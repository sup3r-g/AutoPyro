"""Versioned Rock-Eval formulae with explicit input roles and unit declarations."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

from autopyro.model.core import Column, Project
from autopyro.model.measurements import parse_result, symbolic_lod_operator
from autopyro.model.tabular import classify, project_frame, classify_series, exclusion_mask
from .qc import is_excluded


SOURCE = "https://publications.gc.ca/collections/collection_2016/rncan-nrcan/M183-2-7648-eng.pdf"
FORMULAS = {
    "HI": {"roles": ("TOC", "S2"), "equation": "100*S2/TOC", "unit": "mg HC/g TOC", "toc_numerator": "S2"},
    "OI": {"roles": ("TOC", "S3"), "equation": "100*S3/TOC", "unit": "mg CO2/g TOC", "toc_numerator": "S3"},
    "PI": {"roles": ("S1", "S2"), "equation": "S1/(S1+S2)", "unit": "1"},
    "GP": {"roles": ("S1", "S2"), "equation": "S1+S2", "unit": "mg HC/g rock"},
    "OSI": {"roles": ("TOC", "S1"), "equation": "100*S1/TOC", "unit": "mg HC/g TOC",
            "toc_numerator": "S1", "source": "https://pubs.usgs.gov/publication/70230365"},
}
# Factors convert declared input units to the canonical formula basis.
UNITS = {
    "TOC": {"%wt": 1.0, "g/g rock": 100.0, "mg C/g rock": 0.1},
    "S1": {"mg HC/g rock": 1.0, "ug HC/g rock": 0.001},
    "S2": {"mg HC/g rock": 1.0, "ug HC/g rock": 0.001},
    "S3": {"mg CO2/g rock": 1.0, "ug CO2/g rock": 0.001},
}
STANDARD_UNCERTAINTY_TYPES = {"standard", "standard_uncertainty", "std", "sigma", "стандартная"}


def input_unit(role, unit):
    aliases = {"%": "%wt", "% wt": "%wt",
               "mg/g": "mg CO2/g rock" if role == "S3" else "mg HC/g rock"}
    return aliases.get(unit, unit)


class InputConfigurationError(ValueError):
    def __init__(self, fields):
        self.fields = fields
        super().__init__(next(iter(fields.values())))


def validate_config(project, config):
    if config.get("formula") not in FORMULAS or config.get("version") != 1:
        raise ValueError("Неизвестная формула или версия")
    if (
        not config.get("analysis_context", "").strip()
        or config.get("compatible_confirmed") is not True
    ):
        raise ValueError(
            "Укажите аналитический контекст и подтвердите совместимость исходных измерений"
        )
    columns = {c.id: c for c in project.columns}
    required = FORMULAS[config["formula"]]["roles"]
    if set(config.get("inputs", {})) != set(required):
        raise ValueError("Неверный состав входов формулы")
    bound_ids = [config["inputs"][role].get("column_id") for role in required]
    duplicates = {cid for cid, count in Counter(bound_ids).items() if cid in columns and count > 1}
    errors = {}
    for role in required:
        item = config["inputs"][role]
        if item.get("column_id") not in columns:
            errors[role] = f"Вход {role}: исходная колонка отсутствует"
        elif item["column_id"] in duplicates:
            errors[role] = "Разным входным показателям должны соответствовать разные колонки"
        elif item.get("unit") not in UNITS[role]:
            errors[role] = f"Вход {role}: неподдерживаемая единица"
        else:
            column = columns[item["column_id"]]
            # A conflicting nonempty source unit cannot be silently relabelled.
            if column.unit and input_unit(role, column.unit) != item["unit"]:
                errors[role] = (
                    f"{column.name}: единица колонки «{column.unit}» не совпадает с декларацией «{item['unit']}»"
                )
    if errors:
        raise InputConfigurationError(errors)


def _working_context(project, row_id, column_ids):
    context = []
    for cid in column_ids:
        measurement = project.science.current(row_id, cid)
        analysis = project.science.analyses[measurement.analysis_id] if measurement else None
        context.append((analysis, measurement.basis if measurement else ""))
    return context


def _context_conflicts(context):
    fields = ("method", "laboratory", "preparation", "fraction", "heating_program")
    conflicts = [field for field in fields if len({
        value.strip().casefold() for analysis, _ in context
        if analysis is not None and (value := getattr(analysis, field)).strip()
    }) > 1]
    if len({basis.strip().casefold() for _, basis in context if basis.strip()}) > 1:
        conflicts.append("basis")
    return conflicts


def context_conflicts(project, row_id, column_ids):
    return _context_conflicts(_working_context(project, row_id, column_ids))


def context_diagnostic(project, row_id, config):
    conflicts = context_conflicts(project, row_id, (item["column_id"] for item in config["inputs"].values()))
    if conflicts:
        return {"status": "context_conflict", "fields": conflicts}
    return None


def propagated_uncertainty(project, row_id, config, inputs, value):
    amounts = {}
    issues = {}
    for role, item in config["inputs"].items():
        measurement = project.science.current(row_id, item["column_id"])
        unit = item["unit"]
        if measurement is None or measurement.uncertainty is None:
            issues[role] = "missing"
        elif (not isinstance(measurement.uncertainty_type, str)
              or measurement.uncertainty_type.strip().casefold() not in STANDARD_UNCERTAINTY_TYPES):
            issues[role] = "unsupported_type"
        elif measurement.unit and input_unit(role, measurement.unit) != unit:
            issues[role] = "unit_mismatch"
        else:
            amount = measurement.uncertainty * UNITS[role][unit]
            if math.isfinite(amount) and amount >= 0:
                amounts[role] = amount
            else:
                issues[role] = "invalid_amount"
    if issues:
        return {"status": "unavailable", "inputs": issues}

    formula = config["formula"]
    numerator = FORMULAS[formula].get("toc_numerator")
    if numerator:
        toc = inputs["TOC"]
        if toc >= 1:
            contributions = {"TOC": (value / toc) * amounts["TOC"],
                             numerator: (100 / toc) * amounts[numerator]}
        else:
            contributions = {"TOC": (value * amounts["TOC"]) / toc,
                             numerator: (100 * amounts[numerator]) / toc}
    elif formula == "PI":
        scale = max(inputs["S1"], inputs["S2"])
        first, second = inputs["S1"] / scale, inputs["S2"] / scale
        scaled_sum = first + second
        contributions = {
            "S1": (second / scaled_sum**2 * amounts["S1"]) / scale,
            "S2": (first / scaled_sum**2 * amounts["S2"]) / scale,
        }
    else:
        contributions = amounts
    combined = math.hypot(*contributions.values())
    if not math.isfinite(combined):
        return {"status": "numerical_error"}
    result = {"status": "ok", "standard": combined, "contributions": contributions}
    if ((numerator and amounts["TOC"] >= inputs["TOC"]) or
        (formula == "PI" and math.hypot(*amounts.values()) >= inputs["S1"] + inputs["S2"])):
        result["unstable_denominator"] = True
    return result


def censored_input_bounds(project, row_id, column, role, unit, raw, *, text_numbers):
    """Return canonical bounds or the reason the input cannot bound a result."""
    status, value = classify(raw, numeric=column.kind == "number" or text_numbers)
    factor = UNITS[role][unit]
    if value is not None:
        value = float(value) * factor
        if math.isfinite(value) and value >= 0 and (role != "TOC" or value <= 100):
            return (value, value, True), None
        return None, "invalid_range"
    if status != "censored":
        return None, status
    measurement = project.science.current(row_id, column.id)
    if measurement and measurement.unit and input_unit(role, measurement.unit) != unit:
        return None, "unit_mismatch"
    operator = symbolic_lod_operator(raw)
    if operator:
        if measurement is None or measurement.lod is None:
            return None, "missing_lod"
        limit, closed = measurement.lod, operator != "<"
    else:
        parsed = parse_result(raw, column.kind)
        if parsed["qualifier"] == "above_limit":
            return None, "lower_limit"
        if parsed["qualifier"] != "below_limit":
            return None, "invalid_range_input"
        limit, closed = parsed["limit"], parsed["comparator"] != "<"
    try:
        upper = float(limit) * factor
    except (OverflowError, ValueError):
        return None, "invalid_range_input"
    if not math.isfinite(upper) or upper < 0 or (upper == 0 and (not closed or limit > 0)):
        return None, "invalid_range_input"
    if role == "TOC" and upper > 100:
        return (0.0, 100.0, True), None
    return (0.0, upper, closed), None


def censored_result_range(project, row_id, config, columns, raw_values):
    bounds = {}
    issues = []
    for role, item in config["inputs"].items():
        cid = item["column_id"]
        column = columns[cid]
        if is_excluded(project, row_id, cid, "calculations"):
            interval, issue = None, "qc_excluded"
        elif column.metadata.get("method"):
            interval, issue = None, "interpolation_input"
        else:
            interval, issue = censored_input_bounds(
                project, row_id, column, role, item["unit"], raw_values[cid],
                text_numbers=config.get("text_numbers", False),
            )
        if issue:
            issues.append({"range_issue": issue, "role": role})
        else:
            bounds[role] = interval
    if issues:
        diagnostic = {"status": "censored", **issues[0]}
        if len(issues) > 1:
            diagnostic["range_issues"] = issues
        return diagnostic

    formula = config["formula"]
    definition = FORMULAS[formula]
    first, second = definition["roles"]
    first_low, first_high, first_closed = bounds[first]
    second_low, second_high, second_closed = bounds[second]
    lower_closed = True
    if definition.get("toc_numerator"):
        if first_high == 0:
            return None
        lower = second_low / first_high * 100
        lower_closed = second_low == 0 or first_closed
        upper = 0.0 if second_high == 0 else (second_high / first_low * 100 if first_low > 0 else None)
        upper_closed = upper is not None and (second_high == 0 or second_closed)
    elif formula == "PI":
        if first_high == second_high == 0:
            return None
        if first_high == 0 or second_high == 0:
            lower = upper = float(second_high == 0)
            upper_closed = True
        else:
            lower = production_index(first_low, second_high)
            upper = production_index(first_high, second_low)
            lower_closed = first_low == 0 or second_closed
            upper_closed = second_low == 0 or first_closed
    else:
        lower, upper = first_low + second_low, first_high + second_high
        upper_closed = first_closed and second_closed
    if not math.isfinite(lower) or (upper is not None and not math.isfinite(upper)):
        return None
    if lower == upper and not (lower_closed and upper_closed):
        return None
    return {"status": "censored_range", "range": {
        "lower": lower, "upper": upper, "lower_closed": lower_closed, "upper_closed": upper_closed,
    }}


def censored_diagnostic(project, row_id, config, columns, raw_values):
    if config.get("check_context") is True:
        conflict = context_diagnostic(project, row_id, config)
        if conflict:
            return conflict
    if config.get("show_censored_ranges") is not True:
        return None
    interval = censored_result_range(project, row_id, config, columns, raw_values)
    if interval:
        return interval
    return {"status": "censored", "range_issue": "undefined_range", "role": config["formula"]}


def production_index(s1, s2):
    scale = max(s1, s2)
    first, second = s1 / scale, s2 / scale
    return first / (first + second)


def outside_scope_diagnostic():
    return {"status": "outside_scope", "reason": "Проба не входит в область расчёта"}


def calculate_cell(project, row, config, columns=None):
    columns = columns or {c.id: c for c in project.columns}
    if (
        config.get("scope_row_ids") is not None
        and row.id not in config["scope_row_ids"]
    ):
        return None, outside_scope_diagnostic()
    inputs = {}
    for role, item in config["inputs"].items():
        column = columns[item["column_id"]]
        if is_excluded(project, row.id, column.id, "calculations"):
            return None, {"status": "qc_excluded", "reason": f"{role}: измерение исключено из расчётов решением QC"}
        if column.metadata.get("method"):
            return None, {
                "status": "interpolation_input",
                "reason": f"{role}: интерполированные входы не разрешены",
            }
        status, value = classify(
            row.values[column.id],
            numeric=column.kind == "number" or config.get("text_numbers", False),
        )
        if value is None:
            if status == "censored":
                interval = censored_diagnostic(project, row.id, config, columns, row.values)
                if interval:
                    return None, interval
            return None, {"status": status, "reason": f"{role}: {status}"}
        try:
            value = float(value) * UNITS[role][item["unit"]]
        except (OverflowError, ValueError):
            return None, {
                "status": "numerical_error",
                "reason": f"{role}: переполнение",
            }
        if not math.isfinite(value) or value < 0 or (role == "TOC" and value > 100):
            return None, {
                "status": "invalid_range",
                "reason": f"{role}: значение вне физических пределов",
            }
        inputs[role] = value
    formula = config["formula"]
    numerator = FORMULAS[formula].get("toc_numerator")
    try:
        if numerator:
            if inputs["TOC"] == 0:
                return None, {"status": "zero_denominator", "reason": "TOC = 0"}
            result = inputs[numerator] / inputs["TOC"] * 100
        elif formula == "PI":
            if inputs["S1"] == inputs["S2"] == 0:
                return None, {"status": "zero_denominator", "reason": "S1 + S2 = 0"}
            result = production_index(inputs["S1"], inputs["S2"])
        else:
            result = inputs["S1"] + inputs["S2"]
    except (OverflowError, ZeroDivisionError):
        result = float("nan")
    if not math.isfinite(result):
        return None, {"status": "numerical_error", "reason": "Неконечный результат"}
    info = {"status": "ok", "inputs_canonical": inputs}
    if config.get("check_context") is True:
        conflict = context_diagnostic(project, row.id, config)
        if conflict:
            return None, conflict
    if config.get("propagate_uncertainty") is True:
        info["uncertainty"] = propagated_uncertainty(project, row.id, config, inputs, result)
    return result, info


def calculate_column(project, config, *, row_ids=None, rows=None, checkpoint=None):
    validate_config(project, config)
    columns = {column.id: column for column in project.columns}
    requested_ids = set(row_ids) if row_ids is not None else None
    target_ids = None
    if config.get("scope_row_ids") is not None:
        scope = set(config["scope_row_ids"])
        if rows is None:
            rows = [row for row in project.rows if requested_ids is None or row.id in requested_ids]
        target_ids = [row.id for row in rows]
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("DataFrame index must be unique for orient='index'.")
        rows = [row for row in rows if row.id in scope]
    if checkpoint:
        checkpoint()
    frame = project_frame(project, row_ids=requested_ids, rows=rows,
                          column_ids=[item["column_id"] for item in config["inputs"].values()])
    status = pd.Series("ok", index=frame.index, dtype=object)
    reasons = pd.Series(None, index=frame.index, dtype=object)
    canonical = pd.DataFrame(index=frame.index)
    has_exclusions = bool(project.view.get("qc_exclusions", {}).get("calculations"))
    for role, item in config["inputs"].items():
        if checkpoint:
            checkpoint()
        cid = item["column_id"]
        if has_exclusions:
            excluded = status.eq("ok") & exclusion_mask(project, frame.index, cid, "calculations")
            if excluded.any():
                status.loc[excluded] = "qc_excluded"
                reasons.loc[excluded] = f"{role}: измерение исключено из расчётов решением QC"
        if columns[cid].metadata.get("method"):
            excluded = status.eq("ok")
            status.loc[excluded], reasons.loc[excluded] = "interpolation_input", f"{role}: интерполированные входы не разрешены"
            canonical[role] = np.nan
            continue
        data = classify_series(frame[cid], numeric=columns[cid].kind == "number" or config.get("text_numbers", False),
                               checkpoint=checkpoint)
        missing = status.eq("ok") & data["number"].isna()
        if missing.any():
            missing_status = data.loc[missing, "status"]
            status.loc[missing] = missing_status
            reasons.loc[missing] = role + ": " + missing_status
        # Classification has already rejected numbers that cannot become finite floats.
        with np.errstate(over="ignore", invalid="ignore"):
            canonical[role] = data["number"].astype(float) * UNITS[role][item["unit"]]
        invalid = ~np.isfinite(canonical[role]) | canonical[role].lt(0)
        if role == "TOC":
            invalid |= canonical[role].gt(100)
        invalid &= status.eq("ok")
        if invalid.any():
            status.loc[invalid], reasons.loc[invalid] = "invalid_range", f"{role}: значение вне физических пределов"
    valid = canonical.loc[status.eq("ok")]
    formula = config["formula"]
    numerator = FORMULAS[formula].get("toc_numerator")
    if numerator:
        zero = valid["TOC"].eq(0)
        zero_reason = "TOC = 0"
    elif formula == "PI":
        zero = valid[["S1", "S2"]].max(axis=1).eq(0)
        zero_reason = "S1 + S2 = 0"
    else:
        zero = pd.Series(False, index=valid.index)
        zero_reason = ""
    if zero.any():
        zero_ids = zero.index[zero]
        status.loc[zero_ids] = "zero_denominator"
        reasons.loc[zero_ids] = zero_reason
        valid = valid.loc[~zero]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        if numerator:
            result = valid[numerator] / valid["TOC"] * 100
        elif formula == "PI":
            scale = valid[["S1", "S2"]].max(axis=1)
            first, second = valid["S1"] / scale, valid["S2"] / scale
            result = first / (first + second)
        else:
            result = valid["S1"] + valid["S2"]
    invalid = ~np.isfinite(result)
    if invalid.any():
        invalid_ids = result.index[invalid]
        status.loc[invalid_ids] = "numerical_error"
        reasons.loc[invalid_ids] = "Неконечный результат"
    if not canonical.index.is_unique:
        raise ValueError("DataFrame index must be unique for orient='index'.")
    roles = tuple(canonical.columns)
    values, diagnostics = {}, {}
    for index, (rid, state, reason, value, row_inputs) in enumerate(zip(
        status.index.to_numpy(dtype=object), status.to_numpy(), reasons.to_numpy(),
        result.reindex(frame.index).tolist(), canonical.itertuples(index=False, name=None),
        strict=True,
    )):
        if checkpoint and index % 512 == 0:
            checkpoint()
        if state == "ok":
            values[rid] = value
            diagnostics[rid] = {"status": state, "inputs_canonical": dict(zip(roles, row_inputs))}
        else:
            values[rid] = None
            diagnostics[rid] = {"status": state, "reason": reason}
    if config.get("check_context") is True:
        for index, rid in enumerate(status.index[status.eq("ok")]):
            if checkpoint and index % 512 == 0:
                checkpoint()
            conflict = context_diagnostic(project, rid, config)
            if conflict:
                diagnostics[rid], values[rid] = conflict, None
    if config.get("propagate_uncertainty") is True:
        for index, (rid, value) in enumerate(result.loc[~invalid].items()):
            if checkpoint and index % 512 == 0:
                checkpoint()
            if values[rid] is None:
                continue
            diagnostics[rid]["uncertainty"] = propagated_uncertainty(
                project, rid, config, diagnostics[rid]["inputs_canonical"], value
            )
    if config.get("show_censored_ranges") is True or config.get("check_context") is True:
        for index, (rid, *raw) in enumerate(frame.loc[status.eq("censored")].itertuples(name=None)):
            if checkpoint and index % 512 == 0:
                checkpoint()
            if diagnostics[rid]["status"] != "censored":
                continue
            interval = censored_diagnostic(project, rid, config, columns, dict(zip(frame.columns, raw)))
            if interval:
                diagnostics[rid] = interval
    if target_ids is not None and len(target_ids) != len(frame):
        values = {rid: values.get(rid) for rid in target_ids}
        diagnostics = {rid: diagnostics[rid] if rid in diagnostics else outside_scope_diagnostic()
                       for rid in target_ids}
    if checkpoint:
        checkpoint()
    return values, diagnostics



def compare_lab_results(project, column, formula, values, scope, unit, tolerance, *, text_numbers, inputs=(),
                        checkpoint=None):
    """Compare scoped values in the formula's canonical unit."""
    inputs = tuple(inputs)
    if column.id in inputs:
        raise ValueError("Лабораторная колонка уже используется как вход формулы")
    if checkpoint:
        checkpoint()
    frame = project_frame(project, row_ids=scope, column_ids=[column.id]).reindex(scope)
    result = classify_series(frame[column.id], numeric=column.kind == "number" or text_numbers,
                             checkpoint=checkpoint)
    lab = result["number"].astype(float)
    if unit == "%":
        lab /= 100
    invalid = lab.lt(0)
    if formula == "PI":
        invalid |= lab.gt(1)
    excluded = exclusion_mask(project, result.index, column.id, "calculations")
    result.loc[invalid, "status"] = "invalid_range"
    result.loc[excluded, "status"] = "qc_excluded"
    calculated = pd.Series(values, dtype=float).reindex(result.index)
    valid = result["status"].eq("numeric") & calculated.notna()
    conflicts = {}
    incomplete = set()
    if inputs:
        ids = (*inputs, column.id)
        for index, rid in enumerate(result.index[valid]):
            if checkpoint and index % 512 == 0:
                checkpoint()
            context = _working_context(project, rid, ids)
            fields = _context_conflicts(context)
            if fields:
                conflicts[rid] = tuple(fields)
            elif any(a is None or not a.method.strip() or not a.laboratory.strip() for a, _ in context):
                incomplete.add(rid)
        conflict_ids = list(conflicts)
        result.loc[conflict_ids, "status"] = "context_conflict"
        valid.loc[conflict_ids] = False
    difference = (calculated - lab).where(valid)
    distance = difference.abs()
    # Allow floating-point roundoff at the tolerance boundary, including zero tolerance.
    roundoff = (2 * np.finfo(float).eps) * np.maximum(calculated.abs(), lab.abs())
    within = (distance - tolerance).le(roundoff)
    result["status"] = result["status"].replace("numeric", "calculation_unavailable")
    result.loc[valid & within, "status"] = "match"
    result.loc[valid & ~within, "status"] = "mismatch"
    result["number"] = lab
    result["difference"] = difference
    result["context_fields"] = [conflicts.get(rid, ()) for rid in result.index]
    result["context_incomplete"] = result.index.isin(incomplete)
    if checkpoint:
        checkpoint()
    return result


def make_calculated_column(project, name, config, *, row_ids=None, checkpoint=None):
    values, diagnostics = calculate_column(project, config, row_ids=row_ids, checkpoint=checkpoint)
    definition = FORMULAS[config["formula"]]
    column = Column(
        name,
        "number",
        unit=definition["unit"],
        metadata={
            "calculation": config,
            "equation": definition["equation"],
            "scientific_source": definition.get("source", SOURCE),
            "diagnostics": diagnostics,
        },
    )
    return column, values


def dependent_calculations(columns, changed_columns):
    """Find the transitive formula dependents, including cycles and removed inputs."""
    affected = set(changed_columns)
    dependencies = {c.id: {item["column_id"] for item in c.metadata["calculation"].get("inputs", {}).values()}
                    for c in columns if c.metadata.get("calculation")}
    while True:
        reached = {cid for cid, inputs in dependencies.items() if inputs & affected}
        if reached <= affected:
            return affected
        affected.update(reached)


def recompute(project: Project, *, rows=None, changed_columns=None):
    """Recalculate topologically; return scoped affected IDs, or None for a full pass.

    Invalid definitions/cycles produce explicit nulls. A scoped result includes
    the supplied change set, even if those columns were removed.
    """
    target_rows = project.rows if rows is None else list(rows)
    pending = {c.id: c for c in project.columns if c.metadata.get("calculation")}
    affected = None if changed_columns is None else dependent_calculations(project.columns, changed_columns)
    while pending:
        ready = [
            column
            for column in pending.values()
            if not {
                item["column_id"]
                for item in column.metadata["calculation"].get("inputs", {}).values()
            }
            & set(pending)
        ]
        if not ready:
            for column in pending.values():
                if affected is not None and column.id not in affected:
                    continue
                for row in target_rows:
                    row.values[column.id] = None
                diagnostics = {
                    r.id: {"status": "cyclic_dependency", "reason": "Цикл зависимостей"}
                    for r in target_rows
                }
                if rows is None:
                    column.metadata["diagnostics"] = diagnostics
                else:
                    column.metadata.setdefault("diagnostics", {}).update(diagnostics)
            break
        for column in ready:
            if affected is not None and column.id not in affected:
                pending.pop(column.id)
                continue
            try:
                values, diagnostics = calculate_column(
                    project, column.metadata["calculation"], rows=target_rows if rows is not None else None
                )
            except (ValueError, KeyError, TypeError) as exc:
                values = {}
                diagnostics = {
                    r.id: {"status": "invalid_definition", "reason": str(exc)}
                    for r in target_rows
                }
            for row in target_rows:
                row.values[column.id] = values.get(row.id)
            if rows is None:
                column.metadata["diagnostics"] = diagnostics
            else:
                column.metadata.setdefault("diagnostics", {}).update(diagnostics)
            pending.pop(column.id)
    return affected


def snapshot_diagnostics(diagnostics):
    """Copy standard formula diagnostics without traversing immutable scalars.

    Unrecognized or extended records retain ordinary deepcopy behavior.
    """
    from copy import deepcopy

    result = {}
    for rid, item in diagnostics.items():
        if type(item) is dict and set(item) == {"status", "inputs_canonical"}:
            inputs = item["inputs_canonical"]
            if (type(item["status"]) is str and type(inputs) is dict
                    and all(type(key) is str and type(value) in (float, int, str, bool, type(None))
                            for key, value in inputs.items())):
                result[rid] = {"status": item["status"], "inputs_canonical": inputs.copy()}
                continue
        if (type(item) is dict and set(item) == {"status", "reason"}
                and type(item["status"]) is str and type(item["reason"]) is str):
            result[rid] = item.copy()
        else:
            result[rid] = deepcopy(item)
    return result


def diagnostic_counts(column):
    return dict(
        Counter(
            item["status"] for item in column.metadata.get("diagnostics", {}).values()
        )
    )
