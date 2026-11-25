"""Versioned mechanical data checks and explicit, revision-bound QC decisions."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields as dataclass_fields
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from itertools import groupby
import json
import math
from operator import itemgetter

import pandas as pd

from autopyro.model.core import Project, Row, parse_number
from autopyro.model.measurements import Measurement, MeasurementStore, SourceRecord, snapshot_record, symbolic_lod_operator

PROFILE_KEY = ("qc_profiles", "pyrolysis")
QC_SEVERITY_ORDER = {"error": 0, "critical": 1, "warning": 2}
PROFILE_UNITS = {"TOC": "%wt", "S1": "mg HC/g rock", "S2": "mg HC/g rock",
                 "TMAX": "°C", "HI": "mg HC/g TOC", "OI": "mg CO2/g TOC"}
PROFILE_DEFAULTS = {
    "TOC": (0, 0.1, 84, 100), "S1": (0, 0.2, 2, 100), "S2": (0, 0.5, 100, 1000),
    "TMAX": (350, 400, 500, 600), "HI": (0, 10, 900, 1200), "OI": (0, 2, 5, 500),
}

SCOPES = {"statistics": "Статистика", "calculations": "Расчёты пиролиза",
          "plots": "Графики и карты", "interpretation": "Геометрическая интерпретация",
          "interpolation": "Интерполяция"}
RULES = {
    "value_format": {"version": 1, "description": "Неразобранная запись числового показателя", "source": "Контракт формата AutoPyro", "parameters": {}},
    "unit": {"version": 1, "description": "Единица пиролитического показателя не объявлена или не поддерживается", "source": "autopyro.calculations.UNITS / SCIENTIFIC_METHODS.md", "parameters": {}},
    "range": {"version": 1, "description": "Физически недопустимое значение", "source": "Определения массовой доли, отношения PI и неотрицательного сигнала", "parameters": {"TOC_percent": [0, 100], "PI_fraction": [0, 1]}},
    "approximate_range": {"version": 1, "description": "Приблизительное значение вне физических пределов",
                          "source": "Проверка номинального приблизительного значения AutoPyro",
                          "parameters": {"action": "review_nominal_value"}},
    "context": {"version": 1, "description": "Не указаны метод или лаборатория измерения", "source": "Требования к прослеживаемости AutoPyro", "parameters": {"required": ["method", "laboratory"]}},
    "repeat": {"version": 3, "description": "Повторы с одинаковым основанием расходятся", "source": "Контроль повторных результатов AutoPyro; это предупреждение, а не критерий браковки", "parameters": {"comparison": "common_declared_interval_or_exact"}},
    "repeat_censored": {"version": 1, "description": "Повторы с пороговыми записями расходятся",
                        "source": "Контроль повторных результатов AutoPyro; это предупреждение, а не критерий браковки",
                        "parameters": {"comparison": "common_interval_with_censoring", "nd_upper_closed": True}},
    "compatibility": {"version": 1, "description": "Контексты рабочих S1/S2/S3 различаются", "source": "Требование явной проверки совместимости аналитических измерений AutoPyro", "parameters": {"fields": ["method", "laboratory", "preparation", "fraction", "heating_program"]}},
    "calculation_context": {"version": 1, "description": "Контексты входов расчёта различаются",
                            "source": "Требование явной проверки совместимости аналитических измерений AutoPyro",
                            "parameters": {}},
}


RULES.update({
    "profile_absolute": {"version": 1, "description": "Вне абсолютных порогов профиля", "source": "Пользовательский профиль пиролиза", "parameters": {}},
    "profile_typical": {"version": 1, "description": "Вне типичных порогов профиля", "source": "Пользовательский профиль пиролиза", "parameters": {}},
    "profile_ambiguous": {"version": 1, "description": "Диапазон пересекает порог профиля", "source": "Пользовательский профиль пиролиза", "parameters": {}},
})

_ENGLISH_RULE_DESCRIPTIONS = {
    "approximate_range": "Approximate value outside physical limits",
    "value_format": "Unparsed numeric property record",
    "unit": "Missing or unsupported pyrolysis unit",
    "range": "Physically invalid value",
    "context": "Missing measurement method or laboratory",
    "repeat": "Repeat measurements on the same basis disagree",
    "repeat_censored": "Repeats with censored records disagree",
    "compatibility": "Working S1/S2/S3 contexts differ",
    "calculation_context": "Calculation input contexts differ",
    "profile_absolute": "Outside absolute profile thresholds",
    "profile_typical": "Outside typical profile thresholds",
    "profile_ambiguous": "Range crosses a profile threshold",
}


def rule_description(rule_id, rules, *, english=False):
    description = rules.get(rule_id, {}).get("description") or rule_id
    if english and description == RULES.get(rule_id, {}).get("description"):
        return _ENGLISH_RULE_DESCRIPTIONS.get(rule_id, description)
    return description

PYROLYSIS_ROLES = frozenset({"TOC", "S1", "S2", "S3", "HI", "OI", "PI", "GP", "TMAX"})
_ROLE_ALIASES = {"T-MAX": "TMAX", "T MAX": "TMAX"}


def parse_exclusion_time(value):
    try:
        instant = datetime.fromisoformat(value)
        return instant if instant.tzinfo is None else instant.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def exclusion_time(value):
    instant = parse_exclusion_time(value)
    if instant is None:
        return str(value) if value else "—"
    if instant.tzinfo is None:
        return instant.isoformat(sep=" ", timespec="seconds")
    return instant.strftime("%Y-%m-%d %H:%M:%S UTC")


def role(column):
    if "property_role" in column.metadata:
        name = column.metadata["property_role"]
        if not isinstance(name, str):
            return None
    else:
        group, separator, name = column.name.rpartition(" / ")
        if separator:
            # The importer inserts " / " for grouped headers. A pair of
            # indicators separated by the same delimiter is also a ratio.
            numerator = group.rsplit(" / ", 1)[-1].strip().upper()
            if _ROLE_ALIASES.get(numerator, numerator) in PYROLYSIS_ROLES or numerator == "S0":
                return None
    name = name.strip().upper()
    name = _ROLE_ALIASES.get(name, name)
    return name if name in PYROLYSIS_ROLES else None


def declared_repeat_interval(measurement, center, factor):
    amount = measurement.uncertainty
    kind = measurement.uncertainty_type
    if (not isinstance(kind, str) or type(amount) not in (int, float) or amount < 0
            or (type(amount) is float and not math.isfinite(amount))):
        return None
    from autopyro.schemes.measurement_uncertainty import ABSOLUTE_TYPES, PERCENT_TYPES
    from .calculations import STANDARD_UNCERTAINTY_TYPES

    kind = kind.strip().casefold()
    radius = Fraction(str(amount))
    if kind in ABSOLUTE_TYPES:
        family = ("standard" if kind in STANDARD_UNCERTAINTY_TYPES else "expanded"
                  if kind in {"expanded", "expanded_uncertainty", "расширенная"} else "absolute")
        radius *= factor
    elif kind in PERCENT_TYPES or kind == "relative_fraction":
        family = "relative"
        radius *= abs(center)
        if kind in PERCENT_TYPES:
            radius /= 100
    else:
        return None
    return (center - radius, center + radius), (family, measurement.confidence_level)


def repeat_comparison(peers, numbers, units, factors):
    """Return the comparison that found a difference, or None for overlapping intervals."""
    if any(measurement.uncertainty is None for measurement in peers):
        return "exact"
    declaration = None
    lower = upper = None
    for measurement in peers:
        factor = factors.get(units[measurement.id])
        if factor is None:
            return "exact"
        factor = Fraction(str(factor))
        center = Fraction(str(numbers[measurement.id])) * factor
        interval = declared_repeat_interval(measurement, center, factor)
        if interval is None:
            return "exact"
        (lo, hi), current = interval
        if declaration is not None and current != declaration:
            return "exact"
        declaration = current
        lower = lo if lower is None else max(lower, lo)
        upper = hi if upper is None else min(upper, hi)
    return "interval" if lower > upper else None


def intersect_intervals(intervals):
    # Prefer an open endpoint when equal bounds meet. None denotes +infinity.
    low, low_open = max((low, not closed) for low, _, closed, _ in intervals)
    upper = [(high, closed) for _, high, _, closed in intervals if high is not None]
    if not upper:
        return low, None, not low_open, False
    high, high_closed = min(upper)
    if low > high or (low == high and (low_open or not high_closed)):
        return None
    return low, high, not low_open, high_closed


def is_excluded(project, row_id, column_id, scope):
    measurement = project.science.current(row_id, column_id)
    if measurement is None:
        return False
    exclusion = project.view.get("qc_exclusions", {}).get(scope, {}).get(measurement.id)
    return bool(exclusion and exclusion.get("revision") == measurement.revision)


def selected_exclusions(project, measurement_ids):
    return {scope: {mid: deepcopy(record) for mid, record in records.items() if mid in measurement_ids}
            for scope, records in project.view.get("qc_exclusions", {}).items()}


def active_exclusions(project, measurement_ids=None):
    measurements = project.science.measurements
    for scope, records in project.view.get("qc_exclusions", {}).items():
        for mid, record in records.items():
            measurement = measurements.get(mid)
            if (measurement is not None and record.get("revision") == measurement.revision
                    and (measurement_ids is None or mid in measurement_ids)):
                yield scope, measurement, record


def pyrolysis_profile(project):
    saved = project.view.get(PROFILE_KEY[0], {}).get(PROFILE_KEY[1], {})
    return {"enabled": False, "thresholds": PROFILE_DEFAULTS,
            "source": "Пользовательская таблица порогов", "version": "", "applicability": "", **saved}


class ProfileThresholdError(ValueError):
    def __init__(self, column):
        super().__init__("Пороги должны быть конечными числами в порядке возрастания")
        self.column = column


def profile_thresholds(values):
    thresholds = []
    for index, value in enumerate(values):
        try:
            number = float(parse_number(value))
            if not math.isfinite(number) or (thresholds and number < thresholds[-1]):
                raise ValueError
        except (ValueError, TypeError, OverflowError):
            raise ProfileThresholdError(index) from None
        thresholds.append(number)
    if len(thresholds) != 4:
        raise ValueError("Требуются четыре порога")
    return thresholds


def profile_violations(bounds, thresholds):
    low, high, low_closed, high_closed = bounds
    absolute_min, typical_min, typical_max, absolute_max = thresholds
    for threshold, below, rule in (
        (absolute_min, True, "profile_absolute"), (absolute_max, False, "profile_absolute"),
        (typical_min, True, "profile_typical"), (typical_max, False, "profile_typical"),
    ):
        if below:
            whole = high is not None and (high < threshold or (high == threshold and not high_closed))
            part = low < threshold
        else:
            whole = low > threshold or (low == threshold and not low_closed)
            part = high is None or high > threshold
        if part:
            yield {"rule": rule, "direction": "below" if below else "above",
                   "threshold": threshold, "whole": whole}


def profile_finding(bounds, thresholds):
    first = next(profile_violations(bounds, thresholds), None)
    return (first["rule"] if first["whole"] else "profile_ambiguous") if first else None


def profile_measurement_bounds(measurement, number, role, factor, *, for_repeat=False):
    if number is not None and not pd.isna(number):
        try:
            value = float(number) * factor
        except OverflowError:
            return None
        return (value, value, True, True) if math.isfinite(value) else None
    operator = symbolic_lod_operator(measurement.raw_value)
    if operator:
        limit = measurement.lod
    elif measurement.qualifier in ("below_limit", "above_limit"):
        limit, operator = measurement.limit, measurement.comparator
    elif measurement.qualifier == "non_detect":
        limit, operator = measurement.lod, "<="
    else:
        return None
    if limit is None or operator not in ("<", "<=", "≤", ">", ">=", "≥"):
        return None
    low = -273.15 if role == "TMAX" else 0.0
    try:
        boundary = Fraction(str(limit)) * Fraction(str(factor)) if for_repeat else float(limit) * factor
    except (OverflowError, ValueError):
        return None
    if not for_repeat and not math.isfinite(boundary):
        return None
    maximum = 100.0 if role == "TOC" else 1.0 if role == "PI" else None
    if for_repeat:
        low = Fraction(str(low))
        maximum = Fraction(str(maximum)) if maximum is not None else None
    if operator in (">", ">=", "≥"):
        closed = operator != ">"
        if maximum is not None and (boundary > maximum or (boundary == maximum and not closed)):
            return None
        return max(low, boundary), maximum, boundary < low or closed, maximum is not None
    closed = operator != "<"
    if boundary < low or (boundary == low and not closed):
        return None
    if maximum is not None and boundary > maximum:
        return low, maximum, True, True
    return low, boundary, True, closed


def snapshot_for_checks(project, checkpoint):
    """Isolate QC inputs without copying unrelated project artifacts or import originals."""
    def copy_records(records, copy_record):
        copied = {}
        for index, (key, record) in enumerate(records.items()):
            if index % 1024 == 0:
                checkpoint()
            copied[key] = copy_record(record)
        return copied

    science = project.science
    store = MeasurementStore(
        measurements=copy_records(science.measurements, snapshot_record),
        analyses=copy_records(science.analyses, snapshot_record),
        sources=copy_records(science.sources, lambda source: SourceRecord(
            id=source.id, provenance=deepcopy(source.provenance))),
        working=copy_records(science.working, dict.copy),
    )
    rows = []
    for index, row in enumerate(project.rows):
        if index % 1024 == 0:
            checkpoint()
        rows.append(Row(deepcopy(row.values), id=row.id, source=deepcopy(row.source)))
    view = {key: deepcopy(project.view[key]) for key in ("qc_decisions", "qc_profiles") if key in project.view}
    calculations = project.view.get("qc_exclusions", {}).get("calculations")
    if calculations:
        view["qc_exclusions"] = {"calculations": deepcopy(calculations)}
    checkpoint()
    return Project(columns=deepcopy(project.columns), rows=rows, science=store, view=view, id=project.id)


def _profile_issue(name, bounds, thresholds, profile, english):
    rule = profile_finding(bounds, thresholds)
    if rule is None:
        return None
    low, high, low_closed, high_closed = bounds
    data = {"role": name, "unit": PROFILE_UNITS[name], "thresholds": thresholds,
            "lower": low, "upper": high, "lower_closed": low_closed, "upper_closed": high_closed,
            "violations": list(profile_violations(bounds, thresholds)),
            "provenance": {key: profile[key] for key in ("source", "version", "applicability")}}
    interval = f"{low:.10g}" if low == high else (
        ("[" if low_closed else "(") + f"{low:.10g}, "
        + (f"{high:.10g}" if high is not None else "∞") + ("]" if high_closed else ")")
    )
    data["display_value"] = interval
    description = rule_description(rule, RULES, english=english)
    text = f"{description}: {name} {interval} [{data['unit']}]"
    return rule, "critical" if rule == "profile_absolute" else "warning", text, data


def _measurement_context_frame(project, measurements, columns, checkpoint):
    frame = pd.DataFrame(
        [(m.id, m.sample_id, m.property_id, m.unit, m.basis, m.qualifier, m.value, m.analysis_id)
         for m in measurements],
        columns=["id", "sample_id", "property_id", "unit", "basis", "qualifier", "value", "analysis_id"],
        dtype=object,
    ).set_index("id")
    if checkpoint:
        checkpoint("Подготовка измерений…", 10)
    contexts = RULES["compatibility"]["parameters"]["fields"]
    analyses = pd.DataFrame(
        [{"id": analysis.id, **{name: getattr(analysis, name).strip() for name in contexts}}
         for analysis in project.science.analyses.values()], columns=["id", *contexts], dtype=object,
    ).set_index("id")
    frame = frame.join(analyses, on="analysis_id", validate="many_to_one")
    frame["role"] = frame["property_id"].map({cid: role(column) for cid, column in columns.items()})
    return frame


def _check_working_contexts(project, frame, peer_payload, add, english):
    contexts = RULES["compatibility"]["parameters"]["fields"]
    working_ids = {mid for selected in project.science.working.values() for mid in selected.values()}
    active = frame.loc[frame.index.isin(working_ids) & frame["role"].isin(["S1", "S2", "S3"])
                       & frame["qualifier"].ne("missing")]
    differences = active[contexts].replace("", None).groupby(active["sample_id"], sort=False).nunique()
    incompatible = differences.gt(1).any(axis=1)
    for _, group in active.loc[active["sample_id"].isin(incompatible.index[incompatible])].groupby("sample_id", sort=False):
        peers = [project.science.measurements[mid] for mid in group.index]
        peer_data = peer_payload(peers)
        for measurement in peers:
            add("compatibility", "warning", measurement,
                "Working pyrolysis inputs have different analytical contexts. Check compatibility before calculating."
                if english else "Рабочие пиролитические входы имеют разные аналитические контексты. Проверьте совместимость до расчёта.",
                peer_data=peer_data)


def run_checks(project, *, unchecked_derived=None, profile_coverage=None, checkpoint=None,
               include_peers=False, coverage_details=None, language="ru"):
    """Read-only physical checks and optional project threshold checks."""
    from .calculations import FORMULAS, UNITS, calculate_column, context_conflicts
    english = language == "en"

    def checked_items(items, stage, percent, *, span=0, total=1):
        if checkpoint is None:
            yield from items
            return
        for index, item in enumerate(items):
            if index % 1024 == 0:
                checkpoint(stage, percent + index * span // max(1, total))
            yield item

    columns = {c.id: c for c in project.columns}
    rows = project.row_map()
    measurements = [m for m in project.science.measurements.values()
                    if m.property_id in columns and m.sample_id in rows]
    frame = _measurement_context_frame(project, measurements, columns, checkpoint)
    issues = []
    # These records are shared by many findings. Cache only for this check so
    # subsequent edits still change the revision-bound decision signatures.
    column_payloads = {}
    analysis_payloads = {}
    analysis_json = {}
    measurement_fields = tuple(field.name for field in dataclass_fields(Measurement))
    scalar_types = (str, int, float, bool, type(None))

    def measurement_payload(measurement):
        if type(measurement) is Measurement and not measurement.history:
            payload = {name: getattr(measurement, name) for name in measurement_fields if name != "history"}
            if all(type(value) in scalar_types for value in payload.values()):
                payload["history"] = []
                return payload
        return asdict(measurement)

    def analysis_payload(analysis_id):
        if analysis_id not in analysis_payloads:
            analysis_payloads[analysis_id] = asdict(project.science.analyses[analysis_id])
        return analysis_payloads[analysis_id]

    def peer_payload(peers, *, include_units=False, include_limits=False):
        return [
            {"id": m.id, "revision": m.revision, "value": m.value, "raw": m.raw_value,
             "analysis": analysis_payload(m.analysis_id),
             **({"unit": m.unit, "basis": m.basis, "uncertainty": m.uncertainty,
                 "uncertainty_type": m.uncertainty_type, "confidence_level": m.confidence_level}
                if include_units else {}),
             **({"qualifier": m.qualifier, "limit": m.limit, "comparator": m.comparator, "lod": m.lod}
                if include_limits else {})}
            for m in sorted(peers, key=lambda m: m.id)
        ]

    def append_issue(rule_id, severity, sample_id, property_id, measurement_id, revision,
                     source, message, signature, profile=None):
        identity = measurement_id or f"{sample_id}:{property_id}"
        issue_id = f"{rule_id}:{identity}:{signature}"
        decision = project.view.get("qc_decisions", {}).get(issue_id, {})
        issue = {"id": issue_id, "rule_id": rule_id, "rule_version": RULES[rule_id]["version"],
            "severity": severity, "sample_id": sample_id, "property_id": property_id,
            "measurement_id": measurement_id, "revision": revision, "message": message,
            "source": deepcopy(source), "decision": decision.get("action", "open"),
            "comment": decision.get("comment", "")}
        if decision.get("at"):
            issue["decision_at"] = decision["at"]
        if profile:
            issue["profile"] = profile
        issues.append(issue)

    encode = json.JSONEncoder(sort_keys=True, ensure_ascii=False, allow_nan=False).encode
    rule_json = {key: encode(rule) for key, rule in RULES.items()}

    def digest(payload):
        return hashlib.sha256(encode(payload).encode()).hexdigest()

    def profile_signature(data):
        return {key: value for key, value in data.items() if key != "violations"}

    def add(rule_id, severity, measurement, message, *, peer_data=None, peer_signature=None, profile=None):
        source = project.science.sources.get(measurement.source_id)
        if measurement.property_id not in column_payloads:
            column_payloads[measurement.property_id] = encode(asdict(columns[measurement.property_id]))
        if measurement.analysis_id not in analysis_json:
            analysis = analysis_payloads.get(measurement.analysis_id)
            if analysis is None:
                analysis = asdict(project.science.analyses[measurement.analysis_id])
            analysis_json[measurement.analysis_id] = encode(analysis)
        # Keep the exact canonical JSON used by saved decisions while reusing
        # the serialized fields shared by many findings.
        parts = [
            '"analysis": ' + analysis_json[measurement.analysis_id],
            '"column": ' + column_payloads[measurement.property_id],
            '"measurement": ' + encode(measurement_payload(measurement)),
            '"peers": ' + encode(peer_signature or (peer_data if peer_data is not None else [])),
        ]
        if profile:
            parts.append('"profile": ' + encode(profile_signature(profile)))
        parts.append('"rule": ' + rule_json[rule_id])
        signature = hashlib.sha256(("{" + ", ".join(parts) + "}").encode()).hexdigest()
        append_issue(rule_id, severity, measurement.sample_id, measurement.property_id,
                     measurement.id, measurement.revision, source.provenance if source else {},
                     message, signature, profile)
        if peer_data and include_peers:
            issues[-1]["peers"] = peer_data

    _check_working_contexts(project, frame, peer_payload, add, english)

    rule = RULES["calculation_context"]
    for column in columns.values():
        config = column.metadata.get("calculation", {})
        formula = config.get("formula")
        if formula not in FORMULAS:
            continue
        inputs = config.get("inputs", {})
        if set(inputs) != set(FORMULAS[formula]["roles"]):
            continue
        input_ids = tuple(item.get("column_id") for item in inputs.values())
        diagnostics = column.metadata.get("diagnostics", {})
        scope_ids = config.get("scope_row_ids")
        scope = rows.keys() if scope_ids is None else dict.fromkeys(scope_ids)
        for sid in checked_items(scope, "Проверка контекста расчётов…", 30):
            if sid not in rows:
                continue
            info = diagnostics.get(sid, {})
            status = info.get("status")
            if status not in ("censored_range", "context_conflict") and rows[sid].values.get(column.id) is None:
                continue
            conflicts = context_conflicts(project, sid, input_ids)
            if not conflicts:
                continue
            peers = [measurement for cid in input_ids if (measurement := project.science.current(sid, cid))]
            related = peer_payload(peers, include_units=True)
            signature = digest({"rule": rule, "formula": formula, "inputs": inputs, "peers": related,
                                "fields": conflicts, "context_policy": config.get("check_context") is True})
            text = ("Расчёт пропущен из-за различий аналитического контекста."
                    if status == "context_conflict"
                    else "Результат рассчитан из измерений с различным аналитическим контекстом.")
            if english:
                text = ("Calculation skipped because the analytical contexts differ." if status == "context_conflict"
                        else "The result uses measurements from different analytical contexts.")
            append_issue("calculation_context", "warning", sid, column.id, None, None,
                         rows[sid].source, text, signature)
            issues[-1]["context_fields"] = conflicts
            if include_peers:
                issues[-1]["peers"] = related

    allowed = {**UNITS, "HI": {"mg HC/g TOC": 1}, "OI": {"mg CO2/g TOC": 1},
               "PI": {"1": 1}, "GP": {"mg HC/g rock": 1}, "TMAX": {"°C": 1, "C": 1, "degC": 1}}
    unit = frame["unit"].replace({"%": "%wt", "% wt": "%wt", "mg/g": "mg HC/g rock"})
    unit.loc[frame["unit"].eq("mg/g") & frame["role"].eq("S3")] = "mg CO2/g rock"
    allowed_units = pd.MultiIndex.from_tuples([(name, label) for name, units in allowed.items() for label in units])
    invalid_unit = ~pd.MultiIndex.from_arrays([frame["role"], unit]).isin(allowed_units)
    missing_context = frame["method"].eq("").map({True: "method" if english else "метод", False: ""})
    missing_context += (frame["method"].eq("") & frame["laboratory"].eq("")).map({True: ", ", False: ""})
    missing_context += frame["laboratory"].eq("").map({True: "laboratory" if english else "лаборатория", False: ""})

    def parse_exact(value):
        try:
            return True, parse_number(value)
        except (ValueError, TypeError, OverflowError):
            return False, None

    exact = frame["qualifier"].eq("exact")
    approximate = frame["qualifier"].eq("approximate")
    numeric = exact | approximate
    parsed = pd.DataFrame(frame.loc[numeric, "value"].map(parse_exact).tolist(),
                          index=frame.index[numeric], columns=["valid", "number"], dtype=object).reindex(frame.index)
    numbers = parsed["number"]
    profile = pyrolysis_profile(project)
    valid_lod_ids = {
        m.id for m in measurements if m.qualifier == "unparsed"
        and m.lod is not None and symbolic_lod_operator(m.raw_value)
    }
    nd_with_lod = {m.id for m in measurements if m.qualifier == "non_detect" and m.lod is not None}
    qualified = (frame["qualifier"].isin(["below_limit", "above_limit"])
                 | frame.index.isin(valid_lod_ids | nd_with_lod))
    bounds_by_id = {}
    impossible_limits = set()
    if checkpoint:
        checkpoint("Проверка диапазонов…", 40)
    for mid, name in checked_items(frame.loc[qualified & ~invalid_unit, "role"].items(),
                                   "Проверка диапазонов…", 40):
        measurement = project.science.measurements[mid]
        bounds = profile_measurement_bounds(measurement, None, name, allowed[name][unit[mid]])
        if bounds is None:
            impossible_limits.add(mid)
        else:
            bounds_by_id[mid] = bounds
    if profile.get("enabled") is True:
        thresholds = {name: profile_thresholds(profile["thresholds"][name]) for name in PROFILE_UNITS}
        sample_scope = profile.get("sample_scope")
        profile_rows = rows.keys() if sample_scope is None else rows.keys() & set(sample_scope["row_ids"])
        supported = frame["role"].isin(PROFILE_UNITS)
        if sample_scope is not None:
            supported &= frame["sample_id"].isin(profile_rows)
        missing = frame["qualifier"].eq("missing")
        candidates = supported & ~missing
        usable_exact = exact & ~invalid_unit
        unsupported_units = set(frame.index[supported & invalid_unit]) if coverage_details is not None else ()
        if coverage_details is not None:
            for mid in frame.index[supported & missing]:
                measurement = project.science.measurements[mid]
                coverage_details.append((measurement.sample_id, measurement.property_id, "missing"))
        candidates_data = zip(frame.index[candidates], frame.loc[candidates, "role"],
                              unit[candidates], numbers[candidates])
        for mid, name, label, number in checked_items(
                candidates_data, "Проверка диапазонов…", 45, span=15, total=int(candidates.sum())):
            measurement = project.science.measurements[mid]
            bounds = bounds_by_id.get(mid)
            if usable_exact[mid]:
                bounds = profile_measurement_bounds(measurement, number, name, allowed[name][label])
            if profile_coverage is not None:
                profile_coverage.append((measurement.sample_id, bounds is not None))
            if coverage_details is not None:
                status = ("checked" if bounds is not None else "unit" if mid in unsupported_units
                          else "range" if mid in impossible_limits else "no_bounds")
                coverage_details.append((measurement.sample_id, measurement.property_id, status))
            finding = _profile_issue(name, bounds, thresholds[name], profile, english) if bounds else None
            if finding:
                rule, severity, text, data = finding
                add(rule, severity, measurement, text, profile=data)
        for column in project.columns:
            config = column.metadata.get("calculation", {})
            name = config.get("formula")
            if name not in ("HI", "OI"):
                continue
            diagnostics = column.metadata.get("diagnostics", {})
            if column.unit != PROFILE_UNITS[name]:
                if coverage_details is not None:
                    coverage_details.extend((sid, column.id, "unit") for sid, info in diagnostics.items()
                                            if sid in profile_rows and info["status"] != "outside_scope")
                continue
            censored_ids = [sid for sid, info in diagnostics.items() if info["status"] == "censored" and sid in profile_rows]
            ranges = {}
            if censored_ids and not config.get("show_censored_ranges"):
                _, ranges = calculate_column(project, {**config, "show_censored_ranges": True}, row_ids=censored_ids)
            for sid, info in checked_items(diagnostics.items(), "Проверка диапазонов…", 60):
                if sid not in profile_rows:
                    continue
                info = ranges.get(sid, info)
                if info["status"] == "censored_range":
                    interval = info["range"]
                    bounds = tuple(interval[key] for key in ("lower", "upper", "lower_closed", "upper_closed"))
                elif info["status"] == "ok" and rows[sid].values[column.id] is not None:
                    value = rows[sid].values[column.id]
                    bounds = (value, value, True, True)
                else:
                    if unchecked_derived is not None and info["status"] != "outside_scope":
                        unchecked_derived.append(sid)
                    if coverage_details is not None and info["status"] != "outside_scope":
                        status = "excluded" if info["status"] == "qc_excluded" else "no_bounds"
                        coverage_details.append((sid, column.id, status))
                    continue
                if coverage_details is not None:
                    coverage_details.append((sid, column.id, "checked"))
                finding = _profile_issue(name, bounds, thresholds[name], profile, english)
                if finding:
                    rule, severity, text, data = finding
                    inputs = {key: project.science.current(sid, item["column_id"])
                              for key, item in config["inputs"].items()}
                    signature = digest({"rule": RULES[rule], "profile": profile_signature(data), "diagnostics": info,
                        "inputs": {key: (m.id, m.revision) if m else None for key, m in inputs.items()}})
                    append_issue(rule, severity, sid, column.id, None, None,
                                 rows[sid].source, text, signature, data)
    if checkpoint:
        checkpoint("Проверка формата, контекста и повторов…", 70)
    invalid_range = frame["role"].ne("TMAX") & numbers.lt(0)
    invalid_range |= frame["role"].eq("PI") & unit.eq("1") & numbers.gt(1)
    toc_limit = unit.map({label: 100 / factor for label, factor in UNITS["TOC"].items()})
    invalid_range |= frame["role"].eq("TOC") & numbers.fillna(0).gt(toc_limit.fillna(float("inf")).astype(object))
    invalid_range |= frame["role"].eq("TMAX") & ~invalid_unit & numbers.lt(-273.15)
    recognized = frame["role"].notna()
    canonical_units = {(name, label): next(iter(units))
                       for name, units in allowed.items() for label in units}
    comparison_units = pd.Series(
        [canonical_units.get((name, label), label) for name, label in zip(frame["role"], unit)],
        index=frame.index, dtype=object,
    )
    repeat_groups = frame.groupby([frame["sample_id"], frame["property_id"], comparison_units, frame["basis"]],
                                  sort=False, dropna=False).ngroup()
    comparable = recognized & exact & numbers.notna()
    comparison_numbers = numbers.loc[comparable].copy()
    comparison_groups = repeat_groups.loc[comparable]
    mixed = unit.loc[comparable].groupby(comparison_groups, sort=False).transform("nunique").gt(1)
    for mid in mixed.index[mixed]:
        factor = allowed[frame.at[mid, "role"]][unit[mid]]
        comparison_numbers.at[mid] = Fraction(str(numbers[mid])) * Fraction(str(factor))
    repeat_values = comparison_numbers.groupby(comparison_groups, sort=False)
    disagreement = repeat_values.transform("nunique").gt(1).reindex(frame.index, fill_value=False)
    repeat_bounds = {}
    bounded = ((qualified | frame["qualifier"].eq("non_detect")) & recognized & ~invalid_unit
               & repeat_groups.duplicated(keep=False))
    for mid, name in frame.loc[bounded, "role"].items():
        measurement = project.science.measurements[mid]
        bounds = profile_measurement_bounds(measurement, None, name, allowed[name][unit[mid]], for_repeat=True)
        if bounds is not None:
            repeat_bounds[mid] = bounds
    bounded_groups = repeat_groups.loc[list(repeat_bounds)].unique()
    mixed_members = repeat_groups.isin(bounded_groups) & (
        (comparable & ~invalid_unit) | frame.index.isin(repeat_bounds))
    needs_context = missing_context.ne("") & frame["qualifier"].ne("missing")
    invalid_exact = numeric & parsed["valid"].eq(False)
    unparsed = frame["qualifier"].eq("unparsed") & ~frame.index.isin(valid_lod_ids)
    candidates = unparsed | disagreement | mixed_members | (
        recognized & (needs_context | (invalid_unit & frame["qualifier"].ne("missing"))
                      | invalid_exact | (numeric & invalid_range) | frame.index.isin(impossible_limits))
    )
    if not candidates.any():
        return issues
    checks = pd.DataFrame({
        "group": repeat_groups, "recognized": recognized, "context": missing_context.where(needs_context, ""),
        "invalid_unit": invalid_unit & frame["qualifier"].ne("missing"), "invalid_exact": invalid_exact,
        "invalid_range": numeric & invalid_range, "number": numbers, "disagreement": disagreement,
        "mixed_repeat": mixed_members,
    }, index=frame.index)
    # Keep the original group order even when the first row has no issues.
    ordered = checks.loc[candidates].sort_values("group", kind="stable")
    checked = checked_items(ordered.itertuples(name=None), "Проверка формата, контекста и повторов…",
                            85, span=15, total=len(ordered))
    for _, group in groupby(checked, key=itemgetter(1)):
        peers = []
        mixed_peers = []
        limit_intervals = []
        exact_intervals = []
        for mid, _, has_role, context, bad_unit, bad_exact, bad_range, number, differs, mixed_repeat in group:
            measurement = project.science.measurements[mid]
            column = columns[measurement.property_id]
            if differs:
                peers.append(measurement)
            if mixed_repeat:
                bounds = repeat_bounds.get(mid)
                if bounds is None:
                    factor = Fraction(str(allowed[frame.at[mid, "role"]][unit[mid]]))
                    center = Fraction(str(number)) * factor
                    declared = declared_repeat_interval(measurement, center, factor)
                    lo, hi = declared[0] if declared is not None else (center, center)
                    exact_intervals.append((lo, hi, True, True))
                else:
                    limit_intervals.append(bounds)
                mixed_peers.append(measurement)
            if measurement.qualifier == "unparsed" and mid not in valid_lod_ids:
                add("value_format", "error", measurement, f"Could not parse the value: {measurement.raw_value}" if english
                    else f"Не удалось разобрать запись: {measurement.raw_value}")
            if not has_role:
                continue
            if context:
                add("context", "warning", measurement, ("Missing: " if english else "Не указаны: ") + context)
            if bad_unit:
                add("unit", "warning", measurement,
                    f"{column.name}: check the unit ({measurement.unit or 'not specified'})" if english
                    else f"{column.name}: единица «{measurement.unit or 'не задана'}» требует проверки")
            if bad_exact:
                add("value_format", "error", measurement, f"{column.name}: expected a numeric value" if english
                    else f"{column.name}: ожидалось числовое значение")
            elif mid in impossible_limits:
                add("range", "error", measurement,
                    f"{column.name}: limit {measurement.raw_value} does not define a physically valid range" if english
                    else f"{column.name}: предел {measurement.raw_value} не задаёт физически допустимый диапазон")
            elif bad_range:
                if measurement.qualifier == "approximate":
                    add("approximate_range", "warning", measurement,
                        f"{column.name}: approximate nominal value {number} is outside physical limits; review the source"
                        if english else f"{column.name}: приблизительное номинальное значение {number} вне физических пределов; проверьте источник")
                else:
                    add("range", "error", measurement, f"{column.name}: value {number} is outside physical limits" if english
                        else f"{column.name}: значение {number} вне физических пределов")
        common_limits = intersect_intervals(limit_intervals) if limit_intervals else None
        censored_conflict = limit_intervals and (common_limits is None or any(
            intersect_intervals((common_limits, bounds)) is None for bounds in exact_intervals))
        if censored_conflict:
            peer_data = peer_payload(mixed_peers, include_units=True, include_limits=True)
            signature = digest(peer_data)
            message = ("The values and limits of the repeats have no common interval." if english
                       else "У значений и пределов повторов нет общей области.")
            for measurement in mixed_peers:
                add("repeat_censored", "warning", measurement, message, peer_data=peer_data,
                    peer_signature=signature)
        elif peers:
            first = peers[0]
            comparison = repeat_comparison(peers, numbers, unit, allowed[frame.at[first.id, "role"]])
            if comparison is None:
                continue
            peer_data = peer_payload(peers, include_units=True)
            peer_signature = digest(peer_data)
            if comparison == "interval":
                message = ("Declared uncertainty intervals of the repeats have no common overlap." if english
                           else "У заявленных интервалов неопределённости повторов нет общей части.")
            else:
                message = ("Repeat values differ. Comparable uncertainty is unavailable for the group." if english
                           else "Значения повторов различаются. Сопоставимая неопределённость для группы недоступна.")
            for measurement in peers:
                add("repeat", "warning", measurement, message, peer_data=peer_data, peer_signature=peer_signature)
                issues[-1]["repeat_comparison"] = comparison
    return issues
