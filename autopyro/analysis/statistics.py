"""Descriptive statistics on an explicit row set, without changing measurements.

Definitions: linear quantiles, sample std(ddof=1), SciPy
Pearson correlation and correlation of average ranks for Spearman.
Censored/non-detect values are counted, never substituted.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from itertools import batched, combinations

import numpy as np
import pandas as pd
import scipy
from scipy.stats import NearConstantInputWarning, pearsonr, rankdata

from autopyro.model.audit import read_protocol
from autopyro.model.core import Project, timestamp
from autopyro import __version__
from autopyro.data_io.io import export_table_csv
from autopyro.model.measurements import parse_result
from autopyro.model.tabular import classify, project_frame, classify_series, exclusion_mask


@dataclass(frozen=True)
class StatisticsOptions:
    column_ids: tuple[str, ...]
    group_ids: tuple[str, ...] = ()
    text_numbers: bool = False
    include_interpolated: bool = False
    correlations: bool = True
    min_pairs: int = 3


def snapshot(project: Project, row_ids, column_ids):
    """Copy just requested cells for worker isolation; keep IDs and column semantics."""
    return next(snapshot_steps(project, row_ids, column_ids))


def snapshot_steps(project: Project, row_ids, column_ids, *, batch_size=None):
    """Yield checkpoints, followed by the complete isolated snapshot."""
    from copy import deepcopy
    from autopyro.model.core import Row
    from autopyro.model.measurements import snapshot_mapping, snapshot_record

    row_ids, column_ids = set(row_ids), set(column_ids)
    from .qc import selected_exclusions
    from autopyro.database.refresh import snapshot_status
    for step in project.science.subset_steps(row_ids, column_ids, batch_size=batch_size):
        if step is None:
            yield None
        else:
            science = step
    columns = []
    for column in project.columns:
        if column.id not in column_ids:
            continue
        metadata = dict(column.metadata)
        has_diagnostics = "diagnostics" in metadata
        diagnostics = metadata.pop("diagnostics", None)
        if isinstance(diagnostics, dict):
            diagnostics = {
                rid: item for rid, item in diagnostics.items() if rid in row_ids
            }
        copied = snapshot_record(replace(column, metadata=metadata))
        if has_diagnostics:
            if column.metadata.get("calculation") and isinstance(diagnostics, dict):
                from .calculations import snapshot_diagnostics
                copied.metadata["diagnostics"] = snapshot_diagnostics(diagnostics)
            else:
                copied.metadata["diagnostics"] = deepcopy(diagnostics)
        columns.append(copied)
    rows = []
    batches = batched(project.rows, batch_size) if batch_size else (project.rows,)
    for batch in batches:
        rows.extend(
            Row(id=row.id, values=snapshot_mapping({cid: row.values[cid] for cid in column_ids}),
                source=snapshot_mapping(row.source),
                original={cid: deepcopy(value) for cid, value in row.original.items() if cid in column_ids})
            for row in batch
            if row.id in row_ids
        )
        if batch_size and len(batch) == batch_size:
            yield None
    yield Project(
        id=project.id,
        source=snapshot_mapping(project.source),
        view={"qc_exclusions": selected_exclusions(project, science.measurements),
              "database_snapshot_status": snapshot_status(project, row_ids)},
        science=science,
        columns=columns,
        rows=rows,
    )


def classify_property(project, values, column, options):
    data = classify_series(values, numeric=column.kind == "number" or options.text_numbers)
    if project.view.get("qc_exclusions", {}).get("statistics"):
        excluded = exclusion_mask(project, values.index, column.id, "statistics")
        if excluded.any():
            data.loc[excluded, ["status", "number"]] = ["excluded_qc", None]
    if column.metadata.get("method") and (not options.include_interpolated or column.metadata.get("stale")):
        data.loc[data["status"].ne("missing"), ["status", "number"]] = ["excluded_interpolation", None]
    return data


def calculate_statistics(project: Project, row_ids, options: StatisticsOptions, *, check_cancelled=None, language="ru"):
    columns = {c.id: c for c in project.columns}
    requested = options.column_ids + options.group_ids
    if not options.column_ids or any(cid not in columns for cid in requested):
        raise ValueError("Выберите существующие показатели")
    if len(set(options.column_ids)) != len(options.column_ids) or len(
        set(options.group_ids)
    ) != len(options.group_ids):
        raise ValueError("Повторяющиеся колонки в настройках")
    if options.min_pairs < 3:
        raise ValueError("Для корреляции требуется не менее трёх пар")
    if options.correlations and len(options.column_ids) > 30:
        raise ValueError(
            "Для корреляций выберите не более 30 показателей или отключите корреляции"
        )
    row_ids = set(row_ids)
    frame = project_frame(project, row_ids, dict.fromkeys(requested))
    frame.index = frame.index.astype(object)
    if row_ids.difference(frame.index.to_numpy()):
        raise ValueError("Выборка содержит отсутствующие пробы")
    frame_ids = frame.index.to_numpy()
    if options.group_ids:
        group_values = frame[list(options.group_ids)]
        group_labels = group_values.to_numpy()
        grouped = group_values.groupby(list(options.group_ids), sort=False, dropna=False, observed=True)
        groups = sorted(grouped.indices.values(), key=lambda positions: positions[0])
    else:
        groups = [None] if not frame.empty else []
    notices = []
    summary, availability, correlations, outliers, group_row_ids = [], [], [], [], []
    numerical_warnings = 0
    for cid in options.column_ids:
        column = columns[cid]
        if column.metadata.get("method"):
            if column.metadata.get("stale"):
                notices.append(f"Stale interpolation excluded for {column.name}" if language == "en"
                               else f"{column.name}: устаревшая интерполяция исключена")
            elif not options.include_interpolated:
                notices.append(f"Interpolation excluded by settings for {column.name}" if language == "en"
                               else f"{column.name}: интерполяция исключена настройками")
    classified_columns = {}
    for cid in options.column_ids:
        if check_cancelled is not None:
            check_cancelled()
        column = columns[cid]
        classified = classify_property(project, frame[cid], column, options)
        numbers = classified["number"]
        default_context = ((column.unit or "").strip(), (column.metadata.get("basis") or "").strip())
        contexts = {}
        if project.science.working:
            for index, rid in enumerate(numbers.dropna().index):
                if check_cancelled is not None and index % 1024 == 0:
                    check_cancelled()
                measurement = project.science.current(rid, cid)
                if measurement is not None:
                    context = ((measurement.unit or default_context[0]).strip(),
                               (measurement.basis or default_context[1]).strip())
                    if context != default_context:
                        contexts[rid] = context
        classified_columns[cid] = classified["status"].to_numpy(), numbers.to_numpy(), default_context, contexts
    for positions in groups:
        if positions is None:
            group, member_ids = [], frame_ids
        else:
            # Group keys normalize None to NaN. Read the original values.
            group = group_labels[positions[0]].tolist()
            member_ids = frame_ids[positions]
        row_count = len(member_ids)
        group_row_ids.append(member_ids.tolist())
        correlation_columns = {}
        for cid in options.column_ids:
            if check_cancelled is not None:
                check_cancelled()
            column = columns[cid]
            is_interpolated = bool(column.metadata.get("method"))
            statuses, values, default_context, overrides = classified_columns[cid]
            if positions is not None:
                statuses, values = statuses[positions], values[positions]
            counts = Counter(statuses)
            numeric_mask = statuses == "numeric"
            numbers = values[numeric_mask]
            contexts = ({overrides.get(rid, default_context) for rid in member_ids[numeric_mask]}
                        if overrides else {default_context})
            incompatible = len(contexts) > 1
            excluded_context = len(numbers) if incompatible else 0
            unit, basis = next(iter(contexts), default_context) if not incompatible else ("", "")
            if incompatible:
                numbers = numbers[:0]
            if options.correlations:
                numeric_values = np.full(row_count, np.nan) if incompatible else values.astype(float)
                correlation_columns[cid] = numeric_values, np.isfinite(numeric_values)
            base = {
                "group": group,
                "column_id": cid,
                "property": column.name,
                "unit": unit,
                "basis": basis,
                "rows": row_count,
                "n": len(numbers),
                "missing": counts["missing"],
                "censored": counts["censored"],
                "non_detect": counts["non_detect"],
                "other": counts["other"],
                "excluded_interpolation": counts["excluded_interpolation"],
                "excluded_qc": counts["excluded_qc"],
                "excluded_context": excluded_context,
                "interpolated_used": len(numbers) if is_interpolated else 0,
                "coverage_pct": 100.0 * len(numbers) / row_count,
            }
            availability.append(base)
            stats = {
                **base,
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None,
                "p10": None,
                "q1": None,
                "q3": None,
                "p90": None,
                "iqr": None,
                "outliers": None,
                "summary_status": "incompatible_context" if incompatible else None,
                "numeric_issues": [],
            }
            if len(numbers) == 1:
                only = numbers[0]
                value = float(only)
                stats.update(mean=value, median=value, min=only, max=only,
                             p10=value, q1=value, q3=value, p90=value, iqr=0.0)
            elif len(numbers):
                values = numbers.astype(float)
                probabilities = [0.1, 0.25, 0.5, 0.75, 0.9]
                stats.update(min=numbers.min(), max=numbers.max())
                with np.errstate(over="ignore", invalid="ignore"):
                    quantiles = np.quantile(values, probabilities, method="linear")
                    mean = float(values.mean())
                    std = float(values.std(ddof=1))
                    if not math.isfinite(mean) or not math.isfinite(std) or not np.isfinite(quantiles).all():
                        scale = max(abs(float(stats["min"])), abs(float(stats["max"])))
                        scaled = values / scale
                        if not math.isfinite(mean):
                            mean = float(scaled.mean()) * scale
                        if not math.isfinite(std):
                            std = float(scaled.std(ddof=1)) * scale
                        if not np.isfinite(quantiles).all():
                            quantiles = np.quantile(scaled, probabilities, method="linear") * scale
                q10, q1, median, q3, q90 = map(float, quantiles)
                iqr = q3 - q1
                metrics = dict(mean=mean, median=median, std=std, p10=q10, q1=q1, q3=q3, p90=q90, iqr=iqr)
                for field, value in metrics.items():
                    if value is not None and not math.isfinite(value):
                        stats["numeric_issues"].append(field)
                    else:
                        stats[field] = value
                if len(numbers) >= 4:
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    if math.isfinite(lower) and math.isfinite(upper):
                        outside = (numbers < lower) | (numbers > upper)
                        stats["outliers"] = int(outside.sum())
                        outliers.extend(
                            {"group": group, "row_id": rid, "column_id": cid,
                             "property": column.name, "unit": unit,
                             "value": value, "lower": lower, "upper": upper}
                            for rid, value in zip(member_ids[numeric_mask][outside], numbers[outside], strict=True)
                        )
                    else:
                        stats["numeric_issues"].append("outlier_limits")
                if stats["numeric_issues"]:
                    stats["summary_status"] = "numerical_warning"
                    numerical_warnings += 1
            summary.append(stats)
        if not options.correlations:
            continue
        complete_ranks = {}
        for x_id, y_id in combinations(options.column_ids, 2):
            if check_cancelled is not None:
                check_cancelled()
            x_values, x_present = correlation_columns[x_id]
            y_values, y_present = correlation_columns[y_id]
            paired = x_present & y_present
            n_pairs = int(paired.sum())
            result = {
                "group": group,
                "x_id": x_id,
                "y_id": y_id,
                "x": columns[x_id].name,
                "y": columns[y_id].name,
                "n_pairs": n_pairs,
                "pearson": None,
                "spearman": None,
                "status": "ok",
            }
            if n_pairs < options.min_pairs:
                result["status"] = "insufficient_pairs"
            else:
                complete_pair = n_pairs == row_count
                x, y = (x_values, y_values) if complete_pair else (x_values[paired], y_values[paired])
                if x.min() == x.max() or y.min() == y.max():
                    result["status"] = "constant"
                else:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", NearConstantInputWarning)
                        result["pearson"] = float(pearsonr(x, y).statistic)
                        # Ranks depend on the paired sample. Only pairs that
                        # use every row in the group share cached ranks.
                        if complete_pair:
                            for cid, values in ((x_id, x), (y_id, y)):
                                if cid not in complete_ranks:
                                    complete_ranks[cid] = rankdata(values)
                            ranks = np.column_stack((complete_ranks[x_id], complete_ranks[y_id]))
                        else:
                            ranks = rankdata(np.column_stack((x, y)), axis=0)
                        result["spearman"] = float(np.corrcoef(ranks, rowvar=False)[0, 1])
                    if caught:
                        result["status"] = "numerical_warning"
                    for name in ("pearson", "spearman"):
                        if not math.isfinite(result[name]):
                            result[name] = None
                            result["status"] = "numerical_warning"
            correlations.append(result)
    return {
        "created_at": timestamp(),
        "project_id": project.id,
        "row_ids": sorted(row_ids),
        "group_row_ids": group_row_ids,
        "options": asdict(options),
        "groups": [columns[cid].name for cid in options.group_ids],
        "versions": {
            "autopyro": __version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "notices": notices,
        "numerical_warning_count": numerical_warnings,
        "summary": summary,
        "availability": availability,
        "correlations": correlations,
        "outliers": outliers,
        "methods": {
            "std": "sample, ddof=1; undefined at n<2",
            "quantiles": "linear",
            "correlations": "pairwise complete observations; Pearson and Spearman; no p-values",
            "censoring": "excluded and counted; no substitution or imputation",
            "outliers": "outside Q1-1.5*IQR .. Q3+1.5*IQR, n>=4; no deletion",
        },
    }


_CONTEXT_FIELDS = ("unit", "basis", "method", "laboratory", "instrument",
                   "heating_program", "preparation", "fraction")
_CONTEXT_LABELS = dict(zip(_CONTEXT_FIELDS, (
    "Единица", "Основание измерения", "Метод", "Лаборатория", "Прибор",
    "Программа нагрева", "Подготовка", "Фракция",
), strict=True))


def _measurement_context(science, row_id, column):
    measurement = science.current(row_id, column.id)
    analysis = science.analyses.get(measurement.analysis_id) if measurement else None
    fields = (getattr(measurement, "unit", "") or column.unit, getattr(measurement, "basis", ""),
              *(getattr(analysis, name, "") for name in _CONTEXT_FIELDS[2:]))
    return tuple((value or "").strip() for value in fields)


def compare_statistics(project, a_ids, b_ids, options, *, check_cancelled=None, language="ru"):
    columns = {column.id: column for column in project.columns}
    frame = project_frame(project, set(a_ids) | set(b_ids), options.column_ids)
    comparison = []
    for cid in options.column_ids:
        if check_cancelled is not None:
            check_cancelled()
        column = columns[cid]
        data = classify_property(project, frame[cid], column, options)
        entry = {"property": column.name, "unit": column.unit, "column_id": cid,
                 "group": [], "samples": {}}
        contexts, sides = set(), {}
        for side, ids in (("a", a_ids), ("b", b_ids)):
            selected = data.loc[list(ids)]
            numbers = selected["number"].dropna().astype(float)
            side_contexts = {_measurement_context(project.science, rid, column) for rid in numbers.index}
            sides[side] = side_contexts
            contexts.update(side_contexts)
            median = float(numbers.median()) if len(side_contexts) == 1 else None
            entry.update({f"{side}_used": len(numbers), f"{side}_excluded": len(ids) - len(numbers),
                          f"{side}_unit": " / ".join(sorted({context[0] for context in side_contexts})),
                          f"{side}_median": median if median is not None and math.isfinite(median) else None})
            entry["samples"][side] = list(selected["status"].items())
        a, b = entry["a_median"], entry["b_median"]
        missing, mismatched = [], []
        for field, values in zip(_CONTEXT_FIELDS, zip(*contexts)):
            known = set(values) - {""}
            if len(known) > 1:
                mismatched.append(field)
            if "" in values and (field == "method" or known):
                missing.append(field)
        delta = b - a if a is not None and b is not None and not (missing or mismatched) else None
        entry["delta"] = delta if delta is not None and math.isfinite(delta) else None
        entry["context_missing"] = missing
        entry["context_mismatched"] = mismatched
        entry["context_details"] = {
            field: {side: sorted({context[i] for context in values}) for side, values in sides.items()}
            for i, field in enumerate(_CONTEXT_FIELDS) if field in missing or field in mismatched
        }
        entry["comparison_status"] = (
            "Недостаточно числовых значений" if not entry["a_used"] or not entry["b_used"]
            else "Разный аналитический контекст" if mismatched
            else "Аналитический контекст неизвестен" if missing
            else "Численная неустойчивость" if entry["delta"] is None
            else "OK"
        )
        descriptions = []
        for field, by_side in entry["context_details"].items():
            label = field.replace("_", " ").capitalize() if language == "en" else _CONTEXT_LABELS[field]
            left, right = (", ".join(value or "—" for value in by_side[side]) or "—" for side in ("a", "b"))
            descriptions.append(f"{label}: A = {left} · B = {right}")
        entry["context_explanation"] = "\n".join(descriptions)
        units = {context[0] for context in contexts}
        if units:
            entry["unit"] = " / ".join(sorted(units))
        comparison.append(entry)
    return comparison


def _median(values):
    ordered = sorted(values)
    upper = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[upper]
    left, right = ordered[upper - 1], ordered[upper]
    return (left + right) / 2 if left < 0 < right else left + (right - left) / 2


def repeat_statistics(project, report, *, check_cancelled=None):
    """Summarize independent analyses within each sample and analytical context."""
    options = report["options"]
    columns = {c.id: c for c in project.columns if c.id in options["column_ids"]}
    group_members = {rid: group for group, ids in enumerate(report["group_row_ids"]) for rid in ids}
    exclusions = project.view.get("qc_exclusions", {}).get("statistics", {})
    science = project.science
    by_sample = {}
    for index, measurement in enumerate(science.measurements.values()):
        if check_cancelled is not None and index % 1024 == 0:
            check_cancelled()
        rid, cid = measurement.sample_id, measurement.property_id
        column = columns.get(cid)
        if column is None or rid not in group_members or measurement.qualifier != "exact":
            continue
        if exclusions.get(measurement.id, {}).get("revision") == measurement.revision:
            continue
        analysis = science.analyses[measurement.analysis_id]
        if not analysis.method:
            continue
        _, number = classify(measurement.table_value(), numeric=column.kind == "number" or options["text_numbers"])
        if number is None:
            continue
        context = (measurement.unit or column.unit, measurement.basis or "",
                   analysis.heating_program or "", analysis.preparation or "", analysis.fraction or "")
        channel = (analysis.method, analysis.laboratory or "", analysis.instrument or "")
        channels = by_sample.setdefault((rid, cid, context), {})
        channels.setdefault(channel, {}).setdefault(analysis.id, []).append(float(number))

    summaries = defaultdict(list)
    for (rid, cid, context), channels in by_sample.items():
        if check_cancelled is not None:
            check_cancelled()
        group = group_members[rid]
        centers = {}
        for channel, analyses in channels.items():
            numbers = [_median(values) for values in analyses.values()]
            center = _median(numbers)
            centers[channel] = center, len(numbers)
            if len(numbers) > 1:
                spread = max(numbers) - min(numbers)
                if math.isfinite(spread):
                    summaries[group, cid, context, channel, None].append(
                        dict(row_id=rid, a=center, b=None, a_count=len(numbers), b_count=None, difference=spread)
                    )
        for a, b in combinations(sorted(centers), 2):
            if a[:2] == b[:2] or bool(a[1]) != bool(b[1]):
                continue
            a_value, a_count = centers[a]
            b_value, b_count = centers[b]
            difference = b_value - a_value
            if math.isfinite(difference):
                summaries[group, cid, context, a, b].append(
                    dict(row_id=rid, a=a_value, b=b_value, a_count=a_count, b_count=b_count, difference=difference)
                )

    results = []
    column_count = len(options["column_ids"])
    for (group, cid, context, a, b), samples in summaries.items():
        unit, basis, heating, preparation, fraction = context
        differences = pd.Series([sample["difference"] for sample in samples], dtype=float)
        p10, center, p90 = differences.quantile([0.1, 0.5, 0.9], interpolation="linear")
        results.append({
            "group": report["summary"][group * column_count]["group"],
            "column_id": cid, "property": columns[cid].name, "unit": unit, "basis": basis,
            "heating": heating, "preparation": preparation, "fraction": fraction,
            "method_a": a[0], "lab_a": a[1], "instrument_a": a[2],
            "method_b": b[0] if b else None, "lab_b": b[1] if b else None, "instrument_b": b[2] if b else None,
            "kind": "Парные разности" if b else "Размах повторов",
            "n": len(samples), "median": float(center) if math.isfinite(center) else None,
            "p10": float(p10) if math.isfinite(p10) else None,
            "p90": float(p90) if math.isfinite(p90) else None, "samples": samples,
        })
    return results


def threshold_diagnostics(project, row_ids, options, *, group_row_ids=(), check_cancelled=None):
    """Describe literal qualifiers on working results, including QC-excluded records."""
    science = project.science
    columns = {column.id: column for column in project.columns}
    exclusions = project.view.get("qc_exclusions", {}).get("statistics", {})
    selected = set(row_ids)
    rows = {row.id: row for row in project.rows if row.id in selected}
    scopes = [(None, rows)]
    if options.group_ids:
        scopes.extend(enumerate(group_row_ids))
    diagnostics = []
    for cid in options.column_ids:
        column = columns[cid]
        groups = {}
        scoped_rows = ((group, rows[rid]) for group, ids in scopes for rid in ids)
        for index, (group, row) in enumerate(scoped_rows):
            if check_cancelled is not None and index % 1024 == 0:
                check_cancelled()
            rid, value = row.id, row.values[cid]
            status, _ = classify(value, numeric=False)
            if status == "missing":
                continue
            measurement = science.current(rid, cid)
            analysis = science.analyses.get(measurement.analysis_id) if measurement else None
            method, lab = (analysis.method or "", analysis.laboratory or "") if analysis else ("", "")
            unit = measurement.unit if measurement else column.unit
            basis = measurement.basis if measurement else column.metadata.get("basis", "")
            key = group, method, lab, unit, basis
            if key not in groups:
                groups[key] = {
                    "group": [], "group_index": group, "column_id": cid, "property": column.name, "method": method, "lab": lab,
                    "unit": unit, "basis": basis, "n": 0, "below": 0, "above": 0, "non_detect": 0,
                    "below_min": None, "below_max": None, "above_min": None, "above_max": None,
                    "no_numeric_limit": 0, "excluded_qc": 0, "samples": [],
                    "other_row_ids": [], "other_qc_row_ids": [],
                }
            entry = groups[key]
            entry["n"] += 1
            excluded = measurement is not None and exclusions.get(measurement.id, {}).get("revision") == measurement.revision
            entry["excluded_qc"] += excluded
            if status not in ("censored", "non_detect"):
                entry["other_row_ids"].append(rid)
                if excluded:
                    entry["other_qc_row_ids"].append(rid)
                continue
            limit = None
            if status == "censored":
                status = "below" if value.lstrip().startswith(("<", "≤")) else "above"
                limit = parse_result(value, "text")["limit"]
                if limit is None:
                    entry["no_numeric_limit"] += 1
                else:
                    low, high = f"{status}_min", f"{status}_max"
                    entry[low] = limit if entry[low] is None else min(limit, entry[low])
                    entry[high] = limit if entry[high] is None else max(limit, entry[high])
            entry[status] += 1
            entry["samples"].append({
                "row_id": rid, "value": value, "status": status, "limit": limit,
                "lod": measurement.lod if measurement else None,
                "loq": measurement.loq if measurement else None, "excluded_qc": excluded,
            })
        for entry in groups.values():
            for status in ("below", "above", "non_detect"):
                entry[f"{status}_pct"] = 100 * entry[status] / entry["n"]
            diagnostics.append(entry)
    return diagnostics


def interpolation_pairs(project, column_ids):
    selected = set(column_ids)
    sources = {column.id for column in project.columns if not column.metadata.get("method")}
    return tuple(
        (column.metadata["y_id"], column.id)
        for column in project.columns
        if column.metadata.get("method") and column.metadata.get("y_id") in sources
        and (column.id in selected or column.metadata["y_id"] in selected)
    )


def interpolation_comparison(project, report, pairs, options, *, check_cancelled=None):
    """Compare the same scope before and after filling only blank source cells."""
    if not pairs:
        return []
    columns = {column.id: column for column in project.columns}
    frame = project_frame(project, report["row_ids"], dict.fromkeys(cid for pair in pairs for cid in pair))
    options = replace(options, include_interpolated=True)
    classified = {cid: classify_property(project, frame[cid], columns[cid], options) for cid in frame}
    provenance = {}
    for _, estimate_id in pairs:
        if check_cancelled is not None:
            check_cancelled()
        metadata = columns[estimate_id].metadata
        protocol = metadata.get("interpolation_protocol")
        if isinstance(protocol, str):
            try:
                protocol = read_protocol(protocol)
            except ValueError:
                protocol = None
        protocol = protocol if isinstance(protocol, dict) else {}
        eligibility = protocol.get("eligibility")
        records = protocol.get("proposals", metadata.get("proposals"))
        proposals = {}
        if isinstance(records, list):
            proposals = {record["row_id"]: record for record in records
                         if isinstance(record, dict) and isinstance(record.get("row_id"), str)}
        scope_ids = metadata.get("scope_row_ids")
        scope = {rid for rid in scope_ids if isinstance(rid, str)} if isinstance(scope_ids, list) else None
        provenance[estimate_id] = eligibility if isinstance(eligibility, dict) else {}, proposals, scope
    results = []
    for group_index, ids in enumerate(report["group_row_ids"]):
        group = report["summary"][group_index * len(options.column_ids)]["group"]
        for source_id, estimate_id in pairs:
            if check_cancelled is not None:
                check_cancelled()
            source, estimate = columns[source_id], columns[estimate_id]
            eligibility, archived_proposals, scope = provenance[estimate_id]
            originals, proposals = classified[source_id].loc[ids], classified[estimate_id].loc[ids]
            measured_numbers = originals["number"].dropna().astype(float)
            measured = measured_numbers.tolist()
            added, samples = [], []
            sample_provenance = {}
            counts = dict(stale=0, unit_mismatch=0, invalid=0, source_numeric=0, source_censored=0,
                          source_non_detect=0, source_other=0, source_excluded_qc=0, no_estimate=0)
            samples_to_compare = zip(ids, originals["status"], proposals["status"], proposals["number"], strict=True)
            for index, (rid, original_status, proposal_status, proposal) in enumerate(samples_to_compare):
                if check_cancelled is not None and index % 1024 == 0:
                    check_cancelled()
                if proposal_status == "missing":
                    if original_status != "missing":
                        continue
                    status = "no_estimate"
                elif estimate.metadata.get("stale"):
                    status = "stale"
                elif source.unit != estimate.unit:
                    status = "unit_mismatch"
                elif proposal_status != "numeric":
                    status = "invalid"
                elif original_status != "missing":
                    status = f"source_{original_status}"
                else:
                    status = "added"
                    added.append(float(proposal))
                if status != "added":
                    counts[status] += 1
                samples.append([rid, frame.at[rid, source_id], frame.at[rid, estimate_id], status])
                proposal_record = archived_proposals.get(rid)
                diagnostic = eligibility.get(rid)
                if proposal_record:
                    role = "extrapolated" if proposal_record.get("extrapolated") else "proposed"
                    sample_provenance[rid] = (role, None, proposal_record.get("left_id"),
                                              proposal_record.get("right_id"))
                elif (isinstance(diagnostic, dict) and isinstance(diagnostic.get("status"), str)
                      and diagnostic["status"] != "eligible"):
                    reason = diagnostic.get("reason")
                    sample_provenance[rid] = (diagnostic["status"], reason if isinstance(reason, str) else None,
                                              None, None)
                elif scope is not None and rid not in scope:
                    sample_provenance[rid] = ("outside_scope", None, None, None)
            augmented = measured + added
            before = _median(measured) if measured else None
            after = _median(augmented) if augmented else None
            delta = after - before if before is not None and after is not None else None
            results.append({
                "group": group, "group_index": group_index, "column_id": source_id, "property": source.name,
                "estimate_id": estimate_id, "estimate": estimate.name,
                "unit": source.unit, "rows": len(ids), "measured": len(measured), "added": len(added),
                "before_pct": 100 * len(measured) / len(ids), "after_pct": 100 * len(augmented) / len(ids),
                "before_median": before, "estimate_median": _median(added) if added else None,
                "after_median": after, "delta": delta if delta is not None and math.isfinite(delta) else None,
                "samples": samples, "measured_ids": measured_numbers.index.tolist(),
                "sample_provenance": sample_provenance, **counts,
            })
    return results


TABLES = {
    "summary": [
        ("property", "Показатель"),
        ("unit", "Единица"),
        ("rows", "Всего"),
        ("n", "N чисел"),
        ("mean", "Среднее"),
        ("median", "Медиана"),
        ("std", "SD (n−1)"),
        ("min", "Минимум"),
        ("max", "Максимум"),
        ("p10", "P10"),
        ("q1", "Q1"),
        ("q3", "Q3"),
        ("p90", "P90"),
        ("iqr", "IQR"),
        ("outliers", "Выбросы (IQR)"),
        ("summary_status", "Статус"),
    ],
    "availability": [
        ("property", "Показатель"),
        ("unit", "Единица"),
        ("rows", "Всего"),
        ("n", "N чисел"),
        ("missing", "Пусто"),
        ("censored", "< / >"),
        ("non_detect", "ND"),
        ("other", "Прочие / текст"),
        ("excluded_interpolation", "Искл. интерполяции"),
        ("excluded_qc", "Исключено QC"),
        ("excluded_context", "Искл. несовместимый контекст"),
        ("interpolated_used", "Интерпол. использовано"),
        ("coverage_pct", "Заполненность, %"),
    ],
    "correlations": [
        ("x", "X"),
        ("y", "Y"),
        ("n_pairs", "N пар"),
        ("pearson", "Pearson r"),
        ("spearman", "Spearman ρ"),
        ("status", "Статус"),
    ],
    "outliers": [
        ("row_id", "UUID пробы"),
        ("property", "Показатель"),
        ("unit", "Единица"),
        ("value", "Значение"),
        ("lower", "Нижняя граница"),
        ("upper", "Верхняя граница"),
    ],
    "comparison": [
        ("property", "Показатель"),
        ("a_median", "Медиана A"),
        ("b_median", "Медиана B"),
        ("delta", "Медиана B − A"),
        ("comparison_status", "Статус"),
        ("a_unit", "Единица A"),
        ("b_unit", "Единица B"),
        ("a_used", "A · использовано"),
        ("a_excluded", "A · исключено"),
        ("b_used", "B · использовано"),
        ("b_excluded", "B · исключено"),
        ("context_explanation", "Аналитический контекст"),
    ],
    "repeats": [
        ("property", "Показатель"),
        ("kind", "Вид сравнения"),
        ("n", "Число проб"),
        ("median", "Медиана"),
        ("p10", "P10"),
        ("p90", "P90"),
        ("unit", "Единица"),
        ("method_a", "Метод A"),
        ("lab_a", "Лаборатория A"),
        ("instrument_a", "Прибор A"),
        ("method_b", "Метод B"),
        ("lab_b", "Лаборатория B"),
        ("instrument_b", "Прибор B"),
        ("basis", "Основание измерения"),
        ("heating", "Программа нагрева"),
        ("preparation", "Подготовка"),
        ("fraction", "Фракция"),
    ],
    "thresholds": [
        ("property", "Показатель"),
        ("method", "Метод"),
        ("lab", "Лаборатория"),
        ("unit", "Единица"),
        ("basis", "Основание измерения"),
        ("n", "Непустых записей"),
        ("below", "Ниже порога"),
        ("below_pct", "Ниже порога, %"),
        ("below_min", "Мин. порог < / ≤"),
        ("below_max", "Макс. порог < / ≤"),
        ("above", "Выше порога"),
        ("above_pct", "Выше порога, %"),
        ("above_min", "Мин. порог > / ≥"),
        ("above_max", "Макс. порог > / ≥"),
        ("non_detect", "ND"),
        ("non_detect_pct", "ND, %"),
        ("no_numeric_limit", "Порог без числа"),
        ("excluded_qc", "Исключено QC"),
    ],
    "interpolation": [
        ("property", "Показатель"),
        ("estimate", "Колонка интерполяции"),
        ("unit", "Единица"),
        ("rows", "Всего"),
        ("measured", "Исходных чисел"),
        ("added", "Добавлено оценок"),
        ("before_pct", "До, %"),
        ("after_pct", "После, %"),
        ("before_median", "Исходная медиана"),
        ("estimate_median", "Медиана добавленных оценок"),
        ("after_median", "Медиана после заполнения"),
        ("delta", "Изменение медианы"),
        ("no_estimate", "Пусто без оценки"),
        ("stale", "Устаревшие оценки"),
        ("unit_mismatch", "Разные единицы"),
        ("invalid", "Недоступные оценки"),
        ("source_numeric", "Есть исходное число"),
        ("source_censored", "Есть исходный порог"),
        ("source_non_detect", "Исходное ND"),
        ("source_other", "Прочая исходная запись"),
        ("source_excluded_qc", "Исходное исключено QC"),
    ],
}


_ENGLISH_REPORT_HEADERS = {
    "a_median": "Median A",
    "b_median": "Median B",
    "delta": "Median change",
    "comparison_status": "Status",
    "a_unit": "Unit A",
    "b_unit": "Unit B",
    "a_used": "A · used",
    "a_excluded": "A · excluded",
    "b_used": "B · used",
    "b_excluded": "B · excluded",
    "context_explanation": "Analytical context",
    "kind": "Comparison type",
    "method_a": "Method A",
    "lab_a": "Laboratory A",
    "instrument_a": "Instrument A",
    "method_b": "Method B",
    "lab_b": "Laboratory B",
    "instrument_b": "Instrument B",
    "basis": "Measurement basis",
    "heating": "Heating program",
    "preparation": "Preparation",
    "fraction": "Fraction",
    "method": "Method",
    "lab": "Laboratory",
    "below": "Below threshold",
    "below_pct": "Below threshold, %",
    "below_min": "Min. threshold < / ≤",
    "below_max": "Max. threshold < / ≤",
    "above": "Above threshold",
    "above_pct": "Above threshold, %",
    "above_min": "Min. threshold > / ≥",
    "above_max": "Max. threshold > / ≥",
    "no_numeric_limit": "Threshold without a number",
    "estimate": "Interpolation column",
    "measured": "Source numeric values",
    "added": "Estimates added",
    "before_pct": "Before, %",
    "after_pct": "After, %",
    "before_median": "Source median",
    "estimate_median": "Median of added estimates",
    "after_median": "Median after filling",
    "no_estimate": "Blank with no estimate",
    "stale": "Stale estimates",
    "unit_mismatch": "Different units",
    "invalid": "Unavailable estimates",
    "source_numeric": "Numeric source value present",
    "source_censored": "Censored source value present",
    "source_non_detect": "Source value is ND",
    "source_other": "Other source entry",
    "source_excluded_qc": "Source excluded by QC",
    "property": "Indicator", "unit": "Unit", "rows": "Total", "n": "Numeric N",
    "mean": "Mean", "median": "Median", "min": "Minimum", "max": "Maximum",
    "outliers": "IQR outliers", "summary_status": "Status", "status": "Status",
    "missing": "Missing", "other": "Other / text", "excluded_interpolation": "Interpolation excluded",
    "excluded_qc": "Excluded by QC", "interpolated_used": "Interpolated values used",
    "excluded_context": "Incompatible context excluded",
    "coverage_pct": "Coverage, %", "n_pairs": "Pairs", "row_id": "Sample UUID",
    "value": "Value", "lower": "Lower bound", "upper": "Upper bound",
}

_ENGLISH_REPORT_STATUSES = {
    "ok": "OK", "constant": "Constant indicator", "insufficient_pairs": "Insufficient pairs",
    "numerical_warning": "Numerical warning", "incompatible_context": "Incompatible units or measurement bases",
}


def report_table(report, table_name, *, language="ru"):
    spec = TABLES[table_name]
    groups = [] if table_name in ("comparison", "thresholds") else report["groups"]
    english = language == "en"
    prefix = "Group" if english else "Группа"
    headers = [f"{prefix}: {name}" for name in groups] + [
        _ENGLISH_REPORT_HEADERS.get(key, label) if english else label for key, label in spec
    ]
    rows = [
        entry["group"] + [entry.get(key) if key == "summary_status" else entry[key] for key, _ in spec]
        for entry in report.get(table_name, [])
    ]
    if english:
        for offset, (key, _) in enumerate(spec, len(groups)):
            if key in ("summary_status", "status"):
                for row in rows:
                    row[offset] = _ENGLISH_REPORT_STATUSES.get(row[offset], row[offset])
    return headers, rows


def export_statistics_csv(path, report, table_name, *, row_order=None, group_index=None, language="ru"):
    """Include policy and row count in each record so a CSV remains self-describing."""
    headers, rows = report_table(report, table_name, language=language)
    context = [report["created_at"], report.get("scope", ""), len(report["row_ids"]),
               report["options"]["text_numbers"], report["options"]["include_interpolated"]]
    context_headers = ["Расчёт UTC", "Область", "Строк в области", "Числа в тексте", "Интерполяции"]
    if table_name == "comparison":
        comparison = report["comparison_context"]
        context_headers = ["Расчёт UTC", "Представление A", "Представление B", "Общие пробы",
                           "Числа в тексте", "Интерполяции"]
        context[1:3] = [comparison["a"], comparison["b"], comparison["overlap"]]
        if any(comparison.get(key, "").startswith("set:") for key in ("a_scope", "b_scope")):
            context_headers[1:3] = ["Выборка A", "Выборка B", "Идентификатор A", "Идентификатор B"]
            context[1:3] = [comparison["a"], comparison["b"], comparison["a_scope"], comparison["b_scope"]]
    if table_name == "thresholds" and group_index is not None:
        context[2] = len(report["group_row_ids"][group_index])
        count = len(report["options"]["column_ids"])
        group = report["summary"][group_index * count]["group"]
        context_headers += [f"Группа: {field}" for field in report["groups"]]
        context += group
    if language == "en":
        labels = {"Расчёт UTC": "Calculated UTC", "Область": "Scope", "Строк в области": "Rows in scope",
                  "Числа в тексте": "Numbers in text", "Интерполяции": "Interpolations",
                  "Представление A": "View A", "Представление B": "View B", "Общие пробы": "Shared samples",
                  "Выборка A": "Selection A", "Выборка B": "Selection B",
                  "Идентификатор A": "ID A", "Идентификатор B": "ID B"}
        context_headers = [labels.get(label, label.replace("Группа: ", "Group: ", 1)) for label in context_headers]
    if row_order is not None:
        rows = [rows[index] for index in row_order]
    elif table_name == "thresholds":
        rows = [row for row, entry in zip(rows, report[table_name], strict=True) if entry["group_index"] == group_index]
    from autopyro.data_io.file_output import atomic_output
    with atomic_output(path) as temporary:
        export_table_csv(temporary, context_headers + headers, (context + values for values in rows))
