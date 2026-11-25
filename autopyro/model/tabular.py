"""DataFrame snapshots for batch operations; the project remains the source of truth."""

import math

import pandas as pd

from .measurements import parse_number


def classify(value, *, numeric: bool):
    """Classify a cell without importing statistical calculations or SciPy."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return "missing", None
    if isinstance(value, str):
        text = value.strip()
        if text.casefold() in {"nd", "n.d.", "n/d", "н/о", "не обнаружено"}:
            return "non_detect", None
        if text.startswith(("<", ">", "≤", "≥")):
            return "censored", None
    if not numeric:
        return "other", None
    try:
        number = parse_number(value)
        if number is not None and math.isfinite(float(number)):
            return "numeric", number
    except (TypeError, ValueError, OverflowError):
        pass
    return "other", None


def project_frame(project, row_ids=None, column_ids=None, *, rows=None):
    """Copy cells in project order or in the order of supplied source rows."""
    if rows is None:
        selected = None if row_ids is None else set(row_ids)
        rows = [row for row in project.rows if selected is None or row.id in selected]
    return pd.DataFrame(
        [row.values for row in rows], index=[row.id for row in rows],
        columns=list(column_ids) if column_ids is not None else [c.id for c in project.columns],
        dtype=object,
    )


def classify_series(values, *, numeric, checkpoint=None):
    """Classify stored scalars in one pass and parse literal text as usual."""
    statuses, numbers = [], []
    for index, value in enumerate(values.to_numpy(dtype=object)):
        if checkpoint and index % 512 == 0:
            checkpoint()
        value_type = type(value)
        if value is None:
            status, number = "missing", None
        elif value_type is float:
            status, number = ("numeric", value) if numeric and math.isfinite(value) else ("other", None)
        elif value_type is int and -(2 ** 53) <= value <= 2 ** 53:
            status, number = ("numeric", value) if numeric else ("other", None)
        else:
            status, number = classify(value, numeric=numeric)
        statuses.append(status)
        numbers.append(number)
    return pd.DataFrame({'status': statuses, 'number': numbers}, index=values.index, dtype=object)


def depth_filter_masks(depth, bottom, lower, upper):
    """Return invalid/outside masks; sample intervals use [top, bottom)."""
    invalid = depth.isna()
    top = depth.where(depth.notna(), 0)
    outside = pd.Series(False, index=depth.index)
    if bottom is not None:
        end = bottom.where(bottom.notna(), 0)
        invalid |= bottom.isna() | end.le(top)
        if lower is not None and lower == upper:
            outside = ~(top.le(lower) & end.gt(lower))
        else:
            if lower is not None:
                outside |= end.le(lower)
            if upper is not None:
                outside |= top.ge(upper)
    else:
        if lower is not None:
            outside |= top.lt(lower)
        if upper is not None:
            outside |= top.gt(upper)
    return invalid, outside


def excluded_sample_ids(project, column_ids, scope):
    """Resolve explicit exclusions against the current working measurements."""
    excluded = set()
    for mid, decision in project.view.get("qc_exclusions", {}).get(scope, {}).items():
        measurement = project.science.measurements.get(mid)
        if (measurement is not None and measurement.property_id in column_ids
                and decision.get("revision") == measurement.revision
                and project.science.working.get(measurement.sample_id, {}).get(measurement.property_id) == mid):
            excluded.add(measurement.sample_id)
    return excluded


def exclusion_mask(project, index, column_id, scope):
    """Resolve only explicit exclusions, including their revision and working result."""
    mask = pd.Series(False, index=index, dtype=bool)
    excluded = [rid for rid in excluded_sample_ids(project, {column_id}, scope) if rid in index]
    if len(excluded) == 1:
        mask.loc[excluded[0]] = True
    elif excluded:
        mask.loc[excluded] = True
    return mask
