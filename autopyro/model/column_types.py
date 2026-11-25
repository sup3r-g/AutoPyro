"""Prepare atomic column conversions, including non-working repeat results."""

from copy import deepcopy

from .field_types import FIELD_KINDS, format_duration
from .measurements import derived, now, parse_result, parsed_table_value
from .snapshots import copy_project


def converted_result(value, source_kind, target_kind):
    if source_kind == "duration" and target_kind in ("text", "category", "identifier") and value is not None:
        value = format_duration(value)
    result = parse_result(value, target_kind)
    if result["qualifier"] == "unparsed":
        raise ValueError("Значение не соответствует выбранному типу")
    return result


def prepare_column_conversion(project, column_id, target_kind):
    if target_kind not in FIELD_KINDS:
        raise ValueError("Неизвестный тип колонки")
    before = copy_project(project)
    column = next(column for column in before.columns if column.id == column_id)
    if derived(column):
        raise ValueError("Тип вычисляемой колонки задаётся её расчётом")
    after = copy_project(before)
    target = next(column for column in after.columns if column.id == column_id)
    preview, errors = [], []
    working = {selected.get(column_id) for selected in before.science.working.values()}
    for position, row in enumerate(after.rows):
        value = row.values[column_id]
        try:
            result = converted_result(value, column.kind, target_kind)
            row.values[column_id] = parsed_table_value(result)
            preview.append((position + 1, value, row.values[column_id], ""))
        except (ValueError, TypeError, OverflowError) as exc:
            item = (position + 1, value, None, str(exc))
            preview.append(item)
            errors.append(item)
    positions = {row.id: position + 1 for position, row in enumerate(after.rows)}
    for measurement in after.science.measurements.values():
        if measurement.property_id != column_id:
            continue
        try:
            result = converted_result(measurement.table_value(), column.kind, target_kind)
        except (ValueError, TypeError, OverflowError) as exc:
            # Working values already appear in the row preview.
            if measurement.id not in working:
                errors.append((f"{positions.get(measurement.sample_id, '—')} ({measurement.id})",
                               measurement.table_value(), None, str(exc)))
            continue
        previous = {key: deepcopy(value) for key, value in vars(measurement).items() if key != "history"}
        measurement.history.append({"at": now(), "previous": previous})
        for key, value in result.items():
            setattr(measurement, key, value)
        measurement.revision += 1
    if errors:
        return before, None, preview, errors
    target.kind = target_kind
    if column_id in after.science.properties:
        after.science.properties[column_id].kind = target_kind
    after.refresh(sync_measurements=False, changed_columns={column_id})
    after.validate()
    return before, after, preview, errors
