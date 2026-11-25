"""Apply interpolation as a new column and restore or replay saved calculations."""

from __future__ import annotations

from PySide6.QtCore import Qt

from autopyro.analysis.interpolation import minimum_anchor_count, replay
from autopyro.model.core import Column

from ..i18n import tr, message
from ..table import AddColumnCommand
from ..tasks import TaskCancelled, run_with_progress
from ..widgets.documents import refill_combo, source_token
from .interpolation_results import BACKGROUND_ROWS


def output_contexts(sheet):
    return {sheet.measurement_context[p.row_id][1:] for p in sheet.proposals}


def apply(sheet):
    if not sheet._diagnostics_current or (
        sheet.property("workspaceTracked") and sheet._preview_source != source_token(sheet.owner)
    ):
        sheet.invalidate()
        sheet.summary.setText(tr("Данные или выборка изменились. Рассчитайте предпросмотр заново."))
        return
    if not sheet.proposals:
        return
    contexts = sheet.output_contexts()
    if len(contexts) != 1:
        sheet.set_announced_summary(tr('Для создания колонки выберите область с одной единицей и базой Y.'))
        return
    (unit, basis), = contexts
    name = sheet.name.text().strip()
    if not name:
        sheet.set_name_error(tr('Введите название новой колонки.'))
        return
    if name in {c.name for c in sheet.project.columns}:
        sheet.set_name_error(tr('Колонка с таким названием уже есть.'))
        return
    sheet.set_name_error()
    sheet.column = Column(
        name,
        "number",
        unit=unit,
        metadata={
            **sheet.parameters,
            **({"basis": basis} if basis else {}),
            "stale": False,
            "proposals": [p.record() for p in sheet.proposals],
            "scope_row_ids": sorted(sheet.row_ids),
            "interpolation_protocol": sheet.protocol,
        },
    )
    sheet.values = {p.row_id: p.value for p in sheet.proposals}
    if sheet.property("workspaceTracked"):
        try:
            command = AddColumnCommand(sheet.owner.model, sheet.column, sheet.values)
        except (ValueError, TypeError, KeyError) as exc:
            sheet.invalidate()
            sheet.summary.setText(tr('Ошибка: {0}', message(exc)))
            return
        # Preparing a large inline history snapshot runs the Qt event loop.
        # A direct proxy filter/sort change must still reject this preview.
        if sheet._preview_source != source_token(sheet.owner):
            sheet.invalidate()
            sheet.summary.setText(tr("Данные или выборка изменились. Рассчитайте предпросмотр заново."))
            return
        sheet.owner.model.undo.push(command)
        sheet.invalidate()
        sheet.summary.setText(tr("Колонка создана. Доступна отмена."))
        sheet.announce_summary()
        sheet.refresh_saved_runs()
    else:
        sheet.accept()


def refresh_saved_runs(sheet):
    choices = [(tr("{0} · устарел", column.name) if column.metadata.get("stale") else column.name, column.id)
               for column in sheet.project.columns
               if column.metadata.get("interpolation_protocol")]
    refill_combo(sheet.saved_runs, choices)
    sheet.replay_button.setEnabled(bool(choices))
    sheet.load_run_button.setEnabled(bool(choices))


def load_saved_parameters(sheet):
    sheet.refresh_columns()
    column = next((c for c in sheet.project.columns if c.id == sheet.saved_runs.currentData()), None)
    if column is None:
        return
    values = column.metadata
    constraints = values.get("constraints", {})
    method = values.get("method")
    minimum = minimum_anchor_count(method)
    min_anchors = values.get("min_anchors", minimum)
    gap = values.get("max_gap", 100)
    distance = values.get("max_extrapolation", 10)
    duplicate_index = sheet.duplicates.findData(values.get("duplicate_policy", "reject"))
    if (not isinstance(constraints, dict) or type(min_anchors) is not int
            or not minimum <= min_anchors <= sheet.min_anchors.maximum()
            or type(gap) not in (int, float) or not sheet.gap.minimum() <= gap <= sheet.gap.maximum()
            or type(distance) not in (int, float)
            or not sheet.extrapolation_limit.minimum() <= distance <= sheet.extrapolation_limit.maximum()
            or duplicate_index < 0 or ("max_extrapolation" in values and method != "linear")):
        sheet.set_announced_summary(tr('Сохранённые параметры повреждены или выходят за допустимые пределы.'))
        return
    fields = ((sheet.x, values.get("x_id")), (sheet.y, values.get("y_id")),
              (sheet.method, method),
              (sheet.well, constraints.get("well_id")),
              (sheet.interval_bottom, constraints.get("exclude_intervals_by")))
    groups = values.get("group_ids", ())
    levels = constraints.get("levels", ())
    if any(not isinstance(items, (list, tuple)) or any(not isinstance(item, str) for item in items)
           for items in (groups, levels)):
        sheet.set_announced_summary(tr('Сохранённые параметры повреждены или выходят за допустимые пределы.'))
        return
    if (not values.get("x_id") or not values.get("y_id")
            or any(combo.findData(value) < 0 for combo, value in fields)
            or set(groups) - {c.id for c in sheet.project.columns} or set(levels) - set(sheet._boundary_choices)):
        sheet.set_announced_summary(tr('Часть сохранённых параметров недоступна в текущем проекте.'))
        return
    for combo, value in fields:
        combo.setCurrentIndex(combo.findData(value))
    # The historical confirmation applied to the archived scope.
    sheet.without_well.setChecked(False)
    for widget, identifiers in ((sheet.groups, groups), (sheet.boundaries, levels)):
        for i in range(widget.count()):
            item = widget.item(i)
            item.setCheckState(Qt.CheckState.Checked if item.data(Qt.ItemDataRole.UserRole) in identifiers
                               else Qt.CheckState.Unchecked)
    sheet.gap.setValue(gap)
    sheet.min_anchors.setValue(min_anchors)
    sheet.duplicates.setCurrentIndex(duplicate_index)
    sheet.extrapolate.setChecked("max_extrapolation" in values)
    sheet.extrapolation_limit.setValue(distance)
    sheet.parameter_tabs.setCurrentIndex(0)
    sheet.invalidate()
    if not sheet.input_error():
        sheet.set_announced_summary(tr('Параметры загружены. Предпросмотр использует текущую область и схему.'))
    else:
        sheet.announce_summary()


def run_snapshot_task(function, protocol, title, update):
    percent = 0

    def progress(value):
        nonlocal percent
        percent = value
        checkpoint()

    def checkpoint():
        update(title, percent)

    checkpoint()
    return function(protocol, checkpoint=checkpoint, progress=progress)


def replay_saved(sheet):
    sheet.replay_status.hide()
    sheet.refresh_saved_runs()
    column = next((c for c in sheet.project.columns if c.id == sheet.saved_runs.currentData()), None)
    if column is None:
        return
    try:
        protocol = column.metadata["interpolation_protocol"]
        if isinstance(protocol, str) or (isinstance(protocol, dict) and len(protocol.get("rows", [])) >= BACKGROUND_ROWS):
            title = tr('Проверить сохранённый расчёт')
            proposals = run_with_progress(
                sheet, title, lambda progress: sheet.run_snapshot_task(replay, protocol, title, progress),
                cancellable=True,
            )
        else:
            proposals = replay(protocol)
    except TaskCancelled:
        text = tr('Проверка снимка отменена.')
    except (ValueError, TypeError, KeyError) as exc:
        text = tr('Ошибка: {0}', message(exc))
    else:
        text = tr('Расчёт воспроизведён из снимка: {0} значений. Таблица не изменена.', len(proposals))
    sheet.replay_status.setText(text)
    sheet.replay_status.show()
    sheet.announce_summary(text)
