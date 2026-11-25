"""Manage interval draft, local undo, schema validation and saved schemes."""

from __future__ import annotations

from ..i18n import tr, message
from copy import deepcopy
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import QDialog, QTableWidgetItem
from autopyro.model.core import new_id, parse_number
from autopyro.geology.stratigraphy import STRATAPY_FIELDS, validate_scheme
from ..tasks import TaskCancelled, run_with_progress
from ..widgets.documents import refill_combo
from ..widgets.stratigraphy_labels import CheckedValuesDialog, geological_options
from ..widgets.lithology import LithologyCatalog, LithologyPicker
from ..widgets.interval_history import InsertInterval, DeleteIntervals
from .stratigraphy_binding import SetStratigraphyCommand


def initialize(editor):
    owner = editor.owner
    editor.draft = deepcopy(
        owner.model.project.view.get("stratigraphy_scheme")
        or owner.model.project.view.get("stratigraphy", {}).get("scheme")
    ) or {
        "format": "autopyro.stratigraphy",
        "version": 1,
        "id": new_id(),
        "revision": "1",
        "name": tr('Стратиграфические интервалы'),
        "depth_unit": "m",
        "depth_reference": "",
        "source": tr('Ручной ввод'),
        "intervals": [],
    }


def show_geological_columns(editor, visible):
    for column in range(8, editor.intervals.columnCount()):
        editor.intervals.setColumnHidden(column, not visible)


def has_unsaved_draft(editor):
    if (not editor.draft_undo.isClean()
            or editor.owner.model.project.view.get('stratigraphy_scheme') != editor._saved_scheme):
        return True
    try:
        return editor.style_values() != editor._saved_style
    except ValueError:
        return True


def mark_draft_saved(editor, style):
    editor._saved_style = style
    editor._saved_scheme = editor.owner.model.project.view.get('stratigraphy_scheme')
    editor.draft_undo.setClean()
    editor.update_binding_revision()


def load_draft(editor):
    history = getattr(editor, 'interval_history', None)
    if history:
        with history.suspended():
            editor.populate_draft()
        history.reset()
        history.structure_changed()
    else:
        editor.populate_draft()


def populate_draft(editor):
    fields = [(editor.name, 'name'), (editor.revision, 'revision'), (editor.reference, 'depth_reference')]
    for widget, key in fields:
        with QSignalBlocker(widget):
            widget.setText(editor.draft[key])
    with QSignalBlocker(editor.unit):
        editor.unit.setCurrentText(editor.draft['depth_unit'])
    editor.intervals.setRowCount(0)
    for interval in editor.draft['intervals']:
        editor.insert_interval(editor.intervals.rowCount(), interval)
    editor.refresh_wellbores()


def add_interval(editor, checked=False, interval=None):
    item = interval or {
        "wellbore": "",
        "name": "",
        "level": editor.level.currentData(),
        "top": "",
        "bottom": "",
        "id": new_id(),
    }
    editor.interval_history.stack.push(InsertInterval(editor.interval_history, editor.intervals.rowCount(), item))


def insert_interval(editor, row, item):
    editor.intervals.insertRow(row)
    editor.write_interval(row, item)


def refresh_interval_color_preview(editor, cell):
    if cell.column() == 1 and (color_cell := editor.intervals.item(cell.row(), 5)) is not None:
        editor.intervals.viewport().update(editor.intervals.visualItemRect(color_cell))


def write_interval(editor, row, item):
    color = item.get("color")
    if color is None:
        color = ""
    if not isinstance(color, str):
        from matplotlib.colors import to_hex

        color = to_hex(color, keep_alpha=True)
    for col, key in enumerate(
        ("wellbore", "name", "level", "top", "bottom", "color", "id", "lithology", *STRATAPY_FIELDS)
    ):
        cell = QTableWidgetItem(color if key == "color" else str(item.get(key, "")))
        if col == 6:
            cell.setData(Qt.ItemDataRole.UserRole, deepcopy(item))
        editor.intervals.setItem(row, col, cell)


def remove_intervals(editor):
    rows = sorted({row for selection in editor.intervals.selectedRanges()
                   for row in range(selection.topRow(), selection.bottomRow() + 1)})
    if rows:
        editor.draft_undo.push(DeleteIntervals(editor.interval_history, rows))


def choose_geological_cell(editor, row, column):
    if column == 7:
        editor.choose_lithology()
    elif column in (13, 14):
        from stratapy.setup import formatting

        codes = formatting.minerals_list if column == 13 else formatting.features
        cell = editor.intervals.item(row, column)
        selected = [value for value in cell.text().split(';') if value]
        title = tr('Минералы слоя') if column == 13 else tr('Геологические признаки слоя')
        picker = CheckedValuesDialog(title, geological_options(codes), selected, editor)
        if picker.exec() == picker.DialogCode.Accepted:
            cell.setText(';'.join(picker.values()))


def choose_lithology(editor):
    rows = sorted({row for selection in editor.intervals.selectedRanges()
                   for row in range(selection.topRow(), selection.bottomRow() + 1)})
    if not rows:
        editor.status.setText(tr('Выделите интервалы для назначения литологии.'))
        return
    try:
        custom = editor.column_style_form.rocks_editor.values()
        catalog = LithologyCatalog(custom) if custom else editor.lithology_catalog
        row = editor.intervals.currentRow()
        if row not in rows:
            row = rows[0]
        cell = editor.intervals.item(row, 7)
        picker = LithologyPicker(catalog, current=cell.text(), parent=editor)
        if len(rows) > 1:
            picker.setWindowTitle(tr('Выбор литологии · интервалов: {0}', len(rows)))
        if picker.exec() == QDialog.DialogCode.Accepted:
            editor.lithology_delegate.catalog = catalog
            changes = [(editor.intervals.item(row, 6).text(), 7, picker.selected_code) for row in rows]
            changed = editor.interval_history.change_cells(changes, tr('Назначение литологии'))
            if changed:
                editor.status.setText(tr('Литология назначена интервалам: {0}. Постройте колонку заново.', len(rows)))
            editor.intervals.viewport().update()
    except Exception as exc:
        editor.status.setText(tr('Ошибка: {0}', message(exc)))


def read_scheme(editor):
    steps = editor.read_scheme_steps()
    try:
        while True:
            next(steps)
    except StopIteration as result:
        return validate_scheme(result.value)


def read_scheme_steps(editor):
    scheme = deepcopy({key: value for key, value in editor.draft.items() if key != 'intervals'})
    scheme.update(
        name=editor.name.text(),
        revision=editor.revision.text(),
        depth_reference=editor.reference.text(),
        depth_unit=editor.unit.currentText(),
        intervals=[],
    )
    for row in range(editor.intervals.rowCount()):
        if row and row % 256 == 0:
            yield
        interval = deepcopy(
            editor.intervals.item(row, 6).data(Qt.ItemDataRole.UserRole)
        )
        for col, key in enumerate(
            ("wellbore", "name", "level", "top", "bottom", "color", "id", "lithology", *STRATAPY_FIELDS)
        ):
            value = editor.intervals.item(row, col).text()
            if key in ("lithology", *STRATAPY_FIELDS) and not value and key not in interval:
                continue
            if key == "color":
                if not value.strip():
                    interval.pop("color_manual", None)
                elif value != interval.get("color"):
                    interval["color_manual"] = True
            interval[key] = (
                parse_number(value) if key in ("top", "bottom") else value
            )
        scheme["intervals"].append(interval)
    return scheme


def read_render_scheme(editor):
    if editor.intervals.rowCount() < 1000:
        return editor.read_scheme()
    revision = editor._column_preview_revision

    def prepare():
        steps = editor.read_scheme_steps()
        try:
            while True:
                if revision != editor._column_preview_revision:
                    raise TaskCancelled()
                next(steps)
                yield
        except StopIteration as result:
            return result.value

    title = tr('Подготовка снимка данных…')
    editor._column_render_running = True
    try:
        scheme = run_with_progress(editor, title, lambda update, scheme: validate_scheme(scheme),
                                   cancellable=True, prepare=prepare())
        if revision != editor._column_preview_revision:
            raise TaskCancelled()
        return scheme
    finally:
        editor._column_render_running = False


def refresh_wellbores(editor, *_):
    has_intervals = editor.intervals.rowCount() > 0
    editor.build_column_button.setEnabled(has_intervals)
    editor.save_scheme_button.setEnabled(has_intervals)
    editor.batch_export_button.setEnabled(has_intervals)
    editor.merge_intervals_action.setEnabled(has_intervals)
    editor.diagnose_intervals_action.setEnabled(has_intervals)
    editor.summary_action.setEnabled(has_intervals)
    wells = sorted({cell.text() for row in range(editor.intervals.rowCount())
                    if (cell := editor.intervals.item(row, 0)) and cell.text().strip()})
    refill_combo(editor.column_well, [(well, well) for well in wells])
    editor.column_style_form.available_wells = wells


def save_scheme(editor):
    try:
        scheme = editor.read_scheme()
        style = editor.style_values()
        editor.owner.model.undo.push(SetStratigraphyCommand(
            editor.owner.model, {"scheme": scheme}, scheme_only=True, style=style))
        editor.mark_draft_saved(style)
        active = (editor.owner.model.project.view.get('stratigraphy') or {}).get('scheme')
        if active:
            editor.status.setText(tr('Схема «{0}», редакция {1}, сохранена. Действующая привязка: «{2}», редакция {3}.',
                                   scheme['name'], scheme['revision'], active['name'], active['revision']))
        else:
            editor.status.setText(tr('Схема сохранена без привязки проб. Доступна отмена.'))
        return True
    except Exception as exc:
        editor.status.setText(tr('Ошибка: {0}', message(exc)))
        return False
