"""Bind interpolation parameters to project columns and stratigraphic constraints."""

from __future__ import annotations

from PySide6.QtCharts import QChart
from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtWidgets import QListWidgetItem

from autopyro.analysis.interpolation import minimum_anchor_count

from ..i18n import tr
from ..widgets.documents import refill_combo


WELL_NAMES = ("wellbore / name", "wellbore id", "wellbore", "bore", "ствол",
              "well / name", "well id", "well", "скважина")


def column_label(column):
    return f"{column.name} [{column.unit}]" if column.unit else column.name


def default_well(sheet):
    names = {column.name.strip().casefold(): column.id for column in sheet.project.columns}
    return sheet.bound_well() or next((names[name] for name in WELL_NAMES if name in names), None)


def bound_well(sheet):
    return sheet.project.view.get("stratigraphy", {}).get("binding", {}).get("well_id")


def default_well_label(sheet):
    return tr('Из привязки стратиграфии') if sheet.bound_well() else tr('Не задан')


def selected_well(sheet):
    return sheet.well.currentData() or sheet.bound_well()


def numeric_choices(sheet):
    return [(column_label(column), column.id) for column in sheet.project.columns
            if column.kind == "number" and not column.metadata.get("method")]


def selected_group_ids(sheet):
    return [item.data(Qt.ItemDataRole.UserRole)
            for i in range(sheet.groups.count())
            if (item := sheet.groups.item(i)).checkState() == Qt.CheckState.Checked]


def input_error(sheet):
    x_id, y_id = sheet.x.currentData(), sheet.y.currentData()
    if not x_id or not y_id or x_id == y_id:
        return tr('Выберите разные числовые колонки X и Y')
    axes = {x_id, y_id}
    if axes.intersection((sheet.well.currentData(), sheet.bound_well())):
        return tr('Для ствола выберите колонку, отличную от X и Y.')
    groups = sheet.selected_group_ids()
    if axes.intersection(groups):
        return tr('Исключите колонки X и Y из группировки.')
    bottom_id = sheet.interval_bottom.currentData()
    binding = sheet.project.view.get("stratigraphy", {}).get("binding", {})
    if bottom_id and (bottom_id in axes or bottom_id == binding.get("depth_id")):
        return tr('Выберите отдельную числовую колонку подошвы пробы')
    if binding.get("sample_mode", "point") == "interval":
        if bottom_id is None or bottom_id != binding.get("bottom_id"):
            return tr('Выберите колонку подошвы из привязки стратиграфии')
    if not sheet.selected_well() and not sheet.without_well.isChecked():
        return tr('Выберите ствол или подтвердите расчёт без разделения по стволам.')
    if not sheet.selected_well() and not groups:
        return tr('Выберите хотя бы одну колонку группировки.')
    return ""


def restore_parameters(sheet):
    sheet.parameter_defaults.restore()
    sheet.preview_model.set_rows(())
    sheet.preview_model.sample_labels = {}
    sheet.preview_model.support_details = {}
    sheet.preview_model.group_labels = {}
    sheet.protocol = None
    sheet._diagnostic_rows, sheet._excluded_rows = [], []
    refill_combo(sheet.diagnostic_status, [(tr('Все статусы'), None)])
    sheet._diagnostic_column_id = None
    sheet.show_diagnostics()
    sheet._pending_chart = None
    sheet._chart_dirty = False
    sheet._chart_group_by_anchor = {}
    sheet._support_spans = {}
    sheet.chart_well.hide()
    sheet.chart_excluded.hide()
    sheet.chart_table_button.hide()
    sheet.replace_chart(QChart())
    sheet.invalidate()
    sheet.summary.setText(sheet.input_error() or tr('Рассчитайте предпросмотр. Исходный Y сохраняется.'))


def set_name_error(sheet, text=""):
    sheet.name_error.setText(text)
    sheet.name_error.setVisible(bool(text))
    sheet.name.setAccessibleDescription(text)
    if text:
        sheet.parameter_tabs.setCurrentIndex(0)
        sheet.name.setFocus()
        sheet.name.selectAll()
        sheet.announce_summary(text)


def update_extrapolation_controls(sheet, *args):
    linear = sheet.method.currentData() == "linear"
    sheet.extrapolate.setEnabled(linear)
    if not linear:
        sheet.extrapolate.setChecked(False)
    sheet.extrapolation_limit.setEnabled(linear and sheet.extrapolate.isChecked())


def update_method_controls(sheet):
    was_default = sheet.min_anchors.value() == sheet.min_anchors.minimum()
    minimum = minimum_anchor_count(sheet.method.currentData())
    sheet.min_anchors.setMinimum(minimum)
    if was_default:
        sheet.min_anchors.setValue(minimum)
    sheet.update_extrapolation_controls()


def refresh_columns(sheet):
    sheet.refresh_saved_runs()
    sheet.refresh_boundaries()
    if refill_combo(sheet.well, [(sheet.default_well_label(), None), *[(c.name, c.id) for c in sheet.project.columns]]):
        sheet.invalidate()
    choices = [(tr('Не задан'), None), *sheet.numeric_choices()]
    for combo in (sheet.x, sheet.y):
        if refill_combo(combo, choices):
            sheet.invalidate()
    if refill_combo(sheet.interval_bottom, choices):
        sheet.invalidate()
    groups = [(column.name, column.id) for column in sheet.project.columns]
    if [(sheet.groups.item(i).text(), sheet.groups.item(i).data(Qt.ItemDataRole.UserRole))
            for i in range(sheet.groups.count())] != groups:
        checked = {sheet.groups.item(i).data(Qt.ItemDataRole.UserRole): sheet.groups.item(i).checkState()
                   for i in range(sheet.groups.count())}
        blocker = QSignalBlocker(sheet.groups)
        sheet.groups.clear()
        for label, identifier in groups:
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, identifier)
            item.setCheckState(checked.get(identifier, Qt.CheckState.Unchecked))
            sheet.groups.addItem(item)
        del blocker
        sheet.invalidate()


def refresh_boundaries(sheet):
    scheme = sheet.project.view.get("stratigraphy", {}).get("scheme", {})
    choices = sorted({item["level"] for item in scheme.get("intervals", [])})
    if choices == sheet._boundary_choices:
        return
    checked = {sheet.boundaries.item(i).data(Qt.ItemDataRole.UserRole)
               for i in range(sheet.boundaries.count())
               if sheet.boundaries.item(i).checkState() == Qt.CheckState.Checked}
    blocker = QSignalBlocker(sheet.boundaries)
    sheet.boundaries.clear()
    for level in choices:
        item = QListWidgetItem(tr('Не пересекать границы: {0}', level))
        item.setData(Qt.ItemDataRole.UserRole, level)
        item.setCheckState(Qt.CheckState.Checked
                           if level in checked or level not in (sheet._boundary_choices or [])
                           else Qt.CheckState.Unchecked)
        sheet.boundaries.addItem(item)
    sheet._boundary_choices = choices
    sheet.boundaries.setVisible(bool(choices))
    del blocker
