"""Manage statistics parameters, project bindings and saved comparison scopes."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import QListWidgetItem

from ..i18n import tr


def restore_parameters(sheet):
    sheet._initial_parameters.restore()
    sheet.search_properties(sheet.search.text())
    sheet.request_refresh()


def fill_comparison_views(sheet, selected):
    view = sheet.owner.model.project.view
    choices = [(tr('Представление: {0}', name), f"view:{name}") for name in view.get("named", {})]
    choices += [(tr('Набор проб: {0}', name), f"set:{name}") for name in view.get("sample_sets", {})]
    for combo, scope in zip(sheet.comparison_views, selected):
        with QSignalBlocker(combo):
            combo.clear()
            combo.addItem(tr('Выберите сохранённую выборку'), None)
            for label, value in choices:
                combo.addItem(label, value)
                combo.setItemData(combo.count() - 1, label, Qt.ItemDataRole.ToolTipRole)
            combo.setCurrentIndex(max(0, combo.findData(scope)))


def sync_comparison_views(sheet):
    sheet.fill_comparison_views([combo.currentData() for combo in sheet.comparison_views])
    sheet.request_refresh()


def rename_comparison_scope(sheet, old_scope, new_scope):
    selected = [combo.currentData() for combo in sheet.comparison_views]
    sheet.fill_comparison_views([new_scope if scope == old_scope else scope for scope in selected])
    if sheet.previous_sample_filter and old_scope == f"set:{sheet.previous_sample_filter[3]}":
        sheet.previous_sample_filter = (*sheet.previous_sample_filter[:3], new_scope[4:])


def search_properties(sheet, text):
    for i in range(sheet.properties.count()):
        item = sheet.properties.item(i)
        item.setHidden(text.casefold() not in item.text().casefold())


def checked(widget):
    return tuple(
        widget.item(i).data(Qt.ItemDataRole.UserRole)
        for i in range(widget.count())
        if widget.item(i).checkState() == Qt.CheckState.Checked
    )


def restore_settings(sheet, settings):
    for widget, key in (
        (sheet.properties, "column_ids"),
        (sheet.groups, "group_ids"),
    ):
        if key in settings:
            for i in range(widget.count()):
                item = widget.item(i)
                item.setCheckState(
                    Qt.CheckState.Checked
                    if item.data(Qt.ItemDataRole.UserRole) in settings[key]
                    else Qt.CheckState.Unchecked
                )
    sheet.text_numbers.setChecked(settings.get("text_numbers", False))
    sheet.interpolated.setChecked(settings.get("include_interpolated", False))
    sheet.correlations.setChecked(settings.get("correlations", True))
    sheet.min_pairs.setValue(settings.get("min_pairs", 3))
    index = sheet.scope.findData(settings.get("scope", "filtered"))
    sheet.scope.setCurrentIndex(max(0, index))
    selected = []
    for side in ("a", "b"):
        legacy = settings.get(f"comparison_{side}")
        selected.append(settings.get(f"comparison_scope_{side}", f"view:{legacy}" if legacy is not None else None))
    sheet.fill_comparison_views(selected)


def selected_row_ids(sheet):
    return sheet.owner.analysis_scope.rows()


def selection_changed(sheet, *args):
    if sheet.scope.currentData() == "selected":
        sheet.request_refresh()


def sync_columns(sheet, *args):
    for widget in (sheet.properties, sheet.groups):
        checked = set(sheet.checked(widget))
        widget.blockSignals(True)
        widget.clear()
        for column in sheet.owner.model.project.columns:
            item = QListWidgetItem(
                column.name + (f" [{column.unit}]" if column.unit else "")
            )
            item.setData(Qt.ItemDataRole.UserRole, column.id)
            item.setCheckState(
                Qt.CheckState.Checked
                if column.id in checked
                else Qt.CheckState.Unchecked
            )
            widget.addItem(item)
        widget.blockSignals(False)
    sheet.request_refresh()
