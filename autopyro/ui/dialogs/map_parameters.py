"""Restore map parameters and synchronize project bindings."""

from __future__ import annotations

from copy import deepcopy

from PySide6.QtWidgets import QFormLayout

from autopyro.spatial.mapping import MapSpec

from ..i18n import tr
from .map_layers import LAYER_DEFAULTS


def load_project_state(sheet):
    owner = sheet.owner
    sheet.layers = deepcopy(owner.model.project.view.get("map_layers", []))
    sheet.spec = MapSpec.from_dict(owner.model.project.view.get("map_current", {}))


def focus_map_coordinates(sheet):
    if sheet.location.currentData() == 'layers':
        sheet.location.setCurrentIndex(sheet.location.findData('columns'))
    desktop = sheet.owner.desktop
    if desktop.properties_stack.indexOf(sheet.tabs) >= 0:
        desktop.properties_stack.setCurrentWidget(sheet.tabs)
        desktop.properties_dock.show()
        desktop.properties_dock.raise_()
    sheet.tabs.setCurrentIndex(0)
    field = 'sample_key' if sheet.location.currentData() == 'layer' else 'x_id'
    sheet.data.fields[field][0].setFocus()


def visualization_crs_changed(sheet, *_):
    if sheet.loading or sheet.closed:
        return
    # Limits saved in the old projection must not clip the new projection.
    sheet.map_viewport = None
    sheet.layout_form.load(dict.fromkeys(('xmin', 'xmax', 'ymin', 'ymax')))
    sheet.schedule()


def restore_parameters(sheet):
    sheet.timer.stop()
    sheet.map_viewport = None
    defaults = MapSpec().to_dict()
    defaults["title"] = tr(defaults["title"])
    sheet.loading = True
    for form in (sheet.style, sheet.layout_form):
        for key in form.fields:
            setattr(sheet.spec, key, deepcopy(defaults[key]))
        form.load(defaults)
    sheet.spec.style_maps = {}
    sheet.category_table.setRowCount(0)
    for layer in sheet.layers:
        layer.update(deepcopy(LAYER_DEFAULTS))
        layer.pop('category_styles', None)
    sheet.sync_layers()
    sheet.loading = False
    sheet.schedule()


def sync_columns(sheet, *_):
    for combo in sheet.column_combos:
        previous = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(tr('Не задано'), "")
        for column in sheet.owner.model.project.columns:
            combo.addItem(f"{column.name} [{column.unit}]", column.id)
        if previous and combo.findData(previous) < 0:
            combo.addItem(tr('Удалённая колонка'), previous)
        combo.setCurrentIndex(max(0, combo.findData(previous)))
        combo.blockSignals(False)
    if not sheet.loading:
        sheet.schedule()


def sync_coordinate_controls(sheet, *_):
    mode = sheet.location.currentData()
    enabled = {
        'x_id': mode == 'columns', 'y_id': mode == 'columns',
        'source_crs': mode == 'columns',
        'sample_key': mode == 'layer', 'point_layer': mode == 'layer',
        'layer_key': mode == 'layer',
        'scope': mode != 'layers', 'text_numbers': mode != 'layers',
        'include_interpolated': mode != 'layers',
    }
    for key, active in enabled.items():
        widget = sheet.data.fields[key][0]
        for row in range(sheet.data.form.rowCount()):
            item = sheet.data.form.itemAt(row, QFormLayout.ItemRole.FieldRole)
            field = item.widget() if item else None
            if field is not None and (field is widget or field.isAncestorOf(widget)):
                field.setEnabled(active)
                label = sheet.data.form.labelForField(field)
                if label is not None:
                    label.setEnabled(active)
                break


def remember(sheet, stored_layers=None):
    project = sheet.owner.model.project
    if (
        project.view.get("map_current") != sheet.spec.to_dict()
        or project.view.get("map_layers") != sheet.layers
    ):
        project.view.update(
            map_current=sheet.spec.to_dict(), map_layers=deepcopy(sheet.layers) if stored_layers is None else stored_layers
        )
        sheet.owner.model_changed()
