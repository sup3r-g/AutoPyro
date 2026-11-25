"""Manage map layers, their attributes and category styles."""

from __future__ import annotations

from copy import deepcopy
import json

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import QComboBox, QTableWidgetItem
from matplotlib.colors import to_hex, to_rgba
import numpy as np

from autopyro.spatial.mapping import crs_label, layer_categories, default_point_category

from ..i18n import tr, message


LAYER_DEFAULTS = dict(visible=True, color="#778899", fill="#c7ddeb", alpha=0.35, line_width=1, color_by='',
                      filter={}, label_by='', label_size=8, color_mode='category', color_scale='linear',
                      cmap='viridis', color_min=None, color_max=None, missing_color='#999999', legend_hidden=True)


def sync_layers(sheet):
    for combo in (sheet.point_layer, sheet.site_layer, sheet.attribute_layer, sheet.category_layer, sheet.layer_settings_layer):
        previous = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(tr('Не задано'), "")
        for layer in sheet.layers:
            if 'raster' in layer and combo in (sheet.point_layer, sheet.site_layer):
                continue
            combo.addItem(layer["name"], layer["id"])
        combo.setCurrentIndex(max(0, combo.findData(previous)))
        combo.blockSignals(False)
    sheet.layer_table.blockSignals(True)
    sheet.layer_table.setRowCount(len(sheet.layers))
    for row, layer in enumerate(sheet.layers):
        for col, key in enumerate(
            ("name", "color", "fill", "alpha", "line_width", "id")
        ):
            item = QTableWidgetItem(str(layer[key]))
            if col == 0:
                item.setCheckState(
                    Qt.CheckState.Checked
                    if layer.get("visible", True)
                    else Qt.CheckState.Unchecked
                )
            sheet.layer_table.setItem(row, col, item)
        selector = QComboBox()
        selector.addItem(tr('Не задано'), '')
        fields = layer.get('fields', sorted({key for feature in layer['features'] for key in feature['properties']}))
        for field in fields:
            selector.addItem(field, field)
        selector.setCurrentIndex(max(0, selector.findData(layer.get('color_by', ''))))
        selector.currentIndexChanged.connect(lambda _, lid=layer['id'], combo=selector: sheet.layer_color_attribute_changed(lid, combo.currentData()))
        sheet.layer_table.setCellWidget(row, 6, selector)
    sheet.layer_table.blockSignals(False)
    sheet.sync_layer_fields()
    if sheet.attribute_layer.currentIndex() == 0 and sheet.layers:
        sheet.attribute_layer.setCurrentIndex(1)
    sheet.sync_attributes()
    if sheet.category_layer.currentIndex() == 0 and sheet.layers:
        sheet.category_layer.setCurrentIndex(1)
    sheet.sync_layer_categories()
    if sheet.layer_settings_layer.currentIndex() == 0 and sheet.layers:
        sheet.layer_settings_layer.setCurrentIndex(1)
    sheet.sync_layer_settings()


def layer_color_attribute_changed(sheet, layer_id, field):
    layer = next((item for item in sheet.layers if item['id'] == layer_id), None)
    if layer is not None:
        layer['color_by'] = field
        sheet.category_layer.setCurrentIndex(sheet.category_layer.findData(layer_id))
        sheet.sync_layer_categories()
        if sheet.layer_settings_layer.currentData() == layer_id:
            sheet.sync_layer_settings()
        sheet.schedule()


def sync_layer_categories(sheet, *_):
    layer = next((item for item in sheet.layers if item['id'] == sheet.category_layer.currentData()), None)
    entries = layer_categories(layer) if layer else {}
    sheet.layer_category_info.setText(layer.get('color_by') or tr('Цвет по атрибуту') if layer else '')
    sheet.layer_category_keys = list(entries)
    sheet.layer_category_table.blockSignals(True)
    sheet.layer_category_table.setRowCount(len(entries))
    for row, (key, (label, color)) in enumerate(entries.items()):
        value = json.loads(key)[1]
        for column, text in enumerate((str(value) if value is not None else '—', to_hex(color, keep_alpha=True), label)):
            item = QTableWidgetItem(text)
            if column == 0:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            sheet.layer_category_table.setItem(row, column, item)
    sheet.layer_category_table.blockSignals(False)


def layer_category_edited(sheet, item):
    if item.column() == 0 or item.row() >= len(sheet.layer_category_keys):
        return
    layer = next((value for value in sheet.layers if value['id'] == sheet.category_layer.currentData()), None)
    if layer is None or not layer.get('color_by'):
        return
    color = sheet.layer_category_table.item(item.row(), 1).text()
    try:
        to_rgba(color)
    except ValueError as exc:
        sheet.sync_layer_categories()
        sheet.status.setText(message(exc))
        return
    key = sheet.layer_category_keys[item.row()]
    layer.setdefault('category_styles', {}).setdefault(layer['color_by'], {})[key] = {
        'color': color, 'label': sheet.layer_category_table.item(item.row(), 2).text(),
    }
    sheet.schedule()


def reset_layer_categories(sheet):
    layer = next((item for item in sheet.layers if item['id'] == sheet.category_layer.currentData()), None)
    if layer is not None:
        layer.get('category_styles', {}).pop(layer.get('color_by'), None)
        sheet.sync_layer_categories()
        sheet.schedule()


def select_attribute_layer(sheet):
    row = sheet.layer_table.currentRow()
    if 0 <= row < len(sheet.layers):
        sheet.attribute_layer.setCurrentIndex(sheet.attribute_layer.findData(sheet.layers[row]['id']))
        sheet.category_layer.setCurrentIndex(sheet.category_layer.findData(sheet.layers[row]['id']))
        sheet.layer_settings_layer.setCurrentIndex(sheet.layer_settings_layer.findData(sheet.layers[row]['id']))


def sync_attributes(sheet, *_):
    layer = next((item for item in sheet.layers if item['id'] == sheet.attribute_layer.currentData()), None)
    sheet.attribute_model.set_layer(layer)
    sheet.feature_card.clear()
    name = crs_label(layer['crs']) if layer and layer.get('crs') else tr('Неизвестная CRS')
    sheet.attribute_info.setText(
        f"{tr('Объектов')}: {sheet.attribute_model.rowCount()} · {name}" if layer else tr('Выберите пространственный слой')
    )


def sync_layer_fields(sheet, *_):
    for layer_combo, field_combo in (
        (sheet.point_layer, sheet.layer_key),
        (sheet.site_layer, sheet.site_label),
    ):
        previous = field_combo.currentData()
        layer = next(
            (l for l in sheet.layers if l["id"] == layer_combo.currentData()), None
        )
        fields = (
            sorted({key for f in layer["features"] for key in f["properties"]})
            if layer
            else []
        )
        field_combo.blockSignals(True)
        field_combo.clear()
        field_combo.addItem(tr('Выберите поле'), "")
        for key in fields:
            field_combo.addItem(key, key)
        field_combo.setCurrentIndex(max(0, field_combo.findData(previous)))
        field_combo.blockSignals(False)


def refresh_categories(sheet):
    sheet.category_table.blockSignals(True)
    sheet.category_table.setRowCount(0)
    sheet.entries = []
    for channel, keys in sheet.report['point_categories'].items():
        cid = getattr(sheet.spec, f'{channel}_by')
        for key in keys:
            entry = sheet.spec.style_maps[channel][cid][key]
            row = sheet.category_table.rowCount()
            sheet.category_table.insertRow(row)
            for col, value in enumerate(
                (
                    f"{tr('Цвет') if channel == 'color' else tr('Размер')}: {json.loads(key)[1]}",
                    entry["style"],
                    entry["label"],
                )
            ):
                item = QTableWidgetItem(str(value))
                if col == 0:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                sheet.category_table.setItem(row, col, item)
            sheet.entries.append((channel, cid, key))
    sheet.category_table.blockSignals(False)


def category_changed(sheet, item):
    if item.column() == 0 or item.row() >= len(sheet.entries):
        return
    channel, cid, key = sheet.entries[item.row()]
    entry = sheet.spec.style_maps[channel][cid][key]
    field = 'style' if item.column() == 1 else 'label'
    try:
        value = item.text()
        if field == 'style':
            if channel == 'size':
                try:
                    value = float(value)
                except ValueError:
                    raise ValueError(tr('Размер категории должен быть числом')) from None
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(tr('Размер категории должен быть положительным и конечным'))
            else:
                to_rgba(value)
    except ValueError as exc:
        error = message(exc)
        with QSignalBlocker(sheet.category_table):
            item.setText(str(entry[field]))
            item.setToolTip(error)
        sheet.status.setText(error)
        return
    with QSignalBlocker(sheet.category_table):
        item.setToolTip('')
    entry[field] = value
    sheet.schedule()


def reset_category_parameter(sheet, row, column):
    if column not in (1, 2) or row >= len(sheet.entries):
        return
    channel, cid, key = sheet.entries[row]
    index = list(sheet.spec.style_maps[channel][cid]).index(key)
    entry = default_point_category(channel, key, index)
    sheet.category_table.item(row, column).setText(str(entry['style' if column == 1 else 'label']))


def reset_layer_parameter(sheet, row, column):
    if not 0 <= row < len(sheet.layers) or column not in (0, 1, 2, 3, 4):
        return
    item = sheet.layer_table.item(row, column)
    if column == 0:
        item.setCheckState(Qt.CheckState.Checked)
    else:
        key = {1: 'color', 2: 'fill', 3: 'alpha', 4: 'line_width'}[column]
        item.setText(str(LAYER_DEFAULTS[key]))


def layer_changed(sheet, item):
    row, column = item.row(), item.column()
    key = {0: 'name', 1: 'color', 2: 'fill', 3: 'alpha', 4: 'line_width'}.get(column)
    if key is None or not 0 <= row < len(sheet.layers):
        return
    layer = sheet.layers[row]
    try:
        value = item.text()
        if column in (1, 2):
            to_rgba(value)
        elif column in (3, 4):
            value = float(value)
            if not np.isfinite(value) or (not 0 <= value <= 1 if column == 3 else value <= 0):
                raise ValueError(tr('У слоя alpha 0–1 и положительная толщина линии'))
    except ValueError:
        error = (tr('Укажите допустимый цвет слоя.') if column in (1, 2)
                 else tr('У слоя alpha 0–1 и положительная толщина линии'))
        with QSignalBlocker(sheet.layer_table):
            item.setText(str(layer[key]))
            item.setToolTip(error)
        sheet.status.setText(error)
        return
    with QSignalBlocker(sheet.layer_table):
        item.setToolTip('')
    layer[key] = value
    if column == 0:
        layer['visible'] = item.checkState() == Qt.CheckState.Checked
        for combo in (sheet.point_layer, sheet.site_layer, sheet.attribute_layer, sheet.category_layer, sheet.layer_settings_layer):
            index = combo.findData(layer["id"])
            if index >= 0:
                combo.setItemText(index, layer["name"])
    sheet.schedule()


def remove_layer(sheet):
    selected = {index.row() for index in sheet.layer_table.selectedIndexes()}
    if not selected:
        return
    for i in selected:
        sheet.remember_removed_layer(sheet.layers[i])
    sheet.layers = [
        layer for i, layer in enumerate(sheet.layers) if i not in selected
    ]
    sheet.owner.model.project.view["map_layers"] = deepcopy(sheet.layers)
    sheet.owner.model_changed()
    sheet.sync_layers()
    sheet.schedule()


def add_interpolation_layers(sheet, layers):
    sheet.layers[:0] = layers
    sheet.owner.model.project.view['map_layers'] = deepcopy(sheet.layers)
    sheet.owner.view_changed()
    sheet.sync_layers()
    sheet.schedule()
