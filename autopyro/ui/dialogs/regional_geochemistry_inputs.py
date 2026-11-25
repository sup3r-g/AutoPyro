"""Synchronize map inputs, prepare analysis options and schedule regional calculations."""

from copy import deepcopy

from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtWidgets import QListWidgetItem, QMessageBox

from autopyro.spatial.regional_geochemistry import RegionalOptions, analyze_regions, prepare_input
from autopyro.visualization.plotting import category_key

from ..i18n import tr, message
from ..tasks import run_with_progress


def _regional_map_input(report):
    spec = report['spec']
    bindings = tuple(spec.get(key) for key in (
        'location_mode', 'scope', 'x_id', 'y_id', 'source_crs', 'target_crs',
        'sample_key', 'point_layer', 'layer_key', 'text_numbers', 'include_interpolated',
        'depth_id', 'depth_bottom_id', 'depth_min', 'depth_max', 'strat_id', 'strat_value',
        'date_id', 'date_from', 'date_to',
    ))
    locations = [(point['id'], point['row_ids'], point['x'], point['y']) for point in report['points']]
    layers = {}
    for layer in report['layers']:
        if layer['id'] == spec.get('point_layer') or any(
            (f.get('geometry') or {}).get('type') in ('Polygon', 'MultiPolygon') for f in layer['features']
        ):
            rule = layer.get('filter') or {}
            fields = tuple(layer.get(key) for key in ('name', 'crs', 'source', 'fields', 'features'))
            layers[layer['id']] = fields, rule if rule.get('field') else None
    return bindings, report['row_ids'], locations, layers


def update_regions(sheet):
    previous = sheet.region.currentText()
    sheet.region.clear()
    layer = next((l for l in (sheet.report or {}).get('layers', []) if l['id'] == sheet.layer.currentData()), None)
    if layer:
        fields = sorted(set(layer.get('fields', [])) | {k for f in layer['features'] for k in f['properties']})
        sheet.region.addItems(fields)
        if previous in fields:
            sheet.region.setCurrentText(previous)
        elif 'Paleogeogr' in fields:
            sheet.region.setCurrentText('Paleogeogr')
    sheet.invalidate()


def populate(sheet, key, binding):
    listing = sheet.choices[key]
    listing.blockSignals(True)
    listing.clear()
    cid = sheet.bindings[binding].currentData()
    ids = set((sheet.report or {}).get('row_ids', []))
    values = {category_key(row.values.get(cid)): row.values.get(cid) for row in sheet.project.rows if row.id in ids} if cid else {}
    for value in sorted(values.values(), key=lambda v: str(v)):
        label = repr(value) if isinstance(value, str) and value != value.strip() else str(value)
        item = QListWidgetItem(tr('(Не задан)') if value is None else label)
        item.setData(Qt.ItemDataRole.UserRole, value)
        item.setCheckState(Qt.CheckState.Unchecked)
        listing.addItem(item)
    listing.blockSignals(False)
    sheet.invalidate()


def check_choices(sheet, listing, checked):
    remaining = {category_key(value): value for value in checked}
    with QSignalBlocker(listing):
        for i in range(listing.count()):
            item = listing.item(i)
            key = category_key(item.data(Qt.ItemDataRole.UserRole))
            item.setCheckState(Qt.CheckState.Checked if key in remaining else Qt.CheckState.Unchecked)
            remaining.pop(key, None)
        for value in remaining.values():
            item = QListWidgetItem(tr('(Не задан)') if value is None else str(value))
            item.setData(Qt.ItemDataRole.UserRole, value)
            item.setCheckState(Qt.CheckState.Checked)
            listing.addItem(item)


def set_map_input(sheet, project, report, *, data_revision):
    exclusions = project.view.get('qc_exclusions', {}).get('statistics', {})
    unchanged = (
        project is sheet.project and data_revision == sheet.map_data_revision
        and exclusions == sheet.statistics_exclusions
        and report is not None and sheet.report is not None
        and _regional_map_input(report) == _regional_map_input(sheet.report)
    )
    sheet.project, sheet.report = project, report
    sheet.map_data_revision = data_revision
    sheet.statistics_exclusions = deepcopy(exclusions)
    if report is None:
        sheet.invalidate()
        return
    if unchanged:
        if sheet.isVisible():
            current = sheet.tabs.currentWidget()
            if current is sheet.comparison:
                sheet.comparison.show_selection()
            elif current is sheet.samples_tab:
                sheet.highlight_samples()
            elif current is sheet.spatial_tab:
                sheet.highlight_spatial()
            else:
                sheet.select_on_map(refresh_details=False)
        return
    previous = sheet.layer.currentData()
    sheet.layer.blockSignals(True)
    sheet.layer.clear()
    for layer in report['layers']:
        if any((f.get('geometry') or {}).get('type') in ('Polygon', 'MultiPolygon') for f in layer['features']):
            sheet.layer.addItem(layer['name'], layer['id'])
    sheet.layer.setCurrentIndex(max(0, sheet.layer.findData(previous)))
    sheet.layer.blockSignals(False)
    sheet.update_regions()
    for combo in sheet.bindings.values():
        previous = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(tr('Не задан'), '')
        for column in project.columns:
            combo.addItem(f'{column.name} [{column.unit}]' if column.unit else column.name, column.id)
        combo.setCurrentIndex(max(0, combo.findData(previous)))
        combo.blockSignals(False)
    for key, binding in [('ages', 'age_id'), ('toc_methods', 'toc_method_id'), ('pyrolysis_methods', 'pyrolysis_method_id')]:
        listing = sheet.choices[key]
        checked = [listing.item(i).data(Qt.ItemDataRole.UserRole) for i in range(listing.count())
                   if listing.item(i).checkState() == Qt.CheckState.Checked]
        sheet.populate(key, binding)
        if sheet.bindings[binding].currentData():
            sheet.check_choices(listing, checked)
    if sheet.tabs.currentWidget() is sheet.comparison:
        sheet.comparison.show_selection()


def options(sheet):
    selected = {key: tuple(listing.item(i).data(Qt.ItemDataRole.UserRole) for i in range(listing.count())
                if listing.item(i).checkState() == Qt.CheckState.Checked) for key, listing in sheet.choices.items()}
    number = lambda widget: float(widget.text().strip().replace(',', '.')) if widget.text().strip() else None
    return RegionalOptions(layer_id=sheet.layer.currentData(), region_field=sheet.region.currentText(),
        **{key: combo.currentData() for key, combo in sheet.bindings.items()}, **selected,
        depth_min=number(sheet.depth_min), depth_max=number(sheet.depth_max), stratify_methods=sheet.stratify.isChecked(),
        text_numbers=sheet.text_numbers.isChecked(), include_interpolated=sheet.interpolated.isChecked(),
        by_well=sheet.by_well.isChecked())


def calculate(sheet):
    sheet.invalidate()
    try:
        if not sheet.report:
            raise ValueError(tr('Сначала обновите карту.'))
        options = sheet.options()
        revision = sheet.input_revision
        project, report = run_with_progress(sheet, tr('Подготовка выборки'), lambda: prepare_input(sheet.project, sheet.report, options))
        sheet.run_task(lambda: analyze_regions(project, report, options),
                      lambda result: sheet.show_result(result) if revision == sheet.input_revision else None)
    except Exception as exc:
        QMessageBox.warning(sheet, tr('Ошибка'), message(exc))
