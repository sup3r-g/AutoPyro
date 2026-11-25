"""Save regional scenarios and capture snapshots for A/B comparison."""

from dataclasses import asdict

from PySide6.QtWidgets import QInputDialog, QMessageBox

from autopyro.spatial.regional_comparison import freeze_regional_result

from ..i18n import tr, message


def publish_comparison_selection(sheet, selection):
    if sheet.tabs.currentWidget() is sheet.comparison:
        sheet.map_selection_changed.emit(selection)


def capture_comparison_snapshot(sheet):
    if sheet.result is None or sheet.report is None:
        return
    name, accepted = QInputDialog.getText(sheet, tr('Сохранить снимок расчёта…'), tr('Название'))
    name = name.strip()
    if not accepted or not name or sheet.result is None or sheet.report is None:
        return
    if name in sheet.comparison.snapshots:
        QMessageBox.warning(sheet, tr('Ошибка'), tr('Снимок с таким названием уже существует.'))
        return
    try:
        snapshot = freeze_regional_result(sheet.result, sheet.report, sheet.project.columns)
    except (ValueError, TypeError) as exc:
        QMessageBox.warning(sheet, tr('Ошибка'), message(exc))
        return
    snapshots = sheet.project.view.setdefault('regional_comparison_snapshots', {})
    snapshots[name] = snapshot
    sheet.comparison.snapshots = snapshots
    sheet.comparison.refresh_snapshots(name)
    sheet.scenarios_changed.emit()


def refresh_scenarios(sheet, name=''):
    sheet.scenario.clear()
    sheet.scenario.addItems(sorted(sheet.project.view.get('regional_scenarios', {})))
    if name:
        sheet.scenario.setCurrentText(name)


def save_scenario(sheet):
    name, accepted = QInputDialog.getText(sheet, tr('Сохранить сценарий…'), tr('Название'), text=sheet.scenario.currentText())
    if not accepted or not name.strip():
        return
    try:
        options = asdict(sheet.options())
    except ValueError as exc:
        QMessageBox.warning(sheet, tr('Ошибка'), message(exc))
        return
    sheet.project.view.setdefault('regional_scenarios', {})[name.strip()] = options
    sheet.refresh_scenarios(name.strip())
    sheet.scenarios_changed.emit()


def delete_scenario(sheet):
    name = sheet.scenario.currentText()
    if name in sheet.project.view.get('regional_scenarios', {}):
        del sheet.project.view['regional_scenarios'][name]
        sheet.refresh_scenarios()
        sheet.scenarios_changed.emit()


def load_scenario(sheet):
    saved = sheet.project.view.get('regional_scenarios', {}).get(sheet.scenario.currentText())
    if saved is None:
        return
    layer_index = sheet.layer.findData(saved['layer_id'])
    layer = next((l for l in (sheet.report or {}).get('layers', []) if l['id'] == saved['layer_id']), None)
    fields = set(layer.get('fields', [])) | {key for feature in layer['features'] for key in feature['properties']} if layer else set()
    if layer_index < 0 or saved['region_field'] not in fields or any(
            combo.findData(saved.get(key, '')) < 0 for key, combo in sheet.bindings.items()):
        QMessageBox.warning(sheet, tr('Ошибка'), tr('В сценарии есть недоступный слой, поле или колонка.'))
        return
    sheet.layer.setCurrentIndex(layer_index)
    sheet.region.setCurrentText(saved['region_field'])
    for key, combo in sheet.bindings.items():
        combo.setCurrentIndex(combo.findData(saved.get(key, '')))
    for key, listing in sheet.choices.items():
        sheet.check_choices(listing, saved.get(key, []))
    for key, widget in [('depth_min', sheet.depth_min), ('depth_max', sheet.depth_max)]:
        value = saved.get(key)
        widget.setText('' if value is None else str(value))
    for key, widget, default in [('stratify_methods', sheet.stratify, True), ('text_numbers', sheet.text_numbers, True),
            ('include_interpolated', sheet.interpolated, False), ('by_well', sheet.by_well, False)]:
        widget.setChecked(saved.get(key, default))
    sheet.invalidate()
