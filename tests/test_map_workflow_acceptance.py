"""Exercise the real import/map forms against the local Norwegian example."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QFileDialog, QInputDialog, QPushButton
from matplotlib.backend_bases import MouseEvent

from autopyro.ui import i18n
from autopyro.ui.data_io.import_file import ImportDialog
from autopyro.ui.widgets.spatial import CRSDialog
from autopyro.ui.window import MainWindow
from autopyro.map_tools import filtered_features


ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / 'examples' / 'npd.xlsx', ROOT / 'examples' / 'maps' / 'T3.shp',
         ROOT / 'examples' / 'maps' / 'Norwegian_Wells.shp']
pytestmark = pytest.mark.skipif(not all(path.is_file() for path in FILES), reason='Local NPD and map example files are required')


def test_import_map_style_join_save_export_and_reopen(qtbot, tmp_path, monkeypatch):
    from autopyro.ui.widgets import spatial
    store = QSettings(str(tmp_path / 'preferences.ini'), QSettings.Format.IniFormat)
    store.setValue('general/autosave_enabled', False)
    monkeypatch.setattr(i18n, 'settings', lambda: store)
    monkeypatch.setattr(spatial, 'settings', lambda: store)
    i18n.initialize(store)
    source_paths = [FILES[0], *FILES[1].parent.glob('*')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths if p.is_file()}
    reader = ImportDialog(path=str(FILES[0]))
    qtbot.addWidget(reader)
    reader.show()
    reader.load()
    qtbot.waitUntil(lambda: reader.raw is not None and reader.isEnabled(), timeout=120000)
    assert len(reader.raw[1]) == 32651
    wanted = {'Well', 'Age', 'Depth', 'TOC'}
    for row in range(reader.mapping.rowCount()):
        header = reader.mapping.item(row, 1).text()
        check = reader.mapping.cellWidget(row, 0)
        if check.isChecked() != (header in wanted):
            check.click()
        if header == 'Depth':
            reader.mapping.item(row, 4).setText('m')
    reader.buttons.button(QDialogButtonBox.StandardButton.Ok).click()
    qtbot.waitUntil(lambda: reader.project is not None and reader.result() == reader.DialogCode.Accepted, timeout=120000)
    project = reader.project
    assert {c.name for c in project.columns} == wanted
    assert len(project.rows) == 32651
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.may_replace = lambda: True
    owner.set_project(project, prepared=reader.project_prepared)
    owner.resize(1600, 1000)
    owner.show()
    owner.actions[i18n.tr('Карты…')].trigger()
    dialog = owner.map_window
    warnings = []
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warnings.append(args[2]))
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args, **kwargs: (args[3][0], True))
    for path in FILES[1:]:
        monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args, chosen=str(path), **kwargs: (chosen, ''))
        dialog.import_layer()
        dialog.redraw()
        assert dialog.report is not None, dialog.status.text()
        assert not warnings, warnings
    assert len(dialog.result['layer_features']) == 517
    assert not dialog.report['points']  # Both layers render before any sample binding.
    regions = next(item for item in dialog.layers if item['name'] == 'T3')
    wells = next(item for item in dialog.layers if item['name'] == 'Norwegian_Wells')
    ids = {c.name: c.id for c in project.columns}
    def select(form, key, value):
        combo = form.fields[key][0]
        index = combo.findData(value)
        assert index >= 0, (key, value)
        combo.setCurrentIndex(index)
    select(dialog.data, 'location_mode', 'layer')
    select(dialog.data, 'sample_key', ids['Well'])
    select(dialog.data, 'point_layer', wells['id'])
    select(dialog.data, 'layer_key', 'wlbWellbor')
    def choose_crs():
        picker = QApplication.activeModalWidget()
        assert isinstance(picker, CRSDialog)
        picker.search.setText('EPSG:5940')
        picker.timer.stop()
        picker.filter_tree()
        picker.buttons.button(QDialogButtonBox.StandardButton.Ok).click()
    QTimer.singleShot(0, choose_crs)
    dialog.visualization_crs.showPopup()
    assert dialog.visualization_crs.currentData() == 'EPSG:5940'
    select(dialog.style, 'color_by', ids['TOC'])
    select(dialog.style, 'color_mode', 'numeric')
    select(dialog.filters, 'strat_id', ids['Age'])
    qtbot.keyClicks(dialog.filters.fields['strat_value'][0], 'T3n-T3r')
    dialog.redraw()
    assert dialog.report is not None, dialog.status.text()
    assert len(dialog.report['row_ids']) == 1841
    keyset = {feature['properties']['wlbWellbor'] for feature in wells['features']}
    expected = sum(row.values[ids['Well']] in keyset for row in project.rows if row.values[ids['Age']] == 'T3n-T3r')
    assert len(dialog.report['points']) == expected
    color_field = dialog.layer_table.cellWidget(dialog.layers.index(regions), 6)
    color_field.setCurrentIndex(color_field.findData('Paleogeogr'))
    dialog.category_layer.setCurrentIndex(dialog.category_layer.findData(regions['id']))
    dialog.layer_category_table.item(0, 1).setText('#5b9bd5')
    dialog.layer_category_table.item(0, 2).setText('Shelf')
    dialog.layer_settings_layer.setCurrentIndex(dialog.layer_settings_layer.findData(wells['id']))
    select(dialog.layer_display, 'color_by', 'wlbWaterDe')
    select(dialog.layer_display, 'color_mode', 'numeric')
    select(dialog.layer_rules, 'filter_field', 'wlbWellbor')
    select(dialog.layer_rules, 'filter_op', 'contains')
    qtbot.keyClicks(dialog.layer_rules.fields['filter_value'][0], '7120')
    select(dialog.layer_rules, 'label_by', 'wlbWellbor')
    dialog.layer_table.reordered.emit(dialog.layers.index(wells), 0)
    dialog.layout_form.fields['scale_bar'][0].click()
    dialog.layout_form.fields['north_arrow'][0].click()
    dialog.redraw()
    assert dialog.report is not None, dialog.status.text()
    assert len(filtered_features(wells)) == 42
    assert dialog.result['layer_labels']
    assert dialog.result['decorations']['scale_m'] > 0
    dialog.attribute_layer.setCurrentIndex(dialog.attribute_layer.findData(wells['id']))
    available = {point['values'][ids['Well']] for point in dialog.report['points']}
    projected = {fid: geometry for lid, fid, geometry in dialog.result['layer_features'] if lid == wells['id']}
    candidates = [f for f in filtered_features(wells) if f['properties']['wlbWellbor'] in available]
    # Pick an isolated well and zoom before clicking: global-scale pixel rounding
    # cannot distinguish neighbouring/overlapping well symbols reliably.
    chosen = max(candidates, key=lambda f: min(projected[str(f['id'])].distance(other)
                                              for fid, other in projected.items() if fid != str(f['id'])))
    row = next(i for i in range(dialog.attribute_model.rowCount()) if str(dialog.attribute_model.frame.iat[i, 0]) == str(chosen['id']))
    dialog.attribute_table.doubleClicked.emit(dialog.attribute_model.index(row, 0))
    dialog.canvas.draw()
    dialog.attribute_table.selectionModel().clearSelection()
    geometry = projected[str(chosen['id'])]
    px, py = dialog.result['axes'].transData.transform((geometry.x, geometry.y))
    for name in ('button_press_event', 'button_release_event'):
        dialog.canvas.callbacks.process(name, MouseEvent(name, dialog.canvas, px, py, button=1))
    rows = dialog.attribute_table.selectionModel().selectedRows()
    assert rows and str(dialog.attribute_model.frame.iat[rows[0].row(), 0]) == str(chosen['id'])
    assert 'Связанных замеров:' in dialog.feature_card.toPlainText()
    dialog.attribute_table.doubleClicked.emit(dialog.attribute_model.index(rows[0].row(), 0))
    assert dialog.feature_highlights
    dialog.toolbar.back()
    dialog.toolbar.back()
    view_name = 'Acceptance T3'
    monkeypatch.setattr(QInputDialog, 'getText', lambda *args, **kwargs: (view_name, True))
    button = next(b for b in dialog.findChildren(QPushButton) if b.text() == i18n.tr('Сохранить вид…'))
    button.click()
    snapshot = deepcopy(project.view['map_views'][view_name])
    wells['filter'] = {}
    dialog.saved_view.setCurrentIndex(dialog.saved_view.findData(view_name))
    dialog.saved_view.activated.emit(dialog.saved_view.currentIndex())
    dialog.redraw()
    assert wells['filter']['value'] == '7120'
    assert project.view['map_views'][view_name] == snapshot
    def accept_join():
        modal = QApplication.activeModalWidget()
        assert modal.windowTitle() == i18n.tr('Присоединить области к пробам')
        modal.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()
    QTimer.singleShot(0, accept_join)
    dialog.spatial_join_dialog()
    assert not warnings, warnings
    assert len(project.columns) == 6
    status = project.columns[-1]
    assert status.metadata['spatial_join']['diagnostics']['matched'] == expected
    owner.undo_action.trigger()
    assert len(project.columns) == 4
    owner.redo_action.trigger()
    assert len(project.columns) == 6
    dialog.redraw()
    for suffix in ('png', 'svg', 'pdf'):
        path = tmp_path / f'acceptance-map.{suffix}'
        monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args, chosen=str(path), **kwargs: (chosen, ''))
        dialog.export_button.trigger()
        assert path.stat().st_size > 1000
        assert not warnings, warnings
    svg = ET.parse(tmp_path / 'acceptance-map.svg').getroot()
    assert float(svg.attrib['width'].removesuffix('pt')) == pytest.approx(dialog.spec.width_mm / 25.4 * 72)
    project_path = tmp_path / 'acceptance.apyro'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args, **kwargs: (str(project_path), ''))
    owner.actions[i18n.tr('Сохранить')].trigger()
    assert project_path.is_file()
    owner.actions[i18n.tr('Новый проект')].trigger()
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args, **kwargs: (str(project_path), ''))
    owner.actions[i18n.tr('Открыть проект…')].trigger()
    owner.actions[i18n.tr('Карты…')].trigger()
    restored = owner.map_window
    restored.redraw()
    assert restored.report is not None, restored.status.text()
    assert len(restored.report['points']) == expected
    assert owner.model.project.view['map_views'][view_name] == snapshot
    assert len(owner.model.project.columns) == 6
    assert hashes == {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in hashes}
    summary = dict(source_rows=32651, scoped_rows=1841, located_rows=expected,
                   visible_wells=42, source_files_unchanged=True,
                   checks=['import_form', 'layer_import', 'crs_search', 'sample_binding', 'age_filter',
                           'layer_filter', 'category_editor', 'layer_order', 'labels', 'feature_card',
                           'zoom', 'saved_view', 'spatial_join', 'undo_redo', 'png_svg_pdf', 'project_save_reopen'])
    (tmp_path / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    restored.close()
    owner.may_replace = lambda: True
    owner.dirty = False
