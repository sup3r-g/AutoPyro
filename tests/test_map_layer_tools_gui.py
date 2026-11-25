from copy import deepcopy
import json

import pytest
from shapely.geometry import Point, Polygon, mapping

from autopyro.io import build_project
from autopyro.ui.window import MainWindow


@pytest.fixture
def map_window(qtbot):
    project = build_project(['Well', 'X', 'Y', 'TOC', 'Depth'],
                            [['A', 1, 1, 1, 100], ['B', 0, 5, 2, 200], ['C', 9, 9, 3, 300],
                             ['D', 3, 3, 4, 400], ['E', 20, 20, 5, 500], ['F', None, None, None, 600]],
                            list(range(6)), [(0, 'Well', 'text', ''), (1, 'X', 'number', 'm'),
                                             (2, 'Y', 'number', 'm'), (3, 'TOC', 'number', ''), (4, 'Depth', 'number', 'm')], {})
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.may_replace = lambda: True
    owner.show_maps()
    dialog = owner.map_window
    ids = {c.name: c.id for c in project.columns}
    dialog.data.load(dict(x_id=ids['X'], y_id=ids['Y'], source_crs='EPSG:32631', target_crs='EPSG:32631'))
    dialog.redraw()
    assert dialog.report is not None, dialog.status.text()
    yield owner, dialog, ids
    from shiboken6 import isValid
    if isValid(dialog):
        dialog.close()
    owner.dirty = False


def layer(identifier, geometries):
    return dict(id=identifier, name=identifier, crs='EPSG:32631', visible=True,
                color='#333333', fill='#aaccee', alpha=.6, line_width=1,
                features=[dict(id=str(i), type='Feature', geometry=mapping(geometry),
                               properties={'name': f'Area {i}', 'value': 1 + i * 9}) for i, geometry in enumerate(geometries)])


def test_map_view_pins_auto_crs_and_hides_new_layers(map_window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    owner, dialog, _ = map_window
    original = layer('original', [Point(3, 60)])
    original['crs'] = 'EPSG:4326'
    dormant = layer('dormant', [Point(2, 2)])
    dormant['visible'] = False
    dialog.layers = [original, dormant]
    dialog.sync_layers()
    dialog.data.load(dict(target_crs='auto', point_layer='dormant'))
    dialog.redraw()
    assert not dialog.point_layer.isEnabled()
    assert dialog.report['spec']['target_crs'] == 'EPSG:4326'
    dialog.save_named_view('Original')
    views = owner.model.project.view['map_views']
    saved = deepcopy(views['Original'])
    assert saved['spec']['target_crs'] == 'EPSG:4326'

    owner.analysis_scope.set_mode('all')
    extra = layer('extra', [Point(9, 9)])
    dialog.layers.insert(0, extra)
    dialog.sync_layers()
    dialog.redraw()
    assert dialog.report['spec']['target_crs'] == 'EPSG:32631'
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.No)
    dialog.save_named_view('Original')
    dialog.delete_named_view()
    assert views['Original'] == saved
    dialog.load_named_view()
    dialog.redraw()
    assert [(l['id'], l['visible']) for l in dialog.layers] == [
        ('original', True), ('dormant', False), ('extra', False)]
    assert dialog.report['spec']['target_crs'] == 'EPSG:4326'
    assert owner.analysis_scope.mode == saved['spec']['scope'] == 'filtered'
    assert [lid for lid, _, _ in dialog.result['layer_features']] == ['original']
    assert views['Original'] == saved

    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.Yes)
    dialog.save_named_view('Original')
    assert len(views['Original']['layers']) == 3
    replacement = deepcopy(views['Original'])
    owner.model.undo.undo()
    assert views['Original'] == saved
    owner.model.undo.redo()
    assert views['Original'] == replacement
    dialog.delete_named_view()
    assert 'Original' not in views
    assert not dialog.delete_view_button.isEnabled()
    owner.model.undo.undo()
    assert views['Original'] == replacement
    owner.model.undo.redo()
    assert 'Original' not in views


def test_saved_view_requires_only_visible_and_analytical_layers(map_window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    _, dialog, ids = map_window
    overview = layer('overview', [Point(10, 10)])
    reference = layer('reference', [Point(20, 20)])
    reference['visible'] = False
    sites = layer('sites', [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])])
    sites['visible'] = False
    dialog.layers = [overview, reference, sites]
    dialog.sync_layers()
    dialog.sites.load(dict(site_layer='sites', site_value=ids['TOC']))
    dialog.redraw()
    dialog.save_named_view('Overview')
    dialog.layer_table.selectRow(1)
    dialog.remove_layer_button.click()
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warnings.append(args[2]))

    dialog.load_named_view()
    dialog.redraw()
    assert not warnings
    assert dialog.report and len(dialog.report['sites']) == 1
    assert [layer['id'] for layer in dialog.layers] == ['overview', 'sites']

    dialog.layer_table.selectRow(1)
    dialog.remove_layer_button.click()
    dialog.load_named_view()
    assert len(warnings) == 1


def test_map_rejects_invalid_layer_cells_and_clears_unselected_card(map_window):
    _, dialog, _ = map_window
    item_layer = layer('features', [Point(1, 1), Point(2, 2)])
    item_layer['filter'] = dict(field='value', op='eq', value='1')
    dialog.layers = [item_layer]
    dialog.sync_layers()
    dialog.redraw()
    report = dialog.report
    for col, invalid in ((1, 'bad-color'), (2, '#12'), (3, 'oops'), (3, 'nan'),
                         (3, '1.1'), (4, '-1'), (4, 'inf')):
        item = dialog.layer_table.item(0, col)
        previous = item.text()
        item.setText(invalid)
        assert item.text() == previous
        assert item.toolTip() == dialog.status.text()
        assert dialog.report is report and dialog.export_button.isEnabled()
    dialog.layer_table.item(0, 3).setText('0.4')
    dialog.layer_table.item(0, 0).setText('Renamed')
    dialog.redraw()
    assert item_layer['alpha'] == .4 and item_layer['name'] == 'Renamed'
    assert dialog.report is not None
    dialog.attribute_table.selectRow(0)
    assert dialog.feature_card.toPlainText()
    dialog.attribute_table.clearSelection()
    assert not dialog.feature_card.toPlainText() and not dialog.feature_highlights
    dialog.zoom_to_attribute(dialog.attribute_model.index(1, 0))
    assert 'Объект скрыт' in dialog.status.text()


def test_map_export_worker_keeps_selected_snapshot_when_map_changes(map_window, tmp_path, monkeypatch):
    from threading import Event
    from PySide6.QtCore import QThread, QTimer
    from PySide6.QtWidgets import QFileDialog
    from autopyro.ui.dialogs import map_files as map_module

    _, dialog, ids = map_window
    path = tmp_path / 'map.geojson'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), ''))
    released = Event()
    worker_threads = []
    write_map = map_module.export_map

    def export_worker(path, report):
        worker_threads.append(QThread.currentThread())
        if not released.wait(5):
            raise RuntimeError('Qt timer did not run during export')
        write_map(path, report)

    def change_map():
        dialog.filters.load(dict(depth_id=ids['Depth'], depth_min=300))
        dialog.schedule()
        released.set()

    monkeypatch.setattr(map_module, 'export_map', export_worker)
    QTimer.singleShot(0, change_map)
    dialog.export()
    assert worker_threads and worker_threads[0] != dialog.thread()
    assert len(json.loads(path.read_text(encoding='utf-8'))['features']) == 5
    dialog.redraw()
    assert len(dialog.report['points']) == 3
    assert dialog.export_button.isEnabled()


def test_map_blank_stratigraphy_filter_and_status_use_full_scope(map_window):
    _, dialog, ids = map_window
    dialog.filters.load(dict(strat_id=ids['Well'], strat_value=''))
    dialog.redraw()
    assert len(dialog.report['points']) == 5
    dialog.filters.load(dict(depth_id=ids['Depth'], depth_min=300))
    dialog.redraw()
    assert len(dialog.report['points']) == 3
    assert 'Проб на карте: 3 из 6' in dialog.status.text()
    assert 'После фильтров карты: 4' in dialog.status.text()
    assert 'Без привязки: 1' in dialog.status.text()


def test_large_map_label_placement_runs_in_background_during_navigation(map_window, qtbot, monkeypatch):
    from threading import Event
    from PySide6.QtCore import QThread, QTimer
    from PySide6.QtWidgets import QApplication
    from autopyro.ui.dialogs import map_rendering as ui

    owner, dialog, _ = map_window
    owner.show()
    qtbot.waitUntil(dialog.canvas.isVisible)
    labelled = layer('labels', [Point(i, i % 20) for i in range(2000)])
    labelled['label_by'] = 'name'
    dialog.layers = [labelled]
    dialog.layout_form.load({'scale_bar': True})
    dialog.redraw()
    assert dialog.report is not None, dialog.status.text()
    previous_labels = list(dialog.result['layer_labels'])
    previous_scale = dialog.result['decorations']['scale_m']
    placement = ui.add_layer_labels
    release = Event()
    threads = []

    def place(*args, **kwargs):
        threads.append(QThread.currentThread())
        assert threads[-1] is not QApplication.instance().thread()
        assert release.wait(5)
        return placement(*args, **kwargs)

    monkeypatch.setattr(ui, 'add_layer_labels', place)
    dialog.result['axes'].set_xlim(-1, 12)
    dialog.result['axes'].set_ylim(-1, 20)
    dialog.update_map_layout()
    assert not threads
    QTimer.singleShot(10, release.set)
    dialog.render_map_view()
    assert len(threads) == 1, dialog.status.text()
    assert dialog.result['layer_labels']
    assert all(label.axes is None for label in previous_labels)
    assert all(label.axes is dialog.result['axes'] for label in dialog.result['layer_labels'])
    assert dialog.result['decorations']['scale_m'] < previous_scale
    current_labels = list(dialog.result['layer_labels'])
    dialog.result['axes'].set_xlim(20, 40)
    dialog.update_map_layout()
    dialog.render_map_view()
    assert len(threads) == 2
    assert all(label.axes is None for label in current_labels)
    assert all(label.axes is dialog.result['axes'] for label in dialog.result['layer_labels'])
    dialog.render_map_view()
    assert len(threads) == 2

    def unavailable(*args, **kwargs):
        raise ValueError('Scale unavailable')

    monkeypatch.setattr(ui, 'add_map_decorations', unavailable)
    dialog.result['axes'].set_xlim(30, 50)
    dialog.render_map_view()
    assert dialog.report is not None
    assert dialog.result['layer_labels']
    assert dialog.result['decorations'] == {}
    assert dialog.status.text() == 'Scale unavailable'


def test_large_map_pick_checks_nearby_features_after_zoom(map_window, monkeypatch):
    from types import SimpleNamespace
    from shapely.geometry import box
    from autopyro.ui.dialogs import map_selection as ui

    _, dialog, _ = map_window
    dialog.layers = [layer('polygons', [box(i, 0, i + .8, 1) for i in range(2500)])]
    dialog.sync_layers()
    dialog.redraw()
    assert dialog.report is not None, dialog.status.text()
    axes = dialog.result['axes']
    axes.set_xlim(2490, 2502)
    axes.set_ylim(-2, 2)
    transformed = []
    transform = ui.shapely.transform

    def track(geometry, *args, **kwargs):
        transformed.append(geometry)
        return transform(geometry, *args, **kwargs)

    monkeypatch.setattr(ui.shapely, 'transform', track)
    px, py = axes.transData.transform((2499.4, .5))
    event = SimpleNamespace(x=px, y=py, button=1, inaxes=axes)
    dialog.feature_press(event)
    dialog.feature_release(event)
    assert len(transformed) < 5
    assert dialog.attribute_table.selectionModel().selectedRows()[0].row() == 2499
    assert dialog.attribute_model.feature_rows['2499'] == 2499


def test_large_map_prepares_and_draws_off_gui_thread(map_window, qtbot, monkeypatch):
    from threading import Event
    from PySide6.QtCore import QThread, QTimer
    from PySide6.QtWidgets import QApplication
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from shapely.geometry import box
    from autopyro.ui.dialogs import map_rendering as ui

    owner, dialog, _ = map_window
    owner.show()
    qtbot.waitUntil(dialog.canvas.isVisible)
    dialog.layers = [layer('polygons', [box(i % 100, i // 100, i % 100 + .8, i // 100 + .8)
                                       for i in range(3000)])]
    original_build, original_render = ui.build_map, ui.render_map
    original_draw = FigureCanvasAgg.draw
    stages, ticks = [], []
    prepare_ready, render_ready = Event(), Event()

    def build(*args, **kwargs):
        assert QThread.currentThread() is not QApplication.instance().thread()
        stages.append('prepare')
        assert prepare_ready.wait(5)
        return original_build(*args, **kwargs)

    def render(*args, **kwargs):
        assert QThread.currentThread() is not QApplication.instance().thread()
        stages.append('render')
        assert render_ready.wait(5)
        return original_render(*args, **kwargs)

    monkeypatch.setattr(ui, 'build_map', build)
    monkeypatch.setattr(ui, 'render_map', render)

    def paint(canvas):
        assert QThread.currentThread() is not QApplication.instance().thread()
        stages.append('paint')
        return original_draw(canvas)

    monkeypatch.setattr(FigureCanvasAgg, 'draw', paint)
    timer = QTimer(dialog)
    timer.setInterval(10)
    def heartbeat():
        ticks.append(len(stages))
        if len(stages) == 1:
            prepare_ready.set()
        if len(stages) == 2:
            render_ready.set()

    timer.timeout.connect(heartbeat)
    timer.start()
    dialog.redraw()
    timer.stop()
    assert dialog.report is not None, dialog.status.text()
    assert stages == ['prepare', 'render', 'paint']
    assert 1 in ticks and 2 in ticks
    assert len(dialog.result['layer_features']) == 3000
    assert dialog.export_button.isEnabled()
    point = dialog.report['points'][0]
    x, y = dialog.result['axes'].transData.transform((point['x'], point['y']))
    click = MouseEvent('button_press_event', dialog.canvas, x, y, button=1)
    dialog.canvas.callbacks.process('button_press_event', click)
    assert point['id'] in owner.sample_selection.ids()
    dialog.result['axes'].set_xlim(0, 20)
    dialog.result['axes'].set_ylim(0, 20)
    dialog.update_map_layout()
    dialog.render_map_view()
    assert stages == ['prepare', 'render', 'paint', 'paint']
    assert dialog.result['axes'].get_xlim() == (0, 20)
    assert dialog.selection_overlay.selected_ids == {point['id']}


def test_large_map_discards_frame_changed_during_render(map_window, qtbot, monkeypatch):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from shapely.geometry import box
    from autopyro.ui.dialogs import map_rendering as ui

    _, dialog, _ = map_window
    dialog.layers = [layer('polygons', [box(i, 0, i + .8, 1) for i in range(2000)])]
    original_render = ui.render_map
    from threading import Event
    release = Event()

    def slow_render(*args, **kwargs):
        assert release.wait(5)
        return original_render(*args, **kwargs)

    def change():
        assert QApplication.activeModalWidget() is None
        assert dialog.isEnabled()
        dialog.style.load({'color': '#ff0000'})
        dialog.schedule()
        release.set()

    monkeypatch.setattr(ui, 'render_map', slow_render)
    QTimer.singleShot(50, change)
    dialog.redraw()
    assert dialog.report is None
    qtbot.waitUntil(lambda: dialog.report is not None, timeout=15000)
    assert dialog.report['spec']['color'] == '#ff0000'


def test_large_map_render_error_clears_stale_output_and_can_retry(map_window, monkeypatch):
    from shapely.geometry import box
    from autopyro.ui.dialogs import map_rendering as ui

    _, dialog, _ = map_window
    dialog.layers = [layer('polygons', [box(i, 0, i + .8, 1) for i in range(2000)])]
    render = ui.render_map

    def fail(*args, **kwargs):
        raise ValueError('Rendering failed')

    monkeypatch.setattr(ui, 'render_map', fail)
    dialog.redraw()
    assert dialog.report is None
    assert not dialog.export_button.isEnabled()
    assert not dialog.map_render_running
    assert 'Rendering failed' in dialog.status.text()
    monkeypatch.setattr(ui, 'render_map', render)
    dialog.redraw()
    assert dialog.report is not None
    assert dialog.export_button.isEnabled()


def test_replacing_project_during_large_map_render_does_not_install_old_map(map_window, monkeypatch, qtbot):
    from threading import Event
    from PySide6.QtCore import QTimer
    from autopyro.core import Project
    from autopyro.ui.dialogs import map_rendering as ui
    from shapely.geometry import box
    from shiboken6 import isValid

    owner, dialog, _ = map_window
    dialog.layers = [layer('polygons', [box(i, 0, i + .8, 1) for i in range(2000)])]
    render = ui.render_map
    release = Event()
    replacement = Project()
    replaced_during_render = []

    def slow(*args, **kwargs):
        assert release.wait(5)
        return render(*args, **kwargs)

    def replace_project():
        try:
            owner.set_project(replacement)
            replaced_during_render.append(isValid(dialog) and dialog.closed and dialog.map_render_running)
        finally:
            release.set()

    monkeypatch.setattr(ui, 'render_map', slow)
    QTimer.singleShot(50, replace_project)
    dialog.redraw()
    assert owner.model.project is replacement
    assert owner.map_window is None
    assert replaced_during_render == [True]
    assert dialog.report is None
    qtbot.waitUntil(lambda: not isValid(dialog))


def test_replacing_project_retires_map_and_running_regional_analysis(map_window, qtbot):
    from threading import Event
    from shiboken6 import isValid
    from autopyro.core import Project

    owner, dialog, _ = map_window
    dialog.open_regional_geochemistry()
    regional = dialog.regional_window
    release = Event()
    results = []
    regional.run_task(lambda: release.wait(5), results.append)
    try:
        owner.dirty = False
        owner.may_replace = owner.session.may_replace
        owner.new_project()
        assert owner.map_window is None
        assert isValid(dialog) and isValid(regional)
        assert dialog.isHidden() and regional.isHidden()
    finally:
        release.set()
    qtbot.waitUntil(lambda: not isValid(dialog) and not isValid(regional))
    assert not results


def test_replacing_project_does_not_destroy_running_modal_map_analysis(map_window, monkeypatch, qtbot):
    from threading import Event
    from PySide6.QtCore import QTimer
    from shiboken6 import isValid
    from autopyro.core import Project
    from autopyro.ui.dialogs import map as ui
    from autopyro.ui.tasks import TaskDialog

    owner, dialog, _ = map_window
    release = Event()
    results, children = [], []

    class SpatialDialog(TaskDialog):
        def __init__(self, report, parent):
            super().__init__(parent)
            children.append(self)
            self.run_task(lambda: release.wait(5), results.append)

    replacement = Project()
    monkeypatch.setattr(ui, 'SpatialInterpolationDialog', SpatialDialog)
    QTimer.singleShot(0, lambda: owner.set_project(replacement))
    try:
        dialog.open_interpolation()
        assert owner.model.project is replacement and owner.map_window is None
        assert isValid(children[0]) and children[0].task_running
    finally:
        release.set()
    qtbot.waitUntil(lambda: not isValid(dialog) and not isValid(children[0]))
    assert not results


def test_point_layer_filter_explains_why_hidden_wells_still_bind_samples(map_window):
    _, dialog, ids = map_window
    wells = layer('wells', [Point(1, 1), Point(2, 2)])
    for feature, key in zip(wells['features'], ('A', 'B'), strict=True):
        feature['properties']['well'] = key
    wells['filter'] = dict(field='well', op='eq', value='B')
    dialog.layers = [wells]
    dialog.sync_layers()
    dialog.data.load({'location_mode': 'layer', 'point_layer': 'wells', 'sample_key': ids['Well']})
    dialog.sync_layer_fields()
    dialog.data.load({'layer_key': 'well'})
    dialog.redraw()
    assert len(dialog.report['points']) == 2
    assert len(dialog.result['layer_features']) == 1
    assert 'использует и скрытые скважины' in dialog.layer_binding_hint.text()
    assert not dialog.layer_binding_hint.isHidden()

    dialog.location.setCurrentIndex(dialog.location.findData('columns'))
    assert dialog.layer_binding_hint.isHidden()


def test_layer_filter_numeric_style_order_and_saved_view(map_window, tmp_path):
    owner, dialog, ids = map_window
    dialog.layers = [layer('top', [Point(1, 1), Point(9, 9)]), layer('bottom', [Point(2, 2)])]
    original = deepcopy(dialog.layers[0]['features'])
    dialog.sync_layers()
    dialog.layer_display.load(dict(color_by='value', color_mode='numeric', color_scale='log', color_min=1, color_max=10))
    dialog.layer_rules.load(dict(filter_field='name', filter_op='eq', filter_value='Area 0', label_by='name', legend_hidden=False))
    dialog.layer_settings_changed()
    dialog.redraw()
    assert dialog.report is not None, dialog.status.text()
    assert len(dialog.result['layer_features']) == 2
    assert len(dialog.result['layer_labels']) == 1
    dialog.save_named_view('T3 test')
    saved = deepcopy(owner.model.project.view['map_views']['T3 test'])
    assert 'features' not in saved['layers'][0]
    dialog.layer_table.reordered.emit(0, 1)
    assert [item['id'] for item in dialog.layers] == ['bottom', 'top']
    dialog.layers[1]['filter'] = {}
    dialog.saved_view.setCurrentIndex(dialog.saved_view.findData('T3 test'))
    dialog.load_named_view()
    dialog.redraw()
    assert dialog.report is not None, dialog.status.text()
    assert [item['id'] for item in dialog.layers] == ['top', 'bottom']
    assert dialog.layers[0]['filter']['value'] == 'Area 0'
    assert dialog.layers[0]['features'] == original
    assert owner.model.project.view['map_views']['T3 test'] == saved
    owner.model.project.save(tmp_path / 'views.apyro')
    from autopyro.core import Project
    restored = Project.load(tmp_path / 'views.apyro')
    assert restored.view['map_views']['T3 test'] == saved


def test_layer_only_geojson_exports_visible_filtered_geometry(map_window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    _, dialog, _ = map_window
    regions = layer('regions', [Point(500000, 0), Point(501000, 0)])
    regions['filter'] = dict(field='name', op='eq', value='Area 0')
    hidden = dict(layer('hidden', [Point(502000, 0)]), visible=False)
    dialog.layers = [regions, hidden]
    dialog.sync_layers()
    dialog.data.load({'location_mode': 'layers'})
    dialog.redraw()
    assert len(dialog.result['layer_features']) == 1
    path = tmp_path / 'map.geojson'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), ''))
    dialog.export()
    features = json.loads(path.read_text(encoding='utf-8'))['features']
    assert len(features) == 1
    assert features[0]['properties']['name'] == 'Area 0'
    assert features[0]['geometry']['coordinates'] == pytest.approx([3, 0])

    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warnings.append(args[2]))
    regions['visible'] = False
    dialog.redraw()
    before = path.read_bytes()
    dialog.export()
    assert warnings
    assert path.read_bytes() == before


def test_map_export_describes_contents_for_each_map_mode(map_window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    _, dialog, _ = map_window
    labels = []

    def cancel(*args):
        labels.extend(args[3].split(';;'))
        return '', ''

    monkeypatch.setattr(QFileDialog, 'getSaveFileName', cancel)
    dialog.export()
    assert len(labels) == 6
    assert 'TIFF — изображение без геопривязки' in labels[3]
    assert 'JSON — данные выборки, все слои и настройки' in labels[4]
    assert 'GeoJSON WGS84 — пробы со всеми полями и площадки' in labels[5]

    dialog.layers = [layer('locations', [Point(1, 1)])]
    dialog.data.load({'location_mode': 'layers'})
    dialog.redraw()
    labels.clear()
    dialog.export()
    assert 'GeoJSON WGS84 — видимые векторные объекты с атрибутами' in labels[5]

    path = tmp_path / 'stale-map.geojson'
    warnings = []

    def change_map(*args):
        dialog.schedule()
        return str(path), labels[5]

    monkeypatch.setattr(QFileDialog, 'getSaveFileName', change_map)
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warnings.append(args[2]))
    dialog.export()
    assert warnings == ['Карта изменилась. Откройте экспорт ещё раз.']
    assert not path.exists()


def test_numeric_stratigraphy_code_entered_in_map_filter(map_window, qtbot):
    owner, dialog, ids = map_window
    dialog.filters.load({'strat_id': ids['TOC']})
    qtbot.keyClicks(dialog.filters.fields['strat_value'][0], '1,0')
    dialog.redraw()
    assert dialog.report['row_ids'] == [owner.model.project.rows[0].id]


def test_failed_map_redraw_clears_previous_aggregates(map_window):
    _, dialog, ids = map_window
    dialog.layers = [layer('regions', [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])])]
    dialog.sync_layers()
    dialog.sites.load({'site_layer': 'regions', 'site_value': ids['TOC']})
    dialog.redraw()
    assert dialog.site_table.rowCount() == 1

    dialog.data.load({'y_id': ids['X']})
    dialog.redraw()
    assert dialog.report is None
    assert dialog.site_table.rowCount() == 0
    assert not dialog.export_button.isEnabled()

    dialog.data.load({'y_id': ids['Y']})
    dialog.redraw()
    assert dialog.site_table.rowCount() == 1


@pytest.mark.parametrize('failure', [False, True])
def test_layer_import_keeps_gui_responsive_and_commits_only_success(map_window, qtbot, monkeypatch, failure):
    from threading import Event
    from PySide6.QtCore import QThread, QTimer
    from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox
    from autopyro.ui.dialogs import map_files as ui

    owner, dialog, _ = map_window
    release = Event()
    progress_seen = []
    imported = layer('background', [Point(1, 1)])
    imported['source'] = {}
    before = deepcopy(owner.model.project.view.get('map_layers', []))
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: ('test.geojson', ''))
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args, **kwargs: ('test', True))

    def list_background(path):
        assert QThread.currentThread() is not QApplication.instance().thread()
        return [('test', 'Point')]

    def read_background(*args):
        assert QThread.currentThread() is not QApplication.instance().thread()
        assert release.wait(5), 'The GUI did not process the progress timer'
        if failure:
            raise ValueError('Broken layer')
        return imported

    timer = QTimer(dialog)
    timer.setInterval(10)

    def observe_progress():
        modal = QApplication.activeModalWidget()
        if modal is not None and modal.windowTitle() == 'Импорт слоя…':
            progress_seen.append(True)
            release.set()

    timer.timeout.connect(observe_progress)
    warnings = []
    monkeypatch.setattr(ui, 'list_layers', list_background)
    monkeypatch.setattr(ui, 'read_layer', read_background)
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warnings.append(args[2]))
    timer.start()
    try:
        dialog.import_layer()
    finally:
        release.set()
        timer.stop()
    assert progress_seen
    if failure:
        assert warnings == ['Broken layer']
        assert not dialog.layers
        assert owner.model.project.view.get('map_layers', []) == before
    else:
        assert not warnings
        assert dialog.layers == [imported]


def test_background_import_can_retry_with_declared_crs(map_window, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QInputDialog, QDialog
    from autopyro.ui.dialogs import map_files as ui

    _, dialog, _ = map_window
    imported = layer('declared', [Point(1, 1)])
    imported['source'] = {}
    calls = []
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: ('test.geojson', ''))
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args, **kwargs: ('test', True))
    monkeypatch.setattr(ui, 'list_layers', lambda path: [('test', 'Point')])

    def read(path, name, declared=None):
        calls.append(declared)
        if declared is None:
            raise ValueError('CRS слоя неизвестна; задайте её явно')
        return imported

    original_exec = QDialog.exec

    def choose_crs(widget):
        if widget.windowTitle() == 'Неизвестная CRS':
            return QDialog.DialogCode.Accepted
        return original_exec(widget)

    monkeypatch.setattr(ui, 'read_layer', read)
    monkeypatch.setattr(QDialog, 'exec', choose_crs)
    dialog.import_layer()
    assert calls == [None, 'EPSG:4326']
    assert dialog.layers == [imported]


def test_saved_view_restores_original_layer_after_reimport(map_window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

    owner, dialog, _ = map_window
    path = tmp_path / 'wells.geojson'
    path.write_text(json.dumps(dict(type='FeatureCollection', features=[
        dict(type='Feature', geometry=mapping(Point(3, 0)), properties={'name': 'Well A'}),
    ])), encoding='utf-8')
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(path), ''))
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args, **kwargs: (args[3][0], True))
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warnings.append(args[2]))
    dialog.import_layer()
    original = dialog.layers[0]['id']
    dialog.layers[0]['color'] = '#ff0000'
    dialog.save_named_view('Wells')
    snapshot = owner.model.project.view['map_views']['Wells']
    snapshot.pop('layer_sources')  # Existing views gain the source reference on removal.
    dialog.layer_table.selectRow(0)
    dialog.remove_layer()
    assert not dialog.layers
    dialog.import_layer()
    assert dialog.layers[0]['id'] == original
    assert dialog.layers[0]['color'] != '#ff0000'
    dialog.load_named_view()
    dialog.redraw()
    assert not warnings
    assert dialog.layers[0]['color'] == '#ff0000'
    assert dialog.result['layer_features'][0][0] == original


def test_aggregates_show_boundary_samples_that_spatial_join_leaves_unassigned(map_window):
    owner, dialog, ids = map_window
    dialog.layers = [layer('regions', [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])])]
    dialog.sync_layers()
    dialog.sites.load({'site_layer': 'regions', 'site_value': ids['TOC']})
    dialog.redraw()
    assert dialog.report['sites'][0]['n_boundary'] == 1
    assert dialog.site_table.item(0, 2).text() == '4'
    assert dialog.site_table.horizontalHeaderItem(3).text() == 'Из N на границе'
    assert dialog.site_table.item(0, 3).text() == '1'

    diagnostics = dialog.apply_spatial_join('regions', ['name'])
    assert diagnostics['boundary'] == 1
    boundary_sample = owner.model.project.rows[1]
    attribute, status = owner.model.project.columns[-2:]
    assert boundary_sample.values[attribute.id] is None
    assert boundary_sample.values[status.id] == 'На границе'


def test_spatial_join_adds_auditable_columns_and_can_be_undone(map_window, monkeypatch):
    from PySide6.QtCore import QThread
    from autopyro.ui.widgets import map_layers

    owner, dialog, ids = map_window
    regions = layer('regions', [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], holes=[[(2, 2), (4, 2), (4, 4), (2, 4)]]),
                                Polygon([(8, 8), (12, 8), (12, 12), (8, 12)])])
    dialog.layers = [regions]
    dialog.sync_layers()
    dialog.redraw()
    before = [deepcopy(row.values) for row in owner.model.project.rows]
    assign_regions = map_layers.assign_regions

    def assign_in_worker(*args):
        assert QThread.currentThread() != dialog.thread()
        return assign_regions(*args)

    monkeypatch.setattr(map_layers, 'assign_regions', assign_in_worker)
    diagnostics = dialog.apply_spatial_join('regions', ['name'])
    assert diagnostics == dict(matched=1, boundary=1, overlap=1, outside=2, no_location=1)
    columns = owner.model.project.columns[-2:]
    assert columns[0].metadata['spatial_join']['layer_id'] == 'regions'
    assert owner.model.project.rows[0].values[columns[0].id] == 'Area 0'
    assert all(row.values[columns[0].id] is None for row in owner.model.project.rows[1:])
    assert [row.values[columns[1].id] for row in owner.model.project.rows][1:3] == ['На границе', 'Перекрытие']
    assert [{key: row.values[key] for key in original} for row, original in zip(owner.model.project.rows, before)] == before
    owner.model.undo.undo()
    assert len(owner.model.project.columns) == 5
    assert [row.values for row in owner.model.project.rows] == before
    owner.model.undo.redo()
    assert len(owner.model.project.columns) == 7


def test_feature_card_displays_linked_measurements_and_escaped_values(map_window):
    owner, dialog, ids = map_window
    owner.model.project.rows[1].values[ids['Well']] = 'A'
    wells = layer('wells', [Point(1, 1)])
    wells['features'][0]['properties'].update(well='A', name='<script>alert(1)</script>')
    dialog.layers = [wells]
    dialog.sync_layers()
    dialog.data.load(dict(location_mode='layer', point_layer='wells', sample_key=ids['Well'], layer_key='well'))
    dialog.redraw()
    assert dialog.report is not None, dialog.status.text()
    dialog.attribute_table.selectRow(0)
    text = dialog.feature_card.toPlainText()
    assert 'Связанных замеров: 2' in text
    assert '100–200' in text
    assert '<script>' in text
    assert '&lt;script&gt;' in dialog.feature_card.toHtml()


@pytest.mark.parametrize('show_aggregates', [False, True])
def test_map_legend_toggle_hides_numeric_scales_without_changing_colors(map_window, qtbot, show_aggregates):
    _, dialog, ids = map_window
    regions = layer('regions', [Polygon([(-1, -1), (21, -1), (21, 21), (-1, 21)])])
    regions.update(color_by='value', color_mode='numeric')
    dialog.layers = [regions]
    dialog.sync_layers()
    dialog.style.load({'color_by': ids['TOC'], 'color_mode': 'numeric'})
    dialog.sites.load({'site_layer': 'regions', 'site_value': ids['TOC'], 'show_aggregates': show_aggregates})
    dialog.redraw()
    assert len(dialog.figure.axes) == 3  # Map, sample/aggregate scale and layer scale.
    expected = dialog.result['artist'].get_facecolors().copy()

    legend = dialog.layout_form.fields['legend'][0]
    legend.setChecked(False)
    qtbot.waitUntil(lambda: dialog.report is not None)
    assert dialog.figure.axes == [dialog.result['axes']]
    assert dialog.result['artist'].get_facecolors() == pytest.approx(expected)

    legend.setChecked(True)
    qtbot.waitUntil(lambda: dialog.report is not None)
    assert len(dialog.figure.axes) == 3


def test_map_decorations_follow_zoom_and_legend_placement_is_selectable(map_window):
    owner, dialog, ids = map_window
    dialog.layers = [layer('points', [Point(1, 1), Point(9, 9)])]
    dialog.sync_layers()
    dialog.layout_form.load(dict(scale_bar=True, north_arrow=True, legend_placement='lower_left'))
    assert dialog.layout_form.fields['legend_placement'][0].currentData() == 'lower_left'
    dialog.redraw()
    assert dialog.report is not None, dialog.status.text()
    initial = dialog.result['decorations']['scale_m']
    original_points = dialog.result['artist']
    original_artists = list(dialog.result['decorations']['_artists'])
    ax = dialog.result['axes']
    ax.set_xlim(1, 2)
    ax.set_ylim(1, 2)
    dialog.update_map_layout()
    assert dialog.result['decorations']['scale_m'] < initial
    assert all(artist.axes is None for artist in original_artists)
    assert dialog.result['artist'] is original_points


def test_map_status_follows_spatial_join_undo_and_redo(map_window, qtbot):
    owner, dialog, ids = map_window
    region = layer('regions', [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])])
    region['diagnostics'] = dict(missing_coordinates=2)
    dialog.layers = [region]
    dialog.sync_layers()
    dialog.redraw()
    text = dialog.status.text()
    assert 'Проб на карте: 5 из 6' in text
    assert 'Без привязки: 1' in text
    assert 'Без отображаемой геометрии: 2' in text
    assert 'input_rows' not in text and 'missing_coordinates' not in text
    dialog.apply_spatial_join('regions', ['name'])
    dialog.redraw()
    assert 'Последнее присоединение — regions' in dialog.status.text()
    assert 'Сопоставлено:' in dialog.status.text()
    first_summary = dialog.status.text()

    other = layer('other', [Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])])
    dialog.layers.append(other)
    dialog.sync_layers()
    dialog.redraw()
    dialog.apply_spatial_join('other', ['name'])
    qtbot.waitUntil(lambda: dialog.report is not None)
    assert 'Последнее присоединение — other' in dialog.status.text()

    owner.model.undo.undo()
    qtbot.waitUntil(lambda: dialog.report is not None)
    assert dialog.status.text().splitlines()[-1] == first_summary.splitlines()[-1]
    owner.model.undo.undo()
    qtbot.waitUntil(lambda: dialog.report is not None)
    assert 'Последнее присоединение' not in dialog.status.text()

    owner.model.undo.redo()
    qtbot.waitUntil(lambda: dialog.report is not None)
    assert 'Последнее присоединение — regions' in dialog.status.text()
    owner.model.undo.redo()
    qtbot.waitUntil(lambda: dialog.report is not None)
    assert 'Последнее присоединение — other' in dialog.status.text()


def test_sample_diagnostics_selects_affected_rows(map_window, qtbot):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import QApplication, QDialogButtonBox, QListWidget

    owner, dialog, ids = map_window
    dialog.filters.load({'depth_id': ids['Depth'], 'depth_min': 200})
    dialog.redraw()
    assert 'Исключено фильтрами карты: 1' in dialog.status.text()
    assert dialog.sample_diagnostics_button.isEnabled()
    assert dialog.report['diagnostics']['excluded_rows'] == {
        'depth_filter': [owner.model.project.rows[0].id],
        'coordinate:missing': [owner.model.project.rows[-1].id],
    }
    labels = []

    def select_missing():
        details = QApplication.activeModalWidget()
        listing = details.findChild(QListWidget)
        labels.extend(listing.item(i).text() for i in range(listing.count()))
        listing.setCurrentRow(1)
        buttons = details.findChild(QDialogButtonBox)
        next(button for button in buttons.buttons()
             if buttons.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole).click()

    QTimer.singleShot(0, select_missing)
    dialog.sample_diagnostics_button.click()
    assert labels == ['Вне диапазона глубин или нет числовой глубины · 1', 'Отсутствуют X или Y · 1']
    assert owner.sample_selection.ids() == {owner.model.project.rows[-1].id}


def test_join_dialog_requires_fields_and_explains_missing_polygon_layer(map_window, monkeypatch):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QDialogButtonBox, QListWidget, QMessageBox
    owner, dialog, ids = map_window
    notices = []
    monkeypatch.setattr(QMessageBox, 'information', lambda *args: notices.append(args[2]))
    dialog.spatial_join_dialog()
    assert notices and 'полигональный слой' in notices[0]
    dialog.layers = [layer('regions', [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])])]
    dialog.sync_layers()
    # This call also flushes the pending redraw caused by changing settings.
    dialog.schedule()
    def inspect():
        modal = QApplication.activeModalWidget()
        fields = modal.findChild(QListWidget)
        button = modal.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok)
        assert not fields.selectedItems()
        assert not button.isEnabled()
        fields.item(0).setSelected(True)
        assert button.isEnabled()
        fields.clearSelection()
        assert not button.isEnabled()
        modal.reject()
    QTimer.singleShot(0, inspect)
    dialog.spatial_join_dialog()
    assert len(owner.model.project.columns) == 5
