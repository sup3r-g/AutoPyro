from copy import deepcopy
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt

from autopyro.core import Project
from autopyro.io import build_project
from autopyro.map_dialog import SpatialInterpolationDialog
from autopyro.window import MainWindow


@pytest.fixture
def window(qtbot):
    project=build_project(["X","Y","TOC","Family"],[[0,0,1,"A"],[100,0,3,"B"],[0,100,3,"A"],[100,100,5,"B"],[50,50,3,"A"]],list(range(1,6)),
        [(0,"X","number","m"),(1,"Y","number","m"),(2,"TOC","number","%wt"),(3,"Family","text","")],{})
    owner=MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.may_replace=lambda:True
    yield owner
    if owner.map_window:
        owner.map_window.close()
    owner.dirty=False


def ready_map(window):
    window.show_maps()
    dialog=window.map_window
    x,y,value,family=[c.id for c in window.model.project.columns]
    dialog.data.load({"x_id":x,"y_id":y,"source_crs":"EPSG:32631","target_crs":"EPSG:32631"})
    dialog.style.load({"color_by":family,"size_by":value})
    dialog.redraw()
    assert dialog.report is not None,dialog.status.text()
    return dialog


def test_coordinate_modes_disable_unused_fields_and_preserve_settings(window, tmp_path):
    import json
    from autopyro.spatial.mapping import read_layer

    dialog = ready_map(window)
    x, y = [column.id for column in window.model.project.columns[:2]]
    path = tmp_path / 'coordinates.geojson'
    path.write_text(json.dumps({'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [3 + i / 1000, 60]},
         'properties': {'key': value}}
        for i, value in enumerate((0, 100, 50))
    ]}), encoding='utf-8')
    dialog.layers = [read_layer(path)]
    dialog.sync_layers()
    settings = dict(x_id=x, y_id=y, source_crs='EPSG:32631', target_crs='EPSG:32631',
                    sample_key=x, point_layer=dialog.layers[0]['id'], layer_key='key',
                    text_numbers=True, include_interpolated=True)
    dialog.data.load(settings)
    dialog.sync_layer_fields()
    dialog.data.load({'layer_key': 'key'})

    def check_mode(mode):
        active = {'location_mode', 'target_crs'}
        if mode == 'columns':
            active.update(('x_id', 'y_id', 'source_crs'))
        if mode == 'layer':
            active.update(('sample_key', 'point_layer', 'layer_key'))
        if mode != 'layers':
            active.update(('scope', 'text_numbers', 'include_interpolated'))
        for key, (widget, _) in dialog.data.fields.items():
            assert widget.isEnabled() == (key in active), (mode, key)
            assert widget.parameter_reset_button.isEnabled() == (key in active), (mode, key)
        values = dialog.data.values()
        assert {key: values[key] for key in settings} == settings

    for mode in ('layer', 'layers', 'columns'):
        dialog.location.setCurrentIndex(dialog.location.findData(mode))
        check_mode(mode)  # Updates immediately, before the deferred redraw.
        dialog.redraw()
        assert dialog.report is not None, dialog.status.text()
        assert dialog.report['spec']['location_mode'] == mode
        if mode != 'layers':
            assert len(dialog.report['points']) == 5

    dialog.save_named_view('Columns')
    dialog.location.setCurrentIndex(dialog.location.findData('layers'))
    dialog.load_named_view()
    check_mode('columns')  # Form.load blocks combo signals; schedule must also sync.
    dialog.redraw()
    assert len(dialog.report['points']) == 5


def test_map_first_open_guides_coordinate_setup(window, qtbot):
    window.show()
    window.show_maps()
    dialog = window.map_window
    qtbot.waitUntil(lambda: dialog.status.text() == 'Карта готова к настройке.')
    assert dialog.map_pages.currentWidget() is dialog.map_start
    assert dialog.report is None and not dialog.export_button.isEnabled()
    dialog.tabs.setCurrentIndex(1)
    window.desktop.properties_dock.hide()
    qtbot.mouseClick(dialog.configure_coordinates_button, Qt.MouseButton.LeftButton)
    assert window.desktop.properties_dock.isVisible()
    assert dialog.tabs.currentIndex() == 0
    assert window.focusWidget() is dialog.data.fields['x_id'][0]

    x, y = [column.id for column in window.model.project.columns[:2]]
    dialog.data.load({'x_id': x})
    dialog.redraw()
    assert dialog.map_pages.currentWidget() is dialog.map_start
    dialog.data.load({'y_id': y})
    dialog.redraw()
    assert 'CRS координат неизвестна' in dialog.status.text()
    dialog.data.load({'source_crs': 'EPSG:32631'})
    dialog.redraw()
    assert dialog.map_pages.currentWidget() is dialog.canvas_scroll
    assert len(dialog.report['points']) == 5


def test_map_first_open_can_import_layer_without_sample_coordinates(window, qtbot, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QInputDialog

    path = tmp_path / 'reference.geojson'
    path.write_text('{"type":"FeatureCollection","features":[{"type":"Feature",'
                    '"geometry":{"type":"Point","coordinates":[3,60]},"properties":{}}]}', encoding='utf-8')
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(path), ''))
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args, **kwargs: (args[3][0], True))
    window.show()
    window.show_maps()
    dialog = window.map_window
    qtbot.waitUntil(lambda: dialog.status.text() == 'Карта готова к настройке.')
    qtbot.mouseClick(dialog.import_first_layer_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: dialog.report is not None)
    assert dialog.map_pages.currentWidget() is dialog.canvas_scroll
    assert dialog.report['spec']['location_mode'] == 'layers'
    assert len(dialog.result['layer_features']) == 1
    assert dialog.export_button.isEnabled()


def test_map_commands_stay_discoverable_in_narrow_window(window, qtbot, monkeypatch):
    from PySide6.QtWidgets import QToolButton
    from autopyro.ui.dialogs.map import MapDialog

    invoked = []
    monkeypatch.setattr(MapDialog, 'export', lambda self: invoked.append('export'))
    monkeypatch.setattr(MapDialog, 'open_interpolation', lambda self: invoked.append('interpolate'))
    window.resize(1280, 720)
    window.show()
    dialog = ready_map(window)
    dock = window.desktop.properties_dock
    window.resizeDocks([dock], [385], Qt.Orientation.Horizontal)
    operations, = [button for button in dialog.findChildren(QToolButton) if button.text() == 'Операции карты']
    assert operations.isVisible()
    assert [action.text() for action in operations.menu().actions()] == [
        'Обновить карту', 'Запомнить видимые пределы',
        'Присоединить области к пробам', 'Геохимия по областям…',
    ]
    assert dialog.remove_layer_button.isVisible() and not dialog.remove_layer_button.isEnabled()

    buttons = {button.defaultAction(): button for button in dialog.findChildren(QToolButton)}
    export = buttons[dialog.export_button]
    interpolate = buttons[dialog.interpolate_button]
    assert export.isVisible() and export.isEnabled()
    assert interpolate.isVisible() and interpolate.isEnabled()
    qtbot.mouseClick(export, Qt.MouseButton.LeftButton)
    interpolate.setFocus()
    qtbot.keyClick(interpolate, Qt.Key.Key_Space)
    assert invoked == ['export', 'interpolate']


def test_live_selection_filter_and_project_reload(window,qtbot,tmp_path):
    dialog=ready_map(window)
    assert len(dialog.report["points"])==5
    report = dialog.report
    dialog.on_rectangle(SimpleNamespace(xdata=-1,ydata=-1),SimpleNamespace(xdata=1,ydata=1))
    assert dialog.selected_ids()=={window.model.project.rows[0].id}
    assert dialog.report is report  # Highlighting does not change the map's data snapshot.
    assert dialog.export_button.isEnabled()
    assert dialog.selection_overlay.selected_ids == {window.model.project.rows[0].id}
    dialog.redraw()
    dialog.on_pick(SimpleNamespace(artist=dialog.result["artist"],ind=[1]))
    assert dialog.selected_ids()=={window.model.project.rows[1].id}
    dialog.redraw()
    x=window.model.project.columns[0].id
    window.apply_filter([{"column":x,"operator":"gt","value":0}],"AND")
    dialog.redraw()
    assert len(dialog.report["points"])==3
    window.model.project.save(tmp_path/"map.apyro")
    restored=Project.load(tmp_path/"map.apyro")
    assert restored.view["map_current"]["target_crs"]=="EPSG:32631"
    window.set_project(restored)
    assert window.map_window is None
    assert "Карты…" in window.actions


def test_categories_edit_and_colocated_selection(window,qtbot):
    project=window.model.project
    x,y=[c.id for c in project.columns[:2]]
    project.rows[1].values[x]=project.rows[1].values[y]=0
    dialog=ready_map(window)
    dialog.on_pick(SimpleNamespace(artist=dialog.result["artist"],ind=[0]))
    assert dialog.selected_ids()=={project.rows[0].id,project.rows[1].id}
    dialog.redraw()
    dialog.category_table.item(0,1).setText("#ff0000")
    dialog.redraw()
    assert list(dialog.spec.style_maps["color"][project.columns[3].id].values())[0]["style"]=="#ff0000"


def test_map_category_editor_follows_selection_and_keeps_saved_styles(window):
    dialog = ready_map(window)
    project = window.model.project
    a, b = [row.id for row in project.rows[:2]]
    window.sample_selection.select([a])
    dialog.data.fields['scope'][0].setCurrentIndex(dialog.data.fields['scope'][0].findData('selected'))
    dialog.redraw()
    assert dialog.category_table.rowCount() == 1
    original = dialog.category_table.item(0, 1).text()
    dialog.category_table.item(0, 1).setText('#ff0000')
    dialog.redraw()
    window.sample_selection.select([b])
    dialog.redraw()
    assert dialog.category_table.rowCount() == 1
    assert dialog.category_table.item(0, 2).text() == 'B'
    window.sample_selection.select([a])
    dialog.redraw()
    assert dialog.category_table.item(0, 1).text() == '#ff0000'
    dialog.reset_category_parameter(0, 1)
    assert dialog.category_table.item(0, 1).text() == original


def test_map_category_invalid_style_preserves_map_and_can_be_corrected(window):
    dialog = ready_map(window)
    family = window.model.project.columns[3].id
    dialog.style.load({'size_by': family, 'size_mode': 'category'})
    dialog.redraw()
    for channel, invalid_values, valid_value in (
        ('color', ('not-a-color', '#12'), '#ff0000'),
        ('size', ('oops', '0', '-1', 'nan', 'inf'), '12'),
    ):
        row = next(i for i, entry in enumerate(dialog.entries) if entry[0] == channel)
        item = dialog.category_table.item(row, 1)
        original = item.text()
        styles = deepcopy(dialog.spec.style_maps)
        report, result = dialog.report, dialog.result
        for invalid in invalid_values:
            item.setText(invalid)
            assert dialog.spec.style_maps == styles
            assert item.text() == original
            assert dialog.report is report and dialog.result is result
            assert dialog.export_button.isEnabled()
            assert not dialog.timer.isActive()
            assert dialog.status.text()
            assert item.toolTip() == dialog.status.text()

        item.setText(valid_value)
        assert not item.toolTip()
        dialog.category_table.item(row, 2).setText('Updated label')
        dialog.redraw()
        assert dialog.report is not None, dialog.status.text()
        row = next(i for i, entry in enumerate(dialog.entries) if entry[0] == channel)
        assert dialog.category_table.item(row, 1).text() == str(float(valid_value) if channel == 'size' else valid_value)
        assert dialog.category_table.item(row, 2).text() == 'Updated label'
        assert window.model.project.view['map_current']['style_maps'] == dialog.spec.style_maps
        dialog.reset_category_parameter(row, 1)
        dialog.redraw()
        assert dialog.report is not None, dialog.status.text()


def test_map_fills_viewport_preserving_coordinates_selection_and_export(window, qtbot, tmp_path, monkeypatch):
    import xml.etree.ElementTree as ET
    from PySide6.QtWidgets import QFileDialog

    window.resize(1450, 950)
    window.show()
    dialog = ready_map(window)
    dialog.layout_form.load({'width_mm': 180, 'height_mm': 120, 'dpi': 150})
    dialog.redraw()
    qtbot.waitUntil(lambda: dialog.canvas.size() == dialog.canvas_scroll.viewport().size())
    axes = dialog.result['axes']
    axes.set_xlim(-10, 110)
    axes.set_ylim(-20, 120)
    selected = window.model.project.rows[0].id
    dialog.on_pick(SimpleNamespace(artist=dialog.result['artist'], ind=[0]))
    report = dialog.report
    size = dialog.canvas.size()
    window.resize(1200, 760)
    qtbot.waitUntil(lambda: dialog.canvas.size() != size)
    dialog.canvas.draw()
    assert dialog.canvas.size() == dialog.canvas_scroll.viewport().size()
    assert dialog.canvas_scroll.horizontalScrollBar().maximum() == 0
    assert dialog.canvas_scroll.verticalScrollBar().maximum() == 0
    assert axes.get_xlim() == (-10, 110)
    assert axes.get_ylim() == (-20, 120)
    assert axes.get_aspect() == 1
    assert dialog.selection_overlay.selected_ids == {selected}
    assert dialog.report is report
    assert dialog.spec.width_mm == 180 and dialog.spec.height_mm == 120
    dialog.canvas_view.mode_combo.setCurrentIndex(1)
    qtbot.waitUntil(lambda: not dialog.canvas_view.scroll.widgetResizable())
    assert dialog.canvas.width() == round(180 / 25.4 * dialog.figure._original_dpi)
    assert dialog.canvas.height() == round(120 / 25.4 * dialog.figure._original_dpi)
    dialog.canvas_view.mode_combo.setCurrentIndex(0)
    qtbot.waitUntil(lambda: dialog.canvas.size() == dialog.canvas_scroll.viewport().size())
    assert axes.get_xlim() == (-10, 110)
    assert axes.get_ylim() == (-20, 120)
    assert dialog.selection_overlay.selected_ids == {selected}
    assert dialog.report is report
    path = tmp_path / 'fitted-map.svg'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a: (str(path), 'SVG'))
    dialog.export()
    root = ET.parse(path).getroot()
    assert float(root.attrib['width'].removesuffix('pt')) == pytest.approx(180 / 25.4 * 72)
    assert float(root.attrib['height'].removesuffix('pt')) == pytest.approx(120 / 25.4 * 72)


def test_short_map_viewport_keeps_title_and_footer_clear(window, qtbot):
    window.resize(1450, 950)
    window.show()
    project = window.model.project
    x, y = [column.id for column in project.columns[:2]]
    for row in project.rows:
        row.values[x] += 500_000
        row.values[y] += 6_700_000
    dialog = ready_map(window)
    tall = dialog.canvas.height()
    window.resize(1280, 720)
    qtbot.waitUntil(lambda: dialog.canvas.height() < tall and dialog._layout_extent[2] == tuple(dialog.figure.bbox.size))
    dialog.canvas.draw()
    renderer = dialog.canvas.get_renderer()
    axes = dialog.result['axes']
    footer = dialog.result['footer'].get_window_extent(renderer)
    xlabel = axes.xaxis.label.get_window_extent(renderer)
    title = axes.title.get_window_extent(renderer)
    assert dialog.figure.bbox.ymin <= footer.ymin < footer.ymax < xlabel.ymin
    assert title.ymax <= dialog.figure.bbox.ymax


def test_map_style_edits_preserve_zoom_and_home_extent(window, qtbot):
    dialog = ready_map(window)
    axes = dialog.result['axes']
    home = axes.get_xlim(), axes.get_ylim()
    zoom = ((10, 40), (20, 50))
    axes.set_xlim(zoom[0])
    axes.set_ylim(zoom[1])
    dialog.style.load({'color': '#d55e00'})
    dialog.style.edited.emit()
    assert dialog.report is None
    qtbot.waitUntil(lambda: dialog.report is not None)
    axes = dialog.result['axes']
    assert (axes.get_xlim(), axes.get_ylim()) == zoom
    saved = window.model.project.view['map_current']
    assert all(saved[key] is None for key in ('xmin', 'xmax', 'ymin', 'ymax'))
    dialog.toolbar.home()
    assert (axes.get_xlim(), axes.get_ylim()) == home


def test_map_named_view_overrides_zoom_even_with_same_configured_limits(window):
    dialog = ready_map(window)
    limits = dict(xmin=0, xmax=90, ymin=0, ymax=90)
    dialog.layout_form.load(limits)
    dialog.redraw()
    dialog.save_named_view('Overview')
    axes = dialog.result['axes']
    axes.set_xlim(20, 50)
    axes.set_ylim(20, 50)

    dialog.load_named_view()
    dialog.redraw()
    axes = dialog.result['axes']
    assert axes.get_xlim() == (0, 90)
    assert axes.get_ylim() == (0, 90)


def test_interpolation_links_to_map_crs_picker(window, qtbot, monkeypatch):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QDialog, QPushButton
    from autopyro.ui.widgets.spatial import CRSDialog

    window.show()
    dialog = ready_map(window)
    dialog.data.load({'target_crs': 'EPSG:4326'})
    dialog.redraw()
    opened = []
    accept_selection = False

    def choose_crs(picker):
        assert QApplication.activeModalWidget() is None
        opened.append(picker.windowTitle())
        if not accept_selection:
            return QDialog.DialogCode.Rejected
        picker.selected_crs = 'EPSG:32631'
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(CRSDialog, 'exec', choose_crs)

    def request_crs():
        tool = QApplication.activeModalWidget()
        button = next(button for button in tool.findChildren(QPushButton) if button.text() == 'Выбрать CRS карты…')
        qtbot.mouseClick(button, Qt.MouseButton.LeftButton)

    QTimer.singleShot(0, request_crs)
    dialog.open_interpolation()
    assert len(opened) == 1
    assert dialog.visualization_crs.currentData() == 'EPSG:4326'

    accept_selection = True
    QTimer.singleShot(0, request_crs)
    dialog.open_interpolation()
    assert len(opened) == 2
    qtbot.waitUntil(lambda: dialog.report is not None)
    assert dialog.report['spec']['target_crs'] == 'EPSG:32631'


def test_spatial_tool_requires_explicit_run_and_preserves_input(window,qtbot):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QAccessible

    dialog=ready_map(window)
    before=deepcopy(dialog.report)
    tool=SpatialInterpolationDialog(dialog.report,dialog)
    qtbot.addWidget(tool)
    tool.show()
    assert tool.result is None and not tool.export_button.isEnabled()
    value=window.model.project.columns[2].id
    tool.form.load({"value_id":value,"context":"Same lab/method, formation and depth","max_distance":1000,"grid_size":10})
    tool.calculate()
    qtbot.waitUntil(lambda:tool.isEnabled() and tool.result is not None,timeout=15000)
    assert tool.result["diagnostics"]["loo_valid"]==5
    assert tool.idw_summary.focusPolicy() == Qt.FocusPolicy.StrongFocus
    summary = QAccessible.queryAccessibleInterface(tool.idw_summary).text(QAccessible.Text.Name)
    assert 'Узлов с прогнозом' in summary and 'LOO: прогноз в 5 из 5' in summary
    assert 'RMSE' in summary and 'не доверительный интервал' in summary
    assert tool.figure.get_figwidth() * tool.figure._original_dpi == pytest.approx(tool.canvas.width())
    assert tool.figure.get_figheight() * tool.figure._original_dpi == pytest.approx(tool.canvas.height())
    result = tool.result
    limits = [ax.get_xlim() for ax in tool.figure.axes]
    tool.canvas_view.mode_combo.setCurrentIndex(1)
    assert tool.canvas.width() == round(8 * tool.figure._original_dpi)
    assert tool.canvas.height() == round(6 * tool.figure._original_dpi)
    tool.canvas_view.mode_combo.setCurrentIndex(0)
    qtbot.waitUntil(lambda: tool.canvas.size() == tool.canvas_view.scroll.viewport().size())
    assert tool.result is result
    assert [ax.get_xlim() for ax in tool.figure.axes] == limits
    assert dialog.report==before
    assert tool.save_surface.isEnabled()
    tool.add_surface_to_map()
    dialog.redraw()
    assert dialog.report is not None
    assert [layer['raster']['band'] for layer in dialog.layers[:3]] == [
        'prediction', 'neighbors', 'nearest_distance_m']
    assert window.model.project.view['map_layers'] == dialog.layers
    assert dialog.point_layer.findData(dialog.layers[0]['id']) == -1
    dialog.zoom_to_layer(dialog.layer_table.model().index(0, 0))
    assert dialog.result['axes'].get_xlim()[0] < 0
    tool.form.fields["neighbors"][0].setValue(9)
    assert tool.result is None and not tool.export_button.isEnabled()
    assert tool.idw_summary.isHidden() and not tool.idw_summary.text()


def test_spatial_tool_reports_live_scope_and_values_with_opening_map_filters(window, qtbot, monkeypatch):
    from PySide6.QtCore import QThread
    from autopyro.ui.dialogs import spatial_input

    dialog = ready_map(window)
    project = window.model.project
    value = project.columns[2].id
    dialog.filters.load({'depth_id': value, 'depth_min': 0})
    dialog.redraw()
    opening_report = dialog.report
    tool = SpatialInterpolationDialog(opening_report, dialog)
    qtbot.addWidget(tool)
    assert 'ещё не подготовлена' in tool.input_summary.text()

    chosen = [row.id for row in project.rows[:4]]
    window.sample_selection.select(chosen)
    tool.scope.setCurrentIndex(tool.scope.findData('selected'))
    assert window.model.setData(window.model.index(0, 2), '9')
    dialog.filters.load({'depth_min': 50})
    dialog.redraw()
    assert not dialog.report['row_ids']

    build_map = spatial_input.build_map

    def prepare(*args):
        assert QThread.currentThread() != tool.thread()
        return build_map(*args)

    monkeypatch.setattr(spatial_input, 'build_map', prepare)
    tool.form.load({'value_id': value, 'context': 'Synthetic cohort', 'max_distance': 1000, 'grid_size': 10})
    tool.calculate()
    qtbot.waitUntil(lambda: tool.result is not None, timeout=15000)
    snapshot = tool.result['map_snapshot']
    assert snapshot['spec']['scope'] == 'selected'
    assert snapshot['spec']['depth_min'] == 0
    assert snapshot['row_ids'] == chosen
    assert snapshot['points'][0]['values'][value] == 9
    assert opening_report['points'][0]['values'][value] == 1
    assert tool.scope.currentText() in tool.input_summary.text()
    assert 'Проб в области: 4; после фильтров карты: 4; с координатами: 4.' in tool.input_summary.text()
    tool.scope.setCurrentIndex(tool.scope.findData('all'))
    assert 'ещё не подготовлена' in tool.input_summary.text()
    assert tool.result is None


def test_tin_surface_from_comparison_is_saved_and_exported(window, qtbot, monkeypatch, tmp_path):
    import ctypes
    import json
    from PySide6.QtWidgets import QFileDialog
    from autopyro.spatial.surfaces import _gdal_library

    dialog = ready_map(window)
    tool = SpatialInterpolationDialog(dialog.report, dialog)
    qtbot.addWidget(tool)
    original_rows = [deepcopy(row.values) for row in window.model.project.rows]
    tool.form.load({'value_id': window.model.project.columns[2].id, 'context': 'Synthetic cohort',
                    'grid_size': 10, 'block_size_km': .05})
    tool.kriging_form.load({'range_km': .2})
    tool.views.setCurrentWidget(tool.kriging_tab)
    tool.calculate_kriging(preview=False)
    qtbot.waitUntil(lambda: tool.kriging_result is not None and tool.isEnabled(), timeout=15000)
    tool.comparison_surface.setCurrentIndex(tool.comparison_surface.findData('TIN'))
    assert tool.save_surface.isEnabled() and tool.save_tiff.isEnabled()
    tool.save_surface.click()
    dialog.redraw()
    assert dialog.report is not None
    layers = dialog.layers[:3]
    assert [layer['raster']['method'] for layer in layers] == ['TIN'] * 3
    assert {value for row in layers[1]['raster']['values'] for value in row if value is not None} == {3}
    assert [row.values for row in window.model.project.rows] == original_rows
    project_path = tmp_path / 'tin.apyro'
    window.model.project.save(project_path)
    assert Project.load(project_path).view['map_layers'][:3] == layers

    output = tmp_path / 'tin.tif'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(output), 'GeoTIFF'))
    tool.save_tiff.click()
    qtbot.waitUntil(lambda: output.is_file() and tool.isEnabled(), timeout=15000)
    gdal = _gdal_library()
    metadata = gdal.GDALGetMetadataItem
    metadata.restype = ctypes.c_char_p
    metadata.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    dataset = gdal.GDALOpen(str(output).encode('utf-8'), 0)
    assert dataset
    try:
        assert metadata(dataset, b'INTERPOLATION', None) == b'TIN'
        options = json.loads(metadata(dataset, b'OPTIONS_JSON', None))
        assert options['context'] == 'Synthetic cohort'
        assert {'max_distance', 'power', 'min_neighbors', 'neighbors'}.isdisjoint(options)
        assert json.loads(metadata(dataset, b'VALIDATION_JSON', None)) == {
            'method': 'spatial_blocks', 'block_size_km': .05, 'buffer_km': 0, 'shift_km': 0}
        assert metadata(dataset, b'LOCATION_COUNT', None) == b'5'
        assert layers[0]['raster']['validation']['block_size_km'] == .05
    finally:
        gdal.GDALClose(dataset)

    tool.comparison_surface.setCurrentIndex(tool.comparison_surface.findData('idw-tin-difference'))
    assert tool.save_surface.isEnabled() and tool.save_tiff.isEnabled()
    tool.save_surface.click()
    diff, mask = dialog.layers[:2]
    assert diff['raster']['band'] == 'difference' and diff['visible'] and diff['cmap'] == 'RdBu_r'
    assert diff['raster']['limits'][0] == -diff['raster']['limits'][1]
    assert mask['raster']['band'] == 'coverage' and not mask['visible']
    assert mask['raster']['categories'] == ['Оба без прогноза', 'Только IDW', 'Только TIN', 'Оба метода']
    assert [row.values for row in window.model.project.rows] == original_rows
    mask['visible'] = True
    dialog.redraw()
    assert dialog.report is not None


def test_named_crs_catalog_and_auto_resolution(window, qtbot):
    dialog = ready_map(window)
    for key in ('source_crs', 'target_crs'):
        combo = dialog.data.fields[key][0]
        assert 'WGS 84 / UTM zone 31N' in combo.currentText()
        assert combo.findData('auto') >= 0
        assert combo.completer().filterMode() == Qt.MatchFlag.MatchContains
    assert 'WGS 84 / UTM zone 31N' in dialog.crs_status.text()
    window.model.project.columns[0].metadata['crs'] = 'EPSG:32631'
    dialog.data.load({'source_crs': 'auto', 'target_crs': 'auto'})
    dialog.redraw()
    assert dialog.report is not None, dialog.status.text()
    assert dialog.spec.source_crs == 'auto'
    assert dialog.report['spec']['target_crs'] == 'EPSG:32631'


def test_attribute_table_available_without_valid_map(window, qtbot, tmp_path):
    import geopandas as gpd
    from shapely.geometry import Point
    from autopyro.mapping import read_layer
    path = tmp_path / 'wells.gpkg'
    gpd.GeoDataFrame({'well': ['A', 'B'], 'value': [1, None]},
                     geometry=[Point(3, 60), None], crs='EPSG:4326').to_file(path)
    window.show_maps()
    dialog = window.map_window
    dialog.layers = [read_layer(path)]
    dialog.sync_layers()
    model = dialog.attribute_model
    assert model.rowCount() == 2
    assert model.columnCount() == 4
    assert model.data(model.index(0, 2)) == 'A'
    assert model.data(model.index(1, 3)) == '—'
    assert not (model.flags(model.index(0, 2)) & Qt.ItemFlag.ItemIsEditable)
    assert 'WGS 84' in dialog.attribute_info.text()
    dialog.redraw()
    assert dialog.report is not None, dialog.status.text()
    assert dialog.report['spec']['location_mode'] == 'layers'
    assert not dialog.report['points']
    assert not dialog.interpolate_button.isEnabled()
    assert model.rowCount() == 2
    dialog.layer_table.selectRow(0)
    dialog.remove_layer()
    assert model.rowCount() == 0


def test_layers_only_mode_ignores_sample_bindings_and_supports_navigation(window, qtbot):
    from shapely.geometry import Point, mapping
    dialog = ready_map(window)
    dialog.layers = [dict(id='layer', name='Layer', crs='EPSG:4326', visible=True,
                         color='#333333', fill='#aaccee', alpha=.3, line_width=1,
                         features=[dict(id='5', type='Feature', geometry=mapping(Point(3, 60)), properties={'name': 'A'})])]
    dialog.sync_layers()
    dialog.data.load({'location_mode': 'layers', 'target_crs': 'EPSG:3857'})
    dialog.redraw()
    assert dialog.report is not None, dialog.status.text()
    assert not dialog.report['points']
    assert dialog.report['spec']['target_crs'] == 'EPSG:3857'
    dialog.attribute_table.selectRow(0)
    assert dialog.feature_highlights
    dialog.attribute_table.doubleClicked.emit(dialog.attribute_model.index(0, 0))
    assert dialog.export_button.isEnabled()


def test_layer_color_field_is_saved_and_restored(window, qtbot):
    from shapely.geometry import Point, mapping
    dialog = ready_map(window)
    dialog.layers = [dict(id='layer', name='Layer', crs='EPSG:32631', visible=True,
                         color='#333333', fill='#aaccee', alpha=.3, line_width=1,
                         features=[dict(id=str(i), type='Feature', geometry=mapping(Point(i, i)), properties={'Paleogeogr': value})
                                   for i, value in enumerate(['Шельф', 'Равнина'])])]
    dialog.sync_layers()
    selector = dialog.layer_table.cellWidget(0, 6)
    selector.setCurrentIndex(selector.findData('Paleogeogr'))
    dialog.redraw()
    assert dialog.report is not None, dialog.status.text()
    assert len(dialog.result['axes'].get_legend().get_texts()) >= 2
    assert window.model.project.view['map_layers'][0]['color_by'] == 'Paleogeogr'
    dialog.sync_layers()
    selector = dialog.layer_table.cellWidget(0, 6)
    assert selector.currentData() == 'Paleogeogr'
    selector.setCurrentIndex(0)
    dialog.redraw()
    assert dialog.layers[0]['color_by'] == ''


def test_map_property_sections_menu_reaches_tabs_outside_narrow_dock(window, qtbot):
    window.resize(1280, 720)
    window.show()
    dialog = ready_map(window)
    dock = window.desktop.properties_dock
    window.resizeDocks([dock], [385], Qt.Orientation.Horizontal)
    qtbot.waitUntil(dialog.sections_button.isVisible)
    assert window.desktop.properties_stack.currentWidget() is dialog.tabs
    assert dock.width() < 450
    bar = dialog.tabs.tabBar()
    assert bar.tabRect(dialog.tabs.count() - 1).right() >= bar.width()

    menu = dialog.sections_button.menu()
    actions = menu.actions()
    assert [action.text() for action in actions] == [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())]
    qtbot.mouseClick(dialog.sections_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(menu.isVisible)
    assert [action.text() for action in actions if action.isChecked()] == [dialog.tabs.tabText(0)]
    qtbot.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(actions[-1]).center())
    assert dialog.tabs.currentIndex() == dialog.tabs.count() - 1
    dialog.tabs.setCurrentIndex(1)
    qtbot.mouseClick(dialog.sections_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(menu.isVisible)
    assert [action.text() for action in actions if action.isChecked()] == [dialog.tabs.tabText(1)]
    menu.close()


def test_map_forms_fit_narrow_properties_dock(window, qtbot):
    from PySide6.QtWidgets import QLabel, QScrollArea

    window.resize(1280, 720)
    window.show()
    dialog = ready_map(window)
    dock = window.desktop.properties_dock
    window.resizeDocks([dock], [385], Qt.Orientation.Horizontal)
    qtbot.waitUntil(dialog.tabs.isVisible)
    assert dock.width() < 450
    for index in range(dialog.tabs.count()):
        page = dialog.tabs.widget(index)
        if isinstance(page, QScrollArea):
            dialog.tabs.setCurrentIndex(index)
            qtbot.waitUntil(lambda: page.isVisible() and page.horizontalScrollBar().maximum() == 0)

    checkbox = dialog.layout_form.fields['scale_bar'][0]
    caption = next(label for label in dialog.layout_form.findChildren(QLabel) if label.buddy() is checkbox)
    qtbot.mouseClick(caption, Qt.MouseButton.LeftButton)
    assert dialog.layout_form.values()['scale_bar'] is True
    checkbox.parameter_reset_button.click()
    assert dialog.layout_form.values()['scale_bar'] is False


def test_layer_tables_stay_below_map_when_properties_are_borrowed(window, qtbot):
    from shapely.geometry import Point, mapping
    dialog = ready_map(window)
    dialog.layers = [{
        'id': 'wells', 'name': 'Wells', 'crs': 'EPSG:32631',
        'visible': True, 'color': '#333333', 'fill': '#aaccee', 'alpha': .3,
        'line_width': 1, 'features': [
            {'id': '1', 'type': 'Feature', 'geometry': mapping(Point(0, 0)),
             'properties': {'well': 'A'}},
        ],
    }]
    dialog.sync_layers()
    window.show()
    qtbot.waitUntil(dialog.canvas.isVisible)
    assert dialog.map_splitter.orientation() == Qt.Orientation.Vertical
    assert dialog.map_splitter.widget(1) is dialog.layer_tabs
    assert not dialog.tabs.isAncestorOf(dialog.layer_table)
    assert not dialog.tabs.isAncestorOf(dialog.attribute_table)
    assert dialog.isAncestorOf(dialog.layer_tabs)
    assert dialog.layer_table.isVisible()
    assert dialog.canvas.mapToGlobal(dialog.canvas.rect().topLeft()).y() < dialog.layer_tabs.mapToGlobal(dialog.layer_tabs.rect().topLeft()).y()
    dialog.layer_table.item(0, 1).setText('#ff0000')
    assert dialog.layers[0]['color'] == '#ff0000'
    dialog.layer_table.selectRow(0)
    assert dialog.attribute_layer.currentData() == 'wells'
    dialog.layer_tabs.setCurrentIndex(1)
    assert dialog.attribute_table.isVisible()
    assert dialog.canvas.isVisible()
    assert dialog.attribute_model.data(dialog.attribute_model.index(0, 2)) == 'A'
    dialog.redraw()
    assert dialog.report is not None, dialog.status.text()


def test_attributes_and_map_select_by_fid_in_both_directions(window, qtbot):
    from shapely.geometry import Point, Polygon, mapping
    dialog = ready_map(window)
    def layer(identifier, features):
        return dict(id=identifier, name=identifier, crs='EPSG:32631', visible=True,
                    color='#333333', fill='#aaccee', alpha=.3, line_width=1,
                    features=[dict(id=fid, type='Feature', geometry=mapping(geometry),
                                   properties={'name': fid}) for fid, geometry in features])
    wells = layer('wells', [('10', Point(10, 10)), ('90', Point(80, 80))])
    wells['fields'] = ['name']
    wells['attributes'] = [dict(id=fid, geometry_type=kind, properties={'name': fid})
                           for fid, kind in [('10', 'Point'), ('40', None), ('90', 'Point')]]
    polygon = Polygon([(0, 0), (60, 0), (60, 60), (0, 60)],
                      holes=[[(20, 20), (40, 20), (40, 40), (20, 40)]])
    regions = layer('regions', [('region', polygon)])
    dialog.layers = [wells, regions]
    dialog.sync_layers()
    dialog.redraw()
    limits = dialog.result['axes'].get_xlim(), dialog.result['axes'].get_ylim()
    dialog.attribute_table.selectRow(2)
    assert len(dialog.feature_highlights) == 1
    assert dialog.feature_highlights[0].get_offsets().tolist() == [[80, 80]]
    assert (dialog.result['axes'].get_xlim(), dialog.result['axes'].get_ylim()) == limits
    dialog.redraw()
    assert len(dialog.feature_highlights) == 1
    dialog.attribute_table.selectRow(1)  # Record with no displayable geometry.
    assert not dialog.feature_highlights

    def click(x, y, drag=0):
        px, py = dialog.result['axes'].transData.transform((x, y))
        event = SimpleNamespace(x=px, y=py, button=1, inaxes=dialog.result['axes'])
        dialog.feature_press(event)
        event.x += drag
        dialog.feature_release(event)

    click(80, 80)
    assert dialog.attribute_table.selectionModel().selectedRows()[0].row() == 2
    assert dialog.layer_tabs.currentIndex() == 1
    assert dialog.feature_highlights
    dialog.attribute_layer.setCurrentIndex(dialog.attribute_layer.findData('regions'))
    click(30, 30)  # Polygon hole must not count as a hit.
    assert not dialog.attribute_table.selectionModel().selectedRows()
    click(50, 50, drag=20)  # Rectangle selection must not select a feature.
    assert not dialog.attribute_table.selectionModel().selectedRows()
    click(50, 50)
    assert dialog.attribute_table.selectionModel().selectedRows()[0].row() == 0
    assert len(dialog.feature_highlights) == 1
    regions['visible'] = False
    dialog.redraw()
    assert not dialog.feature_highlights
    click(50, 50)
    assert not dialog.feature_highlights


def test_double_click_attribute_zooms_to_projected_feature(window, qtbot):
    from shapely.geometry import Point, Polygon, mapping
    dialog = ready_map(window)
    dialog.layers = [dict(id='objects', name='Objects', crs='EPSG:32631', visible=True,
                         color='#333333', fill='#aaccee', alpha=.3, line_width=1,
                         features=[dict(id=fid, type='Feature', geometry=mapping(geometry), properties={})
                                   for fid, geometry in [('point', Point(80, 80)),
                                                        ('area', Polygon([(10, 20), (30, 20), (30, 40), (10, 40)]))]])]
    dialog.sync_layers()
    dialog.redraw()
    ax = dialog.result['axes']
    dialog.attribute_table.doubleClicked.emit(dialog.attribute_model.index(1, 0))
    assert ax.get_xlim() == pytest.approx((8, 32))
    assert ax.get_ylim() == pytest.approx((18, 42))
    assert dialog.feature_highlights
    dialog.attribute_table.doubleClicked.emit(dialog.attribute_model.index(0, 0))
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    assert sum(xlim) / 2 == pytest.approx(80)
    assert sum(ylim) / 2 == pytest.approx(80)
    assert 0 < xlim[1] - xlim[0] < 20
    dialog.attribute_table.doubleClicked.emit(dialog.attribute_model.index(0, 0))
    assert ax.get_xlim() == xlim  # Repeated double clicks keep a stable point extent.
    dialog.toolbar.back()
    assert ax.get_xlim() == pytest.approx((8, 32))
    dialog.toolbar.forward()
    assert ax.get_xlim() == xlim


def test_double_click_layer_fits_all_objects_in_map_crs(window, qtbot, monkeypatch):
    from PySide6.QtCore import QThread
    from shapely.geometry import Point, mapping
    from autopyro.mapping import project_geometries
    from autopyro.ui.dialogs import map_selection as map_module
    dialog = ready_map(window)
    points = [Point(3, 60), Point(4, 61)]
    dialog.layers = [dict(id='wells', name='Wells', crs='EPSG:4326', visible=True,
                         color='#333333', fill='#aaccee', alpha=.3, line_width=1,
                         features=[dict(id=str(i), type='Feature', geometry=mapping(point), properties={})
                                   for i, point in enumerate(points)])]
    dialog.sync_layers()
    dialog.redraw()
    ax = dialog.result['axes']
    original = ax.get_xlim(), ax.get_ylim()
    bounds = project_geometries(points, 'EPSG:4326', dialog.report['spec']['target_crs']).total_bounds
    pad = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * .1

    def project_in_worker(*args):
        assert QThread.currentThread() != dialog.thread()
        return project_geometries(*args)

    monkeypatch.setattr(map_module, 'project_geometries', project_in_worker)
    dialog.layer_table.doubleClicked.emit(dialog.layer_table.model().index(0, 0))
    assert ax.get_xlim() == pytest.approx((bounds[0] - pad, bounds[2] + pad))
    assert ax.get_ylim() == pytest.approx((bounds[1] - pad, bounds[3] + pad))
    dialog.toolbar.back()
    assert (ax.get_xlim(), ax.get_ylim()) == original
    dialog.layers[0]['filter'] = dict(field='missing', op='eq', value='x')
    dialog.redraw()
    ax = dialog.result['axes']
    original = ax.get_xlim(), ax.get_ylim()
    dialog.layer_table.doubleClicked.emit(dialog.layer_table.model().index(0, 0))
    assert (ax.get_xlim(), ax.get_ylim()) == original
    assert 'нет объектов после фильтрации' in dialog.status.text()


def test_global_visualization_crs_reprojects_layers_and_samples(window, qtbot):
    from shapely.geometry import Point, mapping
    from autopyro.mapping import transform_geometry
    dialog = ready_map(window)
    dialog.layers = [dict(id=identifier, name=identifier, crs=source, visible=True,
                         color='#333333', fill='#aaccee', alpha=.3, line_width=1,
                         features=[dict(id='1', type='Feature', geometry=mapping(geometry), properties={})])
                     for identifier, source, geometry in [
                         ('geographic', 'EPSG:4326', Point(3, 60)),
                         ('projected', 'EPSG:3857', transform_geometry(Point(3, 60), 'EPSG:4326', 'EPSG:3857'))]]
    before = deepcopy(dialog.layers)
    dialog.sync_layers()
    old_axes = dialog.result['axes']
    old_axes.set_xlim(10, 40)
    old_axes.set_ylim(20, 50)
    dialog.layout_form.load({'xmin': -1, 'xmax': 1, 'ymin': -1, 'ymax': 1})
    assert not dialog.tabs.isAncestorOf(dialog.visualization_crs)
    dialog.visualization_crs.setCurrentIndex(dialog.visualization_crs.findData('EPSG:3857'))
    assert all(dialog.layout_form.values()[key] is None for key in ('xmin', 'xmax', 'ymin', 'ymax'))
    dialog.redraw()
    assert dialog.report is not None, dialog.status.text()
    assert dialog.report['spec']['target_crs'] == 'EPSG:3857'
    assert dialog.result['axes'].get_xlim()[1] > 100_000
    geometries = [item[2] for item in dialog.result['layer_features']]
    assert geometries[0].distance(geometries[1]) < 1e-6
    expected = transform_geometry(Point(0, 0), 'EPSG:32631', 'EPSG:3857')
    assert dialog.report['points'][0]['x'] == pytest.approx(expected.x)
    assert dialog.report['points'][0]['y'] == pytest.approx(expected.y)
    assert dialog.layers == before
    assert window.model.project.view['map_current']['target_crs'] == 'EPSG:3857'


def test_layer_category_editor_persists_colors_labels_and_field_scopes(window, qtbot, tmp_path):
    from shapely.geometry import Point, mapping
    from autopyro.mapping import layer_categories
    dialog = ready_map(window)
    layer = dict(id='regions', name='T3', crs='EPSG:32631', visible=True,
                 color='#333333', fill='#aaccee', alpha=.6, line_width=1, color_by='Paleogeogr',
                 features=[dict(id=str(i), type='Feature', geometry=mapping(Point(i, i)),
                                properties={'Paleogeogr': value, 'other': 'all'})
                           for i, value in enumerate(['Шельф', 'Равнина'])])
    dialog.layers = [layer]
    dialog.sync_layers()
    table = dialog.layer_category_table
    assert table.rowCount() == 2
    key = dialog.layer_category_keys[0]
    assert not table.item(0, 0).flags() & Qt.ItemFlag.ItemIsEditable
    table.item(0, 1).setText('#12ab34')
    table.item(0, 2).setText('Custom shelf')
    dialog.redraw()
    assert dialog.report is not None, dialog.status.text()
    assert any('Custom shelf' in item.get_text().replace('\n', ' ')
               for item in dialog.result['axes'].get_legend().get_texts())
    expected = deepcopy(layer['category_styles'])
    table.item(0, 1).setText('invalid color')
    assert layer['category_styles'] == expected
    selector = dialog.layer_table.cellWidget(0, 6)
    selector.setCurrentIndex(selector.findData('other'))
    assert table.rowCount() == 1
    dialog.reset_layer_categories()
    assert layer['category_styles']['Paleogeogr'] == expected['Paleogeogr']
    selector.setCurrentIndex(selector.findData('Paleogeogr'))
    assert table.item(0, 2).text() == 'Custom shelf'
    dialog.redraw()
    path = tmp_path / 'category-styles.apyro'
    window.model.project.save(path)
    restored_layer = Project.load(path).view['map_layers'][0]
    assert restored_layer['category_styles'] == expected
    assert layer_categories(restored_layer)[key][0] == 'Custom shelf'
    dialog.reset_layer_categories()
    assert layer_categories(layer)[key][0] != 'Custom shelf'

def test_saved_map_is_available_in_report(window, qtbot, tmp_path):
    from shapely.geometry import Point, mapping

    from autopyro.reporting import export_report
    from autopyro.ui.dialogs.report import ReportDialog

    dialog = ready_map(window)
    dialog.layers = [dict(id='test-layer', name='Test layer', crs='EPSG:32631', visible=True,
                         color='#333333', fill='#aaccee', alpha=.3, line_width=1,
                         features=[dict(id='point', type='Feature', geometry=mapping(Point(25, 25)),
                                        properties={})])]
    window.model.project.view['map_layers'] = deepcopy(dialog.layers)
    dialog.redraw()
    dialog.save_named_view('Test view')
    report = ReportDialog(window)
    qtbot.addWidget(report)
    report.section_order = ('figures',)
    report.map_view.setCurrentIndex(report.map_view.findData('Test view'))
    report.build()
    assert report.report is not None, report.status.text()
    figure, = report.report['map_figures']
    assert figure['png'] and figure['spec']['target_crs'] == 'EPSG:32631'
    assert figure['name'].endswith('Test view')
    assert figure['row_ids'] == [row.id for row in window.model.project.rows]
    assert 'Test view' in report.preview.toHtml()
    for format_name in ('pdf', 'docx'):
        path = tmp_path / f'map-report.{format_name}'
        export_report(path, report.report, format_name)
        assert path.stat().st_size > 1000
