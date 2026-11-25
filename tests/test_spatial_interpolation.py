from copy import deepcopy

import numpy as np
import pytest

from autopyro.io import build_project
from autopyro.mapping import MapSpec,build_map
from autopyro.spatial_interpolation import SpatialOptions,compare_idw_settings,interpolate,predict,spatial_blocks
from autopyro.spatial.spatial_interpolation import nested_idw_validation
from autopyro.spatial.spatial_interpolation import preview_sample_support


def test_sample_preview_counts_duplicates_exclusions_and_block_support(data):
    from dataclasses import replace

    report, options = data
    point = report['points'][0]
    report['points'].extend([dict(point, id='duplicate'),
                             dict(point, id='below-limit', values={options.value_id: '<0.1'})])
    before = deepcopy(report)
    result = preview_sample_support(report, options, .025)
    assert (result['mapped'], result['eligible'], result['locations']) == (7, 6, 5)
    assert (result['duplicate_locations'], result['duplicate_samples']) == (1, 1)
    assert result['excluded'] == {'censored': 1}
    assert result['blocks'] == 5
    assert result['supported'] == 5
    validated = spatial_blocks(report, replace(options, duplicate_policy='median'), .025, evaluate=True)
    assert result['with_hull'] == validated['totals']['predicted'] == 1
    assert preview_sample_support(report, replace(options, max_distance=1), .025)['supported'] == 0
    assert preview_sample_support(report, options, 1)['supported'] == 0
    assert report == before
    report['spec']['target_crs'] = 'EPSG:4326'
    geographic = preview_sample_support(report, options, .025)
    assert geographic['metric'] is False and geographic['supported'] is None
    assert geographic['locations'] == 5


def test_sample_preview_explains_collinear_locations_before_calculation(data):
    report, options = data
    for i, point in enumerate(report['points']):
        point.update(x=i * 100, y=0)
    result = preview_sample_support(report, options, .05)
    assert result['locations'] == result['blocks'] == 5
    assert result['supported'] == 5
    assert result['with_hull'] == 0
    assert 'на одной линии' in result['issue']
    with pytest.raises(ValueError, match='на одной линии'):
        interpolate(report, options)


def test_buffer_excludes_near_block_locations_from_outer_and_inner_checks(data):
    from dataclasses import replace
    from autopyro.spatial.spatial_interpolation import _block_training_mask

    xy = np.array([[0, 0], [-3, -4], [-4, -4], [25, 10], [31, 10]])
    block = {'bounds': [0, 0, 25, 25]}
    assert _block_training_mask(xy, block, [0], 0).tolist() == [False, True, True, True, True]
    assert _block_training_mask(xy, block, [0], .005).tolist() == [False, False, True, False, True]

    report, options = data
    coordinates = [(-100, -100), (-100, 100), (100, -100), (100, 100),
                   (1, 1), (40, 10), (10, 40), (-1, 1)]
    report['points'] = [{'id': str(i), 'x': x, 'y': y, 'values': {options.value_id: i ** 2}}
                        for i, (x, y) in enumerate(coordinates)]
    buffered = replace(options, validation_buffer_km=.005)
    variants = [dict(power=power, neighbors=8, max_distance=1000) for power in (1, 3)]
    result = nested_idw_validation(report, buffered, .025, variants)
    held_out = next(p for p in result['locations'] if p['row_ids'] == ['4'])
    fold = result['blocks'][held_out['block'] - 1]
    assert fold['buffered_locations'] == 1
    assert fold['selected'] is not None
    assert held_out['tuned'] is not None
    unbuffered = nested_idw_validation(report, options, .025, variants)
    assert next(p for p in unbuffered['locations'] if p['row_ids'] == ['4'])['baseline'] != held_out['baseline']
    report['points'][-1]['values'][options.value_id] += 1000
    changed = nested_idw_validation(report, buffered, .025, variants)
    assert changed['blocks'][held_out['block'] - 1] == fold
    assert next(p for p in changed['locations'] if p['row_ids'] == ['4']) == held_out
    blocked = spatial_blocks(report, buffered, .025, evaluate=True)
    assert blocked['totals'] == changed['baseline']
    assert preview_sample_support(report, buffered, .025)['with_hull'] == blocked['totals']['predicted']


def test_idw_spread_counts_predictions_and_masks_unsupported_nodes(data, qtbot, monkeypatch, tmp_path):
    from dataclasses import replace
    from PySide6.QtWidgets import QDialog, QLabel, QSpinBox
    from autopyro.spatial.kriging import KrigingOptions, predict_kriging
    from autopyro.spatial.spatial_interpolation import idw_partition_spread, partition_spread
    from autopyro.spatial.surfaces import SPREAD_METHOD, TIN_SPREAD_METHOD, export_geotiff, surface_bands, surface_layers
    from autopyro.spatial.surfaces import KRIGING_SPREAD_METHOD
    from autopyro.ui.dialogs.map import SpatialInterpolationDialog

    report, options = data
    report['points'][-1]['values'][options.value_id] = 20
    options = replace(options, grid_size=11)
    checkpoints = []
    result = idw_partition_spread(report, options, .025, progress=lambda i, n: checkpoints.append((i, n)))
    assert checkpoints == [(i, 5) for i in range(5)]
    assert result['count'][5, 5] == 5
    assert result['spread'][5, 5] == pytest.approx(17)
    assert np.isnan(result['spread'][result['count'] < 2]).all()
    tin = partition_spread(report, replace(options, max_distance=1, min_neighbors=5), .025, method='TIN')
    np.testing.assert_equal(tin['inside_domain'], result['inside_domain'])
    assert tin['count'][5, 5] == 5
    assert tin['spread'][5, 5] == pytest.approx(17)
    assert list(surface_bands(tin, TIN_SPREAD_METHOD)[2]) == ['spread', 'count']
    model = KrigingOptions(model='exponential', range_km=.2, azimuth_deg=30, range_ratio=.5)
    kriging = partition_spread(report, options, .025, method='kriging', kriging_model=model)
    xy = np.array([[point['x'], point['y']] for point in report['points']])
    values = np.array([point['values'][options.value_id] for point in report['points']])
    predictions = []
    for i in range(len(xy)):
        train = np.arange(len(xy)) != i
        predictions.append(predict_kriging(xy[train], values[train], np.array([[50, 50]]), options, model=model)[0][0])
    assert kriging['count'][5, 5] == 5
    assert kriging['spread'][5, 5] == pytest.approx(np.ptp(predictions))
    layers = surface_layers(kriging, KRIGING_SPREAD_METHOD, 'EPSG:32631', 'TOC', {'spread': 'Spread', 'count': 'Count'}, '%wt')
    assert layers[0]['raster']['model']['range_ratio'] == .5
    point = report['points'][-1]
    report['points'].append(dict(point, id='shift-only-block', x=15, y=15))
    shifted = partition_spread(report, options, .025, method='TIN', shifted=True)
    assert shifted['blocks'] == tin['blocks'] + 1
    assert shifted['options']['block_shift_km'] == .0125
    np.testing.assert_equal(shifted['inside_domain'], tin['inside_domain'])
    report['points'].pop()
    buffered = idw_partition_spread(report, replace(options, validation_buffer_km=1), .025)
    assert not buffered['count'].any()
    assert np.isnan(buffered['spread']).all()
    _, _, bands = surface_bands(buffered, SPREAD_METHOD)
    assert np.isnan(bands['spread']).all()
    assert np.all(bands['count'][buffered['inside_domain']] == 0)
    labels = {'spread': 'Spread', 'count': 'Count'}
    layers = surface_layers(buffered, SPREAD_METHOD, 'EPSG:32631', 'TOC', labels, '%wt')
    assert len(layers) == 1 and layers[0]['visible'] and layers[0]['raster']['band'] == 'count'
    assert layers[0]['raster']['options']['validation_buffer_km'] == 1
    path = tmp_path / 'idw-spread.tif'
    export_geotiff(path, buffered, SPREAD_METHOD, 'EPSG:32631', '%wt')
    assert path.read_bytes()[:2] in (b'II', b'MM')
    with pytest.raises(ValueError, match='минимум два'):
        idw_partition_spread(report, options, 1)
    dialog = SpatialInterpolationDialog(report)
    qtbot.addWidget(dialog)
    opened = []

    def inspect(widget):
        assert 'Разброс от' in widget.findChild(QLabel).text()
        row, column = widget.findChildren(QSpinBox)
        assert (row.value(), column.value()) == (6, 6)
        node = next(label for label in widget.findChildren(QLabel)
                    if label.accessibleDescription() == 'Значения в выбранном узле карты разброса')
        assert 'Прогноз дали 5 из 5' in node.text()
        column.setValue(7)
        assert 'X 60.00' in node.text()
        opened.append(widget.windowTitle())
        return 0

    monkeypatch.setattr(QDialog, 'exec', inspect)
    dialog.show_spread(result, report, options, .025)
    dialog.show_spread(tin, report, options, .025)
    dialog.show_spread(kriging, report, options, .025)
    assert opened == ['Карта разброса IDW', 'Карта разброса TIN', 'Карта разброса кригинга']


def test_sample_preview_dialog_does_not_require_running_interpolation(data, qtbot):
    from autopyro.ui.dialogs.map import SpatialInterpolationDialog

    report, options = data
    dialog = SpatialInterpolationDialog(report)
    qtbot.addWidget(dialog)
    dialog.form.load({'value_id': options.value_id, 'block_size_km': .025, 'max_distance': 1})
    dialog.check_sample()
    qtbot.waitUntil(lambda: dialog.sample_preview_table.model() is not None and dialog.isEnabled(), timeout=10000)
    assert dialog.views.currentWidget() is dialog.sample_preview_tab
    assert 'EPSG:32631' in dialog.sample_preview_table.model().index(0, 1).data()
    assert 'ни у одной' in dialog.sample_preview_note.text()
    assert dialog.result is None
    dialog.form.fields['max_distance'][0].setValue(1)
    assert dialog.form.values()['max_distance'] == 1000
    assert dialog.sample_preview_table.model() is None


def test_interpolation_start_requires_variable_and_shows_contextual_actions(data, qtbot):
    from PySide6.QtWidgets import QLabel, QPushButton
    from autopyro.ui.dialogs.map import SpatialInterpolationDialog

    report, _ = data
    report['spec']['target_crs'] = 'EPSG:4326'
    dialog = SpatialInterpolationDialog(report)
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog.value_selector.currentData() == ''
    assert dialog.idw_intro.isVisible()
    assert not dialog.canvas_view.scroll.isVisible()
    labels = [label.text() for label in dialog.findChildren(QLabel)]
    assert any('EPSG:4326' in label for label in labels)
    assert any('вернитесь на карту' in label for label in labels)
    dialog.run_idw_button.click()
    assert dialog.worker is None and 'Выберите показатель' in dialog.status.text()
    block_run = next(button for button in dialog.findChildren(QPushButton) if button.text() == 'Проверить по блокам')
    assert not block_run.isVisible()
    dialog.views.setCurrentIndex(1)
    assert block_run.isVisible()
    assert not dialog.run_idw_button.isVisible()
    assert not dialog.save_tiff.isVisible()


def test_sample_check_can_be_completed_with_keyboard(data, qtbot):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from autopyro.ui.dialogs.map import SpatialInterpolationDialog

    report, options = data
    dialog = SpatialInterpolationDialog(report)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.activateWindow()
    qtbot.waitUntil(dialog.value_selector.hasFocus)
    qtbot.keyClick(dialog.value_selector, Qt.Key.Key_End)
    assert dialog.value_selector.currentData() == options.value_id
    qtbot.keyClick(dialog.value_selector, Qt.Key.Key_Tab)
    assert dialog.preflight_button.hasFocus()
    qtbot.keyClick(dialog.preflight_button, Qt.Key.Key_Space)
    results = dialog.sample_preview_table
    qtbot.waitUntil(lambda: results.hasFocus() and results.model() is not None and dialog.isEnabled(), timeout=10000)
    assert results.accessibleName() == 'Результаты проверки выборки'
    assert results.currentIndex().row() == 0
    qtbot.keyClick(results, Qt.Key.Key_Down)
    assert results.currentIndex().row() == 1
    qtbot.keyClick(results, Qt.Key.Key_Tab)
    assert QApplication.focusWidget() is not results


def test_interpolation_exposes_status_and_announces_success_and_error(data, qtbot, monkeypatch):
    from types import SimpleNamespace
    from PySide6.QtGui import QAccessible
    from autopyro.ui.dialogs import spatial_interpolation as map_ui

    report, options = data
    dialog = map_ui.SpatialInterpolationDialog(report)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.form.load({'value_id': options.value_id})
    table = QAccessible.queryAccessibleInterface(dialog.sample_preview_table)
    assert table.role() == QAccessible.Role.Table
    assert table.text(QAccessible.Text.Name) == 'Результаты проверки выборки'
    radius = QAccessible.queryAccessibleInterface(dialog.variant_fields['max_distance'])
    assert radius.text(QAccessible.Text.Name) == 'Радиусы поиска, км'

    events = []
    monkeypatch.setattr(map_ui, 'QAccessible', SimpleNamespace(
        isActive=lambda: True, updateAccessibility=lambda event: events.append((event.object(), event.message())),
    ))
    dialog.check_sample()
    qtbot.waitUntil(lambda: dialog.sample_preview_table.model() is not None and dialog.isEnabled(), timeout=10000)
    assert events[-1] == (dialog.status, 'Предварительная проверка выборки завершена.')

    def fail(*args):
        raise ValueError('Test worker failure')

    from autopyro.ui.dialogs import spatial_input

    monkeypatch.setattr(spatial_input, 'preview_sample_support', fail)
    dialog.check_sample()
    qtbot.waitUntil(lambda: dialog.isEnabled() and 'Test worker failure' in dialog.status.text(), timeout=10000)
    assert events[-1] == (dialog.status, 'Расчёт не выполнен: Test worker failure')
    accessible = QAccessible.queryAccessibleInterface(dialog.status)
    assert accessible.text(QAccessible.Text.Name) == dialog.status.text()


def test_interpolation_layout_adapts_to_small_logical_window(data, qtbot):
    from PySide6.QtCore import Qt
    from autopyro.ui.dialogs.map import SpatialInterpolationDialog

    report, options = data
    dialog = SpatialInterpolationDialog(report)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.resize(720, 450)
    qtbot.waitUntil(dialog.sections_button.isVisible)
    menu = dialog.sections_button.menu()
    actions = menu.actions()
    assert [action.text() for action in actions] == [dialog.views.tabText(i) for i in range(dialog.views.count())]
    qtbot.mouseClick(dialog.sections_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(menu.isVisible)
    assert actions[0].isChecked()
    comparison = actions[dialog.views.indexOf(dialog.kriging_tab)]
    qtbot.mouseClick(menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(comparison).center())
    assert dialog.views.currentWidget() is dialog.kriging_tab
    qtbot.waitUntil(lambda: dialog.kriging_tab.verticalScrollBar().maximum() > 0)
    assert (dialog.width(), dialog.height()) == (720, 450)
    assert dialog.interpolation_splitter.orientation() == Qt.Orientation.Vertical
    dialog.form.load({'value_id': options.value_id, 'context': options.context, 'grid_size': 10})
    dialog.calculate_kriging(preview=False)
    assert dialog.kriging_result is not None
    assert dialog.kriging_tab.verticalScrollBar().value() > 0
    result_scroll = dialog.kriging_tab.verticalScrollBar().value()
    dialog.comparison_back.click()
    assert dialog.kriging_tab.verticalScrollBar().value() < result_scroll
    assert dialog.views.tabBar().isVisible()
    dialog.resize(2400, 950)
    assert dialog.interpolation_splitter.orientation() == Qt.Orientation.Horizontal


def test_validation_map_distinguishes_probes_locations_and_background(data, qtbot):
    from dataclasses import replace
    from shapely.geometry import Point, box, mapping
    from autopyro.ui.dialogs.map import SpatialInterpolationDialog

    report, options = data
    point = report['points'][0]
    report['points'].extend([dict(point, id='repeat', label='Raw sample label'),
                             dict(point, id='missing', values={options.value_id: None})])
    report['diagnostics']['input_rows'] = 8
    report['layers'] = [
        dict(id='background', name='Background', crs='EPSG:32631', visible=True, color='#333333',
             fill='none', alpha=1, line_width=1, label_by='name', features=[
                 dict(id='well', geometry=mapping(Point(32, 32)), properties={'name': 'Background well'}),
                 dict(id='area', geometry=mapping(box(-10, -10, 110, 110)), properties={'name': 'Area'}),
             ]),
    ]
    before = deepcopy(report)
    result = spatial_blocks(report, replace(options, duplicate_policy='median'), .025, evaluate=True)
    dialog = SpatialInterpolationDialog(report)
    qtbot.addWidget(dialog)
    ax, _ = dialog.draw_blocks(dialog.block_figure, result, report)
    footer, = dialog.block_figure.texts
    assert 'в выборке 8 · с координатами 7 · в проверяемых локациях 6' in footer.get_text()
    assert 'Проверяемых локаций: 5 · С прогнозом: 1' in footer.get_text()
    assert 'EPSG:32631' in footer.get_text()
    assert 'Background well' not in [label.get_text() for label in ax.texts]
    assert 'Raw sample label' not in [label.get_text() for label in ax.texts]
    assert 'Area' in [label.get_text() for label in ax.texts]
    assert all([32, 32] not in item.get_offsets().tolist() for item in ax.collections)
    assert report == before


def test_idw_tin_difference_respects_common_domain_and_coverage():
    from autopyro.spatial.surfaces import compare_idw_tin_surfaces

    grid = dict(x=[0, 1, 2], y=[0, 1], inside_domain=[[True, True, True], [True, True, False]])
    idw = dict(grid, prediction=[[1, 3, np.nan], [4, np.nan, 100]])
    tin = dict(grid, prediction=[[3, np.nan, 5], [1, np.nan, 100]])
    result = compare_idw_tin_surfaces(idw, tin)
    np.testing.assert_equal(result['difference'], [[2, np.nan, np.nan], [-3, np.nan, np.nan]])
    assert np.argwhere(result['idw_only']).tolist() == [[0, 1]]
    assert np.argwhere(result['tin_only']).tolist() == [[0, 2]]
    assert np.argwhere(result['neither']).tolist() == [[1, 1]]
    with pytest.raises(ValueError):
        compare_idw_tin_surfaces(idw, dict(tin, x=[1, 2, 3]))


def test_comparison_dialog_shows_idw_and_tin_on_shared_scale(data, qtbot):
    from types import SimpleNamespace
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QAccessible
    from autopyro.spatial.surfaces import compare_idw_tin_surfaces, surface_bands
    from autopyro.ui.dialogs.map import SpatialInterpolationDialog

    report, options = data
    report['points'][-1]['values'][options.value_id] = 20
    report['points'][-1]['label'] = 'Well-Center'
    dialog = SpatialInterpolationDialog(report)
    qtbot.addWidget(dialog)
    dialog.form.load({'context': options.context, 'value_id': options.value_id,
                      'block_size_km': .025, 'max_distance': 65, 'grid_size': 10})
    dialog.views.setCurrentWidget(dialog.kriging_tab)
    dialog.comparison_surface.setCurrentIndex(dialog.comparison_surface.findData('idw-tin-difference'))
    dialog.calculate_kriging(preview=False)
    qtbot.waitUntil(lambda: dialog.kriging_result is not None and dialog.isEnabled(), timeout=10000)
    axes = dialog.kriging_figure.axes
    assert [ax.get_title() for ax in axes[:3]] == ['IDW', 'TIN', 'TIN − IDW']
    assert axes[0].collections[1].norm is axes[1].collections[1].norm
    delta = axes[2].collections[1]
    assert delta.norm.vmin == -delta.norm.vmax
    assert any(label.get_text() == 'Только TIN' for label in axes[2].get_legend().get_texts())
    assert not dialog.comparison_common_scale.isEnabled()
    saved = dialog.kriging_result
    compared = compare_idw_tin_surfaces(saved['idw_surface'], saved['tin_surface'])
    summary = dialog.comparison_text_summary
    assert summary.focusPolicy() == Qt.FocusPolicy.StrongFocus
    accessible = QAccessible.queryAccessibleInterface(summary)
    assert f"общих {compared['common'].sum()}" in summary.text()
    assert f"только TIN {compared['tin_only'].sum()}" in summary.text()
    assert 'Разность TIN − IDW от' in summary.text()
    assert accessible.text(QAccessible.Text.Name) == summary.text()
    grid = saved['idw_surface']
    row, column = np.argwhere(compared['tin_only'])[0]
    dialog.pick_grid_node(SimpleNamespace(button=1, inaxes=axes[2],
                                          xdata=grid['x'][column], ydata=grid['y'][row]))
    assert [field.value() for field in dialog.grid_indices] == [row + 1, column + 1]
    assert 'IDW: Недоступно' in dialog.grid_node_text.text()
    assert 'TIN − IDW: Недоступно' in dialog.grid_node_text.text()
    row, column = np.argwhere(compared['common'])[0]
    for field, value in zip(dialog.grid_indices, (row + 1, column + 1)):
        field.setValue(value)
    assert f"TIN − IDW: {compared['difference'][row, column]:.5g}" in dialog.grid_node_text.text()
    assert all(marker.get_xdata() == [grid['x'][column]] for marker in dialog.grid_inspection[2])
    for field in dialog.grid_indices:
        field.setValue(5)
    text = dialog.grid_holdout_text.text()
    assert text.splitlines()[1].startswith('Well-Center')
    assert len(text.splitlines()) == 4
    center = next(point for point in saved['idw_tin_holdouts'] if (point['x'], point['y']) == (50, 50))
    assert f"TIN {center['tin_residual']:+.4g}" in text.splitlines()[1]
    assert 'Мало соседей' in text.splitlines()[1]
    assert 'До ближайшей проверочной локации' in dialog.grid_node_text.text()
    dialog.comparison_surface.setCurrentIndex(dialog.comparison_surface.findData('IDW'))
    assert dialog.grid_inspector.isHidden()
    result, method, _, _ = dialog.active_surface()
    _, _, bands = surface_bands(result, method)
    np.testing.assert_equal(bands['prediction'], dialog.kriging_result['idw_surface']['prediction'])
    assert 'IDW. Прогноз в' in summary.text()
    limits = saved['surface_value_range']
    assert dialog.kriging_figure.axes[1].collections[0].get_clim() == limits
    dialog.comparison_surface.setCurrentIndex(dialog.comparison_surface.findData('TIN'))
    assert dialog.kriging_figure.axes[1].collections[0].get_clim() == limits
    dialog.comparison_common_scale.setChecked(False)
    tin = saved['tin_surface']['prediction']
    local = (np.nanmin(tin), np.nanmax(tin))
    assert local != limits
    assert dialog.kriging_figure.axes[1].collections[0].get_clim() == local
    assert dialog.kriging_result is saved


def test_partition_comparison_shifts_the_grid_and_preserves_shared_holdouts(data, qtbot):
    from types import SimpleNamespace
    from PySide6.QtCore import Qt
    from autopyro.ui.dialogs.map import SpatialInterpolationDialog
    report, options = data
    report['points'][0]['label'] = 'Well-A'
    dialog = SpatialInterpolationDialog(report)
    qtbot.addWidget(dialog)
    dialog.form.load({'context': options.context, 'value_id': options.value_id, 'max_distance': options.max_distance})
    dialog.partition_sizes.setText('0,05; 0,1')
    dialog.calculate_partitions()
    qtbot.waitUntil(lambda: len(dialog.partition_results) == 24 and dialog.isEnabled(), timeout=15000)
    rows = dialog.partition_results
    assert dialog.partition_table.model().rowCount() == 24
    for first in range(0, len(rows), 6):
        variants = rows[first:first + 6]
        memberships = [[(tuple(point['row_ids']), point['block']) for point in item['locations']] for item in variants]
        assert all(membership == memberships[0] for membership in memberships)
        forecasts = [np.array([point['predicted'] is not None for point in item['locations']]) for item in variants]
        common_count = np.logical_and.reduce(forecasts).sum()
        assert all(item['common']['predicted'] == common_count for item in variants)
        reference = variants[0]['common']['rmse']
        if reference is not None:
            assert [row['delta_rmse_idw'] for row in variants] == pytest.approx(
                [row['common']['rmse'] - reference for row in variants])
    baseline = spatial_blocks(report, options, .1, evaluate=True)
    shifted = spatial_blocks(report, options, .1, evaluate=True, offset_m=(50, 50))
    assert rows[12]['rmse'] == baseline['totals']['rmse']
    assert rows[18]['rmse'] == shifted['totals']['rmse']
    assert [block['bounds'] for block in rows[12]['blocks']] != [block['bounds'] for block in rows[18]['blocks']]
    dialog.partition_table.selectRow(18)
    assert '0.05' in dialog.partition_figure.axes[0].get_title()
    model = dialog.residual_table.model()
    assert model.rowCount() == len(rows[18]['locations'])
    index = next(i for i, p in enumerate(rows[18]['locations']) if p['predicted'] is None)
    point = rows[18]['locations'][index]
    ax = dialog.partition_figure.axes[0]
    px, py = ax.transData.transform((point['x'], point['y']))
    dialog.pick_residual_location(SimpleNamespace(button=1, inaxes=ax, x=px, y=py))
    assert dialog.residual_table.selectionModel().selectedRows()[0].row() == index
    assert model.index(index, 7).data() == '—'
    assert model.index(index, 2).data() == 'Вне обучающей оболочки'
    assert any(label.get_text() == model.index(index, 2).data() for label in ax.get_legend().get_texts())
    measured = next(i for i, p in enumerate(rows[18]['locations']) if p['predicted'] is not None)
    dialog.residual_table.selectRow(measured)
    expected = rows[18]['locations'][measured]
    assert dialog.residual_marker.get_offsets().tolist() == [[expected['x'], expected['y']]]
    assert float(model.index(measured, 8).data()) == pytest.approx(expected['residual'], rel=1e-5)
    assert any(model.index(i, 0).data() == 'Well-A' for i in range(model.rowCount()))
    dialog.partition_table.sortByColumn(3, Qt.SortOrder.DescendingOrder)
    proxy = dialog.partition_table.model()
    coverage = [proxy.index(i, 3).data(Qt.ItemDataRole.UserRole) for i in range(proxy.rowCount())]
    assert coverage == sorted(coverage, reverse=True)
    dialog.partition_table.selectRow(1)
    assert dialog.residual_locations == rows[proxy.mapToSource(proxy.index(1, 0)).row()]['locations']


def test_ordinary_kriging_uses_same_holdouts(data, qtbot):
    from autopyro.spatial.kriging import KrigingOptions, predict_kriging
    from autopyro.ui.dialogs.map import SpatialInterpolationDialog
    report, options = data
    xy = np.array([[0, 0], [100, 0], [0, 100], [100, 100]])
    model = KrigingOptions(range_km=.2)
    estimates, _, _ = predict_kriging(xy, np.array([1., 2., 3., 4.]), xy, options, model=model)
    assert estimates == pytest.approx([1, 2, 3, 4])
    constant, _, _ = predict_kriging(xy, np.full(4, 7.), np.array([[40, 20]]), options,
                                    model=KrigingOptions(range_km=.2, nugget=.1))
    assert constant == pytest.approx([7])
    dialog = SpatialInterpolationDialog(report)
    qtbot.addWidget(dialog)
    dialog.form.load({'context': 'synthetic cohort', 'value_id': options.value_id, 'block_size_km': .05})
    dialog.kriging_form.load({'range_km': .2})
    dialog.calculate_kriging(preview=False)
    qtbot.waitUntil(lambda: dialog.kriging_result is not None and dialog.isEnabled(), timeout=10000)
    result = dialog.kriging_result
    baseline = spatial_blocks(report, options, .05, evaluate=True)
    assert [block['bounds'] for block in result['blocks']] == [block['bounds'] for block in baseline['blocks']]
    assert [row['common']['predicted'] for row in result['validation']] == [1, 1, 1, 1, 1, 1]
    assert sum(point['pairs'] for point in result['variogram']) == 10
    assert dialog.kriging_table.model().rowCount() == 6
    from autopyro.spatial.surfaces import surface_bands
    _, _, bands = surface_bands(result, 'kriging')
    assert np.isfinite(bands['prediction']).any()
    assert np.isfinite(bands['neighbors']).any()
    dialog.views.setCurrentWidget(dialog.kriging_tab)
    dialog.comparison_surface.setCurrentIndex(dialog.comparison_surface.findData('TIN'))
    surface, method, _, _ = dialog.active_surface()
    assert method == 'TIN' and surface is result['tin_surface']
    _, _, bands = surface_bands(surface, method)
    assert set(bands['neighbors'][np.isfinite(bands['prediction'])]) == {3}
    for name, neighbors in [('RBF', 5), ('nearest', 1), ('Clough-Tocher', 5)]:
        dialog.comparison_surface.setCurrentIndex(dialog.comparison_surface.findData(name))
        grid, method, _, _ = dialog.active_surface()
        assert method == name
        assert set(grid['neighbors'][np.isfinite(grid['prediction'])]) == {neighbors}


def test_directional_variograms_and_anisotropic_comparison(data, qtbot):
    from autopyro.spatial.kriging import KrigingOptions, covariance_coordinates, directional_variograms
    from autopyro.ui.dialogs.map import SpatialInterpolationDialog
    xy = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    groups = directional_variograms(xy, np.array([0., 1., 2., 3.]))
    assert {angle: sum(point['pairs'] for point in bins) for angle, bins in groups.items()} == {
        0: 2, 45: 1, 90: 2, 135: 1}
    model = KrigingOptions(azimuth_deg=90, range_ratio=.5)
    east, north = covariance_coordinates(np.array([[100., 0.], [0., 100.]]), model)
    np.testing.assert_allclose([np.linalg.norm(east), np.linalg.norm(north)], [100, 200])
    report, options = data
    dialog = SpatialInterpolationDialog(report)
    qtbot.addWidget(dialog)
    dialog.form.load({'context': options.context, 'value_id': options.value_id, 'block_size_km': .05})
    dialog.calculate_kriging(preview=True)
    qtbot.waitUntil(lambda: dialog.kriging_result is not None and dialog.isEnabled(), timeout=10000)
    rows = dialog.kriging_table.model().rows
    assert rows == [['Все направления', point['distance_km'], point['semivariance'], point['pairs']]
                    for point in dialog.kriging_result['variogram']]
    assert dialog.kriging_table.accessibleName() == 'Данные эмпирической вариограммы'
    assert '10 пар' in dialog.comparison_text_summary.text()
    dialog.kriging_form.load({'directions': True, 'azimuth_deg': 90, 'range_ratio': .5, 'range_km': .2})
    dialog.calculate_kriging(preview=True)
    qtbot.waitUntil(lambda: dialog.kriging_result is not None and dialog.isEnabled(), timeout=10000)
    rows = dialog.kriging_table.model().rows
    assert sum(row[-1] for row in rows if row[0] == 'Все направления') == 10
    assert sum(row[-1] for row in rows if row[0] != 'Все направления') == 10
    dialog.calculate_kriging(preview=False)
    qtbot.waitUntil(lambda: dialog.kriging_result is not None and dialog.isEnabled(), timeout=10000)
    result = dialog.kriging_result
    assert dialog.kriging_table.accessibleName() == 'Результаты сравнения методов'
    metrics = {row['method']: row for row in result['validation']}
    assert metrics['kriging']['common']['predicted'] == metrics['isotropic-kriging']['common']['predicted'] == 1
    assert dialog.comparison_surface.findData('isotropic-kriging') >= 0
    assert not np.allclose(result['prediction'], result['extra_surfaces']['isotropic-kriging']['prediction'])


@pytest.mark.parametrize('method', ['predict_tin', 'predict_clough_tocher'])
def test_triangulation_reproduces_plane_and_leaves_outside_unpredicted(method):
    from autopyro.spatial import tin
    predict_tin = getattr(tin, method)
    xy = np.array([[0., 0.], [4., 0.], [0., 4.]])
    values = 2 * xy[:, 0] - xy[:, 1] + 5
    predicted, support, _ = predict_tin(xy, values, np.array([[1., 1.], [4., 4.]]), None)
    np.testing.assert_allclose(predicted, [6, np.nan], atol=1e-6, equal_nan=True)
    assert support.tolist() == [3, 0]
    flat, _, _ = predict_tin(np.array([[0., 0.], [1., 0.], [2., 0.]]), values, xy[:1], None)
    assert np.isnan(flat[0])


def test_rbf_smoothing_and_nearest_radius(data):
    from dataclasses import replace
    from autopyro.spatial.rbf import predict_nearest, predict_rbf
    _, options = data
    xy = np.array([[0., 0.], [1000., 0.], [0., 1000.], [1000., 1000.]])
    values = np.array([1., 2., 3., 7.])
    exact, _, _ = predict_rbf(xy, values, xy, options, smoothing=0)
    np.testing.assert_allclose(exact, values)
    smooth, support, _ = predict_rbf(xy, values, np.vstack((xy, [1200, 1200])), options, smoothing=1)
    assert not np.allclose(smooth[:4], values)
    assert np.isnan(smooth[-1]) and support[-1] == 0
    nearest, support, _ = predict_nearest(xy, values, np.array([[1., 0.], [500., 500.]]),
                                          replace(options, max_distance=100))
    np.testing.assert_equal(nearest, [1, np.nan])
    assert support.tolist() == [1, 0]


def test_common_metrics_can_use_a_pair_when_other_methods_lack_forecasts(data, qtbot):
    from dataclasses import replace
    from autopyro.spatial.kriging import KrigingOptions, run_kriging
    report, options = data
    from autopyro.ui.dialogs.spatial_interpolation import SpatialInterpolationDialog
    model = KrigingOptions(range_km=.2)
    result = run_kriging(report, replace(options, max_distance=1, grid_size=10),
                         model, .05, include_tin=True, rbf_smoothing=1)
    assert [row['predicted'] for row in result['validation']] == [0, 0, 1, 1, 0]
    assert all(row['common']['rmse'] is None for row in result['validation'])
    dialog = SpatialInterpolationDialog(report)
    qtbot.addWidget(dialog)
    dialog.completed_kriging(result, model)
    dialog.comparison_pair.setCurrentIndex(dialog.comparison_pair.findData(['TIN', 'RBF']))
    assert [row[6] for row in dialog.kriging_table.model().rows] == [1, 1]
    dialog.comparison_pair.setCurrentIndex(0)
    assert [row[6] for row in dialog.kriging_table.model().rows] == [0] * 5


def test_surface_layer_persistence_and_geotiff(data, tmp_path, monkeypatch):
    import ctypes as c
    from pyproj import CRS
    from autopyro.model.core import Project
    from autopyro.spatial.mapping import render_map
    from autopyro.spatial.surfaces import DIFFERENCE_METHOD, _gdal_library, compare_idw_tin_surfaces, export_geotiff, surface_layers

    report, options = data
    # A simple asymmetric grid catches north/south reversal and cell-center offsets.
    result = {'grid': dict(x=[500000, 500010], y=[10, 20], prediction=[[1, None], [3, 4]],
                           neighbors=[[3, 0], [4, 3]], nearest_distance_m=[[2, None], [1, 2]],
                           inside_domain=[[True, False], [True, True]])}
    labels = dict(prediction='Prediction', neighbors='Neighbors', nearest_distance_m='Distance')
    layers = surface_layers(result, 'IDW', 'EPSG:32631', 'TOC', labels, '%')
    path = tmp_path / 'surfaces.apyro'
    project = Project()
    project.view['map_layers'] = layers
    project.save(path)
    saved = Project.load(path).view['map_layers']
    assert saved == layers
    assert [layer['visible'] for layer in saved] == [True, False, False]
    mapped = build_map(Project(), [], MapSpec(location_mode='layers', target_crs='EPSG:4326'), saved)
    figure, _ = render_map(mapped)
    assert len(figure.axes) >= 2
    monkeypatch.setenv('PATH', '')
    monkeypatch.delenv('ProgramFiles', raising=False)
    output = tmp_path / 'surface.tif'
    export_geotiff(output, result, 'IDW', 'EPSG:32631', '%')
    gdal = _gdal_library()
    ptr, integer = c.c_void_p, c.c_int
    signatures = {
        'GDALGetGeoTransform': (integer, [ptr, c.POINTER(c.c_double)]),
        'GDALGetProjectionRef': (c.c_char_p, [ptr]),
        'GDALGetRasterCount': (integer, [ptr]),
        'GDALGetRasterBand': (ptr, [ptr, integer]),
        'GDALGetDescription': (c.c_char_p, [ptr]),
        'GDALGetRasterNoDataValue': (c.c_double, [ptr, c.POINTER(integer)]),
        'GDALRasterIO': (integer, [ptr, integer, integer, integer, integer, integer, ptr,
                                   integer, integer, integer, integer, integer]),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(gdal, name)
        function.restype, function.argtypes = restype, argtypes
    dataset = gdal.GDALOpen(str(output).encode('utf-8'), 0)
    assert dataset
    try:
        transform = (c.c_double * 6)()
        assert gdal.GDALGetGeoTransform(dataset, transform) == 0
        assert list(transform) == [499995, 10, 0, 25, 0, -10]
        assert CRS.from_wkt(gdal.GDALGetProjectionRef(dataset).decode()).to_epsg() == 32631
        assert gdal.GDALGetRasterCount(dataset) == 3
        for i, key in enumerate(labels, 1):
            band = gdal.GDALGetRasterBand(dataset, i)
            assert gdal.GDALGetDescription(band).decode() == key
            has_nodata = integer()
            assert np.isnan(gdal.GDALGetRasterNoDataValue(band, c.byref(has_nodata)))
            assert has_nodata.value
        pixels = (c.c_double * 4)()
        assert gdal.GDALRasterIO(gdal.GDALGetRasterBand(dataset, 1), 0, 0, 0, 2, 2, pixels, 2, 2, 7, 0, 0) == 0
        np.testing.assert_equal(list(pixels), [3, 4, 1, np.nan])
    finally:
        gdal.GDALClose(dataset)

    grid = dict(x=[500000, 500010, 500020], y=[10, 20], inside_domain=[[True, True, True], [True, True, False]])
    comparison = compare_idw_tin_surfaces(dict(grid, prediction=[[1, 3, np.nan], [4, np.nan, 100]]),
                                          dict(grid, prediction=[[3, np.nan, 5], [1, np.nan, 100]]))
    output = tmp_path / 'tin-minus-idw.tif'
    export_geotiff(output, dict(grid, **comparison), DIFFERENCE_METHOD, 'EPSG:32631', '%')
    dataset = gdal.GDALOpen(str(output).encode('utf-8'), 0)
    assert dataset
    try:
        assert gdal.GDALGetRasterCount(dataset) == 2
        for i, key, expected in [(1, b'difference', [-3, np.nan, np.nan, 2, np.nan, np.nan]),
                                 (2, b'coverage', [3, 0, np.nan, 3, 1, 2])]:
            band = gdal.GDALGetRasterBand(dataset, i)
            assert gdal.GDALGetDescription(band) == key
            pixels = (c.c_double * 6)()
            assert gdal.GDALRasterIO(band, 0, 0, 0, 3, 2, pixels, 3, 2, 7, 0, 0) == 0
            np.testing.assert_equal(list(pixels), expected)
        gdal.GDALGetMetadataItem.restype = c.c_char_p
        gdal.GDALGetMetadataItem.argtypes = [ptr, c.c_char_p, c.c_char_p]
        assert gdal.GDALGetMetadataItem(band, b'CODE_0', None) == b'Neither method'
    finally:
        gdal.GDALClose(dataset)


def test_idw_settings_score_only_common_forecast_locations(data):
    from dataclasses import replace
    report, options = data
    points = [(-100, -100, 10), (-100, 100, 10), (100, -100, 10), (100, 100, 10),
              (0, 0, 30), (40, 0, 0)]
    report['points'] = [{'id': str(i), 'x': x, 'y': y, 'values': {options.value_id: value}}
                        for i, (x, y, value) in enumerate(points)]
    variants = [dict(power=2, neighbors=8, max_distance=radius) for radius in (1000, 130)]
    result = compare_idw_settings(report, options, .025, variants)
    wide, narrow = result['variants']
    assert (wide['metrics']['predicted'], narrow['metrics']['predicted']) == (2, 1)
    common, = result['common_indices']
    assert result['locations'][common]['row_ids'] == ['5']
    for variant, metrics in zip(variants, result['variants']):
        individual = spatial_blocks(report, replace(options, **variant), .025, evaluate=True)
        held_out = next(p for p in individual['locations'] if p['row_ids'] == ['5'])
        assert metrics['common_metrics']['rmse'] == pytest.approx(abs(held_out['residual']))
        assert metrics['common_metrics']['bias'] == pytest.approx(held_out['residual'])
    empty = compare_idw_settings(report, options, .025, [variants[0], dict(variants[1], max_distance=1)])
    assert empty['common_indices'] == []
    assert all(row['common_metrics']['rmse'] is None for row in empty['variants'])


def test_idw_comparison_applies_and_exports_selected_settings(data, qtbot, monkeypatch, tmp_path):
    import json
    from PySide6.QtWidgets import QFileDialog
    from autopyro.ui.dialogs.map import SpatialInterpolationDialog
    report, options = data
    dialog = SpatialInterpolationDialog(report)
    qtbot.addWidget(dialog)
    dialog.form.load({'context': 'synthetic cohort', 'value_id': options.value_id, 'block_size_km': .05})
    radius = dialog.form.fields['max_distance'][0]
    assert radius.value() == 10 and dialog.form.values()['max_distance'] == 10000
    dialog.form.load({'max_distance': 125.01})
    assert radius.value() == pytest.approx(.12501)
    dialog.variant_fields['max_distance'].setText('0,12501')
    dialog.calculate_settings_comparison()
    qtbot.waitUntil(lambda: dialog.settings_result is not None and dialog.isEnabled(), timeout=10000)
    assert dialog.settings_table.model().rowCount() == 3
    dialog.settings_table.selectRow(2)
    dialog.apply_idw_variant()
    assert dialog.form.values()['power'] == 3
    assert dialog.form.values()['max_distance'] == pytest.approx(125.01)
    path = tmp_path / 'comparison.json'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), ''))
    dialog.export_settings_comparison()
    exported = json.loads(path.read_text(encoding='utf-8'))
    assert exported['selected_variant_index'] == 2
    assert exported['block_size_km'] == .05
    assert exported['variants'][2]['parameters']['max_distance'] == pytest.approx(125.01)
    dialog.form.reset_parameter('max_distance')
    assert radius.value() == 10 and dialog.form.values()['max_distance'] == 10000


def test_nested_idw_excludes_the_whole_outer_block(data):
    report, options = data
    coordinates = [(-100, -100), (-100, 100), (100, -100), (100, 100), (0, 0), (1, 1), (40, 0), (0, 40)]
    report['points'] = [{'id': str(i), 'x': x, 'y': y, 'values': {options.value_id: i ** 2}}
                        for i, (x, y) in enumerate(coordinates)]
    variants = [dict(power=power, neighbors=8, max_distance=1000) for power in (1, 3)]
    original = nested_idw_validation(report, options, .025, variants)
    assert original['baseline'] == spatial_blocks(report, options, .025, evaluate=True)['totals']
    block_id = next(point['block'] for point in original['locations'] if point['row_ids'] == ['4'])
    block = original['blocks'][block_id - 1]
    assert block['selected'] is not None
    assert block['locations'] == block['tuned']['predicted'] == 2
    for point in report['points'][4:6]:
        point['values'][options.value_id] += 1000
    updated = nested_idw_validation(report, options, .025, variants)
    changed = updated['blocks'][block_id - 1]
    assert changed['inner_scores'] == block['inner_scores']
    assert changed['selected'] == block['selected']
    assert changed['tuned']['rmse'] != block['tuned']['rmse']
    unavailable = nested_idw_validation(report, options, .025, [variants[0], dict(variants[1], max_distance=1)])
    assert all(row['selected'] is None for row in unavailable['blocks'])
    assert unavailable['tuned']['coverage'] == 0
    assert next(p for p in unavailable['locations'] if p['row_ids'] == ['4'])['reasons']['tuned'] == 'no_inner_selection'
    unavailable_fold = unavailable['blocks'][block_id - 1]
    assert unavailable_fold['inner_locations'] == 6
    assert unavailable_fold['inner_candidate_counts'][0] > 0
    assert unavailable_fold['inner_candidate_counts'][1] == unavailable_fold['inner_common'] == 0
    assert unavailable['tuned_common']['rmse'] is None


def test_nested_idw_dialog_shows_and_exports_both_results(data, qtbot, monkeypatch, tmp_path):
    import json
    from PySide6.QtWidgets import QFileDialog
    from autopyro.ui.dialogs.map import SpatialInterpolationDialog

    report, options = data
    dialog = SpatialInterpolationDialog(report)
    qtbot.addWidget(dialog)
    dialog.form.load({'context': options.context, 'value_id': options.value_id, 'block_size_km': .025})
    dialog.calculate_settings_comparison(nested=True)
    qtbot.waitUntil(lambda: dialog.nested_result is not None and dialog.isEnabled(), timeout=10000)
    assert dialog.nested_summary.model().rowCount() == 2
    assert dialog.nested_folds.model().rowCount() == 5
    assert not dialog.apply_variant.isEnabled()
    path = tmp_path / 'nested.json'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), ''))
    dialog.export_settings_comparison()
    assert json.loads(path.read_text(encoding='utf-8'))['format'] == 'autopyro.idw-nested-validation'
    dialog.variant_fields['power'].setText('1; 4')
    assert dialog.nested_result is None
    assert not dialog.export_variants.isEnabled()


@pytest.mark.parametrize('shifted', [False, True])
def test_nested_methods_share_outer_locations_without_leakage(data, shifted):
    from autopyro.spatial.kriging import KrigingOptions, compare_nested_methods
    from autopyro.spatial.tin import predict_tin

    report, options = data
    coordinates = [(-100, -100), (-100, 100), (100, -100), (100, 100), (0, 0), (1, 1), (40, 0), (0, 40)]
    report['points'] = [{'id': str(i), 'x': x, 'y': y, 'values': {options.value_id: i ** 2}}
                        for i, (x, y) in enumerate(coordinates)]
    variants = [dict(power=power, neighbors=8, max_distance=1000) for power in (1, 3)]
    args = (options, .025, variants, KrigingOptions(), [.03, .08])
    result = compare_nested_methods(report, *args, shifted=shifted)
    offset = (12.5, 12.5) if shifted else (0, 0)
    assert result['offset_m'] == list(offset)
    for name, predictor in [('baseline', predict), ('TIN', predict_tin)]:
        fixed = spatial_blocks(report, options, .025, evaluate=True, predictor=predictor, offset_m=offset)
        assert result['methods'][name]['metrics'] == fixed['totals']
    common = [point for point in result['locations'] if all(v is not None for v in point['predictions'].values())]
    assert common
    for name, stats in result['methods'].items():
        assert stats['common']['predicted'] == len(common)
        assert stats['common']['mae'] == pytest.approx(
            np.mean([abs(point['predictions'][name] - point['value']) for point in common]))

    outer = next(point['block'] for point in result['locations'] if point['row_ids'] == ['4'])
    for point in report['points'][4:6]:
        point['values'][options.value_id] += 1000
    changed = compare_nested_methods(report, *args, shifted=shifted)
    for name in ('IDW', 'kriging'):
        before = result['blocks'][outer - 1]['methods'][name]
        after = changed['blocks'][outer - 1]['methods'][name]
        assert before['selected'] == after['selected'] is not None
        assert before['inner_scores'] == after['inner_scores']
        assert before['metrics']['rmse'] != after['metrics']['rmse']


def test_nested_methods_dialog_uses_ranges_and_shift(data, qtbot):
    from autopyro.ui.dialogs.map import SpatialInterpolationDialog

    report, options = data
    dialog = SpatialInterpolationDialog(report)
    qtbot.addWidget(dialog)
    dialog.form.load({'context': options.context, 'value_id': options.value_id, 'block_size_km': .025})
    dialog.nested_ranges.setText('0,05; 0,1')
    dialog.nested_shift.setChecked(True)
    dialog.calculate_nested_methods()
    qtbot.waitUntil(lambda: dialog.nested_result is not None and dialog.isEnabled(), timeout=10000)
    assert dialog.nested_summary.model().rowCount() == 4
    assert dialog.nested_folds.model().rowCount() == 4 * len(dialog.nested_result['blocks'])
    assert [m['range_km'] for m in dialog.nested_result['candidates']['kriging']] == [.05, .1]
    assert dialog.nested_result['offset_m'] == [12.5, 12.5]
    dialog.kriging_form.fields['nugget'][0].setValue(.1)
    assert dialog.nested_result is None


def test_nested_partitions_use_same_locations_across_grids(data, qtbot):
    from autopyro.spatial.kriging import KrigingOptions, compare_nested_methods
    from autopyro.ui.dialogs.map import SpatialInterpolationDialog

    report, options = data
    coordinates = [(-100, -100), (-100, 100), (100, -100), (100, 100), (0, 0), (1, 1), (40, 0), (0, 40)]
    report['points'] = [dict(report['points'][0], id=str(i), x=x, y=y, values={options.value_id: i ** 2})
                        for i, (x, y) in enumerate(coordinates)]
    dialog = SpatialInterpolationDialog(report)
    qtbot.addWidget(dialog)
    dialog.form.load({'context': options.context, 'value_id': options.value_id, 'max_distance': 1000})
    dialog.partition_sizes.setText('0,025; 0,03')
    dialog.partition_nested.setChecked(True)
    dialog.calculate_partitions()
    qtbot.waitUntil(lambda: len(dialog.partition_results) == 16 and dialog.isEnabled(), timeout=15000)
    rows = dialog.partition_results
    available = [{(p['x'], p['y']) for p in row['locations'] if p['predicted'] is not None} for row in rows]
    shared = set.intersection(*available)
    assert shared
    for row in rows:
        errors = [p['residual'] for p in row['locations'] if (p['x'], p['y']) in shared]
        assert row['shared']['predicted'] == len(shared)
        assert row['shared']['rmse'] == pytest.approx(np.sqrt(np.mean(np.square(errors))))
    _, configured, _ = dialog.calculation_input()
    for first in range(0, 16, 4):
        batch = rows[first:first + 4]
        size, shift = batch[0]['block_size_km'], batch[0]['shift_km']
        single = compare_nested_methods(report, configured, size, dialog.variant_settings(configured),
                                        KrigingOptions(), [25, 50, 100], shifted=bool(shift))
        assert [row['common'] for row in batch] == [stats['common'] for stats in single['methods'].values()]
        assert min(batch, key=lambda row: row['common']['rmse'])['rank'] == 1
        baseline = next(row for row in batch if row['method'] == 'baseline')
        for row in batch:
            assert row['delta_rmse_idw'] == pytest.approx(row['common']['rmse'] - baseline['common']['rmse'])
            assert row['shared_delta_rmse_idw'] == pytest.approx(row['shared']['rmse'] - baseline['shared']['rmse'])
    dialog.partition_table.selectRow(6)
    assert dialog.residual_locations == rows[6]['locations']
    assert dialog.residual_table.model().rowCount() == len(coordinates)
    dialog.nested_ranges.setText('25; 100')
    assert dialog.partition_results == []


def test_partition_summary_keeps_completed_rows_and_explains_skips(data, qtbot, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDialog, QTableView
    from autopyro.ui.dialogs.map import SpatialInterpolationDialog

    report, options = data
    dialog = SpatialInterpolationDialog(report)
    qtbot.addWidget(dialog)
    dialog.form.load({'context': options.context, 'value_id': options.value_id, 'max_distance': 1000})
    dialog.partition_sizes.setText('10000; 0,025')
    dialog.partition_shift.setChecked(False)
    dialog.partition_nested.setChecked(True)
    dialog.preview_partition_setup()
    assert '25, 50, 100' in dialog.partition_setup.text()
    assert dialog.partition_results == []
    dialog.calculate_partitions()
    qtbot.waitUntil(lambda: len(dialog.partition_results) == 5 and dialog.isEnabled(), timeout=15000)
    skipped, *completed = dialog.partition_results
    assert skipped['block_size_km'] == 10000 and skipped['shift_km'] == 0
    assert all(row['shared'] == row['common'] for row in completed)
    assert dialog.partition_table.model().columnCount() == 9
    assert dialog.partition_table.selectionModel().selectedRows()[0].row() == 1
    assert dialog.residual_table.model() is not None
    assert dialog.partition_tuning.isEnabled()
    assert 'Подбор выполнен' in dialog.partition_details.text()

    def inspect_tuning(widget):
        model = widget.findChild(QTableView).model()
        assert model.rowCount() == len(completed[0]['blocks']) * 3
        assert 'Степень 1' in model.index(0, 1).data()
        assert model.rows[0][2] == completed[0]['blocks'][0]['methods']['IDW']['inner_locations']
        inspected.append(True)
        return 0

    inspected = []
    with monkeypatch.context() as patch:
        patch.setattr(QDialog, 'exec', inspect_tuning)
        dialog.partition_tuning.click()
    assert inspected
    def inspect_pair(widget):
        rows = widget.findChild(QTableView).model().rows
        assert len(rows) == 2
        assert rows[0][1] == rows[1][1]
        inspected.append('pair')
        return 0

    with monkeypatch.context() as patch:
        patch.setattr(QDialog, 'exec', inspect_pair)
        dialog.partition_comparison.partition_pair.click()
    assert inspected[-1] == 'pair'
    dialog.partition_table.selectRow(0)
    assert 'минимум три' in dialog.partition_details.text()
    assert not dialog.partition_tuning.isEnabled()
    assert not dialog.partition_comparison.partition_pair.isEnabled()
    assert dialog.residual_table.model() is None
    assert not dialog.partition_figure.axes
    for order in (Qt.SortOrder.AscendingOrder, Qt.SortOrder.DescendingOrder):
        dialog.partition_table.sortByColumn(3, order)
        assert dialog.partition_table.model().index(4, 2).data() == 'Пропущено'

    dialog.partition_sizes.setText('10000')
    dialog.calculate_partitions()
    qtbot.waitUntil(lambda: len(dialog.partition_results) == 1 and dialog.isEnabled(), timeout=15000)
    assert 'Ни одно' in dialog.status.text()


def test_spread_kriging_can_cancel_within_a_grid_fit(data):
    from dataclasses import replace
    from autopyro.spatial.spatial_interpolation import partition_spread
    from autopyro.spatial.kriging import KrigingOptions
    from autopyro.ui.tasks import TaskCancelled

    report, options = data
    options = replace(options, grid_size=25)
    events = []

    def cancel(index, total):
        events.append((index, total))
        if len(events) == 3:
            raise TaskCancelled()

    with pytest.raises(TaskCancelled):
        partition_spread(report, options, .025, method='kriging', progress=cancel)
    assert events == [(0, 5)] * 3

    model = KrigingOptions(range_km=.2, azimuth_deg=35, range_ratio=.5)
    expected = partition_spread(report, options, .025, method='kriging', kriging_model=model)
    actual = partition_spread(report, options, .025, method='kriging', kriging_model=model,
                              progress=lambda *_: None)
    np.testing.assert_equal(actual['spread'], expected['spread'])
    np.testing.assert_equal(actual['count'], expected['count'])


def test_idw_progress_can_stop_during_loo_and_retry(data):
    from autopyro.ui.tasks import TaskCancelled

    report, options = data
    before = deepcopy(report)
    events = []
    def cancel(fraction):
        events.append(fraction)
        if fraction > 0:
            raise TaskCancelled()
    with pytest.raises(TaskCancelled):
        interpolate(report, options, progress=cancel)
    assert 0 < events[-1] < .7
    assert report == before
    events.clear()
    actual = interpolate(report, options, progress=events.append)
    expected = interpolate(report, options)
    assert actual['grid'] == expected['grid']
    assert actual['diagnostics'] == expected['diagnostics']
    assert events == sorted(events)
    assert events[-1] == 1


def test_method_comparison_progress_and_cancellation(data):
    from autopyro.spatial.kriging import KrigingOptions, run_kriging
    from autopyro.ui.tasks import TaskCancelled

    report, options = data
    updates = []

    def cancel_at_validation(fraction):
        updates.append(fraction)
        if fraction >= .6:
            raise TaskCancelled()

    with pytest.raises(TaskCancelled):
        run_kriging(report, options, KrigingOptions(), .025, include_tin=True, progress=cancel_at_validation)
    assert updates == sorted(updates)
    assert updates[-1] == .6
    updates.clear()
    result = run_kriging(report, options, KrigingOptions(), .025, include_tin=True, progress=updates.append)
    assert updates[0] == 0 and updates[-1] == 1
    assert updates == sorted(updates)
    assert len(result['validation']) == 3


def test_nested_progress_can_cancel_inside_first_outer_block(data, monkeypatch):
    from autopyro.spatial import spatial_interpolation as spatial
    from autopyro.spatial.kriging import KrigingOptions, compare_nested_methods
    from autopyro.ui.tasks import TaskCancelled

    report, options = data
    coordinates = [(-100, -100), (-100, 100), (100, -100), (100, 100),
                   (0, 0), (1, 1), (40, 0), (0, 40)]
    report['points'] = [{'id': str(i), 'x': x, 'y': y, 'values': {options.value_id: i ** 2}}
                        for i, (x, y) in enumerate(coordinates)]
    variants = [dict(power=power, neighbors=8, max_distance=1000) for power in (1, 3)]
    calls = []
    original = spatial._domain_predict

    def forecast(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(spatial, '_domain_predict', forecast)
    updates = []

    def cancel_after_first_prediction(fraction):
        updates.append(fraction)
        if calls:
            raise TaskCancelled()

    with pytest.raises(TaskCancelled):
        compare_nested_methods(report, options, .025, variants, KrigingOptions(), [.03, .08],
                               progress=cancel_after_first_prediction)
    assert len(calls) == 1
    assert 0 < updates[-1] < 1 / len(coordinates)
    assert updates == sorted(updates)


def test_partition_progress_can_cancel_and_completed_batch_exports(data, qtbot, monkeypatch, tmp_path):
    import json
    from threading import Event

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QFileDialog, QProgressBar, QPushButton
    from autopyro.ui.dialogs import spatial_interpolation as map_ui

    report, options = data
    dialog = map_ui.SpatialInterpolationDialog(report)
    qtbot.addWidget(dialog)
    dialog.form.load({'context': options.context, 'value_id': options.value_id, 'max_distance': 1000})
    dialog.partition_sizes.setText('10000; 0,025')
    dialog.partition_shift.setChecked(False)
    dialog.partition_nested.setChecked(True)
    from autopyro.ui.dialogs import spatial_partition_comparison as partition_ui

    original = partition_ui.compare_nested_partitions
    released = Event()
    observed = []

    def wait_for_cancel(*args, progress, **kwargs):
        progress(0, 2, 10000, 0, .5)
        released.wait(5)
        progress(1, 2, .025, 0, 0)
        pytest.fail('Cancellation must interrupt the next progress checkpoint')

    def cancel():
        modal = QApplication.activeModalWidget()
        try:
            observed.append(modal.findChild(QProgressBar).value())
            modal.findChild(QPushButton).click()
        finally:
            released.set()

    monkeypatch.setattr(partition_ui, 'compare_nested_partitions', wait_for_cancel)
    QTimer.singleShot(100, cancel)
    dialog.calculate_partitions()
    assert observed == [25]
    assert 'отменена' in dialog.status.text()
    assert not dialog.partition_results and not dialog.export_partition_button.isEnabled()

    monkeypatch.setattr(partition_ui, 'compare_nested_partitions', original)
    dialog.calculate_partitions()
    assert dialog.export_partition_button.isEnabled()
    target = tmp_path / 'spatial-partitions'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(target), 'JSON (*.json)'))
    dialog.export_partition_button.click()
    exported = json.loads(target.with_suffix('.json').read_text(encoding='utf-8'))
    assert exported['block_sizes_km'] == [10000, .025]
    assert exported['map_spec']['target_crs'] == report['spec']['target_crs']
    assert exported['spatial_options']['max_distance'] == 1000
    assert exported['results'][0]['issue']
    tuned = next(row for row in exported['results'] if row['method'] == 'IDW')
    assert tuned['candidates'] == exported['idw_variants']
    assert len(tuned['blocks'][0]['methods']['IDW']['inner_candidate_counts']) == 3
    assert any(point['residual'] is not None for row in exported['results'][1:] for point in row['locations'])
    dialog.partition_sizes.setText('.05')
    assert not dialog.export_partition_button.isEnabled()


def test_block_validation_holds_out_all_locations_in_a_cell(data):
    report, options = data
    coordinates = [(-100, -100), (100, -100), (-100, 100), (100, 100), (0, 0), (1, 1)]
    report['points'] = [{'id': str(i), 'x': x, 'y': y, 'values': {options.value_id: value}}
                        for i, ((x, y), value) in enumerate(zip(coordinates, [10, 10, 10, 10, 6, 14]))]
    preview = spatial_blocks(report, options, .05)
    assert preview['totals'] is None
    result = spatial_blocks(report, options, .05, evaluate=True)
    assert [block['bounds'] for block in preview['blocks']] == [block['bounds'] for block in result['blocks']]
    predicted = [p for p in result['locations'] if p['predicted'] is not None]
    assert [p['predicted'] for p in predicted] == pytest.approx([10, 10])
    assert len({p['block'] for p in predicted}) == 1
    assert result['totals'] == pytest.approx(dict(predicted=2, coverage=2/6, rmse=4, mae=4, bias=0))
    from dataclasses import replace
    unsupported = spatial_blocks(report, replace(options, max_distance=1), .05, evaluate=True)
    assert unsupported['totals'] == dict(predicted=0, coverage=0, rmse=None, mae=None, bias=None)
    assert {p['reason'] for p in unsupported['locations']} == {'outside_hull', 'insufficient_neighbors'}


@pytest.mark.parametrize('nested', [False, True])
def test_settings_comparison_can_cancel_and_retry(data, qtbot, monkeypatch, nested):
    from autopyro.ui.dialogs import spatial_interpolation as module
    from autopyro.ui.tasks import TaskCancelled

    report, options = data
    dialog = module.SpatialInterpolationDialog(report)
    qtbot.addWidget(dialog)
    dialog.form.load({'value_id': options.value_id, 'context': options.context,
                      'block_size_km': .025, 'max_distance': options.max_distance})
    from autopyro.ui.dialogs import spatial_settings_comparison as settings_ui

    original = settings_ui.run_with_progress
    fractions = []
    def stop(parent, title, calculate, **kwargs):
        def update(stage, percent):
            fractions.append(percent)
            if percent > 0:
                raise TaskCancelled()
        return calculate(update)
    monkeypatch.setattr(settings_ui, 'run_with_progress', stop)
    dialog.calculate_settings_comparison(nested=nested)
    assert 'отменено' in dialog.status.text()
    assert dialog.settings_result is None and dialog.nested_result is None
    assert not dialog.export_variants.isEnabled() and not dialog.apply_variant.isEnabled()
    assert fractions == sorted(fractions) and 0 < fractions[-1] < 100
    monkeypatch.setattr(settings_ui, 'run_with_progress', original)
    dialog.calculate_settings_comparison(nested=nested)
    assert (dialog.nested_result if nested else dialog.settings_result) is not None
    assert dialog.export_variants.isEnabled()


def test_block_validation_can_stop_between_blocks_and_retry(data, qtbot, monkeypatch):
    from autopyro.spatial import spatial_interpolation as spatial
    from autopyro.ui.dialogs import spatial_interpolation as module
    from autopyro.ui.tasks import TaskCancelled

    report, options = data
    before = deepcopy(report)
    calls = []
    original = spatial._domain_predict
    def forecast(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    def stop(fraction):
        if calls:
            raise TaskCancelled()
    with monkeypatch.context() as patch:
        patch.setattr(spatial, '_domain_predict', forecast)
        with pytest.raises(TaskCancelled):
            spatial_blocks(report, options, .025, evaluate=True, progress=stop)
    assert len(calls) == 1 and report == before

    dialog = module.SpatialInterpolationDialog(report)
    qtbot.addWidget(dialog)
    dialog.form.load({'value_id': options.value_id, 'context': options.context,
                      'block_size_km': .025, 'max_distance': options.max_distance})
    def cancel(*args, **kwargs):
        raise TaskCancelled()
    from autopyro.ui.dialogs import spatial_blocks as blocks_ui

    with monkeypatch.context() as patch:
        patch.setattr(blocks_ui, 'run_with_progress', cancel)
        dialog.calculate_blocks(evaluate=True)
    assert 'отменена' in dialog.status.text()
    assert dialog.block_result is None and dialog.block_table.model().rowCount() == 0
    dialog.calculate_blocks(evaluate=True)
    assert dialog.block_result['totals'] == spatial_blocks(report, options, .025, evaluate=True)['totals']
    assert report == before


def test_block_preview_and_table_in_idw_dialog(data, qtbot):
    from autopyro.ui.dialogs.map import SpatialInterpolationDialog
    from shapely.geometry import box, mapping
    report, options = data
    report['layers'] = [dict(id='t3', name='T3', crs='EPSG:32631', visible=True, color='#333333',
        fill='#dddddd', alpha=.2, line_width=1, features=[
            dict(id='area', geometry=mapping(box(-1, -1, 101, 101)), properties={'Paleogeogr': 'Shelf'}),
        ])]
    dialog = SpatialInterpolationDialog(report)
    qtbot.addWidget(dialog)
    dialog.form.load({'value_id': options.value_id, 'block_size_km': .05})
    dialog.calculate_blocks(evaluate=False)
    qtbot.waitUntil(lambda: dialog.block_result is not None and dialog.isEnabled(), timeout=10000)
    assert len(dialog.block_patches) == len(dialog.block_result['blocks'])
    assert dialog.views.currentIndex() == 1
    assert dialog.block_table.model().rowCount() == len(dialog.block_patches) + 1
    dialog.block_table.selectRow(1)
    assert dialog.block_patches[0].get_linewidth() == 2.6
    dialog.form.fields['validation_buffer_km'][0].setValue(.005)
    dialog.calculate_blocks(evaluate=False)
    qtbot.waitUntil(lambda: dialog.block_result is not None and dialog.isEnabled(), timeout=10000)
    dialog.block_table.selectRow(1)
    assert dialog.block_buffer_outline.get_linestyle() == '--'
    dialog.form.fields['validation_buffer_km'][0].setValue(0)
    dialog.form.load({'context': 'synthetic cohort'})
    dialog.calculate_blocks(evaluate=True)
    qtbot.waitUntil(lambda: dialog.block_result is not None and dialog.block_result['evaluated'] and dialog.isEnabled(), timeout=10000)
    assert dialog.block_table.model().index(0, 3).data() == '20.0%'
    dialog.region_mode.setCurrentIndex(dialog.region_mode.findData('within'))
    assert dialog.region_field.currentData() == 'Paleogeogr'
    dialog.calculate_blocks(evaluate=True)
    qtbot.waitUntil(lambda: dialog.block_result is not None and dialog.block_result['evaluated'] and dialog.isEnabled(), timeout=10000)
    assert dialog.block_table.model().rowCount() == 3 * (len(dialog.block_patches) + 1)
    dialog.calculate()
    qtbot.waitUntil(lambda: dialog.result is not None and dialog.isEnabled(), timeout=10000)
    assert dialog.figure.axes[0].collections[0].get_clip_path() is not None


def test_region_modes_share_blocks_and_restrict_neighbors(data):
    from dataclasses import replace
    from shapely.geometry import box, mapping
    from autopyro.visualization.plotting import category_key
    report, options = data
    points = [(0, 0, 10), (0, 100, 10), (40, 0, 10), (40, 100, 10),
              (100, 0, 100), (100, 100, 100), (20, 50, 0), (70, 50, 0)]
    report['points'] = [{'id': str(i), 'x': x, 'y': y, 'values': {options.value_id: value}}
                        for i, (x, y, value) in enumerate(points)]
    report['layers'] = [dict(id='t3', crs='EPSG:32631', features=[
        dict(id='left', geometry=mapping(box(-10, -10, 50, 110)), properties={'Paleogeogr': 'Left'}),
        dict(id='right', geometry=mapping(box(50, -10, 110, 110)), properties={'Paleogeogr': 'Right'}),
    ])]
    restricted = replace(options, region_mode='within', region_layer='t3', region_field='Paleogeogr',
                         region_key=category_key('Left'))
    base = spatial_blocks(report, options, .025, evaluate=True)
    within = spatial_blocks(report, restricted, .025, evaluate=True)
    clipped = spatial_blocks(report, replace(restricted, region_mode='clip'), .025, evaluate=True)
    assert within['comparisons']['none'] == base['totals']
    assert within['comparisons']['clip'] == clipped['totals']
    assert [block['bounds'] for block in within['blocks']] == [block['bounds'] for block in base['blocks']]
    assert 0 < within['totals']['coverage'] <= clipped['totals']['coverage'] < base['totals']['coverage']
    by_id = {p['row_ids'][0]: p['predicted'] for p in within['locations']}
    assert by_id['6'] == pytest.approx(10)
    assert by_id['7'] is None
    assert next(p for p in within['locations'] if p['row_ids'] == ['7'])['reason'] == 'outside_region'
    base_value = next(p['predicted'] for p in base['locations'] if p['row_ids'] == ['6'])
    clipped_value = next(p['predicted'] for p in clipped['locations'] if p['row_ids'] == ['6'])
    assert clipped_value == pytest.approx(base_value) and base_value > 10
    surface = interpolate(report, restricted)
    assert surface['region_outline'] is not None
    assert all(row[-1] is None for row in surface['grid']['prediction'])
    clipped_surface = interpolate(report, replace(restricted, region_mode='clip'))
    assert clipped_surface['diagnostics']['loo_eligible'] == 5
    assert all(p['predicted'] is None and not p['eligible_for_validation']
               for p in clipped_surface['loo'] if p['x'] > 50)
    assert next(p for p in clipped_surface['loo'] if p['row_ids'] == ['6'])['predicted'] == clipped_value
    edge = next(i for i, x in enumerate(surface['grid']['x']) if 40 < x < 50)
    assert any(row[edge] is not None for row in clipped_surface['grid']['prediction'])
    assert all(not row[edge] for row in surface['grid']['inside_domain'])
    from autopyro.spatial.kriging import KrigingOptions, run_kriging
    kriging = run_kriging(report, restricted, KrigingOptions(), .025)
    assert not kriging['inside_domain'][:, edge].any()
    assert preview_sample_support(report, restricted, .025)['with_hull'] == within['totals']['predicted']

    # One point is on a polygon boundary and another intersects both polygons.
    report['layers'][0]['features'][1]['geometry'] = mapping(box(48, -10, 110, 110))
    report['points'].extend([
        dict(id='boundary', x=0, y=-10, values={options.value_id: 10}),
        dict(id='overlap', x=49, y=50, values={options.value_id: 10}),
    ])
    check = preview_sample_support(report, restricted, .025)
    assert check['unassigned_region'] == 2
    assert check['inside_region'] - check['candidates'] == 2
    checked = spatial_blocks(report, restricted, .025, evaluate=True)
    assert {p['reason'] for p in checked['locations'] if p['row_ids'][0] in ('boundary', 'overlap')} == {'ambiguous_region'}
    report['layers'][0]['features'][1]['properties']['Paleogeogr'] = 'Left'
    same_category = preview_sample_support(report, restricted, .025)
    assert same_category['unassigned_region'] == 1
    checked = spatial_blocks(report, restricted, .025, evaluate=True)
    assert next(p['reason'] for p in checked['locations'] if p['row_ids'] == ['overlap']) != 'ambiguous_region'


@pytest.fixture
def data():
    project=build_project(["X","Y","TOC"],[[0,0,1],[100,0,3],[0,100,3],[100,100,5],[50,50,3]],list(range(1,6)),
        [(0,"X","number","m"),(1,"Y","number","m"),(2,"TOC","number","%wt")],{})
    x,y,value=[c.id for c in project.columns]
    spec=MapSpec(x_id=x,y_id=y,source_crs="EPSG:32631",target_crs="EPSG:32631")
    report=build_map(project,[r.id for r in project.rows],spec,[])
    return report,SpatialOptions(value_id=value,context="Синтетический метод, одна формация и глубина",max_distance=1000,grid_size=10)


def test_idw_reference_and_local_diagnostics(data):
    report,options=data
    before=deepcopy(report)
    result=interpolate(report,options)
    assert report==before
    assert result["diagnostics"]["loo_valid"]==5
    assert result["diagnostics"]["loo_outside_training_hull"]==4
    assert result["diagnostics"]["rmse"]>0
    assert "NOT calibrated" in result["uncertainty_method"]
    assert all(v is not None for row in result["grid"]["prediction"] for v in row)
    xy=np.array([[0.,0.],[100,0],[0,100],[100,100]])
    values=np.array([1.,3.,3.,5.])
    predicted,_,_=predict(xy,values,np.array([[50.,50.],[0,0]]),options)
    assert predicted.tolist()==pytest.approx([3,1])


def test_duplicates_reject_and_hold_out_whole_location(data):
    report,options=data
    duplicate=deepcopy(report["points"][0])
    duplicate["id"]="duplicate"
    duplicate["row_ids"]=["duplicate"]
    duplicate["values"][options.value_id]=9
    report["points"].append(duplicate)
    with pytest.raises(ValueError,match="повторные координаты"):
        interpolate(report,options)
    from dataclasses import replace
    result=interpolate(report,replace(options,duplicate_policy="mean"))
    assert result["locations"][0]["value"]==5
    assert len(result["loo"][0]["row_ids"])==2
    assert result["diagnostics"]["locations"]==5


def test_mask_and_input_requirements(data):
    from dataclasses import replace
    report,options=data
    report["points"].pop(3) # triangle hull; bounding rectangle's corner is excluded
    result=interpolate(report,options)
    assert result["grid"]["prediction"][-1][-1] is None
    assert result["grid"]["error_proxy"][-1][-1] is None
    sparse = interpolate(report, replace(options, max_distance=1))
    assert sparse['diagnostics']['loo_valid'] == 0
    assert sparse['diagnostics']['grid_valid'] > 0
    assert all(value is None for row in sparse['grid']['error_proxy'] for value in row)
    with pytest.raises(ValueError,match="Сопоставимости|сопоставимости"):
        interpolate(report,replace(options,context=""))
    report["spec"]["target_crs"]="EPSG:4326"
    with pytest.raises(ValueError,match="метрах"):
        interpolate(report,options)


def test_qualifiers_and_collinear_locations(data):
    report,options=data
    report["points"][0]["values"][options.value_id]="<1"
    result=interpolate(report,options)
    assert result["excluded"]=={"censored":1}
    assert result["diagnostics"]["locations"]==4
    for point in report["points"]:
        point["y"]=0
    with pytest.raises(ValueError,match="линии|уникальных|повторные"):
        interpolate(report,options)
