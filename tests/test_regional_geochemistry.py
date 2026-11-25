from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest
from shapely.geometry import Polygon, mapping

from autopyro.io import build_project
from autopyro.mapping import MapSpec, build_map
from autopyro.regional_geochemistry import RegionalOptions, analyze_regions, prepare_input, render_distributions, export_regional_summary


@pytest.fixture
def source():
    headers = ['X', 'Y', 'TOC', 'HI', 'Tmax', 'Well', 'Age', 'Depth', 'Bottom', 'Method TOC', 'Method Pyrolysis']
    rows = [[1, 1, '1', '100', 430, 'A', 'T3', 100, 110, 'A', 'R'],
            [2, 1, '3', '200', 440, 'A', 'T3', 110, 120, 'A', 'S'],
            [3, 1, '<2', 'ND', None, 'B', 'T2-T3', 120, 130, 'B', 'R'],
            [4, 1, 'ND', '300', 450, 'B', 'T3', 130, 140, 'B', 'R'],
            [0, 1, '99', '999', 499, 'C', 'T3', 100, 110, 'A', 'R'],
            [20, 1, '99', '999', 499, 'C', 'T3', 100, 110, 'A', 'R']]
    project = build_project(headers, rows, list(range(len(rows))),
        [(i, name, 'number' if name in ['X', 'Y', 'TOC', 'Tmax', 'Depth', 'Bottom'] else 'text', '') for i, name in enumerate(headers)], {})
    ids = {c.name: c.id for c in project.columns}
    layer = dict(id='areas', name='Areas', crs='EPSG:32631', color='#333333', fill='#aaccee', alpha=.7, line_width=1, fields=['Paleogeogr'], features=[
        dict(id='0', geometry=mapping(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])), properties={'Paleogeogr': 'Shelf'})])
    report = build_map(project, [r.id for r in project.rows], MapSpec(x_id=ids['X'], y_id=ids['Y'], source_crs='EPSG:32631', target_crs='EPSG:32631'), [layer])
    options = RegionalOptions('areas', 'Paleogeogr', toc_id=ids['TOC'], hi_id=ids['HI'], tmax_id=ids['Tmax'],
        well_id=ids['Well'], age_id=ids['Age'], depth_id=ids['Depth'], toc_method_id=ids['Method TOC'], pyrolysis_method_id=ids['Method Pyrolysis'])
    return project, report, options, ids


def test_statistics_qualifiers_methods_and_snapshot(source, tmp_path):
    project, report, options, ids = source
    before = deepcopy(report)
    copied, spatial = prepare_input(project, report, options)
    result = analyze_regions(copied, spatial, options)
    assert report == before
    assert result['diagnostics']['spatial'] == {'matched': 4, 'boundary': 1, 'outside': 1}
    toc = next(r for r in result['summary'] if r['metric'] == 'TOC' and r['method'] == 'A')
    assert (toc['n_total'], toc['n_valid'], toc['n_wells'], toc['mean'], toc['median']) == (2, 2, 1, 2, 2)
    assert toc['std'] == pytest.approx(np.sqrt(2))
    assert toc['q1'] == 1.5
    excluded = next(r for r in result['summary'] if r['metric'] == 'TOC' and r['method'] == 'B')
    assert excluded['excluded'] == {'censored': 1, 'non_detect': 1}
    assert excluded['n_valid'] == 0 and excluded['std'] is None
    json.dumps(result, allow_nan=False)
    export_regional_summary(tmp_path / 'summary.csv', result)
    render_distributions(result, 'HI').savefig(tmp_path / 'plot.png')
    project.rows[0].values[ids['TOC']] = 888
    assert copied.rows[0].values[ids['TOC']] == 1


def test_exact_age_depth_and_separate_method_filters(source):
    project, report, options, ids = source
    result = analyze_regions(project, report, replace(options, ages=('T3',), depth_min=100, depth_max=130, toc_methods=('A',), pyrolysis_methods=('R',)))
    assert result['diagnostics']['filters'] == {'age_filter': 1}
    assert result['diagnostics']['method_filters'] == {'TOC': 1, 'HI': 1, 'Tmax': 1}
    assert next(r for r in result['summary'] if r['metric'] == 'HI')['mean'] == 200
    interval = analyze_regions(project, report, replace(options, depth_bottom_id=ids['Bottom'], depth_min=110, depth_max=120, stratify_methods=False))
    assert interval['assigned_row_ids'] == [project.rows[1].id]
    assert interval['summary'][0]['std'] is None
    assert interval['summary'][0]['mean'] == 3
    with pytest.raises(ValueError):
        analyze_regions(project, report, replace(options, depth_min=200, depth_max=100))


def test_well_medians_keep_raw_samples_and_spatial_issue_keys(source):
    project, report, options, ids = source
    missing = project.rows[-1].id
    report['points'] = [p for p in report['points'] if missing not in p['row_ids']]
    report['spec']['sample_key'] = ids['Well']
    options = replace(options, stratify_methods=False, depth_bottom_id=ids['Bottom'])
    samples = analyze_regions(project, report, options)
    wells = analyze_regions(project, report, replace(options, by_well=True))
    i = next(i for i, row in enumerate(wells['summary']) if row['metric'] == 'HI')
    assert samples['summary'][i]['mean'] == 200
    assert wells['summary'][i]['mean'] == 225
    assert wells['summary'][i]['n_valid'] == 3
    assert wells['summary'][i]['n_observations'] == 2
    assert [item['value'] for item in wells['distributions'][i]['values']] == [150, 300]
    records = wells['distributions'][i]['samples']
    assert len(records) == 4 and records[2]['status'] == 'non_detect'
    assert (records[0]['profile_depth'], records[0]['profile_bottom']) == (100, 110)
    assert {'row_id': missing, 'status': 'no_location', 'key': 'C'} in wells['spatial_issues']
    json.dumps(wells, allow_nan=False)
    report['points'] = []
    empty = analyze_regions(project, report, replace(options, by_well=True))
    assert len(empty['spatial_issues']) == len(project.rows)
    assert all(row['n_observations'] == 0 for row in empty['summary'])


def test_saved_comparison_retains_values_and_well_locations(source, qtbot, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QInputDialog
    from autopyro.core import Project
    from autopyro.regional_comparison import compare_regions, SCENARIO_COLORS
    from autopyro.ui.dialogs.regional_geochemistry import RegionalGeochemistryDialog
    project, report, options, ids = source
    dialog = RegionalGeochemistryDialog(project, report)
    qtbot.addWidget(dialog)
    options = replace(options, stratify_methods=False)
    names = iter(['A shallow', 'B deep'])
    monkeypatch.setattr(QInputDialog, 'getText', lambda *args: (next(names), True))
    for bounds in [(100, 110), (110, 140)]:
        result = analyze_regions(project, report, replace(options, depth_min=bounds[0], depth_max=bounds[1]))
        dialog.show_result(result)
        dialog.capture_comparison_snapshot()
    path = tmp_path / 'comparison.apyro'
    project.save(path)
    # The reopened map scope excludes all of the original observations.
    changed_report = dict(report, row_ids=[], points=[])
    restored = RegionalGeochemistryDialog(Project.load(path), changed_report)
    qtbot.addWidget(restored)
    restored.tabs.setCurrentWidget(restored.comparison)
    selections = []
    restored.map_selection_changed.connect(selections.append)
    restored.comparison.metric_combo.setCurrentText('HI')
    a, b = restored.comparison.pair()
    assert a['conditions']['options']['depth_max'] == 110
    assert a['conditions']['scope_row_ids'] == report['row_ids']
    row = restored.comparison.rows[0]
    assert (row['a']['median'], row['b']['median'], row['delta'], row['common_wells']) == (150, 250, 100, 1)
    lines = restored.comparison.canvas.figure.axes[0].lines
    assert len(lines) == 2 and list(lines[0].get_xdata()) == [100, 100, 200]
    points = selections[-1]['snapshot_points']
    assert {group['color'] for group in points} == {SCENARIO_COLORS['B'], SCENARIO_COLORS['AB']}
    assert {xy for group in points for xy in group['xy']} == {(1, 1), (2, 1), (4, 1)}
    restored.comparison.paired.setChecked(True)
    paired = restored.comparison.rows[0]['paired']
    assert paired['wells'] == [{'well': 'A', 'a': 150, 'b': 200, 'delta': 50,
                               'median_without': None, 'sign_changed': None}]
    assert paired['without_min'] is None and paired['sign_changed'] is None
    assert (paired['n'], paired['median'], paired['q1'], paired['q3']) == (1, 50, 50, 50)
    assert restored.comparison.well_table.model().rowCount() == 1
    assert {group['color'] for group in selections[-1]['snapshot_points']} == {SCENARIO_COLORS['AB']}
    assert selections[-1]['paired_map']['values'] == {row['region_key']: 50}
    legacy = dict(b, version=1)
    assert compare_regions(a, legacy, 'HI', 'pooled', 'pooled')[0]['delta'] == 100
    with pytest.raises(ValueError, match='старые снимки'):
        compare_regions(a, legacy, 'HI', 'pooled', 'pooled', paired=True)
    incompatible = deepcopy(b)
    incompatible['conditions']['options']['by_well'] = True
    assert compare_regions(a, incompatible, 'HI', 'pooled', 'pooled', paired=True)[0]['paired'] == paired
    with pytest.raises(ValueError, match='режим расчёта'):
        compare_regions(a, incompatible, 'HI', 'pooled', 'pooled')
    incompatible = deepcopy(b)
    incompatible['conditions']['region_layer']['fingerprint'] = 'changed'
    with pytest.raises(ValueError, match='геометрия'):
        compare_regions(a, incompatible, 'HI', 'pooled', 'pooled')


@pytest.mark.parametrize('differences', [[-4, 2], [-3, 0, 2], [-4, 1, 1, 6], [0, 0, 0], [1, 2, 3, 9, 9]])
def test_paired_stability_matches_recalculation(differences):
    from statistics import median
    from autopyro.regional_comparison import paired_wells
    left = {'well_medians': {str(i): {'well': i, 'median': 0} for i in range(len(differences))}}
    right = {'well_medians': {str(i): {'well': i, 'median': delta} for i, delta in enumerate(differences)}}
    stats = paired_wells(left, right)
    expected = [median(differences[:i] + differences[i + 1:]) for i in range(len(differences))]
    changes = [np.sign(value) != np.sign(median(differences)) for value in expected]
    assert [row['median_without'] for row in stats['wells']] == expected
    assert [row['sign_changed'] for row in stats['wells']] == changes
    assert (stats['without_min'], stats['without_max']) == (min(expected), max(expected))
    assert stats['sign_changed'] == any(changes)


def test_detail_tabs_and_scenario_roundtrip(source, qtbot, tmp_path, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QInputDialog
    from autopyro.core import Project
    from autopyro.ui.dialogs.regional_geochemistry import RegionalGeochemistryDialog, SPATIAL_COLORS
    project, report, options, ids = source
    dialog = RegionalGeochemistryDialog(project, report)
    qtbot.addWidget(dialog)
    selection, opened = [], []
    dialog.map_selection_changed.connect(selection.append)
    dialog.sample_opened.connect(opened.append)
    dialog.bindings['depth_bottom_id'].setCurrentIndex(dialog.bindings['depth_bottom_id'].findData(ids['Bottom']))
    dialog.stratify.setChecked(False)
    dialog.by_well.setChecked(True)
    dialog.show_result(analyze_regions(project, report, dialog.options()))
    dialog.table.selectRow(1)
    dialog.tabs.setCurrentWidget(dialog.samples_tab)
    assert dialog.samples.model().rowCount() == 4
    dialog.samples.selectRow(2)
    assert selection[-1]['row_ids'] == {project.rows[2].id}
    dialog.samples.doubleClicked.emit(dialog.samples.model().index(2, 0))
    assert opened == [project.rows[2].id]
    dialog.tabs.setCurrentWidget(dialog.profile_tab)
    dialog.profile_metric.setCurrentText('HI')
    ax = dialog.profile_canvas.figure.axes[0]
    limits = ax.get_ylim()
    assert limits[0] > limits[1] and dialog.profile_well.count() == 2
    dialog.profile_well.setCurrentIndex(1)
    assert dialog.profile_canvas.figure.axes[0].get_ylim() == limits
    dialog.tabs.setCurrentWidget(dialog.spatial_tab)
    assert dialog.spatial_table.model().rowCount() == 2
    assert selection[-1]['point_groups'][SPATIAL_COLORS['boundary']] == {project.rows[4].id}
    dialog.spatial_filter.setCurrentIndex(dialog.spatial_filter.findData('outside'))
    assert dialog.spatial_table.model().rowCount() == 1
    dialog.choices['ages'].item(0).setCheckState(Qt.CheckState.Checked)
    dialog.depth_min.setText('110')
    saved = dialog.options()
    monkeypatch.setattr(QInputDialog, 'getText', lambda *args, **kwargs: ('T3', True))
    dialog.save_scenario()
    path = tmp_path / 'regional.apyro'
    project.save(path)
    restored = RegionalGeochemistryDialog(Project.load(path), report)
    qtbot.addWidget(restored)
    restored.load_scenario()
    assert restored.options() == saved


def test_text_interpolation_and_stale_exclusions(source):
    project, report, options, ids = source
    result = analyze_regions(project, report, replace(options, text_numbers=False))
    assert all(row['n_valid'] == 0 for row in result['summary'] if row['metric'] == 'HI')
    col = next(c for c in project.columns if c.id == ids['TOC'])
    col.metadata['method'] = 'linear'
    result = analyze_regions(project, report, options)
    assert all(row['n_valid'] == 0 for row in result['summary'] if row['metric'] == 'TOC')
    result = analyze_regions(project, report, replace(options, include_interpolated=True))
    assert sum(row['n_valid'] for row in result['summary'] if row['metric'] == 'TOC') == 2


def test_dialog_calculates_invalidates_and_exports(source, qtbot, tmp_path, monkeypatch):
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QFileDialog
    from autopyro.ui.dialogs.regional_geochemistry import RegionalGeochemistryDialog
    project, report, options, ids = source
    dialog = RegionalGeochemistryDialog(project, report)
    dialog.setFont(QFont('Segoe UI', 9))
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog.region.currentText() == 'Paleogeogr'
    assert dialog.bindings['hi_id'].currentData() == ids['HI']
    dialog.calculate()
    qtbot.waitUntil(lambda: dialog.result is not None and dialog.isEnabled(), timeout=15000)
    assert dialog.table.rowCount() == 6
    dialog.metric_combo.setCurrentText('HI')
    assert dialog.result is not None
    for kind, suffix in [('summary', 'csv'), ('values', 'csv'), ('json', 'json'), ('figure', 'png')]:
        path = tmp_path / f'{kind}.{suffix}'
        monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, chosen=str(path): (chosen, ''))
        dialog.export(kind)
        assert path.stat().st_size > 20
    dialog.grab().save(str(tmp_path / 'dialog.png'))
    dialog.depth_min.setText('120')
    assert dialog.result is None and not dialog.exports[0].isEnabled()


def test_qc_exclusions_are_revision_aware(source):
    project, report, options, ids = source
    project.sync_measurements()
    measurement = project.science.current(project.rows[0].id, ids['TOC'])
    project.view['qc_exclusions'] = {'statistics': {measurement.id: {'revision': measurement.revision}}}
    copied, spatial = prepare_input(project, report, options)
    result = analyze_regions(copied, spatial, options)
    row = next(r for r in result['summary'] if r['metric'] == 'TOC' and r['method'] == 'A')
    assert row['mean'] == 3 and row['excluded'] == {'qc_excluded': 1}
    measurement.revision += 1
    result = analyze_regions(project, report, options)
    row = next(r for r in result['summary'] if r['metric'] == 'TOC' and r['method'] == 'A')
    assert row['mean'] == 2


def test_summary_selection_highlights_map_and_respects_group_filters(source, qtbot):
    from PySide6.QtCore import QItemSelectionModel
    from matplotlib.collections import PathCollection
    from autopyro.ui.window import MainWindow
    project, report, options, ids = source
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_maps()
    map_dialog = owner.map_window
    map_dialog.layers = deepcopy(report['layers'])
    # Several disjoint polygons with the same category must all be highlighted.
    extra = deepcopy(map_dialog.layers[0]['features'][0])
    extra['id'] = 'second'
    extra['geometry'] = mapping(Polygon([(30, 0), (35, 0), (35, 5), (30, 5)]))
    map_dialog.layers[0]['features'].append(extra)
    coast = deepcopy(extra)
    coast['id'] = 'coast'
    coast['properties']['Paleogeogr'] = 'Coast'
    coast['geometry'] = mapping(Polygon([(40, 0), (45, 0), (45, 5), (40, 5)]))
    map_dialog.layers[0]['features'].append(coast)
    map_dialog.data.load(dict(location_mode='columns', x_id=ids['X'], y_id=ids['Y'], source_crs='EPSG:32631', target_crs='EPSG:32631'))
    map_dialog.redraw()
    assert map_dialog.report is not None
    initial_selection = map_dialog.selected_ids()
    limits = map_dialog.result['axes'].get_xlim(), map_dialog.result['axes'].get_ylim()
    map_dialog.open_regional_geochemistry()
    dialog = map_dialog.regional_window
    assert dialog.isVisible() and not dialog.isModal()
    # Exact age excludes T2-T3; TOC method B still has one ND measurement.
    for i in range(dialog.choices['ages'].count()):
        item = dialog.choices['ages'].item(i)
        if item.data(256) == 'T3':
            from PySide6.QtCore import Qt
            item.setCheckState(Qt.CheckState.Checked)
    dialog.calculate()
    qtbot.waitUntil(lambda: dialog.result is not None and dialog.isEnabled(), timeout=15000)
    summary = dialog.result['summary']
    group_a = next(i for i, r in enumerate(summary) if r['metric'] == 'TOC' and r['method'] == 'A')
    group_b = next(i for i, r in enumerate(summary) if r['metric'] == 'TOC' and r['method'] == 'B')
    dialog.table.selectRow(group_a)
    rings = [a for a in map_dialog.regional_highlights if isinstance(a, PathCollection)]
    assert len(rings) == 1 and len(rings[0].get_offsets()) == 2
    assert len(map_dialog.regional_highlights) == 3  # Both polygons and the sample collection.
    dialog.table.selectRow(group_b)
    rings = [a for a in map_dialog.regional_highlights if isinstance(a, PathCollection)]
    assert len(rings[0].get_offsets()) == 1  # ND is part of n_total; boundary/outside excluded.
    dialog.table.selectionModel().select(dialog.table.model().index(group_a, 0),
        QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
    rings = [a for a in map_dialog.regional_highlights if isinstance(a, PathCollection)]
    assert len(rings[0].get_offsets()) == 3
    assert map_dialog.selected_ids() == initial_selection
    assert (map_dialog.result['axes'].get_xlim(), map_dialog.result['axes'].get_ylim()) == limits
    from types import SimpleNamespace

    def click_region(x, y):
        ax = map_dialog.result['axes']
        px, py = ax.transData.transform((x, y))
        event = SimpleNamespace(x=px, y=py, button=1, inaxes=ax)
        map_dialog.feature_press(event)
        map_dialog.feature_release(event)

    # A second polygon with the same region selects all of its metric/method rows.
    dialog.tabs.setCurrentIndex(1)
    click_region(32, 2)
    selected = {i.row() for i in dialog.table.selectionModel().selectedRows()}
    assert selected == {i for i, r in enumerate(summary) if r['region'] == 'Shelf'}
    assert dialog.tabs.currentWidget() is dialog.table
    rings = [a for a in map_dialog.regional_highlights if isinstance(a, PathCollection)]
    assert len(rings[0].get_offsets()) == 3
    # Selecting a region with no samples replaces the previous rows and keeps its outline.
    click_region(42, 2)
    selected = {i.row() for i in dialog.table.selectionModel().selectedRows()}
    assert selected == {i for i, r in enumerate(summary) if r['region'] == 'Coast'}
    assert len(selected) == 3
    assert len(map_dialog.regional_highlights) == 1
    assert not any(isinstance(a, PathCollection) for a in map_dialog.regional_highlights)
    # Navigation gestures and selections from other layers do not change the summary.
    map_dialog.toolbar.pan()
    click_region(32, 2)
    assert {i.row() for i in dialog.table.selectionModel().selectedRows()} == selected
    map_dialog.toolbar.pan()
    dialog.select_map_region('other-layer', '0')
    assert {i.row() for i in dialog.table.selectionModel().selectedRows()} == selected
    dialog.table.selectRow(group_a)
    dialog.tabs.setCurrentWidget(dialog.samples_tab)
    dialog.samples.selectRow(1)
    rings = [a for a in map_dialog.regional_highlights if isinstance(a, PathCollection)]
    assert len(rings) == 1 and rings[0].get_offsets().tolist() == [[2, 1]]
    dialog.samples.doubleClicked.emit(dialog.samples.model().index(1, 0))
    assert map_dialog.selected_ids() == {project.rows[1].id}
    dialog.depth_min.setText('120')
    assert not map_dialog.regional_highlights
    dialog.calculate()
    qtbot.waitUntil(lambda: dialog.result is not None and dialog.isEnabled(), timeout=15000)
    dialog.table.selectRow(0)
    assert map_dialog.regional_highlights  # The region is outlined even for an empty metric group.
    result, revision = dialog.result, dialog.input_revision
    selected = dialog.table.selectionModel().selectedRows()[0].row()
    map_dialog.style.load({'color': '#d55e00'})
    map_dialog.layers[0]['fill'] = '#bbccdd'
    map_dialog.redraw()
    assert dialog.result is result and dialog.report is map_dialog.report
    assert dialog.input_revision == revision
    assert dialog.table.selectionModel().selectedRows()[0].row() == selected
    assert all(button.isEnabled() for button in dialog.exports)
    assert map_dialog.regional_highlights

    map_dialog.layers[0]['filter'] = dict(field='Paleogeogr', op='ne', value='Absent')
    map_dialog.redraw()
    assert dialog.result is None and dialog.report is map_dialog.report
    assert not dialog.exports[0].isEnabled()
    assert not map_dialog.regional_highlights
    click_region(32, 2)
    assert not dialog.table.selectionModel().selectedRows()
    dialog.calculate()
    qtbot.waitUntil(lambda: dialog.result is not None and dialog.isEnabled(), timeout=15000)
    dialog.table.selectRow(0)
    dialog.close()
    assert not map_dialog.regional_highlights
    map_dialog.highlight_regional_selection({'snapshot_points': [
        {'xy': [(3, 0)], 'crs': 'EPSG:4326', 'color': '#0072b2'}]})
    assert len(map_dialog.regional_highlights) == 1
    assert map_dialog.regional_highlights[0].get_offsets()[0].tolist() == pytest.approx([500000, 0])
    from matplotlib.colors import to_rgba
    from matplotlib.patches import PathPatch
    from autopyro.regional_comparison import freeze_regional_result

    snapshots = {}
    for name, bounds in [('A', (100, 110)), ('B', (110, 140))]:
        opts = replace(options, stratify_methods=False, depth_min=bounds[0], depth_max=bounds[1])
        result = analyze_regions(project, map_dialog.report, opts)
        snapshots[name] = freeze_regional_result(result, map_dialog.report)
    comparison = dialog.comparison
    comparison.snapshots = snapshots
    comparison.refresh_snapshots()
    comparison.metric_combo.setCurrentText('HI')
    comparison.paired.setChecked(True)
    map_dialog.open_regional_geochemistry()
    dialog.tabs.setCurrentWidget(comparison)
    fills = [artist.get_facecolor() for artist in map_dialog.regional_highlights if isinstance(artist, PathPatch)]
    assert len(fills) == 3 and fills.count(to_rgba('#a7a7a7')) == 1
    assert len(map_dialog.result['axes'].child_axes) == 1
    legend = map_dialog.result['axes'].child_axes[0]
    assert [label.get_text() for label in legend.child_axes[0].get_xticklabels()] == ['-50', '0', '50']
    click_region(42, 2)
    selected = comparison.table.selectionModel().selectedRows()[0].row()
    assert comparison.rows[selected]['region'] == 'Coast'
    assert comparison.well_table.model().rowCount() == 0
    comparison.b.setCurrentIndex(comparison.b.findData('A'))
    legend = map_dialog.result['axes'].child_axes[0]
    assert not legend.child_axes
    assert {label.get_text() for label in legend.texts} >= {'0', 'Нет пар'}
    from autopyro.regional_comparison import UNSTABLE_REGION_COLOR
    layer = comparison.paired_map()
    shelf = next(row['region_key'] for row in comparison.rows if row['region'] == 'Shelf')
    layer.update(selected=shelf, unstable={shelf})
    map_dialog.highlight_regional_selection({'paired_map': layer})
    edges = [artist.get_edgecolor() for artist in map_dialog.regional_highlights if isinstance(artist, PathPatch)]
    assert edges.count(to_rgba(UNSTABLE_REGION_COLOR)) == 2
    dialog.tabs.setCurrentWidget(dialog.table)
    assert not map_dialog.result['axes'].child_axes
    toc_column = next(i for i, column in enumerate(project.columns) if column.id == ids['TOC'])
    assert owner.model.setData(owner.model.index(0, toc_column), '42')
    map_dialog.redraw()
    assert dialog.result is None and not dialog.exports[0].isEnabled()

    dialog.calculate()
    qtbot.waitUntil(lambda: dialog.result is not None and dialog.isEnabled(), timeout=15000)
    from autopyro.ui.table.commands import QCCommand
    measurement = project.science.current(project.rows[0].id, ids['TOC'])
    revision = owner.model.data_revision
    owner.model.undo.push(QCCommand(owner.model, ('qc_exclusions', 'statistics'),
                                    {'revision': measurement.revision}, measurement_id=measurement.id))
    assert owner.model.data_revision == revision
    map_dialog.redraw()
    assert dialog.result is None and not dialog.exports[0].isEnabled()
    owner.dirty = False


def test_invalidating_running_analysis_discards_old_result(source, qtbot, monkeypatch):
    from threading import Event
    from autopyro.ui.dialogs import regional_geochemistry_inputs as inputs_ui
    from autopyro.ui.dialogs.regional_geochemistry import RegionalGeochemistryDialog
    project, report, _, _ = source
    dialog = RegionalGeochemistryDialog(project, report)
    qtbot.addWidget(dialog)
    release = Event()
    original = inputs_ui.analyze_regions

    def delayed(*args):
        release.wait(5)
        return original(*args)

    monkeypatch.setattr(inputs_ui, 'analyze_regions', delayed)
    dialog.calculate()
    try:
        dialog.invalidate()
    finally:
        release.set()
    qtbot.waitUntil(lambda: dialog.isEnabled() and not dialog.worker.isRunning(), timeout=10000)
    assert dialog.result is None
    assert not dialog.exports[0].isEnabled()
