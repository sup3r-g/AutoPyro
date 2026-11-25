"""Analysis documents share choices and resolve saved views independently of the table."""

import pytest
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import QDialogButtonBox, QMessageBox

from autopyro.core import Project
from autopyro.io import build_project
from autopyro.mapping import MapSpec, build_map
from autopyro.ui.dialogs.map import SpatialInterpolationDialog
from autopyro.ui.dialogs.report import ReportDialog
from autopyro.ui.dialogs.geometry_editor import GeometryEditor
from autopyro.ui.widgets.documents import source_token
from autopyro.ui.window import MainWindow


@pytest.fixture
def owner(qtbot):
    project = build_project(['Well', 'X', 'Y', 'TOC', 'S1', 'S2', 'S3'],
        [['A', 0, 2, 2, .5, 6, 1.2], ['A', 1, '', 3, .4, 9, 1.4],
         ['A', 2, 6, 4, .8, 12, 1.6], ['B', 3, 8, 5, 1, 15, 2]], [2, 3, 4, 5],
        [(0, 'Well', 'text', ''), (1, 'X', 'number', 'm'), (2, 'Y', 'number', ''),
         (3, 'TOC', 'number', '%wt'), (4, 'S1', 'number', 'mg HC/g rock'),
         (5, 'S2', 'number', 'mg HC/g rock'), (6, 'S3', 'number', 'mg CO2/g rock')], {})
    x = project.columns[1].id
    project.view['named'] = {
        'High': {'rules': [{'column': x, 'operator': 'ge', 'value': 2}], 'combine': 'AND',
                 'sort': [(x, True)], 'sort_column': 1},
        'Gap': {'rules': [{'column': x, 'operator': 'le', 'value': 2}], 'combine': 'AND'},
        'Empty': {'rules': [{'column': x, 'operator': 'lt', 'value': -1}], 'combine': 'AND'},
    }
    window = MainWindow()
    qtbot.addWidget(window, before_close_func=lambda w: setattr(w, 'dirty', False))
    window.set_project(project)
    window.autosave_timer.stop()
    window.may_replace = lambda: True
    yield window
    window.desktop.close_documents()
    window.dirty = False


def choose(combo, mode):
    index = combo.findData(mode)
    assert index >= 0, mode
    combo.setCurrentIndex(index)


def test_missing_saved_set_is_rejected_while_empty_set_is_valid(owner):
    owner.model.project.view['sample_sets'] = {'Empty': []}
    assert owner.analysis_scope.rows('set:Empty') == []
    with pytest.raises(ValueError, match='Сохранённый набор проб отсутствует'):
        owner.analysis_scope.rows('set:Deleted')


@pytest.mark.parametrize('mode', ['view:Review', 'set:Review'])
def test_saved_view_refreshes_when_linked_set_membership_changes(owner, mode):
    project = owner.model.project
    first, second = [row.id for row in project.rows[:2]]
    project.view['sample_sets'] = {'Review': [first]}
    project.view['named']['Review'] = {'rules': [], 'combine': 'AND', 'sample_set': 'Review'}
    scope = owner.analysis_scope
    scope.refresh()
    scope.set_mode(mode)
    assert scope.rows() == [first]
    revision = scope.revision

    project.view['sample_sets']['Review'][:] = [second]
    scope.refresh()

    assert scope.rows() == [second]
    assert scope.revision > revision


def stop_timers(owner):
    for name in ('plot_window', 'map_window', 'statistics_window', 'interpretation_window', 'qc_window'):
        document = getattr(owner, name)
        if document is not None:
            document.timer.stop()
    if owner.interpretation_window is not None:
        owner.interpretation_window.plot.timer.stop()


def test_every_document_has_synchronized_live_view_choices(owner, qtbot):
    for method in ('show_plots', 'show_maps', 'show_statistics', 'show_interpretation',
                   'show_calculations', 'interpolate', 'show_nonpyro', 'show_stratigraphy', 'show_qc'):
        getattr(owner, method)()
        stop_timers(owner)
    report = ReportDialog(owner)
    qtbot.addWidget(report)
    project = owner.model.project
    snapshot = build_map(project, [row.id for row in project.rows], MapSpec(
        x_id=project.columns[1].id, y_id=project.columns[3].id,
        source_crs='EPSG:32631', target_crs='EPSG:32631'), [])
    spatial = SpatialInterpolationDialog(snapshot, owner.map_window)
    qtbot.addWidget(spatial)
    controls = [owner.analysis_scope_combo, owner.statistics_window.scope,
        owner.map_window.data.fields['scope'][0], owner.interpretation_window.scope,
        owner.interpretation_window.plot.scope, owner.calculation_window.scope,
        owner.interpolation_window.scope, owner.nonpyro_window.form.fields['scope'][0],
        owner.stratigraphy_window.scope, owner.qc_window.data_scope, report.scope, spatial.scope]
    expected = {'filtered', 'all', 'selected', 'view:High', 'view:Gap', 'view:Empty'}
    for combo in controls:
        assert {combo.itemData(i) for i in range(combo.count())} == expected
        choose(combo, 'view:High')
        assert all(other.currentData() == 'view:High' for other in controls)
        assert owner.analysis_scope.rows() == [project.rows[i].id for i in (3, 2)]
        choose(combo, 'all')
        assert all(other.currentData() == 'all' for other in controls)
    plot = owner.plot_window
    assert plot.scope.currentData() == 'global'
    choose(plot.scope, 'view:High')
    assert owner.analysis_scope.mode == 'all'
    assert plot.scope_ids() == [project.rows[i].id for i in (3, 2)]
    choose(plot.scope, 'global')
    # Saving a new table view updates controls that are already open.
    properties = owner.desktop.table_properties
    properties.view_name.setText('New view')
    properties.save_view()
    assert all(combo.findData('view:New view') >= 0 for combo in controls)
    choose(report.scope, 'view:New view')
    project.view['named'].pop('New view')
    owner.model_changed()
    assert all(combo.currentData() == 'filtered' for combo in controls)
    assert all(combo.findData('view:New view') < 0 for combo in controls)
    owner.analysis_scope.create_view('From toolbar')
    assert all(combo.currentData() == 'view:From toolbar' for combo in controls)
    owner.analysis_scope.rename_view('From toolbar', 'Renamed from toolbar')
    assert all(combo.currentData() == 'view:Renamed from toolbar' for combo in controls)
    assert all(combo.findData('view:From toolbar') < 0 for combo in controls)
    owner.analysis_scope.delete_view('Renamed from toolbar')
    assert all(combo.currentData() == 'filtered' for combo in controls)
    stop_timers(owner)
    report.close()
    spatial.close()


def test_shared_row_resolution_empty_selection_and_view_edits(owner):
    project = owner.model.project
    x = project.columns[1].id
    owner.apply_filter([{'column': x, 'operator': 'le', 'value': 1}], 'AND')
    current = owner.row_ids()
    owner.show_plots()
    owner.show_maps()
    owner.show_nonpyro()
    readers = [owner.analysis_scope.rows, owner.plot_window.scope_ids,
               owner.map_window.scope_ids, owner.nonpyro_window.row_ids]
    choose(owner.analysis_scope_combo, 'view:High')
    for read in readers:
        assert read() == [project.rows[i].id for i in (3, 2)]
    assert owner.row_ids() == current
    choose(owner.analysis_scope_combo, 'selected')
    assert all(read() == [] for read in readers)
    owner.table.selectionModel().select(owner.proxy.index(1, 2), QItemSelectionModel.SelectionFlag.ClearAndSelect)
    assert all(read() == [project.rows[1].id] for read in readers)
    owner.table.selectionModel().clearSelection()
    assert all(read() == [] for read in readers)
    choose(owner.analysis_scope_combo, 'view:Empty')
    assert all(read() == [] for read in readers)
    choose(owner.analysis_scope_combo, 'view:High')
    token = source_token(owner)
    project.view['named']['High']['rules'][0]['value'] = 3
    owner.model_changed()
    assert source_token(owner) != token
    assert all(read() == [project.rows[3].id] for read in readers)
    project.view['named']['High']['rules'][0]['column'] = 'deleted-column'
    owner.model_changed()
    for read in readers:
        with pytest.raises(ValueError):
            read()
    stop_timers(owner)


def test_live_selection_scope_keeps_proxy_order_without_reading_unselected_rows(owner, monkeypatch):
    project = owner.model.project
    owner.apply_filter([{'column': project.columns[0].id, 'operator': 'eq', 'value': 'A'}], 'AND')
    owner.proxy.sort(1, Qt.SortOrder.DescendingOrder)
    owner.show_plots()
    owner.show_maps()
    choose(owner.analysis_scope_combo, 'selected')
    selection = owner.table.selectionModel()
    for row, column in ((2, 0), (0, 1), (0, 2)):
        selection.select(owner.proxy.index(row, column), QItemSelectionModel.SelectionFlag.Select)
    expected = [project.rows[index].id for index in (2, 0)]

    def read_all_rows():
        pytest.fail('Resolving the live selection should not read the entire table')

    with monkeypatch.context() as patch:
        patch.setattr(owner, 'row_ids', read_all_rows)
        assert owner.analysis_scope.rows() == expected
        assert owner.plot_window.scope_ids() == expected
        assert owner.map_window.scope_ids() == expected
    stop_timers(owner)


def test_interpolation_and_calculations_use_views_and_invalidate_previews(owner):
    owner.interpolate()
    interpolation = owner.interpolation_window
    choose(interpolation.x, owner.model.project.columns[1].id)
    choose(interpolation.y, owner.model.project.columns[2].id)
    interpolation.groups.item(0).setCheckState(Qt.CheckState.Checked)
    choose(interpolation.scope, 'view:Gap')
    interpolation.calculate()
    assert len(interpolation.proposals) == 1
    assert interpolation.proposals[0].row_id == owner.model.project.rows[1].id
    assert interpolation.row_ids == {row.id for row in owner.model.project.rows[:3]}
    owner.show_calculations()
    calculation = owner.calculation_window
    choose(calculation.scope, 'view:High')
    assert not interpolation.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    calculation.context.setText('Test series')
    calculation.confirmed.setChecked(True)
    calculation.calculate()
    assert calculation.prepared, calculation.summary.text()
    ids = {owner.model.project.rows[i].id for i in (2, 3)}
    for column, values in calculation.prepared:
        assert set(column.metadata['calculation']['scope_row_ids']) == ids
    choose(owner.analysis_scope_combo, 'view:Empty')
    assert not calculation.prepared
    assert not calculation.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()


def test_scope_is_saved_in_project_and_new_project_starts_filtered(owner, qtbot, tmp_path):
    choose(owner.analysis_scope_combo, 'view:High')
    owner.path = tmp_path / 'shared-scope.apyro'
    assert owner.save()
    loaded = Project.load(owner.path)
    assert loaded.view['analysis_scope'] == 'view:High'
    owner.set_project(loaded, owner.path)
    assert owner.analysis_scope_combo.currentData() == 'view:High'
    owner.show_maps()
    assert owner.map_window.data.fields['scope'][0].currentData() == 'view:High'
    stop_timers(owner)
    owner.set_project(Project())
    assert owner.analysis_scope.mode == 'filtered'
    assert owner.analysis_scope_combo.findData('view:High') == -1


def test_report_uses_saved_view_even_when_table_has_different_filter(owner, qtbot):
    x = owner.model.project.columns[1].id
    owner.apply_filter([{'column': x, 'operator': 'lt', 'value': 1}], 'AND')
    dialog = ReportDialog(owner)
    qtbot.addWidget(dialog)
    choose(dialog.scope, 'view:High')
    dialog.build()
    assert dialog.report is not None, dialog.status.text()
    assert set(dialog.report['row_ids']) == {owner.model.project.rows[i].id for i in (2, 3)}
    choose(owner.analysis_scope_combo, 'view:Empty')
    assert dialog.report is None
    dialog.build()
    assert dialog.report['row_ids'] == []
    dialog.close()


def test_saved_view_keeps_sample_set_boundary_in_table_and_analysis(owner, monkeypatch, tmp_path):
    project = owner.model.project
    ids = [row.id for row in project.rows]
    project.view['sample_sets'] = {'Review': [ids[0], ids[2], ids[3]]}
    monkeypatch.setattr(owner, 'choose_sample_set', lambda title: 'Review')
    owner.open_sample_set()
    x = project.columns[1].id
    owner.apply_filter([{'column': x, 'operator': 'le', 'value': 2}], 'AND')
    owner.proxy.sort(1, Qt.SortOrder.DescendingOrder)
    expected = [ids[2], ids[0]]

    properties = owner.desktop.table_properties
    properties.view_name.setText('From table')
    properties.save_view()
    owner.analysis_scope.create_view('From analysis')
    owner.close_sample_set()
    assert ids[1] in owner.row_ids()
    for name in ('From table', 'From analysis'):
        assert owner.analysis_scope.rows(f'view:{name}') == expected

    owner.analysis_scope.rename_sample_set('Review', 'Renamed')
    with pytest.raises(ValueError, match='From table.*From analysis'):
        owner.analysis_scope.delete_sample_set('Renamed')
    assert owner.analysis_scope.rows('view:From analysis') == expected
    path = tmp_path / 'view-from-set.apyro'
    project.save(path)
    owner.set_project(Project.load(path), path)
    assert owner.analysis_scope.rows('view:From table') == expected
    properties.saved_views.setCurrentText('From table')
    properties.load_view()
    assert owner.active_sample_set == 'Renamed'
    assert owner.row_ids() == expected

    owner.close_sample_set()
    owner.apply_filter([{'column': x, 'operator': 'eq', 'value': 3}], 'AND')
    owner.show_statistics()
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.Yes)
    owner.statistics_window.go_to_sample(ids[2], 'view:From table')
    assert owner.active_sample_set == 'Renamed'
    assert owner.row_ids() == expected
    owner.statistics_window.restore_sample_filter()
    assert owner.row_ids() == [ids[3]]

    # A broken reference must also fail safely when a file was changed externally.
    del owner.model.project.view['sample_sets']['Renamed']
    properties.load_view()
    assert owner.row_ids() == []
    with pytest.raises(ValueError):
        owner.analysis_scope.rows('view:From table')


def test_plot_and_report_use_saved_sample_set_independently_of_table(owner, qtbot):
    project = owner.model.project
    expected = [project.rows[i].id for i in (3, 0)]
    project.view["sample_sets"] = {"Review": expected}
    owner.apply_filter([{"column": project.columns[1].id, "operator": "lt", "value": 0}], "AND")
    dialog = ReportDialog(owner)
    qtbot.addWidget(dialog)
    choose(dialog.scope, "set:Review")
    dialog.build()
    assert dialog.report is not None, dialog.status.text()
    assert dialog.report["row_ids"] == expected
    assert dialog.report["selection_context"]["sample_set"] == "Review"
    owner.show_plots()
    assert owner.plot_window.scope.currentData() == "global"
    assert set(owner.plot_window.scope_ids()) == set(expected)
    stop_timers(owner)
    dialog.close()


def test_interpretation_statistics_and_map_calculate_the_same_view(owner, qtbot):
    from autopyro.standard_interpretation import standard_schemes

    owner.show_interpretation()
    interpretation = owner.interpretation_window
    interpretation.commit_graphical_scheme(standard_schemes()['S2–TOC'], auto_bind=True)
    choose(interpretation.scope, 'view:High')
    interpretation.calculate()
    expected = {owner.model.project.rows[i].id for i in (2, 3)}
    assert {row['row_id'] for row in interpretation.report['results']} == expected
    owner.show_statistics()
    statistics = owner.statistics_window
    qtbot.waitUntil(lambda: statistics.report is not None and not statistics.timer.isActive(), timeout=15000)
    assert set(statistics.report['row_ids']) == expected
    owner.show_maps()
    dialog = owner.map_window
    dialog.data.load({'x_id': owner.model.project.columns[1].id,
                     'y_id': owner.model.project.columns[3].id,
                     'source_crs': 'EPSG:32631', 'target_crs': 'EPSG:32631'})
    dialog.redraw()
    assert {row['id'] for row in dialog.report['points']} == expected
    choose(owner.analysis_scope_combo, 'view:Empty')
    assert interpretation.report is None
    assert statistics.report is None
    interpretation.calculate()
    assert interpretation.report['results'] == []
    stop_timers(owner)


def test_interpretation_checks_membership_of_the_same_filtered_scope(owner):
    from autopyro.standard_interpretation import standard_schemes

    project = owner.model.project
    axis = project.columns[1].id
    owner.apply_filter([{'column': axis, 'operator': 'lt', 'value': 2}], 'AND')
    owner.show_interpretation()
    dialog = owner.interpretation_window
    dialog.commit_graphical_scheme(standard_schemes()['S2–TOC'], auto_bind=True)
    choose(dialog.scope, 'filtered')
    dialog.calculate()
    saved = project.view['interpretation_result']
    dialog.pin_base.setChecked(True)
    pinned = project.view['interpretation_comparison_base']

    owner.proxy.sort(1, Qt.SortOrder.DescendingOrder)
    dialog.start_calculation()
    dialog.cancel_calculation()
    assert not saved['stale']

    owner.apply_filter([{'column': axis, 'operator': 'ge', 'value': 2}], 'AND')
    dialog.start_calculation()
    dialog.cancel_calculation()
    assert saved['stale_reason'] == {'code': 'binding_changed', 'fields': ['scope']}
    assert project.view['interpretation_result'] is saved
    assert not pinned['stale']

    dialog.calculate()
    current = project.view['interpretation_result']
    assert not current['stale']
    assert {item['row_id'] for item in current['results']} == {row.id for row in project.rows[2:]}
    assert project.view['interpretation_comparison_base'] is pinned


def test_geometry_preview_can_expand_scope_beyond_original_report(owner, qtbot):
    from copy import deepcopy
    from autopyro.standard_interpretation import standard_schemes

    owner.show_interpretation()
    interpretation = owner.interpretation_window
    interpretation.commit_graphical_scheme(standard_schemes()['S2–TOC'], auto_bind=True)
    choose(interpretation.scope, 'view:High')
    interpretation.calculate()
    editor = GeometryEditor(interpretation.scheme, interpretation, samples=interpretation.geometry_samples())
    qtbot.addWidget(editor)
    draft = deepcopy(editor.draft)
    assert len(editor.samples) == 2
    assert editor.data_scope.findData('view:Gap') >= 0
    choose(editor.data_scope, 'all')
    assert interpretation.scope.currentData() == 'all'
    assert len(editor.samples) == 4
    choose(owner.analysis_scope_combo, 'view:Empty')
    assert editor.samples == []
    assert editor.draft == draft
    stop_timers(owner)
    editor.close()


def test_quality_results_follow_scope_without_changing_exclusion_target(owner):
    owner.model.setData(owner.model.index(0, 5), '-1')
    owner.model.setData(owner.model.index(3, 5), '-2')
    owner.show_qc()
    dialog = owner.qc_window
    target = dialog.scope.currentData()
    assert any(issue['sample_id'] == owner.model.project.rows[0].id for issue in dialog.issues)
    choose(dialog.data_scope, 'view:High')
    assert dialog.results_model.issues
    assert all(issue['sample_id'] in {owner.model.project.rows[i].id for i in (2, 3)}
               for issue in dialog.results_model.issues)
    assert dialog.scope.currentData() == target
    choose(owner.analysis_scope_combo, 'view:Empty')
    assert dialog.results_model.rowCount() == 0


def test_scope_change_rejects_pending_spatial_interpolation(owner, qtbot, monkeypatch):
    from threading import Event
    from PySide6.QtCore import QTimer
    from autopyro.ui.dialogs import spatial_idw as module

    owner.show_maps()
    owner.map_window.timer.stop()
    project = owner.model.project
    snapshot = build_map(project, [row.id for row in project.rows], MapSpec(
        x_id=project.columns[1].id, y_id=project.columns[3].id,
        source_crs='EPSG:32631', target_crs='EPSG:32631'), [])
    dialog = SpatialInterpolationDialog(snapshot, owner.map_window)
    qtbot.addWidget(dialog)
    dialog.form.load({'value_id': project.columns[0].id})
    entered, release = Event(), Event()
    def interpolate(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return {'obsolete': True}
    monkeypatch.setattr(module, 'interpolate', interpolate)
    completed = []
    monkeypatch.setattr(dialog, 'completed', completed.append)
    timer = QTimer(dialog)
    def change_scope():
        if entered.is_set():
            timer.stop()
            try:
                choose(owner.analysis_scope_combo, 'view:High')
            finally:
                release.set()
    timer.timeout.connect(change_scope)
    timer.start(10)
    dialog.calculate()
    assert entered.is_set()
    qtbot.waitUntil(dialog.isEnabled)
    assert completed == []
    assert dialog.result is None and not dialog.export_button.isEnabled()
    stop_timers(owner)
    dialog.close()


def test_manage_views_actions_capture_table_rename_and_delete(owner, qtbot, monkeypatch):
    from copy import deepcopy
    from PySide6.QtWidgets import QInputDialog, QMessageBox

    owner.show_maps()
    stop_timers(owner)
    combo = owner.map_window.data.fields['scope'][0]
    create, edit, rename, delete = owner.analysis_views_button.menu().actions()
    assert create.isEnabled() and not rename.isEnabled() and not delete.isEnabled()
    assert len(combo.actions()) == 4
    x = owner.model.project.columns[1].id
    owner.apply_filter([{'column': x, 'operator': 'le', 'value': 1}], 'AND')
    expected = deepcopy(owner.capture_view())
    monkeypatch.setattr(QInputDialog, 'getText', lambda *a, **kw: ('  Shared  ', True))
    create.trigger()
    assert owner.model.project.view['named']['Shared'] == expected
    assert combo.currentData() == 'view:Shared'
    assert rename.isEnabled() and delete.isEnabled()
    assert edit.isEnabled() and combo.actions()[2].isEnabled()
    assert owner.desktop.table_properties.saved_views.currentText() == 'Shared'
    owner.apply_filter([{'column': x, 'operator': 'ge', 'value': 2}], 'AND')
    assert owner.analysis_scope.rows() == [row.id for row in owner.model.project.rows[:2]]
    calls = []
    def renamed(*args, **kwargs):
        calls.append(kwargs['text'])
        return 'Renamed', True
    monkeypatch.setattr(QInputDialog, 'getText', renamed)
    combo.actions()[2].trigger()
    assert calls == ['Shared']
    assert owner.model.project.view['named']['Renamed'] == expected
    assert 'Shared' not in owner.model.project.view['named']
    assert owner.analysis_scope_combo.currentData() == 'view:Renamed'
    assert owner.desktop.table_properties.saved_views.currentText() == 'Renamed'
    owner.dirty = False
    monkeypatch.setattr(QMessageBox, 'question', lambda *a: QMessageBox.StandardButton.No)
    delete.trigger()
    assert 'Renamed' in owner.model.project.view['named'] and not owner.dirty
    monkeypatch.setattr(QMessageBox, 'question', lambda *a: QMessageBox.StandardButton.Yes)
    delete.trigger()
    assert 'Renamed' not in owner.model.project.view['named']
    assert combo.currentData() == 'filtered'
    assert owner.analysis_scope.rows() == owner.row_ids()
    assert not rename.isEnabled() and not delete.isEnabled()
    assert owner.dirty
    stop_timers(owner)


def test_view_names_are_validated_without_overwriting(owner, monkeypatch):
    from copy import deepcopy
    from PySide6.QtWidgets import QInputDialog

    scope = owner.analysis_scope
    before = deepcopy(owner.model.project.view)
    owner.dirty = False
    for value in ('', '   ', ' High '):
        with pytest.raises(ValueError):
            scope.create_view(value)
    with pytest.raises(ValueError):
        scope.rename_view('High', 'Gap')
    scope.rename_view('High', ' High ')
    monkeypatch.setattr(QInputDialog, 'getText', lambda *a, **kw: ('Cancelled', False))
    owner.analysis_views_button.menu().actions()[0].trigger()
    assert owner.model.project.view == before and not owner.dirty


def test_rename_keeps_plot_undo_attached_to_the_same_view(owner):
    choose(owner.analysis_scope_combo, 'view:High')
    owner.show_plots()
    plot = owner.plot_window
    choose(plot.scope, 'view:High')
    plot.timer.stop()
    plot.redraw()
    plot.style.load({'marker_size': 12})
    plot.redraw()
    assert plot.undo_stack.count() > 0
    owner.analysis_scope.rename_view('High', 'Renamed')
    plot.undo_stack.undo()
    assert owner.analysis_scope.mode == 'view:Renamed'
    assert plot.scope.currentData() == 'view:Renamed'
    plot.undo_stack.redo()
    assert owner.analysis_scope.mode == 'view:Renamed'
    stop_timers(owner)


def test_renaming_and_deletion_retarget_saved_ui_scopes_and_survive_storage(owner, tmp_path):
    from copy import deepcopy

    scope = owner.analysis_scope
    view = owner.model.project.view
    for key in ('plot_current', 'interpretation_plot_current', 'statistics', 'interpretation_current'):
        view[key] = {'scope': 'view:High'}
    view['plots'] = {'Saved graph': {'scope': 'view:High'}, 'Other graph': {'scope': 'all'}}
    view['interpretation_result'] = {'scope': 'view:High', 'row_ids': ['archived']}
    archive = deepcopy(view['interpretation_result'])
    original = deepcopy(view['named']['High'])
    choose(owner.analysis_scope_combo, 'view:High')
    rows = scope.rows()
    scope.rename_view('High', 'Новые данные: High')
    assert scope.rows() == rows
    assert list(view['named'])[0] == 'Новые данные: High'
    assert view['named']['Новые данные: High'] == original
    for key in ('plot_current', 'interpretation_plot_current', 'statistics', 'interpretation_current'):
        assert view[key]['scope'] == 'view:Новые данные: High'
    assert view['plots']['Saved graph']['scope'] == 'view:Новые данные: High'
    assert view['plots']['Other graph']['scope'] == 'all'
    assert view['interpretation_result'] == archive
    owner.path = tmp_path / 'renamed-view.apyro'
    assert owner.save()
    loaded = Project.load(owner.path)
    owner.set_project(loaded, owner.path)
    restored_archive = deepcopy(loaded.view['interpretation_result'])
    assert scope.mode == 'view:Новые данные: High'
    assert owner.analysis_views_button.menu().actions()[1].isEnabled()
    scope.delete_view('Новые данные: High')
    assert scope.mode == 'filtered'
    assert loaded.view['plots']['Saved graph']['scope'] == 'filtered'
    assert loaded.view['interpretation_result'] == restored_archive
    assert owner.save()
    assert 'Новые данные: High' not in Project.load(owner.path).view['named']


def test_saved_view_editor_changes_filters_and_sorting_without_changing_table(owner, tmp_path):
    from copy import deepcopy

    x = owner.model.project.columns[1].id
    owner.apply_filter([{'column': x, 'operator': 'le', 'value': 1}], 'AND')
    source = deepcopy(owner.capture_view())
    owner.show_plots()
    owner.show_maps()
    stop_timers(owner)
    choose(owner.analysis_scope_combo, 'view:High')
    desktop = owner.desktop
    document = desktop.documents.currentWidget()
    owner.analysis_views_button.menu().actions()[1].trigger()
    panel = desktop.view_properties
    assert desktop.properties_stack.currentWidget() is panel
    assert desktop.documents.currentWidget() is document
    assert panel.name == 'High'
    assert {key for key, page in panel.pages.items() if panel.isTabVisible(panel.indexOf(page))} == {'filter', 'sort'}
    assert panel.rules[0].value.text() == '2'
    panel.rules[0].value.setText('1')
    panel.apply_filter_button.click()
    assert owner.model.project.view['named']['High']['rules'][0]['value'] == 1
    assert owner.capture_view() == source
    expected = [owner.model.project.rows[i].id for i in (3, 2, 1)]
    assert owner.plot_window.scope_ids() == expected
    assert owner.map_window.scope_ids() == expected
    panel.select_page('sort')
    item = next(panel.sort_columns.item(i) for i in range(panel.sort_columns.count())
                if panel.sort_columns.item(i).data(Qt.ItemDataRole.UserRole) == x)
    panel.reverse_sort(item)
    panel.apply_sort_button.click()
    assert owner.analysis_scope.rows() == list(reversed(expected))
    assert owner.capture_view() == source
    panel.done_button.click()
    assert desktop.properties_stack.currentWidget() is owner.map_window.tabs
    assert desktop.documents.currentWidget() is document
    stop_timers(owner)
    desktop.close_documents()
    owner.path = tmp_path / 'edited-view.apyro'
    assert owner.save()
    loaded = Project.load(owner.path)
    owner.set_project(loaded, owner.path)
    assert owner.analysis_scope.mode == 'view:High'
    assert owner.analysis_scope.rows() == list(reversed(expected))
    assert owner.capture_view()['rules'] == source['rules']


def test_saved_view_editor_handles_validation_drafts_rename_and_scope_switch(owner):
    from copy import deepcopy

    choose(owner.analysis_scope_combo, 'view:High')
    owner.analysis_scope.edit_view()
    panel = owner.desktop.view_properties
    original = deepcopy(owner.model.project.view['named']['High'])
    panel.rules[0].value.setText('invalid-number')
    panel.apply_filter()
    assert panel.filter_error.text()
    assert owner.model.project.view['named']['High'] == original
    panel.rules[0].value.setText('3')
    owner.desktop.refresh_project()
    assert panel.rules[0].value.text() == '3'
    owner.analysis_scope.rename_view('High', 'Renamed')
    assert panel.name == 'Renamed'
    assert panel.rules[0].value.text() == '3'
    panel.apply_filter()
    assert owner.analysis_scope.rows() == [owner.model.project.rows[3].id]
    choose(owner.analysis_scope_combo, 'view:Gap')
    assert panel.name == 'Gap'
    assert panel.rules[0].value.text() == '2'
    panel.reset_filter()
    panel.reset_sort()
    assert len(owner.analysis_scope.rows()) == 4
    owner.analysis_scope.delete_view('Gap')
    assert owner.analysis_scope.mode == 'filtered'
    assert owner.desktop.properties_stack.currentWidget() is not panel
    owner.set_project(Project())
    assert panel.name is None
