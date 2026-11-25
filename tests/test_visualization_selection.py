"""Selection is presentation state; data scopes and exports retain their contracts."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PySide6.QtCore import QItemSelection, QItemSelectionModel, Qt
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from autopyro.io import build_project
from autopyro.plotting import PlotSpec, render_plot
from autopyro.selection_overlay import SelectionOverlay
from autopyro.ui.table.selection import selected_row_ids, select_row_ids
from autopyro.ui.dialogs.map import MapDialog
from autopyro.ui.window import MainWindow


@pytest.fixture
def window(qtbot):
    owner = MainWindow()
    owner.may_replace = lambda: True
    qtbot.addWidget(owner)
    owner.set_project(build_project(['X', 'Y', 'Bottom'], [[1, 1, 2], [2, 3, 4], [3, 5, 6]], [2, 3, 4],
        [(0, 'X', 'number', ''), (1, 'Y', 'number', 'm'), (2, 'Bottom', 'number', 'm')], {}))
    yield owner
    for name in ('plot_window', 'map_window'):
        if getattr(owner, name):
            getattr(owner, name).close()
    owner.dirty = False


def fail(*args, **kwargs):
    raise AssertionError('Selection must not rebuild data or redraw the base canvas')


def ready_plot(owner):
    owner.show_plots()
    dialog = owner.plot_window
    dialog.timer.stop()
    dialog.redraw()
    return dialog


def test_partial_cells_are_rows_without_expanding_selected_indexes(window, monkeypatch):
    selection = window.table.selectionModel()
    selection.select(window.proxy.index(1, 2), QItemSelectionModel.SelectionFlag.Select)
    monkeypatch.setattr(selection, 'selectedIndexes', fail)
    assert selected_row_ids(window) == {window.model.project.rows[1].id}


@pytest.mark.parametrize('ranges,positions', [
    ([], []), ([(0, 0, 0), (2, 2, 2)], [0, 2]),
    ([(0, 1, 0), (1, 2, 2)], [0, 1, 2]),
    ([(1, 2, 0), (1, 2, 2)], [1, 2]),
])
def test_selection_row_union_preserves_gaps_and_proxy_order(window, ranges, positions):
    window.proxy.sort(0, Qt.SortOrder.DescendingOrder)
    selection = QItemSelection()
    for top, bottom, column in ranges:
        selection.select(window.proxy.index(top, column), window.proxy.index(bottom, column))
    window.table.selectionModel().select(selection, QItemSelectionModel.SelectionFlag.ClearAndSelect)
    expected = [window.model.project.rows[
        window.proxy.mapToSource(window.proxy.index(row, 0)).row()].id for row in positions]
    assert selected_row_ids(window, ordered=True) == expected
    assert selected_row_ids(window) == set(expected)


@pytest.mark.parametrize("origin", ["table", "canvas"])
def test_selection_export_count_matches_unique_rows_through_filters(window, origin):
    rows = window.model.project.rows
    expected = {rows[0].id, rows[2].id}
    if origin == "canvas":
        window.sample_selection.select(expected)
    else:
        select_row_ids(window, expected)
    column = window.model.project.columns[0].id
    for minimum in (0, 2, 4, 0):
        window.apply_filter([{"column": column, "operator": "ge", "value": minimum}], "AND")
        count, snapshot = window.sample_selection.export_snapshot()
        exported = [row.id for row in snapshot]
        assert count == len(exported) == len(set(exported)) == 2
        assert set(exported) == expected

    window.table.selectionModel().clearSelection()
    count, snapshot = window.sample_selection.export_snapshot()
    assert count == 0
    assert list(snapshot) == []


def test_undo_added_column_keeps_hidden_source_cell_selected(window):
    from autopyro.model.core import Column
    from autopyro.ui.table import AddColumnCommand

    project = window.model.project
    hidden, visible = project.rows[:2]
    original = project.columns[0]
    extra = Column("New column")
    window.model.undo.push(AddColumnCommand(window.model, extra, {}))
    selection = window.table.selectionModel()
    selection.select(window.proxy.index(0, 0), QItemSelectionModel.SelectionFlag.ClearAndSelect)
    selection.select(window.proxy.index(1, 3), QItemSelectionModel.SelectionFlag.Select)
    window.apply_filter([{"column": original.id, "operator": "gt", "value": 1}], "AND")
    assert window.sample_selection.ids() == {hidden.id, visible.id}

    window.model.undo.undo()
    assert window.sample_selection.ids() == {hidden.id}
    window.apply_filter([], "AND")
    index, = selection.selectedIndexes()
    source = window.proxy.mapToSource(index)
    assert (project.rows[source.row()].id, project.columns[source.column()].id) == (hidden.id, original.id)


def test_batch_selection_emits_once_and_respects_sort_and_filter(window):
    project = window.model.project
    select_row_ids(window, [project.rows[1].id])
    window.table.sortByColumn(0, Qt.SortOrder.DescendingOrder)
    selection = window.table.selectionModel()
    spy = QSignalSpy(selection.selectionChanged)
    wanted = {project.rows[0].id, project.rows[2].id}
    select_row_ids(window, wanted)
    assert spy.count() == 1
    assert selected_row_ids(window) == wanted
    window.apply_filter([{'column': project.columns[0].id, 'operator': 'gt', 'value': 1}], 'AND')
    select_row_ids(window, wanted)
    assert selected_row_ids(window) == {project.rows[2].id}
    select_row_ids(window, [])
    assert selected_row_ids(window) == set()


@pytest.mark.parametrize('target', ['hidden', 'missing', 'empty'])
def test_single_selection_clears_when_target_is_unavailable(window, target):
    project = window.model.project
    window.apply_filter([{'column': project.columns[0].id, 'operator': 'gt', 'value': 1}], 'AND')
    select_row_ids(window, [project.rows[1].id])
    wanted = {'hidden': [project.rows[0].id], 'missing': ['absent-id'], 'empty': []}[target]
    spy = QSignalSpy(window.table.selectionModel().selectionChanged)
    select_row_ids(window, wanted)
    assert selected_row_ids(window) == set()
    assert spy.count() == 1


def test_table_selection_survives_filters_without_expanding_to_whole_rows(window):
    hidden, visible, other = window.model.project.rows
    window.proxy.sort(0, Qt.SortOrder.DescendingOrder)
    selection = window.table.selectionModel()
    for row, column in ((2, 0), (2, 2), (1, 1)):
        selection.select(window.proxy.index(row, column), QItemSelectionModel.SelectionFlag.Select)

    def cells():
        return {(window.row_ids()[index.row()], index.column()) for index in selection.selectedIndexes()}

    window.apply_filter([{'column': window.model.project.columns[0].id, 'operator': 'gt', 'value': 2}], 'AND')
    assert cells() == set()
    assert window.analysis_scope.rows('selected') == [hidden.id, visible.id]
    window.apply_filter([{'column': window.model.project.columns[0].id, 'operator': 'gt', 'value': 1}], 'AND')
    assert cells() == {(visible.id, 1)}
    window.apply_filter([], 'AND')
    assert cells() == {(hidden.id, 0), (hidden.id, 2), (visible.id, 1)}
    assert window.analysis_scope.rows('selected') == [visible.id, hidden.id]

    window.apply_filter([{'column': window.model.project.columns[0].id, 'operator': 'gt', 'value': 1}], 'AND')
    selection.select(window.proxy.index(0, 2), QItemSelectionModel.SelectionFlag.ClearAndSelect)
    window.apply_filter([], 'AND')
    assert cells() == {(other.id, 2)}
    assert window.sample_selection.ids() == {other.id}


def test_edit_and_undo_keep_cells_selected_when_active_filter_hides_row(window):
    project = window.model.project
    target, other, last = project.rows
    rules = [{'column': project.columns[0].id, 'operator': 'ge', 'value': 1}]
    window.apply_filter(rules, 'AND')
    window.proxy.sort(0, Qt.SortOrder.DescendingOrder)
    selection = window.table.selectionModel()
    for row, column in ((2, 0), (2, 2), (1, 1)):
        selection.select(window.proxy.index(row, column), QItemSelectionModel.SelectionFlag.Select)

    def cells():
        return {(window.row_ids()[index.row()], index.column()) for index in selection.selectedIndexes()}

    assert window.proxy.setData(window.proxy.index(2, 0), '0')
    assert window.row_ids() == [last.id, other.id]
    assert cells() == {(other.id, 1)}
    assert window.analysis_scope.rows('selected') == [other.id, target.id]
    assert window.proxy.rules == rules

    window.model.undo.undo()
    assert cells() == {(target.id, 0), (target.id, 2), (other.id, 1)}
    window.model.undo.redo()
    assert cells() == {(other.id, 1)}
    assert window.sample_selection.ids() == {target.id, other.id}


def test_model_reset_restores_hidden_cells_by_row_and_column_identity(window):
    project = window.model.project
    hidden, removed, last = project.rows
    x, y, bottom = project.columns
    selection = window.table.selectionModel()
    selection.select(QItemSelection(window.proxy.index(0, 0), window.proxy.index(0, 1)),
                     QItemSelectionModel.SelectionFlag.ClearAndSelect)
    selection.select(window.proxy.index(1, 2), QItemSelectionModel.SelectionFlag.Select)
    window.apply_filter([{'column': x.id, 'operator': 'gt', 'value': 1}], 'AND')
    assert window.sample_selection.ids() == {hidden.id, removed.id}

    window.model.beginResetModel()
    project.rows = [last, hidden]
    project.columns = [y, bottom, x]
    window.model.endResetModel()
    window.model.structureChanged.emit()

    assert not selection.hasSelection()
    assert window.sample_selection.ids() == {hidden.id}
    window.apply_filter([], 'AND')

    def cells():
        return {(project.rows[source.row()].id, project.columns[source.column()].id)
                for index in selection.selectedIndexes()
                if (source := window.proxy.mapToSource(index)).isValid()}

    assert cells() == {(hidden.id, x.id), (hidden.id, y.id)}
    window.model.beginResetModel()
    project.columns = [bottom, y]
    window.model.endResetModel()
    window.model.structureChanged.emit()
    assert cells() == {(hidden.id, y.id)}


def test_map_selected_scope_reads_selection_once():
    from autopyro.ui import analysis_scope
    calls = []
    def selected(*, ordered=False):
        assert ordered
        calls.append(1)
        return ['1', '9999']
    owner = SimpleNamespace(row_ids=lambda: list(map(str, range(10000))),
                            model=SimpleNamespace(project=None),
                            sample_selection=SimpleNamespace(ids=selected))
    scope = SimpleNamespace(owner=owner, mode='selected')
    owner.analysis_scope = SimpleNamespace(rows=lambda **kwargs: analysis_scope.AnalysisScope.rows(scope, **kwargs))
    dialog = SimpleNamespace(owner=owner)
    assert MapDialog.scope_ids(dialog) == ['1', '9999']
    assert len(calls) == 1
    assert MapDialog.scope_ids(dialog, selected_ids={'2'}) == ['2']
    assert len(calls) == 1


@pytest.mark.parametrize('canvas', ['plot', 'map'])
def test_canvas_selection_keeps_hidden_samples_until_table_selection_changes(window, canvas, monkeypatch, tmp_path):
    project = window.model.project
    hidden, visible, replacement = (row.id for row in project.rows)
    window.apply_filter([{'column': project.columns[0].id, 'operator': 'gt', 'value': 1}], 'AND')
    rules = window.proxy.rules
    if canvas == 'plot':
        window.show_plots()
        dialog = window.plot_window
    else:
        window.show_maps()
        dialog = window.map_window
    spy = QSignalSpy(window.sample_selection.changed)

    dialog.select_rows([hidden])
    assert spy.count() == 1
    assert selected_row_ids(window) == set()
    assert window.analysis_scope.rows('selected') == [hidden]
    assert window.proxy.rules == rules
    monkeypatch.setattr(QMessageBox, 'information', fail)
    monkeypatch.setattr(QInputDialog, 'getText', lambda *args: ('From canvas', True))
    window.save_sample_set()
    assert project.view['sample_sets']['From canvas'] == [hidden]
    path = tmp_path / 'selection.json'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), 'JSON (*.json)'))
    window.export_selected_samples()
    assert [row['id'] for row in json.loads(path.read_text(encoding='utf-8'))['rows']] == [hidden]

    dialog.select_rows([hidden, visible])
    assert selected_row_ids(window) == {visible}
    assert window.analysis_scope.rows('selected') == [visible, hidden]
    window.apply_filter([], 'AND')
    assert selected_row_ids(window) == {hidden, visible}
    assert window.analysis_scope.rows('selected') == [hidden, visible]
    window.apply_filter(rules, 'AND')
    assert selected_row_ids(window) == {visible}
    assert window.sample_selection.ids() == {hidden, visible}

    window.table.selectRow(1)
    assert window.sample_selection.ids() == {replacement}
    window.apply_filter([], 'AND')
    assert selected_row_ids(window) == {replacement}


def test_plot_selection_blits_without_render_or_dirty_state(window, monkeypatch):
    from autopyro.ui.dialogs import plot_rendering as module
    dialog = ready_plot(window)
    result = dialog.result
    limits = result['axes'][0].get_xlim(), result['axes'][0].get_ylim()
    window.dirty = False
    monkeypatch.setattr(module, 'render_plot', fail)
    monkeypatch.setattr(dialog.canvas, 'draw', fail)
    rid = window.model.project.rows[1].id
    select_row_ids(window, [rid])
    assert dialog.selection_overlay.selected_ids == {rid}
    assert dialog.result is result and not dialog.timer.isActive()
    assert not window.dirty and dialog.export_button.isEnabled()
    assert (result['axes'][0].get_xlim(), result['axes'][0].get_ylim()) == limits
    with monkeypatch.context() as unchanged:
        unchanged.setattr(dialog.canvas, 'blit', fail)
        dialog.selection_overlay.set_selected([rid])
    select_row_ids(window, [])
    assert not dialog.selection_overlay.selected_ids


def test_selected_scope_still_rebuilds_and_empty_means_empty(window):
    dialog = ready_plot(window)
    dialog.data.load({'scope': 'selected'})
    dialog.timer.stop()
    dialog.redraw()
    assert dialog.rendered_ids == []
    rid = window.model.project.rows[0].id
    select_row_ids(window, [rid])
    assert dialog.result is None and not dialog.export_button.isEnabled()
    dialog.timer.stop()
    dialog.redraw()
    assert dialog.rendered_ids == [rid]
    select_row_ids(window, [])
    dialog.timer.stop()
    dialog.redraw()
    assert dialog.rendered_ids == []


def test_map_selection_keeps_snapshot_and_does_not_reaggregate(window, monkeypatch):
    from autopyro.ui.dialogs import map_rendering as module
    window.show_maps()
    dialog = window.map_window
    x, y = [c.id for c in window.model.project.columns[:2]]
    dialog.data.load({'x_id': x, 'y_id': y, 'source_crs': 'EPSG:32631', 'target_crs': 'EPSG:32631'})
    dialog.redraw()
    assert dialog.result is not None, dialog.status.text()
    report = dialog.report
    monkeypatch.setattr(module, 'build_map', fail)
    monkeypatch.setattr(dialog.canvas, 'draw', fail)
    rid = window.model.project.rows[2].id
    select_row_ids(window, [rid])
    assert dialog.selection_overlay.selected_ids == {rid}
    assert dialog.report is report and dialog.export_button.isEnabled()
    assert not dialog.timer.isActive()
    monkeypatch.undo()
    dialog.data.load({'scope': 'selected'})
    dialog.redraw()
    assert len(dialog.report['points']) == 1
    other = window.model.project.rows[0].id
    select_row_ids(window, [other])
    assert dialog.report is None and not dialog.export_button.isEnabled()
    dialog.redraw()
    assert dialog.selection_overlay.selected_ids == {other}


@pytest.mark.parametrize('kind', ['scatter', 'depth', 'multi_depth', 'mpl:scatter', 'mpl3d:scatter'])
@pytest.mark.parametrize('compact', [False, True])
def test_overlay_updates_and_survives_axis_redraw(window, kind, compact):
    project = window.model.project
    x, y, bottom = [c.id for c in project.columns]
    spec = PlotSpec(kind=kind, x_id=x, y_id=y, z_id=bottom, track_ids=[x, bottom])
    figure, result = render_plot(project, [r.id for r in project.rows], spec)
    canvas = FigureCanvasAgg(figure)
    overlay = SelectionOverlay(canvas, result, compact=compact)
    canvas.draw()
    original = np.array(canvas.buffer_rgba())
    rid = project.rows[1].id
    overlay.set_selected([rid])
    assert overlay.selected_ids == {rid}
    assert sum(len(layer.artist.get_sizes()) for layer in overlay.layers) >= 1
    if kind != 'mpl3d:scatter':
        assert not np.array_equal(original, np.array(canvas.buffer_rgba()))
        overlay.set_selected([])
        np.testing.assert_array_equal(original, np.array(canvas.buffer_rgba()))
        overlay.set_selected([rid])
    figure.axes[0].set_xlim(0, 10)
    assert overlay.background is None
    canvas.draw()
    assert overlay.selected_ids == {rid}
    overlay.set_selected([])
    assert not overlay.selected_ids
    if kind != 'mpl3d:scatter':
        figure.axes[0].set_xlim(result['artists'][0].get_offsets()[:, 0].min() - .1,
                                result['artists'][0].get_offsets()[:, 0].max() + .1)
        canvas.draw()
    overlay.dispose()
    assert not overlay.locations


def test_interval_overlay_shows_full_range(window):
    project = window.model.project
    x, top, bottom = [c.id for c in project.columns]
    figure, result = render_plot(project, [r.id for r in project.rows],
        PlotSpec(kind='depth', x_id=x, y_id=top, depth_bottom_id=bottom))
    canvas = FigureCanvasAgg(figure)
    overlay = SelectionOverlay(canvas, result)
    canvas.draw()
    overlay.set_selected([project.rows[1].id])
    np.testing.assert_allclose(overlay.layers[0].artist.get_segments()[0], [[2, 3], [2, 4]])
    overlay.dispose()


def test_map_aggregate_members_share_one_highlight(window):
    project = window.model.project
    x, y = [c.id for c in project.columns[:2]]
    figure, result = render_plot(project, [r.id for r in project.rows], PlotSpec(x_id=x, y_id=y))
    canvas = FigureCanvasAgg(figure)
    members = [r.id for r in project.rows[:2]]
    overlay = SelectionOverlay(canvas, {'artist': result['artists'][0],
        'items': [{'x': 1, 'y': 1, 'row_ids': members}]})
    canvas.draw()
    overlay.set_selected(members)
    assert len(overlay.layers[0].artist.get_offsets()) == 1
    overlay.dispose()
