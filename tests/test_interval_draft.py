"""Local history and table/column linkage must not touch project undo or rerender logs."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QUndoCommand
from PySide6.QtWidgets import QTableWidgetSelectionRange

from autopyro.core import Project
from autopyro.ui.dialogs.stratigraphy import StratigraphyDialog
from autopyro.ui.widgets.lithology import LithologyPicker
from autopyro.ui.window import MainWindow


@pytest.fixture
def draft(qtbot):
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(Project())
    owner.may_replace = lambda: True
    dialog = StratigraphyDialog(owner)
    qtbot.addWidget(dialog)
    dialog.reference.setText('MD RKB')
    for i in range(3):
        dialog.add_interval(interval={
            'id': f'layer-{i}', 'wellbore': 'A', 'name': 'Same name', 'level': 'FORMATION',
            'top': i * 10, 'bottom': (i + 1) * 10, 'color': '#aaccff',
            'source_row': 42 + i, 'raw_cells': {'original': f'row {i}'}})
    dialog.draft_undo.clear()
    yield owner, dialog
    owner.dirty = False


def test_cell_edit_shortcuts_are_local_and_redoable(draft, qtbot):
    owner, dialog = draft
    dialog.show()
    dialog.intervals.setCurrentCell(0, 3)
    dialog.intervals.setFocus()
    qtbot.wait(20)
    dialog.intervals.item(0, 3).setText('2')
    assert dialog.draft_undo.count() == 1
    assert len(dialog.draft_undo.command(0).changes) == 1
    qtbot.keyClick(dialog.intervals, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert dialog.intervals.item(0, 3).text() == '0'
    qtbot.keyClick(dialog.intervals, Qt.Key.Key_Z,
                   Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
    assert dialog.intervals.item(0, 3).text() == '2'
    assert owner.model.undo.count() == 0
    assert not owner.model.project.view.get('stratigraphy')


def test_mass_lithology_is_one_undo_command(draft, monkeypatch):
    owner, dialog = draft
    before = dialog.read_scheme()
    dialog.intervals.selectAll()
    def choose(picker):
        picker.selected_code = '607'
        return picker.DialogCode.Accepted
    monkeypatch.setattr(LithologyPicker, 'exec', choose)
    dialog.choose_lithology()
    assert dialog.draft_undo.count() == 1
    assert len(dialog.draft_undo.command(0).changes) == 3
    assert all(item['lithology'] == '607' for item in dialog.read_scheme()['intervals'])
    dialog.draft_undo.undo()
    assert dialog.read_scheme() == before
    dialog.draft_undo.redo()
    assert all(item['lithology'] == '607' for item in dialog.read_scheme()['intervals'])


def test_insert_delete_and_metadata_preserve_identity_and_provenance(draft):
    owner, dialog = draft
    before = dialog.read_scheme()
    dialog.intervals.setCurrentCell(0, 1)
    dialog.remove_intervals()
    assert dialog.intervals.rowCount() == 2
    dialog.draft_undo.undo()
    assert dialog.read_scheme() == before
    dialog.draft_undo.redo()
    dialog.draft_undo.undo()
    dialog.add_interval(interval={'id': 'new', 'wellbore': 'A', 'name': 'New',
                                  'level': 'FORMATION', 'top': 30, 'bottom': 40, 'color': '#ffaa00'})
    dialog.draft_undo.undo()
    assert dialog.read_scheme() == before
    dialog.draft_undo.redo()
    assert dialog.read_scheme()['intervals'][-1]['id'] == 'new'
    old_name = dialog.name.text()
    dialog.name.setText('New name')
    dialog.name.setText('New name completed')
    dialog.draft_undo.undo()
    assert dialog.name.text() == old_name  # Consecutive typing is one metadata edit.
    dialog.draft_undo.redo()
    assert dialog.name.text() == 'New name completed'


def test_column_click_and_table_selection_link_without_rendering(draft, monkeypatch):
    from autopyro.ui.dialogs import stratigraphy_column as module
    owner, dialog = draft
    dialog.show_column()
    def unexpected_render(*args, **kwargs):
        pytest.fail('Selecting layers must not rerun stratapy')
    monkeypatch.setattr(module, 'render_columns', unexpected_render)
    link = dialog.interval_selection
    axis = dialog.column_figure.axes[0]
    event = SimpleNamespace(button=1, inaxes=axis, ydata=15, key=None)
    link.on_click(event)
    assert link.selected_ids() == {'layer-1'}
    assert dialog.column_intervals.selectionModel() is dialog.intervals.selectionModel()
    assert len(link.highlights) == len(dialog.column_figure.axes)
    link.on_click(SimpleNamespace(button=1, inaxes=axis, ydata=25, key='ctrl'))
    assert link.selected_ids() == {'layer-1', 'layer-2'}
    link.zoom_to_selected()
    assert axis.get_ylim() == pytest.approx((30.8, 9.2))
    assert not dialog.column_preview_timer.isActive()
    link.on_click(SimpleNamespace(button=1, inaxes=axis, ydata=25, key='ctrl'))
    assert link.selected_ids() == {'layer-1'}
    dialog.intervals.selectAll()
    assert len(link.highlights) == len(dialog.column_figure.axes)  # Merge adjacent spans.
    assert dialog.draft_undo.count() == 0


def test_overlapping_layers_and_other_wells_are_not_confused(draft):
    owner, dialog = draft
    for interval in [
        {'id': 'overlap', 'wellbore': 'A', 'top': 5, 'bottom': 15},
        {'id': 'other-well', 'wellbore': 'B', 'top': 0, 'bottom': 30}]:
        dialog.add_interval(interval={**interval, 'name': 'Same name', 'level': 'FORMATION', 'color': '#aabbcc'})
    dialog.show_column()
    link = dialog.interval_selection
    link.on_click(SimpleNamespace(button=1, inaxes=dialog.column_figure.axes[0], ydata=7, key=None))
    assert link.selected_ids() == {'layer-0', 'overlap'}
    assert dialog.column_intervals.isRowHidden(dialog.interval_history.rows_by_id['other-well'])
    dialog.column_toolbar.mode = 'pan/zoom'
    link.on_click(SimpleNamespace(button=1, inaxes=dialog.column_figure.axes[0], ydata=25, key=None))
    assert link.selected_ids() == {'layer-0', 'overlap'}


def test_hierarchy_selection_is_specific_to_clicked_track(draft, monkeypatch):
    from autopyro.ui.dialogs import stratigraphy_column as module

    _, dialog = draft
    dialog.add_interval(interval={'id': 'parent', 'wellbore': 'A', 'name': 'Parent',
                                 'level': 'GROUP', 'top': -5, 'bottom': 35, 'color': '#aabbcc'})
    dialog.column_style_form.hierarchy_checks['GROUP'].setChecked(True)
    dialog.show_column()
    link = dialog.interval_selection
    group, formation = dialog.column_figure.axes
    assert not link.view.isColumnHidden(2)
    assert link.view.itemDelegateForColumn(2).displayText('GROUP', None) != 'GROUP'
    assert group.get_ylim() == formation.get_ylim() == (35, -5)
    def unexpected_render(*args, **kwargs):
        pytest.fail('Hierarchy selection must not rerun stratapy')
    monkeypatch.setattr(module, 'render_columns', unexpected_render)
    link.on_click(SimpleNamespace(button=1, inaxes=group, ydata=15, key=None))
    assert link.selected_ids() == {'parent'}
    assert [artist.axes for artist in link.highlights] == [group]
    link.on_click(SimpleNamespace(button=1, inaxes=formation, ydata=15, key=None))
    assert link.selected_ids() == {'layer-1'}
    assert [artist.axes for artist in link.highlights] == [formation]
    link.on_click(SimpleNamespace(button=1, inaxes=group, ydata=15, key='shift'))
    assert link.selected_ids() == {'parent', 'layer-1'}
    assert len(link.highlights) == 2
    link.zoom_to_selected()
    assert group.get_ylim() == formation.get_ylim() == pytest.approx((36.6, -6.6))


def test_comparison_clicks_and_highlights_do_not_cross_wells(draft, monkeypatch):
    from autopyro.ui.dialogs import stratigraphy_column as module

    _, dialog = draft
    for well in ('B', 'excluded'):
        dialog.add_interval(interval={'id': well, 'wellbore': well, 'name': 'Same name',
                                     'level': 'FORMATION', 'top': 0, 'bottom': 40,
                                     'color': '#aabbcc', 'lithology': '607'})
    dialog.column_style_form.load({'comparison_wells': ['B']})
    dialog.show_column()
    link = dialog.interval_selection
    assert not link.view.isColumnHidden(0)
    assert link.view.isRowHidden(dialog.interval_history.rows_by_id['excluded'])
    first, second, lithology = dialog.column_figure.axes
    def unexpected_render(*args, **kwargs):
        pytest.fail('Selection must not rerender the comparison')
    monkeypatch.setattr(module, 'render_columns', unexpected_render)
    link.on_click(SimpleNamespace(button=1, inaxes=second, ydata=15, key=None))
    assert link.selected_ids() == {'B'}
    assert {artist.axes for artist in link.highlights} == {second, lithology}
    link.on_click(SimpleNamespace(button=1, inaxes=first, ydata=15, key='ctrl'))
    assert link.selected_ids() == {'B', 'layer-1'}
    assert {artist.axes for artist in link.highlights} == {first, second, lithology}
    link.zoom_to_selected()
    assert all(axis.get_ylim() == pytest.approx((41.6, -1.6)) for axis in dialog.column_figure.axes)
    assert not dialog.column_preview_timer.isActive()


def test_local_history_works_in_borrowed_workspace_properties(draft, qtbot):
    owner, dialog = draft
    owner.stratigraphy_window = dialog
    owner.show()
    owner.desktop.open_document('stratigraphy', dialog, 'Stratigraphy')
    owner.model.undo.push(QUndoCommand('Project sentinel'))
    dialog.intervals.item(0, 3).setText('2')
    dialog.name.setFocus()
    qtbot.wait(30)
    qtbot.keyClick(dialog.name, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert dialog.intervals.item(0, 3).text() == '0'
    assert owner.model.undo.index() == 1
    qtbot.keyClick(dialog.name, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert owner.model.undo.index() == 1  # Empty draft history must not fall through to project undo.



def test_canvas_event_and_auto_redraw_keep_interval_selection(draft, qtbot):
    from matplotlib.backend_bases import MouseEvent

    owner, dialog = draft
    dialog.show()
    dialog.show_column()
    dialog.column_canvas.draw()
    axis = dialog.column_figure.axes[0]
    x = sum(axis.get_xlim()) / 2
    px, py = axis.transData.transform((x, 15))
    event = MouseEvent('button_press_event', dialog.column_canvas, px, py, button=1)
    dialog.column_canvas.callbacks.process('button_press_event', event)
    assert dialog.interval_selection.selected_ids() == {'layer-1'}
    dialog.interval_selection.on_click(SimpleNamespace(button=1, inaxes=axis, ydata=10, key=None))
    assert dialog.interval_selection.selected_ids() == {'layer-1'}
    dialog.column_preview_timer.setInterval(40)
    dialog.column_style_form.fields['label_size'][0].setValue(12)
    qtbot.waitUntil(lambda: dialog.export_column_button.isEnabled(), timeout=3000)
    assert dialog.interval_selection.selected_ids() == {'layer-1'}
    assert len(dialog.interval_selection.highlights) == len(dialog.column_figure.axes)
    assert not dialog.zoom_intervals_action.icon().isNull()
    assert dialog.zoom_intervals_action.isEnabled()
    dialog.column_intervals.model().setData(dialog.column_intervals.model().index(1, 1), 'Edited beside plot')
    assert dialog.intervals.item(1, 1).text() == 'Edited beside plot'
    assert dialog.draft_undo.count() == 1
    dialog.draft_undo.undo()
    assert dialog.intervals.item(1, 1).text() == 'Same name'
