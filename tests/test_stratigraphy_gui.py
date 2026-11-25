from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt

from autopyro.io import build_project
from autopyro.stratigraphy_dialog import SetStratigraphyCommand, StratigraphyDialog
from autopyro.window import MainWindow


@pytest.fixture
def window(qtbot):
    project = build_project(["Well", "MD", "TOC"], [["A", "10", "1"], ["A", "25", "2"]], [1, 2],
        [(0, "Well", "text", ""), (1, "MD", "number", "m"), (2, "TOC", "number", "%wt")], {})
    w = MainWindow()
    qtbot.addWidget(w)
    w.set_project(project)
    w.may_replace = lambda: True
    yield w
    if w.plot_window:
        w.plot_window.close()
    w.dirty = False


def configure(window, qtbot):
    dialog = StratigraphyDialog(window)
    qtbot.addWidget(dialog)
    dialog.reference.setText("MD RKB")
    dialog.sample_reference.setText("MD RKB")
    dialog.well.setCurrentIndex(1)
    dialog.depth.setCurrentIndex(2)
    for name, top, bottom in [("Upper", 0, 20), ("Lower", 20, 30)]:
        dialog.add_interval(interval={"id": name, "name": name, "wellbore": "A", "level": "FORMATION",
                                      "top": top, "bottom": bottom, "color": "#c7ddeb"})
    return dialog


@pytest.fixture
def narrow_layout_font(qapp):
    from PySide6.QtGui import QFont, QFontDatabase

    original = qapp.font()
    font_path = Path('C:/Windows/Fonts/segoeui.ttf')
    # The Windows offscreen plugin does not load the system fonts by default.
    font_id = QFontDatabase.addApplicationFont(str(font_path)) if font_path.is_file() else -1
    if font_id >= 0:
        qapp.setFont(QFont('Segoe UI', 10))
    yield
    if font_id >= 0:
        qapp.setFont(original)
        QFontDatabase.removeApplicationFont(font_id)


def test_interval_commands_and_depths_fit_narrow_workspace(narrow_layout_font, window, qtbot):
    from PySide6.QtWidgets import QToolBar, QToolButton

    dialog = configure(window, qtbot)
    window.stratigraphy_window = dialog
    window.resize(1280, 720)
    window.show()
    window.show_stratigraphy()
    def workspace_layout_ready():
        # A newly embedded dialog is visible before QStackedWidget assigns
        # its page geometry and the docks finish moving the parameter tabs.
        return (dialog.intervals.isVisible()
                and dialog.size() == dialog.parentWidget().contentsRect().size())

    qtbot.waitUntil(workspace_layout_ready)
    buttons = {button.text(): button for button in dialog.findChildren(QToolButton) if button.menu()}
    for title in ('Импорт', 'Литология', 'Отчёты и экспорт'):
        button = buttons[title]
        assert button.isVisible() and not button.icon().isNull()
        assert all(not action.icon().isNull() for action in button.menu().actions())
    action = dialog.diagnose_intervals_action
    toolbar = next(bar for bar in dialog.findChildren(QToolBar) if action in bar.actions())
    assert toolbar.width() <= 650
    for action in toolbar.actions():
        assert toolbar.widgetForAction(action).isVisible() and not action.icon().isNull()

    def depths_visible(table):
        header = table.horizontalHeader()
        return all(0 <= header.sectionViewportPosition(column)
                   and header.sectionViewportPosition(column) + header.sectionSize(column) <= table.viewport().width()
                   for column in (3, 4))

    assert depths_visible(dialog.intervals)
    dialog.show_column()
    qtbot.waitUntil(dialog.column_intervals.isVisible)
    assert depths_visible(dialog.column_intervals)


def test_results_and_parameter_tabs_follow_each_other(window, qtbot):
    dialog = configure(window, qtbot)

    dialog.result_tabs.setCurrentWidget(dialog.column_page)
    assert dialog.parameter_tabs.currentIndex() == 1
    dialog.result_tabs.setCurrentWidget(dialog.results)
    assert dialog.parameter_tabs.currentIndex() == 2
    dialog.parameter_tabs.setCurrentIndex(0)
    assert dialog.result_tabs.currentWidget() is dialog.intervals
    dialog.parameter_tabs.setCurrentIndex(1)
    assert dialog.result_tabs.currentWidget() is dialog.column_page


def test_database_refresh_previews_changed_fields_and_undoes(window, qtbot, monkeypatch):
    from PySide6.QtWidgets import QCheckBox
    from autopyro.ui.widgets.interval_review import IntervalReviewDialog
    from autopyro.ui.widgets.stratigraphy_labels import geological_label

    dialog = configure(window, qtbot)
    current = dialog.read_scheme()
    source = {'type': 'postgresql_lithostratigraphy',
              'endpoint': {'host': 'localhost', 'port': 5432, 'database': 'geo'}}
    previous = current['intervals'][0]
    previous.update(import_source=source, lithology='607',
                    raw_cells={'wellbore_id': 7, 'source_interval_id': 19, 'interpretation_interval_id': 92})
    dialog.write_interval(0, previous)
    current = dialog.read_scheme()
    incoming = {**current, 'source': source, 'intervals': [{
        **previous, 'id': 'new-db-row', 'top': 1, 'bottom': 19, 'lithology': '608',
        'wellbore': 'A revised', 'level': 'MEMBER', 'name': 'Renamed layer',
        'raw_cells': {**previous['raw_cells'], 'interpretation_interval_id': 93},
    }]}

    def inspect(widget):
        model = widget.table.model()
        assert model.index(0, 1).data() == 'A → A revised'
        assert model.index(0, 2).data() == f"{geological_label('FORMATION')} → {geological_label('MEMBER')}"
        assert model.index(0, 3).data() == 'Upper → Renamed layer'
        assert model.index(0, 4).data() == '0 → 1'
        assert model.index(0, 5).data() == '20 → 19'
        widget.findChild(QCheckBox).setChecked(True)
        return widget.DialogCode.Accepted

    monkeypatch.setattr(IntervalReviewDialog, 'exec', inspect)
    assert dialog.review_merge(current, incoming)
    updated = dialog.read_scheme()['intervals'][0]
    assert (updated['id'], updated['top'], updated['bottom'], updated['lithology']) == ('Upper', 1, 19, '607')
    assert (updated['wellbore'], updated['level'], updated['name']) == ('A revised', 'MEMBER', 'Renamed layer')
    dialog.draft_undo.undo()
    assert dialog.read_scheme() == current


def test_database_import_can_replace_invalid_draft(window, qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from autopyro.database.ui.stratigraphy import StratigraphyDatabaseDialog

    dialog = configure(window, qtbot)
    incoming = dialog.read_scheme()
    incoming['intervals'] = incoming['intervals'][:1]
    dialog.intervals.item(1, 3).setText('unfinished')

    def select_revision(picker):
        picker.scheme = incoming
        return picker.DialogCode.Accepted

    monkeypatch.setattr(StratigraphyDatabaseDialog, 'exec', select_revision)
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.Cancel)
    dialog.import_database_scheme()
    assert dialog.intervals.item(1, 3).text() == 'unfinished'
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.Yes)
    dialog.import_database_scheme()
    assert dialog.intervals.rowCount() == 1
    dialog.draft_undo.undo()
    assert dialog.intervals.item(1, 3).text() == 'unfinished'


def test_database_merge_split_preview_recalculates_and_blocks_overlapping_update(window, qtbot, monkeypatch):
    from PySide6.QtWidgets import QCheckBox, QDialogButtonBox
    from autopyro.ui.widgets.interval_review import IntervalReviewDialog

    dialog = configure(window, qtbot)
    current = dialog.read_scheme()
    source = {'type': 'postgresql_lithostratigraphy',
              'endpoint': {'host': 'localhost', 'port': 5432, 'database': 'geo'}}
    upper, lower = current['intervals']
    upper.update(import_source=source, raw_cells={'wellbore_id': 7, 'source_interval_id': 1})
    dialog.write_interval(0, upper)
    current = dialog.read_scheme()
    incoming = {**current, 'source': source, 'intervals': [{**upper, 'bottom': 25}]}

    def review(widget):
        checkbox = widget.findChild(QCheckBox)
        buttons = widget.findChild(QDialogButtonBox)
        apply = buttons.button(QDialogButtonBox.StandardButton.Ok)
        checkbox.setChecked(True)
        model = widget.proxy.sourceModel()
        assert model.index(0, 6).data() == 'Перекрытие'
        assert 'Lower' in model.index(0, 7).data()
        assert not apply.isEnabled()
        return widget.DialogCode.Rejected

    monkeypatch.setattr(IntervalReviewDialog, 'exec', review)
    assert not dialog.review_merge(current, incoming)
    assert dialog.read_scheme() == current

    incoming['intervals'] = [{**upper, 'bottom': 10},
                             {**lower, 'id': 'split', 'name': 'Split', 'top': 10, 'bottom': 20,
                              'raw_cells': {'wellbore_id': 7, 'source_interval_id': 2}}]

    def review_split(widget):
        model = widget.proxy.sourceModel()
        assert model.index(1, 0).data() == 'Конфликт'
        widget.findChild(QCheckBox).setChecked(True)
        assert model.index(1, 0).data() == 'Добавление'
        assert model.rowCount() == 2  # The proposed split leaves neither a gap nor an overlap.
        assert widget.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).isEnabled()
        return widget.DialogCode.Accepted

    monkeypatch.setattr(IntervalReviewDialog, 'exec', review_split)
    assert dialog.review_merge(current, incoming)
    assert sorted((item['top'], item['bottom']) for item in dialog.read_scheme()['intervals']) == [
        (0, 10), (10, 20), (20, 30)]
    dialog.draft_undo.undo()
    assert dialog.read_scheme() == current


def test_database_import_uses_merge_preview_and_undo(window, qtbot, monkeypatch):
    from autopyro.database.ui import stratigraphy as database
    from autopyro.ui.widgets.interval_review import IntervalReviewDialog

    dialog = configure(window, qtbot)
    original = dialog.read_scheme()
    incoming = {**original, 'source': {'type': 'postgresql_lithostratigraphy'},
                'intervals': [{**original['intervals'][0], 'id': 'from-db', 'wellbore': 'B'}]}
    rows = [{'interpretation_id': 7, 'wellbore_name': 'B', 'version_tag': 'v2',
             'status': 'published', 'created_at': '2026-01-01'}]
    monkeypatch.setattr(database, 'interpretation_catalog', lambda parameters: rows)

    def read(parameters, ids, reference):
        assert ids == [7] and reference == 'MD RKB'
        return incoming

    def select(widget):
        widget.load_catalog()
        widget.table.selectRow(0)
        assert widget.read_button.isEnabled()
        widget.read_selection()
        return widget.result()

    monkeypatch.setattr(database, 'read_interpretations', read)
    monkeypatch.setattr(database.StratigraphyDatabaseDialog, 'run_task',
                        lambda self, work, callback: callback(work()))
    monkeypatch.setattr(database.StratigraphyDatabaseDialog, 'exec', select)
    monkeypatch.setattr(IntervalReviewDialog, 'exec', lambda self: self.DialogCode.Rejected)
    dialog.import_database_scheme()
    assert dialog.read_scheme() == original
    monkeypatch.setattr(IntervalReviewDialog, 'exec', lambda self: self.DialogCode.Accepted)
    dialog.import_database_scheme()
    added = dialog.read_scheme()['intervals'][-1]
    assert added['wellbore'] == 'B'
    assert added['import_source']['type'] == 'postgresql_lithostratigraphy'
    dialog.draft_undo.undo()
    assert dialog.read_scheme() == original


def test_summary_action_combines_pattern_variants_and_keeps_unknown_lithology(window, qtbot, monkeypatch):
    from PySide6.QtWidgets import QTabWidget
    from autopyro.ui.widgets.stratigraphy_summary import StratigraphySummaryDialog

    dialog = configure(window, qtbot)
    dialog.intervals.item(0, 7).setText('607')
    dialog.intervals.item(1, 7).setText('607*')
    dialog.add_interval(interval={'id': 'unknown', 'name': 'Third', 'wellbore': 'A',
                                  'level': 'FORMATION', 'top': 35, 'bottom': 40,
                                  'lithology': 'unknown-code'})
    original = dialog.read_scheme()
    seen = []

    def review(widget):
        tabs = widget.findChild(QTabWidget)
        totals, rocks = [tabs.widget(index).model().sourceModel().rows for index in range(2)]
        assert totals[0][3:] == (35, 5, 0, 0, 5)
        assert [(row[-2], row[-1]) for row in rocks if row[2] == 'Итого по уровню'] == [
            (30, '85.71%'), (5, '14.29%')]
        seen.append(True)
        return widget.DialogCode.Rejected

    monkeypatch.setattr(StratigraphySummaryDialog, 'exec', review)
    dialog.summary_action.trigger()
    assert seen
    assert dialog.read_scheme() == original


@pytest.mark.parametrize('decimal, separator', [(',', ';'), ('.', ',')])
def test_summary_csv_exports_both_sections_with_excel_numbers(window, qtbot, tmp_path, decimal, separator):
    import csv
    from codecs import BOM_UTF8
    from autopyro.ui.widgets.stratigraphy_labels import geological_label
    from autopyro.ui.widgets.stratigraphy_summary import StratigraphySummaryDialog

    dialog = configure(window, qtbot)
    dialog.intervals.item(0, 1).setText('=Слой;"А"')
    dialog.intervals.item(0, 4).setText('18.5')
    dialog.intervals.item(0, 7).setText('607')
    dialog.intervals.item(1, 3).setText('17')
    dialog.add_interval(interval={'id': 'group', 'name': 'Group', 'wellbore': 'A', 'level': 'GROUP',
                                  'top': 0, 'bottom': 30, 'color': '#c7ddeb'})
    widget = StratigraphySummaryDialog(dialog.read_scheme(), dialog.lithology_catalog, dialog)
    qtbot.addWidget(widget)
    path = tmp_path / 'summary.csv'
    widget.write_csv(path, decimal)
    assert path.read_bytes().startswith(BOM_UTF8)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream, delimiter=separator))
    totals = [row for row in rows if row['Тип строки'] == 'Итого по уровню']
    details = [row for row in rows if row['Тип строки'] == 'Подразделение']
    assert all(not row['Подразделение'] for row in totals)
    assert {row['Уровень'] for row in totals if row['Раздел'] == 'Мощность'} == {
        geological_label('GROUP'), geological_label('FORMATION')}
    totals = [row for row in totals if row['Уровень'] == geological_label('FORMATION')]
    assert totals[0]['Раздел'] == 'Мощность'
    assert totals[0]['Покрытая мощность [m]'] == '30'
    assert totals[0]['Перекрытия [m]'] == f'1{decimal}5'
    assert totals[0]['Разрывы [m]'] == '0'
    disputed = next(row for row in totals if row['Литология'] == 'Перекрытие')
    assert disputed['Покрытая мощность [m]'] == ''
    assert disputed['Доля, %'] == '5'
    assert all(not row['Покрытая мощность [m]'] and not row['Мощность [m]'] for row in details)
    assert all(not row['Покрытая мощность [m] · Подразделение'] for row in totals)
    subdivision = next(row for row in details
                       if row['Раздел'] == 'Мощность' and row['Подразделение'] == "'=Слой;\"А\"")
    assert subdivision['Покрытая мощность [m] · Подразделение'] == f'18{decimal}5'


def test_sample_track_reuses_positions_for_style_only_changes(window, qtbot, monkeypatch):
    from autopyro.ui.dialogs import stratigraphy_column as module

    dialog = configure(window, qtbot)
    dialog.sample_track.setChecked(True)
    calculate = module.sample_positions
    calls = []

    def record(project, row_ids, *args, **kwargs):
        calls.append(list(row_ids))
        return calculate(project, row_ids, *args, **kwargs)

    monkeypatch.setattr(module, 'sample_positions', record)
    dialog.show_column()
    dialog.column_style_form.load({'label_size': 12})
    dialog.render_column_preview(automatic=True)
    assert len(calls) == 1

    window.model.setData(window.model.index(0, 1), '11')
    dialog.show_column()
    assert len(calls) == 2
    axis = next(axis for axis in dialog.column_figure.axes if axis._autopyro_kind == 'samples')
    assert sorted(sample.top for rows in axis._autopyro_sample_artists.values() for sample in rows) == [11, 25]

    window.analysis_scope.set_mode('selected')
    dialog.show_column()
    assert calls[-1] == [] and len(calls) == 3
    assert all(axis._autopyro_kind != 'samples' for axis in dialog.column_figure.axes)


def test_compact_interval_select_all_skips_hidden_rows(window, qtbot):
    dialog = configure(window, qtbot)
    dialog.add_interval(interval={'id': 'B', 'name': 'Other well', 'wellbore': 'B',
                                  'level': 'FORMATION', 'top': 0, 'bottom': 20})
    dialog.add_interval(interval={'id': 'A-base', 'name': 'Base', 'wellbore': 'A',
                                  'level': 'FORMATION', 'top': 30, 'bottom': 40})
    dialog.show()
    dialog.show_column()
    compact = dialog.column_intervals
    assert compact.isRowHidden(dialog.interval_history.rows_by_id['B'])
    compact.setFocus()
    qtbot.keyClick(compact, Qt.Key.Key_A, modifier=Qt.KeyboardModifier.ControlModifier)
    assert dialog.interval_selection.selected_ids() == {'Upper', 'Lower', 'A-base'}

    dialog.result_tabs.setCurrentWidget(dialog.intervals)
    qtbot.keyClick(dialog.intervals, Qt.Key.Key_A, modifier=Qt.KeyboardModifier.ControlModifier)
    assert dialog.interval_selection.selected_ids() == {'Upper', 'Lower', 'A-base', 'B'}


@pytest.mark.parametrize('input_method', ['keyboard', 'mouse'])
def test_compact_interval_shift_selection_skips_hidden_wells(window, qtbot, input_method):
    dialog = configure(window, qtbot)
    for key, well, top in [('B', 'B', 0), ('A-base', 'A', 30)]:
        dialog.add_interval(interval={'id': key, 'name': key, 'wellbore': well,
                                      'level': 'FORMATION', 'top': top, 'bottom': top + 10})
    dialog.show()
    dialog.show_column()
    compact = dialog.column_intervals
    lower = compact.model().index(1, 3)
    base = compact.model().index(3, 3)
    qtbot.mouseClick(compact.viewport(), Qt.MouseButton.LeftButton, pos=compact.visualRect(lower).center())
    if input_method == 'keyboard':
        qtbot.keyClick(compact, Qt.Key.Key_Down, modifier=Qt.KeyboardModifier.ShiftModifier)
    else:
        qtbot.mouseClick(compact.viewport(), Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.ShiftModifier, pos=compact.visualRect(base).center())
    assert dialog.interval_selection.selected_ids() == {'Lower', 'A-base'}
    qtbot.keyClick(compact, Qt.Key.Key_Up, modifier=Qt.KeyboardModifier.ShiftModifier)
    assert dialog.interval_selection.selected_ids() == {'Lower'}


def test_column_click_selects_overlaps_and_respects_shared_boundary(window, qtbot):
    dialog = configure(window, qtbot)
    dialog.add_interval(interval={'id': 'overlap', 'name': 'Overlap', 'wellbore': 'A',
                                  'level': 'FORMATION', 'top': 18, 'bottom': 22})
    dialog.show_column()
    selection = dialog.interval_selection
    axis = dialog.column_figure.axes[0]
    selection.on_click(SimpleNamespace(button=1, inaxes=axis, ydata=19, key=None))
    assert selection.selected_ids() == {'Upper', 'overlap'}
    assert len(dialog.intervals.selectionModel().selection()) == 2
    selection.on_click(SimpleNamespace(button=1, inaxes=axis, ydata=20, key=None))
    assert selection.selected_ids() == {'Lower', 'overlap'}
    assert len(dialog.intervals.selectionModel().selection()) == 1


def test_sample_track_opens_project_row_from_point(window, qtbot):
    from matplotlib.backend_bases import MouseEvent
    from autopyro.ui.table.selection import selected_row_ids

    dialog = configure(window, qtbot)
    dialog.sample_track.setChecked(True)
    dialog.show_column()
    assert dialog.export_column_button.isEnabled(), dialog.status.text()
    axis = next(axis for axis in dialog.column_figure.axes if axis._autopyro_kind == 'samples')
    assert axis.get_ylim() == dialog.column_figure.axes[0].get_ylim()
    assert axis.get_title() == 'Пробы'
    x, y = axis.transData.transform((.5, 10))
    event = MouseEvent('button_press_event', dialog.column_canvas, x, y, button=1)
    assert event.inaxes is axis
    dialog.interval_selection.on_click(event)
    assert selected_row_ids(window) == {window.model.project.rows[0].id}
    assert window.desktop.documents.currentWidget() is window.desktop.table_page
    assert not dialog.interval_selection.selected_ids()
    dialog.depth.setCurrentIndex(0)
    dialog.show_column()
    assert dialog.export_column_button.isEnabled()
    assert all(axis._autopyro_kind != 'samples' for axis in dialog.column_figure.axes)
    assert 'Дорожка проб' in dialog.status.text()


def test_revision_comparison_can_open_previous_scheme_after_replacement(window, qtbot, monkeypatch):
    from autopyro.ui.widgets.stratigraphy_revisions import RevisionComparisonDialog

    dialog = configure(window, qtbot)
    dialog.save_scheme()
    replacement = {**dialog.read_scheme(), 'id': 'another-import', 'name': 'Replacement'}
    dialog.stage_import(replacement, None)
    opened = []

    def inspect(picker):
        opened.extend(scheme['id'] for scheme in picker.schemes)
        return picker.DialogCode.Rejected

    monkeypatch.setattr(RevisionComparisonDialog, 'exec', inspect)
    dialog.compare_revisions()
    assert opened == [window.model.project.view['stratigraphy_scheme']['id'], replacement['id']]


def test_revision_comparison_shows_changes_and_navigates_to_historical_column(window, qtbot, monkeypatch, tmp_path):
    import csv
    from threading import get_ident
    from PySide6.QtCore import QLocale
    from PySide6.QtWidgets import QDialog, QFileDialog
    from autopyro.ui.widgets.figure_canvas import FittedFigureCanvas
    from autopyro.ui.widgets.stratigraphy_revisions import RevisionComparisonDialog
    from autopyro.ui.dialogs import stratigraphy_column as module
    import autopyro.ui.widgets.stratigraphy_revisions as revisions

    dialog = configure(window, qtbot)
    dialog.BACKGROUND_BINDING_ROWS = 1
    dialog.BACKGROUND_PREVIEW_INTERVALS = 1
    gui_thread = get_ident()
    compare_samples = revisions.sample_changes
    render_columns = module.render_columns
    render_threads = []
    gui_redraws = []
    draw_canvas = FittedFigureCanvas.draw

    def render(*args, **kwargs):
        render_threads.append(get_ident())
        return render_columns(*args, **kwargs)

    def draw(canvas):
        if canvas.figure.axes:
            gui_redraws.append(True)
        draw_canvas(canvas)

    def record(project, *args, **kwargs):
        assert get_ident() != gui_thread
        assert project is not window.model.project
        return compare_samples(project, *args, **kwargs)

    monkeypatch.setattr(revisions, 'sample_changes', record)
    monkeypatch.setattr(module, 'render_columns', render)
    monkeypatch.setattr(FittedFigureCanvas, 'draw', draw)
    dialog.save_scheme()
    dialog.revision.setText('2')
    dialog.intervals.item(0, 4).setText('9')
    dialog.intervals.item(0, 7).setText('607')
    dialog.intervals.item(1, 3).setText('9')
    dialog.intervals.item(1, 1).setText('Renamed')
    draft = dialog.read_scheme()
    project_view = deepcopy(window.model.project.view)
    viewed = []
    paths = iter([tmp_path / 'intervals.csv', tmp_path / 'samples.csv'])
    monkeypatch.setattr(QFileDialog, 'exec', lambda self: self.DialogCode.Accepted)
    monkeypatch.setattr(QFileDialog, 'selectedFiles', lambda self: [str(next(paths))])

    exec_dialog = QDialog.exec

    def show_column(widget):
        canvas = widget.findChild(FittedFigureCanvas)
        if canvas is None:
            return exec_dialog(widget)
        width, height = map(int, canvas.figure.bbox.size)
        assert canvas.buffer_rgba().shape[:2] == (height, width)
        assert any(list(line.get_ydata()) == [20, 20] for line in canvas.figure.axes[0].lines)
        viewed.append(canvas.figure.axes[0].get_ylim())
        initial_pixels = bytes(canvas.buffer_rgba())
        canvas.figure.axes[0].set_ylim(24, 16)
        canvas.draw_idle()
        assert canvas.navigation_draw_pending
        qtbot.waitUntil(lambda: not canvas.navigation_draw_pending)
        assert bytes(canvas.buffer_rgba()) != initial_pixels
        return QDialog.DialogCode.Rejected

    def review(widget):
        assert not widget.export_button.isEnabled()
        widget.compare()
        intervals = widget.tables[0].model().rows
        assert intervals[0][:2] == ('Подошва, Литология', 'По идентификатору')
        assert '[607]' in intervals[0][-1]
        assert 'Подразделение' in intervals[1][0]
        assert len(widget.tables[1].model().rows) == 2
        assert 'Upper' in widget.tables[1].model().rows[0][-2]
        assert 'Renamed' in widget.tables[1].model().rows[0][-1]
        widget.navigation_buttons[0].click()
        widget.export_button.click()
        widget.tabs.setCurrentIndex(1)
        widget.export_button.click()
        widget.samples.setChecked(False)
        assert not widget.export_button.isEnabled()
        widget.samples.setChecked(True)

        def cancel(*args, **kwargs):
            raise revisions.TaskCancelled()

        with monkeypatch.context() as patch:
            patch.setattr(dialog, 'calculate_binding_report', cancel)
            widget.compare()
        assert widget.tables[0].model().rowCount() == len(intervals)
        assert widget.tables[1].model().rowCount() == 0
        assert 'отменена' in widget.summary.text()
        return widget.DialogCode.Rejected

    monkeypatch.setattr(QDialog, 'exec', show_column)
    monkeypatch.setattr(RevisionComparisonDialog, 'exec', review)
    dialog.compare_revisions()
    assert len(render_threads) == 1 and render_threads[0] != gui_thread
    assert not gui_redraws
    assert viewed == [pytest.approx((20.75, 19.25))]
    assert dialog.read_scheme() == draft
    assert window.model.project.view == project_view
    separator = ';' if QLocale.system().decimalPoint() == ',' else ','
    with (tmp_path / 'intervals.csv').open(encoding='utf-8-sig', newline='') as stream:
        intervals = list(csv.DictReader(stream, delimiter=separator))
    assert intervals[0]['Подошва · До'] == '20'
    assert intervals[0]['Подошва · После'] == '9'
    assert intervals[0]['Литология · После'] == '607'
    assert intervals[0]['Сопоставление'] == 'По идентификатору'
    with (tmp_path / 'samples.csv').open(encoding='utf-8-sig', newline='') as stream:
        samples = list(csv.DictReader(stream, delimiter=separator))
    details = [row for row in samples if row['Тип строки'] == 'Подразделение']
    assert [(row['Сторона сравнения'], row['Подразделение']) for row in details[:2]] == [
        ('До', 'Upper'), ('После', 'Renamed')]
    assert samples[0]['Верх отбора'] == '10'


def test_correlation_requires_existing_intervals_and_can_be_hidden(window, qtbot):
    from autopyro.ui.widgets.stratigraphy_correlation import CorrelationDialog

    owner = configure(window, qtbot)
    intervals = owner.read_scheme()['intervals']
    intervals.append({**intervals[0], 'id': 'B-top', 'wellbore': 'B'})
    settings = {'edge': 'top', 'anchors': {'A': 'Upper', 'B': 'B-top'},
                'horizons': [{'name': 'Marker', 'edge': 'bottom',
                              'anchors': {'A': 'Upper', 'B': 'deleted'}}]}
    dialog = CorrelationDialog(['A', 'B'], intervals, settings, 'm', owner)
    qtbot.addWidget(dialog)
    assert not dialog.apply_button.isEnabled()
    assert 'Marker' in dialog.validation.text()
    dialog.mode.setCurrentIndex(0)
    assert dialog.apply_button.isEnabled()
    assert dialog.values() == {**settings, 'enabled': False}
    dialog.mode.setCurrentIndex(dialog.mode.findData('top'))
    assert not dialog.apply_button.isEnabled()
    selector = dialog.horizon_selectors['B']
    selector.setCurrentIndex(selector.findData('B-top'))
    assert dialog.apply_button.isEnabled()
    assert dialog.values()['horizons'][0]['anchors'] == {'A': 'Upper', 'B': 'B-top'}


def test_disabled_correlation_saves_horizon_removal_with_missing_zero_anchor(window, qtbot):
    from autopyro.ui.widgets.stratigraphy_correlation import CorrelationDialog
    from autopyro.geology.stratigraphy import column_style

    owner = configure(window, qtbot)
    intervals = owner.read_scheme()['intervals']
    intervals.append({**intervals[0], 'id': 'B-top', 'wellbore': 'B'})
    settings = {'edge': 'top', 'anchors': {'A': 'deleted', 'B': 'B-top'},
                'horizons': [{'name': 'Marker', 'edge': 'bottom', 'anchors': {'A': 'Upper', 'B': 'B-top'}}]}
    picker = CorrelationDialog(['A', 'B'], intervals, settings, 'm', owner)
    qtbot.addWidget(picker)
    assert not picker.apply_button.isEnabled()
    picker.remove_horizon_button.click()
    picker.mode.setCurrentIndex(0)
    assert picker.apply_button.isEnabled()
    result = picker.values()
    assert result == {'edge': 'top', 'anchors': settings['anchors'], 'enabled': False}
    column_style({'correlation': result})
    reopened = CorrelationDialog(['A', 'B'], intervals, result, 'm', owner)
    qtbot.addWidget(reopened)
    assert reopened.horizons == []
    reopened.mode.setCurrentIndex(reopened.mode.findData('top'))
    assert not reopened.apply_button.isEnabled()
    reopened.selectors['A'].setCurrentIndex(reopened.selectors['A'].findData('Upper'))
    assert reopened.apply_button.isEnabled()


def test_correlation_picker_aligns_wells_and_keeps_source_depths(window, qtbot, monkeypatch, tmp_path):
    from matplotlib.patches import ConnectionPatch
    from autopyro.ui.widgets.stratigraphy_correlation import CorrelationDialog

    dialog = configure(window, qtbot)
    for key, top, bottom in [('B-top', 100, 115), ('B-base', 115, 140)]:
        dialog.add_interval(interval={'id': key, 'wellbore': 'B', 'level': 'FORMATION',
                                      'name': 'Repeated name', 'top': top, 'bottom': bottom,
                                      'color': '#c7ddeb', 'lithology': '607'})
    form = dialog.column_style_form
    form.load({'comparison_wells': ['B']})
    before = dialog.read_scheme()

    def choose(widget):
        widget.mode.setCurrentIndex(widget.mode.findData('bottom'))
        assert not widget.apply_button.isEnabled()
        for well, key in [('A', 'Upper'), ('B', 'B-top')]:
            selector = widget.selectors[well]
            selector.setCurrentIndex(selector.findData(key))
        assert widget.apply_button.isEnabled()
        for name, edge, anchors in (
            ('Marker top', 'top', {'A': 'Upper', 'B': 'B-top'}),
            ('Marker base', 'bottom', {'A': 'Lower', 'B': 'B-base'}),
        ):
            widget.add_horizon_button.click()
            assert not widget.apply_button.isEnabled()
            widget.horizon_name.setText(name)
            widget.horizon_edge.setCurrentIndex(widget.horizon_edge.findData(edge))
            for well, key in anchors.items():
                selector = widget.horizon_selectors[well]
                selector.setCurrentIndex(selector.findData(key))
            assert widget.apply_button.isEnabled()
        widget.add_horizon_button.click()
        assert not widget.apply_button.isEnabled()
        widget.remove_horizon_button.click()
        assert widget.apply_button.isEnabled()
        return widget.DialogCode.Accepted

    monkeypatch.setattr(CorrelationDialog, 'exec', choose)
    form.correlation_button.click()
    dialog.show_column()
    assert dialog.export_column_button.isEnabled(), dialog.status.text()
    tracks = dialog.column_figure.axes
    assert [axis._autopyro_depth_origin for axis in tracks] == [20, 115, 115]
    assert all(axis.get_ylim() == (25, -20) for axis in tracks)
    connectors = [artist for artist in dialog.column_figure.artists if isinstance(artist, ConnectionPatch)]
    assert [(line.xy1, line.xy2) for line in connectors] == [
        ((1, -20), (0, -15)), ((1, 10), (0, 25))]
    assert sorted(tracks[1].patches[0].get_xy()[:, 1])[0] == -15
    assert all(any(list(line.get_ydata()) == [0, 0] for line in axis.lines) for axis in tracks)
    assert '120 m' in tracks[1].format_coord(.5, 5)
    assert '5 m' in tracks[1].format_coord(.5, 5)
    link = dialog.interval_selection
    link.on_click(SimpleNamespace(button=1, inaxes=tracks[2], ydata=-5, key=None))
    assert link.selected_ids() == {'B-top'}
    assert len(link.highlights) == 2
    assert all(artist.get_y() == -15 for artist in link.highlights)
    path_before_zoom = connectors[0].get_path().vertices.copy()
    link.zoom_to_selected()
    assert tracks[0].get_ylim() == pytest.approx((.6, -15.6))
    assert (connectors[0].get_path().vertices != path_before_zoom).any()
    assert dialog.read_scheme() == before
    assert dialog.style_values()['correlation']['anchors'] == {'A': 'Upper', 'B': 'B-top'}
    dialog.column_well.setCurrentIndex(dialog.column_well.findData('B'))
    form.load({'comparison_wells': ['A']})
    dialog.save_scheme()
    restored = StratigraphyDialog(window)
    qtbot.addWidget(restored)
    assert restored.column_well.currentData() == 'B'
    assert restored.style_values()['correlation'] == form.correlation
    restored.show_column()
    assert restored.export_column_button.isEnabled(), restored.status.text()
    from PySide6.QtWidgets import QFileDialog
    output = tmp_path / 'correlation.svg'
    monkeypatch.setattr(QFileDialog, 'exec', lambda self: self.DialogCode.Accepted)
    monkeypatch.setattr(QFileDialog, 'selectedFiles', lambda self: [str(output)])
    restored.export_column()
    content = output.read_text(encoding='utf-8')
    assert 'Глубина относительно горизонта' in content
    assert 'Marker top' in content and 'Marker base' in content

    configured = restored.style_values()['correlation']

    def hide(widget):
        widget.mode.setCurrentIndex(0)
        assert widget.apply_button.isEnabled()
        return widget.DialogCode.Accepted

    monkeypatch.setattr(CorrelationDialog, 'exec', hide)
    restored.configure_correlation()
    assert restored.style_values()['correlation'] == {**configured, 'enabled': False}
    restored.save_scheme()
    reopened = StratigraphyDialog(window)
    qtbot.addWidget(reopened)
    assert reopened.style_values()['correlation'] == {**configured, 'enabled': False}
    reopened.show_column()
    assert reopened.column_figure.axes[0].get_ylim() == (140, 0)
    assert not any(isinstance(artist, ConnectionPatch) for artist in reopened.column_figure.artists)

    def show_again(widget):
        assert widget.mode.currentData() is None
        widget.mode.setCurrentIndex(widget.mode.findData('bottom'))
        assert widget.apply_button.isEnabled()
        return widget.DialogCode.Accepted

    monkeypatch.setattr(CorrelationDialog, 'exec', show_again)
    reopened.configure_correlation()
    assert reopened.style_values()['correlation'] == configured
    reopened.show_column()
    assert sum(isinstance(artist, ConnectionPatch) for artist in reopened.column_figure.artists) == 2


def test_interval_diagnostics_lists_invalid_cells_and_locates_selected_error(window, qtbot, monkeypatch):
    from PySide6.QtWidgets import QPushButton
    from autopyro.ui.widgets.interval_review import IntervalReviewDialog

    dialog = configure(window, qtbot)
    dialog.intervals.item(0, 3).setText('bad depth')
    dialog.intervals.item(0, 4).setText('')
    dialog.intervals.item(1, 0).setText('')
    dialog.intervals.item(1, 5).setText('bad color')

    def review(widget):
        model = widget.table.model()
        assert [(model.index(row, 0).data(), model.index(row, 1).data())
                for row in range(model.rowCount())] == [
                    ('1', 'Кровля'), ('1', 'Подошва'), ('2', 'Ствол'), ('2', 'Цвет заливки')]
        widget.table.selectRow(3)
        button = next(button for button in widget.findChildren(QPushButton)
                      if button.text() == 'Перейти к ячейке')
        assert not button.icon().isNull()
        button.click()
        return widget.result()

    monkeypatch.setattr(IntervalReviewDialog, 'exec', review)
    dialog.review_intervals()
    assert dialog.intervals.currentItem() is dialog.intervals.item(1, 5)
    assert dialog.intervals.currentItem().text() == 'bad color'


def test_interval_diagnostics_locates_empty_scheme_settings(window, qtbot, monkeypatch):
    from PySide6.QtWidgets import QPushButton
    from autopyro.ui.widgets.interval_review import IntervalReviewDialog

    dialog = configure(window, qtbot)
    dialog.name.clear()
    dialog.revision.clear()
    dialog.reference.setText('  ')
    dialog.intervals.item(0, 3).setText('bad depth')
    dialog.parameter_tabs.setCurrentIndex(1)

    def review(widget):
        model = widget.table.model()
        assert model.rowCount() == 4
        assert [model.index(row, 1).data() for row in range(4)] == [
            'Название схемы', 'Редакция', 'Тип / начало отсчёта интервалов', 'Кровля']
        widget.table.selectRow(2)
        button = next(button for button in widget.findChildren(QPushButton)
                      if button.text() == 'Перейти к полю')
        assert not button.icon().isNull()
        button.click()
        return widget.result()

    monkeypatch.setattr(IntervalReviewDialog, 'exec', review)
    dialog.review_intervals()
    assert dialog.parameter_tabs.currentIndex() == 0
    assert dialog.result_tabs.currentWidget() is dialog.intervals
    assert dialog.reference.selectedText() == '  '


@pytest.mark.parametrize('build_succeeds', [True, False])
def test_diagnostic_jump_selects_rows_after_opening_target_well(window, qtbot, monkeypatch, build_succeeds):
    from autopyro.geology.stratigraphy_review import Finding

    dialog = configure(window, qtbot)
    for index in range(35):
        dialog.add_interval(interval={'id': f'B-{index}', 'name': f'Layer {index}', 'wellbore': 'B',
                                      'level': 'FORMATION', 'top': index, 'bottom': index + 1})
    dialog.show()
    dialog.show_column()
    selection = dialog.interval_selection
    select = selection.select_ids
    calls = []

    def record(ids):
        calls.append((dialog.result_tabs.currentWidget(), set(ids) <= selection.rendered.keys()))
        select(ids)

    monkeypatch.setattr(selection, 'select_ids', record)
    if not build_succeeds:
        monkeypatch.setattr(dialog, 'show_column', lambda: setattr(dialog, '_has_column_preview', True))
    finding = Finding('missing', 'B', 'FORMATION', 34, 35, ('B-34',))
    dialog.locate_finding(finding, True)
    assert selection.selected_ids() == {'B-34'}
    if build_succeeds:
        assert calls == [(dialog.column_page, True)]
        index = dialog.intervals.model().index(dialog.interval_history.rows_by_id['B-34'], 3)
        qtbot.waitUntil(lambda: selection.view.viewport().rect().intersects(selection.view.visualRect(index)))
        assert dialog.column_figure.axes[0].get_ylim() == pytest.approx((35.04, 33.96))
    else:
        assert calls == [(dialog.intervals, False)]


def test_interval_diagnostics_locates_gap_and_keeps_nested_coverage(window, qtbot, monkeypatch):
    from autopyro.stratigraphy_review import diagnose
    from autopyro.ui.widgets.interval_review import IntervalReviewDialog
    from PySide6.QtWidgets import QPushButton

    dialog = configure(window, qtbot)
    dialog.intervals.item(1, 3).setText('25')
    first = dialog.read_scheme()['intervals'][0]
    dialog.add_interval(interval={**first, 'id': 'nested', 'top': 5, 'bottom': 10})
    dialog.add_interval(interval={**first, 'id': 'duplicate'})
    dialog.add_interval(interval={**first, 'id': 'parent', 'level': 'GROUP', 'bottom': 30})
    findings = diagnose(dialog.read_scheme()['intervals'])
    gaps = [(item.top, item.bottom) for item in findings if item.kind == 'gap']
    assert gaps == [(20, 25)]
    assert sum(item.kind == 'duplicate' for item in findings) == 1
    assert sum(item.kind == 'overlap' for item in findings) == 1

    def review(widget):
        row = next(index for index, item in enumerate(findings) if item.kind == 'gap')
        widget.table.selectRow(row)
        button = next(button for button in widget.findChildren(QPushButton)
                      if button.text() == 'Показать на колонке')
        button.click()
        return widget.result()

    monkeypatch.setattr(IntervalReviewDialog, 'exec', review)
    dialog.review_intervals()
    assert dialog.result_tabs.currentWidget() is dialog.column_page
    assert dialog.interval_selection.selected_ids() == {'Upper', 'Lower'}
    assert dialog.column_figure.axes[0].get_ylim() == pytest.approx((25.2, 19.8))


@pytest.mark.parametrize('decision', ['cancel', 'add', 'update'])
def test_merge_import_previews_conflicts_and_undoes_as_one_edit(window, qtbot, monkeypatch, tmp_path, decision):
    from PySide6.QtWidgets import QCheckBox, QFileDialog
    from autopyro.ui.widgets.interval_review import IntervalReviewDialog

    dialog = configure(window, qtbot)
    before = dialog.read_scheme()
    upper, lower = before['intervals']
    incoming = {**before, 'source': {'path': 'incoming.json'}, 'intervals': [
        {**upper, 'id': 'new-upper', 'name': 'Updated'},
        {**lower, 'id': 'same'},
        {**lower, 'id': 'conflict', 'top': 29, 'bottom': 33},
        {**lower, 'id': 'Upper', 'wellbore': 'B', 'raw_cells': {'rock_name': ''}},
    ]}
    incoming['intervals'][-1].pop('color')
    path = tmp_path / 'incoming.json'
    path.write_text(json.dumps(incoming), encoding='utf-8')
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(path), ''))

    def review(widget):
        assert [row[0] for row in widget.proxy.sourceModel().rows] == [
            'Обновление', 'Дубль', 'Конфликт', 'Добавление']
        assert dialog.read_scheme() == before
        widget.findChild(QCheckBox).setChecked(decision == 'update')
        return widget.DialogCode.Rejected if decision == 'cancel' else widget.DialogCode.Accepted

    monkeypatch.setattr(IntervalReviewDialog, 'exec', review)
    count = dialog.draft_undo.count()
    dialog.merge_intervals_action.trigger()
    after = dialog.read_scheme()
    if decision == 'cancel':
        assert after == before
        assert dialog.draft_undo.count() == count
        return
    assert [item['wellbore'] for item in after['intervals']] == ['A', 'A', 'B']
    assert after['intervals'][0]['id'] == 'Upper'
    assert after['intervals'][0]['name'] == ('Updated' if decision == 'update' else 'Upper')
    added = after['intervals'][-1]
    assert added['id'] not in {'Upper', 'Lower'}
    assert added['import_source'] == incoming['source']
    assert dialog.draft_undo.count() == count + 1
    dialog.draft_undo.undo()
    assert dialog.read_scheme() == before
    dialog.draft_undo.redo()
    assert dialog.read_scheme() == after


def test_merge_rejects_depth_mismatch_and_conflicting_incoming_intervals():
    from autopyro.stratigraphy_review import plan_merge

    layer = {'wellbore': 'A', 'level': 'FORMATION', 'name': 'Layer'}
    current = {'depth_reference': 'MD RKB', 'depth_unit': 'm', 'intervals': [
        {**layer, 'id': 'current', 'top': 0, 'bottom': 10},
    ]}
    incoming = {**current, 'intervals': [
        {**layer, 'id': 'x', 'top': 10, 'bottom': 20},
        {**layer, 'id': 'y', 'top': 15, 'bottom': 22},
        {**layer, 'id': 'z', 'top': 22, 'bottom': 30},
    ]}
    assert [entry.kind for entry in plan_merge(current, incoming)] == ['conflict', 'conflict', 'add']
    for change in ({'depth_unit': 'ft'}, {'depth_reference': 'TVDSS'}):
        with pytest.raises(ValueError, match='система глубин'):
            plan_merge(current, {**incoming, **change})


def test_commit_columns_edit_recompute_and_undo(window, qtbot, monkeypatch):
    from autopyro.geology import stratigraphy

    dialog = configure(window, qtbot)
    dialog.preview()
    assert dialog.results.model().rowCount() == 2, dialog.status.text()
    assert not window.model.project.view.get("stratigraphy")
    dialog.BACKGROUND_BINDING_ROWS = 1
    dialog.add_columns()
    project = window.model.project
    assert len(project.columns) == 5, dialog.status.text()
    name, status = project.columns[-2:]
    assert project.rows[0].values[name.id] == "Upper"
    assert not window.model.flags(window.model.index(0, 3)) & Qt.ItemFlag.ItemIsEditable
    assert not window.model.setData(window.model.index(0, 3), "spoof")
    with pytest.raises(ValueError, match="чтения"):
        window.parsed_change(window.model.index(0, 3), "spoof")
    evaluated = []
    evaluate = stratigraphy.evaluate_project

    def record(project, ids, *args, **kwargs):
        evaluated.append(list(ids))
        return evaluate(project, ids, *args, **kwargs)

    monkeypatch.setattr(stratigraphy, 'evaluate_project', record)
    untouched = name.metadata['diagnostics'][project.rows[1].id]
    window.model.setData(window.model.index(0, 1), "22")
    assert project.rows[0].values[name.id] == "Lower"
    window.model.undo.undo()
    assert project.rows[0].values[name.id] == "Upper"
    assert evaluated == [[project.rows[0].id]] * 2
    assert name.metadata['diagnostics'][project.rows[1].id] is untouched
    window.model.undo.undo()
    assert len(project.columns) == 3 and "stratigraphy" not in project.view
    evaluated.clear()
    window.model.undo.redo()
    assert len(project.columns) == 5 and project.rows[0].values[name.id] == "Upper"
    # The intervening edit invalidates the first operation's cached data.
    assert evaluated == [[row.id for row in project.rows]]
    evaluated.clear()
    dialog.add_columns()
    assert len(project.columns) == 5  # Applying twice does not duplicate columns.
    assert not evaluated


def test_binding_with_new_columns_restores_both_states_without_recalculation(window, qtbot, monkeypatch):
    from autopyro.geology import stratigraphy
    from autopyro.table import AddColumnCommand
    from autopyro.ui.widgets.interval_review import IntervalReviewDialog

    dialog = configure(window, qtbot)
    dialog.apply()
    project = window.model.project
    formation = stratigraphy.result_column('FORMATION')
    window.model.undo.push(AddColumnCommand(window.model, formation, {}))
    before_columns = list(project.columns)
    before = deepcopy(project.view['stratigraphy'])
    dialog.revision.setText('2')
    dialog.intervals.item(0, 4).setText('9')
    dialog.intervals.item(1, 3).setText('9')
    dialog.BACKGROUND_BINDING_ROWS = 1
    monkeypatch.setattr(IntervalReviewDialog, 'exec', lambda self: self.DialogCode.Accepted)

    def unexpected_evaluation(*args, **kwargs):
        pytest.fail('The combined operation should use the prepared assignments')

    monkeypatch.setattr(stratigraphy, 'evaluate_project', unexpected_evaluation)
    dialog.add_columns()
    added, = project.columns[len(before_columns):]
    assert project.rows[0].values[formation.id] == 'Lower'
    after_diagnostics = deepcopy(added.metadata['diagnostics'])
    window.model.undo.undo()
    assert project.columns == before_columns
    assert added.id not in project.rows[0].values
    assert project.view['stratigraphy'] == before
    assert project.rows[0].values[formation.id] == 'Upper'
    window.model.undo.redo()
    assert project.columns == [*before_columns, added]
    assert project.rows[0].values[formation.id] == 'Lower'
    assert added.metadata['diagnostics'] == after_diagnostics


def test_cancel_private_draft_and_reused_revision_rejected(window, qtbot):
    dialog = configure(window, qtbot)
    before = deepcopy(window.model.project.view)
    dialog.reject()
    assert window.model.project.view == before
    dialog.apply()
    current = deepcopy(window.model.project.view["stratigraphy"])
    current["scheme"]["intervals"][0]["name"] = "changed"
    with pytest.raises(ValueError, match="revision"):
        SetStratigraphyCommand(window.model, current)
    current["scheme"]["revision"] = "2"
    window.model.undo.push(SetStratigraphyCommand(window.model, current))
    assert window.model.project.view["stratigraphy"]["scheme"]["revision"] == "2"
    window.model.undo.undo()
    assert window.model.project.view["stratigraphy"]["scheme"]["revision"] == "1"


@pytest.mark.parametrize('cancel', [False, True])
def test_large_binding_preview_uses_detached_input_and_handles_cancel(window, qtbot, monkeypatch, cancel):
    from threading import Event, get_ident
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QPushButton
    from autopyro.ui.dialogs import stratigraphy_binding as module

    dialog = configure(window, qtbot)
    dialog.BACKGROUND_BINDING_ROWS = 1
    project = window.model.project
    gui_thread = get_ident()
    evaluate = module.evaluate_project
    started, resume = Event(), Event()
    inputs = []

    def calculate(snapshot, row_ids, *, scheme, binding, checkpoint):
        assert get_ident() != gui_thread
        assert snapshot is not project
        inputs.extend(column.id for column in snapshot.columns)
        assert all(set(row.values) == set(inputs) for row in snapshot.rows)
        started.set()
        assert resume.wait(10)
        checkpoint()
        return evaluate(snapshot, row_ids, scheme, binding, checkpoint=checkpoint)

    timer = QTimer(dialog)
    timer.setInterval(10)

    def continue_on_gui():
        progress = QApplication.activeModalWidget()
        if not started.is_set() or progress is None:
            return
        timer.stop()
        if cancel:
            progress.findChild(QPushButton).click()
        resume.set()

    monkeypatch.setattr(module, 'evaluate_project', calculate)
    timer.timeout.connect(continue_on_gui)
    timer.start()
    dialog.preview()
    timer.stop()
    assert resume.is_set()
    assert len(inputs) == 2
    assert dialog.results.model().rowCount() == (0 if cancel else 2)
    if cancel:
        assert 'отменена' in dialog.status.text()
    else:
        assert 'Upper' in dialog.results.model().index(0, 5).data()


def test_plot_layer_and_stale_preview(window, qtbot, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from autopyro.ui.dialogs.report import ReportDialog

    dialog = configure(window, qtbot)
    dialog.add_interval(interval={'id': 'B-Formation', 'name': 'B Formation', 'wellbore': 'B',
                                  'level': 'FORMATION', 'top': 20, 'bottom': 30, 'color': '#c7ddeb'})
    dialog.apply()
    dialog.preview()
    results = dialog.results.model()
    assert results.rowCount() == 2
    assert results.index(0, 0).data() == window.model.project.rows[0].id
    assert 'Upper' in results.index(0, 5).data()
    dialog.reference.setText("TVD MSL")
    assert dialog.results.model().rowCount() == 0
    dialog.reference.setText("MD RKB")
    window.show_plots()
    previous = window.plot_window.spec
    previous.curve_archive_id = "previous-xy-fit"
    previous.comparison_preview_panel = "previous-panel"
    previous.annotations = [{"type": "text", "text": "Previous X–Y"}]
    window.sample_selection.select([window.model.project.rows[0].id])
    window.analysis_scope.set_mode("selected")
    dialog.show_plot()
    assert window.plot_window is not None, dialog.status.text()
    assert window.plot_window.spec.kind == "depth"
    assert not window.plot_window.spec.curve_archive_id
    assert window.plot_window.spec.comparison_preview_panel is None
    assert window.plot_window.spec.annotations == []
    assert window.plot_window.spec.stratigraphy["level"] == "FORMATION"
    assert window.plot_window.export_button.isEnabled(), window.plot_window.status.text()
    plot = window.plot_window
    assert window.model.setData(window.model.index(1, 0), 'B')
    selected = window.model.project.rows[1].id
    window.sample_selection.select([selected])
    qtbot.waitUntil(lambda: plot.rendered_ids == [selected], timeout=15000)
    assert plot.spec.title == 'Глубинный профиль'

    profile_path = tmp_path / 'profile.svg'
    manifest_path = tmp_path / 'profile.json'
    plot.start_export([('Profile', profile_path, plot.spec)], manifest_path)
    qtbot.waitUntil(lambda: not plot.export_busy(), timeout=30000)
    assert 'B | FORMATION' in profile_path.read_text(encoding='utf-8')
    assert json.loads(manifest_path.read_text(encoding='utf-8'))['plots'][0]['row_ids'] == [selected]

    window.show_maps()
    map_dialog = window.map_window
    _, md, toc = window.model.project.columns
    map_dialog.data.load({'x_id': md.id, 'y_id': toc.id,
                          'source_crs': 'EPSG:32631', 'target_crs': 'EPSG:32631'})
    map_dialog.redraw()
    assert map_dialog.export_button.isEnabled(), map_dialog.status.text()
    map_path = tmp_path / 'map.json'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(map_path), ''))
    map_dialog.export()
    assert json.loads(map_path.read_text(encoding='utf-8'))['row_ids'] == [selected]

    report = ReportDialog(window)
    qtbot.addWidget(report)
    report.build()
    assert report.report is not None, report.status.text()
    report_path = tmp_path / 'report.json'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(report_path), ''))
    report.export('json')
    assert json.loads(report_path.read_text(encoding='utf-8'))['row_ids'] == [selected]


@pytest.mark.parametrize("interval_mode", [False, True])
def test_plot_template_remaps_layer_columns(window, qtbot, tmp_path, monkeypatch, interval_mode):
    from PySide6.QtWidgets import QFileDialog
    if interval_mode:
        from autopyro.core import Column
        from autopyro.table import AddColumnCommand
        bottom = Column("Bottom", "number", unit="m")
        window.model.undo.push(AddColumnCommand(window.model, bottom, {r.id: val for r,val in zip(window.model.project.rows,[12,28])}))
    dialog = configure(window, qtbot)
    if interval_mode:
        dialog.mode.setCurrentIndex(dialog.mode.findData("interval"))
        dialog.bottom.setCurrentIndex(dialog.bottom.findData(bottom.id))
    dialog.apply()
    dialog.show_plot()
    plot = window.plot_window
    config = plot.spec.to_dict()
    mapping = {c.id: f"external-{i}" for i, c in enumerate(window.model.project.columns)}
    config["x_id"], config["y_id"] = mapping[config["x_id"]], mapping[config["y_id"]]
    if config["depth_bottom_id"]:
        config["depth_bottom_id"] = mapping[config["depth_bottom_id"]]
    for key in ("well_id", "depth_id", "bottom_id"):
        if config["stratigraphy"]["binding"].get(key):
            config["stratigraphy"]["binding"][key] = mapping[config["stratigraphy"]["binding"][key]]
    payload = {"format": "autopyro.plot", "version": 1, "spec": config,
        "columns": [{"id": mapping[c.id], "name": c.name, "unit": c.unit} for c in window.model.project.columns]}
    path = tmp_path / "plot.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(path), ""))
    plot.import_spec()
    plot.redraw()
    assert plot.spec.stratigraphy["binding"]["well_id"] == window.model.project.columns[0].id
    if interval_mode:
        assert plot.spec.depth_bottom_id == bottom.id
    assert plot.export_button.isEnabled(), plot.status.text()


def test_interval_mode_columns_plot_selection_and_reload(window, qtbot, tmp_path):
    from autopyro.core import Column, Project
    from autopyro.table import AddColumnCommand
    bottom = Column("Bottom", "number", unit="m")
    window.model.undo.push(AddColumnCommand(window.model, bottom, {r.id: depth for r,depth in zip(window.model.project.rows,[12,28])}))
    dialog = configure(window,qtbot)
    dialog.mode.setCurrentIndex(dialog.mode.findData("interval"))
    dialog.bottom.setCurrentIndex(dialog.bottom.findData(bottom.id))
    assert dialog.bottom.isEnabled()
    dialog.add_columns()
    project = window.model.project
    name = project.columns[-2]
    assert project.rows[0].values[name.id] == "Upper", dialog.status.text()
    window.model.setData(window.model.index(0,3),"22")
    assert project.rows[0].values[name.id] is None
    assert name.metadata["diagnostics"][project.rows[0].id]["status"] == "cross_boundary"
    dialog.show_plot()
    plot = window.plot_window
    assert plot.spec.depth_bottom_id == bottom.id, dialog.status.text()
    assert plot.export_button.isEnabled(), plot.status.text()
    # Rectangle intersects the segment far from its top; selecting by top or midpoint would miss it.
    ax = plot.result["axes"][0]
    plot.on_rectangle(SimpleNamespace(xdata=.9,ydata=20,inaxes=ax),SimpleNamespace(xdata=1.1,ydata=21,inaxes=ax))
    assert plot.selection_ids() == {project.rows[0].id}
    plot.redraw()  # Selection invalidates the preview until it is rebuilt.
    plot.on_pick(SimpleNamespace(artist=plot.result["artists"][0],ind=[1]))
    assert plot.selection_ids() == {project.rows[1].id}
    project.save(tmp_path/"interval.apyro")
    restored = Project.load(tmp_path/"interval.apyro")
    assert restored.view["stratigraphy"]["binding"]["sample_mode"] == "interval"
    plot.kind.setCurrentIndex(plot.kind.findData("hist"))
    assert plot.depth_bottom.currentData() == ""


def test_lithology_editor_roundtrip(window, qtbot):
    dialog = configure(window, qtbot)
    dialog.intervals.item(0, 7).setText("607")
    scheme = dialog.read_scheme()
    assert scheme["intervals"][0]["lithology"] == "607"
    dialog.draft = scheme
    dialog.load_draft()
    assert dialog.intervals.item(0, 7).text() == "607"
    assert dialog.read_scheme() == scheme


def test_legacy_scheme_keeps_revision_when_lithology_is_empty(window, qtbot):
    dialog = configure(window, qtbot)
    legacy = dialog.read_scheme()
    for interval in legacy["intervals"]:
        interval.pop("lithology", None)
    dialog.draft = deepcopy(legacy)
    dialog.load_draft()
    assert dialog.read_scheme() == legacy



def test_import_and_build_without_samples(window, qtbot, monkeypatch, tmp_path):
    from pathlib import Path
    from PySide6.QtWidgets import QFileDialog, QInputDialog
    from autopyro.core import Project

    window.set_project(Project())
    dialog = StratigraphyDialog(window)
    qtbot.addWidget(dialog)
    dialog.reference.setText("MD RKB")
    path = Path(__file__).resolve().parents[1] / "examples" / "stratigraphy-demo.csv"
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(path), ""))
    monkeypatch.setattr(QInputDialog, "getItem", lambda *args, **kwargs: (";", True))
    dialog.import_scheme()
    assert dialog.column_well.currentData() == "Well A", dialog.status.text()
    assert dialog.well.currentData() is None and dialog.depth.currentData() is None
    dialog.show_column()
    assert dialog.column_toolbar.isEnabled(), dialog.status.text()
    assert dialog.result_tabs.currentWidget() is dialog.column_page
    assert len(dialog.column_figure.axes) == 1
    assert len(dialog.column_figure.axes[0].patches) == 2
    assert dialog.column_figure.axes[0].get_ylim() == (3500, 0)
    assert not window.model.project.view.get("stratigraphy")
    dialog.save_scheme()
    assert window.model.project.view["stratigraphy_scheme"]["source"]
    assert dialog.column_toolbar.isEnabled()  # Saving does not invalidate the independent plot.
    saved = tmp_path / "scheme.apyro"
    window.model.project.save(saved)
    restored = Project.load(saved)
    assert not restored.rows and not restored.columns
    assert restored.view["stratigraphy_scheme"] == dialog.read_scheme()
    window.model.undo.undo()
    assert "stratigraphy_scheme" not in window.model.project.view
    window.model.undo.redo()
    window.set_project(restored)
    reopened = StratigraphyDialog(window)
    qtbot.addWidget(reopened)
    reopened.column_well.setCurrentIndex(reopened.column_well.findData("Well B"))
    reopened.show_column()
    assert reopened.column_toolbar.isEnabled(), reopened.status.text()
    assert len(reopened.column_figure.axes[0].patches) == 1


def test_bind_samples_after_independent_column(window, qtbot):
    dialog = configure(window, qtbot)
    dialog.well.setCurrentIndex(0)
    dialog.depth.setCurrentIndex(0)
    dialog.sample_reference.clear()
    dialog.show_column()
    assert dialog.column_toolbar.isEnabled(), dialog.status.text()
    dialog.save_scheme()
    dialog.well.setCurrentIndex(1)
    dialog.depth.setCurrentIndex(2)
    dialog.sample_reference.setText("MD RKB")
    assert dialog.column_toolbar.isEnabled()  # Sample settings do not invalidate the column.
    dialog.preview()
    assert dialog.results.model().rowCount() == 2, dialog.status.text()
    dialog.add_columns()
    assert window.model.project.rows[0].values[window.model.project.columns[-2].id] == "Upper"
    dialog.show_plot()
    assert window.plot_window.export_button.isEnabled(), window.plot_window.status.text()


@pytest.mark.parametrize('ratio', [1, 2])
def test_column_fits_page_preserving_depth_and_export_style(window, qtbot, ratio):
    dialog = configure(window, qtbot)
    dialog.resize(1350, 900)
    dialog.show()
    dialog.column_style_form.load({'width_mm': 130, 'height_mm': 300, 'dpi': 300})
    style = dialog.style_values()
    dialog.show_column()
    qtbot.waitUntil(lambda: dialog.column_canvas.isVisible())
    canvas = dialog.column_canvas
    canvas._set_device_pixel_ratio(ratio)
    canvas.draw()
    assert canvas.figure.dpi == canvas.figure._original_dpi * ratio
    assert canvas.figure.get_figwidth() * canvas.figure._original_dpi == pytest.approx(canvas.width())
    assert canvas.figure.get_figheight() * canvas.figure._original_dpi == pytest.approx(canvas.height())
    axes = dialog.column_figure.axes[0]
    axes.set_ylim(28, 5)
    size = canvas.size()
    dialog.resize(1100, 720)
    qtbot.waitUntil(lambda: canvas.size() != size)
    canvas.draw()
    assert axes.get_ylim() == (28, 5)
    assert canvas.figure.get_figwidth() * canvas.figure._original_dpi == pytest.approx(canvas.width())
    assert canvas.figure.get_figheight() * canvas.figure._original_dpi == pytest.approx(canvas.height())
    assert dialog.style_values() == style
    dialog.canvas_view.mode_combo.setCurrentIndex(1)
    assert canvas.width() == round(style['width_mm'] / 25.4 * canvas.figure._original_dpi)
    assert canvas.height() == round(style['height_mm'] / 25.4 * canvas.figure._original_dpi)
    qtbot.waitUntil(lambda: dialog.canvas_view.scroll.verticalScrollBar().maximum() > 0)
    dialog.canvas_view.mode_combo.setCurrentIndex(0)
    qtbot.waitUntil(lambda: canvas.size() == dialog.canvas_view.scroll.viewport().size())
    assert axes.get_ylim() == (28, 5)
    assert dialog.style_values() == style


@pytest.mark.parametrize('change', ['boundaries', 'depth_column'])
def test_binding_update_previews_saved_and_proposed_membership(window, qtbot, monkeypatch, change):
    from threading import get_ident
    from PySide6.QtWidgets import QPushButton
    from autopyro.geology import stratigraphy
    from autopyro.ui.dialogs import stratigraphy_binding as binding_module
    from autopyro.core import Column
    from autopyro.table import AddColumnCommand
    from autopyro.ui.widgets.interval_review import IntervalReviewDialog

    dialog = configure(window, qtbot)
    dialog.BACKGROUND_BINDING_ROWS = 1
    evaluated = []
    evaluate = stratigraphy.evaluate_project
    prepare = binding_module.prepare_result_columns
    gui_thread = get_ident()

    def project_columns(*args, **kwargs):
        assert get_ident() != gui_thread
        return prepare(*args, **kwargs)

    def record(project, ids, *args, **kwargs):
        evaluated.append(list(ids))
        return evaluate(project, ids, *args, **kwargs)

    monkeypatch.setattr(stratigraphy, 'evaluate_project', record)
    monkeypatch.setattr(binding_module, 'prepare_result_columns', project_columns)
    dialog.add_columns()
    assert not evaluated
    project = window.model.project
    name, status = project.columns[-2:]
    assert name.metadata['diagnostics'] == status.metadata['diagnostics']
    assert name.metadata['diagnostics'][project.rows[0].id] is not status.metadata['diagnostics'][project.rows[0].id]
    first = project.rows[0]
    result_column = next(column for column in project.columns if first.values.get(column.id) == 'Upper')
    if change == 'boundaries':
        dialog.revision.setText('2')
        dialog.intervals.item(0, 4).setText('9')
        dialog.intervals.item(1, 3).setText('9')
    else:
        alternate = Column('Alternate depth', 'number', unit='m')
        window.model.undo.push(AddColumnCommand(window.model, alternate, {row.id: 25 for row in project.rows}))
        dialog.refresh_columns()
        dialog.depth.setCurrentIndex(dialog.depth.findData(alternate.id))
    evaluated.clear()
    saved = deepcopy(project.view['stratigraphy'])
    undo_index = window.model.undo.index()

    def inspect(widget):
        assert project.view['stratigraphy'] == saved
        assert first.values[result_column.id] == 'Upper'
        rows = widget.table.model().rows
        assert len(rows) == 1 and rows[0][0] == first.id
        assert 'Upper' in rows[0][-2] and 'Lower' in rows[0][-1]
        save = next(button for button in widget.findChildren(QPushButton)
                    if button.text() == 'Пересчитать и сохранить')
        assert not save.icon().isNull()
        if accept:
            save.click()
        return widget.result()

    monkeypatch.setattr(IntervalReviewDialog, 'exec', inspect)
    accept = False
    dialog.apply()
    assert project.view['stratigraphy'] == saved
    assert window.model.undo.index() == undo_index
    accept = True
    dialog.apply()
    assert first.values[result_column.id] == 'Lower'
    assert not evaluated
    window.model.undo.undo()
    assert not evaluated
    assert first.values[result_column.id] == 'Upper'
    assert project.view['stratigraphy'] == saved
    window.model.undo.redo()
    assert not evaluated
    assert first.values[result_column.id] == 'Lower'

    window.model.setData(window.model.index(0, 1), '11')
    window.model.undo.undo()
    evaluated.clear()
    window.model.undo.undo()
    assert evaluated == [[row.id for row in project.rows]]
    assert first.values[result_column.id] == 'Upper'


def test_standalone_save_preserves_existing_binding(window, qtbot, monkeypatch):
    from autopyro.geology import stratigraphy

    dialog = configure(window, qtbot)
    dialog.add_columns()
    project = window.model.project
    settings = deepcopy(project.view["stratigraphy"])
    derived_column = project.columns[-2]
    diagnostics = derived_column.metadata['diagnostics']
    data_revision = window.model.data_revision
    window.dirty = False

    def unexpected_evaluation(*args, **kwargs):
        pytest.fail('Saving the scheme alone should not recalculate sample assignments')

    monkeypatch.setattr(stratigraphy, 'evaluate_project', unexpected_evaluation)
    dialog.intervals.item(0, 1).setText("Revised")
    dialog.revision.setText("2")
    dialog.save_scheme()
    assert window.model.project.view["stratigraphy"] == settings
    assert window.model.project.view["stratigraphy_scheme"]["revision"] == "2"
    assert window.dirty
    assert window.model.data_revision == data_revision
    assert derived_column.metadata['diagnostics'] is diagnostics
    assert 'Сохранённая привязка проб' in dialog.binding_revision.text()
    assert 'редакция 1' in dialog.binding_revision.text()
    assert 'редакция 2' in dialog.binding_revision.text()
    dialog.preview()
    assert 'Результаты предпросмотра рассчитаны по черновику' in dialog.binding_revision.text()
    window.model.undo.undo()
    assert window.model.project.view["stratigraphy_scheme"] == settings["scheme"]
    window.model.undo.redo()
    assert project.view['stratigraphy_scheme']['revision'] == '2'
    assert project.view['stratigraphy'] == settings
    assert window.model.data_revision == data_revision
    assert derived_column.metadata['diagnostics'] is diagnostics


def test_replace_import_restores_invalid_draft_and_guards_project_close(window, qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    dialog = configure(window, qtbot)
    incoming = deepcopy(dialog.read_scheme())
    incoming['intervals'].pop()
    dialog.intervals.item(1, 3).setText('unfinished')
    previous_count = dialog.draft_undo.count()
    dialog.stage_import(incoming, None)
    assert dialog.intervals.rowCount() == 1
    assert dialog.draft_undo.count() == previous_count + 1
    dialog.draft_undo.undo()
    assert dialog.intervals.rowCount() == 2
    assert dialog.intervals.item(1, 3).text() == 'unfinished'
    dialog.draft_undo.undo()
    assert dialog.intervals.item(1, 3).text() == '20'
    window.stratigraphy_window = dialog
    window.dirty = False
    window.show_stratigraphy()
    desktop = window.desktop
    desktop.close_document(desktop.documents.indexOf(dialog))
    window.show_stratigraphy()
    assert desktop.documents.currentWidget() is dialog
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.Cancel)
    assert not MainWindow.may_replace(window)
    assert dialog.intervals.rowCount() == 2


def test_project_close_can_save_stratigraphy_draft(window, qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    dialog = configure(window, qtbot)
    window.stratigraphy_window = dialog
    saved = []
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.Save)
    monkeypatch.setattr(window, 'save', lambda: saved.append(window.model.project.view['stratigraphy_scheme']) or True)
    assert MainWindow.may_replace(window)
    assert saved == [dialog.read_scheme()]
    assert not dialog.has_unsaved_draft()


def test_database_merge_preserves_local_lithology_on_exact_update(window, qtbot, monkeypatch):
    from PySide6.QtWidgets import QCheckBox
    from autopyro.ui.widgets.interval_review import IntervalReviewDialog

    dialog = configure(window, qtbot)
    dialog.intervals.item(0, 7).setText('607')
    old = dialog.read_scheme()
    new = {**old, 'source': {'type': 'postgresql_lithostratigraphy'},
           'intervals': [{**old['intervals'][0], 'name': 'Updated', 'lithology': ''}]}

    def accept(widget):
        widget.findChild(QCheckBox).setChecked(True)
        return widget.DialogCode.Accepted

    monkeypatch.setattr(IntervalReviewDialog, 'exec', accept)
    dialog.stage_import(new, old)
    interval = dialog.read_scheme()['intervals'][0]
    assert (interval['name'], interval['lithology']) == ('Updated', '607')



def test_column_style_save_reload_and_reset(window, qtbot, tmp_path):
    from autopyro.core import Project
    from autopyro.stratigraphy import COLUMN_STYLE

    dialog = configure(window, qtbot)
    scheme = dialog.read_scheme()
    form = dialog.column_style_form
    form.load({"width_mm": 180, "height_mm": 250, "dpi": 150, "title": "Styled column",
               "legend": True, "legend_loc": "bottom", "labels": False,
               "depth_min": 5, "depth_max": 28})
    dialog.show_column()
    assert dialog.column_toolbar.isEnabled(), dialog.status.text()
    assert dialog.column_figure.axes[0].get_ylim() == (28, 5)
    assert dialog.read_scheme() == scheme  # Style does not change scheme geometry or revision.
    dialog.save_scheme()
    assert window.model.project.view["stratigraphy_style"]["title"] == "Styled column"
    path = tmp_path / "styled.apyro"
    window.model.project.save(path)
    window.model.undo.undo()
    assert "stratigraphy_style" not in window.model.project.view
    window.model.undo.redo()
    restored = Project.load(path)
    window.set_project(restored)
    reopened = StratigraphyDialog(window)
    qtbot.addWidget(reopened)
    assert reopened.style_values()["title"] == "Styled column"
    reopened.column_style_form.reset_defaults()
    assert reopened.style_values() == COLUMN_STYLE


def test_invalid_custom_scale_disables_export(window, qtbot):
    dialog = configure(window, qtbot)
    dialog.show_column()
    assert dialog.column_toolbar.isEnabled()
    scale = dialog.column_style_form.scale_editor
    scale.mode.setCurrentIndex(scale.mode.findData('custom'))
    scale.ticks.add_row(values=['Fine', 'not a number'])
    dialog.show_column()
    assert not dialog.column_toolbar.isEnabled()
    assert 'Ошибка' in dialog.status.text()


def test_scheme_export_preserves_previous_json_on_failed_replace(window, qtbot, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from autopyro.data_io import file_output

    dialog = configure(window, qtbot)
    path = tmp_path / 'scheme.json'
    path.write_text('previous', encoding='utf-8')
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), ''))

    def fail(*args):
        raise OSError('simulated write failure')

    with monkeypatch.context() as patch:
        patch.setattr(file_output.os, 'replace', fail)
        dialog.export_scheme()
    assert path.read_text(encoding='utf-8') == 'previous'
    assert list(tmp_path.iterdir()) == [path]
    dialog.export_scheme()
    assert json.loads(path.read_text(encoding='utf-8')) == dialog.read_scheme()


def test_column_export_uses_style_dimensions(window, qtbot, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from PIL import Image

    dialog = configure(window, qtbot)
    dialog.column_style_form.load({"width_mm": 127, "height_mm": 177.8, "dpi": 100,
                                   "transparent": True})
    dialog.show_column()
    assert dialog.export_column_button.isEnabled()
    path = tmp_path / "styled.png"
    monkeypatch.setattr(QFileDialog, 'exec', lambda self: self.DialogCode.Accepted)
    monkeypatch.setattr(QFileDialog, 'selectedFiles', lambda self: [str(path)])
    dialog.export_column()
    assert path.exists(), dialog.status.text()
    with Image.open(path) as image:
        assert image.size == (500, 700)
        assert image.getpixel((0, 0))[3] == 0


def test_single_export_cancel_during_write_preserves_existing_file(window, qtbot, tmp_path, monkeypatch):
    from threading import Event, get_ident
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QFileDialog, QPushButton
    from autopyro.ui.dialogs import stratigraphy_column_export as module

    dialog = configure(window, qtbot)
    path = tmp_path / 'column.svg'
    path.write_text('existing export', encoding='utf-8')
    monkeypatch.setattr(QFileDialog, 'exec', lambda self: self.DialogCode.Accepted)
    monkeypatch.setattr(QFileDialog, 'selectedFiles', lambda self: [str(path)])
    writing, resume = Event(), Event()
    gui_thread = get_ident()

    def render(*args, **kwargs):
        assert get_ident() != gui_thread

        def save(temporary, **options):
            temporary.write_text('new export', encoding='utf-8')
            writing.set()
            assert resume.wait(10)

        return SimpleNamespace(axes=[], savefig=save, clear=lambda: None), []

    timer = QTimer(dialog)
    timer.setInterval(10)

    def cancel():
        progress = QApplication.activeModalWidget()
        if writing.is_set() and progress is not None:
            timer.stop()
            progress.findChild(QPushButton).click()
            resume.set()

    monkeypatch.setattr(module, 'render_columns', render)
    timer.timeout.connect(cancel)
    timer.start()
    dialog.export_column()
    timer.stop()
    assert resume.is_set()
    assert path.read_text(encoding='utf-8') == 'existing export'
    assert list(tmp_path.iterdir()) == [path]
    assert 'отменён' in dialog.status.text()
    assert not dialog._batch_export_running


def test_single_export_can_include_samples_independently_of_preview(window, qtbot, tmp_path, monkeypatch):
    from autopyro.ui.widgets.stratigraphy_export import ColumnExportDialog

    dialog = configure(window, qtbot)
    assert not dialog.sample_track.isChecked()
    path = tmp_path / 'samples.svg'

    def choose(picker):
        assert not picker.include_samples.isChecked()
        picker.include_samples.setChecked(True)
        picker.selectNameFilter('SVG (*.svg)')
        picker.filterSelected.emit('SVG (*.svg)')
        picker.selectFile(str(path.with_suffix('')))
        return picker.DialogCode.Accepted

    monkeypatch.setattr(ColumnExportDialog, 'exec', choose)
    dialog.export_column()
    assert 'Пробы' in path.read_text(encoding='utf-8')
    assert not dialog.sample_track.isChecked()

    def use_saved_option(picker):
        assert picker.include_samples.isChecked()
        picker.selectFile(str(path))
        return picker.DialogCode.Accepted

    monkeypatch.setattr(ColumnExportDialog, 'exec', use_saved_option)
    dialog.depth.setCurrentIndex(0)
    path.unlink()
    dialog.export_column()
    assert not path.exists()
    assert 'Ошибка' in dialog.status.text()


def test_batch_exports_separate_columns_and_comparison_with_one_template(window, qtbot, tmp_path, monkeypatch):
    from autopyro.ui.dialogs import stratigraphy_column_export as module
    from PIL import Image
    from PySide6.QtWidgets import QFileDialog
    from autopyro.ui.widgets.stratigraphy_export import ColumnBatchDialog
    from autopyro.ui.widgets.stratigraphy_templates import TEMPLATES_KEY

    dialog = configure(window, qtbot)
    dialog.add_interval(interval={'id': 'other', 'wellbore': 'B/side', 'level': 'FORMATION',
                                  'name': 'Layer', 'top': 10, 'bottom': 50})
    window.model.project.view[TEMPLATES_KEY] = {
        'Print': {'width_mm': 76.2, 'height_mm': 101.6, 'dpi': 100, 'transparent': True}
    }
    original = dialog.style_values()
    scheme = dialog.read_scheme()
    sample_tracks = []
    render = module.render_columns

    def record(*args, **kwargs):
        figure, notices = render(*args, **kwargs)
        sample_tracks.append([axis._autopyro_well for axis in figure.axes
                              if axis._autopyro_kind == 'samples'])
        return figure, notices

    monkeypatch.setattr(module, 'render_columns', record)

    def choose(widget):
        assert not widget.comparison.isEnabled()
        widget.list.item(1).setCheckState(Qt.CheckState.Checked)
        assert widget.includes_comparison()
        widget.template.setCurrentIndex(widget.template.findData('Print'))
        widget.include_samples.setChecked(True)
        return widget.DialogCode.Accepted

    monkeypatch.setattr(ColumnBatchDialog, 'exec', choose)
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *args: str(tmp_path))
    dialog.batch_export_button.click()
    folders = list(tmp_path.iterdir())
    assert len(folders) == 1, dialog.status.text()
    images = sorted(folders[0].glob('*.png'))
    assert [path.name for path in images] == ['001-A.png', '002-B_side.png', 'comparison.png']
    assert sample_tracks == [['A'], [], ['A']]
    for path in images:
        with Image.open(path) as image:
            assert image.size == (300, 400)
            assert image.getpixel((0, 0))[3] == 0
    assert dialog.style_values() == original
    assert dialog.read_scheme() == scheme


def test_batch_keeps_completed_files_when_cancelled(window, qtbot, tmp_path, monkeypatch):
    from threading import Event, get_ident
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QPushButton
    from autopyro.ui.dialogs import stratigraphy_column_export as module

    dialog = configure(window, qtbot)
    scheme = dialog.read_scheme()
    scheme['intervals'].append({**scheme['intervals'][0], 'wellbore': 'B', 'id': 'other'})
    gui_thread = get_ident()
    rendering = []
    writing_second = Event()
    continue_writing = Event()

    def render(subset, well, level, *, options, samples):
        assert get_ident() != gui_thread
        assert {item['wellbore'] for item in subset['intervals']} == {well}
        rendering.append(well)

        def save(path, **kwargs):
            path.write_text('saved', encoding='utf-8')
            if well == 'B':
                writing_second.set()
                assert continue_writing.wait(10)

        return SimpleNamespace(axes=[], savefig=save, clear=lambda: None), []

    timer = QTimer(dialog)
    timer.setInterval(10)

    def cancel_during_write():
        progress = QApplication.activeModalWidget()
        if not writing_second.is_set() or progress is None:
            return
        timer.stop()
        progress.findChild(QPushButton).click()
        continue_writing.set()

    monkeypatch.setattr(module, 'render_columns', render)
    timer.timeout.connect(cancel_during_write)
    timer.start()
    dialog.write_column_batch(tmp_path, 'svg', scheme, 'FORMATION', dialog.style_values(),
                              ['A', 'B'], individual=True, comparison=True)
    timer.stop()
    assert continue_writing.is_set()
    assert rendering == ['A', 'B']
    assert [path.name for path in tmp_path.glob('*/*')] == ['001-A.svg']
    assert 'Экспорт отменён' in dialog.status.text()
    assert '1 из 3' in dialog.status.text()



def test_choose_lithology_updates_scheme_and_cancel_preserves_it(window, qtbot, monkeypatch):
    from autopyro.ui.widgets.lithology import LithologyPicker

    dialog = configure(window, qtbot)
    dialog.intervals.setCurrentCell(0, 7)
    before = dialog.read_scheme()
    def accept(picker):
        picker.search.setText('607')
        picker.list.setCurrentIndex(picker.proxy.index(0, 0))
        picker.accept()
        return picker.result()
    monkeypatch.setattr(LithologyPicker, 'exec', accept)
    dialog.choose_lithology()
    assert dialog.intervals.item(0, 7).text() == '607'
    assert dialog.read_scheme()['intervals'][0]['lithology'] == '607'
    assert dialog.read_scheme()['intervals'][0]['top'] == before['intervals'][0]['top']
    assert dialog.read_scheme()['revision'] == before['revision']
    dialog.show_column()
    assert dialog.column_toolbar.isEnabled(), dialog.status.text()
    selected = dialog.read_scheme()
    monkeypatch.setattr(LithologyPicker, 'exec', lambda picker: picker.DialogCode.Rejected)
    dialog.choose_lithology()
    assert dialog.read_scheme() == selected


def test_double_click_opens_lithology_picker(window, qtbot, monkeypatch):
    from autopyro.ui.widgets.lithology import LithologyPicker

    dialog = configure(window, qtbot)
    dialog.show()
    dialog.intervals.setCurrentCell(0, 7)
    dialog.intervals.scrollToItem(dialog.intervals.item(0, 7))
    opened = []
    def reject(picker):
        opened.append(True)
        return picker.DialogCode.Rejected
    monkeypatch.setattr(LithologyPicker, 'exec', reject)
    rect = dialog.intervals.visualItemRect(dialog.intervals.item(0, 7))
    qtbot.mouseClick(dialog.intervals.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    qtbot.mouseDClick(dialog.intervals.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    assert opened



@pytest.mark.parametrize("code", ["607", ""])
def test_assign_lithology_to_selected_intervals_once(window, qtbot, monkeypatch, code):
    from PySide6.QtWidgets import QTableWidgetSelectionRange
    from autopyro.ui.widgets.lithology import LithologyPicker

    dialog = configure(window, qtbot)
    dialog.add_interval(interval={"id": "third", "name": "Unselected", "wellbore": "A",
                                  "level": "FORMATION", "top": 30, "bottom": 40,
                                  "color": "#c7ddeb", "lithology": "608"})
    dialog.intervals.item(0, 7).setText("609")
    dialog.intervals.item(1, 7).setText("610")
    # Two overlapping ranges in the same rows, selecting different columns.
    dialog.intervals.setCurrentCell(2, 7)
    dialog.intervals.clearSelection()
    dialog.intervals.setRangeSelected(QTableWidgetSelectionRange(0, 0, 1, 2), True)
    dialog.intervals.setRangeSelected(QTableWidgetSelectionRange(0, 5, 1, 7), True)
    before = dialog.read_scheme()
    calls = []
    invalidations = []
    monkeypatch.setattr(dialog, "invalidate_column", lambda: invalidations.append(True))
    def accept(picker):
        calls.append(picker.windowTitle())
        picker.selected_code = code
        return picker.DialogCode.Accepted
    monkeypatch.setattr(LithologyPicker, "exec", accept)
    dialog.choose_lithology()
    after = dialog.read_scheme()
    assert len(calls) == 1 and "2" in calls[0]
    assert len(invalidations) == 1
    assert [item.get("lithology", "") for item in after["intervals"]] == [code, code, "608"]
    for original, updated in zip(before["intervals"], after["intervals"]):
        assert {k: v for k, v in original.items() if k != "lithology"} == {
            k: v for k, v in updated.items() if k != "lithology"}
    dialog.choose_lithology()  # Assigning the same value does not invalidate again.
    assert len(invalidations) == 1


def test_batch_lithology_cancel_and_no_selection(window, qtbot, monkeypatch):
    from autopyro.ui.widgets.lithology import LithologyPicker

    dialog = configure(window, qtbot)
    dialog.intervals.selectAll()
    before = dialog.read_scheme()
    calls = []
    def reject(picker):
        calls.append(True)
        picker.selected_code = "607"
        return picker.DialogCode.Rejected
    monkeypatch.setattr(LithologyPicker, "exec", reject)
    dialog.choose_lithology()
    assert calls == [True]
    assert dialog.read_scheme() == before
    dialog.intervals.clearSelection()  # A stale current row must not be modified.
    dialog.choose_lithology()
    assert calls == [True]
    assert dialog.read_scheme() == before



def test_stratigraphy_workflow_has_separate_steps_and_localized_modes(window, qtbot):
    from PySide6.QtWidgets import QLineEdit
    dialog = configure(window, qtbot)
    assert [dialog.parameter_tabs.tabText(i) for i in range(3)] == [
        '1. Интервалы', '2. Колонка', '3. Замеры']
    assert dialog.level.currentData() == 'FORMATION'
    assert dialog.level.currentText() != 'FORMATION'
    form = dialog.column_style_form
    for key in ('display_mode', 'feature_mode', 'grain_preset'):
        combo = form.fields[key][0]
        assert all(combo.itemText(i) != combo.itemData(i) for i in range(combo.count()))
    assert not any(key in form.fields for key in ('legend_kwargs', 'legend_titles',
                                                 'x_ticks_dict', 'grain_brackets', 'lithologies'))
    assert all(dialog.intervals.isColumnHidden(i) for i in range(8, 16))
    dialog.geological_columns.setChecked(True)
    assert not any(dialog.intervals.isColumnHidden(i) for i in range(8, 16))
    assert not form.fields['grain_preset'][0].isEnabled()
    form.fields['display_mode'][0].setCurrentIndex(form.fields['display_mode'][0].findData('default'))
    assert form.fields['grain_preset'][0].isEnabled()
    assert form.fields['x_axis'][0].isChecked()
    assert dialog.style_values()['display_mode'] == 'default'
    dialog.show_column()
    assert dialog.export_column_button.isEnabled(), dialog.status.text()



def test_column_style_changes_are_debounced_and_preserve_zoom(window, qtbot, monkeypatch):
    from autopyro.ui.dialogs import stratigraphy_column as module

    dialog = configure(window, qtbot)
    dialog.show()
    dialog.show_column()
    qtbot.wait(100)  # Drain the initial Qt resize/paint events before timing edits.
    dialog.column_preview_timer.setInterval(150)
    dialog.column_figure.axes[0].set_ylim(27, 12)
    calls = []
    render = module.render_columns
    def record(*args, **kwargs):
        calls.append(kwargs['options']['label_size'])
        return render(*args, **kwargs)
    monkeypatch.setattr(module, 'render_columns', record)
    spin = dialog.column_style_form.fields['label_size'][0]
    for value in (11, 12, 13):
        spin.setValue(value)
        qtbot.wait(15)
    assert not calls
    assert not dialog.export_column_button.isEnabled()
    assert dialog.column_figure.axes  # Keep the last picture while the timer is pending.
    qtbot.waitUntil(lambda: dialog.export_column_button.isEnabled(), timeout=3000)
    assert calls == [13]
    assert dialog.column_figure.axes[0].get_ylim() == (27, 12)
    assert dialog.parameter_tabs.currentIndex() == 1


@pytest.mark.parametrize('background_render', [False, True])
def test_column_selection_repaints_only_highlights(window, qtbot, monkeypatch, background_render):
    from PySide6.QtWidgets import QApplication

    dialog = configure(window, qtbot)
    if background_render:
        dialog.BACKGROUND_PREVIEW_INTERVALS = 1
    dialog.show()
    dialog.show_column()
    QApplication.processEvents()
    canvas = dialog.column_canvas
    original = bytes(canvas.buffer_rgba())

    def unexpected_draw():
        pytest.fail('Changing interval selection should not redraw the column')

    with monkeypatch.context() as patch:
        patch.setattr(canvas, 'draw_idle', unexpected_draw)
        dialog.interval_selection.select_ids({'Upper'})
        assert bytes(canvas.buffer_rgba()) != original
        dialog.intervals.clearSelection()
        assert bytes(canvas.buffer_rgba()) == original

    dialog.interval_selection.zoom_to_depth(5, 25)
    canvas.draw()
    zoomed = bytes(canvas.buffer_rgba())
    assert zoomed != original
    with monkeypatch.context() as patch:
        patch.setattr(canvas, 'draw_idle', unexpected_draw)
        dialog.interval_selection.select_ids({'Lower'})
        assert bytes(canvas.buffer_rgba()) != zoomed
        dialog.intervals.clearSelection()
        assert bytes(canvas.buffer_rgba()) == zoomed


def test_large_column_batches_navigation_and_resize_redraws(window, qtbot, monkeypatch):
    from threading import get_ident

    dialog = configure(window, qtbot)
    dialog.BACKGROUND_PREVIEW_INTERVALS = 1
    dialog.show()
    dialog.show_column()
    qtbot.wait(180)
    canvas = dialog.column_canvas
    draws = []
    scene = dialog.column_navigation.scene
    original_draw = scene.draw
    gui_thread = get_ident()

    def record_draw():
        draws.append((get_ident(), scene.figure.axes[0].get_ylim()))
        original_draw()

    monkeypatch.setattr(scene, 'draw', record_draw)
    for index in range(5):
        dialog.resize(1100 + index * 10, 820)
        dialog.interval_selection.zoom_to_depth(5 + index, 25)
        qtbot.wait(12)
    assert not draws
    click = SimpleNamespace(button=1, ydata=10, inaxes=canvas.figure.axes[0], key=None)
    dialog.interval_selection.on_click(click)
    assert not dialog.interval_selection.selected_ids()
    qtbot.waitUntil(lambda: len(draws) == 1 and not canvas.navigation_draw_pending)
    qtbot.wait(140)
    assert len(draws) == 1
    thread, limits = draws[0]
    assert thread != gui_thread
    assert limits == pytest.approx((25.64, 8.36))
    dialog.interval_selection.on_click(click)
    assert dialog.interval_selection.selected_ids() == {'Upper'}


@pytest.mark.parametrize('new_message', [None, 'New operation completed'])
def test_large_column_navigation_can_retry_after_worker_failure(window, qtbot, monkeypatch, new_message):
    dialog = configure(window, qtbot)
    dialog.BACKGROUND_PREVIEW_INTERVALS = 1
    dialog.show()
    dialog.show_column()
    canvas = dialog.column_canvas
    navigation = dialog.column_navigation
    qtbot.waitUntil(lambda: not canvas.navigation_draw_pending)
    scene = navigation.scene
    draw_scene = scene.draw
    original_pixels = bytes(canvas.buffer_rgba())
    presented = []
    present = canvas.present
    previous_status = dialog.status.text()

    def record_present(pixels):
        presented.append(True)
        return present(pixels)

    def fail():
        raise ValueError('Navigation render failed')

    def unexpected_gui_draw():
        pytest.fail('A worker failure should not trigger a full GUI redraw')

    monkeypatch.setattr(scene, 'draw', fail)
    monkeypatch.setattr(canvas, 'draw', unexpected_gui_draw)
    monkeypatch.setattr(canvas, 'present', record_present)
    dialog.interval_selection.zoom_to_depth(5, 25)
    qtbot.waitUntil(lambda: 'Navigation render failed' in dialog.status.text())
    qtbot.waitUntil(lambda: navigation.worker is None)
    assert not presented
    assert canvas.navigation_draw_pending
    assert navigation.scene is scene
    if new_message:
        dialog.status.setText(new_message)

    monkeypatch.setattr(scene, 'draw', draw_scene)
    dialog.interval_selection.zoom_to_depth(10, 20)
    qtbot.waitUntil(lambda: not canvas.navigation_draw_pending)
    assert bytes(canvas.buffer_rgba()) != original_pixels
    assert canvas.figure.axes[0].get_ylim() == pytest.approx((20.4, 9.6))
    assert dialog.status.text() == (new_message or previous_status)


def test_large_column_navigation_discards_stale_frame_and_keeps_history(window, qtbot, monkeypatch):
    from threading import Event

    dialog = configure(window, qtbot)
    dialog.BACKGROUND_PREVIEW_INTERVALS = 1
    dialog.show()
    dialog.show_column()
    canvas = dialog.column_canvas
    qtbot.wait(180)
    qtbot.waitUntil(lambda: not canvas.navigation_draw_pending)
    scene = dialog.column_navigation.scene
    started, release = Event(), Event()
    original_draw, original_present = scene.draw, canvas.present
    frames = []

    def slow_first_frame():
        if not started.is_set():
            started.set()
            assert release.wait(10)
        original_draw()

    def record_frame(pixels):
        frames.append(canvas.figure.axes[0].get_ylim())
        original_present(pixels)

    monkeypatch.setattr(scene, 'draw', slow_first_frame)
    monkeypatch.setattr(canvas, 'present', record_frame)
    dialog.interval_selection.zoom_to_depth(5, 25)
    first_view = canvas.figure.axes[0].get_ylim()
    try:
        qtbot.waitUntil(started.is_set)
        dialog.interval_selection.zoom_to_depth(10, 20)
        last_view = canvas.figure.axes[0].get_ylim()
    finally:
        release.set()
    qtbot.waitUntil(lambda: not canvas.navigation_draw_pending)
    assert frames == [last_view]
    dialog.column_toolbar.back()
    qtbot.waitUntil(lambda: not canvas.navigation_draw_pending)
    assert frames == [last_view, first_view]


@pytest.mark.parametrize('change', ['style', 'draft', 'scope'])
def test_large_column_never_presents_frame_superseded_during_render(window, qtbot, monkeypatch, change):
    from threading import Event
    from PySide6.QtCore import QTimer
    from autopyro.ui.dialogs import stratigraphy_column as module

    dialog = configure(window, qtbot)
    dialog.BACKGROUND_PREVIEW_INTERVALS = 1
    dialog.column_preview_timer.setInterval(40)
    dialog.sample_track.setChecked(change == 'scope')
    dialog.show()
    started, release = Event(), Event()
    original_render = module.render_columns
    original_present = dialog.column_canvas.present
    requests, frames = [], []

    def render(scheme, *args, options, samples, **kwargs):
        requests.append((scheme['intervals'][0]['name'], options['label_size'], sum(map(len, samples.values()))))
        if len(requests) == 1:
            started.set()
            assert release.wait(10)
        return original_render(scheme, *args, options=options, samples=samples, **kwargs)

    def present(pixels):
        frames.append(requests[-1])
        original_present(pixels)

    timer = QTimer(dialog)
    timer.setInterval(10)

    def edit_during_render():
        if not started.is_set():
            return
        timer.stop()
        try:
            if change == 'style':
                dialog.column_style_form.fields['label_size'][0].setValue(14)
            elif change == 'draft':
                dialog.intervals.item(0, 1).setText('Updated layer')
            else:
                window.analysis_scope.set_mode('selected')
        finally:
            release.set()

    monkeypatch.setattr(module, 'render_columns', render)
    monkeypatch.setattr(dialog.column_canvas, 'present', present)
    timer.timeout.connect(edit_during_render)
    timer.start()
    dialog.show_column()
    timer.stop()
    assert release.is_set()
    assert not frames
    assert not dialog.export_column_button.isEnabled()
    if change == 'draft':
        assert not dialog.column_preview_timer.isActive()
        dialog.show_column()
        assert frames[0][0] == 'Updated layer'
    elif change == 'style':
        qtbot.waitUntil(lambda: dialog.export_column_button.isEnabled())
        assert frames[0] == ('Upper', 14, 0)
    else:
        qtbot.waitUntil(lambda: dialog.export_column_button.isEnabled())
        assert requests[0][2] == 2
        assert frames[0][2] == 0
    assert len(requests) == 2


def test_resizing_during_large_build_requests_matching_background_frame(window, qtbot, monkeypatch):
    from threading import Event
    from PySide6.QtCore import QTimer
    from autopyro.ui.dialogs import stratigraphy_column as module

    dialog = configure(window, qtbot)
    dialog.BACKGROUND_PREVIEW_INTERVALS = 1
    dialog.show()
    canvas = dialog.column_canvas
    started, release = Event(), Event()
    render_columns = module.render_columns
    original_present, original_draw = canvas.present, canvas.draw
    frames, gui_draws = [], []

    def render(*args, **kwargs):
        started.set()
        assert release.wait(10)
        return render_columns(*args, **kwargs)

    def present(pixels):
        frames.append(pixels.shape)
        original_present(pixels)

    def draw():
        if dialog.column_navigation.scene is not None:
            gui_draws.append(True)
        original_draw()

    timer = QTimer(dialog)
    timer.setInterval(10)

    def resize_during_render():
        if started.is_set():
            timer.stop()
            dialog.resize(dialog.width() + 200, dialog.height() + 100)
            QTimer.singleShot(0, release.set)

    monkeypatch.setattr(module, 'render_columns', render)
    monkeypatch.setattr(canvas, 'present', present)
    monkeypatch.setattr(canvas, 'draw', draw)
    timer.timeout.connect(resize_during_render)
    timer.start()
    dialog.show_column()
    timer.stop()
    assert release.is_set()
    assert canvas.navigation_draw_pending
    qtbot.waitUntil(lambda: not canvas.navigation_draw_pending)
    assert len(frames) >= 2 and frames[0] != frames[-1]
    assert canvas.buffer_rgba().shape == frames[-1]
    assert not gui_draws


def test_large_column_preview_renders_off_gui_thread_and_keeps_picking(window, qtbot, monkeypatch):
    from threading import Event, get_ident
    from matplotlib.backend_bases import MouseEvent
    from PySide6.QtCore import QTimer
    from autopyro.ui.dialogs import stratigraphy_column as module

    dialog = configure(window, qtbot)
    dialog.show()
    dialog.BACKGROUND_PREVIEW_INTERVALS = 1
    gui_thread = get_ident()
    rendering, ui_processed = Event(), Event()
    render_columns = module.render_columns

    def render(*args, **kwargs):
        assert get_ident() != gui_thread
        rendering.set()
        assert ui_processed.wait(10)
        return render_columns(*args, **kwargs)

    timer = QTimer(dialog)
    timer.setInterval(10)
    timer.timeout.connect(lambda: ui_processed.set() if rendering.is_set() else None)
    timer.start()
    monkeypatch.setattr(module, 'render_columns', render)
    dialog.show_column()
    timer.stop()
    assert dialog.export_column_button.isEnabled(), dialog.status.text()
    assert ui_processed.is_set()
    canvas = dialog.column_canvas
    axis = dialog.column_figure.axes[0]
    x, y = axis.transData.transform((.5, 10))
    event = MouseEvent('button_press_event', canvas, x, y, button=1)
    canvas.callbacks.process(event.name, event)
    assert dialog.interval_selection.selected_ids() == {'Upper'}
    dialog.intervals.clearSelection()
    axis.set_ylim(24, 5)
    dialog.render_column_preview(automatic=True)
    assert dialog.column_figure.axes[0].get_ylim() == (24, 5)


def test_auto_preview_defers_hidden_tab_and_resumes_on_return(window, qtbot, monkeypatch):
    from autopyro.ui.dialogs import stratigraphy_column as module

    dialog = configure(window, qtbot)
    dialog.show()
    dialog.show_column()
    selection = dialog.interval_selection
    depth_indexes = selection.depth_indexes
    visibility_updates = []
    set_row_hidden = selection.view.setRowHidden

    def record_visibility(row, hidden):
        visibility_updates.append((row, hidden))
        set_row_hidden(row, hidden)

    monkeypatch.setattr(selection.view, 'setRowHidden', record_visibility)
    reads = []
    read_scheme = dialog.read_scheme

    def record_read():
        reads.append(True)
        return read_scheme()

    monkeypatch.setattr(dialog, 'read_scheme', record_read)
    dialog.column_preview_timer.setInterval(40)
    calls = []
    render = module.render_columns
    def record(*args, **kwargs):
        calls.append(True)
        return render(*args, **kwargs)
    monkeypatch.setattr(module, 'render_columns', record)
    dialog.parameter_tabs.setCurrentIndex(0)
    dialog.column_style_form.fields['label_size'][0].setValue(12)
    qtbot.wait(100)
    assert not calls
    assert dialog.parameter_tabs.currentIndex() == 0
    dialog.parameter_tabs.setCurrentIndex(1)
    qtbot.waitUntil(lambda: bool(calls), timeout=3000)
    assert calls == [True]
    dialog.column_style_form.fields['label_size'][0].setValue(13)
    dialog.hide()
    qtbot.wait(100)
    assert calls == [True]
    dialog.show()
    qtbot.waitUntil(lambda: len(calls) == 2, timeout=3000)
    assert not reads
    assert not visibility_updates
    assert selection.depth_indexes is depth_indexes
    dialog.intervals.item(0, 1).setText('Updated layer')
    dialog.show_column()
    assert reads == [True]
    assert visibility_updates
    assert selection.depth_indexes is not depth_indexes
    assert dialog.interval_selection.rendered['Upper']['name'] == 'Updated layer'


def test_auto_preview_waits_for_valid_input_and_manual_build_cancels_timer(window, qtbot, monkeypatch):
    from autopyro.ui.dialogs import stratigraphy_column as module

    dialog = configure(window, qtbot)
    dialog.show()
    dialog.show_column()
    dialog.column_preview_timer.setInterval(50)
    old_axis = dialog.column_figure.axes[0]
    scale = dialog.column_style_form.scale_editor
    calls = []
    render = module.render_columns
    def record(*args, **kwargs):
        calls.append(True)
        return render(*args, **kwargs)
    monkeypatch.setattr(module, 'render_columns', record)
    scale.mode.setCurrentIndex(scale.mode.findData('custom'))
    qtbot.wait(150)
    assert not calls  # Empty custom ticks are validated before calling stratapy.
    assert dialog.column_figure.axes[0] is old_axis
    assert not dialog.export_column_button.isEnabled()
    scale.mode.setCurrentIndex(scale.mode.findData('preset'))
    dialog.show_column()
    qtbot.wait(150)
    assert calls == [True]  # No delayed duplicate of the manual render.
    assert dialog.export_column_button.isEnabled()


def test_schema_edit_cancels_pending_style_preview(window, qtbot, monkeypatch):
    from autopyro.ui.dialogs import stratigraphy_column as module

    dialog = configure(window, qtbot)
    dialog.show()
    dialog.column_preview_timer.setInterval(40)
    calls = []
    render = module.render_columns
    def record(*args, **kwargs):
        calls.append(True)
        return render(*args, **kwargs)
    monkeypatch.setattr(module, 'render_columns', record)
    dialog.column_style_form.fields['label_size'][0].setValue(11)
    qtbot.wait(100)
    assert not calls  # The first preview still requires an explicit build.
    dialog.show_column()
    dialog.column_style_form.fields['label_size'][0].setValue(12)
    dialog.intervals.item(0, 1).setText('Edited layer')
    qtbot.wait(100)
    assert calls == [True]
    assert not dialog.column_preview_timer.isActive()
    assert not dialog.export_column_button.isEnabled()


def test_hierarchy_style_auto_preview_save_and_restore(window, qtbot):
    dialog = configure(window, qtbot)
    dialog.add_interval(interval={'id': 'parent', 'name': 'Parent', 'wellbore': 'A',
                                 'level': 'GROUP', 'top': -10, 'bottom': 40, 'color': '#aabbcc'})
    dialog.show()
    dialog.show_column()
    dialog.column_figure.axes[0].set_ylim(25, 10)
    dialog.column_preview_timer.setInterval(40)
    dialog.column_style_form.hierarchy_checks['GROUP'].setChecked(True)
    assert not dialog.export_column_button.isEnabled()
    qtbot.waitUntil(lambda: dialog.export_column_button.isEnabled(), timeout=3000)
    assert [axis._autopyro_level for axis in dialog.column_figure.axes] == ['GROUP', 'FORMATION']
    assert all(axis.get_ylim() == (40, -10) for axis in dialog.column_figure.axes)
    dialog.save_scheme()
    saved = window.model.project.view['stratigraphy_style']
    assert saved['hierarchy_levels'] == ['GROUP']
    restored = StratigraphyDialog(window)
    qtbot.addWidget(restored)
    assert restored.column_style_form.hierarchy_checks['GROUP'].isChecked()
    restored.show_column()
    assert [axis._autopyro_level for axis in restored.column_figure.axes] == ['GROUP', 'FORMATION']
    assert not restored.zoom_intervals_action.icon().isNull()


def test_comparison_picker_updates_preview_and_persists(window, qtbot, monkeypatch):
    from autopyro.ui.widgets.stratigraphy_labels import CheckedValuesDialog

    dialog = configure(window, qtbot)
    dialog.add_interval(interval={'id': 'B', 'name': 'B layer', 'wellbore': 'B',
                                 'level': 'FORMATION', 'top': 50, 'bottom': 80, 'color': '#aabbcc'})
    dialog.show()
    dialog.show_column()
    row = dialog.interval_history.rows_by_id['B']
    assert dialog.column_intervals.isRowHidden(row)
    dialog.column_preview_timer.setInterval(40)
    dialog.column_figure.axes[0].set_ylim(25, 10)
    form = dialog.column_style_form
    assert form.available_wells == ['A', 'B']
    assert not form.comparison_button.icon().isNull()
    def choose(picker):
        for row in range(picker.list.count()):
            item = picker.list.item(row)
            item.setCheckState(Qt.CheckState.Checked if item.text() == 'B' else Qt.CheckState.Unchecked)
        return picker.DialogCode.Accepted
    monkeypatch.setattr(CheckedValuesDialog, 'exec', choose)
    form.comparison_button.click()
    assert form.values()['comparison_wells'] == ['B']
    assert not dialog.export_column_button.isEnabled()
    qtbot.waitUntil(lambda: dialog.export_column_button.isEnabled(), timeout=3000)
    assert [axis._autopyro_well for axis in dialog.column_figure.axes] == ['A', 'B']
    assert not dialog.column_intervals.isRowHidden(row)
    assert all(axis.get_ylim() == (80, 0) for axis in dialog.column_figure.axes)
    dialog.column_figure.axes[0].set_ylim(60, 20)
    form.fields['label_size'][0].setValue(12)
    qtbot.waitUntil(lambda: dialog.export_column_button.isEnabled(), timeout=3000)
    assert all(axis.get_ylim() == (60, 20) for axis in dialog.column_figure.axes)
    dialog.save_scheme()
    restored = StratigraphyDialog(window)
    qtbot.addWidget(restored)
    assert restored.style_values()['comparison_wells'] == ['B']
    restored.show_column()
    assert [axis._autopyro_well for axis in restored.column_figure.axes] == ['A', 'B']
    form.reset_defaults()
    qtbot.waitUntil(lambda: dialog.export_column_button.isEnabled(), timeout=3000)
    assert [axis._autopyro_well for axis in dialog.column_figure.axes] == ['A']
    assert dialog.column_intervals.isRowHidden(row)
    assert dialog.column_intervals.isColumnHidden(0)
