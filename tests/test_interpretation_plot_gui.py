"""Embedded interpretation chart, shared controls and document lifetime."""
import pytest
from PySide6.QtCore import Qt

from autopyro.ui.window import MainWindow
from autopyro.standard_interpretation import standard_schemes


@pytest.fixture
def document(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    window.may_replace = lambda: True
    window.show_interpretation()
    dialog = window.interpretation_window
    dialog.commit_graphical_scheme(standard_schemes()['S2–TOC'], auto_bind=True)
    yield window, dialog
    dialog.close()
    if window.plot_window:
        window.plot_window.close()
    window.dirty = False


def ready(qtbot, dialog, *, timeout=20000):
    qtbot.waitUntil(lambda: dialog.report is not None and dialog.plot.result is not None
                   and not dialog.plot.timer.isActive(), timeout=timeout)


def test_switching_catalog_scheme_protects_unapplied_draft(document, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    _, dialog = document
    original = dialog.scheme
    draft = '{"incomplete":'
    dialog.editor.setPlainText(draft)
    dialog.standard_choice.setCurrentIndex(dialog.standard_choice.findData('HI–TOC'))
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.Cancel)
    dialog.load_standard_scheme()
    assert dialog.editor.toPlainText() == draft and dialog.scheme is original

    path = tmp_path / 'draft.json'
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.Save)
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), ''))
    dialog.load_standard_scheme()
    assert path.read_text(encoding='utf-8') == draft
    assert dialog.scheme['name'] == 'HI–TOC'


def test_chart_is_embedded_and_binding_is_fixed(document, qtbot):
    window, dialog = document
    ready(qtbot, dialog)
    plot = dialog.plot
    assert window.plot_window is None
    assert dialog.result_splitter.widget(0) is plot
    assert dialog.result_splitter.widget(1) is dialog.table
    assert dialog.isAncestorOf(plot.canvas)
    assert dialog.parameter_tabs.isAncestorOf(plot.style)
    assert not plot.kind.isVisibleTo(plot.data)
    assert plot.spec.kind == 'scatter'
    assert plot.spec.x_id == dialog.x.currentData()
    assert plot.spec.interpretation['scheme'] == dialog.scheme
    assert plot.rendered_ids == window.row_ids()
    # Named configurations cannot change the interpretation chart's type/binding.
    plot.data.load({'kind': 'hist', 'x_id': '', 'scope': 'selected'})
    plot.read_controls()
    assert plot.spec.kind == 'scatter'
    assert plot.spec.x_id == dialog.x.currentData()
    assert plot.spec.scope == dialog.scope.currentData()
    dialog.editor.insertPlainText(' ')
    assert plot.result is None
    assert not plot.export_button.isEnabled()


def test_svg_scheme_autoscales_and_can_show_source_frame(document, qtbot):
    _, dialog = document
    original = dialog.scheme['id']
    dialog.scheme_group.setCurrentIndex(dialog.scheme_group.findData('TOCo'))
    jarvie = next(name for name in dialog.standard_schemes if name.startswith('TOCo · Jarvie'))
    dialog.standard_choice.setCurrentIndex(dialog.standard_choice.findData(jarvie))
    assert dialog.scheme['id'] == original
    preview = dialog.catalog_preview.text()
    assert 'Справочная диаграмма' in preview
    assert 'Kerogen Yield [mg HC/g rock] — Колонка показателя отсутствует' in preview

    dialog.scheme_group.setCurrentIndex(dialog.scheme_group.findData('S2–TOC'))
    name = next(name for name in dialog.standard_schemes if name.startswith('S2–TOC · Edress'))
    dialog.standard_choice.setCurrentIndex(dialog.standard_choice.findData(name))
    assert dialog.scheme['id'] == original
    preview = dialog.catalog_preview.text()
    assert 'X · TOC [%wt] — TOC' in preview
    assert 'Y · S2 [mg HC/g rock] — S2' in preview
    dialog.apply_catalog_scheme.click()
    ready(qtbot, dialog, timeout=40000)
    assert dialog.report['scheme']['name'] == name
    axis, = dialog.plot.result['axes']
    assert axis.get_xscale() == axis.get_yscale() == 'log'
    assert 0 < axis.get_xlim()[0] < axis.get_xlim()[1]
    assert 0 < axis.get_ylim()[0] < axis.get_ylim()[1]
    assert axis.get_xlim() != pytest.approx((.1, 100))
    assert 'Ожидает независимого решения' in dialog.scheme_info.text()
    dialog.plot.layout_form.fields['interpretation_source_frame'][0].setChecked(True)
    qtbot.waitUntil(lambda: dialog.plot.result is not None
                   and dialog.plot.result['axes'][0].get_xlim() == pytest.approx((.1, 100)), timeout=30000)
    axis, = dialog.plot.result['axes']
    assert axis.get_xlim() == pytest.approx((.1, 100))
    assert axis.get_ylim() == pytest.approx((.1, 100))
    dialog.comparison.setChecked(True)
    qtbot.waitUntil(lambda: dialog.plot.result is not None and len(dialog.plot.result['axes']) == 4, timeout=40000)
    axis = dialog.plot.result['axes'][0]
    assert len(dialog.plot.result['axes']) == 4
    assert axis.get_xscale() == axis.get_yscale() == 'log'
    assert axis.get_xlim() == pytest.approx((.1, 100))
    assert 'Согласие стандартных HI–Tmax и HI–OI' in dialog.table.model().headers
    dialog.select_result_rows([dialog.report['results'][0]['row_id']])
    assert 'Согласие стандартных HI–Tmax и HI–OI' in dialog.explanation.toPlainText()


def test_compact_layout_keeps_chart_and_sample_conclusion_visible(document, qtbot):
    window, dialog = document
    window.resize(1024, 700)
    window.show()
    dialog.text_numbers.setChecked(True)
    ready(qtbot, dialog)
    plot = dialog.plot
    assert plot.canvas_scroll.viewport().height() >= 178
    assert plot.export_button in plot.toolbar.actions()
    assert plot.export_button in plot.plot_actions_menu.actions()
    assert plot.plot_actions_button.isVisible()
    assert plot.brush.isVisible()
    assert not plot.reset_defaults_button.isVisible()
    assert any('Автопределы' in action.text() for action in plot.plot_actions_menu.actions())

    row = window.model.project.rows[0]
    dialog.select_result_rows([row.id])
    index = dialog.table.selectionModel().selectedRows()[0]
    assert index.data() == 'A · 1000'
    assert index.data(Qt.ItemDataRole.UserRole) == row.id
    assert row.id in index.data(Qt.ItemDataRole.ToolTipRole)
    lines = dialog.explanation.toPlainText().splitlines()
    assert lines[0] == 'Проба: A · 1000'
    assert lines[1] == 'Результат: В области'
    assert lines[2].startswith('Области: TOC:')
    assert any(line.startswith('UUID пробы:') for line in lines[3:])


def test_appearance_is_shared_while_scheme_axis_limits_are_local(document, qtbot):
    window, dialog = document
    ready(qtbot, dialog)
    window.show_plots()
    graph = window.plot_window
    graph.redraw()
    original = graph.spec.to_dict()
    dialog.plot.style.load({'marker_size': 12, 'alpha': .5})
    dialog.plot.x_form.load({'minimum': 0, 'maximum': 20})
    dialog.plot.redraw()
    updated = graph.read_controls().to_dict()
    assert updated['marker_size'] == 12 and updated['alpha'] == .5
    assert updated['kind'] == original['kind'] and updated['x_axis'] == original['x_axis']
    graph.style.load({'marker_size': 9})
    graph.redraw()
    assert dialog.plot.style.values()['marker_size'] == 9
    assert dialog.plot.x_form.values()['maximum'] == 20
    assert 'interpretation_plot_current' in window.model.project.view
    assert window.model.project.view['plot_shared_appearance']['style']['marker_size'] == 9
    generation = dialog.plot.generation
    window.desktop.documents.setCurrentWidget(dialog)
    assert dialog.plot.generation > generation
    ready(qtbot, dialog)


def test_reopening_restarts_preview_and_keeps_draft(document, qtbot):
    window, dialog = document
    ready(qtbot, dialog)
    window.desktop.close_document(window.desktop.documents.indexOf(dialog))
    assert dialog.plot.closed
    window.show_interpretation()
    ready(qtbot, dialog)
    assert not dialog.plot.closed
    assert window.plot_window is None


def test_scope_and_numeric_policy_match_interpretation(document, qtbot):
    window, dialog = document
    ready(qtbot, dialog)
    dialog.text_numbers.setChecked(True)
    ready(qtbot, dialog)
    assert any(r['status'] == 'classified' for r in dialog.report['results'])
    assert dialog.plot.spec.text_numbers
    dialog.scope.setCurrentIndex(dialog.scope.findData('selected'))
    window.table.selectRow(0)
    ready(qtbot, dialog)
    assert set(dialog.plot.rendered_ids) == {r['row_id'] for r in dialog.report['results']}
    assert len(dialog.plot.rendered_ids) == 1


def test_point_and_hidden_region_tooltips(document, qtbot, monkeypatch):
    from matplotlib.backend_bases import MouseEvent
    from PySide6.QtWidgets import QToolTip
    from autopyro.interpretation import STATUSES
    from autopyro.ui.i18n import tr

    _, dialog = document
    dialog.text_numbers.setChecked(True)
    ready(qtbot, dialog)
    plot = dialog.plot
    plot.layout_form.load({'interpretation_labels': False})
    plot.schedule()
    ready(qtbot, dialog)
    shown = []
    monkeypatch.setattr(QToolTip, 'showText', lambda *args: shown.append(args[1]))
    artist = plot.result['artists'][0]
    position = artist.get_offset_transform().transform(artist.get_offsets()[0])
    event = MouseEvent('motion_notify_event', plot.canvas, *position)
    plot.canvas.callbacks.process('motion_notify_event', event)
    qtbot.waitUntil(lambda: bool(shown), timeout=2000)
    row_id = artist._autopyro_row_ids[0]
    record = next(r for r in dialog.report['results'] if r['row_id'] == row_id)
    assert row_id in shown[-1]
    assert 'TOC' in shown[-1] and 'S2' in shown[-1]
    assert tr(STATUSES[record['status']]) in shown[-1]
    assert 'Области под указателем:' in shown[-1]
    shown.clear()
    position = plot.figure.axes[0].transData.transform((1.5,4))
    event = MouseEvent('motion_notify_event', plot.canvas, *position)
    plot.canvas.callbacks.process('motion_notify_event', event)
    qtbot.waitUntil(lambda: bool(shown), timeout=2000)
    assert 'TOC: хороший; S2: удовлетворительный' in shown[-1]
    plot.schedule()
    assert plot.hover_event is None and not plot.hover_timer.isActive()


def test_graph_pick_and_result_row_selection_are_bidirectional(document, qtbot):
    from types import SimpleNamespace

    window, dialog = document
    dialog.text_numbers.setChecked(True)
    ready(qtbot, dialog)
    plot = dialog.plot
    artist = next(a for a in plot.result['artists'] if len(a._autopyro_row_ids) >= 2)
    first, second = artist._autopyro_row_ids[:2]
    source_selection = plot.owner.table.selectionModel().selection()
    generation = plot.generation
    plot.on_pick(SimpleNamespace(artist=artist, ind=[0]))
    selected = dialog.table.selectionModel().selectedRows()
    assert [i.data(Qt.ItemDataRole.UserRole) for i in selected] == [first]
    assert plot.selection_overlay.selected_ids == {first}
    dialog.table.selectRow(dialog.result_rows[second])
    assert plot.selection_overlay.selected_ids == {second}
    assert plot.selection_ids() == {second}
    assert plot.generation == generation  # Selection changes do not rerender the graph.
    assert window.table.selectionModel().selection() == source_selection
    dialog.table.selectionModel().clearSelection()
    assert plot.selection_overlay.selected_ids == set()


def test_result_multiselection_survives_recalculation(document, qtbot):
    from PySide6.QtCore import QItemSelection, QItemSelectionModel

    _, dialog = document
    dialog.text_numbers.setChecked(True)
    ready(qtbot, dialog)
    plotted = [rid for a in dialog.plot.result['artists'] for rid in a._autopyro_row_ids]
    wanted = set(plotted[:2])
    model = dialog.table.model()
    selection = QItemSelection()
    for rid in wanted:
        index = model.index(dialog.result_rows[rid], 0)
        selection.select(index, index)
    flags = QItemSelectionModel.SelectionFlag
    dialog.table.selectionModel().select(selection, flags.ClearAndSelect | flags.Rows)
    assert dialog.plot.selection_overlay.selected_ids == wanted
    dialog.schedule()
    ready(qtbot, dialog)
    assert {i.data(Qt.ItemDataRole.UserRole) for i in dialog.table.selectionModel().selectedRows()} == wanted
    assert dialog.plot.selection_overlay.selected_ids == wanted
    # A row with no plotted numeric point stays selectable without a fake highlight.
    missing = next(r['row_id'] for r in dialog.report['results'] if r['status'] == 'insufficient_data')
    dialog.table.selectRow(dialog.result_rows[missing])
    assert dialog.result_selection == {missing}
    assert not dialog.plot.selection_overlay.selected_ids


def test_result_selection_does_not_shrink_selected_source_scope(document, qtbot):
    from PySide6.QtCore import QItemSelection, QItemSelectionModel
    from autopyro.ui.table.selection import selected_row_ids

    window, dialog = document
    dialog.text_numbers.setChecked(True)
    selection = QItemSelection(window.proxy.index(0,0), window.proxy.index(3,0))
    flags = QItemSelectionModel.SelectionFlag
    window.table.selectionModel().select(selection, flags.ClearAndSelect | flags.Rows)
    dialog.scope.setCurrentIndex(dialog.scope.findData('selected'))
    ready(qtbot, dialog)
    source_ids = selected_row_ids(window)
    assert set(dialog.plot.rendered_ids) == source_ids
    rid = dialog.plot.result['artists'][0]._autopyro_row_ids[0]
    dialog.plot.select_rows([rid])
    assert dialog.result_selection == {rid}
    assert selected_row_ids(window) == source_ids
    dialog.schedule()
    ready(qtbot, dialog)
    assert set(dialog.plot.rendered_ids) == source_ids
    assert dialog.plot.selection_overlay.selected_ids == {rid}
    # Filtering/recalculating away a row removes it from result selection.
    dialog.scope.setCurrentIndex(dialog.scope.findData('filtered'))
    well = window.model.project.columns[0].id
    window.apply_filter([{'column': well, 'operator': 'eq', 'value': 'missing', 'negate': False}], 'AND')
    ready(qtbot, dialog)
    assert not dialog.result_selection
    assert not dialog.table.selectionModel().selectedRows()


def test_graph_and_results_stay_visible_with_adjustable_divider(document, qtbot, monkeypatch, tmp_path):
    from PySide6.QtCore import QByteArray, QSettings, Qt
    from autopyro.ui.dialogs import interpretation_chart

    store = QSettings(str(tmp_path / 'layout.ini'), QSettings.Format.IniFormat)
    monkeypatch.setattr(interpretation_chart, 'settings', lambda: store)

    window, dialog = document
    window.resize(1450, 1000)
    window.show()
    ready(qtbot, dialog)
    splitter = dialog.result_splitter
    assert splitter.orientation() == Qt.Orientation.Vertical
    assert not splitter.childrenCollapsible()
    assert splitter.handle(1).isVisible()
    assert dialog.plot.isVisible() and dialog.table.isVisible()
    before = splitter.sizes()
    splitter.moveSplitter(before[0] - 70, 1)
    after = splitter.sizes()
    assert after[0] < before[0] and after[1] > before[1]
    assert min(after) > 0
    assert dialog.plot.isVisible() and dialog.table.isVisible()
    assert store.value(interpretation_chart.SPLITTER_SETTING, type=QByteArray) == splitter.saveState()


def test_quality_and_explanation_follow_selected_sample(document, qtbot):
    _, dialog = document
    dialog.text_numbers.setChecked(True)
    ready(qtbot, dialog)
    rid = dialog.report['results'][0]['row_id']
    generation = dialog.plot.generation
    dialog.plot.select_rows([rid])
    text = dialog.explanation.toPlainText()
    assert rid in text and dialog.scheme['name'] in text
    assert 'Контроль качества' in text and 'Результат:' in text
    assert dialog.plot.generation == generation
    assert dialog.table.model().columnCount() == 9
    dialog.schedule()
    assert 'не актуален' in dialog.explanation.toPlainText()
    ready(qtbot, dialog)
    assert rid in dialog.explanation.toPlainText()
