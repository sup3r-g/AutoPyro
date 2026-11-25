"""Comparison previews retain UUID selection, hover context and the ordinary view."""
from types import SimpleNamespace

import pytest

from autopyro.ui.window import MainWindow
from test_interpretation_comparison import data
from autopyro.standard_interpretation import standard_schemes


@pytest.fixture
def comparison(qtbot):
    project, _, _ = data()
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.may_replace = lambda: True
    window.show_interpretation()
    dialog = window.interpretation_window
    dialog.commit_graphical_scheme(standard_schemes()['S2–TOC'], auto_bind=True)
    dialog.comparison.setChecked(True)
    yield window, dialog
    dialog.close()
    window.dirty = False


def ready(qtbot, dialog, count):
    qtbot.waitUntil(lambda: dialog.report is not None and dialog.plot.result is not None
                   and len(dialog.plot.result['axes']) == count and not dialog.plot.timer.isActive(), timeout=30000)


def test_catalog_requires_explicit_binding_for_column_without_unit(qtbot):
    project, _, _ = data()
    toc = next(column for column in project.columns if column.name == 'TOC')
    toc.unit = ''
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.show_interpretation()
    dialog = window.interpretation_window
    dialog.commit_graphical_scheme(standard_schemes()['S2–TOC'], auto_bind=True)
    assert not dialog.x.currentData()
    assert not dialog.x_unit.text()
    assert dialog.y.currentData()
    assert 'укажите единицу вручную' in dialog.catalog_preview.text()

    dialog.x.setCurrentIndex(dialog.x.findData(toc.id))
    dialog.x_unit.setText('%')
    qtbot.waitUntil(lambda: dialog.report is not None, timeout=30000)
    assert dialog.report['binding']['x_id'] == toc.id
    assert dialog.report['binding']['x_unit'] == '%'
    dialog.close()
    window.dirty = False


def test_changing_comparison_mode_invalidates_saved_result_before_recalculation(comparison, qtbot):
    window, dialog = comparison
    ready(qtbot, dialog, 3)
    view = window.model.project.view
    saved = view['interpretation_result']
    dialog.pin_base.setChecked(True)
    pinned = view['interpretation_comparison_base']

    dialog.comparison.setChecked(False)

    assert saved['stale']
    assert saved['stale_reason'] == {'code': 'binding_changed', 'fields': ['comparison']}
    assert not pinned['stale']
    ready(qtbot, dialog, 1)
    assert not view['interpretation_result']['stale']
    assert not view['interpretation_result']['comparison_panels']
    assert view['interpretation_comparison_base'] is pinned


def test_background_calculation_discards_cancelled_result(comparison, qtbot, monkeypatch):
    from threading import Event, current_thread, main_thread

    _, dialog = comparison
    ready(qtbot, dialog, 3)
    selected = dialog.report['results'][0]['row_id']
    dialog.select_result_rows([selected])
    start = Event()
    release = Event()
    calculate = dialog.calculate_report
    calls = []

    def delayed(project, request, checkpoint):
        calls.append(current_thread())
        if len(calls) == 1:
            start.set()
            # Model a computation finishing just after the user cancels it.
            assert release.wait(8)
            return calculate(project, request)
        return calculate(project, request, checkpoint)

    monkeypatch.setattr(dialog, 'calculate_report', delayed)
    try:
        dialog.schedule()
        assert dialog.table.model().rowCount() == 0
        assert not dialog.export_csv.isEnabled()
        qtbot.waitUntil(start.is_set)
        assert calls[0] is not main_thread()
        dialog.cancel_calculation_button.click()
        release.set()
        qtbot.waitUntil(lambda: dialog.calculation_worker is None)
        assert dialog.report is None
        assert dialog.table.model().rowCount() == 0
        assert 'отменён' in dialog.explanation.toPlainText()
        assert 'отменён' in dialog.status.text()
        assert not dialog.previous_result_button.isHidden()
        assert not dialog.export_json.isEnabled()
        dialog.schedule()
        ready(qtbot, dialog, 3)
        assert len(calls) == 2
        assert dialog.result_selection == {selected}
    finally:
        release.set()


def test_unchanged_recalculation_keeps_project_clean_and_failure_clears_rows(comparison, qtbot, tmp_path):
    window, dialog = comparison
    ready(qtbot, dialog, 3)
    project = window.model.project
    stored = project.view['interpretation_result']
    previous = project.view.get('interpretation_previous_result')
    path = tmp_path / 'unchanged.apyro'
    project.save(path)
    reopened = type(project).load(path)
    assert dialog.report_storage(reopened.view['interpretation_result'], dialog.report) == (None, None)
    revision = window.analysis_revision
    window.dirty = False

    dialog.schedule()
    assert dialog.table.model().rowCount() == 0
    ready(qtbot, dialog, 3)
    assert not window.dirty and window.analysis_revision == revision
    assert project.view['interpretation_result'] is stored
    assert project.view.get('interpretation_previous_result') is previous

    dialog.y_unit.setText('wrong unit')
    dialog.schedule()
    qtbot.waitUntil(lambda: 'не выполнена' in dialog.status.text(), timeout=10000)
    assert dialog.report is None and dialog.table.model().rowCount() == 0
    assert not dialog.export_json.isEnabled()
    assert 'не выполнена' in dialog.explanation.toPlainText()


@pytest.mark.parametrize('phase', ['input', 'table'])
def test_cancelling_chunked_preparation_does_not_install_report(comparison, qtbot, monkeypatch, phase):
    from autopyro.ui.dialogs import interpretation_calculation as module

    window, dialog = comparison
    ready(qtbot, dialog, 3)
    project = window.model.project
    saved = project.view['interpretation_result']
    entered = []

    def pause(*_args):
        entered.append(True)
        while True:
            yield

    if phase == 'input':
        monkeypatch.setattr(module, 'capture_interpretation_steps', pause)
    else:
        monkeypatch.setattr(dialog, 'prepare_report_steps', pause)
    dialog.schedule()
    qtbot.waitUntil(lambda: bool(entered))
    assert dialog.report is None
    assert project.view['interpretation_result'] is saved
    dialog.cancel_calculation_button.click()
    assert dialog.calculation_steps is None
    assert not dialog.calculation_steps_timer.isActive()
    assert dialog.report is None
    assert project.view['interpretation_result'] is saved


def test_comparison_preview_selection_and_explanation(comparison, qtbot):
    window, dialog = comparison
    ready(qtbot, dialog, 3)
    assert len(dialog.comparison_reports) == 3
    assert dialog.standard_choice.currentData() == 'S2–TOC'
    assert dialog.table.model().columnCount() == 13
    plot = dialog.plot
    for artist in plot.result['artists']:
        rid = artist._autopyro_row_ids[0]
        plot.on_pick(SimpleNamespace(artist=artist, ind=[0]))
        assert dialog.result_selection == {rid}
        assert all(len(layer.artist.get_offsets()) == 1 for layer in plot.selection_overlay.layers)
        assert {i.data() for i in dialog.table.selectionModel().selectedRows()} == {rid}
    assert 'Сравнение диаграмм' in dialog.explanation.toPlainText()
    assert 'HI–OI' in dialog.explanation.toPlainText()
    assert 'HI–Tmax' in dialog.explanation.toPlainText()
    assert len(dialog.report['comparisons']) == 3
    import json
    assert 'HI–OI' in json.dumps(dialog.report, ensure_ascii=False)
    assert window.plot_window is None


def test_single_diagram_preview_keeps_selection_and_full_export(comparison, qtbot):
    window, dialog = comparison
    ready(qtbot, dialog, 3)
    plot = dialog.plot
    report = dialog.report
    row_id = window.model.project.rows[0].id
    dialog.select_result_rows([row_id])
    dialog.comparison_view.setCurrentIndex(dialog.comparison_view.findData('HI–OI'))
    ready(qtbot, dialog, 1)
    assert plot.figure.axes[0].get_title() == 'HI–OI'
    assert plot.selection_overlay.selected_ids == {row_id}
    assert {index.data() for index in dialog.table.selectionModel().selectedRows()} == {row_id}
    assert dialog.report is report
    assert len(plot.spec.interpretation_panels) == 3
    assert plot.spec.comparison_preview_panel is None
    dialog.comparison_view.setCurrentIndex(dialog.comparison_view.findData('all'))
    ready(qtbot, dialog, 3)
    assert plot.selection_overlay.selected_ids == {row_id}


def test_save_and_open_next_expert_review(comparison, qtbot, monkeypatch):
    from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLineEdit
    from autopyro.schemes.interpretation_review import save_review

    window, dialog = comparison
    ready(qtbot, dialog, 3)
    for position in range(2):
        save_review(window.model.project, dialog.report, position, 'open', author='Reviewer')
    dialog.calculate()
    dialog.result_filter.setCurrentIndex(dialog.result_filter.findData('review_open'))
    first, second = [row[0] for row in dialog.table.model().rows]
    visited = []

    def inspect(editor):
        visited.append(next(iter(dialog.result_selection)))
        if len(visited) == 1:
            editor.findChild(QLineEdit, 'review_author').setText('Reviewer')
            next_button = next(button for button in editor.findChild(QDialogButtonBox).buttons()
                               if 'следующую пробу' in button.text())
            next_button.click()
        return editor.result()

    monkeypatch.setattr(QDialog, 'exec', inspect)
    dialog.select_result_rows([first])
    dialog.edit_review()
    qtbot.waitUntil(lambda: visited == [first, second])


def test_result_table_sort_and_compact_view_preserve_sample_selection(comparison, qtbot):
    from PySide6.QtCore import Qt

    _, dialog = comparison
    ready(qtbot, dialog, 3)
    selected = dialog.report['results'][0]['row_id']
    dialog.select_result_rows([selected])
    dialog.table.horizontalHeader().setSortIndicator(0, Qt.SortOrder.DescendingOrder)
    model = dialog.table.model()
    labels = [model.index(i, 0).data() for i in range(model.rowCount())]
    assert labels == sorted(labels, key=str.casefold, reverse=True)
    assert dialog.table.selectionModel().selectedRows()[0].data(Qt.ItemDataRole.UserRole) == selected
    dialog.compact_results.setChecked(True)
    assert dialog.table.isColumnHidden(5)
    assert not dialog.table.isColumnHidden(1)


def test_comparison_grid_control_reflows_panels(comparison, qtbot):
    _, dialog = comparison
    dialog.comparison_view.setCurrentIndex(dialog.comparison_view.findData('all'))
    ready(qtbot, dialog, 3)
    plot = dialog.plot
    control = plot.layout_form.fields['comparison_grid'][0]
    assert not control.isHidden()
    control.setChecked(True)
    ready(qtbot, dialog, 3)
    qtbot.waitUntil(lambda: plot.figure.axes[2].get_position().y0 < plot.figure.axes[0].get_position().y0)
    first, second, third = [ax.get_position() for ax in plot.figure.axes]
    assert first.x0 == pytest.approx(third.x0)
    assert first.y0 == pytest.approx(second.y0)
    assert plot.spec.to_dict()['comparison_grid'] is True


def test_compact_interpretation_fits_with_both_docks_open(qtbot):
    from pathlib import Path

    from matplotlib import get_data_path
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    previous_font = app.font()
    QFontDatabase.addApplicationFont(str(Path(get_data_path()) / 'fonts' / 'ttf' / 'DejaVuSans.ttf'))
    app.setFont(QFont('DejaVu Sans', 9))
    try:
        project, _, _ = data()
        window = MainWindow()
        qtbot.addWidget(window)
        window.set_project(project)
        window.may_replace = lambda: True
        window.resize(1024, 768)
        window.show()
        window.show_interpretation()
        dialog = window.interpretation_window
        dialog.commit_graphical_scheme(standard_schemes()['S2–TOC'], auto_bind=True)
        dialog.comparison.setChecked(True)
        ready(qtbot, dialog, 1)
        assert (window.width(), window.height()) == (1024, 768)
        assert window.desktop.project_dock.isVisible()
        assert window.desktop.properties_dock.isVisible()
        assert dialog.table.isVisible() and dialog.explanation.isVisible()
        row_id = window.model.project.rows[0].id
        dialog.plot.select_rows([row_id])
        assert dialog.result_selection == {row_id}
        assert dialog.plot.selection_overlay.selected_ids == {row_id}
        assert 'Таблица · S2–TOC' in dialog.comparison_context.text()
        assert 'График · HI–Tmax' in dialog.comparison_context.text()
        assert dialog.table.model().headers[1] == 'Статус · S2–TOC'
        dialog.close()
        window.dirty = False
    finally:
        app.setFont(previous_font)


def test_combined_filters_keep_graph_and_table_on_the_same_probe(comparison, qtbot):
    window, dialog = comparison
    project = window.model.project
    first, second = project.rows
    oi = next(column for column in project.columns if column.name == 'OI')
    first.values[oi.id] = '4'
    dialog.comparison.setChecked(False)
    ready(qtbot, dialog, 1)
    assert dialog.review_states == ['pending', 'pending']

    dialog.result_filter.setCurrentIndex(dialog.result_filter.findData('boundary'))
    assert dialog.table.model().rowCount() == 2
    dialog.extra_filter_actions['quality'].trigger()
    ready(qtbot, dialog, 1)
    assert dialog.visible_positions == [1]
    assert dialog.plot.rendered_ids == [second.id]
    dialog.plot.select_rows([second.id])
    assert dialog.table.selectionModel().selectedRows()[0].data() == second.id
    dialog.table.clearSelection()
    dialog.table.selectRow(0)
    assert dialog.plot.selection_overlay.selected_ids == {second.id}
    explanation = dialog.explanation.toPlainText()
    assert 'Номинальное измерение на границе' in explanation
    assert 'С предупреждениями контроля качества» — S2–TOC' in explanation

    dialog.result_filter.setCurrentIndex(dialog.result_filter.findData('outside'))
    ready(qtbot, dialog, 1)
    assert not dialog.plot.rendered_ids
    assert dialog.table.model().rowCount() == 0
    assert not dialog.plot.selection_overlay.selected_ids


def test_expert_review_survives_project_save_and_returns_to_queue(comparison, qtbot, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QLineEdit, QPlainTextEdit, QTextBrowser

    from autopyro.schemes.interpretation_review import review_history, review_state

    window, dialog = comparison
    project = window.model.project
    second = project.rows[1]
    oi_column = next(i for i, c in enumerate(project.columns) if c.name == 'OI')
    assert window.model.setData(window.model.index(1, oi_column), '200')
    ready(qtbot, dialog, 3)
    dialog.result_filter.setCurrentIndex(dialog.result_filter.findData('review'))
    assert second.id in dialog.result_rows
    assert dialog.table.model().rows[dialog.result_rows[second.id]][-1] == 'Требует проверки'
    other_pending = set(dialog.result_rows) - {second.id}
    pending_count = len(other_pending)
    assert f'новые: {pending_count + 1} · открытые: 0 · повторные: 0' in dialog.review_queue_count.text()
    dialog.select_result_rows([second.id])
    assert 'Расхождение оценок' in dialog.explanation.toPlainText()

    def enter_decision(editor):
        status = editor.findChild(QComboBox)
        status.setCurrentIndex(status.findData('expert'))
        save = editor.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Save)
        assert not save.isEnabled()
        editor.findChild(QLineEdit, 'expert_conclusion').setText('Смешанный тип')
        editor.findChild(QPlainTextEdit).setPlainText('Обе оценки учитываются')
        assert not save.isEnabled()
        editor.findChild(QLineEdit, 'review_author').setText('Первый эксперт')
        assert save.isEnabled()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, 'exec', enter_decision)
    dialog.review_button.click()
    assert set(dialog.result_rows) == other_pending
    assert f'новые: {pending_count} · открытые: 0 · повторные: 0' in dialog.review_queue_count.text()
    ready(qtbot, dialog, 3)
    assert dialog.result_selection == other_pending
    assert dialog.plot.selection_overlay.selected_ids == other_pending
    assert dialog.report['type_agreement'] == ['equal', 'different']

    path = tmp_path / 'review.apyro'
    project.save(path)
    reopened = type(project).load(path)
    state, decision = review_state(reopened, dialog.report, 1)
    assert state == 'expert'
    assert decision['conclusion'] == 'Смешанный тип'
    assert decision['note'] == 'Обе оценки учитываются'
    assert decision['author'] == 'Первый эксперт'

    assert window.model.setData(window.model.index(1, oi_column), '20')
    ready(qtbot, dialog, 3)
    assert set(dialog.result_rows) == other_pending | {second.id}
    assert review_state(project, dialog.report, 0)[1] is None
    assert review_state(project, dialog.report, 1) == ('stale', decision)
    assert f'новые: {pending_count} · открытые: 0 · повторные: 1' in dialog.review_queue_count.text()
    assert 'требуется повторная проверка' in dialog.table.model().rows[dialog.result_rows[second.id]][-1]
    dialog.select_result_rows([second.id])
    assert 'Предыдущее решение' in dialog.explanation.toPlainText()
    assert 'Причина повторной проверки — HI–OI:' in dialog.explanation.toPlainText()
    assert 'измерения или их неопределённость' in dialog.explanation.toPlainText()
    assert dialog.plot.selection_overlay.selected_ids == {second.id}

    def leave_open(editor):
        status = editor.findChild(QComboBox)
        status.setCurrentIndex(status.findData('open'))
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, 'exec', leave_open)
    dialog.review_button.click()
    assert dialog.result_selection == {second.id}
    assert f'новые: {pending_count} · открытые: 1 · повторные: 0' in dialog.review_queue_count.text()

    def revise_decision(editor):
        history = editor.findChild(QTextBrowser).toPlainText()
        assert 'Первый эксперт' in history and 'Смешанный тип' in history
        status = editor.findChild(QComboBox)
        status.setCurrentIndex(status.findData('accepted'))
        editor.findChild(QLineEdit, 'review_author').setText('Второй эксперт')
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, 'exec', revise_decision)
    dialog.review_button.click()
    project.save(path)
    reopened = type(project).load(path)
    entries = review_history(reopened, dialog.report['scheme']['id'], second.id)
    assert [(entry['author'], entry['status']) for entry in entries] == [
        ('Первый эксперт', 'expert'), ('Первый эксперт', 'open'), ('Второй эксперт', 'accepted')]
    assert entries[0] == decision
    assert entries[-1]['scheme_revision'] == dialog.report['scheme']['revision']


def test_well_summary_keeps_crossing_samples_separate_and_selects_them(qtbot, monkeypatch, tmp_path):
    import csv
    from types import SimpleNamespace

    from PySide6.QtCore import QLocale
    from PySide6.QtWidgets import QFileDialog

    from autopyro.data_io.io import build_project
    from autopyro.ui.widgets.interpretation_summary import InterpretationSummaryDialog, SummaryBindingDialog

    names = ['Ствол', 'Кровля пробы', 'Подошва пробы', 'TOC', 'S2']
    units = ['', 'ft', 'ft', '%wt', 'mg HC/g rock']
    project = build_project(names, [['A', '10', '15', '1', '3'], ['A', '60', '80', '101', '3']],
                            [1, 2], [(i, name, 'number' if i else 'text', unit)
                                     for i, (name, unit) in enumerate(zip(names, units, strict=True))], {})
    well, top, bottom = [column.id for column in project.columns[:3]]
    scheme = {
        'format': 'autopyro.stratigraphy', 'version': 1, 'id': 'test', 'revision': '1',
        'name': 'Test', 'depth_reference': 'MD', 'depth_unit': 'm',
        'intervals': [
            {'id': 'upper', 'wellbore': 'A', 'level': 'FORMATION', 'name': 'Upper', 'top': 0, 'bottom': 20},
            {'id': 'lower', 'wellbore': 'A', 'level': 'FORMATION', 'name': 'Lower', 'top': 20, 'bottom': 40},
        ],
    }
    binding = {'well_id': well, 'depth_id': top, 'bottom_id': bottom, 'depth_reference': 'MD',
               'depth_unit': 'ft', 'sample_mode': 'interval'}
    project.view['stratigraphy_scheme'] = scheme
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.show_interpretation()
    dialog = window.interpretation_window
    dialog.commit_graphical_scheme(standard_schemes()['S2–TOC'], auto_bind=True)
    ready(qtbot, dialog, 1)
    inspected = []
    modes = iter(['interval', 'point', 'interval'])

    def configure(editor):
        for key, column_id in [('well_id', well), ('depth_id', top), ('bottom_id', bottom)]:
            combo = editor.fields[key]
            combo.setCurrentIndex(combo.findData(column_id))
        editor.mode.setCurrentIndex(editor.mode.findData(next(modes)))
        editor.unit.setCurrentText('ft')
        editor.reference.setText('TVD')
        editor.apply_binding()
        assert editor.binding is None and editor.error.text()
        editor.reference.setText('MD')
        editor.apply_binding()
        return editor.result()

    monkeypatch.setattr(SummaryBindingDialog, 'exec', configure)
    csv_path = tmp_path / 'summary.csv'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(csv_path), ''))

    def inspect_summary(summary):
        summary.change_binding()
        assert summary.binding.get('sample_mode', 'point') == 'point'
        assert summary.model.rows[1][:2] == ('Upper [0, 20) m', 2)
        summary.change_binding()
        summary.canvas.draw()
        assert [row[1] for row in summary.model.rows] == [2, 1, 1]
        assert summary.model.rows[1][0] == 'Upper [0, 20) m'
        assert summary.model.rows[2][0] == 'Пересекает границу подразделений'
        assert summary.model.rows[2][4] == '1 (100.0%)'
        summary.export_button.click()
        decimal = QLocale.system().decimalPoint()
        with csv_path.open(encoding='utf-8-sig', newline='') as stream:
            exported = list(csv.DictReader(stream, delimiter=';' if decimal == ',' else ','))
        assert [row['sample_count'] for row in exported] == ['2', '1', '1']
        assert exported[2]['interval_or_status'] == 'Пересекает границу подразделений'
        assert exported[2]['excluded_count'] == '1'
        assert float(exported[2]['excluded_percent'].replace(decimal, '.')) == 100
        assert exported[2]['wellbore'] == 'A' and exported[2]['level'] == 'FORMATION'
        axis = summary.figure.axes[0]
        assert axis.yaxis_inverted()
        assert axis.collections[1].get_segments()[0][:, 1] == pytest.approx([18.288, 24.384])
        assert [track.get_title() for track in summary.figure.axes] == ['Статус', 'Тип / область', 'Контроль качества']
        first = project.rows[0].id
        summary.show_profile_row(project.rows[1].id)
        assert summary.profile_row == project.rows[1].id
        assert 'замечаний QC: 1' in summary.profile_note.text()
        summary.canvas.callbacks.process('pick_event', SimpleNamespace(artist=axis.collections[0], ind=[0]))
        assert summary.profile_row == first
        assert dialog.result_selection == {first}
        assert dialog.plot.selection_overlay.selected_ids == {first}
        assert all(highlight.get_visible() for highlight in summary.highlights.values())
        summary.open_samples(summary.model.index(2, 0))
        inspected.append(True)
        return summary.result()

    monkeypatch.setattr(InterpretationSummaryDialog, 'exec', inspect_summary)
    assert dialog.summary_button.isEnabled()
    dialog.summary_button.click()
    assert inspected
    assert dialog.result_selection == {project.rows[1].id}
    assert dialog.scope.currentData() == 'filtered'
    saved = project.view['interpretation_summary_bindings'][scheme['id']]
    assert saved == dict(binding, text_numbers=False)
    assert 'stratigraphy' not in project.view
    path = tmp_path / 'summary.apyro'
    project.save(path)
    assert type(project).load(path).view['interpretation_summary_bindings'][scheme['id']] == saved
    dialog.close()


def test_well_summary_compares_two_wells_without_merging_intervals(qtbot, monkeypatch, tmp_path):
    import csv

    from PySide6.QtCore import QLocale
    from PySide6.QtWidgets import QFileDialog

    from autopyro.data_io.io import build_project
    from autopyro.schemes.interpretation import evaluate_project
    from autopyro.ui.widgets.interpretation_summary import InterpretationSummaryDialog

    names = ['Ствол', 'MD', 'TOC', 'S2']
    units = ['', 'm', '%wt', 'mg HC/g rock']
    project = build_project(
        names, [['A', '10', '1.5', '3'], ['B', '15', '101', '3'], ['C', '25', '2', '3']],
        [1, 2, 3], [(i, name, 'number' if i else 'text', units[i]) for i, name in enumerate(names)], {})
    well, depth, toc, s2 = [column.id for column in project.columns]
    report = evaluate_project(project, project.row_map(), toc, s2, units[2], units[3],
                              standard_schemes()['S2–TOC'])
    stratigraphy = {
        'format': 'autopyro.stratigraphy', 'version': 1, 'id': 'test', 'revision': '1',
        'name': 'Test', 'depth_reference': 'MD', 'depth_unit': 'm',
        'intervals': [{'id': name, 'wellbore': name, 'level': 'FORMATION',
                       'name': 'Общий горизонт', 'top': 0, 'bottom': 30} for name in ('A', 'B', 'C')],
    }
    binding = {'well_id': well, 'depth_id': depth, 'depth_unit': 'm', 'depth_reference': 'MD'}
    summary = InterpretationSummaryDialog(None, project, report, stratigraphy, binding)
    qtbot.addWidget(summary)
    assert summary.comparison_well.findData('A') == -1
    summary.comparison_well.setCurrentIndex(summary.comparison_well.findData('B'))
    assert summary.model.headers[0] == 'Ствол'
    assert [row[0] for row in summary.model.rows] == ['A', 'A', 'B', 'B']
    assert summary.model.rows[1][1] == summary.model.rows[3][1]
    assert summary.model.rows[1][-1] == '0 (0.0%)'
    assert summary.model.rows[3][-1] == '1 (100.0%)'

    path = tmp_path / 'wells.csv'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), ''))
    summary.export_button.click()
    delimiter = ';' if QLocale.system().decimalPoint() == ',' else ','
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream, delimiter=delimiter))
    assert [row['wellbore'] for row in rows] == ['A', 'A', 'B', 'B']
    assert [row['excluded_count'] for row in rows] == ['0', '0', '1', '1']

    summary.well.setCurrentIndex(summary.well.findData('B'))
    assert summary.comparison_well.currentData() is None
    assert summary.model.headers[0] == 'Интервал / привязка'
    summary.comparison_well.setCurrentIndex(summary.comparison_well.findData('A'))
    summary.table.selectRow(3)
    assert summary.profile_well == 'A'
    assert summary.well.currentData() == 'B'
    summary.open_samples(summary.model.index(3, 0))
    assert summary.selected_ids == (project.rows[0].id,)


def test_depth_profile_shows_hio_range_at_interval_midpoint(qtbot):
    from types import SimpleNamespace

    from autopyro.data_io.io import build_project
    from autopyro.schemes.interpretation import evaluate_project
    from autopyro.ui.widgets.interpretation_summary import InterpretationSummaryDialog

    columns = ['Ствол', 'Кровля пробы', 'Подошва пробы', 'Tmax', 'HI']
    units = ['', 'm', 'm', '°C', 'mg HC/g TOC']
    project = build_project(columns, [['A', '10', '20', '430', '300']], [1], [
        (i, name, 'number' if i else 'text', unit)
        for i, (name, unit) in enumerate(zip(columns, units, strict=True))
    ], {})
    project.sync_measurements()
    well, top, bottom, x, y = [column.id for column in project.columns]
    row_id = project.rows[0].id
    measurement = project.science.current(row_id, y)
    measurement.uncertainty, measurement.uncertainty_type = 10, 'absolute'
    scheme = standard_schemes()['HIo · HI–Tmax']
    report = evaluate_project(project, [row_id], x, y, units[3], units[4], scheme)
    low, high = report['results'][0]['value_bounds']
    assert low < high
    stratigraphy = {
        'format': 'autopyro.stratigraphy', 'version': 1, 'id': 'test', 'revision': '1',
        'name': 'Test', 'depth_reference': 'MD', 'depth_unit': 'm',
        'intervals': [{'id': 'one', 'wellbore': 'A', 'level': 'FORMATION',
                       'name': 'One', 'top': 0, 'bottom': 30}],
    }
    binding = {'well_id': well, 'depth_id': top, 'bottom_id': bottom,
               'depth_reference': 'MD', 'depth_unit': 'm', 'sample_mode': 'interval'}
    summary = InterpretationSummaryDialog(None, project, report, stratigraphy, binding)
    qtbot.addWidget(summary)
    axis = summary.figure.axes[3]
    assert axis.get_title() == 'Диапазон HIo'
    assert axis.yaxis_inverted()
    artist = axis.collections[0]
    assert artist.get_segments()[0][:, 0] == pytest.approx([low, high])
    assert artist.get_segments()[0][:, 1] == pytest.approx([15, 15])
    with qtbot.waitSignal(summary.profile_selected) as selection:
        summary.pick_profile(SimpleNamespace(artist=artist, ind=[0]))
    assert selection.args == [row_id]
    assert 'HIo' in summary.profile_note.text()


def test_revision_comparison_survives_recalculation_and_project_save(comparison, qtbot, monkeypatch, tmp_path):
    from autopyro.schemes.interpretation_review import save_review
    from autopyro.schemes.interpretation_revisions import retain_previous_revision
    from autopyro.ui.widgets.interpretation_revisions import InterpretationRevisionsDialog

    window, dialog = comparison
    ready(qtbot, dialog, 3)
    project = window.model.project
    row_id = project.rows[1].id
    save_review(project, dialog.report, 1, 'accepted')
    original = project.view['interpretation_result']
    oi_column = next(i for i, c in enumerate(project.columns) if c.name == 'OI')
    assert window.model.setData(window.model.index(1, oi_column), '200')
    ready(qtbot, dialog, 3)
    assert project.view['interpretation_previous_result'] is original

    dialog.schedule()
    ready(qtbot, dialog, 3)
    assert project.view['interpretation_previous_result'] is original
    path = tmp_path / 'revisions.apyro'
    project.save(path)
    reopened = type(project).load(path)
    persisted_previous = reopened.view['interpretation_previous_result']
    retain_previous_revision(reopened.view, dialog.report)
    assert reopened.view['interpretation_previous_result'] is persisted_previous

    inspected = []

    def inspect_revisions(revisions):
        assert [change.row_id for change in revisions.changes] == [row_id]
        assert {'inputs', 'outcome', 'agreement', 'review'} <= set(revisions.changes[0].flags)
        assert revisions.table.model().rows[0][2] != revisions.table.model().rows[0][3]
        details = revisions.input_details.toPlainText()
        assert 'HI–OI · X · OI' in details
        assert 'Исходное значение: 20 → 200' in details
        assert 'Рабочее измерение:' in details and 'версия' in details
        revisions.show_sample.click()
        inspected.append(True)
        return revisions.result()

    monkeypatch.setattr(InterpretationRevisionsDialog, 'exec', inspect_revisions)
    assert dialog.revisions_button.isEnabled()
    dialog.revisions_button.click()
    assert inspected
    assert dialog.result_selection == {row_id}


def test_sample_pyrogram_overlay_and_observation_survive_project_save(comparison, qtbot, monkeypatch, tmp_path):
    import json
    from copy import deepcopy

    from PySide6.QtWidgets import QFileDialog, QMessageBox

    from autopyro.ui.widgets.pyrograms import PyrogramDialog

    window, dialog = comparison
    ready(qtbot, dialog, 3)
    project, report = window.model.project, dialog.report
    first, second = project.rows
    dialog.select_result_rows([second.id])
    paths = []
    run = {
        'format': 'autopyro.pyrogram', 'version': 1, 'label': 'First',
        'time_unit': 's', 'temperature_unit': '°C', 'heating_program': 'Programme A',
        'traces': [{
            'channel': 'FID', 'stage': 'Pyrolysis', 'unit': 'mV',
            'time': [0, 1, 2], 'temperature': [300, 310, 305], 'signal': [-2, 4, 1],
            'peaks': [{'label': 'Peak', 'index': 1}],
            'intervals': [{'label': 'Window', 'start': 0, 'end': 1}],
        }],
    }
    repeat = deepcopy(run)
    repeat['label'] = 'Repeat'
    repeat['traces'][0].update(unit='V', signal=[0, 0, 0])
    repeat['traces'].append(dict(repeat['traces'][0], channel='IR'))
    invalid = deepcopy(run)
    invalid['traces'][0]['peaks'][0]['index'] = 3
    for index, payload in enumerate((run, repeat, invalid)):
        path = tmp_path / f'run-{index}.json'
        path.write_text(json.dumps(payload), encoding='utf-8')
        paths.append(str(path))
    choices = iter([*paths, paths[0]])
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (next(choices), ''))
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warnings.append(args[-1]))
    answers = iter([QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.Yes])
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: next(answers))
    inspected = []

    def inspect_pyrogram(view):
        for _ in range(4):
            view.add_run()
        assert view.runs.count() == 2
        assert len(warnings) == 2
        assert [axis.get_ylabel() for axis in view.figure.axes] == ['mV', 'V']
        view.normalize.setChecked(True)
        view.axis.setCurrentIndex(view.axis.findData('temperature'))
        view.canvas.draw()
        axis, = view.figure.axes
        assert list(axis.lines[0].get_xdata()) == [300, 310, 305]
        assert list(axis.lines[0].get_ydata()) == [-.5, 1, .25]
        assert list(axis.lines[2].get_ydata()) == [0, 0, 0]
        assert {text.get_text() for text in axis.texts} == {'Peak', 'Window'}
        view.observation.setPlainText('Наблюдение по сигналу')
        view.save_observation.click()
        view.trace.setCurrentIndex(view.trace.findData(('IR', 'Pyrolysis')))
        view.runs.setCurrentRow(1)
        assert view.detach_button.isEnabled()
        view.detach_button.click()
        assert view.runs.count() == 2
        view.detach_button.click()
        assert view.runs.count() == 1
        assert view.trace.currentData() == ('FID', 'Pyrolysis')
        axis, = view.figure.axes
        assert 'First' in axis.lines[0].get_label()
        inspected.append(True)
        return view.result()

    monkeypatch.setattr(PyrogramDialog, 'exec', inspect_pyrogram)
    assert dialog.pyrogram_button.isEnabled()
    dialog.pyrogram_button.click()
    assert inspected and dialog.report is report
    assert 'Наблюдение по сигналу' in dialog.explanation.toPlainText()
    assert 'pyrograms' not in project.science.samples[first.id].metadata
    metadata = project.science.samples[second.id].metadata
    undo = window.model.undo
    undo.undo()
    assert len(metadata['pyrograms']) == 2
    undo.undo()
    assert 'pyrogram_observation' not in metadata
    undo.undo()
    assert len(metadata['pyrograms']) == 1
    for _ in range(3):
        undo.redo()
    assert len(metadata['pyrograms']) == 1
    assert metadata['pyrogram_observation'] == 'Наблюдение по сигналу'
    audit = [entry for entry in project.history if entry["action"] in ("pyrogram_metadata", "undo_pyrogram_metadata")]
    assert len(audit) == 10
    assert {entry["sample_id"] for entry in audit} == {second.id}
    assert [entry["action"] for entry in audit[-6:]] == ["undo_pyrogram_metadata"] * 3 + ["pyrogram_metadata"] * 3
    observation = next(entry for entry in audit if entry["field"] == "pyrogram_observation")
    assert observation["previous"] is None
    assert observation["value"] == 'Наблюдение по сигналу'
    attachment = next(entry for entry in audit if entry["field"] == "pyrograms")
    assert attachment["previous"] is None
    assert attachment["value"][0]["sha256"] == metadata["pyrograms"][0]["sha256"]
    assert "run" not in attachment["value"][0]
    path = tmp_path / 'pyrograms.apyro'
    project.save(path)
    reopened = type(project).load(path)
    assert [entry for entry in reopened.history if entry["action"] in ("pyrogram_metadata", "undo_pyrogram_metadata")] == audit
    metadata = reopened.science.samples[second.id].metadata
    assert [entry['run'] for entry in metadata['pyrograms']] == [run]
    assert metadata['pyrogram_observation'] == 'Наблюдение по сигналу'
    assert (tmp_path / 'run-1.json').is_file()
    view = PyrogramDialog(None, reopened, second.id)
    qtbot.addWidget(view)
    view.runs.setCurrentRow(0)
    view.detach_button.click()
    assert view.runs.count() == view.trace.count() == 0
    assert not view.detach_button.isEnabled()
    assert 'pyrograms' not in metadata
    assert metadata['pyrogram_observation'] == 'Наблюдение по сигналу'
    assert (tmp_path / 'run-0.json').is_file()


def test_pyrogram_unsaved_observation_close_paths(qtbot, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDialogButtonBox, QMessageBox

    from autopyro.ui.widgets.pyrograms import PyrogramDialog

    project, _, _ = data()
    project.sync_measurements()
    sample = project.science.samples[project.rows[0].id]
    sample.metadata['pyrogram_observation'] = 'Ранее сохранённое'
    dialog = PyrogramDialog(None, project, sample.id)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.observation.setPlainText('Черновик')
    answers = iter([
        QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Discard, QMessageBox.StandardButton.Save,
    ])
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: next(answers))

    assert not dialog.close()
    assert dialog.isVisible() and dialog.observation.toPlainText() == 'Черновик'
    qtbot.keyClick(dialog, Qt.Key.Key_Escape)
    assert not dialog.isVisible()
    assert sample.metadata['pyrogram_observation'] == 'Ранее сохранённое'

    dialog.show()
    assert dialog.observation.toPlainText() == 'Ранее сохранённое'
    dialog.observation.setPlainText('Новая запись')
    with qtbot.waitSignal(dialog.changed):
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Close).click()
    assert not dialog.isVisible()
    assert sample.metadata['pyrogram_observation'] == 'Новая запись'
    dialog.show()
    assert dialog.close()


def test_pinned_comparison_base_survives_recalculation_scope_and_reload(comparison, qtbot, tmp_path):
    import json

    from autopyro.schemes.interpretation_revisions import comparison_base

    window, dialog = comparison
    ready(qtbot, dialog, 3)
    project = window.model.project
    original = dialog.report
    dialog.pin_base.click()
    pinned = comparison_base(project.view)
    assert pinned == original and pinned is not original
    assert dialog.revisions_button.isEnabled()

    oi_column = next(i for i, c in enumerate(project.columns) if c.name == 'OI')
    assert window.model.setData(window.model.index(1, oi_column), '200')
    ready(qtbot, dialog, 3)
    window.table.selectRow(1)
    dialog.scope.setCurrentIndex(dialog.scope.findData('selected'))
    ready(qtbot, dialog, 3)
    assert len(dialog.report['results']) == 1
    assert comparison_base(project.view) is pinned
    assert project.view['interpretation_previous_result']['type_agreement'] == ['equal', 'different']

    path = tmp_path / 'comparison-base.apyro'
    project.save(path)
    reopened = type(project).load(path)
    assert comparison_base(reopened.view) == json.loads(json.dumps(pinned))
    dialog.pin_base.click()
    assert comparison_base(project.view) is project.view['interpretation_previous_result']


def test_hover_uses_its_diagram_and_switching_back_preserves_selection(comparison, qtbot):
    _, dialog = comparison
    ready(qtbot, dialog, 3)
    plot = dialog.plot
    for artist in plot.result['artists']:
        rid = artist._autopyro_row_ids[0]
        binding = artist.axes._autopyro_interpretation
        text = plot.point_tooltip(rid, tuple(artist.get_offsets()[0]), binding)
        assert binding['scheme']['name'] in text
        for key in ('x', 'y'):
            assert binding['scheme'][key]['property'] in text
    rid = plot.result['artists'][0]._autopyro_row_ids[0]
    plot.select_rows([rid])
    dialog.comparison.setChecked(False)
    ready(qtbot, dialog, 1)
    assert plot.selection_overlay.selected_ids == {rid}
    assert plot.spec.interpretation_panels == []


def test_disagreement_filter_exports_visible_rows_and_expert_conclusion(comparison, qtbot, monkeypatch, tmp_path):
    import csv
    import json

    from PySide6.QtWidgets import QFileDialog
    from autopyro.schemes.interpretation import replay_report
    from autopyro.schemes.interpretation_review import save_review

    window, dialog = comparison
    project = window.model.project
    first, second = project.rows
    oi = next(c for c in project.columns if c.name == 'OI')
    second.values[oi.id] = '200'
    dialog.schedule()
    ready(qtbot, dialog, 3)
    report = dialog.report
    assert report['type_agreement'] == ['equal', 'different']

    dialog.result_filter.setCurrentIndex(dialog.result_filter.findData('disagreement'))
    ready(qtbot, dialog, 3)
    assert dialog.report is report
    assert dialog.plot.rendered_ids == [second.id]
    assert all(list(artist._autopyro_row_ids) == [second.id] for artist in dialog.plot.result['artists'])
    dialog.table.selectRow(0)
    assert dialog.plot.selection_overlay.selected_ids == {second.id}
    assert 'Нет общих типов' in dialog.explanation.toPlainText()
    dialog.plot.select_rows([second.id])
    assert {index.data() for index in dialog.table.selectionModel().selectedRows()} == {second.id}

    save_review(project, report, 0, 'open', note='Скрытая проба', author='Эксперт')
    save_review(project, report, 1, 'open', note='Первое решение', author='Эксперт')
    save_review(project, report, 1, 'expert', 'Экспертный тип', 'По двум диаграммам', author='Эксперт')
    project.science.samples[second.id].metadata['pyrogram_observation'] = 'Проверен пик'
    path = tmp_path / 'interpretation.csv'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), ''))
    dialog.save_csv()
    with path.open(encoding='utf-8-sig', newline='') as stream:
        row, = csv.DictReader(stream, delimiter=';')
    assert row['row_id'] == second.id
    assert row['type_agreement'] == 'different'
    assert row['review_state'] == 'expert'
    assert row['review_author'] == 'Эксперт'
    assert row['review_conclusion'] == 'Экспертный тип'
    assert row['pyrogram_observation'] == 'Проверен пик'
    assert row['sample_label'] == second.id
    assert (float(row['x']), row['x_column'], row['x_unit']) == (2, 'TOC', '%wt')
    assert (float(row['y']), row['y_column'], row['y_unit']) == (4, 'S2', 'mg HC/g rock')
    assert row['reason']

    path = tmp_path / 'interpretation.json'
    dialog.save_report()
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert first.id not in path.read_text(encoding='utf-8')
    assert payload['selection'] == {
        'scope': 'visible', 'filter': 'disagreement', 'visible_count': 1,
        'exported_count': 1, 'calculation_count': 2,
    }
    review, = payload['expert_reviews']
    assert review['row_id'] == second.id
    assert [entry['status'] for entry in review['history']] == ['open', 'expert']
    snapshot = payload['report']
    assert snapshot['type_agreement'] == ['different']
    for diagram in (snapshot, *snapshot['comparisons'].values()):
        assert [item['row_id'] for item in diagram['inputs']] == [second.id]
        assert replay_report(diagram) == diagram['results']

    dialog.export_scope.setCurrentIndex(dialog.export_scope.findData('all'))
    path = tmp_path / 'full-interpretation.csv'
    dialog.save_csv()
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream, delimiter=';'))
    assert [row['row_id'] for row in rows] == [first.id, second.id]
    assert rows[0]['review_note'] == 'Скрытая проба'

    path = tmp_path / 'full-interpretation.json'
    dialog.save_report()
    full = json.loads(path.read_text(encoding='utf-8'))
    assert full['selection'] == {
        'scope': 'all', 'filter': 'disagreement', 'visible_count': 1,
        'exported_count': 2, 'calculation_count': 2,
    }
    assert [item['row_id'] for item in full['expert_reviews']] == [first.id, second.id]
    assert replay_report(full['report']) == full['report']['results']
    assert dialog.report is report
    assert dialog.plot.rendered_ids == [second.id]
    assert dialog.plot.selection_overlay.selected_ids == {second.id}

    dialog.comparison.setChecked(False)
    ready(qtbot, dialog, 1)
    assert dialog.result_filter.currentData() == 'all'
    assert dialog.result_filter.findData('disagreement') == -1
    assert dialog.plot.selection_overlay.selected_ids == {second.id}


def test_changed_comparison_input_explains_saved_snapshot(comparison, qtbot):
    window, dialog = comparison
    ready(qtbot, dialog, 3)
    project = window.model.project
    row = project.rows[1]
    oi = next(c for c in project.columns if c.name == 'OI')
    saved = project.view['interpretation_result']
    row.values[oi.id] = '200'
    window.model.refresh(rows=[row])
    assert saved['stale_reason'] == {
        'code': 'input_changed', 'row_id': row.id, 'column_id': oi.id,
        'axis': 'x', 'diagram': 'HI–OI',
    }
    assert 'HI–OI' in dialog.stale_notice.text()
    assert row.id in dialog.stale_notice.text()
    ready(qtbot, dialog, 3)
    assert not project.view['interpretation_result']['stale']
    assert 'Результаты пересчитаны' in dialog.stale_notice.text()
