"""The passport document follows model changes, undo and export validity."""
import json
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QComboBox, QDialog, QFileDialog, QInputDialog, QLineEdit, QPlainTextEdit, QPushButton

from autopyro.analysis.nonpyro import STER_HOP_HOPANES, STER_HOP_STERANES
from autopyro.core import Column, Project, Row
from autopyro.geochemistry_workspace import STATE_KEY, GeochemistryIndex
from autopyro.ui.window import MainWindow
from autopyro.ui.dialogs import (
    geochemistry_workspace_catalog as catalog,
    geochemistry_workspace_comparison as comparison,
    geochemistry_workspace_exports as exports,
)
from autopyro.ui.dialogs.geochemistry_workspace import GeochemistryWorkspaceDialog
from autopyro.ui.dialogs.geochemistry_workspace_display import _status
from autopyro.ui.i18n import tr


@pytest.fixture
def workspace(qtbot):
    name = Column('Sample / Name')
    column = Column('TOC', 'number', unit='%wt')
    project = Project(columns=[name, column], rows=[
        Row({name.id: '<Core A>', column.id: 2}),
        Row({name.id: 'Core B', column.id: 4}),
        Row({name.id: 'Core C', column.id: 6}),
    ])
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.actions['Паспорта и сравнение проб…'].trigger()
    document = owner.geochemistry_workspace_window
    document.fields.item(0).setCheckState(Qt.CheckState.Checked)
    yield owner, document, column
    owner.dirty = False


def select(document, positions):
    document.samples.clearSelection()
    for position in positions:
        document.samples.item(position).setSelected(True)


def test_incompatible_merge_leaves_other_passports_and_can_be_repaired(qtbot, monkeypatch):
    from autopyro.analysis.geochemistry_workspace import link_samples

    material = Column('Sample / Type')
    project = Project(columns=[material], rows=[Row({material.id: 'oil'}) for _ in range(3)])
    original = list(GeochemistryIndex(project).samples)
    project.view[STATE_KEY] = link_samples(project, original[:2], 'Merged')
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_geochemistry_workspace()
    document = owner.geochemistry_workspace_window
    assert document.samples.count() == 2
    assert owner.model.setData(owner.model.index(0, 0), 'rock')
    qtbot.waitUntil(lambda: document.samples.count() == 3)
    assert original[-1] in document.available_ids
    assert not document.repair_links_button.isHidden()
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args: (args[3][0], True))
    document.repair_links_button.click()
    assert not project.view[STATE_KEY]['links']
    owner.model.undo.undo()
    qtbot.waitUntil(lambda: not document.repair_links_button.isHidden())
    owner.model.undo.undo()
    qtbot.waitUntil(lambda: document.samples.count() == 2)
    assert document.repair_links_button.isHidden()
    owner.dirty = False


@pytest.mark.parametrize('correlation', [False, True])
def test_export_rechecks_inputs_after_file_dialog(qtbot, monkeypatch, tmp_path, correlation):
    ratio = Column('Sat-GC / Pr/nC19[a]', 'number', unit='1')
    project = Project(columns=[ratio], rows=[
        Row({ratio.id: value}, source={'sample_type': 'oil'}) for value in (.4, .5)
    ])
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_geochemistry_workspace()
    document = owner.geochemistry_workspace_window
    document.fields.item(0).setCheckState(Qt.CheckState.Checked)
    document.calculate_comparison()
    export = document.export_comparison
    if correlation:
        document.oil_correlation_button.click()
        export = document.export_correlation
        csv_path = tmp_path / 'valid.csv'
        monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(csv_path), ''))
        export()
        assert all(name in csv_path.read_text('utf-8-sig') for name in document.report['request']['names'])
    path = tmp_path / 'stale.json'

    def choose_file(*args):
        assert owner.model.setData(owner.model.index(0, 0), .9)
        return str(path), ''

    monkeypatch.setattr(QFileDialog, 'getSaveFileName', choose_file)
    export()
    assert not path.exists()
    assert 'Пересчитайте сравнение' in document.status.text()
    owner.dirty = False


def test_group_edits_preserve_unrelated_comparison(workspace):
    owner, document, _ = workspace
    from autopyro.analysis.geochemistry_workspace import save_group

    project = owner.model.project
    samples = list(document.index.samples)
    document.calculate_comparison()
    original = document.report
    assert original is not None
    document.commit(save_group(project, 'A reference', [samples[0]]), 'Save group')
    assert document.report is original
    document.a.setCurrentIndex(document.a.findData('group:A reference'))
    document.calculate_comparison()
    assert document.report['request']['saved_groups'] == ['A reference', None]
    document.commit(save_group(project, 'A reference', [samples[2]]), 'Change group')
    assert document.report is None
    project.view[STATE_KEY]['groups']['A reference'] = ['missing-sample']
    document.refresh()
    document.calculate_comparison()
    assert 'отсутствуют пробы missing-sample' in document.status.text()


def test_export_passport_reuses_current_index(workspace, monkeypatch, tmp_path):
    _, document, column = workspace
    path = tmp_path / 'passport.json'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), ''))
    monkeypatch.setattr(exports, 'GeochemistryIndex', lambda *args: pytest.fail('The current index was rebuilt'))
    document.export_passport()
    passport = json.loads(path.read_text('utf-8'))
    measurement, = [row for row in passport['measurements'] if row['column_id'] == column.id]
    assert measurement['value'] == 2


def test_index_tracks_edits_undo_and_working_measurement_changes(workspace, qtbot):
    from autopyro.ui.table.commands import AddMeasurementCommand, SelectMeasurementCommand

    owner, document, column = workspace
    project = owner.model.project
    row = project.rows[0]
    position = project.columns.index(column)

    def settle():
        qtbot.waitUntil(lambda: not document.timer.isActive() and not document._loading)

    def value():
        return document.index.value(document.selected_samples()[0], column.id)

    old_records = document.index.row_records
    old_index = document.index
    old_catalog = document._catalog_rows
    old_item = document.samples.item(0)
    assert owner.model.setData(owner.model.index(0, position), '9')
    settle()
    assert document.index.row_records is old_records
    assert document.index is old_index
    assert document._catalog_rows is old_catalog
    assert document.samples.item(0) is old_item
    assert value()['value'] == 9
    owner.model.undo.undo()
    settle()
    assert value()['value'] == 2
    assert document.index is old_index
    measurement, analysis = project.science.prepare_result(row.id, column.id, 12)
    owner.model.undo.push(AddMeasurementCommand(owner.model, measurement, analysis))
    settle()
    assert document.index.row_records is not old_records
    assert len(document.index.candidates(document.selected_samples()[0], column.id)) == 2
    owner.model.undo.push(SelectMeasurementCommand(owner.model, row.id, column.id, measurement.id))
    settle()
    assert value()['value'] == 12
    owner.model.undo.undo()
    settle()
    assert value()['value'] == 2


def test_imported_material_edit_updates_passport_and_correlation(qtbot):
    name = Column('Sample / Name')
    kind = Column('Sample / Type')
    preparation = Column('Sample / Preparation')
    ratio = Column('Sat-GC / Pr/nC19[a]', 'number', unit='1')
    project = Project(columns=[name, kind, preparation, ratio], rows=[
        Row({name.id: label, kind.id: 'oil', preparation.id: '', ratio.id: value},
            source={'sample_type': 'oil', 'preparation': ''})
        for label, value in (('A', .4), ('B', .5))
    ])
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_geochemistry_workspace()
    document = owner.geochemistry_workspace_window
    document.fields.item(0).setCheckState(Qt.CheckState.Checked)

    def settle():
        qtbot.waitUntil(lambda: not document.timer.isActive() and not document._loading)

    assert owner.model.setData(owner.model.index(0, 1), 'rock')
    settle()
    assert owner.model.setData(owner.model.index(0, 2), 'extract')
    settle()
    assert document.passport['sample']['material']['sample_type'] == 'rock'
    assert document.passport['sample']['material']['preparation'] == 'extract'
    document.mixed.setChecked(True)
    document.calculate_comparison()
    document.oil_correlation_button.click()
    assert document.profile_match['correlation']['materials'] == ['rock_extract', 'oil']
    assert owner.model.setData(owner.model.index(0, 2), '')
    settle()
    assert document.passport['sample']['material']['preparation'] == ''
    document.calculate_comparison()
    assert not document.oil_correlation_button.isEnabled()
    owner.model.undo.undo()
    owner.model.undo.undo()
    owner.model.undo.undo()
    settle()
    assert document.passport['sample']['material']['sample_type'] == 'oil'
    assert project.rows[0].source == {'sample_type': 'oil', 'preparation': ''}
    owner.dirty = False


def test_material_relation_marks_missing_identity_and_recovers_after_undo(qtbot, monkeypatch, tmp_path):
    from autopyro.analysis.geochemistry_workspace import relate_materials

    kind = Column('Sample / Type')
    project = Project(columns=[kind], rows=[
        Row({kind.id: 'rock'}, source={'sample_id': i}) for i in (1, 2)
    ], source={'endpoint': {'host': 'fixture', 'port': 5432, 'database': 'geo'}})
    a, b = GeochemistryIndex(project).samples
    project.view[STATE_KEY] = relate_materials(project, a, b, 'extract')
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_geochemistry_workspace()
    document = owner.geochemistry_workspace_window
    assert owner.model.setData(owner.model.index(1, 0), 'oil')
    qtbot.waitUntil(lambda: not document.timer.isActive() and not document._loading)
    relation, = document.passport['relationships']
    assert relation['missing_sample_ids'] == [b]
    assert 'Связанная проба отсутствует' in document.details.toPlainText()
    path = tmp_path / 'passport.json'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), ''))
    document.export_passport()
    assert json.loads(path.read_text('utf-8'))['relationships'][0]['missing_sample_ids'] == [b]
    owner.model.undo.undo()
    qtbot.waitUntil(lambda: not document.timer.isActive() and not document._loading)
    assert document.passport['relationships'] == [{'parent': a, 'child': b, 'kind': 'extract'}]
    assert 'Связанная проба отсутствует' not in document.details.toPlainText()
    owner.dirty = False


def test_comparison_controls_fit_narrow_properties_pane(qtbot, recwarn):
    name = Column('Sample / Name')
    ratio = Column('Sat-GC / Pr/nC19[a]', 'number', unit='1')
    project = Project(columns=[name, ratio], rows=[
        Row({name.id: f'{side} — ' + 'A long geochemistry sample label ' * 8, ratio.id: value},
            source={'sample_type': 'oil'})
        for side, value in (('A', 0.4), ('B', 0.5))
    ])
    owner = MainWindow()
    qtbot.addWidget(owner)
    windows_font = Path('C:/Windows/Fonts/segoeui.ttf')
    if windows_font.exists() and QFontDatabase.addApplicationFont(str(windows_font)) >= 0:
        owner.setFont(QFont('Segoe UI', 10))
    owner.set_project(project)
    owner.show()
    owner.show_geochemistry_workspace()
    document = owner.geochemistry_workspace_window
    document.parameter_tabs.setCurrentIndex(1)
    page = document.parameter_tabs.currentWidget()
    page.setFixedWidth(340)
    qtbot.waitUntil(lambda: page.viewport().width() <= 340)

    assert page.horizontalScrollBar().maximum() == 0
    assert document.a.toolTip() == document.a.currentText()
    assert document.b.toolTip() == document.b.currentText()
    for control in (document.a, document.b, document.field_preset, document.compare_button,
                    document.correlation_preview_button, document.oil_correlation_button):
        assert control.mapTo(page.widget(), control.rect().topRight()).x() < page.viewport().width()
    document.fields.item(0).setCheckState(Qt.CheckState.Checked)
    document.compare_button.click()
    assert document.oil_correlation_button.isEnabled(), document.status.text()
    document.passport_tabs.setCurrentWidget(document.canvas)
    qtbot.waitUntil(document.canvas.isVisible)
    document.canvas.draw()
    axis, = document.figure.axes
    assert [label.get_text() for label in axis.get_xticklabels()] == ['A\nN = 1', 'B\nN = 1']
    assert all(name in document.status.text() for name in document.report['request']['names'])
    assert not any('constrained_layout not applied' in str(warning.message) for warning in recwarn)
    assert document.minimumSizeHint().width() < sum(
        button.minimumSizeHint().width()
        for button in (document.use_measurement, document.reset_button, document.method_button)
    )
    document.show_model_archive()
    archive = document.model_archive_page
    buttons = archive.findChildren(QPushButton)
    assert len(buttons) == 5
    assert archive.minimumSizeHint().width() < sum(button.minimumSizeHint().width() for button in buttons[:4])
    assert all(button.isVisible() and archive.rect().contains(button.geometry()) for button in buttons)
    owner.dirty = False


def test_comparison_opens_exact_repeat_and_recalculates_after_selection(qtbot, tmp_path, monkeypatch):
    from autopyro.database.geochem import project_with_repeats

    group = 'Sat-GC'
    field = {'source_field_id': 1, 'group_name': group, 'source_label': 'Pr/nC19[a]',
             'unit_label': '1', 'source_dtype': 'Float64'}
    project = project_with_repeats(
        [{'source_record_id': i, 'sample_name': name, 'sample_id': i}
         for i, name in ((1, 'A'), (2, 'B'))],
        [{'source_record_id': sample, 'source_field_id': 1, 'analysis_run_id': run,
          'repeat_number': repeat, 'group_name': group, 'value_float': value}
         for sample, run, repeat, value in ((1, 11, 1, .4), (1, 12, 2, .7), (2, 21, 1, .5))],
        [field], source={'include_repeats': True, 'repeat_number': 1},
    )
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_geochemistry_workspace()
    document = owner.geochemistry_workspace_window
    for i in range(document.fields.count()):
        item = document.fields.item(i)
        if 'Pr/nC19' in item.text():
            item.setCheckState(Qt.CheckState.Checked)
    document.calculate_comparison()
    assert document.comparison_table.item(0, 3).text() == '0.4'
    assert '11' in document.comparison_table.item(0, 3).toolTip()
    document.comparison_table.cellDoubleClicked.emit(0, 3)
    assert document.passport_tabs.currentWidget() is document.measurements
    assert document.measurement_record()['analysis_run_id'] == 11
    for i in range(document.measurements.rowCount()):
        record = document.measurements.item(i, 0).data(Qt.ItemDataRole.UserRole)
        if record.get('analysis_run_id') == 12:
            document.measurements.selectRow(i)
            break
    document.choose_measurement()
    assert document.report is None
    document.calculate_comparison()
    assert document.comparison_table.item(0, 3).text() == '0.7'
    assert '12' in document.comparison_table.item(0, 3).toolTip()
    path = tmp_path / 'repeats.apyro'
    project.save(path)
    restored = Project.load(path)
    index = GeochemistryIndex(restored)
    sample_id = next(iter(index.samples))
    field_id = document.report['results'][0]['field']['id']
    assert len(index.candidates(sample_id, field_id)) == 2
    assert index.value(sample_id, field_id)['measurement']['analysis_run_id'] == 12

    chosen = index.value(sample_id, field_id)['measurement']
    project.view['qc_exclusions'] = {'statistics': {
        chosen['id']: {'revision': chosen['revision'], 'comment': 'Interference <QC>'},
    }}
    document.refresh()
    document.calculate_comparison()
    assert document.comparison_table.item(0, 2).text() == '0 / 1'
    document.comparison_table.cellDoubleClicked.emit(0, 2)
    assert document.measurement_record()['analysis_run_id'] == 12
    statuses = {}
    for row in range(document.measurements.rowCount()):
        record = document.measurements.item(row, 0).data(Qt.ItemDataRole.UserRole)
        if record['column_id'] == field_id:
            statuses[record['analysis_run_id']] = record['comparison_status']
        if record['id'] == chosen['id']:
            assert 'Interference &lt;QC&gt;' in document.measurements.item(row, 7).toolTip()
    assert statuses == {11: 'not_selected', 12: 'qc_excluded'}
    path = tmp_path / 'passport.json'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), ''))
    document.export_passport()
    records = json.loads(path.read_text('utf-8'))['measurements']
    excluded, = [record for record in records if 'statistics_exclusion' in record]
    assert excluded['id'] == chosen['id']
    assert excluded['statistics_exclusion']['comment'] == 'Interference <QC>'
    owner.dirty = False


def test_large_workspace_prepares_initial_catalog_in_worker(qtbot, monkeypatch):
    from threading import get_ident

    column = Column('Sample / Name')
    project = Project(columns=[column], rows=[Row({column.id: str(i)}) for i in range(10000)])
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    threads = []

    def index(*args, **kwargs):
        threads.append(get_ident())
        return GeochemistryIndex(*args, **kwargs)

    monkeypatch.setattr(catalog, 'GeochemistryIndex', index)
    owner.show_geochemistry_workspace()
    document = owner.geochemistry_workspace_window
    assert len(threads) == 1 and threads[0] != get_ident()
    assert document.samples.count() == document.a.count() == document.b.count() == 10000
    assert document.passport['sample']['name'] == '0'
    assert owner.model.setData(owner.model.index(0, 0), 'Edited sample')
    qtbot.waitUntil(lambda: len(threads) == 2 and not document._loading, timeout=15000)
    assert all(thread != get_ident() for thread in threads)
    assert document.passport['sample']['name'] == 'Edited sample'
    assert document.samples.count() == document.a.count() == document.b.count() == 10000
    index = document.index
    first_item = document.samples.item(0)
    owner.analysis_scope.set_mode('all')
    qtbot.waitUntil(lambda: not document.timer.isActive() and not document._loading, timeout=15000)
    assert document.index is index
    assert len(threads) == 2
    assert document.samples.item(0) is first_item
    owner.dirty = False


@pytest.mark.parametrize('change', ['data', 'scope', 'project'])
def test_initial_catalog_rejects_changes_during_widget_population(qtbot, monkeypatch, change):
    column = Column('Sample / Name')
    project = Project(columns=[column], rows=[Row({column.id: str(i)}) for i in range(10000)])
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)

    def progress(parent, title, work, *, prepare=None):
        if prepare is None:
            return work()
        next(prepare)
        if change == 'data':
            assert owner.model.setData(owner.model.index(0, 0), 'Updated sample')
        elif change == 'scope':
            owner.analysis_scope.set_mode('all')
        else:
            replacement = Project(columns=[column], rows=[Row({column.id: 'Replacement sample'})])
            owner.set_project(replacement)
        try:
            next(prepare)
        finally:
            prepare.close()
        pytest.fail('Population continued with stale inputs')

    monkeypatch.setattr(catalog, 'run_with_progress', progress)
    owner.show_geochemistry_workspace()
    document = owner.geochemistry_workspace_window
    assert document.index is None
    assert document.samples.count() == document.fields.count() == document.a.count() == document.b.count() == 0
    assert not document.export_passport_button.isEnabled()
    assert 'изменились' in document.status.text()
    assert document.refresh_button.isEnabled()
    def retry_progress(parent, title, work, *, prepare=None):
        if prepare is None:
            return work()
        for _ in prepare:
            pass
        return work(None)

    monkeypatch.setattr(catalog, 'run_with_progress', retry_progress)
    document.refresh_button.click()
    assert document.samples.count() == (1 if change == 'project' else 10000)
    expected = {'data': 'Updated sample', 'scope': '0', 'project': 'Replacement sample'}
    assert document.passport['sample']['name'] == expected[change]
    owner.dirty = False


def test_menu_document_passport_and_search_preserves_hidden_selection(workspace):
    owner, document, column = workspace
    assert owner.desktop.documents.currentWidget() is document
    assert owner.desktop.properties_stack.currentWidget() is document.parameter_tabs
    assert document.samples.count() == 3
    assert '<Core A>' in document.details.toPlainText()
    from autopyro.nonpyro import METHODS
    assert document.methods.rowCount() == len(METHODS)
    assert document.measurements.rowCount() >= 1
    select(document, [0, 1])
    passport = document.passport
    document.search.setText('Core B')
    assert document.samples.item(0).isHidden()
    assert len(document.selected_samples()) == 2
    assert document.passport is passport


def test_link_undo_group_and_measurement_selection(workspace, monkeypatch, qtbot):
    owner, document, column = workspace
    project = owner.model.project
    before = [(row.id, dict(row.values)) for row in project.rows]
    select(document, [0, 1])
    monkeypatch.setattr(QInputDialog, 'getText', lambda *args: ('Linked core', True))
    document.link_selected()
    assert len(GeochemistryIndex(project).samples) == 2
    assert [(row.id, dict(row.values)) for row in project.rows] == before
    linked = next(s for s in document.index.samples if s.startswith('link:'))
    assert document.index.value(linked, column.id)['status'] == 'ambiguous'
    owner.model.undo.undo()
    document.refresh()
    assert document.samples.count() == 3
    owner.model.undo.redo()
    document.refresh()
    select(document, [0])
    row = next(i for i in range(document.measurements.rowCount())
               if document.measurements.item(i, 0).data(Qt.ItemDataRole.UserRole)['column_id'] == column.id)
    document.measurements.selectRow(row)
    document.choose_measurement()
    assert document.index.value(linked, column.id)['status'] == 'numeric'
    document.save_selected_group()
    assert project.view[STATE_KEY]['groups']['Linked core'] == [linked]
    document.reset_links()
    assert document.samples.count() == 3


def test_compare_export_restore_and_invalidate(workspace, monkeypatch, tmp_path, qtbot):
    owner, document, column = workspace
    monkeypatch.setattr(comparison, 'run_with_progress', lambda parent, title, work: work())
    document.calculate_comparison()
    assert document.report is not None, document.status.text()
    assert document.comparison_table.rowCount() == 1
    assert document.report['results'][0]['difference'] == 2
    assert document.export_comparison_button.isEnabled()
    assert document.figure.axes[0].get_ylabel() == '%wt'
    for suffix in ('json', 'csv'):
        path = tmp_path / ('comparison.' + suffix)
        monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args, p=path: (str(p), ''))
        document.export_comparison()
        assert path.exists()
    assert json.loads((tmp_path / 'comparison.json').read_text('utf-8'))['results'][0]['difference'] == 2
    restored = GeochemistryWorkspaceDialog(owner)
    qtbot.addWidget(restored)
    assert restored.report is not None
    assert restored.export_comparison_button.isEnabled()
    owner.model.project.rows[0].values[column.id] = 3
    owner.model.project.sync_measurements()
    document.export_comparison()
    assert document.report is None
    assert not document.export_comparison_button.isEnabled()
    owner.model.notify_changed()
    qtbot.waitUntil(lambda: restored.report is None)


def test_pca_from_group_comparison_exports_and_invalidates(qtbot, monkeypatch, tmp_path):
    x, y = [Column(name, 'number') for name in ('X', 'Y')]
    project = Project(columns=[x, y], rows=[Row({x.id: a, y.id: b}) for a, b in [(1, 1), (2, 3), (3, 2)]])
    samples = list(GeochemistryIndex(project).samples)
    project.view[STATE_KEY] = {'groups': {'Reference': samples[:2]}}
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_geochemistry_workspace()
    document = owner.geochemistry_workspace_window
    assert not document.pca_button.isEnabled()
    document.a.setCurrentIndex(document.a.findData('group:Reference'))
    document.b.setCurrentIndex(document.b.findData('sample:' + samples[2]))
    for i in range(document.fields.count()):
        document.fields.item(i).setCheckState(Qt.CheckState.Checked)
    document.compare_button.click()
    assert document.pca_button.isEnabled(), document.status.text()
    document.pca_button.click()
    assert document.passport_tabs.currentWidget() is document.pca_page, document.status.text()
    assert document.pca_coefficients.rowCount() == 2
    assert [len(group.get_offsets()) for group in document.pca_canvas.figure.axes[0].collections] == [2, 1]
    monkeypatch.setattr(QInputDialog, 'getInt', lambda *args: (2, True))
    document.clustering_button.click()
    assert document.passport_tabs.currentWidget() is document.clustering_page, document.status.text()
    assert document.cluster_members.rowCount() == 3
    assert len(document.clustering_canvas.figure.axes[0].collections) > 0
    document.kmeans_button.click()
    assert document.kmeans_run is not None, document.status.text()
    assert len(document.clustering_canvas.figure.axes[0].lines) == 2
    monkeypatch.setattr(document, 'dbscan_parameters', lambda *args: (0.01, 2))
    document.dbscan_button.click()
    assert document.dbscan_run['noise_count'] == 3
    assert document.cluster_members.item(0, 3).text() == 'Шум'
    assert not document.cluster_members.isColumnHidden(4)
    output = tmp_path / 'pca.json'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(output), ''))
    document.export_comparison()
    exported = json.loads(output.read_text('utf-8'))
    assert exported['pca']['scores'] == document.pca_run['scores']
    assert exported['clustering']['membership'] == document.clustering_run['membership']
    assert exported['kmeans']['centers'] == document.kmeans_run['centers']
    assert exported['dbscan']['membership'] == document.dbscan_run['membership']
    document.clr.setChecked(True)
    assert document.report is not None and document.pca_run is None and document.dbscan_run is None
    document.clustering_button.click()
    assert document.clustering_run['preprocessing'] == 'clr', document.status.text()
    document.export_comparison()
    assert json.loads(output.read_text('utf-8'))['clustering']['composition']['transform'] == 'clr'
    document.fields.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert not document.clr.isChecked()
    assert document.pca_run is None
    assert not document.passport_tabs.isTabVisible(document.pca_tab)
    assert document.clustering_run is None
    assert document.kmeans_run is None
    assert document.dbscan_run is None
    assert not document.passport_tabs.isTabVisible(document.clustering_tab)
    document.show_model_archive()
    assert document.model_runs.rowCount() == 5
    assert 'CLR применяется' in document.model_passport.toPlainText()
    assert 'Число кластеров: 2' in document.model_passport.toPlainText()
    document.model_runs.selectRow(4)
    assert 'два независимых направления вариации' in document.model_passport.toPlainText()
    document.open_model_run()
    assert document.passport_tabs.currentWidget() is document.pca_page
    assert 'Архив' in document.passport_tabs.tabText(document.pca_tab)
    assert document.pca_run is None
    document.check_model_run()
    assert 'воспроизведён' in document.status.text()
    document.export_model_run()
    archive = json.loads(output.read_text('utf-8'))
    assert archive['run']['result']['format'] == 'autopyro.comparison_pca'
    assert archive['snapshot']['input_digest'] == archive['run']['result']['input_digest']
    owner.set_project(Project())
    owner.show_geochemistry_workspace()
    imported = owner.geochemistry_workspace_window
    imported.show_model_archive()
    assert imported.model_runs.rowCount() == 0
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(output), ''))
    imported.import_model_run()
    assert imported.model_runs.rowCount() == 1, imported.status.text()
    imported.import_model_run()
    assert imported.model_runs.rowCount() == 1
    assert 'уже есть' in imported.status.text()
    imported.open_model_run()
    assert imported.passport_tabs.currentWidget() is imported.pca_page
    assert 'Входы изменились' in imported.status.text()
    owner.dirty = False


@pytest.mark.parametrize('oil_correlation', [False, True])
def test_profile_review_reopens_with_comparison_and_exports(qtbot, monkeypatch, tmp_path, oil_correlation):
    columns = [Column(f'Sat-GC / nC{carbon}[a]', 'number', unit='count') for carbon in range(24, 35)]
    pristane = Column('Sat-GC / Pr/Ph[a]', 'number', unit='1')
    sterane = Column('Sat-GCMS / St29 20S/R[h]', 'number', unit='1')
    isotope = Column('Isotopes / d13C Sat [‰]', 'number', unit='‰')
    project = Project(columns=[*columns, pristane, sterane, isotope], rows=[
        Row({**{c.id: scale * i for i, c in enumerate(columns, 1)}, pristane.id: scale,
             sterane.id: 0.4 if scale == 1 else 'ND', isotope.id: -29 - scale / 2}) for scale in (1, 2)
    ])
    if oil_correlation:
        for row in project.rows:
            row.source['sample_type'] = 'oil'
    button = 'oil_correlation_button' if oil_correlation else 'profile_match_button'
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_geochemistry_workspace()
    document = owner.geochemistry_workspace_window
    for i in range(document.fields.count()):
        document.fields.item(i).setCheckState(Qt.CheckState.Checked)
    document.calculate_comparison()
    assert not document.correlation_isotope_basis.isChecked()
    if oil_correlation:
        document.correlation_isotope_basis.setChecked(True)
    getattr(document, button).click()
    assert document.passport_tabs.currentWidget() is document.profile_page, document.status.text()
    assert document.profile_match['overlap_percent'] == pytest.approx(100)
    assert document.profile_evidence.rowCount() == 3
    assert document.profile_evidence.item(0, 4).text() == '1'
    assert document.profile_evidence.item(1, 4).text() == '—'
    assert document.biodegradation['a'].currentData() == 'not_assessed'
    assert document.biodegradation['b'].currentData() == 'not_assessed'
    for side, value in (('a', 'evidence_noted'), ('b', 'insufficient_evidence')):
        combo = document.biodegradation[side]
        combo.setCurrentIndex(combo.findData(value))
    document.save_profile_review()
    assert 'Введите текст' in document.status.text()
    document.profile_note.setPlainText('Compare other biomarkers before assigning a family.')
    if oil_correlation:
        observed, source = document.biodegradation_observations['a', 'alkane_depletion']
        observed.setCurrentIndex(observed.findData('observed'))
        document.save_profile_review()
        assert 'Укажите источник анализа' in document.status.text()
        assert document.profile_views.currentWidget() is document.biodegradation_table
        source.setText('GC run 17, saturated fraction')
        absent, source = document.biodegradation_observations['b', 'alkane_depletion']
        absent.setCurrentIndex(absent.findData('not_observed'))
        source.setText('GC run 18, saturated fraction')
        observation, reason = document.profile_evidence_notes[pristane.id]
        assert sterane.id not in document.profile_evidence_notes
        observation.setCurrentIndex(observation.findData('b:alkane_depletion'))
        document.save_profile_review()
        assert 'Свяжите пометку' in document.status.text()
        observation.setCurrentIndex(observation.findData('a:alkane_depletion'))
        document.save_profile_review()
        assert 'Поясните влияние' in document.status.text()
        reason.setText('Проверить влияние изменения состава насыщенной фракции.')
    document.save_profile_review()
    output = tmp_path / 'profile.json'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(output), ''))
    document.export_comparison()
    payload = json.loads(output.read_text('utf-8'))
    assert payload['expert_review']['note'] == document.profile_note.toPlainText()
    assert payload['expert_review']['biodegradation'] == {'a': 'evidence_noted', 'b': 'insufficient_evidence'}
    assert payload['alkane_profile_match']['input_digest'] == payload['expert_review']['match']['input_digest']
    if oil_correlation:
        observations = {(item['side'], item['indicator']): item
                        for item in payload['expert_review']['biodegradation_observations']}
        assert observations['a', 'alkane_depletion'] == {
            'side': 'a', 'sample_id': document.report['request']['group_a'][0],
            'indicator': 'alkane_depletion', 'status': 'observed', 'source': 'GC run 17, saturated fraction',
        }
        assert observations['b', 'alkane_depletion']['status'] == 'not_observed'
        assert observations['b', 'unresolved_background']['status'] == 'insufficient_data'
        assert payload['expert_review']['evidence_notes'] == [{
            'field_id': pristane.id, 'side': 'a', 'indicator': 'alkane_depletion',
            'reason': 'Проверить влияние изменения состава насыщенной фракции.',
        }]
        assert payload['alkane_profile_match']['correlation']['expert_review_required']
        assert payload['alkane_profile_match']['correlation']['materials'] == ['oil', 'oil']
        assert payload['alkane_profile_match']['correlation']['isotope_basis'] == {
            'fraction': 'C15+', 'reference': 'VPDB',
        }
        assert payload['alkane_profile_match']['evidence'][2]['difference'] == pytest.approx(-0.5)
    evidence = payload['alkane_profile_match']['evidence']
    assert evidence[0]['difference'] == 1
    assert evidence[1]['b']['status'] == 'non_detect' and evidence[1]['difference'] is None
    path = tmp_path / 'review.apyro'
    project.save(path)
    owner.set_project(Project.load(path))
    owner.show_geochemistry_workspace()
    reopened = owner.geochemistry_workspace_window
    if oil_correlation:
        reopened.correlation_isotope_basis.setChecked(True)
    getattr(reopened, button).click()
    assert reopened.profile_note.toPlainText() == payload['expert_review']['note']
    assert {side: combo.currentData() for side, combo in reopened.biodegradation.items()} == payload['expert_review']['biodegradation']
    if oil_correlation:
        observation, reason = reopened.profile_evidence_notes[pristane.id]
        assert observation.currentData() == 'a:alkane_depletion'
        assert reason.text() == payload['expert_review']['evidence_notes'][0]['reason']
        for key, (combo, source) in reopened.biodegradation_observations.items():
            assert combo.currentData() == observations[key]['status']
            assert source.text() == observations[key]['source']
    reopened.show_review_archive()
    assert reopened.passport_tabs.currentWidget() is reopened.review_archive_page
    assert reopened.review_archive.rowCount() == 1
    assert 'Входы совпадают' in reopened.archived_review_text.toPlainText()
    if oil_correlation:
        assert 'GC run 17, saturated fraction' in reopened.archived_review_text.toPlainText()
        assert payload['expert_review']['evidence_notes'][0]['reason'] in reopened.archived_review_text.toPlainText()
    assert owner.model.setData(owner.model.index(0, 0), 99)
    qtbot.waitUntil(lambda: 'Входы изменились' in reopened.archived_review_text.toPlainText())
    assert payload['expert_review']['note'] in reopened.archived_review_text.toPlainText()
    reopened.export_archived_review()
    historical = json.loads(output.read_text('utf-8'))
    assert historical['biodegradation_observations'] == payload['expert_review']['biodegradation_observations']
    assert historical['evidence_notes'] == payload['expert_review']['evidence_notes']
    assert historical['snapshot']['input_digest'] == payload['alkane_profile_match']['input_digest']
    assert historical['match']['overlap_percent'] == pytest.approx(100)
    owner.dirty = False


def test_binary_oil_mixing_displays_residuals_and_reopens_archive(qtbot, monkeypatch):
    x, y = [Column(name, 'number', unit='mg/kg oil') for name in ('X', 'Y')]
    project = Project(columns=[x, y], rows=[
        Row({x.id: a, y.id: b}, source={'sample_type': 'oil'})
        for a, b in ((10, 40), (50, 0), (20, 30))
    ])
    first, second, mixture = GeochemistryIndex(project).samples
    project.view[STATE_KEY] = {'groups': {'Компоненты': [first, second]}}
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_geochemistry_workspace()
    document = owner.geochemistry_workspace_window
    document.a.setCurrentIndex(document.a.findData('group:Компоненты'))
    document.b.setCurrentIndex(document.b.findData('sample:' + mixture))
    for index in range(document.fields.count()):
        document.fields.item(index).setCheckState(Qt.CheckState.Checked)
    document.calculate_comparison()

    def accept_tolerances(dialog):
        for edit in dialog.findChildren(QLineEdit):
            edit.setText('1')
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, 'exec', accept_tolerances)
    document.oil_mixing_button.click()
    assert document.passport_tabs.currentWidget() is document.oil_mixing_table, document.status.text()
    assert document.oil_mixing_table.item(0, 7).text() == '0'
    assert '25%' in document.status.text()
    document.show_model_archive()
    document.open_model_run()
    assert document.passport_tabs.currentWidget() is document.oil_mixing_table, document.status.text()
    assert '25%' in document.status.text()
    owner.dirty = False


@pytest.mark.parametrize('name', ('Sat-GC / Pr/nC19[a]', 'Sat-GCMS / 25nor(177)/H30ab[h]'))
def test_oil_correlation_with_only_biomarkers_opens_evidence_table(qtbot, name, monkeypatch):
    ratio = Column(name, 'number', unit='1')
    project = Project(columns=[ratio], rows=[
        Row({ratio.id: 0.47}, source={'sample_type': 'oil'}),
        Row({ratio.id: 0.42}, source={'sample_type': 'rock', 'preparation': 'extract'}),
    ])
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_geochemistry_workspace()
    document = owner.geochemistry_workspace_window
    document.fields.item(0).setCheckState(Qt.CheckState.Checked)
    document.mixed.setChecked(True)
    document.calculate_comparison()
    document.oil_correlation_button.click()
    assert document.profile_views.currentWidget() is document.profile_evidence
    assert not document.profile_views.isTabEnabled(0)
    assert document.profile_match['overlap_percent'] is None
    assert document.profile_evidence.item(0, 4).text() == '-0.05'
    shown = []

    def capture(dialog):
        table = dialog.findChild(type(document.profile_evidence))
        dialog.findChild(QComboBox).setCurrentIndex(1)
        shown.extend([[table.item(row, col).text() for col in range(4)] for row in range(table.rowCount())])
        return dialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, 'exec', capture)
    before = document.profile_match
    document.sensitivity_button.click()
    assert shown[0][1:] == ['1', '—', '—']
    assert shown[1][1:] == ['0', '—', name]
    assert shown[2] == [f'A · {name}', '0', '—', name]
    assert document.profile_match is before
    assert not project.view.get('qc_exclusions')
    for row, (side, indicator) in enumerate(document.biodegradation_observations):
        measured = document.biodegradation_table.item(row, 4).text()
        if indicator == 'hopane_changes' and '25nor' in name:
            assert measured.endswith('0.47 [1]' if side == 'a' else '0.42 [1]')
            assert document.biodegradation_observations[side, indicator][0].currentData() == 'insufficient_data'
        else:
            assert not measured
    owner.dirty = False


def test_isotope_confirmation_invalidates_correlation_and_csv_exports_evidence(qtbot, monkeypatch, tmp_path):
    isotope = Column('Isotopes / d13C Sat [per mil]', 'number', unit='per mil')
    peak = Column('Sat-GC / nC24[a]', 'number', unit='count')
    project = Project(columns=[isotope, peak], rows=[
        Row({isotope.id: value, peak.id: 10}, source={'sample_type': 'oil'}) for value in (-28, -27)
    ])
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_geochemistry_workspace()
    dialog = owner.geochemistry_workspace_window
    assert all(dialog.fields.item(i).checkState() == Qt.CheckState.Unchecked
               for i in range(dialog.fields.count()))
    for i in range(dialog.fields.count()):
        dialog.fields.item(i).setCheckState(Qt.CheckState.Checked)
    dialog.calculate_comparison()
    assert not dialog.profile_match_button.isEnabled()
    dialog.correlation_isotope_basis.setChecked(True)
    dialog.oil_correlation_button.click()
    assert dialog.profile_match['overlap_issue']
    assert dialog.profile_match['evidence'][0]['difference'] == 1
    path = tmp_path / 'correlation.csv'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), ''))
    dialog.export_correlation_button.click()
    assert 'd13C Sat' in path.read_text('utf-8-sig')
    dialog.correlation_isotope_basis.setChecked(False)
    assert dialog.profile_match is None
    assert not dialog.export_correlation_button.isEnabled()
    assert dialog.export_comparison_button.isEnabled()
    owner.dirty = False


@pytest.mark.parametrize('missing', [False, True])
def test_correlation_preview_checks_ster_hop_pair_before_comparison(qtbot, monkeypatch, missing):
    peaks = [Column(f'Sat-GCMS / {peak}[h]', 'number', unit='count')
             for peak in (*STER_HOP_STERANES, *STER_HOP_HOPANES)]
    project = Project(columns=peaks, rows=[
        Row({column.id: 10 for column in peaks}, source={'sample_type': 'oil'}),
        Row({column.id: None if missing and column is peaks[-1] else 15 for column in peaks},
            source={'sample_type': 'rock', 'preparation': 'extract'}),
    ])
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_geochemistry_workspace()
    document = owner.geochemistry_workspace_window
    document.mixed.setChecked(True)
    document.field_preset.setCurrentIndex(document.field_preset.findData('ster_hop_from_peaks'))
    document.select_field_preset()
    seen = []
    original_exec = QDialog.exec

    def capture_preview(dialog):
        if dialog.windowTitle() == tr('Готовность пары к корреляции'):
            seen.append(dialog.findChild(QPlainTextEdit).toPlainText())
            return QDialog.DialogCode.Rejected
        return original_exec(dialog)

    monkeypatch.setattr(QDialog, 'exec', capture_preview)

    document.correlation_preview_button.click()

    assert len(seen) == 1, document.status.text()
    assert f'{int(not missing)} из 1' in seen[0]
    assert 'Ster/Hop · 17/16 [h]' in seen[0]
    assert ('B: ' + _status('missing' if missing else 'numeric')) in seen[0]
    assert document.report is None
    assert document.profile_match is None
    assert 'geochemistry_comparison' not in project.view

    document.calculate_comparison()
    assert document.oil_correlation_button.isEnabled() is not missing
    if missing:
        assert 'Сопоставимые показатели: 0' in document.correlation_readiness.text()
    else:
        document.oil_correlation_button.click()
        evidence = document.profile_match['evidence'][0]
        assert evidence['a']['status'] == evidence['b']['status'] == 'numeric'
    owner.dirty = False


def test_prediction_opens_with_holdout_metrics_and_applies_to_unmeasured_samples(qtbot, monkeypatch, tmp_path):
    well = Column('Well / Well Name')
    x, y = [Column(name, 'number') for name in ('X', 'Y')]
    project = Project(columns=[well, x, y], rows=[
        Row({well.id: f'W{i // 3 + 1}', x.id: i, y.id: 2 * i + 1}) for i in range(8)
    ] + [Row({well.id: 'W4', x.id: 8, y.id: None}), Row({well.id: 'W4', x.id: 'ND', y.id: None})])
    samples = list(GeochemistryIndex(project).samples)
    project.view[STATE_KEY] = {'groups': {'Train': samples[:6], 'Test': samples[6:8]}}
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_geochemistry_workspace()
    document = owner.geochemistry_workspace_window
    document.a.setCurrentIndex(document.a.findData('group:Train'))
    document.b.setCurrentIndex(document.b.findData('group:Test'))
    for i in range(document.fields.count()):
        document.fields.item(i).setCheckState(Qt.CheckState.Checked)
    document.compare_button.click()
    assert document.prediction_button.isEnabled(), document.status.text()
    document.prediction_cv.setChecked(True)
    monkeypatch.setattr(QInputDialog, 'getItem', lambda parent, title, prompt, labels, *args: (labels[1], True))
    document.prediction_button.click()
    assert document.passport_tabs.currentWidget() is document.prediction_page, document.status.text()
    assert document.prediction_table.item(1, 2).text() == '15'
    assert document.prediction_table.item(1, 3).text() == 'X'
    folds = document.prediction_run['well_cv']['folds']
    assert document.prediction_cv_table.rowCount() == 2
    assert [fold['train_means'] for fold in folds] == [[4], [1]]
    assert [fold['train_ids'] for fold in folds] == [samples[3:6], samples[:3]]
    assert document.prediction_run['well_cv']['metrics']['RMSE'] == pytest.approx(0, abs=1e-12)
    assert [fold['baseline']['value'] for fold in folds] == [9, 3]
    assert document.prediction_run['well_cv']['baseline_metrics']['RMSE'] == pytest.approx((116 / 3)**.5)
    assert document.prediction_run['baseline']['metrics']['RMSE'] == pytest.approx(65**.5)
    assert document.prediction_cv_table.item(0, 5).text() != '—'
    assert document.prediction_details.isTabVisible(2)
    assert document.prediction_well_table.rowCount() == 2
    assert document.prediction_well_table.item(1, 0).text() == 'Все скважины с равным весом'
    document.show_model_archive()
    assert 'Стандартизация только на A' in document.model_passport.toPlainText()
    assert 'Целевой показатель: Y' in document.model_passport.toPlainText()
    document.open_model_run()
    assert document.passport_tabs.currentWidget() is document.prediction_page
    document.check_model_run()
    assert 'воспроизведён' in document.status.text()
    select(document, [8, 9])
    document.show_model_archive()
    document.apply_archived_prediction()
    assert document.passport_tabs.currentWidget() is document.application_page, document.status.text()
    assert document.application_table.item(0, 1).text() == '17'
    assert document.application_table.item(0, 3).text() == 'X'
    assert document.prediction_application['results'][1]['excluded'] == {x.id: 'non_detect'}
    assert project.rows[8].values[y.id] is None
    output = tmp_path / 'applied.json'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(output), ''))
    document.export_prediction_application()
    assert json.loads(output.read_text('utf-8'))['results'][0]['predicted'] == pytest.approx(17)
    assert owner.model.setData(owner.model.index(8, 1), 7)
    qtbot.waitUntil(lambda: document.prediction_application is None)
    owner.dirty = False


def test_comparison_runs_in_real_worker(workspace):
    owner, document, column = workspace
    document.calculate_comparison()
    assert document.report is not None, document.status.text()
    assert document.report['results'][0]['difference'] == 2


def test_passport_export_is_fresh_and_replacing_project_closes_document(workspace, monkeypatch, tmp_path):
    owner, document, column = workspace
    path = tmp_path / 'passport.json'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), ''))
    document.export_passport()
    payload = json.loads(path.read_text('utf-8'))
    assert payload['sample']['name'] == '<Core A>'
    assert any(item['raw'] == 2 for item in payload['measurements'])
    owner.set_project(Project())
    assert owner.geochemistry_workspace_window is None
    assert owner.desktop.documents.count() == 1


def test_background_comparison_rejects_changed_inputs(workspace, monkeypatch):
    owner, document, column = workspace
    def change_during_calculation(parent, title, work):
        result = work()
        owner.model.project.rows[0].values[column.id] = 100
        owner.model.notify_changed()
        return result
    monkeypatch.setattr(comparison, 'run_with_progress', change_during_calculation)
    document.calculate_comparison()
    assert document.report is None
    assert 'изменились' in document.status.text()
    assert 'geochemistry_comparison' not in owner.model.project.view


def test_import_multiple_sets_and_repeats(qtbot, monkeypatch):
    from autopyro.database.ui import nonpyro
    dialog = nonpyro.NonPyroDatabaseDialog()
    qtbot.addWidget(dialog)
    calls = []
    def read_profiles(parameters, methods, **options):
        calls.append((methods, options))
        return Project()

    monkeypatch.setattr(nonpyro, 'read_profiles', read_profiles)
    monkeypatch.setattr(dialog, 'run_task', lambda work, done: work())
    dialog.combine.setChecked(True)
    dialog.include_repeats.setChecked(True)
    dialog.profile_list.item(1).setCheckState(Qt.CheckState.Checked)
    dialog.read()
    assert len(calls[0][0]) == 2
    assert calls[0][1]['include_repeats'] is True
    assert calls[0][1]['group'] == calls[0][1]['variant'] == ''

def test_demo_passports_show_location_and_search_by_well(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    window.show_geochemistry_workspace()
    document = window.geochemistry_workspace_window
    sample = document.passport['sample']
    assert sample['well'] == ['A']
    assert sample['intervals'][0]['top_md'] == 1000
    assert sample['intervals'][0]['base_md'] is None
    assert '1000' in document.samples.item(0).text()
    for well, count in (('A', 3), ('B', 2)):
        document.search.setText(well)
        assert sum(not document.samples.item(i).isHidden() for i in range(document.samples.count())) == count
    window.dirty = False
