"""Laboratory append acceptance: real files, Qt decisions, repeats and storage."""
import json
import time

import pytest
from openpyxl import Workbook
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import QFileDialog, QMessageBox

from autopyro.core import Project
from autopyro.ui.data_io.import_file import ImportDialog
from autopyro.ui.data_io.append_import import AppendImportDialog
from autopyro.ui.dialogs.measurement import MeasurementDialog
from autopyro.ui.window import MainWindow


def read_laboratory(qtbot, path, *, s2_unit="mg HC/g rock"):
    dialog = ImportDialog(path=path)
    qtbot.addWidget(dialog)
    map_laboratory(qtbot, dialog, s2_unit=s2_unit)
    return dialog.project


def map_laboratory(qtbot, dialog, *, s2_unit="mg HC/g rock"):
    dialog.load()
    qtbot.waitUntil(lambda: dialog.raw is not None and dialog.isEnabled(), timeout=10000)
    for index, unit in ((2, "%wt"), (3, s2_unit)):
        dialog.mapping.cellWidget(index, 3).setCurrentIndex(
            dialog.mapping.cellWidget(index, 3).findData("number"))
        dialog.mapping.item(index, 4).setText(unit)
    dialog.import_data()
    qtbot.waitUntil(lambda: dialog.result() == dialog.DialogCode.Accepted, timeout=10000)


def append_dialog(qtbot, owner, incoming):
    dialog = AppendImportDialog(owner, incoming)
    qtbot.addWidget(dialog)
    for index in (0, 1):
        dialog.mapping.cellWidget(index, 2).setChecked(True)
    return dialog


def compare(qtbot, dialog):
    dialog.calculate()
    qtbot.waitUntil(lambda: dialog.plan is not None and dialog.isEnabled(), timeout=10000)


def commit(qtbot, dialog):
    dialog.apply_import()
    qtbot.waitUntil(lambda: dialog.result() == dialog.DialogCode.Accepted, timeout=10000)


def test_main_window_opens_and_appends_tsv(qtbot, tmp_path, monkeypatch):
    owner = MainWindow()
    qtbot.addWidget(owner)
    original = tmp_path / 'original.TSV'
    original.write_text('Well\tSample\tTOC\tS2\nA\t001\t2\t5\n', encoding='utf-8')
    repeat = tmp_path / 'repeat.tsv'
    repeat.write_text('Well\tSample\tTOC\tS2\nB\t002\t2\t6\n', encoding='utf-8')
    chosen = iter((original, repeat, repeat))
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(next(chosen)), ''))
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: pytest.fail(str(args[-1])))

    def read(dialog):
        assert dialog.separator.currentText() == 'TAB'
        map_laboratory(qtbot, dialog)
        return dialog.result()

    monkeypatch.setattr(ImportDialog, 'exec', read)
    owner.import_file()
    project = owner.model.project
    assert owner.dirty and owner.path is None and len(project.rows) == 1

    confirmed = iter((False, True))

    def review(dialog):
        for index in (0, 1):
            dialog.mapping.cellWidget(index, 2).setChecked(True)
        compare(qtbot, dialog)
        assert dialog.plan.rows[0]['status'] == 'new'
        if next(confirmed):
            commit(qtbot, dialog)
        else:
            dialog.reject()
        return dialog.result()

    monkeypatch.setattr(AppendImportDialog, 'exec', review)
    with monkeypatch.context() as patch:
        patch.setattr(owner, 'may_replace', lambda: pytest.fail('Append must not prompt to replace the project'))
        owner.append_file()
        assert owner.model.project is project and len(project.rows) == 1
        assert not owner.model.undo.canUndo()

        owner.append_file()

    assert owner.model.project is project and len(project.rows) == 2
    assert project.rows[-1].values[project.columns[-1].id] == 6
    owner.model.undo.undo()
    assert len(project.rows) == 1


@pytest.mark.parametrize("background_snapshots,packed_snapshots", [(False, False), (True, False), (True, True)])
def test_xlsx_csv_append_conflicts_repeats_undo_reimport_and_reopen(qtbot, tmp_path, monkeypatch, background_snapshots, packed_snapshots):
    if background_snapshots:
        from autopyro.ui.table import snapshot_policy
        monkeypatch.setattr(snapshot_policy, "large_project", lambda project: True)
    if packed_snapshots:
        import autopyro.audit as audit
        from autopyro.ui.table import undo_state
        monkeypatch.setattr(undo_state, "MIN_PACK_BYTES", 0)
        monkeypatch.setattr(audit, "MIN_PROTOCOL_BYTES", 0)
    started = time.perf_counter()
    timings = {}
    xlsx = tmp_path / "initial.xlsx"
    workbook = Workbook()
    for row in [["Well", "Sample", "TOC", "S2"],
                ["A", "001", 2, 6], ["A", "001", 3, 7],
                ["B", "001", 2, 6], ["C", "002", 2, None], ["D", "003", 2, "ND"]]:
        workbook.active.append(row)
    workbook.save(xlsx)
    workbook.close()
    csv = tmp_path / "repeat.csv"
    csv.write_text("Well;Sample;TOC;S2\nA;001;3;8\nB;001;2;9\n"
                   "C;002;2;10\nD;003;2;11\nE;004;2;12\nE;004;2;13\n", encoding="utf-8")
    original_files = {path: path.read_bytes() for path in (xlsx, csv)}
    project = read_laboratory(qtbot, xlsx)
    incoming = read_laboratory(qtbot, csv)
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.may_replace = lambda: True
    sid = project.rows[1].id
    s2 = project.columns[3].id
    owner.table.selectionModel().select(owner.proxy.index(1, 3), QItemSelectionModel.SelectionFlag.Select)

    def assert_selected_cell():
        source, = (owner.proxy.mapToSource(index) for index in owner.table.selectionModel().selectedIndexes())
        assert (project.rows[source.row()].id, project.columns[source.column()].id) == (sid, s2)

    original = project.science.current(sid, s2).id
    assert project.rows[0].values[project.columns[1].id] == "001"
    timings["read_files_and_table_seconds"] = time.perf_counter() - started
    try:
        started = time.perf_counter()
        dialog = append_dialog(qtbot, owner, incoming)
        dialog.laboratory.setText("Lab B")
        dialog.method.setText("Pyrol6")
        dialog.policy.setCurrentIndex(dialog.policy.findData("fill_missing"))
        compare(qtbot, dialog)
        assert [item["status"] for item in dialog.plan.rows] == [
            "ambiguous", "matched", "matched", "matched", "input_repeat", "input_repeat"]
        # Resolve an ambiguous existing sample and explicitly keep two new samples separate.
        target = dialog.rows.cellWidget(0, 3)
        target.setCurrentIndex(target.findData(sid))
        action = dialog.rows.cellWidget(0, 4)
        action.setCurrentIndex(action.findData("append"))
        for index in (4, 5):
            action = dialog.rows.cellWidget(index, 4)
            action.setCurrentIndex(action.findData("new"))
        commit(qtbot, dialog)
        assert_selected_cell()
        assert [row.values[s2] for row in project.rows] == [6, 7, 6, 10, "ND", 12, 13]
        repeat = next(m for m in project.science.results(sid, s2) if m.id != original)
        assert repeat.value == 8
        assert project.science.analyses[repeat.analysis_id].laboratory == "Lab B"
        assert project.science.sources[repeat.source_id].provenance["row"] == 2
        card = MeasurementDialog(owner.model, sid, s2, owner)
        qtbot.addWidget(card)
        card.table.selectRow(next(i for i in range(card.table.rowCount())
                                  if card.table.item(i, 0).data(Qt.ItemDataRole.UserRole) == repeat.id))
        card.select_result()
        assert project.row_map()[sid].values[s2] == 8
        owner.model.undo.undo()  # Working result.
        owner.model.undo.undo()  # Whole laboratory batch.
        assert_selected_cell()
        assert len(project.rows) == 5 and project.science.current(sid, s2).id == original
        assert repeat.id not in project.science.measurements
        owner.model.undo.redo()
        owner.model.undo.redo()
        assert_selected_cell()
        assert project.science.current(sid, s2).id == repeat.id
        timings["append_select_undo_redo_seconds"] = time.perf_counter() - started

        # A differing unit must be reviewed; it cannot overwrite the original property.
        units_file = tmp_path / "other_units.csv"
        units_file.write_text("Well;Sample;TOC;S2\nB;001;2;9000\n", encoding="utf-8")
        original_files[units_file] = units_file.read_bytes()
        other_units = read_laboratory(qtbot, units_file, s2_unit="ug HC/g rock")
        units = append_dialog(qtbot, owner, other_units)
        errors = []
        monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, message: errors.append(message))
        mapping = units.mapping.cellWidget(3, 1)
        mapping.setCurrentIndex(mapping.findData(s2))
        units.calculate()
        qtbot.waitUntil(lambda: bool(errors) and units.isEnabled(), timeout=10000)
        assert "единица" in errors[0] and units.plan is None
        assert len(project.columns) == 4
        mapping.setCurrentIndex(mapping.findData("new"))
        compare(qtbot, units)
        commit(qtbot, units)
        separate = project.columns[-1]
        assert separate.name == "S2 (2)" and separate.unit == "ug HC/g rock"
        assert project.rows[2].values[s2] == 6 and project.rows[2].values[separate.id] == 9000

        path = tmp_path / "laboratory.apyro"
        project.save(path)
        restored = Project.load(path)
        owner.set_project(restored)
        assert restored.science.current(sid, s2).id == repeat.id
        assert restored.rows[4].values[s2] == "ND"
        counts = (len(restored.rows), len(restored.columns), len(restored.science.measurements))
        for batch in (incoming, other_units):
            again = append_dialog(qtbot, owner, batch)
            compare(qtbot, again)
            assert all(item["status"] == "duplicate" for item in again.plan.rows)
            commit(qtbot, again)
            assert (len(restored.rows), len(restored.columns), len(restored.science.measurements)) == counts
            protocol = list(restored.view["import_runs"].values())[-1]["protocol"]
            assert protocol["skipped"] == len(batch.rows) and protocol["measurements"] == 0
        assert all(path.read_bytes() == content for path, content in original_files.items())
        (tmp_path / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    finally:
        owner.dirty = False

def test_append_two_analyses_to_one_new_sample(qtbot, tmp_path):
    original = tmp_path / 'original.csv'
    original.write_text('Well;Sample;TOC;S2\nA;001;2;5\n', encoding='utf-8')
    repeat = tmp_path / 'repeat.csv'
    repeat.write_text('Well;Sample;TOC;S2\nE;004;2;12\nE;004;2;13\n', encoding='utf-8')
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(read_laboratory(qtbot, original))
    incoming = read_laboratory(qtbot, repeat)
    dialog = append_dialog(qtbot, owner, incoming)
    compare(qtbot, dialog)
    for row, action in enumerate(('new', 'append_new')):
        choice = dialog.rows.cellWidget(row, 4)
        choice.setCurrentIndex(choice.findData(action))
    target = dialog.rows.cellWidget(1, 3)
    target.setCurrentIndex(target.findData(incoming.rows[0].id))
    commit(qtbot, dialog)
    project = owner.model.project
    assert len(project.rows) == 2
    sample_id, s2 = project.rows[-1].id, project.columns[3].id
    assert sorted(m.value for m in project.science.results(sample_id, s2)) == [12, 13]
    owner.model.undo.undo()
    assert len(owner.model.project.rows) == 1
    owner.model.undo.redo()
    path = tmp_path / 'combined.apyro'
    owner.model.project.save(path)
    restored = Project.load(path)
    assert sorted(m.value for m in restored.science.results(sample_id, s2)) == [12, 13]
    again = append_dialog(qtbot, owner, incoming)
    compare(qtbot, again)
    assert [item['status'] for item in again.plan.rows] == ['duplicate', 'duplicate']
    owner.dirty = False
