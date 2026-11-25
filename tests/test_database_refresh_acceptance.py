"""Refresh acceptance against a real SQLite source through the Qt workflow."""
from contextlib import closing
import sqlite3

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QMessageBox, QTextEdit

from autopyro.calculations import make_calculated_column
from autopyro.core import Project
from autopyro.database.readers import read_sqlite
from autopyro.database.ui.refresh import DatabaseRefreshDialog
from autopyro.database.ui.snapshot import DatabaseDialog
from autopyro.io import build_project
from autopyro.ui.data_io.import_file import ImportDialog
from autopyro.ui.i18n import tr
from autopyro.ui.table.commands import QCCommand, SelectMeasurementCommand
from autopyro.ui.window import MainWindow


def read_refresh(qtbot, owner):
    dialog = DatabaseRefreshDialog(owner)
    qtbot.addWidget(dialog)
    if dialog.keys.isEnabled():
        dialog.keys.item(0).setCheckState(Qt.CheckState.Checked)
    dialog.read()
    qtbot.waitUntil(lambda: dialog.plan is not None and dialog.isEnabled(), timeout=10000)
    return dialog


def apply_refresh(qtbot, dialog):
    dialog.apply_refresh()
    qtbot.waitUntil(lambda: dialog.result() == dialog.DialogCode.Accepted, timeout=10000)


def test_main_window_imports_refreshes_and_displays_snapshot_log(qtbot, tmp_path, monkeypatch):
    path = tmp_path / 'source.sqlite'
    with closing(sqlite3.connect(path)) as connection:
        connection.execute('CREATE TABLE lab (Sample TEXT PRIMARY KEY, S2 INTEGER)')
        connection.execute('INSERT INTO lab VALUES (?, ?)', ('A-1', 6))
        connection.commit()

    owner = MainWindow()
    qtbot.addWidget(owner)
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: pytest.fail(str(args[-1])))

    def read_source(dialog):
        dialog.fields['path'].setText(str(path))
        dialog.fields['table'].setText('lab')
        dialog.read()
        qtbot.waitUntil(lambda: dialog.result() == QDialog.DialogCode.Accepted, timeout=10000)
        return dialog.result()

    def map_source(dialog):
        kind = dialog.mapping.cellWidget(1, 3)
        kind.setCurrentIndex(kind.findData('number'))
        dialog.import_data()
        qtbot.waitUntil(lambda: dialog.result() == QDialog.DialogCode.Accepted, timeout=10000)
        return dialog.result()

    monkeypatch.setattr(DatabaseDialog, 'exec', read_source)
    monkeypatch.setattr(ImportDialog, 'exec', map_source)
    original_bytes = path.read_bytes()
    owner.import_database()
    project = owner.model.project
    s2 = project.columns[1].id
    assert owner.dirty and project.source['read_only']
    assert project.rows[0].values[s2] == 6
    assert path.read_bytes() == original_bytes

    with closing(sqlite3.connect(path)) as connection:
        connection.execute('UPDATE lab SET S2 = 8')
        connection.commit()
    updated_bytes = path.read_bytes()

    def refresh(dialog):
        dialog.keys.item(0).setCheckState(Qt.CheckState.Checked)
        dialog.read()
        qtbot.waitUntil(lambda: dialog.plan is not None and dialog.isEnabled(), timeout=10000)
        assert dialog.apply_button.isEnabled()
        apply_refresh(qtbot, dialog)
        return dialog.result()

    monkeypatch.setattr(DatabaseRefreshDialog, 'exec', refresh)
    owner.refresh_database()
    assert owner.model.project is project and project.rows[0].values[s2] == 8
    assert path.read_bytes() == updated_bytes

    log = []

    def inspect_log(dialog):
        text = dialog.findChild(QTextEdit)
        assert text.isReadOnly()
        log.append(text.toPlainText())
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, 'exec', inspect_log)
    owner.database_refresh_log()
    assert tr('применено') in log[-1] and 'A-1' in log[-1] and 'S2' in log[-1]
    owner.model.undo.undo()
    assert project.rows[0].values[s2] == 6
    owner.database_refresh_log()
    assert tr('отменено') in log[-1]


@pytest.mark.parametrize("background_snapshots,packed_snapshots", [(False, False), (True, False), (True, True)])
def test_read_only_refresh_preserves_local_choices_qc_history_and_reopens(qtbot, tmp_path, monkeypatch, background_snapshots, packed_snapshots):
    if background_snapshots:
        from autopyro.ui.table import snapshot_policy
        monkeypatch.setattr(snapshot_policy, "large_project", lambda project: True)
    if packed_snapshots:
        import autopyro.audit as audit
        from autopyro.ui.table import undo_state
        monkeypatch.setattr(undo_state, "MIN_PACK_BYTES", 0)
        monkeypatch.setattr(audit, "MIN_PROTOCOL_BYTES", 0)
    path = tmp_path / "source.sqlite"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE lab (id TEXT PRIMARY KEY, Well TEXT, Sample TEXT, TOC TEXT, S2 TEXT)")
        connection.executemany("INSERT INTO lab VALUES (?,?,?,?,?)", [
            ("001", "A", "001", "2", "6"), ("002", "B", "001", "3", "9"),
            ("003", "C", "002", "2", "ND"), ("004", "D", "003", "1", "4")])
        connection.commit()
    headers, rows, numbers = read_sqlite(path, "lab")
    source = {"type": "sqlite", "path": str(path), "table": "lab", "read_only": True,
              "source_columns": headers, "imported_at": "initial"}
    units = ["", "", "", "%wt", "mg HC/g rock"]
    project = build_project(headers, rows, numbers,
        [(i, name, "text" if i < 3 else "number", units[i]) for i, name in enumerate(headers)], source)
    project.sync_measurements()
    toc, s2 = project.columns[3:5]
    hi, values = make_calculated_column(project, "HI", {
        "formula": "HI", "version": 1, "analysis_context": "synthetic QA", "compatible_confirmed": True,
        "inputs": {"TOC": {"column_id": toc.id, "unit": toc.unit},
                   "S2": {"column_id": s2.id, "unit": s2.unit}}})
    project.columns.append(hi)
    for row in project.rows:
        row.values[hi.id] = values[row.id]
    ids = [row.id for row in project.rows]
    initial = project.science.current(ids[0], s2.id).id
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.may_replace = lambda: True
    try:
        repeat = project.science.add_result(ids[0], s2.id, 8,
                    analysis_fields={"laboratory": "Local lab"}, details={"uncertainty": .3})
        owner.model.undo.push(SelectMeasurementCommand(owner.model, ids[0], s2.id, repeat.id))
        owner.model.undo.push(QCCommand(owner.model, ("qc_exclusions", "calculations"), {
            repeat.id: {"revision": repeat.revision, "comment": "Verify local repeat"}}))
        owner.model.setData(owner.model.index(1, 4), 10)
        assert project.rows[0].values[hi.id] is None
        local_edit = project.science.current(ids[1], s2.id).id
        # Simulate external laboratory changes, independently of the read-only app.
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("UPDATE lab SET S2='8' WHERE id='001'")
            connection.execute("UPDATE lab SET S2='11' WHERE id='002'")
            connection.execute("DELETE FROM lab WHERE id='004'")
            connection.execute("INSERT INTO lab VALUES ('005','E','004','2','12')")
            connection.commit()
        source_bytes = path.read_bytes()
        dialog = read_refresh(qtbot, owner)
        assert path.read_bytes() == source_bytes
        assert [entry["status"] for entry in dialog.plan.cells] == ["converged", "conflict"]
        assert len(dialog.plan.new_rows) == len(dialog.plan.missing_rows) == 1
        assert not dialog.apply_button.isEnabled()
        for index, entry in enumerate(dialog.entries):
            choice = dialog.table.cellWidget(index, 6)
            if entry.get("sample_id") == ids[0]:
                choice.setCurrentIndex(choice.findData("source"))
            elif entry.get("sample_id") in (ids[1], ids[3]):
                choice.setCurrentIndex(choice.findData("keep"))
        assert dialog.apply_button.isEnabled()
        apply_refresh(qtbot, dialog)
        assert path.read_bytes() == source_bytes
        assert [row.id for row in project.rows[:4]] == ids
        new_id = project.rows[-1].id
        assert [row.values[s2.id] for row in project.rows] == [8, 10, "ND", 4, 12]
        assert project.rows[0].values[hi.id] == 400
        assert project.rows[1].values[hi.id] == pytest.approx(1000 / 3)
        assert project.rows[2].values[hi.id] is None
        remote = project.science.current(ids[0], s2.id).id
        assert remote not in (initial, repeat.id)
        assert project.science.current(ids[1], s2.id).id == local_edit
        assert project.science.measurements[repeat.id].uncertainty == .3
        assert project.rows[0].original[s2.id] == "6"
        owner.model.undo.undo()
        assert "database_snapshot" not in project.view and len(project.rows) == 4
        assert project.science.current(ids[0], s2.id).id == repeat.id
        assert project.rows[0].values[hi.id] is None
        owner.model.undo.redo()
        assert project.rows[-1].id == new_id and project.science.current(ids[0], s2.id).id == remote
        assert project.rows[0].values[hi.id] == 400
        saved = tmp_path / "refreshed.apyro"
        project.save(saved)
        restored = Project.load(saved)
        owner.set_project(restored)
        assert restored.science.current(ids[0], s2.id).id == remote
        assert restored.science.measurements[repeat.id].uncertainty == .3
        assert any(not entry["present"] and entry["sample_id"] == ids[3]
                   for entry in restored.view["database_snapshot"]["rows"].values())
        count = len(restored.science.measurements)
        again = read_refresh(qtbot, owner)
        assert [entry["status"] for entry in again.plan.cells] == ["local_only"]
        assert not again.plan.new_rows and not again.plan.missing_rows
        apply_refresh(qtbot, again)
        assert len(restored.science.measurements) == count
        protocol = list(restored.view["database_refresh_runs"].values())[-1]["protocol"]
        assert not protocol["working_changed"]
        assert path.read_bytes() == source_bytes
        for save in (restored.save, restored.save_recovery):
            with pytest.raises(ValueError, match="только для чтения"):
                save(path)
            assert path.read_bytes() == source_bytes
    finally:
        owner.dirty = False
