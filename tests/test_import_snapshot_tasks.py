"""Large import snapshots run in workers; model mutations and undo stay in Qt."""
from contextlib import closing
from copy import deepcopy
import sqlite3

import pytest
from PySide6.QtCore import QThread, QTimer

from autopyro.database.refresh import preview_refresh, prepare_refresh
from autopyro.database.sources import read_snapshot
from autopyro.importing import digest, preview_import, prepare_import, suggest_mapping
from autopyro.io import build_project, read_sqlite
from autopyro.ui.table import commands, snapshot_policy
from autopyro.ui.table import model as table_model


@pytest.fixture
def laboratory(tmp_path):
    path = tmp_path / "source.sqlite"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE lab (Sample TEXT PRIMARY KEY, TOC TEXT, S2 TEXT)")
        connection.executemany("INSERT INTO lab VALUES (?,?,?)", [("001", "2", "6"), ("002", "3", "ND")])
        connection.commit()
    headers, rows, numbers = read_sqlite(path, "lab")
    source = {"type": "sqlite", "path": str(path), "table": "lab", "source_columns": headers, "read_only": True}
    project = build_project(headers, rows, numbers,
        [(0, "Sample", "text", ""), (1, "TOC", "number", "%wt"), (2, "S2", "number", "mg HC/g rock")], source)
    model = table_model.TableModel(project)
    project.columns[0].metadata["tags"] = ["original"]
    measurement = project.science.current(project.rows[0].id, project.columns[2].id)
    measurement.history.append({"previous": {"notes": ["original"]}})
    return path, model


def prepared_change(path, model, kind):
    project = model.project
    if kind == "import":
        incoming = build_project(["Sample", "TOC", "S2"], [["001", "2", "8"], ["003", "2", "12"]], [2, 3],
            [(i, c.name, c.kind, c.unit) for i, c in enumerate(project.columns)], {"sha256": "second"})
        plan = preview_import(project, incoming, suggest_mapping(project, incoming), [incoming.columns[0].id])
        prepared, protocol = prepare_import(project, incoming, plan, working_policy="replace")
        return commands.ImportCommand, prepared, protocol, plan.project_digest
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("UPDATE lab SET S2='8' WHERE Sample='001'")
        connection.execute("INSERT INTO lab VALUES ('003','2','12')")
        connection.commit()
    incoming = read_snapshot(project.source)
    plan = preview_refresh(project, incoming, [project.columns[0].id])
    prepared, protocol = prepare_refresh(project, incoming, plan)
    return commands.RefreshDatabaseCommand, prepared, protocol, plan.project_digest


def test_input_snapshot_runs_in_worker_and_is_independent(laboratory, monkeypatch, qtbot):
    _, model = laboratory
    project = model.project
    project._storage_digest = "private storage marker"
    main_thread = QThread.currentThread()
    threads = []
    copy_project = table_model.copy_project

    def tracked_copy(value):
        threads.append(QThread.currentThread())
        return copy_project(value)

    monkeypatch.setattr(snapshot_policy, "large_project", lambda project: True)
    monkeypatch.setattr(table_model, "copy_project", tracked_copy)
    copied = table_model.snapshot_project(project)
    assert threads and all(thread != main_thread for thread in threads)
    assert digest(copied) == digest(project)
    assert copied._storage_digest == project._storage_digest
    copied.columns[0].metadata["tags"].append("changed")
    assert project.columns[0].metadata["tags"] == ["original"]
    measurement = copied.science.current(copied.rows[0].id, copied.columns[2].id)
    measurement.history[0]["previous"]["notes"].append("changed")
    original = project.science.current(project.rows[0].id, project.columns[2].id)
    assert original.history[0]["previous"]["notes"] == ["original"]


@pytest.mark.parametrize("kind", ["import", "refresh"])
@pytest.mark.parametrize("packed", [False, True])
def test_snapshot_workers_preserve_undo_and_do_not_reset_models_from_worker(laboratory, monkeypatch, qtbot, kind, packed):
    if packed:
        from autopyro.ui.table import undo_state
        monkeypatch.setattr(undo_state, "MIN_PACK_BYTES", 0)
    path, model = laboratory
    command_type, prepared, protocol, expected = prepared_change(path, model, kind)
    main_thread = QThread.currentThread()
    copy_threads, restore_threads, reset_threads = [], [], []
    original_reset = model.beginResetModel
    copy_table_state = commands.copy_table_state
    restore_table_state = commands.restore_table_state

    def tracked_copy(value):
        copy_threads.append(QThread.currentThread())
        return copy_table_state(value)

    def tracked_reset():
        reset_threads.append(QThread.currentThread())
        original_reset()

    def tracked_restore(value):
        restore_threads.append(QThread.currentThread())
        return restore_table_state(value)

    monkeypatch.setattr(snapshot_policy, "large_project", lambda project: True)
    monkeypatch.setattr(commands, "copy_table_state", tracked_copy)
    monkeypatch.setattr(commands, "restore_table_state", tracked_restore)
    monkeypatch.setattr(model, "beginResetModel", tracked_reset)
    command = command_type(model, prepared, protocol, expected)
    if packed:
        assert isinstance(command.before, undo_state.PackedTableState)
        assert isinstance(command.after, undo_state.PackedTableState)
    prepared.columns[0].metadata["tags"].append("outside command")
    sid, pid = model.project.rows[0].id, model.project.columns[2].id
    old_id = model.project.science.current(sid, pid).id
    model.undo.push(command)
    assert len(model.project.rows) == 3
    assert model.project.rows[0].values[pid] == 8
    assert model.project.columns[0].metadata["tags"] == ["original"]
    model.project.columns[0].metadata["tags"].append("active edit")
    model.undo.undo()
    assert len(model.project.rows) == 2 and model.project.rows[0].values[pid] == 6
    assert model.project.science.current(sid, pid).id == old_id
    assert model.project.columns[0].metadata["tags"] == ["original"]
    model.project.columns[0].metadata["tags"].append("edit after undo")
    model.undo.redo()
    assert model.project.rows[0].values[pid] == 8
    assert model.project.columns[0].metadata["tags"] == ["original"]
    assert len(copy_threads) == 2 and all(thread != main_thread for thread in copy_threads)
    assert len(restore_threads) == 3 and all(thread != main_thread for thread in restore_threads)
    assert len(reset_threads) == 3 and all(thread == main_thread for thread in reset_threads)


@pytest.mark.parametrize("kind", ["import", "refresh"])
def test_change_during_modal_snapshot_is_not_committed(laboratory, monkeypatch, qtbot, kind):
    path, model = laboratory
    command_type, prepared, protocol, expected = prepared_change(path, model, kind)
    monkeypatch.setattr(snapshot_policy, "large_project", lambda project: True)
    QTimer.singleShot(0, lambda: model.setData(model.index(0, 2), 13))
    with pytest.raises(ValueError, match="измен"):
        command_type(model, prepared, protocol, expected)
    assert len(model.project.rows) == 2
    assert model.project.rows[0].values[model.project.columns[2].id] == 13
    assert model.undo.count() == 1  # Only the local edit, never the rejected import.


@pytest.mark.parametrize("kind", ["import", "refresh"])
@pytest.mark.parametrize("archived", [False, True])
def test_protocol_copies_stay_in_worker_and_history_is_independent(laboratory, monkeypatch, qtbot, kind, archived):
    from autopyro.audit import read_protocol
    if archived:
        import autopyro.audit as audit
        monkeypatch.setattr(audit, "MIN_PROTOCOL_BYTES", 0)
    path, model = laboratory
    command_type, prepared, protocol, expected = prepared_change(path, model, kind)
    monkeypatch.setattr(snapshot_policy, "large_project", lambda project: True)
    command = command_type(model, prepared, protocol, expected)
    main_thread = QThread.currentThread()
    threads = []
    timestamps = []
    timestamp = commands.timestamp

    def commit_timestamp():
        assert QThread.currentThread() == main_thread
        value = timestamp()
        timestamps.append(value)
        return value

    def tracked_copy(value):
        if value is command.protocol:
            threads.append(QThread.currentThread())
        return deepcopy(value)

    monkeypatch.setattr(commands, "deepcopy", tracked_copy)
    monkeypatch.setattr(commands, "timestamp", commit_timestamp)
    registry = "import_runs" if kind == "import" else "database_refresh_runs"
    model.undo.push(command)
    run = model.project.view[registry][protocol["id"]]
    event = model.project.history[-1]
    assert run["protocol"] == read_protocol(event["protocol"]) == command.protocol
    if archived:
        assert isinstance(event["protocol"], str)
    assert event["action"] == ("import" if kind == "import" else "refresh_database")
    assert event["at"]
    run["protocol"]["rows" if kind == "import" else "changes"].clear()
    assert read_protocol(event["protocol"]) == command.protocol
    details = read_protocol(event["protocol"])
    details["source"]["external edit"] = True
    assert "external edit" not in command.protocol["source"]
    assert "external edit" not in read_protocol(event["protocol"])["source"]
    model.undo.undo()
    undone = model.project.history[-1]
    assert undone["action"] == ("undo_import" if kind == "import" else "undo_refresh_database")
    assert read_protocol(undone["protocol"]) == command.protocol
    assert model.project.view[registry][protocol["id"]]["status"] == "undone"
    model.undo.redo()
    assert model.project.view[registry][protocol["id"]]["status"] == "applied"
    assert read_protocol(model.project.history[-1]["protocol"]) == command.protocol
    if archived:
        assert event["protocol"] is undone["protocol"] is model.project.history[-1]["protocol"]
    assert [event["at"], undone["at"], model.project.history[-1]["at"]] == timestamps
    assert threads and all(thread != main_thread for thread in threads)
