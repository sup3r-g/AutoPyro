"""Three-way refresh against a real temporary SQLite source, always read-only in the app."""
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3

import pytest
from PySide6.QtCore import Qt

from autopyro.core import Project
from autopyro.io import build_project, read_sqlite
from autopyro.database_sources import read_snapshot, descriptor
from autopyro.database_refresh import preview_refresh, prepare_refresh


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "source.sqlite"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute('CREATE TABLE lab (id TEXT PRIMARY KEY, TOC TEXT, S2 TEXT)')
        connection.executemany('INSERT INTO lab VALUES (?,?,?)', [("001", "2", "6"), ("002", "3", "9")])
        connection.commit()
    source = {"type": "sqlite", "path": str(path), "table": "lab", "read_only": True, "imported_at": "initial"}
    headers, rows, numbers = read_sqlite(path, "lab")
    source["source_columns"] = headers
    project = build_project(headers, rows, numbers,
        [(0, "id", "text", ""), (1, "TOC", "number", "%wt"), (2, "S2", "number", "mg HC/g rock")], source)
    project.sync_measurements()
    return path, project


def execute(path, statement, values=()):
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(statement, values)
        connection.commit()


def preview(project):
    incoming = read_snapshot(project.source)
    return incoming, preview_refresh(project, incoming, [project.columns[0].id])


def test_refresh_validation_does_not_search_row_lists_for_each_change(database):
    class NoMembershipScan(list):
        def __contains__(self, entry):
            raise AssertionError("Refresh must not scan row lists to classify every change")

    path, project = database
    execute(path, "UPDATE lab SET S2='8' WHERE id='001'")
    execute(path, "DELETE FROM lab WHERE id='002'")
    execute(path, "INSERT INTO lab VALUES ('003','4','12')")
    incoming, plan = preview(project)
    plan.new_rows = NoMembershipScan(plan.new_rows)
    plan.missing_rows = NoMembershipScan(plan.missing_rows)
    result, protocol = prepare_refresh(project, incoming, plan, {plan.missing_rows[0]["id"]: "keep"})
    assert [row.id for row in result.rows[:2]] == [row.id for row in project.rows]
    assert [row.values[project.columns[2].id] for row in result.rows] == [8, 9, 12]
    assert len(protocol["changes"]) == 3


def test_source_only_change_is_new_measurement_and_never_writes_database(database):
    path, project = database
    execute(path, 'UPDATE lab SET S2=? WHERE id=?', ("8", "001"))
    before_bytes, before_project = path.read_bytes(), asdict(deepcopy(project))
    incoming, plan = preview(project)
    assert len(plan.cells) == 1 and plan.cells[0]["status"] == "source_changed"
    result, protocol = prepare_refresh(project, incoming, plan)
    assert asdict(project) == before_project
    assert path.read_bytes() == before_bytes
    sid, pid = project.rows[0].id, project.columns[2].id
    assert result.rows[0].id == sid and result.rows[0].values[pid] == 8
    assert result.rows[0].original[pid] == "6"
    assert len(result.science.results(sid, pid)) == 2
    assert protocol["working_changed"]
    assert result.view["database_snapshot"]["checked_at"]


def test_refresh_resolves_shared_database_path_outside_row_loops(database, monkeypatch):
    path, original = database
    with closing(sqlite3.connect(path)) as connection:
        connection.executemany('INSERT INTO lab VALUES (?,?,?)',
                               [(str(i), "2", "6") for i in range(3, 53)])
        connection.commit()
    headers, rows, numbers = read_sqlite(path, "lab")
    project = build_project(headers, rows, numbers,
        [(i, c.name, c.kind, c.unit) for i, c in enumerate(original.columns)], original.source)
    project.sync_measurements()
    execute(path, "UPDATE lab SET S2='8'")
    incoming = read_snapshot(project.source)
    resolve = Path.resolve
    resolved = []

    def counted_resolve(self, *args, **kwargs):
        resolved.append(self)
        return resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", counted_resolve)
    plan = preview_refresh(project, incoming, [project.columns[0].id])
    result, _ = prepare_refresh(project, incoming, plan)
    assert len(plan.cells) == len(project.rows)
    assert all(row.values[project.columns[2].id] == 8 for row in result.rows)
    assert len(resolved) <= 10  # Independent of the number of source rows.


def test_source_row_checks_table_and_resolves_alternative_path_spelling(database):
    from autopyro.database.refresh import source_row
    _, project = database
    row = deepcopy(project.rows[0])
    row.source["path"] = str(Path(row.source["path"]).parent / "unused" / ".." / "source.sqlite")
    assert source_row(project, row)
    row.source["table"] = "other_lab"
    assert not source_row(project, row)


def test_unchanged_source_keeps_local_edit_and_can_explicitly_restore_source(database):
    _, project = database
    pid = project.columns[2].id
    project.rows[0].values[pid] = 7
    project.refresh()
    incoming, plan = preview(project)
    assert plan.cells[0]["status"] == "local_only"
    result, _ = prepare_refresh(project, incoming, plan)
    assert result.rows[0].values[pid] == 7
    restored, _ = prepare_refresh(project, incoming, plan, {plan.cells[0]["id"]: "source"})
    assert restored.rows[0].values[pid] == 6
    assert len(restored.science.results(project.rows[0].id, pid)) == 2


@pytest.mark.parametrize("decision,expected", [("keep", 7), ("source", 8)])
def test_two_sided_conflict_requires_a_decision(database, decision, expected):
    path, project = database
    project.rows[0].values[project.columns[2].id] = 7
    project.refresh()
    execute(path, 'UPDATE lab SET S2=? WHERE id=?', ("8", "001"))
    incoming, plan = preview(project)
    assert plan.cells[0]["status"] == "conflict"
    with pytest.raises(ValueError, match="конфликты"):
        prepare_refresh(project, incoming, plan)
    result, _ = prepare_refresh(project, incoming, plan, {plan.cells[0]["id"]: decision})
    assert result.rows[0].values[project.columns[2].id] == expected
    assert len(result.science.results(project.rows[0].id, project.columns[2].id)) == 2


def test_kept_conflict_is_not_repeated_until_source_changes_again(database):
    path, project = database
    project.rows[0].values[project.columns[2].id] = 7
    project.refresh()
    execute(path, 'UPDATE lab SET S2=? WHERE id=?', ("8", "001"))
    incoming, plan = preview(project)
    result, _ = prepare_refresh(project, incoming, plan, {plan.cells[0]["id"]: "keep"})
    incoming2, plan2 = preview(result)
    assert plan2.cells[0]["status"] == "local_only"
    kept, _ = prepare_refresh(result, incoming2, plan2)
    assert len(kept.science.measurements) == len(result.science.measurements)
    execute(path, 'UPDATE lab SET S2=? WHERE id=?', ("10", "001"))
    _, plan3 = preview(kept)
    assert plan3.cells[0]["status"] == "conflict"


def test_removed_rows_are_retained_and_new_rows_have_new_ids(database):
    path, project = database
    execute(path, 'DELETE FROM lab WHERE id=?', ("002",))
    execute(path, 'INSERT INTO lab VALUES (?,?,?)', ("003", "4", "12"))
    incoming, plan = preview(project)
    assert len(plan.new_rows) == len(plan.missing_rows) == 1
    with pytest.raises(ValueError, match="исчезнувших"):
        prepare_refresh(project, incoming, plan)
    result, _ = prepare_refresh(project, incoming, plan, {plan.missing_rows[0]["id"]: "keep"})
    assert len(result.rows) == 3
    assert [row.id for row in result.rows[:2]] == [row.id for row in project.rows]
    assert result.rows[-1].values[project.columns[0].id] == "003"
    assert sum(not row["present"] for row in result.view["database_snapshot"]["rows"].values()) == 1
    _, again = preview(result)
    assert again.missing_rows == [] and again.new_rows == []


def test_source_schema_change_needs_confirmation_and_never_fills_removed_column(database):
    path, project = database
    execute(path, 'ALTER TABLE lab ADD COLUMN S3 TEXT')
    incoming, plan = preview(project)
    assert plan.schema_changes["added"] == ["name:S3"]
    with pytest.raises(ValueError, match="набор"):
        prepare_refresh(project, incoming, plan)
    result, _ = prepare_refresh(project, incoming, plan, accept_schema_changes=True)
    assert len(result.columns) == 3
    execute(path, 'ALTER TABLE lab DROP COLUMN S2')
    incoming2, plan2 = preview(result)
    assert "name:S2" in plan2.schema_changes["removed"]
    retained, _ = prepare_refresh(result, incoming2, plan2, accept_schema_changes=True)
    assert retained.rows[0].values[project.columns[2].id] == 6


def test_keys_are_required_unique_and_not_row_positions(database):
    _, project = database
    incoming = read_snapshot(project.source)
    with pytest.raises(ValueError, match="ключ"):
        preview_refresh(project, incoming)
    incoming.rows[1].original[incoming.columns[0].id] = "001"
    incoming.rows[1].values[incoming.columns[0].id] = "001"
    plan = preview_refresh(project, incoming, [project.columns[0].id])
    assert plan.errors
    with pytest.raises(ValueError, match="ошибки"):
        prepare_refresh(project, incoming, plan)


def test_invalid_remote_value_can_be_kept_only_and_is_preserved_for_audit(database):
    path, project = database
    execute(path, 'UPDATE lab SET S2=? WHERE id=?', ("bad value", "001"))
    incoming, plan = preview(project)
    assert plan.cells[0]["status"] == "invalid_source"
    with pytest.raises(ValueError):
        prepare_refresh(project, incoming, plan, {plan.cells[0]["id"]: "source"})
    result, _ = prepare_refresh(project, incoming, plan, {plan.cells[0]["id"]: "keep"})
    assert result.rows[0].values[project.columns[2].id] == 6
    assert any(m.qualifier == "unparsed" and m.raw_value == "bad value" for m in result.science.measurements.values())


def test_refresh_rejects_stale_preview_or_different_database(database):
    _, project = database
    incoming, plan = preview(project)
    project.rows[0].values[project.columns[2].id] = 12
    with pytest.raises(ValueError, match="изменились"):
        prepare_refresh(project, incoming, plan)
    incoming.source["table"] = "other"
    with pytest.raises(ValueError, match="другому"):
        preview_refresh(project, incoming, [project.columns[0].id])


def test_snapshot_and_decisions_roundtrip_without_login_parameters(database, tmp_path):
    path, project = database
    execute(path, 'UPDATE lab SET S2=? WHERE id=?', ("8", "001"))
    incoming = read_snapshot(project.source, {"user": "private_reader", "password": "never-persist-this"})
    assert "never-persist-this" not in json.dumps(asdict(incoming))
    plan = preview_refresh(project, incoming, [project.columns[0].id])
    result, _ = prepare_refresh(project, incoming, plan)
    saved = tmp_path / "project.apyro"
    result.save(saved)
    loaded = Project.load(saved)
    assert loaded == result
    _, again = preview(loaded)
    assert again.cells == []


def test_source_order_is_not_record_identity(database):
    path, project = database
    execute(path, 'DELETE FROM lab WHERE id=?', ("001",))
    execute(path, 'INSERT INTO lab VALUES (?,?,?)', ("001", "2", "6"))
    incoming, plan = preview(project)
    assert plan.cells == [] and plan.new_rows == [] and plan.missing_rows == []
    result, protocol = prepare_refresh(project, incoming, plan)
    assert [row.id for row in result.rows] == [row.id for row in project.rows]
    assert not protocol["working_changed"]


def test_bad_new_row_must_be_skipped_and_numeric_qualifier_is_not_silently_accepted(database):
    path, project = database
    execute(path, 'INSERT INTO lab VALUES (?,?,?)', ("003", "not a number", "7"))
    execute(path, 'UPDATE lab SET S2=? WHERE id=?', ("<unknown", "001"))
    incoming, plan = preview(project)
    assert plan.new_rows[0]["invalid"] == ["TOC"]
    assert plan.cells[0]["status"] == "invalid_source"
    result, _ = prepare_refresh(project, incoming, plan, {plan.new_rows[0]["id"]: "skip", plan.cells[0]["id"]: "keep"})
    assert len(result.rows) == 2 and result.rows[0].values[project.columns[2].id] == 6


def test_working_selection_is_a_local_choice_even_when_numeric_value_matches(database):
    path, project = database
    sid, pid = project.rows[0].id, project.columns[2].id
    repeated = project.science.add_result(sid, pid, 6, analysis_fields={"method": "different local analysis"})
    project.select_measurement(sid, pid, repeated.id)
    execute(path, 'UPDATE lab SET S2=? WHERE id=?', ("8", "001"))
    _, plan = preview(project)
    assert plan.cells[0]["status"] == "conflict"


@pytest.mark.parametrize("decision", ["keep", "source"])
def test_equal_value_source_selection_recomputes_measurement_scoped_qc(database, decision):
    from autopyro.calculations import make_calculated_column
    path, project = database
    sid = project.rows[0].id
    toc, s2 = project.columns[1:3]
    repeat = project.science.add_result(sid, s2.id, 8, analysis_fields={"laboratory": "Local lab"})
    project.select_measurement(sid, s2.id, repeat.id)
    config = {"formula": "HI", "version": 1, "analysis_context": "synthetic QA",
              "compatible_confirmed": True, "inputs": {
                  "TOC": {"column_id": toc.id, "unit": toc.unit},
                  "S2": {"column_id": s2.id, "unit": s2.unit}}}
    hi, values = make_calculated_column(project, "HI", config)
    project.columns.append(hi)
    for row in project.rows:
        row.values[hi.id] = values[row.id]
    project.view["qc_exclusions"] = {"calculations": {
        repeat.id: {"revision": repeat.revision, "comment": "Check local repeat"}}}
    project.refresh()
    assert project.rows[0].values[hi.id] is None
    execute(path, 'UPDATE lab SET S2=? WHERE id=?', ("8", "001"))
    incoming, plan = preview(project)
    assert plan.cells[0]["status"] == "converged"
    result, protocol = prepare_refresh(project, incoming, plan, {plan.cells[0]["id"]: decision})
    assert result.rows[0].values[s2.id] == 8
    assert protocol["working_changed"] is (decision == "source")
    assert result.rows[0].values[hi.id] == (400 if decision == "source" else None)
    assert (result.science.current(sid, s2.id).id == repeat.id) is (decision == "keep")


def test_snapshot_key_cannot_be_rebound_without_new_import(database):
    _, project = database
    incoming, plan = preview(project)
    result, _ = prepare_refresh(project, incoming, plan)
    with pytest.raises(ValueError, match="закреплён"):
        preview_refresh(result, read_snapshot(result.source), [result.columns[1].id])


@pytest.mark.parametrize("dictionary", [False, True])
@pytest.mark.parametrize("mode,admin,writable,accepted", [("on", False, False, True),
    ("off", False, False, False), ("on", True, False, False), ("on", False, True, False)])
def test_source_account_check_rejects_admin_and_writable_roles(dictionary, mode, admin, writable, accepted):
    from autopyro.database_access import verify_read_only
    class Cursor:
        def __init__(self, value):
            self.value = value
        def fetchone(self):
            return self.value
    class Connection:
        def execute(self, statement):
            assert statement.lstrip().startswith(("SELECT", "SHOW", "WITH"))
            if statement.startswith("SHOW"):
                value = (mode,)
            elif statement.lstrip().startswith("SELECT rolsuper"):
                value = (admin, False, False, False, False)
            else:
                value = (writable,)
            return Cursor({str(i): item for i, item in enumerate(value)} if dictionary else value)
    if accepted:
        verify_read_only(Connection())
    else:
        with pytest.raises(ValueError):
            verify_read_only(Connection())


def test_gui_conflict_decision_undo_and_snapshot_state(database, qtbot):
    from autopyro.window import MainWindow
    from autopyro.database_refresh_dialog import DatabaseRefreshDialog
    path, project = database
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.model.setData(window.model.index(0, 2), 7)
    execute(path, 'UPDATE lab SET S2=? WHERE id=?', ("8", "001"))
    incoming, plan = preview(project)
    dialog = DatabaseRefreshDialog(window)
    qtbot.addWidget(dialog)
    dialog.loaded((incoming, plan))
    assert not dialog.apply_button.isEnabled()
    choice = dialog.table.cellWidget(0, 6)
    choice.setCurrentIndex(choice.findData("source"))
    assert dialog.apply_button.isEnabled()
    dialog.apply_refresh()
    qtbot.waitUntil(lambda: dialog.result() == dialog.DialogCode.Accepted, timeout=10000)
    assert project.rows[0].values[project.columns[2].id] == 8
    assert "database_snapshot" in project.view
    project.view["plots"] = {"keep user styling": {"title": "new title"}}
    window.model.undo.undo()
    assert project.rows[0].values[project.columns[2].id] == 7
    assert "database_snapshot" not in project.view
    assert project.view["plots"]["keep user styling"]["title"] == "new title"
    window.model.undo.redo()
    assert project.rows[0].values[project.columns[2].id] == 8
    assert project.view["database_snapshot"]["checked_at"]
    assert "Снимок проверен" in window.statusBar().currentMessage()
    window.dirty = False


def test_gui_reads_snapshot_in_worker_without_writing_the_source(database, qtbot):
    from autopyro.window import MainWindow
    from autopyro.database_refresh_dialog import DatabaseRefreshDialog
    path, project = database
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    dialog = DatabaseRefreshDialog(window)
    qtbot.addWidget(dialog)
    dialog.keys.item(0).setCheckState(Qt.CheckState.Checked)
    before = path.read_bytes()
    dialog.read()
    qtbot.waitUntil(lambda: dialog.plan is not None and dialog.worker is not None and not dialog.worker.isRunning(), timeout=10000)
    assert path.read_bytes() == before
    assert dialog.plan.cells == []
    dialog.reject()
    assert "database_snapshot" not in project.view


def test_profile_snapshot_uses_source_record_id_and_catalog_field_identity():
    from autopyro.geochem_database import project_from_records
    source = {"type": "postgresql_nonpyro", "endpoint": {"host": "localhost", "port": 15432, "database": "geochemistry"},
              "method_profile": "atomic_ratios", "groups": ["Elem"], "variant": "", "repeat_number": 1,
              "batch_id": None, "well_name": "", "read_only": True}
    fields = [{"source_field_id": 77, "group_name": "Elem", "source_label": "TOC", "source_dtype": "Float64", "unit_label": "%wt"}]
    records = [{"source_record_id": 1001, "sample_id": 20, "import_batch_id": 1, "row_number": 5, "sample_name": "same name"}]
    def make(value):
        results = [{"source_record_id": 1001, "analysis_run_id": 500, "group_name": "Elem", "repeat_number": 1,
                    "source_field_id": 77, "value_float": value, "value_int": None, "value_text": None}]
        return project_from_records(records, results, fields, source=source)
    project, incoming = make(2), make(3)
    project.sync_measurements()
    plan = preview_refresh(project, incoming)
    assert len(plan.cells) == 1 and plan.cells[0]["status"] == "source_changed"
    result, _ = prepare_refresh(project, incoming, plan)
    assert result.rows[0].id == project.rows[0].id
    assert result.rows[0].values[project.columns[-1].id] == 3
    changed_definition = make(3)
    changed_definition.columns[-1].metadata["geochem_field"]["label"] = "Different scientific definition"
    assert preview_refresh(result, changed_definition).errors
    incoming.source["well_name"] = "different selection"
    with pytest.raises(ValueError, match="другому"):
        preview_refresh(project, incoming)


def test_raw_postgres_record_keys_are_stable_and_not_laboratory_names():
    source = {"type": "postgresql", "host": "localhost", "port": "15432", "dbname": "geochemistry",
              "relation": "geochem.source_record.raw_cells", "read_only": True, "source_columns": ["S2"]}
    project = build_project(["S2"], [["6"], ["9"]], [1001, 1002], [(0, "S2", "number", "mg HC/g rock")], source)
    project.sync_measurements()
    incoming = build_project(["S2"], [["10"], ["6"]], [1002, 1001], [(0, "S2", "text", "")], source)
    plan = preview_refresh(project, incoming)
    assert len(plan.cells) == 1
    assert plan.cells[0]["sample_id"] == project.rows[1].id
    result, _ = prepare_refresh(project, incoming, plan)
    assert [r.id for r in result.rows] == [r.id for r in project.rows]
    assert result.rows[1].values[project.columns[0].id] == 10


def test_reader_credentials_are_never_copied_from_incoming_connection_metadata(database):
    path, project = database
    execute(path, 'UPDATE lab SET S2=? WHERE id=?', ("8", "001"))
    execute(path, 'INSERT INTO lab VALUES (?,?,?)', ("003", "4", "12"))
    incoming = read_snapshot(project.source)
    incoming.source.update(password="secret-connection-option", user="private-role")
    for row in incoming.rows:
        row.source.update(password="secret-connection-option", user="private-role", dsn="secret-dsn")
    plan = preview_refresh(project, incoming, [project.columns[0].id])
    result, protocol = prepare_refresh(project, incoming, plan)
    serialized = json.dumps({"project": asdict(result), "protocol": protocol})
    assert "secret-connection-option" not in serialized and "secret-dsn" not in serialized and "private-role" not in serialized


def test_reports_and_statistics_capture_snapshot_state_for_selected_samples_only(database):
    from autopyro.reporting import build_report, ReportOptions, render_html
    from autopyro.statistics import snapshot
    path, project = database
    execute(path, 'DELETE FROM lab WHERE id=?', ("002",))
    incoming, plan = preview(project)
    result, _ = prepare_refresh(project, incoming, plan, {plan.missing_rows[0]["id"]: "keep"})
    sid, pid = project.rows[1].id, project.columns[2].id
    report = build_report(result, [sid], ReportOptions((pid,)))
    assert report["database_snapshot"]["records"] == [{"sample_id": sid, "present": False}]
    assert project.rows[0].id not in json.dumps(report)
    assert "Состояние снимка БД" in render_html(report)
    stats = snapshot(result, [sid], [pid])
    assert stats.view["database_snapshot_status"]["records"] == [{"sample_id": sid, "present": False}]


def test_unchanged_refresh_does_not_invalidate_interpolation(database):
    _, project = database
    from autopyro.core import Column
    column = Column("old interpolation", "number", metadata={"method": "linear", "stale": False})
    project.columns.append(column)
    for row in project.rows:
        row.values[column.id] = 1
    incoming, plan = preview(project)
    result, protocol = prepare_refresh(project, incoming, plan)
    assert not protocol["working_changed"]
    assert not result.columns[-1].metadata["stale"]


def test_project_or_recovery_cannot_overwrite_the_read_only_source(database):
    path, project = database
    before = path.read_bytes()
    for save in (project.save, project.save_recovery):
        with pytest.raises(ValueError, match="только для чтения"):
            save(path)
        assert path.read_bytes() == before


def test_known_invalid_source_keeps_previous_decision_without_duplicate_results(database):
    path, project = database
    execute(path, 'UPDATE lab SET S2=? WHERE id=?', ("bad", "001"))
    incoming, plan = preview(project)
    result, _ = prepare_refresh(project, incoming, plan, {plan.cells[0]["id"]: "keep"})
    incoming2, plan2 = preview(result)
    assert plan2.cells[0]["status"] == "invalid_source" and plan2.cells[0]["action"] == "keep"
    refreshed, _ = prepare_refresh(result, incoming2, plan2)
    assert len(refreshed.science.measurements) == len(result.science.measurements)


def test_refresh_dialog_never_prefills_local_password_for_a_different_endpoint(qtbot, monkeypatch):
    import autopyro.database_refresh_dialog as gui
    from autopyro.window import MainWindow
    source = {"type": "postgresql", "host": "other-host.example", "port": 15432, "dbname": "geochemistry",
              "relation": "geochem.source_record.raw_cells", "read_only": True}
    project = build_project(["S2"], [["6"]], [1001], [(0, "S2", "number", "mg HC/g rock")], source)
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    monkeypatch.setattr(gui, "local_parameters", lambda: {"host": "127.0.0.1", "port": "15432", "dbname": "geochemistry",
                                                         "user": "readonly", "password": "local-private-password"})
    dialog = gui.DatabaseRefreshDialog(window)
    qtbot.addWidget(dialog)
    assert dialog.password.text() == ""
    from PySide6.QtWidgets import QLabel
    assert any("other-host.example:15432" in label.text() for label in dialog.findChildren(QLabel))
