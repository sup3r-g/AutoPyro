"""Acceptance tests for measurement selection, v1 migration and crash-safe storage."""
import json
import sqlite3
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict

import pytest


from autopyro.core import Project
from autopyro.io import build_project
from autopyro.measurements import parse_result
from autopyro.storage import write_lock


@pytest.mark.parametrize("cache_kib", [0, 64 * 1024, 256 * 1024])
def test_write_cache_budget_preserves_project_data(project, tmp_path, monkeypatch, cache_kib):
    from autopyro.database import storage
    monkeypatch.setattr(storage, "WRITE_CACHE_KIB", cache_kib)
    path = tmp_path / "cache-budget.apyro"
    project.save(path)
    restored = Project.load(path)
    assert asdict(restored) == asdict(project)


@pytest.fixture
def project():
    return build_project(["Sample", "TOC", "S2"], [["001", "2,00", "6"], ["001", "1", "ND"]],
                         [2, 3], [(0, "Sample", "text", ""), (1, "TOC", "number", "%wt"),
                                  (2, "S2", "text", "mg HC/g rock")],
                         {"path": "lab.csv", "sha256": "test-source", "sheet": "Lab"})


def test_sync_keeps_original_cells_and_nested_source_metadata_independent(project):
    row = project.rows[0]
    pid = project.columns[1].id
    row.source["analysis_runs"] = {"TOC": {"labels": ["original"]}}
    project.sync_measurements()
    measurement = project.science.current(row.id, pid)
    source = project.science.sources[measurement.source_id]
    analysis = project.science.analyses[measurement.analysis_id]

    row.original[pid] = "99"
    row.source["analysis_runs"]["TOC"]["labels"].append("changed")
    assert source.original[pid] == measurement.raw_value == "2,00"
    assert source.provenance["analysis_runs"]["TOC"]["labels"] == ["original"]
    assert analysis.metadata["analysis_runs"]["TOC"]["labels"] == ["original"]


def test_subset_preserves_scientific_records_and_isolates_nested_data(project):
    project.sync_measurements()
    store = project.science
    sid, pid = project.rows[0].id, project.columns[1].id
    measurement = store.current(sid, pid)
    measurement.history.append({"previous": {"value": [2, 3]}})
    store.samples[sid].metadata["tags"] = ["original"]
    store.analyses[measurement.analysis_id].metadata["settings"] = {"steps": [1, 2]}
    assert asdict(store.subset(store.samples, store.properties)) == asdict(deepcopy(store))
    frozen = store.subset([sid], [pid])
    measurement.history[0]["previous"]["value"].append(4)
    store.samples[sid].metadata["tags"].append("changed")
    store.analyses[measurement.analysis_id].metadata["settings"]["steps"].append(3)
    assert frozen.current(sid, pid).history[0]["previous"]["value"] == [2, 3]
    assert frozen.samples[sid].metadata["tags"] == ["original"]
    assert frozen.analyses[measurement.analysis_id].metadata["settings"]["steps"] == [1, 2]


def test_subset_does_not_copy_original_cells_of_excluded_properties(project):
    project.sync_measurements()
    sid, pid = project.rows[0].id, project.columns[1].id
    measurement = project.science.current(sid, pid)
    source = project.science.sources[measurement.source_id]

    class Unrequested:
        def __deepcopy__(self, memo):
            raise AssertionError("An excluded original cell must not be copied")

    source.original[project.columns[0].id] = Unrequested()
    shared = ["trace"]
    source.original[pid] = shared
    source.provenance["trace"] = shared
    frozen = project.science.subset([sid], [pid])
    copied = frozen.sources[source.id]
    assert set(copied.original) == {pid}
    assert copied.original[pid] is copied.provenance["trace"]
    assert copied.original[pid] is not shared


@pytest.mark.parametrize("raw,qualifier,value,limit", [
    (0, "exact", 0, None), (None, "missing", None, None), ("ND", "non_detect", None, None),
    ("<0,01", "below_limit", None, .01), (">=2", "above_limit", None, 2),
    ("≈1,2", "approximate", 1.2, None)])
def test_result_semantics(raw, qualifier, value, limit):
    result = parse_result(raw, "number")
    assert result["qualifier"] == qualifier
    assert result["value"] == value
    assert result["limit"] == limit
    assert result["raw_value"] == raw


def test_normalized_roundtrip_keeps_raw_text_ids_repeats_and_settings(project, tmp_path):
    project.view = {"current": {"visible": [project.columns[1].id]}, "plots": {"example": {"x_id": project.columns[1].id}}}
    path = tmp_path / "project.apyro"
    project.save(path)
    restored = Project.load(path)
    assert restored == project
    assert path.read_bytes().startswith(b"SQLite format 3\0")
    assert len(restored.science.samples) == 2  # Same name never merges physical samples.
    toc = restored.science.current(restored.rows[0].id, restored.columns[1].id)
    assert toc.raw_value == "2,00"
    assert toc.value == 2
    assert restored.rows[0].values[restored.columns[0].id] == "001"
    nd = restored.science.current(restored.rows[1].id, restored.columns[2].id)
    assert nd.qualifier == "non_detect"
    assert nd.value is None
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("SELECT count(*) FROM measurements").fetchone()[0] == 6


def test_legacy_open_is_read_only_and_first_save_backs_up_exact_bytes(project, tmp_path):
    payload = asdict(project)
    payload.pop("science")
    payload.update(format="autopyro.table", version=1)
    path = tmp_path / "old.apyro"
    original = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    path.write_bytes(original)
    restored = Project.load(path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.bak"))
    assert set(Project.load(path).science.measurements) == set(restored.science.measurements)
    assert [row.id for row in restored.rows] == [row.id for row in project.rows]
    restored.save(path)
    assert Project.load(path) == restored
    backups = list(tmp_path.glob("old.apyro.v1-*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original


def test_failed_migration_keeps_legacy_file_and_cleans_temp(project, tmp_path, monkeypatch):
    import autopyro.storage as storage
    payload = asdict(project)
    payload.pop("science")
    path = tmp_path / "legacy.apyro"
    path.write_text(json.dumps({"format": "autopyro.table", "version": 1, **payload}), encoding="utf-8")
    original = path.read_bytes()
    loaded = Project.load(path)
    monkeypatch.setattr(storage.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("simulated failure")))
    with pytest.raises(OSError, match="simulated"):
        loaded.save(path)
    assert path.read_bytes() == original
    assert next(tmp_path.glob("*.bak")).read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_sqlite_save_leaves_previous_project(project, tmp_path, monkeypatch):
    import autopyro.storage as storage
    path = tmp_path / "project.apyro"
    project.save(path)
    original = path.read_bytes()
    project.rows[0].values[project.columns[1].id] = 9
    monkeypatch.setattr(storage.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("interrupted save")))
    with pytest.raises(OSError):
        project.save(path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_stale_writer_is_rejected_and_save_as_remains_available(project, tmp_path):
    path = tmp_path / "project.apyro"
    project.save(path)
    first, stale = Project.load(path), Project.load(path)
    first.rows[0].values[first.columns[1].id] = 3
    first.save(path)
    stale.rows[0].values[stale.columns[1].id] = 4
    with pytest.raises(ValueError, match="другим процессом"):
        stale.save(path)
    assert Project.load(path).rows[0].values[first.columns[1].id] == 3
    alternative = tmp_path / "local.apyro"
    stale.save(alternative)
    assert Project.load(alternative).rows[0].values[first.columns[1].id] == 4


def test_simultaneous_writer_lock_is_rejected(project, tmp_path):
    path = tmp_path / "project.apyro"
    with write_lock(path):
        with pytest.raises(ValueError, match="другим процессом"):
            project.save(path)
    project.save(path)  # OS lock released after error; persistent sidecar is harmless.


def test_corrupt_selection_is_rejected_instead_of_silently_repaired(project, tmp_path):
    path = tmp_path / "project.apyro"
    project.save(path)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("UPDATE working_measurements SET measurement_id='missing'")
        connection.commit()
    with pytest.raises(ValueError, match="целостность"):
        Project.load(path)


def test_recovery_save_does_not_replace_or_rebind_original(project, tmp_path):
    path = tmp_path / "project.apyro"
    project.save(path)
    before = path.read_bytes()
    project.rows[0].values[project.columns[1].id] = 5
    recovery = tmp_path / "recovery.apyro"
    project.save_recovery(recovery)
    assert path.read_bytes() == before
    assert project._storage_path == path.resolve()
    assert Project.load(recovery).rows[0].values[project.columns[1].id] == 5
    project.save(path)


def test_measurement_from_other_sample_cannot_be_selected(project):
    project.sync_measurements()
    sample_id, other = [row.id for row in project.rows]
    property_id = project.columns[1].id
    wrong = project.science.current(other, property_id)
    with pytest.raises(ValueError, match="другой пробе"):
        project.science.select(sample_id, property_id, wrong.id)


def test_missing_analysis_and_missing_result_are_distinct(project):
    project.sync_measurements()
    sid, pid = project.rows[0].id, project.columns[1].id
    before = len(project.science.results(sid, pid))
    missing = project.science.add_result(sid, pid, None, analysis_fields={"method": "Pyrol6"})
    assert len(project.science.results(sid, pid)) == before + 1
    assert missing.qualifier == "missing"
    assert missing.analysis_id in project.science.analyses


def test_project_api_selects_result_without_qt_and_keeps_it_on_save(project, tmp_path):
    project.sync_measurements()
    sid, pid = project.rows[0].id, project.columns[1].id
    result = project.science.add_result(sid, pid, "5", analysis_fields={"method": "Pyrol6"})
    project.select_measurement(sid, pid, result.id)
    assert project.rows[0].values[pid] == 5
    path = tmp_path / "api.apyro"
    project.save(path)
    assert Project.load(path).science.current(sid, pid).id == result.id


def test_working_result_switch_recalculates_hi_and_undo_preserves_other_results(project, qtbot, tmp_path):
    from autopyro.calculations import make_calculated_column
    from autopyro.table import TableModel, AddColumnCommand, SelectMeasurementCommand
    model = TableModel(project)
    sample_id = project.rows[0].id
    toc, s2 = project.columns[1:]
    config = {"formula": "HI", "version": 1, "analysis_context": "test Pyrol6 whole rock",
              "compatible_confirmed": True, "text_numbers": True,
              "inputs": {"TOC": {"column_id": toc.id, "unit": toc.unit},
                         "S2": {"column_id": s2.id, "unit": s2.unit}}}
    hi, values = make_calculated_column(project, "HI", config)
    model.undo.push(AddColumnCommand(model, hi, values))
    assert hi.id not in project.science.properties
    original = project.science.current(sample_id, s2.id)
    second = project.science.add_result(sample_id, s2.id, "8", analysis_fields={"method": "Pyrol6", "laboratory": "Lab B"},
                                       details={"uncertainty": .3, "uncertainty_type": "standard", "confidence_level": .95})
    assert project.science.current(sample_id, s2.id).id == original.id
    assert project.rows[0].values[hi.id] == 300
    project.view["interpretation_result"] = {"stale": False}
    model.undo.push(SelectMeasurementCommand(model, sample_id, s2.id, second.id))
    assert project.rows[0].values[hi.id] == 400
    assert project.view["interpretation_result"]["stale"]
    assert original.value == "6"  # Explicit text column stays literal.
    assert len(project.science.results(sample_id, s2.id)) == 2
    model.undo.undo()
    assert project.rows[0].values[hi.id] == 300
    model.undo.redo()
    assert project.rows[0].values[hi.id] == 400
    path = tmp_path / "repeat.apyro"
    project.save(path)
    loaded = Project.load(path)
    assert loaded.science.current(sample_id, s2.id).id == second.id
    assert loaded.science.measurements[second.id].uncertainty == .3
    assert loaded.science.analyses[second.analysis_id].laboratory == "Lab B"


def test_measurement_history_tracks_edits_and_undo_without_changing_source(project, qtbot):
    from autopyro.table import TableModel
    model = TableModel(project)
    row, column = project.rows[0], project.columns[1]
    measurement = project.science.current(row.id, column.id)
    source = deepcopy(project.science.sources[measurement.source_id])
    model.setData(model.index(0, 1), "3")
    assert measurement.revision == 2
    assert measurement.history[0]["previous"]["raw_value"] == "2,00"
    model.undo.undo()
    assert measurement.revision == 3
    assert measurement.value == 2
    assert project.science.sources[measurement.source_id] == source
    assert row.original[column.id] == "2,00"


def test_dialog_add_is_explicit_validated_and_undoable(project, qtbot):
    from autopyro.table import TableModel
    from autopyro.measurement_dialog import MeasurementDialog, NewMeasurementDialog
    model = TableModel(project)
    sid, pid = project.rows[0].id, project.columns[1].id
    original_id = project.science.current(sid, pid).id
    dialog = NewMeasurementDialog(model, sid, pid)
    qtbot.addWidget(dialog)
    dialog.fields["raw"].setText("3,5")
    dialog.submit()
    assert dialog.error.text()
    assert len(project.science.results(sid, pid)) == 1
    dialog.fields["method"].setText("Pyrol6")
    dialog.fields["uncertainty"].setText("-1")
    dialog.submit()
    assert len(project.science.results(sid, pid)) == 1
    dialog.fields["uncertainty"].setText("0,2")
    dialog.submit()
    assert len(project.science.results(sid, pid)) == 2
    assert project.science.current(sid, pid).id == original_id
    card = MeasurementDialog(model, sid, pid)
    qtbot.addWidget(card)
    card.table.selectRow(1)
    card.select_result()
    assert project.rows[0].values[pid] == 3.5
    model.undo.undo()  # selection
    assert project.science.current(sid, pid).id == original_id
    model.undo.undo()  # addition
    assert len(project.science.results(sid, pid)) == 1
    model.undo.redo()
    model.undo.redo()
    assert project.rows[0].values[pid] == 3.5


def test_card_switches_properties_without_rescanning_and_shows_analysis_source(project, qtbot):
    from autopyro.measurement_dialog import MeasurementDialog
    from autopyro.measurements import SourceRecord
    from autopyro.table import AddMeasurementCommand, TableModel

    class CountedMeasurements(dict):
        scans = 0

        def values(self):
            self.scans += 1
            return super().values()

    model = TableModel(project)
    row = project.rows[0]
    toc, s2 = project.columns[1:]
    source = SourceRecord("repeat", {"path": "repeat.csv", "row": 4})
    project.science.sources[source.id] = source
    repeat = project.science.add_result(row.id, s2.id, "8", analysis_fields={"source_id": source.id})
    project.science.measurements = CountedMeasurements(project.science.measurements)
    card = MeasurementDialog(model, row.id, toc.id)
    qtbot.addWidget(card)
    assert card.table.rowCount() == 1
    assert project.science.measurements.scans == 1
    card.property_picker.setCurrentIndex(card.property_picker.findData(s2.id))
    assert card.table.rowCount() == 2
    assert project.science.measurements.scans == 1
    card.table.selectRow(1)
    assert card.selected_id() == repeat.id
    assert "repeat.csv" in card.source_details.toPlainText()

    added, analysis = project.science.prepare_result(row.id, s2.id, "9")
    model.undo.push(AddMeasurementCommand(model, added, analysis))
    card.reload()
    assert card.table.rowCount() == 3
    model.undo.undo()
    card.reload()
    assert card.table.rowCount() == 2
    model.undo.redo()
    card.reload()
    assert card.table.rowCount() == 3


def test_numeric_censoring_remains_safe_for_filter_sort_and_interpolation(qtbot):
    from autopyro.table import TableModel, FilterProxy, SelectMeasurementCommand
    from autopyro.interpolation import propose
    data = build_project(["Well", "Depth", "S2"], [["A", "10", "2"], ["A", "20", "4"], ["A", "30", "6"]],
                         [2, 3, 4], [(0, "Well", "text", ""), (1, "Depth", "number", "m"), (2, "S2", "number", "mg/g")], {})
    model = TableModel(data)
    well, depth, s2 = data.columns
    mid = data.science.add_result(data.rows[1].id, s2.id, "<0.01", analysis_fields={"method": "test"}).id
    model.undo.push(SelectMeasurementCommand(model, data.rows[1].id, s2.id, mid))
    proxy = FilterProxy()
    proxy.setSourceModel(model)
    proxy.sort_rules = [(s2.id, False)]
    proxy.sort(2)
    proxy.set_rules([{"column": s2.id, "operator": "gt", "value": 1}])
    assert proxy.rowCount() == 2
    assert propose(data, depth.id, s2.id, [well.id], set(data.row_map()), max_gap=100) == []


def test_window_autosave_recovers_dirty_data_and_keeps_saved_original(project, qtbot, tmp_path):
    from autopyro.window import MainWindow
    window = MainWindow()
    qtbot.addWidget(window)
    path = tmp_path / "project.apyro"
    project.save(path)
    before = path.read_bytes()
    window.set_project(project, path)
    window.model.setData(window.model.index(0, 1), "7")
    window.autosave()
    recovery = window.autosave_path()
    qtbot.waitUntil(lambda: recovery.exists() and window._background_autosave.worker is None, timeout=10000)
    assert path.read_bytes() == before
    window.load_autosave(recovery)
    assert window.dirty
    assert window.model.project.rows[0].values[project.columns[1].id] == 7
    assert window.save()
    assert not recovery.exists()
    assert Project.load(path).rows[0].values[project.columns[1].id] == 7


def test_report_and_statistics_snapshot_keep_selected_measurement_provenance(project, qtbot):
    from autopyro.table import TableModel, SelectMeasurementCommand
    from autopyro.reporting import build_report, ReportOptions
    from autopyro.statistics import snapshot
    model = TableModel(project)
    sid, pid = project.rows[0].id, project.columns[1].id
    result = project.science.add_result(sid, pid, 4, analysis_fields={"method": "Pyrol6", "laboratory": "Lab B"})
    model.undo.push(SelectMeasurementCommand(model, sid, pid, result.id))
    report = build_report(project, [sid], ReportOptions((pid,)))
    science = report["project"]["science"]
    assert science["working"][sid][pid] == result.id
    assert science["analyses"][result.analysis_id]["laboratory"] == "Lab B"
    assert project.rows[1].id not in json.dumps(report)
    stats = snapshot(project, [sid], [pid])
    assert stats.science.current(sid, pid).id == result.id
    assert list(stats.science.properties) == [pid]
    original = next(iter(stats.science.sources.values()))
    assert set(original.original) == {pid}
    project.science.measurements[result.id].value = 999
    assert stats.science.measurements[result.id].value == 4


def test_restoring_recovery_does_not_allow_overwriting_external_changes(project, qtbot, tmp_path):
    from autopyro.window import MainWindow
    window = MainWindow()
    qtbot.addWidget(window)
    path = tmp_path / "project.apyro"
    project.save(path)
    window.set_project(project, path)
    window.model.setData(window.model.index(0, 1), "7")
    window.autosave()
    recovery = window.autosave_path()
    qtbot.waitUntil(lambda: recovery.exists() and window._background_autosave.worker is None, timeout=10000)
    external = Project.load(path)
    external.rows[0].values[project.columns[1].id] = 8
    external.save(path)
    window.load_autosave(recovery)
    with pytest.raises(ValueError, match="другим процессом"):
        window.model.project.save(path)
    assert Project.load(path).rows[0].values[project.columns[1].id] == 8
    window.clear_autosave()
    window.dirty = False


@pytest.mark.parametrize("recovery_case", ["missing_target", "missing_digest"])
def test_recovery_cannot_adopt_an_unverified_target_version(project, qtbot, tmp_path, recovery_case, monkeypatch):
    from autopyro.window import MainWindow
    from PySide6.QtWidgets import QFileDialog

    window = MainWindow()
    qtbot.addWidget(window)
    path = tmp_path / "original.apyro"
    project.save(path)
    window.set_project(project, path)
    window.model.setData(window.model.index(0, 1), "7")
    window.autosave()
    recovery = window.autosave_path()
    qtbot.waitUntil(lambda: recovery.exists() and window._background_autosave.worker is None, timeout=10000)
    try:
        if recovery_case == "missing_digest":
            legacy_recovery = Project.load(recovery)
            legacy_recovery.view["recovery"].pop("source_digest")
            legacy_recovery.save_recovery(recovery)
        else:
            path.unlink()
            window.load_autosave(recovery)
        external = deepcopy(project)
        external.rows[0].values[external.columns[1].id] = 8
        external.save_recovery(path)
        if recovery_case == "missing_digest":
            window.load_autosave(recovery)
        external_bytes = path.read_bytes()
        with pytest.raises(ValueError):
            window.model.project.save(path)
        assert path.read_bytes() == external_bytes
        assert recovery.exists()
        alternative = tmp_path / "recovered-copy.apyro"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(alternative), ""))
        assert window.save(as_new=True)
        assert Project.load(alternative).rows[0].values[project.columns[1].id] == 7
        assert path.read_bytes() == external_bytes
        assert not recovery.exists()
    finally:
        window.clear_autosave()
        window.dirty = False


@pytest.mark.parametrize("restore_first", [False, True])
def test_saving_to_autosave_path_preserves_the_saved_project(project, qtbot, tmp_path, monkeypatch, restore_first):
    from autopyro.window import MainWindow
    from PySide6.QtWidgets import QFileDialog

    path = tmp_path / "original.apyro"
    project.save(path)
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project, path)
    window.model.setData(window.model.index(0, 1), "7")
    window.autosave()
    recovery = window.autosave_path()
    qtbot.waitUntil(lambda: recovery.exists() and window._background_autosave.worker is None, timeout=10000)
    if restore_first:
        window.load_autosave(recovery)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(recovery), ""))
    try:
        assert window.save(as_new=True)
        assert recovery.exists()
        assert Project.load(recovery).rows[0].values[project.columns[1].id] == 7
        assert window.path == recovery
        assert not window.dirty
        assert Project.load(path).rows[0].values[project.columns[1].id] == 2
    finally:
        window.dirty = False
