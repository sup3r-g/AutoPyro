"""Exercise actual window save/autosave/recovery on a disposable wide project.

python -m tools.benchmark_recovery_workflow --rows 30000 --directory tmp/recovery-wide
The directory must be new. No existing projects or user preferences are changed.
"""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QEventLoop, QSettings, QTimer
from PySide6.QtWidgets import QApplication, QFileDialog

from autopyro.calculations import make_calculated_column
from autopyro.database.storage import file_digest
from autopyro.database import storage
from autopyro.ui import i18n
from autopyro.ui.tasks import Worker
from autopyro.ui.table.commands import SelectMeasurementCommand
from autopyro.ui.table.model import prepare_project
from autopyro.ui.window import MainWindow
from tools.benchmark_large_project_storage import fingerprint
from tools.benchmark_workspace_interpolation import dataset, measure
from tools.save_diagnostics import SaveDiagnostics


def wait_for_autosave(owner, timeout_seconds=1200):
    owner.autosave()
    job = owner._background_autosave
    if job is None:
        assert owner.autosave_path().exists()
        return
    errors = []
    job.failed.connect(errors.append)
    loop, timer = QEventLoop(), QTimer()
    started = perf_counter()

    def check():
        if job.worker is None or perf_counter() - started > timeout_seconds:
            loop.quit()

    timer.timeout.connect(check)
    timer.start(25)
    try:
        if job.worker is not None:
            loop.exec()
        if job.worker is not None:
            raise TimeoutError("Background recovery did not finish within the benchmark deadline")
        if errors:
            raise RuntimeError(errors[-1])
        assert owner.autosave_path().exists()
    finally:
        timer.stop()
        job.failed.disconnect(errors.append)


def benchmark(app, directory, rows, extra_columns, *, save_only=False, save_diagnostics=False, repeat_save=False,
              defer_progress_start=False, history_revisions=0):
    result = {"rows": rows, "source_columns": 5 + extra_columns,
              "python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
              "qt_platform": app.platformName(), "stages": [], "verified": False, "save_only": save_only,
              "save_diagnostics": save_diagnostics,
              "write_cache_kib": storage.WRITE_CACHE_KIB,
              "secondary_indexes_after_rows": bool(storage.SECONDARY_INDEXES),
              "repeat_save": repeat_save,
              "history_revisions_per_row": history_revisions,
              "defer_progress_start": defer_progress_start,
              "note": "One process, one pass, no tracemalloc. Shown Qt widgets on the selected backend; "
                      "offscreen does not measure a physical Windows display. Times include queued Qt work. "
                      "Preparation, explicit pre-stage GC, fingerprints and file hashes are outside stage timings. "
                      "Memory peak is cumulative, including preparation and old/new project overlap. "
                      "Reopening a just-written file does not establish cold-cache performance or p95."}
    report = directory / "result.json"

    def record():
        report.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def stage(name, function):
        print(json.dumps({"stage": name, "status": "starting"}), flush=True)
        observed_saves = ("initial_save", "repeat_save", "background_autosave", "save_restored")
        diagnostic = SaveDiagnostics() if save_diagnostics and name in observed_saves else None
        def operation():
            if diagnostic is None:
                return function()
            with diagnostic:
                return function()
        value, metrics = measure(app, name, operation)
        if diagnostic is not None:
            metrics["save_parts"] = diagnostic.summary()
        result["stages"].append(metrics)
        record()
        return value

    print(json.dumps({"status": "preparing_synthetic_fixture"}), flush=True)
    project = prepare_project(dataset(rows, extra_columns))
    if history_revisions:
        depth = project.columns[2]
        for item in project.rows:
            initial_depth = item.values[depth.id]
            for revision in range(1, history_revisions + 1):
                project.science.set_value(item.id, depth.id, initial_depth + revision, depth.kind)
            item.values[depth.id] = initial_depth + history_revisions
    result["initial_measurement_history_entries"] = sum(len(m.history) for m in project.science.measurements.values())
    assert result["initial_measurement_history_entries"] == rows * history_revisions
    toc, s2 = project.columns[3:5]
    hi, values = make_calculated_column(project, "HI", {
        "formula": "HI", "version": 1, "compatible_confirmed": True,
        "analysis_context": "Synthetic recovery performance QA; canonical units",
        "inputs": {"TOC": {"column_id": toc.id, "unit": toc.unit},
                   "S2": {"column_id": s2.id, "unit": s2.unit}}})
    project.columns.append(hi)
    for row in project.rows:
        row.values[hi.id] = values[row.id]
    row = project.rows[3]
    sid, pid, hi_id = row.id, s2.id, hi.id
    original = project.science.current(sid, pid)
    original_data, original_id = asdict(original), original.id
    repeat = project.science.add_result(sid, pid, "8",
        analysis_fields={"method": "Pyrol6", "laboratory": "Synthetic repeat lab"},
        details={"uncertainty": .3, "uncertainty_type": "standard", "confidence_level": .95})
    repeat_id = repeat.id
    result["measurements"] = len(project.science.measurements)
    path = directory / "synthetic.apyro"
    owner = MainWindow()
    owner.autosave_timer.stop()  # The benchmark triggers each recovery explicitly.
    owner.show()
    try:
        stage("attach_table", lambda: owner.set_project(project, path, prepared=True))
        result["qt_window_exposed"] = owner.windowHandle().isExposed()
        if app.platformName() == "windows":
            assert result["qt_window_exposed"], "The native benchmark window was not exposed"
        assert stage("initial_save", owner.save)
        original_file_digest = file_digest(path)
        result["initial_file_bytes"] = path.stat().st_size
        if save_only:
            if repeat_save:
                stage("select_repeat", lambda: owner.model.undo.push(SelectMeasurementCommand(owner.model, sid, pid, repeat_id)))
                assert stage("edit_repeat", lambda: owner.model.setData(owner.model.index(3, 4), "9"))
                assert row.values[hi_id] == 450
                assert asdict(original) == original_data
                assert project._storage_digest == original_file_digest
                assert stage("repeat_save", owner.save)
                new_digest = file_digest(path)
                assert new_digest != original_file_digest
                assert project._storage_digest == new_digest and not owner.dirty
                original_file_digest = new_digest
                result["repeat_file_bytes"] = path.stat().st_size
                result["repeat_binding_verified"] = True
            print(json.dumps({"status": "fingerprinting_saved_state"}), flush=True)
            expected = fingerprint(project)
            project = row = original = repeat = None
            with patch.object(QFileDialog, "getOpenFileName", return_value=(str(path), "")):
                stage("reopen_saved", owner.open_project)
            assert fingerprint(owner.model.project) == expected
            assert file_digest(path) == original_file_digest and not owner.dirty
            result.update(verified=True, final_file_bytes=path.stat().st_size, fingerprint=expected)
            record()
            print(json.dumps({"verified": True, "report": str(report)}), flush=True)
            return result
        stage("select_repeat", lambda: owner.model.undo.push(SelectMeasurementCommand(owner.model, sid, pid, repeat_id)))
        assert row.values[hi_id] == 400
        assert stage("edit_repeat", lambda: owner.model.setData(owner.model.index(3, 4), "9"))
        assert row.values[hi_id] == 450
        stage("undo_edit", owner.model.undo.undo)
        assert row.values[hi_id] == 400
        stage("undo_selection", owner.model.undo.undo)
        assert project.science.current(sid, pid).id == original_id
        assert row.values[hi_id] == original.value / row.values[toc.id] * 100
        stage("redo_selection", owner.model.undo.redo)
        assert row.values[hi_id] == 400
        stage("redo_edit", owner.model.undo.redo)
        assert row.values[hi_id] == 450
        assert asdict(original) == original_data
        project.view["current"] = owner.capture_view()
        print(json.dumps({"status": "fingerprinting_expected_state"}), flush=True)
        expected = fingerprint(project)
        expected_history_entries = sum(len(m.history) for m in project.science.measurements.values())
        recovery = owner.autosave_path()
        stage("background_autosave", lambda: wait_for_autosave(owner))
        assert file_digest(path) == original_file_digest
        result["recovery_file_bytes"] = recovery.stat().st_size
        # Do not retain the old project's scientific store during restoration.
        project = row = original = repeat = None
        stage("restore_autosave", lambda: owner.load_autosave(recovery))
        assert owner.path == path and owner.dirty
        print(json.dumps({"status": "verifying_restored_state"}), flush=True)
        assert fingerprint(owner.model.project) == expected
        assert file_digest(path) == original_file_digest
        result["recovery_verified"] = True
        record()
        assert stage("save_restored", owner.save)
        assert not recovery.exists()
        saved_digest = file_digest(path)
        # Exercise the ordinary Open action; only its file picker is replaced.
        with patch.object(QFileDialog, "getOpenFileName", return_value=(str(path), "")):
            stage("reopen_saved", owner.open_project)
        print(json.dumps({"status": "verifying_reopened_state"}), flush=True)
        assert fingerprint(owner.model.project) == expected
        assert file_digest(path) == saved_digest
        assert not owner.dirty
        current = owner.model.project.science.current(sid, pid)
        assert current.id == repeat_id and current.value == 9 and current.uncertainty == .3
        assert current.confidence_level == .95
        assert asdict(owner.model.project.science.measurements[original_id]) == original_data
        result["final_measurement_history_entries"] = sum(
            len(m.history) for m in owner.model.project.science.measurements.values())
        assert result["final_measurement_history_entries"] == expected_history_entries
        result.update(verified=True, final_file_bytes=path.stat().st_size, fingerprint=expected)
        record()
        print(json.dumps({"verified": True, "report": str(report)}), flush=True)
        return result
    finally:
        if owner._background_autosave is not None:
            owner._background_autosave.shutdown()
        owner.dirty = False
        owner.close()
        app.processEvents()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1300)
    parser.add_argument("--extra-columns", type=int, default=75)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--save-only", action="store_true", help="Measure explicit save/reopen without the autosave/recovery stages")
    parser.add_argument("--save-diagnostics", action="store_true", help="Observe nested save operations and snapshot release")
    parser.add_argument("--write-cache-mib", type=int, help="Benchmark-only SQLite write-cache budget; 0 uses SQLite default")
    parser.add_argument("--repeat-save", action="store_true", help="With save-only: edit a repeat measurement and overwrite the existing project")
    parser.add_argument("--defer-progress-start", action="store_true", help="Experiment: queue progress workers after entering the Qt event loop")
    parser.add_argument("--early-secondary-indexes", action="store_true", help="Control: build search indexes before inserting rows")
    parser.add_argument("--history-revisions", type=int, default=0, help="Synthetic depth measurement revisions per row")
    args = parser.parse_args()
    if args.rows < 4 or args.extra_columns < 0:
        parser.error("rows must be at least 4 and extra-columns non-negative")
    if args.history_revisions < 0:
        parser.error("history-revisions must be non-negative")
    if args.repeat_save and not args.save_only:
        parser.error("--repeat-save requires --save-only")
    if args.write_cache_mib is not None:
        if args.write_cache_mib < 0:
            parser.error("write-cache-mib must be non-negative")
        storage.WRITE_CACHE_KIB = args.write_cache_mib * 1024
    if args.early_secondary_indexes:
        storage.SCHEMA += storage.SECONDARY_INDEXES
        storage.SECONDARY_INDEXES = ""
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    app = QApplication.instance() or QApplication([])
    store = QSettings(str(directory / "benchmark-settings.ini"), QSettings.Format.IniFormat)
    store.setValue("general/autosave_enabled", False)
    with ExitStack() as patches:
        patches.enter_context(patch.object(i18n, "settings", return_value=store))
        if args.defer_progress_start:
            original_start = Worker.start
            def deferred_start(worker):
                QTimer.singleShot(0, lambda: original_start(worker))
            patches.enter_context(patch.object(Worker, "start", deferred_start))
        benchmark(app, directory, args.rows, args.extra_columns, save_only=args.save_only,
                  save_diagnostics=args.save_diagnostics, repeat_save=args.repeat_save,
                  defer_progress_start=args.defer_progress_start, history_revisions=args.history_revisions)


if __name__ == "__main__":
    main()
