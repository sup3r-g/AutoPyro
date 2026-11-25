"""Compare native Qt progress loops with the same snapshot workload."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QEventLoop, QSettings
from PySide6.QtWidgets import QApplication

from autopyro.model.snapshots import copy_project
from autopyro.ui import i18n
from autopyro.ui.tasks import TaskDialog, run_with_progress
from autopyro.ui.window import MainWindow
from tools.benchmark_workspace_interpolation import dataset, measure


def open_and_wait(dialog):
    loop = QEventLoop(dialog)
    dialog.finished.connect(loop.quit)
    dialog.open()
    if dialog.isVisible():
        loop.exec()
    return dialog.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    app = QApplication([])
    if app.platformName() != "windows":
        parser.error("Set QT_QPA_PLATFORM=windows for this experiment")
    args.directory.mkdir(parents=True, exist_ok=False)
    project = dataset(10000, 75)
    project.sync_measurements()
    depth = project.columns[2]
    for row in project.rows:
        initial = row.values[depth.id]
        for offset in range(1, 11):
            project.science.set_value(row.id, depth.id, initial + offset, depth.kind)
        row.values[depth.id] = initial + 10

    settings = QSettings(str(args.directory.resolve() / "settings.ini"), QSettings.Format.IniFormat)
    with patch.object(i18n, "settings", return_value=settings):
        compare(app, project, args.directory)


def compare(app, project, directory):
    def prepare_snapshot():
        snapshot = copy_project(project)
        snapshot.sync_measurements()
        snapshot.validate()
        return len(snapshot.science.measurements)

    owner = MainWindow()
    owner.autosave_timer.stop()
    owner.set_project(project, prepared=True)
    owner.show()
    app.processEvents()
    assert owner.windowHandle().isExposed()
    original_exec = TaskDialog.exec
    report = {"qt_platform": app.platformName(), "rows": 10000, "columns": 80,
              "history_entries": 100000, "runs": []}
    try:
        for mode in ("exec", "open", "open", "exec"):
            wait = original_exec if mode == "exec" else open_and_wait
            with patch.object(TaskDialog, "exec", wait):
                count, metrics = measure(app, mode, lambda: run_with_progress(owner, "Snapshot benchmark", prepare_snapshot))
            assert count == len(project.science.measurements)
            report["runs"].append(metrics)
            (directory / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        owner.close()


if __name__ == "__main__":
    main()
