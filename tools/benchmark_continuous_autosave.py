"""Measure recovery publication and Qt event gaps during ongoing edits."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
from time import perf_counter
from types import SimpleNamespace


def make_fixture(case):
    if case == "plot":
        from tools.benchmark_plot_memory import benchmark_project

        args = SimpleNamespace(rows=30000, columns=12, multi_track=True, science=True, history_revisions=5)
        return benchmark_project(args)[0], 1

    from autopyro.analysis.calculations import make_calculated_column
    from autopyro.ui.table.model import prepare_project
    from tools.benchmark_workspace_interpolation import dataset

    rows, columns, revisions = (1300, 20, 2) if case == "small" else (10000, 80, 10)
    project = prepare_project(dataset(rows, columns - 5))
    depth, toc, s2 = project.columns[2:5]
    for row in project.rows:
        value = row.values[depth.id]
        for revision in range(1, revisions + 1):
            project.science.set_value(row.id, depth.id, value + revision, depth.kind)
        row.values[depth.id] = value + revisions
    hi, values = make_calculated_column(project, "HI", {
        "formula": "HI", "version": 1, "compatible_confirmed": True, "analysis_context": "Synthetic benchmark",
        "inputs": {name: {"column_id": col.id, "unit": col.unit} for name, col in (("TOC", toc), ("S2", s2))},
    })
    project.columns.append(hi)
    for row in project.rows:
        row.values[hi.id] = values[row.id]
    row_id = project.rows[3].id
    repeat = project.science.add_result(
        row_id, s2.id, "8", analysis_fields={"method": "Pyrol6", "laboratory": "Synthetic repeat lab"},
        details={"uncertainty": .3, "uncertainty_type": "standard", "confidence_level": .95},
    )
    project.select_measurement(row_id, s2.id, repeat.id)
    return project, 4


def open_plot(owner, app, project):
    from autopyro.visualization.plotting import PlotSpec

    columns = project.columns
    owner.apply_filter([{"column": columns[2].id, "operator": "eq", "value": "G0"}], "AND")
    owner.show_plots()
    dialog = owner.plot_window
    tracks = [col.id for col in columns[1:2] + columns[3:12]]
    dialog.spec = PlotSpec(kind="multi_depth", x_id=tracks[0], y_id=columns[0].id, track_ids=tracks,
                           scope="filtered", width_mm=768 * 25.4 / 96, height_mm=672 * 25.4 / 96)
    dialog.load_controls()
    dialog.canvas_view.mode_combo.setCurrentIndex(dialog.canvas_view.mode_combo.findData(False))
    dialog.prepare_async()
    from PySide6.QtTest import QTest

    deadline = perf_counter() + 60
    while dialog.result is None and perf_counter() < deadline:
        app.processEvents()
        QTest.qWait(10)
    assert dialog.result["plotted"] == dict.fromkeys(tracks, 15000)
    assert dialog.canvas.size().toTuple() == (768, 672)
    return dialog


def exercise(owner, directory, position, column_number):
    from PySide6.QtCore import QEventLoop, Qt, QTimer
    from tools.benchmark_plot_memory import gap_statistics

    loop = QEventLoop()
    publications, errors, gaps = [], [], []
    value = 1000

    def edit():
        nonlocal value
        value += 1
        assert owner.model.setData(owner.model.index(position, column_number), str(value))

    def heartbeat():
        nonlocal previous
        now = perf_counter()
        gaps.append(now - previous)
        previous = now

    def published(revision):
        path = directory / f"publication-{len(publications) + 1}.apyro"
        os.link(owner.autosave_path(), path)
        publications.append({"revision": revision, "seconds": perf_counter() - started,
                             "live_value": value, "path": path.name})
        if len(publications) == 2:
            edits.stop()
            edit()  # The follow-up must capture this edit without another timer tick.
        elif len(publications) == 3:
            loop.quit()

    def failed(error):
        errors.append(error)
        loop.quit()

    pulse, edits, timeout = QTimer(), QTimer(), QTimer()
    pulse.setTimerType(Qt.TimerType.PreciseTimer)
    pulse.timeout.connect(heartbeat)
    edits.setTimerType(Qt.TimerType.PreciseTimer)
    edits.timeout.connect(edit)
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    started = previous = perf_counter()
    pulse.start(5)
    edit()
    owner.autosave()
    job = owner._background_autosave
    job.saved.connect(published)
    job.failed.connect(failed)
    edits.start(200)
    timeout.start(300_000)
    try:
        loop.exec()
    finally:
        timeout.stop()
        edits.stop()
        pulse.stop()
        job.saved.disconnect(published)
        job.failed.disconnect(failed)
    assert not errors, errors
    assert len(publications) == 3, publications
    return {"publications": publications, "final_value": value, **gap_statistics(gaps)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("small", "wide", "plot"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    directory.mkdir(parents=True)
    os.environ.setdefault("QT_QPA_PLATFORM", "windows")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from autopyro.database.storage import file_digest
    from autopyro.ui import i18n
    from autopyro.ui.window import MainWindow
    from tools.benchmark_large_project_storage import fingerprint

    settings = QSettings(str(directory / "settings.ini"), QSettings.Format.IniFormat)
    i18n.settings = lambda: settings
    app = QApplication([])
    i18n.initialize()
    project, column_number = make_fixture(args.case)
    position = 0 if args.case == "plot" else 3
    row_id, column_id = project.rows[position].id, project.columns[column_number].id
    measurement = project.science.current(row_id, column_id)
    path = directory / "original.apyro"
    project.save(path)
    original_digest = file_digest(path)
    owner = MainWindow()
    owner.autosave_timer.stop()
    owner.set_project(project, path, prepared=True)
    owner.show()
    app.processEvents()
    assert owner.windowHandle().isExposed()
    try:
        plot = open_plot(owner, app, project) if args.case == "plot" else None
        print(f"Recording {args.case} autosaves", flush=True)
        result = exercise(owner, directory, position, column_number)
        result.update(case=args.case, original_rows=len(project.rows),
                      qt_platform=app.platformName(), edit_interval_ms=200)
        if plot is not None:
            artist = plot.result["artists"][0]
            point = artist._autopyro_row_ids.index(row_id)
            assert artist.get_offsets()[point, 0] == result["final_value"]
            assert plot.canvas.size().toTuple() == (768, 672)
            result.update(tracks=len(plot.figure.axes), filtered_rows=owner.proxy.rowCount(),
                          canvas=plot.canvas.size().toTuple())
        previous_value = 0
        for item in result["publications"]:
            copy = directory / item["path"]
            with closing(sqlite3.connect(copy.as_uri() + "?mode=ro", uri=True)) as db:
                raw_row, = db.execute("SELECT payload FROM table_rows WHERE position=?", (position,)).fetchone()
                raw_measurement, = db.execute("SELECT payload FROM measurements WHERE id=?", (measurement.id,)).fetchone()
            saved = json.loads(raw_row)["values"][column_id]
            record = json.loads(raw_measurement)
            assert previous_value < saved <= item["live_value"] and saved == record["value"]
            item.update(saved_value=saved, measurement_revisions=len(record["history"]))
            previous_value = saved
        assert previous_value == result["final_value"]
        owner.load_autosave(owner.autosave_path())
        # A later frame can update display state after the snapshot was taken.
        project.view["current"] = owner.model.project.view["current"]
        assert fingerprint(owner.model.project) == fingerprint(project)
        assert file_digest(path) == original_digest
        result.update(recovery_verified=True, original_unchanged=True)
        (directory / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
    finally:
        owner.dirty = False
        owner.close()
        app.aboutToQuit.emit()


if __name__ == "__main__":
    main()
