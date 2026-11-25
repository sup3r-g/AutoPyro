"""Measure edits with pyrolysis calculations and live statistics/plot windows."""
import argparse
import json
import platform
import sqlite3
import time
from contextlib import contextmanager, ExitStack
from functools import wraps
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from autopyro.calculations import FORMULAS, make_calculated_column
from autopyro.core import Column, Project, Row
from autopyro.plotting import PlotSpec
from autopyro.ui.window import MainWindow


@contextmanager
def profile_stages(plot, statistics):
    from autopyro.ui.dialogs import plot as plot_module, statistics as statistics_module

    stages = {}

    def measured(name, function):
        @wraps(function)
        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                stages.setdefault(name, []).append(time.perf_counter() - started)
        return timed

    def measured_snapshot(function):
        @wraps(function)
        def steps(*args, **kwargs):
            iterator = function(*args, **kwargs)
            advance = measured("statistics_snapshot", lambda: next(iterator))
            while True:
                step = advance()
                yield step
                if step is not None:
                    return
        return steps

    targets = [(statistics_module, "calculate_statistics", "statistics_compute"),
               (statistics, "display_report", "statistics_display"),
               (plot_module, "render_plot", "plot_prepare"),
               (plot.canvas, "draw", "canvas_draw"),
               (plot, "refresh_categories", "plot_categories")]
    with ExitStack() as stack:
        stack.enter_context(patch.object(statistics_module, "snapshot_steps",
                                         measured_snapshot(statistics_module.snapshot_steps)))
        for owner, attribute, name in targets:
            stack.enter_context(patch.object(owner, attribute, measured(name, getattr(owner, attribute))))
        yield stages


def input_project(size, source=None):
    units = {"Depth": "m", "TOC": "%wt", "S1": "mg HC/g rock",
             "S2": "mg HC/g rock", "S3": "mg CO2/g rock"}
    columns = [Column(name, "number", unit=unit) for name, unit in units.items()]
    if source is None:
        values = [[index, 2., .5, 6., 1.2] for index in range(size)]
    else:
        # Read only the requested cells, without loading the wide source project.
        with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            catalog = [json.loads(row[0]) for row in connection.execute("SELECT payload FROM table_columns")]
            by_name = {column["name"]: column for column in catalog}
            paths = [f'$.values."{by_name[name]["id"]}"' for name in units]
            expressions = ", ".join("json_extract(payload, ?)" for _ in paths)
            values = connection.execute(
                f"SELECT {expressions} FROM table_rows ORDER BY position LIMIT ?", [*paths, size]).fetchall()
    project = Project(columns=columns, rows=[Row(dict(zip([c.id for c in columns], row))) for row in values])
    project.sync_measurements()
    by_name = {column.name: column for column in columns}
    for formula, definition in FORMULAS.items():
        config = {"formula": formula, "version": 1, "compatible_confirmed": True,
                  "analysis_context": "Computational performance QA only. Canonical input units assumed, not independently validated.",
                  "inputs": {role: {"column_id": by_name[role].id, "unit": by_name[role].unit}
                             for role in definition["roles"]}}
        column, calculated = make_calculated_column(project, formula, config)
        project.columns.append(column)
        for row in project.rows:
            row.values[column.id] = calculated[row.id]
    return project


def wait_ready(app, statistics, plot):
    started = time.perf_counter()
    max_gap = 0
    previous = started
    while True:
        app.processEvents()
        now = time.perf_counter()
        max_gap = max(max_gap, now - previous)
        previous = now
        worker_ready = statistics.worker is None or not statistics.worker.isRunning()
        if (statistics.report is not None and worker_ready and not statistics.timer.isActive()
                and plot.result is not None and not plot.timer.isActive()):
            return max_gap
        if now - started > 120:
            raise TimeoutError("Live windows did not finish within 120 seconds")
        time.sleep(.005)


def benchmark(app, size, source):
    project = input_project(size, source)
    window = MainWindow()
    try:
        window.set_project(project, prepared=True)
        window.show()
        window.show_statistics()
        statistics = window.statistics_window
        for index in range(statistics.properties.count()):
            item = statistics.properties.item(index)
            column = next(c for c in project.columns if c.id == item.data(Qt.ItemDataRole.UserRole))
            item.setCheckState(Qt.CheckState.Checked if column.name in ("HI", "GP") else Qt.CheckState.Unchecked)
        statistics.request_refresh()
        window.show_plots()
        plot = window.plot_window
        by_name = {column.name: column for column in project.columns}
        plot.spec = PlotSpec(x_id=by_name["Depth"].id, y_id=by_name["HI"].id)
        plot.load_controls()
        plot.schedule()
        wait_ready(app, statistics, plot)
        target = next(index for index, row in enumerate(project.rows)
                      if isinstance(row.values[by_name["TOC"].id], (int, float))
                      and row.values[by_name["TOC"].id] > 0)
        results = []
        for value in (7., 8., 9.):
            with profile_stages(plot, statistics) as stages:
                started = time.perf_counter()
                assert window.model.setData(window.model.index(target, 3), str(value))
                edit_time = time.perf_counter() - started
                event_gap = wait_ready(app, statistics, plot)
            assert project.rows[target].values[by_name["HI"].id] == value / project.rows[target].values[by_name["TOC"].id] * 100
            results.append({"edit_seconds": edit_time, "settled_seconds": time.perf_counter() - started,
                            "max_gap_while_settling_seconds": event_gap, "stage_seconds": stages})
        return {"rows": len(project.rows), "measurements": results}
    finally:
        if window.statistics_window and window.statistics_window.worker:
            window.statistics_window.worker.wait()
        window.dirty = False
        window.close()
        window.deleteLater()
        app.processEvents()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    result = {"platform": platform.platform(), "qt_platform": app.platformName(),
              "source": str(args.source) if args.source else "synthetic",
              "note": "Five input columns, HI/OI/PI/GP, statistics for HI/GP and a live Depth-HI plot. Source values use assumed canonical units for computational QA only. Source scientific records are not imported. Event gaps cover settling after setData, synchronous edit duration is recorded separately.",
              "runs": []}
    for size in args.sizes:
        run = benchmark(app, size, args.source)
        result["runs"].append(run)
        print(json.dumps(run), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
