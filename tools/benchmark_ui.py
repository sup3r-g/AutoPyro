"""Measure synchronous Qt actions and delivery delay of a queued event."""
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication

from autopyro.core import Column, Project, Row
from autopyro.plotting import PlotSpec
from autopyro.ui.window import MainWindow


def measure(app, action, repeats):
    durations, event_delays = [], []
    for _ in range(repeats):
        app.processEvents()
        delivered = []
        started = time.perf_counter()
        QTimer.singleShot(0, lambda: delivered.append(time.perf_counter() - started))
        action()
        durations.append(time.perf_counter() - started)
        app.processEvents()
        assert delivered
        event_delays.append(delivered[0])
    return {"seconds": durations, "median_seconds": statistics.median(durations),
            "queued_event_delay_seconds": event_delays,
            "max_event_delay_seconds": max(event_delays)}


def benchmark(app, size, repeats):
    columns = [Column("Well"), Column("Depth", "number", unit="m"),
               Column("Value", "number")]
    project = Project(columns=columns, rows=[
        Row({columns[0].id: f"W{index % 10}", columns[1].id: index,
             columns[2].id: index % 100}) for index in range(size)])
    project.sync_measurements()
    window = MainWindow()
    try:
        window.set_project(project, prepared=True)
        window.show()
        app.processEvents()
        window.table.sortByColumn(1, Qt.SortOrder.AscendingOrder)
        rules = [{"column": columns[0].id, "operator": "eq", "value": "W0"}]
        filter_result = measure(app, lambda: window.apply_filter(rules, "AND"), repeats)
        assert window.proxy.rowCount() == (size + 9) // 10
        counter = 0

        def edit():
            nonlocal counter
            counter += 1
            assert window.model.setData(window.model.index(0, 2), str(100 + counter))

        edit_result = measure(app, edit, repeats)
        window.show_plots()
        plot = window.plot_window
        plot.timer.stop()
        plot.spec = PlotSpec(x_id=columns[1].id, y_id=columns[2].id, scope="filtered")
        plot.load_controls()
        plot.timer.stop()
        plot_result = measure(app, plot.redraw, repeats)
        assert plot.result is not None, plot.status.text()
        return {"project_rows": size, "visible_rows": window.proxy.rowCount(),
                "filter": filter_result, "edit": edit_result, "plot": plot_result}
    finally:
        window.dirty = False
        window.close()
        window.deleteLater()
        app.processEvents()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 30000, 100000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    result = {"python": sys.version, "platform": platform.platform(),
              "qt_platform": app.platformName(), "repeats": args.repeats,
              "note": "Synthetic three-column project, no derived columns. Active numeric sort, W0 filter. Plot uses visible samples. Event delay measures one queued callback across each synchronous action, not continuous latency or p95.",
              "measurements": []}
    for size in args.sizes:
        measurement = benchmark(app, size, args.repeats)
        result["measurements"].append(measurement)
        print(json.dumps(measurement), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
