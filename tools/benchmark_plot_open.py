"""Time and profile opening/restoring the plot workspace on a three-row project."""
import argparse
import cProfile
import io
import json
import platform
import pstats
import time
from pathlib import Path
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from autopyro.plotting import PlotSpec
from autopyro.ui.dialogs.plot import PlotDialog
from autopyro.ui.window import MainWindow
from tools.benchmark_dependent_edit import input_project


def open_plot(app, profile=None):
    owner = MainWindow()
    project = input_project(3)
    columns = {column.name: column.id for column in project.columns}
    project.view["plot_current"] = PlotSpec(x_id=columns["Depth"], y_id=columns["HI"]).to_dict()
    owner.set_project(project, prepared=True)
    stages = {}

    def timed(name, function):
        def call(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                stages.setdefault(name, []).append(time.perf_counter() - started)
        return call

    try:
        with patch.object(PlotDialog, "load_controls", timed("load_controls", PlotDialog.load_controls)), \
             patch.object(PlotDialog, "redraw", timed("redraw", PlotDialog.redraw)):
            started = time.perf_counter()
            if profile is None:
                owner.show_plots()
            else:
                profile.runcall(owner.show_plots)
            created = time.perf_counter() - started
            plot = owner.plot_window
            while plot.result is None or plot.timer.isActive():
                app.processEvents()
                if time.perf_counter() - started > 30:
                    raise TimeoutError(plot.status.text())
                time.sleep(.005)
            assert plot.spec.y_id == columns["HI"]
            assert sum(plot.result["plotted"].values()) == 3
            return {"create_seconds": created, "ready_seconds": time.perf_counter() - started,
                    "stages": stages}
    finally:
        owner.dirty = False
        owner.close()
        owner.deleteLater()
        app.processEvents()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    runs = [open_plot(app) for _ in range(3)]
    profile = cProfile.Profile()
    open_plot(app, profile)
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(25)
    result = {"platform": platform.platform(), "qt_platform": app.platformName(), "runs": runs,
              "profile": stream.getvalue(),
              "note": "Three fresh windows, three synthetic rows, saved Depth-HI spec. Separate profile covers synchronous show_plots only."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
