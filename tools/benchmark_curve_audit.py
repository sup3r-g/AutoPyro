"""Isolated curve-fitting audit timings (run without competing benchmarks).

python -m tools.benchmark_curve_audit --gui-rows 100000 --output benchmarks/curve-audit.json
"""
import argparse
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter, sleep

import numpy as np
import scipy

from autopyro.curve_fitting import fit_curve, standard_model
from autopyro.curve_bootstrap import bootstrap_bands
from autopyro.model_comparison import FitCandidate, compare_models


def measure(function, repeats=3):
    function()  # Warm imports/allocators before collecting timings.
    times = []
    for _ in range(repeats):
        started = perf_counter()
        function()
        times.append(perf_counter()-started)
    return {"median_seconds": median(times), "samples_seconds": times}


def measure_gui(size):
    from tempfile import TemporaryDirectory
    from PySide6.QtCore import QSettings, QTimer
    from PySide6.QtWidgets import QApplication
    from autopyro.core import Column, Project, Row
    from autopyro.ui import i18n
    from autopyro.ui.window import MainWindow

    app = QApplication.instance() or QApplication([])
    previous_settings = i18n.settings
    with TemporaryDirectory() as directory:
        store = QSettings(str(Path(directory)/"benchmark.ini"), QSettings.Format.IniFormat)
        i18n.settings = lambda: store
        i18n.initialize(store)
        window = MainWindow()
        cx, cy = Column(name="X", kind="number"), Column(name="Y", kind="number")
        x = np.linspace(0, 10, size)
        rows = [Row(values={cx.id: float(v), cy.id: float(2*v+1+np.sin(v))}) for v in x]
        # Exclude project import/measurement preparation: this measures the fitting UI.
        window.set_project(Project(columns=[cx, cy], rows=rows), prepared=True)
        window.show()
        window.show_curve_fitting()
        dialog = window.curve_fitting_window
        app.processEvents()
        drawn = []
        dialog.canvas.mpl_connect("draw_event", lambda event: drawn.append(perf_counter())
                                 if dialog.result is not None and len(dialog.figure.axes) == 2 else None)
        ticks = [perf_counter()]
        timer = QTimer()
        timer.setInterval(20)
        timer.timeout.connect(lambda: ticks.append(perf_counter()))
        timer.start()
        started = perf_counter()
        dialog.calculate()
        prepared = perf_counter()-started
        deadline = perf_counter()+60
        try:
            while not drawn or dialog.worker.isRunning():
                app.processEvents()
                if perf_counter() > deadline:
                    raise RuntimeError("Fitting UI benchmark timed out")
                sleep(.001)
            app.processEvents()
            ticks.append(perf_counter())
            result = {"rows": size, "prepare_seconds": prepared, "first_plot_seconds": drawn[0]-started,
                      "max_event_gap_seconds": max(np.diff(ticks)),
                      "diagnostics_deferred": dialog.residual_table.model() is None}
        finally:
            timer.stop()
            if dialog.worker and dialog.worker.isRunning():
                dialog._bootstrap_cancel.set()
                dialog.worker.wait()
            window.dirty = False
            window.close()
            app.aboutToQuit.emit()
            i18n.settings = previous_settings
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui-rows", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    x = np.linspace(0, 10, 100_000)
    y = 2*x+1+np.random.default_rng(42).normal(0, .3, len(x))
    models = [FitCandidate("line", standard_model()),
              FitCandidate("quadratic", standard_model("polynomial", degree=2))]
    report = {"numpy": np.__version__, "scipy": scipy.__version__, "python": platform.python_version(),
              "ols_100000": measure(lambda: fit_curve(x, y)),
              "odr_100000": measure(lambda: fit_curve(x, y, method="odr")),
              "huber_100000": measure(lambda: fit_curve(x, y, loss="huber")),
              "group_cv_100000_2_models": measure(lambda: compare_models(x, y, models, groups=np.arange(len(x))//100,
                                                                         strategy="group", folds=5))}
    fit = fit_curve(x[:10_000], y[:10_000], loss="huber", bounds=([0, -10], [2, 10]))
    report["bootstrap_200_10000"] = measure(lambda: bootstrap_bands(fit, np.linspace(0, 10, 400), resamples=200), repeats=1)
    if args.gui_rows:
        report["gui"] = measure_gui(args.gui_rows)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
