"""Time shared property-panel switching separately from plotting and input loading."""
import argparse
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from PySide6.QtWidgets import QApplication

from autopyro.ui.window import MainWindow
from tools.benchmark_dependent_edit import input_project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('Repeats must be positive')
    app = QApplication.instance() or QApplication([])
    owner = MainWindow()
    owner.may_replace = lambda: True
    try:
        owner.set_project(input_project(16), prepared=True)
        owner.show()
        app.processEvents()
        started = perf_counter()
        owner.show_plots()
        dialog = owner.plot_window
        dialog.timer.stop()
        created_s = perf_counter() - started
        dialog.redraw()
        app.processEvents()
        runs = {kind: [] for kind in ('mpl:plot', 'mpl:imshow', 'mpl:errorbar',
                                      'mpl3d:scatter', 'mpl3d:plot_surface', 'scatter')}
        for _ in range(args.repeats):
            for kind, durations in runs.items():
                started = perf_counter()
                dialog.kind.setCurrentIndex(dialog.kind.findData(kind))
                dialog.timer.stop()  # Measure the panel, not the renderer.
                dialog.read_controls()
                app.processEvents()
                durations.append(perf_counter() - started)
        result = {'platform': platform.platform(), 'python': platform.python_version(),
                  'qt_platform': app.platformName(), 'window_create_s': created_s,
                  'note': 'One fresh window, synthetic 16-row project. Switching and layout, no plot redraw. '
                          'No baseline comparison, no RSS or native Qt latency claim.',
                  'switches': {kind: {'runs_s': durations, 'median_s': float(np.median(durations)),
                                      'p95_s': float(np.percentile(durations, 95))}
                               for kind, durations in runs.items()}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps({'window_create_s': created_s, 'switch_p95_s':
                         {kind: data['p95_s'] for kind, data in result['switches'].items()}}))
    finally:
        if owner.plot_window:
            owner.plot_window.close()
        owner.dirty = False
        owner.close()


if __name__ == '__main__':
    main()
