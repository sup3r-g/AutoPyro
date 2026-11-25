"""Separate snapshot, cold/warm preparation and renderer costs; optional Qt heartbeat."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
from time import perf_counter, sleep

import numpy as np

from autopyro.core import Column, Row, Project
from autopyro.plot_data import PlotDataCache, capture_input, input_token
from autopyro.plotting import PlotSpec, render_plot


def inputs(size):
    x, y, group = Column('X', 'number'), Column('Y', 'number'), Column('Group', 'text')
    project = Project(columns=[x, y, group], rows=[Row({x.id: i, y.id: 1 + (i % 997) / 100,
                      group.id: 'G' + str(i % 12)}) for i in range(size)])
    spec = PlotSpec(x_id=x.id, y_id=y.id, color_by=x.id, color_mode='numeric', shape_by=group.id)
    return project, spec


def measure(size):
    project, spec = inputs(size)
    ids = [r.id for r in project.rows]
    started = perf_counter()
    source = capture_input(project, ids, [spec], input_token(project, 0, [spec]))
    captured = perf_counter()
    cache = PlotDataCache()
    prepared = cache.prepare(source, ids, spec)
    cold = perf_counter()
    warm = []
    misses = cache.misses
    for _ in range(20):
        before = perf_counter()
        prepared = cache.prepare(source, ids, spec)
        warm.append(perf_counter() - before)
    assert cache.misses == misses
    renders = {'existing': [], 'prepared': []}
    for _ in range(3):
        for name in renders:
            before = perf_counter()
            figure, result = render_plot(project, ids, deepcopy(spec), prepared=prepared if name == 'prepared' else None)
            renders[name].append(perf_counter() - before)
            assert result['plotted'][spec.x_id] == size
            figure.clear()
    return {'rows': size, 'capture_s': captured - started, 'cold_prepare_s': cold - captured,
            'warm_runs_s': warm, 'warm_p95_s': float(np.percentile(warm, 95)),
            'renderer_runs_s': renders, 'parsed_cache_bytes': cache.bytes,
            'hits': cache.hits, 'misses': cache.misses}


def measure_gui(size, output):
    import gc
    from PySide6.QtWidgets import QApplication
    from autopyro.ui.window import MainWindow
    app = QApplication.instance() or QApplication([])
    owner = MainWindow()
    owner.may_replace = lambda: True
    project, spec = inputs(size)
    owner.set_project(project, prepared=True)
    owner.show()
    owner.show_plots()
    dialog = owner.plot_window
    dialog.timer.stop()
    dialog.spec = spec
    dialog.load_controls()
    dialog.schedule()
    dialog.timer.stop()
    failures = []
    collections, gc_start = [], None
    def record_gc(phase, info):
        nonlocal gc_start
        if phase == 'start':
            gc_start = perf_counter()
        elif gc_start is not None:
            collections.append({'generation': info['generation'], 's': perf_counter() - gc_start})
    gc.callbacks.append(record_gc)
    installs = []
    original_redraw = dialog.redraw
    def tracked_redraw(*args, **kwargs):
        before = perf_counter()
        try:
            return original_redraw(*args, **kwargs)
        finally:
            installs.append(perf_counter() - before)
    dialog.redraw = tracked_redraw
    dialog.export_service.failed.disconnect(dialog.export_failed)
    dialog.export_service.failed.connect(failures.append)
    def wait_until(condition, started):
        previous, gap = started, 0
        while not condition():
            app.processEvents()
            now = perf_counter()
            gap = max(gap, now - previous)
            previous = now
            if now - started > 90:
                raise TimeoutError(dialog.status.text())
            sleep(.005)
        return perf_counter() - started, gap
    try:
        started = perf_counter()
        dialog.prepare_async()
        request_s = perf_counter() - started
        ready_s, preview_gap_s = wait_until(lambda: dialog.result is not None and dialog.preparation.worker is None, started)
        from matplotlib.image import imsave
        imsave(output.with_name(output.stem + '-preview.png'), np.asarray(dialog.canvas.buffer_rgba()))
        started = perf_counter()
        dialog.prepare_async()
        warm_ready_s, warm_gap_s = wait_until(lambda: dialog.result is not None and not dialog.preview.busy
                                             and dialog.preparation.worker is None, started)
        ax = dialog.figure.axes[0]
        low, high = ax.get_xlim()
        ax.set_xlim(low + (high - low) / 4, high - (high - low) / 4)
        started = perf_counter()
        dialog.canvas.draw()
        navigation_s, navigation_gap_s = wait_until(lambda: not dialog.preview.busy, started)
        path = output.parent / f'plot-cache-gui-{size}.png'
        started = perf_counter()
        jobs = dialog.make_export_jobs([('Benchmark', path, dialog.spec)])
        dialog.export_service.start(jobs)
        export_request_s = perf_counter() - started
        export_s, export_gap_s = wait_until(lambda: dialog.export_service.worker is None, started)
        assert not failures and path.exists(), failures
        return {'rows': size, 'qt_platform': app.platformName(), 'request_s': request_s,
                'preview_ready_s': ready_s, 'preview_max_gap_s': preview_gap_s,
                'warm_preview_s': warm_ready_s, 'warm_preview_max_gap_s': warm_gap_s,
                'navigation_s': navigation_s, 'navigation_max_gap_s': navigation_gap_s,
                'frame_install_runs_s': installs,
                'longest_gc': sorted(collections, key=lambda item: item['s'], reverse=True)[:5],
                'export_request_s': export_request_s, 'export_s': export_s, 'export_max_gap_s': export_gap_s}
    finally:
        gc.callbacks.remove(record_gc)
        dialog.close()
        dialog.preparation.shutdown()
        dialog.preview.shutdown()
        dialog.export_service.shutdown()
        owner.dirty = False
        owner.close()
        app.processEvents()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[1000, 10000, 100000])
    parser.add_argument('--gui-size', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if any(size < 1 for size in args.sizes) or (args.gui_size is not None and args.gui_size < 1):
        parser.error('Sizes must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {'platform': platform.platform(), 'python': platform.python_version(), 'measurements': [],
              'note': 'Synthetic table-only numeric color and 12 marker groups. No MeasurementStore/RSS. '
                      '20 warm preparation calls, 3 renderer calls without canvas drawing; GUI numbers are single runs.'}
    for size in args.sizes:
        data = measure(size)
        result['measurements'].append(data)
        print(json.dumps({k: data[k] for k in ('rows', 'capture_s', 'cold_prepare_s', 'warm_p95_s', 'parsed_cache_bytes')}), flush=True)
    if args.gui_size:
        result['gui'] = measure_gui(args.gui_size, args.output)
        print(json.dumps(result['gui']), flush=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
