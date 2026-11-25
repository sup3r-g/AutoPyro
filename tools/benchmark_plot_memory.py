"""Fresh-process preview latency and sampled Windows process-tree memory.

Run each size in a fresh interpreter. Sampling includes the GUI and its owned
preview, export and recovery children. Working sets can count shared pages more than once;
private bytes measure committed private memory, not just resident pages.
"""
import argparse
from collections import Counter
from contextlib import ExitStack, closing
from copy import deepcopy
import ctypes
from ctypes import wintypes
import gc
import json
from math import ceil
import os
from pathlib import Path
import platform
import sqlite3
from tempfile import TemporaryDirectory
from threading import Event, Thread, get_ident
from time import perf_counter, sleep
from unittest.mock import patch


class Memory:
    def __init__(self):
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('peak_ws', 'ws', 'peak_paged', 'paged',
                'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile', 'private')]
        self.Counters = Counters
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.info = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
        self.info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        self.info.restype = wintypes.BOOL

    def read(self, pid):
        handle = self.kernel.OpenProcess(0x0400 | 0x0010, False, pid)
        if not handle:
            return None
        try:
            data = self.Counters()
            data.cb = ctypes.sizeof(data)
            if not self.info(handle, ctypes.byref(data), data.cb):
                return None
            return {'pid': pid, 'working_set_bytes': data.ws, 'private_bytes': data.private}
        finally:
            self.kernel.CloseHandle(handle)


def gap_statistics(intervals):
    ordered = sorted(intervals)
    return {
        'event_gap_count': len(ordered),
        'p95_event_gap_s': ordered[ceil(.95 * len(ordered)) - 1] if ordered else None,
        'max_event_gap_s': ordered[-1] if ordered else 0,
        'event_gaps_over_100ms': sum(gap > .1 for gap in ordered),
    }


def pause_trace(origin, pauses, spans):
    """Relate observed GUI gaps to work on the same monotonic clock."""
    trace = []
    for pause in pauses:
        start, end = pause['start'], pause['end']
        overlaps = []
        for span in spans:
            overlap = min(end, span['end']) - max(start, span['start'])
            if overlap > 0:
                overlaps.append({'name': span['name'], 'thread': span['thread'],
                                 'start_s': span['start'] - origin,
                                 'end_s': span['end'] - origin, 'overlap_s': overlap})
        trace.append({'phase': pause['phase'], 'start_s': start - origin,
                      'end_s': end - origin, 'duration_s': end - start,
                      'before_dispatch_s': pause['dispatch'] - start,
                      'dispatch_s': end - pause['dispatch'], 'overlaps': overlaps})
    return trace


def benchmark_project(args):
    from autopyro.core import Column, Project, Row

    columns = [Column('X', 'number'), Column('Y', 'number'), Column('Group', 'text')]
    columns += [Column('Extra' + str(i), 'number') for i in range(args.columns - 3)]
    groups = 2 if args.multi_track else 12
    rows = [Row({c.id: ('G' + str(i % groups) if n == 2 else i if n == 0 else 1 + (i % 997) / 100)
                 for n, c in enumerate(columns)}) for i in range(args.rows)]
    project = Project(columns=columns, rows=rows)
    science_setup_s = None
    history_setup_s = None
    if args.science:
        started = perf_counter()
        project.sync_measurements()
        science_setup_s = perf_counter() - started
    if args.history_revisions:
        started = perf_counter()
        column = columns[1]
        for row in rows:
            original = row.values[column.id]
            for revision in range(1, args.history_revisions + 1):
                project.science.set_value(row.id, column.id, original + revision, column.kind)
            row.values[column.id] = original + args.history_revisions
        history_setup_s = perf_counter() - started
    return project, science_setup_s, history_setup_s


def measure_live_gc(args):
    """Measure full collections with different parts of a snapshot held live."""
    from autopyro.model import snapshots

    project, _, _ = benchmark_project(args)
    meter = Memory()
    states = {}

    def probe(name):
        samples = []
        for _ in range(4):
            started = perf_counter()
            collected = gc.collect(2)
            samples.append({'s': perf_counter() - started, 'collected': collected})
        memory = meter.read(os.getpid())
        tracked = Counter(type(value).__name__ for value in gc.get_objects())
        states[name] = {'full_gc': samples, 'memory': memory,
                        'tracked_objects': tracked.total(),
                        'tracked_types': dict(tracked.most_common(12))}

    probe('original')
    memo = snapshots._measurement_memo(project.science)
    probe('measurement_memo')
    snapshots._table_memo(project.rows, project.science, memo)
    copied = deepcopy(project, memo)
    probe('snapshot_and_memo')
    del memo
    probe('snapshot')
    assert copied.rows == project.rows and copied.science == project.science
    assert copied.rows[0] is not project.rows[0]
    del copied
    probe('after_release')
    return {'rows': args.rows, 'columns': args.columns,
            'history_revisions': args.history_revisions, 'states': states,
            'verified': True}


def measure(args, recovery_path=None, autosave_spans=(), writer_pids=()):
    # Keep imports inside this function: spawn must not import Qt/the desktop
    # through the benchmark entry point before entering the render process.
    if args.qt_platform is None:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    else:
        os.environ['QT_QPA_PLATFORM'] = args.qt_platform
    from PySide6.QtWidgets import QApplication
    from autopyro.core import Project
    from autopyro.plotting import PlotSpec
    from autopyro.ui.window import MainWindow
    app = QApplication.instance() or QApplication([])
    if args.qt_platform is not None:
        assert app.platformName() == args.qt_platform
    owner = MainWindow()
    owner.may_replace = lambda: True
    project, science_setup_s, history_setup_s = benchmark_project(args)
    columns, rows = project.columns, project.rows
    owner.set_project(project, prepared=True)
    owner.show()
    app.processEvents()
    owner.autosave_timer.stop()  # Start recovery at a reproducible point if requested.
    if recovery_path is not None:
        owner.autosave_path = lambda: recovery_path
    track_ids = [c.id for c in (columns[1:2] + columns[3:12])] if args.multi_track else []
    expected_scope = None
    filter_apply_s = None
    if track_ids:
        expected_scope = frozenset(rows[position].id for position in range(0, args.rows, 2))
        started = perf_counter()
        owner.apply_filter([{'column': columns[2].id, 'operator': 'eq', 'value': 'G0'}], 'AND')
        assert owner.proxy.rowCount() == len(expected_scope)
        filter_apply_s = perf_counter() - started
    gc.collect()
    meter, stop = Memory(), Event()
    records, phase = [], 'baseline'
    def pids():
        result = {os.getpid(), *writer_pids}
        dialog = owner.plot_window
        if dialog:
            for service in (dialog.preview, dialog.export_service):
                worker = service.worker
                process = worker.process if worker else None
                if process is not None and process.pid is not None:
                    result.add(process.pid)
        return result
    def snapshot():
        processes = [data for pid in pids() if (data := meter.read(pid)) is not None]
        return {'phase': phase, 'processes': processes,
                'working_set_bytes': sum(p['working_set_bytes'] for p in processes),
                'private_bytes': sum(p['private_bytes'] for p in processes)}
    baseline = snapshot()
    trace_origin = perf_counter()
    long_pauses, plot_spans = [], []
    measurements_before = len(project.science.measurements)
    collections, collection_start = [], None
    installs = []
    def record_gc(event, info):
        nonlocal collection_start
        if event == 'start':
            collection_start = perf_counter(), phase, get_ident()
        elif collection_start is not None:
            started, name, thread = collection_start
            ended = perf_counter()
            if args.trace_pauses:
                plot_spans.append({'name': f'GC gen {info["generation"]}', 'thread': thread,
                                   'start': started, 'end': ended})
            collections.append({'phase': name, 's': ended - started,
                                'generation': info['generation'], 'thread': thread,
                                'collected': info['collected'], 'uncollectable': info['uncollectable']})
    gc.callbacks.append(record_gc)
    def sample():
        while not stop.is_set():
            records.append(snapshot())
            stop.wait(.02)
    sampler = Thread(target=sample)
    sampler.start()
    stages = []
    preparation = {}
    requested_at = None
    def capture_ready(generation, source):
        preparation.setdefault(phase, {})['snapshot_ready_s'] = perf_counter() - requested_at
    def numeric_ready(generation, prepared):
        preparation.setdefault(phase, {})['numeric_ready_s'] = perf_counter() - requested_at
    def wait(condition, started, timeout=180):
        previous, intervals = started, []
        while not condition():
            dispatch = perf_counter() if args.trace_pauses else None
            app.processEvents()
            now = perf_counter()
            intervals.append(now - previous)
            if args.trace_pauses and now - previous > .1:
                long_pauses.append({'phase': phase, 'start': previous, 'dispatch': dispatch, 'end': now})
            previous = now
            if now - started > timeout:
                raise TimeoutError(f'{phase}: {owner.plot_window.status.text()}')
            sleep(.005)
        records.append(snapshot())
        return perf_counter() - started, intervals
    def record(name, seconds, intervals):
        if recovery_path is None:
            assert owner._background_autosave is None, 'Autosave interfered with the plot benchmark'
        samples = [r for r in records if r['phase'] == name]
        gc_samples = [c for c in collections if c['phase'] == name]
        longest = max(gc_samples, key=lambda c: c['s']) if gc_samples else None
        gc_by_generation = {}
        for generation in range(3):
            runs = [c for c in gc_samples if c['generation'] == generation]
            gc_by_generation[str(generation)] = {
                'runs': len(runs), 'total_s': sum(c['s'] for c in runs),
                'max_s': max((c['s'] for c in runs), default=0),
                'collected': sum(c['collected'] for c in runs),
                'uncollectable': sum(c['uncollectable'] for c in runs)}
        stages.append({'name': name, 's': seconds, **gap_statistics(intervals), 'end': snapshot(),
                       'longest_gc': longest,
                       'gc_by_generation': gc_by_generation,
                       'frame_install_runs_s': [s for n, s in installs if n == name],
                       'transfer': dict(owner.plot_window.preview.worker.last_transfer),
                       'preparation': preparation.get(name, {}),
                       'sampled_peak_working_set_bytes': max(r['working_set_bytes'] for r in samples),
                       'sampled_peak_private_bytes': max(r['private_bytes'] for r in samples)})
        if track_ids:
            assert dialog.result['plotted'] == dict.fromkeys(track_ids, len(expected_scope))
            assert len(dialog.figure.axes) == len(track_ids)
            assert dialog.prepared.scope == expected_scope
        if args.canvas_size:
            width, height = args.canvas_size
            assert dialog.canvas.size().toTuple() == (width, height)
            ratio = dialog.canvas.device_pixel_ratio
            assert dialog.canvas.buffer_rgba().shape == (round(height * ratio), round(width * ratio), 4)
    try:
        phase = 'first'
        opened = perf_counter()
        owner.show_plots()
        dialog = owner.plot_window
        dialog.preparation.capture.ready.connect(capture_ready)
        dialog.preparation.ready.connect(numeric_ready)
        original_redraw = dialog.redraw
        def timed_redraw(*values, **options):
            started = perf_counter()
            try:
                return original_redraw(*values, **options)
            finally:
                ended = perf_counter()
                installs.append((phase, ended - started))
                if args.trace_pauses:
                    plot_spans.append({'name': 'frame_install', 'thread': get_ident(),
                                       'start': started, 'end': ended})
        dialog.redraw = timed_redraw
        open_s = perf_counter() - opened
        dialog.timer.stop()
        if track_ids:
            dialog.spec = PlotSpec(kind='multi_depth', x_id=track_ids[0], y_id=columns[0].id,
                                   track_ids=track_ids, scope='filtered', width_mm=500)
        else:
            dialog.spec = PlotSpec(x_id=columns[0].id, y_id=columns[1].id,
                                   color_by=columns[0].id, color_mode='numeric', shape_by=columns[2].id)
        if args.canvas_size:
            dialog.spec.width_mm, dialog.spec.height_mm = (pixels * 25.4 / 96 for pixels in args.canvas_size)
        dialog.load_controls()
        if args.canvas_size:
            dialog.canvas_view.mode_combo.setCurrentIndex(dialog.canvas_view.mode_combo.findData(False))
        started = perf_counter()
        requested_at = started
        dialog.prepare_async()
        seconds, intervals = wait(lambda: dialog.result is not None and dialog.preparation.worker is None, started)
        record(phase, seconds, intervals)
        first_from_open_s = perf_counter() - opened
        if not track_ids:
            assert dialog.result['plotted'][columns[0].id] == args.rows
        captured_measurements = len(dialog.prepared.project.science.measurements)
        assert captured_measurements == 0
        for i in range(args.repeats):
            phase = 'repeat-' + str(i + 1)
            started = perf_counter()
            requested_at = started
            dialog.prepare_async()
            seconds, intervals = wait(lambda: dialog.result is not None and dialog.preparation.worker is None
                                and not dialog.preview.busy, started)
            record(phase, seconds, intervals)
        recovery = None
        edited_rows = {}
        for edit in range(args.edits):
            phase = 'edit' if args.edits == 1 else 'edit-' + str(edit + 1)
            edit_started = perf_counter()
            position = 2 * (edit % len(expected_scope)) if track_ids else edit % args.rows
            value = 123 + edit
            if value == rows[position].values[columns[1].id]:
                value += 1
            assert owner.model.setData(owner.model.index(position, 1), str(value))
            edited_rows[position] = value
            edit_call_s = perf_counter() - edit_started
            dialog.timer.stop()
            started = perf_counter()
            requested_at = started
            dialog.prepare_async()
            overlap = recovery_path is not None and edit == args.edits - 1
            if overlap:
                recovery_started = perf_counter()
                if args.autosave_immediate:
                    with patch.object(dialog, 'isVisible', return_value=False):
                        owner.autosave()
                else:
                    owner.autosave()
                job = owner._background_autosave
                assert job is not None, 'Background autosave did not queue'
                published = []
                errors = []
                job.saved.connect(lambda revision: published.append((revision, perf_counter())))
                job.failed.connect(errors.append)
            seconds, intervals = wait(lambda: dialog.result is not None and dialog.preparation.worker is None
                                and not dialog.preview.busy, started)
            record(phase, seconds, intervals)
            stages[-1]['edit_call_s'] = edit_call_s
            stages[-1]['max_interaction_gap_s'] = max(edit_call_s, stages[-1]['max_event_gap_s'])
            if track_ids:
                row_id = rows[position].id
                assert dialog.prepared.local.at[row_id, track_ids[0]] == value
                artist = dialog.result['artists'][0]
                point = artist._autopyro_row_ids.index(row_id)
                assert artist.get_offsets()[point, 0] == value
            if overlap:
                phase = 'autosave-tail'
                _, tail_intervals = wait(lambda: bool(published or errors) and job.worker is None, perf_counter(), timeout=300)
                assert not errors, errors
                assert len(published) == 1 and published[0][0] == owner.analysis_revision
                recovery = {'published_s': published[0][1] - recovery_started,
                            **gap_statistics(intervals + tail_intervals),
                            'bytes': recovery_path.stat().st_size}
        cache_bytes = dialog.preparation.cache.bytes
        canvas = {'width': dialog.canvas.width(), 'height': dialog.canvas.height(),
                  'device_pixel_ratio': dialog.canvas.device_pixel_ratio}
        # Diagnostic only, after all interactive measurements. This is not an
        # application GC policy or part of the stage latency/pause results.
        gc_probe_s = []
        if args.gc_probe:
            phase = 'gc-probe'
            for _ in range(5):
                started = perf_counter()
                gc.collect(2)
                gc_probe_s.append(perf_counter() - started)
        phase = 'closed'
        dialog.close()
        wait(lambda: dialog.preview.worker is None and dialog.preparation.worker is None, perf_counter())
        gc.collect()
        app.processEvents()
        closed = snapshot()
        stop.set()
        sampler.join()
        if recovery is not None:
            with closing(sqlite3.connect(recovery_path.as_uri() + '?mode=ro', uri=True)) as db:
                assert db.execute('PRAGMA quick_check').fetchone() == ('ok',)
                assert db.execute('SELECT count(*) FROM table_rows').fetchone() == (args.rows,)
                working_count, = db.execute('SELECT count(*) FROM working_measurements').fetchone()
                assert working_count == args.rows * args.columns
                for position, value in edited_rows.items():
                    payload, = db.execute('SELECT payload FROM table_rows WHERE position=?',
                                          (position,)).fetchone()
                    assert json.loads(payload)['values'][columns[1].id] == value
                if args.history_revisions:
                    history_entries, = db.execute(
                        "SELECT sum(json_array_length(payload, '$.history')) FROM measurements"
                    ).fetchone()
                    assert history_entries == args.rows * args.history_revisions + args.edits
                    recovery['history_entries'] = history_entries
            recovery['verified'] = True
            recovery['working_measurements'] = working_count
            from tools.benchmark_large_project_storage import fingerprint

            project.view['current'] = owner.capture_view(include_sample_set=True)
            expected = fingerprint(project)
            owner.load_autosave(recovery_path)
            assert owner.dirty and owner.recovered_from == recovery_path
            assert fingerprint(owner.model.project) == expected
            recovery['restored_fingerprint'] = expected
        return {'rows': args.rows, 'columns': args.columns, 'measurements_before': measurements_before,
                'science_setup_s': science_setup_s,
                'history_setup_s': history_setup_s,
                'history_revisions': args.history_revisions,
                'track_count': len(track_ids),
                'filtered_rows': len(expected_scope) if track_ids else None,
                'filter_apply_s': filter_apply_s,
                'measurements_after_edit': len(project.science.measurements),
                'preview_snapshot_measurements': captured_measurements,
                'qt_platform': app.platformName(), 'gui_thread': get_ident(),
                'canvas': canvas,
                'open_s': open_s, 'first_from_open_s': first_from_open_s,
                'first_after_window_s': first_from_open_s - open_s,
                'baseline': baseline, 'stages': stages, 'closed_after_gc': closed,
                'parsed_cache_bytes': cache_bytes,
                'gc_probe_s': gc_probe_s,
                'autosave': recovery,
                'long_pause_trace': pause_trace(trace_origin, long_pauses, [*plot_spans, *autosave_spans])
                if args.trace_pauses else None,
                'sampled_peak_private_bytes': max(r['private_bytes'] for r in records),
                'sampled_peak_working_set_bytes': max(r['working_set_bytes'] for r in records)}
    finally:
        gc.callbacks.remove(record_gc)
        stop.set()
        sampler.join()
        if owner._background_autosave is not None:
            owner._background_autosave.shutdown()
        if owner.plot_window:
            owner.plot_window.close()
            owner.plot_window.preparation.shutdown()
            owner.plot_window.preview.shutdown()
        owner.dirty = False
        owner.close()
        app.processEvents()


def experimental_analysis_copy(record, memo):
    """Copy a plain Analysis directly for the historical snapshot experiment."""
    copied = object.__new__(type(record))
    memo[id(record)] = copied
    copied.__dict__.update(deepcopy(vars(record), memo))
    return copied


def measure_autosave(args, recovery_path):
    """Trace transfer and publication without changing the writer process."""
    from autopyro.database import recovery_writer
    from autopyro.ui import autosave

    phases, spans, writer_pids = {}, [], []
    popen = recovery_writer.Popen

    def start_writer(*values, **options):
        process = popen(*values, **options)
        writer_pids.append(process.pid)
        return process

    def timed(name, operation):
        def run(*values, **options):
            started = perf_counter()
            try:
                return operation(*values, **options)
            finally:
                ended = perf_counter()
                phases[name] = ended - started
                if args.trace_pauses:
                    spans.append({'name': name, 'thread': get_ident(), 'start': started, 'end': ended})
        return run

    with ExitStack() as stack:
        stack.enter_context(patch.object(recovery_writer, 'Popen', start_writer))
        stack.enter_context(patch.object(recovery_writer, '_send_project',
                                         timed('snapshot_transfer', recovery_writer._send_project)))
        stack.enter_context(patch.object(autosave, 'write_recovery', timed('recovery_writer', autosave.write_recovery)))
        result = measure(args, recovery_path, spans, writer_pids)
    result['autosave']['phase_s'] = phases
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=100000)
    parser.add_argument('--columns', type=int, default=3)
    parser.add_argument('--science', action='store_true')
    parser.add_argument('--qt-platform', choices=('offscreen', 'windows'),
                        help='Override QT_QPA_PLATFORM, which defaults to offscreen')
    parser.add_argument('--history-revisions', type=int, default=0,
                        help='Create this many revisions per row in the edited numeric column')
    parser.add_argument('--multi-track', action='store_true',
                        help='Plot ten depth tracks for the first of two groups')
    parser.add_argument('--canvas-size', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
                        help='Hold every frame to these logical pixels in actual-size mode')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--edits', type=int, default=1)
    parser.add_argument('--autosave', action='store_true',
                        help='Start background recovery alongside the final edited preview')
    parser.add_argument('--autosave-immediate', action='store_true',
                        help='Control: start autosave without giving the active graph a head start')
    parser.add_argument('--trace-pauses', action='store_true',
                        help='Correlate event-loop gaps over 100 ms with timed work')
    parser.add_argument('--trace-copy-collections', action='store_true',
                        help='Historical probe for the former in-process writer')
    parser.add_argument('--experimental-analysis-copy', action='store_true',
                        help='Historical probe for the former in-process writer')
    parser.add_argument('--gc-probe', action='store_true',
                        help='Time five explicit full collections after the interactive stages')
    parser.add_argument('--gc-live-probe', action='store_true',
                        help='Measure full GC with the source, memo and snapshot held live, without Qt')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.autosave_immediate and not args.autosave:
        parser.error('--autosave-immediate requires --autosave')
    if args.trace_copy_collections or args.experimental_analysis_copy:
        parser.error('Deepcopy probes require the historical benchmark and in-process recovery writer')
    if os.name != 'nt':
        parser.error('This benchmark uses Windows process memory counters')
    if args.rows < 1 or args.columns < 3 or args.repeats < 1 or args.edits < 1:
        parser.error('rows/repeats/edits must be positive; columns must be at least 3')
    if args.multi_track and args.columns < 12:
        parser.error('--multi-track requires at least 12 columns')
    if args.canvas_size and any(not 114 <= pixels <= 3779 for pixels in args.canvas_size):
        parser.error('--canvas-size must be within the editor size controls (114 to 3779 pixels)')
    if args.history_revisions < 0 or args.history_revisions and not args.science:
        parser.error('--history-revisions must be nonnegative and requires --science when positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.gc_live_probe:
        if args.autosave or args.gc_probe or not args.science:
            parser.error('--gc-live-probe requires --science and runs separately from autosave and --gc-probe')
        result = measure_live_gc(args)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps({name: state['tracked_objects'] for name, state in result['states'].items()}))
        return
    if args.autosave and args.rows * args.columns < 100_000:
        parser.error('--autosave requires at least 100,000 table cells')
    from PySide6.QtCore import QSettings
    with TemporaryDirectory(prefix='plot-bench-', dir=args.output.parent.resolve()) as directory:
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
            QSettings.setPath(QSettings.Format.IniFormat, scope, directory)
        if args.autosave:
            result = measure_autosave(args, Path(directory) / 'recovery.apyro')
        else:
            result = measure(args)
    data = {'platform': platform.platform(), 'python': platform.python_version(),
            'repeats': args.repeats, 'edits': args.edits,
            'experimental_analysis_copy': args.experimental_analysis_copy,
            'autosave_immediate': args.autosave_immediate,
            'sampling_s': .02, 'note': 'Fresh GUI; owned processes only; sampled peaks can miss short spikes. '
            'Sum of working sets can double-count shared pages. '
            'No cold disk-cache guarantee.', **result}
    args.output.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(json.dumps({k: data[k] for k in ('rows', 'columns', 'first_from_open_s',
        'sampled_peak_private_bytes', 'sampled_peak_working_set_bytes')}), flush=True)


if __name__ == '__main__':
    main()
