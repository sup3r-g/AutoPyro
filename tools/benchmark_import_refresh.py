"""Measure file import and snapshot refresh, including Qt heartbeat gaps.

Run: python -m tools.benchmark_import_refresh --sizes 1000 5000 --output tmp/import-before.json
Use --memory for a separate run with tracemalloc overhead, --profile-size for cProfile.
"""
import argparse
from contextlib import closing
import cProfile
import gc
import json
import os
from pathlib import Path
import platform
import sqlite3
import sys
from tempfile import TemporaryDirectory
import time
import tracemalloc

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from autopyro.database.refresh import preview_refresh, prepare_refresh
from autopyro.database.sources import read_snapshot
from autopyro.importing import preview_import, prepare_import, suggest_mapping, digest
from autopyro.io import build_project, read_file, read_sqlite, file_source
from autopyro.ui.tasks import Worker


def process_memory():
    """Windows working set, including the peak since process start (not per stage)."""
    if sys.platform != "win32":
        return {}
    import ctypes

    class Counters(ctypes.Structure):
        _fields_ = [("size", ctypes.c_ulong), ("page_faults", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in (
                "peak_working_set", "working_set", "peak_paged_pool", "paged_pool",
                "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile")]

    counters = Counters()
    counters.size = ctypes.sizeof(counters)
    get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    get_info.restype = ctypes.c_int
    if not get_info(ctypes.c_void_p(-1), ctypes.byref(counters), counters.size):
        return {}
    return {"working_set_bytes": counters.working_set,
            "process_peak_working_set_bytes": counters.peak_working_set}


def measure(name, function, *, memory=False, profile_path=None):
    gc.collect()
    loop, timer = QEventLoop(), QTimer()
    timer.setInterval(10)
    gaps, outcome = [], {}
    previous = time.perf_counter()

    def tick():
        nonlocal previous
        current = time.perf_counter()
        gaps.append(current - previous)
        previous = current

    def run():
        if profile_path is None:
            return function()
        profiler = cProfile.Profile()
        try:
            return profiler.runcall(function)
        finally:
            profiler.dump_stats(str(profile_path))

    worker = Worker(run)
    worker.succeeded.connect(lambda value: outcome.update(value=value))
    worker.failed.connect(lambda error: outcome.update(error=error))
    worker.finished.connect(loop.quit)
    timer.timeout.connect(tick)
    if memory:
        tracemalloc.start()
    started = previous = time.perf_counter()
    timer.start()
    worker.start()
    loop.exec()
    worker.wait()
    elapsed = time.perf_counter() - started
    tick()
    timer.stop()
    peak = tracemalloc.get_traced_memory()[1] if memory else None
    if memory:
        tracemalloc.stop()
    if "error" in outcome:
        raise RuntimeError(outcome["error"])
    return outcome["value"], {"stage": name, "seconds": elapsed,
        "max_qt_event_gap_seconds": max(gaps), "peak_python_bytes": peak}


def benchmark(size, directory, *, memory=False, profile=False):
    path = directory / f"source-{size}.sqlite"
    records = [(f"{i:07d}", f"W{i % 20}", "2", "ND" if i % 20 == 0 else "6") for i in range(size)]
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE lab (Sample TEXT PRIMARY KEY, Well TEXT, TOC TEXT, S2 TEXT)")
        connection.executemany("INSERT INTO lab VALUES (?,?,?,?)", records)
        connection.commit()
    source = {"type": "sqlite", "path": str(path), "table": "lab", "read_only": True}
    headers, rows, numbers = read_sqlite(path, "lab")
    source["source_columns"] = headers
    selections = [(i, name, "text" if i < 2 else "number", ["", "", "%wt", "mg HC/g rock"][i])
                  for i, name in enumerate(headers)]
    project = build_project(headers, rows, numbers, selections, source)
    project.sync_measurements()
    original_digest = digest(project)
    csv = directory / f"repeat-{size}.csv"
    csv.write_text("Sample;Well;TOC;S2\n" + "".join(
        f"{i:07d};W{i % 20};2;8\n" for i in range(size + size // 10)), encoding="utf-8")
    stages = []

    def stage(name, function):
        profile_path = directory.parent / f"import-{size}-{name}.prof" if profile else None
        value, timing = measure(name, function, memory=memory, profile_path=profile_path)
        timing.update(process_memory())
        stages.append(timing)
        print(json.dumps({"rows": size, **timing}), flush=True)
        return value

    def read_append():
        incoming = build_project(*read_file(csv), selections, file_source(csv))
        incoming.sync_measurements()
        return incoming

    incoming = stage("append_read", read_append)
    plan = stage("append_preview", lambda: preview_import(project, incoming,
        suggest_mapping(project, incoming), [incoming.columns[0].id]))
    result, protocol = stage("append_prepare", lambda: prepare_import(project, incoming, plan))
    assert len(result.rows) == size + size // 10 and protocol["new_samples"] == size // 10
    assert result.rows[1].values[project.columns[3].id] == 6
    del result, protocol, plan, incoming
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("UPDATE lab SET S2='8'")
        connection.execute("DELETE FROM lab WHERE CAST(Sample AS INTEGER) >= ?", (size - size // 10,))
        connection.executemany("INSERT INTO lab VALUES (?,?,?,?)", [
            (f"{i:07d}", f"W{i % 20}", "2", "8") for i in range(size, size + size // 10)])
        connection.commit()
    source_bytes = path.read_bytes()
    remote = stage("refresh_read", lambda: read_snapshot(source))
    plan = stage("refresh_preview", lambda: preview_refresh(project, remote, [project.columns[0].id]))
    decisions = {entry["id"]: "keep" for entry in plan.missing_rows}
    result, protocol = stage("refresh_prepare", lambda: prepare_refresh(project, remote, plan, decisions))
    assert len(result.rows) == size + size // 10
    assert result.rows[1].values[project.columns[3].id] == 8
    assert len(plan.missing_rows) == size // 10 and protocol["working_changed"]
    assert digest(project) == original_digest and path.read_bytes() == source_bytes
    return {"rows": size, "columns": 4, "initial_measurements": size * 4,
            "new_rows_each_batch": size // 10, "missing_refresh_rows": size // 10, "stages": stages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000])
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--profile-size", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(size < 10 for size in args.sizes):
        parser.error("sizes must be at least 10")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    result = {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
              "memory_instrumentation": args.memory, "heartbeat_interval_ms": 10,
              "process_memory_initial": process_memory(),
              "process_memory_note": "Windows working set includes native allocations and existing inputs; peak is cumulative since process start, not a stage allocation peak.",
              "note": "Synthetic four-column data, one pass per stage in a Qt worker; GUI construction and GUI-side input copies, undo snapshots and disk project save excluded. First append_prepare includes cold scientific-module imports. Python allocation peaks exclude existing inputs and native allocations. Profiling changes timing.",
              "measurements": []}
    with TemporaryDirectory(dir=args.output.parent) as directory:
        for size in args.sizes:
            result["measurements"].append(benchmark(size, Path(directory), memory=args.memory,
                                                   profile=size == args.profile_size))
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    app.processEvents()


if __name__ == "__main__":
    main()
