"""Measure main-thread input copying and undo commands for laboratory append."""
import argparse
import cProfile
import gc
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from autopyro.importing import digest, preview_import, prepare_import, suggest_mapping
from autopyro.io import build_project
from autopyro.ui.table.commands import ImportCommand
from autopyro.ui.table.model import TableModel, snapshot_project


def measure(app, name, function, *, profile_path=None):
    gc.collect()
    app.processEvents()
    timer = QTimer()
    timer.setInterval(10)
    gaps = []
    collections, collecting = [], {}
    main_thread = threading.get_ident()
    previous = time.perf_counter()

    def gc_event(phase, info):
        key = (threading.get_ident(), info["generation"])
        if phase == "start":
            collecting[key] = time.perf_counter()
        elif key in collecting:
            collections.append({"generation": key[1], "thread": key[0],
                                "seconds": time.perf_counter() - collecting.pop(key)})

    def tick():
        nonlocal previous
        current = time.perf_counter()
        gaps.append(current - previous)
        previous = current

    timer.timeout.connect(tick)
    gc.callbacks.append(gc_event)
    timer.start()
    started = previous = time.perf_counter()
    profiler = cProfile.Profile() if profile_path else None
    try:
        value = profiler.runcall(function) if profiler else function()
    finally:
        gc.callbacks.remove(gc_event)
    elapsed = time.perf_counter() - started
    tick()
    timer.stop()
    if profiler:
        profiler.dump_stats(str(profile_path))
    timing = {"stage": name, "seconds": elapsed, "max_qt_event_gap_seconds": max(gaps),
              "gc_collections": len(collections),
              "gc_total_seconds": sum(item["seconds"] for item in collections),
              "gc_max_seconds": max((item["seconds"] for item in collections), default=0),
              "gc_main_seconds": sum(item["seconds"] for item in collections if item["thread"] == main_thread),
              "gc_worker_seconds": sum(item["seconds"] for item in collections if item["thread"] != main_thread)}
    print(json.dumps(timing), flush=True)
    return value, timing


def dataset(size, *, incoming=False):
    return build_project(["Sample", "Well", "TOC", "S2"],
        [[f"{i:07d}", f"W{i % 20}", "2", "8" if incoming else "6"] for i in range(size)],
        list(range(2, size + 2)), [(0, "Sample", "text", ""), (1, "Well", "text", ""),
         (2, "TOC", "number", "%wt"), (3, "S2", "number", "mg HC/g rock")],
        {"sha256": "second" if incoming else "first"})


def benchmark(app, size, *, profile_directory=None):
    project = dataset(size)
    model = TableModel(project)
    original_ids = [row.id for row in project.rows]
    incoming = dataset(size + size // 10, incoming=True)
    incoming.sync_measurements()
    plan = preview_import(project, incoming, suggest_mapping(project, incoming), [incoming.columns[0].id])
    prepared, protocol = prepare_import(project, incoming, plan)
    stages = []

    def stage(name, function):
        profile_path = profile_directory / f"commit-{size}-{name}.prof" if profile_directory else None
        value, timing = measure(app, name, function, profile_path=profile_path)
        stages.append(timing)
        return value

    snapshot = stage("input_snapshot", lambda: snapshot_project(project))
    assert digest(snapshot) == plan.project_digest
    del snapshot
    command = stage("command_constructor", lambda: ImportCommand(model, prepared, protocol, plan.project_digest))
    stage("first_redo", lambda: model.undo.push(command))
    assert len(project.rows) == size + size // 10
    stage("undo", model.undo.undo)
    assert [row.id for row in project.rows] == original_ids
    assert len(project.science.measurements) == size * 4
    stage("redo", model.undo.redo)
    assert len(project.rows) == size + size // 10
    assert [row.id for row in project.rows[:size]] == original_ids
    assert project.rows[0].values[project.columns[3].id] == 6
    model.undo.clear()
    return {"rows": size, "columns": 4, "initial_measurements": size * 4, "stages": stages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[30000])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true", help="Profile each stage, including Qt-entered callbacks")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    result = {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
              "heartbeat_interval_ms": 10, "profiled": args.profile,
              "note": "Synthetic narrow project. Calls execute on the GUI thread. Preparation of the append plan/result, window construction, external views and rendering are excluded. One pass. Profiling changes timings and may include worker callbacks entered through Qt; GC callback records collections in both threads.",
              "measurements": []}
    for size in args.sizes:
        result["measurements"].append(benchmark(app, size,
            profile_directory=args.output.parent if args.profile else None))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
