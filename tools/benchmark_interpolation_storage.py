"""Measure snapshot retention, SQLite save/open and Python allocations in fresh processes.

Save: python tools/benchmark_interpolation_storage.py save tmp/packed.apyro --rows 3000
Baseline: same with a different path and --inline. Open: ... open tmp/packed.apyro
Uses the real Qt table command, without mounting charts or the full desktop.
tracemalloc timings are diagnostic, not ordinary application latency.
"""
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import tracemalloc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from autopyro.audit import archive_protocol
from autopyro.core import Column, Project
from autopyro.interpolation import prepare, replay
from autopyro.io import build_project
from autopyro.ui.table import AddColumnCommand, TableModel


def graph_bytes(roots):
    """Shallow sizes of distinct plain JSON objects; shared strings count once."""
    seen, pending, size = set(), list(roots), 0
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        size += sys.getsizeof(item)
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return size


def make_project(rows):
    records = [[f"Well {i // 1000}", str(i % 1000),
                "" if i % 5 == 2 else str((i % 1000) ** 1.2)] for i in range(rows)]
    return build_project(["Well", "Depth", "Y"], records, list(range(2, rows + 2)),
                         [(0, "Well", "text", ""), (1, "Depth", "number", "m"),
                          (2, "Y", "number", "mg/g")], {"type": "synthetic benchmark"})


def prepare_column(project, inline):
    group, x, y = project.columns
    parameters = dict(x_id=x.id, y_id=y.id, group_ids=[group.id], method="pchip", max_gap=10,
                      constraints={"well_id": group.id, "analytical_context": True})
    started = perf_counter()
    proposals, protocol = prepare(project, row_ids=set(project.row_map()), **parameters)
    prepare_seconds = perf_counter() - started
    started = perf_counter()
    stored = protocol if inline else archive_protocol(protocol)
    archive_seconds = perf_counter() - started
    column = Column("Interpolated Y", "number", unit=y.unit, metadata={
        **parameters, "stale": False, "scope_row_ids": sorted(project.row_map()),
        "proposals": [asdict(p) for p in proposals], "interpolation_protocol": stored})
    # Neither a raw protocol nor eligibility diagnostics escape this scope when packed.
    return column, {p.row_id: p.value for p in proposals}, {
        "prepare_seconds": prepare_seconds, "archive_seconds": archive_seconds,
        "proposals": len(proposals), "archived": isinstance(stored, str)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("save", "open"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--inline", action="store_true")
    args = parser.parse_args()
    app = QApplication([])
    TableModel(Project())  # Warm scientific imports before measuring allocations.
    result = {"phase": args.phase, "python": platform.python_version(), "platform": platform.platform(),
              "note": "tracemalloc; warm imports; no charts; additional Python allocations only"}
    if args.phase == "save":
        if args.path.exists():
            parser.error("Use a fresh output path; existing files are not overwritten")
        if args.rows < 10:
            parser.error("--rows must be at least 10")
        project = make_project(args.rows)
        model = TableModel(project)
        gc.collect()
        tracemalloc.start()
        column, values, metrics = prepare_column(project, args.inline)
        result.update(metrics)
        started = perf_counter()
        model.undo.push(AddColumnCommand(model, column, values))
        result["commit_seconds"] = perf_counter() - started
        gc.collect()
        result["retained_python_bytes"], result["prepare_peak_python_bytes"] = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        started = perf_counter()
        project.save(args.path)
        result["save_seconds"] = perf_counter() - started
        _, result["save_peak_python_bytes"] = tracemalloc.get_traced_memory()
    else:
        tracemalloc.start()
        started = perf_counter()
        project = Project.load(args.path)
        result["open_seconds"] = perf_counter() - started
        gc.collect()
        result["retained_python_bytes"], result["open_peak_python_bytes"] = tracemalloc.get_traced_memory()
        result["archived"] = isinstance(project.columns[-1].metadata["interpolation_protocol"], str)
    tracemalloc.stop()
    result.update(rows=len(project.rows), measurements=len(project.science.measurements),
                  file_bytes=args.path.stat().st_size,
                  snapshot_graph_bytes=graph_bytes([
                      project.columns[-1].metadata["interpolation_protocol"],
                      project.history[-1]["metadata"]["interpolation_protocol"]]))
    # Replay after the allocation measurement, without touching the project.
    started = perf_counter()
    result["reproduced_values"] = len(replay(project.columns[-1].metadata["interpolation_protocol"]))
    result["replay_seconds_untraced"] = perf_counter() - started
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
