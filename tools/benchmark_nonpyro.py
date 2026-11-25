"""Repeatable non-pyrolysis benchmark. Run with python -m tools.benchmark_nonpyro."""
import argparse
import gc
import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path

from autopyro.core import Column, Project, Row
from autopyro.nonpyro import evaluate_project


def benchmark(size):
    columns = [Column("H/C", "number", unit="1"), Column("O/C", "number", unit="1")]
    project = Project(columns=columns)
    project.rows = [Row({columns[0].id: 1.2, columns[1].id: 0.1}) for _ in range(size)]
    config = {
        "version": 1, "method": "atomic_ratios", "confirmed": True,
        "context": "Synthetic performance benchmark",
        "scope_row_ids": [row.id for row in project.rows],
        "inputs": {key: {"column_id": column.id, "unit": "1"}
                   for key, column in zip(("hc", "oc"), columns)},
    }
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    report = evaluate_project(project, config["scope_row_ids"], config)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert report["summary"] == {"profile": size}
    assert all(item["values"] == {"HC": 1.2, "OC": 0.1} for item in report["results"])
    return {"rows": size, "seconds": elapsed, "peak_python_bytes": peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 30000, 100000])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"python": sys.version, "platform": platform.platform(),
              "processor": platform.processor(), "columns": 2,
              "note": "Synthetic in-memory data. Timing includes tracemalloc overhead. Peak is Python allocations during evaluation, excluding input construction.",
              "measurements": []}
    for size in args.sizes:
        measurement = benchmark(size)
        result["measurements"].append(measurement)
        print(json.dumps(measurement), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
