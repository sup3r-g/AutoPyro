"""Measure building a 100-sample report from an in-memory project."""
import argparse
import gc
import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path

from autopyro.core import Column, Project, Row
from autopyro.reporting import ReportOptions, build_report
from autopyro.plotting import PlotSpec


def benchmark(size, grouped=False):
    columns = [Column("Depth", "number", unit="m"), Column("Value", "number")]
    if grouped:
        columns.append(Column("Family"))
    project = Project(columns=columns)
    project.rows = [Row({columns[0].id: index, columns[1].id: index % 100})
                    for index in range(size)]
    if grouped:
        for index, row in enumerate(project.rows):
            row.values[columns[2].id] = f"Family {index % 10}"
    project.sync_measurements()
    ids = [row.id for row in project.rows[:100]][::-1]
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    plots = []
    if grouped:
        spec = PlotSpec(x_id=columns[0].id, y_id=columns[1].id,
                        color_by=columns[2].id, shape_by=columns[2].id,
                        size_by=columns[2].id, size_mode="category", dpi=72)
        plots = [("Families", spec.to_dict())]
    report = build_report(project, ids, ReportOptions((columns[1].id,)), plots)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert report["row_ids"] == ids
    assert len(report["project"]["science"]["measurements"]) == len(ids) * len(columns)
    return {"project_rows": size, "report_rows": len(ids), "seconds": elapsed,
            "peak_python_bytes": peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000, 30000])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grouped", action="store_true", help="Include a categorical scatter plot")
    args = parser.parse_args()
    result = {"python": sys.version, "platform": platform.platform(),
              "grouped_plot": args.grouped,
              "note": "One run per size. Grouped mode includes PNG rendering. Tracemalloc overhead included. Peak Python allocations exclude input construction.",
              "measurements": []}
    for size in args.sizes:
        measurement = benchmark(size, args.grouped)
        result["measurements"].append(measurement)
        print(json.dumps(measurement), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
