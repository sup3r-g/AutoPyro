"""Profile worker snapshots without running Qt windows or statistical calculations."""
import argparse
import cProfile
import io
import json
import platform
import pstats
import time
from copy import deepcopy
from pathlib import Path

from autopyro.statistics import snapshot
from autopyro.calculations import snapshot_diagnostics
from tools.benchmark_dependent_edit import input_project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = input_project(args.size)
    ids = [row.id for row in project.rows]
    runs = {}
    profile = cProfile.Profile()
    for label, names in (("derived", ("HI", "GP")), ("measured", ("TOC", "S2"))):
        columns = [column.id for column in project.columns if column.name in names]
        durations = []
        for _ in range(3):
            started = time.perf_counter()
            frozen = snapshot(project, ids, columns)
            durations.append(time.perf_counter() - started)
            assert len(frozen.rows) == len(ids)
        runs[label] = durations
        profile.runcall(snapshot, project, ids, columns)
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(22)
    diagnostics = [column.metadata["diagnostics"] for column in project.columns
                   if column.metadata.get("calculation")]
    diagnostic_runs = {"deepcopy": [], "formula_snapshot": []}
    for _ in range(3):
        for label, copy_function in (("deepcopy", deepcopy), ("formula_snapshot", snapshot_diagnostics)):
            started = time.perf_counter()
            copied = [copy_function(items) for items in diagnostics]
            diagnostic_runs[label].append(time.perf_counter() - started)
            assert copied == diagnostics
    result = {"platform": platform.platform(), "rows": args.size, "seconds": runs,
              "diagnostics_seconds": diagnostic_runs,
              "profile": stream.getvalue(), "note": "Synthetic project, three unprofiled snapshots per mode. Separate cumulative cProfile run includes both modes."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"seconds": runs, "diagnostics_seconds": diagnostic_runs}), flush=True)


if __name__ == "__main__":
    main()
