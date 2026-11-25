"""Measure initial and repeated table preparation on a synthetic wide project.

python -m tools.benchmark_measurement_sync --rows 1000 --extra-columns 75 --output tmp/prepare.json
Dataset construction and validation are outside the timed stages. No project file is opened.
"""
import argparse
import cProfile
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

from tools.benchmark_workspace_interpolation import dataset
from tools.benchmark_import_refresh import process_memory
from autopyro.ui.table.model import prepare_project


def table_digest(project):
    digest = hashlib.sha256()
    for row in project.rows:
        digest.update(json.dumps([row.id, row.values, row.original, row.source],
                                 ensure_ascii=False, allow_nan=False).encode("utf-8"))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--extra-columns", type=int, default=75)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.rows < 1 or args.extra_columns < 0:
        parser.error("rows must be positive and extra-columns non-negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    project = dataset(args.rows, args.extra_columns)
    before = table_digest(project)
    result = {"rows": args.rows, "source_columns": len(project.columns),
              "python": sys.version, "platform": platform.platform(),
              "processor": platform.processor(), "profiled": args.profile,
              "note": "One pass; no tracemalloc; dataset construction and validation excluded. "
                      "Process working-set peak includes construction and is cumulative. No GUI responsiveness measurement.",
              "stages": []}
    for name in ("initial_preparation", "existing_measurements"):
        gc.collect()
        profiler = cProfile.Profile() if args.profile else None
        started = perf_counter()
        if profiler:
            profiler.runcall(prepare_project, project)
        else:
            prepare_project(project)
        metrics = {"stage": name, "seconds": perf_counter() - started, **process_memory()}
        if profiler:
            profiler.dump_stats(str(args.output.with_suffix(f".{name}.prof")))
        result["stages"].append(metrics)
        print(json.dumps(metrics), flush=True)
        assert table_digest(project) == before
        assert len(project.science.measurements) == args.rows * len(project.columns)
        project.science.validate(project)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["verified"] = True
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
