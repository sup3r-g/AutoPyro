"""Measure retained project data after reopening a disposable synthetic SQLite file.

python -m tools.benchmark_science_loading --rows 1000 --extra-columns 75 --output tmp/load-before.json
No existing user project is read or overwritten. Construction/save/checks are outside load timing.
"""
import argparse
import gc
import json
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

from autopyro.core import Project
from autopyro.importing import digest
from tools.benchmark_import_memory import graph_size
from tools.benchmark_import_refresh import process_memory
from tools.benchmark_workspace_interpolation import dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--extra-columns", type=int, default=75)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rows < 1 or args.extra_columns < 0:
        parser.error("rows must be positive and extra-columns non-negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="science-load-", dir=args.output.parent) as directory:
        path = Path(directory) / "synthetic.apyro"
        project = dataset(args.rows, args.extra_columns)
        project.save(path)
        expected = digest(project)
        file_bytes = path.stat().st_size
        del project
        gc.collect()
        started = perf_counter()
        loaded = Project.load(path)
        elapsed = perf_counter() - started
        gc.collect()
        # Capture the working set before graph traversal allocates its visited set.
        memory = process_memory()
        size = graph_size((loaded.columns, loaded.rows, loaded.science, loaded.source), set())
        assert digest(loaded) == expected
        result = {"rows": args.rows, "source_columns": len(loaded.columns),
                  "measurements": len(loaded.science.measurements), "seconds": elapsed,
                  "retained_graph_bytes": size, "file_bytes": file_bytes, **memory,
                  "python": sys.version, "platform": platform.platform(),
                  "processor": platform.processor(), "verified": True,
                  "note": "One pass; no tracemalloc; load includes validation. Retained graph counts "
                          "each reachable object once. Cumulative process peak includes construction/save. "
                          "File was just saved: this is not a cold filesystem-cache test. "
                          "No GUI responsiveness or physical-display measurement."}
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
