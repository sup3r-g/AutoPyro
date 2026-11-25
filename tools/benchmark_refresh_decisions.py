"""Isolate refresh decision validation from copying, hashing and source reads."""
import argparse
import json
from pathlib import Path
import platform
import sys

from autopyro.database.refresh import RefreshPlan, _validate_decisions
from tools.benchmark_import_refresh import measure, QApplication


def benchmark(size):
    count = size // 10
    # Same shapes/counts as benchmark_import_refresh: 90% changed, 10% new/missing.
    cells = [{"id": f"sample-{i}:S2", "sample_id": f"sample-{i}", "property_id": "S2",
              "key": str(i), "base": "6", "local": 6, "source": "8",
              "status": "source_changed", "action": "source", "remote_changed": True}
             for i in range(size - count)]
    new_rows = [{"id": f"new:{i}", "key": str(i), "action": "add", "invalid": [],
                 "values": {"S2": "8"}, "source": {"row": i}} for i in range(count)]
    missing = [{"id": f"removed:{i}", "sample_id": f"sample-{i}", "key": str(i), "action": "unresolved"}
               for i in range(count)]
    plan = RefreshPlan("", "", {}, {}, cells, new_rows, missing, {}, [], "synthetic")
    decisions = {item["id"]: "keep" for item in missing}
    _, timing = measure("validate_decisions", lambda: _validate_decisions(plan, decisions))
    result = {"source_rows": size, "cells": len(cells), "new_rows": count, "missing_rows": count, **timing}
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 30000, 100000])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(size < 10 for size in args.sizes):
        parser.error("sizes must be at least 10")
    app = QApplication.instance() or QApplication([])
    result = {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
              "note": "Synthetic plan with production entry shapes; one validation pass per size in a Qt worker. Plan construction, hashing, source reads and project copies excluded. No profiler/tracemalloc.",
              "measurements": []}
    for size in args.sizes:
        result["measurements"].append(benchmark(size))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    app.processEvents()


if __name__ == "__main__":
    main()
