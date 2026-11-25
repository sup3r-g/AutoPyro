"""Separate measurement memo construction from the remaining project copy.

python -m tools.benchmark_snapshot_parts --rows 3000 --output tmp/snapshot-parts.json
Only synthetic inputs; no project file is read or written.
"""
import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import platform
import sys
import tracemalloc
from time import perf_counter

from autopyro import snapshots
from autopyro.measurements import Measurement
from tools.benchmark_import_refresh import QApplication, measure, process_memory
from tools.benchmark_workspace_interpolation import dataset


class GCObservations:
    """Observe automatic GC without changing thresholds or disabling collection."""
    def __init__(self):
        self.started = {}
        self.generations = [{"count": 0, "seconds": 0., "max_seconds": 0., "collected": 0}
                            for _ in range(3)]

    def callback(self, phase, info):
        generation = info["generation"]
        if phase == "start":
            self.started[generation] = perf_counter()
        elif generation in self.started:
            elapsed = perf_counter() - self.started.pop(generation)
            entry = self.generations[generation]
            entry["count"] += 1
            entry["seconds"] += elapsed
            entry["max_seconds"] = max(entry["max_seconds"], elapsed)
            entry["collected"] += info["collected"]


def memo_sizes(memo):
    """Shallow accounting only; referenced graphs and allocator reserve are excluded."""
    values = {}
    attribute_bytes = 0
    for value in memo.values():
        group = values.setdefault(type(value).__name__, {"entries": 0, "shallow_bytes": 0})
        group["entries"] += 1
        group["shallow_bytes"] += sys.getsizeof(value)
        if type(value) is Measurement:
            attribute_bytes += sys.getsizeof(vars(value))
    return {"entries": len(memo), "dict_bytes": sys.getsizeof(memo),
            "key_bytes": sum(sys.getsizeof(key) for key in memo), "values": values,
            "measurement_attribute_dict_bytes": attribute_bytes,
            "note": "Sizes per entry may count shared objects repeatedly; not a deep graph or resident-memory size."}


def experimental_measurement_memo(store):
    """Benchmark-only candidate; retain ordinary copying for all nonempty values."""
    records = [r for r in store.measurements.values()
               if type(r) is Measurement and not hasattr(r, "__deepcopy__")]
    memo = {}
    for record in records:
        memo[id(record)] = object.__new__(Measurement)
    for record in records:
        values = vars(memo[id(record)])
        for name, value in vars(record).items():
            if type(value) is list and not value:
                key = id(value)
                if key not in memo:
                    memo[key] = []
                    memo.setdefault(id(memo), []).append(value)
                value = memo[key]
            elif type(value) not in snapshots._SCALAR_TYPES:
                value = deepcopy(value, memo)
            values[name] = value
    return memo


def experimental_scalar_memo(store):
    """Bulk-copy scalar fields for the normal layout, retaining the generic fallback."""
    records = [r for r in store.measurements.values()
               if type(r) is Measurement and not hasattr(r, "__deepcopy__")]
    memo = {}
    for record in records:
        memo[id(record)] = object.__new__(Measurement)
    for record in records:
        source = vars(record)
        values = vars(memo[id(record)])
        if source and next(reversed(source)) == "history":
            scalars = source.copy()
            history = scalars.pop("history")
            if set(map(type, scalars.values())).issubset(snapshots._SCALAR_TYPE_SET):
                values.update(scalars)
                values["history"] = deepcopy(history, memo)
                continue
        for name, value in source.items():
            values[name] = value if type(value) in snapshots._SCALAR_TYPES else deepcopy(value, memo)
    return memo


def experimental_single_pass_memo(store):
    """Copy each plain record once; recursive deepcopy may already own later ones."""
    records = [r for r in store.measurements.values()
               if type(r) is Measurement and not hasattr(r, "__deepcopy__")]
    memo = {}
    for record in records:
        key = id(record)
        if key in memo:
            continue
        copied = object.__new__(Measurement)
        memo[key] = copied
        values = vars(copied)
        for name, value in vars(record).items():
            values[name] = value if type(value) in snapshots._SCALAR_TYPES else deepcopy(value, memo)
    return memo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--extra-columns", type=int, default=75)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--diagnostics", action="store_true", help="Observe GC and shallow memo sizes; does not change GC policy")
    parser.add_argument("--allocations", action="store_true", help="Trace net retained Python allocations and peak; adds timing overhead")
    parser.add_argument("--plain-mappings", action="store_true", help="Control: do not pre-copy flat scalar mappings")
    parser.add_argument("--whole", action="store_true", help="Measure copy_project as one operation without GC between parts")
    parser.add_argument("--experimental-empty-lists", action="store_true", help="Benchmark-only candidate: copy empty measurement lists directly")
    parser.add_argument("--experimental-scalar-fields", action="store_true", help="Benchmark-only candidate: bulk-copy normal scalar measurement fields")
    parser.add_argument("--experimental-single-pass", action="store_true", help="Benchmark-only candidate: register and fill each record in one pass")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rows < 1 or args.extra_columns < 0:
        parser.error("rows must be positive and extra-columns non-negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if sum((args.experimental_empty_lists, args.experimental_scalar_fields, args.experimental_single_pass)) > 1:
        parser.error("Choose only one measurement-copy experiment")
    if args.experimental_empty_lists:
        snapshots._measurement_memo = experimental_measurement_memo
    if args.experimental_scalar_fields:
        snapshots._measurement_memo = experimental_scalar_memo
    if args.experimental_single_pass:
        snapshots._measurement_memo = experimental_single_pass_memo
    app = QApplication.instance() or QApplication([])
    project = dataset(args.rows, args.extra_columns)
    project.sync_measurements()
    result = {"rows": args.rows, "columns": len(project.columns),
              "measurements": len(project.science.measurements), "profiled": args.profile,
              "shared_scalar_mappings": not args.plain_mappings,
              "whole_project": args.whole,
              "fast_empty_lists": args.experimental_empty_lists,
              "bulk_scalar_fields": args.experimental_scalar_fields,
              "single_pass_records": args.experimental_single_pass,
              "gc_diagnostics": args.diagnostics, "allocation_tracing": args.allocations,
              "python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
              "note": "Fresh process, one pass. Input construction, checks and explicit pre-stage GC excluded. "
                      "Automatic GC and Qt event gaps included. Peak is cumulative, including input construction. "
                      "No disk or shown desktop; profiled timings include profiler overhead.", "stages": []}
    def stage(name, function):
        gc.collect()
        profile = args.output.with_suffix(f".{name}.prof") if args.profile else None
        observations = GCObservations()
        def observed_function():
            if args.diagnostics:
                gc.callbacks.append(observations.callback)
            try:
                return function()
            finally:
                if args.diagnostics:
                    gc.callbacks.remove(observations.callback)
        if args.allocations:
            tracemalloc.start()
        try:
            value, metrics = measure(name, observed_function, profile_path=profile)
            metrics.update(process_memory())
            if args.allocations:
                current, peak = tracemalloc.get_traced_memory()
                sites = tracemalloc.take_snapshot().statistics("lineno")[:8]
                metrics["allocations"] = {"retained_bytes": current, "peak_bytes": peak,
                    "top_retained_sites": [{"location": str(site.traceback), "bytes": site.size,
                                             "blocks": site.count} for site in sites],
                    "note": "Net retained Python allocations, not total allocation traffic; tracing adds overhead."}
        finally:
            if args.allocations:
                tracemalloc.stop()
        if args.diagnostics:
            metrics["gc_by_generation"] = observations.generations
        result["stages"].append(metrics)
        print(json.dumps(metrics), flush=True)
        return value
    if args.whole:
        if args.plain_mappings:
            # This benchmark process only: retain the former copy algorithm.
            snapshots._table_memo = lambda rows, store, memo: None
        copied = stage("whole_project", lambda: snapshots.copy_project(project))
    else:
        memo = stage("measurement_records", lambda: snapshots._measurement_memo(project.science))
        if args.diagnostics:
            result["measurement_memo"] = memo_sizes(memo)
        def remainder():
            if not args.plain_mappings:
                snapshots._table_memo(project.rows, project.science, memo)
            return deepcopy(project, memo)
        copied = stage("remaining_project", remainder)
        if args.diagnostics:
            result["complete_memo"] = memo_sizes(memo)
    # Compare records in turn rather than materializing a second full asdict tree.
    assert copied.columns == project.columns and copied.rows == project.rows
    assert copied.science == project.science and copied.history == project.history
    assert copied.view == project.view and copied.source == project.source and copied.id == project.id
    row, column = copied.rows[0], copied.columns[3]
    copied.science.current(row.id, column.id).history.append({"qa": ["changed"]})
    assert not project.science.current(row.id, column.id).history
    row.original[column.id] = "changed"
    assert project.rows[0].original[column.id] == "2"
    result["verified"] = True
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
