"""Measure live interpolation undo/redo retention without saving a project."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.benchmark_large_project_storage import interpolation_column
from tools.benchmark_workspace_interpolation import dataset
from tools.benchmark_import_refresh import process_memory
from PySide6.QtWidgets import QApplication
from autopyro.audit import read_protocol
from autopyro.protocol_json import protocol_json_chunks
from autopyro.interpolation import replay
from autopyro import interpolation
from autopyro.ui.table import AddColumnCommand, TableModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--redos", type=int, default=3)
    parser.add_argument("--inline-protocol", action="store_true")
    parser.add_argument("--legacy-protocol", action="store_true", help="Control: retain the previous full measurement records")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rows < 100 or args.redos < 1:
        parser.error("rows must be at least 100; redos must be positive")
    if args.legacy_protocol:
        interpolation.COMPACT_MEASUREMENT_THRESHOLD = sys.maxsize
    app = QApplication.instance() or QApplication([])
    project = dataset(args.rows)
    model = TableModel(project)
    for analysis in project.science.analyses.values():
        analysis.method, analysis.laboratory = "Synthetic Pyrol6", "QA"
    original_ids = [row.id for row in project.rows]
    source_values = [dict(row.values) for row in project.rows]
    column, values = interpolation_column(project, inline=args.inline_protocol)
    result = {"rows": args.rows, "measurements": len(project.science.measurements),
              "redos": args.redos, "archived": isinstance(column.metadata["interpolation_protocol"], str),
              "legacy_protocol": args.legacy_protocol,
              "stages": [], "note": "No full desktop or project file; synthetic source; no tracemalloc. "
              "Command times include progress dialog workers, not GC/verification. Peak is cumulative."}

    def measure(name, function):
        gc.collect()
        started = perf_counter()
        function()
        elapsed = perf_counter() - started
        app.processEvents()
        gc.collect()
        item = {"stage": name, "seconds": elapsed, "events": len(project.history), **process_memory()}
        result["stages"].append(item)
        print(json.dumps(item), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def check(applied):
        assert [row.id for row in project.rows] == original_ids
        for row, original in zip(project.rows, source_values, strict=True):
            assert all(row.values[cid] == value for cid, value in original.items())
            if applied:
                assert row.values[column.id] == values.get(row.id)
            else:
                assert column.id not in row.values

    measure("prepared", lambda: None)
    measure("apply", lambda: model.undo.push(AddColumnCommand(model, column, values)))
    check(True)
    for i in range(args.redos):
        measure(f"undo_{i + 1}", model.undo.undo)
        check(False)
        measure(f"redo_{i + 1}", model.undo.redo)
        check(True)
    events = [e["metadata"]["interpolation_protocol"] for e in project.history if e.get("action") == "add_column"]
    assert len(events) == args.redos + 1
    protocol = column.metadata["interpolation_protocol"]
    if isinstance(protocol, str):
        assert all(event is protocol for event in events)
    else:
        assert len({id(event) for event in events}) == len(events)
        assert all(event is not protocol and event["checksum"] == protocol["checksum"] for event in events)
    # Replay and archive inflation happen after memory/time samples above.
    details = read_protocol(protocol) if isinstance(protocol, str) else protocol
    assert [asdict(p) for p in replay(details)] == details["proposals"]
    result.update(verified=True, replay_verified=True, proposals=len(values), history_events=len(project.history),
                  protocol_version=details["version"],
                  protocol_bytes=sum(map(len, protocol_json_chunks(details))),
                  stored_archive_bytes=len(protocol) if isinstance(protocol, str) else None)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
