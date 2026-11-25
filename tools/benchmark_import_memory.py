"""Track reachable data and process memory across laboratory batches and undo."""
import argparse
from dataclasses import is_dataclass
import gc
import json
from pathlib import Path
import platform
import sys
import time

from tools.benchmark_import_refresh import QApplication, process_memory
from autopyro.importing import digest, preview_import, prepare_import, suggest_mapping
from autopyro.io import build_project
from autopyro.ui.table.commands import ImportCommand
from autopyro.ui.table import commands as table_commands
from autopyro.ui.table.model import TableModel


def graph_size(roots, seen):
    pending = [roots]
    total = 0
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        total += sys.getsizeof(value)
        if isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            pending.extend(value)
        elif is_dataclass(value):
            pending.append(vars(value))
    return total


def checkpoint(model, name):
    gc.collect()
    memory = process_memory()
    project = model.project
    commands = [model.undo.command(i) for i in range(model.undo.count())]
    roots = [
        ("active_data", (project.columns, project.rows, project.science, project.source)),
        ("history", project.history), ("view", project.view),
        ("undo_states", [(c.before, c.after) for c in commands]),
        ("undo_protocols", [(c.protocol, getattr(c, "history_protocol", None)) for c in commands]),
    ]
    seen = set()
    sizes = {label: graph_size(values, seen) for label, values in roots}
    result = {"stage": name, "measurements": len(project.science.measurements),
              "undo_commands": model.undo.count(), "history_events": len(project.history),
              "graph_bytes": sizes, "total_graph_bytes": sum(sizes.values()), **memory}
    print(json.dumps(result), flush=True)
    return result


def dataset(rows, columns, batch):
    headers = ["Sample", "Well", "TOC", "S2"] + [f"Bench{i}" for i in range(columns - 4)]
    units = ["", "", "%wt", "mg HC/g rock"] + [""] * (columns - 4)
    values = [[f"{i:07d}", f"W{i % 20}", "2", str(6 + batch)] + [str(j + batch) for j in range(columns - 4)]
              for i in range(rows)]
    return build_project(headers, values, list(range(2, rows + 2)),
        [(i, name, "text" if i < 2 else "number", units[i]) for i, name in enumerate(headers)],
        {"sha256": f"synthetic-batch-{batch}"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--columns", type=int, default=12)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=1, help="Undo/redo cycles after loading the batches")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plain-undo", action="store_true", help="Control run with uncompressed undo snapshots")
    parser.add_argument("--inline-history", action="store_true", help="Control run with full protocol copies in events")
    args = parser.parse_args()
    if args.rows < 1 or args.columns < 4 or args.batches < 1 or args.cycles < 1:
        parser.error("positive rows/batches and at least four columns are required")
    if args.plain_undo:
        table_commands.pack_table_state = lambda state: state
    if args.inline_history:
        table_commands.archive_protocol = lambda protocol: protocol
    app = QApplication.instance() or QApplication([])
    model = TableModel(dataset(args.rows, args.columns, 0))
    ids = [row.id for row in model.project.rows]
    result = {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
              "rows": args.rows, "columns": args.columns, "batches": args.batches, "cycles": args.cycles,
              "plain_undo": args.plain_undo, "inline_history": args.inline_history,
              "note": "Synthetic fixed samples, repeats in each batch. Graph sizes use getsizeof on known containers/dataclasses with shared objects counted once in listed root order; Qt/native/class memory excluded. Working set captured before graph traversal; cumulative process peaks can include prior observer allocations. No tracemalloc/profiler. GUI external views excluded.",
              "checkpoints": [checkpoint(model, "initial")], "batch_seconds": [], "cycle_seconds": []}
    for batch in range(1, args.batches + 1):
        started = time.perf_counter()
        incoming = dataset(args.rows, args.columns, batch)
        incoming.sync_measurements()
        plan = preview_import(model.project, incoming, suggest_mapping(model.project, incoming), [incoming.columns[0].id])
        prepared, protocol = prepare_import(model.project, incoming, plan)
        model.undo.push(ImportCommand(model, prepared, protocol, plan.project_digest))
        del incoming, plan, prepared, protocol
        result["batch_seconds"].append(time.perf_counter() - started)
        assert [row.id for row in model.project.rows] == ids
        assert len(model.project.science.measurements) == args.rows * args.columns * (batch + 1)
        result["checkpoints"].append(checkpoint(model, f"batch_{batch}"))
    for cycle in range(1, args.cycles + 1):
        started = time.perf_counter()
        model.undo.undo()
        elapsed = time.perf_counter() - started
        assert len(model.project.science.measurements) == args.rows * args.columns * args.batches
        if cycle == 1 or cycle == args.cycles:
            result["checkpoints"].append(checkpoint(model, f"undo_{cycle}"))
        started = time.perf_counter()
        model.undo.redo()
        elapsed += time.perf_counter() - started
        result["cycle_seconds"].append(elapsed)
        assert len(model.project.science.measurements) == args.rows * args.columns * (args.batches + 1)
        if cycle == 1 or cycle == args.cycles:
            result["checkpoints"].append(checkpoint(model, f"redo_{cycle}"))
    before = digest(model.project)
    model.undo.clear()
    app.processEvents()
    assert digest(model.project) == before
    result["checkpoints"].append(checkpoint(model, "undo_cleared"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
