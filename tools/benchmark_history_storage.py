"""Save/load a long journal in separate processes to measure transient memory."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time
import tracemalloc

from autopyro.audit import archive_protocol, MAGIC, read_protocol
from autopyro.core import Project
from autopyro.database.storage import file_digest
from autopyro.importing import preview_import, prepare_import, suggest_mapping
from tools.benchmark_import_memory import dataset
from tools.benchmark_import_refresh import process_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("save", "load"))
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=int, default=500)
    parser.add_argument("--memory", action="store_true")
    args = parser.parse_args()
    if args.mode == "save":
        project, incoming = dataset(200, 12, 0), dataset(200, 12, 1)
        project.sync_measurements()
        plan = preview_import(project, incoming, suggest_mapping(project, incoming), [incoming.columns[0].id])
        project, protocol = prepare_import(project, incoming, plan)
        archived = archive_protocol(protocol)
        assert isinstance(archived, str) and archived.startswith(MAGIC)
        project.history.extend({"at": str(i), "action": "import" if i % 2 else "undo_import", "protocol": archived}
                               for i in range(args.events))
        args.project.parent.mkdir(parents=True, exist_ok=True)
        operation = lambda: project.save(args.project)
    else:
        before = file_digest(args.project)
        operation = lambda: Project.load(args.project)
    initial = process_memory()
    if args.memory:
        tracemalloc.start()
    started = time.perf_counter()
    loaded = operation()
    elapsed = time.perf_counter() - started
    peak = tracemalloc.get_traced_memory()[1] if args.memory else None
    if args.memory:
        tracemalloc.stop()
    memory = process_memory()
    if args.mode == "load":
        assert file_digest(args.project) == before
        project = loaded
    assert len(project.history) == args.events + 1
    assert len(project.science.measurements) == 4800
    values = [event["protocol"] for event in project.history if "protocol" in event]
    assert len(values) == args.events and all(value is values[0] for value in values)
    assert len(read_protocol(values[0])["rows"]) == 200
    result = {"mode": args.mode, "events": args.events, "file_bytes": args.project.stat().st_size,
              "seconds": elapsed, "peak_python_bytes": peak, "memory_initial": initial, **memory,
              "python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
              "note": "200 samples, 12 columns, one real core append, 500 repeated actions by default. Each invocation is a fresh process. Load includes cold scientific imports. Tracemalloc changes timing. Native peak is cumulative for the whole process. Original file hash checked after loading."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
