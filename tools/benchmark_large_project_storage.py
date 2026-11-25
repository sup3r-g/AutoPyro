"""Save/reopen a disposable large interpolation project in separate processes.

No tracemalloc or shown desktop. Preparation and verification are outside timing.
Use a fresh path for save; open requires its generated verification manifest.
"""
import argparse
from contextlib import closing
import cProfile
from dataclasses import asdict, fields, is_dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sqlite3
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from autopyro.audit import archive_protocol
from autopyro.core import Column, Project
from autopyro.database.storage import file_digest
from autopyro.database import storage
from autopyro.interpolation import prepare, replay
from autopyro.protocol_json import protocol_json_chunks
from autopyro.ui.table import AddColumnCommand, TableModel
from tools.benchmark_import_refresh import process_memory
from tools.benchmark_workspace_interpolation import dataset


def fingerprint(project):
    """Hash every stored field, batching large protocols rather than copying them."""
    digest = hashlib.sha256()

    def add(value):
        for chunk in protocol_json_chunks(value, sort_keys=True):
            digest.update(chunk)

    def metadata(value):
        add({k: v for k, v in value.items() if k != "interpolation_protocol"})
        protocol = value.get("interpolation_protocol")
        if isinstance(protocol, dict):
            add(protocol)
        else:
            add({"interpolation_protocol": protocol})

    add({"id": project.id, "view": project.view, "source": project.source})
    for column in project.columns:
        add({k: v for k, v in vars(column).items() if k != "metadata"})
        metadata(column.metadata)
    for row in project.rows:
        add({**{item.name: getattr(row, item.name) for item in fields(row)}, **vars(row)})
    for name in ("samples", "properties", "sources", "analyses", "measurements"):
        add({"collection": name})
        records = getattr(project.science, name)
        for key in sorted(records):
            add({"key": key, "record": vars(records[key])})
    for sample in sorted(project.science.working):
        add({"sample": sample, "working": project.science.working[sample]})
    for event in project.history:
        add({k: v for k, v in event.items() if k != "metadata"})
        metadata(event.get("metadata", {}))
    return digest.hexdigest()


def use_eager_control():
    """This process only: reproduce former record copies and eager bulk lists."""
    connect = storage.sqlite3.connect

    class EagerConnection:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def executemany(self, sql, records):
            return self.connection.executemany(sql, list(records))

    storage.sqlite3.connect = lambda *args, **kwargs: EagerConnection(connect(*args, **kwargs))
    class EagerRecordEncoder:
        def __init__(self, **options):
            pass

        def encode(self, record):
            return storage.encode(asdict(record) if is_dataclass(record) else record)

    storage._RecordEncoder = EagerRecordEncoder


def use_plain_records():
    """This process only: create a JSON encoder and inspect fields per record."""
    class PlainRecordEncoder(json.JSONEncoder):
        def default(self, value):
            if is_dataclass(value) and not isinstance(value, type):
                return {field.name: getattr(value, field.name) for field in fields(value)}
            return super().default(value)

    class PerRecordEncoder:
        def __init__(self, **options):
            self.options = options

        def encode(self, record):
            return json.dumps(record, cls=PlainRecordEncoder, **self.options)

    storage._RecordEncoder = PerRecordEncoder


def interpolation_column(project, *, inline=False):
    """Prepare the same selected-input PCHIP fixture for storage/history checks."""
    well, _, depth, _, value = project.columns[:5]
    scope = {row.id for row in project.rows if row.values[depth.id] < 900}
    parameters = dict(x_id=depth.id, y_id=value.id, group_ids=[well.id], method="pchip", max_gap=10,
                      constraints={"well_id": well.id, "analytical_context": True})
    proposals, protocol = prepare(project, row_ids=scope, **parameters)
    stored = protocol if inline else archive_protocol(protocol)
    column = Column("Interpolated S2", "number", unit=value.unit, metadata={
        **parameters, "stale": False, "scope_row_ids": sorted(scope),
        "proposals": [asdict(p) for p in proposals], "interpolation_protocol": stored})
    return column, {p.row_id: p.value for p in proposals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("save", "open"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--extra-columns", type=int, default=0, help="Numeric source columns in addition to the standard five")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", action="store_true", help="Verify replay after load timing/memory measurement")
    parser.add_argument("--table-only", action="store_true", help="Do not prepare an interpolation protocol")
    parser.add_argument("--eager-control", action="store_true", help="Table-only save using former eager lists/asdict")
    parser.add_argument("--plain-strings", action="store_true", help="Open control: omit sharing strings from nested protocols")
    parser.add_argument("--profile", action="store_true", help="Profile the storage operation; timings include profiling overhead")
    parser.add_argument("--inline-protocol", action="store_true", help="Save control: keep a plain interpolation protocol instead of archiving")
    parser.add_argument("--plain-records", action="store_true", help="Save control: use the previous per-record encoder without field caching")
    parser.add_argument("--write-cache-mib", type=int, help="Save experiment: SQLite page-cache budget in MiB; 0 uses SQLite's default")
    parser.add_argument("--plain-row-keys", action="store_true", help="Open control: keep separately decoded column IDs in row maps")
    args = parser.parse_args()
    manifest = Path(str(args.path) + ".verification.json")
    if args.phase == "save" and (args.path.exists() or manifest.exists()):
        parser.error("Use a fresh path; existing files are not overwritten")
    if args.rows < 100:
        parser.error("--rows must be at least 100")
    if args.extra_columns < 0:
        parser.error("--extra-columns must be non-negative")
    if args.eager_control and (args.phase != "save" or not args.table_only):
        parser.error("--eager-control requires table-only save")
    if args.replay and args.table_only:
        parser.error("--replay requires interpolation")
    if args.plain_strings and args.phase != "open":
        parser.error("--plain-strings requires open")
    if args.inline_protocol and (args.phase != "save" or args.table_only):
        parser.error("--inline-protocol requires full save")
    if args.plain_records and (args.phase != "save" or args.eager_control):
        parser.error("--plain-records requires save without --eager-control")
    if args.write_cache_mib is not None and (args.phase != "save" or args.write_cache_mib < 0):
        parser.error("--write-cache-mib requires save and a non-negative value")
    if args.plain_row_keys and args.phase != "open":
        parser.error("--plain-row-keys requires open")
    if args.eager_control:
        use_eager_control()
    if args.plain_records:
        use_plain_records()
    if args.write_cache_mib is not None:
        storage.WRITE_CACHE_KIB = args.write_cache_mib * 1024
    if args.plain_row_keys:
        storage._share_row_column_keys = lambda record, columns: record
    if args.plain_strings:
        class PlainStrings:
            loads = staticmethod(json.loads)
            existing = staticmethod(lambda value: value)
        storage.JsonStringPool = PlainStrings
    app = QApplication.instance() or QApplication([])
    if args.phase == "save":
        project = dataset(args.rows, args.extra_columns)
        model = TableModel(project)
        for analysis in project.science.analyses.values():
            analysis.method, analysis.laboratory = "Synthetic Pyrol6", "QA"
        if not args.table_only:
            column, values = interpolation_column(project, inline=args.inline_protocol)
            model.undo.push(AddColumnCommand(model, column, values))
            model.undo.undo()
            model.undo.redo()  # Two independent protocol events, as in the GUI acceptance.
        expected = fingerprint(project)
        args.path.parent.mkdir(parents=True, exist_ok=True)
        operation = lambda: project.save(args.path)
    else:
        expected = json.loads(manifest.read_text(encoding="utf-8"))["fingerprint"]
        before = file_digest(args.path)
        operation = lambda: Project.load(args.path)
    gc.collect()
    initial = process_memory()
    print(json.dumps({"phase": args.phase, "status": "starting_storage", **initial}), flush=True)
    started = perf_counter()
    profiler = cProfile.Profile() if args.profile else None
    loaded = profiler.runcall(operation) if profiler else operation()
    elapsed = perf_counter() - started
    memory = process_memory()  # Excludes the verification/replay below.
    if profiler:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(args.output.with_suffix(".prof")))
    if args.phase == "open":
        project = loaded
    assert fingerprint(project) == expected
    if args.phase == "open":
        assert file_digest(args.path) == before
    else:
        manifest.write_text(json.dumps({"fingerprint": expected}), encoding="utf-8")
    result = {"phase": args.phase, "rows": len(project.rows), "columns": len(project.columns),
              "source_columns": len(project.science.properties),
              "measurements": len(project.science.measurements), "history_events": len(project.history),
              "archived": isinstance(project.columns[-1].metadata.get("interpolation_protocol"), str),
              "eager_control": args.eager_control,
              "shared_json_strings": not args.plain_strings,
              "profiled": args.profile,
              "forced_inline_protocol": args.inline_protocol,
              "cached_record_encoder": not args.plain_records and not args.eager_control,
              "write_cache_mib": args.write_cache_mib,
              "configured_write_cache_kib": storage.WRITE_CACHE_KIB if args.phase == "save" else None,
              "shared_row_column_keys": not args.plain_row_keys,
              "seconds": elapsed, "initial_memory": initial, **memory, "file_bytes": args.path.stat().st_size,
              "python": sys.version, "platform": platform.platform(), "verified": True,
              "note": "Fresh process per phase; no tracemalloc/GUI latency measurement. Save/load includes validation. "
                      "Fingerprint covers all stored fields. Peak is cumulative, including fixture preparation on save. "
                      "Loading a just-written file does not establish cold filesystem-cache performance."}
    with closing(sqlite3.connect(args.path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        result["storage_version"] = connection.execute("PRAGMA user_version").fetchone()[0]
        if result["storage_version"] >= 4:
            result["stored_nested_protocols"], result["nested_protocol_bytes"] = connection.execute(
                "SELECT COUNT(*),COALESCE(SUM(byte_count),0) FROM stored_protocols").fetchone()
            result["nested_references"] = connection.execute(
                "SELECT (SELECT COUNT(*) FROM column_protocols)+(SELECT COUNT(*) FROM history_nested_protocols)").fetchone()[0]
    if args.replay:
        started = perf_counter()
        reproduced = replay(project.columns[-1].metadata["interpolation_protocol"])
        assert [asdict(p) for p in reproduced] == project.columns[-1].metadata["proposals"]
        result.update(replay_seconds=perf_counter() - started, replay_verified=True,
                      reproduced_values=len(reproduced))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
