"""Versioned local SQLite projects, atomic replacement and optimistic concurrency."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from concurrent.futures import CancelledError
from contextlib import contextmanager, closing
from dataclasses import fields, is_dataclass
from pathlib import Path

from autopyro.model.measurements import MeasurementStore
from autopyro.model.audit import share_history_protocols
from .history import SCHEMA as HISTORY_SCHEMA, read_history, write_history
from .json_strings import JsonStringPool
from .nested_protocols import SCHEMA as NESTED_SCHEMA, ProtocolReader, ProtocolWriter

FORMAT_VERSION = 4
APPLICATION_ID = 0x41505952
WRITE_CACHE_KIB = 256 * 1024
READ_CACHE_KIB = 32 * 1024
INTEGRITY_CACHE_KIB = 128 * 1024
LARGE_PROJECT_BYTES = 512 * 1024 * 1024
SCIENTIFIC_READ_BATCH_SIZE = 64
_decode_json_object = json.JSONDecoder().raw_decode


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class _RecordEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._field_names = {}

    def default(self, value):
        record_type = type(value)
        # Custom metaclasses can redefine class equality/hash. Inspect those
        # records normally rather than using the type as a cache key.
        cacheable = type(record_type) is type
        names = self._field_names.get(record_type) if cacheable else None
        if names is not None:
            # Keep the encoding dictionary local to this record. Calling vars()
            # here would materialize one retained dict per compact snapshot record.
            return {name: getattr(value, name) for name in names}
        if is_dataclass(value) and not isinstance(value, type):
            # Read declared fields directly; asdict would first deep-copy the
            # entire inline protocol inside a column's metadata.
            names = tuple(field.name for field in fields(value))
            if cacheable and len(self._field_names) < 32:
                self._field_names[record_type] = names
            return {name: getattr(value, name) for name in names}
        return super().default(value)


def encode_record(record):
    return json.dumps(record, cls=_RecordEncoder, ensure_ascii=False, allow_nan=False)


def _decode_sqlite_record(payload):
    """Decode one compact JSON record, preserving the general decoder fallback."""
    if type(payload) is str and payload.startswith("{"):
        record, end = _decode_json_object(payload)
        if end == len(payload):
            return record
    # Older or external files may contain whitespace or a UTF-encoded BLOB.
    # Also let json.loads reject trailing data as it did before.
    return json.loads(payload)


def _share_flat_text(values, share):
    if type(values) is dict:
        for key, value in values.items():
            if type(value) is str:
                values[key] = share(value)


def _share_row_cells(record, column_ids, share):
    """Reuse immutable IDs and text while keeping each row's maps independent."""
    if type(record) is dict:
        canonical_key = column_ids.get
        for field in ("values", "original"):
            values = record.get(field)
            if type(values) is dict:
                record[field] = {canonical_key(key, key): share(value) if type(value) is str else value
                                 for key, value in values.items()}
        _share_flat_text(record.get("source"), share)


def _share_source_cells(record, share):
    if type(record) is dict:
        _share_flat_text(record.get("original"), share)
        _share_flat_text(record.get("provenance"), share)


def _share_measurement_cells(record, share):
    if type(record) is not dict:
        return
    raw, value = record.get("raw_value"), record.get("value")
    if type(raw) is str:
        raw = record["raw_value"] = share(raw)
    if type(value) is str:
        record["value"] = raw if type(raw) is str and value == raw else share(value)


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def write_lock(path):
    """OS lock is released on crash. Keep the sidecar inode to avoid lock races."""
    lock = Path(str(path) + ".lock").open("a+b")
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt
            if lock.seek(0, os.SEEK_END) == 0:
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise ValueError("Проект сохраняется другим процессом") from exc
        else:
            import fcntl
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ValueError("Проект сохраняется другим процессом") from exc
        acquired = True
        yield
    finally:
        if acquired:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


SCHEMA = """
CREATE TABLE metadata (key TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE table_columns (id TEXT PRIMARY KEY, position INTEGER UNIQUE NOT NULL, payload TEXT NOT NULL);
CREATE TABLE table_rows (id TEXT PRIMARY KEY, position INTEGER UNIQUE NOT NULL, payload TEXT NOT NULL);
CREATE TABLE samples (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE properties (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE source_records (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE analyses (id TEXT PRIMARY KEY, sample_id TEXT NOT NULL REFERENCES samples(id),
    source_id TEXT REFERENCES source_records(id), payload TEXT NOT NULL);
CREATE TABLE measurements (id TEXT PRIMARY KEY, sample_id TEXT NOT NULL REFERENCES samples(id),
    property_id TEXT NOT NULL REFERENCES properties(id), analysis_id TEXT NOT NULL REFERENCES analyses(id),
    source_id TEXT REFERENCES source_records(id), payload TEXT NOT NULL,
    UNIQUE(id, sample_id, property_id));
CREATE TABLE working_measurements (sample_id TEXT NOT NULL REFERENCES samples(id),
    property_id TEXT NOT NULL REFERENCES properties(id), measurement_id TEXT NOT NULL,
    PRIMARY KEY(sample_id, property_id),
    FOREIGN KEY(measurement_id, sample_id, property_id) REFERENCES measurements(id, sample_id, property_id));
"""

SECONDARY_INDEXES = """
CREATE INDEX measurement_sample_property ON measurements(sample_id, property_id);
CREATE INDEX analysis_sample ON analyses(sample_id);
"""


def _write_database(path, project, *, cancelled=None):
    # Encoder options/field descriptions are reusable; JSON's recursion markers
    # are local to each encode() call. Keep this cache scoped to one save.
    record_json = _RecordEncoder(ensure_ascii=False, allow_nan=False).encode
    with closing(sqlite3.connect(path)) as connection:
        if cancelled is not None:
            connection.set_progress_handler(cancelled.is_set, 1000)
        connection.execute("PRAGMA foreign_keys=ON")
        if WRITE_CACHE_KIB:
            # Private temporary database only. Keep hot foreign-key/index pages
            # in memory; durability and the atomic replacement stay unchanged.
            connection.execute(f"PRAGMA cache_size={-WRITE_CACHE_KIB}")
        connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
        connection.execute(f"PRAGMA user_version={FORMAT_VERSION}")
        connection.executescript(SCHEMA + HISTORY_SCHEMA + NESTED_SCHEMA)
        nested = ProtocolWriter(connection)
        for key in ("id", "view", "source"):
            connection.execute("INSERT INTO metadata VALUES (?,?)", (key, encode(getattr(project, key))))
        connection.execute("INSERT INTO metadata VALUES (?,?)", ("history_event_count", encode(len(project.history))))
        write_history(connection, project.history, nested=nested)
        for i, column in enumerate(project.columns):
            metadata, protocol_id = nested.extract(column.metadata)
            payload = {field.name: getattr(column, field.name) for field in fields(column)}
            payload["metadata"] = metadata
            connection.execute("INSERT INTO table_columns VALUES (?,?,?)", (column.id, i, record_json(payload)))
            if protocol_id is not None:
                connection.execute("INSERT INTO column_protocols VALUES (?,?)", (column.id, protocol_id))
        connection.executemany("INSERT INTO table_rows VALUES (?,?,?)",
                               ((row.id, i, record_json(row)) for i, row in enumerate(project.rows)))
        store = project.science
        for table, collection in (("samples", store.samples), ("properties", store.properties), ("source_records", store.sources)):
            connection.executemany(f"INSERT INTO {table} VALUES (?,?)",
                                   ((record.id, record_json(record)) for record in collection.values()))
        # sqlite consumes iterators one record at a time. Do not retain another
        # project-sized list of JSON strings alongside the scientific records.
        connection.executemany("INSERT INTO analyses VALUES (?,?,?,?)", (
            (a.id, a.sample_id, a.source_id, record_json(a)) for a in store.analyses.values()))
        connection.executemany("INSERT INTO measurements VALUES (?,?,?,?,?,?)", (
            (m.id, m.sample_id, m.property_id, m.analysis_id, m.source_id, record_json(m)) for m in store.measurements.values()))
        connection.executemany("INSERT INTO working_measurements VALUES (?,?,?)", (
            (sid, pid, mid) for sid, selections in store.working.items() for pid, mid in selections.items()))
        # Build search indexes in bulk. Unique indexes and foreign keys have
        # remained active throughout insertion; the final schema is unchanged.
        for statement in SECONDARY_INDEXES.split(";"):
            if statement.strip():
                connection.execute(statement)
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("Нарушена целостность проекта")
        connection.commit()


def _backup_previous(path, version):
    # Content-addressed name permits repeated migrations without replacing an earlier backup.
    backup = path.with_name(path.name + f".v{version}-{file_digest(path)[:12]}.bak")
    if backup.exists():
        if file_digest(backup) != file_digest(path):
            raise ValueError("Резервная копия старого проекта повреждена")
        return
    try:
        with backup.open("xb") as target, path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                target.write(block)
            target.flush()
            os.fsync(target.fileno())
    except BaseException:
        backup.unlink(missing_ok=True)
        raise


def save_project(project, path, *, recovery=False, cancelled=None):
    if cancelled is not None and cancelled.is_set():
        raise CancelledError()
    path = Path(path).resolve()
    if project.source.get("type") == "sqlite":
        source = Path(project.source["path"]).resolve()
        if path == source or path.exists() and source.exists() and path.samefile(source):
            raise ValueError("Источник SQLite доступен только для чтения. Сохраните локальный проект в другой файл")
    # JSON validation also catches non-finite values hidden in settings/source metadata.
    project.sync_measurements()
    project.validate()
    share_history_protocols(project.history)
    if cancelled is not None and cancelled.is_set():
        raise CancelledError()
    with write_lock(path):
        if not recovery and getattr(project, "_storage_path", None) == path:
            if not path.exists() or file_digest(path) != project._storage_digest:
                raise ValueError("Проект изменён другим процессом. Откройте актуальную версию или сохраните под другим именем.")
        fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        os.close(fd)
        try:
            try:
                if cancelled is None:
                    _write_database(temporary, project)
                else:
                    _write_database(temporary, project, cancelled=cancelled)
            except sqlite3.OperationalError as exc:
                if (cancelled is not None and cancelled.is_set()
                        and getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_INTERRUPT):
                    raise CancelledError() from exc
                raise
            if cancelled is not None and cancelled.is_set():
                raise CancelledError()
            with open(temporary, "r+b") as stream:
                os.fsync(stream.fileno())
            if path.exists():
                with path.open("rb") as stream:
                    legacy = stream.read(16) != b"SQLite format 3\x00"
                previous = 1 if legacy else None
                if not legacy:
                    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as old:
                        if old.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID:
                            previous = old.execute("PRAGMA user_version").fetchone()[0]
                            if previous not in (2, 3, FORMAT_VERSION):
                                raise ValueError("Неизвестный формат или версия проекта")
                if previous in (1, 2, 3) and not recovery:
                    _backup_previous(path, previous)
            if cancelled is not None and cancelled.is_set():
                raise CancelledError()
            os.replace(temporary, path)
            if not recovery:
                project._storage_path = path
                project._storage_digest = file_digest(path)
        finally:
            Path(temporary).unlink(missing_ok=True)


def load_payload(path):
    try:
        return _load_payload(path)
    except sqlite3.Error as exc:
        raise ValueError(str(exc)) from exc


def read_project_id(path):
    """Read the project identity when a full load failed."""
    try:
        path = Path(path).resolve()
        with path.open("rb") as stream:
            header = stream.read(16)
        if header == b"SQLite format 3\x00":
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
                if connection.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID:
                    return None
                row = connection.execute("SELECT payload FROM metadata WHERE key='id'").fetchone()
                identifier = json.loads(row[0]) if row is not None else None
        else:
            if header.strip() and not header.lstrip().startswith(b"{"):
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("format") != "autopyro.table" or data.get("version") != 1:
                return None
            identifier = data.get("id")
        return identifier if isinstance(identifier, str) and identifier else None
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return None


def _load_payload(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        sqlite = stream.read(16) == b"SQLite format 3\x00"
        large_sqlite = sqlite and os.fstat(stream.fileno()).st_size >= LARGE_PROJECT_BYTES
    if not sqlite:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.pop("format", None) != "autopyro.table" or data.pop("version", None) != 1:
            raise ValueError("Неизвестный формат или версия проекта")
        return data, True
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        # Large foreign-key checks revisit index pages before any records are restored.
        cache_kib = INTEGRITY_CACHE_KIB if large_sqlite else READ_CACHE_KIB
        connection.execute(f"PRAGMA cache_size={-cache_kib}")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if connection.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID or version not in (2, 3, FORMAT_VERSION):
            raise ValueError("Неизвестный формат или версия проекта")
        if (connection.execute("PRAGMA quick_check").fetchone()[0] != "ok"
                or connection.execute("PRAGMA foreign_key_check").fetchone() is not None):
            raise ValueError("Нарушена целостность проекта")
        if large_sqlite:
            connection.execute(f"PRAGMA cache_size={-READ_CACHE_KIB}")
            connection.execute("PRAGMA shrink_memory")
        strings = JsonStringPool()
        share_cell = strings.cell
        nested = ProtocolReader(connection, strings.loads) if version == FORMAT_VERSION else None
        data = {key: strings.loads(value) if key == "history" else json.loads(value)
                for key, value in connection.execute("SELECT key,payload FROM metadata")}
        if version >= 3:
            if "history" in data:
                raise ValueError("Нарушена целостность журнала проекта")
            data["history"] = read_history(connection, data.pop("history_event_count", None),
                                           loads=strings.text_loader(), nested=nested)
        for name, table in (("columns", "table_columns"), ("rows", "table_rows")):
            loads = strings.loads if name == "columns" else _decode_sqlite_record
            data[name] = []
            column_ids = {column["id"]: column["id"] for column in data["columns"]} if name == "rows" else {}
            for identifier, value in connection.execute(f"SELECT id,payload FROM {table} ORDER BY position"):
                record = loads(value)
                if name == "columns" and nested is not None:
                    if record.get("id") != identifier:
                        raise ValueError("Нарушена целостность проекта")
                    nested.restore(record.get("metadata"), "column", identifier)
                elif name == "rows":
                    _share_row_cells(record, column_ids, share_cell)
                data[name].append(record)
        if nested is not None:
            nested.validate_used()
        existing_cell = strings.existing_cell

        def records(table, share=None):
            cursor = connection.execute(f"SELECT id,payload FROM {table}")
            while batch := cursor.fetchmany(SCIENTIFIC_READ_BATCH_SIZE):
                for key, value in batch:
                    record = _decode_sqlite_record(value)
                    if share is not None:
                        share(record, existing_cell)
                    yield strings.existing(key), record

        store = {}
        share_by_table = {"source_records": _share_source_cells, "measurements": _share_measurement_cells}
        for name, table in (("samples", "samples"), ("properties", "properties"), ("sources", "source_records"), ("analyses", "analyses"), ("measurements", "measurements")):
            store[name] = records(table, share_by_table.get(table))
        working_rows = connection.execute("SELECT sample_id,property_id,measurement_id FROM working_measurements")
        data["science"] = MeasurementStore.from_sqlite_payloads(store, working_rows)
    return data, False
