"""SQLite v4: content-addressed nested protocols, with independent JSON copies."""
import hashlib
from io import BytesIO
import json

from autopyro.model.audit import MAGIC, MIN_PROTOCOL_BYTES
from autopyro.model.protocol_json import protocol_json_chunks

PART_BYTES = 1024 * 1024  # Fixed by SQLite v4; changing this requires a migration.
# This bounds plain storage, not archive decompression. The archive limit in
# audit.py stays unchanged. Larger or extended values retain inline storage.
MAX_STORED_BYTES = 1024 * 1024 * 1024
ERROR = "Нарушена целостность проекта"
SCHEMA = """
CREATE TABLE stored_protocols (id TEXT PRIMARY KEY, kind TEXT NOT NULL,
    byte_count INTEGER NOT NULL, part_count INTEGER NOT NULL);
CREATE TABLE stored_protocol_parts (protocol_id TEXT NOT NULL REFERENCES stored_protocols(id),
    position INTEGER NOT NULL, payload BLOB NOT NULL, PRIMARY KEY(protocol_id, position));
CREATE TABLE column_protocols (column_id TEXT PRIMARY KEY REFERENCES table_columns(id),
    protocol_id TEXT NOT NULL REFERENCES stored_protocols(id));
CREATE TABLE history_nested_protocols (position INTEGER PRIMARY KEY REFERENCES history_events(position),
    protocol_id TEXT NOT NULL REFERENCES stored_protocols(id));
"""


def _parts(value):
    if type(value) is str:
        chunks = (value[i:i + PART_BYTES].encode("utf-8") for i in range(0, len(value), PART_BYTES))
    else:
        chunks = protocol_json_chunks(value)
    pending = bytearray()
    for chunk in chunks:
        offset = 0
        while offset < len(chunk):
            length = min(PART_BYTES - len(pending), len(chunk) - offset)
            pending.extend(chunk[offset:offset + length])
            offset += length
            if len(pending) == PART_BYTES:
                yield bytes(pending)
                pending.clear()
    if pending:
        yield bytes(pending)


def _digest(kind):
    return hashlib.sha256(kind.encode("ascii") + b"\0")


def _copy_json(value):
    # Only freshly decoded JSON enters this function: scalars are immutable,
    # containers have no custom copy behavior or shared/cyclic references.
    if type(value) is dict:
        return {key: _copy_json(item) for key, item in value.items()}
    if type(value) is list:
        return [_copy_json(item) for item in value]
    return value


class ProtocolWriter:
    def __init__(self, connection):
        self.connection = connection
        self._written = set()

    def store(self, value):
        if type(value) is str and value.startswith(MAGIC):
            kind = "archive"
        elif type(value) is dict:
            kind = "json"
        else:
            return None
        digest, size = _digest(kind), 0
        try:
            for part in _parts(value):
                size += len(part)
                if size > MAX_STORED_BYTES:
                    return None
                digest.update(part)
        except (TypeError, ValueError, RecursionError):
            # Let the ordinary record encoder handle supported extensions or
            # report invalid data. No registry rows have been written yet.
            return None
        if kind == "json" and size < MIN_PROTOCOL_BYTES:
            return None
        identifier = digest.hexdigest()
        if identifier in self._written:
            return identifier
        count = (size + PART_BYTES - 1) // PART_BYTES
        self.connection.execute("INSERT INTO stored_protocols VALUES (?,?,?,?)", (identifier, kind, size, count))
        verified, actual_size, actual_count = _digest(kind), 0, 0
        for position, part in enumerate(_parts(value)):
            verified.update(part)
            actual_size += len(part)
            actual_count += 1
            if actual_size > size or actual_count > count:
                raise ValueError(ERROR)
            self.connection.execute("INSERT INTO stored_protocol_parts VALUES (?,?,?)", (identifier, position, part))
        if actual_size != size or actual_count != count or verified.hexdigest() != identifier:
            raise ValueError(ERROR)  # Inputs changed between hashing and writing.
        self._written.add(identifier)
        return identifier

    def extract(self, metadata):
        if type(metadata) is not dict or "interpolation_protocol" not in metadata:
            return metadata, None
        identifier = self.store(metadata["interpolation_protocol"])
        if identifier is None:
            return metadata, None
        return {key: None if key == "interpolation_protocol" else value
                for key, value in metadata.items()}, identifier


class ProtocolReader:
    def __init__(self, connection, loads=json.loads):
        self.connection, self.loads = connection, loads
        self._references = {
            "column": dict(connection.execute("SELECT column_id,protocol_id FROM column_protocols")),
            "history": dict(connection.execute("SELECT position,protocol_id FROM history_nested_protocols")),
        }
        self._cache, self._used = {}, set()

    def _read(self, identifier):
        row = self.connection.execute(
            "SELECT kind,byte_count,part_count FROM stored_protocols WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise ValueError(ERROR)
        kind, size, count = row
        if (kind not in ("json", "archive") or type(size) is not int or not 0 < size <= MAX_STORED_BYTES
                or type(count) is not int or count != (size + PART_BYTES - 1) // PART_BYTES):
            raise ValueError(ERROR)
        digest, actual_size, actual_count = _digest(kind), 0, 0
        with BytesIO() as stream:
            for position, part in self.connection.execute(
                    "SELECT position,payload FROM stored_protocol_parts WHERE protocol_id=? ORDER BY position", (identifier,)):
                expected_size = min(PART_BYTES, size - actual_size)
                if (position != actual_count or type(part) is not bytes or len(part) != expected_size
                        or expected_size <= 0):
                    raise ValueError(ERROR)
                actual_count += 1
                actual_size += len(part)
                digest.update(part)
                stream.write(part)
            if actual_count != count or actual_size != size or digest.hexdigest() != identifier:
                raise ValueError(ERROR)
            raw = stream.getvalue()
        try:
            value = self.loads(raw) if kind == "json" else raw.decode("utf-8")
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise ValueError(ERROR) from exc
        if (kind == "json" and type(value) is not dict
                or kind == "archive" and not value.startswith(MAGIC)):
            raise ValueError(ERROR)
        return value

    def restore(self, metadata, owner, key):
        identifier = self._references[owner].get(key)
        if identifier is None:
            return metadata
        if type(metadata) is not dict or "interpolation_protocol" not in metadata or metadata["interpolation_protocol"] is not None:
            raise ValueError(ERROR)
        self._used.add((owner, key))
        if identifier in self._cache:
            value = _copy_json(self._cache[identifier])
        else:
            value = self._cache[identifier] = self._read(identifier)
        metadata["interpolation_protocol"] = value
        return metadata

    def validate_used(self):
        references = {(owner, key) for owner, records in self._references.items() for key in records}
        identifiers = {row[0] for row in self.connection.execute("SELECT id FROM stored_protocols")}
        if self._used != references or set(self._cache) != identifiers:
            raise ValueError(ERROR)
