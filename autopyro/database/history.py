"""SQLite journal: ordered events refer to shared immutable protocol payloads."""
import hashlib
import json

from autopyro.model.audit import MAGIC


SCHEMA = """
CREATE TABLE history_protocols (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE history_events (position INTEGER PRIMARY KEY, payload TEXT NOT NULL,
    protocol_id TEXT REFERENCES history_protocols(id));
"""
ERROR = "Нарушена целостность журнала проекта"


def write_history(connection, history, *, nested=None):
    identifiers = {}
    for position, event in enumerate(history):
        protocol_id = None
        payload = event
        value = event.get("protocol") if isinstance(event, dict) else None
        if type(value) is str and value.startswith(MAGIC):
            protocol_id = identifiers.get(value)
            if protocol_id is None:
                protocol_id = hashlib.sha256(value.encode("utf-8")).hexdigest()
                identifiers[value] = protocol_id
                connection.execute("INSERT INTO history_protocols VALUES (?,?)", (protocol_id, value))
            payload = {key: None if key == "protocol" else item for key, item in event.items()}
        nested_id = None
        if nested is not None and type(payload) is dict and "metadata" in payload:
            metadata, nested_id = nested.extract(payload["metadata"])
            if nested_id is not None:
                payload = {**payload, "metadata": metadata}
        # Inline protocols can be hundreds of MiB each. Insert this event before
        # encoding the next, inside the caller's atomic transaction.
        connection.execute("INSERT INTO history_events VALUES (?,?,?)",
                           (position, json.dumps(payload, ensure_ascii=False, allow_nan=False), protocol_id))
        if nested_id is not None:
            connection.execute("INSERT INTO history_nested_protocols VALUES (?,?)", (position, nested_id))


def read_history(connection, expected_count, *, loads=json.loads, nested=None):
    if type(expected_count) is not int or expected_count < 0:
        raise ValueError(ERROR)
    protocols = dict(connection.execute("SELECT id,payload FROM history_protocols"))
    for identifier, payload in protocols.items():
        if (type(payload) is not str or not payload.startswith(MAGIC)
                or hashlib.sha256(payload.encode("utf-8")).hexdigest() != identifier):
            raise ValueError(ERROR)
    result, used = [], set()
    for position, payload, identifier in connection.execute(
            "SELECT position,payload,protocol_id FROM history_events ORDER BY position"):
        if position != len(result):
            raise ValueError(ERROR)
        event = loads(payload)
        if identifier is not None:
            if (identifier not in protocols or type(event) is not dict
                    or "protocol" not in event or event["protocol"] is not None):
                raise ValueError(ERROR)
            event["protocol"] = protocols[identifier]
            used.add(identifier)
        if nested is not None:
            metadata = event.get("metadata") if type(event) is dict else None
            nested.restore(metadata, "history", position)
        result.append(event)
    if len(result) != expected_count or used != set(protocols):
        raise ValueError(ERROR)
    return result
