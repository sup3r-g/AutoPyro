"""Nested protocol deduplication, independent JSON objects and v3 migration."""
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import sqlite3

import pytest

from autopyro import audit
from autopyro.core import Column, Project
from autopyro.database import nested_protocols as nested, storage
from autopyro.database.json_strings import JsonStringPool
from autopyro.interpolation import prepare, replay
from autopyro.io import build_project


@pytest.fixture
def project(monkeypatch):
    monkeypatch.setattr(nested, "MIN_PROTOCOL_BYTES", 0)
    result = build_project(["Well", "X", "Y"], [["A", "0", "2"], ["A", "1", ""], ["A", "2", "6"]],
                           [2, 3, 4], [(0, "Well", "text", ""), (1, "X", "number", "m"),
                                       (2, "Y", "number", "mg/g")], {})
    result.sync_measurements()
    well, x, y = result.columns
    proposals, protocol = prepare(result, x.id, y.id, [well.id], set(result.row_map()), method="pchip",
                                   constraints={"well_id": well.id, "analytical_context": True})
    column = Column("Interpolated Y", "number", unit=y.unit,
                    metadata={"method": "pchip", "interpolation_protocol": protocol})
    result.columns.append(column)
    values = {p.row_id: p.value for p in proposals}
    for row in result.rows:
        row.values[column.id] = values.get(row.id)
    result.history.extend({"action": "add_column", "metadata": {
        "interpolation_protocol": deepcopy(protocol)}} for _ in range(2))
    return result


def test_same_content_is_stored_and_decoded_once_but_returns_independent_protocols(project, tmp_path, monkeypatch):
    path = tmp_path / "shared.apyro"
    before = deepcopy(project)
    project.save(path)
    assert project == before
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
        assert connection.execute("SELECT COUNT(*) FROM stored_protocols").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM column_protocols").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM history_nested_protocols").fetchone()[0] == 2
    decodes = []
    loads = JsonStringPool.loads

    def tracked(self, payload):
        value = loads(self, payload)
        if type(value) is dict and "algorithm" in value:
            decodes.append(value["algorithm"])
        return value

    monkeypatch.setattr(JsonStringPool, "loads", tracked)
    original = path.read_bytes()
    restored = Project.load(path)
    assert restored == project and path.read_bytes() == original
    assert len(decodes) == 1
    column = restored.columns[-1].metadata["interpolation_protocol"]
    first, second = [event["metadata"]["interpolation_protocol"] for event in restored.history[-2:]]
    assert first is not second and first is not column and second is not column
    mid = next(iter(column["measurements"]))
    assert first["measurements"][mid] is not second["measurements"][mid]
    assert first["measurements"][mid]["sample_id"] is column["measurements"][mid]["sample_id"]
    expected = replay(column)
    first["rows"][0]["values"][restored.columns[2].id] = "changed history"
    assert replay(second) == replay(column) == expected


def test_large_json_parts_preserve_order_numbers_and_mutability(project, tmp_path):
    protocol = {"last": None, "rows": [{"text": "Лаборатория 🌍" * 120000,
                                            "values": [2**100, -0.0, True, "ND", "<0.1"]}], "first": 1}
    project.columns[-1].metadata["interpolation_protocol"] = protocol
    for event in project.history[-2:]:
        event["metadata"]["interpolation_protocol"] = deepcopy(protocol)
    path = tmp_path / "parts.apyro"
    project.save(path)
    with closing(sqlite3.connect(path)) as connection:
        size, count = connection.execute("SELECT byte_count,part_count FROM stored_protocols").fetchone()
        assert count > 1 and count == (size + nested.PART_BYTES - 1) // nested.PART_BYTES
        assert connection.execute("SELECT MAX(length(payload)) FROM stored_protocol_parts").fetchone()[0] <= nested.PART_BYTES
    restored = Project.load(path)
    assert asdict(restored) == asdict(project)
    assert list(restored.columns[-1].metadata["interpolation_protocol"]) == list(protocol)


def test_same_embedded_checksum_does_not_merge_different_content(project, tmp_path):
    other = project.history[-1]["metadata"]["interpolation_protocol"]
    other["rows"][0]["values"][project.columns[2].id] = 99
    path = tmp_path / "distinct.apyro"
    project.save(path)
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM stored_protocols").fetchone()[0] == 2
    assert Project.load(path) == project


@pytest.mark.parametrize("left,right", [({"v": 1}, {"v": True}), ({"v": 0.0}, {"v": -0.0}),
                                        ({"a": 1, "b": 2}, {"b": 2, "a": 1})])
def test_json_types_signed_zero_and_key_order_are_not_collapsed(project, tmp_path, left, right):
    project.columns[-1].metadata["interpolation_protocol"] = left
    project.history[-2]["metadata"]["interpolation_protocol"] = deepcopy(left)
    project.history[-1]["metadata"]["interpolation_protocol"] = right
    path = tmp_path / "exact-json.apyro"
    project.save(path)
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM stored_protocols").fetchone()[0] == 2
    restored = Project.load(path)
    assert json.dumps(restored.columns[-1].metadata["interpolation_protocol"]) == json.dumps(left)
    assert json.dumps(restored.history[-1]["metadata"]["interpolation_protocol"]) == json.dumps(right)


@pytest.mark.parametrize("custom_record", [False, True])
def test_json_extensions_keep_previous_storage_semantics(project, tmp_path, custom_record):
    @dataclass
    class Context:
        label: str
        values: tuple

    extension = Context("Лаборатория", (1, "ND")) if custom_record else (1, "ND")
    project.columns[-1].metadata["interpolation_protocol"] = {"extension": extension}
    expected = json.loads(json.dumps(asdict(project), ensure_ascii=False))
    path = tmp_path / "extensions.apyro"
    project.save(path)
    restored = Project.load(path)
    assert json.loads(json.dumps(asdict(restored), ensure_ascii=False)) == expected
    with closing(sqlite3.connect(path)) as connection:
        references = connection.execute("SELECT COUNT(*) FROM column_protocols").fetchone()[0]
        assert references == (0 if custom_record else 1)


@pytest.mark.parametrize("damage", ["payload", "gap", "missing_part", "extra_part", "part_count", "byte_count",
                                    "oversized", "kind", "part_type", "missing_ref", "ambiguous", "unused", "owner"])
def test_corrupted_nested_storage_is_rejected_without_changing_file(project, tmp_path, damage):
    path = tmp_path / "damaged.apyro"
    project.save(path)
    with closing(sqlite3.connect(path)) as connection:
        identifier, count = connection.execute("SELECT id,part_count FROM stored_protocols").fetchone()
        if damage == "payload":
            part = connection.execute("SELECT payload FROM stored_protocol_parts LIMIT 1").fetchone()[0]
            connection.execute("UPDATE stored_protocol_parts SET payload=? WHERE position=0", (b"!" + part[1:],))
        elif damage == "gap":
            connection.execute("UPDATE stored_protocol_parts SET position=3")
        elif damage == "missing_part":
            connection.execute("DELETE FROM stored_protocol_parts")
        elif damage == "extra_part":
            connection.execute("INSERT INTO stored_protocol_parts VALUES (?,?,?)", (identifier, count, b"x"))
        elif damage == "part_count":
            connection.execute("UPDATE stored_protocols SET part_count=part_count+1")
        elif damage == "byte_count":
            connection.execute("UPDATE stored_protocols SET byte_count=byte_count+1")
        elif damage == "oversized":
            connection.execute("UPDATE stored_protocols SET byte_count=?", (nested.MAX_STORED_BYTES + 1,))
        elif damage == "kind":
            connection.execute("UPDATE stored_protocols SET kind='unknown'")
        elif damage == "part_type":
            connection.execute("UPDATE stored_protocol_parts SET payload='wrong type'")
        elif damage == "missing_ref":
            connection.execute("UPDATE column_protocols SET protocol_id='missing'")
        elif damage == "unused":
            nested.ProtocolWriter(connection).store({"unused": True})
        else:
            column_id = project.columns[-1].id
            payload = json.loads(connection.execute("SELECT payload FROM table_columns WHERE id=?", (column_id,)).fetchone()[0])
            if damage == "owner":
                payload["id"] = "different column"
            else:
                payload["metadata"]["interpolation_protocol"] = {"ambiguous": True}
            connection.execute("UPDATE table_columns SET payload=? WHERE id=?", (json.dumps(payload), column_id))
        connection.commit()
    before = path.read_bytes()
    with pytest.raises(ValueError):
        Project.load(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("failure", ["io", "changed"])
def test_failed_protocol_write_keeps_previous_file_and_can_be_retried(project, tmp_path, monkeypatch, failure):
    path = tmp_path / "atomic.apyro"
    project.save(path)
    before = path.read_bytes()
    parts, calls = nested._parts, []

    def failing(value):
        calls.append(1)
        for part in parts(value):
            yield b"!" + part[1:] if len(calls) == 2 and failure == "changed" else part
            if len(calls) == 2 and failure == "io":
                raise OSError("interrupted protocol write")

    monkeypatch.setattr(nested, "_parts", failing)
    with pytest.raises(OSError if failure == "io" else ValueError):
        project.save(path)
    assert path.read_bytes() == before and not list(tmp_path.glob("*.tmp"))
    monkeypatch.setattr(nested, "_parts", parts)
    project.save(path)
    assert Project.load(path) == project


def test_registry_limit_falls_back_to_inline_without_changing_archive_limit(project, tmp_path, monkeypatch):
    monkeypatch.setattr(nested, "MAX_STORED_BYTES", 1)
    path = tmp_path / "inline.apyro"
    project.save(path)
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM stored_protocols").fetchone()[0] == 0
    assert Project.load(path) == project
    assert replay(Project.load(path).columns[-1].metadata["interpolation_protocol"])
    assert audit.MAX_PROTOCOL_BYTES == 256 * 1024 * 1024


def test_v3_read_only_open_recovery_and_interrupted_migration_keep_exact_backup(project, tmp_path, write_v3_project, monkeypatch):
    path = write_v3_project(project, tmp_path / "legacy.apyro")
    before = path.read_bytes()
    loaded = Project.load(path)
    assert loaded == project and path.read_bytes() == before
    recovery = tmp_path / "recovery.apyro"
    loaded.save_recovery(recovery)
    assert Project.load(recovery) == loaded and path.read_bytes() == before
    assert not list(tmp_path.glob("*.v3-*.bak"))
    replace = storage.os.replace

    def fail(source, target):
        raise OSError("interrupted migration")

    monkeypatch.setattr(storage.os, "replace", fail)
    with pytest.raises(OSError, match="interrupted"):
        loaded.save(path)
    backup, = tmp_path.glob("*.v3-*.bak")
    assert backup.read_bytes() == before and path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))
    monkeypatch.setattr(storage.os, "replace", replace)
    loaded.save(path)
    assert Project.load(backup) == project and Project.load(path) == project
    loaded.save(path)
    assert list(tmp_path.glob("*.v3-*.bak")) == [backup]
