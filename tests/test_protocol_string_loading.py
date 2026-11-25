"""Equal strings may be shared; mutable protocols and invalid links must not be."""
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict
import json
import math
import sqlite3

import pytest

from autopyro import audit
from autopyro.core import Column, Project
from autopyro.database import json_strings
from autopyro.database.json_strings import JsonStringPool
from autopyro.interpolation import prepare, replay
from autopyro.io import build_project


def test_json_values_order_and_mutability_are_preserved():
    pool = JsonStringPool()
    value = {"row-идентификатор": [["ND", "<0.1", "≈2"], "row-идентификатор"],
             "values": [2**100, -0.0, 1e-100, True, False, None, {"unknown": "missing-reference"}]}
    text = json.dumps(value, ensure_ascii=False)
    first, second = pool.loads(text), pool.loads(text)
    assert first == second == value
    assert list(first) == list(value)
    assert math.copysign(1, first["values"][1]) == -1
    key = next(iter(first))
    assert key is next(iter(second)) and first[key][1] is key
    assert first[key][0][0] is second[key][0][0]
    first[key][0].append("changed")
    first["values"][-1]["unknown"] = "changed"
    assert second == value
    assert pool.loads('["ND", ["ND"]]')[0] is pool.loads('{"x":"ND"}')["x"]


def test_pool_is_bounded_local_and_skips_long_free_text(monkeypatch):
    monkeypatch.setattr(json_strings, "MAX_SHARED_STRINGS", 2)
    pool = JsonStringPool()
    first = pool.string("a unique identifier 0001")
    pool.string("a unique identifier 0002")
    third = "a unique identifier 0003"
    assert pool.string(third) is third and len(pool._strings) == 2
    decoded = json.loads(json.dumps(first))
    assert decoded is not first and pool.string(decoded) is first
    assert JsonStringPool().string(decoded) is decoded  # No process-global interning.
    fresh = JsonStringPool()
    long = "long free text " * 100
    assert fresh.string(long) is long and fresh._strings == {}


@pytest.fixture
def project():
    project = build_project(["Well", "X", "Y"],
                            [["A", "0", "2"], ["A", "1", ""], ["A", "2", "6"],
                             ["A", ".25", "ND"], ["A", ".5", "<0.1"]], list(range(2, 7)),
                            [(0, "Well", "text", ""), (1, "X", "number", "m"),
                             (2, "Y", "number", "mg/g")], {})
    project.sync_measurements()
    well, x, y = project.columns
    proposals, protocol = prepare(project, x.id, y.id, [well.id], set(project.row_map()), method="pchip",
                                   constraints={"well_id": well.id, "analytical_context": True})
    column = Column("Interpolated Y", "number", unit=y.unit,
                    metadata={"method": "pchip", "interpolation_protocol": protocol})
    project.columns.append(column)
    values = {p.row_id: p.value for p in proposals}
    for row in project.rows:
        row.values[column.id] = values.get(row.id)
    project.history.extend({"action": "add_column", "metadata": {
        "interpolation_protocol": deepcopy(protocol)}} for _ in range(2))
    return project


@pytest.mark.parametrize("version", [2, 3, 4])
def test_inline_protocols_share_strings_not_objects_and_replay(project, tmp_path, write_v2_project, write_v3_project, version):
    path = tmp_path / "protocols.apyro"
    if version == 2:
        write_v2_project(project, path)
    elif version == 3:
        write_v3_project(project, path)
    else:
        project.save(path)
    before = path.read_bytes()
    loaded = Project.load(path)
    assert asdict(loaded) == asdict(project) and path.read_bytes() == before
    protocol = loaded.columns[-1].metadata["interpolation_protocol"]
    first, second = [e["metadata"]["interpolation_protocol"] for e in loaded.history[-2:]]
    assert protocol is not first and first is not second and protocol is not second
    assert protocol["rows"] is not first["rows"]
    mid = next(iter(protocol["measurements"]))
    assert mid is next(iter(first["measurements"]))
    assert mid is loaded.science.measurements[mid].id
    assert protocol["measurements"][mid] is not first["measurements"][mid]
    assert protocol["measurements"][mid]["sample_id"] is first["measurements"][mid]["sample_id"]
    expected = replay(protocol)
    assert expected == replay(first) == replay(second)
    first["rows"][0]["values"][loaded.columns[2].id] = "changed history"
    assert replay(protocol) == replay(second) == expected
    assert Project.load(path).columns[-1].metadata["interpolation_protocol"] == protocol
    assert path.read_bytes() == before


def test_nested_archives_share_text_without_inflating_on_open(project, tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "MIN_PROTOCOL_BYTES", 0)
    archived = audit.archive_protocol(project.columns[-1].metadata["interpolation_protocol"])
    assert isinstance(archived, str)
    project.columns[-1].metadata["interpolation_protocol"] = archived
    for event in project.history[-2:]:
        event["metadata"]["interpolation_protocol"] = archived
    path = tmp_path / "archives.apyro"
    project.save(path)

    def forbidden(_):
        raise AssertionError("Nested archive must stay lazy while loading")

    monkeypatch.setattr(audit, "read_protocol", forbidden)
    loaded = Project.load(path)
    text = loaded.columns[-1].metadata["interpolation_protocol"]
    assert text == archived
    assert all(e["metadata"]["interpolation_protocol"] is text for e in loaded.history[-2:])


@pytest.mark.parametrize("field", ["id", "analysis_id"])
def test_pool_does_not_repair_invalid_scientific_payloads(project, tmp_path, field):
    path = tmp_path / "invalid.apyro"
    project.save(path)
    mid = next(iter(project.science.measurements))
    with closing(sqlite3.connect(path)) as connection:
        payload = json.loads(connection.execute("SELECT payload FROM measurements WHERE id=?", (mid,)).fetchone()[0])
        payload[field] = "missing-reference"
        connection.execute("UPDATE measurements SET payload=? WHERE id=?", (json.dumps(payload), mid))
        connection.commit()
    before = path.read_bytes()
    with pytest.raises(ValueError):
        Project.load(path)
    assert path.read_bytes() == before
