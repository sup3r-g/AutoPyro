"""Frozen interpolation inputs reproduce all methods independently of a project."""
from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from autopyro.core import Project
from autopyro.interpolation import prepare, propose, replay
from autopyro.io import build_project


@pytest.fixture
def data():
    headers = ["Well", "Depth", "Y", "Unused"]
    records = [["A", "0", "0", "private"], ["A", "1", "", "private"],
               ["A", "2", "4", "private"], ["A", "3", "", "private"],
               ["A", "4", "5", "private"], ["A", "5", "ND", "private"],
               ["A", "6", "12", "private"], ["B", "1", "99", "private"]]
    project = build_project(headers, records, list(range(2, 10)),
                         [(i, name, "text" if i in (0, 3) else "number", "m" if i == 1 else "")
                          for i, name in enumerate(headers)], {"type": "test"})
    project.sync_measurements()
    return project


def parameters(project, method="linear"):
    well, x, y, _ = project.columns
    return dict(x_id=x.id, y_id=y.id, group_ids=[well.id],
                row_ids={r.id for r in project.rows[:-1]}, method=method, max_gap=3)


@pytest.mark.parametrize("method", ["linear", "nearest", "pchip"])
def test_replay_all_anchors_isolated_provenance_and_roundtrip(data, method, tmp_path):
    options = parameters(data, method)
    original = deepcopy(data)
    proposals, protocol = prepare(data, **options)
    assert proposals == propose(data, **options)
    assert data == original
    assert len(proposals) == 2
    assert len(protocol["supports"]) == 1
    assert [a["row_id"] for a in protocol["supports"][0]["anchors"]] == [data.rows[i].id for i in (0, 2, 4, 6)]
    assert replay(json.loads(json.dumps(protocol))) == proposals
    assert protocol["environment"]["scipy"]
    assert protocol["columns"][1]["unit"] == "m"
    row = protocol["rows"][0]
    measurement = data.science.current(row["id"], data.columns[2].id)
    assert protocol["measurements"][measurement.id]["revision"] == measurement.revision
    assert protocol["analyses"][measurement.analysis_id] == asdict(data.science.analyses[measurement.analysis_id])
    assert "history" not in protocol["measurements"][measurement.id]
    encoded = json.dumps(protocol)
    assert "private" not in encoded
    assert data.rows[-1].id not in encoded
    frozen = deepcopy(protocol)
    data.rows.clear()
    data.science.measurements[measurement.id].raw_value = 999
    data.columns[1].unit = "km"
    assert protocol == frozen
    assert replay(protocol) == proposals
    # Persist the snapshot in the same column metadata used by the GUI.
    original.columns[2].metadata["interpolation_protocol"] = protocol
    path = tmp_path / "protocol.apyro"
    original.save(path)
    restored = Project.load(path)
    assert replay(restored.columns[2].metadata["interpolation_protocol"]) == proposals


def test_pchip_replay_needs_anchors_outside_the_bracketing_pair(data):
    proposals, protocol = prepare(data, **parameters(data, "pchip"))
    first = proposals[0]
    assert (first.left_id, first.right_id) == (data.rows[0].id, data.rows[2].id)
    # A remote point changes the neighbouring slope and hence the first result.
    data.rows[4].values[data.columns[2].id] = 20
    changed = propose(data, **parameters(data, "pchip"))
    assert first.value != changed[0].value
    assert replay(protocol) == proposals


@pytest.mark.parametrize("damage", ["input", "anchor", "result", "version", "checksum"])
def test_replay_rejects_damage(data, damage):
    _, protocol = prepare(data, **parameters(data))
    if damage == "input":
        protocol["rows"][0]["values"][data.columns[2].id] = 20
    elif damage == "anchor":
        protocol["supports"][0]["anchors"].pop()
    elif damage == "result":
        protocol["proposals"][0]["value"] += 1
    elif damage == "version":
        protocol["version"] = 999
    else:
        protocol["checksum"] = "wrong"
    with pytest.raises(ValueError, match="протокола"):
        replay(protocol)


@pytest.mark.parametrize("gap", [float("nan"), float("inf"), -1, 0])
def test_invalid_gap(data, gap):
    options = parameters(data)
    options["max_gap"] = gap
    with pytest.raises(ValueError, match="параметры"):
        prepare(data, **options)


def test_empty_scope_replays_without_anchors(data):
    options = parameters(data)
    options["row_ids"] = set()
    proposals, protocol = prepare(data, **options)
    assert replay(protocol) == proposals == []
    assert protocol["rows"] == protocol["supports"] == []


def test_replay_checks_recomputed_results_and_policy_not_only_checksum(data):
    from autopyro.interpolation import _checksum

    _, protocol = prepare(data, **parameters(data))
    protocol["proposals"][0]["value"] += 1
    protocol["checksum"] = _checksum(protocol)
    with pytest.raises(ValueError, match="не совпадает"):
        replay(protocol)
    protocol["policies"]["extrapolation"] = True
    protocol["checksum"] = _checksum(protocol)
    with pytest.raises(ValueError, match="версия"):
        replay(protocol)


@pytest.mark.parametrize("legacy", [None, {}, {"method": "linear", "proposals": []}])
def test_legacy_metadata_cannot_fabricate_a_snapshot(legacy):
    with pytest.raises(ValueError, match="версия"):
        replay(legacy)


def test_snapshot_records_selected_repeat_and_its_analytical_context(data):
    y = data.columns[2]
    row = data.rows[2]
    old = data.science.current(row.id, y.id)
    repeat = data.science.add_result(row.id, y.id, 8,
                                     analysis_fields={"method": "Lab method", "laboratory": "Lab B"},
                                     details={"uncertainty": .2, "uncertainty_type": "standard"})
    data.select_measurement(row.id, y.id, repeat.id)
    proposals, protocol = prepare(data, **parameters(data, "pchip"))
    assert old.id not in protocol["measurements"]
    assert protocol["measurements"][repeat.id]["uncertainty"] == .2
    assert protocol["analyses"][repeat.analysis_id]["laboratory"] == "Lab B"
    assert next(r for r in protocol["rows"] if r["id"] == row.id)["working"][y.id] == repeat.id
    data.select_measurement(row.id, y.id, old.id)
    assert replay(protocol) == proposals
