"""Encoding v3 omits only declared nulls; scientific inputs and replay stay exact."""
from copy import deepcopy
from dataclasses import fields, MISSING
import json
import math
import sys

import pytest

from autopyro import audit, interpolation
from autopyro.core import Project
from autopyro.io import build_project
from autopyro.measurements import Measurement


@pytest.fixture
def data():
    project = build_project(["Well", "X", "Y"],
                            [["A", "0", "2"], ["A", "1", ""], ["A", "2", "6"],
                             ["A", "3", "ND"], ["A", "4", "<0.1"], ["A", "5", "10"]],
                            list(range(2, 8)), [(0, "Well", "text", ""), (1, "X", "number", "m"),
                                                (2, "Y", "number", "mg/g")], {})
    project.sync_measurements()
    for analysis in project.science.analyses.values():
        analysis.method, analysis.laboratory = "Pyrol6", "QA"
    measurement = project.science.current(project.rows[2].id, project.columns[2].id)
    measurement.limit, measurement.lod, measurement.loq = 0, .1, .2
    measurement.uncertainty, measurement.confidence_level = -0.0, .95
    measurement.uncertainty_type = "standard"
    return project


def options(data, method, constrained):
    well, x, y = data.columns
    result = dict(x_id=x.id, y_id=y.id, group_ids=[well.id], row_ids=set(data.row_map()), method=method, max_gap=10)
    if constrained:
        result["constraints"] = {"well_id": well.id, "analytical_context": True}
    return result


@pytest.mark.parametrize("method", ["linear", "nearest", "pchip"])
@pytest.mark.parametrize("constrained", [False, True])
def test_v3_expands_to_identical_measurements_and_replays_old_and_new(data, monkeypatch, method, constrained, tmp_path):
    before = deepcopy(data)
    args = options(data, method, constrained)
    monkeypatch.setattr(interpolation, "COMPACT_MEASUREMENT_THRESHOLD", sys.maxsize)
    expected, old = interpolation.prepare(data, **args)
    monkeypatch.setattr(interpolation, "COMPACT_MEASUREMENT_THRESHOLD", 0)
    actual, compact = interpolation.prepare(data, **args)
    assert compact["version"] == 3 and old["version"] == (2 if constrained else 1)
    assert compact["algorithm"] == old["algorithm"]
    assert actual == expected == interpolation.replay(old) == interpolation.replay(compact)
    assert data == before
    for key in ("parameters", "policies", "columns", "rows", "analyses", "sources", "supports", "proposals"):
        assert compact[key] == old[key]
    expanded = {mid: {**compact["measurement_defaults"], **record} for mid, record in compact["measurements"].items()}
    assert expanded == old["measurements"]
    for mid, record in compact["measurements"].items():
        assert record["id"] == mid
        assert record["created_at"] == old["measurements"][mid]["created_at"]
        assert all(key not in record or record[key] is not None for key in compact["measurement_defaults"])
    selected = data.science.current(data.rows[2].id, data.columns[2].id)
    record = compact["measurements"][selected.id]
    assert record["limit"] == 0 and record["lod"] == .1 and record["loq"] == .2
    assert math.copysign(1, record["uncertainty"]) == -1 and record["confidence_level"] == .95
    assert len(json.dumps(compact)) < len(json.dumps(old))
    monkeypatch.setattr(audit, "MIN_PROTOCOL_BYTES", 0)
    archived = audit.archive_protocol(compact)
    assert isinstance(archived, str)
    assert interpolation.replay(archived) == expected
    data.columns[-1].metadata["interpolation_protocol"] = archived
    path = tmp_path / "compact.apyro"
    data.save(path)
    assert interpolation.replay(Project.load(path).columns[-1].metadata["interpolation_protocol"]) == expected


def test_v3_null_defaults_are_explicit_not_factory_generated():
    declared = {field.name: field for field in fields(Measurement)}
    for name, value in interpolation.NULL_MEASUREMENT_DEFAULTS.items():
        assert value is None and declared[name].default is None
        assert declared[name].default_factory is MISSING
    assert {"id", "created_at", "revision", "qualifier", "unit", "basis"} <= interpolation.REQUIRED_MEASUREMENT_FIELDS
    assert interpolation.REQUIRED_MEASUREMENT_FIELDS | interpolation.NULL_MEASUREMENT_DEFAULTS.keys() == set(declared) - {"history"}


@pytest.mark.parametrize("damage", ["missing_defaults", "changed_defaults", "extra_default", "id", "created_at", "revision", "qc_status", "mismatched_id"])
def test_v3_rejects_ambiguous_defaults_or_missing_trace_even_with_new_checksum(data, monkeypatch, damage):
    monkeypatch.setattr(interpolation, "COMPACT_MEASUREMENT_THRESHOLD", 0)
    _, protocol = interpolation.prepare(data, **options(data, "pchip", True))
    if damage == "missing_defaults":
        protocol.pop("measurement_defaults")
    elif damage == "changed_defaults":
        protocol["measurement_defaults"]["lod"] = 0
    elif damage == "extra_default":
        protocol["measurement_defaults"]["id"] = None
    else:
        record = next(iter(protocol["measurements"].values()))
        if damage == "mismatched_id":
            record["id"] = "different-id"
        else:
            record.pop(damage)
    protocol["checksum"] = interpolation._checksum(protocol)
    with pytest.raises(ValueError, match="протокола"):
        interpolation.replay(protocol)


@pytest.mark.parametrize("constrained", [False, True])
def test_empty_compact_scope_remains_replayable(data, monkeypatch, constrained):
    monkeypatch.setattr(interpolation, "COMPACT_MEASUREMENT_THRESHOLD", 0)
    args = options(data, "pchip", constrained)
    args["row_ids"] = set()
    proposals, protocol = interpolation.prepare(data, **args)
    assert protocol["version"] == (2 if constrained else 1)
    protocol.update(version=3, measurement_defaults=dict(interpolation.NULL_MEASUREMENT_DEFAULTS))
    protocol["checksum"] = interpolation._checksum(protocol)
    assert proposals == interpolation.replay(protocol) == []


def test_compaction_is_skipped_when_default_declaration_would_expand_json(data, monkeypatch):
    monkeypatch.setattr(interpolation, "COMPACT_MEASUREMENT_THRESHOLD", 0)
    for record in data.science.measurements.values():
        record.limit, record.lod, record.loq = 0, .1, .2
        record.uncertainty, record.confidence_level = 0, .95
    proposals, protocol = interpolation.prepare(data, **options(data, "pchip", True))
    assert protocol["version"] == 2 and "measurement_defaults" not in protocol
    assert interpolation.replay(protocol) == proposals


def test_compact_unconstrained_mode_keeps_legacy_provenance_extensions(data, monkeypatch):
    measurement = next(iter(data.science.measurements.values()))
    measurement.extra = {"notes": ["custom provenance"]}
    monkeypatch.setattr(interpolation, "COMPACT_MEASUREMENT_THRESHOLD", 0)
    proposals, protocol = interpolation.prepare(data, **options(data, "pchip", False))
    assert protocol["version"] == 3 and protocol["algorithm"] == interpolation.ALGORITHM
    assert protocol["measurements"][measurement.id]["extra"] == measurement.extra
    assert interpolation.replay(protocol) == proposals
