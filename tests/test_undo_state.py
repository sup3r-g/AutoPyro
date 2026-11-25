"""Internal compressed states retain identifiers, graphs and independent restores."""
from dataclasses import FrozenInstanceError, asdict

import pytest

from autopyro.core import Column, Project, Row
from autopyro.ui.table import undo_state


class CopyAwareExtension:
    def __init__(self, count=0):
        self.count = count

    def __deepcopy__(self, memo):
        result = CopyAwareExtension(self.count + 1)
        memo[id(self)] = result
        return result


@pytest.fixture
def state():
    column = Column("S2", "number", unit="mg HC/g rock")
    project = Project(columns=[column], rows=[Row({column.id: "ND"})])
    project.sync_measurements()
    measurement = project.science.current(project.rows[0].id, column.id)
    measurement.uncertainty = .3
    measurement.extra = {"notes": ["original"]}
    return project.columns, project.rows, project.science, {"present": True}


def test_packed_state_matches_original_and_restores_independent_objects(state, monkeypatch):
    monkeypatch.setattr(undo_state, "MIN_PACK_BYTES", 0)
    packed = undo_state.pack_table_state(state)
    assert isinstance(packed, undo_state.PackedTableState)
    assert packed.cell_count == 1 and packed.measurement_count == 1
    first, second = packed.restore(), packed.restore()
    assert asdict(first[2]) == asdict(state[2])
    measurement = next(iter(first[2].measurements.values()))
    original = next(iter(state[2].measurements.values()))
    assert measurement.id == original.id and measurement.created_at == original.created_at
    assert measurement.qualifier == "non_detect" and measurement.uncertainty == .3
    measurement.extra["notes"].append("changed")
    assert original.extra["notes"] == ["original"]
    assert next(iter(second[2].measurements.values())).extra["notes"] == ["original"]
    with pytest.raises(FrozenInstanceError):
        packed.payload = b"changed"


def test_packed_state_retains_mutable_aliases_and_cycles(state, monkeypatch):
    monkeypatch.setattr(undo_state, "MIN_PACK_BYTES", 0)
    measurement = next(iter(state[2].measurements.values()))
    shared = {"notes": ["original"]}
    state[0][0].metadata["shared"] = shared
    state[1][0].source["shared"] = shared
    state[3]["shared"] = shared
    measurement.extra = {"shared": shared, "self": measurement}
    copied = undo_state.pack_table_state(state).restore()
    result = next(iter(copied[2].measurements.values()))
    alias = copied[0][0].metadata["shared"]
    assert alias is copied[1][0].source["shared"] is copied[3]["shared"] is result.extra["shared"]
    assert alias is not shared and result.extra["self"] is result


def test_small_state_remains_plain_and_restore_still_copies(state, monkeypatch):
    monkeypatch.setattr(undo_state, "MIN_PACK_BYTES", 10**9)
    assert undo_state.pack_table_state(state) is state
    copied = undo_state.restore_table_state(state)
    copied[0][0].name = "changed"
    assert state[0][0].name == "S2"


def test_non_pickleable_extension_keeps_plain_snapshot(state, monkeypatch):
    class LocalExtension:
        def __init__(self):
            self.notes = ["original"]
    monkeypatch.setattr(undo_state, "MIN_PACK_BYTES", 0)
    measurement = next(iter(state[2].measurements.values()))
    measurement.extra = LocalExtension()
    assert undo_state.pack_table_state(state) is state
    copied = undo_state.restore_table_state(state)
    result = next(iter(copied[2].measurements.values()))
    result.extra.notes.append("changed")
    assert measurement.extra.notes == ["original"]


def test_pickleable_extension_keeps_custom_deepcopy_semantics(state, monkeypatch):
    monkeypatch.setattr(undo_state, "MIN_PACK_BYTES", 0)
    measurement = next(iter(state[2].measurements.values()))
    measurement.extra = CopyAwareExtension()
    assert undo_state.pack_table_state(state) is state
    copied = undo_state.restore_table_state(state)
    result = next(iter(copied[2].measurements.values()))
    assert result.extra.count == 1 and measurement.extra.count == 0
