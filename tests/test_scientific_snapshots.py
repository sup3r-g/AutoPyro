"""Fast scientific snapshots must retain ordinary deepcopy graph semantics."""
from copy import deepcopy
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from threading import Event
import math
import pickle

import pytest

from autopyro.importing import digest
from autopyro.io import build_project
from autopyro.measurements import Measurement
from autopyro.snapshots import copy_project, copy_table_state


@pytest.fixture
def project():
    project = build_project(["Sample", "TOC", "S2"], [["001", "2,00", "<0.1"], ["001", "3", "ND"]], [2, 3],
        [(0, "Sample", "text", ""), (1, "TOC", "number", "%wt"), (2, "S2", "number", "mg HC/g rock")],
        {"path": "лаборатория.csv", "sha256": "synthetic"})
    project.sync_measurements()
    project._storage_path = Path("original.apyro")
    measurement = project.science.current(project.rows[0].id, project.columns[2].id)
    measurement.uncertainty = .03
    measurement.history.append({"previous": {"raw_value": "0,08", "value": .08}})
    measurement.extra = {"notes": ["original"]}
    return project


def test_large_worker_copy_can_be_cancelled_and_validation_reaches_last_record(project, monkeypatch):
    from autopyro.model import work_batches

    records = project.science.measurements
    original = next(iter(records.values()))
    for index in range(8200):
        record = replace(original, id=str(index))
        records[record.id] = record

    cancelled = Event()
    monkeypatch.setattr(work_batches, "sleep", lambda _: cancelled.set())
    with ThreadPoolExecutor(max_workers=1) as worker:
        with pytest.raises(CancelledError):
            worker.submit(copy_project, project, cancelled=cancelled).result()
        worker.submit(project.validate).result()
        records["wrong-key"] = record
        with pytest.raises(ValueError, match="Идентификатор записи"):
            worker.submit(project.validate).result()


def test_project_copy_matches_deepcopy_without_generating_ids_or_timestamps(project, monkeypatch):
    import autopyro.measurements as records
    reference = deepcopy(project)

    def unexpected_constructor(*args, **kwargs):
        raise AssertionError("A snapshot must not create a new analytical identity")

    monkeypatch.setattr(records.Measurement, "__init__", unexpected_constructor)
    copied = copy_project(project)
    assert asdict(copied) == asdict(reference)
    assert digest(copied) == digest(project)
    assert copied._storage_path == project._storage_path
    sid, pid = project.rows[0].id, project.columns[2].id
    copied.science.current(sid, pid).extra["notes"].append("changed")
    copied.science.current(sid, pid).history[0]["previous"]["value"] = 99
    assert project.science.current(sid, pid).extra == {"notes": ["original"]}
    assert project.science.current(sid, pid).history[0]["previous"]["value"] == .08


@pytest.mark.parametrize("extended", [False, True])
def test_sample_snapshot_preserves_shared_metadata_and_extensions(project, extended):
    first, second = project.science.samples.values()
    shared = {"label": "original"}
    first.metadata = second.metadata = project.view["sample_metadata"] = shared
    if extended:
        first.peer = second
        shared["owner"] = first
    copied = copy_project(project)
    a, b = copied.science.samples.values()
    assert a.metadata is b.metadata is copied.view["sample_metadata"]
    assert a is not first and b is not second
    if extended:
        assert a.peer is b and a.metadata["owner"] is a
    a.metadata["label"] = "changed"
    assert shared["label"] == "original"


def test_empty_sample_metadata_alias_and_copy_hook_order(project):
    first, second = project.science.samples.values()
    first.metadata = second.metadata = {}
    project.view["metadata"] = first.metadata
    copied = copy_project(project)
    a, b = copied.science.samples.values()
    assert a.metadata is b.metadata is copied.view["metadata"]
    assert a.metadata is not first.metadata

    class AnnotateSample:
        def __deepcopy__(self, memo):
            first.metadata["label"] = "new"
            return None

    project.columns[0].metadata["hook"] = AnnotateSample()
    copied = copy_project(project)
    assert copied.science.samples[first.id].metadata == {"label": "new"}


def test_sample_pickle_accepts_old_state_and_preserves_shared_empty_metadata(project, monkeypatch):
    from autopyro.model.measurements import Sample

    first, second = project.science.samples.values()
    first.metadata = second.metadata = {}
    for legacy in (False, True):
        with monkeypatch.context() as patches:
            if legacy:
                patches.setattr(Sample, "__reduce_ex__", object.__reduce_ex__)
            payload = pickle.dumps((first, second, first.metadata), protocol=4)
        a, b, metadata = pickle.loads(payload)
        assert (a.id, b.id) == (first.id, second.id)
        assert a.metadata is b.metadata is metadata
        assert metadata == {} and metadata is not first.metadata


@pytest.mark.parametrize("nested", [False, True])
def test_history_previous_states_preserve_aliases_and_isolation(project, nested):
    first, second = list(project.science.measurements.values())[:2]
    previous = {"value": 7, "raw_value": "007"}
    if nested:
        previous["notes"] = ["original"]
        previous["self"] = previous
    first.history = [{"at": "first", "previous": previous}]
    second.history = [{"at": "second", "previous": previous}]
    project.view["previous"] = previous

    copied = copy_project(project)
    owned = copied.view["previous"]
    assert owned is copied.science.measurements[first.id].history[0]["previous"]
    assert owned is copied.science.measurements[second.id].history[0]["previous"]
    assert owned is not previous
    assert list(owned) == list(previous)
    owned["value"] = 8
    assert previous["value"] == 7
    if nested:
        assert owned["self"] is owned
        owned["notes"].append("changed")
        assert previous["notes"] == ["original"]


def test_experimental_analysis_copy_preserves_metadata_aliases_and_cycles(project, monkeypatch):
    from copy import _deepcopy_dispatch
    from autopyro.model.measurements import Analysis
    from tools.benchmark_plot_memory import experimental_analysis_copy

    first, second = list(project.science.analyses.values())[:2]
    shared = {"notes": ["original"], "owner": first}
    first.metadata = second.metadata = shared
    project.science.analysis_state = vars(first)
    expected = copy_project(project)
    monkeypatch.setitem(_deepcopy_dispatch, Analysis, experimental_analysis_copy)
    copied = copy_project(project)

    owned = copied.science.analyses[first.id]
    other = copied.science.analyses[second.id]
    assert owned.metadata is other.metadata is copied.science.analysis_state["metadata"]
    assert owned.metadata["owner"] is owned
    expected_owner = expected.science.analyses[first.id]
    assert (copied.science.analysis_state is vars(owned)) == (
        expected.science.analysis_state is vars(expected_owner))
    assert list(vars(owned)) == list(vars(expected_owner))
    owned.metadata["notes"].append("changed")
    assert shared["notes"] == ["original"]


def test_history_copy_preserves_hook_order_across_revisions(project):
    measurement = next(iter(project.science.measurements.values()))
    later_state = {"value": 7}

    class UpdateLaterState:
        def __deepcopy__(self, memo):
            later_state["value"] = 8
            return "copied note"

    measurement.history = [
        {"note": UpdateLaterState()},
        {"at": "later", "previous": later_state},
    ]
    copied = copy_project(project).science.measurements[measurement.id]
    assert copied.history == [{"note": "copied note"}, {"at": "later", "previous": {"value": 8}}]
    assert copied.history[1]["previous"] is not later_state


@pytest.mark.parametrize("snapshot", [False, True])
def test_table_state_copy_is_independent_and_preserves_cross_layer_aliases(project, snapshot):
    measurement = project.science.current(project.rows[0].id, project.columns[2].id)
    shared = {"notes": ["original"]}
    project.columns[0].metadata["shared"] = shared
    measurement.extra = shared
    project.rows[0].source["shared"] = shared
    state = (project.columns, project.rows, project.science)
    if snapshot:
        state += ({"shared": shared},)
    copied = copy_table_state(state)
    alias = copied[0][0].metadata["shared"]
    assert alias is copied[1][0].source["shared"]
    assert alias is copied[2].measurements[measurement.id].extra
    if snapshot:
        assert alias is copied[3]["shared"]
    assert alias is not shared
    alias["notes"].append("changed")
    assert shared["notes"] == ["original"]


def test_copy_preserves_cycles_between_records_and_the_project(project):
    first, second = list(project.science.measurements.values())[:2]
    first.extra = {"self": first, "other": second, "project": project}
    second.extra = {"other": first}
    copied = copy_project(project)
    a, b = copied.science.measurements[first.id], copied.science.measurements[second.id]
    assert a is not first and b is not second
    assert a.extra["self"] is a and a.extra["other"] is b
    assert b.extra["other"] is a and a.extra["project"] is copied


def test_measurement_subclass_slots_use_normal_deepcopy(project):
    @dataclass
    class ExtendedMeasurement(Measurement):
        __slots__ = ("extra_slot",)
    original = next(iter(project.science.measurements.values()))
    extended = ExtendedMeasurement(**asdict(original))
    extended.extra_slot = ["original"]
    project.science.measurements[original.id] = extended
    project.view["shared_slot"] = extended.extra_slot
    copied = copy_project(project)
    result = copied.science.measurements[original.id]
    assert type(result) is ExtendedMeasurement
    assert result.extra_slot is copied.view["shared_slot"]
    result.extra_slot.append("changed")
    assert extended.extra_slot == ["original"]


def test_nested_custom_copy_hook_is_respected(project):
    class CopyAwareList(list):
        def __deepcopy__(self, memo):
            result = CopyAwareList(["custom copy"])
            memo[id(self)] = result
            return result
    measurement = next(iter(project.science.measurements.values()))
    measurement.extra = CopyAwareList(["original"])
    copied = copy_project(project)
    assert copied.science.measurements[measurement.id].extra == ["custom copy"]
    assert measurement.extra == ["original"]


def test_measurement_with_custom_copy_hook_uses_that_hook(project):
    original = next(iter(project.science.measurements.values()))
    calls = []

    def custom_copy(memo):
        calls.append(original.id)
        result = Measurement(original.sample_id, original.property_id, original.analysis_id,
                             id=original.id, raw_value="custom", value="custom")
        memo[id(original)] = result
        return result

    original.__deepcopy__ = custom_copy
    copied = copy_project(project)
    assert calls == [original.id]
    assert copied.science.measurements[original.id].value == "custom"
    assert original.value != "custom"


def test_scalar_subclass_mutable_attributes_are_not_shared(project):
    class AnnotatedInt(int):
        pass
    measurement = next(iter(project.science.measurements.values()))
    measurement.extra = AnnotatedInt(7)
    measurement.extra.notes = ["original"]
    copied = copy_project(project)
    result = copied.science.measurements[measurement.id].extra
    assert type(result) is AnnotatedInt and result == 7
    result.notes.append("changed")
    assert measurement.extra.notes == ["original"]


@pytest.mark.parametrize("table", [False, True])
def test_flat_maps_preserve_order_values_and_cross_layer_aliases(project, table):
    flat = {"raw": "001", "none": None, 7: -0.0, "large": 2**100, "boolean": True}
    project.rows[0].original = flat
    source = next(iter(project.science.sources.values()))
    source.original = flat
    project.columns[0].metadata["shared"] = flat
    project.view["shared"] = flat
    if table:
        columns, rows, store, view = copy_table_state((project.columns, project.rows, project.science, project.view))
    else:
        result = copy_project(project)
        columns, rows, store, view = result.columns, result.rows, result.science, result.view
    owned = rows[0].original
    assert owned == flat and list(owned) == list(flat) and owned is not flat
    assert math.copysign(1, owned[7]) == -1
    assert store.sources[source.id].original is owned
    assert columns[0].metadata["shared"] is owned and view["shared"] is owned
    owned["new"] = "changed"
    assert "new" not in flat
    selected = store.working[rows[0].id]
    assert selected == project.science.working[rows[0].id]
    selected.clear()
    assert project.science.working[rows[0].id]


@pytest.mark.parametrize("key_subclass", [False, True])
def test_flat_maps_do_not_share_mutable_attributes_of_scalar_subclasses(project, key_subclass):
    class AnnotatedStr(str):
        pass
    text = AnnotatedStr("annotated")
    text.notes = ["original"]
    project.rows[0].original = {text: "raw"} if key_subclass else {"raw": text}
    copied = copy_project(project).rows[0].original
    owned = next(iter(copied)) if key_subclass else copied["raw"]
    assert type(owned) is AnnotatedStr and owned is not text
    owned.notes.append("changed")
    assert text.notes == ["original"]


def test_flat_map_candidates_keep_custom_dict_hooks_and_shared_nested_lists(project):
    calls = []
    class CustomMap(dict):
        def __deepcopy__(self, memo):
            calls.append(1)
            result = CustomMap({"copied": deepcopy(self["raw"], memo)})
            memo[id(self)] = result
            return result
    nested = ["original"]
    source = next(iter(project.science.sources.values()))
    source.original = CustomMap(raw=nested)
    project.rows[0].original = {"nested": nested}
    project.view["shared"] = source.original
    copied = copy_project(project)
    assert calls == [1]
    owned = copied.science.sources[source.id].original
    assert type(owned) is CustomMap and copied.view["shared"] is owned
    assert owned["copied"] is copied.rows[0].original["nested"]
    owned["copied"].append("changed")
    assert nested == ["original"]


def test_nonflat_map_cycle_remains_independent(project):
    value = {"self": None}
    value["self"] = value
    project.rows[0].original = value
    project.view["shared"] = value
    copied = copy_project(project)
    owned = copied.rows[0].original
    assert owned is not value and owned["self"] is owned
    assert copied.view["shared"] is owned


@pytest.mark.parametrize("table", [False, True])
@pytest.mark.parametrize("experimental", [False, True])
def test_empty_histories_preserve_shared_and_distinct_lists(project, table, experimental, monkeypatch):
    if experimental:
        from autopyro import snapshots
        from tools.benchmark_snapshot_parts import experimental_measurement_memo
        monkeypatch.setattr(snapshots, "_measurement_memo", experimental_measurement_memo)
    first, second, third = list(project.science.measurements.values())[:3]
    first.history = second.history = []
    third.history = []
    project.view["history"] = first.history
    if table:
        _, _, store, view = copy_table_state((project.columns, project.rows, project.science, project.view))
    else:
        copied = copy_project(project)
        store, view = copied.science, copied.view
    owned = store.measurements[first.id].history
    assert owned is store.measurements[second.id].history and owned is view["history"]
    assert owned is not first.history and owned is not store.measurements[third.id].history
    owned.append({"qa": ["changed"]})
    assert first.history == second.history == third.history == []
    assert store.measurements[third.id].history == []


@pytest.mark.parametrize("experimental", [False, True])
def test_empty_history_subclass_keeps_custom_copy_hook(project, experimental, monkeypatch):
    if experimental:
        from autopyro import snapshots
        from tools.benchmark_snapshot_parts import experimental_measurement_memo
        monkeypatch.setattr(snapshots, "_measurement_memo", experimental_measurement_memo)
    class CustomHistory(list):
        def __deepcopy__(self, memo):
            result = CustomHistory(["custom"])
            memo[id(self)] = result
            return result
    record = next(iter(project.science.measurements.values()))
    record.history = CustomHistory()
    copied = copy_project(project)
    assert copied.science.measurements[record.id].history == ["custom"]
    assert record.history == []


@pytest.mark.parametrize("layout", ["normal", "extension", "reordered"])
def test_scalar_field_experiment_retains_history_aliases_and_field_order(project, monkeypatch, layout):
    from autopyro import snapshots
    from tools.benchmark_snapshot_parts import experimental_scalar_memo
    record = list(project.science.measurements.values())[1]
    record.history.append({"previous": {"notes": ["original"]}})
    project.view["history"] = record.history
    if layout == "extension":
        record.extra = "scalar extension after history"
    elif layout == "reordered":
        values = vars(record).copy()
        history = values.pop("history")
        record.__dict__ = {"history": history, **values}
    expected = deepcopy(project)
    monkeypatch.setattr(snapshots, "_measurement_memo", experimental_scalar_memo)
    copied = copy_project(project)
    owned = copied.science.measurements[record.id]
    assert asdict(copied) == asdict(expected)
    assert list(vars(owned)) == list(vars(record))
    assert owned.history is copied.view["history"] and owned.history is not record.history
    owned.history[-1]["previous"]["notes"].append("changed")
    assert record.history[-1]["previous"]["notes"] == ["original"]


@pytest.mark.parametrize("cycles", [False, True])
def test_single_pass_experiment_preserves_data_and_recursive_references(project, monkeypatch, cycles):
    from autopyro import snapshots
    from tools.benchmark_snapshot_parts import experimental_single_pass_memo
    first, second = list(project.science.measurements.values())[:2]
    if cycles:
        first.history.append({"peer": second, "project": project})
        second.history.append({"peer": first})
    reference = deepcopy(project)
    monkeypatch.setattr(snapshots, "_measurement_memo", experimental_single_pass_memo)
    copied = copy_project(project)
    if cycles:
        a, b = copied.science.measurements[first.id], copied.science.measurements[second.id]
        assert a.history[-1]["peer"] is b
        assert b.history[-1]["peer"] is a
        assert a.history[-1]["project"] is copied
        assert a is not first and b is not second
    else:
        assert asdict(copied) == asdict(reference)
    assert copied.rows == project.rows and copied.columns == project.columns
