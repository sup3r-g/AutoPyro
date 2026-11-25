"""Sharing decoded strings must preserve complete scientific records and errors."""
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict
import json
import sqlite3

import pytest

from autopyro.data_io.io import build_project
from autopyro.database import json_strings, storage
from autopyro.model.core import Project
from autopyro.model.measurements import MeasurementStore


@pytest.fixture
def project():
    project = build_project(["Sample", "TOC", "S2"],
                            [["001", "2.00", "<0.1"], ["001", "2.00", "ND"]], [2, 3],
                            [(0, "Sample", "text", ""), (1, "TOC", "number", "%wt"),
                             (2, "S2", "number", "mg HC/g rock")], {"path": "лаборатория.csv"})
    project.sync_measurements()
    first, second = project.rows
    toc = project.columns[1]
    project.science.set_value(first.id, toc.id, 3, "number")
    first.values[toc.id] = 3
    repeat = project.science.add_result(second.id, toc.id, "~2.5",
                                       analysis_fields={"laboratory": "Lab", "method": "Pyrol6"},
                                       details={"basis": "custom basis", "uncertainty": .1})
    repeat.qc_status = "expert pending"
    return project


def decoded(project):
    return json.loads(json.dumps(asdict(project.science), ensure_ascii=False))


def test_large_sqlite_open_rejects_a_dangling_foreign_key(project, tmp_path, monkeypatch):
    path = tmp_path / "broken-reference.apyro"
    project.save(path)
    monkeypatch.setattr(storage, "LARGE_PROJECT_BYTES", 0)
    assert Project.load(path).science == project.science

    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute(
            "UPDATE working_measurements SET measurement_id = ? "
            "WHERE rowid = (SELECT min(rowid) FROM working_measurements)",
            ("absent-measurement",),
        )
        connection.commit()
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Нарушена целостность проекта"):
        Project.load(path)
    assert path.read_bytes() == before


def test_sqlite_shares_repeated_cell_text_across_independent_records(tmp_path):
    text = "Повторяющееся описание лабораторной пробы"
    project = build_project(["Sample"], [[text], [text]], [2, 3], [(0, "Sample", "text", "")], {})
    path = tmp_path / "shared-cells.apyro"
    project.save(path)

    loaded = Project.load(path)
    first, second = loaded.rows
    pid = loaded.columns[0].id
    shared = first.values[pid]
    for row in loaded.rows:
        measurement = loaded.science.current(row.id, pid)
        source = loaded.science.sources[measurement.source_id]
        assert shared is row.values[pid] is row.original[pid]
        assert shared is source.original[pid] is measurement.raw_value is measurement.value
        assert source.original is not row.original
    first.values[pid] = "Другое описание"
    assert second.values[pid] == text and first.original[pid] == text


def test_cell_text_pool_is_bounded_and_only_shares_short_text(monkeypatch):
    monkeypatch.setattr(json_strings, "MAX_SHARED_CELL_STRINGS", 1)
    pool = json_strings.JsonStringPool()
    first = b"first literal value".decode()
    assert pool.cell(first) is first
    assert pool.existing_cell(first.encode().decode()) is first
    overflow = b"uncached literal value".decode()
    assert pool.cell(overflow) is overflow
    duplicate = overflow.encode().decode()
    assert pool.cell(duplicate) is duplicate

    fresh_pool = json_strings.JsonStringPool()
    long_text = "a" * (json_strings.MAX_SHORT_STRING_LENGTH + 1)
    assert fresh_pool.cell(long_text) is long_text
    assert fresh_pool.existing_cell(long_text.encode().decode()) is not long_text


def test_sqlite_working_rows_can_revisit_a_sample(project):
    payload = decoded(project)
    working = payload.pop("working")
    columns = [column.id for column in project.columns]
    interleaved = [(sid, pid, selected[pid]) for pid in columns for sid, selected in working.items()]
    restored = MeasurementStore.from_sqlite_payloads(
        {name: records.items() for name, records in payload.items()}, iter(interleaved),
    )
    assert list(restored.working) == list(working)
    assert restored.working == working
    assert all(list(selected) == columns for selected in restored.working.values())


def test_measurement_payloads_accept_reordered_fields_and_legacy_defaults(project):
    payload = decoded(project)
    mid, record = next(iter(payload['measurements'].items()))
    reordered = dict(reversed(list(record.items())))
    reordered.pop('confidence_level')
    payload['measurements'][mid] = reordered
    restored = MeasurementStore.from_dict(payload)
    assert asdict(restored.measurements[mid]) == {**record, 'confidence_level': None}
    reordered['unknown_field'] = 123
    with pytest.raises(TypeError, match='unknown_field'):
        MeasurementStore.from_dict(payload)


@pytest.mark.parametrize("value", [True, float("inf"), float("nan")])
def test_project_validation_still_rejects_invalid_numbers(project, value):
    project.rows[0].values[project.columns[1].id] = value
    with pytest.raises(ValueError, match="Логическое значение|NaN и бесконечность"):
        project.validate()


def test_decoded_store_preserves_values_payload_and_independent_records(project):
    payload = decoded(project)
    before = deepcopy(payload)
    store = MeasurementStore.from_dict(payload)
    assert asdict(store) == before
    assert payload == before
    store.validate(project)
    results = list(store.measurements.values())
    assert len({id(m) for m in results}) == len(results)
    assert len({id(m.history) for m in results}) == len(results)
    edited = store.current(project.rows[0].id, project.columns[1].id)
    assert edited.history and edited.revision == 2
    assert any(m.basis == "custom basis" and m.qc_status == "expert pending" for m in results)
    # Repeated foreign keys share immutable strings, not result objects or lists.
    for m in results:
        assert m.sample_id is store.samples[m.sample_id].id
        assert m.property_id is store.properties[m.property_id].id
        assert m.analysis_id is store.analyses[m.analysis_id].id
        if m.source_id is not None:
            assert m.source_id is store.sources[m.source_id].id
    before_other = deepcopy(results[1].history)
    results[0].history.append({"note": "one result only"})
    assert results[1].history == before_other


@pytest.mark.parametrize("corruption", ["key", "analysis", "source", "property", "working"])
def test_invalid_references_are_not_repaired_by_string_sharing(project, corruption):
    payload = decoded(project)
    mid = next(iter(payload["measurements"]))
    measurement = payload["measurements"][mid]
    if corruption == "key":
        measurement["id"] = "different-id"
    elif corruption == "analysis":
        measurement["analysis_id"] = "missing-analysis"
    elif corruption == "source":
        measurement["source_id"] = "missing-source"
    elif corruption == "property":
        measurement["property_id"] = "missing-property"
    else:
        sid = project.rows[0].id
        payload["working"][sid][project.columns[1].id] = "missing-result"
    store = MeasurementStore.from_dict(payload)
    assert asdict(store) == payload
    with pytest.raises(ValueError):
        store.validate(project)


@pytest.mark.parametrize("field", ["lod", "loq", "uncertainty", "limit"])
def test_present_measurement_limits_are_validated(project, field):
    measurement = next(iter(project.science.measurements.values()))
    setattr(measurement, field, -1)
    with pytest.raises(ValueError, match=field + ":"):
        project.validate()


def test_streamed_sqlite_roundtrip_preserves_full_project_and_remains_read_only(project, tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "SCIENTIFIC_READ_BATCH_SIZE", 2)
    path = tmp_path / "science.apyro"
    project.save(path)
    before = path.read_bytes()
    restored = Project.load(path)
    assert asdict(restored) == asdict(project)
    assert path.read_bytes() == before
    assert len(restored.science.results(project.rows[1].id, project.columns[1].id)) == 2
    for sid, selected in restored.science.working.items():
        assert sid is restored.science.samples[sid].id
        for pid, mid in selected.items():
            assert pid is restored.science.properties[pid].id
            assert mid is restored.science.measurements[mid].id
    # A subsequent edit has its own history; the original project is independent.
    restored.rows[0].values[project.columns[1].id] = 4
    restored.sync_measurements(rows=[restored.rows[0]])
    assert project.science.current(project.rows[0].id, project.columns[1].id).value == 3
