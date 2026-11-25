"""Record encoding compatibility and failures partway through SQLite writing."""
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict, dataclass, InitVar, make_dataclass
import json
import sqlite3
import gc
import weakref
from typing import ClassVar

import pytest

from autopyro.core import Project
from autopyro.database import storage
from autopyro.io import build_project


@dataclass
class Context:
    label: str
    values: tuple


@pytest.fixture
def project():
    result = build_project(["Sample", "TOC", "S2"],
                           [["001", "2", "ND"], ["001", "3", "<0.1"]], [2, 3],
                           [(0, "Sample", "text", ""), (1, "TOC", "number", "%wt"),
                            (2, "S2", "number", "mg HC/g rock")], {})
    result.sync_measurements()
    return result


def test_record_encoder_matches_previous_format_including_nested_dataclasses(project, tmp_path):
    nested = {"context": Context("Лаборатория 🌍", (2**100, -0.0, None, True)),
              "qualifiers": ["ND", "<0.1", "≈2"]}
    project.columns[1].metadata["custom"] = nested
    records = [*project.columns, *project.rows]
    for name in ("samples", "properties", "sources", "analyses", "measurements"):
        records.extend(getattr(project.science, name).values())
    expected = [json.dumps(asdict(record), ensure_ascii=False, allow_nan=False) for record in records]
    assert [storage.encode_record(record) for record in records] == expected
    before = deepcopy(project)
    path = tmp_path / "records.apyro"
    project.save(path)
    assert project == before
    assert project.columns[1].metadata["custom"] is nested
    loaded = Project.load(path)
    assert loaded.columns[1].metadata["custom"] == json.loads(json.dumps(asdict(project.columns[1])))["metadata"]["custom"]
    assert loaded.science == project.science


@pytest.mark.parametrize("representation", ["whitespace", "utf8_blob", "extra_record"])
def test_sqlite_measurement_decoder_preserves_record_boundaries(project, tmp_path, representation):
    path = tmp_path / "decoder.apyro"
    project.save(path)
    with closing(sqlite3.connect(path)) as connection:
        mid, payload = connection.execute("SELECT id, payload FROM measurements LIMIT 1").fetchone()
        if representation == "whitespace":
            payload = f"  {payload}\n"
        elif representation == "utf8_blob":
            payload = payload.encode("utf-8-sig")
        else:
            payload += "{}"
        connection.execute("UPDATE measurements SET payload = ? WHERE id = ?", (payload, mid))
        connection.commit()
    if representation == "extra_record":
        with pytest.raises(json.JSONDecodeError, match="Extra data"):
            Project.load(path)
    else:
        assert Project.load(path).science == project.science


def test_reused_encoder_keeps_field_layouts_separate_and_excludes_transient_fields():
    @dataclass
    class Base:
        value: int
        temporary: InitVar[str] = ""
        label: ClassVar[str] = "not stored"

    @dataclass
    class Child(Base):
        extra: tuple = ()

    class UnhashableType(type):
        __hash__ = None

    @dataclass
    class Custom(metaclass=UnhashableType):
        value: str

    first = make_dataclass("SameName", [("number", int)])
    second = make_dataclass("SameName", [("text", str)])
    child = Child(3, temporary="ignored", extra=("ND", 2**100))
    child.dynamic = "not stored either"
    records = [first(1), second("Лаборатория"), child, Base(2), first(4), second("<0.1")]
    encoder = storage._RecordEncoder(ensure_ascii=False, allow_nan=False)
    for record in records * 2:
        assert encoder.encode(record) == json.dumps(asdict(record), ensure_ascii=False, allow_nan=False)
    assert encoder.encode(Custom("kept")) == json.dumps({"value": "kept"})
    with pytest.raises(TypeError):
        encoder.encode(Base)  # A dataclass type is not a record instance.


def test_measurement_encoding_ignores_transient_fields_and_preserves_field_order(project):
    encoder = storage._RecordEncoder(ensure_ascii=False, allow_nan=False)
    first, second = list(project.science.measurements.values())[:2]
    for measurement in (first, second):
        assert encoder.encode(measurement) == storage.encode(asdict(measurement))
    second.sample_id = vars(second).pop('sample_id')
    assert encoder.encode(second) == storage.encode(asdict(second))
    first.transient = object()
    assert encoder.encode(first) == storage.encode(asdict(first))


def test_reused_encoder_recovers_after_invalid_record_without_retaining_values():
    encoder = storage._RecordEncoder(ensure_ascii=False, allow_nan=False)
    invalid = {"cycle": []}
    invalid["cycle"].append(invalid)
    with pytest.raises(ValueError):
        encoder.encode(invalid)
    with pytest.raises(ValueError):
        encoder.encode(Context("bad", (float("nan"),)))
    valid = Context("good", ("ND", -0.0))
    expected = json.dumps(asdict(valid), ensure_ascii=False, allow_nan=False)
    assert encoder.encode(valid) == expected
    valid.label = "changed"
    assert encoder.encode(valid) == json.dumps(asdict(valid), ensure_ascii=False, allow_nan=False)
    reference = weakref.ref(valid)
    del valid
    gc.collect()
    assert reference() is None


def test_bulk_indexes_preserve_search_columns_and_unique_constraint(project, tmp_path):
    path = tmp_path / "indexes.apyro"
    project.save(path)
    with sqlite3.connect(path) as connection:
        for name, table, columns in (
            ("measurement_sample_property", "measurements", ["sample_id", "property_id"]),
            ("analysis_sample", "analyses", ["sample_id"]),
        ):
            assert [r[2] for r in connection.execute(f"PRAGMA index_info({name})")] == columns
            plan = connection.execute(f"EXPLAIN QUERY PLAN SELECT id FROM {table} WHERE sample_id=?", ("sample",)).fetchall()
            assert any(name in row[3] for row in plan)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert any(row[2] for row in connection.execute("PRAGMA index_list(measurements)"))


def test_index_build_failure_keeps_previous_file_and_binding(project, tmp_path, monkeypatch):
    path = tmp_path / "index-failure.apyro"
    project.save(path)
    before = path.read_bytes()
    digest = project._storage_digest
    connect = sqlite3.connect

    class FailingConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql.strip().startswith("CREATE INDEX analysis_sample"):
                raise sqlite3.OperationalError("index build failed")
            return super().execute(sql, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(storage.sqlite3, "connect", lambda *a, **kw: connect(*a, **{**kw, "factory": FailingConnection}))
        with pytest.raises(sqlite3.OperationalError, match="index build failed"):
            project.save(path)
    assert path.read_bytes() == before
    assert project._storage_digest == digest
    assert not list(tmp_path.glob("*.tmp"))
    project.save(path)
    assert Project.load(path) == project


@pytest.mark.parametrize("location", ["history", "column", "source", "measurement"])
def test_late_encoding_failure_keeps_previous_file_and_allows_retry(project, tmp_path, location):
    path = tmp_path / "atomic.apyro"
    project.save(path)
    original = path.read_bytes()
    previous = Project.load(path)
    if location == "history":
        target = {"action": "late", "details": {}}
        project.history.append(target)
    elif location == "column":
        target = project.columns[-1].metadata
    elif location == "source":
        target = list(project.science.sources.values())[-1].provenance
    else:
        target = {}
        list(project.science.measurements.values())[-1].history.append(target)
    target["unsupported"] = float("nan")
    with pytest.raises(ValueError):
        project.save(path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
    assert Project.load(path) == previous
    del target["unsupported"]
    project.save(path)  # The connection, transaction and write lock were released.
    assert Project.load(path) == project
