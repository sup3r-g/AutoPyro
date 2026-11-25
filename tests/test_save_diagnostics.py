"""Benchmark observation must retain exceptions, ownership and original callables."""
import gc
import weakref
import json
import sqlite3

import pytest

from autopyro.core import Project
from tools.save_diagnostics import SaveDiagnostics, SaveTimeline
from autopyro.database import storage


def test_nested_spans_do_not_double_count_same_thread_children():
    diagnostic = SaveDiagnostics()
    child = diagnostic.wrap("child", lambda: 7)
    assert diagnostic.wrap("parent", child)() == 7
    spans = diagnostic.summary()["spans"]
    assert spans[1]["parent"] == 0
    assert spans[0]["seconds"] >= spans[1]["seconds"]
    assert spans[0]["exclusive_seconds"] == pytest.approx(spans[0]["seconds"] - spans[1]["seconds"])


def test_patches_restore_after_error_and_preserve_exception():
    original = Project.save
    callbacks = list(gc.callbacks)
    diagnostic = SaveDiagnostics()
    with pytest.raises(RuntimeError, match="QA failure"):
        with diagnostic:
            assert Project.save is not original
            diagnostic.wrap("fail", lambda: (_ for _ in ()).throw(RuntimeError("QA failure")))()
    assert Project.save is original
    assert gc.callbacks == callbacks
    assert diagnostic.summary()["spans"][0]["seconds"] >= 0


def test_weak_observer_does_not_keep_snapshot_alive():
    class Snapshot:
        pass
    diagnostic = SaveDiagnostics()
    value = diagnostic.wrap("copy_project", Snapshot)()
    reference = weakref.ref(value)
    assert not diagnostic.summary()["snapshot_release"]
    del value
    gc.collect()
    assert reference() is None
    event = diagnostic.summary()["snapshot_release"][0]
    assert not event["before_worker_return"]
    assert event["root_finalizer_to_worker_return_seconds"] is None


def test_save_diagnostics_observes_project_session(qtbot, tmp_path):
    from autopyro.ui.window import MainWindow

    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.path = tmp_path / "project.apyro"
    diagnostic = SaveDiagnostics()
    with diagnostic:
        assert owner.save()
    spans = {span["name"] for span in diagnostic.summary()["spans"]}
    assert {"progress_wait", "worker_function", "copy_project", "project_save"} <= spans
    assert Project.load(owner.path).id == owner.model.project.id


def test_sql_diagnostics_preserve_streaming_rows_encoding_and_commit():
    original_connect, original_encode = sqlite3.connect, storage._RecordEncoder.encode
    diagnostic = SaveDiagnostics()
    consumed = []
    with diagnostic:
        connection = sqlite3.connect(":memory:")
        try:
            connection.execute("CREATE TABLE payload (id INTEGER PRIMARY KEY, body TEXT)")
            encode = storage._RecordEncoder(ensure_ascii=False, allow_nan=False).encode
            def rows():
                for i in range(10):
                    consumed.append(i)
                    yield i, encode({"raw": "001", "value": i})
            connection.executemany("INSERT INTO payload VALUES (?,?)", rows())
            connection.execute("CREATE INDEX payload_body ON payload(body)")
            connection.commit()
            assert consumed == list(range(10)) and not connection.in_transaction
            assert [json.loads(body)["value"] for body, in connection.execute("SELECT body FROM payload ORDER BY id")] == list(range(10))
            with pytest.raises(sqlite3.IntegrityError):
                connection.executemany("INSERT INTO payload VALUES (?,?)", [(0, "duplicate")])
        finally:
            connection.close()
    assert sqlite3.connect is original_connect and storage._RecordEncoder.encode is original_encode
    report = diagnostic.summary()
    assert sum(group["calls"] for group in report["record_encoding"]) == 10
    assert any(span["name"] == "sqlite_commit" for span in report["spans"])
    assert any(span["name"] == "secondary_index_build" for span in report["spans"])
    group = report["record_encoding"][0]
    assert report["spans"][group["parent"]]["name"] == "executemany:payload"


@pytest.mark.parametrize("positional", [False, True])
def test_sql_observer_respects_explicit_connection_factory(positional):
    class CustomConnection(sqlite3.Connection):
        pass
    with SaveDiagnostics():
        connection = (sqlite3.connect(":memory:", 5., 0, "", True, CustomConnection)
                      if positional else sqlite3.connect(":memory:", factory=CustomConnection))
        try:
            assert type(connection) is CustomConnection
        finally:
            connection.close()


def test_foreign_key_observer_preserves_violation_rows():
    diagnostic = SaveDiagnostics()
    with diagnostic:
        connection = sqlite3.connect(":memory:")
        try:
            connection.executescript("CREATE TABLE parent(id PRIMARY KEY); CREATE TABLE child(pid REFERENCES parent(id)); INSERT INTO child VALUES (7);")
            rows = connection.execute("PRAGMA foreign_key_check").fetchall()
            assert rows == [("child", 1, "parent", 0)]
        finally:
            connection.close()
    assert [s["name"] for s in diagnostic.spans] == ["foreign_key_check_execute", "foreign_key_check_fetchall"]


def test_timeline_bounds_gaps_and_calculates_overlap(monkeypatch):
    import tools.save_diagnostics as module
    ticks = iter(range(1, 13))
    monkeypatch.setattr(module, "perf_counter", lambda: next(ticks))
    timeline = SaveTimeline()
    timeline.previous = 0
    for _ in range(12):
        timeline.tick()
    assert len(timeline.gaps) == 8
    timeline.gaps = [(4., 1., 5.)]
    timeline.collections = [(2., 4.)]
    report = timeline.summary(1., [{"name": "copy", "start": 1., "end": 3., "thread": 11}])
    gap = report["largest_qt_gaps"][0]
    assert gap["start"] == 0 and gap["end"] == 4
    assert gap["generation2_gc_seconds"] == 2
    assert gap["overlapping_spans"][0]["seconds"] == 2
