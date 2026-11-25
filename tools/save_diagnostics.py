"""Benchmark-only observation of existing save operations; no writer replacement."""
from contextlib import ExitStack
from functools import wraps
from threading import get_ident, local, Lock
from time import perf_counter
from unittest.mock import patch
import weakref
import gc
import heapq

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from autopyro import snapshots
from autopyro.core import Project
from autopyro.database import storage
from autopyro.ui import autosave, project_session, tasks
from autopyro.ui.table import project_table


class SaveTimeline:
    """Bounded Qt gaps and generation-2 GC intervals, without changing GC policy."""
    def __init__(self):
        self.gaps = []
        self.collections = []
        self.gc_start = None
        self.timer = None
        self.previous = None

    def tick(self):
        now = perf_counter()
        if self.previous is not None:
            heapq.heappush(self.gaps, (now - self.previous, self.previous, now))
            if len(self.gaps) > 8:
                heapq.heappop(self.gaps)
        self.previous = now

    def observe_gc(self, phase, info):
        if info["generation"] != 2:
            return
        if phase == "start":
            self.gc_start = perf_counter()
        elif self.gc_start is not None:
            self.collections.append((self.gc_start, perf_counter()))
            self.gc_start = None

    def start(self):
        gc.callbacks.append(self.observe_gc)
        if QApplication.instance() is not None:
            self.timer = QTimer()
            self.timer.setInterval(10)
            self.timer.timeout.connect(self.tick)
            self.previous = perf_counter()
            self.timer.start()

    def stop(self):
        if self.timer is not None:
            self.tick()
            self.timer.stop()
        gc.callbacks.remove(self.observe_gc)

    def summary(self, origin, spans):
        gaps = []
        for duration, start, end in sorted(self.gaps, reverse=True):
            overlapping = [{"span": i, "name": s["name"], "thread": s["thread"],
                            "seconds": min(end, s["end"]) - max(start, s["start"])}
                           for i, s in enumerate(spans) if s["start"] < end and s["end"] > start]
            gc_seconds = sum(max(0., min(end, b) - max(start, a)) for a, b in self.collections)
            gaps.append({"start": start - origin, "end": end - origin, "seconds": duration,
                         "overlapping_spans": overlapping, "generation2_gc_seconds": gc_seconds})
        return {"largest_qt_gaps": gaps, "generation2_gc": [
            {"start": a - origin, "end": b - origin, "seconds": b - a} for a, b in self.collections],
            "note": "Overlap is temporal, not proof of causation. Nested spans overlap each other. "
                    "Timer covers the save context, not queued events after it. Only generation 2 GC is recorded."}

class SaveDiagnostics:
    def __init__(self):
        self.spans = []
        self.finalized = []
        self.threads = local()
        self.patches = ExitStack()
        self.guard = Lock()
        self.encodings = []
        self.timeline = SaveTimeline()

    def encoder(self, function):
        @wraps(function)
        def observed(*args, **kwargs):
            groups = getattr(self.threads, "encodings", None)
            if groups is None:
                groups = self.threads.encodings = {}
            stack = getattr(self.threads, "stack", [])
            parent = stack[-1] if stack else None
            group = groups.get(parent)
            if group is None:
                group = groups[parent] = {"thread": get_ident(), "parent": parent, "calls": 0, "seconds": 0.}
                with self.guard:
                    self.encodings.append(group)
            started = perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                group["seconds"] += perf_counter() - started
                group["calls"] += 1
        return observed

    def connect(self, function):
        diagnostic = self
        base = storage.sqlite3.Connection
        class ObservedConnection(base):
            def execute(self, sql, *args, **kwargs):
                if sql.strip().rstrip(";").casefold() == "pragma foreign_key_check":
                    class CheckCursor(storage.sqlite3.Cursor):
                        def execute(self, *a, **kw):
                            return diagnostic.wrap("foreign_key_check_execute", super().execute)(*a, **kw)

                        def fetchall(self):
                            return diagnostic.wrap("foreign_key_check_fetchall", super().fetchall)()
                    return self.cursor(factory=CheckCursor).execute(sql, *args, **kwargs)
                if sql.lstrip().upper().startswith("CREATE INDEX "):
                    return diagnostic.wrap("secondary_index_build", super().execute)(sql, *args, **kwargs)
                return super().execute(sql, *args, **kwargs)

            def executemany(self, sql, records):
                words = sql.split()
                table = words[2] if len(words) > 2 and words[:2] == ["INSERT", "INTO"] else "other"
                return diagnostic.wrap("executemany:" + table, super().executemany)(sql, records)

            def commit(self):
                return diagnostic.wrap("sqlite_commit", super().commit)()

        @wraps(function)
        def observed(*args, **kwargs):
            if len(args) < 6:  # Respect an explicit positional factory as well.
                kwargs.setdefault("factory", ObservedConnection)
            return function(*args, **kwargs)
        return observed

    def wrap(self, name, function):
        @wraps(function)
        def observed(*args, **kwargs):
            stack = getattr(self.threads, "stack", None)
            if stack is None:
                stack = self.threads.stack = []
            span = {"name": name, "thread": get_ident(), "start": perf_counter(),
                    "parent": stack[-1] if stack else None, "child_seconds": 0.}
            with self.guard:
                index = len(self.spans)
                self.spans.append(span)
            stack.append(index)
            try:
                value = function(*args, **kwargs)
                if name == "copy_project":
                    # A weak observer does not keep the large snapshot alive.
                    weakref.finalize(value, self.snapshot_finalized, get_ident(), stack[0])
                return value
            finally:
                span["end"] = perf_counter()
                span["seconds"] = span["end"] - span["start"]
                stack.pop()
                if stack:
                    self.spans[stack[-1]]["child_seconds"] += span["seconds"]
        return observed

    def snapshot_finalized(self, thread, root_span):
        self.finalized.append({"thread": thread, "root_span": root_span, "at": perf_counter()})

    def progress(self, function):
        @wraps(function)
        def observed(parent, title, task, **kwargs):
            return function(parent, title, self.wrap("worker_function", task), **kwargs)
        return self.wrap("progress_wait", observed)

    def __enter__(self):
        self.timeline.start()
        targets = [(Project, "sync_measurements", "sync_measurements"),
                   (Project, "validate", "validate"), (Project, "save", "project_save"),
                   (project_session, "copy_project", "copy_project"),
                   (autosave, "write_recovery", "recovery_writer"),
                   (tasks.Worker, "run", "qt_worker_run"),
                   (tasks.TaskDialog, "exec", "progress_dialog_exec"),
                   (tasks.TaskDialog, "showEvent", "progress_dialog_show"),
                   (Project, "save_recovery", "project_save_recovery"),
                   (snapshots, "_measurement_memo", "copy_measurements"),
                   (snapshots, "_plain_measurements", "select_copy_measurements"),
                   (snapshots, "_allocate_measurement_copies", "allocate_measurement_copies"),
                   (snapshots, "_table_memo", "copy_table_mappings"),
                   (storage, "_write_database", "sqlite_write"),
                   (storage, "file_digest", "file_digest"),
                   (storage, "_backup_previous", "backup_previous"),
                   (storage.os, "fsync", "fsync"), (storage.os, "replace", "atomic_replace")]
        try:
            for owner, attribute, name in targets:
                self.patches.enter_context(patch.object(owner, attribute, self.wrap(name, getattr(owner, attribute))))
            for module in (project_table, project_session):
                self.patches.enter_context(patch.object(module, "run_with_progress", self.progress(module.run_with_progress)))
            self.patches.enter_context(patch.object(storage.sqlite3, "connect", self.connect(storage.sqlite3.connect)))
            self.patches.enter_context(patch.object(storage._RecordEncoder, "encode", self.encoder(storage._RecordEncoder.encode)))
        except BaseException:
            self.patches.close()
            self.timeline.stop()
            raise
        return self

    def __exit__(self, *error):
        try:
            self.patches.close()
        finally:
            self.timeline.stop()

    def summary(self):
        origin = min((s["start"] for s in self.spans), default=0.)
        spans = [{**s, "start": s["start"] - origin, "end": s["end"] - origin,
                  "exclusive_seconds": max(0., s["seconds"] - s["child_seconds"])} for s in self.spans]
        release = []
        for event in self.finalized:
            root = self.spans[event["root_span"]]
            release.append({"thread": event["thread"], "at": event["at"] - origin,
                            "before_worker_return": event["at"] <= root["end"],
                            "root_finalizer_to_worker_return_seconds":
                                root["end"] - event["at"] if event["at"] <= root["end"] else None})
        return {"spans": spans, "snapshot_release": release, "record_encoding": self.encodings,
                "timeline": self.timeline.summary(origin, self.spans),
                "note": "Wall times; nested spans overlap. Exclusive excludes only same-thread children. "
                        "sqlite_write includes encoding, SQL and commit. Root finalization is a release boundary, "
                        "not a deep destructor profiler; missing events mean release was not observed. "
                        "Encoding is aggregated per thread/parent, not one span per record. "
                        "executemany includes lazy generator encoding; subtracting encoding still includes generator/observer overhead. "
                        "No GC policy change, no retained snapshot."}
