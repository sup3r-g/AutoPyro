"""Transfer a recovery snapshot to a private SQLite writer process."""

from concurrent.futures import CancelledError
from copy import deepcopy
from dataclasses import fields
from itertools import batched, islice
import os
from pathlib import Path
import pickle
import subprocess
from subprocess import Popen
import sys
from tempfile import TemporaryDirectory, TemporaryFile
from threading import Lock
from time import perf_counter, sleep

from autopyro.model.core import Column, Project, Row
from autopyro.model.measurements import (
    Analysis, Measurement, MeasurementStore, PropertyDefinition, Sample, SourceRecord, snapshot_record,
)


_STORE_RECORD_TYPES = {
    "samples": Sample, "properties": PropertyDefinition, "sources": SourceRecord,
    "analyses": Analysis, "measurements": Measurement,
}


def _check_cancelled(cancelled):
    if cancelled.is_set():
        raise CancelledError()


class RecoverySnapshot:
    """Source records as of dispatch, retaining old values only for concurrent edits."""

    def __init__(self, project, view, recovery):
        self.project = project
        self.view = {**project.view, "current": view, "recovery": recovery}
        self.history_size = len(project.history)
        self.measurement_count = len(project.science.measurements)
        self.working_count = len(project.science.working)
        self.rows = {}
        self.measurements = {}
        self.working = {}
        self.columns = None
        self.header_sent = False
        self.transferred = False
        self.lock = Lock()

    def preserve_cells(self, rows, changes):
        """Called by the GUI before a row-local edit or working-repeat change."""
        project = self.project
        store = project.science
        with self.lock:
            if self.transferred:
                return True
            # These calculations can rewrite rows outside the edit's scope.
            if any(c.metadata.get("nonpyro") or c.metadata.get("stratigraphy") for c in project.columns):
                return False
            if any(row.id not in store.samples for row in rows):
                return False
            if not self.header_sent and self.columns is None:
                self.columns = deepcopy(project.columns)
                self.view = deepcopy(self.view)
            for row in rows:
                if row.id not in self.rows:
                    self.rows[row.id] = deepcopy(row)
                if row.id not in self.working:
                    self.working[row.id] = deepcopy(store.working.get(row.id))
            if changes is None:
                changes = ((row.id, cid) for row in rows for cid in store.working.get(row.id, {}))
            for row_id, column_id in changes:
                measurement = store.current(row_id, column_id)
                if measurement is not None and measurement.id not in self.measurements:
                    self.measurements[measurement.id] = snapshot_record(measurement)
            return True


def _send_project(stream, snapshot, cancelled):
    # Only the private child reads this pipe. Each record batch has its own
    # pickle memo, so Qt's process never builds a second measurement store.
    project = snapshot.project
    pickle.dump(project.id, stream, protocol=pickle.HIGHEST_PROTOCOL)

    def record_values(records, record_type):
        # Pickling the dataclasses directly materializes each live record's
        # instance dict. Stream the same declared fields as the SQLite writer.
        names = tuple(item.name for item in fields(record_type))
        for record in records:
            yield tuple(getattr(record, name) for name in names)

    def store_values(items, record_type):
        names = tuple(item.name for item in fields(record_type))
        for key, record in items:
            yield (key, *(getattr(record, name) for name in names))

    def original_items(collection, previous, count):
        # Table edits can append records. Read only keys that existed at dispatch.
        keys = tuple(collection)
        for key in islice(keys, count):
            yield key, previous.get(key, collection[key])

    store = project.science
    collections = (
        record_values((snapshot.rows.get(row.id, row) for row in project.rows), Row),
        islice(project.history, snapshot.history_size),
        project.source.items(),
        *(store_values(getattr(store, name).items(), record_type)
          for name, record_type in _STORE_RECORD_TYPES.items() if name != "measurements"),
        store_values(original_items(store.measurements, snapshot.measurements, snapshot.measurement_count), Measurement),
        original_items(store.working, snapshot.working, snapshot.working_count),
    )
    next_yield = perf_counter() + .004
    # Resolve headers under the same lock as the GUI preservation hook.
    with snapshot.lock:
        columns = snapshot.columns if snapshot.columns is not None else project.columns
        header = (
            pickle.dumps(tuple(record_values(columns, Column)), protocol=pickle.HIGHEST_PROTOCOL),
            pickle.dumps(tuple(snapshot.view.items()), protocol=pickle.HIGHEST_PROTOCOL),
        )
        snapshot.header_sent = True
    for payload in header:
        stream.write(payload)
        pickle.dump(None, stream, protocol=pickle.HIGHEST_PROTOCOL)
    for records in collections:
        batches = batched(records, 128)
        while True:
            _check_cancelled(cancelled)
            with snapshot.lock:
                batch = next(batches, None)
                if batch is None:
                    break
                payload = pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL)
            stream.write(payload)
            if perf_counter() >= next_yield:
                sleep(0.001)
                next_yield = perf_counter() + .004
        pickle.dump(None, stream, protocol=pickle.HIGHEST_PROTOCOL)
    snapshot.transferred = True


def _receive_project(stream):
    project_id = pickle.load(stream)

    def records():
        while (batch := pickle.load(stream)) is not None:
            yield from batch

    columns = [Column(*values) for values in records()]
    view = dict(records())
    rows = [Row(*values) for values in records()]
    history, source = list(records()), dict(records())
    collections = {
        name: {key: record_type(*values) for key, *values in records()}
        for name, record_type in _STORE_RECORD_TYPES.items()
    }
    science = MeasurementStore(**collections, working=dict(records()))
    return Project(id=project_id, columns=columns, rows=rows, history=history,
                   view=view, source=source, science=science)


def write_recovery(snapshot, staging, *, cancelled, transferred):
    """Prepare a staged copy. The GUI decides whether to publish it."""
    _check_cancelled(cancelled)
    staging.parent.mkdir(parents=True, exist_ok=True)
    # If cancelled during SQLite work, the child and all its temporary files
    # are disposed of together without touching the last published recovery.
    with TemporaryDirectory(prefix=".autopyro-recovery-", dir=staging.parent) as directory:
        output = Path(directory) / "recovery.apyro"
        with TemporaryFile(dir=directory) as stderr:
            entry = (["--write-recovery"] if "__compiled__" in globals()
                     else ["-m", "autopyro.database.recovery_writer"])
            with Popen(
                [sys.executable, *entry, str(output)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ) as process:
                try:
                    try:
                        with process.stdin as stream:
                            _send_project(stream, snapshot, cancelled)
                        transferred()
                    except BrokenPipeError:
                        pass  # Read the child error after it exits.
                    while process.poll() is None:
                        if cancelled.wait(0.02):
                            raise CancelledError()
                    _check_cancelled(cancelled)
                    if process.returncode:
                        stderr.seek(0)
                        detail = stderr.read().decode("utf-8", errors="replace").strip()
                        raise OSError(detail or f"Recovery writer exited with code {process.returncode}")
                    os.replace(output, staging)
                finally:
                    if process.poll() is None:
                        process.kill()


def writer_main(output):
    try:
        _receive_project(sys.stdin.buffer).save_recovery(Path(output))
    except Exception as exc:
        sys.stderr.buffer.write(str(exc).encode("utf-8", errors="replace"))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(writer_main(sys.argv[1]))
