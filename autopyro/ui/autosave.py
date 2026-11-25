"""Consistent background recovery copies without blocking the Qt event loop."""

import os
from tempfile import gettempdir
from concurrent.futures import CancelledError
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from uuid import uuid4

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QApplication

from autopyro.database.recovery_writer import RecoverySnapshot, write_recovery
from autopyro.model.core import Project, timestamp
from .i18n import tr
from .tasks import Worker


def recovery_directory():
    return Path(gettempdir()) / "AutoPyro-recovery"


def find_newer_recovery(path, project_id):
    """Load the newest valid recovery for the selected project and path."""
    candidate = _find_recovery(path, project_id, newer_than=Path(path).stat().st_mtime_ns)
    return (candidate.path, candidate.project) if candidate is not None else None


@dataclass(frozen=True)
class RecoveryCandidate:
    path: Path
    project: Project
    modified_ns: int
    identity_verified: bool


def find_recovery_after_open_error(path):
    from autopyro.database.storage import read_project_id

    # Damage or copying the original back can make its mtime newer than the copy.
    return _find_recovery(path, read_project_id(path))


def _find_recovery(path, project_id, *, newer_than=None):
    path = Path(path)
    target = path.resolve()
    prefix = os.path.normcase(path.name + ".autosave-")
    candidates = []
    for candidate in path.parent.glob("*.apyro"):
        if not os.path.normcase(candidate.name).startswith(prefix):
            continue
        try:
            candidate_modified = candidate.stat().st_mtime_ns
        except OSError:
            continue
        if newer_than is None or candidate_modified > newer_than:
            candidates.append((candidate_modified, candidate))

    for modified_ns, candidate in sorted(candidates, reverse=True):
        try:
            project = Project.load(candidate)
            if project_id is not None and project.id != project_id:
                continue
            if restore_recovery_binding(project) == target:
                return RecoveryCandidate(candidate, project, modified_ns, project_id is not None)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            # A newer copy may have been moved, damaged or written for another project.
            continue
    return None


def recovery_metadata(project, target):
    return {
        "target": str(target.resolve()) if target else None,
        "created_at": timestamp(),
        "source_digest": getattr(project, "_storage_digest", None),
    }


def restore_recovery_binding(project):
    recovery = project.view.pop("recovery", None)
    if not isinstance(recovery, dict):
        raise ValueError(tr('Выберите файл автосохранения'))
    target = Path(recovery["target"]) if recovery.get("target") else None
    if target:
        target = target.resolve()
        project._storage_path = target
        # An older recovery may lack a digest. Preserve that missing digest
        # so the restored project cannot silently overwrite a changed target.
        project._storage_digest = recovery.get("source_digest")
    return target


class BackgroundAutosave(QObject):
    failed = Signal(str)
    saved = Signal(int)
    idle = Signal()
    _transfer_finished = Signal(object)

    def __init__(self):
        super().__init__(QApplication.instance())
        self.worker = None
        self.cancelled = Event()
        self._write_done = Event()
        self._write_done.set()
        self.staging = None
        self.snapshot = None
        self._sealed = False
        self._transfer_finished.connect(self._seal_transfer, Qt.ConnectionType.QueuedConnection)
        QApplication.instance().aboutToQuit.connect(self.shutdown)

    def cancel(self):
        self.cancelled.set()
        # Safe to wait in a worker even after Qt disposes the old QThread.
        return self._write_done

    def cancel_transfer(self):
        if self.snapshot is not None and not self._sealed:
            self.cancel()

    def _seal_transfer(self, snapshot):
        # A GUI edit may be between mutation and its changed signal when the
        # worker sends the last batch. Seal only once the GUI can dispatch.
        if self.snapshot is snapshot:
            self._sealed = True

    def cleanup(self):
        if self.staging is not None:
            for path in (self.staging, Path(str(self.staging) + ".lock")):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    # A failed cleanup must not retain a finished worker or
                    # prevent the next recovery attempt after an I/O failure.
                    pass
            self.staging = None

    def start(self, project, path, view, recovery, revision):
        if self.worker is not None:
            return False
        path = Path(path)
        staging = path.with_name(path.name + ".pending-" + uuid4().hex)
        self.staging = staging
        self.snapshot = snapshot = RecoverySnapshot(project, view, recovery)
        self._sealed = False
        self.cancelled = cancelled = Event()
        self._write_done = done = Event()
        outcome = {}

        def write():
            try:
                write_recovery(snapshot, staging, cancelled=cancelled,
                               transferred=lambda: self._transfer_finished.emit(snapshot))
            except (CancelledError, RuntimeError):
                # An edit can resize a collection while a cancelled generation
                # is still transmitting a batch.
                if not cancelled.is_set():
                    raise
            finally:
                done.set()

        worker = self.worker = Worker(write, self)
        worker.failed.connect(lambda error: outcome.update(error=error))

        def finished():
            try:
                # Publication happens on the GUI thread, serialized with edits,
                # explicit saves, project replacement and clearing recovery.
                if not cancelled.is_set():
                    if "error" in outcome:
                        self.failed.emit(outcome["error"])
                    elif staging.exists():
                        os.replace(staging, path)
                        self.saved.emit(revision)
            except OSError as exc:
                self.failed.emit(str(exc))
            finally:
                self.cleanup()
                self.snapshot = None
                self.worker = None
                worker.function = None
                worker.deleteLater()
                self.idle.emit()

        worker.finished.connect(finished)
        worker.start()
        return True

    def shutdown(self):
        self.cancel()
        if self.worker is not None:
            self.worker.wait()
        self.cleanup()
