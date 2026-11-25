"""Project file lifecycle and recovery coordinated independently of window layout."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import Event

from PySide6.QtCore import QObject, QElapsedTimer, QTimer
from PySide6.QtWidgets import QFileDialog, QMessageBox

from autopyro.model.core import Project, new_id
from autopyro.model.snapshots import copy_project
from .autosave import (
    BackgroundAutosave, find_newer_recovery, find_recovery_after_open_error, recovery_directory,
    recovery_metadata, restore_recovery_binding,
)
from .i18n import message, tr
from .table.model import prepare_project
from .table.snapshot_policy import large_snapshot
from .tasks import run_with_progress


class ProjectSession(QObject):
    """Own file state and timers; let the window bind projects and capture views."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.path = None
        self.dirty = False
        self.session_id = new_id()
        self.recovered_from = None
        self._damaged_source = None
        self._save_cancelled = None
        self._background_autosave = None
        self._autosaved_revision = None
        self._autosave_again = False
        self._autosave_enabled = True
        self._preserved_edit = False
        self._autosave_deferred = QTimer(self)
        self._autosave_deferred.setSingleShot(True)
        self._autosave_deferred.setInterval(100)
        self._autosave_deferred.timeout.connect(window.autosave)
        self._autosave_wait = QElapsedTimer()
        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(window.autosave)

    def cancel_pending(self):
        """Prevent pending snapshots from acknowledging a newer project state."""
        if self._save_cancelled is not None:
            self._save_cancelled.set()
        if self._background_autosave is not None:
            self._background_autosave.cancel()
        self._autosave_again = False
        self._autosaved_revision = None

    def prepare_replacement(self):
        self._stop_autosave_deferral()
        self.cancel_pending()
        self._damaged_source = None

    def mark_changed(self):
        if self._save_cancelled is not None:
            self._save_cancelled.set()
        job = self._background_autosave
        if job is not None and job.worker is not None:
            self._autosave_again = True
            if not self._preserved_edit:
                job.cancel_transfer()
        self._autosaved_revision = None
        self.dirty = True

    @contextmanager
    def preserve_recovery_edit(self, rows, changes):
        job = self._background_autosave
        snapshot = job.snapshot if job is not None and not job.cancelled.is_set() else None
        if snapshot is not None and not snapshot.preserve_cells(rows, changes):
            job.cancel_transfer()
        previous = self._preserved_edit
        self._preserved_edit = snapshot is not None
        try:
            yield
        finally:
            self._preserved_edit = previous

    def configure_autosave(self, enabled, minutes):
        self._autosave_enabled = enabled
        self.autosave_timer.setInterval(minutes * 60_000)
        if enabled:
            self.autosave_timer.start()
        else:
            self.autosave_timer.stop()
            self.stop()

    def stop(self):
        self._stop_autosave_deferral()
        self._autosave_again = False
        if self._background_autosave is not None:
            self._background_autosave.cancel()

    def confirm_documents_idle(self, title):
        if title is None:
            return True
        QMessageBox.information(
            self.window,
            title,
            tr('Дождитесь завершения расчёта перед сменой проекта или выходом.'),
        )
        return False

    def may_replace(self):
        if not self.confirm_documents_idle(self.window.analysis_documents.project_close_blocker()):
            return False
        if self.window.stratigraphy_window is not None and self.window.stratigraphy_window.has_unsaved_draft():
            response = QMessageBox.question(
                self.window, tr('Несохранённый черновик стратиграфии'),
                tr('Черновик стратиграфии содержит несохранённые правки. Сохранить схему в проекте?'),
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if response == QMessageBox.StandardButton.Save:
                if not self.window.stratigraphy_window.save_scheme():
                    self.window.show_stratigraphy()
                    return False
                return self.window.save()
            if response != QMessageBox.StandardButton.Discard:
                return False
        if not self.dirty:
            return True
        response = QMessageBox.question(
            self.window,
            tr('Несохранённые изменения'),
            tr('Сохранить текущий проект?'),
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if response == QMessageBox.StandardButton.Cancel:
            return False
        if response == QMessageBox.StandardButton.Save:
            return self.window.save()
        return True

    def new_project(self):
        if self.window.may_replace():
            self.window.set_project(Project())

    def load_demo(self):
        if not self.window.may_replace():
            return
        from autopyro.data_io.io import build_project

        rows = [
            ["A", "Shale", "1000", "1.2", "2.4", "435"],
            ["A", "Shale", "1010", "1.5", None, "438"],
            ["A", "Shale", "1020", "2.0", "6.0", "441"],
            ["B", "Shale", "1000", "0.8", "1.0", "430"],
            ["B", "Shale", "1010", "<0.1", "0.3", "ND"],
        ]
        headers = ["Скважина", "Литология", "MD", "TOC", "S2", "Tmax"]
        settings = [
            (
                i,
                name,
                "number" if i in (2, 4) else "text",
                {2: "m", 3: "%wt", 4: "mg HC/g rock", 5: "°C"}.get(i, ""),
            )
            for i, name in enumerate(headers)
        ]
        self.window.set_project(
            build_project(
                headers, rows, list(range(2, 7)), settings, {"type": "synthetic demo"}
            )
        )
        self.window.model_changed()

    def open_project(self):
        if not self.window.may_replace():
            return
        path, _ = QFileDialog.getOpenFileName(
            self.window, tr('Открыть проект'), "", "AutoPyro (*.apyro)"
        )
        if path:
            try:
                def read():
                    try:
                        project = Project.load(path)
                    except (OSError, ValueError, AttributeError, KeyError, TypeError) as exc:
                        return None, find_recovery_after_open_error(path), message(exc)
                    return prepare_project(project), find_newer_recovery(path, project.id), None

                project, recovery, error = run_with_progress(self.window, tr("Открытие проекта…"), read)
                if project is None:
                    self._offer_damaged_project_recovery(Path(path), recovery, error)
                    return
                if recovery is not None:
                    response = QMessageBox.question(self.window, tr('Автосохранение'), tr('Найдена более новая копия. Восстановить её?'),
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
                    if response == QMessageBox.StandardButton.Cancel:
                        return
                    if response == QMessageBox.StandardButton.Yes:
                        recovery_path, recovered = recovery
                        loaded = run_with_progress(self.window, tr("Открытие проекта…"),
                                                   lambda: prepare_project(recovered))
                        self._apply_recovery(loaded, recovered._storage_path, recovery_path)
                        return
                self.window.set_project(project, path, prepared=True)
            except Exception as exc:
                QMessageBox.warning(self.window, tr('Открытие'), message(exc))

    def _offer_damaged_project_recovery(self, path, candidate, error):
        if candidate is None:
            QMessageBox.warning(self.window, tr('Открытие'), error)
            return
        modified = datetime.fromtimestamp(candidate.modified_ns // 1_000_000_000)
        text = tr('Не удалось открыть проект «{0}».\n{1}\n\nАвтокопия: {2}\nИзменена: {3}',
                  str(path), error, str(candidate.path), modified.isoformat(sep=" "))
        if not candidate.identity_verified:
            text += "\n\n" + tr('Не удалось прочитать UUID исходного проекта. Найденная копия совпадает по имени и записанному пути.')
        text += "\n\n" + tr('Открыть копию как отдельный проект? При сохранении выберите новое имя.')
        response = QMessageBox.question(self.window, tr('Восстановление'), text,
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                                        QMessageBox.StandardButton.Cancel)
        if response == QMessageBox.StandardButton.Yes:
            project = run_with_progress(self.window, tr("Открытие проекта…"),
                                        lambda: prepare_project(candidate.project))
            project._storage_path = project._storage_digest = None
            self._apply_recovery(project, None, candidate.path)
            self._damaged_source = path.resolve()

    def save(self, checked=False, as_new=False):
        if self._save_cancelled is not None:
            return False
        path = self.path
        if as_new or path is None:
            suggested = path or "project.apyro"
            if self._damaged_source is not None and path is None:
                suggested = self._damaged_source.with_stem(self._damaged_source.stem + "-recovered")
            value, _ = QFileDialog.getSaveFileName(
                self.window,
                tr('Сохранить проект'),
                str(suggested),
                "AutoPyro (*.apyro)",
            )
            if not value:
                return False
            path = Path(value)
            if path.suffix.lower() != ".apyro":
                # The file dialog only confirmed the name entered by the user.
                path = path.with_suffix(".apyro")
                destination = path.resolve()
                if destination.exists() and QMessageBox.question(
                    self.window, tr('Сохранить проект'), tr('Заменить файл «{0}»?', str(destination)),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                ) != QMessageBox.StandardButton.Yes:
                    return False
        project = self.window.model.project
        cancelled = self._save_cancelled = Event()
        self._autosave_again = False
        try:
            if self._damaged_source is not None:
                for protected in (self._damaged_source, self.recovered_from):
                    if path.resolve() == protected.resolve() or (
                            path.exists() and protected.exists() and path.samefile(protected)):
                        raise ValueError(tr('Выберите новое имя, чтобы сохранить исходный проект и автокопию.'))
            pending = self._background_autosave.cancel() if self._background_autosave is not None else None
            if pending is not None and not pending.is_set():
                run_with_progress(self.window, tr("Подготовка снимка данных…"), pending.wait)
            if cancelled.is_set():
                raise ValueError(tr("Данные изменились во время подготовки снимка. Повторите операцию"))
            # Preserve existing measurement objects and the legacy API bridge.
            # This short synchronization stays on the GUI thread; the expensive
            # copy, validation, encoding, fsync and replacement use a worker.
            project.sync_measurements()
            project.view["current"] = deepcopy(self.window.capture_view(include_sample_set=True))

            def write():
                snapshot = copy_project(project)
                if cancelled.is_set():
                    raise ValueError(tr("Данные изменились во время подготовки снимка. Повторите операцию"))
                snapshot.save(path)
                return snapshot._storage_path, snapshot._storage_digest

            saved_path, saved_digest = run_with_progress(self.window, tr('Сохранение'), write)
            project._storage_path, project._storage_digest = saved_path, saved_digest
            if self.window.model.project is not project:
                return False
            # A coherent snapshot may finish writing after a timer changed the
            # live project. Adopt its file binding, but keep newer edits dirty.
            if not cancelled.is_set():
                self.clear_autosave(preserve=path)
            self.path, self.dirty = path, cancelled.is_set()
            self.window.update_status()
            return not self.dirty
        except Exception as exc:
            QMessageBox.warning(self.window, tr('Сохранение'), message(exc))
            return False
        finally:
            self._save_cancelled = None

    def autosave_path(self):
        if self.path:
            return self.path.with_name(self.path.name + f".autosave-{self.session_id}.apyro")
        return recovery_directory() / f"{self.window.model.project.id}-{self.session_id}.apyro"

    def _stop_autosave_deferral(self):
        self._autosave_deferred.stop()
        self._autosave_wait.invalidate()

    def autosave(self):
        if not self.dirty or self._save_cancelled is not None:
            self._stop_autosave_deferral()
            return
        path = self.window.autosave_path()
        if self._autosaved_revision == self.window.analysis_revision and path.exists():
            self._stop_autosave_deferral()
            return
        if self._background_autosave is None:
            self._background_autosave = BackgroundAutosave()
            self._background_autosave.failed.connect(self.autosave_failed)
            self._background_autosave.saved.connect(self.autosave_saved)
            self._background_autosave.idle.connect(self.autosave_idle)
        if self._background_autosave.worker is not None:
            return
        project = self.window.model.project
        plot = self.window.plot_window
        if (large_snapshot(project) and plot is not None and plot.isVisible() and not plot.closed
                and (plot.timer.isActive() or plot.capture_request is not None
                     or plot.preparation.worker is not None or plot.preview.busy)):
            # Give an interactive frame a head start, with a fixed deadline
            # so repeated edits cannot indefinitely postpone recovery.
            if not self._autosave_wait.isValid():
                self._autosave_wait.start()
            if self._autosave_wait.elapsed() < 5000:
                self._autosave_deferred.start()
                return
        self._stop_autosave_deferral()
        try:
            self._background_autosave.start(
                project, path, deepcopy(self.window.capture_view(include_sample_set=True)),
                recovery_metadata(project, self.path), self.window.analysis_revision,
            )
        except Exception as exc:
            self.autosave_failed(message(exc))

    def autosave_failed(self, error):
        self._autosave_again = False
        self.window.statusBar().showMessage(tr('Не удалось выполнить автосохранение: {0}', error))

    def autosave_saved(self, revision):
        if not self._autosave_again:
            self._autosaved_revision = revision

    def autosave_idle(self):
        again, self._autosave_again = self._autosave_again, False
        if again and self._autosave_enabled and self.dirty and self._save_cancelled is None:
            self._autosave_deferred.start()

    def clear_autosave(self, *, preserve=None):
        self._stop_autosave_deferral()
        self._autosave_again = False
        if self._background_autosave is not None:
            self._background_autosave.cancel()
        self._autosaved_revision = None
        self.remove_recovery_files(self.recovery_files(), preserve=(preserve,))
        self.recovered_from = None
        self._damaged_source = None

    def recovery_files(self):
        path = self.window.autosave_path()
        # A damaged project's recovery copy remains available for inspection.
        return (path,) if self._damaged_source is not None else (path, self.recovered_from)

    def remove_recovery_files(self, paths, *, preserve=()):
        try:
            protected = {Path(path).resolve() for path in preserve if path is not None}
            for path in paths:
                if path is not None and path.resolve() not in protected:
                    path.unlink(missing_ok=True)
        except OSError as exc:
            self.window.statusBar().showMessage(tr('Не удалось удалить копию восстановления: {0}', message(exc)))

    def load_autosave(self, path):
        project = run_with_progress(self.window, tr("Открытие проекта…"), lambda: prepare_project(Project.load(path)))
        target = restore_recovery_binding(project)
        self._apply_recovery(project, target, path)

    def _apply_recovery(self, project, target, path):
        self.window.set_project(project, target, prepared=True, recovery_path=Path(path))
        self.window.model_changed()

    def restore_autosave(self):
        if not self.window.may_replace():
            return
        directory = str(self.path.parent if self.path else recovery_directory())
        path, _ = QFileDialog.getOpenFileName(self.window, tr('Восстановить автосохранение'), directory, "AutoPyro (*.apyro)")
        if path:
            try:
                self.load_autosave(path)
            except Exception as exc:
                QMessageBox.warning(self.window, tr('Восстановление'), message(exc))

