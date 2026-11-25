from __future__ import annotations

from threading import Event

from .i18n import tr, message

from PySide6.QtCore import QEventLoop, QThread, QTimer, Signal
from PySide6.QtWidgets import QDialog, QLabel, QMessageBox, QProgressBar, QPushButton, QVBoxLayout
from .theme import icon


class TaskCancelled(Exception):
    pass


class Worker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)
    progress = Signal(str, int)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function

    def run(self):
        try:
            result = self.function()
        except Exception as exc:
            self.failed.emit(message(exc))
        else:
            self.succeeded.emit(result)


class TaskDialog(QDialog):
    project_tasks_finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self._task_pending = False
        self._project_closing = False
        self._task_cancelled = None

    @property
    def task_running(self):
        return self._task_pending or (self.worker is not None and self.worker.isRunning())

    def project_close_blocker(self):
        return self.windowTitle() if self.task_running else None

    def prepare_project_close(self):
        self._project_closing = True
        if self._task_cancelled is not None:
            self._task_cancelled.set()

    def run_task(self, function, callback, *, on_finished=None, disable_dialog=True, cancelled=None, on_error=None):
        if self.task_running or self._project_closing:
            return
        if self.worker is not None:
            self.worker.deleteLater()
        self._task_pending = True
        self._task_cancelled = cancelled
        if disable_dialog:
            self.setEnabled(False)
        worker = self.worker = Worker(function, self)
        outcome = {}
        worker.succeeded.connect(lambda value: outcome.update(result=value))
        worker.failed.connect(lambda error: outcome.update(error=error))

        def finished():
            nonlocal callback
            deliver, callback = callback, None
            received = outcome.copy()
            outcome.clear()
            worker.function = None
            self._task_pending = False
            self._task_cancelled = None
            if disable_dialog:
                self.setEnabled(True)
            try:
                if self._project_closing or (cancelled is not None and cancelled.is_set()):
                    return
                if "error" in received:
                    if on_error is None:
                        QMessageBox.warning(self, tr('Ошибка'), message(received["error"]))
                    else:
                        on_error(received["error"])
                elif "result" in received:
                    deliver(received["result"])
            finally:
                if not self._project_closing and on_finished is not None:
                    on_finished()
                self.project_tasks_finished.emit()

        worker.finished.connect(finished)
        worker.start()

    def reject(self):
        if self.task_running:
            return
        super().reject()

    def closeEvent(self, event):
        if self.task_running:
            event.ignore()
        else:
            super().closeEvent(event)


def run_with_progress(parent, title, function, *, cancellable=False, prepare=None, preparation_label=None, modal=True):
    """Run a worker with an optional GUI-side iterator whose return value is its last argument."""
    dialog = TaskDialog(parent)
    dialog.setWindowTitle(title)
    layout = QVBoxLayout(dialog)
    label = QLabel(title)
    label.setWordWrap(True)
    layout.addWidget(label)
    progress = QProgressBar()
    progress.setRange(0, 0)
    layout.addWidget(progress)
    outcome = {}
    worker = dialog.worker = Worker(function, dialog)
    cancelled = Event()
    preparation_timer = QTimer(dialog)
    dialog._task_pending = True
    if cancellable:
        progress.setRange(0, 100)
        cancel = QPushButton(icon('x'), tr('Отмена'))
        layout.addWidget(cancel)
        last_progress = None

        def request_cancel():
            cancelled.set()
            if preparation_timer.isActive():
                preparation_timer.stop()
                dialog.accept()
                return
            cancel.setEnabled(False)
            label.setText(tr('Отмена после завершения текущего этапа…'))

        def update(stage, percent):
            nonlocal last_progress
            if cancelled.is_set():
                raise TaskCancelled()
            progress_state = (stage, percent)
            if progress_state != last_progress:
                last_progress = progress_state
                worker.progress.emit(stage, percent)

        def display_progress(stage, percent):
            if not cancelled.is_set():
                label.setText(stage)
                progress.setValue(percent)

        cancel.clicked.connect(request_cancel)
        worker.progress.connect(display_progress)
        worker.function = lambda: function(update)
    worker.succeeded.connect(lambda value: outcome.update(result=value))
    worker.failed.connect(lambda error: outcome.update(error=error))
    worker.finished.connect(dialog.accept)
    if prepare is None:
        worker.start()
    else:
        progress.setRange(0, 0)
        label.setText(preparation_label or title)

        def advance_preparation():
            try:
                next(prepare)
            except StopIteration as done:
                preparation_timer.stop()
                prepared = done.value
                if cancellable:
                    progress.setRange(0, 100)
                    worker.function = lambda: function(update, prepared)
                else:
                    worker.function = lambda: function(prepared)
                worker.start()
            except TaskCancelled:
                preparation_timer.stop()
                cancelled.set()
                dialog.accept()
            except Exception as exc:
                preparation_timer.stop()
                outcome['error'] = message(exc)
                dialog.accept()

        preparation_timer.timeout.connect(advance_preparation)
        preparation_timer.start(0)
    if modal:
        dialog.exec()
    else:
        loop = QEventLoop()
        dialog.finished.connect(loop.quit)
        dialog.show()
        loop.exec()
    worker.wait()
    worker.function = None
    dialog.deleteLater()
    if cancelled.is_set():
        raise TaskCancelled()
    if "error" in outcome:
        raise ValueError(outcome["error"])
    return outcome["result"]
