"""Coalesced preparation jobs; the latest data revision wins."""
from threading import Event
from time import perf_counter

from PySide6.QtCore import QObject, Signal, QTimer
from PySide6.QtWidgets import QApplication

from autopyro.visualization.plot_data import PlotDataCache, PreparationCancelled, capture_steps
from .tasks import Worker


class InputCapture(QObject):
    ready = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, parent):
        super().__init__(parent)
        self.timer = QTimer(self)
        self.timer.setInterval(0)
        self.timer.timeout.connect(self._step)
        self.steps = None
        self.generation = 0

    def start(self, generation, project, ids, specs, token, previous=None):
        self.cancel()
        self.generation = generation
        self.steps = capture_steps(project, ids, specs, token, previous)
        self.timer.start()

    def cancel(self):
        self.timer.stop()
        self.steps = None

    def _step(self):
        deadline = perf_counter() + .006
        try:
            while True:
                next(self.steps)
                if perf_counter() >= deadline:
                    break
        except StopIteration as finished:
            self.cancel()
            self.ready.emit(self.generation, finished.value)
        except Exception as exc:
            self.cancel()
            self.failed.emit(self.generation, str(exc))


class PlotPreparation(QObject):
    ready = Signal(int, object)
    failed = Signal(int, str)
    idle = Signal()

    def __init__(self):
        super().__init__(QApplication.instance())
        self.cache = PlotDataCache()
        self.worker = None
        self.cancelled = Event()
        self.pending = None
        self.stopped = False
        self.capture = InputCapture(self)
        QApplication.instance().aboutToQuit.connect(self.shutdown)

    def request(self, generation, source, ids, spec):
        self.cancelled.set()
        self.pending = generation, source, ids, spec
        if self.worker is None:
            self._start()

    def invalidate(self):
        self.cancelled.set()
        self.pending = None
        self.capture.cancel()

    def release(self):
        self.invalidate()
        self.cache = PlotDataCache(self.cache.max_bytes)

    def _start(self):
        if self.pending is None or self.stopped:
            return
        generation, source, ids, spec = self.pending
        self.pending = None
        self.cancelled = cancelled = Event()
        cache = self.cache.fork()
        def prepare():
            try:
                return cache.prepare(source, ids, spec, cancelled), cache
            except PreparationCancelled:
                return None
        worker = self.worker = Worker(prepare, self)
        outcome = {}
        worker.succeeded.connect(lambda value: outcome.update(result=value))
        worker.failed.connect(lambda error: outcome.update(error=error))
        def finished():
            self.worker = None
            worker.deleteLater()
            if not cancelled.is_set() and not self.stopped:
                if outcome.get('result') is not None:
                    prepared, self.cache = outcome['result']
                    self.ready.emit(generation, prepared)
                elif 'error' in outcome:
                    self.failed.emit(generation, outcome['error'])
            outcome.clear()
            worker.function = None
            self._start()
            if self.worker is None and self.capture.steps is None:
                self.idle.emit()
        worker.finished.connect(finished)
        worker.start()

    def shutdown(self):
        self.stopped = True
        self.invalidate()
        if self.worker is not None:
            self.worker.wait()
