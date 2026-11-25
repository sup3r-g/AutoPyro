"""Bounded cache and latest-request scheduling for interactive joint bands."""

from collections import OrderedDict

from PySide6.QtCore import QObject, QThreadPool, Signal, Slot


class _BandWorkerSignals(QObject):
    finished = Signal(object)


class JointBandCache(QObject):
    ready = Signal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cache = OrderedDict()
        self._generation = 0
        self._pending = None
        self._running = None

    def clear(self):
        self._generation += 1
        self._cache.clear()
        self._pending = None

    def discard_pending(self):
        self._pending = None

    def request(self, key, compute):
        if key in self._cache:
            self._cache.move_to_end(key)
            self._pending = None
            return self._cache[key]
        token = self._generation, key
        self._pending = None if token == self._running else (token, compute)
        self._start_pending()
        return None

    def _start_pending(self):
        if self._running is not None or self._pending is None:
            return
        token, compute = self._pending
        self._pending = None
        self._running = token
        signals = _BandWorkerSignals()
        signals.finished.connect(self._finish)

        def run():
            try:
                outcome = compute()
            except Exception as exc:
                outcome = str(exc)
            signals.finished.emit((token, outcome))

        QThreadPool.globalInstance().start(run)

    @Slot(object)
    def _finish(self, payload):
        (generation, key), outcome = payload
        self._running = None
        if generation == self._generation:
            self._cache[key] = outcome
            if len(self._cache) > 32:
                self._cache.popitem(last=False)
            self.ready.emit(key, outcome)
        self._start_pending()
