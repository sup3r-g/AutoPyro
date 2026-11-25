"""A persistent preview process; Qt only installs completed frames."""
import multiprocessing
from io import BytesIO
from queue import Queue, Empty
from time import perf_counter, sleep

from matplotlib.backend_bases import DrawEvent, FigureCanvasBase
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QApplication

from autopyro.visualization.plot_preview import (preview_process, preview_input, send_preview_message, receive_preview_response,
                            restore_preview_membership)


class _CooperativeBuffer(BytesIO):
    """Pickle frames yield the GIL periodically instead of starving Qt timers."""
    def __init__(self, initial=b''):
        super().__init__(initial)
        self.last_yield = perf_counter()

    def _yield(self):
        if perf_counter() - self.last_yield >= .004:
            sleep(0)
            self.last_yield = perf_counter()

    def write(self, data):
        self._yield()
        return super().write(data)

    def read(self, size=-1):
        self._yield()
        return super().read(size)


def _preparation_key(prepared):
    # The revision/QC/layer token describes immutable values. Retain the key,
    # not the PreparedPlot with its DataFrames and full row snapshot.
    return (prepared.source.token, prepared.scope, prepared.policy,
            frozenset(prepared.parsed), frozenset(prepared.category_values))


def release_figure(figure):
    """Detach a retired scene and release its data without a full GC in Qt."""
    # Clearing an old figure must not update the shared Qt canvas/toolbar.
    FigureCanvasBase(figure)
    for ax in list(figure.axes):
        for artist in (*ax.collections, *ax.images):
            colorbar = getattr(artist, 'colorbar', None)
            if colorbar is not None:
                colorbar.remove()
                colorbar.mappable = None
    for ax in list(figure.axes):
        for container in list(ax.containers):
            container.remove()
        # Removing a contour also removes its labels; read each family after
        # previous removals rather than keeping a stale combined artist list.
        for family in ('collections', 'images', 'lines', 'patches', 'texts'):
            for artist in list(getattr(ax, family)):
                artist.remove()
        # Ticks point back to the Axes. Drop the retired lists before removing
        # the Axes so repeated previews do not accumulate whole tick graphs.
        for axis in ax._axis_map.values():
            axis.majorTicks.clear()
            axis.minorTicks.clear()
        # Do not rebuild empty Axes/ticks just to discard them (Axes.clear).
        figure.delaxes(ax)
    figure.clear()


class PreviewCanvas(FigureCanvasQTAgg):
    """Reuse Qt painting/blitting while requesting expensive draws remotely."""
    def __init__(self, figure):
        self.request_draw = None
        super().__init__(figure)

    def draw(self):
        if self.request_draw is None:
            return super().draw()
        self.request_draw()

    def present(self, pixels):
        height, width = pixels.shape[:2]
        # Let Matplotlib manage its renderer cache key: versions differ in
        # whether it uses floating Figure bounds or physical integer pixels.
        # Copying into an independently keyed renderer can be erased by the
        # next highlight draw calling get_renderer().
        renderer = self.get_renderer()
        target = np.asarray(renderer.buffer_rgba())
        if target.shape != pixels.shape:
            self.figure.set_size_inches(np.nextafter(width / self.figure.dpi, np.inf),
                                        np.nextafter(height / self.figure.dpi, np.inf), forward=False)
            renderer = self.get_renderer()
            target = np.asarray(renderer.buffer_rgba())
        target[:] = pixels
        self._draw_pending = False
        self.figure.stale = False
        DrawEvent('draw_event', self, self.renderer)._process()
        self.update()


class _PreviewWorker(QThread):
    completed = Signal(int, object, object)

    def __init__(self, parent):
        super().__init__(parent)
        self.jobs = Queue()
        self.context = multiprocessing.get_context('spawn')
        self.wanted = self.context.Value('q', 0)
        self.process = None
        self.last_transfer = {}

    def _start_process(self):
        connection, child = self.context.Pipe()
        self.process = self.context.Process(target=preview_process, args=(child, self.wanted))
        try:
            self.process.start()
        except Exception:
            connection.close()
            self.process = None
            raise
        finally:
            child.close()
        return connection

    def run(self):
        connection = None
        cached = None
        row_ids = None
        row_lookup = None
        try:
            # Import Agg while Qt builds the editor and prepares the input.
            try:
                connection = self._start_process()
            except Exception:
                pass  # Retry on the first request and report its error normally.
            while True:
                job = self.jobs.get()
                # Only the latest queued request needs to be drawn.
                try:
                    while True:
                        job = self.jobs.get_nowait()
                except Empty:
                    pass
                if job is None:
                    break
                serial = job[0]
                try:
                    if self.process is None:
                        connection = self._start_process()
                    # Both serialization and deserialization run outside Qt.
                    prepared = job[1]
                    key = _preparation_key(prepared)
                    started = perf_counter()
                    if cached == key:
                        job = (job[0], None, *job[2:])
                    else:
                        # This contains UUID strings only, not an Index/DataFrame
                        # or owning snapshot. Reuse it for every frame of this input.
                        row_ids = prepared.local.index.to_numpy(dtype=object)
                        row_lookup = {rid: position for position, rid in enumerate(row_ids)}
                        job = (job[0], preview_input(prepared), *job[2:])
                    input_pack_s = perf_counter() - started
                    sent = send_preview_message(connection, job, _CooperativeBuffer, cooperate=lambda: sleep(0))
                    cached = key
                    result, received = receive_preview_response(connection, _CooperativeBuffer)
                    sleep(0)
                    if result[1] is not None:
                        if serial != self.wanted.value:
                            release_figure(result[1][0])
                            result = serial, None, None
                        else:
                            started = perf_counter()
                            restore_preview_membership(result[1][1], row_ids, row_lookup)
                            received['restore_ids_s'] = perf_counter() - started
                    sleep(0)
                    self.last_transfer = {'input_pack_s': input_pack_s, 'encode_s': sent['encode_s'],
                        'request_bytes': sent['message_bytes'], 'request_metadata_bytes': sent['metadata_bytes'],
                        'request_buffers': sent['buffers'], 'request_send_s': sent['send_s'], **received}
                    self.completed.emit(*result)
                except Exception as exc:
                    self.completed.emit(serial, None, str(exc))
                    if self.process is not None and self.process.pid is not None:
                        self.process.terminate()
                        self.process.join()
                    self.process = None
                    cached = None
                    row_ids = None
                    row_lookup = None
                    if connection is not None:
                        connection.close()
                    connection = None
                job = result = prepared = None
        finally:
            if connection is not None:
                try:
                    send_preview_message(connection, None)
                except (EOFError, OSError):
                    pass
                connection.close()
            if self.process is not None and self.process.pid is not None:
                self.process.join(1)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join()


class PlotPreview(QObject):
    ready = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self):
        super().__init__(QApplication.instance())
        self.worker = None
        self.serial = 0
        self.generation = 0
        self.busy = False
        self.stopped = False
        self.retiring = False
        self.pending = None
        QApplication.instance().aboutToQuit.connect(self.shutdown)

    def warmup(self):
        """Start imports early; reopening waits for the old worker to retire."""
        self.stopped = False
        if self.worker is None:
            self.worker = _PreviewWorker(self)
            self.worker.completed.connect(self._completed)
            self.worker.finished.connect(self._finished)
            self.worker.start()

    def request(self, generation, prepared, spec, views, ratio, palette):
        if self.stopped:
            return
        if self.worker is None:
            self.warmup()
        self.serial += 1
        self.generation = generation
        self.busy = True
        if self.retiring:
            self.pending = generation, prepared, spec, views, ratio, palette
            return
        self.worker.wanted.value = self.serial
        self.worker.jobs.put((self.serial, prepared, spec, views, ratio, palette))

    def _completed(self, serial, frame, error):
        if self.stopped or serial != self.serial:
            # Cancellation may arrive after decoding but before Qt dispatch.
            if frame is not None:
                release_figure(frame[0])
            return
        self.busy = False
        self.pending = None
        if error:
            self.failed.emit(self.generation, error)
        elif frame is not None:
            self.ready.emit(self.generation, frame)

    def cancel(self):
        self.serial += 1
        self.busy = False
        self.pending = None
        if self.worker is not None:
            self.worker.wanted.value = self.serial

    def release(self):
        self.stopped = True
        self.cancel()
        if self.worker is not None:
            self.retiring = True
            self.worker.jobs.put(None)

    def _finished(self):
        restart = self.retiring and not self.stopped
        pending = self.pending
        self.pending = None
        self.worker.deleteLater()
        self.worker = None
        self.retiring = False
        if restart:
            self.warmup()
            if pending is not None:
                self.request(*pending)

    def shutdown(self):
        self.release()
        if self.worker is not None and not self.worker.wait(2000):
            process = self.worker.process
            if process is not None and process.pid is not None:
                process.terminate()
            self.worker.wait()
