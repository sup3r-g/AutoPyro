"""Matplotlib canvas that uses all the space assigned by its Qt layout."""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backend_bases import DrawEvent
from ..theme import icon
from PySide6.QtCore import QObject, QSignalBlocker, QTimer, Signal, Qt
from PySide6.QtWidgets import QApplication, QComboBox, QScrollArea, QSizePolicy

from ..i18n import message, tr
from ..tasks import Worker


class CanvasView(QObject):
    """Screen-only zoom mode shared by synchronous and remote plot canvases."""
    changed = Signal()

    def __init__(self, canvas, size_mm=None, scroll=None):
        self.scroll = scroll if scroll is not None else QScrollArea()
        super().__init__(self.scroll)
        self.canvas = canvas
        initial = tuple(value * 25.4 for value in canvas.figure.get_size_inches())
        self.size_mm = size_mm or (lambda: initial)
        self.fit = True
        self.project_owner = None
        self.view_key = None
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(canvas)
        self.refresh()

    def bind_project(self, owner, key):
        """Restore a document's view mode without creating a project edit."""
        if owner is None or not hasattr(owner, 'view_changed'):
            return
        self.project_owner, self.view_key = owner, key
        modes = owner.model.project.view.get('visualization_view_modes', {})
        self.fit = not isinstance(modes, dict) or modes.get(key) != 'actual'
        blocker = QSignalBlocker(self.mode_combo)
        self.mode_combo.setCurrentIndex(0 if self.fit else 1)
        del blocker
        self.refresh()

    def add_to_toolbar(self, toolbar):
        self.mode_combo = QComboBox(toolbar)
        self.mode_combo.addItem(icon('maximize'), tr('Подогнать под окно'), True)
        self.mode_combo.addItem(icon('zoom-in'), tr('Реальный размер'), False)
        self.mode_combo.setCurrentIndex(0 if self.fit else 1)
        self.mode_combo.setToolTip(tr('Масштаб просмотра. Реальный размер использует размеры рисунка и экранный DPI; размеры экспорта не меняются.'))
        self.mode_combo.setWhatsThis(self.mode_combo.toolTip())
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        if toolbar.actions():
            toolbar.insertWidget(toolbar.actions()[0], self.mode_combo)
        else:
            toolbar.addWidget(self.mode_combo)

    def _mode_changed(self):
        self.fit = self.mode_combo.currentData()
        if self.project_owner is not None:
            view = self.project_owner.model.project.view
            modes = view.get('visualization_view_modes')
            if not isinstance(modes, dict):
                modes = view['visualization_view_modes'] = {}
            mode = 'fit' if self.fit else 'actual'
            if modes.get(self.view_key) != mode:
                modes[self.view_key] = mode
                self.project_owner.view_changed()
        self.refresh()
        self.canvas.draw_idle()
        self.changed.emit()

    def set_canvas(self, canvas):
        self.scroll.takeWidget()
        self.canvas = canvas
        self.scroll.setWidget(canvas)
        self.refresh()

    def refresh(self):
        if self.fit:
            self.canvas.setMinimumSize(1, 1)
            self.canvas.setMaximumSize(16777215, 16777215)
            self.scroll.setWidgetResizable(True)
        else:
            self.scroll.setWidgetResizable(False)
            dpi = self.canvas.figure._original_dpi
            width, height = self.size_mm()
            self.canvas.setFixedSize(round(width / 25.4 * dpi), round(height / 25.4 * dpi))


class FittedFigureCanvas(FigureCanvasQTAgg):
    navigation_draw_requested = Signal()

    def __init__(self, figure):
        self._navigation_timer = None
        super().__init__(figure)
        self._navigation_timer = QTimer(self)
        self._navigation_timer.setSingleShot(True)
        self._navigation_timer.setInterval(100)
        self._navigation_timer.timeout.connect(self.flush_navigation_draw)
        self._defer_navigation_draw = False
        self.navigation_rendering = False
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(1, 1)

    def defer_navigation_draws(self, enabled):
        self._navigation_timer.stop()
        self._defer_navigation_draw = enabled

    @property
    def navigation_draw_pending(self):
        return self._defer_navigation_draw and (
            self._navigation_timer.isActive() or self._draw_pending or self.navigation_rendering)

    def draw_idle(self):
        if self._navigation_timer is not None and self._defer_navigation_draw:
            self._navigation_timer.start()
        else:
            super().draw_idle()

    def flush_navigation_draw(self):
        if self.isVisible():
            self.navigation_draw_requested.emit()

    def fit_to_view(self):
        if not self.isVisible():
            return
        # A renderer may apply export dimensions/DPI to the existing Figure.
        # Restore screen DPI and logical widget dimensions for the preview.
        dpi = self.figure._original_dpi
        self.figure.set_dpi(dpi * self.device_pixel_ratio)
        self.figure.set_size_inches(self.width() / dpi, self.height() / dpi, forward=False)

    def draw(self):
        if self._defer_navigation_draw and self.navigation_rendering:
            # A queued idle draw can otherwise wait on Agg's render lock in Qt.
            self._draw_pending = False
            return
        if self._navigation_timer is not None:
            self._navigation_timer.stop()
        self.fit_to_view()
        super().draw()

    def present(self, pixels):
        """Paint an Agg frame prepared off the GUI thread, if it still fits."""
        self._navigation_timer.stop()
        self.fit_to_view()
        width, height = self.figure.bbox.size
        if pixels.shape[:2] != (int(height), int(width)):
            if self._defer_navigation_draw:
                # The viewport changed while the worker rendered. Keep the old
                # buffer until a frame for the new viewport is available.
                self._draw_pending = False
                self.draw_idle()
            else:
                self.draw()
            return False
        target = self.get_renderer().buffer_rgba()
        target.cast('B')[:] = pixels.cast('B')
        self._draw_pending = False
        self.figure.stale = False
        DrawEvent('draw_event', self, self.renderer)._process()
        self.update()
        return True

    def showEvent(self, event):
        super().showEvent(event)
        self.fit_to_view()
        self.draw_idle()

    def hideEvent(self, event):
        self._navigation_timer.stop()
        super().hideEvent(event)


class FigureNavigation(QObject):
    """Render updated views on an independent Agg canvas."""

    failed = Signal(str)
    recovered = Signal()

    def __init__(self, canvas, *, blocked=None):
        super().__init__(canvas)
        self.canvas = canvas
        self.blocked = blocked
        self.scene = self.worker = self.pending = None
        self._failed = False
        canvas.navigation_draw_requested.connect(self.request)

    def bind_status(self, label):
        previous = error_text = None

        def show_error(error):
            nonlocal previous, error_text
            if label.text() != error_text:
                previous = label.text()
            error_text = tr('Ошибка: {0}', message(error))
            label.setText(error_text)

        def restore_status():
            nonlocal previous, error_text
            if label.text() == error_text:
                label.setText(previous)
            previous = error_text = None

        self.failed.connect(show_error)
        self.recovered.connect(restore_status)

    def clear(self):
        self.scene = self.pending = None
        self.canvas.navigation_rendering = False

    def view(self):
        canvas = self.canvas
        dpi = canvas.figure._original_dpi
        return ((canvas.width() / dpi, canvas.height() / dpi), dpi * canvas.device_pixel_ratio,
                tuple((axis.get_xlim(), axis.get_ylim()) for axis in canvas.figure.axes))

    def request(self):
        if self.blocked is not None and self.blocked():
            return
        scene = self.scene
        if scene is None:
            self.canvas.draw()
            return
        self.canvas.navigation_rendering = True
        self.pending = (scene, self.view())
        self.start()

    def start(self):
        if self.worker is not None or self.pending is None:
            return
        if not self.canvas.isVisible():
            self.pending = None
            self.canvas.navigation_rendering = False
            return
        scene, view = self.pending
        self.pending = None

        def draw():
            size, dpi, limits = view
            figure = scene.figure
            figure.set_size_inches(*size, forward=False)
            figure.set_dpi(dpi)
            for axis, (xlim, ylim) in zip(figure.axes, limits, strict=True):
                axis.set_xlim(xlim)
                axis.set_ylim(ylim)
            scene.draw()
            return scene, view, scene.buffer_rgba()

        worker = self.worker = Worker(draw, QApplication.instance())
        worker.scene = scene
        worker.view = view
        worker.succeeded.connect(self.present)
        worker.failed.connect(self.render_failed)
        worker.finished.connect(self.finish)
        worker.finished.connect(worker.deleteLater)
        QApplication.instance().aboutToQuit.connect(worker.wait)
        worker.start()

    def present(self, result):
        scene, view, pixels = result
        # A newer gesture may already have changed the axes before its timer fires.
        if scene is self.scene and self.canvas.isVisible() and view == self.view():
            self.pending = None
            self.canvas.navigation_rendering = False
            presented = self.canvas.present(pixels)
            if presented and self._failed:
                self._failed = False
                self.recovered.emit()

    def render_failed(self, error):
        worker = self.sender()
        if worker.scene is self.scene and worker.view == self.view():
            self.pending = None
            # Keep picking suspended until a frame matches the current axes.
            # The next navigation request retries using the independent scene.
            self._failed = True
            self.failed.emit(error)

    def finish(self):
        self.worker = None
        self.start()
