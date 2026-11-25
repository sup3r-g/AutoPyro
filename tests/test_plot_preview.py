"""Remote rasterization preserves pixels, navigation and UUID interactions."""
from copy import deepcopy
from types import SimpleNamespace
import gc
import weakref

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import MouseEvent, MouseButton
from PySide6.QtCore import QTimer

from autopyro.io import build_project
from autopyro.plotting import render_plot
from autopyro.ui.window import MainWindow


@pytest.fixture
def dialog(qtbot):
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.may_replace = lambda: True
    owner.set_project(build_project(['X', 'Y', 'Z'], [[1, 2, 3], [2, 4, 6], [3, 1, 9]], [2, 3, 4],
        [(0, 'X', 'number', ''), (1, 'Y', 'number', ''), (2, 'Z', 'number', '')], {}))
    owner.show_plots()
    plot = owner.plot_window
    plot.timer.stop()
    yield plot
    plot.close()
    qtbot.waitUntil(lambda: plot.preview.worker is None and plot.preparation.worker is None, timeout=15000)
    owner.dirty = False


def ready(dialog, qtbot):
    qtbot.waitUntil(lambda: dialog.result is not None and not dialog.preview.busy
                   and dialog.preparation.worker is None, timeout=20000)


def test_preview_pixels_match_agg_without_gui_rendering(dialog, qtbot, monkeypatch):
    from autopyro.ui.dialogs import plot_rendering as module
    ids = dialog.scope_ids()
    reference, _ = render_plot(dialog.owner.model.project, ids, deepcopy(dialog.spec))
    canvas = FigureCanvasAgg(reference)
    canvas.draw()
    pixels = np.asarray(canvas.buffer_rgba()).copy()
    def fail(*args, **kwargs):
        raise AssertionError('GUI must not construct or rasterize the preview')
    monkeypatch.setattr(module, 'render_plot', fail)
    monkeypatch.setattr(type(reference), 'draw', fail)
    ticks = []
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start()
    dialog.prepare_async()
    ready(dialog, qtbot)
    timer.stop()
    assert len(ticks) > 5
    np.testing.assert_array_equal(np.asarray(dialog.canvas.buffer_rgba()), pixels)
    assert len(dialog.result['artists'][0]._autopyro_row_ids) == 3
    serial = dialog.preview.serial
    rid = dialog.result['artists'][0]._autopyro_row_ids[0]
    dialog.select_rows([rid])
    assert dialog.selection_overlay.selected_ids == {rid}
    dialog.select_rows([])
    assert dialog.preview.serial == serial
    np.testing.assert_array_equal(np.asarray(dialog.canvas.buffer_rgba()), pixels)
    reference.clear()


def test_navigation_history_and_picking_survive_frames(dialog, qtbot):
    dialog.prepare_async()
    ready(dialog, qtbot)
    process = dialog.preview.worker.process.pid
    dialog.toolbar.pan()
    ax = dialog.figure.axes[0]
    original = ax.get_xlim()
    dialog.toolbar.push_current()
    ax.set_xlim(1.25, 2.75)
    dialog.toolbar.push_current()
    dialog.canvas.draw()
    qtbot.waitUntil(lambda: not dialog.preview.busy, timeout=10000)
    assert dialog.preview.worker.process.pid == process
    assert dialog.figure.axes[0].get_xlim() == (1.25, 2.75)
    assert dialog.figure.axes[0].get_navigate_mode() == 'PAN'
    dialog.toolbar.back()
    qtbot.waitUntil(lambda: not dialog.preview.busy and dialog.figure.axes[0].get_xlim() == original, timeout=10000)
    dialog.toolbar.forward()
    qtbot.waitUntil(lambda: not dialog.preview.busy and dialog.figure.axes[0].get_xlim() == (1.25, 2.75), timeout=10000)
    dialog.toolbar.pan()
    artist = dialog.result['artists'][0]
    dialog.on_pick(SimpleNamespace(artist=artist, ind=[0]))
    assert dialog.selection_ids() == {artist._autopyro_row_ids[0]}
    dialog.toolbar.home()
    qtbot.waitUntil(lambda: not dialog.preview.busy and dialog.figure.axes[0].get_xlim() == original, timeout=10000)
    artist = dialog.result['artists'][0]
    x, y = artist.axes.transData.transform(artist.get_offsets()[1])
    MouseEvent('button_press_event', dialog.canvas, x, y, button=1)._process()
    MouseEvent('button_release_event', dialog.canvas, x, y, button=1)._process()
    assert dialog.selection_ids() == {artist._autopyro_row_ids[1]}


def test_cancel_or_edit_rejects_old_preview(dialog, qtbot):
    dialog.prepare_async()
    qtbot.waitUntil(lambda: dialog.preview.busy, timeout=10000)
    dialog.cancel_preparation()
    assert dialog.result is None
    dialog.owner.model.setData(dialog.owner.model.index(0, 1), '987')
    dialog.timer.stop()
    dialog.prepare_async()
    ready(dialog, qtbot)
    assert max(dialog.result['artists'][0].get_offsets()[:, 1]) == 987


def test_3d_preview_keeps_rotation_and_projection(dialog, qtbot):
    dialog.spec.kind = 'mpl3d:scatter'
    dialog.spec.z_id = dialog.owner.model.project.columns[2].id
    dialog.load_controls()
    dialog.prepare_async()
    ready(dialog, qtbot)
    dialog.figure.axes[0].view_init(elev=15, azim=35)
    dialog.canvas.draw()
    qtbot.waitUntil(lambda: not dialog.preview.busy, timeout=10000)
    ax = dialog.figure.axes[0]
    assert (ax.elev, ax.azim) == (15, 35)
    assert ax.M is not None


def test_high_dpi_frame_has_matching_transforms(dialog, qtbot):
    dialog.canvas._set_device_pixel_ratio(2)
    dialog.prepare_async()
    ready(dialog, qtbot)
    assert dialog.figure.dpi == 192
    assert dialog.canvas.buffer_rgba().shape[1] == int(dialog.figure.get_figwidth() * 192)
    np.testing.assert_allclose(dialog.figure.dpi_scale_trans.transform((1, 1)), (192, 192))


@pytest.mark.parametrize('ratio', [1, 2])
def test_visible_preview_fills_viewport_and_resizes_without_changing_export(dialog, qtbot, ratio):
    owner = dialog.owner
    owner.resize(1400, 950)
    owner.show()
    qtbot.waitUntil(lambda: dialog.canvas_scroll.isVisible())
    dialog.canvas._set_device_pixel_ratio(ratio)
    dimensions = dialog.spec.width_mm, dialog.spec.height_mm
    dialog.prepare_async()
    ready(dialog, qtbot)

    def fitted():
        size = dialog.canvas_scroll.viewport().size()
        return (dialog.canvas.size() == size and dialog.result is not None and not dialog.preview.busy
                and abs(dialog.figure.get_figwidth() * 96 - size.width()) < 1
                and abs(dialog.figure.get_figheight() * 96 - size.height()) < 1)

    qtbot.waitUntil(fitted, timeout=20000)
    size = dialog.canvas.size()
    assert dialog.canvas_scroll.horizontalScrollBar().maximum() == 0
    assert dialog.canvas_scroll.verticalScrollBar().maximum() == 0
    assert abs(dialog.canvas.buffer_rgba().shape[1] - size.width() * ratio) <= 1
    dialog.figure.axes[0].set_xlim(1.25, 2.75)
    selected = dialog.result['artists'][0]._autopyro_row_ids[0]
    dialog.select_rows([selected])
    history = dialog.undo_stack.count()
    owner.resize(1150, 750)
    qtbot.waitUntil(lambda: dialog.canvas.size() != size, timeout=10000)
    qtbot.waitUntil(fitted, timeout=20000)
    assert dialog.figure.axes[0].get_xlim() == (1.25, 2.75)
    assert dialog.selection_overlay.selected_ids == {selected}
    assert (dialog.spec.width_mm, dialog.spec.height_mm) == dimensions
    assert dialog.undo_stack.count() == history


def test_remote_preview_switches_view_modes_without_changing_plot_settings(dialog, qtbot):
    dialog.owner.show()
    dialog.layout_form.load({'width_mm': 400, 'height_mm': 300})
    dialog.prepare_async()
    ready(dialog, qtbot)
    axes = dialog.figure.axes[0]
    axes.set_xlim(1.25, 2.75)
    selected = dialog.result['artists'][0]._autopyro_row_ids[0]
    dialog.select_rows([selected])
    config = dialog.spec.to_dict()
    serial = dialog.preview.serial
    dialog.canvas_view.mode_combo.setCurrentIndex(1)
    qtbot.waitUntil(lambda: dialog.preview.serial > serial and not dialog.preview.busy, timeout=20000)
    assert dialog.canvas.width() == round(400 / 25.4 * 96)
    assert dialog.canvas.height() == round(300 / 25.4 * 96)
    assert abs(dialog.canvas.buffer_rgba().shape[1] - 400 / 25.4 * dialog.figure.dpi) <= 1
    assert dialog.canvas_scroll.horizontalScrollBar().maximum() > 0
    assert dialog.canvas_scroll.verticalScrollBar().maximum() > 0
    assert dialog.figure.axes[0].get_xlim() == (1.25, 2.75)
    assert dialog.selection_overlay.selected_ids == {selected}
    serial = dialog.preview.serial
    dialog.canvas_view.mode_combo.setCurrentIndex(0)
    qtbot.waitUntil(lambda: dialog.preview.serial > serial and not dialog.preview.busy, timeout=20000)
    assert dialog.canvas.size() == dialog.canvas_scroll.viewport().size()
    assert abs(dialog.figure.get_figwidth() * 96 - dialog.canvas.width()) < 1
    assert dialog.figure.axes[0].get_xlim() == (1.25, 2.75)
    assert dialog.selection_overlay.selected_ids == {selected}
    assert dialog.spec.to_dict() == config


def test_mouse_drag_defers_frame_installation(dialog, qtbot):
    dialog.canvas._button = MouseButton.LEFT
    dialog.prepare_async()
    qtbot.waitUntil(lambda: dialog.preview.busy, timeout=10000)
    qtbot.waitUntil(lambda: not dialog.preview.busy, timeout=10000)
    assert dialog.result is None
    dialog.canvas._button = None
    ready(dialog, qtbot)


def test_failed_render_can_be_followed_by_a_valid_request(dialog, qtbot):
    dialog.redraw()
    broken = deepcopy(dialog.spec)
    broken.kind = 'invalid-preview-kind'
    failures = []
    dialog.preview.failed.connect(lambda *args: failures.append(args))
    dialog.preview.request(dialog.generation, dialog.prepared, broken, None, 1, None)
    qtbot.waitUntil(lambda: bool(failures), timeout=10000)
    dialog.prepare_async()
    ready(dialog, qtbot)
    assert dialog.result['artists']


@pytest.mark.parametrize('immediate', [False, True])
def test_close_releases_frame_and_reopening_starts_a_new_preview(dialog, qtbot, immediate):
    dialog.prepare_async()
    ready(dialog, qtbot)
    old_process = dialog.preview.worker.process
    dialog.close()
    assert dialog.result is None and dialog.prepared is None and dialog.captured_input is None
    assert not dialog.figure.axes and not dialog.selectors
    assert not hasattr(dialog.canvas, 'renderer')
    if not immediate:
        qtbot.waitUntil(lambda: dialog.preview.worker is None, timeout=15000)
    dialog.owner.show_plots()
    dialog.timer.stop()
    dialog.prepare_async()
    ready(dialog, qtbot)
    assert not old_process.is_alive()
    assert dialog.preview.worker.process is not old_process
    assert len(dialog.result['artists'][0]._autopyro_row_ids) == 3


@pytest.mark.parametrize('mode', ['pan', 'zoom'])
def test_close_during_drag_drops_axes_and_mouse_state(dialog, qtbot, mode):
    dialog.prepare_async()
    ready(dialog, qtbot)
    getattr(dialog.toolbar, mode)()
    x, y = dialog.figure.axes[0].transData.transform((2, 3))
    MouseEvent('button_press_event', dialog.canvas, x, y, button=1)._process()
    assert getattr(dialog.toolbar, '_' + mode + '_info') is not None
    dialog.close()
    assert dialog.canvas._button is None
    assert dialog.toolbar._pan_info is None and dialog.toolbar._zoom_info is None
    assert not dialog.canvas.widgetlock.locked()
    dialog.owner.show_plots()
    dialog.timer.stop()
    dialog.prepare_async()
    ready(dialog, qtbot)


def test_closed_editor_does_not_retain_axes_or_prepared_frame(dialog, qtbot):
    dialog.prepare_async()
    ready(dialog, qtbot)
    axes = weakref.ref(dialog.figure.axes[0])
    frame = weakref.ref(dialog.prepared.local)
    dialog.close()
    qtbot.waitUntil(lambda: dialog.preview.worker is None and dialog.preparation.worker is None, timeout=15000)
    gc.collect()
    assert axes() is None and frame() is None


def test_idle_transport_does_not_retain_the_gui_snapshot(dialog, qtbot):
    dialog.prepare_async()
    ready(dialog, qtbot)
    source = weakref.ref(dialog.captured_input)
    prepared = weakref.ref(dialog.prepared)
    dialog.prepared = dialog.captured_input = None
    dialog.preparation.release()
    gc.collect()
    assert source() is None and prepared() is None


def test_rebinding_a_column_replaces_the_process_input(dialog, qtbot):
    dialog.prepare_async()
    ready(dialog, qtbot)
    dialog.prepare_async()
    ready(dialog, qtbot)
    small_request = dialog.preview.worker.last_transfer['request_bytes']
    dialog.data.load({'x_id': dialog.owner.model.project.columns[2].id, 'text_numbers': True})
    dialog.prepare_async()
    ready(dialog, qtbot)
    assert max(dialog.result['artists'][0].get_offsets()[:, 0]) == 9
    assert dialog.preview.worker.last_transfer['request_bytes'] > small_request


def test_replaced_scene_releases_point_buffers_and_ticks_without_gc(dialog, qtbot):
    dialog.style.load({'color_by': dialog.owner.model.project.columns[0].id, 'color_mode': 'numeric'})
    dialog.prepare_async()
    ready(dialog, qtbot)
    old_figure = dialog.figure  # Keep the old graph alive deliberately.
    old_axis = old_figure.axes[0].xaxis
    assert old_axis.majorTicks
    offsets = weakref.ref(dialog.result['artists'][0].get_offsets())
    dialog.prepare_async()
    ready(dialog, qtbot)
    assert not old_figure.axes
    assert not old_axis.majorTicks and not old_axis.minorTicks
    assert dialog.figure.axes[0].xaxis.majorTicks
    assert offsets() is None


def test_multi_depth_preview_keeps_tick_templates_and_picking(dialog, qtbot):
    from autopyro.plotting import PlotSpec

    first, depth, second = dialog.owner.model.project.columns
    dialog.spec = PlotSpec(kind='multi_depth', x_id=first.id, y_id=depth.id,
                           track_ids=[first.id, second.id])
    dialog.load_controls()
    dialog.prepare_async()
    ready(dialog, qtbot)

    assert len(dialog.figure.axes) == 2
    for axis in (ax.xaxis for ax in dialog.figure.axes):
        assert len(axis.majorTicks) == 1
        assert len(axis.get_major_ticks(3)) == 3
    expected = {row.id for row in dialog.owner.model.project.rows}
    assert all(set(artist._autopyro_row_ids) == expected for artist in dialog.result['artists'])


def test_obsolete_queued_frame_releases_its_scene(dialog, qtbot):
    dialog.prepare_async()
    ready(dialog, qtbot)
    active = dialog.figure
    obsolete, result = render_plot(dialog.owner.model.project, dialog.scope_ids(), deepcopy(dialog.spec))
    dialog.preview._completed(dialog.preview.serial - 1, (obsolete, result, dialog.spec, None), None)
    assert not obsolete.axes and dialog.figure is active
