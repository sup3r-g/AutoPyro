"""Coordinate data capture, preview rendering and canvas lifetime."""

from __future__ import annotations

from ..i18n import language, tr, message
from copy import deepcopy
from ..widgets.navigation import NavigationToolbar as PlotToolbar
from matplotlib.widgets import RectangleSelector
from PySide6.QtCore import QTimer
from autopyro.visualization.plot_data import capture_input, input_token
from autopyro.visualization.plot_export import freeze_palette
from ..plot_preparation import PlotPreparation
from ..plot_preview import PlotPreview, PreviewCanvas, release_figure
from autopyro.visualization.selection_overlay import SelectionOverlay
from autopyro.visualization.plotting import layout_notice, render_plot
from .plot_editor import PlotConfigCommand


def install(dialog):
    dialog.generation = 0
    dialog.prepared = None
    dialog.captured_input = None
    dialog.preparation = PlotPreparation()
    dialog.preparation.ready.connect(dialog.preparation_ready)
    dialog.preparation.failed.connect(dialog.preparation_failed)
    dialog.preparation.idle.connect(dialog.preparation_idle)
    dialog.preparation.capture.ready.connect(dialog.input_captured)
    dialog.preparation.capture.failed.connect(dialog.preparation_failed)
    dialog.preview = PlotPreview()
    dialog.preview.ready.connect(dialog.preview_ready)
    dialog.preview.failed.connect(dialog.preparation_failed)
    dialog.preview.warmup()
    dialog.preview_request = None
    dialog.capture_request = None
    dialog.timer = QTimer(dialog)
    dialog.timer.setSingleShot(True)
    dialog.timer.setInterval(220)
    dialog.timer.timeout.connect(dialog.prepare_async)


def mark_frame(dialog, text=None):
    if text and dialog.figure.axes:
        dialog.frame_notice.setText(text)
        dialog.frame_notice.adjustSize()
        dialog.frame_notice.raise_()
        dialog.frame_notice.show()
    else:
        dialog.frame_notice.hide()


def make_toolbar(dialog, canvas):
    return PlotToolbar(canvas, dialog, coordinates=False)


def replace_figure(dialog, figure, *, remote=False, navigation=False):
    if remote:
        old_figure = dialog.figure
        old_axes = dialog.figure.axes
        dialog.figure = figure
        dialog.canvas.figure = figure
        figure.set_canvas(dialog.canvas)
        figure._original_dpi = 96
        dialog.connect_canvas_events()
        dialog.toolbar.reconnect_canvas()
        if navigation:
            dialog.toolbar.remap_history(old_axes, figure.axes)
        else:
            dialog.toolbar.update()
        # 3.10 stores this on Axes; 3.11+ reads it from canvas.toolbar.
        for ax in figure.axes:
            if hasattr(ax, '_navigate_mode'):
                ax.set_navigate_mode(dialog.toolbar.mode.name if dialog.toolbar.mode else None)
        release_figure(old_figure)
        return
    # UltraPlot wraps canvas methods and requires its own Figure. Recreate
    # the canvas and toolbar when switching providers to keep navigation and
    # picking connected to the correct figure without retaining wrappers.
    previous_canvas, previous_toolbar = dialog.canvas, dialog.toolbar
    dialog.figure = figure
    dialog.canvas = PreviewCanvas(figure)
    dialog.canvas.resize(previous_canvas.size())
    dialog.connect_canvas_events()
    dialog.toolbar = dialog.make_toolbar(dialog.canvas)
    previous_toolbar.parentWidget().layout().replaceWidget(previous_toolbar, dialog.toolbar)
    previous_toolbar.hide()
    previous_toolbar.deleteLater()
    dialog.canvas_view.set_canvas(dialog.canvas)
    dialog.canvas_view.add_to_toolbar(dialog.toolbar)
    dialog.canvas_view.mode_combo.setMaximumWidth(145)
    previous_canvas.hide()
    previous_canvas.deleteLater()


def schedule(dialog, *args):
    if dialog.loading or dialog.closed:
        return
    if dialog.interpretation_owner is None and dialog.timer.isActive():
        dialog.timer.start()
        return
    dialog.clear_hover()
    dialog.generation += 1
    dialog.preparation.invalidate()
    dialog.preview.cancel()
    dialog.canvas.request_draw = lambda: None
    dialog.result = None
    dialog.export_button.setEnabled(False)
    dialog.mark_frame(tr('Предыдущий график · обновление…'))
    dialog.retry_plot_action.setEnabled(False)
    dialog.status.setText(tr('Данные или настройки изменились; обновление…'))
    document = dialog.interpretation_owner
    if document is not None and (document.report is None or document.editor_dirty):
        dialog.timer.stop()
        if dialog.selection_overlay is not None:
            dialog.selection_overlay.dispose()
            dialog.selection_overlay = None
        for selector in dialog.selectors:
            selector.disconnect_events()
        dialog.selectors = []
        dialog.rendered_ids = []
        dialog.toolbar.cancel_drag()
        dialog.toolbar.update()
        dialog.figure.clear()
        dialog.mark_frame()
        dialog.canvas.draw_idle()
        dialog.status.setText(tr('График появится после расчёта интерпретации'))
        return
    dialog.timer.start()


def capture_plot_input(dialog, ids, specs):
    project = dialog.owner.model.project
    token = input_token(project, dialog.owner.model.data_revision, specs)
    source = capture_input(project, ids, specs, token, dialog.captured_input)
    dialog.captured_input = source
    return source


def prepare_async(dialog):
    if dialog.closed:
        return
    dialog.generation += 1
    dialog.preparation.invalidate()
    dialog.preview.cancel()
    dialog.canvas.request_draw = lambda: None
    dialog.result = None
    dialog.export_button.setEnabled(False)
    try:
        dialog.update_enabled()
        dialog.read_controls()
        ids = dialog.scope_ids()
        spec = deepcopy(dialog.spec)
        token = input_token(dialog.owner.model.project, dialog.owner.model.data_revision, [spec])
        dialog.capture_request = dialog.generation, ids, spec, token
        dialog.cancel_prepare_action.setEnabled(True)
        dialog.status.setText(tr('Подготовка данных графика…'))
        dialog.preparation.capture.start(dialog.generation, dialog.owner.model.project, ids, [spec], token, dialog.captured_input)
    except (ValueError, TypeError, KeyError) as exc:
        dialog.preparation_failed(dialog.generation, str(exc))


def input_captured(dialog, generation, source):
    if dialog.closed or generation != dialog.generation:
        return
    _, ids, spec, token = dialog.capture_request
    dialog.capture_request = None
    if input_token(dialog.owner.model.project, dialog.owner.model.data_revision, [spec]) != token:
        dialog.schedule()
        return
    dialog.captured_input = source
    dialog.preparation.request(generation, source, ids, spec)


def preparation_ready(dialog, generation, prepared):
    if dialog.closed or generation != dialog.generation:
        return
    current = input_token(dialog.owner.model.project, dialog.owner.model.data_revision, [dialog.spec])
    if current != prepared.source.token:
        dialog.schedule()
        return
    dialog.prepared = prepared
    dialog.request_preview()


def preview_spec(dialog):
    """Fit the on-screen scene without changing saved/export dimensions."""
    spec = deepcopy(dialog.spec)
    spec.render_language = language()
    choice = 'all'
    if spec.interpretation_panels and dialog.interpretation_owner:
        choice = dialog.interpretation_owner.comparison_view.currentData()
        if choice not in ('auto', 'all'):
            spec.comparison_preview_panel = choice
    if dialog.canvas_view.fit and dialog.canvas_scroll.isVisible():
        size = dialog.canvas_scroll.viewport().size()
        spec.width_mm = max(30, size.width() * 25.4 / 96)
        spec.height_mm = max(30, size.height() * 25.4 / 96)
        if spec.interpretation_panels:
            if choice == 'auto':
                from autopyro.schemes.interpretation_comparison import comparison_shape

                rows, columns = comparison_shape(spec, len(spec.interpretation_panels))
                first = spec.interpretation_panels[0]['name']
                already_compact = (dialog.result is not None and len(dialog.figure.axes) == 1
                                   and dialog.figure.axes[0].get_title() == first)
                # Keep a small buffer when returning to all panels because their
                # longer status can shrink the viewport by a line.
                buffer = 20 if already_compact else 0
                cramped = (size.width() < 275 * columns + buffer or size.height() < 210 * rows + buffer)
                if cramped:
                    spec.comparison_preview_panel = first
        if (dialog.interpretation_owner is not None or spec.interpretation_panels) and size.height() > 100:
            spec.bottom = max(spec.bottom, min(.40, 42 / size.height()))
            spec.top = min(spec.top, max(.65, 1 - 26 / size.height()))
            if spec.bottom >= spec.top:
                spec.bottom, spec.top = dialog.spec.bottom, dialog.spec.top
    return spec


def request_preview(dialog):
    if dialog.closed or dialog.prepared is None:
        return
    dialog.canvas_view.refresh()
    navigation = dialog.result is not None
    spec = dialog.preview_spec()
    views = [deepcopy(ax._get_view()) for ax in dialog.figure.axes] if navigation else None
    if views and spec.interpretation_panels:
        titles = [spec.comparison_preview_panel] if spec.comparison_preview_panel else [
            panel['name'] for panel in spec.interpretation_panels]
        if titles != [ax.get_title() for ax in dialog.figure.axes]:
            views = None
    dialog.preview_request = dialog.generation, navigation
    dialog.cancel_prepare_action.setEnabled(True)
    # Keep completed multi-line diagnostics during fit/navigation refreshes.
    # Replacing them with a one-line progress message resizes the viewport
    # and starts an endless fit -> status -> resize -> fit loop.
    if not navigation:
        dialog.status.setText(tr('Отрисовка предпросмотра…'))
    dialog.preview.request(dialog.generation, dialog.prepared, spec, views,
                         dialog.canvas.device_pixel_ratio, freeze_palette(dialog.spec.cmap))


def preview_ready(dialog, generation, frame):
    if dialog.closed or generation != dialog.generation:
        return
    dialog.present_preview(generation, dialog.preview.serial, frame)


def present_preview(dialog, generation, serial, frame):
    if dialog.closed or generation != dialog.generation or serial != dialog.preview.serial:
        return
    expected = dialog.preview_spec()
    size_changed = any(abs(actual - target / 25.4) * 96 > 1 for actual, target in
                       zip(frame[0].get_size_inches(), (expected.width_mm, expected.height_mm)))
    if size_changed or frame[2].comparison_preview_panel != expected.comparison_preview_panel:
        release_figure(frame[0])
        dialog.request_preview()
        return
    # A toolbar/3D drag still owns the old Axes until the mouse is released.
    if dialog.canvas._button is not None:
        QTimer.singleShot(25, lambda: dialog.present_preview(generation, serial, frame))
        return
    if input_token(dialog.owner.model.project, dialog.owner.model.data_revision, [dialog.spec]) != dialog.prepared.source.token:
        dialog.schedule()
        return
    _, navigation = dialog.preview_request
    dialog.redraw(dialog.prepared, preview=frame, navigation=navigation)
    dialog.cancel_prepare_action.setEnabled(False)


def preparation_failed(dialog, generation, error):
    if not dialog.closed and generation == dialog.generation:
        dialog.capture_request = None
        dialog.cancel_prepare_action.setEnabled(False)
        dialog.result = None
        dialog.export_button.setEnabled(False)
        dialog.mark_frame(tr('Предыдущий график · обновление не удалось'))
        dialog.retry_plot_action.setEnabled(True)
        dialog.status.setText(tr('График не обновлён: {0}', message(error)))


def preparation_idle(dialog):
    dialog.cancel_prepare_action.setEnabled(dialog.preview.busy)


def cancel_preparation(dialog):
    dialog.generation += 1
    dialog.timer.stop()
    dialog.preparation.invalidate()
    dialog.preview.cancel()
    dialog.capture_request = None
    dialog.cancel_prepare_action.setEnabled(False)
    dialog.result = None
    dialog.export_button.setEnabled(False)
    dialog.mark_frame(tr('Предыдущий график · обновление отменено'))
    dialog.retry_plot_action.setEnabled(True)
    dialog.status.setText(tr('Подготовка отменена.'))


def redraw(dialog, prepared=None, *, preview=None, navigation=False):
    """Render on Qt. Explicit synchronous calls retain their existing contract."""
    dialog.timer.stop()
    if dialog.selection_overlay is not None:
        dialog.selection_overlay.dispose()
        dialog.selection_overlay = None
    try:
        dialog.update_enabled()
        dialog.read_controls()
        dialog.shared_refresh_pending = False
        selected = dialog.selection_ids()
        ids = dialog.scope_ids()
        if prepared is None:
            dialog.generation += 1
            dialog.preparation.invalidate()
            dialog.preview.cancel()
            dialog.canvas.request_draw = None
            source = dialog.capture_plot_input(ids, [dialog.spec])
            cache = dialog.preparation.cache.fork()
            prepared = cache.prepare(source, ids, dialog.spec)
            dialog.preparation.cache = cache
        dialog.prepared = prepared
        dialog.canvas_view.refresh()
        display_spec = dialog.preview_spec()
        if preview is None:
            dialog.figure.set_size_inches(display_spec.width_mm / 25.4, display_spec.height_mm / 25.4)
        for selector in dialog.selectors:
            selector.disconnect_events()
        dialog.selectors = []
        if preview is None:
            figure, result = render_plot(dialog.owner.model.project, ids, display_spec,
                                         figure=dialog.figure, prepared=prepared)
            dialog.spec.style_maps = display_spec.style_maps
            if figure is not dialog.figure:
                dialog.replace_figure(figure)
        else:
            figure, result, rendered_spec, pixels = preview
            dialog.spec.style_maps = rendered_spec.style_maps
            dialog.replace_figure(figure, remote=True, navigation=navigation)
            dialog.canvas.request_draw = dialog.request_navigation
        dialog.selection_overlay = SelectionOverlay(dialog.canvas, result, interval_width=dialog.spec.interval_width)
        dialog.selection_overlay.set_selected(selected, redraw=False)
        if preview is None:
            dialog.canvas.draw()
            result['layout_warning'] = layout_notice(figure, dialog.canvas.get_renderer(), display_spec.render_language)
        else:
            dialog.canvas.present(pixels)
        dialog.clear_hover()
        from autopyro.visualization.plot_hover import HoverIndex
        dialog.hover_index = HoverIndex(result, dialog.spec.interpretation)
        dialog.result, dialog.rendered_ids = result, ids
        dialog.refresh_categories()
        config = dialog.spec.to_dict()
        if (
            dialog.last_config is not None
            and config != dialog.last_config
            and not dialog.replaying
        ):
            dialog.undo_stack.push(PlotConfigCommand(dialog, dialog.last_config, config))
        dialog.last_config = deepcopy(config)
        dialog.replaying = False
        if dialog.owner.model.project.view.get(dialog.settings_key) != config:
            dialog.owner.model.project.view[dialog.settings_key] = config
            view = dialog.owner.model.project.view
            if view.get("plot_palettes") != dialog.spec.style_maps:
                view["plot_palettes"] = dialog.shared_palette = deepcopy(dialog.spec.style_maps)
                dialog.sync_peer_appearance()
            dialog.owner.view_changed()
        for ax in result["axes"]:
            if dialog.spec.kind in ("scatter", "depth", "multi_depth", "mpl:scatter"):
                dialog.selectors.append(
                    RectangleSelector(
                        ax,
                        dialog.on_rectangle,
                        useblit=False,
                        button=[1],
                        minspanx=3,
                        minspany=3,
                        spancoords="pixels",
                        interactive=False,
                    )
                )
        dialog.enable_brush(dialog.brush.isChecked())
        labels = {
            "missing": tr('пусто'),
            "censored": "< / >",
            "non_detect": "ND",
            "other": tr('текст'),
            "nonpositive_log": tr('≤0 на лог. оси'),
            "excluded_interpolation": tr('интерполяции'),
            "excluded_calculation": tr('расчётные показатели'),
            "excluded_original": tr('исходные показатели'),
            "qc_excluded": tr('исключено по QC'),
            "invalid_depth_interval": tr('некорректный интервал отбора'),
        }
        columns = {column.id: column.name for column in dialog.owner.model.project.columns}
        summary = []
        for cid, count in result['plotted'].items():
            excluded = result['excluded'][cid]
            details = result.get('excluded_by_column', {}).get(cid)
            reasons = (
                f"{columns.get(column, column)} · {labels.get(reason, reason)}: {n}"
                for column, values in (details or {cid: excluded}).items()
                for reason, n in values.items()
            )
            label_id = result.get('summary_columns', {}).get(cid, cid)
            label = columns.get(label_id, tr('Выбранные показатели'))
            summary.append(tr('{0}: {1} точек, исключено {2}', label, count, sum(excluded.values()))
                           + (' (' + ', '.join(reasons) + ')' if excluded else ''))
        counts = '; '.join(summary)
        dialog.status.setText(
            tr('Строк в области: {0} · {1}', len(ids), counts)
            + ("\n" + "; ".join(result["notices"]) if result["notices"] else "")
            + ("\n" + result['layout_warning'] if result.get('layout_warning') else "")
        )
        if dialog.interpretation_owner is not None:
            dialog.interpretation_owner.update_preview_context(result['axes'])
        dialog.mark_frame()
        dialog.retry_plot_action.setEnabled(False)
        dialog.export_button.setEnabled(not dialog.export_busy())
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        dialog.preparation_failed(dialog.generation, exc)


def request_navigation(dialog):
    if dialog.result is not None and not dialog.closed and not dialog.viewport_timer.isActive():
        dialog.request_preview()


def release_plot(dialog):
    """Drop plot-owned data while retaining the editor for reopening."""
    dialog.viewport_timer.stop()
    dialog.closed = True
    dialog.mark_frame()
    dialog.clear_hover()
    dialog.generation += 1
    dialog.preparation.release()
    dialog.preview.release()
    dialog.canvas.request_draw = lambda: None
    dialog.export_capture.cancel()
    dialog.export_request = dialog.capture_request = None
    dialog.export_service.cancel()
    dialog.prepared = dialog.captured_input = None
    dialog.result = None
    dialog.rendered_ids = []
    if dialog.selection_overlay is not None:
        dialog.selection_overlay.dispose()
        dialog.selection_overlay = None
    for selector in dialog.selectors:
        if dialog.canvas.widgetlock.isowner(selector):
            dialog.canvas.widgetlock.release(selector)
        selector.disconnect_events()
    dialog.selectors = []
    dialog.canvas._button = None
    dialog.toolbar.cancel_drag()
    dialog.toolbar.update()
    dialog.figure.clear()
    if hasattr(dialog.canvas, 'renderer'):
        del dialog.canvas.renderer
    dialog.canvas._lastKey = None
    dialog.timer.stop()
