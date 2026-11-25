"""Schedule and render map frames, including background drawing and viewport updates."""

from __future__ import annotations

from copy import deepcopy

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QTableWidgetItem
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from shiboken6 import isValid

from autopyro.spatial.map_tools import (
    add_layer_labels,
    add_map_decorations,
    fit_map_vertical_margins,
    place_layer_label,
)
from autopyro.spatial.mapping import MapSpec, crs_label, crs, build_map, render_map
from autopyro.visualization.selection_overlay import SelectionOverlay

from ..i18n import tr, message
from ..tasks import run_with_progress


def initialize(sheet):
    sheet.report = None
    sheet.result = None
    sheet.map_viewport = None
    sheet.map_generation = 0
    sheet.map_render_running = False
    sheet.map_export_running = False
    sheet.map_redraw_pending = False
    sheet.large_map = False
    sheet.selection_overlay = None


def initialize_timers(sheet):
    sheet.timer = QTimer(sheet)
    sheet.timer.setSingleShot(True)
    sheet.timer.setInterval(180)
    sheet.timer.timeout.connect(sheet.redraw)
    sheet.viewport_timer = QTimer(sheet)
    sheet.viewport_timer.setSingleShot(True)
    sheet.viewport_timer.setInterval(40)
    sheet.viewport_timer.timeout.connect(sheet.update_map_layout)


def schedule_map_layout(sheet, *_):
    if sheet.result and not sheet.closed:
        sheet.viewport_timer.start()


def update_map_layout(sheet):
    if not sheet.result or not sheet.report or sheet.closed:
        return
    if sheet.map_render_running:
        sheet.viewport_timer.start()
        return
    ax = sheet.result['axes']
    limits = ax.get_xlim(), ax.get_ylim(), tuple(sheet.figure.bbox.size)
    if limits == sheet._layout_extent:
        return
    sheet._layout_extent = limits
    if sheet.large_map:
        fit_map_vertical_margins(ax, sheet.result['footer'])
        sheet.canvas.draw_idle()
        return
    for artist in [*sheet.result.get('layer_labels', []), *sheet.result.get('decorations', {}).get('_artists', [])]:
        artist.remove()
    fit_map_vertical_margins(ax, sheet.result['footer'])
    spec = MapSpec.from_dict(sheet.report['spec'])
    sheet.result['layer_labels'] = add_layer_labels(ax, reversed(sheet.result.get('_label_candidates', [])), spec.font_family)
    try:
        sheet.result['decorations'] = add_map_decorations(ax, crs(spec.target_crs), scale_bar=spec.scale_bar,
                                                        north_arrow=spec.north_arrow, font=spec.font_family) if spec.scale_bar or spec.north_arrow else {}
    except (ValueError, RuntimeError) as exc:
        sheet.result['decorations'] = {}
        sheet.status.setText(message(exc))
    sheet.canvas.draw_idle()


def schedule(sheet, *_):
    if sheet.closed:
        return
    sheet.sync_coordinate_controls()
    if sheet.loading:
        return
    sheet.map_generation += 1
    sheet.report = None
    sheet.result = None
    sheet.export_button.setEnabled(False)
    sheet.interpolate_button.setEnabled(False)
    sheet.sample_diagnostics_button.setEnabled(False)
    sheet.timer.start()


def redraw(sheet):
    if sheet.closed:
        return
    sheet.sync_coordinate_controls()
    if sheet.map_render_running:
        sheet.map_redraw_pending = True
        return
    sheet.map_generation += 1
    generation = sheet.map_generation
    sheet.clear_regional_highlights()
    sheet.timer.stop()
    sheet.feature_highlights = []
    if sheet.selection_overlay is not None:
        sheet.selection_overlay.dispose()
        sheet.selection_overlay = None
    sheet.report = None
    sheet.result = None
    sheet.export_button.setEnabled(False)
    sheet.interpolate_button.setEnabled(False)
    sheet.sample_diagnostics_button.setEnabled(False)
    previous_viewport, sheet.map_viewport = sheet.map_viewport, None
    if previous_viewport is not None:
        previous_axes, previous_key, previous_home = previous_viewport
        previous_extent = previous_axes.get_xlim(), previous_axes.get_ylim()
    try:
        for form in sheet.forms():
            for key, value in form.values().items():
                setattr(sheet.spec, key, value)
        mode = sheet.spec.location_mode
        sheet.update_layer_binding_hint()
        if not sheet.layers and (mode == 'layers' or (mode == 'columns' and not (sheet.spec.x_id and sheet.spec.y_id))):
            sheet.map_pages.setCurrentWidget(sheet.map_start)
            sheet.clear_map_display(tr('Карта готова к настройке.'))
            return
        sheet.map_pages.setCurrentWidget(sheet.canvas_scroll)
        selected = sheet.selected_ids()
        sheet.canvas_view.refresh()
        sheet.canvas.fit_to_view()
        if sheet.selector:
            sheet.selector.disconnect_events()
        row_ids = sheet.scope_ids()
        object_count = sum(len(layer.get('features', [])) for layer in sheet.layers)
        raster_nodes = sum(len(row) for layer in sheet.layers
                           for row in layer.get('raster', {}).get('values', []))
        layers_only = mode == 'layers' or (mode == 'columns' and not sheet.spec.x_id and not sheet.spec.y_id)
        point_count = 0 if layers_only else len(row_ids)
        large = point_count + object_count + raster_nodes >= 2000
        pixels = None
        stored_layers = None
        if large:
            report, figure, result, pixels, home, stored_layers = sheet.render_large_map(row_ids, previous_viewport)
            if not isValid(sheet) or generation != sheet.map_generation or sheet.closed:
                return
            previous = sheet.figure
            sheet.figure = sheet.canvas.figure = figure
            figure.set_canvas(sheet.canvas)
            figure._original_dpi = previous._original_dpi
            sheet.connect_canvas_events()
            sheet.toolbar.reconnect_canvas()
            for axis in figure.axes:
                if hasattr(axis, '_navigate_mode'):
                    axis.set_navigate_mode(sheet.toolbar.mode.name if sheet.toolbar.mode else None)
            FigureCanvasBase(previous)
            previous.clear()
            sheet.spec.style_maps = report['spec']['style_maps']
        else:
            sheet.canvas.defer_navigation_draws(False)
            report = build_map(sheet.owner.model.project, row_ids, sheet.spec, sheet.layers)
            _, result = render_map(report, figure=sheet.figure)
            home = result['axes'].get_xlim(), result['axes'].get_ylim()
        axes = result['axes']
        viewport_key = tuple(report['spec'][key] for key in ('target_crs', 'xmin', 'xmax', 'ymin', 'ymax'))
        sheet.toolbar.update()
        sheet._layout_extent = *home, tuple(sheet.figure.bbox.size)
        sheet.report, sheet.result = report, result
        if previous_viewport is not None and previous_key == viewport_key and previous_extent != previous_home:
            axes.set_xlim(home[0])
            axes.set_ylim(home[1])
            sheet.toolbar.push_current()
            axes.set_xlim(previous_extent[0])
            axes.set_ylim(previous_extent[1])
            sheet.toolbar.push_current()
            if pixels is None:
                sheet.update_map_layout()
        sheet.map_viewport = axes, viewport_key, home
        sheet.selector = RectangleSelector(
            axes,
            sheet.on_rectangle,
            useblit=True,
            button=[1],
            minspanx=4,
            minspany=4,
            spancoords="pixels",
        )
        sheet.selection_overlay = SelectionOverlay(sheet.canvas, result)
        sheet.selection_overlay.set_selected(selected, redraw=False)
        if pixels is None:
            sheet.canvas.draw()
        else:
            sheet.canvas.defer_navigation_draws(True)
            sheet.canvas.present(pixels)
        sheet.large_map = large
        result['_label_view'] = tuple(sheet.figure.bbox.size), tuple(
            (axis.get_xlim(), axis.get_ylim()) for axis in sheet.figure.axes)
        sheet._layout_extent = result['axes'].get_xlim(), result['axes'].get_ylim(), tuple(sheet.figure.bbox.size)
        result['axes'].callbacks.connect('xlim_changed', sheet.schedule_map_layout)
        result['axes'].callbacks.connect('ylim_changed', sheet.schedule_map_layout)
        sheet.highlight_attributes()
        sheet.crs_status.setText(
            (f"{tr('Координаты')}: {crs_label(report['crs_operation']['source_crs'])}\n" if report['crs_operation']['source_crs'] else '') +
            f"{tr('Карта')}: {crs_label(report['spec']['target_crs'])}"
        )
        sheet.export_button.setEnabled(not sheet.map_export_running)
        sheet.interpolate_button.setEnabled(bool(report['points']))
        sheet.sample_diagnostics_button.setEnabled(bool(report['diagnostics']['excluded_rows']))
        sheet.refresh_categories()
        sheet.site_table.setRowCount(len(report["sites"]))
        for row, site in enumerate(report["sites"]):
            for col, key in enumerate(
                (
                    "label",
                    "id",
                    "n_samples",
                    "n_boundary",
                    "n_valid",
                    "method",
                    "value",
                    "family",
                    "family_status",
                )
            ):
                sheet.site_table.setItem(
                    row,
                    col,
                    QTableWidgetItem(
                        str(site[key]) if site[key] is not None else "—"
                    ),
                )
        sheet.remember(stored_layers)
        sheet.status.setText(sheet.map_status(report, result))
        if sheet.regional_window is not None:
            model = sheet.owner.model
            sheet.regional_window.set_map_input(model.project, report, data_revision=model.data_revision)
    except Exception as exc:
        if isValid(sheet) and generation == sheet.map_generation and not sheet.closed:
            sheet.clear_map_display(tr('Карта не построена: {0}', message(exc)))


def render_large_map(sheet, row_ids, previous_viewport):
    project, layers, spec = sheet.owner.model.project, sheet.layers, deepcopy(sheet.spec)
    screen_dpi = sheet.figure._original_dpi
    size = sheet.canvas.width() / screen_dpi, sheet.canvas.height() / screen_dpi
    dpi = screen_dpi * sheet.canvas.device_pixel_ratio
    viewport = None
    if previous_viewport is not None:
        axes, key, home = previous_viewport
        extent = axes.get_xlim(), axes.get_ylim()
        if extent != home:
            viewport = key, extent

    def render():
        report = build_map(deepcopy(project), row_ids, spec, deepcopy(layers))
        figure = Figure(figsize=size, dpi=dpi)
        scene = FigureCanvasAgg(figure)
        _, result = render_map(report, figure=figure)
        axes = result['axes']
        home = axes.get_xlim(), axes.get_ylim()
        key = tuple(report['spec'][field] for field in ('target_crs', 'xmin', 'xmax', 'ymin', 'ymax'))
        if viewport is not None and viewport[0] == key:
            axes.set_xlim(viewport[1][0])
            axes.set_ylim(viewport[1][1])
            for artist in [*result['layer_labels'], *result['decorations'].get('_artists', [])]:
                artist.remove()
            fit_map_vertical_margins(axes, result['footer'])
            result['layer_labels'] = add_layer_labels(axes, reversed(result['_label_candidates']), spec.font_family)
            result['decorations'] = add_map_decorations(
                axes, crs(report['spec']['target_crs']), scale_bar=spec.scale_bar,
                north_arrow=spec.north_arrow, font=spec.font_family,
            ) if spec.scale_bar or spec.north_arrow else {}
        scene.draw()
        return report, figure, result, scene.buffer_rgba(), home, deepcopy(report['layers'])

    return sheet.run_map_task(render)


def run_map_task(sheet, render):
    sheet.map_render_running = True
    sheet.canvas.defer_navigation_draws(True)
    sheet.canvas.navigation_rendering = True
    try:
        return run_with_progress(sheet.owner, tr('Обновить карту'), render, modal=False)
    finally:
        sheet.map_render_running = False
        if isValid(sheet):
            sheet.canvas.navigation_rendering = False
            sheet.canvas.defer_navigation_draws(sheet.large_map)
            if not sheet.closed and (sheet.map_redraw_pending or sheet.timer.isActive()):
                sheet.map_redraw_pending = False
                sheet.timer.start()
            sheet.project_tasks_finished.emit()


def render_map_view(sheet):
    if sheet.closed or sheet.map_render_running or not sheet.result:
        return
    if not sheet.large_map:
        sheet.canvas.draw()
        return
    sheet.canvas.fit_to_view()
    generation = sheet.map_generation
    source = sheet.figure
    view = tuple(source.bbox.size), tuple((ax.get_xlim(), ax.get_ylim()) for ax in source.axes)
    old_labels = tuple(sheet.result['layer_labels'])
    old_decorations = tuple(sheet.result['decorations'].get('_artists', []))
    candidates = sheet.result['_label_candidates']
    spec = MapSpec.from_dict(sheet.report['spec'])
    axes_index = source.axes.index(sheet.result['axes'])
    relayout = view != sheet.result.get('_label_view')

    def render():
        figure, labels, decorations = deepcopy((source, old_labels, old_decorations))
        scene = FigureCanvasAgg(figure)
        axes = figure.axes[axes_index]
        placements = None
        decoration_error = None
        if relayout:
            # Copied Matplotlib artists retain removal callbacks tied to the source lists.
            for artist in (*labels, *decorations):
                artist.set_visible(False)
            labels = add_layer_labels(axes, reversed(candidates), spec.font_family)
            if spec.scale_bar or spec.north_arrow:
                try:
                    add_map_decorations(axes, crs(spec.target_crs), scale_bar=spec.scale_bar,
                                        north_arrow=spec.north_arrow, font=spec.font_family)
                except (ValueError, RuntimeError) as exc:
                    decoration_error = message(exc)
            placements = [(label.xy, label.get_text(), label.get_fontsize(), label.get_position()) for label in labels]
        scene.draw()
        return scene.buffer_rgba(), placements, decoration_error

    try:
        pixels, placements, decoration_error = sheet.run_map_task(render)
        if isValid(sheet) and generation == sheet.map_generation and not sheet.closed:
            current = tuple(source.bbox.size), tuple((ax.get_xlim(), ax.get_ylim()) for ax in source.axes)
            if view == current:
                if placements is not None:
                    axes = sheet.result['axes']
                    for artist in (*old_labels, *old_decorations):
                        artist.remove()
                    sheet.result['layer_labels'] = []
                    sheet.result['decorations'] = {}
                    sheet.result['layer_labels'] = [
                        place_layer_label(axes, anchor, text, size, offset, spec.font_family)
                        for anchor, text, size, offset in placements
                    ]
                    if decoration_error is not None:
                        sheet.status.setText(decoration_error)
                    elif spec.scale_bar or spec.north_arrow:
                        sheet.result['decorations'] = add_map_decorations(
                            axes, crs(spec.target_crs), scale_bar=spec.scale_bar,
                            north_arrow=spec.north_arrow, font=spec.font_family,
                        )
                sheet.result['_label_view'] = view
                sheet.canvas.present(pixels)
            else:
                sheet.canvas.draw_idle()
    except Exception as exc:
        if isValid(sheet) and generation == sheet.map_generation:
            sheet.status.setText(tr('Карта не построена: {0}', message(exc)))


def clear_map_display(sheet, status):
    sheet.crs_status.clear()
    sheet.site_table.setRowCount(0)
    sheet.category_table.setRowCount(0)
    sheet.entries = []
    sheet.map_viewport = None
    sheet.report = sheet.result = None
    sheet.export_button.setEnabled(False)
    sheet.interpolate_button.setEnabled(False)
    sheet.sample_diagnostics_button.setEnabled(False)
    sheet.large_map = False
    sheet.canvas.defer_navigation_draws(False)
    sheet.figure.clear()
    sheet.canvas.draw_idle()
    sheet.status.setText(status)
    if sheet.regional_window is not None:
        model = sheet.owner.model
        sheet.regional_window.set_map_input(model.project, None, data_revision=model.data_revision)
