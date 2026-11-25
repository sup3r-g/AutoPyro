"""Navigate the map and synchronize selection of samples and spatial features."""

from __future__ import annotations

from shapely.geometry import Point
import numpy as np
import shapely

from autopyro.spatial.map_tools import filtered_features
from autopyro.spatial.mapping import crs, geometry_artists, layer_frame, project_geometries
from autopyro.spatial.surfaces import raster_coordinates

from ..i18n import tr, message
from ..tasks import run_with_progress


def connect_canvas_events(sheet):
    sheet.canvas.mpl_connect('resize_event', sheet.schedule_map_layout)
    sheet.canvas.mpl_connect('pick_event', sheet.on_pick)
    sheet.canvas.mpl_connect('button_press_event', sheet.feature_press)
    sheet.canvas.mpl_connect('button_release_event', sheet.feature_release)


def highlight_attributes(sheet, *_):
    for artist in sheet.feature_highlights:
        artist.remove()
    sheet.feature_highlights = []
    sheet.feature_card.clear()
    if not sheet.result:
        return
    selected = {str(sheet.attribute_model.frame.iat[index.row(), 0])
                for index in sheet.attribute_table.selectionModel().selectedRows()}
    ax = sheet.result['axes']
    limits = ax.get_xlim(), ax.get_ylim()
    style = dict(color='#ff3300', fill='none', alpha=1, line_width=2.5)
    for layer_id, fid, geometry in sheet.result['layer_features']:
        if layer_id != sheet.attribute_layer.currentData() or fid not in selected:
            continue
        before = set(ax.get_children())
        geometry_artists(ax, geometry, style)
        for artist in set(ax.get_children()) - before:
            artist.set_zorder(20)
            if hasattr(artist, 'set_sizes'):
                artist.set_sizes([110])
                artist.set_facecolor('none')
                artist.set_edgecolor('#ff3300')
                artist.set_linewidth(2.5)
            sheet.feature_highlights.append(artist)
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    sheet.canvas.draw_idle()
    selected_rows = sheet.attribute_table.selectionModel().selectedRows()
    if selected_rows:
        sheet.show_feature_card(sheet.attribute_layer.currentData(), str(sheet.attribute_model.frame.iat[selected_rows[0].row(), 0]))


def zoom_to_attribute(sheet, index):
    if not index.isValid():
        return
    if not sheet.result:
        sheet.status.setText(tr('Сначала постройте карту.'))
        return
    layer_id = sheet.attribute_layer.currentData()
    fid = str(sheet.attribute_model.frame.iat[index.row(), 0])
    features = [(key, geometry) for lid, key, geometry in sheet.result['layer_features'] if lid == layer_id]
    geometry = next((geometry for key, geometry in features if key == fid), None)
    if geometry is None:
        sheet.status.setText(tr('Объект скрыт или не имеет отображаемой геометрии.'))
        return
    xmin, ymin, xmax, ymax = geometry.bounds
    span = max(xmax - xmin, ymax - ymin)
    if span == 0:
        bounds = np.array([item.bounds for _, item in features])
        span = max(np.ptp(bounds[:, [0, 2]]), np.ptp(bounds[:, [1, 3]])) * .25
        if span == 0:
            target = crs(sheet.report['spec']['target_crs'])
            span = (.0001 if target.is_geographic else 500) / target.axis_info[0].unit_conversion_factor
    sheet.attribute_table.selectRow(index.row())
    sheet.zoom_to_bounds((xmin, ymin, xmax, ymax), span)


def zoom_to_layer(sheet, index):
    if not sheet.result or not index.isValid() or not 0 <= index.row() < len(sheet.layers):
        return
    report = sheet.report
    layer_id = sheet.layers[index.row()]['id']
    layer = next((layer for layer in report['layers'] if layer['id'] == layer_id), None)
    if layer is None:
        return
    target = report['spec']['target_crs']

    def bounds():
        if 'raster' in layer:
            x, y = raster_coordinates(layer, target)
            return x.min(), y.min(), x.max(), y.max()
        features = filtered_features(layer)
        if features:
            frame = layer_frame(dict(layer, features=features))
            return project_geometries(frame.geometry, layer['crs'], target).total_bounds
        return None

    try:
        extent = run_with_progress(sheet, tr('Приближение к слою'), bounds)
    except Exception as exc:
        sheet.status.setText(message(exc))
        return
    if sheet.report is report:
        if extent is None:
            sheet.status.setText(tr('В слое нет объектов после фильтрации.'))
        else:
            sheet.zoom_to_bounds(extent)


def zoom_to_bounds(sheet, bounds, span=None):
    xmin, ymin, xmax, ymax = bounds
    if span is None:
        span = max(xmax - xmin, ymax - ymin)
        if span == 0:
            target = crs(sheet.report['spec']['target_crs'])
            span = (.0001 if target.is_geographic else 500) / target.axis_info[0].unit_conversion_factor
    padding = span * .1
    ax = sheet.result['axes']
    xlim, ylim = (xmin - padding, xmax + padding), (ymin - padding, ymax + padding)
    if ax.get_xlim() == xlim and ax.get_ylim() == ylim:
        return
    sheet.toolbar.push_current()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    sheet.toolbar.push_current()
    sheet.canvas.draw_idle()


def feature_press(sheet, event):
    sheet.map_press = (event.x, event.y) if event.button == 1 else None


def feature_release(sheet, event):
    start, sheet.map_press = sheet.map_press, None
    if (not sheet.result or sheet.toolbar.mode or start is None or event.button != 1
            or event.inaxes is not sheet.result['axes']
            or np.hypot(event.x - start[0], event.y - start[1]) > 5):
        return
    ax = sheet.result['axes']
    point = Point(event.x, event.y)
    corners = np.array([[event.x - 6, event.y - 6], [event.x + 6, event.y + 6]])
    bounds = ax.transData.inverted().transform(corners)
    search = shapely.box(bounds[:, 0].min(), bounds[:, 1].min(), bounds[:, 0].max(), bounds[:, 1].max())
    nearby = sheet.result['feature_index'].query(search)
    candidates = []
    for order in nearby:
        layer_id, fid, geometry = sheet.result['layer_features'][order]
        screen = shapely.transform(geometry, ax.transData.transform)
        distance = screen.distance(point)
        if distance <= 6:
            candidates.append((layer_id != sheet.attribute_layer.currentData(),
                               screen.geom_type in ('Polygon', 'MultiPolygon'), distance, -order, layer_id, fid))
    if not candidates:
        return
    *_, layer_id, fid = min(candidates)
    if sheet.regional_window is not None and sheet.regional_window.isVisible():
        sheet.regional_window.select_map_region(layer_id, fid)
    sheet.attribute_layer.setCurrentIndex(sheet.attribute_layer.findData(layer_id))
    row = sheet.attribute_model.feature_rows.get(fid)
    if row is not None:
        sheet.attribute_table.selectRow(row)
        sheet.attribute_table.scrollTo(sheet.attribute_model.index(row, 0))
        sheet.layer_tabs.setCurrentIndex(1)


def selection_changed(sheet, *_):
    if sheet.closed or sheet.loading:
        return
    if sheet.data.fields['scope'][0].currentData() == 'selected' or sheet.result is None or sheet.timer.isActive():
        sheet.schedule()
    elif sheet.selection_overlay is not None:
        sheet.selection_overlay.set_selected(sheet.selected_ids())


def selected_ids(sheet):
    return sheet.owner.sample_selection.ids()


def scope_ids(sheet, selected_ids=None):
    return sheet.owner.analysis_scope.rows(selected_ids=selected_ids)


def select_rows(sheet, ids):
    hidden = sheet.owner.sample_selection.select(ids)
    if hidden:
        sheet.status.setText(tr('Выбранных проб скрыто текущим фильтром таблицы: {0}', hidden))


def on_pick(sheet, event):
    if (
        not sheet.result
        or sheet.toolbar.mode
        or event.artist is not sheet.result["artist"]
    ):
        return
    items = sheet.result["items"]
    chosen = [items[int(i)] for i in event.ind]
    coordinates = {(item["x"], item["y"]) for item in chosen}
    sheet.select_rows(
        {
            rid
            for item in items
            if (item["x"], item["y"]) in coordinates
            for rid in item["row_ids"]
        }
    )


def on_rectangle(sheet, start, end):
    if (
        not sheet.result
        or sheet.toolbar.mode
        or any(
            value is None
            for value in (start.xdata, start.ydata, end.xdata, end.ydata)
        )
    ):
        return
    xmin, xmax = sorted((start.xdata, end.xdata))
    ymin, ymax = sorted((start.ydata, end.ydata))
    sheet.select_rows(
        {
            rid
            for item in sheet.result["items"]
            if xmin <= item["x"] <= xmax and ymin <= item["y"] <= ymax
            for rid in item["row_ids"]
        }
    )


def capture_limits(sheet):
    if sheet.result:
        ax = sheet.result["axes"]
        sheet.layout_form.load(
            dict(
                zip(
                    ("xmin", "xmax", "ymin", "ymax"),
                    [*sorted(ax.get_xlim()), *sorted(ax.get_ylim())],
                )
            )
        )
        sheet.schedule()
