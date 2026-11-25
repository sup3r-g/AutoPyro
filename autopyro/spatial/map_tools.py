"""Layer display rules, strict spatial attribution and map decorations."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from matplotlib.colors import Normalize, LogNorm, to_rgba
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_agg import RendererAgg

from autopyro.visualization.plot_catalog import colormaps


def numeric(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def filtered_features(layer):
    rule = layer.get('filter') or {}
    field, op, expected = rule.get('field'), rule.get('op', 'eq'), rule.get('value', '')
    if not field:
        return list(layer['features'])
    def matches(feature):
        value = feature['properties'].get(field)
        missing = value is None or (isinstance(value, float) and not math.isfinite(value))
        if op == 'isnull':
            return missing
        if op == 'notnull':
            return not missing
        if missing:
            return False
        if op in ('gt', 'ge', 'lt', 'le'):
            left, right = numeric(value), numeric(expected)
            if left is None or right is None:
                return False
            return {'gt': left > right, 'ge': left >= right, 'lt': left < right, 'le': left <= right}[op]
        if op == 'contains':
            return str(expected).casefold() in str(value).casefold()
        equal = numeric(value) == numeric(expected) if isinstance(value, (int, float)) else str(value) == str(expected)
        if op not in ('eq', 'ne'):
            raise ValueError(f'Неизвестный оператор фильтра: {op}')
        return equal if op == 'eq' else not equal
    return [feature for feature in layer['features'] if matches(feature)]


def numeric_layer_style(layer, features):
    field = layer.get('color_by')
    values = [numeric(f['properties'].get(field)) for f in features]
    logarithmic = layer.get('color_scale') == 'log'
    valid = [v for v in values if v is not None and (v > 0 or not logarithmic)]
    missing = to_rgba(layer.get('missing_color', '#999999'))
    if not valid:
        return {str(f['id']): missing for f in features}, None
    lower = layer.get('color_min')
    upper = layer.get('color_max')
    low = min(valid) if lower is None else float(lower)
    high = max(valid) if upper is None else float(upper)
    if not np.isfinite([low, high]).all() or low > high or (logarithmic and low <= 0):
        raise ValueError('Некорректные пределы числовой шкалы слоя')
    if low == high:
        if lower is not None or upper is not None:
            raise ValueError('Пределы числовой шкалы должны различаться')
        low, high = (low / 2, high * 2) if logarithmic else (low - max(abs(low) * .01, 1), high + max(abs(high) * .01, 1))
    norm = (LogNorm if logarithmic else Normalize)(low, high, clip=True)
    mapper = ScalarMappable(norm=norm, cmap=colormaps[layer.get('cmap', 'viridis')])
    mapped = iter(mapper.to_rgba(np.asarray(valid, dtype=float)))
    colors = {str(f['id']): missing if v is None or (logarithmic and v <= 0) else tuple(next(mapped))
              for f, v in zip(features, values, strict=True)}
    return colors, mapper


def summarize_polygon_matches(joined, polygons=None):
    """Summarize a left spatial join indexed by point position.

    Supply polygon geometries when boundary hits need separate treatment.
    """
    matches = joined['index_right'].groupby(level=0, sort=False).agg(['count', 'first'])
    if polygons is not None:
        intersecting = joined.loc[joined['index_right'].notna()]
        polygon_positions = intersecting['index_right'].to_numpy(dtype=int)
        touches = shapely.touches(polygons.iloc[polygon_positions].array, intersecting.geometry.array)
        boundary = pd.Series(touches, index=intersecting.index).groupby(level=0, sort=False).any()
        matches['boundary'] = boundary.reindex(matches.index, fill_value=False)
    return matches


def assign_regions(report, layer_id, fields):
    """Never choose arbitrarily between overlaps or boundary hits."""
    from .mapping import layer_frame, project_geometries
    layer = next(item for item in report['layers'] if item['id'] == layer_id)
    available = set(layer.get('fields', [])) | {key for feature in layer['features'] for key in feature['properties']}
    if not fields or set(fields) - available:
        raise ValueError('Выберите существующие атрибуты полигонального слоя')
    features = filtered_features(layer)
    geometries = project_geometries(layer_frame(dict(layer, features=features)).geometry,
                                    layer['crs'], report['spec']['target_crs'])
    if not geometries.geom_type.isin(['Polygon', 'MultiPolygon']).all():
        raise ValueError('Для присоединения атрибутов выберите полигональный слой')
    polygons = gpd.GeoDataFrame(geometry=list(geometries), crs=report['spec']['target_crs'])
    points = report['points']
    locations = gpd.GeoDataFrame(geometry=gpd.points_from_xy([p['x'] for p in points], [p['y'] for p in points],
                                                           crs=polygons.crs))
    joined = gpd.sjoin(locations, polygons, how='left', predicate='intersects')
    records = {rid: {'status': 'no_location', 'values': {field: None for field in fields}} for rid in report['row_ids']}
    matches = summarize_polygon_matches(joined, geometries)
    for position, count, polygon, boundary in matches.itertuples(name=None):
        status = 'boundary' if boundary else 'overlap' if count > 1 else 'matched' if count else 'outside'
        values = {field: deepcopy(features[int(polygon)]['properties'].get(field)) if status == 'matched' else None
                  for field in fields}
        for rid in points[position]['row_ids']:
            records[rid] = {'status': status, 'values': values}
    return records, dict(Counter(record['status'] for record in records.values()))


def renderer_for(figure):
    if hasattr(figure.canvas, 'get_renderer'):
        return figure.canvas.get_renderer()
    return RendererAgg(round(figure.bbox.width), round(figure.bbox.height), figure.dpi)


def fit_map_vertical_margins(ax, footer):
    figure = ax.figure
    renderer = renderer_for(figure)
    height = figure.bbox.height
    if height <= 1:
        return
    # Measure axis text in pixels so the margins also work in a short viewport.
    text_bounds = ax.get_tightbbox(renderer, bbox_extra_artists=[])
    gap = renderer.points_to_pixels(4)
    footer_height = footer.get_window_extent(renderer).ymax - figure.bbox.ymin if footer is not None else 0
    bottom = max(.17, (ax.bbox.ymin - text_bounds.ymin + footer_height + gap) / height)
    top_margin = max(.12, (text_bounds.ymax - ax.bbox.ymax + gap) / height)
    # Keep space for the map itself even when the requested text cannot fit.
    if bottom + top_margin > .85:
        factor = .85 / (bottom + top_margin)
        bottom *= factor
        top_margin *= factor
    figure.subplots_adjust(bottom=bottom, top=1 - top_margin)


def place_layer_label(ax, anchor, text, size, offset, font):
    return ax.annotate(text, anchor, xytext=offset, textcoords='offset points',
                       fontsize=size, fontfamily=font, ha='left' if offset[0] > 0 else 'right',
                       zorder=8, clip_on=True,
                       bbox=dict(facecolor='white', edgecolor='none', alpha=.65, pad=1))


def add_layer_labels(ax, candidates, font):
    ax.apply_aspect()
    renderer = renderer_for(ax.figure)
    occupied, artists = [], []
    for geometry, text, size in candidates:
        anchor = geometry.representative_point()
        for offset in [(4, 4), (4, -12), (-4, 4), (-4, -12), (4, 16), (4, -24)]:
            artist = place_layer_label(ax, (anchor.x, anchor.y), text, size, offset, font)
            box = artist.get_window_extent(renderer).expanded(1.05, 1.1)
            if ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1) and not any(box.overlaps(other) for other in occupied):
                occupied.append(box)
                artists.append(artist)
                break
            artist.remove()
    return artists


def add_map_decorations(ax, target, *, scale_bar=False, north_arrow=False, font='DejaVu Sans'):
    from .mapping import transformer
    inverse = transformer(target.to_wkt(), target.geodetic_crs.to_wkt())
    forward = transformer(target.geodetic_crs.to_wkt(), target.to_wkt())
    geod = target.get_geod()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    width, height = xmax - xmin, ymax - ymin
    result = {'_artists': []}
    if scale_bar:
        x, y = xmin + width * .08, ymin + height * .07
        lon, lat = inverse.transform(x, y, errcheck=True)
        def distance(delta):
            lon2, lat2 = inverse.transform(x + delta, y, errcheck=True)
            return abs(geod.inv(lon, lat, lon2, lat2)[2])
        maximum = distance(width * .2)
        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError('Масштабная линейка недоступна для текущего охвата')
        power = 10 ** math.floor(math.log10(maximum))
        length = max(value for value in (power, 2 * power, 5 * power) if value <= maximum)
        low, high = 0., width * .2
        for _ in range(40):
            middle = (low + high) / 2
            if distance(middle) < length:
                low = middle
            else:
                high = middle
        line, = ax.plot([x, x + high], [y, y], color='black', linewidth=3, zorder=9)
        text = ax.text(x + high / 2, y + height * .015, f'{length / 1000:g} km' if length >= 1000 else f'{length:g} m',
                ha='center', fontsize=8, fontfamily=font, zorder=9,
                bbox=dict(facecolor='white', alpha=.7, edgecolor='none'))
        result['scale_m'] = length
        result['_artists'].extend([line, text])
    if north_arrow:
        cx, cy = xmin + width / 2, ymin + height / 2
        lon, lat = inverse.transform(cx, cy, errcheck=True)
        # Geodetic forward azimuth zero gives true north, including polar maps.
        lon2, lat2, _ = geod.fwd(lon, lat, 0, 100)
        nx, ny = forward.transform(lon2, lat2, errcheck=True)
        direction = np.array([(nx - cx) / width, (ny - cy) / height])
        direction /= np.linalg.norm(direction)
        tip = np.array([.9, .85]) + direction * .07
        arrow = ax.annotate('N', xy=tip, xytext=(.9, .85), xycoords='axes fraction', textcoords='axes fraction',
                    ha='center', va='center', fontfamily=font, fontsize=10,
                    arrowprops=dict(arrowstyle='-|>', color='black'), zorder=9)
        result['north_direction'] = direction.tolist()
        result['_artists'].append(arrow)
    return result
