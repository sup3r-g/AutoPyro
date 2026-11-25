"""GeoPandas vector layers, CRS discovery, exact joins and site aggregates."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
import pyproj
from pyproj import CRS, Transformer, network
import shapely
from shapely.geometry import Point, mapping, shape
from autopyro.visualization.plot_catalog import colormaps, validate_marker
from matplotlib.collections import PathCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Patch

from autopyro.model.core import new_id, timestamp
from autopyro.model.measurements import parse_number
from autopyro.data_io.io import file_source
from autopyro.visualization.plotting import COLORS, category_key, category_keys
from autopyro.analysis.statistics import classify
from autopyro.model.tabular import project_frame, classify_series, exclusion_mask, depth_filter_masks
from autopyro import __version__
from .map_tools import filtered_features, numeric_layer_style, add_layer_labels, add_map_decorations, renderer_for
from .map_tools import fit_map_vertical_margins, summarize_polygon_matches

# PROJ must not download grid files implicitly while opening a local project.
network.set_network_enabled(False)


@dataclass
class MapSpec:
    location_mode: str = "columns"
    x_id: str = ""
    y_id: str = ""
    source_crs: str = "auto"
    target_crs: str = "auto"
    point_layer: str = ""
    sample_key: str = ""
    layer_key: str = ""
    scope: str = "filtered"
    color_by: str = ""
    color_mode: str = "category"
    size_by: str = ""
    size_mode: str = "numeric"
    color: str = "#0072B2"
    missing_color: str = "#999999"
    marker_size: float = 7
    marker: str = "o"
    edge_color: str = "#253341"
    edge_width: float = 0.5
    size_min: float = 4
    size_max: float = 16
    alpha: float = 0.85
    cmap: str = "viridis"
    color_min: float | None = None
    color_max: float | None = None
    style_maps: dict = field(default_factory=dict)
    text_numbers: bool = False
    include_interpolated: bool = False
    label_by: str = ""
    depth_id: str = ""
    depth_bottom_id: str = ""
    depth_min: float | None = None
    depth_max: float | None = None
    strat_id: str = ""
    strat_value: str = ""
    date_id: str = ""
    date_from: str = ""
    date_to: str = ""
    site_layer: str = ""
    site_value: str = ""
    site_family: str = ""
    site_label: str = ""
    aggregate: str = "mean"
    show_aggregates: bool = False
    title: str = "Локальная карта"
    legend: bool = True
    legend_size: int = 8
    legend_title: str = ''
    legend_placement: str = 'outside'
    scale_bar: bool = False
    north_arrow: bool = False
    grid: bool = True
    grid_color: str = "#aaaaaa"
    grid_alpha: float = 0.3
    grid_width: float = 0.6
    background: str = "#ffffff"
    font_family: str = "DejaVu Sans"
    label_size: int = 10
    annotation_size: int = 8
    tick_size: int = 8
    x_label: str = ""
    y_label: str = ""
    xmin: float | None = None
    xmax: float | None = None
    ymin: float | None = None
    ymax: float | None = None
    width_mm: float = 200
    height_mm: float = 160
    dpi: int = 300

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = deepcopy(value)
        if 'legend_position' in value:
            value.setdefault('legend_placement', value.pop('legend_position'))
        return cls(**value)


def local_path(path):
    value = Path(path).resolve()
    if not value.is_file():
        raise ValueError("Выберите существующий локальный пространственный файл")
    return value


def list_layers(path):
    return [
        (str(name), str(kind))
        for name, kind in gpd.list_layers(local_path(path)).itertuples(index=False, name=None)
        if pd.notna(kind)
    ]


def crs(value):
    result = CRS.from_user_input(value)
    if len(result.axis_info) != 2 or not (result.is_geographic or result.is_projected):
        raise ValueError(
            "Карта требует двумерную географическую или проецированную CRS"
        )
    return result


def crs_label(value):
    definition = crs(value)
    authority = definition.to_authority()
    return f"{':'.join(authority)} — {definition.name}" if authority else definition.name


@lru_cache(maxsize=1)
def available_crs():
    """Use the installed PROJ catalogue; no online registry is required."""
    return tuple(
        (f"{info.auth_name}:{info.code} — {info.name}", f"{info.auth_name}:{info.code}")
        for info in pyproj.database.query_crs_info(allow_deprecated=False)
        if info.type in (pyproj.enums.PJType.GEOGRAPHIC_2D_CRS, pyproj.enums.PJType.PROJECTED_CRS)
    )


def resolve_crs(project, spec, layers):
    """Resolve only explicit metadata or semantic longitude/latitude columns."""
    result = deepcopy(spec)
    result.style_maps = spec.style_maps
    if spec.location_mode == 'layers' or (spec.location_mode == 'columns' and not spec.x_id and not spec.y_id and layers):
        result.location_mode = 'layers'
        first = next((layer for layer in layers if layer.get('visible', True)), layers[0] if layers else {})
        result.target_crs = first.get('crs', 'EPSG:4326') if spec.target_crs == 'auto' else spec.target_crs
        result.source_crs = result.target_crs
        result.show_aggregates = False
        result.color_by = result.size_by = ''
        crs(result.target_crs)
        return result
    point_layer = (next((item for item in layers if item['id'] == spec.point_layer), None)
                   if spec.location_mode == 'layer' else None)
    if spec.location_mode == 'layer' and point_layer:
        result.source_crs = point_layer['crs']
    elif spec.source_crs == 'auto':
        columns = {c.id: c for c in project.columns}
        x, y = columns.get(spec.x_id), columns.get(spec.y_id)
        definitions = [c.metadata.get('crs') for c in (x, y) if c and c.metadata.get('crs')]
        if definitions:
            if any(crs(value) != crs(definitions[0]) for value in definitions):
                raise ValueError('CRS колонок X и Y противоречат друг другу')
            result.source_crs = crs(definitions[0]).to_string()
        elif x and y and x.name.strip().casefold() in ('longitude', 'lon', 'long', 'долгота') and y.name.strip().casefold() in ('latitude', 'lat', 'широта'):
            result.source_crs = 'EPSG:4326'
        else:
            raise ValueError('CRS координат неизвестна; выберите её из списка')
    if spec.target_crs == 'auto':
        reference = point_layer or next((layer for layer in layers if layer.get('visible', True)), {})
        result.target_crs = reference.get('crs', result.source_crs)
    crs(result.source_crs)
    crs(result.target_crs)
    return result


def layer_frame(layer):
    """Rebuild a GeoDataFrame while keeping project snapshots JSON-compatible."""
    features = layer['features']
    frame = gpd.GeoDataFrame.from_features(features, crs=layer.get('crs')) if features else gpd.GeoDataFrame(geometry=[], crs=layer.get('crs'))
    frame.index = pd.Index([f['id'] for f in features], name='fid')
    return frame


def project_geometries(geometries, source, target):
    operation = transformer(source, target)
    result = gpd.GeoSeries(geometries, crs=crs(source)).transform(
        lambda coordinates: np.column_stack(operation.transform(
            coordinates[:, 0], coordinates[:, 1], errcheck=True,
        )),
    ).set_crs(crs(target), allow_override=True)
    if (result.is_empty | ~result.is_valid).any() or not np.isfinite(shapely.get_coordinates(result.array)).all():
        raise ValueError('После преобразования получена пустая или некорректная геометрия')
    return result


@lru_cache(maxsize=32)
def transformer(source, target):
    return Transformer.from_crs(
        crs(source), crs(target), always_xy=True, allow_ballpark=False, only_best=True
    )


def transform_geometry(geometry, source, target):
    return project_geometries([geometry], source, target).iloc[0]


def geographic_limits(source):
    # always_xy uses longitude first even when EPSG lists latitude first.
    axes = source.axis_info
    xfactor = next(
        (
            a.unit_conversion_factor
            for a in axes
            if a.direction.lower() in ("east", "west")
        ),
        axes[0].unit_conversion_factor,
    )
    yfactor = next(
        (
            a.unit_conversion_factor
            for a in axes
            if a.direction.lower() in ("north", "south")
        ),
        axes[-1].unit_conversion_factor,
    )
    return math.pi / xfactor, math.pi / 2 / yfactor


def check_coordinate_units(project, spec, source):
    known = {
        "m": ("linear", 1.0),
        "metre": ("linear", 1.0),
        "meter": ("linear", 1.0),
        "м": ("linear", 1.0),
        "ft": ("linear", 0.3048),
        "us-ft": ("linear", 1200 / 3937),
        "deg": ("angular", math.pi / 180),
        "degree": ("angular", math.pi / 180),
        "°": ("angular", math.pi / 180),
        "grad": ("angular", math.pi / 200),
        "gon": ("angular", math.pi / 200),
    }
    columns = {c.id: c for c in project.columns}
    expected = "angular" if source.is_geographic else "linear"
    for cid in (spec.x_id, spec.y_id):
        if cid not in columns:
            raise ValueError("Колонка координат отсутствует")
        unit = known.get(columns[cid].unit.strip().casefold())
        if unit and (
            unit[0] != expected
            or not math.isclose(
                unit[1], source.axis_info[0].unit_conversion_factor, rel_tol=1e-10
            )
        ):
            raise ValueError("Единица колонки координат противоречит исходной CRS")


def scalar(value):
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def read_layer(path, layer=None, declared_crs=None):
    path = local_path(path)
    frame = gpd.read_file(path, layer=layer, engine='pyogrio', rows=100001, fid_as_index=True)
    if not isinstance(frame, gpd.GeoDataFrame):
        raise ValueError("Слой не содержит геометрии")
    if len(frame) > 100000:
        raise ValueError("Слой превышает лимит 100 000 объектов; данные не усечены")
    source_crs = frame.crs or declared_crs
    if not source_crs:
        raise ValueError("CRS слоя неизвестна; задайте её явно")
    if frame.crs and declared_crs and crs(frame.crs) != crs(declared_crs):
        raise ValueError("Объявленная CRS противоречит CRS файла")
    source_definition = crs(source_crs)
    source_crs = source_definition.to_wkt()
    frame = frame.set_crs(source_definition)
    fields = [str(c) for c in frame.columns if c != frame.geometry.name]
    attributes = []
    geographic_range = (
        geographic_limits(source_definition)
        if source_definition.is_geographic
        else None
    )
    features, counts = [], Counter()
    records = frame.drop(columns=frame.geometry.name).to_dict('records') if fields else ({} for _ in frame.index)
    for fid, geometry, record in zip(frame.index, frame.geometry.array, records):
        properties = {key: scalar(record[key]) for key in fields}
        attributes.append({'id': str(fid), 'properties': properties, 'geometry_type': geometry.geom_type if geometry is not None else None})
        if geometry is None:
            counts["null_geometry"] += 1
            continue
        if geometry.is_empty:
            counts["empty_geometry"] += 1
            continue
        # Some point layers use -DBL_MAX for missing XY. This is finite and
        # topologically valid, but even an identity projection ruins map bounds.
        if (shapely.get_coordinates(geometry) == -np.finfo(np.float64).max).any():
            counts["missing_coordinates"] += 1
            continue
        if (
            not geometry.is_valid
            or not np.isfinite(shapely.get_coordinates(geometry)).all()
        ):
            raise ValueError(
                f"Объект {fid}: некорректная геометрия; автоматическое исправление не выполняется"
            )
        if geographic_range:
            xmax, ymax = geographic_range
            xmin, ymin, xhigh, yhigh = geometry.bounds
            if xmin < -xmax or xhigh > xmax or ymin < -ymax or yhigh > ymax:
                raise ValueError(
                    f"Объект {fid}: координаты за диапазоном географической CRS"
                )
        if geometry.has_z:
            counts["z_not_displayed"] += 1
        features.append(
            {
                "type": "Feature",
                "id": str(fid),
                "geometry": mapping(geometry),
                "properties": properties,
            }
        )
    source = file_source(path, layer=layer)
    if path.suffix.lower() == ".shp":
        source["sidecars"] = [
            file_source(p)
            for suffix in (".dbf", ".shx", ".prj", ".cpg")
            if (p := path.with_suffix(suffix)).is_file()
        ]
    return {
        "id": new_id(),
        "name": layer or path.stem,
        "crs": source_crs,
        "features": features,
        "fields": fields,
        "attributes": attributes,
        "source": source,
        "diagnostics": dict(counts),
        "visible": True,
        "color": "#778899",
        "fill": "#c7ddeb",
        "alpha": 0.35,
        "line_width": 1,
    }


def number(project, row, cid, spec):
    from autopyro.analysis.qc import is_excluded
    if is_excluded(project, row.id, cid, "plots"):
        return "qc_excluded", None
    column = next((c for c in project.columns if c.id == cid), None)
    if column is None:
        raise ValueError("Привязанная колонка отсутствует")
    if column.metadata.get("method") and (
        column.metadata.get("stale") or not spec.include_interpolated
    ):
        return "interpolation_excluded", None
    return classify(
        row.values.get(cid), numeric=column.kind == "number" or spec.text_numbers
    )


def _map_numbers(project, frame, cid, spec):
    column = next((column for column in project.columns if column.id == cid), None)
    if column is None:
        raise ValueError("Привязанная колонка отсутствует")
    if column.metadata.get("method") and (column.metadata.get("stale") or not spec.include_interpolated):
        data = pd.DataFrame({"status": "interpolation_excluded", "number": None}, index=frame.index, dtype=object)
    else:
        data = classify_series(frame[cid], numeric=column.kind == "number" or spec.text_numbers)
    data.loc[exclusion_mask(project, frame.index, cid, "plots"), ["status", "number"]] = ["qc_excluded", None]
    return data


def _calendar_dates(values):
    """Parse distinct ISO timestamps once, retaining each timestamp's local date."""
    def parse(value):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except (ValueError, TypeError):
            return None
    text = values.map(str)
    return text.map({value: parse(value) for value in text.unique()}).astype(object)


def _record_exclusions(groups, reasons):
    if groups is not None:
        for row_id, reason in reasons.dropna().items():
            groups.setdefault(reason, []).append(row_id)


def filter_rows(project, row_ids, spec, *, excluded_rows=None):
    ids = list(row_ids)
    if len(set(ids)) != len(ids) or set(ids) - set(project.row_map()):
        raise ValueError("Некорректная выборка проб")
    columns = {c.id: c for c in project.columns}
    for cid in (
        spec.depth_id,
        spec.depth_bottom_id,
        spec.strat_id,
        spec.date_id,
        spec.color_by,
        spec.size_by,
        spec.label_by,
        spec.site_value,
        spec.site_family,
    ):
        if cid and cid not in columns:
            raise ValueError("Колонка настройки карты удалена")
    for lower, upper in (
        (spec.depth_min, spec.depth_max),
        (spec.xmin, spec.xmax),
        (spec.ymin, spec.ymax),
        (spec.color_min, spec.color_max),
    ):
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("Минимум должен быть не больше максимума")
    lower = date.fromisoformat(spec.date_from) if spec.date_from else None
    upper = date.fromisoformat(spec.date_to) if spec.date_to else None
    if lower and upper and lower > upper:
        raise ValueError("Начальная дата позже конечной")
    if (spec.depth_min is not None or spec.depth_max is not None) and not spec.depth_id:
        raise ValueError("Выберите колонку для фильтра глубины")
    if spec.depth_bottom_id:
        metadata = {c.id: c for c in project.columns}
        if spec.depth_id not in metadata or spec.depth_id == spec.depth_bottom_id:
            raise ValueError("Для интервала глубин выберите разные верх и низ")
        units = [metadata[cid].unit for cid in (spec.depth_id, spec.depth_bottom_id)]
        if units[0] and units[1] and units[0] != units[1]:
            raise ValueError("Верх и низ отбора должны иметь одинаковые единицы")
    if (lower or upper) and not spec.date_id:
        raise ValueError("Выберите колонку даты")
    row_map = project.row_map()
    strat_id = spec.strat_id if spec.strat_value != '' else ''
    if not strat_id and spec.depth_min is None and spec.depth_max is None and not (lower or upper):
        return [row_map[rid] for rid in ids], {}
    required = list(dict.fromkeys(cid for cid in [strat_id, spec.depth_id, spec.depth_bottom_id, spec.date_id] if cid))
    frame = project_frame(project, ids, required).reindex(pd.Index(ids, dtype=object))
    rejected = pd.Series(None, index=frame.index, dtype=object)
    if strat_id:
        values = frame[strat_id]
        matches = values.map(lambda value: value is None) if spec.strat_value is None else values.eq(spec.strat_value)
        if columns[strat_id].kind == 'number' and isinstance(spec.strat_value, str):
            try:
                code = parse_number(spec.strat_value)
            except ValueError:
                code = None
            if code is not None:
                matches |= values.eq(code)
        rejected.loc[~matches] = "stratigraphy_filter"
    if spec.depth_min is not None or spec.depth_max is not None:
        active = frame.loc[rejected.isna()]
        depth = _map_numbers(project, active, spec.depth_id, spec)["number"]
        bottom = _map_numbers(project, active, spec.depth_bottom_id, spec)["number"] if spec.depth_bottom_id else None
        invalid, outside = depth_filter_masks(depth, bottom, spec.depth_min, spec.depth_max)
        if bottom is not None:
            rejected.loc[invalid.index[invalid]] = "invalid_depth_interval"
        depth_rejected = (outside | invalid) & rejected.loc[active.index].isna()
        rejected.loc[active.index[depth_rejected]] = "depth_filter"
    if lower or upper:
        dates = _calendar_dates(frame.loc[rejected.isna(), spec.date_id])
        invalid = dates.isna()
        rejected.loc[dates.index[invalid]] = "invalid_date"
        dates = dates.loc[~invalid]
        outside = pd.Series(False, index=dates.index)
        if lower:
            outside |= dates.lt(lower)
        if upper:
            outside |= dates.gt(upper)
        rejected.loc[dates.index[outside]] = "date_filter"
    selected = frame.index[rejected.isna()]
    _record_exclusions(excluded_rows, rejected)
    return [row_map[rid] for rid in selected], rejected.dropna().value_counts(sort=False).to_dict()



def _transform_coordinates(operation, x, y):
    """Use a strict batch transform, isolating failures to individual source points."""
    try:
        return operation.transform(x, y, errcheck=True)
    except (ValueError, pyproj.exceptions.ProjError):
        projected = []
        for sx, sy in zip(x, y, strict=True):
            try:
                point = operation.transform(sx, sy, errcheck=True)
            except (ValueError, pyproj.exceptions.ProjError):
                point = (np.nan, np.nan)
            projected.append(point)
        return np.asarray(projected, dtype=float).reshape(-1, 2).T


def sample_points(project, rows, spec, layers, *, excluded_rows=None):
    ids = [row.id for row in rows]
    index = pd.Index(ids, dtype=object)
    rejected = pd.Series(None, index=index, dtype=object)
    if spec.location_mode == "columns":
        if not spec.x_id or not spec.y_id or spec.x_id == spec.y_id:
            raise ValueError("Выберите разные колонки X и Y")
        source = crs(spec.source_crs)
        check_coordinate_units(project, spec, source)
        frame = pd.DataFrame([row.values for row in rows], index=index,
                             columns=[spec.x_id, spec.y_id], dtype=object)
        x_data = _map_numbers(project, frame, spec.x_id, spec)
        y_data = _map_numbers(project, frame, spec.y_id, spec)
        coordinates = pd.DataFrame({"x": x_data["number"], "y": y_data["number"]}, dtype=object)
        missing_x = coordinates["x"].isna()
        missing_y = ~missing_x & coordinates["y"].isna()
        rejected.loc[missing_x] = "coordinate:" + x_data.loc[missing_x, "status"]
        rejected.loc[missing_y] = "coordinate:" + y_data.loc[missing_y, "status"]
    elif spec.location_mode == "layer":
        layer = next((item for item in layers if item["id"] == spec.point_layer), None)
        if layer is None or not spec.sample_key or not spec.layer_key:
            raise ValueError("Выберите точечный слой и поля точного сопоставления")
        if spec.sample_key not in {column.id for column in project.columns}:
            raise ValueError("Колонка ключа проб отсутствует")
        source = crs(layer["crs"])
        features = []
        for feature in layer["features"]:
            geometry = shape(feature["geometry"])
            key = feature["properties"].get(spec.layer_key)
            if key is not None:
                features.append((category_key(key), geometry))
        locations = pd.DataFrame(features, columns=["key", "geometry"], dtype=object)
        duplicates = locations["key"].duplicated(keep=False)
        unique = locations.loc[~duplicates]
        unique = unique.loc[unique["geometry"].map(lambda geometry: geometry.geom_type == "Point")]
        sample_keys = pd.Series([row.values.get(spec.sample_key) for row in rows], dtype=object)
        keys = pd.DataFrame({"id": ids, "key": category_keys(sample_keys)}, dtype=object)
        joined = keys.merge(unique, on="key", how="left", sort=False, validate="many_to_one", indicator=True).set_index("id")
        matched = joined['_merge'].eq('both')
        rejected.loc[joined.index[~matched]] = "unmatched_or_ambiguous_location"
        if excluded_rows is not None and not matched.all():
            unmatched = joined.loc[~matched, 'key']
            known = set(locations['key'])
            known.update(category_key(feature['properties'][spec.layer_key]) for feature in layer.get('attributes', [])
                         if feature['properties'].get(spec.layer_key) is not None)
            reasons = pd.Series('location_key_unmatched', index=unmatched.index)
            reasons.loc[unmatched.isin(known)] = 'location_geometry_missing'
            reasons.loc[unmatched.isin(locations.loc[duplicates, 'key'])] = 'location_key_duplicate'
            reasons.loc[unmatched.eq(category_key(None))] = 'location_key_missing'
            _record_exclusions(excluded_rows, reasons)
        geometries = joined.loc[matched, 'geometry']
        coordinates = pd.DataFrame({'x': geometries.map(lambda geometry: geometry.x),
                                    'y': geometries.map(lambda geometry: geometry.y)}, dtype=object).reindex(index)
    else:
        raise ValueError("Неизвестный способ привязки координат")
    operation = transformer(source, spec.target_crs)
    if source.is_geographic:
        xlimit, ylimit = geographic_limits(source)
        active = coordinates.loc[rejected.isna()]
        valid = active["x"].between(-xlimit, xlimit) & active["y"].between(-ylimit, ylimit)
        rejected.loc[active.index[~valid]] = "invalid_longitude_latitude"
    active = coordinates.loc[rejected.isna()]
    x, y = active["x"].to_numpy(dtype=float), active["y"].to_numpy(dtype=float)
    px, py = _transform_coordinates(operation, x, y)
    valid = np.isfinite(px) & np.isfinite(py)
    rejected.loc[active.index[~valid]] = "coordinate_transform_error"
    projected = dict(zip(active.index[valid], zip(px[valid], py[valid]), strict=True))
    points = [{"id": row.id, "row_ids": [row.id], "x": float(projected[row.id][0]), "y": float(projected[row.id][1]),
               "values": deepcopy(row.values), "label": str(row.values.get(spec.label_by) or "")}
              for row in rows if row.id in projected]
    excluded = rejected.dropna().value_counts(sort=False).to_dict()
    _record_exclusions(excluded_rows, rejected[rejected.ne('unmatched_or_ambiguous_location')])
    return (
        points,
        dict(excluded),
        {
            "source_crs": source.to_wkt(),
            "target_crs": crs(spec.target_crs).to_wkt(),
            "operation": operation.description,
            "accuracy_m": operation.accuracy,
            "axis_order": "always_xy",
            "network": False,
        },
    )


def site_aggregates(project, points, spec, layers):
    if spec.aggregate not in ("mean", "median", "min", "max"):
        raise ValueError("Агрегация: mean, median, min или max")
    layer = next((item for item in layers if item["id"] == spec.site_layer), None)
    if layer is None:
        raise ValueError("Выберите полигональный слой площадок")
    layer = dict(layer, features=filtered_features(layer))
    sites = []
    projected = project_geometries(layer_frame(layer).geometry, layer['crs'], spec.target_crs)
    for feature, geometry in zip(layer["features"], projected):
        if geometry.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError("Слой площадок должен содержать только полигоны")
        sites.append(
            {
                "id": feature["id"],
                "geometry": mapping(geometry),
                "properties": deepcopy(feature["properties"]),
                "points": [],
                "n_boundary": 0,
            }
        )
    geometries = [shape(site["geometry"]) for site in sites]
    excluded = Counter()
    locations = gpd.GeoDataFrame(geometry=gpd.points_from_xy(
        [p['x'] for p in points], [p['y'] for p in points], crs=spec.target_crs,
    ))
    polygons = gpd.GeoDataFrame(geometry=geometries, crs=spec.target_crs)
    joined = gpd.sjoin(locations, polygons, how='left', predicate='intersects')
    matches = summarize_polygon_matches(joined, polygons.geometry)
    for point_index, count, polygon, boundary in matches.itertuples(name=None):
        if count == 1:
            site = sites[int(polygon)]
            site['points'].append(points[point_index])
            site['n_boundary'] += int(boundary)
        else:
            excluded['ambiguous_site' if count else 'outside_sites'] += 1
    row_map = project.row_map()
    records = []
    for site_index, site in enumerate(sites):
        for point in site["points"]:
            row = row_map[point["id"]]
            status, value = number(project, row, spec.site_value, spec) if spec.site_value else (None, None)
            family = row.values.get(spec.site_family) if spec.site_family else None
            records.append((site_index, point["id"], value, status, category_key(family) if family is not None else None))
    frame = pd.DataFrame(records, columns=["site", "id", "value", "status", "family"], dtype=object)
    frame["value"] = frame["value"].astype(float)
    aggregates = frame.groupby("site", sort=False).agg(
        value=("value", spec.aggregate), n_valid=("value", "count"), row_ids=("id", list),
    ).to_dict("index")
    invalid_counts = frame.loc[frame["value"].isna() & frame["status"].notna()].groupby(
        ["site", "status"], sort=False,
    ).size()
    family_counts = frame.loc[frame["family"].notna()].groupby(["site", "family"], sort=False).size()
    for site_index, (site, geometry) in enumerate(zip(sites, geometries)):
        aggregate = aggregates.get(site_index, {"value": None, "n_valid": 0, "row_ids": []})
        value = aggregate["value"] if aggregate["n_valid"] else None
        invalid = invalid_counts.loc[site_index].to_dict() if site_index in invalid_counts.index.get_level_values(0) else {}
        families = family_counts.loc[site_index].to_dict() if site_index in family_counts.index.get_level_values(0) else {}
        center = geometry.representative_point()
        winners = (
            [key for key, count in families.items() if count == max(families.values())]
            if families
            else []
        )
        site.update(
            x=center.x,
            y=center.y,
            row_ids=aggregate["row_ids"],
            label=str(site["properties"].get(spec.site_label) or site["id"]),
            n_samples=0,
            n_valid=int(aggregate["n_valid"]),
            method=spec.aggregate,
            excluded=dict(invalid),
            value=float(value) if value is not None else None,
            family=json.loads(winners[0])[1] if len(winners) == 1 else None,
            family_status="mode"
            if len(winners) == 1
            else "tie"
            if winners
            else "missing",
            family_counts=dict(families),
        )
        site.pop("points")
        site["n_samples"] = len(site["row_ids"])
        if site["value"] is not None and not np.isfinite(site["value"]):
            raise ValueError("Численное переполнение при агрегации площадки")
    return sites, dict(excluded)


def _point_category_keys(points, column_id):
    return sorted({category_key(point['values'].get(column_id)) for point in points})


def default_point_category(channel, key, index):
    value = json.loads(key)[1]
    return {
        'label': str(value) if value is not None else 'Нет значения',
        'style': COLORS[index % len(COLORS)] if channel == 'color' else [6, 9, 12, 15][index % 4],
    }


MAP_LAYER_DATA = {'id', 'features', 'attributes', 'fields', 'source', 'diagnostics', 'crs'}


def saved_map_layers(snapshot, layers, *, keep_extra=False):
    """Restore a view's presentation using the project's current layer data."""
    spec = MapSpec.from_dict(snapshot['spec'])
    by_id = {layer['id']: layer for layer in layers}
    required = {layer['id'] for layer in snapshot['layers'] if layer.get('visible', True)}
    if spec.location_mode == 'layer':
        required.add(spec.point_layer)
    if spec.location_mode != 'layers' and (spec.location_mode == 'layer' or spec.x_id or spec.y_id):
        required.add(spec.site_layer)
    missing = (required - {None, ''}) - by_id.keys()
    ordered = []
    for saved in snapshot['layers']:
        if saved['id'] in by_id:
            data = {key: value for key, value in by_id[saved['id']].items() if key in MAP_LAYER_DATA}
            ordered.append({**data, **deepcopy(saved)})
    if keep_extra:
        saved_ids = {layer['id'] for layer in snapshot['layers']}
        ordered.extend(dict(layer, visible=False) for layer in layers if layer['id'] not in saved_ids)
    return spec, ordered, missing


def build_map(project, row_ids, spec, layers):
    spec = resolve_crs(project, spec, layers)
    crs(spec.target_crs)
    for key, value in spec.to_dict().items():
        if isinstance(value, (float, int)) and not math.isfinite(value):
            raise ValueError(f"{key}: требуется конечное число")
    for color in (
        spec.color,
        spec.missing_color,
        spec.edge_color,
        spec.background,
        spec.grid_color,
    ):
        to_rgba(color)
    for layer in layers:
        to_rgba(layer["color"])
        to_rgba(layer["fill"])
        if not (
            np.isfinite(layer["alpha"])
            and 0 <= layer["alpha"] <= 1
            and np.isfinite(layer["line_width"])
            and layer["line_width"] > 0
        ):
            raise ValueError("У слоя alpha 0–1 и положительная толщина линии")
    if not (
        0 <= spec.alpha <= 1
        and 0 < spec.size_min <= spec.size_max
        and spec.marker_size > 0
    ):
        raise ValueError("Некорректный размер или прозрачность символов")
    validate_marker(spec.marker)
    if (
        spec.edge_width < 0
        or spec.grid_width <= 0
        or not 0 <= spec.grid_alpha <= 1
    ):
        raise ValueError("Проверьте символ, обводку и сетку карты")
    if spec.color_mode not in ("category", "numeric") or spec.size_mode not in (
        "category",
        "numeric",
    ):
        raise ValueError("Оформление: category или numeric")
    colormaps[spec.cmap]
    only_layers = spec.location_mode == 'layers'
    excluded_rows = {}
    if only_layers:
        rows, filtered, points, excluded = [], {}, [], {}
        operation = dict(source_crs=None, target_crs=crs(spec.target_crs).to_wkt(), network=False, axis_order='always_xy')
    else:
        rows, filtered = filter_rows(project, row_ids, spec, excluded_rows=excluded_rows)
        points, excluded, operation = sample_points(project, rows, spec, layers, excluded_rows=excluded_rows)
    row_map = project.row_map()
    for point in points:
        point["color_value"] = (
            number(project, row_map[point["id"]], spec.color_by, spec)[1]
            if spec.color_by and spec.color_mode == "numeric"
            else None
        )
        point["size_value"] = (
            number(project, row_map[point["id"]], spec.size_by, spec)[1]
            if spec.size_by and spec.size_mode == "numeric"
            else None
        )
    sites, site_excluded = (
        site_aggregates(project, points, spec, layers) if spec.site_layer and not only_layers else ([], {})
    )
    if spec.show_aggregates and not spec.site_layer:
        raise ValueError("Для отображения агрегатов выберите площадки")
    point_categories = {}
    for channel, cid, mode in (
        ("color", spec.color_by, spec.color_mode),
        ("size", spec.size_by, spec.size_mode),
    ):
        if cid and mode == "category" and not only_layers and not spec.show_aggregates:
            keys = _point_category_keys(points, cid)
            if len(keys) > 100:
                raise ValueError("Категориальное оформление текущей карты ограничено 100 значениями")
            entries = spec.style_maps.setdefault(channel, {}).setdefault(cid, {})
            point_categories[channel] = keys
            for key in keys:
                if key not in entries:
                    entries[key] = default_point_category(channel, key, len(entries))
                entry = entries[key]
                if channel == "color":
                    to_rgba(entry["style"])
                elif (
                    not np.isfinite(float(entry["style"])) or float(entry["style"]) <= 0
                ):
                    raise ValueError(
                        "Размер категории должен быть положительным и конечным"
                    )
    return {
        "format": "autopyro.map",
        "version": 1,
        "created_at": timestamp(),
        "project_id": project.id,
        "source": deepcopy(project.source),
        "columns": [
            {
                "id": c.id,
                "name": c.name,
                "unit": c.unit,
                "kind": c.kind,
                "interpolated": bool(c.metadata.get("method")),
            }
            for c in project.columns
        ],
        "spec": spec.to_dict(),
        "row_ids": [r.id for r in rows],
        "points": points,
        "point_categories": point_categories,
        "sites": sites,
        "layers": deepcopy(layers),
        "diagnostics": {
            "input_rows": len(row_ids),
            "filtered": filtered,
            "coordinates": excluded,
            "excluded_rows": excluded_rows,
            "site_assignment": site_excluded,
            "layers": {layer['id']: dict(layer['diagnostics']) for layer in layers if layer.get('diagnostics')},
        },
        "crs_operation": operation,
        "versions": {
            "autopyro": __version__,
            "numpy": np.__version__,
            "geopandas": gpd.__version__,
            "pyproj": pyproj.__version__,
            "PROJ": pyproj.proj_version_str,
            "pyogrio": pyogrio.__version__,
            "GDAL": str(pyogrio.__gdal_version__),
            "shapely": shapely.__version__,
        },
    }


def _polygon_path(geometry):
    from shapely.geometry.polygon import orient

    polygon = orient(geometry, sign=1)
    vertices, codes = [], []
    for ring in [polygon.exterior, *polygon.interiors]:
        coords = [(x, y) for x, y, *_ in ring.coords]
        vertices.extend(coords)
        codes.extend(
            [MplPath.MOVETO]
            + [MplPath.LINETO] * (len(coords) - 2)
            + [MplPath.CLOSEPOLY]
        )
    return MplPath(vertices, codes)


def geometry_artists(ax, geometry, style, point_buffer=None):
    if geometry.geom_type == "Point":
        if point_buffer is not None:
            point_buffer.append((geometry.x, geometry.y))
        else:
            ax.scatter(
                [geometry.x],
                [geometry.y],
                s=25,
                c=style["color"],
                alpha=style["alpha"],
                zorder=style.get('zorder', 1),
            )
    elif geometry.geom_type in ("LineString", "LinearRing"):
        x, y = geometry.xy
        ax.plot(
            x,
            y,
            color=style["color"],
            alpha=style["alpha"],
            linewidth=style["line_width"],
            zorder=style.get('zorder', 1),
        )
    elif geometry.geom_type == "Polygon":
        ax.add_patch(
            PathPatch(
                _polygon_path(geometry),
                facecolor=style["fill"],
                edgecolor=style["color"],
                alpha=style["alpha"],
                linewidth=style["line_width"],
                zorder=style.get('zorder', 0),
            )
        )
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            geometry_artists(ax, part, style, point_buffer)


def layer_categories(layer):
    field = layer.get('color_by')
    if not field or layer.get('color_mode') == 'numeric':
        return {}
    keys = sorted({category_key(feature['properties'].get(field)) for feature in layer['features']})
    overrides = layer.get('category_styles', {}).get(field, {})
    result = {}
    for i, key in enumerate(keys):
        entry = overrides.get(key, {})
        label = str(json.loads(key)[1]) if json.loads(key)[1] is not None else 'Нет значения'
        color = colormaps['turbo'](i / max(1, len(keys) - 1))
        result[key] = (entry.get('label', label), to_rgba(entry.get('color', color)))
    return result


def _draw_map_legends(ax, report, spec, items, layer_handles, numeric_legends):
    figure = ax.figure
    handles = []
    if (
        spec.legend
        and spec.color_by
        and spec.color_mode == "category"
        and not spec.show_aggregates
        and items
    ):
        import textwrap

        entries = spec.style_maps["color"][spec.color_by]
        keys = report.get('point_categories', {}).get('color')
        if keys is None:
            keys = _point_category_keys(items, spec.color_by)
        handles = [
            Line2D(
                [],
                [],
                marker=spec.marker,
                linestyle="none",
                color=entries[key]["style"],
                label=textwrap.fill(entries[key]["label"], 18),
            )
            for key in keys
        ]
    legend_artist = None
    if spec.legend and (handles or layer_handles):
        legend_artist = ax.legend(
            handles=handles + layer_handles,
            title=spec.legend_title or None,
            loc={'outside': 'upper left', 'upper_right': 'upper right', 'lower_left': 'lower left', 'lower_right': 'lower right'}.get(spec.legend_placement, 'upper left'),
            **(dict(bbox_to_anchor=(.73, .88), bbox_transform=figure.transFigure) if spec.legend_placement == 'outside' else {}),
            prop={"family": spec.font_family, "size": spec.legend_size},
        )
    if spec.legend and numeric_legends:
        top = .86
        if legend_artist is not None and spec.legend_placement == 'outside':
            top -= legend_artist.get_window_extent(renderer_for(figure)).height / figure.bbox.height + .03
        spacing = min(.14, max(.06, (top - .16) / len(numeric_legends)))
        for index, (label, mapper) in enumerate(numeric_legends):
            bar_ax = figure.add_axes([.75, top - (index + 1) * spacing, .21, .018])
            bar = figure.colorbar(mapper, cax=bar_ax, orientation='horizontal')
            bar.set_label(label, fontsize=spec.legend_size, fontfamily=spec.font_family)
            bar.ax.tick_params(labelsize=spec.legend_size)


def render_map(report, *, figure=None, selected_ids=(), show_footer=True):
    spec = MapSpec.from_dict(report["spec"])
    if not (
        30 <= spec.width_mm <= 1000
        and 30 <= spec.height_mm <= 1000
        and 36 <= spec.dpi <= 1200
        and 5 <= spec.legend_size <= 30
    ):
        raise ValueError("Размер карты 30–1000 мм, DPI 36–1200, легенда 5–30 pt")
    figure = figure or Figure(
        figsize=(spec.width_mm / 25.4, spec.height_mm / 25.4), dpi=96
    )
    figure.clear()
    figure.set_facecolor(spec.background)
    ax = figure.add_subplot(111)
    ax.set_facecolor(spec.background)
    figure.subplots_adjust(
        left=0.13,
        right=0.70
        if spec.legend and (any(layer.get('visible', True) and (layer.get('color_by') or layer.get('raster')) for layer in report['layers'])
                            or (spec.color_by and spec.color_mode == 'category' and not spec.show_aggregates and report['points']))
        else 0.84,
        bottom=0.17,
        top=0.88,
    )
    layer_features = []
    layer_handles = []
    numeric_legends, label_candidates = [], []
    for layer_index, layer in enumerate(reversed(report["layers"])):
        if not layer.get("visible", True):
            continue
        if 'raster' in layer:
            from .surfaces import raster_coordinates
            raster = layer['raster']
            x, y = raster_coordinates(layer, spec.target_crs)
            values = np.ma.masked_invalid(np.asarray(raster['values'], dtype=float))
            categories = raster.get('categories')
            cmap, norm = layer.get('cmap', 'viridis'), None
            if categories:
                colors = raster['colors']
                cmap = ListedColormap(colors)
                norm = BoundaryNorm(np.arange(len(colors) + 1) - .5, len(colors))
                layer_handles.extend(Patch(facecolor=color, label=label) for color, label in zip(colors, categories))
            elif raster.get('limits'):
                norm = Normalize(*raster['limits'])
            mesh = ax.pcolormesh(x, y, values, shading='flat', cmap=cmap, norm=norm,
                                 alpha=layer['alpha'], zorder=1 + layer_index / max(1, len(report['layers'])))
            if not categories:
                label = layer['name'] + (f" [{raster['unit']}]" if raster['unit'] else '')
                numeric_legends.append((label, mesh))
            continue
        point_buffer = []
        point_colors = []
        features = filtered_features(layer)
        categories = layer_categories(layer)
        if categories and not layer.get('legend_hidden', True):
            visible_keys = {category_key(f['properties'].get(layer['color_by'])) for f in features}
            categories = {key: entry for key, entry in categories.items() if key in visible_keys}
        numeric_colors, mapper = numeric_layer_style(layer, features) if layer.get('color_by') and layer.get('color_mode') == 'numeric' else ({}, None)
        if mapper is not None:
            numeric_legends.append((f"{layer['name']} · {layer['color_by']}", mapper))
        import textwrap
        layer_handles.extend(Patch(facecolor=color, edgecolor=layer['color'], alpha=layer['alpha'],
                                   label=textwrap.fill(f"{layer['name']} · {layer['color_by']}: {label}", 26))
                             for label, color in categories.values())
        projected = project_geometries(layer_frame(dict(layer, features=features)).geometry, layer['crs'], spec.target_crs)
        # Batch polygon layers while keeping the drawing order of mixed layers.
        batch_polygons = projected.geom_type.isin(('Polygon', 'MultiPolygon')).all()
        paths, facecolors, edgecolors = [], [], []
        zorder = 1 + layer_index / max(1, len(report['layers']))
        for feature, geometry in zip(features, projected):
            layer_features.append((layer['id'], str(feature['id']), geometry))
            color = numeric_colors.get(str(feature['id']), categories[category_key(feature['properties'].get(layer['color_by']))][1] if categories else layer['color'])
            style = dict(layer, color=color, fill=color) if categories or numeric_colors else dict(layer)
            style['zorder'] = zorder
            text = feature['properties'].get(layer.get('label_by'))
            if text is not None and str(text).strip():
                label_candidates.append((geometry, str(text), int(layer.get('label_size', 8))))
            if batch_polygons:
                polygons = geometry.geoms if geometry.geom_type == 'MultiPolygon' else (geometry,)
                for polygon in polygons:
                    paths.append(_polygon_path(polygon))
                    facecolors.append(to_rgba(style['fill'], layer['alpha']))
                    edgecolors.append(to_rgba(style['color'], layer['alpha']))
                continue
            previous_count = len(point_buffer)
            geometry_artists(
                ax,
                geometry,
                style,
                point_buffer,
            )
            point_colors.extend([color] * (len(point_buffer) - previous_count))
        if paths:
            ax.add_collection(PathCollection(paths, facecolors=facecolors, edgecolors=edgecolors,
                                             linewidths=layer['line_width'], zorder=zorder))
        if point_buffer:
            ax.scatter(
                *np.array(point_buffer).T,
                s=25,
                c=point_colors,
                alpha=layer["alpha"],
                zorder=zorder,
            )
    items = report["sites"] if spec.show_aggregates else report["points"]
    if spec.show_aggregates:
        values = [s["value"] for s in items]
        colors = (
            values
            if any(v is not None for v in values)
            else [spec.missing_color] * len(items)
        )
        counts = [math.sqrt(s["n_samples"]) for s in items]
        sizes = [
            (
                spec.size_min
                + (spec.size_max - spec.size_min)
                * (n - min(counts))
                / (max(counts) - min(counts))
                if max(counts) > min(counts)
                else spec.marker_size
            )
            ** 2
            for n in counts
        ]
        norm = (
            Normalize(
                spec.color_min
                if spec.color_min is not None
                else min(v for v in values if v is not None),
                spec.color_max
                if spec.color_max is not None
                else max(v for v in values if v is not None),
            )
            if any(v is not None for v in values)
            else None
        )
    else:
        colors, sizes = [], []
        numeric_colors = [p["color_value"] for p in items]
        numbers = [v for v in numeric_colors if v is not None]
        norm = (
            Normalize(
                spec.color_min if spec.color_min is not None else min(numbers),
                spec.color_max if spec.color_max is not None else max(numbers),
            )
            if numbers and spec.color_mode == "numeric" and spec.color_by
            else None
        )
        size_values = [p["size_value"] for p in items]
        numeric_sizes = [v for v in size_values if v is not None]
        for i, point in enumerate(items):
            if spec.color_by and spec.color_mode == "category":
                entry = spec.style_maps["color"][spec.color_by][
                    category_key(point["values"].get(spec.color_by))
                ]
                colors.append(entry["style"])
            else:
                colors.append(
                    numeric_colors[i]
                    if norm
                    else spec.color
                    if not spec.color_by
                    else spec.missing_color
                )
            diameter = spec.marker_size
            if spec.size_by and spec.size_mode == "category":
                diameter = float(
                    spec.style_maps["size"][spec.size_by][
                        category_key(point["values"].get(spec.size_by))
                    ]["style"]
                )
            elif (
                spec.size_by
                and size_values[i] is not None
                and numeric_sizes
                and max(numeric_sizes) > min(numeric_sizes)
            ):
                diameter = spec.size_min + (spec.size_max - spec.size_min) * (
                    size_values[i] - min(numeric_sizes)
                ) / (max(numeric_sizes) - min(numeric_sizes))
            sizes.append(diameter**2)
    kwargs = {}
    if norm is not None:
        if norm.vmin == norm.vmax:
            # A colorbar would expand a constant range on its own.
            padding = abs(norm.vmin) * 0.1 or 0.1
            norm.vmin, norm.vmax = norm.vmin - padding, norm.vmax + padding
        colors = np.array(
            [np.nan if value is None else value for value in colors], dtype=float
        )
        kwargs = {
            "norm": norm,
            "cmap": colormaps[spec.cmap].with_extremes(bad=spec.missing_color),
            "plotnonfinite": True,
        }
    artist = ax.scatter(
        [p["x"] for p in items],
        [p["y"] for p in items],
        s=sizes,
        c=colors,
        alpha=spec.alpha,
        marker=spec.marker,
        edgecolors=spec.edge_color,
        linewidths=spec.edge_width,
        picker=5,
        zorder=4,
        **kwargs,
    )
    artist._autopyro_members = [p["row_ids"] for p in items]
    if spec.legend and norm is not None and items:
        cid = spec.site_value if spec.show_aggregates else spec.color_by
        column = next(c for c in report["columns"] if c["id"] == cid)
        figure.colorbar(
            artist,
            ax=ax,
            pad=0.02,
            label=(spec.aggregate + " · " if spec.show_aggregates else "")
            + column["name"]
            + f" [{column['unit']}]",
        )
    _draw_map_legends(ax, report, spec, items, layer_handles, numeric_legends)
    selected = set(selected_ids)
    for item, size in zip(items, sizes):
        if selected.intersection(item["row_ids"]):
            ax.scatter(
                [item["x"]],
                [item["y"]],
                s=size + 70,
                facecolors="none",
                edgecolors="#d62728",
                linewidths=1.5,
                zorder=6,
            )
        if spec.show_aggregates:
            ax.annotate(
                item["label"]
                + "\n"
                + (
                    f"N={item['n_samples']}, n={item['n_valid']}\n{item['method']}: {item['value']:.4g}"
                    if item["value"] is not None
                    else f"N={item['n_samples']}, n=0"
                ),
                (item["x"], item["y"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=spec.annotation_size,
            )
        elif item["label"]:
            ax.annotate(
                item["label"],
                (item["x"], item["y"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=spec.annotation_size,
            )
    ax.autoscale_view()
    if spec.xmin is not None or spec.xmax is not None:
        ax.set_xlim(left=spec.xmin, right=spec.xmax)
    if spec.ymin is not None or spec.ymax is not None:
        ax.set_ylim(bottom=spec.ymin, top=spec.ymax)
    target = crs(spec.target_crs)
    angular = (
        "°"
        if target.axis_info[0].unit_name == "degree"
        else target.axis_info[0].unit_name
    )
    ax.set_xlabel(
        spec.x_label
        or (
            f"Долгота, {angular}"
            if target.is_geographic
            else f"X, {target.axis_info[0].unit_name}"
        ),
        fontsize=spec.label_size,
        fontfamily=spec.font_family,
    )
    ax.set_ylabel(
        spec.y_label
        or (
            f"Широта, {angular}"
            if target.is_geographic
            else f"Y, {target.axis_info[1].unit_name}"
        ),
        fontsize=spec.label_size,
        fontfamily=spec.font_family,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(
        spec.grid,
        alpha=spec.grid_alpha,
        color=spec.grid_color,
        linewidth=spec.grid_width,
    )
    ax.tick_params(labelsize=spec.tick_size)
    ax.set_title(spec.title, fontfamily=spec.font_family, fontsize=spec.label_size + 2)
    footer = None
    if show_footer:
        crs_name = crs_label(target)
        footer = figure.text(
            0.13,
            0.025,
            f"{crs_name} | проб: {len(report['points'])} | {report['created_at']}\nЛокальные данные; выборка и исключения сохранены в снимке карты",
            fontsize=7,
            fontfamily=spec.font_family,
            verticalalignment='bottom',
        )
    fit_map_vertical_margins(ax, footer)
    layer_labels = add_layer_labels(ax, reversed(label_candidates), spec.font_family)
    decorations = add_map_decorations(ax, target, scale_bar=spec.scale_bar, north_arrow=spec.north_arrow, font=spec.font_family) if spec.scale_bar or spec.north_arrow else {}
    feature_index = shapely.STRtree([geometry for _, _, geometry in layer_features])
    return figure, {
        'axes': ax, 'artist': artist, 'items': items,
        'layer_features': layer_features, 'feature_index': feature_index,
        'layer_labels': layer_labels, 'decorations': decorations,
        '_label_candidates': label_candidates, 'footer': footer,
    }


def _write_map(path, report):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in ('.json', '.geojson'):
        # Filtered sample identities are for navigation in the source project.
        report = dict(report, diagnostics={key: value for key, value in report['diagnostics'].items()
                                           if key != 'excluded_rows'})
    if suffix == ".json":
        path.write_text(
            json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2),
            encoding="utf-8",
        )
    elif suffix == ".geojson":
        features = []
        for point in report["points"]:
            features.append(
                {
                    "type": "Feature",
                    "id": point["id"],
                    "geometry": mapping(Point(point["x"], point["y"])),
                    "properties": {
                        "row_ids": point["row_ids"],
                        "values": point["values"],
                    },
                }
            )
        for site in report["sites"]:
            features.append(
                {
                    "type": "Feature",
                    "geometry": deepcopy(site['geometry']),
                    "properties": {
                        k: v for k, v in site.items() if k not in ("geometry", "x", "y")
                    },
                }
            )
        geometries = project_geometries(
            [shape(feature['geometry']) for feature in features], report['spec']['target_crs'], 'EPSG:4326',
        )
        for feature, geometry in zip(features, geometries):
            feature['geometry'] = mapping(geometry)
        if report['spec']['location_mode'] == 'layers':
            for layer in report['layers']:
                if not layer.get('visible', True) or 'raster' in layer:
                    continue
                visible = filtered_features(layer)
                geometries = project_geometries(
                    [shape(feature['geometry']) for feature in visible], layer['crs'], 'EPSG:4326',
                )
                features.extend(
                    dict(feature, id=f"{layer['id']}:{feature['id']}", geometry=mapping(geometry))
                    for feature, geometry in zip(visible, geometries, strict=True)
                )
        if not features:
            raise ValueError('Нет объектов для экспорта в GeoJSON')
        path.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": features,
                    "autopyro": {
                        "columns": report["columns"],
                        "diagnostics": report["diagnostics"],
                        "spec": report["spec"],
                        "created_at": report["created_at"],
                    },
                },
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    elif suffix in (".png", ".svg", ".pdf", ".tif", ".tiff"):
        spec = MapSpec.from_dict(report["spec"])
        if spec.width_mm * spec.height_mm * (spec.dpi / 25.4) ** 2 > 64_000_000:
            raise ValueError("Размер экспорта превышает 64 мегапикселя")
        figure, _ = render_map(report)
        figure.savefig(path, dpi=spec.dpi)
        figure.clear()
    else:
        raise ValueError("Экспорт карты: PNG, SVG, PDF, TIFF, JSON или GeoJSON")


def export_map(path, report):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=path.suffix)
    os.close(fd)
    try:
        _write_map(Path(temporary), report)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
