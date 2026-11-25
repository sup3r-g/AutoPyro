"""Serializable plotting specification and a Qt-independent Matplotlib renderer."""

from __future__ import annotations

import json
import math
import textwrap
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.markers import MarkerStyle
from matplotlib.colors import BoundaryNorm, LogNorm, Normalize, to_rgba
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.ticker import (
    AutoMinorLocator,
    MultipleLocator,
    NullLocator,
    LogLocator,
    ScalarFormatter,
)

from autopyro.model.core import Project, parse_number
from autopyro.model.tabular import classify, project_frame, classify_series, exclusion_mask
from autopyro.geology.chronostratigraphic_colors import stratigraphic_color
from .plot_catalog import COLORS, MARKERS, KINDS, MERGED_KINDS, colormaps, validate_marker


from .plot_inputs import native_columns
from .plot_passports import TEMPLATES


@dataclass
class AxisSpec:
    label: str = ""
    minimum: float | None = None
    maximum: float | None = None
    scale: str = "linear"
    log_base: float = 10
    inverted: bool = False
    label_size: int = 11
    tick_size: int = 9
    color: str = "#222222"
    rotation: float = 0
    major_step: float | None = None
    minor_divisions: int = 2
    number_format: str = "auto"
    major_grid: bool = True
    minor_grid: bool = False
    label_pad: float = 4


@dataclass
class PlotSpec:
    kind: str = "scatter"
    render_language: str = "ru"
    x_id: str = ""
    y_id: str = ""
    z_id: str = ""
    library_bindings: dict[str, str] = field(default_factory=dict)
    library_options: str = "{}"
    depth_bottom_id: str = ""
    interval_width: float = 2
    track_ids: list[str] = field(default_factory=list)
    track_axes: dict[str, AxisSpec] = field(default_factory=dict)
    x_axis: AxisSpec = field(default_factory=AxisSpec)
    y_axis: AxisSpec = field(default_factory=AxisSpec)
    z_axis: AxisSpec = field(default_factory=AxisSpec)
    camera_3d: tuple[float, float, float] | None = None
    scope: str = "filtered"
    text_numbers: bool = False
    include_interpolated: bool = False
    include_calculated: bool = True
    include_original: bool = True
    filter_rules: list[dict] = field(default_factory=list)
    filter_combine: str = "AND"
    title: str = ""
    subtitle: str = ""
    title_size: int = 13
    font_family: str = "DejaVu Sans"
    color_by: str = ""
    shape_by: str = ""
    size_by: str = ""
    color_mode: str = "category"
    size_mode: str = "numeric"
    style_maps: dict = field(default_factory=dict)
    color: str = "#0072B2"
    edge_color: str = "#202020"
    marker: str = "o"
    marker_size: float = 7
    edge_width: float = 0.6
    alpha: float = 0.8
    missing_color: str = "#999999"
    size_min: float = 4
    size_max: float = 14
    color_min: float | None = None
    color_max: float | None = None
    size_value_min: float | None = None
    size_value_max: float | None = None
    cmap: str = "viridis"
    colorbar_label: str = ""
    color_scale: str = "linear"
    color_class_edges: str = ""
    colorbar_location: str = "right"
    legend: bool = True
    legend_position: str = "best"
    legend_order: str = "default"
    legend_columns: int = 1
    legend_size: int = 8
    legend_frame: bool = True
    legend_visible_only: bool = False
    grid_color: str = "#b0b0b0"
    grid_alpha: float = 0.45
    grid_width: float = 0.6
    grid_style: str = "--"
    grid_below: bool = True
    background: str = "#ffffff"
    width_mm: float = 180
    height_mm: float = 130
    square_plot_area: bool = False
    data_aspect: float | None = None
    dpi: int = 300
    transparent: bool = False
    left: float = 0.14
    right: float = 0.95
    bottom: float = 0.16
    top: float = 0.84
    wspace: float = 0.35
    bins: int = 20
    density: bool = False
    connect: bool = False
    line_group_by: str = ""
    connect_across_gaps: bool = True
    line_width: float = 1
    label_by: str = ""
    annotation: str = ""
    annotations: list[dict] = field(default_factory=list)
    label_size: int = 8
    show_uncertainty: bool = True
    measurement_error_bars: bool = False
    qc_outline: bool = False
    show_result_type: bool = False
    curve_archive_id: str = ""
    passport_id: str = ""
    interpretation_panels: list[dict] = field(default_factory=list)
    comparison_preview_panel: str | None = None
    comparison_grid: bool = False
    interpretation: dict | None = None
    interpretation_alpha: float = 0.18
    interpretation_labels: bool = True
    interpretation_info: bool = True
    interpretation_source_frame: bool = False
    stratigraphy: dict | None = None
    stratigraphy_alpha: float = 0.16
    stratigraphy_labels: bool = True
    stratigraphy_info: bool = True
    stratigraphy_label_size: int = 8

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        panels = data.get('interpretation_panels', [])
        if (not isinstance(panels, list) or panels and len(panels) not in (3, 4)
                or any(not isinstance(panel, dict) or not isinstance(panel.get('name'), str)
                       or not ('binding' in panel or 'error' in panel) for panel in panels)):
            raise ValueError('Некорректное описание диаграмм сравнения')
        data["x_axis"] = AxisSpec(**data.get("x_axis", {}))
        data["y_axis"] = AxisSpec(**data.get("y_axis", {}))
        data["z_axis"] = AxisSpec(**data.get("z_axis", {}))
        data["track_axes"] = {
            key: AxisSpec(**value) for key, value in data.get("track_axes", {}).items()
        }
        camera = data.get('camera_3d')
        if camera is not None:
            if (not isinstance(camera, (list, tuple)) or len(camera) != 3
                    or not all(isinstance(angle, (int, float)) and math.isfinite(angle) for angle in camera)):
                raise ValueError('Положение камеры должно содержать три конечных угла')
            data['camera_3d'] = tuple(camera)
        from .plot_annotations import validate_annotations
        validate_annotations(data.get('annotations', []))
        spec = cls(**data)
        from .plot_controls import merge_line_kind
        merge_line_kind(spec)
        return spec


def category_key(value):
    return json.dumps([type(value).__name__, value], ensure_ascii=False)


def category_keys(values):
    """Encode repeated strings once; mixed types retain their scalar identity."""
    if len(values) >= 256 and values.map(type).isin([str, type(None)]).all():
        return values.map({value: category_key(value) for value in values.unique()})
    return values.map(category_key)


def category_label(value):
    return "(нет данных)" if value is None or value == "" else str(value)


def ensure_style_maps(project: Project, spec: PlotSpec, *, prepared=None):
    """Assign once over the whole project, never over a filtered subset."""
    for channel, cid in (
        ("color", spec.color_by),
        ("shape", spec.shape_by),
        ("size", spec.size_by),
    ):
        if not cid:
            continue
        if (channel == "color" and spec.color_mode == "numeric") or (
            channel == "size" and spec.size_mode == "numeric"
        ):
            continue
        if channel in ("shape", "size") and spec.kind not in (
            "scatter",
            "depth",
            "multi_depth",
            "mpl:scatter",
            "mpl3d:scatter",
            "mpl3d:scatter3D",
        ):
            continue
        mapping = spec.style_maps.setdefault(channel, {}).setdefault(cid, {})
        if channel == "color":
            cycle = rcParams['axes.prop_cycle'].by_key()['color']
        if prepared is None:
            categories = project_frame(project, column_ids=[cid]).rename(columns={cid: "value"})
            categories["key"] = categories["value"].map(category_key)
            values = categories.drop_duplicates("key").itertuples(index=False, name=None)
        else:
            values = prepared.categories(cid)
        for index, (value, key) in enumerate(values):
            if channel == "color" and key in mapping:
                entry = mapping[key]
                # Compare actual colors because reopening can decode HEX as RGB.
                # The original palette was assigned in project row order.
                if "automatic" not in entry:
                    entry["automatic"] = (
                        to_rgba(entry["style"]) == to_rgba(cycle[index % len(cycle)])
                    )
                if entry["automatic"] and (color := stratigraphic_color(value)):
                    entry["style"] = color
            if key not in mapping:
                if len(mapping) >= 100:
                    raise ValueError(
                        "Более 100 категорий оформления: выберите другой показатель или числовую шкалу"
                    )
                i = len(mapping)
                style = (
                    (stratigraphic_color(value) or
                     cycle[i % len(cycle)])
                    if channel == "color"
                    else (
                        MarkerStyle.filled_markers[i % len(MarkerStyle.filled_markers)]
                        if channel == "shape"
                        else 4 + (i % 6) * 2
                    )
                )
                mapping[key] = {"label": category_label(value), "style": style}
                if channel == "color":
                    mapping[key]["automatic"] = True


def color_class_edges(spec):
    try:
        edges = [float(parse_number(part)) for part in spec.color_class_edges.split(';')]
        valid = (len(edges) >= 3 and all(map(math.isfinite, edges))
                 and all(low < high for low, high in zip(edges, edges[1:])))
    except (AttributeError, OverflowError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError('Введите не менее трёх конечных возрастающих границ классов через точку с запятой')
    if len(edges) - 1 > colormaps[spec.cmap].N:
        raise ValueError('Число классов превышает число цветов палитры')
    return edges


def validate_spec(spec):
    if not isinstance(spec.filter_rules, list) or spec.filter_combine not in ("AND", "OR"):
        raise ValueError("Некорректный локальный фильтр графика")
    if spec.data_aspect is not None and (
        type(spec.data_aspect) not in (float, int)
        or not math.isfinite(spec.data_aspect) or spec.data_aspect <= 0
    ):
        raise ValueError("Соотношение масштабов осей должно быть конечным положительным числом")
    if spec.color_scale not in ("linear", "log", "classes"):
        raise ValueError("Неизвестная нормировка цветовой шкалы")
    numeric_color = spec.color_by and spec.color_mode == "numeric"
    if spec.color_scale == "log" and numeric_color:
        if any(value is not None and value <= 0 for value in (spec.color_min, spec.color_max)):
            raise ValueError("Границы логарифмической цветовой шкалы должны быть положительными")
    if spec.legend_order not in ("default", "ascending", "descending"):
        raise ValueError("Неизвестный порядок категорий в легенде")
    if spec.colorbar_location not in ("right", "left", "top", "bottom", "none"):
        raise ValueError("Неизвестное положение цветовой шкалы")
    if spec.annotations:
        from .plot_annotations import validate_annotations
        validate_annotations(spec.annotations)
    for key, value in vars(spec).items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError(f"Параметр {key} должен быть конечным числом")
    if spec.kind not in KINDS and spec.kind not in MERGED_KINDS:
        raise ValueError("Неизвестный тип графика")
    axes = {'X': spec.x_axis, 'Y': spec.y_axis, 'Z': spec.z_axis}
    axes.update({f'X ({column})': axis for column, axis in spec.track_axes.items()})
    for name, axis in axes.items():
        if type(axis.label_pad) not in (float, int) or not 0 <= axis.label_pad <= 60:
            raise ValueError(f"Отступ подписи оси {name} должен быть от 0 до 60 pt")
        if axis.scale not in ("linear", "log"):
            raise ValueError("Поддерживаются линейная и логарифмическая шкалы")
        if not math.isfinite(axis.log_base) or axis.log_base <= 0 or axis.log_base == 1:
            raise ValueError("Основание логарифма должно быть положительным и не равным 1")
        for value in (axis.minimum, axis.maximum, axis.major_step):
            if value is not None and not math.isfinite(value):
                raise ValueError("Пределы и шаг оси должны быть конечными")
        if (
            axis.minimum is not None
            and axis.maximum is not None
            and axis.minimum >= axis.maximum
        ):
            raise ValueError(f"Минимум оси {name} должен быть меньше максимума")
        if axis.scale == "log" and any(
            v is not None and v <= 0 for v in (axis.minimum, axis.maximum)
        ):
            raise ValueError(
                f"Для логарифмической оси {name} пределы должны быть положительными"
            )
        if axis.major_step is not None and (
            axis.major_step <= 0 or axis.scale == "log"
        ):
            raise ValueError(
                "Фиксированный шаг делений задаётся положительным числом на линейной оси"
            )
        to_rgba(axis.color)
    for color in (
        spec.color,
        spec.edge_color,
        spec.missing_color,
        spec.grid_color,
        spec.background,
    ):
        to_rgba(color)
    validate_marker(spec.marker)
    if (
        not 0 <= spec.alpha <= 1
        or not 0 <= spec.grid_alpha <= 1
    ):
        raise ValueError("Некорректный маркер или прозрачность")
    if not 0 <= spec.interpretation_alpha <= 1:
        raise ValueError("Прозрачность слоя должна быть от 0 до 1")
    if (
        not 0 <= spec.stratigraphy_alpha <= 1
        or not 5 <= spec.stratigraphy_label_size <= 30
    ):
        raise ValueError(
            "Прозрачность стратиграфии от 0 до 1, размер подписей от 5 до 30"
        )
    if not (0 < spec.size_min <= spec.size_max and spec.marker_size > 0):
        raise ValueError(
            "Размеры маркеров должны быть положительными; минимум ≤ максимума"
        )
    if spec.interval_width <= 0:
        raise ValueError("Толщина интервальной пробы должна быть положительной")
    if spec.legend_size <= 0 or spec.legend_columns < 1:
        raise ValueError(
            "Размер текста легенды и число колонок должны быть положительными"
        )
    if spec.width_mm < 30 or spec.height_mm < 30 or not 36 <= spec.dpi <= 1200:
        raise ValueError("Размер рисунка ≥30 мм, DPI от 36 до 1200")
    if not (0 <= spec.left < spec.right <= 1 and 0 <= spec.bottom < spec.top <= 1):
        raise ValueError("Некорректные поля рисунка")
    if spec.bins < 1:
        raise ValueError("Число интервалов должно быть положительным")
    for low, high in (
        (None, None) if numeric_color and spec.color_scale == "classes" else (spec.color_min, spec.color_max),
        (spec.size_value_min, spec.size_value_max),
    ):
        if low is not None and high is not None and low >= high:
            raise ValueError("Минимум шкалы должен быть меньше максимума")
    colormaps[spec.cmap]
    for channel, cid in (
        ("color", spec.color_by),
        ("shape", spec.shape_by),
        ("size", spec.size_by),
    ):
        if (
            not cid
            or (channel == "color" and spec.color_mode == "numeric")
            or (channel == "size" and spec.size_mode == "numeric")
        ):
            continue
        if channel in ("shape", "size") and spec.kind not in (
            "scatter",
            "depth",
            "multi_depth",
        ):
            continue
        for entry in spec.style_maps.get(channel, {}).get(cid, {}).values():
            if channel == "color":
                to_rgba(entry["style"])
            elif channel == "shape":
                validate_marker(entry['style'])
            elif channel == "size":
                diameter = float(entry["style"])
                if not math.isfinite(diameter) or not 0 < diameter <= 100:
                    raise ValueError(
                        "Размер категории должен быть от 0 до 100 pt (не включая 0)"
                    )


def is_calculated_column(column):
    return any(column.metadata.get(key) for key in ("calculation", "nonpyro", "stratigraphy"))


def excluded_column_status(column, spec):
    if column.metadata.get("method"):
        return "excluded_interpolation" if column.metadata.get("stale") or not spec.include_interpolated else None
    if is_calculated_column(column):
        return None if spec.include_calculated else "excluded_calculation"
    return None if spec.include_original else "excluded_original"


def numeric(project, row, cid, spec):
    from autopyro.analysis.qc import is_excluded
    if is_excluded(project, row.id, cid, "plots"):
        return "qc_excluded", None
    column = next((c for c in project.columns if c.id == cid), None)
    if column is None:
        return "missing_column", None
    excluded = excluded_column_status(column, spec)
    if excluded is not None:
        return excluded, None
    return classify(
        row.values.get(cid), numeric=column.kind == "number" or spec.text_numbers
    )


def plot_columns(project, frame, column_ids, spec, *, prepared=None):
    """Classify each required column once, preserving plotting exclusion priority."""
    if prepared is not None:
        return prepared.columns(frame, column_ids)
    columns = {column.id: column for column in project.columns}
    result = {}
    for cid in dict.fromkeys(column_ids):
        column = columns.get(cid)
        if column is None:
            data = pd.DataFrame({"status": "missing_column", "number": None}, index=frame.index, dtype=object)
        elif (excluded := excluded_column_status(column, spec)) is not None:
            data = pd.DataFrame({"status": excluded, "number": None}, index=frame.index, dtype=object)
        else:
            data = classify_series(frame[cid], numeric=column.kind == "number" or spec.text_numbers)
        data.loc[exclusion_mask(project, frame.index, cid, "plots"), ["status", "number"]] = ["qc_excluded", None]
        result[cid] = data
    return result


def legend_entries(mapping, order):
    entries = mapping.items()
    if order == "default":
        return entries
    return sorted(entries, key=lambda item: item[1]["label"].casefold(), reverse=order == "descending")


def draw_colorbar(figure, ax, artist, spec, fallback="", **kwargs):
    if spec.colorbar_location == "none":
        return
    if isinstance(artist.norm, BoundaryNorm) and len(artist.norm.boundaries) <= 12:
        kwargs.setdefault('ticks', artist.norm.boundaries)
    bar = figure.colorbar(artist, ax=ax, location=spec.colorbar_location, **kwargs)
    bar.set_label(spec.colorbar_label or fallback, fontsize=spec.legend_size, fontfamily=spec.font_family)
    bar.ax.tick_params(labelsize=spec.legend_size, labelfontfamily=spec.font_family)


def apply_scatter_aspect(ax, spec):
    if spec.data_aspect is not None:
        ax.set_aspect(spec.data_aspect, adjustable="box")
    elif spec.square_plot_area:
        ax.set_box_aspect(1)


def axis_label(project, cid):
    column = next((c for c in project.columns if c.id == cid), None)
    return column.name + (f" [{column.unit}]" if column.unit else "") if column else ""


def _axis(ax, config, dimension, fallback, spec, categorical=False, *, column=None, language="ru"):
    axis = getattr(ax, f'{dimension}axis')
    setter = getattr(ax, f'set_{dimension}scale')
    scale = config.scale if not categorical else "linear"
    setter(scale, **({"base": config.log_base} if scale == "log" else {}))
    if scale == 'log' and config.log_base < 1:
        # Matplotlib's locator expects an increasing base; the transform and
        # formatter still retain the user's base. Both bases share tick positions.
        axis.set_major_locator(LogLocator(base=1 / config.log_base))
        axis.set_minor_locator(LogLocator(base=1 / config.log_base, subs='auto'))
    label = getattr(ax, f'set_{dimension}label')
    caption = config.label or fallback
    if spec.show_result_type and not categorical and column is not None and column.kind == "number":
        if column.metadata.get("method"):
            result_type = "interpolation" if language == "en" else "интерполяция"
        elif is_calculated_column(column):
            result_type = "calculation" if language == "en" else "расчёт"
        else:
            result_type = "source data" if language == "en" else "исходные данные"
        caption = f"{caption} · {result_type}"
    label(
        caption,
        fontsize=config.label_size,
        labelpad=config.label_pad,
        color=config.color,
        fontfamily=spec.font_family,
    )
    ax.tick_params(
        axis=dimension,
        labelsize=config.tick_size,
        labelrotation=config.rotation,
        labelfontfamily=spec.font_family,
        colors=config.color,
    )
    if not categorical:
        limits = getattr(ax, f'set_{dimension}lim')
        limits(config.minimum, config.maximum)
        if config.major_step is not None:
            low, high = getattr(ax, f'get_{dimension}lim')()
            if abs(high - low) / config.major_step > 1000:
                raise ValueError(
                    "Слишком мелкий шаг делений: более 1000 делений на оси"
                )
            axis.set_major_locator(MultipleLocator(config.major_step))
        if config.scale == "linear":
            if config.minor_divisions:
                axis.set_minor_locator(AutoMinorLocator(config.minor_divisions))
            if config.number_format in ("plain", "scientific"):
                formatter = ScalarFormatter(useOffset=False, useMathText=True)
                formatter.set_scientific(config.number_format == "scientific")
                if config.number_format == "scientific":
                    formatter.set_powerlimits((0, 0))
                axis.set_major_formatter(formatter)
        if not config.minor_divisions:
            axis.set_minor_locator(NullLocator())
    if config.inverted:
        getattr(ax, f'invert_{dimension}axis')()
    for which, visible in (("major", config.major_grid), ("minor", config.minor_grid)):
        ax.grid(
            visible,
            axis=dimension,
            which=which,
            **(
                {
                    "color": spec.grid_color,
                    "alpha": spec.grid_alpha,
                    "linewidth": spec.grid_width,
                    "linestyle": spec.grid_style,
                }
                if visible
                else {}
            ),
        )


def _range(project, cid, spec, minimum=None, maximum=None, data=None, *, logarithmic=False):
    if data is None:
        data = plot_columns(project, project_frame(project, column_ids=[cid]), [cid], spec)[cid]
    values = data["number"].dropna().tolist()
    if logarithmic:
        values = [value for value in values if value > 0]
        low = minimum if minimum is not None else min(values, default=maximum / 10 if maximum else 1)
        high = maximum if maximum is not None else max(values, default=minimum * 10 if minimum else 10)
        if low == high and minimum is None:
            low /= 10
        elif low == high and maximum is None:
            high *= 10
        if not 0 < low < high or not math.isfinite(high):
            raise ValueError("Пределы шкалы не соответствуют данным")
        return LogNorm(low, high)
    low = minimum if minimum is not None else (min(values) if values else 0)
    high = maximum if maximum is not None else (max(values) if values else 1)
    if low == high:
        low, high = low - 0.5, high + 0.5
    if low >= high:
        raise ValueError("Пределы шкалы не соответствуют данным")
    return Normalize(low, high, clip=True)


def numeric_color_norm(project, spec, data):
    if spec.color_scale == 'classes':
        return BoundaryNorm(color_class_edges(spec), colormaps[spec.cmap].N, clip=True)
    return _range(project, spec.color_by, spec, spec.color_min, spec.color_max, data,
                  logarithmic=spec.color_scale == 'log')


def draw_interpretation_layer(ax, project, spec, *, language="ru"):
    from autopyro.schemes.interpretation import inverse, validate_binding, validate_scheme
    from shapely.geometry.polygon import orient

    binding = spec.interpretation
    scheme = binding["scheme"]
    if (
        spec.kind != "scatter"
        or spec.x_id != binding["x_id"]
        or spec.y_id != binding["y_id"]
    ):
        raise ValueError(
            "Интерпретационный слой связан с другими осями; удалите слой или восстановите X/Y"
        )
    validate_binding(
        project, spec.x_id, spec.y_id, binding["x_unit"], binding["y_unit"], scheme
    )
    regions, lines = validate_scheme(scheme)
    # Scatter artists already supplied the limits. Scientific overlays must not
    # expand them; keep full-scheme fallback when no valid points were plotted.
    point_limits = ax.dataLim.frozen() if np.isfinite(ax.dataLim.get_points()).all() else None
    label_regions = []

    def coordinates(ring):
        coords = list(ring.coords)
        output = []
        nonlinear = any(scheme[key].get("transform") == "log10" for key in ("x", "y"))
        for a, b in zip(coords[:-1], coords[1:], strict=True):
            for t in np.linspace(0, 1, 65 if nonlinear else 2)[:-1]:
                point = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
                output.append(
                    (inverse(point[0], scheme["x"]), inverse(point[1], scheme["y"]))
                )
        output.append(
            (inverse(coords[-1][0], scheme["x"]), inverse(coords[-1][1], scheme["y"]))
        )
        if not np.isfinite(output).all():
            raise ValueError(
                "Обратное преобразование геометрии дало неконечные координаты"
            )
        return output

    for entry, geometry in regions:
        polygons = (
            list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
        )
        label_parts = []
        for polygon in polygons:
            polygon = orient(polygon, sign=1)
            vertices, codes = [], []
            rings = []
            for ring in (polygon.exterior, *polygon.interiors):
                values = coordinates(ring)
                rings.append(values)
                vertices.extend(values)
                codes.extend(
                    [MplPath.MOVETO]
                    + [MplPath.LINETO] * (len(values) - 2)
                    + [MplPath.CLOSEPOLY]
                )
            color = entry.get("color", "#777777")
            ax.add_patch(
                PathPatch(
                    MplPath(vertices, codes),
                    facecolor=color,
                    edgecolor=color,
                    alpha=spec.interpretation_alpha,
                    linewidth=spec.line_width,
                    zorder=0.3,
                )
            )
            if spec.interpretation_labels:
                from shapely.geometry import Polygon
                label_parts.append(Polygon(rings[0], rings[1:]))
        if label_parts:
            import shapely
            label_regions.append((entry["label"], shapely.union_all(label_parts)))
    if label_regions:
        from autopyro.schemes.interpretation_labels import RegionLabels
        ax.add_artist(RegionLabels(label_regions, fontsize=spec.label_size,
                                   fontfamily=spec.font_family))
    for entry, geometry in lines:
        segments = (
            geometry.geoms if geometry.geom_type == "MultiLineString" else [geometry]
        )
        for segment in segments:
            values = coordinates(segment)
            ax.plot(
                [p[0] for p in values],
                [p[1] for p in values],
                color=entry.get("color", "#555555"),
                linestyle="--" if entry.get("dashed") else "-",
                linewidth=spec.line_width,
                zorder=0.4,
            )
    if point_limits is not None:
        ax.dataLim.set(point_limits)
    ax.autoscale_view()
    if spec.interpretation_info:
        statuses = ({
            "demonstration": "demonstration",
            "user_supplied": "user supplied",
            "validated": "validated by the author",
        } if language == "en" else {
            "demonstration": "демонстрационная",
            "user_supplied": "пользовательская",
            "validated": "проверена автором",
        })
        status = statuses[scheme["scientific_status"]]
        ax.text(
            0.99,
            0.01,
            f"{scheme['id']} · rev {scheme['revision']}\n{status}",
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=6,
            fontfamily=spec.font_family,
        )


def fit_depth_legend(ax, legend, columns):
    """Wrap labels against measured font widths, including legend handles/padding."""
    from matplotlib.backends.backend_agg import RendererAgg

    renderer = RendererAgg(1, 1, ax.figure.dpi)
    texts = legend.get_texts()
    labels = [text.get_text() for text in texts]
    font_pixels = texts[0].get_fontsize() * ax.figure.dpi / 72
    available = ax.get_window_extent().width - 2 * legend.borderaxespad * font_pixels
    width = max(1, int(available / font_pixels / .65 / columns))
    for characters in range(width, 0, -1):
        for text, label in zip(texts, labels, strict=True):
            text.set_text("\n".join(textwrap.fill(line, width=characters) for line in label.split("\n")))
        if legend.get_window_extent(renderer).width <= available:
            break


def _track_samples(spec, x_id, x_axis, paired, index, numeric_data):
    """Resolve usable coordinates and record the first exclusion for each sample."""
    data = pd.DataFrame(index=index)
    rejected = pd.Series(None, index=index, dtype=object)
    details = {}
    for axis_name, cid in [("x", x_id), *([("y", spec.y_id)] if paired else [])]:
        column_data = numeric_data[cid].loc[index]
        data[axis_name] = column_data["number"]
        missing = rejected.isna() & data[axis_name].isna()
        rejected.loc[missing] = column_data.loc[missing, "status"]
        if missing.any():
            details[cid] = rejected.loc[missing].value_counts(sort=False).to_dict()
    for axis_name, cid, axis in [('x', x_id, x_axis), *([('y', spec.y_id, spec.y_axis)] if paired else [])]:
        if axis.scale == 'log':
            nonpositive = rejected.isna() & (data[axis_name] <= 0)
            if nonpositive.any():
                rejected.loc[nonpositive] = 'nonpositive_log'
                details.setdefault(cid, {})['nonpositive_log'] = int(nonpositive.sum())
    bottoms = {}
    if spec.depth_bottom_id:
        bottom_data = numeric_data[spec.depth_bottom_id].loc[index]
        missing = rejected.isna() & bottom_data["number"].isna()
        rejected.loc[missing] = bottom_data.loc[missing, "status"]
        invalid = rejected.isna() & (bottom_data["number"] <= data["y"])
        rejected.loc[invalid] = "invalid_depth_interval"
        if (missing | invalid).any():
            details[spec.depth_bottom_id] = rejected.loc[missing | invalid].value_counts(sort=False).to_dict()
        bottoms = bottom_data.loc[rejected.isna(), "number"].astype(float)
    data = data.loc[rejected.isna()]
    return data, bottoms, rejected, details


def _draw_sample_markers(ax, project, spec, data, selected_frame, numeric_data, bottoms,
                         color_norm, size_norm, result):
    """Draw pickable points or depth intervals and record their sample identifiers."""
    if spec.shape_by:
        shapes = spec.style_maps["shape"][spec.shape_by]
        markers = category_keys(selected_frame.loc[data.index, spec.shape_by]).map(
            lambda key: shapes[key]["style"])
        point_groups = data.groupby(markers, sort=False)
    else:
        point_groups = () if data.empty or spec.marker is None else ((spec.marker, data),)
    for marker_index, (marker, group) in enumerate(point_groups):
        point_ids = group.index
        x = group["x"].to_numpy(dtype=float)
        y = group["y"].to_numpy(dtype=float)
        if color_norm is not None:
            colors = numeric_data[spec.color_by].loc[point_ids, "number"].to_numpy(
                dtype=float, na_value=np.nan
            )
        elif spec.color_by:
            styles = spec.style_maps["color"][spec.color_by]
            colors = category_keys(selected_frame.loc[point_ids, spec.color_by]).map(
                lambda key: styles[key]["style"]).tolist()
        else:
            colors = [spec.color]
        if size_norm is not None:
            values = numeric_data[spec.size_by].loc[point_ids, "number"].to_numpy(
                dtype=float, na_value=np.nan
            )
            diameters = np.where(
                np.isnan(values), spec.marker_size,
                spec.size_min + np.asarray(size_norm(values)) * (spec.size_max - spec.size_min),
            )
            sizes = diameters**2
        elif spec.size_by:
            styles = spec.style_maps["size"][spec.size_by]
            sizes = category_keys(selected_frame.loc[point_ids, spec.size_by]).map(
                lambda key: float(styles[key]["style"])**2).to_numpy(dtype=float)
        else:
            sizes = spec.marker_size**2
        kwargs = {}
        if color_norm is not None:
            kwargs.update(
                norm=color_norm,
                cmap=colormaps[spec.cmap].with_extremes(bad=spec.missing_color),
                plotnonfinite=True,
            )
        if spec.depth_bottom_id:
            bottom = bottoms.loc[point_ids].to_numpy()
            segments = np.stack((np.column_stack((x, y)), np.column_stack((x, bottom))), axis=1)
            artist = LineCollection(
                segments,
                colors=colors if color_norm is None else None,
                linewidths=np.sqrt(sizes) if spec.size_by else spec.interval_width,
                alpha=spec.alpha,
                picker=5,
                **{
                    key: value
                    for key, value in kwargs.items()
                    if key != "plotnonfinite"
                },
            )
            if color_norm is not None:
                artist.set_array(np.ma.masked_invalid(colors))
            ax.add_collection(artist)
            ax.autoscale_view()
            artist._autopyro_intervals = np.column_stack((x, y, bottom))
        else:
            artist = ax.scatter(
                x, y,
                # One color and size let Matplotlib reuse a uniform
                # styling instead of drawing a separate path per point.
                c=colors,
                s=sizes,
                marker=marker,
                edgecolors=spec.edge_color,
                linewidths=spec.edge_width,
                alpha=spec.alpha,
                picker=5,
                **kwargs,
            )
        artist._autopyro_row_ids = point_ids.tolist()
        result["artists"].append(artist)
        if color_norm is not None and marker_index == 0:
            draw_colorbar(ax.figure, ax, artist, spec, axis_label(project, spec.color_by), pad=0.03)
            if not np.isfinite(colors).any():
                result["notices"].append(
                    "Нет числовых значений цвета: использован цвет пропусков"
                )


def render_plot(
    project: Project, row_ids, spec: PlotSpec, *, figure=None, selected_ids=(), prepared=None, axes=None,
    language=None, include_membership=False,
):
    language = language or spec.render_language
    english = language == "en"
    all_samples = "All samples" if english else "Все пробы"
    if prepared is not None:
        project = prepared.project
    validate_spec(spec)
    if spec.filter_rules:
        from autopyro.model.filtering import leaf_rules, matches_rules

        columns = {column.id for column in project.columns}
        if any(rule["column"] not in columns for rule in leaf_rules(spec.filter_rules)):
            raise ValueError("Колонка локального фильтра удалена")
        wanted = set(row_ids)
        row_ids = [row.id for row in project.rows
                   if row.id in wanted and matches_rules(project, row, spec.filter_rules, spec.filter_combine)]
    passport = None
    if spec.passport_id:
        from .plot_passports import plot_passport
        passport = plot_passport(spec, {column.id: column for column in project.columns})
    hidden = set()
    if spec.color_by and spec.color_mode == "category" and not (spec.interpretation or spec.interpretation_panels):
        mapping = spec.style_maps.get("color", {}).get(spec.color_by, {})
        hidden = {key for key, entry in mapping.items() if entry.get("hidden") is True}
        if hidden:
            from .plot_controls import controls_for

            if ":" not in spec.kind or "channels" in controls_for(spec.kind).capabilities:
                frame = (prepared.frame(row_ids, [spec.color_by]) if prepared is not None
                         else project_frame(project, row_ids, [spec.color_by]))
                row_ids = frame.index[~category_keys(frame[spec.color_by]).isin(hidden)].tolist()
    curve_record = None
    if spec.curve_archive_id:
        from .curve_layer import archived_curve
        if spec.kind != 'scatter' or spec.interpretation or spec.interpretation_panels or spec.stratigraphy:
            raise ValueError('Сохранённая кривая доступна для обычной диаграммы X–Y')
        curve_record = archived_curve(project, spec.curve_archive_id)
        if curve_record is None:
            raise ValueError('Архив сохранённой кривой отсутствует в проекте')
    if spec.interpretation_panels:
        from autopyro.schemes.interpretation_comparison import render_comparison
        if len(spec.interpretation_panels) not in (3, 4) or spec.kind != 'scatter':
            raise ValueError('Сравнение требует трёх или четырёх диаграмм X–Y')
        return render_comparison(project, row_ids, spec, figure=figure, selected_ids=selected_ids,
                                 prepared=prepared, language=language, include_membership=include_membership)
    if spec.interpretation_source_frame and spec.interpretation:
        scheme = spec.interpretation['scheme']
        # This display option does not change the coordinates used by the calculation.
        x_axis, y_axis = (
            replace(config, minimum=scheme[key].get('minimum'), maximum=scheme[key].get('maximum'))
            for key, config in (('x', spec.x_axis), ('y', spec.y_axis))
        )
        spec = replace(spec, x_axis=x_axis, y_axis=y_axis)
    if ":" in spec.kind:
        from .library_plot import render_library_plot
        return render_library_plot(project, row_ids, spec, figure=figure, selected_ids=selected_ids,
                                   prepared=prepared, language=language, include_membership=include_membership)
    if figure is not None and type(figure) is not Figure:
        figure = Figure(figsize=(spec.width_mm / 25.4, spec.height_mm / 25.4), dpi=96)
    ensure_style_maps(project, spec, prepared=prepared)
    validate_spec(spec)
    columns = {c.id: c for c in project.columns}
    x_ids, paired = native_columns(spec, columns)
    if spec.kind in ("box", "violin") and spec.y_axis.scale == "log":
        raise ValueError("Ось категорий box/violin должна быть линейной")
    if (
        spec.kind in ("hist", "box", "violin")
        and spec.color_by
        and spec.color_mode == "numeric"
    ):
        raise ValueError(
            "Для распределений используйте категориальную группировку цветом"
        )
    figure = figure or Figure(
        figsize=(spec.width_mm / 25.4, spec.height_mm / 25.4), dpi=96
    )
    if axes is not None:
        if spec.kind != 'scatter' or len(axes) != 1 or axes[0].figure is not figure:
            raise ValueError('Внешняя ось допустима только для одной диаграммы X–Y')
    else:
        figure.clear()
        figure.set_facecolor(spec.background)
        axes = figure.subplots(
            1, len(x_ids), squeeze=False, sharey=spec.kind == "multi_depth"
        )[0]
        figure.subplots_adjust(
            left=spec.left,
            right=spec.right,
            bottom=spec.bottom,
            top=spec.top,
            wspace=spec.wspace,
        )
        figure.suptitle(
            spec.title, fontsize=spec.title_size, fontfamily=spec.font_family, y=0.98
        )
        if spec.subtitle:
            figure.text(
                0.5,
                0.92,
                spec.subtitle,
                ha="center",
                fontsize=max(7, spec.title_size - 3),
                fontfamily=spec.font_family,
            )
    selected_ids, row_ids = set(selected_ids), set(row_ids)
    result = {
        "axes": list(axes),
        "artists": [],
        "excluded": {},
        "excluded_by_column": {},
        "plotted": {},
        "notices": [],
    }
    if include_membership:
        result['plotted_row_ids'] = {}
    if passport is not None:
        result['passport'] = passport
        if passport.get('depth_reference') == 'MD':
            description = 'measured depth (MD)' if english else 'измеренная глубина (MD)'
        else:
            description = 'descriptive diagram' if english else 'описательная диаграмма'
        result['notices'].append(f"{passport['name']} · {passport['id']} · {description}")
    input_ids = list(dict.fromkeys(cid for cid in [*x_ids, spec.y_id if paired else "",
        spec.depth_bottom_id, spec.color_by, spec.size_by, spec.line_group_by,
        spec.shape_by if paired else "", spec.label_by if paired else ""] if cid))
    frame = prepared.frame if prepared is not None else lambda ids=None, cids=None: project_frame(project, ids, cids)
    selected_frame = frame(row_ids, input_ids)
    global_ids = list(dict.fromkeys(cid for cid, enabled in [
        (spec.color_by, spec.color_mode == "numeric"),
        (spec.size_by, paired and spec.size_mode == "numeric"),
    ] if cid and enabled))
    numeric_inputs = {*x_ids, spec.y_id if paired else '', spec.depth_bottom_id}
    numeric_data = plot_columns(project, selected_frame,
                                [cid for cid in input_ids if cid in numeric_inputs and cid not in global_ids],
                                spec, prepared=prepared)
    if global_ids:
        numeric_data.update(plot_columns(project, frame(None, global_ids), global_ids, spec, prepared=prepared))
    color_norm = (
        numeric_color_norm(project, spec, numeric_data[spec.color_by])
        if spec.color_by and spec.color_mode == "numeric"
        else None
    )
    size_norm = (
        _range(project, spec.size_by, spec, spec.size_value_min, spec.size_value_max, numeric_data[spec.size_by])
        if paired and spec.size_by and spec.size_mode == "numeric"
        else None
    )
    for ax, x_id in zip(axes, x_ids, strict=True):
        x_axis = (
            spec.track_axes.get(x_id, spec.x_axis)
            if spec.kind == "multi_depth"
            else spec.x_axis
        )
        ax.set_facecolor(spec.background)
        if spec.kind == "scatter":
            apply_scatter_aspect(ax, spec)
        ax.set_axisbelow(spec.grid_below)
        index = selected_frame.index
        data, bottoms, rejected, details = _track_samples(spec, x_id, x_axis, paired, index, numeric_data)
        result["excluded_by_column"][x_id] = details
        result["excluded"][x_id] = rejected.dropna().value_counts(sort=False).to_dict()
        result["plotted"][x_id] = len(data)
        if include_membership:
            result['plotted_row_ids'][x_id] = data.index.tolist()
        if paired:
            _draw_sample_markers(ax, project, spec, data, selected_frame, numeric_data, bottoms,
                                 color_norm, size_norm, result)
            if spec.connect:
                group_id = spec.line_group_by or (spec.color_by if spec.color_mode == 'category' else '')
                if not group_id:
                    raise ValueError("Для соединения точек выберите группу линий или категорию цвета")
                sort_axis = 'y' if spec.kind in ('depth', 'multi_depth') else 'x'
                keys = selected_frame.loc[data.index, group_id].map(category_key)
                if not spec.connect_across_gaps:
                    # Split at excluded rows whose position along the profile
                    # (or X on a scatter plot) is known within the same group.
                    sort_id = spec.y_id if sort_axis == 'y' else x_id
                    ordered = numeric_data[sort_id].loc[index, 'number'].dropna().sort_values(kind='stable')
                    ordered_groups = selected_frame.loc[ordered.index, group_id].map(category_key)
                    gaps = rejected.loc[ordered.index].notna().groupby(ordered_groups, sort=False).cumsum()
                    keys = [keys, gaps.reindex(data.index).fillna(0)]
                for key, group in data.groupby(keys, sort=False):
                    group = group.sort_values(sort_axis, kind="stable")
                    color = spec.color
                    if group_id == spec.color_by and spec.color_mode == 'category':
                        category = key if spec.connect_across_gaps else key[0]
                        color = spec.style_maps["color"][group_id][category]["style"]
                    ax.plot(
                        group["x"].to_numpy(dtype=float),
                        group["y"].to_numpy(dtype=float),
                        color=color,
                        alpha=spec.alpha,
                        linewidth=spec.line_width,
                        zorder=1,
                    )
            if selected_ids:
                highlights = data.loc[data.index.isin(selected_ids)]
                if not highlights.empty:
                    if spec.depth_bottom_id:
                        ax.vlines(
                            highlights["x"], highlights["y"], bottoms.loc[highlights.index],
                            colors="#d62728",
                            linewidths=spec.interval_width + 2,
                            zorder=10,
                        )
                    else:
                        ax.scatter(
                            highlights["x"], highlights["y"],
                            s=(spec.marker_size + 5) ** 2,
                            facecolors="none",
                            edgecolors="#d62728",
                            linewidths=1.5,
                            zorder=10,
                        )
            if spec.label_by:
                labels = selected_frame.loc[data.index, spec.label_by]
                for label, x, y in zip(labels, data["x"], data["y"], strict=True):
                    ax.annotate(
                        str(label or ""),
                        (float(x), float(y)),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=spec.label_size,
                        fontfamily=spec.font_family,
                    )
        else:
            keys = (selected_frame.loc[data.index, spec.color_by].map(category_key) if spec.color_by
                    else pd.Series("", index=data.index, dtype=object))
            groups = data["x"].astype(float).groupby(keys, sort=False).agg(list).to_dict()
            for position, (key, values) in enumerate(groups.items(), start=1):
                entry = (
                    spec.style_maps["color"][spec.color_by][key]
                    if spec.color_by
                    else {"label": all_samples, "style": spec.color}
                )
                if spec.kind == "hist":
                    continue
                if spec.kind == "box":
                    box = ax.boxplot(
                        [values],
                        positions=[position],
                        orientation="horizontal",
                        patch_artist=True,
                        widths=0.55,
                        boxprops={"facecolor": entry["style"], "alpha": spec.alpha},
                        medianprops={
                            "color": spec.edge_color,
                            "linewidth": spec.line_width,
                        },
                    )
                    for line in box["whiskers"] + box["caps"]:
                        line.set_color(spec.edge_color)
                elif len(values) >= 2 and min(values) != max(values):
                    violin = ax.violinplot(
                        [values],
                        positions=[position],
                        orientation="horizontal",
                        showmedians=True,
                    )
                    for body in violin["bodies"]:
                        body.set_facecolor(entry["style"])
                        body.set_edgecolor(spec.edge_color)
                        body.set_alpha(spec.alpha)
                else:
                    ax.scatter(
                        values,
                        [position] * len(values),
                        color=entry["style"],
                        s=spec.marker_size**2,
                    )
                    result["notices"].append(
                        f"{entry['label']}: для violin недостаточно разброса; показаны точки"
                    )
            if spec.kind == "hist" and not data.empty:
                edges = np.histogram_bin_edges(data["x"].to_numpy(dtype=float), bins=spec.bins)
                for key, values in groups.items():
                    entry = (
                        spec.style_maps["color"][spec.color_by][key]
                        if spec.color_by
                        else {"label": all_samples, "style": spec.color}
                    )
                    ax.hist(
                        values,
                        bins=edges,
                        color=entry["style"],
                        edgecolor=spec.edge_color,
                        linewidth=spec.edge_width,
                        alpha=spec.alpha,
                        density=spec.density,
                        label=entry["label"],
                    )
        qc_outlines = []
        if spec.qc_outline and paired and not spec.interpretation and not spec.depth_bottom_id:
            from .plot_qc import draw_qc_outlines

            qc_outlines = draw_qc_outlines(ax, project, result['artists'], (x_id, spec.y_id))
        if spec.measurement_error_bars and paired and not spec.interpretation:
            from .uncertainty_plot import draw_measurement_error_bars

            x = data["x"].to_numpy(dtype=float)
            y = data["y"].to_numpy(dtype=float)
            if spec.depth_bottom_id:
                y = y / 2 + bottoms.to_numpy() / 2
                column_ids, axis_specs = (x_id,), (x_axis,)
            else:
                column_ids, axis_specs = (x_id, spec.y_id), (x_axis, spec.y_axis)
            points = zip(data.index, x, y, strict=True)
            result['notices'].extend(draw_measurement_error_bars(
                ax, project, points, column_ids, axis_specs, columns))
        if spec.interpretation:
            from .uncertainty_plot import draw_uncertainty
            crossings, skipped_log = draw_uncertainty(
                ax, project, spec, [artist for artist in result['artists'] if artist.axes is ax])
            if skipped_log:
                result['notices'].append(f'Интервалы неопределённости с неположительной границей на логарифмической оси: {skipped_log}')
            result.setdefault('uncertainty_crossings', {}).update(crossings)
            if crossings:
                result['notices'].append(f'Интервальная интерпретация неоднозначна: {len(crossings)}; отмечено оранжевым ромбом')
            draw_interpretation_layer(ax, project, spec, language=language)
            current_scheme = project.view.get("interpretation_schemes", {}).get(
                spec.interpretation["scheme"]["id"]
            )
            if current_scheme and current_scheme != spec.interpretation["scheme"]:
                result["notices"].append(
                    "Слой показывает сохранённую редакцию схемы, отличающуюся от текущей"
                )
        curve_line = None
        if curve_record is not None:
            from .curve_layer import curve_snapshot, draw_archived_curve
            curve_line = draw_archived_curve(ax, curve_record, spec, columns, language=language)
            archive_note = ('archived curve, points from the current scope' if english
                            else 'кривая из архива, точки из текущей области')
            result['notices'].append(f'{curve_line.get_label()} · {archive_note}')
            result['curve_archive'] = curve_snapshot(curve_record)
        _axis(ax, x_axis, "x", axis_label(project, x_id), spec, column=columns[x_id], language=language)
        _axis(
            ax,
            spec.y_axis,
            "y",
            axis_label(project, spec.y_id)
            if paired
            else (
                ("Density" if english else "Плотность")
                if spec.kind == "hist" and spec.density
                else ("Sample count" if english else "Число проб")
                if spec.kind == "hist"
                else axis_label(project, spec.color_by)
            ),
            spec,
            categorical=spec.kind in ("box", "violin"),
            column=columns.get(spec.y_id) if paired else None,
            language=language,
        )
        if spec.kind in ("box", "violin"):
            labels = [
                spec.style_maps["color"][spec.color_by][key]["label"] if spec.color_by else all_samples
                for key in groups
            ]
            ax.set_yticks(range(1, len(labels) + 1), labels)
        if spec.annotation:
            ax.text(
                0.02,
                0.98,
                spec.annotation,
                transform=ax.transAxes,
                va="top",
                fontsize=spec.label_size,
                fontfamily=spec.font_family,
            )
        if spec.legend and paired:
            handles = [curve_line] if curve_line is not None else []
            handles.extend(qc_outlines)
            for channel, cid in (
                ("color", spec.color_by),
                ("shape", spec.shape_by),
                ("size", spec.size_by),
            ):
                if (
                    not cid
                    or (channel == "color" and color_norm is not None)
                    or (channel == "size" and size_norm is not None)
                ):
                    continue
                visible = set(category_keys(selected_frame.loc[data.index, cid])) if spec.legend_visible_only else None
                for key, entry in legend_entries(spec.style_maps[channel][cid], spec.legend_order):
                    if ((channel == "color" and key in hidden)
                            or visible is not None and key not in visible):
                        continue
                    handles.append(
                        Line2D(
                            [],
                            [],
                            linestyle="none",
                            marker=entry["style"]
                            if channel == "shape"
                            else spec.marker,
                            color=entry["style"] if channel == "color" else spec.color,
                            markersize=float(entry["style"])
                            if channel == "size"
                            else spec.marker_size,
                            label=f"{columns[cid].name}: {entry['label']}",
                        )
                    )
                    if spec.depth_bottom_id:
                        handles[-1].set_marker("")
                        handles[-1].set_linestyle("-")
                        handles[-1].set_linewidth(
                            float(entry["style"])
                            if channel == "size"
                            else spec.interval_width
                        )
            if size_norm is not None:
                for value, diameter in (
                    (size_norm.vmin, spec.size_min),
                    (size_norm.vmax, spec.size_max),
                ):
                    handles.append(
                        Line2D(
                            [],
                            [],
                            linestyle="none",
                            marker=spec.marker,
                            color=spec.color,
                            markersize=diameter,
                            label=f"{columns[spec.size_by].name}: {value:g}",
                        )
                    )
                    if spec.depth_bottom_id:
                        handles[-1].set_marker("")
                        handles[-1].set_linestyle("-")
                        handles[-1].set_linewidth(diameter)
            if handles:
                ax.legend(
                    handles=handles,
                    loc=spec.legend_position,
                    ncols=spec.legend_columns,
                    prop={"family": spec.font_family, "size": spec.legend_size},
                    frameon=spec.legend_frame,
                )
        elif spec.legend and spec.kind == "hist" and not data.empty:
            handles, labels = ax.get_legend_handles_labels()
            if spec.legend_order != "default":
                positions = sorted(range(len(labels)), key=lambda i: labels[i].casefold(),
                                   reverse=spec.legend_order == "descending")
                handles = [handles[i] for i in positions]
                labels = [labels[i] for i in positions]
            ax.legend(
                handles, labels,
                loc=spec.legend_position,
                ncols=spec.legend_columns,
                prop={"family": spec.font_family, "size": spec.legend_size},
                frameon=spec.legend_frame,
            )
    if spec.annotations:
        from .plot_annotations import draw_annotations
        result['notices'].extend(draw_annotations(axes, spec))
    # Inversion must be applied once to a shared Y axis, not once per track.
    if spec.kind == "multi_depth" and spec.y_axis.inverted:
        axes[0].yaxis.set_inverted(True)
    if spec.stratigraphy:
        from autopyro.geology.stratigraphy import draw_layer

        result["notices"].extend(draw_layer(axes, project, row_ids, spec, language=language))
    # Stratigraphic tracks can narrow these axes after the legends are created.
    if spec.depth_bottom_id and spec.legend:
        for ax in axes:
            legend = ax.get_legend()
            if legend is not None:
                fit_depth_legend(ax, legend, spec.legend_columns)
    return figure, result


def validate_export_size(path, spec):
    path = Path(path)
    if path.suffix.lower() not in (".png", ".tif", ".tiff", ".svg", ".pdf"):
        raise ValueError("Поддерживаются PNG, TIFF, SVG и PDF")
    if (path.suffix.lower() in (".png", ".tif", ".tiff")
            and (spec.width_mm / 25.4 * spec.dpi) * (spec.height_mm / 25.4 * spec.dpi) > 64_000_000):
        raise ValueError("Растровый экспорт ограничен 64 млн пикселей; уменьшите размер или DPI")


def layout_notice(figure, renderer, language):
    bounds = figure.get_tightbbox(renderer).transformed(figure.dpi_scale_trans)
    canvas = figure.bbox
    if (bounds.x0 < canvas.x0 - 1 or bounds.y0 < canvas.y0 - 1
            or bounds.x1 > canvas.x1 + 1 or bounds.y1 > canvas.y1 + 1):
        return ('Some labels extend outside the figure. Increase the margins or reduce the font size.'
                if language == 'en' else
                'Часть подписей выходит за границы рисунка. Увеличьте поля или уменьшите размер шрифта.')
    return ''


def _write_plot(path, project, row_ids, spec, *, prepared=None, include_membership=False):
    """Write a caller-owned staging file. Publication belongs to the caller."""
    validate_export_size(path, spec)
    figure, result = render_plot(project, row_ids, spec, prepared=prepared,
                                 include_membership=include_membership)
    _save_rendered_plot(figure, path, spec, layout_warnings=result.setdefault('layout_warnings', []))
    return result


def _save_rendered_plot(figure, target, spec, *, layout_warnings=None, **kwargs):
    def check_layout(event):
        warning = layout_notice(figure, event.renderer, spec.render_language)
        if warning and warning not in layout_warnings:
            layout_warnings.append(warning)

    callback = (figure.canvas.mpl_connect('draw_event', check_layout)
                if layout_warnings is not None else None)
    try:
        figure.set_size_inches(spec.width_mm / 25.4, spec.height_mm / 25.4)
        figure.savefig(target, dpi=spec.dpi, transparent=spec.transparent,
                       facecolor="none" if spec.transparent else spec.background, **kwargs)
    finally:
        if callback is not None:
            figure.canvas.mpl_disconnect(callback)
        figure.clear()


def save_plot(path, project, row_ids, spec, *, prepared=None):
    from autopyro.data_io.file_output import atomic_output
    with atomic_output(path) as temporary:
        return _write_plot(temporary, project, row_ids, spec, prepared=prepared)
