"""Explicit wellbore/depth binding for point and interval samples."""

from __future__ import annotations

import hashlib
import json
import textwrap
from bisect import bisect_left, bisect_right
from collections import defaultdict
from copy import deepcopy

import pandas as pd
from matplotlib.colors import is_color_like

from autopyro.model.core import Column, new_id, parse_number, timestamp
from autopyro.model.tabular import classify, classify_series
from .chronostratigraphic_colors import stratigraphic_color

LEVELS = ("GROUP", "SUBGROUP", "FORMATION", "MEMBER")
UNITS = {"m": 1.0, "ft": 0.3048}
STATUSES = {
    "matched": "Определено",
    "ambiguous": "Перекрытие интервалов",
    "cross_boundary": "Пересекает границу подразделений",
    "partial": "Неполное покрытие",
    "invalid_interval": "Некорректные границы пробы",
    "outside": "Вне интервалов",
    "unknown_wellbore": "Нет интервалов ствола",
    "insufficient_data": "Недостаточно данных",
    "interpolation_excluded": "Интерполяция исключена",
}


# Optional native stratapy input fields; tops and thicknesses remain owned by the scheme.
STRATAPY_FIELDS = ("bottom_grain", "top_grain", "connection_type", "erosion",
                   "lenses", "minerals", "features", "contact")
COLUMN_STYLE = {
    "width_mm": 152.4, "height_mm": 228.6, "dpi": 100, "ppi": 400,
    "display_mode": "log", "feature_mode": "off", "grain_preset": "clastic",
    "unit_borders": True, "spines": False, "consec_units": False,
    "legend": False, "legend_loc": "right", "legend_columns": 1, "legend_border": False,
    "x_axis": False, "x_label": "", "y_label": "Глубина", "y_axis": True,
    "labels": True, "label_size": 10, "font_family": "DejaVu Sans",
    "axis_size": 12, "tick_size": 10, "legend_size": 10, "title_size": 12,
    "alpha": 1.0, "transparent": False, "grid": False, "info": True,
    "title": "", "depth_min": None, "depth_max": None, "xmax": None,
    "mineral_size": 1.0, "feature_size": 1.0,
    "legend_kwargs": {}, "x_ticks_dict": None, "grain_brackets": None,
    "legend_titles": ["Lithologies", "Minerals", "Sedimentary Structures", "Palaeontological Features",
                      "Tectonic Structures", "Bed Contacts", "Sample Indicators"],
    "lithologies": {}, "hierarchy_levels": [], "comparison_wells": [], "correlation": {},
}


def column_style(options=None):
    """Validate persisted/UI options before stratapy can silently replace invalid values."""
    import math

    options = {} if options is None else options
    if not isinstance(options, dict) or set(options) - COLUMN_STYLE.keys():
        raise ValueError("Неизвестные параметры оформления колонки")
    style = deepcopy(COLUMN_STYLE)
    style.update(deepcopy(options))
    correlation = style['correlation']
    if not isinstance(correlation, dict) or (correlation and (
            set(correlation) - {'edge', 'anchors', 'horizons', 'enabled'}
            or not isinstance(correlation.get('enabled', True), bool)
            or not valid_horizon(correlation))):
        raise ValueError('Некорректные настройки опорного горизонта')
    horizons = correlation.get('horizons', [])
    if not isinstance(horizons, list) or any(
            not isinstance(horizon, dict) or set(horizon) != {'name', 'edge', 'anchors'}
            or not isinstance(horizon['name'], str) or not horizon['name'].strip()
            or not valid_horizon(horizon) or len(horizon['anchors']) < 2 for horizon in horizons):
        raise ValueError('Некорректные настройки дополнительных горизонтов')
    if len({item['name'].strip().casefold() for item in horizons}) != len(horizons):
        raise ValueError('Названия дополнительных горизонтов должны быть разными')
    wells = style['comparison_wells']
    if (not isinstance(wells, list) or not all(isinstance(well, str) and well.strip() for well in wells)
            or len(wells) != len(set(wells))):
        raise ValueError("Выберите разные непустые названия стволов для сравнения")
    levels = style["hierarchy_levels"]
    if (not isinstance(levels, list) or not all(isinstance(level, str) and level in LEVELS for level in levels)
            or len(levels) != len(set(levels))):
        raise ValueError("Выберите разные существующие стратиграфические уровни")
    choices = {"display_mode": ("log", "default", "grainsize"),
               "feature_mode": ("off", "default", "merge", "semi-merge"),
               "grain_preset": ("clastic", "volcanic", "sedimentary", "dunham", "carbonate"),
               "legend_loc": ("top", "bottom", "left", "right")}
    for key, values in choices.items():
        if style[key] not in values:
            raise ValueError(f"Недопустимое значение {key}")
    for key, default in COLUMN_STYLE.items():
        value = style[key]
        if isinstance(default, bool) and not isinstance(value, bool):
            raise ValueError(f"{key}: требуется логическое значение")
        if isinstance(default, str) and not isinstance(value, str):
            raise ValueError(f"{key}: требуется строка")
    ranges = {"width_mm": (30, 1000), "height_mm": (30, 1000), "dpi": (50, 600),
              "ppi": (50, 1200), "legend_columns": (1, 10), "alpha": (0, 1),
              "mineral_size": (.1, 10), "feature_size": (.1, 10)}
    ranges.update({key: (5, 48) for key in
                   ("label_size", "axis_size", "tick_size", "legend_size", "title_size")})
    for key, (low, high) in ranges.items():
        value = style[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not low <= value <= high):
            raise ValueError(f"{key}: допустимый диапазон {low}–{high}")
    for key in ("dpi", "ppi", "legend_columns"):
        if not isinstance(style[key], int):
            raise ValueError(f"{key}: требуется целое число")
    for key in ("depth_min", "depth_max", "xmax"):
        value = style[key]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value)):
            raise ValueError(f"{key}: требуется конечное число")
    if style["xmax"] is not None and style["xmax"] <= 0:
        raise ValueError("xmax должен быть положительным")
    if (style["depth_min"] is not None and style["depth_max"] is not None
            and style["depth_min"] >= style["depth_max"]):
        raise ValueError("Минимальная глубина должна быть меньше максимальной")
    for key in ("legend_kwargs", "x_ticks_dict", "grain_brackets", "lithologies"):
        if style[key] is not None and not isinstance(style[key], dict):
            raise ValueError(f"{key}: требуется объект JSON")
    if style["legend_kwargs"] is None:
        raise ValueError("legend_kwargs: требуется объект JSON")
    if style["x_ticks_dict"] is not None:
        ticks = style["x_ticks_dict"]
        if not ticks or any(not isinstance(k, str) or isinstance(v, bool)
                            or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                            for k, v in ticks.items()):
            raise ValueError("x_ticks_dict: нужны названия и неотрицательные числовые значения")
    if style["grain_brackets"] is not None and any(
            not isinstance(v, list) or not all(isinstance(k, str) for k in v)
            for v in style["grain_brackets"].values()):
        raise ValueError("grain_brackets: нужны списки названий делений")
    if (not isinstance(style["legend_titles"], list) or len(style["legend_titles"]) != 7
            or not all(isinstance(title, str) for title in style["legend_titles"])):
        raise ValueError("legend_titles: нужны семь заголовков разделов легенды")
    if not isinstance(style["lithologies"], dict) or any(
            not isinstance(key, str) or not isinstance(value, (list, tuple)) or len(value) != 3
            or value[0] is not None and not isinstance(value[0], str)
            or not isinstance(value[2], str)
            for key, value in style["lithologies"].items()):
        raise ValueError("lithologies: нужны тройки [узор/код/путь, цвет, название]")
    return style


def scheme_hash(scheme):
    return hashlib.sha256(
        json.dumps(scheme, ensure_ascii=False, sort_keys=True, allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def validate_scheme(scheme):
    if scheme.get("format") != "autopyro.stratigraphy" or scheme.get("version") != 1:
        raise ValueError("Неизвестный формат стратиграфической схемы")
    for key in ("id", "revision", "name", "depth_reference"):
        if not isinstance(scheme.get(key), str) or not scheme[key].strip():
            raise ValueError(
                f"Укажите {key}; тип глубины и начало отсчёта должны быть заданы явно"
            )
    if scheme.get("depth_unit") not in UNITS:
        raise ValueError("Единицы глубины: m или ft")
    intervals = scheme.get("intervals")
    if not isinstance(intervals, list) or not intervals:
        raise ValueError("Добавьте интервалы")
    ids = set()
    for item in intervals:
        if not isinstance(item.get("id"), str) or not item["id"] or item["id"] in ids:
            raise ValueError("Каждому интервалу нужен уникальный ID")
        ids.add(item["id"])
        if error := next(interval_field_errors(item), None):
            raise ValueError(error[1])
    return scheme


def interval_color(item):
    color = item.get("color")
    if color is None or (isinstance(color, str) and (
            not color.strip() or (color == "#c7ddeb" and not item.get("color_manual")))):
        return stratigraphic_color(item.get("name")) or "#c7ddeb"
    return color


def interval_field_errors(item):
    for key in ("wellbore", "name"):
        if not isinstance(item.get(key), str) or not item[key].strip():
            yield key, "Укажите название ствола и подразделения"
    if item.get("level") not in LEVELS:
        yield "level", "Допустимые уровни: " + ", ".join(LEVELS)
    depths_valid = True
    for key in ("top", "bottom"):
        value = item.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            depths_valid = False
            yield key, "Границы должны быть конечными числами"
            continue
        try:
            parse_number(value)
        except ValueError as exc:
            depths_valid = False
            yield key, str(exc)
    if depths_valid and item["top"] >= item["bottom"]:
        yield "bottom", "Кровля должна быть меньше подошвы"
    if not is_color_like(interval_color(item)):
        yield "color", "Некорректный цвет интервала"
    if not isinstance(item.get("lithology", ""), str):
        yield "lithology", "Литология stratapy должна быть строкой"


def _interval_tracks(scheme):
    validate_scheme(scheme)
    groups = defaultdict(list)
    for item in scheme["intervals"]:
        groups[item["wellbore"], item["level"]].append(item)
    return groups.values()


def overlap_pairs(scheme):
    """Different hierarchy levels are deliberately not conflicts."""
    pairs = []
    for intervals in _interval_tracks(scheme):
        active = []
        for item in sorted(intervals, key=lambda v: v["top"]):
            active = [
                previous for previous in active if previous["bottom"] > item["top"]
            ]
            pairs.extend((previous["id"], item["id"]) for previous in active)
            active.append(item)
    return pairs


def overlap_count(scheme):
    """Count same-track overlaps without storing the pairs."""
    count = 0
    for intervals in _interval_tracks(scheme):
        tops = sorted(item['top'] for item in intervals)
        bottoms = sorted(item['bottom'] for item in intervals)
        ended = 0
        for started, top in enumerate(tops):
            while ended < started and bottoms[ended] <= top:
                ended += 1
            count += started - ended
    return count


def _track_has_overlap(intervals):
    """Check a single wellbore and level, stopping at the first overlap."""
    previous_bottom = float('-inf')
    for item in sorted(intervals, key=lambda item: item['top']):
        if item['top'] < previous_bottom:
            return True
        previous_bottom = item['bottom']
    return False


def validate_binding(project, scheme, binding):
    validate_scheme(scheme)
    columns = {c.id: c for c in project.columns}
    well_id, depth_id = binding.get("well_id"), binding.get("depth_id")
    mode = binding.get("sample_mode", "point")
    if mode not in ("point", "interval"):
        raise ValueError("Режим проб: point или interval")
    inputs = [well_id, depth_id] + (
        [binding.get("bottom_id")] if mode == "interval" else []
    )
    if len(set(inputs)) != len(inputs) or any(cid not in columns for cid in inputs):
        raise ValueError("Выберите разные существующие колонки ствола и глубин проб")
    if binding.get("depth_unit") not in UNITS:
        raise ValueError("Объявите единицу глубины проб: m или ft")
    if any(
        columns[cid].unit and columns[cid].unit != binding["depth_unit"]
        for cid in inputs[1:]
    ):
        raise ValueError("Объявленная единица противоречит единице колонки глубины")
    if binding.get("depth_reference") != scheme["depth_reference"]:
        raise ValueError(
            "Тип глубины и начало отсчёта проб и интервалов должны совпадать"
        )
    if any(columns[cid].metadata.get("stratigraphy") for cid in inputs):
        raise ValueError(
            "Стратиграфические результаты нельзя использовать как вход своей привязки"
        )


def interval_membership(intervals, top, bottom, factor):
    """Union coverage and overlap length; touching endpoints have zero length.

    Lengths are returned in the sample's declared unit. Per-interval fractions
    may sum above 1 where source intervals overlap; union coverage cannot.
    """
    hits = [item for item in intervals if item["top"] < bottom and item["bottom"] > top]
    events = defaultdict(int)
    overlaps = []
    for item in hits:
        start, end = max(top, item["top"]), min(bottom, item["bottom"])
        events[start] += 1
        events[end] -= 1
        overlaps.append(
            {
                "interval_id": item["id"],
                "name": item["name"],
                "length": (end - start) / factor,
                "fraction": (end - start) / (bottom - top),
            }
        )
    active, previous, covered, ambiguous = 0, top, 0.0, 0.0
    for position in sorted(events):
        if active:
            covered += position - previous
        if active > 1:
            ambiguous += position - previous
        active += events[position]
        previous = position
    # Full coverage is determined from interval endpoints, not rounded fractions.
    cursor = top
    for item in sorted(hits, key=lambda v: v["top"]):
        if item["top"] > cursor:
            break
        cursor = max(cursor, item["bottom"])
    status = (
        "outside"
        if not hits
        else "ambiguous"
        if ambiguous > 0
        else "partial"
        if cursor < bottom
        else "matched"
        if len(hits) == 1
        else "cross_boundary"
    )
    return {
        "status": status,
        "interval_ids": [item["id"] for item in hits],
        "names": [item["name"] for item in hits],
        "overlaps": overlaps,
        "covered_length": covered / factor,
        "uncovered_length": max(0.0, bottom - top - covered) / factor,
        "ambiguous_length": ambiguous / factor,
        "coverage_fraction": min(1.0, covered / (bottom - top)),
    }


class _DepthIndex:
    """Bound each lookup by sorted tops and the maximum preceding bottom."""

    def __init__(self, intervals):
        self.items = sorted(enumerate(intervals), key=lambda pair: pair[1]['top'])
        self.tops = [item['top'] for _, item in self.items]
        self.ends = []
        edge = float('-inf')
        for _, item in self.items:
            edge = max(edge, item['bottom'])
            self.ends.append(edge)

    def hits(self, top, bottom=None):
        first = bisect_right(self.ends, top)
        last = bisect_right(self.tops, top) if bottom is None else bisect_left(self.tops, bottom)
        hits = [(order, item) for order, item in self.items[first:last] if item['bottom'] > top]
        return [item for _, item in sorted(hits, key=lambda pair: pair[0])]


def evaluate_project(project, row_ids, scheme, binding, *, checkpoint=None, _row_map=None):
    validate_binding(project, scheme, binding)
    ids = list(row_ids)
    row_map = project.row_map() if _row_map is None else _row_map
    if len(set(ids)) != len(ids) or set(ids) - row_map.keys():
        raise ValueError("Некорректная выборка проб")
    columns = {c.id: c for c in project.columns}
    well_id, depth_id = binding["well_id"], binding["depth_id"]
    mode = binding.get("sample_mode", "point")
    depth_ids = [depth_id] + ([binding["bottom_id"]] if mode == "interval" else [])
    index = defaultdict(lambda: defaultdict(list))
    for item in scheme["intervals"]:
        index[item["wellbore"]][item["level"]].append(item)
    index = {well: {rank: _DepthIndex(items) for rank, items in ranks.items()}
             for well, ranks in index.items()}
    factor = UNITS[binding["depth_unit"]] / UNITS[scheme["depth_unit"]]
    interpolated = any(columns[cid].metadata.get("method") for cid in [well_id, *depth_ids])
    numeric = [columns[cid].kind == "number" or binding.get("text_numbers", False) for cid in depth_ids]
    if len(ids) >= 256 and not interpolated:
        depth_columns = [
            classify_series(
                pd.Series([row_map[rid].values.get(cid) for rid in ids], dtype=object),
                numeric=allow_numbers,
            )["number"]
            for cid, allow_numbers in zip(depth_ids, numeric, strict=True)
        ]
        readings = zip(*depth_columns, strict=True)
    elif interpolated:
        readings = ((None,) * len(depth_ids) for _ in ids)
    else:
        readings = (
            tuple(classify(row_map[rid].values.get(cid), numeric=allow_numbers)[1]
                  for cid, allow_numbers in zip(depth_ids, numeric, strict=True))
            for rid in ids
        )
    results = []
    for number, (rid, depths) in enumerate(zip(ids, readings, strict=True)):
        if checkpoint and number % 256 == 0:
            checkpoint()
        row = row_map[rid]
        well = row.values.get(well_id)
        depth = depths[0]
        bottom = depths[1] if mode == "interval" else None
        invalid = ""
        if interpolated:
            invalid = "interpolation_excluded"
        elif (
            not isinstance(well, str)
            or not well.strip()
            or any(value is None for value in depths)
        ):
            invalid = "insufficient_data"
        elif mode == "interval" and bottom <= depth:
            invalid = "invalid_interval"
        elif well not in index:
            invalid = "unknown_wellbore"
        if not invalid:
            depth *= factor
            if bottom is not None:
                bottom *= factor
        levels = {}
        for level in LEVELS:
            track = index[well].get(level) if not invalid else None
            if mode == "interval" and not invalid:
                levels[level] = interval_membership(
                    track.hits(depth, bottom) if track else [], depth, bottom, factor
                )
                continue
            matches = track.hits(depth) if track else []
            levels[level] = {
                "status": invalid
                or (
                    "matched"
                    if len(matches) == 1
                    else "ambiguous"
                    if matches
                    else "outside"
                ),
                "interval_ids": [item["id"] for item in matches],
                "names": [item["name"] for item in matches],
            }
            if mode == "interval":
                levels[level].update(
                    overlaps=[],
                    covered_length=None,
                    uncovered_length=None,
                    ambiguous_length=None,
                    coverage_fraction=None,
                )
        results.append(
            {
                "row_id": rid,
                "wellbore": well,
                "depth": row.values.get(depth_id),
                "bottom": row.values.get(binding["bottom_id"])
                if mode == "interval"
                else None,
                "levels": levels,
            }
        )
    return {
        "created_at": timestamp(),
        "scheme": deepcopy(scheme),
        "scheme_hash": scheme_hash(scheme),
        "binding": deepcopy(binding),
        "row_ids": ids,
        "results": results,
        "sample_mode": mode,
        "policy": (
            "point depth; top <= depth < bottom"
            if mode == "point"
            else "sample [top,bottom); positive-length intersections; union coverage; matched only if contained in one unambiguous interval"
        )
        + "; exact wellbore name; no MD/TVD/datum conversion",
    }


def assignment_value(result, output):
    if output == "status":
        return STATUSES[result["status"]]
    return result["names"][0] if result["status"] == "matched" else None


def prepare_result_columns(report, definitions, *, checkpoint=None):
    projections = {}
    for column_id, config in definitions.items():
        level, output = config.get("level"), config.get("output")
        if level not in LEVELS or output not in ("unit", "status"):
            continue
        values, diagnostics = [], {}
        for index, item in enumerate(report["results"]):
            if checkpoint and index % 256 == 0:
                checkpoint()
            result = item["levels"][level]
            values.append(assignment_value(result, output))
            diagnostics[item["row_id"]] = deepcopy(result)
        projections[column_id] = (config, values, diagnostics)
    return projections


def capture_result_columns(project, settings):
    if not settings:
        return None
    try:
        scheme, binding = settings["scheme"], settings["binding"]
        source = {"scheme_hash": scheme_hash(scheme), "revision": scheme["revision"], "binding": binding}
    except (KeyError, TypeError, ValueError):
        return None
    row_ids = [row.id for row in project.rows]
    row_keys = set(row_ids)
    projections = {}
    for column in project.columns:
        config = column.metadata.get("stratigraphy")
        if not config:
            continue
        diagnostics = column.metadata.get("diagnostics")
        if (column.metadata.get("stratigraphy_source") != source or not isinstance(diagnostics, dict)
                or diagnostics.keys() != row_keys):
            return None
        projections[column.id] = (deepcopy(config), [row.values.get(column.id) for row in project.rows],
                                  diagnostics.copy())
    return {"row_ids": row_ids, "binding": binding, "scheme_hash": source["scheme_hash"],
            "scheme": {"revision": scheme["revision"]}, "_column_results": projections}


def recompute(project, *, rows=None, prepared_report=None):
    targets = [c for c in project.columns if c.metadata.get("stratigraphy")]
    if not targets:
        return
    scoped = rows is not None and all(
        isinstance(column.metadata.get("diagnostics"), dict)
        and len(column.metadata["diagnostics"]) == len(project.rows)
        for column in targets
    )
    target_rows = list(rows) if scoped else project.rows
    row_map = {row.id: row for row in target_rows}
    settings = project.view.get("stratigraphy") or {}
    row_ids = [row.id for row in target_rows]
    try:
        report = prepared_report if (
            not scoped and prepared_report is not None
            and prepared_report["row_ids"] == row_ids
            and prepared_report["binding"] == settings.get("binding")
            and prepared_report["scheme_hash"] == scheme_hash(settings.get("scheme"))
            and ("results" in prepared_report or all(
                column.id in prepared_report["_column_results"]
                and prepared_report["_column_results"][column.id][0] == column.metadata["stratigraphy"]
                for column in targets
            ))
        ) else evaluate_project(
            project,
            row_ids,
            settings.get("scheme", {}),
            settings.get("binding", {}),
            _row_map=row_map,
        )
        error = None
    except (ValueError, KeyError, TypeError) as exc:
        report, error = None, str(exc)
    source = (
        {"scheme_hash": report["scheme_hash"], "revision": report["scheme"]["revision"],
         "binding": report["binding"]}
        if report else {"error": error}
    )
    if scoped and any(column.metadata.get("stratigraphy_source") != source for column in targets):
        recompute(project)
        return
    projections = report.get("_column_results", {}) if report is prepared_report and report is not None else {}
    for column in targets:
        config = column.metadata["stratigraphy"]
        projection = projections.get(column.id)
        if projection is not None and projection[0] == config:
            _, values, diagnostics = projection
            for row, value in zip(target_rows, values, strict=True):
                row.values[column.id] = value
            column.metadata["diagnostics"] = diagnostics.copy()
            column.metadata["stratigraphy_source"] = deepcopy(source)
            continue
        diagnostics = column.metadata["diagnostics"] if scoped else {}
        if (
            report
            and config.get("level") in LEVELS
            and config.get("output") in ("unit", "status")
        ):
            for item in report["results"]:
                result = item["levels"][config["level"]]
                row_map[item["row_id"]].values[column.id] = assignment_value(result, config["output"])
                diagnostics[item["row_id"]] = deepcopy(result)
        else:
            for row in row_map.values():
                row.values[column.id] = None
                diagnostics[row.id] = {
                    "status": "invalid_definition",
                    "reason": error or "Некорректный уровень или вид результата",
                }
        column.metadata["diagnostics"] = diagnostics
        column.metadata["stratigraphy_source"] = deepcopy(source)


def result_column(level, *, status=False):
    if level not in LEVELS:
        raise ValueError("Неизвестный уровень")
    return Column(
        f"{level}" + (" · статус" if status else ""),
        metadata={
            "stratigraphy": {"level": level, "output": "status" if status else "unit"}
        },
    )


def from_records(headers, rows, numbers, *, name, depth_reference, source):
    """Import the existing DB view's exported field names, without guessing MD/TVD."""
    required = [
        "wellbore_name",
        "unit_name",
        "unit_level",
        "top_depth_m",
        "bottom_depth_m",
    ]
    if any(headers.count(key) != 1 for key in required):
        raise ValueError("Нужны уникальные заголовки: " + ", ".join(required))
    indices = [headers.index(key) for key in required]
    intervals = []
    if len(rows) != len(numbers):
        raise ValueError("Номера исходных строк не соответствуют данным")
    for row, number in zip(rows, numbers):
        values = [row[i] if i < len(row) else None for i in indices]
        if all(value is None or value == "" for value in values):
            continue
        well, unit, level, top, bottom = values
        intervals.append(
            {
                "id": new_id(),
                "wellbore": well,
                "name": unit,
                "level": level,
                "top": parse_number(top),
                "bottom": parse_number(bottom),
                "source_row": number,
                "raw_cells": dict(zip(headers, row)),
            }
        )
        if "depth_reference" in headers:
            declared = row[headers.index("depth_reference")]
            if declared and declared != depth_reference:
                raise ValueError(
                    f"Строка {number}: система глубин файла противоречит объявленной"
                )
        for key in STRATAPY_FIELDS:
            if key in headers:
                value = row[headers.index(key)]
                if value is not None and value != "":
                    intervals[-1][key] = value
        if "lithology" in headers:
            rock = row[headers.index("lithology")]
            intervals[-1]["lithology"] = str(rock) if rock is not None else ""
    scheme = {
        "format": "autopyro.stratigraphy",
        "version": 1,
        "id": new_id(),
        "revision": "1",
        "name": name,
        "depth_unit": "m",
        "depth_reference": depth_reference,
        "source": deepcopy(source),
        "intervals": intervals,
    }
    return validate_scheme(scheme)


def draw_layer(axes, project, row_ids, spec, *, language="ru"):
    """One wellbore and hierarchy level per log; never mix depth systems."""
    english = language == "en"
    layer = spec.stratigraphy
    scheme, binding = layer["scheme"], layer["binding"]
    validate_binding(project, scheme, binding)
    columns = {c.id: c for c in project.columns}
    input_ids = [binding["well_id"], binding["depth_id"]]
    if binding.get("sample_mode", "point") == "interval":
        input_ids.append(binding["bottom_id"])
    if any(columns[cid].metadata.get("method") for cid in input_ids):
        raise ValueError(
            "Стратиграфический фон не привязывается к интерполированным стволам или глубинам"
        )
    if spec.kind not in ("depth", "multi_depth") or spec.y_id != binding["depth_id"]:
        raise ValueError(
            "Стратиграфический слой требует глубинного графика с привязанной осью Y"
        )
    if (
        binding.get("sample_mode", "point") == "interval"
        and spec.depth_bottom_id != binding["bottom_id"]
    ):
        raise ValueError(
            "Для интервальной привязки на графике нужна соответствующая колонка низа отбора"
        )
    level = layer["level"]
    if level not in LEVELS:
        raise ValueError("Укажите уровень стратиграфического слоя")
    selected = set(row_ids)
    wells = {
        row.values.get(binding["well_id"]) for row in project.rows if row.id in selected
    }
    if len(wells) != 1:
        if not wells:
            return ["The selection is empty. The stratigraphic layer is hidden." if english
                    else "Пустая выборка: стратиграфический фон не показан"]
        raise ValueError(
            "Для стратиграфического фона выберите пробы ровно одного ствола"
        )
    well = next(iter(wells))
    if not isinstance(well, str) or not well.strip():
        raise ValueError("Не задан ствол проб")
    intervals = [
        item
        for item in scheme["intervals"]
        if item["wellbore"] == well and item["level"] == level
    ]
    factor = UNITS[scheme["depth_unit"]] / UNITS[binding["depth_unit"]]
    if spec.y_axis.scale == "log" and any(item["top"] <= 0 for item in intervals):
        raise ValueError(
            "Стратиграфические границы логарифмической оси должны быть положительными"
        )
    if intervals:
        draw_columns(axes, intervals, factor, level, spec, language=language)
    notices = []
    if not intervals:
        notices.append("No intervals for the selected wellbore and level" if english
                       else "Для выбранного ствола и уровня нет интервалов")
    if _track_has_overlap(intervals):
        notices.append("The stratigraphic layer has overlapping intervals at the same level" if english
                       else "Стратиграфический слой содержит перекрытия на одном уровне")
    current = project.view.get("stratigraphy", {}).get("scheme")
    if current and scheme_hash(current) != scheme_hash(scheme):
        notices.append(
            "The stratigraphic layer shows a saved revision that differs from the current revision"
            if english else "Стратиграфический слой показывает сохранённую редакцию, отличающуюся от текущей"
        )
    if spec.stratigraphy_info:
        figure = axes[0].figure
        width = max(
            15,
            int(
                figure.get_figwidth()
                * 72
                * (spec.right - spec.left)
                / spec.stratigraphy_label_size
                / 0.65
            ),
        )
        info = f"{well} | {level} | {scheme['name']} rev {scheme['revision']} | {scheme['depth_reference']}"
        figure.text(
            spec.left,
            0.015,
            textwrap.fill(info, width=width),
            va="bottom",
            fontsize=spec.stratigraphy_label_size,
            fontfamily=spec.font_family,
        )
    return notices


def valid_horizon(horizon):
    anchors = horizon.get('anchors')
    return (horizon.get('edge') in ('top', 'bottom') and isinstance(anchors, dict)
            and all(isinstance(well, str) and isinstance(key, str) and key for well, key in anchors.items()))


def render_columns(scheme, wellbore, level, *, figure=None, options=None, samples=None):
    """Build a complete column from a scheme without a project or sample binding."""
    from matplotlib.figure import Figure
    from autopyro.visualization.plotting import PlotSpec

    style = column_style(options)
    validate_scheme(scheme)
    if level not in LEVELS:
        raise ValueError("Укажите уровень стратиграфической колонки")
    selected_levels = [rank for rank in LEVELS if rank == level or rank in style['hierarchy_levels']]
    wells = list(dict.fromkeys([wellbore, *style['comparison_wells']]))
    comparison = len(wells) > 1
    by_well = {well: {rank: [] for rank in selected_levels} for well in wells}
    for item in scheme['intervals']:
        if item['wellbore'] in by_well and item['level'] in selected_levels:
            by_well[item['wellbore']][item['level']].append(item)
    intervals = [item for ranks in by_well.values() for group in ranks.values() for item in group]
    if not intervals:
        raise ValueError("Для выбранного ствола и уровня нет интервалов")
    missing_levels = [(well, rank) for well, ranks in by_well.items() for rank, group in ranks.items() if not group]
    by_well = {well: {rank: group for rank, group in ranks.items() if group}
               for well, ranks in by_well.items() if any(ranks.values())}
    has_overlap = any(_track_has_overlap(group) for ranks in by_well.values() for group in ranks.values())
    origins = {}
    correlation = style['correlation']
    if not correlation.get('enabled', True):
        correlation = {}
    if correlation:
        by_id = {item['id']: item for item in scheme['intervals']}
        for well in by_well:
            anchor = by_id.get(correlation['anchors'].get(well))
            if anchor is None or anchor['wellbore'] != well:
                raise ValueError(f"Выберите опорный интервал для ствола: {well}")
            origins[well] = anchor[correlation['edge']]
        by_well = {
            well: {rank: [dict(item, top=item['top'] - origins[well], bottom=item['bottom'] - origins[well])
                          for item in group] for rank, group in ranks.items()}
            for well, ranks in by_well.items()
        }
        limits_intervals = [item for ranks in by_well.values() for group in ranks.values() for item in group]
    else:
        limits_intervals = intervals
    figure = figure if figure is not None else Figure()
    figure.clear()
    figure.set_size_inches(style["width_mm"] / 25.4, style["height_mm"] / 25.4)
    figure.set_dpi(style["dpi"])
    figure.patch.set_alpha(0 if style["transparent"] else 1)
    bounds = [0.16, 0.16 if style["x_axis"] else 0.12, 0.78, 0.72]
    if style["legend"]:
        location = style["legend_loc"]
        if location == "right":
            bounds[2] = .52
        elif location == "left":
            bounds[0], bounds[2] = .4, .54
        elif location == "top":
            bounds[3] = .56
        else:
            bounds[1], bounds[3] = .32, .52
    axis = figure.add_axes(bounds)
    top = style["depth_min"] if style["depth_min"] is not None else min(i["top"] for i in limits_intervals)
    bottom = style["depth_max"] if style["depth_max"] is not None else max(i["bottom"] for i in limits_intervals)
    samples = {well: rows for well, rows in (samples or {}).items() if well in by_well and rows}
    for well, rows in samples.items():
        origin = origins.get(well, 0)
        if style['depth_min'] is None:
            top = min(top, min(row.top for row in rows) - origin)
        if style['depth_max'] is None:
            bottom = max(bottom, max(row.top if row.bottom is None else row.bottom for row in rows) - origin)
    if top >= bottom:
        raise ValueError("Минимальная глубина должна быть меньше максимальной")
    axis.set_ylim(bottom, top)
    spec = PlotSpec(stratigraphy_alpha=style["alpha"], stratigraphy_labels=style["labels"],
                    stratigraphy_label_size=style["label_size"], font_family=style["font_family"])
    import matplotlib as mpl
    from stratapy.setup import formatting

    old_sizes = formatting.fontsizes.copy()
    old_lithologies = formatting.lithologies.copy()
    horizon_notices = []
    try:
        if style["lithologies"]:
            import stratapy as sp

            if set(style["lithologies"]) & formatting.default_lithologies.keys():
                raise ValueError("Для пользовательской литологии укажите новый код")
            sp.update_lithologies({key: tuple(value) for key, value in style["lithologies"].items()})
            if set(style["lithologies"]) - formatting.lithologies.keys():
                raise ValueError("Не удалось создать пользовательскую литологию stratapy")
        formatting.fontsizes.update({"x_axis_label": style["axis_size"],
                                    "y_axis_label": style["axis_size"],
                                    "x_tick_labels": style["tick_size"],
                                    "y_tick_labels": style["tick_size"],
                                    "grain_brackets": style["tick_size"],
                                    "legend_entry": style["legend_size"],
                                    "legend_subtitle": style["legend_size"]})
        with mpl.rc_context({"font.family": style["font_family"]}):
            draw_well_columns(figure, axis, by_well, level, spec, style, comparison=comparison,
                              samples=samples, origins=origins)
            for track in figure.axes:
                if origins:
                    track._autopyro_depth_origin = origins[track._autopyro_well]
                    track.axhline(0, color='#3879a6', linestyle='--', linewidth=1.25, zorder=20)
            if correlation.get('horizons'):
                horizon_notices = draw_correlation_horizons(figure, correlation['horizons'], by_id, origins)
            label = style['y_label']
            if origins and label == COLUMN_STYLE['y_label']:
                label = 'Глубина относительно горизонта'
            axis.set_ylabel(f"{label} [{scheme['depth_unit']}]" if label else "",
                            fontsize=style["axis_size"])
            figure.suptitle(style["title"] or (scheme['name'] if comparison else f"{wellbore} · {scheme['name']}"),
                            fontsize=style["title_size"])
            if style["info"]:
                figure.text(0.16, 0.02, f"{scheme['depth_reference']} | rev {scheme['revision']}")
    finally:
        formatting.lithologies.clear()
        formatting.lithologies.update(old_lithologies)
        formatting.fontsizes.clear()
        formatting.fontsizes.update(old_sizes)
    notices = (["Стратиграфическая колонка содержит перекрытия на одном уровне"]
               if has_overlap else [])
    notices.extend(f"Нет интервалов уровня: {rank}" + (f" · {well}" if comparison else '')
                   for well, rank in missing_levels)
    notices.extend(horizon_notices)
    return figure, notices


def draw_correlation_horizons(figure, horizons, by_id, origins):
    from matplotlib.patches import ConnectionPatch
    from matplotlib.transforms import Bbox, TransformedBbox

    tracks = defaultdict(list)
    for axis in figure.axes:
        tracks[axis._autopyro_well].append(axis)
    wells = list(tracks)
    frame = figure.axes[0].get_position()
    clip = TransformedBbox(Bbox.from_extents(0, frame.y0, 1, frame.y1), figure.transFigure)
    colors = ('#a03c36', '#217454', '#794d99', '#9a620e', '#346b83', '#795846')
    notices = []
    for number, horizon in enumerate(horizons):
        color = colors[number % len(colors)]
        depths = {}
        for well in wells:
            key = horizon['anchors'].get(well)
            if key is None:
                continue
            item = by_id.get(key)
            if item is None or item['wellbore'] != well:
                notices.append(f"Интервал горизонта «{horizon['name']}» недоступен в стволе {well}.")
                continue
            depth = item[horizon['edge']] - origins[well]
            depths[well] = depth
            for axis in tracks[well]:
                axis.axhline(depth, color=color, linewidth=1.15, zorder=21)
        label_well = next(iter(depths), None)
        for left, right in zip(wells, wells[1:]):
            if left not in depths or right not in depths:
                continue
            last, first = tracks[left][-1], tracks[right][0]
            connector = ConnectionPatch(
                (1, depths[left]), (0, depths[right]),
                last.get_yaxis_transform(), first.get_yaxis_transform(),
                color=color, linewidth=1.15, zorder=22, clip_on=True,
            )
            connector.set_clip_box(clip)
            figure.add_artist(connector)
        if label_well is not None:
            axis = tracks[label_well][-1]
            axis.annotate(horizon['name'], (.98, depths[label_well]),
                          xycoords=axis.get_yaxis_transform(), ha='right',
                          xytext=(-2, 3), textcoords='offset points',
                          color=color, fontsize=8, zorder=23, annotation_clip=True,
                          bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85, 'pad': 1})
    return notices


def draw_well_columns(figure, axis, by_well, level, spec, style, *, comparison, samples, origins):
    """Lay out well blocks, then join all their tracks on one depth scale."""
    bounds = list(axis.get_position().bounds)
    bottom, top = axis.get_ylim()
    # Each well gets its own tracks and legend region on the common display scale.
    count = len(by_well)
    for index, (well, by_level) in enumerate(by_well.items()):
        local_axis = axis if index == 0 else figure.add_axes(bounds)
        local_bounds = [bounds[0] / count + index / count, bounds[1], bounds[2] / count, bounds[3]]
        sample_bounds = None
        if well in samples:
            width = local_bounds[2]
            local_bounds[2] = width * .79
            sample_bounds = [local_bounds[0] + width * .82, bounds[1], width * .18, bounds[3]]
        local_axis.set_position(local_bounds)
        local_axis.set_ylim(bottom, top)
        local_style = dict(style)
        if comparison:
            anchor, location = {'right': ((.72, .5), 'center left'),
                                'left': ((.01, .5), 'center left'),
                                'top': ((.5, .9), 'upper center'),
                                'bottom': ((.5, .08), 'lower center')}[style['legend_loc']]
            legend = {'bbox_to_anchor': anchor, 'loc': location, **style['legend_kwargs']}
            anchor = list(legend['bbox_to_anchor'] or anchor)
            anchor[0] = (index + anchor[0]) / count
            if len(anchor) == 4:
                anchor[2] /= count
            legend['bbox_to_anchor'] = tuple(anchor)
            local_style['legend_kwargs'] = legend
            figure.text((index + .5) / count, .94, well, ha='center',
                        fontsize=style['title_size'])
        start = len(figure.axes) - 1
        if len(by_level) == 1:
            rank, group = next(iter(by_level.items()))
            track_style = local_style if rank == level else dict(
                local_style, display_mode='log', feature_mode='off', x_axis=False, x_label='', legend=False)
            draw_columns([local_axis], group, 1.0, rank, spec, standalone=True, options=track_style,
                         include_lithology=rank == level)
        else:
            draw_hierarchy(figure, local_axis, by_level, level, spec, local_style)
        for track in figure.axes[start:]:
            track._autopyro_well = well
        if sample_bounds is not None:
            from .stratigraphy_samples import draw_sample_track

            sample_axis = figure.add_axes(sample_bounds)
            sample_axis._autopyro_kind = 'samples'
            sample_axis._autopyro_well = well
            draw_sample_track(sample_axis, samples[well], origins.get(well, 0))
    for track in figure.axes:
        if track is not axis and not track.get_shared_y_axes().joined(axis, track):
            track.sharey(axis)
        track.tick_params(axis='y', which='both', left=track is axis and style['y_axis'],
                          labelleft=track is axis and style['y_axis'])
    axis.set_ylim(bottom, top)


def draw_hierarchy(figure, first_axis, by_level, primary_level, spec, style):
    """Allocate shared-depth tracks; parse each level once, even if it has lithology."""
    primary = by_level.get(primary_level, [])
    has_lithology = any(item.get('lithology', '').strip() for item in primary)
    count = len(by_level) + int(has_lithology)
    position = first_axis.get_position().frozen()
    limits = first_axis.get_ylim()
    gap = position.width * .035
    width = (position.width - gap * (count - 1)) / count
    axes = {}
    for index, rank in enumerate(by_level):
        bounds = [position.x0 + index * (width + gap), position.y0, width, position.height]
        axis = first_axis if index == 0 else figure.add_axes(bounds)
        axis.set_position(bounds)
        axis.set_ylim(limits)
        axes[rank] = axis
    lithology_axis = None
    if has_lithology:
        lithology_axis = figure.add_axes([position.x0 + (count - 1) * (width + gap),
                                         position.y0, width, position.height])
        lithology_axis.set_ylim(limits)
    for rank, group in by_level.items():
        track_style = dict(style)
        targets = [axes[rank]]
        if rank == primary_level and lithology_axis is not None:
            targets.append(lithology_axis)
        if rank != primary_level:
            track_style.update(display_mode='log', feature_mode='off', x_axis=False,
                               x_label='', legend=False)
        draw_columns([axes[rank]], group, 1.0, rank, spec, standalone=True,
                     options=track_style, target_axes=targets,
                     include_lithology=rank == primary_level)
    for axis in [*axes.values(), *([lithology_axis] if lithology_axis is not None else [])]:
        if axis is not first_axis and not axis.get_shared_y_axes().joined(first_axis, axis):
            axis.sharey(first_axis)
        axis.set_ylim(limits)
        axis.tick_params(axis='y', which='both', left=axis is first_axis and style['y_axis'],
                         right=False, labelleft=axis is first_axis and style['y_axis'])


def draw_columns(axes, intervals, factor, level, spec, *, standalone=False, options=None,
                 target_axes=None, include_lithology=True, language="ru"):
    """Render native stratapy logs on the same depth scale as the sample tracks."""
    import csv
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import stratapy as sp

    style = column_style(options) if standalone else None
    intervals = sorted(intervals, key=lambda item: (item["top"], item["bottom"]))
    # Use explicit tops and thicknesses: gaps and overlaps must never be inferred away.
    with TemporaryDirectory(prefix="autopyro-strata-") as directory:
        path = Path(directory) / "intervals.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["height/age", "rock", "thickness", *STRATAPY_FIELDS])
            writer.writerows(
                # A valid placeholder preserves grain sizes and features while loading.
                # stratapy's "no" input discards these fields; actual rocks are assigned below.
                (item["top"] * factor, "607", (item["bottom"] - item["top"]) * factor,
                 *(item.get(key, "") for key in STRATAPY_FIELDS))
                for item in intervals
            )
        load_options = {key: style[key] for key in ("grain_preset", "x_ticks_dict", "grain_brackets")} if style else {}
        log = sp.load(str(path), **load_options)
    # Auto-detection treats a single row as height. Our data always increase downwards.
    log.y_mode = "age"
    lithologies = dict(log.lithologies)
    names = {value[-1].lower(): key for key, value in lithologies.items()}
    rocks = []
    for item in intervals:
        rock = item.get("lithology", "").strip().lower()
        rock = names.get(rock, rock) if rock else "no"
        if rock not in lithologies:
            raise ValueError(f"Неизвестная литология stratapy: {item['lithology']}")
        rocks.append(rock)

    figure = axes[0].figure
    if not hasattr(figure.canvas, "get_renderer"):
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        FigureCanvasAgg(figure)
    limit = axes[0].get_ylim()
    positions = [ax.get_position().frozen() for ax in axes]
    left, right = min(p.x0 for p in positions), max(p.x1 for p in positions)
    width = right - left
    has_lithology = include_lithology and any(item.get("lithology", "").strip() for item in intervals)
    reserved = width if standalone else width * (0.36 if has_lithology else 0.24)
    gap = width * 0.025
    if not standalone:
        for ax, position in zip(axes, positions):
            ax.set_position([left + (position.x0 - left) * (width - reserved) / width,
                             position.y0, position.width * (width - reserved) / width,
                             position.height])
    column_width = (reserved - gap * (2 if has_lithology else 1)) / (2 if has_lithology else 1)
    base_df = log.original_df.copy(deep=True)
    for index, lithology in enumerate([False, True] if has_lithology else [False]):
        bounds = [right - reserved + gap + index * (column_width + gap),
                  positions[0].y0, column_width, positions[0].height]
        if target_axes is not None:
            column = target_axes[index]
            column.set_label("lithology" if lithology else "stratigraphy")
        elif standalone and index == 0:
            column = axes[0]
            column.set_position(bounds)
            column.set_label("stratigraphy")
        else:
            column = figure.add_axes(bounds, label="lithology" if lithology else "stratigraphy")
        column._autopyro_level = level
        column._autopyro_kind = "lithology" if lithology else "stratigraphy"
        log.original_df = base_df.copy(deep=True)
        log.lithologies = dict(lithologies)
        if lithology:
            log.original_df["rock"] = rocks
        else:
            keys = [f"autopyro_{i}" for i in range(len(intervals))]
            log.original_df["rock"] = keys
            log.lithologies.update({
                key: ("fill", interval_color(item), "", "k", item["name"])
                for key, item in zip(keys, intervals)
            })
        plot_options = dict(display_mode="log", feature_mode="off", legend=False,
                            x_axis=False, y_label="", consec_units=False,
                            dpi=max(1, round(figure.dpi)))
        if style:
            native_keys = ("display_mode", "feature_mode", "unit_borders", "spines", "consec_units",
                           "legend_loc", "legend_columns", "legend_border", "legend_kwargs",
                           "ppi", "x_axis", "x_label", "xmax", "mineral_size", "feature_size", "legend_titles")
            plot_options.update({key: deepcopy(style[key]) for key in native_keys})
            anchors = {"right": ((.72, .5), "center left"), "left": ((.01, .5), "center left"),
                       "top": ((.5, .9), "upper center"), "bottom": ((.5, .08), "lower center")}
            anchor, location = anchors[style["legend_loc"]]
            plot_options["legend_kwargs"] = {"bbox_to_anchor": anchor, "loc": location,
                                             **plot_options["legend_kwargs"]}
            # One native legend describes the lithologies (or units when lithology is absent).
            plot_options["legend"] = style["legend"] and (lithology or not has_lithology)
        log.plot(fig=figure, ax=column, **plot_options)
        if style:
            column.patch.set_alpha(0 if style["transparent"] else 1)
            if style["grid"]:
                column.grid(True, axis="y", alpha=.3)
            else:
                column.grid(False, axis="y")
        column.set_yscale(axes[0].get_yscale())
        column.set_ylim(limit)
        if column is not axes[0]:
            column.sharey(axes[0])
        column.tick_params(axis="y", which="both", left=standalone and index == 0 and style["y_axis"],
                           right=False, labelleft=standalone and index == 0 and style["y_axis"])
        lithology_title = "Lithology" if language == "en" else "Литология"
        column.set_title(lithology_title if lithology else level,
                         fontsize=spec.stratigraphy_label_size, fontfamily=spec.font_family)
        for artist in [*column.patches, *column.images]:
            artist.set_alpha(spec.stratigraphy_alpha)
            artist.set_clip_on(True)
            artist.set_clip_box(column.bbox)
        for line in column.lines:
            line.set_clip_on(True)
            line.set_clip_path(column.patch)
        if spec.stratigraphy_labels:
            label_width = max(5, int(column.get_position().width * figure.get_figwidth() * 72
                                    / spec.stratigraphy_label_size / 0.65))
            labels = [item.get("lithology", "") if lithology else item["name"]
                      for item in intervals]
            existing_texts = len(column.texts)
            log.add_labels(labels, fontsize=spec.stratigraphy_label_size)
            for text, item, label in zip(list(column.texts)[existing_texts:], intervals, labels):
                midpoint = (item["top"] + item["bottom"]) * factor / 2
                text.set_visible(min(limit) <= midpoint <= max(limit))
                text.set_position((0.5, midpoint))
                text.set_transform(column.get_yaxis_transform())
                text.set_text(textwrap.fill(label, width=label_width))
                text.set_ha("center")
                text.set_fontfamily(spec.font_family)
                text.set_clip_on(True)
                text.set_clip_path(column.patch)
