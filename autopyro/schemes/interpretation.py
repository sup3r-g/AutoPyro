"""Planar geometric interpretation with explicit coordinates, units and boundaries.

Schemes supply their own regions. This engine does not invent geochemical fields.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from copy import deepcopy
from typing import Protocol

import numpy as np
import shapely
from shapely.geometry import LineString, Point, box, shape
from shapely.validation import explain_validity

from autopyro.model.core import timestamp
from autopyro.model.tabular import classify


STATUSES = {
    "classified": "В области",
    "boundary": "На границе",
    "ambiguous": "Неоднозначно",
    "outside": "Вне областей",
    "insufficient_data": "Недостаточно данных",
    "not_applicable": "Вне области применимости",
}


@dataclass
class InterpretationInput:
    row_id: str
    x: float | None
    y: float | None
    x_bounds: tuple[float, float] | None = None
    y_bounds: tuple[float, float] | None = None


@dataclass
class InterpretationResult:
    row_id: str
    status: str
    regions: list[str] = field(default_factory=list)
    boundary_regions: list[str] = field(default_factory=list)
    line_hits: list[str] = field(default_factory=list)
    uncertainty_regions: list[str] = field(default_factory=list)
    distances: dict = field(default_factory=dict)
    reason: str = ""
    value: float | None = None
    value_bounds: list[float] | None = None
    possible_types: list[str] = field(default_factory=list)
    interval_details: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)


class InterpretationEngine(Protocol):
    def evaluate(
        self, samples: list[InterpretationInput], scheme: dict
    ) -> list[InterpretationResult]: ...


def _result_record(result):
    record = asdict(result)
    if not record['thresholds']:
        del record['thresholds']
    return record


def transform(value, axis):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Неконечная координата")
    if axis.get("minimum") is not None and value < axis["minimum"]:
        raise ValueError("Координата ниже области применимости")
    if axis.get("maximum") is not None and value > axis["maximum"]:
        raise ValueError("Координата выше области применимости")
    if axis.get("transform", "identity") == "log10":
        if value <= 0:
            raise ValueError("Неположительная координата для log10")
        value = math.log10(value / axis.get("reference", 1.0))
    return (value - axis.get("origin", 0.0)) / axis.get("scale", 1.0)


def inverse(value, axis):
    value = value * axis.get("scale", 1.0) + axis.get("origin", 0.0)
    return (
        10**value * axis.get("reference", 1.0)
        if axis.get("transform", "identity") == "log10"
        else value
    )


def validate_scheme(scheme):
    if not isinstance(scheme, dict):
        raise ValueError("Схема должна быть объектом JSON")
    if scheme.get("format") != "autopyro.interpretation" or scheme.get("version") != 1:
        raise ValueError("Неизвестный формат схемы")
    for key in ("id", "revision", "name", "source"):
        if not isinstance(scheme.get(key), str) or not scheme[key].strip():
            raise ValueError(f"Обязательное поле схемы: {key}")
    if scheme.get("scientific_status") not in (
        "demonstration",
        "user_supplied",
        "validated",
    ):
        raise ValueError(
            "Укажите scientific_status: demonstration / user_supplied / validated"
        )
    if scheme.get("coordinate_space") != "analysis":
        raise ValueError(
            "Границы должны быть заданы в явно объявленном пространстве analysis"
        )
    if scheme.get("boundary_policy", "flag") not in ("flag", "include", "exclude"):
        raise ValueError("Неизвестная политика границ")
    tolerance = scheme.get("tolerance", 0.0)
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("Допуск границы должен быть конечным и неотрицательным")
    if scheme.get("distance_metric") not in (None, "analysis_euclidean"):
        raise ValueError("Неизвестная метрика расстояния")
    if scheme.get("distance_metric") and not scheme.get("metric_note", "").strip():
        raise ValueError("Для расстояний обоснуйте нормировку осей в metric_note")
    for axis in (scheme.get("x", {}), scheme.get("y", {})):
        if not isinstance(axis, dict):
            raise ValueError("Определение оси должно быть объектом JSON")
        if not axis.get("property") or not axis.get("unit"):
            raise ValueError("Для каждой оси укажите property и unit")
        if axis.get("transform", "identity") not in ("identity", "log10"):
            raise ValueError("Неизвестное преобразование координат")
        for key in ("reference", "origin", "scale", "minimum", "maximum"):
            value = axis.get(key)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("Неконечный параметр преобразования")
            if (
                key in ("reference", "origin", "scale")
                and key in axis
                and value is None
            ):
                raise ValueError("Параметры преобразования не могут быть null")
        if axis.get("scale", 1) <= 0 or axis.get("reference", 1) <= 0:
            raise ValueError("Масштаб и reference должны быть положительными")
        if (
            axis.get("minimum") is not None
            and axis.get("maximum") is not None
            and axis["minimum"] >= axis["maximum"]
        ):
            raise ValueError("Некорректная область применимости")
    regions, lines, ids = [], [], set()
    for entries, allowed, target in (
        (scheme.get("regions", []), ("Polygon", "MultiPolygon"), regions),
        (scheme.get("lines", []), ("LineString", "MultiLineString"), lines),
    ):
        if not isinstance(entries, list):
            raise ValueError("Области и линии должны быть списками")
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(
                entry.get("geometry"), dict
            ):
                raise ValueError(
                    "Геометрический объект должен содержать GeoJSON geometry"
                )
            if (
                not isinstance(entry.get("id"), str)
                or not entry["id"]
                or entry["id"] in ids
                or not isinstance(entry.get("label"), str)
                or not entry["label"]
            ):
                raise ValueError("Объекты должны иметь уникальные id и подписи")
            ids.add(entry["id"])
            try:
                geometry = shape(entry["geometry"])
            except (
                shapely.errors.ShapelyError,
                ValueError,
                TypeError,
                KeyError,
            ) as exc:
                raise ValueError(f"Невозможно прочитать геометрию: {exc}") from exc
            if geometry.geom_type not in allowed or geometry.is_empty or geometry.has_z:
                raise ValueError("Неверный тип, пустая или трёхмерная геометрия")
            coordinates = shapely.get_coordinates(geometry)
            if not np.isfinite(coordinates).all() or not geometry.is_valid:
                raise ValueError(
                    f"Некорректная геометрия {entry['id']}: {explain_validity(geometry)}"
                )
            if geometry.geom_type in ("Polygon", "MultiPolygon") and geometry.area <= 0:
                raise ValueError("Площадь области должна быть положительной")
            if "passport" in entry:
                passport = entry["passport"]
                if (target is not regions or not isinstance(passport, dict)
                        or any(key not in ("source", "description", "limitations")
                               or not isinstance(value, str) for key, value in passport.items())):
                    raise ValueError(f"Некорректный паспорт области {entry['id']}")
            if "threshold" in entry:
                from .interpretation_thresholds import prepare_threshold

                prepare_threshold(entry, geometry, scheme.get("strategy", "regions"))
            target.append((entry, geometry))
    if not regions and not lines:
        raise ValueError("Добавьте хотя бы одну область или линию")
    from autopyro.curves.curve_interpretation import prepare_curves
    prepare_curves(scheme, lines)
    if "reference_cases" in scheme:
        from .reference_cases import validate_reference_cases

        validate_reference_cases(scheme["reference_cases"])
    return regions, lines


class GeometricEngine:
    def __init__(self, *, legacy_point_hio=False):
        self.legacy_point_hio = legacy_point_hio

    def evaluate(self, samples, scheme, *, checkpoint=None):
        regions, lines = validate_scheme(scheme)
        if scheme.get("strategy", "regions") != "regions":
            from autopyro.curves.curve_interpretation import evaluate_curves, prepare_curves
            return evaluate_curves(samples, scheme, prepare_curves(scheme, lines), checkpoint=checkpoint,
                                   legacy_point_hio=self.legacy_point_hio)
        region_tree = shapely.STRtree([geometry for _, geometry in regions])
        from .interpretation_thresholds import prepare_threshold

        strategy = scheme.get("strategy", "regions")
        thresholds = [prepare_threshold(entry, geometry, strategy)
                      for entry, geometry in lines if "threshold" in entry]
        reference_lines = [(entry, geometry) for entry, geometry in lines if "threshold" not in entry]
        tolerance = scheme.get("tolerance", 0.0)
        measure_distances = bool(scheme.get("distance_metric"))
        boundaries = [geometry.boundary for _, geometry in regions]
        policy = scheme.get("boundary_policy", "flag")
        results = []
        for index, sample in enumerate(samples):
            if checkpoint is not None and index % 64 == 0:
                checkpoint()
            result = InterpretationResult(sample.row_id, "outside")
            results.append(result)
            if sample.x is None or sample.y is None:
                result.status, result.reason = (
                    "insufficient_data",
                    "Отсутствует допустимое числовое значение X или Y",
                )
                continue
            try:
                point = Point(
                    transform(sample.x, scheme["x"]), transform(sample.y, scheme["y"])
                )
            except (ValueError, OverflowError):
                result.status, result.reason = (
                    "not_applicable",
                    "Координаты вне области применимости или преобразования",
                )
                continue
            inside, boundary = [], []
            if measure_distances:
                region_indices = range(len(regions))
            elif tolerance:
                x, y = point.x, point.y
                region_indices = sorted(region_tree.query(box(x - tolerance, y - tolerance,
                                                              x + tolerance, y + tolerance)))
            else:
                region_indices = sorted(region_tree.query(point))
            for i in region_indices:
                entry, geometry = regions[i]
                distance = boundaries[i].distance(point)
                on_boundary = distance <= tolerance
                if on_boundary:
                    boundary.append(entry["id"])
                elif geometry.contains(point):
                    inside.append(entry["id"])
                if measure_distances:
                    result.distances[entry["id"]] = distance
            result.boundary_regions = boundary
            result.regions = inside + (boundary if policy == "include" else [])
            result.line_hits = [
                entry["id"]
                for entry, geometry in reference_lines
                if geometry.distance(point) <= tolerance
            ]
            if len(inside) > 1 or (policy == "include" and len(result.regions) > 1):
                result.status = "ambiguous"
            elif boundary or result.line_hits:
                result.status = "boundary"
            elif inside:
                result.status = "classified"
            bounds = None
            if sample.x_bounds is not None or sample.y_bounds is not None:
                try:
                    xb = sample.x_bounds or (sample.x, sample.x)
                    yb = sample.y_bounds or (sample.y, sample.y)
                    if (
                        xb[0] > xb[1]
                        or yb[0] > yb[1]
                        or not xb[0] <= sample.x <= xb[1]
                        or not yb[0] <= sample.y <= yb[1]
                    ):
                        raise ValueError("Интервал не содержит измерение")
                    bounds = (
                        transform(xb[0], scheme["x"]),
                        transform(yb[0], scheme["y"]),
                        transform(xb[1], scheme["x"]),
                        transform(yb[1], scheme["y"]),
                    )
                    footprint = (
                        box(*bounds)
                        if bounds[0] != bounds[2] and bounds[1] != bounds[3]
                        else LineString(
                            [(bounds[0], bounds[1]), (bounds[2], bounds[3])]
                        )
                        if bounds[0] != bounds[2] or bounds[1] != bounds[3]
                        else point
                    )
                    result.uncertainty_regions = [regions[i][0]['id'] for i in
                                                  sorted(region_tree.query(footprint, predicate='intersects'))]
                    covers = region_tree.query(footprint, predicate='covered_by')
                    if (
                        len(result.uncertainty_regions) > 1
                        or (result.regions and len(covers) == 0)
                        or (result.status == "outside" and result.uncertainty_regions)
                    ):
                        result.status = "ambiguous"
                except (ValueError, OverflowError):
                    result.status, result.reason = (
                        "not_applicable",
                        "Недопустимый интервал неопределённости",
                    )
                    continue
            if thresholds:
                nominal = (point.x, point.y, point.x, point.y)
                for threshold in thresholds:
                    relation = threshold.relations(nominal, tolerance)[0]
                    possible = threshold.relations(bounds, tolerance) if bounds is not None else [relation]
                    result.thresholds[threshold.id] = dict(relation=relation, possible_relations=possible)
                    if relation == 'boundary':
                        result.line_hits.append(threshold.id)
                relations = [item['relation'] for item in result.thresholds.values()]
                threshold_ambiguous = any(len(item['possible_relations']) > 1 for item in result.thresholds.values())
                if threshold_ambiguous:
                    result.status = 'ambiguous'
                elif result.status != 'ambiguous' and 'boundary' in relations:
                    result.status = 'boundary'
                elif result.status == 'outside' and any(r != 'outside_domain' for r in relations):
                    result.status = 'classified'
                elif result.status == 'outside' and not regions and not reference_lines:
                    result.status = 'not_applicable'
            rule = {
                "flag": "отмечать отдельно",
                "include": "включать",
                "exclude": "исключать",
            }[policy]
            result.reason = (
                f"Внутри: {', '.join(inside) or '—'}; границы: {', '.join(boundary) or '—'}; "
                f"линии: {', '.join(result.line_hits) or '—'}; правило границ: {rule}"
            )
            if sample.x_bounds is not None or sample.y_bounds is not None:
                result.reason += "; области диапазона неопределённости: " + (
                    ", ".join(result.uncertainty_regions) or "—"
                )
            if thresholds and threshold_ambiguous:
                result.reason += "; интервал допускает несколько исходов порогового правила"
        return results


def unit_key(value):
    return {"%": "%wt", "% wt": "%wt", "C": "°C", "degC": "°C"}.get(value, value)


def validate_axis_binding(column, declared, axis):
    from .interpretation_quality import column_property_name, property_name

    if column_property_name(column) != property_name(axis['property']):
        raise ValueError(f"Для оси нужен показатель {axis['property']}, выбрана колонка «{column.name}»")
    if not declared:
        raise ValueError(f"Укажите единицу для колонки «{column.name}»")
    if unit_key(declared) != unit_key(axis["unit"]):
        raise ValueError(
            f"Единица {column.name} не соответствует схеме; выполните перевод отдельно"
        )
    if column.unit and unit_key(column.unit) != unit_key(declared):
        raise ValueError("Единица исходной колонки противоречит декларации")


def validate_binding(project, x_id, y_id, x_unit, y_unit, scheme):
    validate_scheme(scheme)
    columns = {c.id: c for c in project.columns}
    if x_id == y_id or x_id not in columns or y_id not in columns:
        raise ValueError("Выберите две разные существующие колонки")
    for cid, declared, axis in (
        (x_id, x_unit, scheme["x"]),
        (y_id, y_unit, scheme["y"]),
    ):
        validate_axis_binding(columns[cid], declared, axis)


def evaluate_project(
    project,
    row_ids,
    x_id,
    y_id,
    x_unit,
    y_unit,
    scheme,
    *,
    text_numbers=False,
    engine=None,
    checkpoint=None,
):
    scheme = deepcopy(scheme)
    validate_binding(project, x_id, y_id, x_unit, y_unit, scheme)
    from .interpretation_review import review_scheme_hash
    columns = {c.id: c for c in project.columns}
    row_ids = set(row_ids)
    if row_ids - set(project.row_map()):
        raise ValueError("Выборка содержит отсутствующие пробы")
    from .interpretation_quality import InterpretationQuality, apply_quality, RANGES
    quality = InterpretationQuality(project, x_id, y_id, x_unit, y_unit, scheme, text_numbers)
    samples, input_values = [], []
    selected_rows = [row for row in project.rows if row.id in row_ids]
    axis_data = {}
    if checkpoint is not None:
        checkpoint()
    quality_results = quality.check_many(selected_rows, axis_data=axis_data)
    from autopyro.analysis.qc import is_excluded
    from .measurement_uncertainty import measurement_interval
    for index, row in enumerate(selected_rows):
        if checkpoint is not None and index % 256 == 0:
            checkpoint()
        issues, blocked = quality_results[row.id]
        values, statuses = [], []
        for cid in (x_id, y_id):
            column = columns[cid]
            if cid in axis_data:
                status, value = next(axis_data[cid])
            elif is_excluded(project, row.id, cid, "interpretation"):
                status, value = "qc_excluded", None
            elif column.metadata.get("method"):
                status, value = "interpolation_excluded", None
            else:
                status, value = classify(
                    row.values[cid], numeric=column.kind == "number" or text_numbers
                )
            if cid in blocked:
                status, value = 'range_excluded', None
            values.append(value)
            statuses.append(status)
        bounds, uncertainty, measurements = {}, {}, {}
        for key, cid, unit, value in zip(('x', 'y'), (x_id, y_id), (x_unit, y_unit), values, strict=True):
            interval, metadata, warning = measurement_interval(project, row.id, cid, value, unit)
            bounds[key] = interval
            if metadata is not None:
                uncertainty[key] = metadata
            measurement = project.science.current(row.id, cid)
            if measurement is not None:
                measurements[key] = {"id": measurement.id, "revision": measurement.revision}
            if warning:
                issues.append(dict(column_id=cid, property=scheme[key]['property'], code='uncertainty_unchecked',
                                   severity='warning', message=f"{scheme[key]['property']}: {warning}"))
        samples.append(InterpretationInput(row.id, *values, x_bounds=bounds['x'], y_bounds=bounds['y']))
        input_values.append(
            {
                "row_id": row.id,
                "x": row.values[x_id],
                "y": row.values[y_id],
                "statuses": statuses,
                "quality": issues,
                "quality_blocked": bool(blocked),
                "x_bounds": list(bounds["x"]) if bounds["x"] is not None else None,
                "y_bounds": list(bounds["y"]) if bounds["y"] is not None else None,
                "uncertainty": uncertainty,
                "measurements": measurements,
            }
        )
    if checkpoint is not None:
        checkpoint()
    if engine is None:
        engine = GeometricEngine()
        results = list(engine.evaluate(samples, scheme, checkpoint=checkpoint))
    else:
        results = list(engine.evaluate(samples, scheme))
    if checkpoint is not None:
        checkpoint()
    by_id = {item.row_id: item for item in results}
    if (
        len(by_id) != len(samples)
        or len(results) != len(samples)
        or set(by_id) != {s.row_id for s in samples}
    ):
        raise ValueError(
            "Ядро должно вернуть ровно один результат для каждой входной пробы"
        )
    results = apply_quality([by_id[sample.row_id] for sample in samples], input_values)
    if any(item.status not in STATUSES for item in results):
        raise ValueError("Ядро вернуло неизвестный статус интерпретации")
    region_ids = {r["id"] for r in scheme.get("regions", [])}
    line_ids = {r["id"] for r in scheme.get("lines", [])}
    if any(
        (set(r.regions + r.boundary_regions + r.uncertainty_regions) - region_ids)
        or (set(r.line_hits + r.possible_types) - line_ids)
        or (r.thresholds.keys() - line_ids)
        for r in results
    ):
        raise ValueError("Ядро вернуло области или линии, отсутствующие в схеме")
    records = []
    for index, result in enumerate(results):
        if checkpoint is not None and index % 256 == 0:
            checkpoint()
        records.append(_result_record(result))
    return {
        "created_at": timestamp(),
        "stale": False,
        "scheme": scheme,
        "scheme_hash": scheme_hash(scheme),
        "review_scheme_hash": review_scheme_hash(scheme),
        "binding": {
            "x_id": x_id,
            "y_id": y_id,
            "x_unit": x_unit,
            "y_unit": y_unit,
            "text_numbers": text_numbers,
        },
        "axis_column_names": {"x": columns[x_id].name, "y": columns[y_id].name},
        "project_id": project.id,
        "inputs": input_values,
        "quality_policy": {"version": 2, "source": "Пользовательская таблица диапазонов; SI трактуется как S1",
                           "columns": [dict(id=column.id, name=column.name, property=role)
                                       for column, role, _, _ in quality.columns],
                           "ranges": deepcopy(RANGES), "typical_action": "warn", "absolute_axis_action": "exclude"},
        "interval_policy": {
            "method": "adaptive_lipschitz_v1",
            "hio_requires_unique_type": True,
            "parameters": {"interval_max_evaluations": scheme.get('interval_max_evaluations', 32),
                           "distance_tolerance": scheme.get('distance_tolerance', 0),
                           "hi_resolution": scheme.get('hi_resolution', 0),
                           "max_distance": scheme.get('max_distance')},
        } if scheme.get('strategy') in ('nearest_curve', 'initial_hi') else None,
        "shapely_version": shapely.__version__,
        "engine": type(engine).__qualname__,
        "definition_hash": definition_hash(project, x_id, y_id),
        "results": records,
    }


def scheme_hash(scheme):
    return hashlib.sha256(
        json.dumps(scheme, sort_keys=True, ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def interval_marker_ids(result):
    """Describe a marker using the same result that drives the table."""
    return list(dict.fromkeys([
        *result.get("uncertainty_regions", ()),
        *(result.get("possible_types") or result.get("line_hits", ())),
        *(key for key, item in result.get("thresholds", {}).items()
          if len(item["possible_relations"]) > 1),
    ]))


def interpretation_binding_key(binding):
    return scheme_hash({key: binding.get(key) for key in
                        ("scheme", "x_id", "y_id", "x_unit", "y_unit", "text_numbers")})


def saved_interval_markers(project, binding, uncertain_ids):
    report = project.view.get("interpretation_result")
    covered, markers = set(), {}
    if not report or report.get("stale") or not uncertain_ids:
        return covered, markers
    candidates = (report, *report.get("comparisons", {}).values())
    for snapshot in candidates:
        if (snapshot.get("stale") or snapshot["scheme"] != binding["scheme"]
                or any(snapshot["binding"].get(key) != binding.get(key)
                       for key in ("x_id", "y_id", "x_unit", "y_unit", "text_numbers"))):
            continue
        for result, sample in zip(snapshot["results"], snapshot["inputs"], strict=True):
            row_id = result["row_id"]
            if row_id in uncertain_ids:
                covered.add(row_id)
                if result["status"] == "ambiguous" and (
                    sample.get("x_bounds") is not None or sample.get("y_bounds") is not None
                ):
                    markers[row_id] = interval_marker_ids(result)
        break
    return covered, markers


def replay_report(report, *, engine=None):
    if report.get("scheme_hash") != scheme_hash(report["scheme"]):
        raise ValueError("Контрольная сумма схемы не соответствует снимку")
    if report.get("engine") != "GeometricEngine" and engine is None:
        raise ValueError("Для воспроизведения требуется исходное пользовательское ядро")
    samples = []
    for item in report["inputs"]:
        values = [
            classify(item[key], numeric=True)[1] if status == "numeric" else None
            for key, status in zip(("x", "y"), item["statuses"], strict=True)
        ]
        samples.append(InterpretationInput(item["row_id"], *values,
            x_bounds=item.get("x_bounds"), y_bounds=item.get("y_bounds")))
    from .interpretation_quality import apply_quality
    replay_scheme = deepcopy(report['scheme'])
    policy = report.get('interval_policy')
    if policy is not None:
        if not isinstance(policy, dict) or policy.get('method') != 'adaptive_lipschitz_v1':
            raise ValueError('Неизвестный метод интервального воспроизведения')
        parameters = policy.get('parameters')
        if not isinstance(parameters, dict) or parameters.keys() != {
            'interval_max_evaluations', 'distance_tolerance', 'hi_resolution', 'max_distance',
        }:
            raise ValueError('Некорректные параметры интервального воспроизведения')
        if any(key in replay_scheme and replay_scheme[key] != value for key, value in parameters.items()):
            raise ValueError('Параметры интервального воспроизведения противоречат схеме')
        replay_scheme.update(parameters)
    legacy_point_hio = not (policy or {}).get('hio_requires_unique_type', False)
    results = list((engine or GeometricEngine(legacy_point_hio=legacy_point_hio)).evaluate(samples, replay_scheme))
    return [_result_record(result) for result in apply_quality(results, report['inputs'])]


def definition_hash(project, x_id, y_id):
    payload = {
        column.id: {
            "kind": column.kind,
            "unit": column.unit,
            "calculation": column.metadata.get("calculation"),
            "method": column.metadata.get("method"),
        }
        for column in project.columns
        if column.id in (x_id, y_id)
    }
    return scheme_hash(payload)


def mark_saved_result_stale(project, code, **details):
    """Keep the first invalidation cause with the saved snapshot."""
    report = project.view.get("interpretation_result")
    if report and not report.get("stale"):
        report.update(stale=True, stale_reason={"code": code, **details})


def _snapshot_change(project, report, rows, *, partial):
    binding = report["binding"]
    x_id, y_id = binding["x_id"], binding["y_id"]
    scheme = report["scheme"]
    current = project.view.get("interpretation_schemes", {}).get(scheme["id"], scheme)
    if scheme_hash(current) != report["scheme_hash"]:
        return dict(code="scheme_changed", scheme_id=scheme["id"], scheme_name=scheme["name"],
                    old_revision=scheme["revision"], new_revision=current["revision"])
    if report.get("definition_hash") != definition_hash(project, x_id, y_id):
        return dict(code="axis_definition_changed", column_ids=[x_id, y_id])
    columns = {column.id: column for column in project.columns}
    for key, column_id in (('x', x_id), ('y', y_id)):
        try:
            validate_axis_binding(columns[column_id], binding[f'{key}_unit'], scheme[key])
        except (KeyError, ValueError):
            return dict(code='axis_definition_changed', column_ids=[column_id])
    inputs = report["inputs"]
    if partial:
        inputs = [item for item in inputs if item["row_id"] in rows]
    exclusions = project.view.get("qc_exclusions", {}).get("interpretation", {})
    for item in inputs:
        row_id = item["row_id"]
        if row_id not in rows:
            return dict(code="sample_removed", row_id=row_id)
        for key, cid, status in zip(("x", "y"), (x_id, y_id), item["statuses"], strict=True):
            details = dict(row_id=row_id, column_id=cid, axis=key)
            if rows[row_id].values[cid] != item[key]:
                return dict(code="input_changed", **details)
            measurement = project.science.current(row_id, cid)
            excluded = (measurement is not None
                        and exclusions.get(measurement.id, {}).get("revision") == measurement.revision)
            if excluded != (status == "qc_excluded"):
                return dict(code="qc_changed", **details)
            metadata = item.get("uncertainty", {}).get(key)
            has_uncertainty = measurement is not None and measurement.uncertainty is not None
            if bool(metadata) != has_uncertainty or (metadata and (
                measurement.uncertainty != metadata["uncertainty"]
                or measurement.uncertainty_type != metadata["type"]
                or measurement.confidence_level != metadata["confidence_level"]
                or unit_key(measurement.unit or binding[f"{key}_unit"]) != unit_key(metadata["unit"])
            )):
                return dict(code="uncertainty_changed", **details)
            if "measurements" in item:
                reference = item["measurements"].get(key)
            elif metadata:
                reference = {"id": metadata["measurement_id"], "revision": metadata["revision"]}
            else:
                continue
            current_reference = ({"id": measurement.id, "revision": measurement.revision}
                                 if measurement is not None else None)
            if reference != current_reference:
                return dict(code="measurement_changed", previous=reference, current=current_reference, **details)
    from .interpretation_quality import InterpretationQuality

    quality = InterpretationQuality(project, x_id, y_id, binding["x_unit"], binding["y_unit"],
                                    scheme, binding.get("text_numbers", False))
    current_quality = quality.check_many(rows[item["row_id"]] for item in inputs)
    for item in inputs:
        row_id = item["row_id"]
        issues, blocked = current_quality[row_id]
        previous = [issue for issue in item.get("quality", ()) if issue["code"] != "uncertainty_unchecked"]
        issues_changed = issues != previous and sorted(map(scheme_hash, issues)) != sorted(map(scheme_hash, previous))
        if issues_changed or bool(blocked) != bool(item.get("quality_blocked")):
            return dict(code="qc_changed", row_id=row_id)
    return None


def check_saved_binding(project, settings, *, scope_rows=None):
    report = project.view.get("interpretation_result")
    if report and not report.get("stale"):
        expected = dict(report["binding"], scheme_id=report["scheme"]["id"])
        expected["comparison"] = bool(report.get("comparison_panels") or report.get("comparisons"))
        if "scope" in report:
            expected["scope"] = report["scope"]
        changed = [key for key, value in expected.items() if key in settings and settings[key] != value]
        if scope_rows is not None and "scope" in expected and "scope" not in changed:
            inputs = report["inputs"]
            row_ids = set(scope_rows)
            if len(inputs) != len(row_ids) or any(item["row_id"] not in row_ids for item in inputs):
                changed.append("scope")
        if changed:
            mark_saved_result_stale(project, "binding_changed", fields=changed)


def check_saved_result(project, *, changed_rows=None):
    report = project.view.get("interpretation_result")
    if not report or report.get("stale"):
        return
    try:
        check_saved_binding(project, project.view.get("interpretation_current", {}))
        if report.get("stale"):
            return
        rows = project.row_map() if changed_rows is None else {row.id: row for row in changed_rows}
        qc_reason = None
        for name, snapshot in [(None, report), *report.get("comparisons", {}).items()]:
            reason = _snapshot_change(project, snapshot, rows, partial=changed_rows is not None)
            if reason:
                if reason["code"] == "qc_changed" and "axis" not in reason:
                    # An ancillary QC warning can be a consequence of an axis edit
                    # in another diagram. Prefer the direct explanation.
                    if qc_reason is None:
                        qc_reason = dict(reason, diagram=name)
                    continue
                mark_saved_result_stale(project, **reason, diagram=name)
                return
        if qc_reason is not None:
            mark_saved_result_stale(project, **qc_reason)
        elif changed_rows is None and report.get("scope") == "all" and len(report["inputs"]) != len(rows):
            mark_saved_result_stale(project, "binding_changed", fields=["scope"])
    except (ValueError, KeyError, TypeError):
        mark_saved_result_stale(project, "snapshot_unverifiable")
