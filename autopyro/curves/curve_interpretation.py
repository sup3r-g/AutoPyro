"""Vectorized nearest-curve interpretation in the declared analysis metric."""
import numpy as np
import shapely
from shapely.geometry import LineString

from autopyro.schemes.interpretation import InterpretationResult, inverse, transform


def prepare_curves(scheme, lines):
    strategy = scheme.get("strategy", "regions")
    if strategy not in ("regions", "nearest_curve", "initial_hi"):
        raise ValueError("Неизвестная стратегия интерпретации")
    for key in ("distance_tolerance", "hi_resolution"):
        value = scheme.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} должен быть конечным неотрицательным числом")
    budget = scheme.get('interval_max_evaluations', 32)
    if isinstance(budget, bool) or not isinstance(budget, int) or not 5 <= budget <= 1024:
        raise ValueError('Бюджет интервального поиска должен быть от 5 до 1024')
    maximum = scheme.get("max_distance")
    if maximum is not None and (isinstance(maximum, bool) or not isinstance(maximum, (int, float)) or not np.isfinite(maximum) or maximum < 0):
        raise ValueError("max_distance должен быть конечным неотрицательным числом")
    if strategy == "regions":
        return lines
    lines = [(entry, geometry) for entry, geometry in lines if not entry.get('reference_only')]
    if not lines or scheme.get("distance_metric") != "analysis_euclidean":
        raise ValueError("Для кривых нужны линии и явная метрика расстояния")
    if strategy != "initial_hi":
        return lines
    if len(lines) < 2 or any(g.geom_type != "LineString" for _, g in lines):
        raise ValueError("HIo требует не менее двух LineString")
    count = scheme.get("intermediate_curves", 99)
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 999:
        raise ValueError("Число промежуточных кривых должно быть от 0 до 999")
    alignment = scheme.get("curve_alignment", "vertices")
    if alignment not in ("vertices", "normalized_arclength"):
        raise ValueError("Неизвестное согласование опорных кривых")
    if alignment == "normalized_arclength":
        # Direction and vertex density of PDF paths must not affect correspondence.
        coordinates = []
        for _, geometry in lines:
            if geometry.coords[0][0] > geometry.coords[-1][0]:
                geometry = shapely.reverse(geometry)
            points = shapely.line_interpolate_point(geometry, np.linspace(0, 1, 257), normalized=True)
            coordinates.append(shapely.get_coordinates(points))
    else:
        coordinates = [np.asarray(g.coords) for _, g in lines]
    if len({a.shape for a in coordinates}) != 1:
        raise ValueError("Опорные кривые должны иметь соответствующие вершины")
    curves = []
    for index, (left, right) in enumerate(zip(coordinates, coordinates[1:])):
        for weight in np.linspace(0, 1, count + 2)[:-1]:
            coords = left * (1 - weight) + right * weight
            entry = dict(lines[index][0], parent_ids=[lines[index][0]['id']])
            if weight > 0:
                entry['parent_ids'].append(lines[index + 1][0]['id'])
            curves.append((entry, LineString(coords)))
    curves.append((dict(lines[-1][0], parent_ids=[lines[-1][0]['id']]), lines[-1][1]))
    return curves


def evaluate_curves(samples, scheme, curves, *, checkpoint=None, legacy_point_hio=False):
    from shapely.geometry import Polygon, shape
    from .curve_intervals import IntervalCurveSearch, analysis_bounds, footprint

    results = [InterpretationResult(s.row_id, "outside") for s in samples]
    valid, coordinates, intervals = [], [], {}
    for index, sample in enumerate(samples):
        if checkpoint is not None and index % 256 == 0:
            checkpoint()
        result = results[index]
        if sample.x is None or sample.y is None:
            result.status = "insufficient_data"
            continue
        try:
            point = (transform(sample.x, scheme["x"]), transform(sample.y, scheme["y"]))
            if sample.x_bounds is not None or sample.y_bounds is not None:
                intervals[index] = analysis_bounds(sample, scheme)
            coordinates.append(point)
            valid.append(index)
        except (ValueError, OverflowError, TypeError) as exc:
            result.status = "not_applicable"
            result.reason = 'Недопустимые координаты или интервал: ' + str(exc)

    geometries = np.array([g for _, g in curves], dtype=object)
    source_lines = [entry for entry in scheme['lines'] if not entry.get('reference_only')]
    source_geometries = np.array([shape(entry['geometry']) for entry in source_lines], dtype=object)
    envelope = None
    if scheme.get("no_extrapolation") and scheme["strategy"] == "initial_hi":
        paths = [np.asarray(g.coords) for g in source_geometries]
        paths = [a if a[0, 0] <= a[-1, 0] else a[::-1] for a in paths]
        envelope = shapely.union_all([shapely.make_valid(Polygon(np.vstack([a, b[::-1]])))
                                      for a, b in zip(paths, paths[1:])]).buffer(1e-6)
        shapely.prepare(envelope)
    initial_hi = None
    if scheme['strategy'] == 'initial_hi':
        initial_hi = np.array([inverse(min(g.coords, key=lambda p: (p[0], -p[1]))[1], scheme['y'])
                               for g in geometries])
    search = IntervalCurveSearch(geometries, scheme, initial_hi)
    # Type classification uses the original reference curves, independently of
    # interpolated HIo trajectories and their left-parent provenance labels.
    type_search = IntervalCurveSearch(source_geometries, dict(scheme, max_distance=float('inf')))
    maximum = scheme.get('max_distance')
    maximum = float('inf') if maximum is None else maximum
    source_labels = {entry['id']: entry['label'] for entry in source_lines}
    tolerance = max(1e-12, scheme.get('distance_tolerance', 0))
    batch_size = max(1, 250_000 // len(curves))
    for start in range(0, len(valid), batch_size):
        if checkpoint is not None:
            checkpoint()
        points = shapely.points(coordinates[start:start + batch_size])
        distances = shapely.distance(points[:, None], geometries[None, :])
        for offset, row in enumerate(distances):
            if checkpoint is not None and offset % 32 == 0:
                checkpoint()
            index = valid[start + offset]
            result = results[index]
            bounds = intervals.get(index)
            area = footprint(bounds) if bounds is not None else points[offset]
            if envelope is not None and not envelope.covers(area):
                result.status = "not_applicable"
                result.reason = "Точка или часть интервала вне семейства траекторий; экстраполяция HIo запрещена"
                continue
            minimum = float(row.min())
            if minimum > maximum:
                result.status = "not_applicable"
                result.reason = "Точка вне допустимого расстояния от кривых; экстраполяция запрещена"
                continue
            winners = np.flatnonzero(row <= minimum + tolerance)
            if bounds is not None:
                winners, details, error = search.search(bounds)
                if error:
                    result.status, result.reason = 'not_applicable', error
                    continue
                result.interval_details = dict(details)
            result.line_hits = list(dict.fromkeys(curves[i][0]['id'] for i in winners))
            result.status = 'classified' if len(winners) == 1 else 'ambiguous'
            result.distances = {'nearest': minimum}
            names = list(dict.fromkeys(curves[i][0]['label'] for i in winners))
            result.reason = ('Возможные ближайшие кривые: ' if bounds is not None else 'Ближайшая кривая: ') + ', '.join(names)
            result.possible_types = list(result.line_hits)
            if initial_hi is not None:
                result.line_hits = list(dict.fromkeys(parent for i in winners
                    for parent in curves[i][0].get('parent_ids', [curves[i][0]['id']])))
                result.reason = 'Родительские кривые семейства HIo: ' + ', '.join(source_labels[i] for i in result.line_hits)
                values = initial_hi[winners]
                result.value_bounds = [float(values.min()), float(values.max())]
                if bounds is not None:
                    types, type_details, _ = type_search.search(bounds)
                    result.possible_types = [source_lines[i]['id'] for i in types]
                    result.interval_details['types_conservative'] = type_details['conservative']
                    result.interval_details['type_evaluations'] = type_details['evaluations']
                    result.interval_details['type_budget_exhausted'] = type_details['budget_exhausted']
                else:
                    type_distances = shapely.distance(points[offset], source_geometries)
                    types = np.flatnonzero(type_distances <= type_distances.min() + tolerance)
                    result.possible_types = [source_lines[i]['id'] for i in types]
                narrow = float(np.ptp(values)) <= max(1e-9, scheme.get('hi_resolution', 0))
                unique_type = len(result.possible_types) == 1 or (legacy_point_hio and bounds is None)
                if narrow and unique_type:
                    result.value = float(initial_hi[int(np.argmin(row))])
                    result.status = 'classified'
                    result.reason += f"; модельная оценка HIo = {result.value:g} {scheme['y']['unit']}"
                else:
                    result.status = 'ambiguous'
                    result.reason += f"; HIo: {values.min():g}–{values.max():g}; неопределённость HIo или типа вещества"
            if bounds is not None:
                result.reason += '; интерпретация всего интервала в нормированном пространстве'
                if result.interval_details.get('conservative') or result.interval_details.get('types_conservative'):
                    result.reason += '; консервативная оболочка возможных результатов'
                if result.interval_details.get('budget_exhausted') or result.interval_details.get('type_budget_exhausted'):
                    result.reason += '; бюджет уточнения исчерпан, исключить дополнительные варианты не удалось'
                if result.status == 'ambiguous':
                    result.reason += '; интервал допускает несколько типов или исходных HI'
    return results
