"""Explicit IDW tool with leave-one-location-out diagnostics, never automatic."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd
import scipy
from scipy.spatial import cKDTree
import shapely
from shapely.geometry import MultiPoint, mapping

from autopyro.model.core import timestamp
from .mapping import crs, layer_frame, project_geometries
from .map_tools import filtered_features
from autopyro.visualization.plotting import category_key
from autopyro.model.tabular import classify_series


@dataclass(frozen=True)
class SpatialOptions:
    value_id: str
    context: str
    duplicate_policy: str = "reject"
    power: float = 2
    neighbors: int = 8
    min_neighbors: int = 3
    max_distance: float = 10000
    grid_size: int = 60
    region_mode: str = 'none'
    region_layer: str = ''
    region_field: str = ''
    region_key: str = ''
    validation_buffer_km: float = 0


def predict(xy, values, queries, options):
    tree = cKDTree(xy)
    distances, indices = tree.query(
        queries,
        k=min(options.neighbors, len(xy)),
        distance_upper_bound=options.max_distance,
    )
    if distances.ndim == 1:
        distances, indices = distances[:, None], indices[:, None]
    valid = np.isfinite(distances)
    safe = np.minimum(indices, len(values) - 1)
    local = values[safe]
    # Relative distances avoid overflow of d**(-power) close to a location.
    nearest = np.min(np.where(valid, distances, np.inf), axis=1)
    relative = np.divide(
        np.maximum(nearest, 1e-300)[:, None],
        np.maximum(distances, 1e-300),
        where=valid,
        out=np.zeros_like(distances),
    )
    weights = np.where(valid, relative**options.power, 0.0)
    totals = weights.sum(axis=1)
    means = np.divide(
        (weights * local).sum(axis=1),
        totals,
        out=np.full(len(queries), np.nan),
        where=totals > 0,
    )
    support = valid.sum(axis=1)
    means[support < options.min_neighbors] = np.nan
    exact = valid[:, 0] & (distances[:, 0] == 0)
    means[exact] = values[safe[exact, 0]]
    return means, support, nearest


def _region_domain(report, options):
    if options.region_mode == 'none':
        return None
    if options.region_mode not in ('clip', 'within'):
        raise ValueError('Неизвестный режим областей IDW')
    layer = next((layer for layer in report['layers'] if layer['id'] == options.region_layer), None)
    features = filtered_features(layer) if layer else []
    if not features or not any(options.region_field in item['properties'] for item in features):
        raise ValueError('Выберите полигональный слой и поле области IDW')
    geometries = project_geometries(layer_frame(dict(layer, features=features)).geometry,
                                    layer['crs'], report['spec']['target_crs'])
    if not geometries.geom_type.isin(['Polygon', 'MultiPolygon']).all():
        raise ValueError('Выберите полигональный слой и поле области IDW')
    keys = np.array([category_key(value) if value is not None and str(value).strip() else ''
                     for item in features for value in [item['properties'].get(options.region_field)]], dtype=object)
    selected = keys == options.region_key if options.region_key else keys != ''
    if not selected.any():
        raise ValueError('Выбранная область IDW отсутствует в слое')
    # Remove internal boundaries and overlaps between features of the same category.
    category_keys = list(dict.fromkeys(keys))
    areas = [shapely.union_all(geometries.array[keys == key]) for key in category_keys]
    active = set(keys[selected])
    return (shapely.union_all([area for key, area in zip(category_keys, areas) if key in active]),
            shapely.STRtree(areas), np.array([key if key in active else '' for key in category_keys]))


def _classify_domain(domain, xy):
    outline, tree, keys = domain
    points = shapely.points(xy)
    regions = np.full(len(points), '', dtype=object)
    positions, polygons = tree.query(points, predicate='intersects')
    unique = np.bincount(positions, minlength=len(points))[positions] == 1
    positions, polygons = positions[unique], polygons[unique]
    interior = shapely.contains(tree.geometries[polygons], points[positions])
    regions[positions[interior]] = keys[polygons[interior]]
    return shapely.covers(outline, points), regions


def _domain_predict(xy, values, queries, options, mode, allowed, train_regions, query_regions, predictor=predict):
    means = np.full(len(queries), np.nan)
    support = np.zeros(len(queries), dtype=int)
    distance = np.full(len(queries), np.inf)
    if mode == 'within':
        groups = [(train_regions == key, allowed & (query_regions == key))
                  for key in set(query_regions[allowed]) if key]
    else:
        groups = [(np.ones(len(xy), dtype=bool), allowed)]
    for train, query in groups:
        if train.any() and query.any():
            means[query], support[query], distance[query] = predictor(xy[train], values[train], queries[query], options)
    if mode == 'within' and predictor is predict:
        means[support < options.min_neighbors] = np.nan
    return means, support, distance


def _validate_options(options):
    if not np.isfinite(options.validation_buffer_km) or not 0 <= options.validation_buffer_km <= 10000:
        raise ValueError('Буфер проверки должен быть от 0 до 10 000 км.')
    if options.duplicate_policy not in ("reject", "mean", "median"):
        raise ValueError("Повторные координаты: reject, mean или median")
    if not (
        np.isfinite(options.power)
        and 0.1 <= options.power <= 10
        and np.isfinite(options.max_distance)
        and options.max_distance > 0
    ):
        raise ValueError(
            "Степень IDW 0.1–10; радиус должен быть положительным и конечным"
        )
    if not (
        3 <= options.min_neighbors <= options.neighbors <= 100
        and 10 <= options.grid_size <= 200
    ):
        raise ValueError("Соседей 3–100, размер сетки 10–200")


def is_metric_projection(target):
    return target.is_projected and all(abs(axis.unit_conversion_factor - 1) <= 1e-12 for axis in target.axis_info)


def _location_samples(report, options):
    if options.value_id not in {c["id"] for c in report["columns"]}:
        raise ValueError("Выберите показатель интерполяции")
    column = next(c for c in report["columns"] if c["id"] == options.value_id)
    if column.get("interpolated"):
        raise ValueError(
            "Пространственная интерполяция не использует интерполированную колонку"
        )
    frame = pd.DataFrame(
        [(point["id"], point["x"], point["y"], point["values"].get(options.value_id))
         for point in report["points"]], columns=["id", "x", "y", "raw"], dtype=object,
    )
    classified = classify_series(frame["raw"], numeric=column["kind"] == "number" or report["spec"]["text_numbers"])
    excluded = classified.loc[classified["number"].isna(), "status"].value_counts(sort=False).to_dict()
    frame["value"] = classified["number"]
    frame = frame.loc[frame["value"].notna()].copy()
    return frame, excluded


def _locations(report, options: SpatialOptions):
    if not is_metric_projection(crs(report["spec"]["target_crs"])):
        raise ValueError("Для интерполяции выберите проецированную CRS с единицами в метрах")
    _validate_options(options)
    frame, excluded = _location_samples(report, options)
    groups = frame.groupby(["x", "y"], sort=False)
    duplicates = int((groups.size() > 1).sum())
    if duplicates and options.duplicate_policy == "reject":
        raise ValueError("Есть повторные координаты: отфильтруйте выборку или явно выберите mean/median по месту")
    if not 4 <= groups.ngroups <= 1000:
        raise ValueError("Требуется от 4 до 1000 уникальных мест с числовыми значениями")
    frame["value"] = frame["value"].astype(float)
    locations = frame.groupby(["x", "y"], sort=False).agg(
        value=("value", "median" if options.duplicate_policy == "median" else "mean"),
        row_ids=("id", list), n_samples=("id", "size"),
    ).reset_index().to_dict("records")
    return locations, excluded, duplicates


def _error_metrics(errors):
    valid = errors[np.isfinite(errors)]
    count = len(valid)
    return dict(predicted=count, coverage=count / len(errors) if len(errors) else None,
                rmse=float(np.sqrt(np.mean(valid ** 2))) if count else None,
                mae=float(np.mean(np.abs(valid))) if count else None,
                bias=float(np.mean(valid)) if count else None)


def _block_groups(xy, block_size_km, offset_m):
    if not np.isfinite(block_size_km) or not .001 <= block_size_km <= 10000:
        raise ValueError('Размер блока должен быть от 0,001 до 10 000 км')
    size = block_size_km * 1000
    origin = np.asarray(offset_m, dtype=float)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError('Сдвиг блоков должен содержать две конечные координаты')
    cells = np.floor((xy - origin) / size).astype(np.int64)
    groups = {}
    for i, cell in enumerate(cells):
        groups.setdefault(tuple(cell), []).append(i)
    for number, ((col, row), members) in enumerate(sorted(groups.items()), 1):
        left, bottom = float(col * size + origin[0]), float(row * size + origin[1])
        block = dict(id=number, bounds=[left, bottom, left + size, bottom + size], locations=len(members))
        yield block, members


def _block_training_mask(xy, block, members, buffer_km):
    train = np.ones(len(xy), dtype=bool)
    train[members] = False
    if buffer_km:
        bounds = np.asarray(block['bounds']).reshape(2, 2)
        delta = np.maximum(np.maximum(bounds[0] - xy, xy - bounds[1]), 0)
        train &= np.hypot(delta[:, 0], delta[:, 1]) > buffer_km * 1000
    return train


def _training_hull_mask(xy, train, members, min_neighbors):
    if train.sum() >= min_neighbors:
        hull = MultiPoint(xy[train]).convex_hull
        if hull.geom_type == 'Polygon':
            return shapely.covers(hull, shapely.points(xy[members]))
    return np.zeros(len(members), dtype=bool)


def _regional_hull_mask(xy, queries, allowed, mode, train_regions, query_regions):
    if mode != 'within':
        return allowed
    covered = np.zeros(len(queries), dtype=bool)
    for key in set(query_regions[allowed]) - {''}:
        members = allowed & (query_regions == key)
        hull = MultiPoint(xy[train_regions == key]).convex_hull
        if hull.geom_type == 'Polygon':
            covered[members] = shapely.covers(hull, shapely.points(queries[members]))
    return covered


def _validation_geometry(xy, train, members, mode, inside_region, regions):
    hull = _training_hull_mask(xy, train, members, 3)
    reasons = np.full(len(members), '', dtype=object)
    reasons[~hull] = 'insufficient_training' if train.sum() < 3 else 'outside_hull'
    if mode == 'within':
        local = _regional_hull_mask(xy[train], xy[members], hull, mode, regions[train], regions[members])
        reasons[hull & ~local] = 'outside_regional_hull'
        reasons[regions[members] == ''] = 'ambiguous_region'
    if mode != 'none':
        reasons[~inside_region[members]] = 'outside_region'
    return reasons


def _forecast_reasons(geometry, estimates, support, required_neighbors):
    reasons = geometry.copy()
    missing = (reasons == '') & ~np.isfinite(estimates)
    reasons[missing] = 'method_no_prediction'
    if required_neighbors is not None:
        reasons[missing & (support < required_neighbors)] = 'insufficient_neighbors'
    return reasons


def preview_sample_support(report, options, block_size_km):
    """Inspect eligible locations and IDW support without fitting a surface."""
    frame, excluded = _location_samples(report, options)
    counts = frame.groupby(['x', 'y'], sort=False).size()
    xy = counts.index.to_frame(index=False).to_numpy(dtype=float)
    metric = is_metric_projection(crs(report['spec']['target_crs']))
    diagnostics = report.get('diagnostics', {})
    result = dict(
        target_crs=report['spec']['target_crs'], metric=metric, mapped=len(report['points']),
        eligible=len(frame), locations=len(counts), duplicate_locations=int((counts > 1).sum()),
        duplicate_samples=int((counts - 1).sum()), excluded=excluded,
        input_rows=diagnostics.get('input_rows'), map_filtered=sum(diagnostics.get('filtered', {}).values()),
        map_excluded=dict(diagnostics.get('coordinates', {})),
        block_size_km=block_size_km, buffer_km=options.validation_buffer_km,
        radius_m=options.max_distance, min_neighbors=options.min_neighbors,
        blocks=None, inside_region=None, candidates=None, supported=None, with_hull=None, issue=None,
    )
    if not metric:
        return result
    blocks = list(_block_groups(xy, block_size_km, (0, 0)))
    result['blocks'] = len(blocks)
    if not 4 <= len(xy) <= 1000:
        result['issue'] = 'Требуется от 4 до 1000 уникальных мест с числовыми значениями'
        return result
    try:
        _validate_options(options)
        domain = _region_domain(report, options)
    except ValueError as exc:
        result['issue'] = str(exc)
        return result
    if MultiPoint(xy).convex_hull.geom_type != 'Polygon':
        result['issue'] = 'Места расположены на одной линии; двумерная интерполяция невозможна'
    inside, regions = _classify_domain(domain, xy) if domain else (np.ones(len(xy), dtype=bool), None)
    result['inside_region'] = int(inside.sum()) if domain else None
    eligible = inside & (regions != '') if options.region_mode == 'within' else inside
    result['unassigned_region'] = int((inside & ~eligible).sum())
    result.update(candidates=int(eligible.sum()), supported=0, with_hull=0)
    zeros = np.zeros(len(xy))
    for block, members in blocks:
        train = _block_training_mask(xy, block, members, options.validation_buffer_km)
        _, support, _ = _domain_predict(xy[train], zeros[train], xy[members], options, options.region_mode,
                                        eligible[members], regions[train] if domain else None,
                                        regions[members] if domain else None)
        enough = support >= options.min_neighbors
        result['supported'] += int(enough.sum())
        hull = _training_hull_mask(xy, train, members, options.min_neighbors)
        if domain:
            hull = _regional_hull_mask(xy[train], xy[members], hull & eligible[members],
                                       options.region_mode, regions[train], regions[members])
        result['with_hull'] += int((enough & hull).sum())
    return result


def spatial_blocks(report, options: SpatialOptions, block_size_km, *, evaluate=False, compare_modes=True,
                   predictor=predict, offset_m=(0, 0), progress=None, required_neighbors=None):
    """Assign each eligible location to a square and optionally hold out one square at a time."""
    if evaluate:
        _validate_context(options)
    locations, excluded, _ = _locations(report, options)
    xy = np.array([[item['x'], item['y']] for item in locations])
    values = np.array([item['value'] for item in locations])
    domain = _region_domain(report, options)
    inside_region, region_keys = _classify_domain(domain, xy) if domain else (None, None)
    modes = ('none', 'clip', 'within') if domain and compare_modes else (options.region_mode,)
    if predictor is predict:
        required_neighbors = options.min_neighbors
    rows, held_out = [], []
    residuals = {mode: np.full(len(xy), np.nan) for mode in modes}
    groups = list(_block_groups(xy, block_size_km, offset_m))
    for index, (block, members) in enumerate(groups):
        if progress is not None:
            progress(index / len(groups))
        number = block['id']
        train = _block_training_mask(xy, block, members, options.validation_buffer_km)
        block['buffered_locations'] = int(len(xy) - len(members) - train.sum())
        if evaluate:
            predictions, comparisons, reasons = {}, {}, {}
            for mode in modes:
                geometry = _validation_geometry(xy, train, members, mode, inside_region, region_keys)
                if train.sum() < options.min_neighbors:
                    geometry[geometry == ''] = 'insufficient_training'
                estimates, support, _ = _domain_predict(xy[train], values[train], xy[members], options, mode, geometry == '',
                    region_keys[train] if domain else None, region_keys[members] if domain else None, predictor)
                errors = estimates - values[members]
                residuals[mode][members] = errors
                comparisons[mode] = _error_metrics(errors)
                predictions[mode] = estimates
                reasons[mode] = _forecast_reasons(geometry, estimates, support, required_neighbors)
            if domain:
                block['comparisons'] = comparisons
            block.update(comparisons[options.region_mode])
            estimates = predictions[options.region_mode]
            errors = estimates - values[members]
            for i, estimate, error, reason in zip(members, estimates, errors, reasons[options.region_mode]):
                held_out.append(dict(locations[i], block=number,
                                     predicted=float(estimate) if np.isfinite(estimate) else None,
                                     residual=float(error) if np.isfinite(error) else None, reason=reason or None))
        else:
            held_out.extend(dict(locations[i], block=number) for i in members)
        rows.append(block)
    totals = {mode: _error_metrics(errors) for mode, errors in residuals.items()} if evaluate else {}
    return dict(block_size_km=block_size_km, buffer_km=options.validation_buffer_km,
                evaluated=evaluate, blocks=rows, locations=held_out,
                totals=totals.get(options.region_mode), comparisons=totals if domain else {},
                region_mode=options.region_mode, excluded=dict(excluded))


def surface_options(options, method):
    parameters = asdict(options)
    if method != 'IDW':
        parameters.pop('power')
    if method in ('TIN', 'RBF', 'Clough-Tocher', 'nearest'):
        parameters.pop('neighbors')
        parameters.pop('min_neighbors')
    if method in ('TIN', 'RBF', 'Clough-Tocher'):
        parameters.pop('max_distance')
    return parameters


def partition_spread(report, options, block_size_km, *, method='IDW', shifted=False, kriging_model=None, progress=None):
    """Map the prediction range across fits that omit one occupied block each."""
    from .tin import predict_tin

    model_metadata = None
    if method == 'kriging':
        from .kriging import KrigingOptions, predict_kriging

        model = kriging_model if kriging_model is not None else KrigingOptions()
        model.validate()
        predictor, min_neighbors = partial(predict_kriging, model=model), options.min_neighbors
        model_metadata = asdict(model)
    else:
        predictor, min_neighbors = {'IDW': (predict, options.min_neighbors), 'TIN': (predict_tin, 3)}[method]
    _validate_context(options)
    locations, _, _ = _locations(report, options)
    xy = np.array([[point['x'], point['y']] for point in locations])
    values = np.array([point['value'] for point in locations])
    hull = MultiPoint(xy).convex_hull
    if hull.geom_type != 'Polygon':
        raise ValueError('Места расположены на одной линии; двумерная интерполяция невозможна')
    shift_km = block_size_km / 2 if shifted else 0
    groups = list(_block_groups(xy, block_size_km, (shift_km * 1000,) * 2))
    if len(groups) < 2:
        raise ValueError('Для карты разброса нужны минимум два занятых блока.')
    x, y = (np.linspace(low, high, options.grid_size) for low, high in zip(xy.min(axis=0), xy.max(axis=0)))
    queries = np.column_stack([axis.ravel() for axis in np.meshgrid(x, y)])
    points = shapely.points(queries)
    inside = shapely.covers(hull, points)
    domain = _region_domain(report, options)
    regions = query_regions = None
    if domain:
        _, regions = _classify_domain(domain, xy)
        covered, query_regions = _classify_domain(domain, queries)
        inside = _regional_hull_mask(xy, queries, inside & covered, options.region_mode, regions, query_regions)
    count = np.zeros(len(queries), dtype=int)
    low, high = np.full(len(queries), np.inf), np.full(len(queries), -np.inf)
    for index, (block, members) in enumerate(groups):
        if progress is not None:
            progress(index, len(groups))
        train = _block_training_mask(xy, block, members, options.validation_buffer_km)
        if train.sum() < min_neighbors:
            continue
        training_hull = MultiPoint(xy[train]).convex_hull
        if training_hull.geom_type != 'Polygon':
            continue
        train_regions = regions[train] if domain else None
        allowed = _regional_hull_mask(xy[train], queries, inside & shapely.covers(training_hull, points),
                                      options.region_mode, train_regions, query_regions)
        if method == 'kriging' and progress is not None:
            predictor = partial(predict_kriging, model=model, checkpoint=lambda: progress(index, len(groups)))
        predictions, _, _ = _domain_predict(xy[train], values[train], queries, options, options.region_mode,
                                             allowed, train_regions, query_regions, predictor)
        finite = np.isfinite(predictions)
        count[finite] += 1
        low[finite] = np.minimum(low[finite], predictions[finite])
        high[finite] = np.maximum(high[finite], predictions[finite])
    shape = (len(y), len(x))
    return dict(x=x, y=y, spread=np.where(count >= 2, high - low, np.nan).reshape(shape),
                count=np.where(inside, count, np.nan).reshape(shape), locations=locations,
                inside_domain=inside.reshape(shape), outline=mapping(domain[0]) if domain else None,
                blocks=len(groups), method=method, model=model_metadata,
                options=dict(surface_options(options, method), block_size_km=block_size_km, block_shift_km=shift_km))


def idw_partition_spread(report, options, block_size_km, *, progress=None):
    return partition_spread(report, options, block_size_km, progress=progress)


def _idw_variants(options, variants):
    if not 2 <= len(variants) <= 24:
        raise ValueError('Для сравнения задайте от 2 до 24 вариантов IDW')
    for variant in variants:
        if set(variant) != {'power', 'neighbors', 'max_distance'}:
            raise ValueError('Варианты IDW задают степень, радиус и максимум соседей')
        yield replace(options, **variant)


def compare_idw_settings(report, options, block_size_km, variants, *, progress=None):
    """Compare fixed settings on identical holdouts and on their common forecast subset."""
    results, residuals = [], []
    for index, (variant, configured) in enumerate(zip(variants, list(_idw_variants(options, variants)))):
        def advance(fraction):
            progress((index + fraction) / len(variants))

        validation = spatial_blocks(report, configured, block_size_km, evaluate=True, compare_modes=False,
                                    progress=advance if progress is not None else None)
        results.append(dict(parameters=dict(variant), metrics=validation['totals'],
                            predictions=[point['predicted'] for point in validation['locations']]))
        residuals.append(np.asarray([point['residual'] for point in validation['locations']], dtype=float))
    common = np.isfinite(residuals).all(axis=0)
    locations = [{key: point[key] for key in ('x', 'y', 'block', 'value', 'row_ids')}
                 for point in validation['locations']]
    for result, errors in zip(results, residuals):
        result['common_metrics'] = _error_metrics(errors[common])
    return dict(format='autopyro.idw-settings-comparison', version=1, created_at=timestamp(),
                options=asdict(options), block_size_km=block_size_km, blocks=validation['blocks'],
                locations=locations, common_indices=np.flatnonzero(common).tolist(), variants=results,
                map_snapshot=deepcopy(report))


def nested_idw_validation(report, options, block_size_km, variants, *, progress=None):
    """Tune by inner-block MAE, then evaluate on each untouched outer block."""
    candidates = list(_idw_variants(options, variants))
    for candidate in candidates:
        _validate_options(candidate)
    result = _nested_validation(report, options, block_size_km, {
        'tuned': [(candidate, predict, candidate.min_neighbors) for candidate in candidates],
        'baseline': [(options, predict, options.min_neighbors)],
    }, progress=progress)
    for block in result['blocks']:
        methods = block.pop('methods')
        block.update({key: methods['tuned'][key] for key in
                      ('selected', 'inner_common', 'inner_scores', 'inner_locations', 'inner_candidate_counts')})
        block.update({name: stats['metrics'] for name, stats in methods.items()})
    for point in result['locations']:
        point.update(point.pop('predictions'))
    for name, scores in result.pop('methods').items():
        result[name] = scores['metrics']
        result[f'{name}_common'] = scores['common']
    result.pop('offset_m')
    return dict(result, format='autopyro.idw-nested-validation', version=1,
                variants=[dict(variant) for variant in variants])


class UnsupportedPartition(ValueError):
    """The data or work limit prevents nested validation on this partition."""


def _nested_validation(report, options, block_size_km, methods, *, offset_m=(0, 0), progress=None):
    _validate_context(options)
    locations, excluded, _ = _locations(report, options)
    xy = np.array([[point['x'], point['y']] for point in locations])
    values = np.array([point['value'] for point in locations])
    groups = list(_block_groups(xy, block_size_km, offset_m))
    if len(groups) < 3:
        raise UnsupportedPartition('Для вложенной проверки нужны минимум три занятых блока')
    tuning = {name: candidates for name, candidates in methods.items() if len(candidates) > 1}
    if len(groups) * (len(groups) - 1) * sum(map(len, tuning.values())) > 100_000:
        raise UnsupportedPartition('Слишком много внутренних проверок. Увеличьте размер блока или сократите число вариантов.')
    domain = _region_domain(report, options)
    inside, regions = _classify_domain(domain, xy) if domain else (None, None)

    def forecast(train, members, allowed, spec):
        configured, predictor, _ = spec
        if train.sum() < configured.min_neighbors:
            return np.full(len(members), np.nan), np.zeros(len(members), dtype=int)
        estimates, support, _ = _domain_predict(xy[train], values[train], xy[members], configured,
                               options.region_mode, allowed,
                               regions[train] if domain else None,
                               regions[members] if domain else None, predictor)
        return estimates, support

    estimates = {name: np.full(len(xy), np.nan) for name in methods}
    blocks, held_out = [], []
    candidate_count = sum(map(len, tuning.values()))
    for outer_index, (block, outer) in enumerate(groups):
        if progress is not None:
            progress(outer_index / len(groups))
        outer_train = _block_training_mask(xy, block, outer, options.validation_buffer_km)
        block['buffered_locations'] = int(len(xy) - len(outer) - outer_train.sum())
        inner_errors = {name: np.full((len(candidates), len(xy)), np.nan)
                        for name, candidates in tuning.items()}
        # Reserve the last step of each outer fold for its final predictions.
        outer_steps = len(groups) + 1
        for inner_index, (inner_block, inner) in enumerate(groups):
            if progress is not None:
                progress((outer_index + inner_index / outer_steps) / len(groups))
            # Keep the outer buffer out of both inner training and inner scoring.
            inner = [i for i in inner if outer_train[i]]
            if not inner:
                continue
            train = outer_train & _block_training_mask(xy, inner_block, inner, options.validation_buffer_km)
            allowed = _validation_geometry(xy, train, inner, options.region_mode, inside, regions) == ''
            if allowed.any():
                candidate_index = 0
                for name, candidates in tuning.items():
                    for index, candidate in enumerate(candidates):
                        if progress is not None:
                            progress((outer_index + (inner_index + candidate_index / candidate_count)
                                      / outer_steps) / len(groups))
                        inner_errors[name][index, inner] = forecast(train, inner, allowed, candidate)[0] - values[inner]
                        candidate_index += 1
        geometry = _validation_geometry(xy, outer_train, outer, options.region_mode, inside, regions)
        allowed = geometry == ''
        fold, reasons = {}, {}
        inner_locations = int(outer_train.sum())
        for method_index, (name, candidates) in enumerate(methods.items()):
            if progress is not None:
                progress((outer_index + (len(groups) + method_index / len(methods))
                          / outer_steps) / len(groups))
            count, scores, selected = None, [], 0
            candidate_counts = []
            if name in tuning:
                errors = inner_errors[name]
                finite = np.isfinite(errors)
                common = finite.all(axis=0)
                count = int(common.sum())
                candidate_counts = finite.sum(axis=1).tolist()
                scores = [_error_metrics(row[common]) for row in errors]
                selected = min(range(len(candidates)), key=lambda index: scores[index]['mae']) if count else None
            if selected is not None:
                configured, _, required_neighbors = candidates[selected]
                prediction, support = forecast(outer_train, outer, allowed, candidates[selected])
                estimates[name][outer] = prediction
                if outer_train.sum() < configured.min_neighbors:
                    reasons[name] = np.where(allowed, 'insufficient_training', geometry)
                else:
                    reasons[name] = _forecast_reasons(geometry, prediction, support, required_neighbors)
            else:
                reasons[name] = np.where(allowed, 'no_inner_selection', geometry)
            fold[name] = dict(selected=selected, inner_common=count, inner_scores=scores,
                              inner_locations=inner_locations if name in tuning else None,
                              inner_candidate_counts=candidate_counts,
                              metrics=_error_metrics(estimates[name][outer] - values[outer]))
        blocks.append(dict(block, methods=fold))
        for position, index in enumerate(outer):
            held_out.append(dict(locations[index], block=block['id'],
                                 predictions={name: float(row[index]) if np.isfinite(row[index]) else None
                                              for name, row in estimates.items()},
                                 reasons={name: status[position] or None for name, status in reasons.items()}))
    errors = {name: row - values for name, row in estimates.items()}
    common = np.logical_and.reduce([np.isfinite(row) for row in errors.values()])
    return dict(created_at=timestamp(), options=asdict(options), offset_m=list(offset_m),
                block_size_km=block_size_km, selection='inner_common_mae_first_on_tie',
                target_crs=report['spec']['target_crs'], locations=held_out, blocks=blocks, excluded=dict(excluded),
                methods={name: dict(metrics=_error_metrics(row), common=_error_metrics(row[common]))
                         for name, row in errors.items()})


def _validate_context(options):
    if not options.context.strip():
        raise ValueError(
            "Укажите обоснование сопоставимости выборки: метод, стратиграфия и глубина"
        )


def interpolate(report, options: SpatialOptions, *, progress=None):
    if progress is not None:
        progress(0)
    _validate_context(options)
    locations, excluded, duplicates = _locations(report, options)
    xy = np.array([[p["x"], p["y"]] for p in locations])
    values = np.array([p["value"] for p in locations])
    hull = MultiPoint(xy).convex_hull
    if hull.geom_type != "Polygon" or hull.area == 0:
        raise ValueError(
            "Места расположены на одной линии; двумерная интерполяция невозможна"
        )
    domain = _region_domain(report, options)
    in_region, regions = _classify_domain(domain, xy) if domain else (None, None)
    eligible = in_region if domain else np.ones(len(xy), dtype=bool)
    if options.region_mode == 'within':
        eligible = eligible & (regions != '')
    loo = []
    residuals = np.full(len(xy), np.nan)
    for i, location in enumerate(locations):
        if progress is not None:
            progress(.7 * i / len(locations))
        mask = np.arange(len(xy)) != i
        if domain:
            members = slice(i, i + 1)
            hull_mask = _training_hull_mask(xy, mask, [i], options.min_neighbors)
            hull_mask = _regional_hull_mask(xy[mask], xy[members], hull_mask,
                                            options.region_mode, regions[mask], regions[members])
            predicted, support, _ = _domain_predict(xy[mask], values[mask], xy[i:i+1], options,
                options.region_mode, hull_mask & eligible[members], regions[mask], regions[members])
            outside = not bool(hull_mask[0])
        else:
            predicted, support, _ = predict(xy[mask], values[mask], xy[i:i+1], options)
            outside = not MultiPoint(xy[mask]).convex_hull.covers(shapely.Point(xy[i]))
        residual = predicted[0] - values[i] if np.isfinite(predicted[0]) else None
        if residual is not None:
            residuals[i] = residual
        loo.append(
            {
                **location,
                "predicted": float(predicted[0]) if residual is not None else None,
                "residual": float(residual) if residual is not None else None,
                "neighbors": int(support[0]),
                "outside_training_hull": outside,
                "eligible_for_validation": bool(eligible[i]),
            }
        )
    valid = np.isfinite(residuals)
    if progress is not None:
        progress(.7)
    gx = np.linspace(xy[:, 0].min(), xy[:, 0].max(), options.grid_size)
    gy = np.linspace(xy[:, 1].min(), xy[:, 1].max(), options.grid_size)
    xx, yy = np.meshgrid(gx, gy)
    queries = np.column_stack((xx.ravel(), yy.ravel()))
    inside = shapely.covers(hull, shapely.points(queries))
    query_regions = None
    if domain:
        covered, query_regions = _classify_domain(domain, queries)
        inside = _regional_hull_mask(xy, queries, inside & covered, options.region_mode, regions, query_regions)
    allowed = inside if domain else np.ones(len(queries), dtype=bool)
    predicted, support, distance = _domain_predict(xy, values, queries, options, options.region_mode,
                                                   allowed, regions, query_regions)
    predicted[~inside] = np.nan
    if progress is not None:
        progress(.85)
    mse = np.full(len(queries), np.nan)
    if valid.sum() >= 4:
        mse, _, _ = _domain_predict(xy[valid], residuals[valid] ** 2, queries, options, options.region_mode,
                                    inside, regions[valid] if domain else None, query_regions)
    error = np.sqrt(np.maximum(0, mse))
    error[~inside | ~np.isfinite(predicted)] = np.nan
    if progress is not None:
        progress(1)

    def grid(array):
        return [
            [float(value) if np.isfinite(value) else None for value in row]
            for row in array.reshape(xx.shape)
        ]

    return {
        "format": "autopyro.spatial-idw",
        "version": 1,
        "created_at": timestamp(),
        "options": asdict(options),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "map_snapshot": deepcopy(report),
        "locations": locations,
        "excluded": dict(excluded),
        "duplicate_locations": duplicates,
        "region_outline": mapping(domain[0]) if domain else None,
        "loo": loo,
        "diagnostics": {
            "locations": len(xy),
            "loo_valid": int(valid.sum()),
            "loo_eligible": int(eligible.sum()),
            "rmse": float(np.sqrt(np.mean(residuals[valid] ** 2))) if valid.any() else None,
            "mae": float(np.mean(np.abs(residuals[valid]))) if valid.any() else None,
            "bias": float(np.mean(residuals[valid])) if valid.any() else None,
            "loo_outside_training_hull": sum(
                item["eligible_for_validation"] and item["outside_training_hull"] for item in loo
            ),
            "grid_valid": int(np.isfinite(predicted).sum()),
            "grid_total": len(queries),
        },
        "grid": {
            "x": gx.tolist(),
            "y": gy.tolist(),
            "prediction": grid(predicted),
            "error_proxy": grid(error),
            "neighbors": grid(support),
            "nearest_distance_m": grid(distance),
            "inside_domain": inside.reshape(xx.shape).tolist(),
        },
        "uncertainty_method": "sqrt(IDW of squared leave-one-location-out residuals); empirical local error proxy, NOT calibrated prediction SD or confidence interval",
        "validation_note": "Whole locations held out, including co-located samples. Fixed user parameters. "
                           + ("Regional LOO is restricted to the selected domain and training convex hull."
                              if domain else "LOO is not spatial block validation; boundary holdouts may extrapolate."),
        "domain": "convex hull AND radius/min-neighbor support"
                  + (" AND selected polygon mask" if domain else "")
                  + (" AND same-region training hull and neighbors" if options.region_mode == 'within' else ""),
    }
