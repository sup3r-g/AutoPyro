"""Local ordinary kriging with manually specified variogram parameters."""
from dataclasses import asdict, dataclass, replace
from functools import partial
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist
import shapely
from shapely.geometry import MultiPoint, mapping

from .spatial_interpolation import (UnsupportedPartition, _classify_domain, _domain_predict, _error_metrics, _idw_variants,
                                    _locations, _nested_validation, _region_domain, _regional_hull_mask, _validate_context,
                                    _validate_options, predict, spatial_blocks, surface_options)
from .rbf import predict_nearest, predict_rbf
from .tin import predict_clough_tocher, predict_tin
from .surfaces import finite_surface_range


@dataclass(frozen=True)
class KrigingOptions:
    model: str = 'spherical'
    range_km: float = 100
    sill: float = 1
    nugget: float = 0
    azimuth_deg: float = 0
    range_ratio: float = 1

    def validate(self):
        if (self.model not in ('spherical', 'exponential')
                or not np.isfinite([self.range_km, self.sill, self.nugget]).all()
                or self.range_km <= 0 or not 0 <= self.nugget < self.sill):
            raise ValueError('Кригинг требует положительной дальности и наггета от нуля до порога, не включая порог.')
        if not np.isfinite(self.azimuth_deg) or not 0 < self.range_ratio <= 1:
            raise ValueError('Укажите конечный азимут и отношение дальностей больше нуля и не больше единицы.')


def correlation(distance, model):
    ratio = distance / (1000 * model.range_km)
    if model.model == 'exponential':
        return np.exp(-3 * ratio)
    ratio = np.minimum(ratio, 1)
    return 1 - 1.5 * ratio + .5 * ratio ** 3


def covariance_coordinates(xy, model):
    if model.range_ratio == 1:
        return xy
    angle = np.deg2rad(model.azimuth_deg % 180)
    east, north = np.sin(angle), np.cos(angle)
    return xy @ np.array([[east, north / model.range_ratio], [north, -east / model.range_ratio]])


def predict_kriging(xy, values, queries, options, *, model, checkpoint=None):
    distances, indices = cKDTree(xy).query(queries, k=np.arange(1, min(options.neighbors, len(xy)) + 1),
                                          distance_upper_bound=options.max_distance)
    valid = np.isfinite(distances)
    support = valid.sum(axis=1)
    estimates = np.full(len(queries), np.nan)
    partial_sill = 1 - model.nugget / model.sill
    coordinates = covariance_coordinates(xy - xy[0], model)
    query_coordinates = covariance_coordinates(queries - xy[0], model)
    for position, i in enumerate(np.flatnonzero(support >= options.min_neighbors)):
        if checkpoint is not None and position % 128 == 0:
            checkpoint()
        selected = indices[i, valid[i]]
        local = coordinates[selected]
        count = len(selected)
        system = np.ones((count + 1, count + 1))
        system[:-1, :-1] = partial_sill * correlation(cdist(local, local), model)
        np.fill_diagonal(system, 1)
        system[-1, -1] = 0
        covariance_distance = np.linalg.norm(local - query_coordinates[i], axis=1)
        rhs = np.r_[partial_sill * correlation(covariance_distance, model), 1]
        try:
            weights = np.linalg.solve(system, rhs)[:-1]
        except np.linalg.LinAlgError:
            continue
        estimates[i] = weights @ values[selected]
    return estimates, support, distances[:, 0]


def _variogram_pairs(xy, values, regions):
    left, right = np.triu_indices(len(xy), 1)
    distances = pdist(xy)
    differences = pdist(values[:, None], metric='sqeuclidean') * .5
    vectors = xy[right] - xy[left]
    if regions is not None:
        same = (regions[left] != '') & (regions[left] == regions[right])
        distances, differences, vectors = distances[same], differences[same], vectors[same]
    return distances, differences, vectors


def _variogram_bins(distances, differences, limits):
    if not len(distances):
        return []
    bins = len(limits) - 1
    groups = np.clip(np.searchsorted(limits, distances, side='right') - 1, 0, bins - 1)
    return [dict(distance_km=float(distances[groups == i].mean() / 1000),
                 semivariance=float(differences[groups == i].mean()), pairs=int((groups == i).sum()))
            for i in range(bins) if (groups == i).any()]


def empirical_variogram(xy, values, regions=None, bins=12):
    distances, differences, _ = _variogram_pairs(xy, values, regions)
    limits = np.linspace(0, distances.max(initial=0), bins + 1)
    return _variogram_bins(distances, differences, limits)


def directional_variograms(xy, values, regions=None, bins=12):
    distances, differences, vectors = _variogram_pairs(xy, values, regions)
    limits = np.linspace(0, distances.max(initial=0), bins + 1)
    azimuths = np.degrees(np.arctan2(vectors[:, 0], vectors[:, 1])) % 180
    sectors = np.floor((azimuths + 22.5) / 45).astype(int) % 4
    result = {}
    for sector in range(4):
        mask = sectors == sector
        result[sector * 45] = _variogram_bins(distances[mask], differences[mask], limits)
    return result


def _method_specs(options, model, include_tin, rbf_smoothing, include_cubic):
    methods = [('IDW', options, predict), ('kriging', options, partial(predict_kriging, model=model))]
    if model.range_ratio < 1:
        isotropic = replace(model, range_ratio=1)
        methods.append(('isotropic-kriging', options, partial(predict_kriging, model=isotropic)))
    triangulation = replace(options, min_neighbors=3)
    if include_tin:
        methods.append(('TIN', triangulation, predict_tin))
    if rbf_smoothing is not None:
        if not np.isfinite(rbf_smoothing) or rbf_smoothing < 0:
            raise ValueError('Сглаживание RBF должно быть конечным и неотрицательным.')
        methods.extend([('RBF', triangulation, partial(predict_rbf, smoothing=rbf_smoothing)),
                        ('nearest', triangulation, predict_nearest)])
    if include_cubic:
        methods.append(('Clough-Tocher', triangulation, predict_clough_tocher))
    return methods


def compare_nested_methods(report, options, size, idw_variants, model, ranges_km, *, shifted=False, progress=None):
    """Compare tuned IDW and kriging with TIN and the current fixed IDW."""
    idw = list(_idw_variants(options, idw_variants))
    for candidate in idw:
        _validate_options(candidate)
    ranges = tuple(dict.fromkeys(ranges_km))
    if not 2 <= len(ranges) <= 12 or any(not .001 <= value <= 10000 for value in ranges):
        raise ValueError('Укажите от 2 до 12 разных дальностей кригинга от 0,001 до 10 000 км.')
    kriging = [replace(model, range_km=value) for value in ranges]
    for candidate in kriging:
        candidate.validate()
    methods = {
        'IDW': [(candidate, predict, candidate.min_neighbors) for candidate in idw],
        'kriging': [(options, partial(predict_kriging, model=candidate), options.min_neighbors) for candidate in kriging],
        'TIN': [(replace(options, min_neighbors=3), predict_tin, None)],
        'baseline': [(options, predict, options.min_neighbors)],
    }
    shift = size * 500 if shifted else 0
    result = _nested_validation(report, options, size, methods, offset_m=(shift, shift), progress=progress)
    return dict(result, format='autopyro.nested-spatial-methods', version=1,
                candidates={'IDW': [dict(item) for item in idw_variants], 'kriging': [asdict(item) for item in kriging]})


def paired_validation(rows):
    residuals = {row['method']: {(p['x'], p['y']): p['residual'] for p in row['locations']
                                if p['residual'] is not None} for row in rows if row.get('method')}
    pairs = {}
    for first, second in combinations(residuals, 2):
        common = sorted(residuals[first].keys() & residuals[second].keys())
        pairs[first, second] = {
            method: _error_metrics(np.asarray([residuals[method][xy] for xy in common]))
            for method in (first, second)
        }
    return pairs


def _validate_methods(report, methods, size, offset=(0, 0), progress=None):
    checks = []
    for index, (name, settings, predictor) in enumerate(methods):
        def advance(fraction):
            progress((index + fraction) / len(methods))

        required = (settings.min_neighbors if name in ('kriging', 'isotropic-kriging')
                    else 1 if name == 'nearest' else None)
        checks.append(spatial_blocks(report, settings, size, evaluate=True, compare_modes=False,
                                     predictor=predictor, offset_m=offset,
                                     progress=advance if progress is not None else None, required_neighbors=required))
    errors = [np.asarray([point['residual'] for point in check['locations']], dtype=float) for check in checks]
    common = np.isfinite(errors).all(axis=0)
    return [
        dict(method=name, **check['totals'], common=_error_metrics(error[common]),
             blocks=check['blocks'], locations=check['locations'], buffer_km=check['buffer_km'])
        for (name, _, _), check, error in zip(methods, checks, errors)
    ]


def compare_block_partitions(report, options, model, sizes, *, shifted, rbf_smoothing, progress=None):
    model.validate()
    methods = _method_specs(options, model, True, rbf_smoothing, True)
    results = []
    for size, shift_km, checkpoint in _partition_steps(sizes, shifted, progress):
        offset = shift_km * 1000
        checks = _validate_methods(report, methods, size, (offset, offset), checkpoint)
        _compare_partition_to_idw(checks, 'IDW')
        results.extend(dict(check, block_size_km=size, shift_km=offset / 1000) for check in checks)
    return results


def _compare_partition_to_idw(rows, baseline_method):
    baseline = next(row for row in rows if row['method'] == baseline_method)
    for row in rows:
        for subset, key in [('common', 'delta_rmse_idw'), ('shared', 'shared_delta_rmse_idw')]:
            if subset not in row:
                continue
            current, reference = row[subset]['rmse'], baseline[subset]['rmse']
            row[key] = current - reference if current is not None and reference is not None else None


def _partitions(sizes, shifted):
    sizes = tuple(dict.fromkeys(sizes))
    if not 1 <= len(sizes) <= 8:
        raise ValueError('Укажите от 1 до 8 размеров блоков.')
    if any(not np.isfinite(size) or not .001 <= size <= 10000 for size in sizes):
        raise ValueError('Размер блока должен быть от 0,001 до 10 000 км')
    return [(size, shift) for size in sizes for shift in ((False, True) if shifted else (False,))]


def _partition_steps(sizes, shifted, progress):
    partitions = _partitions(sizes, shifted)
    for index, (size, shift) in enumerate(partitions):
        shift_km = size / 2 if shift else 0
        checkpoint = partial(progress, index, len(partitions), size, shift_km) if progress is not None else None
        if checkpoint is not None:
            checkpoint(0)
        yield size, shift_km, checkpoint


def compare_nested_partitions(report, options, model, sizes, idw_variants, ranges_km, *, shifted, progress=None):
    """Repeat tuning per partition and compare both local and across-partition intersections."""
    partitions = []
    for size, shift_km, checkpoint in _partition_steps(sizes, shifted, progress):
        try:
            partitions.append(compare_nested_methods(report, options, size, idw_variants, model, ranges_km,
                                                     shifted=bool(shift_km), progress=checkpoint))
        except UnsupportedPartition as exc:
            partitions.append(dict(method=None, issue=str(exc), block_size_km=size, shift_km=shift_km))
    completed = [partition for partition in partitions if 'issue' not in partition]
    # The order of locations changes when the blocks change. Match by coordinates.
    shared = set.intersection(*[
        {(point['x'], point['y']) for point in partition['locations']
         if all(value is not None for value in point['predictions'].values())}
        for partition in completed
    ]) if completed else set()
    rows = []
    for partition in partitions:
        if 'issue' in partition:
            rows.append(partition)
            continue
        methods = partition['methods']
        rmse = [stats['common']['rmse'] for stats in methods.values() if stats['common']['rmse'] is not None]
        checks = []
        for name, stats in methods.items():
            locations, residuals = [], []
            for point in partition['locations']:
                estimate = point['predictions'][name]
                residual = estimate - point['value'] if estimate is not None else None
                locations.append(dict(point, predicted=estimate, residual=residual, reason=point['reasons'][name]))
                if (point['x'], point['y']) in shared:
                    residuals.append(residual)
            error = stats['common']['rmse']
            checks.append(dict(method=name, **stats['metrics'], common=stats['common'],
                             rank=1 + sum(other < error for other in rmse) if error is not None else None,
                             shared=_error_metrics(np.asarray(residuals, dtype=float)),
                             candidates=partition['candidates'].get(name),
                             block_size_km=partition['block_size_km'], shift_km=partition['offset_m'][0] / 1000,
                             buffer_km=options.validation_buffer_km, blocks=partition['blocks'], locations=locations))
        _compare_partition_to_idw(checks, 'baseline')
        rows.extend(checks)
    return rows


def run_kriging(report, options, model, block_size_km, *, preview=False, include_tin=False,
                rbf_smoothing=None, include_cubic=False, directions=False, progress=None):
    if progress is not None:
        progress(0)
    model.validate()
    if not preview:
        _validate_context(options)
    locations, _, _ = _locations(report, options)
    xy = np.array([[point['x'], point['y']] for point in locations])
    values = np.array([point['value'] for point in locations])
    domain = _region_domain(report, options)
    _, regions = _classify_domain(domain, xy) if domain else (None, None)
    unit = next(column['unit'] for column in report['columns'] if column['id'] == options.value_id)
    variogram_regions = regions if options.region_mode == 'within' else None
    result = dict(model=asdict(model), unit=unit,
                  variogram=empirical_variogram(xy, values, variogram_regions))
    if directions:
        result['directions'] = directional_variograms(xy, values, variogram_regions)
    if progress is not None:
        progress(.1)
    if preview:
        return result
    hull = MultiPoint(xy).convex_hull
    if hull.geom_type != 'Polygon':
        raise ValueError('Места расположены на одной линии; двумерная интерполяция невозможна')
    methods = _method_specs(options, model, include_tin, rbf_smoothing, include_cubic)
    x = np.linspace(xy[:, 0].min(), xy[:, 0].max(), options.grid_size)
    y = np.linspace(xy[:, 1].min(), xy[:, 1].max(), options.grid_size)
    xx, yy = np.meshgrid(x, y)
    queries = np.column_stack((xx.ravel(), yy.ravel()))
    inside = shapely.covers(hull, shapely.points(queries))
    query_regions = None
    if domain:
        covered, query_regions = _classify_domain(domain, queries)
        inside = _regional_hull_mask(xy, queries, inside & covered, options.region_mode, regions, query_regions)
    def surface_for(name, settings, method):
        prediction, neighbors, distance = _domain_predict(xy, values, queries, settings, settings.region_mode,
                                                           inside, regions, query_regions, method)
        return dict(x=x, y=y, prediction=prediction.reshape(xx.shape), locations=locations,
                    options=surface_options(settings, name), neighbors=neighbors.reshape(xx.shape),
                    validation=dict(method='spatial_blocks', block_size_km=block_size_km,
                                    buffer_km=settings.validation_buffer_km, shift_km=0),
                    nearest_distance_m=distance.reshape(xx.shape), inside_domain=inside.reshape(xx.shape),
                    outline=mapping(domain[0]) if domain else None)

    surfaces = {}
    for index, (name, settings, predictor) in enumerate(methods):
        fraction = .1 + .5 * index / len(methods)
        if progress is not None:
            progress(fraction)
            if name in ('kriging', 'isotropic-kriging'):
                predictor = partial(predictor, checkpoint=partial(progress, fraction))
        surfaces[name] = surface_for(name, settings, predictor)
    result.update(surfaces['kriging'])
    result['surface_value_range'] = finite_surface_range(*(grid['prediction'] for grid in surfaces.values()))
    result['idw_surface'] = surfaces['IDW']
    if 'TIN' in surfaces:
        result['tin_surface'] = surfaces['TIN']
    result['extra_surfaces'] = {name: grid for name, grid in surfaces.items() if name not in ('IDW', 'kriging', 'TIN')}
    if 'RBF' in surfaces:
        surfaces['RBF']['model'] = dict(kernel='thin_plate_spline', smoothing=rbf_smoothing, coordinate_unit='km')
    if 'isotropic-kriging' in surfaces:
        surfaces['isotropic-kriging']['model'] = asdict(replace(model, range_ratio=1))
    validation_progress = (lambda fraction: progress(.6 + .4 * fraction)) if progress is not None else None
    validations = _validate_methods(report, methods, block_size_km, progress=validation_progress)
    result['pairwise_validation'] = paired_validation(validations)
    if include_tin:
        point_sets = {row['method']: row['locations'] for row in validations if row['method'] in ('IDW', 'TIN')}
        tin_points = {(point['x'], point['y']): point for point in point_sets['TIN']}
        result['idw_tin_holdouts'] = []
        for point in point_sets['IDW']:
            tin = tin_points[point['x'], point['y']]
            result['idw_tin_holdouts'].append(dict(
                x=point['x'], y=point['y'], row_ids=point['row_ids'], block=point['block'],
                idw_residual=point['residual'], tin_residual=tin['residual'],
                idw_reason=point['reason'], tin_reason=tin['reason'],
            ))
    result.update(validation=[dict({k: v for k, v in row.items() if k not in ('locations', 'blocks')},
                                   negative_cells=int(np.count_nonzero(surfaces[row['method']]['prediction'] < 0)))
                              for row in validations],
                  blocks=validations[0]['blocks'], block_size_km=block_size_km)
    if progress is not None:
        progress(1)
    return result
