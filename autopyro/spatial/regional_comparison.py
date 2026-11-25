"""Frozen results for descriptive comparisons between two regional scenarios."""
from copy import deepcopy
import hashlib
import json
import math
from statistics import median

import numpy as np

from .map_tools import filtered_features
from autopyro.visualization.plotting import category_key


SCENARIO_COLORS = {'A': '#0072b2', 'B': '#d55e00', 'AB': '#8e44ad'}
UNSTABLE_REGION_COLOR = '#9c1467'


def _median(values):
    value = median(values)
    if math.isfinite(value):
        return value
    return 2 * median(number / 2 for number in values)


def region_fingerprint(layer, field):
    features = sorted(layer['features'], key=lambda feature: str(feature['id']))
    definition = [layer['crs'], [(feature['id'], feature['geometry'], feature['properties'].get(field))
                                for feature in features]]
    return hashlib.sha256(json.dumps(definition, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def freeze_regional_result(result, report, columns=()):
    options = result['options']
    layer = next(layer for layer in report['layers'] if layer['id'] == options['layer_id'])
    field = options['region_field']
    locations = {rid: (point['x'], point['y']) for point in report['points'] for rid in point['row_ids']}
    groups = []
    for entry in result['distributions']:
        row = deepcopy(result['summary'][entry['summary_index']])
        wells = {}
        well_values = {}
        for sample in entry['samples']:
            well = sample['well']
            if sample['value'] is None or well is None or not str(well).strip():
                continue
            key = category_key(well)
            sites = wells.setdefault(key, set())
            well_values.setdefault(key, [well, []])[1].append(sample['value'])
            if sample['row_id'] in locations:
                sites.add(locations[sample['row_id']])
        row['wells'] = {key: [list(xy) for xy in sorted(sites)] for key, sites in wells.items()}
        row['well_medians'] = {key: dict(well=well, median=_median(values))
                               for key, (well, values) in well_values.items()}
        row['values'] = [item['value'] for item in entry['values']]
        groups.append(row)
    conditions = deepcopy({key: value for key, value in result.items()
                           if key not in ('summary', 'distributions', 'spatial_issues', 'format', 'version')})
    features = filtered_features(layer)
    conditions['region_layer'] = dict(id=layer['id'], crs=layer['crs'], fingerprint=region_fingerprint(layer, field),
        feature_ids=[str(feature['id']) for feature in features],
        features=[dict(id=str(feature['id']), geometry=deepcopy(feature['geometry']),
                       region_key=category_key(feature['properties'].get(field))) for feature in features])
    column_ids = {value for key, value in options.items() if key.endswith('_id')}
    conditions['columns'] = {column.id: column.name for column in columns if column.id in column_ids}
    return dict(format='autopyro.regional_comparison_snapshot', version=3,
                conditions=conditions, groups=groups)


def method_key(group):
    return 'pooled' if group['pooled'] else category_key(group['method'])


def compare_regions(a, b, metric, method_a, method_b, *, paired=False):
    left, right = a['conditions'], b['conditions']
    opts_a, opts_b = left['options'], right['options']
    if (opts_a['layer_id'], opts_a['region_field'], left['region_layer']['fingerprint']) != (
            opts_b['layer_id'], opts_b['region_field'], right['region_layer']['fingerprint']):
        raise ValueError('Для сравнения нужны одинаковый слой, геометрия и поле области.')
    shared_binding = opts_a.get('well_id') and opts_a.get('well_id') == opts_b.get('well_id')
    if paired:
        if min(a['version'], b['version']) < 2:
            raise ValueError('Для парного сравнения пересчитайте и заново сохраните старые снимки.')
        if not shared_binding:
            raise ValueError('Для парного сравнения выберите одну колонку скважины в обоих сценариях.')
    elif bool(opts_a.get('by_well')) != bool(opts_b.get('by_well')):
        raise ValueError('Для сравнения выберите один режим расчёта в обоих сценариях.')
    groups = [{group['region_key']: group for group in snapshot['groups']
               if group['metric'] == metric and method_key(group) == method}
              for snapshot, method in ((a, method_a), (b, method_b))]
    units = {group['unit'] or '' for side in groups for group in side.values()}
    if len(units) > 1:
        raise ValueError('Единицы показателя в сценариях различаются.')
    regions = {group['region_key']: group['region_label'] for snapshot in (b, a)
               for group in snapshot['groups'] if group['metric'] == metric}
    unit = next(iter(units), '')
    rows = []
    for key, label in regions.items():
        group_a, group_b = (side.get(key) for side in groups)
        median_a = group_a['median'] if group_a else None
        median_b = group_b['median'] if group_b else None
        delta = median_b - median_a if median_a is not None and median_b is not None else None
        wells_a = group_a['wells'] if group_a else {}
        wells_b = group_b['wells'] if group_b else {}
        common = wells_a.keys() & wells_b.keys()
        row = dict(region_key=key, region=label, unit=unit, a=group_a, b=group_b,
                   delta=delta if delta is None or math.isfinite(delta) else None,
                   common_wells=len(common) if shared_binding else None)
        if paired:
            row['paired'] = paired_wells(group_a, group_b)
        rows.append(row)
    return rows


def paired_wells(group_a, group_b):
    left = group_a['well_medians'] if group_a else {}
    right = group_b['well_medians'] if group_b else {}
    pairs = []
    for key in sorted(left.keys() & right.keys()):
        a, b = left[key]['median'], right[key]['median']
        delta = b - a
        if not math.isfinite(delta):
            raise ValueError('Численное переполнение при расчёте статистики областей')
        pairs.append(dict(well=left[key]['well'], a=a, b=b, delta=delta))
    stats = dict(n=len(pairs), median=None, q1=None, q3=None, minimum=None, maximum=None)
    if pairs:
        differences = [pair['delta'] for pair in pairs]
        q1, middle, q3 = np.quantile(differences, [.25, .5, .75])
        if not all(map(math.isfinite, (q1, middle, q3))):
            raise ValueError('Численное переполнение при расчёте статистики областей')
        stats.update(median=float(middle), q1=float(q1), q3=float(q3),
                     minimum=min(differences), maximum=max(differences))
    stats.update(leave_one_out(pairs, stats['median']))
    return dict(wells=pairs, **stats)


def leave_one_out(pairs, full_median):
    for pair in pairs:
        pair.update(median_without=None, sign_changed=None)
    if len(pairs) < 2:
        return dict(without_min=None, without_max=None, sign_changed=None)
    ordered = sorted(pairs, key=lambda pair: pair['delta'])
    values = [pair['delta'] for pair in ordered]
    # Removing a rank shifts each central position by at most one.
    low, high = (len(values) - 2) // 2, (len(values) - 1) // 2
    sign = np.sign(full_median)
    medians = []
    for index, pair in enumerate(ordered):
        without = _median([values[low + (index <= low)], values[high + (index <= high)]])
        pair.update(median_without=without, sign_changed=bool(np.sign(without) != sign))
        medians.append(without)
    return dict(without_min=min(medians), without_max=max(medians),
                sign_changed=any(pair['sign_changed'] for pair in pairs))


def comparison_map_points(row, snapshots, *, paired=False, well_key=None):
    sides = [(row[key] or {}).get('wells', {}) for key in ('a', 'b')]
    well_ids = [snapshot['conditions']['options'].get('well_id') for snapshot in snapshots]
    common = sides[0].keys() & sides[1].keys() if well_ids[0] and well_ids[0] == well_ids[1] else set()
    if paired and well_key is not None:
        common &= {well_key}
    groups = []
    for name, wells, snapshot in zip(('A', 'B'), sides, snapshots):
        categories = [('AB', common)]
        if not paired:
            categories.insert(0, (name, wells.keys() - common))
        for label, keys in categories:
            coordinates = sorted({tuple(xy) for key in keys for xy in wells[key]})
            if coordinates:
                groups.append(dict(color=SCENARIO_COLORS[label], xy=coordinates,
                                   crs=snapshot['conditions']['map_spec']['target_crs']))
    return groups


def render_comparison(figure, row, metric):
    figure.clear()
    ax = figure.add_subplot()
    for name, key in (('A', 'a'), ('B', 'b')):
        values = np.sort((row[key] or {}).get('values', []))
        if len(values):
            x, counts = np.unique(values, return_counts=True)
            ax.step(np.r_[x[0], x], np.r_[0, counts.cumsum() / len(values)],
                    where='post', label=f'{name} (n={len(values)})', color=SCENARIO_COLORS[name])
    unit = row['unit']
    ax.set(xlabel=f'{metric} [{unit}]' if unit else metric, ylabel='ECDF', ylim=(0, 1.03), title=row['region'])
    if ax.lines:
        ax.legend()
    ax.grid(alpha=.25)
