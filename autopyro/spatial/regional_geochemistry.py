"""Read-only descriptive comparison of working geochemical values by region."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.colors import to_hex

from autopyro.model.core import timestamp
from autopyro.data_io.io import export_table_csv
from .map_tools import assign_regions, filtered_features
from .mapping import layer_categories
from autopyro.visualization.plotting import category_key, category_keys
from autopyro.analysis.statistics import snapshot
from autopyro.model.tabular import project_frame, classify_series, exclusion_mask, depth_filter_masks


@dataclass(frozen=True)
class RegionalOptions:
    layer_id: str
    region_field: str
    toc_id: str = ''
    hi_id: str = ''
    tmax_id: str = ''
    well_id: str = ''
    age_id: str = ''
    ages: tuple = ()
    depth_id: str = ''
    depth_bottom_id: str = ''
    depth_min: float | None = None
    depth_max: float | None = None
    toc_method_id: str = ''
    toc_methods: tuple = ()
    pyrolysis_method_id: str = ''
    pyrolysis_methods: tuple = ()
    stratify_methods: bool = True
    text_numbers: bool = True
    include_interpolated: bool = False
    by_well: bool = False

    def metrics(self):
        return [(name, cid) for name, cid in [('TOC', self.toc_id), ('HI', self.hi_id), ('Tmax', self.tmax_id)] if cid]

    def columns(self):
        return {cid for cid in [self.toc_id, self.hi_id, self.tmax_id, self.well_id, self.age_id,
                               self.depth_id, self.depth_bottom_id, self.toc_method_id, self.pyrolysis_method_id] if cid}


def validate(project, report, options):
    columns = {c.id: c for c in project.columns}
    if not options.metrics() or options.columns() - columns.keys():
        raise ValueError('Выберите существующие колонки показателей и фильтров')
    if options.by_well and not options.well_id:
        raise ValueError('Для сравнения по скважинам выберите колонку скважины')
    if len({cid for _, cid in options.metrics()}) != len(options.metrics()):
        raise ValueError('Для TOC, HI и Tmax выберите разные колонки')
    if options.ages and not options.age_id or options.toc_methods and not options.toc_method_id or options.pyrolysis_methods and not options.pyrolysis_method_id:
        raise ValueError('Выберите колонку для заданного фильтра')
    limits = [value for value in (options.depth_min, options.depth_max) if value is not None]
    if not all(math.isfinite(value) for value in limits) or (len(limits) == 2 and limits[0] > limits[1]):
        raise ValueError('Проверьте пределы глубины')
    if limits and not options.depth_id:
        raise ValueError('Выберите колонку глубины')
    if options.depth_bottom_id:
        if not options.depth_id or options.depth_id == options.depth_bottom_id:
            raise ValueError('Для интервала выберите разные колонки верха и низа')
        upper, lower = columns[options.depth_id], columns[options.depth_bottom_id]
        if upper.unit and lower.unit and upper.unit != lower.unit:
            raise ValueError('Верх и низ глубины должны иметь одинаковые единицы')
    layer = next((item for item in report['layers'] if item['id'] == options.layer_id), None)
    fields = set(layer.get('fields', [])) | {key for f in layer['features'] for key in f['properties']} if layer else set()
    if not layer or options.region_field not in fields:
        raise ValueError('Выберите полигональный слой и поле палеогеографической области')
    row_ids = report['row_ids']
    if len(set(row_ids)) != len(row_ids) or set(row_ids) - project.row_map().keys():
        raise ValueError('Выборка карты не соответствует таблице замеров')
    return columns, layer


def prepare_input(project, report, options):
    """Freeze just this analysis's cells, QC decisions and spatial source."""
    _, layer = validate(project, report, options)
    key = report['spec'].get('sample_key')
    columns = options.columns() | ({key} if key in {c.id for c in project.columns} else set())
    copied = snapshot(project, report['row_ids'], columns)
    spatial = dict(spec=deepcopy(report['spec']), row_ids=list(report['row_ids']), layers=[deepcopy(layer)],
                   points=[{key: deepcopy(point[key]) for key in ('id', 'row_ids', 'x', 'y')} for point in report['points']])
    spatial['location_source'] = deepcopy(report.get('location_source', next(
        ({key: item.get(key) for key in ('id', 'name', 'source', 'crs')} for item in report['layers']
         if item['id'] == report['spec'].get('point_layer')), {})))
    return copied, spatial


def _select_samples(project, report, options):
    frame = project_frame(project, report['row_ids'], sorted(options.columns())).reindex(pd.Index(report['row_ids'], dtype=object))
    rejected = pd.Series(None, index=frame.index, dtype=object)
    if options.ages:
        allowed = {category_key(value) for value in options.ages}
        rejected.loc[~category_keys(frame[options.age_id]).isin(allowed)] = 'age_filter'
    if options.depth_min is not None or options.depth_max is not None:
        active = frame.loc[rejected.isna()]
        depth = classify_series(active[options.depth_id], numeric=True)['number']
        depth.loc[exclusion_mask(project, active.index, options.depth_id, 'statistics')] = None
        bottom = None
        if options.depth_bottom_id:
            bottom = classify_series(active[options.depth_bottom_id], numeric=True)['number']
            bottom.loc[exclusion_mask(project, active.index, options.depth_bottom_id, 'statistics')] = None
        invalid, outside = depth_filter_masks(depth, bottom, options.depth_min, options.depth_max)
        rejected.loc[invalid.index[invalid]] = 'invalid_depth'
        rejected.loc[outside.index[outside & ~invalid]] = 'depth_filter'
    return frame.loc[rejected.isna()], rejected.dropna().value_counts(sort=False).to_dict()


def _metric_rows(project, frame, region_keys, columns, metric, cid, options):
    """Prepare one metric without duplicating unrelated source columns."""
    method_id = options.toc_method_id if metric == 'TOC' else options.pyrolysis_method_id
    allowed = options.toc_methods if metric == 'TOC' else options.pyrolysis_methods
    method = frame[method_id] if method_id else pd.Series(None, index=frame.index, dtype=object)
    if not method_id:
        method.iloc[:] = None
    method_keys = category_keys(method)
    selected = method_keys.isin({category_key(value) for value in allowed}) if allowed else pd.Series(True, index=frame.index)
    active = frame.loc[selected]
    metadata = columns[cid].metadata
    if metadata.get('stale') or metadata.get('method') and not options.include_interpolated:
        status = 'stale_excluded' if metadata.get('stale') else 'interpolation_excluded'
        data = pd.DataFrame({'status': status, 'number': None}, index=active.index, dtype=object)
    else:
        data = classify_series(active[cid], numeric=columns[cid].kind == 'number' or options.text_numbers)
        data.loc[exclusion_mask(project, active.index, cid, 'statistics'), ['status', 'number']] = ['qc_excluded', None]
    well = active[options.well_id] if options.well_id else pd.Series(None, index=active.index, dtype=object)
    if not options.well_id:
        well.iloc[:] = None
    has_well = well.map(lambda value: value is not None and bool(str(value).strip()))
    well_keys = category_keys(well).where(has_well, None)
    table = pd.DataFrame({
        'row_id': active.index, 'region_key': region_keys.loc[active.index], 'metric': metric,
        'method_key': method_keys.loc[active.index] if options.stratify_methods else 'pooled',
        'method': method.loc[active.index], 'method_category': method_keys.loc[active.index],
        'well': well, 'well_key': well_keys, 'status': data['status'], 'value': data['number'],
        'raw': active[cid], 'age': active[options.age_id] if options.age_id else None,
        'depth': active[options.depth_id] if options.depth_id else None,
        'bottom': active[options.depth_bottom_id] if options.depth_bottom_id else None,
    }, dtype=object)
    table['value'] = table['value'].astype(float)
    for column, cid in [('profile_depth', options.depth_id), ('profile_bottom', options.depth_bottom_id)]:
        table[column] = None
        if cid:
            numbers = classify_series(active[cid], numeric=True)['number']
            table[column] = numbers.mask(exclusion_mask(project, active.index, cid, 'statistics'), None)
    return table, int((~selected).sum())


def _summarize_metrics(table, regions, columns, options):
    keys = ['region_key', 'metric', 'method_key']
    grouped = table.groupby(keys, sort=False)
    statistics = grouped.agg(n_total=('row_id', 'size'), n_valid=('value', 'count'),
                             n_wells_total=('well_key', 'nunique'))
    valid = table.loc[table['value'].notna()]
    observations = valid
    if options.by_well:
        observations = (valid.dropna(subset=['well_key']).groupby(keys + ['well_key'], sort=False)
                        .agg(value=('value', 'median'), well=('well', 'first')).reset_index())
        observations['row_id'] = None
    observed = observations.groupby(keys, sort=False)
    statistics = statistics.join(observed['value'].agg(n_observations='size', mean='mean', std='std', min='min', max='max'))
    statistics['n_observations'] = statistics['n_observations'].fillna(0).astype(int)
    quantiles = observed['value'].quantile([.1, .25, .5, .75, .9], interpolation='linear').unstack(-1)
    quantiles = quantiles.reindex(columns=[.1, .25, .5, .75, .9])
    statistics = statistics.join(quantiles.rename(columns={.1: 'p10', .25: 'q1', .5: 'median', .75: 'q3', .9: 'p90'}))
    valid_groups = valid.groupby(keys, sort=False)
    statistics['n_wells'] = valid_groups['well_key'].nunique().reindex(statistics.index, fill_value=0)
    numeric_fields = ['mean', 'median', 'std', 'min', 'max', 'q1', 'q3', 'p10', 'p90']
    defined = statistics.loc[statistics['n_observations'].gt(0), numeric_fields].drop(columns='std')
    defined_std = statistics.loc[statistics['n_observations'].gt(1), 'std']
    if not np.isfinite(defined.to_numpy(dtype=float)).all() or not np.isfinite(defined_std.to_numpy(dtype=float)).all():
        raise ValueError('Численное переполнение при расчёте статистики областей')
    statistics = statistics.astype(object).where(statistics.notna(), None).to_dict('index')
    distributions_by_group = {key: group[['row_id', 'value', 'well']].to_dict('records') for key, group in observed}
    samples_by_group = {}
    details, by_region = {}, {}
    for key, group in grouped:
        samples = group[['row_id', 'well', 'age', 'depth', 'bottom', 'method', 'metric', 'raw', 'value', 'status',
                         'profile_depth', 'profile_bottom']]
        samples_by_group[key] = samples.astype(object).where(samples.notna(), None).to_dict('records')
        details[key] = dict(method=group.iloc[0]['method'] if options.stratify_methods else None,
                            row_ids=group['row_id'].tolist(),
                            methods=group['method_category'].value_counts(sort=False).to_dict(),
                            excluded=group.loc[group['status'].ne('numeric'), 'status'].value_counts(sort=False).to_dict())
        by_region.setdefault(key[:2], []).append(key)
    summary, distributions = [], []
    for region_key, region in regions.items():
        for metric, cid in options.metrics():
            matching = by_region.get((region_key, metric), [None])
            matching = sorted(matching, key=lambda key: category_key(details[key]['method']) if key is not None else '')
            for index, key in enumerate(matching):
                stats = statistics[key] if key is not None else dict.fromkeys(numeric_fields)
                detail = details[key] if key is not None else dict(method=None, methods={}, excluded={}, row_ids=[])
                record = dict(region_key=region_key, region=region['value'], region_label=region['label'], metric=metric,
                              column_id=cid, unit=columns[cid].unit, pooled=not options.stratify_methods, **detail)
                record.update({field: stats.get(field, 0) for field in ['n_total', 'n_valid', 'n_wells', 'n_wells_total', 'n_observations']})
                if not options.well_id:
                    record['n_wells'] = record['n_wells_total'] = None
                record.update({field: stats[field] for field in numeric_fields})
                summary.append(record)
                distributions.append(dict(summary_index=len(summary) - 1, color=region['color'],
                    linestyle=['-', '--', ':', '-.'][index % 4],
                    marker=['o', 's', '^', 'v', 'D', 'P', 'X', '*', '<', '>', 'h'][index % 11],
                    values=distributions_by_group.get(key, []), samples=samples_by_group.get(key, [])))
    return summary, distributions


def analyze_regions(project, report, options):
    columns, layer = validate(project, report, options)
    if any(columns[cid].metadata.get('stale') for cid in (options.depth_id, options.depth_bottom_id) if cid):
        raise ValueError('Фильтр глубины ссылается на устаревшую колонку')
    assignments, _ = assign_regions(report, options.layer_id, [options.region_field])
    colors = layer_categories(dict(layer, color_by=options.region_field, color_mode='category'))
    regions = {}
    for feature in filtered_features(layer):
        value = feature['properties'].get(options.region_field)
        if value is not None and str(value).strip():
            key = category_key(value)
            regions[key] = dict(value=value, label=colors[key][0], color=to_hex(colors[key][1]))
    frame, global_exclusions = _select_samples(project, report, options)
    selected_ids = frame.index.tolist()
    spatial_status = pd.Series([assignments[rid]['status'] for rid in frame.index], index=frame.index, dtype=object)
    region = pd.Series([assignments[rid]['values'][options.region_field] for rid in frame.index], index=frame.index, dtype=object)
    missing_region = region.map(lambda value: value is None or not str(value).strip())
    spatial_status.loc[spatial_status.eq('matched') & missing_region] = 'missing_region'
    spatial_counts = spatial_status.value_counts(sort=False).to_dict()
    key_column = report['spec'].get('sample_key')
    if key_column not in columns:
        key_column = options.well_id
    source_rows = project.row_map()
    spatial_issues = [dict(row_id=rid, status=status, key=source_rows[rid].values.get(key_column))
                      for rid, status in spatial_status.items() if status != 'matched']
    frame = frame.loc[spatial_status.eq('matched')]
    assigned_ids = frame.index.tolist()
    region_keys = category_keys(region.loc[frame.index])
    metric_frames, method_exclusions = [], {}
    for metric, cid in options.metrics():
        metric_frame, excluded = _metric_rows(project, frame, region_keys, columns, metric, cid, options)
        metric_frames.append(metric_frame)
        if excluded:
            method_exclusions[metric] = excluded
    table = pd.concat(metric_frames, ignore_index=True)
    summary, distributions = _summarize_metrics(table, regions, columns, options)
    return dict(format='autopyro.regional_geochemistry', version=1, created_at=timestamp(), project_id=project.id,
                source=deepcopy(project.source), layer_name=layer['name'], region_field=options.region_field,
                layer_source=deepcopy(layer.get('source', {})),
                location_source=deepcopy(report.get('location_source', next(
                    ({key: item.get(key) for key in ('id', 'name', 'source', 'crs')} for item in report['layers']
                     if item['id'] == report['spec'].get('point_layer')), {}))),
                options=asdict(options), map_spec=deepcopy(report['spec']), layer_filter=deepcopy(layer.get('filter', {})),
                scope_row_ids=list(report['row_ids']), filtered_row_ids=selected_ids, assigned_row_ids=assigned_ids,
                summary=summary, distributions=distributions, spatial_issues=spatial_issues,
                depth_unit=columns[options.depth_id].unit if options.depth_id else '',
                diagnostics=dict(scope_rows=len(report['row_ids']), filtered_rows=len(selected_ids), assigned_rows=len(assigned_ids),
                                 filters=dict(global_exclusions), spatial=dict(spatial_counts), method_filters=dict(method_exclusions)),
                definitions=dict(std='sample, ddof=1; undefined at n<2', quantiles='linear',
                                 weighting=('median per well within each group; rows without well ID omitted'
                                            if options.by_well else 'one working table row per observation; no well-level averaging'),
                                 comparison='descriptive only; repeated measurements are not assumed independent',
                                 qualifiers='missing, non-detect, censored, unparsed and explicit QC exclusions are not substituted'))


def render_distributions(result, metric, figure=None, translate=lambda text: text):
    figure = figure or Figure()
    figure.clear()
    selected = [(result['summary'][entry['summary_index']], entry) for entry in result['distributions']
                if result['summary'][entry['summary_index']]['metric'] == metric and entry['values']]
    # Long region and method labels may wrap to four lines. Give every group
    # enough physical space; the dialog scrolls the canvas instead of shrinking it.
    figure.set_size_inches(14, max(6, len(selected) * .8))
    left, right = figure.subplots(1, 2)
    labels, arrays, colors = [], [], []
    import textwrap
    for row, entry in selected:
        values = np.sort([item['value'] for item in entry['values']])
        method = translate('Все методы') if row['pooled'] else str(row['method']) if row['method'] is not None else translate('Метод не задан')
        if isinstance(row['method'], str) and row['method'] != row['method'].strip():
            method = repr(row['method'])
        label = f"{row.get('region_label', row['region'])} · {method} (n={row.get('n_observations', row['n_valid'])})"
        left.step(values, np.arange(1, len(values) + 1) / len(values), where='post', color=entry['color'], linestyle=entry['linestyle'],
                  marker=entry.get('marker', 'o'), markersize=3, markevery=max(1, len(values) // 10), label=textwrap.fill(label, 32))
        labels.append(textwrap.fill(label, 28))
        arrays.append(values)
        colors.append(entry['color'])
    if arrays:
        boxes = right.boxplot(arrays, orientation='horizontal', tick_labels=labels, patch_artist=True, showfliers=True)
        for box, color in zip(boxes['boxes'], colors):
            box.set_facecolor(color)
            box.set_alpha(.55)
        right.invert_yaxis()
        left.legend(fontsize=7, loc='lower right')
    else:
        left.text(.5, .5, translate('Нет числовых данных'), ha='center', transform=left.transAxes)
    unit = next((row['unit'] for row in result['summary'] if row['metric'] == metric), '')
    label = f'{metric} [{unit}]' if unit else metric
    for ax in (left, right):
        ax.set_xlabel(label)
        ax.grid(alpha=.25)
        ax.tick_params(labelsize=7)
    left.set_ylabel('ECDF')
    left.set_ylim(0, 1.02)
    right.set_title(translate('Медиана / IQR / 1,5 IQR / выбросы'), fontsize=9)
    figure.suptitle(metric)
    figure.tight_layout()
    return figure


def export_regional_summary(path, result):
    fields = ['region', 'metric', 'unit', 'method', 'pooled', 'n_total', 'n_valid', 'n_wells', 'n_wells_total',
              'mean', 'median', 'std', 'min', 'max', 'p10', 'q1', 'q3', 'p90']
    if result['options'].get('by_well'):
        fields.insert(9, 'n_observations')
    statuses = sorted({key for row in result['summary'] for key in row['excluded']})
    export_table_csv(path, fields + statuses, (
        [row[field] for field in fields] + [row['excluded'].get(key, 0) for key in statuses]
        for row in result['summary']
    ))
