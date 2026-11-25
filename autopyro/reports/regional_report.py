"""PDF and CSV exports of one frozen paired regional comparison."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from textwrap import wrap

from matplotlib import colormaps
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from shapely.geometry import shape

from autopyro.data_io.io import export_table_csv
from autopyro.spatial.map_tools import add_layer_labels
from autopyro.spatial.mapping import crs_label, geometry_artists, project_geometries
from autopyro.visualization.plotting import category_key
from autopyro.spatial.regional_comparison import UNSTABLE_REGION_COLOR, region_fingerprint


def report_features(snapshots, map_report):
    left, right = (snapshot['conditions'] for snapshot in snapshots)
    layers = [conditions['region_layer'] for conditions in (left, right)]
    needed = {fid for layer in layers for fid in layer['feature_ids']}
    features = {feature['id']: feature for layer in layers for feature in layer.get('features', [])}
    if needed - features.keys():
        layer = next((item for item in (map_report or {}).get('layers', []) if item['id'] == layers[0]['id']), None)
        field = left['options']['region_field']
        if layer and region_fingerprint(layer, field) == layers[0]['fingerprint']:
            for feature in layer['features']:
                fid = str(feature['id'])
                if fid in needed:
                    features[fid] = dict(geometry=feature['geometry'], region_key=category_key(feature['properties'].get(field)))
        if needed - features.keys():
            raise ValueError('Для отчёта откройте исходный слой областей или пересохраните снимки.')
    features = [features[fid] for fid in sorted(needed)]
    geometries = project_geometries([shape(feature['geometry']) for feature in features],
                                   layers[0]['crs'], left['map_spec']['target_crs']) if features else []
    return [(feature['region_key'], geometry) for feature, geometry in zip(features, geometries)]


def _status(value, tr):
    if value is None:
        return tr('Недостаточно пар для проверки')
    return tr('Знак меняется') if value else tr('Знак сохраняется')


def export_pairs_csv(path, rows, metric, names=('', ''), methods=('', '')):
    headers = ['scenario_a', 'scenario_b', 'method_a', 'method_b', 'region', 'well', 'metric', 'unit',
               'median_a', 'median_b', 'delta_b_minus_a',
               'median_without_well', 'sign_changed']
    export_table_csv(path, headers, ([*names, *methods, row['region'], pair['well'], metric, row['unit'], pair['a'], pair['b'],
                                    pair['delta'], pair['median_without'], pair['sign_changed']]
                                    for row in rows for pair in row['paired']['wells']))


def _map_page(pdf, rows, geometries, title, tr):
    figure = Figure(figsize=(11.7, 8.3))
    ax = figure.add_axes([.08, .16, .84, .73], aspect='equal')
    stats = {row['region_key']: row['paired'] for row in rows}
    labels = {row['region_key']: row['region'] for row in rows}
    values = [item['median'] for item in stats.values() if item['median'] is not None]
    limit = max(map(abs, values), default=0)
    cmap, norm = colormaps['RdBu_r'], Normalize(-1, 1)
    largest = {}
    for key, geometry in geometries:
        item = stats.get(key, {})
        value = item.get('median')
        fill = '#a7a7a7' if value is None else cmap(norm(value / limit if limit else 0))
        changed = item.get('sign_changed')
        geometry_artists(ax, geometry, dict(fill=fill, color=UNSTABLE_REGION_COLOR if changed else '#555555',
                                           line_width=2 if changed else .6, alpha=1, zorder=2 if changed else 1))
        if key in labels and (key not in largest or geometry.area > largest[key].area):
            largest[key] = geometry
    ax.autoscale_view()
    ax.set_axis_off()
    candidates = [(geometry, '\n'.join(wrap(labels[key], 26)) + f"\nn={stats[key]['n']}", 8)
                  for key, geometry in largest.items()]
    add_layer_labels(ax, candidates, 'DejaVu Sans')
    figure.suptitle(title, x=.05, ha='left', fontsize=15)
    if limit:
        bar = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=figure.add_axes([.08, .11, .44, .022]),
                              orientation='horizontal', ticks=[-1, 0, 1])
        bar.set_ticklabels([f'{-limit:.6g}', '0', f'{limit:.6g}'])
    elif values:
        figure.text(.08, .11, '0', bbox=dict(facecolor=cmap(norm(0)), edgecolor='#555555'))
    figure.text(.56, .11, tr('Нет пар'), bbox=dict(facecolor='#a7a7a7', edgecolor='none'))
    if any(item['sign_changed'] for item in stats.values()):
        figure.text(.70, .11, tr('Знак меняется'), color=UNSTABLE_REGION_COLOR)
    pdf.savefig(figure)


def _table_pages(pdf, title, headers, rows, widths):
    def lines(cells):
        return [wrap('—' if value is None else f'{value:.8g}' if isinstance(value, float) else str(value),
                     width) or [''] for value, width in zip(cells, widths)]

    def page():
        figure = Figure(figsize=(11.7, 8.3))
        figure.text(.05, .94, title, fontsize=15)
        figure.text(.95, .04, str(pdf.get_pagecount() + 1), ha='right', fontsize=9)
        return figure, write(figure, .88, lines(headers), bold=True)

    def write(figure, y, cells, bold=False):
        x = .05
        for column, width in zip(cells, widths):
            for i, line in enumerate(column):
                figure.text(x, y - i * .019, line, va='top', fontsize=8, family='DejaVu Sans Mono',
                            weight='bold' if bold else 'normal', parse_math=False)
            x += (width + 2) * 4.82 / (11.7 * 72)
        return y - .019 * max(map(len, cells)) - .004

    figure, y = page()
    for row in rows:
        cells = lines(row)
        while any(cells):
            available = int((y - .075) / .019)
            if available < min(max(map(len, cells)), 8):
                pdf.savefig(figure)
                figure, y = page()
                available = int((y - .075) / .019)
            y = write(figure, y, [column[:available] for column in cells])
            cells = [column[available:] for column in cells]
    pdf.savefig(figure)


def _condition_rows(snapshot, name, method, tr):
    conditions = snapshot['conditions']
    options = conditions['options']
    source = conditions['source']
    yield tr('Название'), name
    yield tr('Метод'), method
    yield tr('Время расчёта'), conditions['created_at']
    yield tr('Источник'), source.get('path', '—')
    yield 'SHA-256', source.get('sha256', '—')
    location = conditions.get('location_source') or {}
    if location:
        yield tr('Источник координат'), (location.get('source') or {}).get('path') or location.get('name', '—')
    yield tr('Полигональный слой'), conditions['layer_name']
    yield tr('Поле области'), options['region_field']
    yield tr('Фильтр слоя'), json.dumps(conditions['layer_filter'], ensure_ascii=False)
    yield tr('Геометрия слоя SHA-256'), conditions['region_layer']['fingerprint']
    yield 'CRS', crs_label(conditions['map_spec']['target_crs'])
    yield tr('Строк в выборке карты'), len(conditions['scope_row_ids'])
    yield tr('После фильтров'), len(conditions['filtered_row_ids'])
    yield tr('Выборка карты SHA-256'), sha256('\n'.join(conditions['scope_row_ids']).encode()).hexdigest()
    columns = conditions.get('columns', {})
    labels = {'toc_id': 'TOC', 'hi_id': 'HI', 'tmax_id': 'Tmax', 'well_id': 'Скважина', 'age_id': 'Возраст',
              'ages': 'Фильтр возраста', 'depth_id': 'Глубина / верх интервала', 'depth_bottom_id': 'Низ интервала',
              'depth_min': 'Глубина от', 'depth_max': 'Глубина до', 'toc_method_id': 'Метод TOC',
              'pyrolysis_method_id': 'Метод пиролиза (HI, Tmax)', 'toc_methods': 'Фильтр методов TOC',
              'pyrolysis_methods': 'Фильтр методов пиролиза', 'stratify_methods': 'Разделять распределения по методам',
              'text_numbers': 'Распознавать числа в текстовых колонках',
              'include_interpolated': 'Включать интерполированные значения', 'by_well': 'По скважинам (медиана замеров каждой скважины)'}
    for key, value in options.items():
        if key in ('layer_id', 'region_field'):
            continue
        if key.endswith('_id') and value:
            value = columns.get(value, value)
        if isinstance(value, bool):
            display = tr('Да') if value else tr('Нет')
        elif value is None or value == '' or value == [] or value == ():
            display = '—'
        else:
            display = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, tuple)) else value
        yield tr(labels.get(key, key)), display


def export_paired_pdf(path, rows, snapshots, names, methods, metric, map_report, tr=lambda text: text):
    geometries = report_features(snapshots, map_report)
    unit = rows[0]['unit']
    metric_label = f'{metric} [{unit}]' if unit else metric
    output = BytesIO()
    with PdfPages(output) as pdf:
        _map_page(pdf, rows, geometries, f"{tr('Медиана парных разниц B − A')} · {metric_label}", tr)
        headers = ['Область', 'Пар скважин', 'Медиана разниц', 'Q1', 'Q3', 'Мин. без пары', 'Макс. без пары', 'Устойчивость знака']
        data = [[row['region'], *(row['paired'][key] for key in
                 ('n', 'median', 'q1', 'q3', 'without_min', 'without_max')), _status(row['paired']['sign_changed'], tr)]
                for row in rows]
        _table_pages(pdf, f"{tr('Сводка по областям')} · {metric_label}", list(map(tr, headers)), data,
                     [27, 8, 12, 11, 11, 13, 13, 25])
        headers = ['Область', 'Скважина', 'Медиана A', 'Медиана B', 'B − A',
                   'Медиана без этой скважины', 'Устойчивость знака']
        pairs = [[row['region'], *(pair[key] for key in ('well', 'a', 'b', 'delta', 'median_without')),
                  _status(pair['sign_changed'], tr)] for row in rows for pair in row['paired']['wells']]
        _table_pages(pdf, f"{tr('Парные скважины')} · {metric_label}", list(map(tr, headers)), pairs,
                     [24, 21, 12, 12, 12, 15, 26])
        for side, snapshot, name, method in zip(('A', 'B'), snapshots, names, methods):
            _table_pages(pdf, f"{tr('Условия расчёта')} {side}", [tr('Параметр'), tr('Значение')],
                         _condition_rows(snapshot, name, method, tr), [53, 79])
    Path(path).write_bytes(output.getvalue())
