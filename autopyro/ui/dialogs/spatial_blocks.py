"""Run and display spatial block validation for IDW."""

from __future__ import annotations

from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import shapely

from autopyro.spatial.mapping import crs_label, render_map
from autopyro.spatial.spatial_interpolation import spatial_blocks

from ..i18n import tr, message
from ..tasks import TaskCancelled, run_with_progress
from ..widgets.results import ResultsModel
from .spatial_labels import FORECAST_REASONS, IDW_REGION_MODES


def calculate_blocks(sheet, *, evaluate):
    try:
        snapshot, options, size = sheet.calculation_input()
        generation = sheet.generation
        title = tr('Подготовка блоков IDW…')
        sheet.set_interpolation_status(title)
        if not evaluate:
            sheet.run_interpolation_task(lambda: spatial_blocks(snapshot, options, size),
                                        lambda result: sheet.completed_blocks(result, snapshot)
                                        if generation == sheet.generation else None)
            return
        sheet.block_result = None
        sheet.block_patches = []
        sheet.block_buffer_outline = None
        sheet.block_figure.clear()
        sheet.block_canvas.draw_idle()
        sheet.set_block_rows([])

        def calculate(update):
            result = spatial_blocks(snapshot, options, size, evaluate=True,
                                    progress=lambda fraction: update(title, int(100 * fraction)))
            update(title, 100)
            return result

        result = run_with_progress(sheet, title, calculate, cancellable=True)
        if generation == sheet.generation:
            sheet.completed_blocks(result, snapshot)
    except TaskCancelled:
        sheet.set_interpolation_status(tr('Блочная проверка IDW отменена.'))
    except Exception as exc:
        sheet.set_interpolation_status(tr('Расчёт не выполнен: {0}', message(exc)))


def set_block_rows(sheet, rows):
    old = sheet.block_table.model()
    headers = ['Блок', 'Мест', 'С прогнозом', 'Доля с прогнозом', 'RMSE', 'MAE', 'Смещение', 'Режим',
               'Исключено буфером']
    sheet.block_table.setModel(ResultsModel(headers, rows, sheet.block_table))
    if old is not None:
        old.deleteLater()
    sheet.block_table.selectionModel().selectionChanged.connect(sheet.highlight_block)
    sheet.block_table.resizeColumnsToContents()


def highlight_block(sheet, *_):
    indexes = sheet.block_table.selectionModel().selectedRows()
    selected = sheet.block_table.model().rows[indexes[0].row()][0] if indexes else None
    for i, patch in enumerate(sheet.block_patches, 1):
        patch.set_linewidth(2.6 if i == selected else 1)
        patch.set_edgecolor('#b3261e' if i == selected else '#466482')
    if sheet.block_buffer_outline is not None:
        sheet.block_buffer_outline.remove()
        sheet.block_buffer_outline = None
    if isinstance(selected, int) and sheet.block_result and sheet.block_result['buffer_km']:
        block = sheet.block_result['blocks'][selected - 1]
        buffered = shapely.box(*block['bounds']).buffer(sheet.block_result['buffer_km'] * 1000)
        x, y = buffered.exterior.xy
        sheet.block_buffer_outline, = sheet.block_patches[selected - 1].axes.plot(
            x, y, linestyle='--', color='#b3261e', linewidth=1.4, zorder=13)
    sheet.block_canvas.draw_idle()


def draw_blocks(sheet, figure, result, snapshot, *, residual_limit=None):
    layers = [
        layer if 'raster' in layer else dict(layer, features=[
            feature for feature in layer['features']
            if feature['geometry']['type'] not in ('Point', 'MultiPoint')
        ])
        for layer in snapshot['layers']
    ]
    preview = dict(snapshot, points=[], sites=[], layers=layers,
                   spec=dict(snapshot['spec'], legend=False, show_aggregates=False))
    _, drawing = render_map(preview, figure=figure, show_footer=False)
    ax = drawing['axes']
    patches = []
    size = result['block_size_km'] * 1000
    for block in result['blocks']:
        left, bottom, _, _ = block['bounds']
        patch = Rectangle((left, bottom), size, size, facecolor='#46648218', edgecolor='#466482', zorder=10)
        ax.add_patch(patch)
        patches.append(patch)
        if len(result['blocks']) <= 30:
            ax.annotate(str(block['id']), (left, bottom + size), xytext=(3, -3), textcoords='offset points',
                        ha='left', va='top', fontsize=8, zorder=12)
    points = result['locations']
    evaluated = residual_limit is not None or result['evaluated']
    mapped = len(snapshot['points'])
    source = snapshot.get('diagnostics', {}).get('input_rows', mapped)
    sample_count = tr('Пробы: в выборке {0} · с координатами {1} · в проверяемых локациях {2}',
                      source, mapped, sum(point['n_samples'] for point in points))
    location_count = tr('Проверяемых локаций: {0}', len(points))
    if evaluated:
        location_count += ' · ' + tr('С прогнозом: {0}', sum(point['predicted'] is not None for point in points))
    target = snapshot['spec']['target_crs']
    figure.subplots_adjust(bottom=.27)
    figure.text(.13, .02, '\n'.join((sample_count, location_count, crs_label(target))),
                fontsize=8, fontfamily=snapshot['spec']['font_family'])
    point_groups = [(points, '#0072b2', tr('Проверяемые места'), 'o')]
    if residual_limit is not None:
        valid = [point for point in points if point['residual'] is not None]
        if valid:
            limit = residual_limit or 1
            artist = ax.scatter([p['x'] for p in valid], [p['y'] for p in valid], c=[p['residual'] for p in valid],
                                cmap='RdBu_r', norm=Normalize(-limit, limit), s=22, zorder=11)
            figure.colorbar(artist, ax=ax, label=tr('Прогноз − измерение'))
        point_groups = []
    elif result['evaluated']:
        point_groups = [([p for p in points if p['predicted'] is not None], '#0072b2', tr('С прогнозом'), 'o')]
    if evaluated:
        missing = {}
        for point in points:
            if point['predicted'] is None:
                missing.setdefault(point.get('reason') or 'unknown', []).append(point)
        for reason, (label, color, marker) in FORECAST_REASONS.items():
            if reason in missing:
                point_groups.append((missing[reason], color, tr(label), marker))
    for visible, color, label, marker in point_groups:
        if visible:
            ax.scatter([p['x'] for p in visible], [p['y'] for p in visible], color=color, s=26,
                       marker=marker, zorder=11, label=label)
    bounds = np.asarray([block['bounds'] for block in result['blocks']])
    pad = size * .05
    if residual_limit is None:
        pad += result.get('buffer_km', 0) * 1000
    ax.set_xlim(bounds[:, 0].min() - pad, bounds[:, 2].max() + pad)
    ax.set_ylim(bounds[:, 1].min() - pad, bounds[:, 3].max() + pad)
    ax.set_aspect('equal')
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='best', fontsize=8)
    return ax, patches


def completed_blocks(sheet, result, snapshot):
    ax, sheet.block_patches = sheet.draw_blocks(sheet.block_figure, result, snapshot)
    sheet.block_buffer_outline = None
    points = result['locations']
    ax.set_title(tr('Блочная проверка IDW') + f" · {result['block_size_km']:g} " + tr('км'))

    def cells(label, count, item, mode, buffered=None):
        coverage = f"{item['coverage']:.1%}" if 'coverage' in item else None
        return [label, count, item.get('predicted'), coverage,
                *(item.get(key) for key in ('rmse', 'mae', 'bias')), tr(IDW_REGION_MODES[mode]), buffered]

    mode = result['region_mode']
    totals = result['comparisons'] or {mode: result['totals'] or {}}
    rows = [cells(tr('Всего'), len(points), item, key) for key, item in totals.items()]
    for block in result['blocks']:
        rows.extend(cells(block['id'], block['locations'], item, key, block['buffered_locations'])
                    for key, item in block.get('comparisons', {mode: block}).items())
    sheet.set_block_rows(rows)
    sheet.views.setCurrentIndex(1)
    sheet.block_canvas.draw_idle()
    sheet.set_interpolation_status(tr('Занятых блоков: {0}. Мест: {1}.', len(result['blocks']), len(points)))
    sheet.block_result = result
