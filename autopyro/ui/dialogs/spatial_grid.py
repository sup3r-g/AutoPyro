"""Inspect nodes of the IDW and TIN difference grid and nearby holdout locations."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker
from PySide6.QtGui import QAccessible, QAccessibleAnnouncementEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSpinBox
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
import numpy as np

from autopyro.spatial.surfaces import COVERAGE_COLORS, compare_idw_tin_surfaces, finite_surface_range

from ..i18n import tr
from .spatial_labels import FORECAST_REASONS


def draw_idw_tin_difference(sheet, result):
    grid = result['idw_surface']
    comparison = compare_idw_tin_surfaces(grid, result['tin_surface'])
    axes = sheet.kriging_figure.subplots(1, 3, sharex=True, sharey=True, subplot_kw={'aspect': 'equal'})
    limits = finite_surface_range(comparison['idw'], comparison['tin'])
    norm = Normalize(*(limits or (0, 1)))
    for ax, key, label in zip(axes, ('idw', 'tin', 'difference'), ('IDW', 'TIN', 'TIN − IDW')):
        ax.set_title(label)
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.scatter([p['x'] for p in grid['locations']], [p['y'] for p in grid['locations']],
                   s=7, color='black', zorder=5)
        if key != 'difference':
            artist = sheet.draw_surface_grid(ax, grid, comparison[key], cmap='viridis', norm=norm)
    if limits is not None:
        sheet.kriging_figure.colorbar(artist, ax=axes[:2].tolist(), location='bottom', label=result['unit'])
    delta = comparison['difference']
    limit = float(np.max(np.abs(delta[np.isfinite(delta)]), initial=0)) or 1
    delta_artist = sheet.draw_surface_grid(axes[2], grid, delta, cmap='RdBu_r', norm=Normalize(-limit, limit))
    if comparison['common'].any():
        sheet.kriging_figure.colorbar(delta_artist, ax=axes[2], location='bottom',
                                     label=result['unit'], ticks=[0] if not np.any(delta[comparison['common']]) else None)
    else:
        axes[2].text(.5, .5, tr('Нет общего покрытия'), transform=axes[2].transAxes,
                     ha='center', va='center', zorder=6, bbox=dict(facecolor='white', alpha=.9))
    categories = np.full(delta.shape, np.nan)
    handles = []
    colors = [COVERAGE_COLORS[index] for index in (1, 2, 0)]
    for index, (key, label, color) in enumerate(zip(
            ('idw_only', 'tin_only', 'neither'), (tr('Только IDW'), tr('Только TIN'), tr('Оба без прогноза')), colors)):
        if comparison[key].any():
            categories[comparison[key]] = index
            handles.append(Patch(facecolor=color, label=label))
    if handles:
        overlay = axes[2].pcolormesh(grid['x'], grid['y'], np.ma.masked_invalid(categories),
                                     shading='nearest', cmap=ListedColormap(colors), norm=Normalize(-.5, 2.5))
        overlay.set_clip_path(delta_artist.get_clip_path())
        axes[2].legend(handles=handles, loc='upper center', fontsize=7)
    markers = [ax.plot([], [], marker='s', markersize=8, markerfacecolor='none',
                       markeredgecolor='#242424', markeredgewidth=1.6, zorder=12)[0] for ax in axes]
    sheet.grid_inspection = (grid, comparison, markers, result['unit'])
    holdouts = result['idw_tin_holdouts']
    labels = sheet.sample_location_labels(sheet.snapshot)
    sheet.grid_holdouts = [(point, ', '.join(dict.fromkeys(labels[rid] for rid in point['row_ids'] if rid in labels)))
                          for point in holdouts]
    sheet.grid_holdout_xy = np.asarray([[point['x'], point['y']] for point in holdouts])
    for field, size in zip(sheet.grid_indices, comparison['common'].shape):
        with QSignalBlocker(field):
            field.setRange(1, size)
    sheet.inspect_grid_node(announce=False)
    counts = [int(comparison[key].sum()) for key in ('common', 'idw_only', 'tin_only', 'neither')]
    summary = tr('Узлы сетки: общих {0}, только IDW {1}, только TIN {2}, без прогноза {3}.', *counts)
    limits = finite_surface_range(delta)
    if limits is not None:
        summary += ' ' + tr('Разность TIN − IDW от {0:+.4g} до {1:+.4g}.', *limits)
        if result['unit']:
            summary += ' ' + tr('Единицы: {0}.', result['unit'])
    else:
        summary += ' ' + tr('Нет общего покрытия')
    return summary


def add_grid_index_fields(layout):
    indices = QHBoxLayout()
    fields = []
    for name in (tr('Строка сетки'), tr('Столбец сетки')):
        label = QLabel(name)
        field = QSpinBox()
        field.setAccessibleName(name)
        label.setBuddy(field)
        indices.addWidget(label)
        indices.addWidget(field)
        fields.append(field)
    indices.addStretch()
    layout.addLayout(indices)
    return fields


def pick_grid_indices(event, grid, markers, fields):
    if event.button != 1 or event.xdata is None or event.ydata is None:
        return False
    if event.inaxes not in [marker.axes for marker in markers]:
        return False
    indices = []
    for coordinates, position in zip((grid['y'], grid['x']), (event.ydata, event.xdata)):
        half_step = (coordinates[1] - coordinates[0]) / 2
        if not coordinates[0] - half_step <= position <= coordinates[-1] + half_step:
            return False
        indices.append(int(np.abs(coordinates - position).argmin()) + 1)
    for field, index in zip(fields, indices):
        with QSignalBlocker(field):
            field.setValue(index)
    return True


def pick_grid_node(sheet, event):
    if sheet.grid_inspection is None:
        return
    grid, _, markers, _ = sheet.grid_inspection
    if sheet.pick_grid_indices(event, grid, markers, sheet.grid_indices):
        sheet.inspect_grid_node()


def inspect_grid_node(sheet, *_, announce=True):
    if sheet.grid_inspection is None:
        return
    grid, comparison, markers, unit = sheet.grid_inspection
    row, column = (field.value() - 1 for field in sheet.grid_indices)
    x, y = grid['x'][column], grid['y'][row]
    for marker in markers:
        marker.set_data([x], [y])
    text = tr('X {0:.2f} м, Y {1:.2f} м.', x, y)
    if grid['inside_domain'][row, column]:
        for label, key in [('IDW', 'idw'), ('TIN', 'tin'), ('TIN − IDW', 'difference')]:
            value = comparison[key][row, column]
            display = f'{value:.5g}' if np.isfinite(value) else tr('Недоступно')
            text += f' {label}: {display}.'
        if unit:
            text += ' ' + tr('Единицы: {0}.', unit)
    else:
        text += ' ' + tr('Узел вне расчётной области.')
    distances = np.linalg.norm(sheet.grid_holdout_xy - [x, y], axis=1)
    nearest = np.argsort(distances, kind='stable')[:3]
    text += ' ' + tr('До ближайшей проверочной локации {0:.3g} км.', distances[nearest[0]] / 1000)
    lines = [tr('Ближайшие проверочные локации · остаток = прогноз − измерение')]
    for index in nearest:
        point, name = sheet.grid_holdouts[index]
        if not name:
            name = tr('X {0:.2f} м, Y {1:.2f} м.', point['x'], point['y'])
        residuals = []
        for method in ('idw', 'tin'):
            residual = point[f'{method}_residual']
            reason = FORECAST_REASONS.get(point[f'{method}_reason'], FORECAST_REASONS['unknown'])
            residuals.append(f'{residual:+.4g}' if residual is not None else tr(reason[0]))
        lines.append(tr('{0} · {1:.3g} км · блок {2} · IDW {3} · TIN {4}',
                        name, distances[index] / 1000, point['block'], *residuals))
    sheet.grid_holdout_text.setText('\n'.join(lines))
    changed = text != sheet.grid_node_text.text()
    sheet.grid_node_text.setText(text)
    if announce and changed and QAccessible.isActive():
        QAccessible.updateAccessibility(QAccessibleAnnouncementEvent(sheet.grid_node_text, text))
    sheet.kriging_canvas.draw_idle()
