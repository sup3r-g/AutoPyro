"""Render an existing fit archive as a layer on a live X–Y chart."""
from copy import deepcopy

import numpy as np


def archived_curve(project, archive_id):
    return next((record for record in project.view.get('curve_fit_results', [])
                 if record.get('id') == archive_id), None)


def curve_snapshot(record):
    keys = ('id', 'name', 'at', 'stale', 'settings', 'formula', 'source_digest',
            'source_digest_version', 'axis_labels', 'axis_units', 'grid', 'curve')
    return {key: deepcopy(record[key]) for key in keys if key in record}


def curve_axes(record, columns):
    settings = record.get('settings', {})
    ids = (settings.get('x'), settings.get('y'))
    if any(cid not in columns for cid in ids):
        raise ValueError('Показатели сохранённой кривой отсутствуют в проекте')
    labels = [f'{columns[cid].name} [{columns[cid].unit}]' if columns[cid].unit else columns[cid].name
              for cid in ids]
    units = [columns[cid].unit for cid in ids]
    matches = record['axis_units'] == units if 'axis_units' in record else record.get('axis_labels') == labels
    if not matches:
        raise ValueError('Единицы или подписи осей сохранённой кривой изменились')
    return ids


def draw_archived_curve(ax, record, spec, columns, *, language="ru"):
    if curve_axes(record, columns) != (spec.x_id, spec.y_id):
        raise ValueError('Для сохранённой кривой выберите исходные показатели X и Y')
    grid = np.asarray(record.get('grid'), dtype=float)
    curve = np.asarray(record.get('curve'), dtype=float)
    if grid.ndim != 1 or curve.shape != grid.shape or len(grid) < 2:
        raise ValueError('Некорректная сетка сохранённой кривой')
    visible = np.isfinite(grid) & np.isfinite(curve)
    if spec.x_axis.scale == 'log':
        visible &= grid > 0
    if spec.y_axis.scale == 'log':
        visible &= curve > 0
    if not (visible[1:] & visible[:-1]).any():
        raise ValueError('Сохранённая кривая не отображается при выбранных шкалах')
    method = str(record['settings'].get('method') or '').upper()
    name = record.get('name', record['id'])
    label = f'{name} ({method})' if method else name
    if record.get('stale'):
        label += ' · stale archive' if language == "en" else ' · архив устарел'
    line, = ax.plot(np.where(visible, grid, np.nan), np.where(visible, curve, np.nan),
                    color='#b34d00', linestyle='--', linewidth=1.5, label=label, zorder=2.5)
    line.set_gid('archived-curve')
    return line
