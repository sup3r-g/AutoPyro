"""Connect regional geochemistry analysis and draw its map overlays."""

from __future__ import annotations

from PySide6.QtWidgets import QMessageBox
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pyproj.exceptions import ProjError
import numpy as np
import shapely

from autopyro.spatial.map_tools import filtered_features
from autopyro.spatial.mapping import geometry_artists, project_geometries, transformer
from autopyro.spatial.regional_comparison import UNSTABLE_REGION_COLOR, region_fingerprint
from autopyro.visualization.plotting import category_key

from ..i18n import tr, message


def open_regional_geochemistry(sheet):
    from .regional_geochemistry import RegionalGeochemistryDialog
    if sheet.timer.isActive():
        sheet.timer.stop()
        sheet.redraw()
    snapshots = sheet.owner.model.project.view.get('regional_comparison_snapshots')
    if not sheet.report or not (sheet.report.get('row_ids') or snapshots):
        QMessageBox.information(sheet, tr('Геохимия по областям'), tr('В текущей выборке карты нет проб.'))
        return
    if sheet.regional_window is None:
        model = sheet.owner.model
        sheet.regional_window = RegionalGeochemistryDialog(model.project, sheet.report, sheet, data_revision=model.data_revision)
        sheet.regional_window.map_selection_changed.connect(sheet.highlight_regional_selection)
        sheet.regional_window.sample_opened.connect(sheet.open_regional_sample)
        sheet.regional_window.scenarios_changed.connect(sheet.owner.view_changed)
        sheet.regional_window.project_tasks_finished.connect(sheet.project_tasks_finished)
        sheet.destroyed.connect(sheet.regional_window.wait_for_worker)
    sheet.regional_window.show()
    sheet.regional_window.raise_()
    sheet.regional_window.activateWindow()


def open_regional_sample(sheet, row_id):
    index = next((sheet.owner.proxy.index(i, 0) for i, rid in enumerate(sheet.owner.row_ids()) if rid == row_id), None)
    if index is None:
        QMessageBox.information(sheet.regional_window, tr('Пробы'), tr('Проба скрыта фильтром исходной таблицы.'))
        return
    sheet.select_rows({row_id})
    sheet.owner.table.setCurrentIndex(index)
    sheet.owner.table.scrollTo(index)
    sheet.owner.raise_()
    sheet.owner.activateWindow()


def clear_regional_highlights(sheet):
    for artist in sheet.regional_highlights:
        artist.remove()
    sheet.regional_highlights = []
    sheet.canvas.draw_idle()


def highlight_regional_selection(sheet, selection):
    sheet.clear_regional_highlights()
    if not selection or not sheet.report or not sheet.result:
        return
    layer = next((l for l in sheet.report['layers'] if l['id'] == selection.get('layer_id')), None)
    ax = sheet.result['axes']
    limits = ax.get_xlim(), ax.get_ylim()
    auto = ax.get_autoscalex_on(), ax.get_autoscaley_on()
    before = set(ax.get_children())
    if selection.get('paired_map'):
        sheet.draw_paired_map(ax, selection['paired_map'])
    if layer is not None:
        features = [f for f in filtered_features(layer)
                    if category_key(f['properties'].get(selection['region_field'])) in selection['region_keys']]
        geometries = project_geometries([shapely.geometry.shape(f['geometry']) for f in features],
                                       layer['crs'], sheet.report['spec']['target_crs'])
        for geometry in geometries:
            geometry_artists(ax, geometry, dict(color='#ff3300', fill='none', alpha=1, line_width=3, zorder=21))
    groups = selection.get('point_groups', {'#ff3300': selection.get('row_ids', set())})
    for color, ids in groups.items():
        points = [p for p in sheet.report['points'] if ids.intersection(p['row_ids'])] if ids else []
        if points:
            ax.scatter([p['x'] for p in points], [p['y'] for p in points], s=110,
                       facecolors='none', edgecolors=color, linewidths=2, zorder=24)
    for group in selection.get('snapshot_points', []):
        x, y = np.asarray(group['xy'], dtype=float).T
        try:
            operation = transformer(group['crs'], sheet.report['spec']['target_crs'])
            x, y = (np.asarray(axis) for axis in operation.transform(x, y))
        except (ValueError, ProjError) as exc:
            sheet.status.setText(message(exc))
            continue
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.any():
            ax.scatter(x[valid], y[valid], s=110, facecolors='none',
                       edgecolors=group['color'], linewidths=2, zorder=24)
    sheet.regional_highlights = list(set(ax.get_children()) - before)
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_autoscalex_on(auto[0])
    ax.set_autoscaley_on(auto[1])
    sheet.canvas.draw_idle()


def draw_paired_map(sheet, ax, comparison):
    layer_id, field = comparison['layer_id'], comparison['field']
    layer = next((item for item in sheet.report['layers'] if item['id'] == layer_id), None)
    features = [(fid, geometry) for lid, fid, geometry in sheet.result['layer_features'] if lid == layer_id]
    if not layer or not features or region_fingerprint(layer, field) != comparison['fingerprint']:
        sheet.status.setText(tr('Для карты разниц нужен видимый слой с геометрией из снимков A и B.'))
        return
    values = comparison['values']
    missing = '#a7a7a7'
    cmap = colormaps['RdBu_r']
    finite = [value for value in values.values() if value is not None]
    limit = max(map(abs, finite), default=0)
    norm = Normalize(-1, 1)
    keys = {str(f['id']): category_key(f['properties'].get(field)) for f in layer['features']}
    has_missing = False
    has_unstable = False
    unstable_regions = comparison.get('unstable', set())
    for fid, geometry in features:
        key = keys[fid]
        value = values.get(key)
        has_missing |= value is None
        fill = missing if value is None else cmap(norm(value / limit if limit else 0))
        selected = key == comparison['selected']
        unstable = key in unstable_regions
        has_unstable |= unstable
        geometry_artists(ax, geometry, dict(fill=fill, color='#673782' if selected else '#555555',
                                           line_width=2.2 if selected else .6, alpha=1, zorder=3.2 if selected else 3))
        if unstable:
            geometry_artists(ax, geometry, dict(fill='none', color=UNSTABLE_REGION_COLOR,
                                               line_width=3 if selected else 2.2, alpha=1, zorder=22))
    title = tr('Медиана парных разниц B − A') + f" · {comparison['metric']}"
    if comparison['unit']:
        title += f" [{comparison['unit']}]"
    legend = ax.inset_axes([.035, .80, .31, .17], zorder=30)
    legend.set(facecolor='white', xticks=[], yticks=[])
    legend.patch.set_alpha(.92)
    legend.text(.05, .83, title, fontsize=8, va='center', transform=legend.transAxes)
    if limit:
        bar_ax = legend.inset_axes([.1, .50, .8, .15])
        bar = ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=bar_ax,
                                 orientation='horizontal', ticks=[-1, 0, 1])
        bar.set_ticklabels([f'{-limit:.4g}', '0', f'{limit:.4g}'])
        bar.ax.tick_params(labelsize=7, length=2)
    elif finite:
        legend.text(.05, .52, '0', fontsize=8, va='center', transform=legend.transAxes,
                    bbox=dict(facecolor=cmap(norm(0)), edgecolor='#666666', pad=2))
    if has_missing:
        legend.text(.05, .12, tr('Нет пар'), fontsize=8, va='center', transform=legend.transAxes,
                    bbox=dict(facecolor=missing, edgecolor='none', pad=2))
    if has_unstable:
        legend.text(.32 if has_missing else .05, .12, tr('Знак меняется'), fontsize=8, va='center',
                    color=UNSTABLE_REGION_COLOR, transform=legend.transAxes)
    count = comparison.get('selected_count')
    if count is not None:
        legend.text(.96, .12, f'n={count}', fontsize=8, va='center', ha='right', transform=legend.transAxes)
