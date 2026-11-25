"""Calculate variograms and compare interpolation methods, surfaces and validation metrics."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker
from matplotlib.colors import Normalize
from matplotlib.path import Path as MplPath
import numpy as np
import shapely

from autopyro.spatial.kriging import KrigingOptions, correlation, run_kriging
from autopyro.spatial.mapping import geometry_artists
from autopyro.spatial.surfaces import finite_surface_range

from ..i18n import tr, message
from ..tasks import TaskCancelled, run_with_progress
from ..widgets.results import ResultsModel
from .spatial_labels import INTERPOLATION_METHODS


def clear_kriging(sheet):
    sheet.generation += 1
    sheet.kriging_result = None
    sheet.comparison_pair.hide()
    sheet.comparison_back.hide()
    sheet.grid_inspection = None
    sheet.grid_inspector.hide()
    sheet.comparison_common_scale.setEnabled(False)
    sheet.comparison_text_summary.clear()
    sheet.comparison_text_summary.hide()
    index = sheet.comparison_surface.findData('isotropic-kriging')
    if index >= 0:
        with QSignalBlocker(sheet.comparison_surface):
            sheet.comparison_surface.removeItem(index)
    old = sheet.kriging_table.model()
    sheet.kriging_table.setModel(None)
    if old is not None:
        old.deleteLater()
    sheet.kriging_figure.clear()
    sheet.kriging_canvas.draw_idle()
    sheet.clear_partitions()
    sheet.update_surface_actions()


def calculate_kriging(sheet, *, preview):
    try:
        snapshot, options, size = sheet.calculation_input()
        settings = sheet.kriging_form.values()
        smoothing = settings.pop('rbf_smoothing')
        directions = settings.pop('directions')
        model = KrigingOptions(**settings)
        sheet.clear_kriging()
        generation = sheet.generation
        sheet.set_interpolation_status(tr('Выполняется расчёт вариограммы и кригинга…'))
        title = tr('Выполняется расчёт вариограммы и кригинга…')

        def calculate(update):
            result = run_kriging(snapshot, options, model, size, preview=preview, include_tin=True,
                                 rbf_smoothing=smoothing, include_cubic=True, directions=directions,
                                 progress=lambda fraction: update(title, int(100 * fraction)))
            update(title, 100)
            return result

        result = run_with_progress(sheet, title, calculate, cancellable=True)
        if generation == sheet.generation:
            sheet.completed_kriging(result, model)
            sheet.views.setCurrentWidget(sheet.kriging_tab)
            sheet.kriging_tab.ensureWidgetVisible(sheet.kriging_table, 0, 0)
            sheet.kriging_table.setFocus()
    except TaskCancelled:
        sheet.set_interpolation_status(tr('Сравнение методов отменено.'))
    except Exception as exc:
        sheet.set_interpolation_status(tr('Расчёт не выполнен: {0}', message(exc)))


def redraw_comparison_surface(sheet, *_):
    if sheet.kriging_result is not None:
        sheet.completed_kriging(sheet.kriging_result, KrigingOptions(**sheet.kriging_result['model']))


def comparison_grid(sheet, result, method):
    if method == 'kriging':
        return result
    if method == 'IDW':
        return result['idw_surface']
    if method == 'TIN':
        return result['tin_surface']
    return result['extra_surfaces'][method]


def draw_variogram(sheet, ax, result, model):
    variogram = result['variogram']
    if variogram:
        ax.scatter([point['distance_km'] for point in variogram], [point['semivariance'] for point in variogram],
                   s=[12 + 4 * np.sqrt(point['pairs']) for point in variogram], label=tr('Эмпирическая вариограмма'))
    directional = result.get('directions', {})
    for angle, points in directional.items():
        count = sum(point['pairs'] for point in points)
        ax.plot([point['distance_km'] for point in points], [point['semivariance'] for point in points],
                marker='o', markersize=3, linestyle=':', label=f'{angle}° (n={count})')
    distance = np.linspace(0, max([model.range_km, *(point['distance_km'] for point in variogram)]), 150)
    label = tr('Большая дальность') if model.range_ratio < 1 else tr('Заданная модель')
    ax.plot(distance, model.sill - (model.sill - model.nugget) * correlation(distance * 1000, model), label=label)
    if model.range_ratio < 1:
        ax.plot(distance, model.sill - (model.sill - model.nugget) * correlation(distance * 1000 / model.range_ratio, model),
                linestyle='--', label=tr('Малая дальность'))
    unit = result['unit']
    ylabel = tr('Полудисперсия') + (f' [{unit}²]' if unit else '')
    ax.set(xlabel=tr('Расстояние, км'), ylabel=ylabel)
    ax.legend(fontsize=8)
    if 'prediction' not in result:
        headers = ['Направление', 'Расстояние, км', ylabel, 'Число пар']
        series = [(tr('Все направления'), variogram)]
        series.extend((f'{angle}°', points) for angle, points in directional.items())
        rows = [[label, point['distance_km'], point['semivariance'], point['pairs']]
                for label, points in series for point in points]
        sheet.set_kriging_rows(headers, rows)


def draw_surface_grid(ax, grid, values, **style):
    artist = ax.pcolormesh(grid['x'], grid['y'], np.ma.masked_invalid(values), shading='nearest', **style)
    if grid['outline']:
        geometry_artists(ax, shapely.geometry.shape(grid['outline']),
                         dict(fill='none', color='#555555', alpha=1, line_width=.8))
        paths = [patch.get_path() for patch in ax.patches]
        artist.set_clip_path(MplPath.make_compound_path(*paths), ax.transData)
        ax.set_xlim(grid['x'][0], grid['x'][-1])
        ax.set_ylim(grid['y'][0], grid['y'][-1])
    return artist


def completed_kriging(sheet, result, model):
    sheet.kriging_figure.clear()
    has_surface = 'prediction' in result
    method = sheet.comparison_surface.currentData()
    pair = has_surface and method == 'idw-tin-difference'
    sheet.grid_inspection = None
    sheet.grid_inspector.setVisible(pair)
    sheet.comparison_common_scale.setEnabled(has_surface and not pair)
    sheet.kriging_table.setAccessibleName(tr('Результаты сравнения методов') if has_surface
                                         else tr('Данные эмпирической вариограммы'))
    sheet.comparison_text_summary.setAccessibleDescription(tr('Текстовая сводка поверхности') if has_surface
                                                          else tr('Текстовая сводка вариограммы'))
    if pair:
        summary = sheet.draw_idw_tin_difference(result)
    else:
        summary = ''
        sheet.draw_variogram(sheet.kriging_figure.add_subplot(121 if has_surface else 111), result, model)
    if has_surface:
        if model.range_ratio < 1 and sheet.comparison_surface.findData('isotropic-kriging') < 0:
            sheet.comparison_surface.addItem(tr(INTERPOLATION_METHODS['isotropic-kriging']), 'isotropic-kriging')
        if not pair:
            grid = sheet.comparison_grid(result, method)
            surface = sheet.kriging_figure.add_subplot(122, aspect='equal')
            has_values = np.isfinite(grid['prediction']).any()
            limits = result['surface_value_range'] if sheet.comparison_common_scale.isChecked() else None
            norm = Normalize(*limits) if limits is not None else None
            artist = sheet.draw_surface_grid(surface, grid, grid['prediction'], cmap='viridis', norm=norm)
            surface.scatter([p['x'] for p in result['locations']], [p['y'] for p in result['locations']], s=7, c='black')
            negative_y, negative_x = np.where(grid['prediction'] < 0)
            if len(negative_x):
                surface.scatter(grid['x'][negative_x], grid['y'][negative_y], marker='x', color='#d72622',
                                s=12, label=tr('Отрицательные ячейки'), zorder=5)
                surface.legend(fontsize=7)
            surface.set_title(sheet.kriging_method_label(method, model.range_ratio < 1))
            summary = tr('{0}. Прогноз в {1} из {2} узлов расчётной области.',
                         surface.get_title(), int(np.isfinite(grid['prediction']).sum()), int(grid['inside_domain'].sum()))
            if has_values:
                sheet.kriging_figure.colorbar(artist, ax=surface, label=result['unit'])
                summary += ' ' + tr('Прогноз от {0:.4g} до {1:.4g}.', *finite_surface_range(grid['prediction']))
                if result['unit']:
                    summary += ' ' + tr('Единицы: {0}.', result['unit'])
            else:
                surface.text(.5, .5, tr('На сетке нет прогноза для этого метода'),
                             transform=surface.transAxes, ha='center', va='center',
                             wrap=True, zorder=6, bbox=dict(facecolor='white', alpha=.9))
    else:
        pairs = sum(point['pairs'] for point in result['variogram'])
        summary = (tr('Эмпирическая вариограмма: {0} пар в {1} интервалах расстояния.',
                      pairs, len(result['variogram'])) if pairs else tr('Для эмпирической вариограммы нет пар.'))
    sheet.comparison_text_summary.setText(summary)
    sheet.comparison_text_summary.setVisible(bool(summary))
    sheet.kriging_canvas.draw_idle()
    if pair:
        sheet.set_interpolation_status(tr('Карта разности TIN − IDW построена.'))
    else:
        sheet.set_interpolation_status(tr('Методы проверены на одинаковых блоках.') if has_surface else tr('Вариограмма построена.'))
    sheet.kriging_result = result
    sheet.comparison_back.show()
    sheet.comparison_pair.setVisible(has_surface)
    if has_surface:
        selected_pair = sheet.comparison_pair.currentData()
        with QSignalBlocker(sheet.comparison_pair):
            sheet.comparison_pair.clear()
            sheet.comparison_pair.addItem(tr('Общие локации всех методов'))
            for methods in result['pairwise_validation']:
                names = [sheet.kriging_method_label(name, model.range_ratio < 1) for name in methods]
                sheet.comparison_pair.addItem(' / '.join(names), list(methods))
            sheet.comparison_pair.setCurrentIndex(max(0, sheet.comparison_pair.findData(selected_pair)))
        sheet.refresh_comparison_metrics()
    sheet.update_surface_actions()


def refresh_comparison_metrics(sheet, *_):
    result = sheet.kriging_result
    if not result or 'prediction' not in result:
        return
    pair = sheet.comparison_pair.currentData()
    paired = result['pairwise_validation'][tuple(pair)] if pair else None
    headers = ['Метод', 'С прогнозом', 'Доля с прогнозом', 'RMSE', 'MAE', 'Смещение',
               'Общие места', 'RMSE на общих местах', 'MAE на общих местах', 'Смещение на общих местах',
               'Отрицательные ячейки']
    rows = []
    for row in result['validation']:
        if paired is not None and row['method'] not in paired:
            continue
        common = row['common'] if paired is None else paired[row['method']]
        label = sheet.kriging_method_label(row['method'], result['model']['range_ratio'] < 1)
        rows.append([label, row['predicted'], f"{row['coverage']:.1%}", row['rmse'], row['mae'], row['bias'],
                     *(common[key] for key in ('predicted', 'rmse', 'mae', 'bias')), row['negative_cells']])
    sheet.set_kriging_rows(headers, rows)


def kriging_method_label(method, anisotropic=False):
    if method == 'kriging' and anisotropic:
        return tr('Анизотропный кригинг')
    return tr(INTERPOLATION_METHODS.get(method, method))


def set_kriging_rows(sheet, headers, rows):
    old = sheet.kriging_table.model()
    sheet.kriging_table.setModel(ResultsModel(headers, rows, sheet.kriging_table))
    if old is not None:
        old.deleteLater()
    sheet.kriging_table.resizeColumnsToContents()
