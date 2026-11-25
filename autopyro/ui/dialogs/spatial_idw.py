"""Calculate and display IDW predictions and leave-one-location-out diagnostics."""

from __future__ import annotations

from matplotlib.path import Path as MplPath
import numpy as np
import shapely

from autopyro.spatial.mapping import crs_label, geometry_artists
from autopyro.spatial.spatial_interpolation import interpolate
from autopyro.spatial.surfaces import finite_surface_range

from ..i18n import tr, message
from ..tasks import TaskCancelled, run_with_progress


def set_idw_preview_visible(sheet, visible):
    sheet.idw_intro.setVisible(not visible)
    sheet.idw_summary.setVisible(visible)
    if not visible:
        sheet.idw_summary.clear()
    sheet.toolbar.setVisible(visible)
    sheet.canvas_view.scroll.setVisible(visible)


def calculate(sheet):
    try:
        sheet.invalidate(keep_comparison=True)
        snapshot, options, _ = sheet.calculation_input()
        generation = sheet.generation
        sheet.views.setCurrentIndex(0)
        title = tr('Выполняются IDW и проверка по целым местам…')
        sheet.set_interpolation_status(title)

        def calculate(update):
            return interpolate(snapshot, options, progress=lambda fraction: update(title, int(100 * fraction)))

        result = run_with_progress(sheet, title, calculate, cancellable=True)
        if generation == sheet.generation:
            sheet.completed(result)
    except TaskCancelled:
        sheet.set_interpolation_status(tr('Расчёт IDW отменён.'))
    except Exception as exc:
        sheet.set_interpolation_status(tr('Расчёт не выполнен: {0}', message(exc)))


def completed(sheet, result):
    sheet.result = result
    sheet.figure.clear()
    axes = sheet.figure.subplots(1, 2)
    grid = result["grid"]
    column = next(c for c in result['map_snapshot']['columns'] if c['id'] == result['options']['value_id'])
    unit = column['unit']
    unit_label = f"{column['name']} [{unit}]" if unit else column['name']
    ranges = {}
    origin_x, origin_y = grid["x"][0], grid["y"][0]
    for ax, key, title in zip(
        axes,
        ("prediction", "error_proxy"),
        (tr('IDW · оценка значения'), tr('Локальная ошибка по LOO · не ДИ')),
    ):
        data = np.array(
            [[np.nan if v is None else v for v in row] for row in grid[key]]
        )
        ranges[key] = finite_surface_range(data)
        artist = ax.pcolormesh(
            np.array(grid["x"]) - origin_x,
            np.array(grid["y"]) - origin_y,
            np.ma.masked_invalid(data),
            shading="nearest",
            cmap="viridis",
        )
        if result.get('region_outline'):
            from shapely.affinity import translate
            outline = translate(shapely.geometry.shape(result['region_outline']), -origin_x, -origin_y)
            first = len(ax.patches)
            geometry_artists(ax, outline, dict(fill='none', color='#555555', alpha=1, line_width=.8, zorder=3))
            paths = [patch.get_path() for patch in list(ax.patches)[first:]]
            artist.set_clip_path(MplPath.make_compound_path(*paths), ax.transData)
        ax.scatter(
            [p["x"] - origin_x for p in result["locations"]],
            [p["y"] - origin_y for p in result["locations"]],
            s=8,
            c="black",
        )
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("X − X₀, m")
        ax.set_ylabel("Y − Y₀, m")
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        if ranges[key] is not None:
            sheet.figure.colorbar(artist, ax=ax, label=unit_label)
        else:
            label = (tr('На сетке нет прогноза для этого метода') if key == 'prediction'
                     else tr('На сетке нет локальной оценки ошибки LOO'))
            ax.text(.5, .5, label, transform=ax.transAxes, ha='center', va='center',
                    wrap=True, bbox=dict(facecolor='white', alpha=.9))
    sheet.figure.suptitle(
        f"IDW · {column['name']} · {crs_label(sheet.snapshot['spec']['target_crs'])}",
        fontsize=11,
    )
    options = result["options"]
    sheet.figure.text(
        0.06,
        0.025,
        tr('X₀={0:.3f} m; Y₀={1:.3f} m. p={2}; k={3}; R={4:g} км.\nОшибка по LOO — эмпирическая оценка; не доверительный интервал.',
           origin_x, origin_y, options['power'], options['neighbors'], options['max_distance'] / 1000),
        fontsize=7,
    )
    sheet.figure.tight_layout(rect=(0, 0.13, 1, 0.91))
    d = result['diagnostics']
    summary = [tr('{0}. Узлов с прогнозом: {1} из {2}.', unit_label, d['grid_valid'], d['grid_total'])]
    if ranges['prediction'] is not None:
        summary[0] += ' ' + tr('Прогноз от {0:.4g} до {1:.4g}.', *ranges['prediction'])
    summary.append(tr('LOO: прогноз в {0} из {1} допустимых локаций ({2} всего). Вне обучающей оболочки: {3}.',
                      d['loo_valid'], d['loo_eligible'], d['locations'], d['loo_outside_training_hull']))
    if d['loo_valid']:
        summary[-1] += ' ' + tr('RMSE {0:.4g}, MAE {1:.4g}, смещение {2:.4g}.', d['rmse'], d['mae'], d['bias'])
    if ranges['error_proxy'] is not None:
        summary.append(tr('Локальная оценка ошибки LOO от {0:.4g} до {1:.4g}. Это эмпирическая оценка, не доверительный интервал.',
                          *ranges['error_proxy']))
    else:
        summary.append(tr('На сетке нет локальной оценки ошибки LOO'))
    summary[-1] += ' ' + tr('Для пространственно независимой оценки используйте блочную проверку.')
    sheet.idw_summary.setText('\n'.join(summary))
    sheet.set_idw_preview_visible(True)
    sheet.canvas.draw()
    sheet.export_button.setEnabled(True)
    sheet.update_surface_actions()
    sheet.set_interpolation_status(tr('Расчёт IDW и проверка LOO завершены.'))
