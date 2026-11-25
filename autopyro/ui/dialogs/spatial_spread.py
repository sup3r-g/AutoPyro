"""Compute and inspect prediction spread under spatial block exclusion."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAccessible, QAccessibleAnnouncementEvent
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFileDialog, QLabel, QMessageBox, QVBoxLayout
from matplotlib.figure import Figure
import numpy as np

from autopyro.spatial.kriging import KrigingOptions
from autopyro.spatial.spatial_interpolation import partition_spread
from autopyro.spatial.surfaces import (
    KRIGING_SPREAD_METHOD,
    SPREAD_METHOD,
    TIN_SPREAD_METHOD,
    export_geotiff,
    finite_surface_range,
    surface_layers,
)

from ..i18n import tr, message
from ..tasks import TaskCancelled, run_with_progress
from ..widgets.figure_canvas import FittedFigureCanvas
from ..theme.controls import add_command_button


def calculate_spread(sheet):
    try:
        snapshot, options, size = sheet.calculation_input()
        generation = sheet.generation
        method = sheet.spread_method.currentData()
        shifted = sheet.spread_shift.isChecked()
        title = tr('Карта разброса {0}', tr('кригинга') if method == 'kriging' else method)
        model = None
        if method == 'kriging':
            settings = sheet.kriging_form.values()
            settings.pop('rbf_smoothing')
            settings.pop('directions')
            model = KrigingOptions(**settings)

        def calculate(update):
            def progress(index, total):
                update(tr('Исключается блок {0} из {1}…', index + 1, total), 100 * index // total)
            return partition_spread(snapshot, options, size, method=method, shifted=shifted,
                                    kriging_model=model, progress=progress)

        result = run_with_progress(sheet, title, calculate, cancellable=True)
        if generation == sheet.generation:
            sheet.show_spread(result, snapshot, options, size)
    except TaskCancelled:
        sheet.set_interpolation_status(tr('Расчёт разброса отменён.'))
    except Exception as exc:
        sheet.set_interpolation_status(tr('Расчёт не выполнен: {0}', message(exc)))


def show_spread(sheet, result, snapshot, options, size):
    dialog = QDialog(sheet)
    method = result['method']
    export_method = {'IDW': SPREAD_METHOD, 'TIN': TIN_SPREAD_METHOD, 'kriging': KRIGING_SPREAD_METHOD}[method]
    method_label = tr('Кригинг') if method == 'kriging' else method
    name_for_title = tr('кригинга') if method == 'kriging' else method
    spread_label = tr('Разброс прогнозов {0}', name_for_title)
    dialog.setWindowTitle(tr('Карта разброса {0}', name_for_title))
    dialog.resize(QSize(950, 630).boundedTo(sheet.screen().availableGeometry().size() - QSize(24, 48)))
    layout = QVBoxLayout(dialog)
    count = result['count']
    text = tr('Блоков {0}, размер {1:g} км, буфер {2:g} км. Сдвиг по X и Y {3:g} км.',
              result['blocks'], size, options.validation_buffer_km, result['options']['block_shift_km'])
    text += '\n' + tr('В области {0} узлов: без прогноза {1}, с одним прогнозом {2}, с двумя или более {3}.',
                      int(np.isfinite(count).sum()), int((count == 0).sum()),
                      int((count == 1).sum()), int((count >= 2).sum()))
    text += '\n' + tr('Разброс — максимум минус минимум среди доступных прогнозов. Для оценки нужны хотя бы два прогноза.')
    if method == 'TIN':
        text += '\n' + tr('Для TIN нужны как минимум три неколлинеарные обучающие локации.')
    if method == 'kriging':
        model = result['model']
        family = tr('Сферическая') if model['model'] == 'spherical' else tr('Экспоненциальная')
        text += '\n' + tr('Вариограмма: {0}, дальность {1:g} км, порог {2:g}, наггет {3:g}, азимут {4:g}°, отношение дальностей {5:g}.',
                          family, model['range_km'], model['sill'], model['nugget'],
                          model['azimuth_deg'], model['range_ratio'])
    bounds = finite_surface_range(result['spread'])
    if bounds is not None:
        text += ' ' + tr('Разброс от {0:.4g} до {1:.4g}.', *bounds)
    note = QLabel(text)
    note.setWordWrap(True)
    note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByKeyboard)
    note.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    layout.addWidget(note)
    fields = sheet.add_grid_index_fields(layout)
    for field, length in zip(fields, count.shape):
        field.setRange(1, length)
        field.setValue((length + 1) // 2)
    node_text = QLabel()
    node_text.setAccessibleDescription(tr('Значения в выбранном узле карты разброса'))
    node_text.setWordWrap(True)
    node_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByKeyboard)
    node_text.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    layout.addWidget(node_text)
    figure = Figure(layout='constrained')
    canvas = FittedFigureCanvas(figure)
    layout.addWidget(canvas, 1)
    axes = figure.subplots(1, 2, sharex=True, sharey=True, subplot_kw={'aspect': 'equal'})
    variable = next(column for column in snapshot['columns'] if column['id'] == options.value_id)
    unit = variable['unit']
    for ax, key, title, label in zip(axes, ('spread', 'count'), (spread_label, tr('Доступные прогнозы')),
                                     (unit, tr('Число моделей'))):
        values = result[key]
        artist = sheet.draw_surface_grid(ax, result, values, cmap='viridis',
                                        vmin=0, vmax=result['blocks'] if key == 'count' else None)
        ax.set_title(title)
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.scatter([point['x'] for point in result['locations']], [point['y'] for point in result['locations']],
                   s=7, color='black', zorder=5)
        if np.isfinite(values).any():
            figure.colorbar(artist, ax=ax, location='bottom', label=label)
    markers = [ax.plot([], [], marker='s', markersize=8, markerfacecolor='none',
                       markeredgecolor='#242424', markeredgewidth=1.6, zorder=12)[0] for ax in axes]

    def show_node(*_, announce=True):
        row, column = (field.value() - 1 for field in fields)
        x, y = result['x'][column], result['y'][row]
        for marker in markers:
            marker.set_data([x], [y])
        detail = tr('X {0:.2f} м, Y {1:.2f} м.', x, y)
        models = count[row, column]
        spread = result['spread'][row, column]
        if np.isfinite(models):
            detail += ' ' + tr('Прогноз дали {0} из {1} моделей.', int(models), result['blocks'])
            if np.isfinite(spread):
                detail += ' ' + tr('Разброс в узле {0:.4g}.', spread)
                if unit:
                    detail += ' ' + tr('Единицы: {0}.', unit)
            else:
                detail += ' ' + tr('Для разброса нужны хотя бы два прогноза.')
        else:
            detail += ' ' + tr('Узел вне расчётной области.')
        changed = detail != node_text.text()
        node_text.setText(detail)
        if announce and changed and QAccessible.isActive():
            QAccessible.updateAccessibility(QAccessibleAnnouncementEvent(node_text, detail))
        canvas.draw_idle()

    def pick_node(event):
        if sheet.pick_grid_indices(event, result, markers, fields):
            show_node()

    for field in fields:
        field.valueChanged.connect(show_node)
    canvas.mpl_connect('button_press_event', pick_node)
    show_node(announce=False)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    source_crs = snapshot['spec']['target_crs']
    can_save = bool(np.isfinite(count).any())

    def save_layers():
        labels = {'spread': spread_label, 'count': tr('Число моделей')}
        sheet.map_dialog.add_interpolation_layers(
            surface_layers(result, export_method, source_crs, f"{method_label} · {variable['name']}", labels, unit))
        sheet.set_interpolation_status(tr('Растровые слои добавлены на карту.'))

    def save_tiff():
        title = tr('Экспорт GeoTIFF…')
        path, _ = QFileDialog.getSaveFileName(dialog, title, f'{method.lower()}-spread.tif', 'GeoTIFF (*.tif *.tiff)')
        if not path:
            return
        if not Path(path).suffix:
            path += '.tif'
        try:
            run_with_progress(dialog, title, lambda: export_geotiff(path, result, export_method, source_crs, unit))
        except (ValueError, OSError) as exc:
            QMessageBox.warning(dialog, tr('Ошибка'), message(exc))
        else:
            sheet.set_interpolation_status(tr('GeoTIFF сохранён: {0}', path))

    add = add_command_button(buttons, 'map', tr('Добавить слои на карту'), QDialogButtonBox.ButtonRole.ActionRole)
    add.setEnabled(can_save and sheet.map_dialog is not None)
    add.clicked.connect(save_layers)
    export = add_command_button(buttons, 'log-out', tr('Экспорт GeoTIFF…'), QDialogButtonBox.ButtonRole.ActionRole)
    export.setEnabled(can_save)
    export.clicked.connect(save_tiff)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.exec()
    dialog.deleteLater()
