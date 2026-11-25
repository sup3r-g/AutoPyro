"""Export interpolation results and publish surfaces as map layers."""

from __future__ import annotations

from pathlib import Path
import json

from PySide6.QtWidgets import QFileDialog, QMessageBox
import numpy as np
import pandas as pd

from autopyro.data_io.io import export_frame_csv
from autopyro.spatial.surfaces import DIFFERENCE_METHOD, export_geotiff, surface_layers

from ..i18n import tr, message


def active_surface(sheet):
    if sheet.views.currentIndex() == 0 and sheet.result is not None:
        result = sheet.result
        column = next(c for c in result['map_snapshot']['columns'] if c['id'] == result['options']['value_id'])
        return result, 'IDW', result['map_snapshot']['spec']['target_crs'], column
    if sheet.views.currentWidget() is sheet.kriging_tab and sheet.kriging_result and 'prediction' in sheet.kriging_result:
        method = sheet.comparison_surface.currentData()
        if method == 'idw-tin-difference':
            if sheet.grid_inspection is None:
                return None
            grid, comparison, _, _ = sheet.grid_inspection
            result = dict(grid, **comparison)
            result['options'] = {'value_id': grid['options']['value_id'], 'IDW': grid['options'],
                                 'TIN': sheet.kriging_result['tin_surface']['options']}
            method = DIFFERENCE_METHOD
        else:
            result = sheet.comparison_grid(sheet.kriging_result, method)
        column = next(c for c in sheet.snapshot['columns'] if c['id'] == result['options']['value_id'])
        return result, method, sheet.snapshot['spec']['target_crs'], column
    return None


def update_surface_actions(sheet, *_):
    idw = sheet.views.currentIndex() == 0
    sheet.export_button.setVisible(idw)
    sheet.export_button.setEnabled(idw and sheet.result is not None)
    surface_tab = idw or sheet.views.currentWidget() is sheet.kriging_tab
    sheet.save_surface.setVisible(surface_tab)
    sheet.save_tiff.setVisible(surface_tab)
    sheet.layout_interpolation_actions()
    active = sheet.active_surface() is not None
    sheet.save_surface.setEnabled(active and sheet.map_dialog is not None)
    sheet.save_tiff.setEnabled(active)


def add_surface_to_map(sheet):
    active = sheet.active_surface()
    if not active or sheet.map_dialog is None:
        return
    result, method, source_crs, column = active
    label = ('TIN − IDW' if method == DIFFERENCE_METHOD
             else sheet.kriging_method_label(method, (result.get('model') or {}).get('range_ratio', 1) < 1))
    name = f"{label} · {column['name']}"
    labels = {'prediction': tr('Прогноз'), 'neighbors': tr('Число соседей'),
              'nearest_distance_m': tr('Расстояние до ближайшей точки'),
              'difference': tr('Разность прогнозов'), 'coverage': tr('Маска покрытия'),
              'coverage_categories': [tr('Оба без прогноза'), tr('Только IDW'), tr('Только TIN'), tr('Оба метода')]}
    layers = surface_layers(result, method, source_crs, name, labels, column['unit'])
    if not layers:
        sheet.set_interpolation_status(tr('В сетке нет данных для сохранения.'))
        return
    sheet.map_dialog.add_interpolation_layers(layers)
    sheet.set_interpolation_status(tr('Растровые слои добавлены на карту.'))


def export_surface_tiff(sheet):
    active = sheet.active_surface()
    if not active:
        return
    filename = 'tin-minus-idw.tif' if active[1] == DIFFERENCE_METHOD else 'surface.tif'
    path, _ = QFileDialog.getSaveFileName(sheet, tr('Экспорт GeoTIFF…'), filename, 'GeoTIFF (*.tif *.tiff)')
    if path:
        if not Path(path).suffix:
            path += '.tif'
        result, method, source_crs, column = active
        sheet.run_interpolation_task(lambda: export_geotiff(path, result, method, source_crs, column['unit']),
                      lambda _: sheet.set_interpolation_status(tr('GeoTIFF сохранён: {0}', path)))


def export(sheet):
    if not sheet.result:
        return
    path, _ = QFileDialog.getSaveFileName(
        sheet,
        tr('Результат IDW'),
        "spatial-idw.json",
        tr('JSON (*.json);;CSV сетки (*.csv);;PNG (*.png);;SVG (*.svg)'),
    )
    if not path:
        return
    try:
        suffix = Path(path).suffix.lower()
        if suffix == ".json":
            Path(path).write_text(
                json.dumps(
                    sheet.result, ensure_ascii=False, allow_nan=False, indent=2
                ),
                encoding="utf-8",
            )
        elif suffix == ".csv":
            grid = sheet.result["grid"]
            coordinates = pd.MultiIndex.from_product(
                [grid["y"], grid["x"]], names=["Y_m", "X_m"],
            ).to_frame(index=False)
            coordinates["CRS"] = sheet.snapshot["spec"]["target_crs"]
            coordinates["prediction"] = np.asarray(grid["prediction"], dtype=object).ravel()
            coordinates["LOO_error_proxy_NOT_CI"] = np.asarray(grid["error_proxy"], dtype=object).ravel()
            headers = ["CRS", "X_m", "Y_m", "prediction", "LOO_error_proxy_NOT_CI"]
            export_frame_csv(path, coordinates[headers])
        elif suffix in (".png", ".svg"):
            sheet.figure.savefig(path, dpi=300)
        else:
            raise ValueError(tr('Выберите JSON, CSV, PNG или SVG'))
        sheet.set_interpolation_status(
            tr('Сохранено: {0}. JSON содержит диагностику, LOO и входной снимок.', path)
        )
    except Exception as exc:
        QMessageBox.warning(sheet, "IDW", message(exc))
