"""Import local spatial layers and export the current map."""

from __future__ import annotations

from copy import deepcopy

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMessageBox,
    QVBoxLayout,
)
from shiboken6 import isValid

from autopyro.spatial.mapping import export_map, list_layers, read_layer

from ..i18n import tr, message
from ..tasks import run_with_progress
from ..widgets.spatial import configure_crs_combo, CRSCombo


def import_layer(sheet):
    path, _ = QFileDialog.getOpenFileName(
        sheet,
        tr('Локальный пространственный слой'),
        "",
        tr('GeoPackage / GeoJSON / Shapefile / FlatGeobuf (*.gpkg *.geojson *.json *.shp *.fgb);;Все файлы (*)'),
    )
    if not path:
        return
    try:
        layers = run_with_progress(sheet, tr('Чтение файла'), lambda: list_layers(path))
        if not layers:
            raise ValueError(tr('В файле нет пространственных слоёв'))
        layer, ok = QInputDialog.getItem(
            sheet,
            tr('Слой'),
            tr('Выберите слой'),
            [name for name, kind in layers],
            editable=False,
        )
        if not ok:
            return
        try:
            imported = run_with_progress(sheet, tr('Импорт слоя…'), lambda: read_layer(path, layer))
        except ValueError as exc:
            if ('CRS слоя неизвестна' not in str(exc)
                    and str(exc) != message('CRS слоя неизвестна; задайте её явно')):
                raise
            chooser = QDialog(sheet)
            chooser.setWindowTitle(tr('Неизвестная CRS'))
            chooser.resize(650, 150)
            chooser_layout = QVBoxLayout(chooser)
            chooser_layout.addWidget(QLabel(tr('Выберите CRS исходного слоя')))
            selector = CRSCombo()
            configure_crs_combo(selector, 'EPSG:4326')
            selector.removeItem(selector.findData('auto'))
            chooser_layout.addWidget(selector)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            buttons.accepted.connect(chooser.accept)
            buttons.rejected.connect(chooser.reject)
            chooser_layout.addWidget(buttons)
            if chooser.exec() != QDialog.DialogCode.Accepted:
                return
            declared = selector.currentData() if selector.currentText() == selector.itemText(selector.currentIndex()) else selector.currentText()
            imported = run_with_progress(sheet, tr('Импорт слоя…'), lambda: read_layer(path, layer, declared))
        sheet.restore_imported_layer_id(imported)
        sheet.layers.append(imported)
        sheet.owner.model.project.view["map_layers"] = deepcopy(sheet.layers)
        sheet.owner.model.project.event(
            "map_layer_import", layer_id=imported["id"], source=imported["source"]
        )
        sheet.owner.model_changed()
        sheet.sync_layers()
        sheet.attribute_layer.setCurrentIndex(sheet.attribute_layer.findData(imported['id']))
        sheet.schedule()
    except Exception as exc:
        QMessageBox.warning(sheet, tr('Карта'), message(exc))


def export(sheet):
    snapshot = sheet.report
    if not snapshot or sheet.map_export_running:
        return
    geojson = (tr('GeoJSON WGS84 — видимые векторные объекты с атрибутами')
               if snapshot['spec']['location_mode'] == 'layers'
               else tr('GeoJSON WGS84 — пробы со всеми полями и площадки'))
    filters = ';;'.join((
        f"{tr('PNG — изображение карты')} (*.png)",
        f"{tr('SVG — векторный рисунок карты')} (*.svg)",
        f"{tr('PDF — карта для просмотра и печати')} (*.pdf)",
        f"{tr('TIFF — изображение без геопривязки')} (*.tif *.tiff)",
        f"{tr('JSON — данные выборки, все слои и настройки')} (*.json)",
        f'{geojson} (*.geojson)',
    ))
    path, _ = QFileDialog.getSaveFileName(
        sheet,
        tr('Экспорт карты'),
        "map.png",
        filters,
    )
    if path:
        if sheet.report is not snapshot:
            QMessageBox.warning(sheet, tr('Экспорт карты'), tr('Карта изменилась. Откройте экспорт ещё раз.'))
            return
        ax = sheet.result['axes']
        spec = dict(snapshot['spec'])
        spec.update(zip(('xmin', 'xmax', 'ymin', 'ymax'), [*ax.get_xlim(), *ax.get_ylim()]))
        report = dict(snapshot, spec=spec)
        sheet.map_export_running = True
        sheet.export_button.setEnabled(False)
        try:
            run_with_progress(sheet.owner, tr('Экспорт карты'), lambda: export_map(path, report), modal=False)
        except Exception as exc:
            if isValid(sheet) and not sheet.closed:
                QMessageBox.warning(sheet.owner, tr('Экспорт карты'), message(exc))
        finally:
            if isValid(sheet):
                sheet.map_export_running = False
                sheet.export_button.setEnabled(bool(sheet.report) and not sheet.closed)
                sheet.project_tasks_finished.emit()
