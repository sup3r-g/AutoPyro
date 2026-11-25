"""Display and export regional statistics and synchronize region selection with the map."""

import json

from PySide6.QtCore import QItemSelection, QItemSelectionModel, QSignalBlocker
from PySide6.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem

from autopyro.data_io.io import export_table_csv
from autopyro.spatial.regional_comparison import region_fingerprint
from autopyro.spatial.regional_geochemistry import render_distributions, export_regional_summary
from autopyro.visualization.plotting import category_key

from ..i18n import tr, message


def show_result(sheet, result):
    sheet.result = result
    sheet.comparison.capture.setEnabled(True)
    fields = ['region', 'metric', 'unit', 'method', 'n_total', 'n_valid', 'n_wells', 'mean', 'median', 'std', 'min', 'max', 'p10', 'q1', 'q3', 'p90', 'excluded']
    labels = ['Область', 'Показатель', 'Единицы', 'Метод', 'Всего', 'Чисел', 'Скважин', 'Среднее', 'Медиана', 'σ', 'Мин.', 'Макс.', 'P10', 'Q1', 'Q3', 'P90', 'Исключения']
    if result['options'].get('by_well'):
        fields.insert(7, 'n_observations')
        labels.insert(7, 'Медиан скважин')
    sheet.table.setColumnCount(len(fields))
    sheet.table.setHorizontalHeaderLabels([tr(s) for s in labels])
    sheet.table.setRowCount(len(result['summary']))
    exclusions = {'missing': tr('Пропуски'), 'non_detect': tr('Не обнаружено'), 'censored': tr('За пределом измерения'),
        'other': tr('Текст или неконечное число'), 'qc_excluded': tr('Исключено QC'),
        'stale_excluded': tr('Устаревшие значения'), 'interpolation_excluded': tr('Интерполяция исключена')}
    for i, row in enumerate(result['summary']):
        for j, key in enumerate(fields):
            value = row[key]
            if key == 'method':
                value = tr('Все методы') if row['pooled'] else value
                if isinstance(value, str) and value != value.strip():
                    value = repr(value)
            elif key == 'excluded':
                value = '; '.join(f'{exclusions.get(k, k)}: {n}' for k, n in value.items()) or '—'
            text = '—' if value is None else f'{value:.6g}' if isinstance(value, float) else str(value)
            sheet.table.setItem(i, j, QTableWidgetItem(text))
    sheet.table.resizeColumnsToContents()
    d = result['diagnostics']
    reasons = {'age_filter': 'Возраст', 'depth_filter': 'Глубина', 'invalid_depth': 'Глубина отсутствует или некорректна',
        'matched': 'Однозначно внутри области', 'boundary': 'На границе', 'overlap': 'В перекрытии областей',
        'outside': 'Вне областей', 'no_location': 'Без координат', 'missing_region': 'Не задана область'}
    lines = [f"{tr('Строк в выборке карты')}: {d['scope_rows']}", f"{tr('После фильтров')}: {d['filtered_rows']}", f"{tr('Привязано к областям')}: {d['assigned_rows']}"]
    for counts in (d['filters'], d['spatial']):
        lines.extend(f'{tr(reasons.get(k, k))}: {v}' for k, v in counts.items())
    lines.append(tr('Исключено фильтром метода') + ': ' + str(d['method_filters']))
    sheet.diagnostics.setText('\n\n'.join(lines))
    for button in sheet.exports:
        button.setEnabled(True)
    sheet.refresh_spatial()
    sheet.plot()


def select_on_map(sheet, refresh_details=True):
    if refresh_details:
        sheet.refresh_samples()
    indexes = {index.row() for index in sheet.table.selectionModel().selectedRows()}
    if not sheet.result or not indexes:
        sheet.map_selection_changed.emit(None)
        return
    rows = [sheet.result['summary'][i] for i in sorted(indexes)]
    sheet.map_selection_changed.emit(dict(layer_id=sheet.result['options']['layer_id'],
        region_field=sheet.result['options']['region_field'], region_keys={r['region_key'] for r in rows},
        row_ids={rid for r in rows for rid in r['row_ids']}))


def select_map_region(sheet, layer_id, feature_id):
    if sheet.tabs.currentWidget() is sheet.comparison and sheet.comparison.paired.isChecked():
        comparison = sheet.comparison.paired_map()
        if sheet.report and comparison and layer_id == comparison['layer_id']:
            layer = next((item for item in sheet.report['layers'] if item['id'] == layer_id), None)
            field = comparison['field']
            if layer and region_fingerprint(layer, field) == comparison['fingerprint']:
                feature = next((f for f in layer['features'] if str(f['id']) == str(feature_id)), None)
                if feature:
                    sheet.comparison.select_region(category_key(feature['properties'].get(field)))
        return
    if not sheet.result or not sheet.report or layer_id != sheet.result['options']['layer_id']:
        return
    layer = next((l for l in sheet.report['layers'] if l['id'] == layer_id), None)
    feature = next((f for f in layer['features'] if str(f['id']) == str(feature_id)), None) if layer else None
    if feature is None:
        return
    field = sheet.result['options']['region_field']
    region_key = category_key(feature['properties'].get(field))
    rows = [i for i, row in enumerate(sheet.result['summary']) if row['region_key'] == region_key]
    selection = QItemSelection()
    for row in rows:
        selection.select(sheet.table.model().index(row, 0), sheet.table.model().index(row, sheet.table.columnCount() - 1))
    # Emit the map update once after selecting every metric/method of this region.
    with QSignalBlocker(sheet.table):
        sheet.table.selectionModel().select(selection, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        if rows:
            index = sheet.table.model().index(rows[0], 0)
            sheet.table.selectionModel().setCurrentIndex(index, QItemSelectionModel.SelectionFlag.NoUpdate)
            sheet.table.scrollTo(index)
            sheet.tabs.setCurrentWidget(sheet.table)
    sheet.select_on_map()


def plot(sheet, *_):
    if sheet.result:
        render_distributions(sheet.result, sheet.metric_combo.currentText(), sheet.canvas.figure, tr)
        w, h = sheet.canvas.figure.get_size_inches() * sheet.canvas.figure.dpi
        sheet.canvas.resize(int(w), int(h))
        sheet.canvas.draw_idle()


def export(sheet, kind):
    if not sheet.result:
        return
    filters = {'summary': 'CSV (*.csv)', 'values': 'CSV (*.csv)', 'json': 'JSON (*.json)', 'figure': 'PNG (*.png);;SVG (*.svg);;PDF (*.pdf)'}
    path, _ = QFileDialog.getSaveFileName(sheet, tr('Экспорт'), '', filters[kind])
    if not path:
        return
    try:
        if kind == 'summary':
            export_regional_summary(path, sheet.result)
        elif kind == 'values':
            rows = []
            for entry in sheet.result['distributions']:
                summary = sheet.result['summary'][entry['summary_index']]
                values = entry.get('samples', entry['values'])
                rows.extend([summary['region'], summary['metric'], summary['unit'], v.get('method', summary['method']), v['row_id'], v['well'], v['value']]
                            for v in values if v['value'] is not None)
            export_table_csv(path, ['region', 'metric', 'unit', 'method', 'row_id', 'well', 'value'], rows)
        elif kind == 'json':
            with open(path, 'w', encoding='utf-8') as stream:
                json.dump(sheet.result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        else:
            sheet.canvas.figure.savefig(path, bbox_inches='tight', dpi=180)
    except Exception as exc:
        QMessageBox.warning(sheet, tr('Ошибка'), message(exc))
