"""Prepare the interpolation sample and show preflight diagnostics."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QLabel, QTableView, QVBoxLayout, QWidget

from autopyro.spatial.map_tools import filtered_features
from autopyro.spatial.mapping import MapSpec, crs_label, build_map
from autopyro.spatial.spatial_interpolation import SpatialOptions, preview_sample_support
from autopyro.visualization.plotting import category_key

from ..i18n import tr, message
from ..tasks import run_with_progress
from ..widgets.results import ResultsModel


def build_sample_preview(sheet):
    content = QWidget()
    layout = QVBoxLayout(content)
    sheet.sample_preview_note = QLabel()
    sheet.sample_preview_note.setWordWrap(True)
    layout.addWidget(sheet.sample_preview_note)
    sheet.sample_preview_table = QTableView()
    sheet.sample_preview_table.setAccessibleName(tr('Результаты проверки выборки'))
    sheet.sample_preview_table.setTabKeyNavigation(False)
    sheet.sample_preview_table.horizontalHeader().setStretchLastSection(True)
    layout.addWidget(sheet.sample_preview_table, 1)
    sheet.clear_sample_preview()
    sheet.sample_preview_tab = sheet.add_interpolation_tab(content, tr('Проверка выборки'), 'shield-check')


def clear_sample_preview(sheet):
    sheet.sample_preview_note.setText(tr('Выберите показатель и нажмите «Проверить выборку».'))
    old = sheet.sample_preview_table.model()
    sheet.sample_preview_table.setModel(None)
    if old is not None:
        old.deleteLater()


def check_sample(sheet):
    sheet.clear_sample_preview()
    sheet.views.setCurrentWidget(sheet.sample_preview_tab)
    try:
        snapshot, options, size = sheet.calculation_input()
        generation = sheet.generation
        sheet.set_interpolation_status(tr('Проверяется выборка…'))
        sheet.run_interpolation_task(lambda: preview_sample_support(snapshot, options, size),
                      lambda result: sheet.show_sample_preview(result, options) if generation == sheet.generation else None)
    except Exception as exc:
        sheet.set_interpolation_status(tr('Расчёт не выполнен: {0}', message(exc)))


def show_sample_preview(sheet, result, options):
    labels = {'missing': tr('Пропущенные значения'), 'non_detect': tr('Не обнаружено'),
              'censored': tr('Значения с пределом измерения'), 'other': tr('Нечисловые значения')}
    rows = [
        [tr('CRS карты'), crs_label(result['target_crs'])],
        [tr('Пробы в выбранной выборке'), result['input_rows']],
        [tr('Исключено фильтрами карты'), result['map_filtered']],
        [tr('Пробы с координатами'), result['mapped']],
        [tr('Пробы с числовым значением'), result['eligible']],
        [tr('Уникальные локации скважин'), result['locations']],
        [tr('Локации с повторными пробами'), result['duplicate_locations']],
        [tr('Дополнительные пробы в этих локациях'), result['duplicate_samples']],
        [tr('Исключено при привязке координат'), sum(result['map_excluded'].values())],
        [tr('Исключено по значению показателя'), sum(result['excluded'].values())],
        *[[labels.get(key, key), count] for key, count in result['excluded'].items()],
        [tr('Размер блока, км'), result['block_size_km']],
        [tr('Буфер проверки, км'), result['buffer_km']],
        [tr('Занятые блоки'), result['blocks']],
        [tr('Радиус поиска, км'), result['radius_m'] / 1000],
        [tr('Минимум соседей'), result['min_neighbors']],
    ]
    if result['inside_region'] is not None:
        rows.append([tr('Локации внутри выбранной области'), result['inside_region']])
        if options.region_mode == 'within':
            rows.append([tr('Локации на границе или в перекрытии областей'), result['unassigned_region']])
    if result['supported'] is not None:
        rows.extend([
            [tr('Достаточно соседей вне своего блока'), tr('{0} из {1}', result['supported'], result['candidates'])],
            [tr('Также внутри оболочки обучающих локаций'), tr('{0} из {1}', result['with_hull'], result['candidates'])],
        ])
    notes = [tr('Соседи проверяются для IDW на блоках без сдвига. Из обучения исключается весь блок проверяемой локации.')]
    notes.append(tr('Для TIN нужна обучающая оболочка из трёх неколлинеарных локаций. '
                    'Пакетная проверка оценивает каждый размер и сдвиг отдельно, а подбор требует общих внутренних прогнозов.'))
    if result['buffer_km']:
        notes.append(tr('Из обучения также исключаются локации в буфере вокруг блока.'))
    if not result['metric']:
        notes.append(tr('Для интерполяции выберите проецированную CRS с единицами в метрах'))
    if result['issue']:
        notes.append(message(result['issue']))
    if result['duplicate_locations'] and options.duplicate_policy == 'reject':
        notes.append(tr('Повторы блокируют расчёт. Выберите среднее или медиану по месту либо измените выборку.'))
    if result['candidates'] == 0:
        notes.append(tr('В выбранной области нет локаций, доступных для проверки.'))
    elif result['supported'] == 0:
        notes.append(tr('При текущем радиусе ни у одной проверяемой локации не хватает соседей вне своего блока.'))
    elif result['with_hull'] == 0:
        notes.append(tr('Локации с достаточным числом соседей лежат вне обучающих оболочек.'))
    sheet.sample_preview_note.setText('\n'.join(notes))
    sheet.sample_preview_table.setModel(ResultsModel(['Параметр', 'Значение'], rows, sheet.sample_preview_table))
    sheet.sample_preview_table.resizeColumnsToContents()
    sheet.sample_preview_table.setCurrentIndex(sheet.sample_preview_table.model().index(0, 0))
    sheet.sample_preview_table.setFocus()
    sheet.set_interpolation_status(tr('Предварительная проверка выборки завершена.'))


def selected_region_layer(sheet):
    return next((layer for layer in sheet.snapshot['layers'] if layer['id'] == sheet.region_layer.currentData()), None)


def update_region_fields(sheet):
    layer = sheet.selected_region_layer()
    fields = sorted({key for feature in layer['features'] for key in feature['properties']}) if layer else []
    with QSignalBlocker(sheet.region_field):
        sheet.region_field.clear()
        for field in fields:
            sheet.region_field.addItem(field, field)
        sheet.region_field.setCurrentIndex(max(0, sheet.region_field.findData('Paleogeogr')))
    sheet.update_region_keys()


def update_region_keys(sheet):
    layer = sheet.selected_region_layer()
    field = sheet.region_field.currentData()
    values = {}
    if layer:
        for feature in filtered_features(layer):
            value = feature['properties'].get(field)
            if value is not None and str(value).strip():
                values[category_key(value)] = str(value)
    with QSignalBlocker(sheet.region_key):
        sheet.region_key.clear()
        sheet.region_key.addItem(tr('Все области'), '')
        for key, label in sorted(values.items(), key=lambda item: item[1]):
            sheet.region_key.addItem(label, key)


def update_region_controls(sheet):
    enabled = sheet.region_mode.currentData() != 'none'
    for widget in (sheet.region_layer, sheet.region_field, sheet.region_key):
        widget.setEnabled(enabled)


def calculation_input(sheet):
    sheet.input_summary.setText(tr('Выборка для проверки или расчёта ещё не подготовлена.'))
    values = sheet.form.values()
    if not values['value_id']:
        sheet.value_selector.setFocus()
        raise ValueError(tr('Выберите показатель интерполяции'))
    values.pop('scope', None)
    size = values.pop('block_size_km')
    if sheet.owner is not None:
        spec = MapSpec.from_dict(sheet.snapshot['spec'])
        spec.scope = sheet.owner.analysis_scope.mode
        model = sheet.owner.model
        project, revision = model.project, model.data_revision
        row_ids, layers = sheet.owner.analysis_scope.rows(), sheet.snapshot['layers']
        prepared = run_with_progress(
            sheet, tr('Подготовка выборки'), lambda: build_map(project, row_ids, spec, layers))
        if model.project is not project or model.data_revision != revision:
            raise ValueError(tr('Данные изменились. Повторите подготовку выборки.'))
        sheet.snapshot = prepared
    rows = len(sheet.snapshot['row_ids'])
    sheet.input_summary.setText(tr('Последняя подготовленная выборка — {0}. Проб в области: {1}; после фильтров карты: {2}; с координатами: {3}.',
                                  sheet.scope.currentText(), sheet.snapshot.get('diagnostics', {}).get('input_rows', rows),
                                  rows, len(sheet.snapshot['points'])))
    return sheet.snapshot, SpatialOptions(**values), size
