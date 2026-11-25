"""Compare IDW settings and nested validation results."""

from ..i18n import tr, message
import json
from itertools import product
from math import prod
from pathlib import Path
import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)
from ..tasks import TaskCancelled, run_with_progress
from autopyro.spatial.spatial_interpolation import compare_idw_settings, nested_idw_validation
from autopyro.spatial.kriging import KrigingOptions, compare_nested_methods
from ..widgets.results import ResultsModel
from ..theme import icon


class SpatialSettingsComparison(QWidget):
    invalidated = Signal()

    def __init__(self, dialog):
        super().__init__(dialog)
        self.dialog = dialog
        self.settings_result = None
        self.nested_result = None
        layout = QVBoxLayout(self)
        note = QLabel(tr('Укажите варианты через точку с запятой. Пустое поле использует текущую настройку. Размер блоков, минимум соседей и режим областей общие.'))
        note.setWordWrap(True)
        layout.addWidget(note)
        self.variant_fields = {}
        for key, label, default in [('power', tr('Степени IDW'), '1; 2; 3'),
                                    ('max_distance', tr('Радиусы поиска, км'), ''),
                                    ('neighbors', tr('Максимумы соседей'), '')]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            edit = QLineEdit(default)
            edit.setAccessibleName(label)
            edit.textChanged.connect(self.clear_settings_comparison)
            row.addWidget(edit, 1)
            layout.addLayout(row)
            self.variant_fields[key] = edit
        controls = QHBoxLayout()
        controls.addWidget(QLabel(tr('Дальности кригинга, км')))
        self.nested_ranges = QLineEdit('25; 50; 100')
        self.nested_ranges.setAccessibleName(tr('Дальности кригинга, км'))
        self.nested_ranges.textChanged.connect(self.clear_settings_comparison)
        controls.addWidget(self.nested_ranges, 1)
        self.nested_shift = QCheckBox(tr('Сдвиг на полблока по X и Y'))
        self.nested_shift.toggled.connect(self.clear_settings_comparison)
        controls.addWidget(self.nested_shift)
        layout.addLayout(controls)
        self.settings_table = QTableView()
        self.settings_table.setAccessibleName(tr('Результаты сравнения настроек IDW'))
        self.settings_table.setTabKeyNavigation(False)
        self.settings_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.settings_table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        layout.addWidget(self.settings_table, 1)
        self.nested_summary = QTableView()
        self.nested_folds = QTableView()
        self.nested_summary.setAccessibleName(tr('Сводка вложенной проверки'))
        self.nested_folds.setAccessibleName(tr('Внешние блоки вложенной проверки'))
        self.nested_summary.setTabKeyNavigation(False)
        self.nested_folds.setTabKeyNavigation(False)
        self.nested_note = QLabel()
        self.nested_note.setWordWrap(True)
        layout.addWidget(self.nested_note)
        layout.addWidget(self.nested_summary, 1)
        layout.addWidget(self.nested_folds, 2)
        buttons = QHBoxLayout()
        run = QPushButton(icon('files'), tr('Сравнить настройки IDW'))
        run.clicked.connect(self.calculate_settings_comparison)
        buttons.addWidget(run)
        nested = QPushButton(icon('shield-check'), tr('Вложенная проверка IDW'))
        nested.clicked.connect(lambda: self.calculate_settings_comparison(nested=True))
        buttons.addWidget(nested)
        compare = QPushButton(icon('files'), tr('Вложенное сравнение методов'))
        compare.clicked.connect(self.calculate_nested_methods)
        buttons.addWidget(compare)
        buttons.addStretch()
        layout.addLayout(buttons)
        result_actions = QHBoxLayout()
        self.apply_variant = QPushButton(icon('check'), tr('Применить настройки'))
        self.apply_variant.clicked.connect(self.apply_idw_variant)
        result_actions.addWidget(self.apply_variant)
        self.export_variants = QPushButton(icon('save'), tr('Сохранить сравнение JSON…'))
        self.export_variants.clicked.connect(self.export_settings_comparison)
        result_actions.addWidget(self.export_variants)
        result_actions.addStretch()
        layout.addLayout(result_actions)
        self.clear_settings_comparison()


    def clear_settings_comparison(self):
        self.dialog.generation += 1
        self.invalidated.emit()
        self.settings_result = None
        self.nested_result = None
        self.settings_table.show()
        for widget in (self.nested_note, self.nested_summary, self.nested_folds):
            widget.hide()
        self.apply_variant.setEnabled(False)
        self.export_variants.setEnabled(False)
        self.set_settings_rows([])

    def set_settings_rows(self, rows):
        headers = ['Степень IDW', 'Радиус, км', 'Максимум соседей', 'С прогнозом', 'Доля с прогнозом',
                   'RMSE', 'MAE', 'Смещение', 'Общие места', 'RMSE на общих местах',
                   'MAE на общих местах', 'Смещение на общих местах']
        old = self.settings_table.model()
        self.settings_table.setModel(ResultsModel(headers, rows, self.settings_table))
        if old is not None:
            old.deleteLater()
        self.settings_table.resizeColumnsToContents()

    def variant_settings(self, options):
        values = {}
        for key, edit in self.variant_fields.items():
            current = getattr(options, key)
            if not edit.text().strip():
                values[key] = [current]
                continue
            convert = int if key == 'neighbors' else float
            numbers = [convert(value.strip().replace(',', '.')) for value in edit.text().split(';')]
            field = self.dialog.form.fields[key][0]
            if any(not np.isfinite(number) or not field.minimum() <= number <= field.maximum() for number in numbers):
                raise ValueError(tr('Настройки варианта IDW вне допустимого диапазона.'))
            if key != 'neighbors':
                numbers = [round(number, field.decimals()) for number in numbers]
            scale = self.dialog.form.number_scales.get(key, 1)
            values[key] = list(dict.fromkeys(number * scale for number in numbers))
        if not 2 <= prod(map(len, values.values())) <= 24:
            raise ValueError(tr('Для сравнения задайте от 2 до 24 вариантов IDW'))
        return [dict(zip(values, combination)) for combination in product(*values.values())]

    def calculate_settings_comparison(self, *, nested=False):
        try:
            snapshot, options, size = self.dialog.calculation_input()
            variants = self.variant_settings(options)
            self.clear_settings_comparison()
            generation = self.dialog.generation
            calculate = nested_idw_validation if nested else compare_idw_settings
            completed = self.completed_nested if nested else self.completed_settings
            result = self.run_settings_comparison(lambda progress: calculate(
                snapshot, options, size, variants, progress=progress))
            if generation == self.dialog.generation:
                completed(result)
        except TaskCancelled:
            self.dialog.set_interpolation_status(tr('Сравнение методов отменено.'))
        except Exception as exc:
            self.dialog.set_interpolation_status(tr('Расчёт не выполнен: {0}', message(exc)))

    def run_settings_comparison(self, calculate):
        title = tr('Проверка вариантов на одинаковых блоках…')
        self.dialog.set_interpolation_status(title)

        def run(update):
            result = calculate(lambda fraction: update(title, int(100 * fraction)))
            update(title, 100)
            return result

        return run_with_progress(self.dialog, title, run, cancellable=True)

    def completed_nested(self, result):
        self.nested_note.setText(tr('В каждом внешнем блоке проверяются настройки с минимальной MAE на общих внутренних местах. Без общих внутренних прогнозов блок остаётся без подбора.'))
        metrics = ['С прогнозом', 'Доля с прогнозом', 'RMSE', 'MAE', 'Смещение']
        summary = [[label, *self.nested_scores(result[key]), *self.nested_scores(result[f'{key}_common'], coverage=False)]
                   for label, key in [(tr('С подбором'), 'tuned'), (tr('Текущие настройки'), 'baseline')]]
        folds = []
        for block in result['blocks']:
            index = block['selected']
            selected = result['variants'][index] if index is not None else None
            parameters = ([selected['power'], selected['max_distance'] / 1000, selected['neighbors']]
                          if selected is not None else [None, None, None])
            inner_mae = block['inner_scores'][index]['mae'] if index is not None else None
            folds.append([block['id'], block['locations'], block['inner_common'], inner_mae, *parameters,
                          *self.nested_scores(block['tuned']), block['baseline']['predicted'], block['baseline']['rmse']])
        self.show_nested_tables(result, 'Настройки', summary,
                                ['Блок', 'Места', 'Общие внутренние места', 'Внутренняя MAE',
                                 'Степень IDW', 'Радиус, км', 'Максимум соседей', *metrics,
                                 'С прогнозом при текущих настройках', 'RMSE текущих настроек'], folds)
        selected_count = sum(block['selected'] is not None for block in result['blocks'])
        self.dialog.set_interpolation_status(tr('Подбор выполнен для {0} из {1} блоков. Общих внешних мест с прогнозом: {2}.',
                               selected_count, len(folds), result['tuned_common']['predicted']))

    @staticmethod
    def nested_scores(score, *, coverage=True):
        values = [score['predicted']]
        if coverage:
            values.append(f"{score['coverage']:.1%}")
        return values + [score['rmse'], score['mae'], score['bias']]

    def show_nested_tables(self, result, summary_label, summary, fold_headers, folds):
        self.nested_result = result
        self.settings_table.hide()
        self.nested_note.show()
        summary_headers = [summary_label, 'С прогнозом', 'Доля с прогнозом', 'RMSE', 'MAE', 'Смещение',
                           'Общие места', 'RMSE на общих местах', 'MAE на общих местах', 'Смещение на общих местах']
        for table, headers, rows in ((self.nested_summary, summary_headers, summary),
                                     (self.nested_folds, fold_headers, folds)):
            old = table.model()
            table.setModel(ResultsModel(headers, rows, table))
            if old is not None:
                old.deleteLater()
            table.resizeColumnsToContents()
            table.show()
        self.export_variants.setEnabled(True)

    def calculate_nested_methods(self):
        try:
            snapshot, options, size = self.dialog.calculation_input()
            variants = self.variant_settings(options)
            ranges = self.kriging_ranges()
            settings = self.dialog.kriging_form.values()
            settings.pop('rbf_smoothing')
            settings.pop('directions')
            model = KrigingOptions(**settings)
            shifted = self.nested_shift.isChecked()
            self.clear_settings_comparison()
            generation = self.dialog.generation
            result = self.run_settings_comparison(lambda progress: compare_nested_methods(
                snapshot, options, size, variants, model, ranges, shifted=shifted, progress=progress))
            if generation == self.dialog.generation:
                self.completed_nested_methods(result)
        except TaskCancelled:
            self.dialog.set_interpolation_status(tr('Сравнение методов отменено.'))
        except Exception as exc:
            self.dialog.set_interpolation_status(tr('Расчёт не выполнен: {0}', message(exc)))

    def kriging_ranges(self):
        return [float(item.strip().replace(',', '.')) for item in self.nested_ranges.text().split(';')]

    @staticmethod
    def nested_method_labels():
        return {'IDW': tr('IDW с подбором'), 'kriging': tr('Кригинг с подбором'),
                'TIN': 'TIN', 'baseline': tr('IDW с текущими настройками')}

    def completed_nested_methods(self, result):
        labels = self.nested_method_labels()
        summary = [[labels[name], *self.nested_scores(stats['metrics']),
                    *self.nested_scores(stats['common'], coverage=False)]
                   for name, stats in result['methods'].items()]
        folds = []
        for block in result['blocks']:
            for name, stats in block['methods'].items():
                index = stats['selected']
                settings = result['candidates'][name][index] if name in result['candidates'] and index is not None else {}
                radius = settings.get('max_distance')
                folds.append([block['id'], labels[name], block['locations'], stats['inner_common'],
                              settings.get('power'), radius / 1000 if radius is not None else None,
                              settings.get('neighbors'), settings.get('range_km'),
                              *self.nested_scores(stats['metrics'])])
        self.nested_note.setText(tr('Для IDW подбираются заданные варианты, для кригинга — дальность. Остальные параметры кригинга берутся из вкладки «Сравнение методов». Подбор использует общие внутренние прогнозы каждого метода.'))
        self.show_nested_tables(result, 'Метод', summary,
                                ['Блок', 'Метод', 'Места', 'Общие внутренние места', 'Степень IDW', 'Радиус, км',
                                 'Максимум соседей', 'Дальность вариограммы, км', 'С прогнозом', 'Доля с прогнозом',
                                 'RMSE', 'MAE', 'Смещение'], folds)
        self.dialog.set_interpolation_status(tr('Общие внешние места с прогнозом всех методов: {0} из {1}.',
                               result['methods']['IDW']['common']['predicted'], len(result['locations'])))

    def completed_settings(self, result):
        self.settings_result = result
        rows = []
        for variant in result['variants']:
            settings = variant['parameters']
            scores, common = variant['metrics'], variant['common_metrics']
            rows.append([settings['power'], settings['max_distance'] / 1000, settings['neighbors'],
                         scores['predicted'], f"{scores['coverage']:.1%}", scores['rmse'], scores['mae'], scores['bias'],
                         common['predicted'], common['rmse'], common['mae'], common['bias']])
        self.set_settings_rows(rows)
        self.settings_table.selectRow(0)
        self.apply_variant.setEnabled(True)
        self.export_variants.setEnabled(True)
        self.dialog.set_interpolation_status(tr('Общие места с прогнозом: {0} из {1}.',
                               len(result['common_indices']), len(result['locations'])))

    def apply_idw_variant(self):
        selected = self.settings_table.selectionModel().selectedRows()
        if not self.settings_result or not selected:
            return
        parameters = self.settings_result['variants'][selected[0].row()]['parameters']
        self.dialog.form.load(parameters)
        self.dialog.invalidate(keep_comparison=True)
        self.dialog.set_interpolation_status(tr('Настройки применены. Запустите расчёт поверхности.'))

    def export_settings_comparison(self):
        if self.settings_result is None and self.nested_result is None:
            return
        filename = 'idw-nested-validation.json' if self.nested_result else 'idw-comparison.json'
        if self.nested_result and self.nested_result['format'] == 'autopyro.nested-spatial-methods':
            filename = 'nested-spatial-methods.json'
        path, _ = QFileDialog.getSaveFileName(self.dialog, tr('Сохранить сравнение JSON…'), filename, 'JSON (*.json)')
        if path:
            try:
                indexes = self.settings_table.selectionModel().selectedRows()
                selected = indexes[0].row() if indexes else None
                result = self.nested_result or dict(self.settings_result, selected_variant_index=selected)
                Path(path).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            except (ValueError, TypeError, OSError) as exc:
                QMessageBox.warning(self.dialog, tr('Ошибка'), message(exc))

