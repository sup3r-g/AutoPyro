"""Compare spatial partitions and inspect location residuals."""

from ..i18n import tr, message
from dataclasses import asdict
from pathlib import Path
import numpy as np
from autopyro.data_io.file_output import write_json
from autopyro.model.core import timestamp
from matplotlib.figure import Figure
from PySide6.QtCore import QSortFilterProxyModel, Qt
from PySide6.QtWidgets import (
    QDialog,
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QPushButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)
from ..tasks import TaskCancelled, run_with_progress
from ..theme import icon
from ..widgets.figure_canvas import FittedFigureCanvas
from autopyro.spatial.spatial_interpolation import SpatialOptions
from autopyro.spatial.kriging import KrigingOptions, compare_block_partitions, compare_nested_partitions, paired_validation
from ..widgets.results import ResultsModel
from .spatial_labels import IDW_REGION_MODES, FORECAST_REASONS


class PartitionResultsModel(ResultsModel):
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid():
            value = self.rows[index.row()][index.column()]
            if role == Qt.ItemDataRole.UserRole:
                return value
            if role == Qt.ItemDataRole.DisplayRole and value is not None:
                if index.column() == 3:
                    return f'{value:.1%}'
                if self.headers[index.column()] == 'Δ RMSE к IDW':
                    return f'{value:+.4g}' if value else '0'
        return super().data(index, role)


class PartitionSortProxy(QSortFilterProxyModel):
    def lessThan(self, left, right):
        a, b = left.data(self.sortRole()), right.data(self.sortRole())
        if a is None or b is None:
            if a is b:
                return False
            return (b is None) if self.sortOrder() == Qt.SortOrder.AscendingOrder else (a is None)
        return super().lessThan(left, right)


class SpatialPartitionComparison(QWidget):
    def __init__(self, dialog):
        super().__init__(dialog)
        self.dialog = dialog
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel(tr('Размеры блоков, км')))
        self.partition_sizes = QLineEdit('25; 50')
        self.partition_sizes.setAccessibleName(tr('Размеры блоков, км'))
        self.partition_sizes.setPlaceholderText('25; 50')
        self.partition_sizes.textChanged.connect(self.clear_partitions)
        controls.addWidget(self.partition_sizes)
        self.partition_shift = QCheckBox(tr('Сдвиг на полблока по X и Y'))
        self.partition_shift.setChecked(True)
        self.partition_shift.toggled.connect(self.clear_partitions)
        controls.addWidget(self.partition_shift)
        run = QPushButton(icon('files'), tr('Сравнить разбиения'))
        run.clicked.connect(self.calculate_partitions)
        controls.addWidget(run)
        layout.addLayout(controls)
        self.partition_nested = QCheckBox(tr('Подбирать параметры на внутренних блоках'))
        self.partition_nested.setToolTip(tr('Варианты IDW и дальности кригинга берутся из вкладки «Сравнение настроек IDW».'))
        self.partition_nested.setWhatsThis(tr('Для вложенной проверки нужны хотя бы три занятых блока в каждом разбиении и общие внутренние прогнозы для подбора.'))
        self.partition_nested.toggled.connect(self.clear_partitions)
        modes = QHBoxLayout()
        modes.addWidget(self.partition_nested)
        setup = QPushButton(icon('settings-2'), tr('Показать параметры пакета'))
        setup.clicked.connect(self.preview_partition_setup)
        modes.addWidget(setup)
        self.export_partition_button = QPushButton(icon('log-out'), tr('Экспорт пакета JSON…'))
        self.export_partition_button.setEnabled(False)
        self.export_partition_button.clicked.connect(self.export_partitions)
        modes.addWidget(self.export_partition_button)
        modes.addStretch()
        layout.addLayout(modes)
        self.partition_setup = QLabel()
        self.partition_setup.setWordWrap(True)
        layout.addWidget(self.partition_setup)
        self.partition_details = QLabel()
        self.partition_details.setWordWrap(True)
        layout.addWidget(self.partition_details)
        self.partition_tuning = QPushButton(icon('shield-check'), tr('Диагностика подбора…'))
        self.partition_tuning.setEnabled(False)
        self.partition_tuning.clicked.connect(self.show_partition_tuning)
        actions = QHBoxLayout()
        actions.addWidget(self.partition_tuning)
        self.partition_pair = QPushButton(icon('files'), tr('Сравнить пару методов…'))
        self.partition_pair.clicked.connect(self.show_partition_pair)
        actions.addWidget(self.partition_pair)
        self.partition_back = QPushButton(icon('undo-2'), tr('К настройкам сравнения'))
        self.partition_back.clicked.connect(lambda: self.focus_in_scroll_area(self.partition_sizes))
        actions.addWidget(self.partition_back)
        actions.addStretch()
        layout.addLayout(actions)
        inspection = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(inspection, 1)
        self.partition_figure = Figure(figsize=(8, 5), dpi=96)
        self.partition_canvas = FittedFigureCanvas(self.partition_figure)
        inspection.addWidget(self.partition_canvas)
        self.partition_canvas.mpl_connect('button_press_event', self.pick_residual_location)
        self.partition_table = QTableView()
        self.partition_table.setAccessibleName(tr('Сводка сравнения разбиений'))
        self.partition_table.setTabKeyNavigation(False)
        self.partition_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.partition_table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        inspection.addWidget(self.partition_table)
        self.residual_table = QTableView()
        self.residual_table.setAccessibleName(tr('Остатки по скважинам и локациям'))
        self.residual_table.setTabKeyNavigation(False)
        self.residual_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.residual_table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.residual_table.setToolTip(tr('Остатки по локациям выбранного метода и разбиения'))
        inspection.addWidget(self.residual_table)
        inspection.setSizes([330, 145, 180])
        self.residual_locations = []
        self.residual_marker = None
        self.partition_results = []
        self.partition_export = None
        self.reset_partition_notes()

    def focus_in_scroll_area(self, widget):
        widget.setFocus()
        page = self.parentWidget()
        while page is not None and not isinstance(page, QScrollArea):
            page = page.parentWidget()
        if page is not None:
            page.ensureWidgetVisible(widget, 0, 0)

    def reset_partition_notes(self):
        self.partition_setup.setText(tr('Размеры через точку с запятой. Выберите строку результата, чтобы увидеть блоки и остатки.'))
        self.partition_details.clear()
        self.partition_tuning.setEnabled(False)
        self.partition_pair.setEnabled(False)
        self.partition_back.hide()

    def clear_partitions(self, *_):
        self.dialog.generation += 1
        self.partition_results = []
        self.partition_export = None
        self.export_partition_button.setEnabled(False)
        self.reset_partition_notes()
        old = self.partition_table.model()
        self.partition_table.setModel(None)
        if old is not None:
            old.deleteLater()
        self.clear_residual_locations()
        self.partition_figure.clear()
        self.partition_canvas.draw_idle()

    def describe_partition_setup(self, options, model, smoothing, variants, ranges):
        setup = tr('IDW · степень {0}, соседи {1}–{2}, радиус {3:g} км.',
                   options.power, options.min_neighbors, options.neighbors, options.max_distance / 1000)
        setup += ' ' + tr('Режим областей: {0}.', tr(IDW_REGION_MODES[options.region_mode]))
        setup += ' ' + tr('Буфер проверки: {0:g} км.', options.validation_buffer_km)
        family = self.dialog.kriging_form.fields['model'][0].currentText()
        setup += ' ' + tr('Кригинг · {0}, порог {1:g}, наггет {2:g}, азимут {3:g}°, отношение дальностей {4:g}.',
                          family, model.sill, model.nugget, model.azimuth_deg, model.range_ratio)
        if variants:
            def listed(values):
                return ', '.join(f'{value:g}' for value in dict.fromkeys(values))

            setup += ' ' + tr('Для вложенной проверки нужны хотя бы три занятых блока в каждом разбиении и общие внутренние прогнозы для подбора.')
            return setup + ' ' + tr('Подбор IDW ({0} вариантов) · степени {1}, радиусы {2} км, максимумы соседей {3}. Дальности кригинга, км: {4}.',
                                    len(variants), listed(item['power'] for item in variants),
                                    listed(item['max_distance'] / 1000 for item in variants),
                                    listed(item['neighbors'] for item in variants), listed(ranges))
        return setup + ' ' + tr('Дальность кригинга {0:g} км. Сглаживание RBF {1:g}.', model.range_km, smoothing)

    def preview_partition_setup(self):
        try:
            values = self.dialog.form.values()
            values.pop('scope', None)
            values.pop('block_size_km')
            options = SpatialOptions(**values)
            settings = self.dialog.kriging_form.values()
            smoothing = settings.pop('rbf_smoothing')
            settings.pop('directions')
            nested = self.partition_nested.isChecked()
            self.partition_setup.setText(self.describe_partition_setup(
                options, KrigingOptions(**settings), smoothing,
                self.dialog.variant_settings(options) if nested else None, self.dialog.kriging_ranges() if nested else None))
        except (ValueError, TypeError) as exc:
            self.partition_setup.setText(message(exc))

    def calculate_partitions(self):
        try:
            snapshot, options, _ = self.dialog.calculation_input()
            sizes = [float(value.strip().replace(',', '.')) for value in self.partition_sizes.text().split(';')]
            settings = self.dialog.kriging_form.values()
            smoothing = settings.pop('rbf_smoothing')
            settings.pop('directions')
            model = KrigingOptions(**settings)
            shifted = self.partition_shift.isChecked()
            variants, ranges = None, None
            nested = self.partition_nested.isChecked()
            if nested:
                variants, ranges = self.dialog.variant_settings(options), self.dialog.kriging_ranges()
            self.clear_partitions()
            self.partition_setup.setText(self.describe_partition_setup(options, model, smoothing, variants, ranges))
            generation = self.dialog.generation
            title = tr('Проверка разных разбиений…')
            self.dialog.set_interpolation_status(title)

            def calculate(update):
                def progress(index, total, size, shift, fraction):
                    stage = tr('Разбиение {0} из {1} · блок {2:g} км, сдвиг {3:g} км.',
                               index + 1, total, size, shift)
                    update(stage, int(100 * (index + fraction) / total))

                if nested:
                    rows = compare_nested_partitions(snapshot, options, model, sizes, variants, ranges,
                                                     shifted=shifted, progress=progress)
                else:
                    rows = compare_block_partitions(snapshot, options, model, sizes, shifted=shifted,
                                                    rbf_smoothing=smoothing, progress=progress)
                update(tr('Подготовка сводки…'), 100)
                return rows

            results = run_with_progress(self.dialog, title, calculate, cancellable=True)
            if generation == self.dialog.generation:
                self.partition_export = dict(
                    format='autopyro.spatial-partition-comparison', version=1, created_at=timestamp(),
                    mode='nested' if nested else 'fixed', block_sizes_km=list(dict.fromkeys(sizes)),
                    half_block_shift=shifted, spatial_options=asdict(options), kriging_model=asdict(model),
                    idw_variants=variants, kriging_ranges_km=ranges,
                    rbf_smoothing=None if nested else smoothing, map_spec=snapshot['spec'],
                    map_diagnostics=snapshot.get('diagnostics', {}),
                    variable=next(c for c in snapshot['columns'] if c['id'] == options.value_id), results=results,
                )
                self.completed_partitions(results, snapshot)
                self.export_partition_button.setEnabled(True)
                self.focus_in_scroll_area(self.partition_table)
        except TaskCancelled:
            self.dialog.set_interpolation_status(tr('Пакетная проверка отменена.'))
        except Exception as exc:
            self.dialog.set_interpolation_status(tr('Расчёт не выполнен: {0}', message(exc)))

    def export_partitions(self):
        payload = self.partition_export
        if payload is None:
            return
        title = tr('Экспорт пакета JSON…')
        path, _ = QFileDialog.getSaveFileName(self.dialog, title, 'spatial-partitions.json', 'JSON (*.json)')
        if not path:
            return
        if not Path(path).suffix:
            path += '.json'
        try:
            run_with_progress(self.dialog, title, lambda: write_json(path, payload))
            self.dialog.set_interpolation_status(tr('Пакет сохранён: {0}', path))
        except (ValueError, TypeError, OSError) as exc:
            self.dialog.set_interpolation_status(tr('Не удалось сохранить пакет: {0}', message(exc)))

    def completed_partitions(self, results, snapshot):
        self.partition_results = results
        headers = ['Блок, км', 'Сдвиг, км', 'Метод', 'Покрытие', 'Общие места', 'RMSE', 'MAE', 'Δ RMSE к IDW']
        anisotropic = any(row['method'] == 'isotropic-kriging' for row in results)
        completed = [i for i, row in enumerate(results) if 'issue' not in row]
        skipped = len(results) - len(completed)
        labels = self.dialog.nested_method_labels() if skipped or 'shared' in results[0] else None
        if labels:
            headers.append('Ранг RMSE')
        rows = []
        for row in results:
            values = [row['block_size_km'], row['shift_km']]
            if 'issue' in row:
                values.extend([tr('Пропущено')] + [None] * (len(headers) - 3))
            else:
                method = labels[row['method']] if labels else self.dialog.kriging_method_label(row['method'], anisotropic)
                common = row['common']
                values.extend([method, row['coverage'], common['predicted'], common['rmse'], common['mae'],
                               row.get('delta_rmse_idw')])
                if labels:
                    values.append(row['rank'])
            rows.append(values)
        self.partition_table.setToolTip(tr('RMSE, MAE, разность и ранг рассчитаны на общих местах выбранного разбиения. Отрицательная разность означает меньшую RMSE, чем у IDW с текущими настройками.'))
        proxy = PartitionSortProxy(self.partition_table)
        proxy.setSourceModel(PartitionResultsModel(headers, rows, proxy))
        proxy.setSortRole(Qt.ItemDataRole.UserRole)
        self.partition_table.setModel(proxy)
        self.partition_table.setSortingEnabled(True)
        self.partition_table.sortByColumn(-1, Qt.SortOrder.AscendingOrder)
        self.partition_table.resizeColumnsToContents()
        limit = max((abs(point['residual']) for row in results for point in row.get('locations', ())
                     if point['residual'] is not None), default=0)
        self.partition_table.selectionModel().selectionChanged.connect(
            lambda *_: self.show_partition(snapshot, limit))
        self.partition_table.selectRow(completed[0] if completed else 0)
        self.partition_back.show()
        if not completed:
            status = tr('Ни одно разбиение не проверено. Выберите строку, чтобы увидеть причину.')
        elif labels:
            first = results[completed[0]]
            status = tr('Общие места с прогнозом всех методов во всех выполненных разбиениях: {0} из {1}.',
                        first['shared']['predicted'], len(first['locations']))
        else:
            status = tr('Сравнение разбиений завершено.')
        if skipped:
            status += ' ' + tr('Пропущено разбиений: {0}.', skipped)
        self.dialog.set_interpolation_status(status)

    def selected_partition_row(self):
        selection = self.partition_table.selectionModel()
        indexes = selection.selectedRows() if selection is not None else []
        return self.partition_results[self.partition_table.model().mapToSource(indexes[0]).row()] if indexes else None

    def show_partition_pair(self):
        row = self.selected_partition_row()
        if not row or 'issue' in row:
            return
        partition = [item for item in self.partition_results if 'issue' not in item
                     and (item['block_size_km'], item['shift_km']) == (row['block_size_km'], row['shift_km'])]
        pairs = list(paired_validation(partition).items())
        labels = (self.dialog.nested_method_labels() if 'shared' in row else {
            item['method']: self.dialog.kriging_method_label(
                item['method'], any(entry['method'] == 'isotropic-kriging' for entry in partition))
            for item in partition})
        dialog = QDialog(self)
        dialog.setWindowTitle(tr('Блок {0:g} км, сдвиг {1:g} км', row['block_size_km'], row['shift_km']))
        layout = QVBoxLayout(dialog)
        selector = QComboBox()
        selector.setAccessibleName(tr('Локации для сравнения ошибок'))
        selector.addItems([' / '.join(labels[method] for method in methods) for methods, _ in pairs])
        layout.addWidget(selector)
        table = QTableView()
        table.setTabKeyNavigation(False)
        model = ResultsModel(['Метод', 'Общие места', 'RMSE', 'MAE', 'Смещение'], [], table)
        table.setAccessibleName(tr('Результаты сравнения методов'))
        table.setModel(model)
        layout.addWidget(table)

        def update(index):
            model.set_rows([[labels[name], stats['predicted'], stats['rmse'], stats['mae'], stats['bias']]
                            for name, stats in pairs[index][1].items()])
            table.resizeColumnsToContents()

        selector.currentIndexChanged.connect(update)
        update(0)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()
        dialog.deleteLater()

    def show_partition_tuning(self):
        row = self.selected_partition_row()
        if not row or not row.get('candidates'):
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(tr('Диагностика подбора'))
        dialog.resize(850, 510)
        layout = QVBoxLayout(dialog)
        note = QLabel(tr('Для каждого внешнего блока показано покрытие каждого кандидата на внутренних локациях. '
                         'Подбор использует только локации с прогнозом всех кандидатов.'))
        note.setWordWrap(True)
        layout.addWidget(note)
        table = QTableView(dialog)
        table.setAccessibleName(tr('Диагностика подбора'))
        table.setTabKeyNavigation(False)
        headers = ['Внешний блок', 'Вариант', 'Внутренних локаций', 'С прогнозом',
                   'Общие внутренние места', 'MAE на общих местах', 'Выбран']
        rows = []
        for block in row['blocks']:
            stats = block['methods'][row['method']]
            for index, (candidate, count) in enumerate(zip(row['candidates'], stats['inner_candidate_counts'])):
                if row['method'] == 'IDW':
                    label = tr('Степень {0:g}, радиус {1:g} км, максимум соседей {2}',
                               candidate['power'], candidate['max_distance'] / 1000, candidate['neighbors'])
                else:
                    label = tr('Дальность {0:g} км', candidate['range_km'])
                rows.append([block['id'], label, stats['inner_locations'], count, stats['inner_common'],
                             stats['inner_scores'][index]['mae'],
                             tr('Да') if stats['selected'] == index else None])
        table.setModel(ResultsModel(headers, rows, table))
        table.resizeColumnsToContents()
        layout.addWidget(table)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()
        dialog.deleteLater()

    def show_partition(self, snapshot, limit):
        row = self.selected_partition_row()
        self.partition_tuning.setEnabled(bool(row and row.get('candidates')))
        self.partition_pair.setEnabled(bool(row and 'issue' not in row))
        if row is None or 'issue' in row:
            self.partition_details.setText(message(row['issue']) if row else '')
            self.clear_residual_locations()
            self.partition_figure.clear()
            self.partition_canvas.draw_idle()
            return
        def scores(metrics):
            return [f'{metrics[key]:.4g}' if metrics[key] is not None else '—' for key in ('rmse', 'mae', 'bias')]
        detail = tr('Блоков {0}. Прогнозов {1} из {2}. По всем прогнозам: RMSE {3}, MAE {4}, смещение {5}.',
                    len(row['blocks']), row['predicted'], len(row['locations']), *scores(row))
        if row.get('buffer_km'):
            detail += ' ' + tr('Буфер проверки: {0:g} км.', row['buffer_km'])
        if row.get('candidates'):
            folds = [block['methods'][row['method']] for block in row['blocks']]
            counts = [fold['inner_common'] for fold in folds]
            detail += ' ' + tr('Подбор выполнен в {0} из {1} блоков. Общих внутренних мест на блок: {2}–{3}.',
                               sum(fold['selected'] is not None for fold in folds), len(folds), min(counts), max(counts))
        if 'shared' in row:
            detail += ' ' + tr('На местах, общих для всех выполненных разбиений ({0}): RMSE {1}, MAE {2}, смещение {3}.',
                               row['shared']['predicted'], *scores(row['shared']))
            delta = row.get('shared_delta_rmse_idw')
            if delta is not None:
                detail += ' ' + tr('Δ RMSE к текущему IDW на этих же местах: {0:+.4g}.', delta)
        self.partition_details.setText(detail)
        ax, _ = self.dialog.draw_blocks(self.partition_figure, row, snapshot, residual_limit=limit)
        method = self.partition_table.selectionModel().selectedRows(2)[0].data()
        ax.set_title(f"{method} · {row['block_size_km']:g} {tr('км')} · {tr('Сдвиг, км')} {row['shift_km']:g}")
        self.set_residual_locations(row['locations'], snapshot, ax)
        self.partition_canvas.draw_idle()

    def clear_residual_locations(self):
        self.residual_locations = []
        self.residual_marker = None
        old = self.residual_table.model()
        self.residual_table.setModel(None)
        if old is not None:
            old.deleteLater()

    @staticmethod
    def sample_location_labels(snapshot):
        sample_key = snapshot['spec'].get('sample_key')
        labels = {}
        for point in snapshot['points']:
            label = point['values'].get(sample_key)
            if label is None or not str(label).strip():
                label = point.get('label', '')
            if label is not None and str(label).strip():
                labels[point['id']] = str(label)
        return labels

    def set_residual_locations(self, locations, snapshot, ax):
        self.clear_residual_locations()
        self.residual_locations = locations
        labels = self.sample_location_labels(snapshot)
        headers = ['Скважина / локация', 'Блок', 'Статус прогноза', 'X', 'Y', 'Проб', 'Измерение', 'Прогноз', 'Остаток']
        rows = []
        for point in locations:
            label = ', '.join(dict.fromkeys(labels[rid] for rid in point['row_ids'] if rid in labels))
            reason = FORECAST_REASONS.get(point.get('reason'), FORECAST_REASONS['unknown'])
            status = tr(reason[0]) if point['predicted'] is None else tr('С прогнозом')
            rows.append([label or None, point['block'], status, point['x'], point['y'], point['n_samples'],
                         point['value'], point['predicted'], point['residual']])
        self.residual_table.setModel(ResultsModel(headers, rows, self.residual_table))
        self.residual_table.resizeColumnsToContents()
        self.residual_marker = ax.scatter([], [], s=95, facecolors='none', edgecolors='#efad16',
                                          linewidths=2, zorder=15)
        self.residual_table.selectionModel().selectionChanged.connect(self.highlight_residual_location)

    def highlight_residual_location(self, *_):
        if self.residual_marker is None:
            return
        indexes = self.residual_table.selectionModel().selectedRows()
        offsets = np.empty((0, 2))
        if indexes:
            point = self.residual_locations[indexes[0].row()]
            offsets = [[point['x'], point['y']]]
        self.residual_marker.set_offsets(offsets)
        self.partition_canvas.draw_idle()

    def pick_residual_location(self, event):
        if self.residual_marker is None or event.button != 1 or event.inaxes is not self.residual_marker.axes:
            return
        xy = [(point['x'], point['y']) for point in self.residual_locations]
        pixels = event.inaxes.transData.transform(xy)
        distances = np.hypot(pixels[:, 0] - event.x, pixels[:, 1] - event.y)
        index = int(distances.argmin())
        if distances[index] <= 8:
            self.residual_table.selectRow(index)
            self.residual_table.scrollTo(self.residual_table.model().index(index, 0))

