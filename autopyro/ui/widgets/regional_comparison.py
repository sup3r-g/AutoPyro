"""Comparison tab for named, frozen regional results."""
import json
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QSignalBlocker, Signal, Qt
from pyproj.exceptions import ProjError
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QSplitter, QStackedWidget, QTableView, QVBoxLayout, QWidget)

from autopyro.spatial.regional_comparison import SCENARIO_COLORS, compare_regions, comparison_map_points, method_key, render_comparison
from autopyro.visualization.plotting import category_key
from autopyro.reports.regional_report import export_paired_pdf, export_pairs_csv
from ..i18n import message, tr
from .results import ResultsModel


class RegionalComparison(QWidget):
    capture_requested = Signal()
    report_requested = Signal()
    selection_changed = Signal(object)

    def __init__(self, snapshots, parent=None):
        super().__init__(parent)
        self.snapshots = snapshots
        self.rows = []
        layout = QVBoxLayout(self)
        self.capture = QPushButton(tr('Сохранить снимок расчёта…'))
        self.capture.setEnabled(False)
        self.capture.clicked.connect(self.capture_requested)
        export = QPushButton(tr('Экспорт снимков A и B…'))
        export.clicked.connect(self.export_pair)
        actions = QHBoxLayout()
        actions.addWidget(self.capture)
        actions.addWidget(export)
        layout.addLayout(actions)
        exports = QHBoxLayout()
        self.pdf_button = QPushButton(tr('Отчёт PDF…'))
        self.csv_button = QPushButton(tr('Пары скважин CSV…'))
        self.pdf_button.clicked.connect(self.report_requested)
        self.csv_button.clicked.connect(self.export_csv)
        exports.addWidget(self.pdf_button)
        exports.addWidget(self.csv_button)
        layout.addLayout(exports)
        selectors = QFormLayout()
        layout.addLayout(selectors)
        self.a, self.b = QComboBox(), QComboBox()
        self.method_a, self.method_b = QComboBox(), QComboBox()
        for name, snapshot, method in [('A', self.a, self.method_a), ('B', self.b, self.method_b)]:
            line = QHBoxLayout()
            line.addWidget(snapshot, 2)
            line.addWidget(method, 1)
            selectors.addRow(name, line)
            snapshot.currentIndexChanged.connect(self.refresh_methods)
            method.currentIndexChanged.connect(self.compare)
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(['TOC', 'HI', 'Tmax'])
        self.metric_combo.currentIndexChanged.connect(self.refresh_methods)
        selectors.addRow(tr('Показатель'), self.metric_combo)
        self.paired = QCheckBox(tr('Парное сравнение по общим скважинам'))
        self.paired.toggled.connect(self.compare)
        layout.addWidget(self.paired)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.legend = QLabel(
            f"<span style='color:{SCENARIO_COLORS['A']}'>● A</span> · "
            f"<span style='color:{SCENARIO_COLORS['B']}'>● B</span> · "
            f"<span style='color:{SCENARIO_COLORS['AB']}'>● {tr('Общие скважины')}</span>")
        layout.addWidget(self.legend)
        split = QSplitter(Qt.Orientation.Vertical)
        self.table = QTableView()
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        split.addWidget(self.table)
        self.details = QStackedWidget()
        self.canvas = FigureCanvasQTAgg(Figure(layout='constrained'))
        self.details.addWidget(self.canvas)
        self.well_table = QTableView()
        self.well_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.well_table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.details.addWidget(self.well_table)
        split.addWidget(self.details)
        layout.addWidget(split, 1)
        self.refresh_snapshots()

    def refresh_snapshots(self, captured=None):
        names = sorted(self.snapshots)
        selected = [self.a.currentData(), self.b.currentData()]
        if captured is not None:
            selected[1 if selected[0] else 0] = captured
        elif not any(selected):
            selected = (names + [None, None])[:2]
        for combo, name in zip((self.a, self.b), selected):
            with QSignalBlocker(combo):
                combo.clear()
                combo.addItem(tr('Выберите снимок'), None)
                for snapshot in names:
                    combo.addItem(snapshot, snapshot)
                combo.setCurrentIndex(max(0, combo.findData(name)))
        self.refresh_methods()

    def pair(self):
        return tuple(self.snapshots.get(combo.currentData()) for combo in (self.a, self.b))

    def refresh_methods(self, *_):
        metric = self.metric_combo.currentText()
        for combo, snapshot in zip((self.method_a, self.method_b), self.pair()):
            previous = combo.currentData()
            with QSignalBlocker(combo):
                combo.clear()
                if snapshot is None:
                    continue
                groups = [group for group in snapshot['groups'] if group['metric'] == metric]
                for group in sorted(groups, key=lambda group: not bool(group['values'])):
                    key = method_key(group)
                    if combo.findData(key) >= 0:
                        continue
                    method = group['method']
                    if group['pooled']:
                        label = tr('Все методы')
                    elif method is None:
                        label = tr('Метод не задан')
                    elif isinstance(method, str) and method != method.strip():
                        label = repr(method)
                    else:
                        label = str(method)
                    combo.addItem(label, key)
                combo.setCurrentIndex(max(0, combo.findData(previous)))
        self.compare()

    def compare(self, *_):
        a, b = self.pair()
        paired = self.paired.isChecked()
        self.details.setCurrentWidget(self.well_table if paired else self.canvas)
        self.legend.setVisible(not paired)
        self.rows = []
        if a is None or b is None:
            status = tr('Сохраните два расчёта и выберите снимки A и B.')
        else:
            try:
                self.rows = compare_regions(a, b, self.metric_combo.currentText(),
                                            self.method_a.currentData(), self.method_b.currentData(), paired=paired)
                if paired:
                    status = tr('Карта окрашивает области по медиане парных разниц. Выберите область на карте или в сводке, чтобы увидеть скважины.')
                else:
                    status = tr('Разница медиан B − A. Выберите область для просмотра распределений и скважин.')
                if not self.rows:
                    status = tr('Для выбранного показателя и методов нет групп.')
            except ValueError as exc:
                status = message(exc)
        self.status.setText(status)
        self.pdf_button.setEnabled(paired and bool(self.rows))
        self.csv_button.setEnabled(paired and bool(self.rows))
        if paired:
            columns = ['Область', 'Единицы', 'Пар скважин', 'Медиана разниц', 'Q1', 'Q3', 'Мин.', 'Макс.']
            columns += ['Диапазон медиан без одной скважины', 'Устойчивость знака']
            keys = ['n', 'median', 'q1', 'q3', 'minimum', 'maximum']
            data = []
            for row in self.rows:
                stats = row['paired']
                lower, upper = stats['without_min'], stats['without_max']
                interval = f'{lower:.6g} … {upper:.6g}' if lower is not None else None
                status = self.sign_status(stats['sign_changed'])
                if stats['n'] < 2:
                    status = tr('Недостаточно пар для проверки')
                data.append([row['region'], row['unit'], *(stats[key] for key in keys), interval, status])
        else:
            columns = ['Область', 'Единицы', 'Медиана A', 'Q1–Q3 A', 'Проб A (числ./всего)', 'Скважин A',
                       'Медиана B', 'Q1–Q3 B', 'Проб B (числ./всего)', 'Скважин B', 'B − A', 'Общие скважины']
            data = []
            for row in self.rows:
                cells = [row['region'], row['unit']]
                for key in ('a', 'b'):
                    group = row[key] or {}
                    q1, q3 = group.get('q1'), group.get('q3')
                    interval = f'{q1:.6g}–{q3:.6g}' if q1 is not None and q3 is not None else None
                    counts = f"{group['n_valid']}/{group['n_total']}" if group else None
                    cells.extend((group.get('median'), interval, counts, group.get('n_wells')))
                cells.extend((row['delta'], row['common_wells']))
                data.append(cells)
        self.set_rows(self.table, columns, data)
        self.table.selectionModel().selectionChanged.connect(self.show_selection)
        if self.rows:
            self.table.selectRow(0)
        else:
            self.show_selection()

    def set_rows(self, table, columns, data):
        old = table.model()
        table.setModel(ResultsModel(columns, data, table))
        if old is not None:
            old.deleteLater()
        table.resizeColumnsToContents()

    def show_selection(self, *_):
        indexes = self.table.selectionModel().selectedRows()
        row = self.rows[indexes[0].row()] if indexes else None
        paired = self.paired.isChecked()
        if paired:
            pairs = row['paired']['wells'] if row else []
            columns = ['Скважина', 'Медиана A', 'Медиана B', 'B − A', 'Медиана без этой скважины', 'Устойчивость знака']
            self.set_rows(self.well_table, columns,
                          [[pair[key] for key in ('well', 'a', 'b', 'delta', 'median_without')] +
                           [self.sign_status(pair['sign_changed'])] for pair in pairs])
            self.well_table.selectionModel().selectionChanged.connect(self.publish_map)
        elif row:
            render_comparison(self.canvas.figure, row, self.metric_combo.currentText())
        else:
            self.canvas.figure.clear()
        if not paired:
            self.canvas.draw_idle()
        self.publish_map()

    @staticmethod
    def sign_status(changed):
        if changed is None:
            return None
        return tr('Знак меняется') if changed else tr('Знак сохраняется')

    def publish_map(self, *_):
        indexes = self.table.selectionModel().selectedRows()
        row = self.rows[indexes[0].row()] if indexes else None
        paired = self.paired.isChecked()
        well_key = None
        if paired and row:
            wells = self.well_table.selectionModel().selectedRows()
            if wells:
                well_key = category_key(row['paired']['wells'][wells[0].row()]['well'])
        points = comparison_map_points(row, self.pair(), paired=paired, well_key=well_key) if row else []
        layer = self.paired_map()
        if layer is not None:
            layer['selected'] = row['region_key'] if row else None
            layer['selected_count'] = row['paired']['n'] if row else None
        self.selection_changed.emit(dict(snapshot_points=points, paired_map=layer) if points or layer else None)

    def paired_map(self):
        if not self.paired.isChecked() or not self.rows:
            return None
        a, _ = self.pair()
        conditions = a['conditions']
        return dict(layer_id=conditions['options']['layer_id'], field=conditions['options']['region_field'],
                    fingerprint=conditions['region_layer']['fingerprint'], metric=self.metric_combo.currentText(),
                    unit=self.rows[0]['unit'], values={row['region_key']: row['paired']['median'] for row in self.rows},
                    unstable={row['region_key'] for row in self.rows if row['paired']['sign_changed']})

    def select_region(self, key):
        for i, row in enumerate(self.rows):
            if row['region_key'] == key:
                self.table.selectRow(i)
                self.table.scrollTo(self.table.model().index(i, 0))
                return

    def export_pdf(self, map_report):
        if not self.pdf_button.isEnabled():
            return
        path, _ = QFileDialog.getSaveFileName(self, tr('Отчёт PDF…'), 'regional-AB.pdf', 'PDF (*.pdf)')
        if path:
            try:
                export_paired_pdf(path, self.rows, self.pair(), (self.a.currentData(), self.b.currentData()),
                                  (self.method_a.currentText(), self.method_b.currentText()),
                                  self.metric_combo.currentText(), map_report, tr)
            except (OSError, TypeError, ValueError, ProjError) as exc:
                QMessageBox.warning(self, tr('Ошибка'), message(exc))

    def export_csv(self):
        if not self.csv_button.isEnabled():
            return
        path, _ = QFileDialog.getSaveFileName(self, tr('Пары скважин CSV…'), 'regional-AB-pairs.csv', 'CSV (*.csv)')
        if path:
            try:
                export_pairs_csv(path, self.rows, self.metric_combo.currentText(),
                                 (self.a.currentData(), self.b.currentData()),
                                 (self.method_a.currentText(), self.method_b.currentText()))
            except (OSError, TypeError, ValueError) as exc:
                QMessageBox.warning(self, tr('Ошибка'), message(exc))

    def export_pair(self):
        a, b = self.pair()
        if a is None or b is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, tr('Экспорт снимков A и B…'), 'regional-AB.json', 'JSON (*.json)')
        if path:
            try:
                payload = {'A': {'name': self.a.currentData(), 'snapshot': a},
                           'B': {'name': self.b.currentData(), 'snapshot': b},
                           'comparison': {'metric': self.metric_combo.currentText(), 'method_a': self.method_a.currentData(),
                                          'method_b': self.method_b.currentData(), 'difference': 'B - A',
                                          'paired_by_well': self.paired.isChecked()}}
                Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            except (ValueError, TypeError, OSError) as exc:
                QMessageBox.warning(self, tr('Ошибка'), message(exc))
