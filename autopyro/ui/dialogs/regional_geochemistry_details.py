"""Display selected samples, well depth profiles and spatial binding diagnostics."""

from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QComboBox, QLabel, QTableView, QVBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from autopyro.visualization.plotting import category_key

from ..i18n import tr
from ..widgets.results import ResultsModel


SAMPLE_STATUS = {'numeric': 'Учтено', 'missing': 'Пропуски', 'non_detect': 'Не обнаружено',
    'censored': 'За пределом измерения', 'other': 'Текст или неконечное число', 'qc_excluded': 'Исключено QC',
    'stale_excluded': 'Устаревшие значения', 'interpolation_excluded': 'Интерполяция исключена'}


SPATIAL_STATUS = {'boundary': 'На границе', 'overlap': 'В перекрытии областей',
    'outside': 'Вне областей', 'no_location': 'Без координат', 'missing_region': 'Не задана область'}


SPATIAL_COLORS = {'boundary': '#e68600', 'overlap': '#a534b6', 'outside': '#0072b2', 'missing_region': '#555555'}


def build_detail_tabs(sheet):
    sheet.samples = QTableView()
    sheet.samples.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
    sheet.samples.doubleClicked.connect(lambda index: sheet.sample_opened.emit(sheet.sample_records[index.row()]['row_id']))
    sheet.sample_records = []
    sheet.samples_tab = QWidget()
    layout = QVBoxLayout(sheet.samples_tab)
    hint = QLabel(tr('Выберите строки сводки. Двойной щелчок по пробе открывает исходную строку.'))
    hint.setWordWrap(True)
    layout.addWidget(hint)
    layout.addWidget(sheet.samples)
    sheet.tabs.addTab(sheet.samples_tab, tr('Пробы'))

    sheet.profile_tab = QWidget()
    layout = QVBoxLayout(sheet.profile_tab)
    sheet.profile_metric = QComboBox()
    sheet.profile_metric.addItems(['TOC', 'HI', 'Tmax'])
    sheet.profile_well = QComboBox()
    layout.addWidget(sheet.profile_metric)
    layout.addWidget(sheet.profile_well)
    sheet.profile_canvas = FigureCanvasQTAgg(Figure(layout='constrained'))
    layout.addWidget(NavigationToolbar2QT(sheet.profile_canvas, sheet))
    layout.addWidget(sheet.profile_canvas, 1)
    sheet.profile_metric.currentTextChanged.connect(sheet.refresh_profile)
    sheet.profile_well.currentIndexChanged.connect(sheet.draw_profile)
    sheet.profile_points = {}
    sheet.tabs.addTab(sheet.profile_tab, tr('Профили глубины'))

    sheet.spatial_tab = QWidget()
    layout = QVBoxLayout(sheet.spatial_tab)
    sheet.spatial_filter = QComboBox()
    sheet.spatial_filter.addItem(tr('Все проблемы привязки'), '')
    for key, label in SPATIAL_STATUS.items():
        sheet.spatial_filter.addItem(tr(label), key)
        if key in SPATIAL_COLORS:
            sheet.spatial_filter.setItemData(sheet.spatial_filter.count() - 1, QColor(SPATIAL_COLORS[key]), Qt.ItemDataRole.DecorationRole)
    layout.addWidget(sheet.spatial_filter)
    sheet.spatial_table = QTableView()
    layout.addWidget(sheet.spatial_table)
    sheet.spatial_records = []
    sheet.spatial_table.doubleClicked.connect(lambda index: sheet.sample_opened.emit(sheet.spatial_records[index.row()]['row_id']))
    sheet.spatial_filter.currentIndexChanged.connect(sheet.refresh_spatial)
    sheet.tabs.addTab(sheet.spatial_tab, tr('Контроль привязки'))
    sheet.tabs.currentChanged.connect(sheet.detail_tab_changed)


def refresh_samples(sheet):
    indexes = {index.row() for index in sheet.table.selectionModel().selectedRows()}
    sheet.sample_records = [sample for entry in sheet.result['distributions']
                           if entry['summary_index'] in indexes for sample in entry.get('samples', [])] if sheet.result else []
    if sheet.tabs.currentWidget() is sheet.samples_tab:
        headers = ['Скважина', 'Возраст', 'Глубина', 'Низ интервала', 'Показатель', 'Метод', 'Исходное значение', 'Значение', 'Статус']
        keys = ['well', 'age', 'depth', 'bottom', 'metric', 'method', 'raw', 'value']
        by_well = sheet.result and sheet.result['options'].get('by_well')
        rows = []
        for sample in sheet.sample_records:
            status = SAMPLE_STATUS[sample['status']]
            well = sample['well']
            if by_well and sample['value'] is not None and (well is None or not str(well).strip()):
                status = tr('Нет ключа скважины')
            rows.append([sample[key] for key in keys] + [tr(status)])
        sheet.replace_model(sheet.samples, headers, rows)
        sheet.samples.selectionModel().selectionChanged.connect(sheet.highlight_samples)
    if sheet.tabs.currentWidget() is sheet.profile_tab:
        sheet.refresh_profile()


def replace_model(sheet, view, headers, rows):
    old = view.model()
    view.setModel(ResultsModel(headers, rows, view))
    if old is not None:
        old.deleteLater()


def highlight_samples(sheet, *_):
    if sheet.tabs.currentWidget() is not sheet.samples_tab:
        return
    ids = {sheet.sample_records[i.row()]['row_id'] for i in sheet.samples.selectionModel().selectedRows()}
    if ids:
        sheet.map_selection_changed.emit(dict(row_ids=ids))
    else:
        sheet.select_on_map(refresh_details=False)


def refresh_profile(sheet, *_):
    metric = sheet.profile_metric.currentText()
    sheet.profile_points = {}
    previous = sheet.profile_well.currentData()
    for sample in sheet.sample_records:
        if sample['metric'] != metric or sample['value'] is None or sample['well'] is None or not str(sample['well']).strip():
            continue
        depth = sample['profile_depth']
        bottom = sample['profile_bottom']
        if depth is not None and (not sheet.result['options']['depth_bottom_id'] or bottom is not None and bottom > depth):
            key = category_key(sample['well'])
            sheet.profile_points.setdefault(key, []).append((sample, depth, bottom))
    with QSignalBlocker(sheet.profile_well):
        sheet.profile_well.clear()
        for key, points in sheet.profile_points.items():
            sheet.profile_well.addItem(str(points[0][0]['well']), key)
        sheet.profile_well.setCurrentIndex(max(0, sheet.profile_well.findData(previous)))
    sheet.draw_profile()


def draw_profile(sheet, *_):
    figure = sheet.profile_canvas.figure
    figure.clear()
    ax = figure.add_subplot()
    points = sheet.profile_points.get(sheet.profile_well.currentData(), [])
    if points:
        values = [sample['value'] for sample, _, _ in points]
        depths = [depth if bottom is None else depth + (bottom - depth) / 2 for _, depth, bottom in points]
        ax.scatter(values, depths, s=16)
        intervals = [(sample['value'], depth, bottom) for sample, depth, bottom in points if bottom is not None]
        if intervals:
            x, top, bottom = zip(*intervals)
            ax.vlines(x, top, bottom, alpha=.5)
        all_depths = [value for entries in sheet.profile_points.values() for _, top, bottom in entries
                      for value in (top, bottom) if value is not None]
        lo, hi = min(all_depths), max(all_depths)
        padding = (hi - lo) * .03 or 1
        ax.set_ylim(hi + padding, lo - padding)
        ax.set_title(sheet.profile_well.currentText())
    else:
        ax.text(.5, .5, tr('Выберите строки сводки с числовыми значениями, глубиной и скважиной.'),
                ha='center', va='center', wrap=True, transform=ax.transAxes)
    depth_unit = sheet.result.get('depth_unit', '') if sheet.result else ''
    metric = sheet.profile_metric.currentText()
    unit = next((row['unit'] for row in sheet.result['summary'] if row['metric'] == metric), '') if sheet.result else ''
    ax.set_xlabel(f'{metric} [{unit}]' if unit else metric)
    ax.set_ylabel(f"{tr('Глубина')} [{depth_unit}]" if depth_unit else tr('Глубина'))
    ax.grid(alpha=.25)
    sheet.profile_canvas.draw_idle()


def refresh_spatial(sheet, *_):
    status = sheet.spatial_filter.currentData()
    issues = sheet.result.get('spatial_issues', []) if sheet.result else []
    sheet.spatial_records = [row for row in issues if not status or row['status'] == status]
    sheet.replace_model(sheet.spatial_table, ['Ключ привязки', 'Статус'],
                       [[row['key'], tr(SPATIAL_STATUS[row['status']])] for row in sheet.spatial_records])
    if sheet.tabs.currentWidget() is sheet.spatial_tab:
        sheet.highlight_spatial()


def highlight_spatial(sheet):
    groups = {color: {r['row_id'] for r in sheet.spatial_records if r['status'] == status}
              for status, color in SPATIAL_COLORS.items()}
    sheet.map_selection_changed.emit(dict(point_groups=groups) if sheet.result else None)


def detail_tab_changed(sheet, *_):
    current = sheet.tabs.currentWidget()
    if current is sheet.comparison:
        sheet.comparison.show_selection()
    elif current is sheet.spatial_tab:
        sheet.highlight_spatial()
    elif current is sheet.samples_tab:
        sheet.refresh_samples()
        sheet.highlight_samples()
    elif current is sheet.profile_tab:
        sheet.refresh_profile()
        sheet.select_on_map(refresh_details=False)
    else:
        sheet.select_on_map(refresh_details=False)
