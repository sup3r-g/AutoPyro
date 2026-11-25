"""Build the regional geochemistry worksheet and connect its main controls."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from ..i18n import tr
from ..widgets.regional_comparison import RegionalComparison
from ..theme import icon


def build(sheet, project, report):
    root = QVBoxLayout(sheet)
    _intro(root)
    split = QSplitter()
    root.addWidget(split, 1)
    calculate = _settings_panel(sheet, project, report, split)
    _result_tabs(sheet, project, split)
    sheet.build_detail_tabs()
    _scenario_controls(sheet, root)
    split.setSizes([380, 850])
    _footer(sheet, root, calculate)
    sheet.layer.currentIndexChanged.connect(sheet.update_regions)
    sheet.region.currentIndexChanged.connect(sheet.invalidate)
    for combo in sheet.bindings.values():
        combo.currentIndexChanged.connect(sheet.invalidate)
    sheet.update_regions()


def _intro(root):
    note = QLabel(tr('Используется текущая выборка карты. Фильтры ниже сужают её. Области определяются по положению проб; границы, перекрытия и точки вне областей исключаются.'))
    note.setWordWrap(True)
    root.addWidget(note)
    hint = QLabel(tr('Выбор строки подсвечивает область и пробы на карте; щелчок по области выделяет её строки сводки. Ctrl/Shift — несколько строк.'))
    hint.setWordWrap(True)
    root.addWidget(hint)


def _settings_panel(sheet, project, report, split):
    panel = QWidget()
    settings = QVBoxLayout(panel)
    form = QFormLayout()
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    settings.addLayout(form)
    _bindings(sheet, project, report, form)
    _filters(sheet, settings, form)
    return _attach_settings(sheet, split, panel, settings)


def _bindings(sheet, project, report, form):
    sheet.layer = QComboBox()
    for layer in report['layers']:
        if any(f.get('geometry', {}).get('type') in ('Polygon', 'MultiPolygon') for f in layer['features']):
            sheet.layer.addItem(layer['name'], layer['id'])
    sheet.region = QComboBox()
    form.addRow(tr('Полигональный слой'), sheet.layer)
    form.addRow(tr('Поле области'), sheet.region)
    sheet.bindings = {}
    for key, label, guess in [
        ('toc_id', 'TOC', 'TOC'), ('hi_id', 'HI', 'HI'), ('tmax_id', 'Tmax', 'Tmax'),
        ('well_id', 'Скважина', 'Well'), ('age_id', 'Возраст', 'Age'),
        ('depth_id', 'Глубина / верх интервала', 'Depth'), ('depth_bottom_id', 'Низ интервала (необязательно)', ''),
        ('toc_method_id', 'Метод TOC', 'Method TOC'), ('pyrolysis_method_id', 'Метод пиролиза (HI, Tmax)', 'Method Pyrolysis')]:
        combo = QComboBox()
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(18)
        combo.addItem(tr('Не задан'), '')
        for col in project.columns:
            combo.addItem(f'{col.name} [{col.unit}]' if col.unit else col.name, col.id)
        found = next((c.id for c in project.columns if c.name.casefold() == guess.casefold()), '')
        combo.setCurrentIndex(max(0, combo.findData(found)))
        sheet.bindings[key] = combo
        form.addRow(tr(label), combo)


def _filters(sheet, settings, form):
    sheet.depth_min, sheet.depth_max = QLineEdit(), QLineEdit()
    for label, widget in [('Глубина от (пусто — без ограничения)', sheet.depth_min), ('Глубина до (единицы колонки)', sheet.depth_max)]:
        form.addRow(tr(label), widget)
        widget.textChanged.connect(sheet.invalidate)
    sheet.choices = {}
    for key, binding, label in [('ages', 'age_id', 'Возраст: отметьте значения; пусто — все'),
            ('toc_methods', 'toc_method_id', 'Методы TOC: пусто — все'),
            ('pyrolysis_methods', 'pyrolysis_method_id', 'Методы пиролиза: пусто — все')]:
        heading = QLabel(tr(label))
        heading.setWordWrap(True)
        settings.addWidget(heading)
        listing = QListWidget()
        listing.setMaximumHeight(115)
        sheet.choices[key] = listing
        settings.addWidget(listing)
        listing.itemChanged.connect(sheet.invalidate)
        sheet.bindings[binding].currentIndexChanged.connect(lambda _, k=key, b=binding: sheet.populate(k, b))
        sheet.populate(key, binding)
    sheet.stratify = QCheckBox(tr('Разделять распределения по методам'))
    sheet.stratify.setChecked(True)
    sheet.text_numbers = QCheckBox(tr('Распознавать числа в текстовых колонках'))
    sheet.text_numbers.setChecked(True)
    sheet.interpolated = QCheckBox(tr('Включать интерполированные значения'))
    sheet.by_well = QCheckBox(tr('По скважинам (медиана замеров каждой скважины)'))
    for widget in (sheet.stratify, sheet.text_numbers, sheet.interpolated, sheet.by_well):
        settings.addWidget(widget)
        widget.toggled.connect(sheet.invalidate)


def _attach_settings(sheet, split, panel, settings):
    calculate = QPushButton(tr('Рассчитать сравнение'))
    calculate.clicked.connect(sheet.calculate)
    settings.addStretch()
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(panel)
    split.addWidget(scroll)
    return calculate


def _result_tabs(sheet, project, split):
    sheet.tabs = QTabWidget()
    split.addWidget(sheet.tabs)
    sheet.table = QTableWidget()
    sheet.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    sheet.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    sheet.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
    sheet.table.itemSelectionChanged.connect(sheet.select_on_map)
    sheet.tabs.addTab(sheet.table, tr('Сводная статистика'))
    chart = QWidget()
    chart_layout = QVBoxLayout(chart)
    sheet.metric_combo = QComboBox()
    sheet.metric_combo.addItems(['TOC', 'HI', 'Tmax'])
    sheet.metric_combo.currentTextChanged.connect(sheet.plot)
    chart_layout.addWidget(sheet.metric_combo)
    sheet.canvas = FigureCanvasQTAgg(Figure())
    chart_layout.addWidget(NavigationToolbar2QT(sheet.canvas, sheet))
    chart_scroll = QScrollArea()
    chart_scroll.setWidget(sheet.canvas)
    chart_layout.addWidget(chart_scroll, 1)
    sheet.tabs.addTab(chart, tr('Распределения'))
    sheet.diagnostics = QLabel()
    sheet.diagnostics.setWordWrap(True)
    sheet.diagnostics.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    sheet.tabs.addTab(sheet.diagnostics, icon('shield-check'), tr('Диагностика'))
    sheet.comparison = RegionalComparison(project.view.get('regional_comparison_snapshots', {}))
    sheet.comparison.capture_requested.connect(sheet.capture_comparison_snapshot)
    sheet.comparison.report_requested.connect(lambda: sheet.comparison.export_pdf(sheet.report))
    sheet.comparison.selection_changed.connect(sheet.publish_comparison_selection)
    sheet.tabs.addTab(sheet.comparison, tr('Сравнение A и B'))


def _scenario_controls(sheet, root):
    scenarios = QHBoxLayout()
    root.addLayout(scenarios)
    sheet.scenario = QComboBox()
    scenarios.addWidget(sheet.scenario, 1)
    for label, action in [(tr('Сохранить сценарий…'), sheet.save_scenario),
                          (tr('Загрузить сценарий'), sheet.load_scenario),
                          (tr('Удалить сценарий'), sheet.delete_scenario)]:
        button = QPushButton(label)
        button.clicked.connect(action)
        scenarios.addWidget(button)
    sheet.refresh_scenarios()


def _footer(sheet, root, calculate):
    footer = QLabel(tr('Сравнение описательное. В режиме по скважинам каждая медиана имеет одинаковый вес. Пробы без ключа скважины исключаются из этого расчёта.'))
    footer.setWordWrap(True)
    root.addWidget(footer)
    buttons = QHBoxLayout()
    root.addLayout(buttons)
    buttons.addWidget(calculate)
    sheet.exports = []
    for label, kind in [('Сводка CSV…', 'summary'), ('Значения CSV…', 'values'), ('Отчёт JSON…', 'json'), ('График…', 'figure')]:
        button = QPushButton(tr(label))
        button.clicked.connect(lambda _, k=kind: sheet.export(k))
        buttons.addWidget(button)
        sheet.exports.append(button)
