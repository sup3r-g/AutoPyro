"""Build depth interpolation parameters, preview tabs and commands."""

from __future__ import annotations

from PySide6.QtCharts import QChartView
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from autopyro.analysis.interpolation import minimum_anchor_count

from ..i18n import tr
from ..theme import icon
from ..widgets.documents import analysis_document, parameter_page, refill_combo
from ..widgets.parameters import add_parameter_tools
from .interpolation_navigation import SampleNavigationTable
from .interpolation_results import InterpolationDiagnosticsModel, InterpolationPreviewModel


def build(sheet):
    settings, layout = analysis_document(sheet, parameter_title=tr('Основные'), parameter_icon='calculator')
    sheet.parameter_tabs.setAccessibleName(tr('Параметры расчёта'))
    advanced = parameter_page(sheet.parameter_tabs, tr('Дополнительно'), 'settings-2')
    archive = parameter_page(sheet.parameter_tabs, tr('Сохранённые расчёты'), 'logs')
    form, advanced_form, numeric_choices = _basic_parameters(sheet, settings)
    _advanced_parameters(sheet, advanced, advanced_form)
    _binding_controls(sheet, settings, form, advanced, advanced_form, numeric_choices)
    _run_button(sheet, layout)
    _archive_controls(sheet, archive)
    _proposals_view(sheet)
    _chart_view(sheet, layout)
    _diagnostic_view(sheet, layout)
    _commands(sheet, layout)
    _connect_inputs(sheet)
    _set_tab_order(sheet)
    settings.addStretch()
    advanced.addStretch()
    sheet.invalidate()


def _basic_parameters(sheet, settings):
    inputs = QGroupBox(tr('Параметры расчёта'))
    form = QFormLayout(inputs)
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    advanced_form = QFormLayout()
    advanced_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    advanced_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    sheet.x, sheet.y = QComboBox(), QComboBox()
    sheet.scope = QComboBox()
    form.addRow(tr('Область'), sheet.scope)
    numeric_choices = sheet.numeric_choices()
    for combo in (sheet.x, sheet.y):
        refill_combo(combo, [(tr('Не задан'), None), *numeric_choices])
    sheet.method = QComboBox()
    for label, method in [
        (tr('Линейная'), "linear"),
        (tr('Ближайшее'), "nearest"),
        ("PCHIP", "pchip"),
        (tr('Кубический сплайн'), "spline"),
    ]:
        sheet.method.addItem(label, method)
    sheet.gap = QDoubleSpinBox()
    sheet.gap.setRange(0.000001, 1e12)
    sheet.gap.setDecimals(6)
    sheet.gap.setValue(100)
    sheet.min_anchors = QSpinBox()
    sheet.min_anchors.setRange(minimum_anchor_count(sheet.method.currentData()), 1_000_000)
    sheet.duplicates = QComboBox()
    sheet.duplicates.addItem(tr('Остановить расчёт'), "reject")
    sheet.duplicates.addItem(tr('Исключить все строки с повторяющимся X'), "exclude")
    sheet.duplicates.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    sheet.duplicates.setMinimumContentsLength(12)
    sheet.extrapolate = QCheckBox(tr('Линейная'))
    sheet.extrapolation_limit = QDoubleSpinBox()
    sheet.extrapolation_limit.setDecimals(6)
    sheet.extrapolation_limit.setRange(0.000001, 1e12)
    sheet.extrapolation_limit.setValue(10)
    sheet.extrapolation_limit.setEnabled(False)
    sheet.name = QLineEdit(tr('Интерполированное значение'))
    for label, widget in [
        (tr('Координата X'), sheet.x),
        (tr('Показатель Y'), sheet.y),
        (tr('Метод'), sheet.method),
        (tr('Макс. разрыв (единицы X)'), sheet.gap),
        (tr('Новая колонка'), sheet.name),
    ]:
        form.addRow(label, widget)
    sheet.name_error = QLabel()
    sheet.name_error.setWordWrap(True)
    sheet.name_error.hide()
    form.addRow(sheet.name_error)
    sheet.name.textChanged.connect(lambda: sheet.set_name_error())
    settings.addWidget(inputs)
    return form, advanced_form, numeric_choices


def _advanced_parameters(sheet, advanced, advanced_form):
    for label, widget in [
        (tr('Минимум опор'), sheet.min_anchors),
        (tr('Повторяющиеся координаты X'), sheet.duplicates),
        (tr('Экстраполяция'), sheet.extrapolate),
        (tr('Предел экстраполяции (единицы X)'), sheet.extrapolation_limit),
    ]:
        caption = QLabel(label)
        caption.setWordWrap(True)
        caption.setBuddy(widget)
        advanced_form.addRow(caption, widget)
    advanced.addLayout(advanced_form)


def _binding_controls(sheet, settings, form, advanced, advanced_form, numeric_choices):
    project = sheet.project
    note = QLabel(tr('Дополнительные поля разделяют пробы внутри ствола.'))
    note.setWordWrap(True)
    grouping = QGroupBox(tr('Группировка'))
    grouping_layout = QVBoxLayout(grouping)
    grouping_layout.addWidget(note)
    sheet.groups = QListWidget()
    sheet.groups.setAccessibleName(tr('Группировка'))
    sheet.groups.setFixedHeight(90)
    for column in project.columns:
        item = QListWidgetItem(column.name)
        item.setData(Qt.ItemDataRole.UserRole, column.id)
        item.setCheckState(Qt.CheckState.Unchecked)
        sheet.groups.addItem(item)
    grouping_layout.addWidget(sheet.groups)
    settings.addWidget(grouping)
    sheet.well = QComboBox()
    sheet.well.addItem(sheet.default_well_label(), None)
    for column in project.columns:
        sheet.well.addItem(column.name, column.id)
    sheet.well.setCurrentIndex(max(0, sheet.well.findData(sheet.default_well())))
    form.insertRow(3, tr('Ствол'), sheet.well)
    sheet.without_well = QCheckBox(tr('Расчёт без разделения по стволам'))
    form.insertRow(4, sheet.without_well)
    sheet.interval_bottom = QComboBox()
    refill_combo(sheet.interval_bottom, [(tr('Не задан'), None), *numeric_choices])
    for combo in (sheet.x, sheet.y, sheet.well, sheet.interval_bottom):
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(12)
    advanced_form.addRow(tr('Колонка подошвы'), sheet.interval_bottom)
    sheet.boundaries = QListWidget()
    sheet.boundaries.setAccessibleName(tr('Границы стратиграфии'))
    sheet.boundaries.setMaximumHeight(100)
    sheet._boundary_choices = None
    sheet.refresh_boundaries()
    advanced.addWidget(sheet.boundaries)
    safety = QLabel(tr('Аналитические контексты разделяются автоматически. Для смешанной выборки укажите подошву. Для исключений QC выберите область «Интерполяция».'))
    safety.setWordWrap(True)
    advanced.addWidget(safety)


def _run_button(sheet, layout):
    sheet.preview_button = QPushButton(icon('calculator'), tr('Рассчитать предпросмотр'))
    sheet.preview_button.clicked.connect(sheet.calculate)
    layout.addWidget(sheet.preview_button)


def _archive_controls(sheet, archive):
    sheet.saved_runs = QComboBox()
    sheet.saved_runs.setAccessibleName(tr('Сохранённые расчёты'))
    sheet.replay_button = QPushButton(icon('shield-check'), tr('Проверить снимок'))
    sheet.replay_button.clicked.connect(sheet.replay_saved)
    sheet.load_run_button = QPushButton(icon('folder-open'), tr('Загрузить параметры'))
    sheet.load_run_button.clicked.connect(sheet.load_saved_parameters)
    archive.addWidget(sheet.saved_runs)
    archive.addWidget(sheet.replay_button)
    archive.addWidget(sheet.load_run_button)
    sheet.replay_status = QLabel()
    sheet.replay_status.setTextFormat(Qt.TextFormat.PlainText)
    sheet.replay_status.setWordWrap(True)
    sheet.replay_status.hide()
    sheet.saved_runs.currentIndexChanged.connect(sheet.replay_status.hide)
    archive.addWidget(sheet.replay_status)
    archive_note = QLabel(tr('Проверка воспроизводит исторический снимок. Для расчёта по текущим данным загрузите параметры и запустите предпросмотр.'))
    archive_note.setWordWrap(True)
    archive.addWidget(archive_note)
    archive.addStretch()
    sheet.refresh_saved_runs()


def _proposals_view(sheet):
    sheet.preview = SampleNavigationTable()
    sheet.preview_model = InterpolationPreviewModel(sheet.preview)
    sheet.preview.setModel(sheet.preview_model)
    sheet.preview.setAccessibleName(tr('Предлагаемые значения'))
    sheet.preview.horizontalHeader().setStretchLastSection(True)
    for column, width in enumerate((110, 100, 110, 150, 100, 180)):
        sheet.preview.setColumnWidth(column, width)
    if hasattr(sheet.owner, "open_source_cell"):
        sheet.preview.sample_requested.connect(sheet.open_proposal_sample)
        sheet.preview.setToolTip(tr('Enter или двойной щелчок открывает исходную ячейку пробы.'))
    sheet.preview_tabs = QTabWidget()
    sheet.preview_tabs.setAccessibleName(tr('Результаты'))
    sheet.preview_tabs.addTab(sheet.preview, icon('table-2'), tr('Предлагаемые значения'))


def _chart_view(sheet, layout):
    sheet.chart_view = QChartView()
    sheet.chart_view.setAccessibleName(tr('Сравнение с измерениями'))
    sheet.chart_view.setAccessibleDescription(tr('Откройте «Данные графика», чтобы прочитать значения в таблице.'))
    sheet.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
    if hasattr(sheet.owner, "open_source_cell"):
        sheet.chart_view.setToolTip(tr('Нажмите на точку, чтобы открыть исходную пробу.'))
    sheet.preview_tabs.addTab(sheet.chart_view, icon('chart-no-axes-combined'), tr('Сравнение с измерениями'))
    sheet.chart_well = QComboBox()
    sheet.chart_well.setAccessibleName(tr('Ствол или группа'))
    sheet.chart_well.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    sheet.chart_well.setMinimumContentsLength(24)
    sheet.chart_well.currentIndexChanged.connect(sheet.update_chart_filter)
    sheet.chart_well.hide()
    layout.addWidget(sheet.chart_well)
    sheet.chart_excluded = QCheckBox(tr('Показать исключённые измерения'))
    sheet.chart_excluded.toggled.connect(sheet.update_chart_filter)
    sheet.chart_excluded.hide()
    layout.addWidget(sheet.chart_excluded)
    sheet.chart_table_button = QPushButton(icon('table-2'), tr('Данные графика…'))
    sheet.chart_table_button.clicked.connect(sheet.show_chart_data)
    sheet.chart_table_button.hide()
    layout.addWidget(sheet.chart_table_button)


def _diagnostic_view(sheet, layout):
    sheet.diagnostics = SampleNavigationTable()
    sheet.diagnostics.setAccessibleName(tr('Ограничения'))
    sheet.diagnostic_model = InterpolationDiagnosticsModel(
        [tr('Проба'), tr('Ствол'), "X", tr('Статус'), tr('Причина'), tr('UUID пробы')], [], sheet,
    )
    sheet.diagnostics.setModel(sheet.diagnostic_model)
    for column, width in enumerate((115, 105, 75, 170, 150, 270)):
        sheet.diagnostics.setColumnWidth(column, width)
    sheet._diagnostic_column_id = None
    sheet._diagnostics_current = False
    can_navigate = hasattr(sheet.owner, "open_source_cell")
    if can_navigate:
        sheet.diagnostics.sample_requested.connect(sheet.open_diagnostic_sample)
    diagnostic_tab = QWidget()
    diagnostic_layout = QVBoxLayout(diagnostic_tab)
    sheet.only_excluded = QCheckBox(tr('Только исключённые'))
    sheet._diagnostic_rows, sheet._excluded_rows = [], []
    sheet.only_excluded.toggled.connect(sheet.show_diagnostics)
    diagnostic_layout.addWidget(sheet.only_excluded)
    sheet.diagnostic_status = QComboBox()
    sheet.diagnostic_status.setAccessibleName(tr('Статус'))
    sheet.diagnostic_status.addItem(tr('Все статусы'), None)
    sheet.diagnostic_status.currentIndexChanged.connect(sheet.show_diagnostics)
    diagnostic_layout.addWidget(sheet.diagnostic_status)
    sheet.diagnostic_count = QLabel(tr('Показано проб: {0} из {1}', 0, 0))
    sheet.diagnostic_count.setWordWrap(True)
    diagnostic_layout.addWidget(sheet.diagnostic_count)
    sheet.export_diagnostics_button = QPushButton(icon('log-out'), tr('Экспорт диагностики CSV…'))
    sheet.export_diagnostics_button.setToolTip(tr('В CSV попадут строки, видимые с текущими фильтрами.'))
    sheet.export_diagnostics_button.setEnabled(False)
    sheet.export_diagnostics_button.clicked.connect(sheet.export_diagnostics)
    diagnostic_layout.addWidget(sheet.export_diagnostics_button)
    sheet.export_status = QLabel()
    sheet.export_status.setTextFormat(Qt.TextFormat.PlainText)
    sheet.export_status.setWordWrap(True)
    sheet.export_status.hide()
    diagnostic_layout.addWidget(sheet.export_status)
    sheet.diagnostic_hint = None
    if can_navigate:
        sheet.diagnostic_hint = QLabel(tr('Enter или двойной щелчок открывает исходную ячейку пробы.'))
        sheet.diagnostic_hint.setWordWrap(True)
        sheet.diagnostic_hint.hide()
        diagnostic_layout.addWidget(sheet.diagnostic_hint)
    diagnostic_layout.addWidget(sheet.diagnostics)
    sheet.preview_tabs.addTab(diagnostic_tab, icon('circle-question-mark'), tr('Ограничения'))
    sheet.preview_tabs.currentChanged.connect(sheet.render_pending_chart)
    layout.addWidget(sheet.preview_tabs)


def _commands(sheet, layout):
    sheet.summary = QLabel(
        tr('Будут заполнены только внутренние пропуски. Исходный Y сохраняется.')
    )
    sheet.summary.setWordWrap(True)
    layout.addWidget(sheet.summary)
    sheet.validate_button = QPushButton(icon('files'), tr('Сравнить методы на опорах…'))
    sheet.validate_button.clicked.connect(sheet.validate_methods)
    layout.addWidget(sheet.validate_button)
    sheet.validation_status = QLabel()
    sheet.validation_status.setWordWrap(True)
    sheet.validation_status.hide()
    layout.addWidget(sheet.validation_status)
    add_parameter_tools(sheet, reset=sheet.restore_parameters, layout=layout,
                        exclude=(sheet.chart_well, sheet.chart_excluded, sheet.saved_runs, sheet.diagnostic_status))
    sheet.buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    create_column = sheet.buttons.button(QDialogButtonBox.StandardButton.Ok)
    create_column.setText(tr('Создать колонку'))
    create_column.setIcon(icon('columns-3'))
    create_column.setEnabled(False)
    close = sheet.buttons.button(QDialogButtonBox.StandardButton.Cancel)
    close.setText(tr('Закрыть'))
    close.setIcon(icon('x'))
    sheet.buttons.accepted.connect(sheet.apply)
    sheet.buttons.rejected.connect(sheet.reject)
    layout.addWidget(sheet.buttons)


def _connect_inputs(sheet):
    for widget in (sheet.x, sheet.y, sheet.method):
        widget.currentIndexChanged.connect(sheet.invalidate)
    sheet.gap.valueChanged.connect(sheet.invalidate)
    sheet.method.currentIndexChanged.connect(sheet.update_method_controls)
    sheet.min_anchors.valueChanged.connect(sheet.invalidate)
    sheet.duplicates.currentIndexChanged.connect(sheet.invalidate)
    sheet.extrapolate.toggled.connect(sheet.update_extrapolation_controls)
    sheet.extrapolate.toggled.connect(sheet.invalidate)
    sheet.extrapolation_limit.valueChanged.connect(sheet.invalidate)
    sheet.groups.itemChanged.connect(sheet.invalidate)
    sheet.well.currentIndexChanged.connect(sheet.invalidate)
    sheet.without_well.toggled.connect(sheet.invalidate)
    sheet.interval_bottom.currentIndexChanged.connect(sheet.invalidate)
    sheet.boundaries.itemChanged.connect(sheet.invalidate)
    if sheet.owner is not None and hasattr(sheet.owner, 'analysis_scope'):
        sheet.owner.analysis_scope.bind(sheet.scope, sheet.invalidate)
    else:
        sheet.scope.addItem(tr('Текущая таблица'), 'filtered')
        sheet.scope.setEnabled(False)


def _set_tab_order(sheet):
    for controls in (
        (sheet.scope, sheet.x, sheet.y, sheet.well, sheet.without_well, sheet.method, sheet.gap, sheet.name, sheet.groups),
        (sheet.min_anchors, sheet.duplicates, sheet.extrapolate, sheet.extrapolation_limit,
         sheet.interval_bottom, sheet.boundaries),
        (sheet.only_excluded, sheet.diagnostic_status, sheet.export_diagnostics_button, sheet.diagnostics),
    ):
        order = []
        for control in controls:
            order.append(control)
            if reset := getattr(control, "parameter_reset_button", None):
                order.append(reset)
        for first, second in zip(order, order[1:]):
            QWidget.setTabOrder(first, second)
