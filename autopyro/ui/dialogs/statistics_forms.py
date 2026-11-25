"""Build the statistics workspace and connect its controls."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..i18n import tr
from ..plot_export import PlotExport
from ..table.view import PinnedTable
from ..theme import icon
from ..widgets.documents import analysis_document, wrapped_check
from ..widgets.groups import group_layout
from ..widgets.parameters import add_parameter_tools, ParameterDefaults


def build(sheet):
    control_layout, layout = analysis_document(sheet)
    _scope_controls(sheet, control_layout)
    policy_anchor = _policy_controls(sheet, control_layout)
    properties_label, groups_label = _property_controls(sheet, control_layout)
    result_toolbar = _result_tables(sheet, layout)
    _result_actions(sheet, layout)
    _chart_controls(sheet, layout)
    _sample_controls(sheet, layout)
    _status_controls(sheet, layout)
    _export_controls(sheet, result_toolbar, control_layout)
    _connect_and_restore(sheet)
    group_layout(control_layout, [
        (policy_anchor, tr('Обработка данных')),
        (properties_label, tr('Показатели')),
        (groups_label, tr('Группировка')),
        (sheet.refresh, None),
    ])
    add_parameter_tools(sheet, defaults=sheet._initial_parameters, reset=sheet.restore_parameters, layout=control_layout)
    control_layout.addStretch()
    sheet.owner.analysis_scope.bind(sheet.scope, sheet.request_refresh)


def _scope_controls(sheet, control_layout):
    owner = sheet.owner
    form = QFormLayout()
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    sheet.scope = QComboBox()
    sheet.scope.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    sheet.scope.setMinimumContentsLength(12)
    for label, mode in [
        (tr('Все пробы проекта'), "all"),
        (tr('Текущая отфильтрованная таблица'), "filtered"),
        (tr('Выделенные строки (любая выбранная ячейка)'), "selected"),
    ]:
        sheet.scope.addItem(label, mode)
    for name in owner.model.project.view.get("named", {}):
        sheet.scope.addItem(tr('Представление: {0}', name), f"view:{name}")
    sheet.scope.setCurrentIndex(1)
    sheet.min_pairs = QSpinBox()
    sheet.min_pairs.setRange(3, 1000000)
    sheet.min_pairs.setValue(3)
    form.addRow(tr('Область'), sheet.scope)
    form.addRow(tr('Минимум пар'), sheet.min_pairs)
    control_layout.addLayout(form)


def _policy_controls(sheet, control_layout):
    sheet.text_numbers = QCheckBox(tr('Читать числа в текстовых колонках'))
    sheet.text_numbers.setToolTip(
        tr('Только в расчёте: 1,2 → 1.2. <LOD и ND остаются исключёнными. Таблица не меняется.')
    )
    sheet.interpolated = QCheckBox(tr('Включить актуальные интерполяции отдельно'))
    sheet.interpolated.setToolTip(
        tr('Интерполированные колонки считаются отдельно, без заполнения исходных. Устаревшие всегда исключены.')
    )
    sheet.correlations = QCheckBox(tr('Рассчитывать корреляции (до 30 показателей)'))
    sheet.correlations.setChecked(True)
    policies = []
    for widget in (sheet.text_numbers, sheet.interpolated, sheet.correlations):
        tooltip = widget.toolTip()
        policies.append(wrapped_check(control_layout, widget))
        if tooltip:
            widget.setToolTip(tooltip)
    return policies[0]


def _property_controls(sheet, control_layout):
    owner = sheet.owner
    properties_label = QLabel(tr('Показатели (единицы не пересчитываются)'))
    control_layout.addWidget(properties_label)
    sheet.properties = QListWidget()
    sheet.groups = QListWidget()
    sheet.properties.setMinimumHeight(150)
    sheet.groups.setMinimumHeight(100)
    search = sheet.search = QLineEdit()
    search.setPlaceholderText(tr('Найти показатель…'))
    search.textChanged.connect(sheet.search_properties)
    control_layout.addWidget(search)
    for column in owner.model.project.columns:
        label = column.name + (f" [{column.unit}]" if column.unit else "")
        if column.metadata.get("method"):
            label += tr(' · интерполяция') + (
                tr(' (устарела)') if column.metadata.get("stale") else ""
            )
        for target, checked in (
            (
                sheet.properties,
                column.kind == "number" and not column.metadata.get("method"),
            ),
            (sheet.groups, False),
        ):
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, column.id)
            item.setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )
            target.addItem(item)
    control_layout.addWidget(sheet.properties, 2)
    groups_label = QLabel(tr('Группировка (пустая категория сохраняется)'))
    control_layout.addWidget(groups_label)
    control_layout.addWidget(sheet.groups, 1)
    sheet.refresh = QPushButton(icon('refresh-cw'), tr('Пересчитать'))
    sheet.refresh.clicked.connect(sheet.request_refresh)
    control_layout.addWidget(sheet.refresh)
    return properties_label, groups_label


def _result_tables(sheet, layout):
    sheet.tabs = QTabWidget()
    sheet.tables = {}
    for name, label, image in [
        ("summary", tr('Сводка'), 'table-2'),
        ("availability", tr('Заполненность'), 'columns-3'),
        ("correlations", tr('Корреляции'), 'chart-no-axes-combined'),
        ("outliers", tr('Потенциальные выбросы'), 'shield-check'),
    ]:
        table = PinnedTable()
        table.setAlternatingRowColors(True)
        table.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        sheet.tables[name] = table
        sheet.tabs.addTab(table, icon(image), label)
    sheet.add_comparison_tab()
    repeats = sheet.tables["repeats"] = PinnedTable()
    repeats.setAlternatingRowColors(True)
    repeats.doubleClicked.connect(sheet.show_repeat_samples)
    sheet.tabs.addTab(repeats, icon('flask-conical'), tr('Повторные анализы'))
    thresholds = sheet.tables["thresholds"] = PinnedTable()
    thresholds.setAlternatingRowColors(True)
    thresholds.doubleClicked.connect(sheet.show_threshold_samples)
    sheet.tabs.addTab(thresholds, icon('shield-check'), tr('Пороги и ND'))
    interpolation = sheet.tables["interpolation"] = PinnedTable()
    interpolation.setAlternatingRowColors(True)
    interpolation.doubleClicked.connect(sheet.show_interpolation_samples)
    sheet.tabs.addTab(interpolation, icon('files'), tr('Сравнение с интерполяцией'))
    sheet.tab_picker = QComboBox()
    sheet.tab_picker.setAccessibleName(tr('Раздел статистики'))
    sheet.tab_picker.addItems([sheet.tabs.tabText(index) for index in range(sheet.tabs.count())])
    sheet.tabs.tabBar().hide()
    sheet.tab_picker.currentIndexChanged.connect(sheet.tabs.setCurrentIndex)
    sheet.tabs.currentChanged.connect(sheet.tab_picker.setCurrentIndex)
    result_toolbar = QHBoxLayout()
    result_toolbar.addWidget(sheet.tab_picker, 1)
    layout.addLayout(result_toolbar)
    for table in sheet.tables.values():
        table.frozen.doubleClicked.connect(table.doubleClicked)
        table.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        table.setSortingEnabled(True)
    sheet.result_group = QComboBox()
    sheet.result_group.setAccessibleName(tr('Группа'))
    sheet.result_group_values = []
    sheet.result_group_fields = ()
    sheet.result_group.currentIndexChanged.connect(sheet.filter_result_groups)
    sheet.result_group.hide()
    layout.addWidget(sheet.result_group)
    layout.addWidget(sheet.tabs, 1)
    sheet.tables["summary"].doubleClicked.connect(sheet.show_distribution)
    sheet.tables["availability"].doubleClicked.connect(sheet.show_availability_samples)
    sheet.tables["correlations"].doubleClicked.connect(sheet.show_correlation_matrix)
    sheet.tables["outliers"].doubleClicked.connect(sheet.go_to_outlier)
    return result_toolbar


def _result_actions(sheet, layout):
    sheet.navigation_hint = QLabel()
    sheet.navigation_hint.setWordWrap(True)
    sheet.navigation_hint.hide()
    layout.addWidget(sheet.navigation_hint)
    sheet.result_actions = {
        "summary": (tr('Показать распределение'), sheet.show_distribution, 'chart-column'),
        "correlations": (tr('Открыть тепловую карту'), sheet.show_correlation_matrix, 'chart-no-axes-combined'),
        "outliers": (tr('Показать пробу в таблице'), sheet.go_to_outlier, 'table-2'),
        "availability": (tr('Показать пробы'), sheet.show_availability_samples, 'table-2'),
        "comparison": (tr('Показать пробы'), sheet.show_comparison_samples, 'table-2'),
        "repeats": (tr('Показать пробы'), sheet.show_repeat_samples, 'table-2'),
        "thresholds": (tr('Показать пробы'), sheet.show_threshold_samples, 'table-2'),
        "interpolation": (tr('Показать пробы'), sheet.show_interpolation_samples, 'table-2'),
    }
    sheet.result_action_button = QPushButton()
    sheet.result_action_button.clicked.connect(sheet.activate_result_action)
    layout.addWidget(sheet.result_action_button, alignment=Qt.AlignmentFlag.AlignLeft)
    sheet.pin_identifiers = QCheckBox(tr('Закрепить идентифицирующие колонки'))
    sheet.pin_identifiers.setChecked(True)
    sheet.pin_identifiers.toggled.connect(sheet.update_result_pins)
    layout.addWidget(sheet.pin_identifiers)


def _chart_controls(sheet, layout):
    sheet.scatter_matrix = QPushButton(icon('chart-no-axes-combined'), tr('Матрица диаграмм рассеяния'))
    sheet.scatter_matrix.clicked.connect(sheet.show_scatter_matrix)
    sheet.scatter_matrix.hide()
    layout.addWidget(sheet.scatter_matrix, alignment=Qt.AlignmentFlag.AlignLeft)
    sheet.matrix_artifact = None
    sheet.save_matrix_button = QPushButton(icon('save'), tr('Сохранить матрицу…'))
    sheet.save_matrix_button.clicked.connect(sheet.save_matrix)
    sheet.save_matrix_button.hide()
    layout.addWidget(sheet.save_matrix_button, alignment=Qt.AlignmentFlag.AlignLeft)
    sheet.load_matrix_style_button = QPushButton(icon('folder-open'), tr('Загрузить оформление'))
    sheet.load_matrix_style_button.clicked.connect(sheet.load_matrix_style)
    sheet.load_matrix_style_button.hide()
    layout.addWidget(sheet.load_matrix_style_button, alignment=Qt.AlignmentFlag.AlignLeft)
    sheet.matrix_export = PlotExport()
    sheet.matrix_export_progress = QProgressDialog(tr('Выполняется фоновый экспорт…'), '', 0, 0, sheet)
    sheet.matrix_export_progress.setCancelButton(QPushButton(icon('x'), tr('Отмена')))
    sheet.matrix_export_progress.setWindowTitle(tr('Экспорт'))
    sheet.matrix_export_progress.canceled.connect(sheet.matrix_export.cancel)
    sheet.matrix_export.busyChanged.connect(sheet.matrix_export_busy_changed)
    sheet.matrix_export.completed.connect(sheet.matrix_export_completed)
    sheet.matrix_export.failed.connect(sheet.matrix_export_failed)
    sheet.load_matrix_button = QPushButton(icon('folder-open'), tr('Открыть матрицу…'))
    sheet.load_matrix_button.clicked.connect(sheet.load_matrix)
    sheet.load_matrix_button.hide()
    layout.addWidget(sheet.load_matrix_button, alignment=Qt.AlignmentFlag.AlignLeft)
    sheet.distribution_view = QScrollArea()
    sheet.distribution_view.setWidgetResizable(True)
    sheet.distribution_view.hide()
    layout.addWidget(sheet.distribution_view, 1)


def _sample_controls(sheet, layout):
    sheet.sample_label = QLabel()
    sheet.sample_label.setTextFormat(Qt.TextFormat.PlainText)
    sheet.sample_label.setWordWrap(True)
    sheet.sample_label.hide()
    layout.addWidget(sheet.sample_label)
    sheet.sample_table = QTableView()
    sheet.sample_table.setAlternatingRowColors(True)
    sheet.sample_table.doubleClicked.connect(sheet.go_to_listed_sample)
    sheet.sample_table.hide()
    layout.addWidget(sheet.sample_table, 1)
    sheet.sample_action_button = QPushButton(icon('table-2'), tr('Показать пробу в таблице'))
    sheet.sample_action_button.clicked.connect(lambda: sheet.go_to_listed_sample(sheet.sample_table.currentIndex()))
    sheet.sample_action_button.hide()
    layout.addWidget(sheet.sample_action_button, alignment=Qt.AlignmentFlag.AlignLeft)


def _status_controls(sheet, layout):
    owner = sheet.owner
    sheet.status = QLabel(tr('Выберите показатели'))
    sheet.status.setTextFormat(Qt.TextFormat.PlainText)
    sheet.previous_sample_filter = None
    sheet.restore_sample_filter_button = QPushButton(icon('rotate-ccw'), tr('Вернуть прежний фильтр'))
    sheet.restore_sample_filter_button.clicked.connect(sheet.restore_sample_filter)
    owner.statusBar().addPermanentWidget(sheet.restore_sample_filter_button)
    sheet.restore_sample_filter_button.hide()
    sheet.destroyed.connect(sheet.restore_sample_filter_button.deleteLater)
    owner.proxy.filtersChanged.connect(sheet.clear_sample_filter_restore)
    sheet.status.setWordWrap(True)
    sheet.policy = QLabel()
    sheet.policy.setTextFormat(Qt.TextFormat.PlainText)
    sheet.policy.setWordWrap(True)
    layout.addWidget(sheet.status)
    layout.addWidget(sheet.policy)
    sheet.stale_results = QLabel(tr('Предыдущие результаты доступны только для просмотра.'))
    sheet.stale_results.setWordWrap(True)
    sheet.stale_results.hide()
    layout.addWidget(sheet.stale_results)
    sheet.cancel_calculation_button = QPushButton(icon('x'), tr('Отменить расчёт'))
    sheet.cancel_calculation_button.setEnabled(False)
    sheet.cancel_calculation_button.clicked.connect(sheet.cancel_calculation)
    layout.addWidget(sheet.cancel_calculation_button, alignment=Qt.AlignmentFlag.AlignLeft)


def _export_controls(sheet, result_toolbar, control_layout):
    sheet.copy = QPushButton(icon('copy'), tr('Копировать'))
    sheet.export = QPushButton(icon('table-2'), 'CSV…')
    sheet.export_json = QPushButton(icon('file-text'), 'JSON…')
    for button, label, callback in [
        (sheet.copy, tr('Копировать вкладку'), sheet.copy_current),
        (sheet.export, tr('Экспорт вкладки CSV…'), sheet.export_current),
        (sheet.export_json, tr('Снимок рассчитанных разделов JSON…'), sheet.export_snapshot),
    ]:
        button.setAccessibleName(label)
        button.setToolTip(label)
        button.clicked.connect(callback)
        result_toolbar.addWidget(button)
    close = QPushButton(icon('x'), tr('Закрыть'))
    close.clicked.connect(sheet.close)
    control_layout.addWidget(close)


def _connect_and_restore(sheet):
    owner = sheet.owner
    sheet._initial_parameters = ParameterDefaults(sheet)
    sheet.restore_settings(owner.model.project.view.get("statistics", {}))
    for widget in (sheet.scope, *sheet.comparison_views):
        widget.currentIndexChanged.connect(sheet.request_refresh)
    sheet.min_pairs.valueChanged.connect(sheet.request_policy_refresh)
    for widget in (sheet.text_numbers, sheet.interpolated, sheet.correlations):
        widget.toggled.connect(sheet.request_policy_refresh)
    sheet.properties.itemChanged.connect(sheet.request_refresh)
    sheet.groups.itemChanged.connect(sheet.request_refresh)
    owner.model.changed.connect(sheet.request_refresh)
    owner.model.columnsInserted.connect(sheet.sync_columns)
    owner.model.columnsRemoved.connect(sheet.sync_columns)
    owner.proxy.layoutChanged.connect(sheet.request_refresh)
    owner.proxy.modelReset.connect(sheet.request_refresh)
    owner.proxy.filtersChanged.connect(sheet.request_refresh)
    owner.table.selectionModel().selectionChanged.connect(sheet.selection_changed)
    owner.analysis_scope.saved_scopes_changed.connect(sheet.sync_comparison_views)
    owner.analysis_scope.saved_scope_renamed.connect(sheet.rename_comparison_scope)
    sheet.tabs.currentChanged.connect(sheet.tab_changed)
    sheet.tab_changed()
    sheet.request_refresh()


def add_comparison_tab(sheet):
    page = QWidget()
    layout = QVBoxLayout(page)
    form = QFormLayout()
    sheet.comparison_views = (QComboBox(), QComboBox())
    for combo in sheet.comparison_views:
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(24)
    form.addRow(tr('Выборка A'), sheet.comparison_views[0])
    form.addRow(tr('Выборка B'), sheet.comparison_views[1])
    layout.addLayout(form)
    sheet.fill_comparison_views((None, None))
    sheet.comparison_status = QLabel()
    sheet.comparison_status.setWordWrap(True)
    layout.addWidget(sheet.comparison_status)
    table = sheet.tables["comparison"] = PinnedTable()
    table.setAlternatingRowColors(True)
    table.doubleClicked.connect(sheet.show_comparison_samples)
    layout.addWidget(table, 2)
    sheet.comparison_context_hint = QLabel()
    sheet.comparison_context_hint.setTextFormat(Qt.TextFormat.PlainText)
    sheet.comparison_context_hint.setWordWrap(True)
    sheet.comparison_context_hint.hide()
    layout.addWidget(sheet.comparison_context_hint)
    sheet.tabs.addTab(page, icon('files'), tr('Сравнение'))
