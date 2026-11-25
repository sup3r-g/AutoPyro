"""Build the interpretation worksheet and connect its controls."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QTableView,
    QTextBrowser,
    QSplitter,
    QToolButton,
)

from ..i18n import tr
from ..theme import icon
from ..theme.controls import apply_control_icons
from ..widgets.documents import analysis_document, parameter_page
from ..widgets.groups import group_layout
from ..widgets.parameters import add_parameter_tools, ParameterDefaults
from .interpretation_results import RESULT_FILTERS
from .interpretation_schemes import restore_selection


RESULT_FILTER_ICONS = {
    'review': 'shield-check',
    'review_pending': 'clock',
    'review_open': 'circle-question-mark',
    'review_stale': 'refresh-cw',
    'ambiguous': 'circle-question-mark',
    'boundary': 'maximize',
    'quality': 'shield-check',
    'outside': 'map',
    'not_applicable': 'x',
    'disagreement': 'files',
}


def build(sheet):
    control_layout, layout = analysis_document(sheet)
    scheme_action = _scheme_controls(sheet, control_layout)
    form = _binding_controls(sheet, control_layout)
    _filter_controls(sheet, control_layout)
    editor_layout, note = _scheme_editor(sheet)
    _results_view(sheet, layout)
    _result_actions(sheet, control_layout, layout)
    _export_controls(sheet, control_layout)
    _calculation_controls(sheet, control_layout, layout)
    sheet.sync_columns()
    sheet._initial_parameters = ParameterDefaults(sheet, exclude=(sheet.editor, sheet.explanation))
    restore_selection(sheet)
    _connect_inputs(sheet)
    sheet.create_plot()
    sheet.schedule()
    group_layout(control_layout, [(scheme_action, tr('Схема')), (form, None)])
    group_layout(editor_layout, [(note, tr('Редактор схемы'))])
    add_parameter_tools(sheet, defaults=sheet._initial_parameters, changed=sheet.schedule, layout=control_layout)
    control_layout.addStretch()
    sheet.owner.analysis_scope.bind(sheet.scope, sheet.schedule)
    apply_control_icons(sheet)


def _scheme_controls(sheet, control_layout):
    scheme_actions = []
    for label, callback, image in [
        (tr('Загрузить схему JSON…'), sheet.load_scheme, 'folder-open'),
        (tr('Сохранить JSON…'), sheet.save_scheme, 'save'),
    ]:
        button = QPushButton(icon(image), label)
        button.clicked.connect(callback)
        control_layout.addWidget(button)
        scheme_actions.append(button)
    for label, callback, image in [
        (tr('Новая схема…'), sheet.new_graphical_scheme, 'file-plus'),
        (tr('Графический редактор…'), sheet.open_geometry_editor, 'mouse-pointer-2'),
    ]:
        button = QPushButton(icon(image), label)
        button.clicked.connect(callback)
        control_layout.addWidget(button)
    from autopyro.schemes.standard_interpretation import standard_schemes
    sheet.standard_schemes = standard_schemes()
    sheet.scheme_group = QComboBox()
    sheet.scheme_group.setAccessibleName(tr('Группа диаграмм'))
    sheet.scheme_group.addItem(tr('Основные схемы'), None)
    for group in sorted({scheme['group'] for scheme in sheet.standard_schemes.values() if 'group' in scheme}):
        sheet.scheme_group.addItem(group, group)
    control_layout.addWidget(sheet.scheme_group)
    sheet.standard_choice = QComboBox()
    sheet.standard_choice.setAccessibleName(tr('Схема из каталога'))
    sheet.standard_choice.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    sheet.standard_choice.setMinimumContentsLength(20)
    control_layout.addWidget(sheet.standard_choice)
    sheet.catalog_preview = QLabel()
    sheet.catalog_preview.setTextFormat(Qt.TextFormat.PlainText)
    sheet.catalog_preview.setWordWrap(True)
    control_layout.addWidget(sheet.catalog_preview)
    sheet.apply_catalog_scheme = QPushButton(icon('check'), tr('Применить схему из каталога'))
    sheet.apply_catalog_scheme.setAutoDefault(False)
    sheet.apply_catalog_scheme.clicked.connect(sheet.load_standard_scheme)
    control_layout.addWidget(sheet.apply_catalog_scheme)
    sheet.standard_choice.currentIndexChanged.connect(sheet.update_catalog_preview)
    sheet.scheme_group.currentIndexChanged.connect(sheet.fill_standard_schemes)
    sheet.fill_standard_schemes()
    return scheme_actions[0]


def _binding_controls(sheet, control_layout):
    owner = sheet.owner
    sheet.comparison = QCheckBox(tr('Показать стандартные схемы сравнения'))
    sheet.comparison.setToolTip(tr(
        'Добавляет стандартные HI–Tmax, HI–OI и S2–TOC. Согласие типов считается только по стандартным HI–Tmax и HI–OI.'))
    sheet.comparison.setChecked(bool(owner.model.project.view.get('interpretation_comparison', False)))
    control_layout.addWidget(sheet.comparison)
    sheet.comparison_view = QComboBox()
    sheet.comparison_view.setAccessibleName(tr('Диаграммы в предпросмотре'))
    sheet.comparison_view.hide()
    control_layout.addWidget(sheet.comparison_view)
    form = QFormLayout()
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    sheet.x, sheet.y = QComboBox(), QComboBox()
    for combo in (sheet.x, sheet.y):
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(12)
    sheet.x_unit, sheet.y_unit = QLineEdit(), QLineEdit()
    sheet.scope = QComboBox()
    sheet.scope.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    sheet.scope.setMinimumContentsLength(12)
    sheet.scope.addItem(tr('Текущая отфильтрованная таблица'), "filtered")
    sheet.scope.addItem(tr('Весь проект'), "all")
    sheet.scope.addItem(tr('Выделенные строки'), "selected")
    for label, widget in [
        ("X", sheet.x),
        (tr('Единица X'), sheet.x_unit),
        ("Y", sheet.y),
        (tr('Единица Y'), sheet.y_unit),
        (tr('Область'), sheet.scope),
    ]:
        form.addRow(label, widget)
    sheet.text_numbers = QCheckBox(tr('Читать числа в текстовых колонках'))
    form.addRow(sheet.text_numbers)
    control_layout.addLayout(form)
    sheet.scheme_info = QLabel(
        tr('Загрузите схему. Границы не генерируются автоматически.')
    )
    sheet.scheme_info.setWordWrap(True)
    sheet.scheme_info.setTextFormat(Qt.TextFormat.PlainText)
    control_layout.addWidget(sheet.scheme_info)
    return form


def _filter_controls(sheet, control_layout):
    sheet.result_filter = QComboBox()
    sheet.result_filter.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    sheet.result_filter.setMinimumContentsLength(12)
    sheet.result_filter.addItem(tr('Все пробы'), 'all')
    for mode, label in RESULT_FILTERS.items():
        if mode != 'disagreement':
            sheet.result_filter.addItem(tr(label), mode)
    sheet.result_filter.setAccessibleName(tr('Фильтр результатов'))
    sheet.result_filter.setToolTip(tr('Фильтр применяется к таблице и графику. При сравнении достаточно совпадения в одном из отчётов.'))
    sheet.result_filter.setEnabled(False)
    control_layout.addWidget(sheet.result_filter)
    sheet.extra_filters = QToolButton()
    sheet.extra_filters.setText(tr('Дополнительные условия'))
    sheet.extra_filters.setIcon(icon('funnel'))
    sheet.extra_filters.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    sheet.extra_filters.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    sheet.extra_filters.setToolTip(tr('Проба должна соответствовать каждому выбранному условию.'))
    menu = QMenu(sheet.extra_filters)
    sheet.extra_filter_actions = {}
    for mode, label in RESULT_FILTERS.items():
        action = menu.addAction(icon(RESULT_FILTER_ICONS[mode]), tr(label))
        action.setCheckable(True)
        action.setVisible(mode != 'disagreement')
        sheet.extra_filter_actions[mode] = action
    sheet.extra_filters.setMenu(menu)
    menu.triggered.connect(sheet.filter_results)
    sheet.extra_filters.setEnabled(False)
    control_layout.addWidget(sheet.extra_filters)
    sheet.filtered_count = QLabel()
    control_layout.addWidget(sheet.filtered_count)
    sheet.review_queue_count = QLabel()
    sheet.review_queue_count.setWordWrap(True)
    sheet.review_queue_count.setAccessibleName(tr('Состояние экспертной очереди'))
    sheet.review_queue_count.setToolTip(tr('Счётчики относятся ко всем рассчитанным пробам текущей области данных.'))
    sheet.review_queue_count.hide()
    control_layout.addWidget(sheet.review_queue_count)
    sheet.result_filter.currentIndexChanged.connect(sheet.filter_results)


def _scheme_editor(sheet):
    editor_layout = parameter_page(sheet.parameter_tabs, tr('Схема'), 'layers')
    note = QLabel(tr('Редактор JSON: при изменении геометрии увеличьте revision.\nКоординаты областей заданы в analysis, после преобразования осей.'))
    note.setWordWrap(True)
    editor_layout.addWidget(note)
    sheet.editor = QPlainTextEdit()
    sheet.editor.setAccessibleName(tr('Редактор JSON схемы'))
    sheet.editor.setMinimumHeight(300)
    editor_layout.addWidget(sheet.editor, 1)
    apply = QPushButton(icon('check'), tr('Применить редакцию'))
    apply.clicked.connect(sheet.apply_editor)
    editor_layout.addWidget(apply)
    check_cases = QPushButton(icon('shield-check'), tr('Проверить эталонные пробы'))
    check_cases.setToolTip(tr('Эталоны задаются в массиве reference_cases в JSON схемы.'))
    check_cases.clicked.connect(sheet.check_reference_cases)
    editor_layout.addWidget(check_cases)
    return editor_layout, note


def _results_view(sheet, layout):
    sheet.table = QTableView()
    header = sheet.table.horizontalHeader()
    header.setSectionsClickable(True)
    header.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
    header.setSortIndicatorShown(True)
    header.sortIndicatorChanged.connect(sheet.filter_results)
    sheet.table.setAlternatingRowColors(True)
    sheet.table.setWordWrap(False)
    sheet.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    sheet.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    sheet.comparison_context = QLabel()
    sheet.comparison_context.setTextFormat(Qt.TextFormat.PlainText)
    sheet.comparison_context.setWordWrap(True)
    sheet.comparison_context.hide()
    layout.addWidget(sheet.comparison_context)
    sheet.result_splitter = QSplitter(Qt.Orientation.Vertical)
    sheet.result_splitter.setChildrenCollapsible(False)
    sheet.result_splitter.setHandleWidth(7)
    sheet.result_splitter.setAccessibleName(tr('График и результаты интерпретации'))
    sheet.table.setMinimumHeight(95)
    sheet.result_splitter.addWidget(sheet.table)
    sheet.explanation = QTextBrowser()
    sheet.explanation.setAccessibleName(tr('Объяснение результата выбранной пробы'))
    sheet.explanation.setMinimumHeight(65)
    sheet.explanation.setPlainText(tr('Выберите пробу на графике или в таблице результатов.'))
    sheet.result_splitter.addWidget(sheet.explanation)
    layout.addWidget(sheet.result_splitter, 1)


def _result_actions(sheet, control_layout, layout):
    sheet.review_button = QPushButton(icon('shield-check'), tr('Экспертная проверка…'))
    sheet.review_button.setEnabled(False)
    sheet.review_button.clicked.connect(sheet.edit_review)
    sheet.summary_button = QPushButton(icon('chart-column'), tr('Сводка по стволам…'))
    sheet.summary_button.setEnabled(False)
    sheet.summary_button.clicked.connect(sheet.show_summary)
    sheet.revisions_button = QPushButton(icon('files'), tr('Сравнить расчёты…'))
    sheet.revisions_button.setEnabled(False)
    sheet.revisions_button.clicked.connect(sheet.show_revisions)
    sheet.pyrogram_button = QPushButton(icon('spline'), tr('Пирограммы…'))
    sheet.pyrogram_button.setEnabled(False)
    sheet.pyrogram_button.clicked.connect(sheet.show_pyrograms)
    result_actions = QGridLayout()
    for position, button in enumerate((sheet.review_button, sheet.pyrogram_button,
                                       sheet.summary_button, sheet.revisions_button)):
        result_actions.addWidget(button, position // 2, position % 2)
    layout.addLayout(result_actions)
    sheet.compact_results = QCheckBox(tr('Компактная таблица результатов'))
    sheet.compact_results.toggled.connect(sheet.update_result_columns)
    control_layout.addWidget(sheet.compact_results)
    sheet.previous_result_button = QPushButton(icon('logs'), tr('Просмотреть предыдущий расчёт…'))
    sheet.previous_result_button.hide()
    sheet.previous_result_button.clicked.connect(sheet.show_previous_result)
    layout.addWidget(sheet.previous_result_button)
    sheet.pin_base = QPushButton(icon('save'), tr('Закрепить текущий расчёт как базу'))
    sheet.pin_base.setCheckable(True)
    sheet.pin_base.setEnabled(False)
    sheet.pin_base.toggled.connect(sheet.set_comparison_base)
    control_layout.addWidget(sheet.pin_base)


def _export_controls(sheet, control_layout):
    sheet.export_scope = QComboBox()
    sheet.export_scope.addItem(tr('Видимые пробы'), 'visible')
    sheet.export_scope.addItem(tr('Все рассчитанные пробы'), 'all')
    sheet.export_scope.setToolTip(tr('Область выгрузки отчёта. Экспорт графика использует текущий фильтр.'))
    export_options = QFormLayout()
    export_options.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    export_options.addRow(tr('Экспорт результатов'), sheet.export_scope)
    control_layout.addLayout(export_options)
    sheet.export_json = QPushButton()
    sheet.export_csv = QPushButton()
    sheet.export_json.setIcon(icon('file-text'))
    sheet.export_csv.setIcon(icon('log-out'))
    for button in (sheet.export_json, sheet.export_csv):
        button.setToolTip(tr('Выгрузить пробы выбранной области вместе с экспертными выводами.'))
    sheet.export_scope.currentIndexChanged.connect(sheet.update_export_labels)
    sheet.update_export_labels()
    sheet.export_json.clicked.connect(sheet.save_report)
    sheet.export_csv.clicked.connect(sheet.save_csv)
    control_layout.addWidget(sheet.export_json)
    control_layout.addWidget(sheet.export_csv)


def _calculation_controls(sheet, control_layout, layout):
    refresh = QPushButton(icon('refresh-cw'), tr('Пересчитать'))
    refresh.clicked.connect(sheet.schedule)
    control_layout.addWidget(refresh)
    sheet.cancel_calculation_button = QPushButton(icon('x'), tr('Отменить расчёт'))
    sheet.cancel_calculation_button.hide()
    sheet.cancel_calculation_button.clicked.connect(sheet.cancel_calculation)
    control_layout.addWidget(sheet.cancel_calculation_button)
    sheet.status = QLabel(tr('Схема не выбрана'))
    sheet.status.setWordWrap(True)
    sheet.stale_notice = QLabel()
    sheet.stale_notice.setTextFormat(Qt.TextFormat.PlainText)
    sheet.stale_notice.setWordWrap(True)
    sheet.stale_notice.hide()
    layout.addWidget(sheet.stale_notice)
    layout.addWidget(sheet.status)


def _connect_inputs(sheet):
    owner = sheet.owner
    sheet.editor.textChanged.connect(sheet.editor_changed)
    for widget in (sheet.x, sheet.y, sheet.scope):
        widget.currentIndexChanged.connect(sheet.schedule)
    for widget in (sheet.x_unit, sheet.y_unit):
        widget.textEdited.connect(sheet.schedule)
    sheet.text_numbers.toggled.connect(sheet.schedule)
    sheet.comparison.toggled.connect(sheet.schedule)
    owner.model.changed.connect(sheet.schedule)
    owner.proxy.layoutChanged.connect(sheet.schedule)
    owner.proxy.modelReset.connect(sheet.schedule)
    owner.proxy.filtersChanged.connect(sheet.schedule)
    owner.model.columnsInserted.connect(sheet.sync_columns)
    owner.model.columnsRemoved.connect(sheet.sync_columns)
    owner.model.headerDataChanged.connect(sheet.update_catalog_preview)
    owner.table.selectionModel().selectionChanged.connect(sheet.selection_changed)
