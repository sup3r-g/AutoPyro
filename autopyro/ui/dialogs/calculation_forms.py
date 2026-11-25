"""Build the calculation worksheet and adapt it to narrow layouts."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from autopyro.analysis.calculations import FORMULAS, UNITS

from ..i18n import tr
from ..theme import icon
from ..widgets.documents import analysis_document, parameter_page, wrapped_check
from ..widgets.groups import group_layout, group_form
from ..widgets.parameters import add_parameter_tools


def build(editor):
    settings, layout = analysis_document(editor)
    _workspace_layout(editor, layout)
    _input_fields(editor, settings)
    _output_fields(editor, settings)
    _lab_fields(editor)
    note = _policy_controls(editor, settings)
    _preview_filters(editor, layout)
    _result_navigation(editor, layout)
    _results_table(editor, layout)
    _finish(editor, settings, note)


def _workspace_layout(editor, layout):
    editor._splitter = editor.parameter_tabs.parentWidget()
    editor._results_panel = layout.parentWidget()
    editor._compact_results = QScrollArea(editor.parameter_tabs)
    editor._compact_results.setWidgetResizable(True)
    editor._compact_results.hide()
    for panel in (editor.parameter_tabs, editor._results_panel):
        panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)


def _input_fields(editor, settings):
    owner = editor.owner
    form = QFormLayout()
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    editor.scope = QComboBox()
    editor.scope.addItems([tr('Текущая отфильтрованная таблица'), tr('Весь проект')])
    form.addRow(tr('Область (фиксируется при создании)'), editor.scope)
    editor.context = QLineEdit()
    editor.context.setPlaceholderText(
        tr('Например: Pyrol6, цельная порода, серия лаборатории…')
    )
    form.addRow(tr('Аналитический контекст'), editor.context)
    for field in (editor.scope, editor.context):
        form.labelForField(field).setWordWrap(True)
    editor.inputs = {}
    editor.input_feedback = {}
    editor._input_rows = []
    for role, units in UNITS.items():
        holder = QHBoxLayout()
        editor._input_rows.append(holder)
        column = QComboBox()
        column.setAccessibleName(tr('{0}: исходная колонка', role))
        column.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        column.setMinimumContentsLength(10)
        column.addItem(tr('Не выбрано'), "")
        for c in owner.model.project.columns:
            if not c.metadata.get("calculation") and not c.metadata.get("method"):
                column.addItem(c.name + (f" [{c.unit}]" if c.unit else ""), c.id)
        matches = [
            c.id
            for c in owner.model.project.columns
            if c.name.split(" / ")[-1].split(" [")[0].strip().casefold()
            == role.casefold()
        ]
        if len(matches) == 1:
            column.setCurrentIndex(max(0, column.findData(matches[0])))
        unit = QComboBox()
        unit.setAccessibleName(tr('{0}: единица входа', role))
        unit.addItems(units)
        holder.addWidget(column, 2)
        holder.addWidget(unit, 1)
        field = QVBoxLayout()
        field.addLayout(holder)
        feedback = QLabel()
        feedback.setWordWrap(True)
        feedback.setTextFormat(Qt.TextFormat.PlainText)
        feedback.hide()
        field.addWidget(feedback)
        form.addRow(role, field)
        editor.inputs[role] = (column, unit)
        editor.input_feedback[role] = feedback
        column.currentIndexChanged.connect(lambda _, key=role: editor.input_column_changed(key))
        unit.currentIndexChanged.connect(editor.invalidate)
        editor.input_column_changed(role)
    group_form(form, [(editor.scope, tr('Область и контекст')), (editor.inputs[next(iter(editor.inputs))][0], tr('Входные показатели'))])
    settings.addLayout(form)


def _output_fields(editor, settings):
    owner = editor.owner
    editor.outputs = QTableWidget(len(FORMULAS), 3)
    editor.outputs.setAccessibleName(tr('Вычисляемые показатели'))
    editor.outputs.setHorizontalHeaderLabels(
        [tr('Создать'), tr('Формула'), tr('Название новой колонки')]
    )
    existing = {c.name for c in owner.model.project.columns}
    for i, (name, definition) in enumerate(FORMULAS.items()):
        check = QCheckBox(name)
        check.setChecked(name not in ("OI", "OSI") and all(
            editor.inputs[role][0].currentData() for role in definition["roles"]
        ))
        check.toggled.connect(editor.invalidate)
        editor.outputs.setCellWidget(i, 0, check)
        item = QTableWidgetItem(definition["equation"])
        item.setToolTip(f"{definition['equation']} [{definition['unit']}]")
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        editor.outputs.setItem(i, 1, item)
        editor.outputs.setItem(
            i,
            2,
            QTableWidgetItem(name if name not in existing else tr('{0} (расчёт)', name)),
        )
    editor.outputs.itemChanged.connect(editor.invalidate)
    editor.outputs.resizeColumnsToContents()
    editor.outputs.horizontalHeader().setStretchLastSection(True)
    editor.outputs.setMinimumHeight(195)
    settings.addWidget(editor.outputs)


def _lab_fields(editor):
    lab_form = QFormLayout()
    lab_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    lab_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    editor.lab_inputs = {}
    for formula in ("HI", "PI"):
        column = QComboBox()
        column.setAccessibleName(tr('{0} из лаборатории', formula))
        column.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        column.setMinimumContentsLength(10)
        unit = QComboBox()
        unit.setAccessibleName(tr('{0}: единица лабораторной колонки', formula))
        unit.addItem(FORMULAS[formula]["unit"], FORMULAS[formula]["unit"])
        if formula == "PI":
            unit.setItemText(0, tr('Доля (0–1)'))
            unit.addItem("%", "%")
        tolerance = QDoubleSpinBox()
        tolerance.setDecimals(12 if formula == "PI" else 8)
        tolerance.setRange(0, 1 if formula == "PI" else 1e12)
        tolerance.setValue(0.0001 if formula == "PI" else 0.1)
        tolerance.setToolTip(tr('Абсолютный допуск: {0}', tr('доля (0–1)') if formula == "PI" else FORMULAS[formula]["unit"]))
        tolerance.setAccessibleName(tr('{0}: абсолютный допуск', formula))
        lab_form.addRow(tr('{0} из лаборатории', formula), column)
        lab_form.addRow(unit.accessibleName(), unit)
        lab_form.addRow(tr('Допуск [{0}]', tr('доля') if formula == "PI" else FORMULAS[formula]["unit"]), tolerance)
        for field in (column, unit, tolerance):
            lab_form.labelForField(field).setWordWrap(True)
        editor.lab_inputs[formula] = column, unit, tolerance
        column.currentIndexChanged.connect(
            lambda _, key=formula: editor.lab_column_changed(key)
        )
        unit.currentIndexChanged.connect(editor.invalidate)
        tolerance.valueChanged.connect(editor.invalidate)
    lab_settings = parameter_page(editor.parameter_tabs, tr('Сверка с лабораторией'), 'files')
    lab_settings.addLayout(lab_form)
    lab_settings.addStretch()
    group_form(lab_form, [(editor.lab_inputs["HI"][0], tr('Сверка с лабораторией'))])
    editor.refresh_columns()


def _policy_controls(editor, settings):
    editor.text_numbers = QCheckBox(
        tr('Читать числовые записи в текстовых колонках')
    )
    editor.confirmed = QCheckBox(
        tr('Подтверждаю совместимость методов, фракций и единиц выбранных измерений')
    )
    wrapped_check(settings, editor.text_numbers)
    wrapped_check(settings, editor.confirmed)
    editor.propagate_uncertainty = QCheckBox(
        tr('Рассчитать стандартную неопределённость (входы некоррелированы)')
    )
    wrapped_check(settings, editor.propagate_uncertainty)
    editor.propagate_uncertainty.toggled.connect(editor.invalidate)
    editor.show_censored_ranges = QCheckBox(tr('Показывать диапазоны для входов ниже предела'))
    wrapped_check(settings, editor.show_censored_ranges)
    editor.show_censored_ranges.toggled.connect(editor.invalidate)
    note = QLabel(
        tr('TOC нормируется к %wt; S1/S2 — mg HC/g rock; S3 — mg CO2/g rock. PI — индекс продукции, не степень преобразования керогена. Интерполированные входы не используются.')
    )
    note.setWordWrap(True)
    settings.addWidget(note)
    return note


def _preview_filters(editor, layout):
    editor.preview_button = QPushButton(icon('calculator'), tr('Предпросмотр расчётов'))
    editor.preview_button.clicked.connect(editor.calculate)
    editor._actions = QHBoxLayout()
    editor._actions.addWidget(editor.preview_button)
    editor.mismatches_only = QCheckBox(tr('Расхождения с лабораторией'))
    editor.mismatches_only.setEnabled(False)
    editor.mismatches_only.toggled.connect(editor.render_preview)
    filters = QHBoxLayout()
    wrapped_check(filters, editor.mismatches_only)
    editor.unavailable_only = QCheckBox(tr('Сравнение недоступно'))
    editor.unavailable_only.setEnabled(False)
    editor.unavailable_only.toggled.connect(editor.render_preview)
    wrapped_check(filters, editor.unavailable_only)
    editor.reset_preview_filters = QPushButton(icon('rotate-ccw'), tr('Сбросить фильтры'))
    editor.reset_preview_filters.setAutoDefault(False)
    editor.reset_preview_filters.clicked.connect(editor.clear_preview_filters)
    filters.addWidget(editor.reset_preview_filters)
    layout.addLayout(filters)
    editor._result_filters = filters
    editor.filter_hint = QLabel(tr('Показаны пробы с расхождениями или недоступной сверкой.'))
    editor.filter_hint.setWordWrap(True)
    layout.addWidget(editor.filter_hint)


def _result_navigation(editor, layout):
    sample_navigation = QHBoxLayout()
    editor.sample_search = QLineEdit()
    search_label = tr('Проба, скважина, глубина или UUID…')
    editor.sample_search.setPlaceholderText(search_label)
    editor.sample_search.setAccessibleName(search_label)
    editor.sample_search.setMinimumWidth(130)
    editor.sample_search.setClearButtonEnabled(True)
    editor.sample_search.installEventFilter(editor)
    editor.sample_search.textChanged.connect(editor.render_preview)
    editor.open_sample = QPushButton(icon('table-2'), tr('Показать пробу в таблице'))
    editor.open_sample.setAutoDefault(False)
    editor.open_sample.clicked.connect(lambda: editor.jump_to_sample(editor.preview.currentRow(), 0))
    sample_navigation.addWidget(editor.sample_search, 1)
    sample_navigation.addWidget(editor.open_sample)
    layout.addLayout(sample_navigation)
    editor._sample_navigation = sample_navigation
    navigation = QHBoxLayout()
    editor.previous_page = QPushButton(icon('undo-2'), tr('Назад'))
    editor.next_page = QPushButton(icon('redo-2'), tr('Далее'))
    editor.page_label = QLabel()
    editor.page_label.setWordWrap(True)
    editor.previous_page.clicked.connect(lambda: editor.change_page(-1))
    editor.next_page.clicked.connect(lambda: editor.change_page(1))
    editor.export_button = QPushButton(icon('log-out'), tr('Выгрузить сверку…'))
    editor.export_button.clicked.connect(editor.export_comparison)
    for widget in (editor.previous_page, editor.next_page, editor.page_label, editor.export_button):
        navigation.addWidget(widget)
    layout.addLayout(navigation)
    editor._page_navigation = navigation
    archive_actions = QHBoxLayout()
    editor.save_comparison_button = QPushButton(icon('save'), tr('Сохранить сверку в проекте'))
    editor.save_comparison_button.clicked.connect(editor.save_comparison)
    editor.saved_comparisons_button = QPushButton(icon('logs'), tr('Сохранённые сверки…'))
    editor.saved_comparisons_button.clicked.connect(editor.open_saved_comparisons)
    archive_actions.addWidget(editor.save_comparison_button)
    archive_actions.addWidget(editor.saved_comparisons_button)
    layout.addLayout(archive_actions)
    editor._archive_actions = archive_actions


def _results_table(editor, layout):
    editor.preview = QTableWidget()
    editor.preview.setAccessibleName(tr('Предпросмотр расчётов'))
    editor.preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    editor.preview.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    editor.preview.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    editor.preview.verticalHeader().setAccessibleName(tr('Проба'))
    editor.preview.verticalHeader().setFixedWidth(155)
    editor.preview.cellActivated.connect(editor.jump_to_sample)
    editor.preview.currentItemChanged.connect(editor.show_preview_detail)
    layout.addWidget(editor.preview, 1)
    editor.preview_detail = QLabel()
    editor.preview_detail.setTextFormat(Qt.TextFormat.PlainText)
    editor.preview_detail.setWordWrap(True)
    editor.preview_detail.setTextInteractionFlags(
        Qt.TextInteractionFlag.TextSelectableByKeyboard | Qt.TextInteractionFlag.TextSelectableByMouse
    )
    layout.addWidget(editor.preview_detail)
    editor.summary = QLabel()
    editor.summary.setWordWrap(True)
    editor.summary.setTextFormat(Qt.TextFormat.PlainText)
    editor.summary.setTextInteractionFlags(
        Qt.TextInteractionFlag.TextSelectableByKeyboard | Qt.TextInteractionFlag.TextSelectableByMouse
    )
    layout.addWidget(editor.summary)
    editor.buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    editor.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
        tr('Создать вычисляемые колонки')
    )
    editor.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr('Закрыть'))
    editor.buttons.accepted.connect(editor.apply)
    editor.buttons.rejected.connect(editor.reject)
    editor._actions.addWidget(editor.buttons)
    editor.layout().addLayout(editor._actions)


def _finish(editor, settings, note):
    owner = editor.owner
    editor.context.textChanged.connect(editor.invalidate)
    editor.scope.currentIndexChanged.connect(editor.invalidate)
    editor.text_numbers.toggled.connect(editor.invalidate)
    editor.confirmed.toggled.connect(editor.invalidate)
    editor.invalidate()
    group_layout(settings, [(editor.outputs, tr('Вычисляемые показатели')), (note, None)])
    owner.analysis_scope.bind(editor.scope, editor.invalidate)
    add_parameter_tools(editor, reset=editor.restore_parameters, tables=(editor.outputs,), exclude=(editor.sample_search,))
    utilities = QHBoxLayout()
    utilities.addStretch()
    for button in (editor.reset_defaults_button, editor.help_button):
        editor.buttons.removeButton(button)
        utilities.addWidget(button)
    editor.layout().insertLayout(1, utilities)
    editor.preview_button.setDefault(True)
    settings.addStretch()
    editor._layout_ready = True
    editor._adapt_layout()


def _adapt_layout(editor, width=None):
    width = editor.width() if width is None else width
    margins = editor.layout().contentsMargins()
    inner_width = width - margins.left() - margins.right()
    index = editor.parameter_tabs.indexOf(editor._compact_results)
    if width < 920 and index < 0:
        editor._compact_results.setWidget(editor._results_panel)
        editor.parameter_tabs.addTab(editor._compact_results, icon('table-2'), tr('Результаты'))
        if editor._preview_source is not None:
            editor.parameter_tabs.setCurrentWidget(editor._compact_results)
    elif width >= 920 and index >= 0:
        editor.parameter_tabs.removeTab(index)
        editor._splitter.addWidget(editor._compact_results.takeWidget())
        editor._splitter.setSizes([340, width - 340])
        editor._results_panel.show()
    pane_width = inner_width - (10 if width < 920 else 340)
    input_direction = (QBoxLayout.Direction.TopToBottom if width < 560
                       else QBoxLayout.Direction.LeftToRight)
    for row in editor._input_rows:
        row.setDirection(input_direction)
    button_width = sum(button.minimumSizeHint().width() for button in editor.buttons.buttons())
    editor.buttons.setOrientation(Qt.Orientation.Vertical if inner_width < button_width + 12
                                else Qt.Orientation.Horizontal)
    for row in (editor._result_filters, editor._sample_navigation, editor._page_navigation,
                editor._archive_actions, editor._actions):
        required = sum(row.itemAt(i).minimumSize().width() for i in range(row.count()))
        required += row.spacing() * (row.count() - 1)
        available = inner_width if row is editor._actions else pane_width
        reserve = 24 if row is editor._actions else 0
        row.setDirection(QBoxLayout.Direction.TopToBottom if available < required + reserve
                         else QBoxLayout.Direction.LeftToRight)
