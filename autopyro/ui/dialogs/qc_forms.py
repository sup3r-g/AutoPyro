"""Build the QC workspace, its controls and compact layout."""

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from autopyro.analysis.qc import RULES, SCOPES

from ..i18n import tr
from ..theme import icon
from ..widgets.documents import analysis_document
from ..widgets.parameters import add_parameter_tools
from .qc_results import QCIssuesModel


def build(editor):
    settings, layout = analysis_document(editor)
    _filter_controls(editor, settings)
    _results_view(editor, settings, layout)
    actions, actions_layout = _decision_controls(editor)
    _exclusion_controls(editor, actions_layout)
    _detail_visibility(editor, layout, actions)
    _status_controls(editor, layout, settings)
    _connect_inputs(editor, settings)


def _filter_controls(editor, settings):
    controls = QFormLayout()
    controls.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    controls.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    editor.data_scope = QComboBox()
    controls.addRow(tr('Область'), editor.data_scope)
    editor.severity = QComboBox()
    for label, key in ((tr('Все уровни'), "all"), (tr('Ошибки'), "error"),
                       (tr('Критические предупреждения'), "critical"), (tr('Предупреждения'), "warning")):
        editor.severity.addItem(label, key)
    editor.state = QComboBox()
    for label, key in ((tr('Все решения'), "all"), (tr('Не рассмотрены'), "open"), (tr('Приняты с комментарием'), "acknowledged")):
        editor.state.addItem(label, key)
    editor.rule_filter = QComboBox()
    editor.rule_filter.addItem(tr('Все проверки'), "all")
    for key, rule in RULES.items():
        editor.rule_filter.addItem(tr(rule["description"]), key)
    editor.exclusion_filter = QComboBox()
    editor.exclusion_filter.addItem(tr('Все измерения'), False)
    editor.exclusion_filter.addItem(tr('С действующим исключением'), True)
    editor.property_filter = QLineEdit()
    editor.property_filter.setPlaceholderText(tr('Показатель…'))
    editor.sample = QLineEdit()
    editor.sample.setPlaceholderText(tr('Проба, скважина, глубина или UUID…'))
    editor.source = QLineEdit()
    editor.source.setPlaceholderText(tr('Файл / источник…'))
    for label, widget in ((tr('Уровень'), editor.severity), (tr('Решение'), editor.state), (tr('Проверка'), editor.rule_filter),
                          (tr('Исключения'), editor.exclusion_filter),
                          (tr('Показатель'), editor.property_filter), (tr('Проба'), editor.sample),
                          (tr('Источник'), editor.source)):
        controls.addRow(label, widget)
    settings.addLayout(controls)
    editor.profile_button = QPushButton(icon('sliders-horizontal'), tr('Профиль порогов пиролиза…'))
    editor.profile_button.clicked.connect(editor.edit_profile)
    settings.addWidget(editor.profile_button)
    editor.profile_status = QLabel()
    editor.profile_status.setWordWrap(True)
    settings.addWidget(editor.profile_status)
    editor.coverage_button = QPushButton(icon('shield-check'), tr('Охват проверок…'))
    editor.coverage_button.clicked.connect(editor.show_coverage)
    editor.exclusions_button = QPushButton(icon('eye-off'), tr('Действующие исключения…'))
    editor.exclusions_button.clicked.connect(editor.manage_exclusions)
    settings.insertWidget(0, editor.exclusions_button)
    settings.insertWidget(1, editor.coverage_button)


def _results_view(editor, settings, layout):
    owner = editor.owner
    editor.table = QTableView()
    editor.results_model = QCIssuesModel(owner.model.project, editor.table)
    editor.table.setModel(editor.results_model)
    editor.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
    editor.table.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
    editor.table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
    editor.table.selectionModel().currentChanged.connect(editor.show_details)
    editor.table.selectionModel().selectionChanged.connect(editor.show_details)
    editor.table.doubleClicked.connect(lambda index: editor.go_to_sample())
    header = editor.table.horizontalHeader()
    header.setSectionsClickable(True)
    header.setSortIndicatorShown(True)
    header.setSortIndicator(0, Qt.SortOrder.AscendingOrder)
    header.sortIndicatorChanged.connect(editor.render)
    for column, width in enumerate((96, 88, 128, 130, 160, 88, 160)):
        editor.table.setColumnWidth(column, width)
    editor.table.setColumnHidden(3, True)
    editor.table.setColumnHidden(6, True)
    editor.table.setColumnWidth(7, 170)
    editor.table.setColumnHidden(7, True)
    editor.table.setHorizontalScrollMode(QTableView.ScrollMode.ScrollPerPixel)
    header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
    columns_button = QPushButton(icon('columns-3'), tr('Колонки замечаний'))
    columns_menu = QMenu(columns_button)
    for column in (3, 6, 7):
        action = columns_menu.addAction(icon('columns-3'), editor.results_model.headers[column])
        action.setCheckable(True)
        action.toggled.connect(lambda visible, column=column: editor.table.setColumnHidden(column, not visible))
    columns_button.setMenu(columns_menu)
    settings.addWidget(columns_button)
    split = QSplitter(Qt.Orientation.Vertical)
    split.addWidget(editor.table)
    layout.addWidget(split, 1)
    editor.details = QTextEdit()
    editor.details.setReadOnly(True)
    editor.details.setAccessibleName(tr('Детали замечания'))
    split.addWidget(editor.details)
    split.setSizes([290, 150])


def _decision_controls(editor):
    editor.comment = QLineEdit()
    editor.comment.setPlaceholderText(tr('Комментарий к замечанию'))
    editor.comment.setAccessibleName(tr('Комментарий к замечанию'))
    editor.acknowledge = QPushButton(icon('check'), tr('Принять'))
    editor.acknowledge.clicked.connect(lambda: editor.decide("acknowledged"))
    editor.reopen = QPushButton(icon('rotate-ccw'), tr('Вернуть'))
    editor.reopen.setToolTip(tr('Вернуть на рассмотрение'))
    editor.reopen.clicked.connect(lambda: editor.decide("open"))
    editor.go = QPushButton(icon('table-2'), tr('К пробе'))
    editor.go.setToolTip(tr('Показать пробу в таблице'))
    editor.go.clicked.connect(editor.go_to_sample)
    editor.source_button = QPushButton(icon('database'), tr('Источник…'))
    editor.source_button.clicked.connect(editor.show_source)
    editor.restore_filter_button = QPushButton(icon('rotate-ccw'), tr('Вернуть вид таблицы'))
    editor.restore_filter_button.clicked.connect(editor.restore_table_filter)
    editor.restore_filter_button.hide()
    actions = QWidget()
    actions_layout = QVBoxLayout(actions)
    actions_layout.setContentsMargins(0, 0, 0, 0)
    editor.decision_actions = QBoxLayout(QBoxLayout.Direction.LeftToRight)
    editor.decision_actions.addWidget(editor.comment, 1)
    decision_buttons = QHBoxLayout()
    for widget in (editor.acknowledge, editor.reopen, editor.go, editor.source_button):
        decision_buttons.addWidget(widget)
    editor.decision_actions.addLayout(decision_buttons)
    actions_layout.addLayout(editor.decision_actions)
    return actions, actions_layout


def _exclusion_controls(editor, actions_layout):
    editor.scope = QComboBox()
    for key, label in SCOPES.items():
        editor.scope.addItem(tr(label), key)
    editor.scope.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    editor.scope.setMinimumContentsLength(13)
    editor.scope.setAccessibleName(tr('Вид анализа для исключения'))
    editor.scope.currentIndexChanged.connect(editor.show_details)
    editor.exclusion_comment = QLineEdit()
    editor.exclusion_comment.setPlaceholderText(tr('Причина исключения'))
    editor.exclusion_comment.setAccessibleName(tr('Причина исключения'))
    editor.exclude_button = QPushButton(icon('eye-off'), tr('Исключить'))
    editor.include_button = QPushButton(icon('x'), tr('Снять'))
    editor.include_button.setToolTip(tr('Снять исключение'))
    editor.exclude_button.clicked.connect(lambda: editor.exclude(True))
    editor.include_button.clicked.connect(lambda: editor.exclude(False))
    editor.exclusion_actions = QBoxLayout(QBoxLayout.Direction.LeftToRight)
    editor.exclusion_actions.addWidget(editor.exclusion_comment, 1)
    exclusion_buttons = QHBoxLayout()
    for widget in (editor.scope, editor.exclude_button, editor.include_button):
        exclusion_buttons.addWidget(widget)
    editor.exclusion_actions.addLayout(exclusion_buttons)
    actions_layout.addLayout(editor.exclusion_actions)


def _detail_visibility(editor, layout, actions):
    compact_controls = QHBoxLayout()
    for label, widget in ((tr('Детали замечания'), editor.details), (tr('Действия с замечанием'), actions)):
        toggle = QCheckBox(label)
        toggle.setChecked(True)
        toggle.toggled.connect(widget.setVisible)
        compact_controls.addWidget(toggle)
    compact_controls.addStretch()
    layout.addLayout(compact_controls)
    layout.addWidget(actions)


def _status_controls(editor, layout, settings):
    editor.status = QLabel()
    editor.status.setWordWrap(True)
    editor.status_row = QBoxLayout(QBoxLayout.Direction.LeftToRight)
    editor.status_row.addWidget(editor.status, 1)
    status_buttons = QHBoxLayout()
    status_buttons.addStretch()
    editor.batch_button = QPushButton(icon('shield-check'), tr('Групповое решение…'))
    editor.batch_button.setToolTip(tr('Выберите несколько строк с Ctrl или Shift. Ctrl+A выбирает текущую выдачу.'))
    editor.batch_button.clicked.connect(editor.batch_decision)
    status_buttons.addWidget(editor.batch_button)
    editor.measurement_button = QPushButton(icon('flask-conical'), tr('Измерение…'))
    editor.measurement_button.clicked.connect(editor.open_measurement)
    editor.measurement_button.hide()
    status_buttons.addWidget(editor.measurement_button)
    editor.retry = QPushButton(icon('refresh-cw'), tr('Обновить проверки'))
    editor.retry.clicked.connect(editor.refresh_checks)
    editor.retry.hide()
    status_buttons.addWidget(editor.retry)
    editor.status_row.addLayout(status_buttons)
    layout.addLayout(editor.status_row)
    layout.addWidget(editor.restore_filter_button, alignment=Qt.AlignmentFlag.AlignRight)
    close = QPushButton(icon('x'), tr('Закрыть'))
    close.clicked.connect(editor.close)
    settings.addWidget(close)


def _connect_inputs(editor, settings):
    owner = editor.owner
    for combo in (editor.severity, editor.state, editor.rule_filter, editor.exclusion_filter):
        combo.currentIndexChanged.connect(editor.render)
    editor.filter_timer = QTimer(editor)
    editor.filter_timer.setSingleShot(True)
    editor.filter_timer.setInterval(100)
    editor.filter_timer.timeout.connect(editor.render)
    for field in (editor.property_filter, editor.sample, editor.source):
        field.setClearButtonEnabled(True)
        field.textChanged.connect(lambda: editor.filter_timer.start())
    editor.timer = QTimer(editor)
    editor.timer.setSingleShot(True)
    editor.timer.setInterval(150)
    editor.timer.timeout.connect(editor.reload)
    owner.model.changed.connect(editor.schedule)
    editor.finished.connect(lambda result: editor.timer.stop())
    editor.finished.connect(lambda result: editor.restore_table_filter())
    editor.reload()
    owner.analysis_scope.bind(editor.data_scope, editor.render)
    for combo in (editor.data_scope, editor.severity, editor.state, editor.rule_filter, editor.exclusion_filter):
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(10)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    add_parameter_tools(editor, changed=editor.reload, layout=settings)
    settings.addStretch()


def resize_layout(editor, event):
    if not hasattr(editor, "status_row"):
        return
    compact = event.size().width() < 560
    if compact == editor._compact_table:
        return
    editor._compact_table = compact
    direction = QBoxLayout.Direction.TopToBottom if compact else QBoxLayout.Direction.LeftToRight
    editor.decision_actions.setDirection(direction)
    editor.exclusion_actions.setDirection(direction)
    editor.status_row.setDirection(direction)
    editor.table.verticalHeader().setVisible(not compact)
    widths = (80, 92, 104, 84) if compact else (96, 88, 128, 88)
    for column, width in zip((0, 1, 2, 5), widths):
        editor.table.setColumnWidth(column, width)
    if compact:
        QTimer.singleShot(0, editor.adjust_compact_docks)


def adjust_compact_docks(editor):
    owner = editor.owner
    desktop = owner.desktop
    if editor._compact_table and owner.width() <= 1100 and desktop.documents.currentWidget() is editor:
        docks = [desktop.project_dock, desktop.properties_dock]
        owner.resizeDocks(docks, [dock.minimumWidth() for dock in docks], Qt.Orientation.Horizontal)
