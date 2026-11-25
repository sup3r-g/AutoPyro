"""Build the stratigraphy editor, interval table, toolbars and column page."""

from __future__ import annotations

from ..i18n import tr
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QSplitter,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QTableWidget,
    QTableView,
    QTabWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from autopyro.geology.stratigraphy import STRATAPY_FIELDS, LEVELS, interval_color
from ..widgets.parameters import add_parameter_tools, ParameterDefaults
from ..widgets.documents import analysis_document, parameter_page
from ..theme import icon
from ..widgets.stratigraphy_labels import GeologicalDelegate, geological_options
from ..widgets.styles import StyleDelegate
from ..widgets.lithology import LithologyCatalog, LithologyDelegate
from ..widgets.interval_history import IntervalHistory
from ..widgets.interval_selection import IntervalSelection
from .stratigraphy_binding import BindingPreviewModel


def build(editor):
    settings, layout = analysis_document(editor, parameter_icon='layers')
    _scheme_controls(editor, settings)
    binding_group, binding_layout = _binding_inputs(editor)
    _interval_results(editor, layout)
    _column_page(editor)
    _interval_actions(editor, layout)
    _binding_actions(editor, binding_layout)
    _finish_controls(editor, layout, settings, binding_group)
    _history_bar(editor, layout)


def _scheme_controls(editor, settings):
    owner = editor.owner
    form = QFormLayout()
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    editor.scope = QComboBox()
    editor.name, editor.revision = QLineEdit(), QLineEdit()
    editor.reference = QLineEdit()
    editor.reference.setPlaceholderText(
        tr('Например: MD, метры от RKB; требуется подтверждённая система глубин')
    )
    editor.unit = QComboBox()
    editor.unit.addItems(["m", "ft"])
    editor.mode = QComboBox()
    editor.mode.addItem(tr('Точечная глубина'), "point")
    editor.mode.addItem(tr('Интервал отбора [от, до)'), "interval")
    editor.well, editor.depth, editor.bottom = QComboBox(), QComboBox(), QComboBox()
    editor.well.addItem(tr('Выберите колонку ствола'), None)
    editor.depth.addItem(tr('Выберите глубину / верх отбора'), None)
    editor.bottom.addItem(tr('Выберите низ отбора'), None)
    for column in owner.model.project.columns:
        if not column.metadata.get("stratigraphy"):
            editor.well.addItem(column.name, column.id)
            editor.depth.addItem(f"{column.name} [{column.unit}]", column.id)
            editor.bottom.addItem(f"{column.name} [{column.unit}]", column.id)
    editor.sample_unit = QComboBox()
    editor.sample_unit.addItems(["m", "ft"])
    editor.sample_reference = QLineEdit()
    editor.text_numbers = QCheckBox(tr('Читать числа в текстовой колонке глубины'))
    editor.column_well = QComboBox()
    editor.level = QComboBox()
    for label, code in geological_options(LEVELS):
        editor.level.addItem(label, code)
    editor.level.setCurrentIndex(editor.level.findData("FORMATION"))
    for label, widget in [
        (tr('Название схемы'), editor.name),
        (tr('Редакция'), editor.revision),
        (tr('Тип / начало отсчёта интервалов'), editor.reference),
        (tr('Единица интервалов'), editor.unit),
        (tr('Ствол колонки'), editor.column_well),
        (tr('Основной уровень'), editor.level),
    ]:
        form.addRow(label, widget)
    settings.addLayout(form)


def _binding_inputs(editor):
    binding_group = QGroupBox(tr('Привязка проб и замеров (необязательно)'))
    binding_layout = QVBoxLayout(binding_group)
    form = QFormLayout()
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    for label, widget in [
        (tr('Область проб и замеров'), editor.scope),
        (tr('Режим проб'), editor.mode),
        (tr('Ствол проб (точное имя)'), editor.well),
        (tr('Глубина / верх отбора'), editor.depth),
        (tr('Низ отбора'), editor.bottom),
        (tr('Единица глубины проб'), editor.sample_unit),
        (tr('Тип / начало отсчёта проб'), editor.sample_reference),
    ]:
        form.addRow(label, widget)
    form.addRow(editor.text_numbers)
    binding_layout.addLayout(form)
    note = QLabel(
        tr('Кровля включается, подошва исключается: [top, bottom). Перекрытия одного уровня дают неоднозначность.\nИнтервальные пробы: полное попадание, пересечение границ, неполное покрытие или неоднозначность; без замены серединой.\nMD/TVD и начало отсчёта не преобразуются; стволы и боковые ответвления не объединяются.')
    )
    note.setWordWrap(True)
    binding_layout.addWidget(note)
    return binding_group, binding_layout


def _interval_results(editor, layout):
    tabs = editor.result_tabs = QTabWidget()
    editor.binding_revision = QLabel()
    editor.binding_revision.setTextFormat(Qt.TextFormat.PlainText)
    editor.binding_revision.setWordWrap(True)
    layout.addWidget(editor.binding_revision)
    layout.addWidget(tabs)
    editor.intervals = QTableWidget(0, 8 + len(STRATAPY_FIELDS))
    editor.intervals.setItemDelegateForColumn(5, StyleDelegate(
        editor.intervals, automatic_color=lambda index: interval_color({
            "name": index.siblingAtColumn(1).data(),
        }),
    ))
    editor.lithology_catalog = LithologyCatalog()
    editor.lithology_delegate = LithologyDelegate(editor.lithology_catalog, editor.intervals)
    editor.intervals.setItemDelegateForColumn(7, editor.lithology_delegate)
    editor.intervals.cellDoubleClicked.connect(editor.choose_geological_cell)
    for column, codes in [(2, LEVELS), (10, ('', 'concave', 'convex', 'sawtooth', 'sawtooth_r')),
                          (15, ('', 'sharp', 'gradational', 'erosional'))]:
        editor.intervals.setItemDelegateForColumn(column, GeologicalDelegate(editor.intervals, codes))
    for column in (13, 14):
        editor.intervals.setItemDelegateForColumn(column, GeologicalDelegate(editor.intervals, multiple=True))
    editor.intervals.setHorizontalHeaderLabels(
        [
            tr('Ствол'), tr('Подразделение'),
            tr('Уровень'), tr('Кровля'),
            tr('Подошва'), tr('Цвет заливки'),
            tr('Идентификатор интервала'), tr('Литология'),
            tr('Размер зёрен у подошвы'), tr('Размер зёрен у кровли'),
            tr('Форма профиля слоя'), tr('Эрозия'), tr('Линзы'),
            tr('Минералы'), tr('Геологические признаки'), tr('Тип контакта'),
        ]
    )
    for column, width, tooltip in (
        (0, 102, tr('Ствол скважины')),
        (1, 156, tr('Название подразделения')),
        (2, 106, tr('Стратиграфический уровень')),
        (3, 84, tr('Глубина кровли')),
        (4, 84, tr('Глубина подошвы')),
        (5, 110, tr('Цвет заливки. Очистите поле для автоматического подбора по ОСШ.')),
        (7, 160, tr('Литология')),
    ):
        editor.intervals.setColumnWidth(column, width)
        editor.intervals.horizontalHeaderItem(column).setToolTip(tooltip)
    editor.intervals.setColumnHidden(6, True)
    editor.intervals.horizontalHeaderItem(7).setToolTip(
        tr('Двойной щелчок — выбор породы из каталога с предпросмотром узоров.'))
    tabs.addTab(editor.intervals, icon('layers'), tr('Интервалы'))
    editor.results = QTableView()
    editor.results.setModel(BindingPreviewModel(editor.results))
    tabs.addTab(editor.results, icon('tags'), tr('Результаты привязки'))


def _column_page(editor):
    tabs = editor.result_tabs

    from ..widgets.navigation import NavigationToolbar
    from ..widgets.figure_canvas import CanvasView, FittedFigureCanvas, FigureNavigation
    from matplotlib.figure import Figure

    editor.column_page = QWidget()
    column_layout = QVBoxLayout(editor.column_page)
    editor.column_figure = Figure(figsize=(6, 9))
    editor.column_canvas = FittedFigureCanvas(editor.column_figure)
    editor.column_navigation = FigureNavigation(editor.column_canvas, blocked=lambda: editor._column_render_running)
    editor.column_toolbar = NavigationToolbar(editor.column_canvas, editor.column_page)
    editor.canvas_view = CanvasView(editor.column_canvas,
        lambda: (editor.column_style_form.fields['width_mm'][0].value(),
                 editor.column_style_form.fields['height_mm'][0].value()))
    editor.canvas_view.add_to_toolbar(editor.column_toolbar)
    editor.column_toolbar.setEnabled(False)
    column_layout.addWidget(editor.column_toolbar)
    editor.sample_track = QCheckBox(tr('Показать дорожку проб и замеров'))
    editor.sample_track.setToolTip(tr('Использует текущую привязку и область проб. Щелчок открывает строку в таблице проекта.'))
    editor.sample_track.toggled.connect(editor.schedule_column_preview)
    column_layout.addWidget(editor.sample_track)
    editor.export_column_button = QPushButton(icon('log-out'), tr('Экспорт колонки с заданными размерами…'))
    editor.export_column_button.setEnabled(False)
    editor.export_column_button.clicked.connect(editor.export_column)
    editor.batch_export_button = QPushButton(icon('files'), tr('Экспорт комплекта…'))
    editor.batch_export_button.clicked.connect(editor.export_column_batch)
    export_actions = QHBoxLayout()
    export_actions.addWidget(editor.export_column_button, 1)
    export_actions.addWidget(editor.batch_export_button)
    column_layout.addLayout(export_actions)
    editor._export_include_samples = False
    editor.interval_selection = IntervalSelection(editor)
    editor.column_intervals = editor.interval_selection.view
    column_splitter = QSplitter()
    column_splitter.addWidget(editor.column_intervals)
    column_splitter.addWidget(editor.canvas_view.scroll)
    column_splitter.setSizes([390, 650])
    column_layout.addWidget(column_splitter, 1)
    tabs.addTab(editor.column_page, icon('layers'), tr('Колонка'))
    tabs.tabBar().moveTab(tabs.indexOf(editor.column_page), 1)


def _interval_actions(editor, layout):
    tabs = editor.result_tabs
    row = QToolBar(editor)
    row.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    layout.addWidget(row)
    for image, caption, tooltip, slot in [
        ('plus', tr('Добавить'), tr('Добавить интервал'), editor.add_interval),
        ('trash', tr('Удалить'), tr('Удалить строки'), editor.remove_intervals),
        ('layers', tr('Выбрать породу…'), tr('Выбрать литологию…'), editor.choose_lithology),
    ]:
        action = row.addAction(icon(image), caption)
        action.setToolTip(tooltip)
        action.triggered.connect(slot)
    editor.diagnose_intervals_action = row.addAction(icon('shield-check'), tr('Проверить интервалы…'))
    editor.diagnose_intervals_action.triggered.connect(editor.review_intervals)
    file_actions = QToolBar(editor)
    layout.addWidget(file_actions)
    imports = editor.interval_menu(file_actions, tr('Импорт'), 'import')
    imports.addAction(icon('import'), tr('Заменить интервалы из файла…')).triggered.connect(editor.import_scheme)
    editor.merge_intervals_action = imports.addAction(icon('list-plus'), tr('Добавить из файла…'))
    editor.merge_intervals_action.triggered.connect(lambda: editor.import_scheme(merge=True))
    imports.addAction(icon('database'), tr('Загрузить или обновить из PostgreSQL…')).triggered.connect(
        editor.import_database_scheme)
    rocks = editor.interval_menu(file_actions, tr('Литология'), 'layers')
    rocks.addAction(icon('layers'), tr('Сопоставить породы…')).triggered.connect(editor.map_lithologies)
    rocks.addAction(icon('settings-2'), tr('Правила пород…')).triggered.connect(editor.manage_lithology_rules)
    reports = editor.interval_menu(file_actions, tr('Отчёты и экспорт'), 'file-text')
    editor.summary_action = reports.addAction(icon('chart-column'), tr('Сводка мощностей…'))
    editor.summary_action.triggered.connect(editor.show_thickness_summary)
    revisions_action = reports.addAction(icon('files'), tr('Сравнить редакции…'))
    revisions_action.triggered.connect(editor.compare_revisions)
    reports.addAction(icon('log-out'), tr('Экспорт интервалов…')).triggered.connect(editor.export_scheme)
    editor.geological_columns = QCheckBox(tr('Геологические свойства слоёв'))
    editor.geological_columns.toggled.connect(editor.show_geological_columns)
    file_actions.addWidget(editor.geological_columns)
    tabs.currentChanged.connect(lambda: row.setVisible(tabs.currentWidget() is editor.intervals))
    tabs.currentChanged.connect(lambda: file_actions.setVisible(tabs.currentWidget() is editor.intervals))
    editor.show_geological_columns(False)


def _binding_actions(editor, binding_layout):
    for text, slot, image in [
        (tr('Проверить привязку'), editor.preview, 'shield-check'),
        (tr('Сохранить схему и привязку'), editor.apply, 'save'),
        (tr('Создать колонки уровня'), editor.add_columns, 'columns-3'),
        (tr('Показать график замеров с колонкой'), editor.show_plot, 'chart-no-axes-combined'),
    ]:
        button = QPushButton(icon(image), text)
        button.clicked.connect(slot)
        binding_layout.addWidget(button)


def _finish_controls(editor, layout, settings, binding_group):
    owner = editor.owner
    editor.column_actions = QWidget()
    actions = QHBoxLayout(editor.column_actions)
    actions.setContentsMargins(0, 0, 0, 0)
    editor.build_column_button = QPushButton(icon('layers'), tr('Построить колонку'))
    editor.build_column_button.clicked.connect(editor.show_column)
    editor.save_scheme_button = QPushButton(icon('save'), tr('Сохранить схему и оформление'))
    editor.save_scheme_button.clicked.connect(editor.save_scheme)
    actions.addWidget(editor.build_column_button)
    actions.addWidget(editor.save_scheme_button)
    layout.addWidget(editor.column_actions)
    editor.parameter_tabs.currentChanged.connect(lambda index: editor.column_actions.setVisible(index != 2))
    editor.status = QLabel(
        tr('Изменения черновика сохраняются только кнопками сохранения / создания колонок.')
    )
    editor.status.setWordWrap(True)
    editor.column_navigation.bind_status(editor.status)
    layout.addWidget(editor.status)
    editor.parameter_tabs.setTabText(0, tr('1. Интервалы'))
    editor.create_column_style()
    binding_page = parameter_page(editor.parameter_tabs, tr('3. Замеры'), 'tags')
    binding_page.addWidget(binding_group)
    binding_page.addStretch()
    editor.parameter_tabs.currentChanged.connect(
        lambda index: editor.result_tabs.setCurrentWidget((editor.intervals, editor.column_page, editor.results)[index])
        if 0 <= index < 3 else None)
    editor.result_tabs.currentChanged.connect(editor.parameter_tabs.setCurrentIndex)
    editor.load_draft()
    style_form = editor.column_style_form
    anchors = style_form.correlation.get('anchors', {})
    primary = next((well for well in anchors if well not in style_form.comparison_wells), None)
    if primary is not None:
        index = editor.column_well.findData(primary)
        if index >= 0:
            editor.column_well.setCurrentIndex(index)
    owner.analysis_scope.bind(editor.scope, editor.invalidate_preview)
    editor._initial_parameters = ParameterDefaults(editor, exclude=(editor.name, editor.revision, editor.reference, editor.unit,
                                                              editor.canvas_view.mode_combo, editor.style_templates,
                                                              editor.column_style_form, editor.column_style_form.details))
    binding = owner.model.project.view.get("stratigraphy", {}).get("binding", {})
    for combo, key in [
        (editor.well, "well_id"),
        (editor.depth, "depth_id"),
        (editor.bottom, "bottom_id"),
    ]:
        combo.setCurrentIndex(max(0, combo.findData(binding.get(key))))
    editor.mode.setCurrentIndex(
        max(0, editor.mode.findData(binding.get("sample_mode", "point")))
    )
    editor.bottom.setEnabled(editor.mode.currentData() == "interval")
    editor.sample_unit.setCurrentText(binding.get("depth_unit", "m"))
    editor.sample_reference.setText(binding.get("depth_reference", ""))
    editor.text_numbers.setChecked(binding.get("text_numbers", False))
    editor.intervals.itemChanged.connect(editor.invalidate_column)
    editor.intervals.itemChanged.connect(lambda cell: editor.refresh_wellbores() if cell.column() == 0 else None)
    editor.intervals.itemChanged.connect(editor.refresh_interval_color_preview)
    for field in (editor.name, editor.revision, editor.reference):
        field.textChanged.connect(editor.invalidate_column)
    editor.sample_reference.textChanged.connect(editor.invalidate_preview)
    editor.unit.currentIndexChanged.connect(editor.invalidate_column)
    for combo in (
        editor.well,
        editor.depth,
        editor.bottom,
        editor.mode,
        editor.sample_unit,
    ):
        combo.currentIndexChanged.connect(editor.invalidate_preview)
    editor.mode.currentIndexChanged.connect(
        lambda: editor.bottom.setEnabled(editor.mode.currentData() == "interval")
    )
    editor.column_well.currentIndexChanged.connect(editor.invalidate_column)
    editor.level.currentIndexChanged.connect(editor.invalidate_column)
    editor.text_numbers.toggled.connect(editor.invalidate_preview)
    add_parameter_tools(editor, defaults=editor._initial_parameters, changed=editor.invalidate_column, layout=settings)
    close_button = QPushButton(icon('x'), tr('Закрыть'))
    close_button.clicked.connect(editor.reject)
    settings.addWidget(close_button)
    settings.addStretch()
    editor.canvas_view.bind_project(owner, 'stratigraphy_column')
    editor.result_tabs.currentChanged.connect(editor.resume_column_preview)
    editor.interval_history = IntervalHistory(editor)
    editor.draft_undo = editor.interval_history.stack
    editor._saved_style = editor.style_values()
    editor._saved_scheme = editor.owner.model.project.view.get('stratigraphy_scheme')
    editor.update_binding_revision()


def _history_bar(editor, layout):
    history_bar = QToolBar(editor)
    history_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    layout.insertWidget(0, history_bar)
    undo = QAction(icon('undo-2'), tr('Отменить правку'), editor)
    undo.triggered.connect(editor.draft_undo.undo)
    editor.draft_undo.canUndoChanged.connect(undo.setEnabled)
    undo.setEnabled(editor.draft_undo.canUndo())
    undo.setToolTip(tr('Отменить правку черновика (Ctrl+Z)'))
    redo = QAction(icon('redo-2'), tr('Повторить правку'), editor)
    redo.triggered.connect(editor.draft_undo.redo)
    editor.draft_undo.canRedoChanged.connect(redo.setEnabled)
    redo.setEnabled(editor.draft_undo.canRedo())
    redo.setToolTip(tr('Повторить правку черновика (Ctrl+Shift+Z)'))
    history_bar.addAction(undo)
    history_bar.addAction(redo)
    editor.zoom_intervals_action = history_bar.addAction(icon('maximize'), tr('Показать выделенные'))
    editor.zoom_intervals_action.setEnabled(False)
    editor.zoom_intervals_action.triggered.connect(editor.interval_selection.zoom_to_selected)
    editor.column_intervals.setToolTip(tr('Выделение связано с колонкой. Ctrl добавляет или снимает выделение слоёв.'))


def interval_menu(toolbar, title, image):
    button = QToolButton(toolbar)
    button.setText(title)
    button.setIcon(icon(image))
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    menu = QMenu(button)
    button.setMenu(menu)
    toolbar.addWidget(button)
    return menu
