"""Build the plot editor controls, toolbar and canvas layout."""

from __future__ import annotations

from ..i18n import tr
from copy import deepcopy
from ..theme import icon
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QCompleter, QFormLayout, QLabel, QListWidget, QMenu,
    QPushButton, QScrollArea, QSizePolicy, QSplitter, QTabWidget, QTableWidget,
    QToolBar, QToolButton, QVBoxLayout, QWidget,
)
from ..plot_preview import PreviewCanvas
from autopyro.visualization.plot_catalog import library_kinds, library_parameters
from autopyro.visualization.plot_controls import controls_for, ROLE_LABELS, CHOICE_LABELS
from autopyro.visualization.plot_passports import INTERVAL_PASSPORTS, PASSPORTS, PROFILE_PASSPORTS, TABLET_TITLE
from autopyro.visualization.plotting import KINDS, MARKERS, TEMPLATES, PlotSpec
from ..widgets.forms import Form, axis_form
from ..widgets.parameters import add_parameter_tools
from ..widgets.parameter_reset import install_cell_reset_menu
from ..widgets.styles import StyleDelegate
from ..widgets.figure_canvas import CanvasView


def build(dialog):
    layout = QVBoxLayout(dialog)
    compact = dialog.interpretation_owner is not None
    top = _build_commands(dialog, layout, compact)
    splitter = QSplitter(dialog)
    dialog.tabs = QTabWidget()
    splitter.addWidget(dialog.tabs)
    dialog.tabs.setUsesScrollButtons(True)
    _build_data(dialog)
    _build_axes_and_markers(dialog)
    _build_layout(dialog)
    _build_viewer(dialog, splitter, layout, top, compact)


def _build_commands(dialog, layout, compact):
    top = QToolBar(dialog)
    top.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
    dialog.plot_actions_menu = QMenu(tr('Действия с графиком'), top)
    dialog.template = QComboBox()
    dialog.template.addItem(tr('Свободный график'))
    for title in PASSPORTS:
        dialog.template.addItem(tr(title), title)
    # Pyrolysis presets now belong to the interpretation scheme selector.
    # Keep the combo as an API for loading existing axis presets.
    dialog.template.hide()
    undo = dialog.undo_stack.createUndoAction(dialog.plot_actions_menu, tr('Отменить оформление'))
    undo.setIcon(icon('undo-2'))
    dialog.plot_actions_menu.addAction(undo)
    redo = dialog.undo_stack.createRedoAction(dialog.plot_actions_menu, tr('Повторить оформление'))
    redo.setIcon(icon('redo-2'))
    dialog.plot_actions_menu.addAction(redo)
    dialog.plot_actions_menu.addSeparator()
    for icon_name, label, callback in [
        ('save', tr('Сохранить график…'), dialog.save_named),
        ('folder-open', tr('Открыть график…'), dialog.load_named),
        ('log-out', tr('Экспорт…'), dialog.export_plot),
        ('file-text', tr('JSON оформления…'), dialog.export_spec),
        ('import', tr('Загрузить JSON…'), dialog.import_spec),
    ]:
        actions = top if callback == dialog.export_plot else dialog.plot_actions_menu
        action = actions.addAction(icon(icon_name), label, callback)
        if callback == dialog.export_plot:
            dialog.export_button = action
    if not compact:
        track_export = dialog.plot_actions_menu.addAction(
            icon('log-out'), tr('Экспортировать отдельный трек…'), dialog.export_track,
        )
        dialog.plot_actions_menu.aboutToShow.connect(lambda: track_export.setEnabled(
            dialog.result is not None and dialog.spec.kind == 'multi_depth' and not dialog.export_busy()))
    dialog.plot_actions_menu.addAction(icon('save'), tr('Сохранить снимок графика…'), dialog.export_snapshot)
    dialog.plot_actions_menu.addAction(icon('log-out'), tr('Экспортировать из снимка графика…'), dialog.replay_snapshot)
    dialog.plot_actions_menu.addAction(icon('files'), tr('Пакетный экспорт всех…'), lambda: dialog.batch_export())
    dialog.plot_actions_menu.addAction(icon('log-out'), tr('Экспорт выбранных графиков…'), lambda: dialog.export_selected_plots())
    dialog.plot_actions_menu.addAction(icon('layout-template'), tr('Собрать многопанельный рисунок…'),
                                     lambda: dialog.export_selected_plots(panels=True))
    dialog.plot_actions_menu.setToolTipsVisible(True)
    for action in dialog.plot_actions_menu.actions():
        purpose = {
            tr('JSON оформления…'): tr('Переносит настройки графика без значений проб.'),
            tr('Сохранить снимок графика…'): tr('Сохраняет настройки и данные для повторного экспорта без проекта.'),
        }.get(action.text())
        if purpose:
            action.setToolTip(purpose)
    dialog.plot_actions_menu.addAction(icon('file-text'), tr('Аннотации на графике…'), lambda: dialog.edit_annotations())
    dialog.retry_plot_action = dialog.plot_actions_menu.addAction(icon('refresh-cw'), tr('Обновить график'), dialog.schedule)
    dialog.retry_plot_action.setEnabled(False)
    if dialog.interpretation_owner is None:
        dialog.plot_actions_menu.addAction(icon('plus'), tr('Добавить сохранённую кривую…'), dialog.add_archived_curve)
        dialog.plot_actions_menu.addAction(icon('x'), tr('Убрать сохранённую кривую'), dialog.remove_archived_curve)
        dialog.plot_actions_menu.addAction(icon('save'), tr('Зафиксировать набор проб графика…'), dialog.freeze_scope)
        dialog.plot_actions_menu.addAction(icon('x'), tr('Убрать паспорт диаграммы'), dialog.remove_passport)
        dialog.plot_actions_menu.addAction(icon('layers'), tr('Убрать слой интерпретации'), dialog.remove_interpretation)
        dialog.plot_actions_menu.addAction(icon('layers'), tr('Убрать стратиграфический слой'), dialog.remove_stratigraphy)
    dialog.plot_actions_button = QToolButton(top)
    dialog.plot_actions_button.setText(tr('Действия с графиком'))
    dialog.plot_actions_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    dialog.plot_actions_button.setIcon(icon('ellipsis'))
    dialog.plot_actions_button.setToolTip(tr('Действия с графиком'))
    dialog.plot_actions_button.setAccessibleName(tr('Действия с графиком'))
    dialog.plot_actions_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    dialog.plot_actions_button.setMenu(dialog.plot_actions_menu)
    if not compact:
        top.addWidget(dialog.plot_actions_button)
    task_actions = dialog.plot_actions_menu
    dialog.cancel_prepare_action = task_actions.addAction(icon('x'), tr('Отменить подготовку'), dialog.cancel_preparation)
    dialog.cancel_prepare_action.setEnabled(False)
    dialog.cancel_export_action = task_actions.addAction(icon('x'), tr('Отменить экспорт'), dialog.cancel_export)
    dialog.cancel_export_action.setEnabled(False)
    if not compact:
        templates = dialog.plot_actions_menu.addMenu(icon('layout-template'), tr('Шаблоны осей'))
        for title in TEMPLATES:
            templates.addAction(icon('chart-no-axes-combined'), tr(title),
                                lambda checked=False, title=title: dialog.use_template(title))
        profiles = templates.addMenu(icon('chart-no-axes-combined'), tr('Профили по MD'))
        for title in PROFILE_PASSPORTS:
            profiles.addAction(icon('chart-no-axes-combined'), tr(title),
                               lambda checked=False, title=title: dialog.use_template(title))
        profiles.addAction(icon('chart-no-axes-combined'), tr('Планшет TOC, S1, S2 и Tmax'),
                           lambda: dialog.use_template(TABLET_TITLE))
        intervals = templates.addMenu(icon('chart-no-axes-combined'), tr('Интервальные профили по MD'))
        for title, (_, role, _, _) in INTERVAL_PASSPORTS.items():
            intervals.addAction(icon('chart-no-axes-combined'), role,
                                lambda checked=False, title=title: dialog.use_template(title))
    if compact:
        top.hide()
    else:
        layout.addWidget(top)
    return top


def _build_data(dialog):
    owner = dialog.owner
    interpretation_owner = dialog.interpretation_owner
    dialog.data = _form_page(dialog, tr('Данные'), 'table-2')
    dialog.data.section(tr('Источник данных'))
    groups = KINDS.groups()
    kinds = [(tr(label), key) for _, entries in groups for key, label in entries]
    dialog.kind = dialog.data.combo("kind", tr('Вид'), kinds)
    offset = 0
    for title, entries in groups:
        dialog.kind.insertItem(offset, tr(title), None)
        heading = dialog.kind.model().item(offset)
        heading.setEnabled(False)
        heading.setSelectable(False)
        font = heading.font()
        font.setBold(True)
        heading.setFont(font)
        offset += len(entries) + 1
    dialog.kind.setEditable(True)
    dialog.kind.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    dialog.kind.lineEdit().setPlaceholderText(tr('Найти вид графика'))
    dialog.kind.completer().setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
    dialog.kind.completer().setFilterMode(Qt.MatchFlag.MatchContains)
    dialog.kind.lineEdit().editingFinished.connect(
        lambda: dialog.kind.lineEdit().setText(dialog.kind.itemText(dialog.kind.currentIndex())))
    dialog.kind.setMaxVisibleItems(24)
    dialog.kind.setCurrentIndex(dialog.kind.findData('scatter'))
    dialog.plot_mode = QComboBox()
    dialog.plot_mode.addItem(tr('Группировка проб и слои'), False)
    dialog.plot_mode.addItem(tr('Библиотечный вид Matplotlib'), True)
    dialog.data.form.addRow(tr('Режим'), dialog.plot_mode)
    dialog.plot_mode.currentIndexChanged.connect(dialog.kind_changed)
    dialog.variant = QComboBox()
    dialog.variant_family = None
    dialog.data.form.addRow(tr('Вариант построения'), dialog.variant)
    dialog.variant.currentIndexChanged.connect(dialog.kind_changed)
    scopes = owner.analysis_scope.choices()
    if interpretation_owner is None:
        scopes.insert(0, (tr('Общая область анализа'), 'global'))
    dialog.scope = dialog.data.combo("scope", tr('Область'), scopes)
    dialog.scope.setToolTip(tr('Выберите область графика или используйте общую область анализа.'))
    dialog.local_filter_button = QPushButton(icon('funnel'), tr('Локальный фильтр графика…'))
    dialog.local_filter_button.clicked.connect(dialog.edit_local_filter)
    dialog.local_filter_button.setVisible(interpretation_owner is None)
    dialog.data.form.addRow(dialog.local_filter_button)
    dialog.x = dialog.data.combo("x_id", tr('X / значение'), [])
    dialog.y = dialog.data.combo("y_id", tr('Y / глубина'), [])
    dialog.z = dialog.data.combo("z_id", tr('Z / значение'), [])
    dialog.kind_note = QLabel()
    dialog.kind_note.setWordWrap(True)
    dialog.data.form.addRow(dialog.kind_note)
    dialog.depth_bottom = dialog.data.combo(
        "depth_bottom_id", tr('Низ отбора (интервальные пробы)'), []
    )
    dialog.data.section(tr('Отображение данных'))
    interval_note = QLabel(
        tr('При заданном низе отбора Y — верх пробы; вертикальный отрезок показывает весь интервал.\nТолщина задаётся на вкладке «Маркеры»; размер по свойству меняет толщину.')
    )
    interval_note.setWordWrap(True)
    dialog.interval_note = interval_note
    dialog.data.form.addRow(interval_note)
    dialog.label_by = dialog.data.combo("label_by", tr('Подписи точек'), [])
    dialog.data.check("text_numbers", tr('Читать числа в текстовых колонках'))
    dialog.data.check("include_interpolated", tr('Включить актуальные интерполяции'))
    dialog.data.check("include_calculated", tr('Включить расчётные показатели'), True)
    dialog.data.check("include_original", tr('Включить исходные показатели'), True)
    dialog.data.number("bins", tr('Интервалы гистограммы'), 20, 1, 500, integer=True)
    dialog.data.check("density", tr('Нормировать гистограмму на плотность'))
    dialog.data.check("connect", tr('Соединять точки внутри группы'))
    dialog.line_group_by = dialog.data.combo("line_group_by", tr('Группа линий'), [])
    dialog.line_group_by.setToolTip(tr('Без привязки используется категориальная группа цвета.'))
    dialog.data.check("connect_across_gaps", tr('Соединять через исключённые пробы'), True)
    dialog.data.fields['connect'][0].toggled.connect(dialog.update_enabled)
    dialog.data.number("line_width", tr('Толщина линий'), 1, 0.1, 10)
    dialog.tracks = QListWidget()
    dialog.data.section(tr('Треки и матрица'))
    dialog.tracks.setMaximumHeight(145)
    dialog.tracks.setDragDropMode(QListWidget.DragDropMode.InternalMove)
    dialog.tracks_label = QLabel(tr('Треки планшета\n(порядок: перетаскивание)'))
    dialog.data.form.addRow(dialog.tracks_label, dialog.tracks)
    dialog.data.section(tr('Дополнительные показатели'))
    dialog.library_form = Form()
    dialog.library_form.section(tr('Показатели выбранного вида'))
    dialog.library_inputs = {}
    for key in dict.fromkeys(key for kind in library_kinds() for key in library_parameters(kind) if key != 'matrix'):
        dialog.library_inputs[key] = dialog.library_form.combo(key, tr(ROLE_LABELS.get(key.lower(), 'Дополнительный показатель: ' + key)), [])
    dialog.data.form.addRow(dialog.library_form)
    dialog.kind_options = Form()
    dialog.kind_options.section(tr('Параметры выбранного вида'))
    dialog.option_definitions = {}
    for kind in library_kinds():
        for option in controls_for(kind).options:
            dialog.option_definitions.setdefault(option.key, option)
    for key, option in dialog.option_definitions.items():
        if option.choices:
            dialog.kind_options.combo(key, tr(option.label), [(tr(CHOICE_LABELS.get(value, value)), value) for value in option.choices])
        else:
            dialog.kind_options.number(key, tr(option.label), option.default, option.minimum, option.maximum,
                                     integer=type(option.default) is int)
    dialog.data.form.addRow(dialog.kind_options)
    dialog.expert_options = QCheckBox(tr('Экспертные параметры библиотеки'))
    dialog.data.form.addRow(dialog.expert_options)
    dialog.expert_form = Form()
    dialog.library_options = dialog.expert_form.text('options', tr('Дополнительные параметры (JSON)'), '{}')
    dialog.library_options.setToolTip(tr('Необязательные параметры для опытных пользователей. Могут переопределять оформление.'))
    dialog.data.form.addRow(dialog.expert_form)
    dialog.expert_options.toggled.connect(dialog.expert_form.setVisible)
    dialog.expert_form.hide()
    dialog.library_options.textEdited.connect(dialog.load_kind_options)
    _finish_form(dialog.data)
    dialog.data.sections.addStretch()


def _build_axes_and_markers(dialog):
    dialog.x_form = _form_page(dialog, tr('Ось X'), 'arrow-down-wide-narrow', axis_form)
    dialog.y_form = _form_page(dialog, tr('Ось Y'), 'arrow-down-wide-narrow', axis_form)
    dialog.x_target = QComboBox()
    dialog.x_target.addItem(tr('Общие настройки X'), "")
    dialog.x_form.sections.itemAt(0).widget().layout().insertRow(0, tr('Ось X планшета'), dialog.x_target)
    dialog.previous_x_target = ""
    dialog.x_target.currentIndexChanged.connect(dialog.switch_x_target)
    _finish_form(dialog.x_form)
    _finish_form(dialog.y_form)
    dialog.z_form = _form_page(dialog, tr('Ось Z'), 'arrow-down-wide-narrow', axis_form)
    _finish_form(dialog.z_form)
    dialog.style = _form_page(dialog, tr('Маркеры'), 'mouse-pointer-2')
    dialog.style.section(tr('Привязки оформления'))
    dialog.color_by = dialog.style.combo("color_by", tr('Цвет по свойству'), [])
    dialog.style.combo(
        "color_mode",
        tr('Режим цвета'),
        [(tr('Категории'), "category"), (tr('Числовая шкала'), "numeric")],
    )
    dialog.shape_by = dialog.style.combo("shape_by", tr('Форма по свойству'), [])
    dialog.size_by = dialog.style.combo("size_by", tr('Размер по свойству'), [])
    dialog.style.combo(
        "size_mode",
        tr('Режим размера'),
        [(tr('Числовая шкала'), "numeric"), (tr('Категории'), "category")],
    )
    dialog.style.section(tr('Символы'))
    dialog.style.text("color", tr('Основной цвет'), "#0072B2", color=True)
    dialog.style.text("edge_color", tr('Обводка маркера'), "#202020", color=True)
    dialog.style.combo("marker", tr('Маркер'), MARKERS)
    dialog.style.number("marker_size", tr('Диаметр маркера, pt'), 7, 1, 60)
    dialog.style.number(
        "interval_width", tr('Толщина интервальной пробы, pt'), 2, 0.1, 30
    )
    dialog.style.number("edge_width", tr('Толщина обводки, pt'), 0.6, 0, 10)
    dialog.style.number("alpha", tr('Прозрачность (1 — непрозр.)'), 0.8, 0, 1)
    dialog.style.section(tr('Цвет и размер по значениям'))
    dialog.style.text("missing_color", tr('Цвет пропусков шкалы'), "#999999", color=True)
    dialog.style.number("size_min", tr('Минимальный диаметр, pt'), 4, 1, 60)
    dialog.style.number("size_max", tr('Максимальный диаметр, pt'), 14, 1, 60)
    for key, label in [
        ("size_value_min", tr('Минимум показателя размера')),
        ("size_value_max", tr('Максимум показателя размера')),
        ("color_min", tr('Минимум цветовой шкалы')),
        ("color_max", tr('Максимум цветовой шкалы')),
    ]:
        dialog.style.text(key, label + tr(' (авто)'), optional=True)
    dialog.style.combo(
        "cmap",
        tr('Цветовая шкала'),
        [],
    )
    dialog.style.text("colorbar_label", tr('Подпись цветовой шкалы'))
    dialog.style.combo("color_scale", tr('Нормировка цветовой шкалы'), [
        (tr('Линейная'), 'linear'), (tr('Логарифмическая'), 'log'), (tr('Числовые классы'), 'classes'),
    ])
    class_edges = dialog.style.text("color_class_edges", tr('Границы классов (через ;)'))
    class_edges.setPlaceholderText("0; 1,5; 10")
    class_edges.setToolTip(tr(
        'Укажите не менее трёх возрастающих границ. '
        'Значение на внутренней границе относится к следующему классу. '
        'Значения вне диапазона получают цвет крайнего класса.'
    ))
    dialog.style.combo("colorbar_location", tr('Положение цветовой шкалы'), [
        (tr('Справа'), 'right'), (tr('Слева'), 'left'), (tr('Сверху'), 'top'),
        (tr('Снизу'), 'bottom'), (tr('Скрыть'), 'none'),
    ])
    _finish_form(dialog.style)
    dialog.category_table = QTableWidget(0, 3)
    dialog.category_table.setHorizontalHeaderLabels(
        [tr('Канал / категория'), tr('Стиль'), tr('Подпись легенды')]
    )
    dialog.category_entries = []
    install_cell_reset_menu(dialog.category_table, dialog.reset_category_parameter)
    dialog.category_table.setItemDelegateForColumn(1, StyleDelegate(dialog.category_table, lambda: dialog.category_entries))
    dialog.tabs.addTab(dialog.category_table, icon('tags'), tr('Категории'))


def _build_layout(dialog):
    dialog.layout_form = _form_page(dialog, tr('Оформление'), 'sliders-horizontal')
    dialog.layout_form.section(tr('Заголовки и подписи'))
    for key, label in [
        ("title", tr('Заголовок')),
        ("subtitle", tr('Подзаголовок')),
        ("annotation", tr('Аннотация (\\n — новая строка)')),
    ]:
        dialog.layout_form.text(key, label)
    dialog.layout_form.font_selector("font_family", tr('Шрифт'), "DejaVu Sans")
    dialog.layout_form.number(
        "title_size", tr('Размер заголовка, pt'), 13, 6, 60, integer=True
    )
    dialog.layout_form.number(
        "label_size", tr('Размер подписей точек, pt'), 8, 5, 30, integer=True
    )
    dialog.layout_form.section(tr('Легенда'))
    dialog.layout_form.check("legend", tr('Показывать легенду'), True)
    dialog.layout_form.combo(
        "legend_position",
        tr('Положение легенды'),
        [
            "best",
            "upper right",
            "upper left",
            "lower left",
            "lower right",
            "center left",
            "center right",
            "upper center",
            "lower center",
            "center",
        ],
    )
    dialog.layout_form.number(
        "legend_columns", tr('Колонки легенды'), 1, 1, 10, integer=True
    )
    dialog.layout_form.combo("legend_order", tr('Порядок категорий в легенде'), [
        (tr('По умолчанию'), 'default'), (tr('По возрастанию'), 'ascending'),
        (tr('По убыванию'), 'descending'),
    ])
    dialog.layout_form.number(
        "legend_size", tr('Размер легенды, pt'), 8, 5, 30, integer=True
    )
    dialog.layout_form.check("legend_frame", tr('Рамка легенды'), True)
    dialog.layout_form.check("legend_visible_only", tr('Только видимые категории'))
    dialog.layout_form.number(
        "interpretation_alpha", tr('Прозрачность интерпретационного слоя'), 0.18, 0, 1
    )
    dialog.layout_form.check(
        "interpretation_labels", tr('Подписи интерпретационных областей'), True
    )
    dialog.layout_form.check(
        "interpretation_info", tr('Идентификатор и редакция схемы на рисунке'), True
    )
    dialog.layout_form.check('show_uncertainty', tr('Показывать интервалы неопределённости'), True)
    dialog.layout_form.check('measurement_error_bars', tr('Интервалы неопределённости измерений'))
    dialog.layout_form.check('qc_outline', tr('Обводка измерений с отметкой QC'))
    dialog.layout_form.check('show_result_type', tr('Тип результата на осях'))
    source_frame = dialog.layout_form.check('interpretation_source_frame', tr('Рамка источника'))
    dialog.show_field(source_frame, dialog.interpretation_owner is not None)
    dialog.layout_form.section(tr('Стратиграфия'))
    dialog.layout_form.number(
        "stratigraphy_alpha", tr('Прозрачность стратиграфической колонки'), 0.16, 0, 1
    )
    dialog.layout_form.check(
        "stratigraphy_labels", tr('Подписи стратиграфических интервалов'), True
    )
    dialog.layout_form.check(
        "stratigraphy_info", tr('Схема и система глубин под графиком'), True
    )
    dialog.layout_form.number(
        "stratigraphy_label_size",
        tr('Размер подписей стратиграфии, pt'),
        8,
        5,
        30,
        integer=True,
    )
    dialog.layout_form.section(tr('Сетка и фон'))
    for key, label, default in [
        ("grid_color", tr('Цвет сетки'), "#b0b0b0"),
        ("background", tr('Фон'), "#ffffff"),
    ]:
        dialog.layout_form.text(key, label, default, color=True)
    dialog.layout_form.number("grid_alpha", tr('Прозрачность сетки'), 0.45, 0, 1)
    dialog.layout_form.number("grid_width", tr('Толщина сетки'), 0.6, 0.1, 5)
    dialog.layout_form.combo("grid_style", tr('Линия сетки'), ["--", "-", ":", "-."])
    dialog.layout_form.check("grid_below", tr('Сетка под точками'), True)
    dialog.layout_form.section(tr('Размер и экспорт'))
    dialog.size_unit = QComboBox()
    for label, scale in (("mm", 1), ("cm", 10), ("in", 25.4)):
        dialog.size_unit.addItem(label, scale)
    dialog.layout_form.form.addRow(tr('Единицы размера'), dialog.size_unit)
    dialog.layout_form.number("width_mm", tr('Ширина'), 180, 30, 1000).setSuffix(" mm")
    dialog.layout_form.number("height_mm", tr('Высота'), 130, 30, 1000).setSuffix(" mm")
    dialog.size_unit.currentIndexChanged.connect(dialog.change_size_unit)
    dialog.layout_form.number("dpi", tr('DPI экспорта'), 300, 36, 1200, integer=True)
    dialog.layout_form.check("transparent", tr('Прозрачный фон при экспорте'))
    dialog.layout_form.section(tr('Поля рисунка'))
    dialog.layout_form.check("square_plot_area", tr('Квадратная область построения'))
    dialog.layout_form.check("comparison_grid", tr('Панели сравнения в две колонки'))
    aspect = dialog.layout_form.text("data_aspect", tr('Отношение масштабов Y/X'), optional=True)
    aspect.setToolTip(tr('1 задаёт одинаковый масштаб X и Y. Число имеет приоритет над квадратной областью построения.'))
    for key, label, default in [
        ("left", tr('Поле слева'), 0.14),
        ("right", tr('Правая граница'), 0.95),
        ("bottom", tr('Поле снизу'), 0.16),
        ("top", tr('Верхняя граница'), 0.84),
    ]:
        dialog.layout_form.number(key, label + tr(' (доля)'), default, 0, 1, decimals=3)
    dialog.layout_form.number("wspace", tr('Промежуток между треками'), 0.35, 0, 3)
    _finish_form(dialog.layout_form)


def _build_viewer(dialog, splitter, layout, top, compact):
    viewer = QWidget()
    viewer_layout = QVBoxLayout(viewer)
    dialog.figure = Figure(figsize=(180 / 25.4, 130 / 25.4), dpi=96)
    dialog.canvas = PreviewCanvas(dialog.figure)
    dialog.toolbar = dialog.make_toolbar(dialog.canvas)
    viewer_layout.addWidget(dialog.toolbar)
    dialog.canvas_scroll = QScrollArea()
    dialog.frame_notice = QLabel(dialog.canvas_scroll.viewport())
    dialog.frame_notice.setStyleSheet('QLabel { color: #241600; background: #ffdf91; padding: 6px; }')
    dialog.frame_notice.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    dialog.frame_notice.move(8, 8)
    dialog.frame_notice.hide()
    dialog.canvas_view = CanvasView(dialog.canvas, lambda: (dialog.spec.width_mm, dialog.spec.height_mm), dialog.canvas_scroll)
    dialog.canvas_view.add_to_toolbar(dialog.toolbar)
    dialog.viewport_timer = QTimer(dialog)
    dialog.viewport_timer.setSingleShot(True)
    dialog.viewport_timer.setInterval(220)
    dialog.viewport_timer.timeout.connect(dialog.request_navigation)
    dialog.canvas_view.changed.connect(dialog.viewport_timer.start)
    dialog.canvas_scroll.viewport().installEventFilter(dialog)
    viewer_layout.addWidget(dialog.canvas_scroll, 1)
    dialog.brush = QCheckBox(tr('Рамка'))
    dialog.brush.setAccessibleName(tr('Выделять рамкой → таблица'))
    dialog.canvas_view.mode_combo.setMaximumWidth(145)
    dialog.brush.setToolTip(tr('Выделять рамкой → таблица'))
    dialog.brush.toggled.connect(dialog.enable_brush)
    if compact:
        dialog.canvas_view.mode_combo.setMaximumWidth(140)
        for key in ('back', 'forward'):
            action = dialog.toolbar._actions[key]
            dialog.toolbar.removeAction(action)
            dialog.plot_actions_menu.addAction(action)
        dialog.export_button.setIcon(icon('log-out'))
        dialog.plot_actions_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        first_action = dialog.toolbar.actions()[0]
        dialog.toolbar.insertWidget(first_action, dialog.plot_actions_button)
        dialog.toolbar.insertWidget(first_action, dialog.brush)
        dialog.toolbar.addAction(dialog.export_button)
        dialog.plot_actions_menu.addAction(dialog.export_button)
        dialog.plot_actions_menu.addSeparator()
    else:
        top.addWidget(dialog.brush)
    dialog.plot_actions_menu.addAction(icon('save'), tr('Сохранить текущий масштаб осей'), dialog.capture_limits)
    visible_export = dialog.plot_actions_menu.addAction(icon('log-out'), tr('Экспортировать видимые границы осей…'),
                                                      dialog.export_visible_limits)
    dialog.plot_actions_menu.aboutToShow.connect(lambda: visible_export.setEnabled(dialog.can_export_visible_limits()))
    dialog.plot_actions_menu.addAction(icon('maximize'), tr('Автопределы'), dialog.reset_limits)
    splitter.addWidget(viewer)
    splitter.setSizes([440, 900])
    layout.addWidget(splitter, 1)
    dialog.status = QLabel(tr('Настройте график'))
    dialog.status.setWordWrap(True)
    layout.addWidget(dialog.status)


def initialize_editor(dialog):
    owner = dialog.owner
    interpretation_owner = dialog.interpretation_owner
    dialog.connect_canvas_events()
    dialog.sync_columns()
    # Plain names match only when unambiguous; group-specific source fields stay distinct.
    saved = owner.model.project.view.get(dialog.settings_key)
    if saved:
        try:
            dialog.spec = PlotSpec.from_dict(saved)
        except (TypeError, ValueError):
            dialog.status.setText(
                tr('Сохранённая конфигурация недоступна; создан новый график')
            )
    else:
        if interpretation_owner is None:
            dialog.spec.scope = 'global'
        dialog.spec.style_maps = deepcopy(
            owner.model.project.view.get("plot_palettes", {})
        )
        numeric_columns = [
            c.id for c in owner.model.project.columns if c.kind == "number"
        ]
        dialog.spec.x_id = numeric_columns[0] if numeric_columns else ""
        dialog.spec.y_id = (
            numeric_columns[1] if len(numeric_columns) > 1 else dialog.spec.x_id
        )
    dialog.load_controls()
    dialog.sync_shared_appearance()
    for form in (dialog.data, dialog.library_form, dialog.kind_options, dialog.expert_form, dialog.style, dialog.x_form, dialog.y_form, dialog.z_form, dialog.layout_form):
        form.edited.connect(dialog.schedule)
    dialog.tracks.itemChanged.connect(dialog.tracks_changed)
    dialog.tracks.model().rowsMoved.connect(dialog.tracks_changed)
    dialog.category_table.itemChanged.connect(dialog.category_changed)
    dialog.category_table.cellDoubleClicked.connect(dialog.choose_category_color)
    dialog.kind.currentIndexChanged.connect(dialog.kind_changed)
    owner.model.changed.connect(dialog.schedule)
    owner.proxy.layoutChanged.connect(dialog.table_scope_changed)
    owner.proxy.modelReset.connect(dialog.table_scope_changed)
    owner.proxy.filtersChanged.connect(dialog.table_scope_changed)
    owner.model.columnsInserted.connect(dialog.sync_columns)
    owner.model.columnsRemoved.connect(dialog.sync_columns)
    owner.sample_selection.changed.connect(dialog.selection_changed)
    dialog.loading = False
    dialog.update_enabled()
    dialog.schedule()
    add_parameter_tools(dialog, reset=dialog.restore_parameters, reset_text=tr('Сбросить оформление'),
                        exclude=(dialog.canvas_view.mode_combo,))
    dialog.plot_actions_menu.addAction(dialog.reset_defaults_button.icon(), dialog.reset_defaults_button.text(),
                                     dialog.reset_defaults_button.click)
    dialog.plot_actions_menu.addAction(dialog.whats_this_action)
    dialog.reset_defaults_button.hide()
    dialog.help_button.hide()
    dialog.canvas_view.bind_project(owner, dialog.settings_key.removesuffix('_current'))
    if interpretation_owner is None:
        owner.analysis_scope.saved_scopes_changed.connect(dialog.refresh_scopes)
        owner.analysis_scope.saved_scope_renamed.connect(dialog.rename_scope)
        owner.analysis_scope.changed.connect(dialog.analysis_scope_changed)
        dialog.refresh_scopes()
    else:
        owner.analysis_scope.bind(dialog.scope, dialog.schedule)


def _form_page(dialog, title, icon_name, factory=Form):
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
    dialog.tabs.addTab(scroll, icon(icon_name), title)
    widget = factory(scroll.viewport())
    scroll.setWidget(widget)
    return widget


def _finish_form(widget):
    widget.add_reset_button()
    for form in widget.findChildren(QFormLayout):
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
