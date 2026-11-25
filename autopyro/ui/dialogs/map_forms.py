"""Build the map controls, layer panels and canvas."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QToolButton,
)
from matplotlib.figure import Figure

from ..i18n import tr
from ..theme import icon
from ..widgets.figure_canvas import CanvasView, FittedFigureCanvas
from ..widgets.forms import Form
from ..widgets.map_layers import LayerOrderTable
from ..widgets.navigation import NavigationToolbar, add_tab_sections_menu
from ..widgets.parameter_reset import install_cell_reset_menu
from ..widgets.parameters import add_parameter_tools
from ..widgets.spatial import AttributeModel, configure_crs_combo
from ..widgets.styles import StyleDelegate
from .map_rendering import initialize_timers


def build(sheet):
    layout = QVBoxLayout(sheet)
    splitter = QSplitter()
    layout.addWidget(splitter, 1)
    sheet.tabs = QTabWidget()
    sheet.tabs.setMinimumWidth(380)
    splitter.addWidget(sheet.tabs)
    _coordinates(sheet)
    _sample_style(sheet)
    _filters(sheet)
    _sites(sheet)
    _layout_controls(sheet)
    _sample_categories(sheet)
    _layer_panels(sheet)
    _site_results(sheet)
    sheet.sections_button = add_tab_sections_menu(sheet.tabs, tr('Разделы свойств карты'))
    _viewer(sheet, splitter)
    _commands(sheet, layout)
    _status(sheet, layout)
    initialize_timers(sheet)
    _connect_and_restore(sheet)


def _coordinates(sheet):
    sheet.data = Form(compact=True)
    sheet.data.section(tr('Источник координат'))
    sheet.location = sheet.data.combo(
        "location_mode",
        tr('Координаты'),
        [(tr('Из колонок проб'), "columns"), (tr('По ключу в точечном слое'), "layer"),
         (tr('Только пространственные слои'), 'layers')],
    )
    sheet.location.currentIndexChanged.connect(sheet.sync_coordinate_controls)
    sheet.data.combo(
        "scope",
        tr('Выборка'),
        [
            (tr('Текущая таблица'), "filtered"),
            (tr('Все пробы'), "all"),
            (tr('Выделенные пробы'), "selected"),
        ],
    )
    sheet.column_combos = []
    for key, label in [
        ("x_id", tr('X / долгота')),
        ("y_id", tr('Y / широта')),
        ("sample_key", tr('Ключ пробы / ствола')),
    ]:
        combo = sheet.data.combo(key, label, [])
        sheet.column_combos.append(combo)
    for key, label in [('source_crs', tr('CRS координат проб')), ('target_crs', tr('CRS карты'))]:
        combo = sheet.data.combo(key, label, [])
        configure_crs_combo(combo, getattr(sheet.spec, key))
        sheet.data.defaults[key] = 'auto'
        combo.editTextChanged.connect(sheet.data.edited)
    sheet.visualization_crs = sheet.data.fields['target_crs'][0]
    crs_row = sheet.data.form.takeRow(sheet.visualization_crs)
    crs_row.labelItem.widget().deleteLater()
    sheet.crs_status = QLabel()
    sheet.crs_status.setWordWrap(True)
    sheet.data.form.addRow(sheet.crs_status)
    sheet.point_layer = sheet.data.combo("point_layer", tr('Точечный слой'), [])
    sheet.point_layer.setToolTip(tr('При привязке по ключу используются также скважины, скрытые фильтром или видимостью слоя.'))
    sheet.layer_key = sheet.data.combo("layer_key", tr('Поле ключа слоя'), [])
    sheet.point_layer.currentIndexChanged.connect(sheet.sync_layer_fields)
    sheet.data.check("text_numbers", tr('Читать числа в текстовых колонках'))
    sheet.data.check(
        "include_interpolated", tr('Показывать актуальные интерполированные значения')
    )
    sheet.add_form(sheet.data, tr('Координаты'), 'map')


def _sample_style(sheet):
    sheet.style = Form(compact=True)
    sheet.style.section(tr('Привязки оформления'))
    for key, label in [
        ("color_by", tr('Цвет / геохимическое семейство')),
        ("size_by", tr('Размер по свойству')),
        ("label_by", tr('Подпись')),
    ]:
        combo = sheet.style.combo(key, label, [])
        sheet.column_combos.append(combo)
    sheet.style.combo(
        "color_mode",
        tr('Цвет'),
        [(tr('Категории'), "category"), (tr('Числовая шкала'), "numeric")],
    )
    sheet.style.combo(
        "size_mode",
        tr('Размер'),
        [(tr('Числовая шкала'), "numeric"), (tr('Категории'), "category")],
    )
    sheet.style.section(tr('Символы'))
    for key, label, value in [
        ("color", tr('Основной цвет'), "#0072B2"),
        ("missing_color", tr('Цвет пропусков'), "#999999"),
    ]:
        sheet.style.text(key, label, value, color=True)
    for key, label, value in [
        ("marker_size", tr('Диаметр, pt'), 7),
        ("size_min", tr('Минимум размера, pt'), 4),
        ("size_max", tr('Максимум размера, pt'), 16),
    ]:
        sheet.style.number(key, label, value, 1, 60)
    sheet.style.number("alpha", tr('Непрозрачность'), 0.85, 0, 1)
    from autopyro.visualization.plotting import MARKERS

    sheet.style.combo("marker", tr('Символ'), MARKERS)
    sheet.style.text("edge_color", tr('Обводка символа'), "#253341", color=True)
    sheet.style.number("edge_width", tr('Толщина обводки, pt'), 0.5, 0, 10)
    sheet.style.section(tr('Цветовая шкала'))
    sheet.style.combo("cmap", tr('Цветовая шкала'), [])
    sheet.style.text("color_min", tr('Минимум шкалы цвета (авто)'), optional=True)
    sheet.style.text("color_max", tr('Максимум шкалы цвета (авто)'), optional=True)
    sheet.add_form(sheet.style, tr('Символы'), 'mouse-pointer-2')


def _filters(sheet):
    sheet.filters = Form(compact=True)
    sheet.filters.section(tr('Источники фильтров'))
    for key, label in [
        ("depth_id", tr('Глубина / верх отбора')),
        ("depth_bottom_id", tr('Низ отбора (необязательно)')),
        ("strat_id", tr('Стратиграфия / подразделение')),
        ("date_id", tr('Дата измерения / отбора')),
    ]:
        combo = sheet.filters.combo(key, label, [])
        sheet.column_combos.append(combo)
    sheet.filters.section(tr('Ограничения выборки'))
    sheet.filters.text("depth_min", tr('Глубина от (единица колонки)'), optional=True)
    sheet.filters.text("depth_max", tr('Глубина до (включительно)'), optional=True)
    sheet.filters.text("strat_value", tr('Точное значение подразделения')).setPlaceholderText(
        tr('Пусто — без фильтра'))
    sheet.filters.text("date_from", tr('Дата от, YYYY-MM-DD'))
    sheet.filters.text("date_to", tr('Дата до, YYYY-MM-DD'))
    label = QLabel(
        tr('Фильтры дополняют область таблицы. Если задан низ отбора, учитывается положительная длина пересечения; при от=до — попадание точки в [верх, низ). Даты сравниваются по календарной дате ISO-записи.')
    )
    label.setWordWrap(True)
    sheet.filters.form.addRow(label)
    sheet.add_form(sheet.filters, tr('Фильтры'), 'funnel')


def _sites(sheet):
    sheet.sites = Form(compact=True)
    sheet.sites.section(tr('Слой площадок'))
    sheet.site_layer = sheet.sites.combo(
        "site_layer", tr('Полигональный слой площадок'), []
    )
    sheet.site_label = sheet.sites.combo("site_label", tr('Поле имени площадки'), [])
    sheet.site_layer.currentIndexChanged.connect(sheet.sync_layer_fields)
    for key, label in [
        ("site_value", tr('Показатель агрегации')),
        ("site_family", tr('Геохимическое семейство')),
    ]:
        combo = sheet.sites.combo(key, label, [])
        sheet.column_combos.append(combo)
    sheet.sites.section(tr('Агрегация'))
    sheet.sites.combo(
        "aggregate",
        tr('Способ расчёта'),
        [
            (tr('Среднее'), "mean"),
            (tr('Медиана'), "median"),
            (tr('Минимум'), "min"),
            (tr('Максимум'), "max"),
        ],
    )
    sheet.sites.check("show_aggregates", tr('Показать агрегаты вместо отдельных проб'))
    label = QLabel(
        tr('Проба принадлежит площадке только при единственном покрывающем полигоне. Общие границы и перекрытия дают неоднозначность. N — все пробы; n — числовые значения. Цвет агрегата — значение, размер — N; семейство показано в таблице (мода, ничья отдельно).')
    )
    label.setWordWrap(True)
    sheet.sites.form.addRow(label)
    sheet.add_form(sheet.sites, tr('Площадки'), 'map')


def _layout_controls(sheet):
    sheet.layout_form = Form(compact=True)
    sheet.layout_form.section(tr('Заголовки и подписи'))
    for key, label, value in [
        ("title", tr('Заголовок'), tr('Локальная карта')),
        ("x_label", tr('Подпись X'), ""),
        ("y_label", tr('Подпись Y'), ""),
    ]:
        sheet.layout_form.text(key, label, value)
    sheet.layout_form.section(tr('Диапазон координат'))
    for key, label in [
        ("xmin", tr('Минимум X')),
        ("xmax", tr('Максимум X')),
        ("ymin", tr('Минимум Y')),
        ("ymax", tr('Максимум Y')),
    ]:
        sheet.layout_form.text(key, label + tr(' (авто)'), optional=True)
    sheet.layout_form.section(tr('Легенда и сетка'))
    sheet.layout_form.check("legend", tr('Легенда'), True)
    sheet.layout_form.check("grid", tr('Сетка'), True)
    sheet.layout_form.number(
        "legend_size", tr('Шрифт легенды, pt'), 8, 5, 30, integer=True
    )
    sheet.layout_form.section(tr('Размер и экспорт'))
    sheet.layout_form.number("width_mm", tr('Ширина, мм'), 200, 30, 1000)
    sheet.layout_form.number("height_mm", tr('Высота, мм'), 160, 30, 1000)
    sheet.layout_form.number("dpi", tr('DPI экспорта'), 300, 36, 1200, integer=True)
    sheet.layout_form.section(tr('Шрифты'))
    sheet.layout_form.font_selector("font_family", tr('Шрифт'), "DejaVu Sans")
    sheet.layout_form.number(
        "label_size", tr('Размер подписей осей, pt'), 10, 5, 30, integer=True
    )
    sheet.layout_form.number(
        "tick_size", tr('Размер делений, pt'), 8, 5, 30, integer=True
    )
    sheet.layout_form.number(
        "annotation_size", tr('Подписи объектов, pt'), 8, 5, 30, integer=True
    )
    sheet.layout_form.section(tr('Сетка и фон'))
    sheet.layout_form.text("background", tr('Фон'), "#ffffff", color=True)
    sheet.layout_form.text("grid_color", tr('Цвет сетки'), "#aaaaaa", color=True)
    sheet.layout_form.number("grid_alpha", tr('Прозрачность сетки'), 0.3, 0, 1)
    sheet.layout_form.number("grid_width", tr('Толщина сетки'), 0.6, 0.1, 5)
    sheet.add_form(sheet.layout_form, tr('Оформление'), 'sliders-horizontal')
    sheet.layout_form.section(tr('Легенда и масштаб'))
    sheet.layout_form.text('legend_title', tr('Заголовок легенды'))
    sheet.layout_form.combo('legend_placement', tr('Положение легенды'), [(tr('Справа от карты'), 'outside'),
                           (tr('Вверху справа'), 'upper_right'), (tr('Внизу слева'), 'lower_left'), (tr('Внизу справа'), 'lower_right')])
    sheet.layout_form.check('scale_bar', tr('Масштабная линейка'))
    sheet.layout_form.check('north_arrow', tr('Стрелка истинного севера'))


def _sample_categories(sheet):
    sheet.category_table = QTableWidget(0, 3)
    sheet.category_table.setHorizontalHeaderLabels(
        [tr('Канал / категория'), tr('Стиль'), tr('Подпись')]
    )
    sheet.entries = []
    install_cell_reset_menu(sheet.category_table, sheet.reset_category_parameter)
    sheet.category_table.setItemDelegateForColumn(1, StyleDelegate(sheet.category_table, lambda: sheet.entries))
    sheet.tabs.addTab(sheet.category_table, icon('tags'), tr('Категории'))


def _layer_panels(sheet):
    sheet.layer_tabs = QTabWidget()
    sheet.layer_tabs.setMinimumHeight(140)
    _layer_list(sheet)
    attributes_layout = _layer_attributes(sheet)
    _layer_categories(sheet)
    sheet.build_layer_tools()
    attributes_layout.removeWidget(sheet.attribute_table)
    attribute_splitter = QSplitter()
    attribute_splitter.addWidget(sheet.attribute_table)
    attribute_splitter.addWidget(sheet.feature_card)
    attribute_splitter.setSizes([650, 300])
    attributes_layout.addWidget(attribute_splitter, 1)


def _layer_list(sheet):
    sheet.layer_table = LayerOrderTable(0, 7)
    sheet.layer_table.reordered.connect(sheet.reorder_layers)
    sheet.layer_table.doubleClicked.connect(sheet.zoom_to_layer)
    install_cell_reset_menu(sheet.layer_table, sheet.reset_layer_parameter)
    for column in (1, 2):
        sheet.layer_table.setItemDelegateForColumn(column, StyleDelegate(sheet.layer_table))
    sheet.layer_table.setHorizontalHeaderLabels(
        [tr('Показывать / имя'), tr('Обводка'), tr('Заливка'), "Alpha", tr('Линия'), "ID", tr('Цвет по атрибуту')]
    )
    sheet.layer_table.setColumnHidden(5, True)
    sheet.layer_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    layer_panel = QWidget()
    layer_layout = QVBoxLayout(layer_panel)
    order_buttons = QHBoxLayout()
    import_button = QPushButton(icon('import'), tr('Импорт слоя…'))
    import_button.clicked.connect(sheet.import_layer)
    order_buttons.addWidget(import_button)
    sheet.remove_layer_button = QPushButton(icon('trash'), tr('Удалить выбранный слой'))
    sheet.remove_layer_button.setEnabled(False)
    sheet.remove_layer_button.clicked.connect(sheet.remove_layer)
    sheet.layer_table.itemSelectionChanged.connect(lambda: sheet.remove_layer_button.setEnabled(
        bool(sheet.layer_table.selectedIndexes())))
    order_buttons.addWidget(sheet.remove_layer_button)
    for text, offset, image in [(tr('Вверх'), -1, 'upload'), (tr('Вниз'), 1, 'download')]:
        button = QPushButton(icon(image), text)
        button.clicked.connect(lambda _, step=offset: sheet.reorder_layers(sheet.layer_table.currentRow(), sheet.layer_table.currentRow() + step))
        order_buttons.addWidget(button)
    order_buttons.addStretch()
    layer_layout.addLayout(order_buttons)
    layer_layout.addWidget(sheet.layer_table)
    sheet.layer_tabs.addTab(layer_panel, icon('layers'), tr('Слои'))


def _layer_attributes(sheet):
    attributes = QWidget()
    attributes_layout = QVBoxLayout(attributes)
    sheet.attribute_layer = QComboBox()
    attributes_layout.addWidget(sheet.attribute_layer)
    sheet.attribute_info = QLabel()
    sheet.attribute_info.setWordWrap(True)
    attributes_layout.addWidget(sheet.attribute_info)
    sheet.attribute_model = AttributeModel(sheet)
    sheet.attribute_table = QTableView()
    sheet.attribute_table.setModel(sheet.attribute_model)
    sheet.feature_highlights = []
    sheet.map_press = None
    sheet.attribute_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
    sheet.attribute_table.selectionModel().selectionChanged.connect(sheet.highlight_attributes)
    sheet.attribute_table.doubleClicked.connect(sheet.zoom_to_attribute)
    sheet.attribute_table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
    sheet.attribute_table.setHorizontalScrollMode(QTableView.ScrollMode.ScrollPerPixel)
    header = sheet.attribute_table.horizontalHeader()
    header.setResizeContentsPrecision(100)
    header.setMinimumSectionSize(45)
    header.setMaximumSectionSize(400)
    header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    attributes_layout.addWidget(sheet.attribute_table)
    sheet.layer_tabs.addTab(attributes, icon('table-2'), tr('Атрибуты'))
    sheet.attribute_layer.currentIndexChanged.connect(sheet.sync_attributes)
    sheet.layer_table.itemSelectionChanged.connect(sheet.select_attribute_layer)
    return attributes_layout


def _layer_categories(sheet):
    category_panel = QWidget()
    category_layout = QVBoxLayout(category_panel)
    sheet.category_layer = QComboBox()
    category_layout.addWidget(sheet.category_layer)
    sheet.layer_category_info = QLabel()
    category_layout.addWidget(sheet.layer_category_info)
    sheet.layer_category_table = QTableWidget(0, 3)
    sheet.layer_category_table.setHorizontalHeaderLabels([tr('Значение'), tr('Цвет'), tr('Подпись')])
    sheet.layer_category_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    sheet.layer_category_table.setItemDelegateForColumn(1, StyleDelegate(sheet.layer_category_table))
    category_layout.addWidget(sheet.layer_category_table)
    reset_categories = QPushButton(icon('rotate-ccw'), tr('Сбросить категории'))
    reset_categories.clicked.connect(sheet.reset_layer_categories)
    category_layout.addWidget(reset_categories)
    sheet.layer_category_keys = []
    sheet.layer_tabs.addTab(category_panel, icon('tags'), tr('Категории слоя'))
    sheet.category_layer.currentIndexChanged.connect(sheet.sync_layer_categories)
    sheet.layer_category_table.itemChanged.connect(sheet.layer_category_edited)


def _site_results(sheet):
    site_results = QWidget()
    site_results_layout = QVBoxLayout(site_results)
    boundary_note = QLabel(tr('N включает пробы на границе одного полигона. Если пробу покрывают несколько полигонов, она не входит в N. При присоединении областей граничным пробам атрибуты не назначаются.'))
    boundary_note.setWordWrap(True)
    site_results_layout.addWidget(boundary_note)
    sheet.site_table = QTableWidget(0, 9)
    sheet.site_table.setHorizontalHeaderLabels(
        [
            tr('Площадка'),
            "ID",
            tr('N проб'),
            tr('Из N на границе'),
            tr('n чисел'),
            tr('Способ'),
            tr('Значение'),
            tr('Семейство'),
            tr('Статус семейства'),
        ]
    )
    sheet.site_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    site_results_layout.addWidget(sheet.site_table, 1)
    sheet.tabs.addTab(site_results, icon('chart-column'), tr('Агрегаты'))


def _viewer(sheet, splitter):
    viewer = QWidget()
    vl = QVBoxLayout(viewer)
    crs_bar = QHBoxLayout()
    crs_bar.addWidget(QLabel(tr('CRS всей визуализации')))
    crs_bar.addWidget(sheet.visualization_crs, 1)
    sheet.visualization_crs.setToolTip(tr('Все слои и пробы преобразуются в выбранную CRS.'))
    vl.addLayout(crs_bar)
    sheet.build_view_tools(vl)
    sheet.figure = Figure(figsize=(8, 6), dpi=96)
    sheet.canvas = FittedFigureCanvas(sheet.figure)
    sheet.canvas.navigation_draw_requested.connect(sheet.render_map_view)
    sheet.toolbar = NavigationToolbar(sheet.canvas, sheet)
    vl.addWidget(sheet.toolbar)
    sheet.canvas_view = CanvasView(sheet.canvas, lambda: (sheet.spec.width_mm, sheet.spec.height_mm))
    sheet.canvas_view.add_to_toolbar(sheet.toolbar)
    sheet.canvas_scroll = sheet.canvas_view.scroll
    sheet.map_pages = QStackedWidget()
    sheet.map_start = QWidget()
    start_layout = QVBoxLayout(sheet.map_start)
    start_layout.addStretch()
    start_hint = QLabel(tr('Выберите колонки X и Y для проб или импортируйте локальный пространственный слой.'))
    start_hint.setWordWrap(True)
    start_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
    start_layout.addWidget(start_hint)
    start_actions = QHBoxLayout()
    start_actions.addStretch()
    sheet.configure_coordinates_button = QPushButton(icon('map'), tr('Настроить координаты'))
    sheet.configure_coordinates_button.clicked.connect(sheet.focus_map_coordinates)
    start_actions.addWidget(sheet.configure_coordinates_button)
    sheet.import_first_layer_button = QPushButton(icon('import'), tr('Импорт слоя…'))
    sheet.import_first_layer_button.clicked.connect(sheet.import_layer)
    start_actions.addWidget(sheet.import_first_layer_button)
    start_actions.addStretch()
    start_layout.addLayout(start_actions)
    start_layout.addStretch()
    sheet.map_pages.addWidget(sheet.map_start)
    sheet.map_pages.addWidget(sheet.canvas_scroll)
    vl.addWidget(sheet.map_pages, 1)
    sheet.map_splitter = QSplitter(Qt.Orientation.Vertical)
    sheet.map_splitter.setChildrenCollapsible(False)
    sheet.map_splitter.addWidget(viewer)
    sheet.map_splitter.addWidget(sheet.layer_tabs)
    sheet.map_splitter.setStretchFactor(0, 3)
    sheet.map_splitter.setStretchFactor(1, 1)
    sheet.map_splitter.setSizes([600, 220])
    splitter.addWidget(sheet.map_splitter)
    splitter.setSizes([430, 900])


def _commands(sheet, layout):
    commands = QHBoxLayout()
    layout.addLayout(commands)
    operations = QToolButton(sheet)
    operations.setText(tr('Операции карты'))
    operations.setIcon(icon('map'))
    operations.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    operations.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    menu = QMenu(operations)
    operations.setMenu(menu)
    commands.addWidget(operations)
    commands.addStretch(1)
    for image, text, slot in [
        ('refresh-cw', tr('Обновить карту'), sheet.redraw),
        ('save', tr('Запомнить видимые пределы'), sheet.capture_limits),
        ('layers', tr('Присоединить области к пробам'), sheet.spatial_join_dialog),
        ('chart-column', tr('Геохимия по областям…'), sheet.open_regional_geochemistry),
    ]:
        action = menu.addAction(icon(image), text)
        action.triggered.connect(slot)
    sheet.export_button = QAction(icon('log-out'), tr('Экспорт карты…'), sheet)
    sheet.interpolate_button = QAction(icon('spline'), tr('Пространственная интерполяция…'), sheet)
    for action, slot in ((sheet.export_button, sheet.export), (sheet.interpolate_button, sheet.open_interpolation)):
        action.triggered.connect(slot)
        action.setEnabled(False)
        button = QToolButton(sheet)
        button.setDefaultAction(action)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        commands.addWidget(button)


def _status(sheet, layout):
    sheet.status = QLabel(
        tr('Укажите колонки координат и CRS либо точный ключ в слое скважин. Онлайн-подложки не требуются.')
    )
    sheet.status.setWordWrap(True)
    sheet.status.setMaximumHeight(70)
    status_row = QHBoxLayout()
    status_row.addWidget(sheet.status, 1)
    sheet.sample_diagnostics_button = QPushButton(icon('shield-check'), tr('Диагностика проб…'))
    sheet.sample_diagnostics_button.setEnabled(False)
    sheet.sample_diagnostics_button.clicked.connect(sheet.show_sample_diagnostics)
    status_row.addWidget(sheet.sample_diagnostics_button)
    layout.addLayout(status_row)


def _connect_and_restore(sheet):
    owner = sheet.owner
    sheet.sync_columns()
    sheet.sync_layers()
    for form in sheet.forms():
        form.load(sheet.spec.to_dict())
        form.edited.connect(sheet.schedule)
    sheet.sync_layer_fields()
    sheet.data.load({"layer_key": sheet.spec.layer_key})
    sheet.category_table.itemChanged.connect(sheet.category_changed)
    sheet.layer_table.itemChanged.connect(sheet.layer_changed)
    owner.model.changed.connect(sheet.schedule)
    owner.proxy.layoutChanged.connect(sheet.schedule)
    owner.proxy.modelReset.connect(sheet.schedule)
    owner.proxy.filtersChanged.connect(sheet.schedule)
    owner.model.columnsInserted.connect(sheet.sync_columns)
    owner.model.columnsRemoved.connect(sheet.sync_columns)
    owner.sample_selection.changed.connect(sheet.selection_changed)
    sheet.connect_canvas_events()
    sheet.loading = False
    sheet.visualization_crs.currentIndexChanged.connect(sheet.visualization_crs_changed)
    sheet.visualization_crs.editTextChanged.connect(sheet.visualization_crs_changed)
    sheet.schedule()
    add_parameter_tools(sheet, reset=sheet.restore_parameters, reset_text=tr('Сбросить оформление'),
                        exclude=(sheet.canvas_view.mode_combo,))
    sheet.canvas_view.bind_project(owner, 'map')
    owner.analysis_scope.bind(sheet.data.fields['scope'][0], sheet.schedule)


def forms(sheet):
    return sheet.data, sheet.style, sheet.filters, sheet.sites, sheet.layout_form


def add_form(sheet, form, title, icon_name):
    form.add_reset_button()
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(form)
    sheet.tabs.addTab(scroll, icon(icon_name), title)
