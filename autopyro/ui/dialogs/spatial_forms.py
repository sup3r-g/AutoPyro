"""Build the spatial interpolation controls and arrange the worksheet."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from matplotlib.figure import Figure

from autopyro.spatial.mapping import crs_label, crs
from autopyro.spatial.spatial_interpolation import is_metric_projection

from ..i18n import tr
from ..theme import icon
from ..widgets.figure_canvas import CanvasView, FittedFigureCanvas
from ..widgets.forms import Form
from ..widgets.navigation import NavigationToolbar, add_tab_sections_menu
from ..widgets.parameters import add_parameter_tools
from .spatial_labels import IDW_REGION_MODES, INTERPOLATION_METHODS


def build(sheet, report, parent):
    layout = QVBoxLayout(sheet)
    splitter = sheet.interpolation_splitter = QSplitter()
    splitter.setChildrenCollapsible(False)
    layout.addWidget(splitter, 1)
    context = _settings_panel(sheet, report, splitter)
    _idw_view(sheet)
    _block_view(sheet)
    sheet.build_settings_comparison()
    sheet.build_kriging()
    sheet.build_partition_comparison()
    sheet.build_sample_preview()
    sheet.sections_button = add_tab_sections_menu(sheet.views, tr('Вкладки пространственной интерполяции'))
    splitter.addWidget(sheet.views)
    splitter.setSizes([430, 800])
    _footer(sheet, layout)
    _finish(sheet, report, parent, context)


def _settings_panel(sheet, report, splitter):
    controls = QWidget()
    controls_layout = QVBoxLayout(controls)
    _projection_controls(sheet, report, controls_layout)
    context = _source_controls(sheet, report)
    _idw_parameters(sheet, report, controls_layout)
    _attach_settings(sheet, controls, controls_layout, splitter)
    return context


def _projection_controls(sheet, report, controls_layout):
    target = report['spec']['target_crs']
    projection = QLabel(tr('CRS карты: {0}', crs_label(target)))
    projection.setWordWrap(True)
    controls_layout.addWidget(projection)
    if not is_metric_projection(crs(target)):
        warning = QLabel(tr('Для расчёта вернитесь на карту и выберите подходящую проецированную CRS в метрах.'))
        warning.setWordWrap(True)
        controls_layout.addWidget(warning)
        if sheet.map_dialog is not None:
            configure_crs = QPushButton(icon('map'), tr('Выбрать CRS карты…'))
            configure_crs.setAutoDefault(False)
            configure_crs.clicked.connect(sheet.configure_map_crs)
            controls_layout.addWidget(configure_crs)


def _source_controls(sheet, report):
    sheet.form = Form()
    sheet.form.section(tr('Источник данных'))
    sheet.scope = sheet.form.combo('scope', tr('Область'), [])
    if sheet.owner is not None:
        source_note = tr('CRS, привязка координат, фильтры карты и слои зафиксированы при открытии окна. Перед каждой проверкой или расчётом выбранная область и значения проб считываются из проекта заново.')
    else:
        source_note = tr('Для проверки и расчёта используется переданный снимок карты.')
    source = QLabel(source_note)
    source.setWordWrap(True)
    sheet.form.form.addRow(source)
    sheet.input_summary = QLabel()
    sheet.input_summary.setWordWrap(True)
    sheet.form.form.addRow(sheet.input_summary)
    sheet.value_selector = sheet.form.combo(
        "value_id",
        tr('Показатель'),
        [(tr('Выберите показатель интерполяции'), '')] +
        [(f"{c['name']} [{c['unit']}]" if c['unit'] else c['name'], c["id"])
         for c in report["columns"] if not c.get('interpolated')],
    )
    sheet.preflight_button = QPushButton(icon('shield-check'), tr('Проверить выборку'))
    sheet.preflight_button.setAutoDefault(False)
    sheet.preflight_button.clicked.connect(sheet.check_sample)
    sheet.form.form.addRow(sheet.preflight_button)
    context = sheet.form.text("context", tr('Сопоставимость:\nметод, стратиграфия, глубина'))
    context.setPlaceholderText(tr('Например, TOC верхнего триаса, 2000–3500 м, один метод анализа'))
    sheet.form.combo(
        "duplicate_policy",
        tr('Повторные координаты'),
        [
            (tr('Отклонить'), "reject"),
            (tr('Среднее по месту (явно)'), "mean"),
            (tr('Медиана по месту (явно)'), "median"),
        ],
    )
    return context


def _idw_parameters(sheet, report, controls_layout):
    sheet.form.section(tr('Метод интерполяции'))
    sheet.form.number("power", tr('Степень IDW'), 2, 0.1, 10)
    sheet.form.number("neighbors", tr('Максимум соседей'), 8, 3, 100, integer=True)
    sheet.form.number("min_neighbors", tr('Минимум соседей'), 3, 3, 100, integer=True)
    sheet.form.number("max_distance", tr('Радиус поиска, км'), 10, .001, 10000, decimals=5, storage_scale=1000)
    sheet.form.section(tr('Расчётная сетка'))
    sheet.form.number(
        "grid_size", tr('Узлов сетки по каждой оси'), 60, 10, 200, integer=True
    )
    sheet.form.section(tr('Палеогеографические области IDW'))
    sheet.region_mode = sheet.form.combo('region_mode', tr('Режим областей IDW'),
                                      [(tr(label), key) for key, label in IDW_REGION_MODES.items()])
    polygons = [(layer['name'], layer['id']) for layer in report['layers'] if layer['features'] and
                all(f['geometry']['type'] in ('Polygon', 'MultiPolygon') for f in layer['features'])]
    sheet.region_layer = sheet.form.combo('region_layer', tr('Полигональный слой'), polygons)
    sheet.region_field = sheet.form.combo('region_field', tr('Поле области'), [])
    sheet.region_key = sheet.form.combo('region_key', tr('Область'), [])
    preferred = next((i for i, (name, _) in enumerate(polygons) if name.casefold() == 't3'), 0)
    sheet.region_layer.setCurrentIndex(preferred)
    sheet.region_layer.currentIndexChanged.connect(sheet.update_region_fields)
    sheet.region_field.currentIndexChanged.connect(sheet.update_region_keys)
    sheet.region_mode.currentIndexChanged.connect(sheet.update_region_controls)
    sheet.update_region_fields()
    sheet.update_region_controls()
    sheet.form.section(tr('Блочная проверка IDW'))
    sheet.form.number('block_size_km', tr('Размер блока, км'), 25, .001, 10000, decimals=3)
    buffer_field = sheet.form.number('validation_buffer_km', tr('Буфер проверки, км'), 0, 0, 10000, decimals=3)
    buffer_field.setToolTip(tr('Дополнительный отступ от границы проверочного блока. Ноль исключает только сам блок.'))
    sheet.form.edited.connect(sheet.invalidate)
    controls_layout.addWidget(sheet.form)


def _attach_settings(sheet, controls, controls_layout, splitter):
    note = QLabel(tr('Для расчёта нужно от 4 до 1000 уникальных локаций с числовыми значениями.'))
    note.setWordWrap(True)
    controls_layout.addWidget(note)
    controls_layout.addStretch()
    scroll = sheet.interpolation_settings_scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(controls)
    scroll.setMinimumWidth(controls.minimumSizeHint().width() + scroll.verticalScrollBar().sizeHint().width()
                           + 2 * scroll.frameWidth())
    splitter.addWidget(scroll)


def _idw_view(sheet):
    sheet.figure = Figure(figsize=(8, 6), dpi=96)
    sheet.canvas = FittedFigureCanvas(sheet.figure)
    viewer = QWidget()
    viewer_layout = QVBoxLayout(viewer)
    surface_actions = QHBoxLayout()
    sheet.run_idw_button = QPushButton(icon('calculator'), tr('Рассчитать поверхность IDW'))
    sheet.run_idw_button.clicked.connect(sheet.calculate)
    surface_actions.addWidget(sheet.run_idw_button)
    surface_actions.addStretch()
    viewer_layout.addLayout(surface_actions)
    sheet.idw_intro = QLabel(tr('Выберите показатель, например TOC, и нажмите «Проверить выборку». '
                               'Проверка покажет повторы, исключения и доступность соседей. '
                               'Укажите, почему пробы сопоставимы, и запустите расчёт поверхности.'))
    sheet.idw_intro.setWordWrap(True)
    sheet.idw_intro.setAlignment(Qt.AlignmentFlag.AlignCenter)
    viewer_layout.addWidget(sheet.idw_intro, 1)
    sheet.idw_summary = QLabel()
    sheet.idw_summary.setWordWrap(True)
    sheet.idw_summary.setTextInteractionFlags(
        Qt.TextInteractionFlag.TextSelectableByKeyboard | Qt.TextInteractionFlag.TextSelectableByMouse)
    sheet.idw_summary.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    sheet.idw_summary.setAccessibleDescription(tr('Текстовая сводка IDW и проверки LOO'))
    viewer_layout.addWidget(sheet.idw_summary)
    sheet.toolbar = NavigationToolbar(sheet.canvas, sheet)
    viewer_layout.addWidget(sheet.toolbar)
    sheet.canvas_view = CanvasView(sheet.canvas)
    sheet.canvas_view.add_to_toolbar(sheet.toolbar)
    viewer_layout.addWidget(sheet.canvas_view.scroll, 1)
    sheet.views = QTabWidget()
    sheet.views.setUsesScrollButtons(True)
    sheet.views.tabBar().setAccessibleName(tr('Вкладки пространственной интерполяции'))
    sheet.add_interpolation_tab(viewer, tr('Поверхности IDW'), 'map')


def _block_view(sheet):
    block_view = QWidget()
    block_layout = QVBoxLayout(block_view)
    block_actions = QHBoxLayout()
    preview = QPushButton(icon('map'), tr('Предпросмотр блоков'))
    preview.clicked.connect(lambda: sheet.calculate_blocks(evaluate=False))
    block_actions.addWidget(preview)
    blocks = QPushButton(icon('shield-check'), tr('Проверить по блокам'))
    blocks.clicked.connect(lambda: sheet.calculate_blocks(evaluate=True))
    block_actions.addWidget(blocks)
    block_actions.addStretch()
    block_layout.addLayout(block_actions)
    spread_controls = QHBoxLayout()
    sheet.spread_method = QComboBox()
    for method, label in [('IDW', 'IDW'), ('TIN', 'TIN'), ('kriging', tr('Кригинг'))]:
        sheet.spread_method.addItem(label, method)
    sheet.spread_method.setAccessibleName(tr('Метод для карты разброса'))
    spread_controls.addWidget(sheet.spread_method)
    sheet.spread_shift = QCheckBox(tr('Сдвиг на полблока по X и Y'))
    spread_controls.addWidget(sheet.spread_shift)
    spread = QPushButton(icon('map'), tr('Карта разброса…'))
    spread.clicked.connect(sheet.calculate_spread)
    spread_controls.addWidget(spread)
    spread_controls.addStretch()
    block_layout.addLayout(spread_controls)
    sheet.block_figure = Figure(figsize=(8, 5), dpi=96)
    sheet.block_canvas = FittedFigureCanvas(sheet.block_figure)
    block_layout.addWidget(NavigationToolbar(sheet.block_canvas, sheet))
    block_layout.addWidget(sheet.block_canvas, 2)
    sheet.block_note = QLabel(tr('Прогноз проверяется внутри выпуклой оболочки оставшихся мест. Доля рассчитана от всех мест блока.'))
    sheet.block_note.setWordWrap(True)
    block_layout.addWidget(sheet.block_note)
    sheet.block_table = QTableView()
    sheet.block_table.setAccessibleName(tr('Результаты блочной проверки IDW'))
    sheet.block_table.setTabKeyNavigation(False)
    sheet.block_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
    sheet.block_table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
    block_layout.addWidget(sheet.block_table, 1)
    sheet.block_patches = []
    sheet.block_buffer_outline = None
    sheet.add_interpolation_tab(block_view, tr('Блочная проверка IDW'), 'shield-check')


def _footer(sheet, layout):
    footer = QWidget()
    footer.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    sheet.interpolation_actions = QBoxLayout(QBoxLayout.Direction.LeftToRight, footer)
    sheet.interpolation_actions.setContentsMargins(0, 0, 0, 0)
    buttons = QHBoxLayout()
    final_buttons = QHBoxLayout()
    sheet.interpolation_actions.addLayout(buttons)
    sheet.interpolation_actions.addLayout(final_buttons)
    layout.addWidget(footer)
    sheet.export_button = QPushButton(icon('log-out'), tr('Экспорт результата…'))
    sheet.export_button.setEnabled(False)
    sheet.export_button.clicked.connect(sheet.export)
    buttons.addWidget(sheet.export_button)
    sheet.save_surface = QPushButton(icon('map'), tr('Добавить слои на карту'))
    sheet.save_surface.clicked.connect(sheet.add_surface_to_map)
    buttons.addWidget(sheet.save_surface)
    sheet.save_tiff = QPushButton(icon('log-out'), tr('Экспорт GeoTIFF…'))
    sheet.save_tiff.clicked.connect(sheet.export_surface_tiff)
    final_buttons.addWidget(sheet.save_tiff)
    sheet.views.currentChanged.connect(sheet.update_surface_actions)
    sheet.update_surface_actions()
    close = QPushButton(icon('x'), tr('Закрыть'))
    close.clicked.connect(sheet.reject)
    final_buttons.addWidget(close)
    sheet.status = QLabel()
    sheet.status.setWordWrap(True)
    sheet.status.setMaximumHeight(70)
    sheet.status.setAccessibleDescription(tr('Состояние расчёта интерполяции'))
    layout.addWidget(sheet.status)


def _finish(sheet, report, parent, context):
    add_parameter_tools(sheet, changed=sheet.invalidate, exclude=(sheet.canvas_view.mode_combo,))
    sheet.canvas_view.bind_project(getattr(parent, 'owner', parent), 'spatial_interpolation')
    if sheet.owner is not None:
        sheet.owner.analysis_scope.bind(sheet.scope, sheet.invalidate)
    else:
        sheet.scope.addItem(tr('Снимок карты'), report['spec'].get('scope', 'filtered'))
        sheet.scope.setEnabled(False)
    sheet.set_idw_preview_visible(False)
    sheet.status.setText(tr('Расчёт не запускался.'))
    sheet.input_summary.setText(tr('Выборка для проверки или расчёта ещё не подготовлена.'))
    QWidget.setTabOrder(sheet.scope, sheet.value_selector)
    QWidget.setTabOrder(sheet.value_selector, sheet.preflight_button)
    QWidget.setTabOrder(sheet.preflight_button, context)
    sheet.value_selector.setFocus()


def build_kriging(sheet):
    widget = QWidget()
    layout = QVBoxLayout(widget)
    _kriging_parameters(sheet, layout)
    _surface_controls(sheet, layout)
    _grid_inspector(sheet, layout)
    _comparison_results(sheet, layout)
    sheet.kriging_tab = sheet.add_interpolation_tab(widget, tr('Сравнение методов'), 'files')


def _kriging_parameters(sheet, layout):
    note = QLabel(tr('Параметры вариограммы задаются вручную и фиксируются для всех блоков. Эмпирическая вариограмма описывает текущую выборку.'))
    note.setWordWrap(True)
    layout.addWidget(note)
    sheet.kriging_form = Form()
    sheet.kriging_form.combo('model', tr('Модель вариограммы'),
                           [(tr('Сферическая'), 'spherical'), (tr('Экспоненциальная'), 'exponential')])
    sheet.kriging_form.number('range_km', tr('Дальность вариограммы, км'), 100, .001, 10000, decimals=3)
    sheet.kriging_form.number('sill', tr('Полный порог вариограммы'), 1, .000001, 1e12, decimals=6)
    sheet.kriging_form.number('nugget', tr('Наггет'), 0, 0, 1e12, decimals=6)
    sheet.kriging_form.check('directions', tr('Направленные вариограммы'))
    sheet.kriging_form.number('azimuth_deg', tr('Азимут большой дальности от севера сетки, °'), 0, 0, 180)
    ratio = sheet.kriging_form.number('range_ratio', tr('Малая дальность / большая дальность'), 1, .01, 1, decimals=3)
    ratio.setToolTip(tr('Анизотропия меняет ковариацию. Радиус поиска остаётся круговым.'))
    sheet.kriging_form.number('rbf_smoothing', tr('Сглаживание RBF'), 1, 0, 1e6, decimals=4)
    sheet.kriging_form.edited.connect(sheet.clear_kriging)
    sheet.kriging_form.edited.connect(sheet.clear_nested_methods)
    layout.addWidget(sheet.kriging_form)
    buttons = QHBoxLayout()
    for label, preview, image in [(tr('Построить вариограмму'), True, 'chart-no-axes-combined'), (tr('Сравнить методы интерполяции'), False, 'files')]:
        button = QPushButton(icon(image), label)
        if not preview:
            sheet.comparison_run = button
        button.clicked.connect(lambda _, preview=preview: sheet.calculate_kriging(preview=preview))
        buttons.addWidget(button)
    layout.addLayout(buttons)


def _surface_controls(sheet, layout):
    sheet.comparison_surface = QComboBox()
    sheet.comparison_surface.setAccessibleName(tr('Поверхность для просмотра'))
    for method, label in INTERPOLATION_METHODS.items():
        if method != 'isotropic-kriging':
            sheet.comparison_surface.addItem(tr(label), method)
    sheet.comparison_surface.addItem('IDW', 'IDW')
    sheet.comparison_surface.addItem(tr('IDW и TIN · карта разности'), 'idw-tin-difference')
    sheet.comparison_surface.currentIndexChanged.connect(sheet.redraw_comparison_surface)
    surface_controls = QHBoxLayout()
    surface_controls.addWidget(sheet.comparison_surface, 1)
    sheet.comparison_common_scale = QCheckBox(tr('Общая цветовая шкала'))
    sheet.comparison_common_scale.setChecked(True)
    sheet.comparison_common_scale.setEnabled(False)
    sheet.comparison_common_scale.setToolTip(tr('Одинаковые цвета обозначают одинаковые значения для всех методов. Снимите флажок, чтобы рассмотреть детали выбранной поверхности.'))
    sheet.comparison_common_scale.toggled.connect(sheet.redraw_comparison_surface)
    surface_controls.addWidget(sheet.comparison_common_scale)
    layout.addLayout(surface_controls)
    sheet.comparison_text_summary = QLabel()
    sheet.comparison_text_summary.setWordWrap(True)
    sheet.comparison_text_summary.setTextInteractionFlags(
        Qt.TextInteractionFlag.TextSelectableByKeyboard | Qt.TextInteractionFlag.TextSelectableByMouse)
    sheet.comparison_text_summary.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    sheet.comparison_text_summary.setAccessibleDescription(tr('Текстовая сводка поверхности'))
    sheet.comparison_text_summary.hide()
    layout.addWidget(sheet.comparison_text_summary)


def _grid_inspector(sheet, layout):
    sheet.grid_inspector = QWidget()
    inspector_layout = QVBoxLayout(sheet.grid_inspector)
    inspector_layout.setContentsMargins(0, 0, 0, 0)
    sheet.grid_indices = sheet.add_grid_index_fields(inspector_layout)
    for field in sheet.grid_indices:
        field.valueChanged.connect(sheet.inspect_grid_node)
    sheet.grid_node_text = QLabel()
    sheet.grid_node_text.setWordWrap(True)
    sheet.grid_node_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByKeyboard)
    sheet.grid_node_text.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    inspector_layout.addWidget(sheet.grid_node_text)
    sheet.grid_holdout_text = QLabel()
    sheet.grid_holdout_text.setWordWrap(True)
    sheet.grid_holdout_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByKeyboard)
    sheet.grid_holdout_text.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    sheet.grid_holdout_text.setAccessibleDescription(tr('Остатки на ближайших проверочных локациях'))
    inspector_layout.addWidget(sheet.grid_holdout_text)
    layout.addWidget(sheet.grid_inspector)
    sheet.grid_inspector.hide()
    sheet.grid_inspection = None


def _comparison_results(sheet, layout):
    tin_note = QLabel(tr('RBF использует тонкопластинчатый сплайн и координаты в км. Радиус поиска действует для IDW, кригинга и ближайшего соседа.'))
    tin_note.setWordWrap(True)
    layout.addWidget(tin_note)
    sheet.kriging_figure = Figure(layout='constrained')
    sheet.kriging_canvas = FittedFigureCanvas(sheet.kriging_figure)
    sheet.kriging_canvas.mpl_connect('button_press_event', sheet.pick_grid_node)
    layout.addWidget(sheet.kriging_canvas, 3)
    sheet.comparison_pair = QComboBox()
    sheet.comparison_pair.setAccessibleName(tr('Локации для сравнения ошибок'))
    sheet.comparison_pair.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    sheet.comparison_pair.setMinimumContentsLength(24)
    sheet.comparison_pair.currentIndexChanged.connect(sheet.refresh_comparison_metrics)
    sheet.comparison_pair.hide()
    layout.addWidget(sheet.comparison_pair)
    sheet.comparison_back = QPushButton(icon('undo-2'), tr('К настройкам сравнения'))
    sheet.comparison_back.clicked.connect(sheet.focus_comparison_setup)
    sheet.comparison_back.hide()
    layout.addWidget(sheet.comparison_back, alignment=Qt.AlignmentFlag.AlignLeft)
    sheet.kriging_table = QTableView()
    sheet.kriging_table.setAccessibleName(tr('Результаты сравнения методов'))
    sheet.kriging_table.setTabKeyNavigation(False)
    layout.addWidget(sheet.kriging_table, 1)
    sheet.kriging_result = None
    sheet.comparison_pair.hide()
    sheet.comparison_back.hide()


def resize_layout(sheet, event):
    if not hasattr(sheet, 'views') or not sheet.views.count():
        return
    splitter = sheet.interpolation_splitter
    margins = sheet.layout().contentsMargins()
    page_width = max(sheet.views.widget(i).widget().minimumSizeHint().width() for i in range(sheet.views.count()))
    threshold = (sheet.interpolation_settings_scroll.minimumWidth() + page_width + splitter.handleWidth()
                 + margins.left() + margins.right())
    compact = event.size().width() < threshold
    orientation = Qt.Orientation.Vertical if compact else Qt.Orientation.Horizontal
    if splitter.orientation() != orientation:
        splitter.setOrientation(orientation)
        splitter.setSizes([1, 2] if compact else [430, 800])
    sheet.layout_interpolation_actions()


def add_interpolation_tab(sheet, content, label, icon_name):
    page = QScrollArea()
    page.setWidgetResizable(True)
    page.setFrameShape(QFrame.Shape.NoFrame)
    page.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    page.setWidget(content)
    sheet.views.addTab(page, icon(icon_name), label)
    return page


def layout_interpolation_actions(sheet):
    if not hasattr(sheet, 'interpolation_actions'):
        return
    actions = sheet.interpolation_actions
    required = sum(actions.itemAt(i).minimumSize().width() for i in range(actions.count())) + actions.spacing()
    margins = sheet.layout().contentsMargins()
    available = sheet.width() - margins.left() - margins.right()
    actions.setDirection(QBoxLayout.Direction.TopToBottom if required > available else QBoxLayout.Direction.LeftToRight)


def focus_comparison_setup(sheet):
    sheet.comparison_run.setFocus()
    sheet.kriging_tab.ensureWidgetVisible(sheet.comparison_run, 0, 0)
