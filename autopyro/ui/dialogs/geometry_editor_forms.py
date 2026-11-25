"""Build the geometry editor controls, canvas and keyboard actions."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSplitter,
    QTableWidget,
    QVBoxLayout,
    QWidget,
    QToolButton,
)
from matplotlib.figure import Figure

from ..i18n import tr
from ..theme import icon
from ..widgets.figure_canvas import CanvasView, FittedFigureCanvas
from ..widgets.forms import ColorCombo
from ..widgets.navigation import NavigationToolbar as GeometryToolbar
from ..widgets.parameters import add_parameter_tools


def build(sheet, parent):
    layout = QVBoxLayout(sheet)
    _metadata_controls(sheet, layout)
    splitter = QSplitter()
    _object_panel(sheet, splitter)
    _viewer_panel(sheet, splitter)
    splitter.setSizes([380, 940])
    layout.addWidget(splitter, 1)
    _commands(sheet, layout)
    _connect_and_restore(sheet, parent)


def _metadata_controls(sheet, layout):
    metadata = QHBoxLayout()
    sheet.name, sheet.revision, sheet.source = QLineEdit(), QLineEdit(), QLineEdit()
    sheet.revision.setMaximumWidth(100)
    for label, widget in [
        (tr('Название'), sheet.name),
        (tr('Редакция'), sheet.revision),
        (tr('Источник'), sheet.source),
    ]:
        metadata.addWidget(QLabel(label))
        metadata.addWidget(widget)
        widget.editingFinished.connect(sheet.metadata_changed)
    sheet.author_status = QComboBox()
    for label, value in [
        (tr('Пользовательская'), "user_supplied"),
        (tr('Демонстрация'), "demonstration"),
        (tr('Проверена автором'), "validated"),
    ]:
        sheet.author_status.addItem(label, value)
    sheet.author_status.currentIndexChanged.connect(sheet.metadata_changed)
    metadata.addWidget(sheet.author_status)
    layout.addLayout(metadata)
    info = QLabel(
        tr('Все вершины и оси редактора заданы в analysis. Измеренные точки преобразуются в это пространство. Изменения применятся к проекту только кнопкой «Применить редакцию».')
    )
    info.setWordWrap(True)
    layout.addWidget(info)
    coordinates = []
    for key in ("x", "y"):
        axis = sheet.draft[key]
        value = (
            f"log10(value / {axis.get('reference', 1):g})"
            if axis.get("transform") == "log10"
            else "value"
        )
        coordinates.append(
            f"{key.upper()} ({axis['property']}, {axis['unit']}): ({value} − {axis.get('origin', 0):g}) / {axis.get('scale', 1):g}"
        )
    coordinate_info = QLabel("; ".join(coordinates))
    coordinate_info.setWordWrap(True)
    layout.addWidget(coordinate_info)


def _object_panel(sheet, splitter):
    left = QWidget()
    controls = QVBoxLayout(left)
    _undo_controls(sheet, controls)
    _object_controls(sheet, controls)
    _vertex_controls(sheet, controls)
    splitter.addWidget(left)


def _undo_controls(sheet, controls):
    history = QHBoxLayout()
    for action, shortcut, image in [
        (sheet.undo_stack.createUndoAction(sheet, tr('Отменить')), 'Ctrl+Z', 'undo-2'),
        (sheet.undo_stack.createRedoAction(sheet, tr('Повторить')), 'Ctrl+Shift+Z', 'redo-2'),
    ]:
        action.setShortcut(shortcut)
        action.setIcon(icon(image))
        sheet.addAction(action)
        button = QToolButton()
        button.setDefaultAction(action)
        history.addWidget(button)
    history.addStretch()
    controls.addLayout(history)


def _object_controls(sheet, controls):
    sheet.objects = QListWidget()
    sheet.objects.currentRowChanged.connect(sheet.select_object)
    controls.addWidget(QLabel(tr('Области и линии')))
    controls.addWidget(sheet.objects, 1)
    properties = QFormLayout()
    sheet.object_label = QLineEdit()
    sheet.object_color = ColorCombo('')
    sheet.object_color.parameter_default_state = ('combo', None, '#0072B2')
    properties.addRow(tr('Подпись'), sheet.object_label)
    color_row = QHBoxLayout()
    color_row.addWidget(sheet.object_color)
    choose_color = QPushButton(icon('sliders-horizontal'), tr('Цвет…'))
    choose_color.clicked.connect(sheet.choose_color)
    color_row.addWidget(choose_color)
    properties.addRow(tr('Цвет'), color_row)
    sheet.object_label.editingFinished.connect(sheet.object_properties_changed)
    sheet.object_color.editingFinished.connect(sheet.object_properties_changed)
    sheet.threshold_axis = QComboBox()
    sheet.threshold_axis.addItem(tr('Опорная линия'), None)
    sheet.threshold_axis.addItem(tr('Порог Y от X'), 'y')
    sheet.threshold_axis.addItem(tr('Порог X от Y'), 'x')
    sheet.threshold_less, sheet.threshold_greater = QLineEdit(), QLineEdit()
    sheet.apply_threshold = QPushButton(icon('check'), tr('Применить правило порога'))
    sheet.apply_threshold.clicked.connect(sheet.threshold_changed)
    properties.addRow(tr('Правило линии'), sheet.threshold_axis)
    properties.addRow(tr('Подпись для меньших значений'), sheet.threshold_less)
    properties.addRow(tr('Подпись для больших значений'), sheet.threshold_greater)
    properties.addRow(sheet.apply_threshold)
    sheet.region_passport = QPushButton(icon('file-text'), tr('Паспорт области…'))
    sheet.region_passport.clicked.connect(sheet.edit_region_passport)
    properties.addRow(sheet.region_passport)
    controls.addLayout(properties)


def _vertex_controls(sheet, controls):
    sheet.paths = QComboBox()
    sheet.paths.currentIndexChanged.connect(sheet.select_path)
    controls.addWidget(sheet.paths)
    sheet.vertex_table = QTableWidget(0, 2)
    sheet.vertex_table.setHorizontalHeaderLabels(["X (analysis)", "Y (analysis)"])
    sheet.vertex_table.itemChanged.connect(sheet.table_changed)
    sheet.vertex_table.itemSelectionChanged.connect(sheet.vertex_selection_changed)
    controls.addWidget(sheet.vertex_table, 2)
    edit_row = QHBoxLayout()
    for label, callback, image in [
        (tr('Вставить вершину'), sheet.insert_vertex, 'plus'),
        (tr('Удалить вершину'), sheet.delete_vertex, 'trash'),
    ]:
        button = QPushButton(icon(image), label)
        button.clicked.connect(callback)
        edit_row.addWidget(button)
    controls.addLayout(edit_row)
    remove_row = QHBoxLayout()
    for label, callback, image in [
        (tr('Удалить контур / часть'), sheet.delete_path, 'trash'),
        (tr('Удалить объект'), sheet.delete_object, 'trash'),
    ]:
        button = QPushButton(icon(image), label)
        button.clicked.connect(callback)
        remove_row.addWidget(button)
    controls.addLayout(remove_row)


def _viewer_panel(sheet, splitter):
    viewer = QWidget()
    view = QVBoxLayout(viewer)
    _canvas(sheet, view)
    _drawing_controls(sheet, view)
    _axis_limits(sheet, view)
    _overlays(sheet, view)
    splitter.addWidget(viewer)


def _canvas(sheet, view):
    sheet.figure = Figure(figsize=(8, 6), dpi=96)
    sheet.canvas = FittedFigureCanvas(sheet.figure)
    sheet.ax = sheet.figure.subplots()
    sheet.figure.subplots_adjust(left=0.13, right=0.97, bottom=0.13, top=0.94)
    sheet.toolbar = GeometryToolbar(sheet.canvas, sheet)
    sheet.canvas_view = CanvasView(sheet.canvas)
    sheet.canvas_view.add_to_toolbar(sheet.toolbar)
    view.addWidget(sheet.toolbar)


def _drawing_controls(sheet, view):
    draw_row = QHBoxLayout()
    sheet.mode = QComboBox()
    for label, value, image in [
        (tr('Выбор / перенос вершины'), "move", 'mouse-pointer-2'),
        (tr('Новая область'), "polygon", 'map'),
        (tr('Новая линия'), "line", 'spline'),
        (tr('Новое отверстие'), "hole", 'layers'),
        (tr('Новая часть объекта'), "part", 'plus'),
    ]:
        sheet.mode.addItem(icon(image), label, value)
    sheet.mode.currentIndexChanged.connect(sheet.mode_changed)
    draw_row.addWidget(sheet.mode)
    finish = QPushButton(icon('check'), tr('Завершить контур (Enter)'))
    finish.clicked.connect(sheet.finish_drawing)
    cancel = QPushButton(icon('x'), tr('Сбросить контур (Esc)'))
    cancel.clicked.connect(sheet.cancel_drawing)
    draw_row.addWidget(finish)
    draw_row.addWidget(cancel)
    view.addLayout(draw_row)


def _axis_limits(sheet, view):
    limits = QHBoxLayout()
    sheet.limits = [QLineEdit() for _ in range(4)]
    for label, widget in zip(
        ("X min", "X max", "Y min", "Y max"), sheet.limits, strict=True
    ):
        widget.setMaximumWidth(85)
        limits.addWidget(QLabel(label))
        limits.addWidget(widget)
    apply_limits = QPushButton(icon('maximize'), tr('Пределы'))
    apply_limits.clicked.connect(sheet.apply_limits)
    limits.addWidget(apply_limits)
    fit = QPushButton(icon('maximize'), tr('Вписать всё'))
    fit.clicked.connect(sheet.fit_view)
    limits.addWidget(fit)
    view.addLayout(limits)


def _overlays(sheet, view):
    sheet.show_samples = QCheckBox(tr('Предпросмотр классификации проб (до 2000)'))
    sheet.data_scope = QComboBox()
    sheet.data_scope.setAccessibleName(tr('Область'))
    view.addWidget(sheet.data_scope)
    sheet.show_samples.setChecked(bool(sheet.samples))
    sheet.show_samples.toggled.connect(sheet.redraw)
    view.addWidget(sheet.show_samples)
    sheet.show_coverage = QCheckBox(tr('Домен и разрывы покрытия'))
    sheet.show_coverage.setToolTip(tr('Синий контур — домен, оранжевый — разрывы, красный — области вне домена.'))
    sheet.show_coverage.toggled.connect(sheet.redraw)
    view.addWidget(sheet.show_coverage)
    domain_button = QPushButton(icon('map'), tr('Границы домена…'))
    domain_button.clicked.connect(sheet.edit_coverage_domain)
    view.addWidget(domain_button)
    view.addWidget(sheet.canvas_view.scroll, 1)
    sheet.help = QLabel(
        tr('Перенос: потяните вершину. Рисование: ЛКМ — вершина, ПКМ или Enter — завершить. Для панорамирования и масштаба используйте панель Matplotlib.')
    )
    sheet.help.setWordWrap(True)
    view.addWidget(sheet.help)


def _commands(sheet, layout):
    sheet.diagnostics = QLabel()
    sheet.diagnostics.setWordWrap(True)
    layout.addWidget(sheet.diagnostics)
    sheet.buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    sheet.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
        tr('Применить редакцию')
    )
    sheet.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr('Отмена'))
    sheet.buttons.button(QDialogButtonBox.StandardButton.Ok).setAutoDefault(False)
    sheet.buttons.button(QDialogButtonBox.StandardButton.Cancel).setAutoDefault(
        False
    )
    sheet.buttons.accepted.connect(sheet.apply)
    sheet.buttons.rejected.connect(sheet.reject)
    layout.addWidget(sheet.buttons)


def _connect_and_restore(sheet, parent):
    sheet.canvas.mpl_connect("button_press_event", sheet.on_press)
    sheet.canvas.mpl_connect("motion_notify_event", sheet.on_motion)
    sheet.canvas.mpl_connect("button_release_event", sheet.on_release)
    sheet.canvas.mpl_connect("key_press_event", sheet.on_key)
    sheet.set_draft(sheet.draft)
    sheet.fit_view()
    add_parameter_tools(sheet, reset=sheet.restore_parameters, reset_text=tr('Сбросить вид'),
                        exclude=(sheet.name, sheet.revision, sheet.source, sheet.author_status,
                                 sheet.objects, sheet.paths, sheet.object_label, sheet.canvas_view.mode_combo))
    sheet.canvas_view.bind_project(getattr(parent, 'owner', parent), 'geometry_editor')
    sheet.project_owner = getattr(parent, 'owner', parent)
    if sheet.project_owner is not None and hasattr(sheet.project_owner, 'analysis_scope'):
        sheet.project_owner.analysis_scope.bind(sheet.data_scope, sheet.scope_samples_changed)
        sheet.scope_samples_changed()
    else:
        sheet.data_scope.hide()
