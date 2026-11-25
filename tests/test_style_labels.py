"""Translated style names must never be passed to the renderer as values."""

import re

import pytest
import matplotlib as mpl
from matplotlib.colors import ListedColormap, get_named_colors_mapping
from matplotlib.legend import Legend
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QStyleOptionViewItem, QTableWidget, QTableWidgetItem

from autopyro.io import build_project
from autopyro.plot_catalog import colormap_names, markers, load_extensions
from autopyro.plotting import PlotSpec, render_plot
from autopyro.ui import i18n
from autopyro.ui.widgets.forms import ColorCombo, Form
from autopyro.ui.widgets.parameters import add_parameter_tools
from autopyro.ui.widgets.style_labels import color_label, colormap_label, legend_label, marker_label
from autopyro.ui.widgets.styles import StyleDelegate


@pytest.fixture(params=['ru', 'en'])
def locale(request, qapp, tmp_path, monkeypatch):
    store = QSettings(str(tmp_path / 'style-language.ini'), QSettings.Format.IniFormat)
    store.setValue(i18n.SETTINGS_KEY, request.param)
    monkeypatch.setattr(i18n, 'settings', lambda: store)
    i18n.initialize()
    yield request.param
    store.setValue(i18n.SETTINGS_KEY, 'ru')
    i18n.initialize()


def test_localized_selectors_preserve_values_and_render(locale, qtbot):
    form = Form()
    qtbot.addWidget(form)
    color = form.text('color', i18n.tr('Цвет'), 'red', color=True)
    marker = form.combo('marker', i18n.tr('Маркер'), [])
    cmap = form.combo('cmap', i18n.tr('Цветовая шкала'), [])
    legend = form.combo('legend_position', i18n.tr('Положение легенды'), [])
    form.load({'color': 'tab:blue', 'marker': '^', 'cmap': 'RdBu_r', 'legend_position': 'upper right'})
    assert marker.currentText() == ('Треугольник вверх' if locale == 'ru' else 'Triangle up')
    assert legend.currentText() == ('Вверху справа' if locale == 'ru' else 'Upper right')
    assert 'Красный — синий' in cmap.currentText() if locale == 'ru' else 'Red — blue' in cmap.currentText()
    assert 'обратная' in cmap.currentText() if locale == 'ru' else 'reversed' in cmap.currentText()
    assert color.text() == 'tab:blue'
    assert color.currentText() == color_label('tab:blue')
    assert {marker.itemData(i) for i in range(marker.count())} == set(markers())
    assert all(not marker.itemIcon(i).isNull() for i in range(marker.count()))
    up = marker.itemIcon(marker.findData('^')).pixmap(32, 20).toImage()
    down = marker.itemIcon(marker.findData('v')).pixmap(32, 20).toImage()
    assert up != down
    assert {cmap.itemData(i) for i in range(cmap.count())} >= set(colormap_names())
    assert {legend.itemData(i) for i in range(legend.count())} == set(Legend.codes)
    if locale == 'en':
        assert not re.search('[А-Яа-яЁё]', ' '.join(marker.itemText(i) for i in range(marker.count())))
        assert not re.search('[А-Яа-яЁё]', ' '.join(cmap.itemText(i) for i in range(cmap.count())))
    project = build_project(['X', 'Y'], [['1', '2'], ['2', '3']], [2, 3],
                            [(0, 'X', 'number', ''), (1, 'Y', 'number', '')], {})
    values = form.values()
    assert values == {'color': 'tab:blue', 'marker': '^', 'cmap': 'RdBu_r', 'legend_position': 'upper right'}
    spec = PlotSpec(x_id=project.columns[0].id, y_id=project.columns[1].id, **values)
    spec = PlotSpec.from_dict(spec.to_dict())
    figure, result = render_plot(project, project.row_map(), spec)
    assert result['plotted'][spec.x_id] == 2
    figure.clear()


def test_color_aliases_and_custom_hex_survive_load_and_reset(locale, qtbot):
    form = Form()
    qtbot.addWidget(form)
    color = form.text('color', i18n.tr('Цвет'), 'r', color=True)
    add_parameter_tools(form)
    color.setText('#aabbcc')
    assert color.text() == '#aabbcc'
    assert color.currentText() == color_label('#aabbcc')
    color.parameter_reset_button.click()
    assert color.text() == 'r'
    color.setEditText('xkcd:sky blue')
    assert color.text() == 'xkcd:sky blue'


def test_line_style_names_previews_and_saved_values(locale, qtbot):
    form = Form()
    qtbot.addWidget(form)
    values = ['--', '-', ':', '-.']
    combo = form.combo('grid_style', 'Line', values)
    expected = (['Штриховая', 'Сплошная', 'Пунктирная', 'Штрихпунктирная'] if locale == 'ru'
                else ['Dashed', 'Solid', 'Dotted', 'Dash-dot'])
    assert [combo.itemText(i) for i in range(combo.count())] == expected
    previews = []
    for index, value in enumerate(values):
        assert combo.itemData(index) == value
        assert combo.itemData(index, Qt.ItemDataRole.ToolTipRole) == value
        assert not combo.itemIcon(index).isNull()
        previews.append(combo.itemIcon(index).pixmap(64, 20).toImage())
        form.load({'grid_style': value})
        assert form.values()['grid_style'] == value
    assert all(a != b for i, a in enumerate(previews) for b in previews[i + 1:])
    form.load({'grid_style': 'dashed'})
    assert combo.currentText() == expected[0]
    assert not combo.itemIcon(combo.currentIndex()).isNull()
    assert form.values()['grid_style'] == 'dashed'
    form.reset_parameter('grid_style')
    assert form.values()['grid_style'] == '--'


def test_canvas_view_modes_are_localized(locale, qtbot):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
    from autopyro.ui.widgets.figure_canvas import CanvasView, FittedFigureCanvas

    view = CanvasView(FittedFigureCanvas(Figure()))
    qtbot.addWidget(view.scroll)
    toolbar = NavigationToolbar2QT(view.canvas, view.scroll)
    view.add_to_toolbar(toolbar)
    expected = ['Подогнать под окно', 'Реальный размер'] if locale == 'ru' else ['Fit to window', 'Actual size']
    assert [view.mode_combo.itemText(i) for i in range(2)] == expected
    assert view.fit and view.mode_combo.currentData() is True


def test_view_management_commands_are_localized(locale, qtbot):
    from PySide6.QtWidgets import QWidget
    from autopyro.ui.analysis_scope import AnalysisScope

    parent = QWidget()
    qtbot.addWidget(parent)
    scope = AnalysisScope(parent)
    menu = scope.create_menu(parent)
    expected = (['Создать представление…', 'Редактировать представление', 'Переименовать представление…', 'Удалить представление…']
                if locale == 'ru' else ['Create view…', 'Edit view', 'Rename view…', 'Delete view…'])
    assert [action.text() for action in menu.actions()] == expected


def test_explorer_commands_are_localized(locale, qtbot):
    from autopyro.ui.window import MainWindow

    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.may_replace = lambda: True
    menu = owner.desktop.create_project_menu(('project', ''))
    qtbot.addWidget(menu)
    texts = [action.text() for action in menu.actions()]
    assert ('Открыть модуль' if locale == 'ru' else 'Open module') in texts
    assert ('Открыть таблицу' if locale == 'ru' else 'Open table') in texts
    assert ('Развернуть всё' if locale == 'ru' else 'Expand all') in texts
    owner.dirty = False


def test_common_navigation_labels_are_localized(locale, qtbot):
    from matplotlib.figure import Figure
    from autopyro.ui.widgets.figure_canvas import FittedFigureCanvas
    from autopyro.ui.widgets.navigation import NavigationToolbar

    canvas = FittedFigureCanvas(Figure())
    qtbot.addWidget(canvas)
    toolbar = NavigationToolbar(canvas, canvas)
    expected = (['Исходный вид', 'Предыдущий вид', 'Следующий вид', 'Перемещение', 'Масштабирование']
                if locale == 'ru' else ['Original view', 'Previous view', 'Next view', 'Pan', 'Zoom'])
    assert [action.text() for action in toolbar._actions.values()] == expected
    assert all(action.toolTip() for action in toolbar._actions.values())


def test_style_delegate_displays_names_and_commits_raw_values(locale, qtbot):
    table = QTableWidget(1, 1)
    qtbot.addWidget(table)
    table.setItem(0, 0, QTableWidgetItem('red'))
    delegate = StyleDelegate(table)
    index = table.model().index(0, 0)
    option = QStyleOptionViewItem()
    delegate.initStyleOption(option, index)
    assert option.text == color_label('red')
    editor = delegate.createEditor(table.viewport(), option, index)
    assert isinstance(editor, ColorCombo)
    delegate.setEditorData(editor, index)
    assert editor.text() == 'red'
    editor.setText('navy')
    delegate.setModelData(editor, table.model(), index)
    assert table.item(0, 0).text() == 'navy'

    table.item(0, 0).setText('^')
    delegate = StyleDelegate(table, lambda: [('shape',)])
    delegate.initStyleOption(option, index)
    assert option.text == marker_label('^')
    assert not option.icon.isNull()
    editor = delegate.createEditor(table.viewport(), option, index)
    delegate.setEditorData(editor, index)
    assert not editor.itemIcon(editor.currentIndex()).isNull()
    editor.setCurrentIndex(editor.findData('v'))
    delegate.setModelData(editor, table.model(), index)
    assert table.item(0, 0).text() == 'v'


def test_runtime_colors_and_colormaps_receive_localized_labels(locale, qtbot, monkeypatch):
    monkeypatch.setitem(get_named_colors_mapping(), 'registered-test-color', '#123456')
    name = 'registered_test_scale'
    load_extensions()
    mpl.colormaps.register(ListedColormap(['red', 'blue'], name=name))
    try:
        color = ColorCombo('registered-test-color')
        qtbot.addWidget(color)
        assert color.currentData() == 'registered-test-color'
        assert color.currentText() != 'registered-test-color'
        form = Form()
        qtbot.addWidget(form)
        cmap = form.combo('cmap', i18n.tr('Цветовая шкала'), [])
        assert cmap.itemText(cmap.findData(name)) == colormap_label(name)
        assert marker_label('$A$') != '$A$'
        assert legend_label('right') == ('Справа' if locale == 'ru' else 'Right')
    finally:
        mpl.colormaps.unregister(name)
