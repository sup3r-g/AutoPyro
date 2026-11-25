"""Visualization fonts are selected from runtime catalogs, not free text."""

from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QComboBox

from autopyro.io import build_project
from autopyro.ui import i18n
from autopyro.ui.widgets import fonts
from autopyro.ui.widgets.forms import Form
from autopyro.ui.window import MainWindow


def test_font_catalog_combines_system_and_renderer_fonts(qtbot, monkeypatch):
    monkeypatch.setattr(fonts.QFontDatabase, 'families', lambda: ['System Font', 'Shared Font'])
    monkeypatch.setattr(fonts.font_manager.fontManager, 'ttflist',
                        [SimpleNamespace(name='Bundled Font'), SimpleNamespace(name='Shared Font')])
    form = Form()
    qtbot.addWidget(form)
    selector = form.font_selector('font_family', 'Font', 'Bundled Font')
    assert not selector.isEditable()
    assert [selector.itemData(i) for i in range(selector.count())] == ['Bundled Font', 'Shared Font', 'System Font']
    notifications = []
    form.edited.connect(lambda: notifications.append(True))
    selector.setCurrentIndex(selector.findData('System Font'))
    assert notifications == [True]
    assert form.values()['font_family'] == 'System Font'
    form.load({'font_family': 'Shared Font'})
    assert form.values()['font_family'] == 'Shared Font'
    assert notifications == [True]  # Loading a saved style does not trigger redraw.
    form.load({'font_family': 'Font from another computer'})
    assert form.values()['font_family'] == 'Font from another computer'
    assert not selector.isEditable()
    form.reset_defaults()
    assert form.values()['font_family'] == 'Bundled Font'


@pytest.mark.parametrize('method,attribute,form_name', [
    ('show_plots', 'plot_window', 'layout_form'),
    ('show_maps', 'map_window', 'layout_form'),
    ('show_stratigraphy', 'stratigraphy_window', 'column_style_form'),
])
def test_all_visualization_modules_use_complete_font_selectors(qtbot, tmp_path, monkeypatch,
                                                              method, attribute, form_name):
    store = QSettings(str(tmp_path / 'fonts.ini'), QSettings.Format.IniFormat)
    monkeypatch.setattr(i18n, 'settings', lambda: store)
    i18n.initialize()
    window = MainWindow()
    qtbot.addWidget(window)
    window.may_replace = lambda: True
    project = build_project(['X', 'Y'], [['1', '2'], ['2', '3']], [2, 3],
                            [(0, 'X', 'number', ''), (1, 'Y', 'number', '')], {})
    window.set_project(project)
    getattr(window, method)()
    dialog = getattr(window, attribute)
    if hasattr(dialog, 'timer'):
        dialog.timer.stop()
    form = getattr(dialog, form_name)
    selector, kind = form.fields['font_family']
    assert isinstance(selector, QComboBox)
    assert not selector.isEditable()
    assert kind == 'combo'
    assert {selector.itemData(i) for i in range(selector.count())} == set(fonts.font_families())
    assert form.values()['font_family'] == 'DejaVu Sans'
    selected = next(name for name in fonts.font_families() if name != 'DejaVu Sans')
    selector.setCurrentIndex(selector.findData(selected))
    if attribute == 'stratigraphy_window':
        assert dialog.style_values()['font_family'] == selected
    elif attribute == 'plot_window':
        assert dialog.read_controls().font_family == selected
        dialog.load_controls()
        assert selector.currentData() == selected
    else:
        dialog.redraw()
        assert dialog.spec.font_family == selected
        form.load(dialog.spec.to_dict())
        assert selector.currentData() == selected
    form.reset_defaults()
    assert form.values()['font_family'] == 'DejaVu Sans'
    window.dirty = False
