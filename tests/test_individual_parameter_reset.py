"""Individual resets preserve other settings and drive the existing preview."""

import pytest
from PySide6.QtCore import QEvent, QPoint, QSettings
from PySide6.QtGui import QHelpEvent
from PySide6.QtWidgets import QApplication, QCheckBox, QDialog, QLineEdit, QTableWidget, QTableWidgetItem, QToolTip, QVBoxLayout

from autopyro.io import build_project
from autopyro.ui import i18n
from autopyro.ui.window import MainWindow
from autopyro.ui.widgets.forms import axis_form
from autopyro.ui.widgets.parameters import add_parameter_tools, set_parameter_help


def test_reset_one_form_parameter_uses_declared_default_and_emits_once(qtbot):
    form = axis_form()
    qtbot.addWidget(form)
    form.add_reset_button()
    form.load({'label_size': 25, 'color': 'red', 'scale': 'log', 'log_base': 2})
    notifications = []
    form.edited.connect(lambda: notifications.append(True))
    form.fields['label_size'][0].parameter_reset_button.click()
    assert form.values()['label_size'] == 11
    assert form.values()['color'] == 'red'
    assert form.values()['scale'] == 'log'
    assert form.values()['log_base'] == 2
    assert notifications == [True]
    form.fields['scale'][0].parameter_reset_button.click()
    assert form.values()['scale'] == 'linear'
    assert not form.fields['log_base'][0].isEnabled()
    assert form.values()['log_base'] == 2
    assert len(notifications) == 2


def test_plain_text_reset_notifies_edit_handlers_without_resetting_neighbours(qtbot):
    dialog = QDialog()
    qtbot.addWidget(dialog)
    layout = QVBoxLayout(dialog)
    first, second = QLineEdit('first default'), QLineEdit('second default')
    layout.addWidget(first)
    layout.addWidget(second)
    add_parameter_tools(dialog)
    edits = []
    first.textEdited.connect(edits.append)
    first.setText('changed')
    second.setText('keep')
    first.parameter_reset_button.click()
    assert first.text() == 'first default'
    assert second.text() == 'keep'
    assert edits == ['first default']


def test_hover_shows_detailed_help_without_whats_this_mode(qtbot):
    form = axis_form()
    qtbot.addWidget(form)
    form.add_reset_button()
    form.show()
    field = form.fields['log_base'][0]
    form.load({'scale': 'log'})
    event = QHelpEvent(QEvent.Type.ToolTip, QPoint(4, 4), field.mapToGlobal(QPoint(4, 4)))
    QApplication.sendEvent(field, event)
    assert QToolTip.isVisible()
    assert QToolTip.text() == field.whatsThis()
    assert 'положительное' in QToolTip.text()
    QToolTip.hideText()
    password = QLineEdit()
    qtbot.addWidget(password)
    password.setEchoMode(QLineEdit.EchoMode.Password)
    set_parameter_help(password, 'password', 'Password', 'must-stay-private')
    assert 'must-stay-private' not in password.toolTip()


def test_parameter_table_resets_only_the_requested_cell(qtbot):
    dialog = QDialog()
    qtbot.addWidget(dialog)
    layout = QVBoxLayout(dialog)
    table = QTableWidget(1, 3)
    name = QTableWidgetItem('default name')
    table.setItem(0, 0, name)
    table.setItem(0, 1, QTableWidgetItem('keep neighbour'))
    checkbox = QCheckBox()
    checkbox.setChecked(True)
    table.setCellWidget(0, 2, checkbox)
    layout.addWidget(table)
    add_parameter_tools(dialog, tables=(table,))
    name.setText('changed name')
    table.item(0, 1).setText('changed neighbour')
    checkbox.setChecked(False)
    table.reset_parameter_cell(0, 0)
    assert name.text() == 'default name'
    assert table.item(0, 1).text() == 'changed neighbour'
    assert not checkbox.isChecked()
    table.reset_parameter_cell(0, 2)
    assert checkbox.isChecked()
    assert table.cellWidget(0, 2) is checkbox


def test_application_preference_resets_to_factory_value(window):
    from autopyro.ui.dialogs.settings import SettingsDialog
    from autopyro.ui.settings import load_preferences
    store = i18n.settings()
    store.setValue('general/autosave_minutes', 45)
    dialog = SettingsDialog(window)
    assert dialog.autosave_minutes.value() == 45
    dialog.autosave_minutes.parameter_reset_button.click()
    assert dialog.autosave_minutes.value() == load_preferences(use_saved=False).autosave_minutes
    assert store.value('general/autosave_minutes') == 45  # Draft stays uncommitted.
    dialog.reject()


@pytest.mark.parametrize('method,attribute', [('show_plots', 'plot_window'), ('show_maps', 'map_window')])
def test_category_style_reset_keeps_custom_legend_label(window, method, attribute):
    getattr(window, method)()
    dialog = getattr(window, attribute)
    dialog.timer.stop()
    x, y = [column.id for column in window.model.project.columns]
    if attribute == 'map_window':
        dialog.data.load({'x_id': x, 'y_id': y, 'source_crs': 'EPSG:4326', 'target_crs': 'EPSG:4326'})
    dialog.style.load({'color_by': x})
    dialog.redraw()
    assert dialog.result is not None, dialog.status.text()
    table = dialog.category_table
    original_color = table.item(0, 1).text()
    table.item(0, 1).setText('red')
    table.item(0, 2).setText('custom label')
    table.reset_parameter_cell(0, 1)
    assert table.item(0, 1).text() == original_color
    assert table.item(0, 2).text() == 'custom label'
    dialog.timer.stop()


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    store = QSettings(str(tmp_path / 'individual-reset.ini'), QSettings.Format.IniFormat)
    monkeypatch.setattr(i18n, 'settings', lambda: store)
    i18n.initialize()
    window = MainWindow()
    qtbot.addWidget(window)
    window.may_replace = lambda: True
    window.set_project(build_project(['X', 'Y'], [['1', '2'], ['2', '3']], [2, 3],
                                    [(0, 'X', 'number', ''), (1, 'Y', 'number', '')], {}))
    yield window
    window.dirty = False


@pytest.mark.parametrize('method,attribute,form_name,key,default', [
    ('show_plots', 'plot_window', 'style', 'marker_size', 7),
    ('show_maps', 'map_window', 'style', 'marker_size', 7),
    ('show_stratigraphy', 'stratigraphy_window', 'column_style_form', 'dpi', 100),
])
def test_visualization_field_resets_without_touching_data(window, method, attribute, form_name, key, default):
    getattr(window, method)()
    dialog = getattr(window, attribute)
    if hasattr(dialog, 'timer'):
        dialog.timer.stop()
    form = getattr(dialog, form_name)
    before = [dict(row.values) for row in window.model.project.rows]
    form.load({key: default + 5, 'alpha': .25})
    form.fields[key][0].parameter_reset_button.click()
    assert form.values()[key] == default
    assert form.values()['alpha'] == .25
    assert [dict(row.values) for row in window.model.project.rows] == before
    if hasattr(dialog, 'timer'):
        assert dialog.timer.isActive()
        dialog.timer.stop()
