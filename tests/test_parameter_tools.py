"""Reset parameter drafts without clearing source data or scientific schemes."""
from copy import deepcopy

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QDialog, QLineEdit, QTableWidget, QTableWidgetItem, QVBoxLayout, QWhatsThis

from autopyro.io import build_project
from autopyro.plotting import AxisSpec, PlotSpec
from autopyro.mapping import MapSpec
from autopyro.ui import i18n
from autopyro.ui.window import MainWindow
from autopyro.ui.widgets.forms import Form, axis_form
from autopyro.ui.widgets.parameters import add_parameter_tools, set_parameter_help


def test_parameter_help_inventories_widget_tree_once_per_annotation(qtbot, monkeypatch):
    from PySide6.QtWidgets import QFormLayout, QLabel, QWidget
    from autopyro.ui.widgets.parameters import annotate_parameters

    dialog = QDialog()
    qtbot.addWidget(dialog)
    form = QFormLayout(dialog)
    fields = [QLineEdit() for _ in range(30)]
    for index, field in enumerate(fields):
        form.addRow(f"Field {index}", field)
    queries = []
    find_children = dialog.findChildren

    def find(kind, *args, **kwargs):
        queries.append(kind)
        return find_children(kind, *args, **kwargs)

    monkeypatch.setattr(dialog, "findChildren", find)
    annotate_parameters(dialog)
    assert queries.count(QFormLayout) == queries.count(QLabel) == queries.count(QWidget) == 1
    assert all(field.whatsThis() and field.toolTip() for field in fields)
    assert fields[0].accessibleName() == "Field 0"
    added = QLineEdit()
    form.addRow("Added later", added)
    annotate_parameters(dialog)
    assert queries.count(QFormLayout) == queries.count(QLabel) == queries.count(QWidget) == 2
    assert added.accessibleName() == "Added later"
    assert added.whatsThis() and form.labelForField(added).toolTip()


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    store = QSettings(str(tmp_path / "parameters.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(i18n, "settings", lambda: store)
    i18n.initialize()
    project = build_project(
        ["Well", "X", "Y"], [["A", "10", "20"], ["A", "11", "21"], ["A", "12", "22"]],
        [2, 3, 4], [(0, "Well", "text", ""), (1, "X", "number", "m"), (2, "Y", "number", "m")], {},
    )
    widget = MainWindow()
    qtbot.addWidget(widget)
    widget.set_project(project)
    widget.may_replace = lambda: True
    yield widget
    widget.dirty = False
    QWhatsThis.leaveWhatsThisMode()


def test_form_resets_declared_defaults_with_one_notification(qtbot):
    form = axis_form()
    qtbot.addWidget(form)
    form.load({"minimum": 123, "scale": "log", "major_step": 5,
               "label_size": 25, "color": "#ff0000", "minor_grid": True})
    assert form.values()['major_step'] is None
    assert not form.fields['major_step'][0].isEnabled()
    assert not form.fields['number_format'][0].isEnabled()
    notifications = []
    form.edited.connect(lambda: notifications.append(True))
    form.reset_defaults()
    assert form.values() == vars(AxisSpec())
    assert notifications == [True]
    assert form.fields['major_step'][0].isEnabled()
    assert form.fields['number_format'][0].isEnabled()
    assert "логарифмической" in form.fields["minimum"][0].whatsThis()


def test_form_load_preserves_existing_signal_block(qtbot):
    form = axis_form()
    qtbot.addWidget(form)
    field = form.fields["label"][0]
    field.blockSignals(True)
    form.load({"label": "new"})
    assert field.signalsBlocked()
    field.blockSignals(False)
    empty = Form()
    qtbot.addWidget(empty)
    combo = empty.combo("unpopulated", "Empty", [])
    empty.reset_defaults()
    assert combo.count() == 0


def test_generic_reset_skips_result_tables_and_help_masks_password(qtbot):
    dialog = QDialog()
    qtbot.addWidget(dialog)
    layout = QVBoxLayout(dialog)
    field = QLineEdit("default")
    layout.addWidget(field)
    table = QTableWidget(1, 1)
    table.setItem(0, 0, QTableWidgetItem("source measurement"))
    layout.addWidget(table)
    add_parameter_tools(dialog)
    field.setText("changed")
    dialog.reset_defaults_button.click()
    assert field.text() == "default"
    assert table.item(0, 0).text() == "source measurement"
    password = QLineEdit()
    password.setEchoMode(QLineEdit.EchoMode.Password)
    set_parameter_help(password, "password", "Password", "private-secret")
    assert "private-secret" not in password.whatsThis()
    dialog.help_button.click()
    assert QWhatsThis.inWhatsThisMode()
    QWhatsThis.leaveWhatsThisMode()


def test_plot_reset_preserves_bindings_schemes_and_data(window, qtbot):
    from autopyro.ui.dialogs.plot import PlotDialog

    dialog = PlotDialog(window)
    qtbot.addWidget(dialog)
    dialog.timer.stop()
    original_rows = deepcopy(window.model.project.rows)
    x, y = dialog.x.currentData(), dialog.y.currentData()
    scheme = {"scheme_id": "preserved-test"}
    dialog.spec.interpretation = deepcopy(scheme)
    dialog.style.load({"marker_size": 35, "alpha": .2, "color": "#ff0000"})
    dialog.x_form.load({"minimum": "not a number", "scale": "log", "color": "#0000ff"})
    dialog.layout_form.load({"dpi": 600, "legend": False, "width_mm": 400})
    dialog.reset_defaults_button.click()
    dialog.timer.stop()
    assert (dialog.x.currentData(), dialog.y.currentData()) == (x, y)
    assert dialog.style.values()["marker_size"] == PlotSpec().marker_size
    assert dialog.layout_form.values()["dpi"] == PlotSpec().dpi
    assert dialog.x_form.values() == vars(AxisSpec())
    assert dialog.spec.interpretation == scheme
    assert window.model.project.rows == original_rows
    dialog.closed = True


def test_map_reset_preserves_crs_filters_and_loaded_layers(window, qtbot):
    from autopyro.ui.dialogs.map import MapDialog

    dialog = MapDialog(window)
    qtbot.addWidget(dialog)
    dialog.timer.stop()
    original = deepcopy(window.model.project.rows)
    dialog.data.load({"x_id": window.model.project.columns[1].id, "y_id": window.model.project.columns[2].id,
                      "source_crs": "EPSG:3857", "target_crs": "EPSG:32636"})
    dialog.filters.load({"depth_min": 100})
    layers = [{"id": "layer", "name": "keep layer", "features": []}]
    dialog.layers = deepcopy(layers)
    dialog.style.load({"marker_size": 30, "alpha": .1})
    dialog.layout_form.load({"dpi": 900, "xmin": "invalid", "legend": False})
    dialog.reset_defaults_button.click()
    dialog.timer.stop()
    assert dialog.style.values()["marker_size"] == MapSpec().marker_size
    assert dialog.layout_form.values()["xmin"] is None
    assert dialog.layout_form.values()["dpi"] == MapSpec().dpi
    assert dialog.data.values()["source_crs"] == "EPSG:3857"
    assert dialog.data.values()["target_crs"] == "EPSG:32636"
    assert dialog.filters.values()["depth_min"] == 100
    assert dialog.layers[0]["id"] == layers[0]["id"]
    assert dialog.layers[0]["features"] == layers[0]["features"]
    assert dialog.layers[0]["name"] == layers[0]["name"]
    assert dialog.layers[0]["alpha"] == .35
    assert window.model.project.rows == original
    assert "проецированная" in dialog.data.fields["target_crs"][0].whatsThis()
    dialog.closed = True


def test_statistics_reset_uses_defaults_instead_of_saved_configuration(window, qtbot):
    from autopyro.ui.dialogs.statistics import StatisticsDialog

    window.model.project.view["statistics"] = {"text_numbers": True, "correlations": False, "min_pairs": 12, "scope": "all"}
    dialog = StatisticsDialog(window)
    qtbot.addWidget(dialog)
    dialog.timer.stop()
    assert dialog.min_pairs.value() == 12
    dialog.reset_defaults_button.click()
    dialog.timer.stop()
    assert dialog.min_pairs.value() == 3
    assert not dialog.text_numbers.isChecked()
    assert dialog.correlations.isChecked()
    assert dialog.scope.currentData() == "filtered"
    dialog.closed = True


def test_geometry_view_reset_keeps_draft_and_revision(window, qtbot):
    from autopyro.ui.dialogs.geometry_editor import GeometryEditor, new_scheme
    from autopyro.scheme_geometry import add_geometry

    scheme = new_scheme()
    add_geometry(scheme, "Polygon", [(0, 0), (1, 0), (0, 1)], "region", "Keep region")
    dialog = GeometryEditor(scheme, window)
    qtbot.addWidget(dialog)
    draft = deepcopy(dialog.draft)
    dialog.reset_defaults_button.click()
    assert dialog.draft == draft
    assert scheme["revision"] != dialog.draft["revision"]


def test_table_view_reset_preserves_rows_and_named_views(window):
    original = deepcopy(window.model.project.rows)
    window.model.project.view["named"] = {"keep": {"rules": []}}
    window.apply_filter([{"column": window.model.project.columns[1].id, "operator": "gt", "value": 10}], "AND")
    window.table.setColumnHidden(0, True)
    window.actions[i18n.tr("Сбросить параметры представления")].trigger()
    assert window.proxy.rowCount() == len(original)
    assert not window.table.isColumnHidden(0)
    assert "keep" in window.model.project.view["named"]
    assert window.model.project.rows == original
