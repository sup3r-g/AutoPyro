"""Independent geochemical workspaces preserve the existing method protocols."""

import pytest

from autopyro.core import Project
from autopyro.nonpyro import METHODS
from autopyro.ui.geochemistry import modules
from autopyro.ui.window import MainWindow


def test_modules_cover_every_method_once():
    methods = [method for _, methods in modules().values() for method in methods]
    assert len(methods) == len(set(methods)) == len(METHODS)
    assert set(methods) == set(METHODS)


@pytest.mark.parametrize("module_id", modules())
def test_menu_opens_scoped_document_with_analysis_only(qtbot, module_id):
    from PySide6.QtWidgets import QPushButton, QGroupBox
    window = MainWindow()
    qtbot.addWidget(window)
    title, methods = modules()[module_id]
    window.actions[title + "…"].trigger()
    document = window.geochemistry_windows[module_id]
    assert window.desktop.documents.currentWidget() is document
    assert window.desktop.properties_stack.currentWidget() is document.parameter_tabs
    assert [document.method.itemData(i) for i in range(document.method.count())] == list(methods)
    assert document.windowTitle() == title
    assert not hasattr(document, 'open_database')
    assert not any('БД' in button.text() for button in document.findChildren(QPushButton))
    assert not document.calculate_button.isEnabled()
    assert [document.parameter_tabs.tabText(i) for i in range(3)] == [
        '1. Метод', '2. Привязки', '3. Условия']
    for key, role in METHODS[methods[0]]['roles'].items():
        card = document.parameter_tabs.findChild(QGroupBox, 'geochemistryBinding_' + key)
        assert card and card.title() != key
    document.form.fields["context"][0].setText("Draft")
    window.show_nonpyro(module_id)
    assert window.geochemistry_windows[module_id] is document
    assert document.form.values()["context"] == "Draft"
    window.dirty = False


def test_independent_drafts_applicability_and_project_replacement(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show_nonpyro("gas")
    gas = window.geochemistry_windows["gas"]
    gas.method.setCurrentIndex(gas.method.findData("gas_isotopes"))
    gas.form.fields["context"][0].setText("Gas context")
    assert not gas.form.fields["isotope_reference"][0].isHidden()
    assert gas.form.fields["material"][0].isHidden()
    window.show_nonpyro("petrography")
    petrography = window.geochemistry_windows["petrography"]
    petrography.form.fields["context"][0].setText("Petrography context")
    petrography.method.setCurrentIndex(petrography.method.findData("microscopy"))
    window.show_nonpyro("gas")
    assert gas.method.currentData() == "gas_isotopes"
    assert gas.form.values()["context"] == "Gas context"
    assert window.desktop.active_payload(("table", "")) == ("geochemistry", "gas")
    window.set_project(Project())
    assert window.geochemistry_windows == {}
    assert window.desktop.documents.count() == 1
    window.dirty = False


def test_legacy_saved_profiles_appear_as_separate_sheets(qtbot):
    project = Project()
    project.view["nonpyro_configs"] = {"sofer": {}, "gas_isotopes": {}, "atomic_ratios": {}}
    project.view["geochemistry_current"] = {"gas": {"method": "gas_isotopes"}}
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.desktop.refresh_project()
    sheets = window.desktop.tree.topLevelItem(0).child(1)
    assert {sheets.child(i).text(0) for i in range(sheets.childCount())} == {
        modules()[key][0] for key in ("oil_isotopes", "gas", "elemental")}
    window.show_nonpyro("gas")
    assert window.geochemistry_windows["gas"].method.currentData() == "gas_isotopes"
    window.dirty = False


def test_database_import_opens_table_before_analysis(qtbot, monkeypatch):
    from types import SimpleNamespace
    from PySide6.QtWidgets import QDialog
    from autopyro.database.ui import nonpyro

    window = MainWindow()
    qtbot.addWidget(window)
    window.may_replace = lambda: True
    project = Project(source={"method_profile": "gas_isotopes"})
    monkeypatch.setattr(nonpyro, "NonPyroDatabaseDialog", lambda **kwargs: SimpleNamespace(
        exec=lambda: QDialog.DialogCode.Accepted, project=project, deleteLater=lambda: None))
    window.actions['Импорт геохимии из БД…'].trigger()
    assert window.model.project is project
    assert window.geochemistry_windows == {}
    assert window.desktop.documents.currentWidget() is window.desktop.table_page
    window.show_nonpyro('gas')
    document = window.geochemistry_windows["gas"]
    assert document.method.currentData() == "gas_isotopes"
    assert window.desktop.documents.currentWidget() is document
    assert window.desktop.properties_stack.currentWidget() is document.parameter_tabs
    window.dirty = False


def test_import_cancellation_and_unsaved_change_guard(qtbot, monkeypatch):
    from types import SimpleNamespace
    from PySide6.QtWidgets import QDialog
    from autopyro.database.ui import nonpyro

    window = MainWindow()
    qtbot.addWidget(window)
    original = window.model.project
    incoming = Project()
    for accepted in (False, True):
        monkeypatch.setattr(nonpyro, 'NonPyroDatabaseDialog', lambda **kwargs: SimpleNamespace(
            exec=lambda: QDialog.DialogCode.Accepted if accepted else QDialog.DialogCode.Rejected,
            project=incoming, deleteLater=lambda: None))
        window.may_replace = lambda: False
        window.import_geochemistry()
        assert window.model.project is original
    window.dirty = False


def test_database_append_keeps_project_and_supports_undo(qtbot, monkeypatch):
    from types import SimpleNamespace
    from PySide6.QtWidgets import QDialog
    from autopyro.database.ui import nonpyro
    from autopyro.model.core import Column, Row
    from autopyro.ui.data_io import database_actions

    first, second = Column('Sample / Name'), Column('Sample / Name')
    original = Project(columns=[first], rows=[Row({first.id: 'Existing'})])
    incoming = Project(columns=[second], rows=[Row({second.id: 'Incoming'})])
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(original)
    monkeypatch.setattr(database_actions, 'run_with_progress', lambda parent, title, work: work())
    monkeypatch.setattr(nonpyro, 'NonPyroDatabaseDialog', lambda **kwargs: SimpleNamespace(
        exec=lambda: QDialog.DialogCode.Accepted, project=incoming, deleteLater=lambda: None))
    window.actions['Добавить геохимию из БД в проект…'].trigger()
    assert window.model.project is original
    assert [row.values[first.id] for row in original.rows] == ['Existing', 'Incoming']
    assert len(original.columns) == 1
    window.model.undo.undo()
    assert [row.values[first.id] for row in original.rows] == ['Existing']
    window.model.undo.redo()
    assert len(original.rows) == 2
    window.dirty = False
