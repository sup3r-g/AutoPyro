"""Conversions must preview without mutation and undo the complete typed state."""

from copy import deepcopy

import pytest

from autopyro.data_io.io import build_project
from autopyro.model.column_types import prepare_column_conversion
from autopyro.model.core import Project
from autopyro.ui.window import MainWindow


@pytest.fixture
def owner(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    project = build_project(["ID", "Value"], [["001", "27:30:00"], ["002", "09:00:00"]], [2, 3],
                            [(0, "ID", "identifier", ""), (1, "Value", "text", "")], {})
    window.set_project(project)
    window.may_replace = lambda: True
    yield window
    window.dirty = False


def panel_for(window, column):
    window.desktop.show_object_properties("column", column.id)
    return window.desktop.table_properties.summary.column_type


def test_inline_preview_apply_undo_redo_and_persistence(owner, qtbot, tmp_path):
    column = owner.model.project.columns[1]
    panel = panel_for(owner, column)
    panel.type_combo.setCurrentIndex(panel.type_combo.findData("duration"))
    # A scheduled explorer refresh must not erase the conversion draft.
    owner.desktop.refresh_project()
    assert panel.type_combo.currentData() == "duration"
    original = deepcopy(owner.model.project)
    panel.preview_button.click()
    assert panel.apply_button.isEnabled(), panel.status.text()
    assert owner.model.project == original
    assert panel.table.model().rowCount() == 2
    panel.apply_button.click()
    project = owner.model.project
    assert project.columns[1].kind == "duration"
    assert project.rows[0].values[column.id] == 99000
    assert project.science.properties[column.id].kind == "duration"
    assert project.science.current(project.rows[0].id, column.id).table_value() == 99000
    assert project.rows[0].original[column.id] == "27:30:00"
    owner.model.undo.undo()
    assert project.columns[1].kind == "text"
    assert project.rows[0].values[column.id] == "27:30:00"
    assert project.science == original.science
    owner.model.undo.redo()
    saved = tmp_path / "converted.apyro"
    project.save(saved)
    assert Project.load(saved).columns[1].kind == "duration"


def test_errors_and_stale_preview_cannot_change_project(owner):
    project = owner.model.project
    column = project.columns[1]
    panel = panel_for(owner, column)
    panel.type_combo.setCurrentIndex(panel.type_combo.findData("integer"))
    panel.preview_button.click()
    assert not panel.apply_button.isEnabled()
    assert panel.table.model().rowCount() == 2
    assert project.columns[1].kind == "text"
    assert owner.model.undo.count() == 0
    panel.type_combo.setCurrentIndex(panel.type_combo.findData("category"))
    panel.preview_button.click()
    assert panel.apply_button.isEnabled()
    owner.model.setData(owner.model.index(0, 1), "changed")
    panel.apply_button.click()
    assert project.columns[1].kind == "text"
    assert project.rows[0].values[column.id] == "changed"


def test_non_working_repeat_is_checked_and_original_sources_are_kept(owner):
    project = owner.model.project
    column = project.columns[0]
    measurement, analysis = project.science.prepare_result(project.rows[0].id, column.id, "bad")
    project.science.measurements[measurement.id] = measurement
    project.science.analyses[analysis.id] = analysis
    before, after, rows, errors = prepare_column_conversion(project, column.id, "integer")
    assert after is None and len(errors) == 1
    assert measurement.id in errors[0][0]
    assert rows[0][2] == 1
    assert project.columns[0].kind == "identifier"
    assert before == project


def test_derived_columns_are_protected(owner):
    column = owner.model.project.columns[1]
    column.metadata["calculation"] = {"formula": "S1"}
    panel = panel_for(owner, column)
    assert not panel.isEnabled()
    with pytest.raises(ValueError):
        prepare_column_conversion(owner.model.project, column.id, "number")


def test_conversion_recalculates_dependents_and_undo_restores_them(owner):
    from autopyro.analysis.calculations import FORMULAS
    from autopyro.model.core import Column

    project = build_project(["S1", "S2"], [["2", "3"]], [2],
                            [(0, "S1", "number", "mg HC/g rock"), (1, "S2", "number", "mg HC/g rock")], {})
    s1, s2 = project.columns
    calculated = Column("GP", "number", unit=FORMULAS["GP"]["unit"], metadata={"calculation": {
        "formula": "GP", "version": 1, "analysis_context": "Synthetic whole rock",
        "compatible_confirmed": True,
        "inputs": {role: {"column_id": column.id, "unit": column.unit} for role, column in (("S1", s1), ("S2", s2))}}})
    project.columns.append(calculated)
    project.rows[0].values[calculated.id] = None
    owner.set_project(project)
    assert project.rows[0].values[calculated.id] == 5
    original_science = deepcopy(project.science)
    panel = panel_for(owner, s1)
    panel.type_combo.setCurrentIndex(panel.type_combo.findData("text"))
    panel.preview_button.click()
    assert panel.apply_button.isEnabled(), panel.status.text()
    panel.apply_button.click()
    assert project.rows[0].values[calculated.id] is None
    owner.model.undo.undo()
    assert project.rows[0].values[calculated.id] == 5
    assert project.science == original_science
