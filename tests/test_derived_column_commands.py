"""Derived-column undo recalculates dependencies without rewriting source results."""
from copy import deepcopy

from autopyro.calculations import make_calculated_column
from autopyro.core import Column
from autopyro.io import build_project
from autopyro.ui.table import AddColumnCommand, FilterProxy, TableModel


def test_derived_add_undo_redo_preserves_measurements_and_recalculates_dependents(qtbot, monkeypatch):
    project = build_project(["TOC", "S1", "S2"], [["2", ".5", "6"]], [2],
                            [(0, "TOC", "number", "%wt"), (1, "S1", "number", "mg HC/g rock"),
                             (2, "S2", "number", "mg HC/g rock")], {})
    model = TableModel(project)
    toc, s1, s2 = project.columns
    gp, values = make_calculated_column(project, "GP", {
        "formula": "GP", "version": 1, "analysis_context": "Synthetic whole rock",
        "compatible_confirmed": True,
        "inputs": {c.name: {"column_id": c.id, "unit": c.unit} for c in (s1, s2)}})
    hi = Column("HI from GP", "number", metadata={"calculation": {
        "formula": "HI", "version": 1, "analysis_context": "Synthetic whole rock",
        "compatible_confirmed": True,
        "inputs": {"TOC": {"column_id": toc.id, "unit": toc.unit},
                   "S2": {"column_id": gp.id, "unit": gp.unit}}}})
    project.columns.append(hi)
    project.rows[0].values[hi.id] = None
    model.refresh()
    assert hi.metadata["diagnostics"][project.rows[0].id]["status"] == "invalid_definition"
    proxy = FilterProxy()
    proxy.setSourceModel(model)
    proxy.set_rules([{"column": hi.id, "operator": "gt", "value": 300}])
    assert proxy.rowCount() == 0
    science = deepcopy(project.science)
    originals = deepcopy(project.rows[0].original)
    sync = project.sync_measurements
    calls = []

    def tracked(**kwargs):
        calls.append(kwargs)
        return sync(**kwargs)

    monkeypatch.setattr(project, "sync_measurements", tracked)
    model.undo.push(AddColumnCommand(model, gp, values))
    assert project.rows[0].values[hi.id] == 325
    assert proxy.rowCount() == 1
    assert project.science == science
    model.undo.undo()
    assert project.rows[0].values[hi.id] is None
    assert proxy.rowCount() == 0
    assert hi.metadata["diagnostics"][project.rows[0].id]["status"] == "invalid_definition"
    model.undo.redo()
    assert project.rows[0].values[hi.id] == 325
    assert proxy.rowCount() == 1
    assert calls == []
    assert project.science == science
    assert project.rows[0].original == originals
    # Editing original data still performs the normal synchronization and recalculation.
    assert model.setData(model.index(0, 2), "8")
    assert calls
    assert project.science.current(project.rows[0].id, s2.id).value == 8
    assert project.rows[0].values[hi.id] == 425


def test_original_column_add_still_creates_scientific_results(qtbot):
    project = build_project(["Sample"], [["001"]], [2], [(0, "Sample", "text", "")], {})
    model = TableModel(project)
    column = Column("S2", "number", unit="mg HC/g rock")
    model.undo.push(AddColumnCommand(model, column, {project.rows[0].id: 6}))
    measurement = project.science.current(project.rows[0].id, column.id)
    assert measurement is not None and measurement.value == 6
    model.undo.undo()
    assert column.id not in project.rows[0].values
    model.undo.redo()
    assert project.science.current(project.rows[0].id, column.id).id == measurement.id
