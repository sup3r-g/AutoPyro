"""Selective derived refresh matches full recalculation, including invalid graphs."""
from copy import deepcopy

import pytest

from autopyro import calculations, nonpyro, stratigraphy
from autopyro.core import Column
from autopyro.io import build_project
from autopyro.ui.table import AddColumnCommand, TableModel
from autopyro.ui.table.commands import EditCommand


def formula(name, kind, **inputs):
    return Column(name, "number", unit=calculations.FORMULAS[kind]["unit"], metadata={"calculation": {
        "formula": kind, "version": 1, "analysis_context": "Synthetic whole rock",
        "compatible_confirmed": True,
        "inputs": {role: {"column_id": c.id, "unit": c.unit} for role, c in inputs.items()}}})


@pytest.fixture
def graph():
    project = build_project(["TOC", "S1", "S2", "S3"],
                            [["2", ".5", "6", "1"], ["2", ".5", "ND", "1"], ["0", ".5", "6", "1"]],
                            [2, 3, 4], [(0, "TOC", "number", "%wt"),
                                        (1, "S1", "number", "mg HC/g rock"),
                                        (2, "S2", "number", "mg HC/g rock"),
                                        (3, "S3", "number", "mg CO2/g rock")], {})
    toc, s1, s2, s3 = project.columns
    upstream = formula("GP", "GP", S1=s1, S2=s2)
    second = formula("GP2", "GP", S1=upstream, S2=s2)
    hi = formula("HI", "HI", TOC=toc, S2=second)
    oi = formula("OI", "OI", TOC=toc, S3=s3)
    project.columns.extend((second, hi, oi))
    for row in project.rows:
        row.values.update({c.id: None for c in (second, hi, oi)})
    project.refresh()
    return project, upstream


class FullRefreshModel(TableModel):
    def refresh_derived_structure(self, column_id):
        self.refresh(invalidate_interpolation=False, sync_measurements=False)

    def refresh(self, **kwargs):
        kwargs.pop("changed_columns", None)
        return super().refresh(**kwargs)


def results(project):
    return ([row.values for row in project.rows],
            {c.id: c.metadata.get("diagnostics") for c in project.columns})


def test_bulk_edit_and_undo_recalculate_only_dependent_formulas(qtbot, monkeypatch, graph):
    project, upstream = graph
    model = TableModel(project)
    model.undo.push(AddColumnCommand(model, upstream, {}))
    _, s1, _, _, _, hi, oi = project.columns[:7]
    target = project.rows[0]
    original_hi = target.values[hi.id]
    original_oi_diagnostic = oi.metadata["diagnostics"][target.id]
    calls = []
    calculate = calculations.calculate_column

    def tracked(owner, config, **kwargs):
        calls.append(config["formula"])
        return calculate(owner, config, **kwargs)

    monkeypatch.setattr(calculations, "calculate_column", tracked)
    model.undo.push(EditCommand(model, [(target.id, s1.id, 1.5)]))
    assert calls == ["GP", "GP", "HI"]
    assert target.values[hi.id] == original_hi + 50
    assert oi.metadata["diagnostics"][target.id] is original_oi_diagnostic

    calls.clear()
    model.undo.undo()
    assert calls == ["GP", "GP", "HI"]
    assert target.values[hi.id] == original_hi


def test_mixed_batch_matches_full_refresh_and_limits_each_formula_to_its_rows(qtbot, monkeypatch, graph):
    project, upstream = graph
    model = TableModel(project)
    model.undo.push(AddColumnCommand(model, upstream, {}))
    reference = TableModel(deepcopy(project))
    original_refresh = reference.project.refresh

    def full_refresh(**kwargs):
        kwargs.pop("edited_cells", None)
        return original_refresh(**kwargs)

    monkeypatch.setattr(reference.project, "refresh", full_refresh)
    _, s1, _, s3 = project.columns[:4]
    first, second = project.rows[:2]
    changes = [(first.id, s1.id, 1.5), (second.id, s3.id, 4)]
    calls = []
    calculate = calculations.calculate_column

    def tracked(owner, config, **kwargs):
        if owner is project:
            calls.append((config["formula"], [row.id for row in kwargs["rows"]]))
        return calculate(owner, config, **kwargs)

    monkeypatch.setattr(calculations, "calculate_column", tracked)
    model.undo.push(EditCommand(model, changes))
    reference.undo.push(EditCommand(reference, changes))
    expected = [("GP", [first.id]), ("GP", [first.id]), ("HI", [first.id]), ("OI", [second.id])]
    assert calls == expected
    assert results(project) == results(reference.project)
    for action in ("undo", "redo"):
        calls.clear()
        getattr(model.undo, action)()
        getattr(reference.undo, action)()
        assert calls == expected
        assert results(project) == results(reference.project)


def test_scoped_recalculation_uses_resolved_rows_without_scanning_project(graph):
    project, _ = graph
    toc, _, _, s3 = project.columns[:4]
    oi = project.columns[-1]
    target, other = project.rows[:2]
    other_result = other.values[oi.id]
    target.values[s3.id] = 4

    class AddressedRows(list):
        def __iter__(self):
            pytest.fail("Scoped recalculation must not scan source rows")

    project.rows = AddressedRows(project.rows)
    calculations.recompute(project, rows=[target])
    assert target.values[oi.id] == 100 * 4 / target.values[toc.id]
    assert other.values[oi.id] == other_result


def test_add_undo_redo_recalculates_transitive_dependents_but_not_unrelated_formula(qtbot, monkeypatch, graph):
    project, upstream = graph
    model = TableModel(project)
    reference = FullRefreshModel(deepcopy(project))
    science = deepcopy(project.science)
    originals = deepcopy([r.original for r in project.rows])
    calls = []
    calculate = calculations.calculate_column

    def tracked(owner, config, **kwargs):
        if owner is project:
            calls.append(config["formula"])
        return calculate(owner, config, **kwargs)

    monkeypatch.setattr(calculations, "calculate_column", tracked)
    model.undo.push(AddColumnCommand(model, upstream, {}))
    reference.undo.push(AddColumnCommand(reference, deepcopy(upstream), {}))
    for action in (None, "undo", "redo"):
        if action:
            calls.clear()
            getattr(model.undo, action)()
            getattr(reference.undo, action)()
        assert results(project) == results(reference.project)
        assert "OI" not in calls and "GP" in calls and "HI" in calls
        assert project.science == science
        assert [r.original for r in project.rows] == originals
    hi = next(c for c in project.columns if c.name == "HI")
    assert project.rows[0].values[hi.id] == 625
    assert project.rows[1].values[hi.id] is None  # ND is never an ordinary number.
    assert hi.metadata["diagnostics"][project.rows[2].id]["status"] == "zero_denominator"


def test_new_dependent_of_existing_cycle_retains_cyclic_diagnostic(graph):
    project, a = graph
    s2 = project.columns[2]
    b = formula("Cycle B", "GP", S1=a, S2=s2)
    a.metadata["calculation"]["inputs"]["S1"] = {"column_id": b.id, "unit": b.unit}
    project.columns.extend((a, b))
    calculations.recompute(project)
    child = formula("Child of cycle", "HI", TOC=project.columns[0], S2=a)
    project.columns.append(child)
    reference = deepcopy(project)
    calculations.recompute(reference)
    calculations.recompute(project, changed_columns=[child.id])
    assert results(project) == results(reference)
    assert all(item["status"] == "cyclic_dependency" for item in child.metadata["diagnostics"].values())


def test_row_scoped_dependency_refresh_matches_full_formula_refresh(graph):
    project, upstream = graph
    project.columns.append(upstream)
    calculations.recompute(project)
    project.rows[0].values[project.columns[1].id] = 1
    reference = deepcopy(project)
    calculations.recompute(reference, rows=[reference.rows[0]])
    calculations.recompute(project, rows=[project.rows[0]], changed_columns=[project.columns[1].id])
    assert results(project) == results(reference)


@pytest.mark.parametrize("engine", ["nonpyro", "stratigraphy"])
def test_global_engines_keep_full_pyro_recalculation(qtbot, monkeypatch, graph, engine):
    project, _ = graph
    global_column = Column("Other engine", metadata={engine: {"synthetic": True}})
    project.columns.append(global_column)
    for row in project.rows:
        row.values[global_column.id] = None
    monkeypatch.setattr(nonpyro if engine == "nonpyro" else stratigraphy, "recompute", lambda _, **kwargs: None)
    model = TableModel(project)
    calls = []
    calculate = calculations.calculate_column

    def tracked(owner, config, **kwargs):
        calls.append(config["formula"])
        return calculate(owner, config, **kwargs)

    monkeypatch.setattr(calculations, "calculate_column", tracked)
    unrelated = Column("Interpolation", metadata={"method": "linear"})
    model.undo.push(AddColumnCommand(model, unrelated, {}))
    assert "OI" in calls
