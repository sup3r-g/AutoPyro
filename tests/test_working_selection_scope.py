"""Working-result changes keep row-local formulas scoped and global engines safe."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from autopyro.calculations import make_calculated_column
from autopyro.core import Column
from autopyro.io import build_project
from autopyro.ui.table import TableModel
from autopyro.ui.table.commands import SelectMeasurementCommand


@pytest.fixture
def model():
    project = build_project(["Sample", "TOC", "S2"],
        [["001", "2", "6"], ["001", "3", "ND"], ["001", "4", "<0.1"]], [2, 3, 4],
        [(0, "Sample", "text", ""), (1, "TOC", "number", "%wt"),
         (2, "S2", "number", "mg HC/g rock")], {})
    model = TableModel(project)
    toc, s2 = project.columns[1:]
    config = {"formula": "HI", "version": 1, "compatible_confirmed": True,
              "analysis_context": "Synthetic QA",
              "inputs": {"TOC": {"column_id": toc.id, "unit": toc.unit},
                         "S2": {"column_id": s2.id, "unit": s2.unit}}}
    hi, values = make_calculated_column(project, "HI", config)
    project.columns.append(hi)
    for row in project.rows:
        row.values[hi.id] = values[row.id]
    return model


def test_selection_undo_redo_only_synchronizes_target_and_notifies_its_row(model, qtbot, monkeypatch):
    project = model.project
    row = project.rows[0]
    s2, hi = project.columns[2:]
    original = project.science.current(row.id, s2.id)
    repeat = project.science.add_result(row.id, s2.id, "8", details={"uncertainty": .3})
    before = deepcopy(asdict(project.science))
    other_rows = deepcopy(project.rows[1:])
    sync_calls, notifications = [], []
    sync = project.sync_measurements

    def record_sync(*, rows=None):
        sync_calls.append(rows)
        return sync(rows=rows)

    monkeypatch.setattr(project, "sync_measurements", record_sync)
    model.dataChanged.connect(lambda first, last, roles: notifications.append((first.row(), last.row())))
    project.view["interpretation_result"] = {"stale": False}
    project.view["nonpyro_results"] = {"qa": {"stale": False}}
    for action, expected, value in (
        (lambda: model.undo.push(SelectMeasurementCommand(model, row.id, s2.id, repeat.id)), repeat, 400),
        (model.undo.undo, original, 300), (model.undo.redo, repeat, 400)):
        action()
        assert sync_calls[-1] == [row]
        assert notifications[-1] == (0, 0)
        assert project.science.current(row.id, s2.id) is expected
        assert row.values[hi.id] == value
        assert project.rows[1:] == other_rows
        assert project.view["interpretation_result"]["stale"]
        assert project.view["nonpyro_results"]["qa"]["stale"]
    after = asdict(project.science)
    before.pop("working")
    after.pop("working")
    assert before == after


@pytest.mark.parametrize(("engine", "full_refresh"), [("nonpyro", True), ("stratigraphy", False)])
def test_selection_refresh_respects_engine_scope(model, qtbot, monkeypatch, engine, full_refresh):
    project = model.project
    row = project.rows[0]
    s2 = project.columns[2]
    # Existing engines run independently; this marker makes their presence
    # visible without requiring a scientific schema for this dispatch test.
    column = Column("Derived output", metadata={engine: {"qa": True}})
    project.columns.append(column)
    for item in project.rows:
        item.values[column.id] = "old"
    from autopyro import nonpyro, stratigraphy
    module = nonpyro if engine == "nonpyro" else stratigraphy
    monkeypatch.setattr(module, "recompute", lambda p, **kwargs: None)
    refresh = project.refresh
    scopes, notifications = [], []

    def record_refresh(**kwargs):
        scopes.append(kwargs.get("rows"))
        return refresh(**kwargs)

    monkeypatch.setattr(project, "refresh", record_refresh)
    model.dataChanged.connect(lambda first, last, roles: notifications.append((first.row(), last.row())))
    repeat = project.science.add_result(row.id, s2.id, "8")
    model.undo.push(SelectMeasurementCommand(model, row.id, s2.id, repeat.id))
    assert scopes == [None if full_refresh else [row]]
    assert notifications == [(0, 2)]


def test_undo_selection_restores_absence_without_creating_a_result(model, qtbot):
    project = model.project
    row = project.rows[0]
    s2, hi = project.columns[2:]
    project.select_measurement(row.id, s2.id, None)
    repeat = project.science.add_result(row.id, s2.id, "8")
    result_ids = set(project.science.measurements)
    model.undo.push(SelectMeasurementCommand(model, row.id, s2.id, repeat.id))
    assert row.values[hi.id] == 400
    model.undo.undo()
    assert project.science.current(row.id, s2.id) is None
    assert row.values[s2.id] is None and row.values[hi.id] is None
    assert set(project.science.measurements) == result_ids
    model.undo.redo()
    assert project.science.current(row.id, s2.id) is repeat


def test_selection_marks_interpolation_stale_without_rerunning_it(model, qtbot):
    project = model.project
    row = project.rows[0]
    s2 = project.columns[2]
    predicted = Column("Prediction", "number", metadata={"method": "linear", "stale": False})
    project.columns.append(predicted)
    for item in project.rows:
        item.values[predicted.id] = 123
    notifications = []
    model.dataChanged.connect(lambda first, last, roles: notifications.append((first.row(), last.row())))
    repeat = project.science.add_result(row.id, s2.id, "8")
    model.undo.push(SelectMeasurementCommand(model, row.id, s2.id, repeat.id))
    assert predicted.metadata["stale"]
    assert all(item.values[predicted.id] == 123 for item in project.rows)
    assert notifications == [(0, 2)]  # Every displayed stale marker needs refreshing.
