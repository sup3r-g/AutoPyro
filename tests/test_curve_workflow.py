import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from autopyro.core import Column, Project, Row
from autopyro.curve_fitting import fit_curve, standard_model, custom_model
from autopyro.curve_constraints import parameter_model, parameter_samples
from autopyro.curve_validation import validation_plan
from autopyro.model_comparison import FitCandidate, compare_models
from autopyro.curve_bootstrap import bootstrap_bands
from autopyro.curve_outputs import prepare_output_columns, invalidate_curve_outputs
from autopyro.curve_archive import fit_payload, report_html


def test_group_splits_never_share_a_group():
    groups = np.repeat(["a", "b", "c", "d", "e"], [14, 7, 5, 3, 2])
    plan = validation_plan(len(groups), 3, strategy="group", groups=groups, seed=42)
    for fold in range(3):
        train, test = plan.split(fold)
        assert not set(groups[train]) & set(groups[test])
        assert test.any() and train.any()
    assert_allclose(plan.fold_ids, validation_plan(len(groups), 3, strategy="group", groups=groups, seed=42).fold_ids)


def test_time_splits_ties_gap_and_initial_training_rows():
    times = np.repeat(np.arange(20), 2)
    plan = validation_plan(len(times), 3, strategy="time", times=times, gap=2)
    for fold in range(3):
        train, test = plan.split(fold)
        assert times[train].max()+2 < times[test].min()
    for value in np.unique(times):
        assert len(set(plan.fold_ids[times == value])) == 1
    x = np.arange(len(times), dtype=float)
    result = compare_models(x, x*x+.1*np.sin(x), [FitCandidate("line", standard_model())],
                            strategy="time", times=times, gap=2, folds=3)
    predicted = result.ranking[0].predictions
    assert np.all(np.isnan(predicted[result.fold_ids < 0]))
    valid = result.fold_ids >= 0
    assert result.ranking[0].metrics["RMSE"] == pytest.approx(np.sqrt(np.mean((x[valid]**2+.1*np.sin(x[valid])-predicted[valid])**2)))
    assert result.ranking[0].metrics["N_evaluated"] == valid.sum()


@pytest.mark.parametrize("options", [{"strategy": "group", "groups": [None]*12},
                                    {"strategy": "group", "groups": ["one"]*12},
                                    {"strategy": "time", "times": ["bad"]*12},
                                    {"strategy": "time", "times": list(range(12)), "gap": 20}])
def test_invalid_split_metadata(options):
    with pytest.raises(ValueError):
        validation_plan(12, 3, **options)


def test_fixed_and_tied_parameters_reduce_degrees_of_freedom():
    model = parameter_model(standard_model("polynomial", degree=2), "c0=0; c2=2*c1")
    assert model.parameters == ("c1",)
    x = np.linspace(0, 2, 30)
    result = fit_curve(x, 3*x+6*x*x, model, initial=[1])
    assert_allclose(result.parameters, [3], atol=1e-7)
    expanded = parameter_samples(model, [[1], [2], [3]])
    assert_allclose(expanded["c0"], 0)
    assert_allclose(expanded["c2"], [2, 4, 6])
    bootstrap = bootstrap_bands(result, [0, 1, 2], resamples=30)
    intervals = {r["parameter"]: r for r in bootstrap.parameter_intervals(model)}
    assert intervals["c0"]["lower"] == intervals["c0"]["upper"] == 0
    assert intervals["c2"]["median"] == pytest.approx(2*intervals["c1"]["median"])


@pytest.mark.parametrize("rules", ["a=b;b=a", "b=x", "b=__import__('os')", "missing=2", "a=1;b=2", "b=1;b=2"])
def test_invalid_parameter_links(rules):
    with pytest.raises(ValueError):
        parameter_model(standard_model(), rules)


def test_multistart_escapes_bad_frequency_basin_and_is_reproducible():
    x = np.linspace(0, 6, 100)
    y = 2*np.sin(1.4*x)
    model = custom_model("a*sin(b*x)", ("a", "b"))
    options = dict(initial=[1, 7], bounds=([0, .1], [3, 8]))
    single = fit_curve(x, y, model, **options)
    multiple = fit_curve(x, y, model, starts=16, start_seed=9, **options)
    repeated = fit_curve(x, y, model, starts=16, start_seed=9, **options)
    assert multiple.objective < single.objective/1000
    assert_allclose(multiple.parameters, [2, 1.4], atol=1e-5)
    assert_allclose(multiple.parameters, repeated.parameters)
    assert len(multiple.start_diagnostics) == 16


def test_output_columns_include_unlabelled_rows_and_keep_provenance():
    x, y = Column(name="X", kind="number"), Column(name="Y", kind="number", unit="mg")
    rows = [Row(values={x.id: float(i), y.id: float(2*i+1) if i < 5 else None}) for i in range(6)]
    project = Project(columns=[x, y], rows=rows)
    settings = dict(x=x.id, y=y.id, sx=None, sy=None)
    result = fit_curve(np.arange(5.), 2*np.arange(5.)+1)
    record, columns = prepare_output_columns(project, rows, result, settings, [r.id for r in rows[:5]])
    assert len(columns) == 6
    assert columns[0][1][rows[-1].id] == pytest.approx(11)
    assert columns[1][1][rows[-1].id] is None
    assert all(c.unit == "mg" for c, _ in columns)
    project.view["curve_fit_runs"] = {record["id"]: record}
    project.columns.extend(c for c, _ in columns)
    invalidate_curve_outputs(project, None, True)
    assert not record["stale"]
    rows[0].values[x.id] = 99
    invalidate_curve_outputs(project, [rows[0]], True)
    assert record["stale"]
    assert all(c.metadata["stale"] for c, _ in columns)


def test_curve_sources_match_when_batch_edits_different_rows_and_columns():
    x, y = Column("X", "number"), Column("Y", "number")
    note = Column("Note")
    rows = [Row({x.id: i, y.id: 2*i+1, note.id: ""}) for i in range(6)]
    project = Project(columns=[x, y, note], rows=rows)
    project.sync_measurements()
    fitted = rows[:5]
    result = fit_curve(np.arange(5.), 2*np.arange(5.)+1)
    record, _ = prepare_output_columns(project, fitted, result, {"x": x.id, "y": y.id}, [row.id for row in fitted])
    project.view["curve_fit_runs"] = {record["id"]: record}

    rows[0].values[note.id] = "Review comment"
    rows[-1].values[x.id] = 99
    project.refresh(rows=[rows[0], rows[-1]], edited_cells={rows[0].id: [note.id], rows[-1].id: [x.id]})

    assert not record["stale"]


def test_snapshot_and_report_are_standalone_and_escape_user_content():
    x = np.arange(10.)
    result = fit_curve(x, 2*x+1+.1*np.sin(x))
    grid = np.linspace(0, 9, 30)
    record = fit_payload(result, {"x": "X", "y": "Y"}, [str(i) for i in range(10)], grid, result.bands(grid))
    record.update(name="<script>bad()</script>", at="2026-11-05", stale=False)
    json.dumps(record, allow_nan=False)
    html = report_html([record])
    assert "<script>bad()" not in html
    assert "&lt;script&gt;" in html
    assert "data:image/svg+xml;base64," in html
    assert "RMSE" in html
