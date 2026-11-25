from types import SimpleNamespace
from threading import Event

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import least_squares

from autopyro.curves.curve_fitting import custom_model, standard_model
from autopyro.curves.joint_fitting import JointIdentifiabilityError, fit_joint_curve
from autopyro.curves.joint_bootstrap import bootstrap_joint_fit
from autopyro.curves.curve_bootstrap import BootstrapCancelled
from autopyro.curves.joint_sources import joint_source_digest, source_schema, update_joint_status
from autopyro.curves.joint_validation import compare_joint_sharing
from autopyro.model.core import Column, Project, Row
from autopyro.reports.joint_fit_figures import capture_joint_plots, render_joint_figures


def test_joint_report_bootstrap_extreme_finite_bounds_are_serializable():
    import json
    from autopyro.curves.joint_bootstrap import JointBootstrapResult

    result = SimpleNamespace(
        model=standard_model(), predict=lambda series, grid: np.zeros_like(grid),
        labels=["A"], x=np.array([0., 1.]), y=np.array([0., 0.]),
        indices=lambda series: np.arange(2),
    )
    coefficients = np.zeros((100, 1, 2))
    coefficients[:, 0, 1] = 1e308
    coefficients[:3, 0, 1] = -1e308
    bootstrap = JointBootstrapResult(coefficients, np.zeros((100, 1)), 100, 0, None)
    plots = capture_joint_plots(result, bootstrap, confidence=.95, simultaneous=False)
    for key in ("confidence_band", "prediction_band"):
        lower, upper = np.asarray(plots["series"][0][key])/1e308
        assert_allclose(lower, -.05)
        assert_allclose(upper, 1.)
    json.dumps(plots, allow_nan=False)
    fit = {"series": ["A"], "counts": [2], "axes": ["X", "Y"], "plots": plots, "bootstrap_failed": False}
    assert render_joint_figures(fit, "en", lambda *args: None)[0]["png"].startswith("iVBORw0KGgo")


def test_prediction_metrics_keep_errors_when_r2_overflows():
    from autopyro.curves.curve_validation import prediction_metrics

    metrics = prediction_metrics(np.array([0., 1e-100, 2e-100]), np.full(3, 1e100))
    assert np.isnan(metrics["R2"])
    assert metrics["RMSE"] == pytest.approx(1e100)
    assert metrics["MAE"] == pytest.approx(1e100)


def test_joint_report_sampling_keeps_extremes_and_small_series():
    x = np.r_[np.arange(3999, -1, -1), [1, 2]]
    fitted = x.astype(float)
    y = fitted.copy()
    y[999] += .5    # Large residual within its block, without being a Y extreme.
    y[1999] += 50
    y[2999] -= 50
    result = SimpleNamespace(x=x, y=y, fitted=fitted, labels=("A", "B"),
                             indices=lambda i: (np.arange(4000), np.arange(4000, 4002))[i],
                             predict=lambda i, grid: grid)
    large, small = capture_joint_plots(result, None, confidence=.95, simultaneous=False)["series"]
    assert len(large["x"]) == len(set(large["x"])) <= 2000
    assert large["sampling"] == "x_block_extrema"
    assert {x[0], x[999], x[1999], x[2999], x[3999]} <= set(large["x"])
    assert small["x"] == [1, 2] and "sampling" not in small


@pytest.mark.parametrize("dtype", [float, str, object])
def test_joint_cv_compares_sharing_on_known_series(dtype):
    x = np.tile(np.arange(12.), 2).astype(dtype)
    labels = np.repeat(["A", "B"], 12)
    y = (2*x.astype(float)+np.repeat([1., 8.], 12)).astype(dtype)
    candidates = [("b",), ("a",)]
    scores = compare_joint_sharing(x, y, labels, standard_model(), candidates, folds=4, seed=42)
    assert [score.shared for score in scores] == [("a",), ("b",)]
    assert scores[0].metrics["RMSE"] == pytest.approx(0, abs=1e-12)
    assert scores[1].metrics["RMSE"] > 1
    with pytest.raises(ValueError, match="не меньше точек"):
        compare_joint_sharing(x, y, labels, standard_model(), candidates, folds=13)


@pytest.mark.parametrize("field,value,error", [
    ("x", np.nan, "конечные пары"),
    ("y", np.inf, "конечные пары"),
    ("sy", 0., "Погрешности должны"),
    ("sy", -1., "Погрешности должны"),
    ("sy", np.nan, "Погрешности должны"),
    ("sy", np.inf, "Погрешности должны"),
    ("sy", [1., 2., 3.], "Погрешности должны"),
])
def test_joint_cv_rejects_invalid_data_before_fitting(monkeypatch, field, value, error):
    from autopyro.curves import joint_validation

    def unexpected_fit(*args, **kwargs):
        pytest.fail("Invalid input reached the fitting solver")

    monkeypatch.setattr(joint_validation, "fit_joint_curve", unexpected_fit)
    x = np.tile(np.arange(12.), 2)
    data = {"x": x, "y": 2*x+np.repeat([1., 8.], 12), "sy": None}
    if field == "sy":
        data[field] = value
    else:
        data[field] = data[field].copy()
        data[field][0] = value
    with pytest.raises(ValueError, match=error):
        compare_joint_sharing(
            **data, series=np.repeat(["A", "B"], 12), model=standard_model(),
            candidates=[("a",)], folds=2,
        )


@pytest.mark.parametrize("failure", [RuntimeError, ArithmeticError])
def test_joint_cv_continues_after_candidate_solver_failure(monkeypatch, failure):
    from autopyro.curves import joint_validation

    original_fit = joint_validation.fit_joint_curve
    warning = "Совместная подгонка не сошлась. Уточните начальные значения или границы"

    def fail_one_candidate(*args, **kwargs):
        if kwargs["shared"] == ("b",):
            raise failure(warning)
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(joint_validation, "fit_joint_curve", fail_one_candidate)
    x = np.tile(np.arange(12.), 2)
    y = 2*x+np.repeat([1., 8.], 12)
    scores = compare_joint_sharing(
        x, y, np.repeat(["A", "B"], 12), standard_model(),
        [("b",), ("a",)], folds=2,
    )
    assert scores[0].shared == ("a",)
    assert scores[0].metrics["RMSE"] == pytest.approx(0, abs=1e-12)
    assert scores[1].shared == ("b",)
    assert scores[1].metrics is None
    assert scores[1].error == warning


@pytest.mark.parametrize("seed", [None, True, -1, 1.5, [1, 2]])
def test_joint_cv_rejects_invalid_seed(seed):
    x = np.tile(np.arange(12.), 2)
    y = 2*x+np.repeat([1., 8.], 12)
    with pytest.raises(ValueError, match="Seed должен быть неотрицательным целым числом"):
        compare_joint_sharing(
            x, y, np.repeat(["A", "B"], 12), standard_model(), [("a",)], folds=2, seed=seed,
        )


@pytest.mark.parametrize("invalid_x", [0., -1.])
def test_joint_cv_checks_model_domain_before_fitting(monkeypatch, invalid_x):
    from autopyro.curves import joint_validation

    def unexpected_fit(*args, **kwargs):
        pytest.fail("Invalid model coordinates reached the fitting solver")

    monkeypatch.setattr(joint_validation, "fit_joint_curve", unexpected_fit)
    x = np.tile(np.arange(1., 13.), 2)
    x[0] = invalid_x
    with pytest.raises(ValueError, match="Некорректные координаты прогноза"):
        compare_joint_sharing(
            x, np.ones_like(x), np.repeat(["A", "B"], 12),
            standard_model("power"), [("a",)], folds=2,
        )


def test_joint_cv_preserves_series_order_for_parameter_bounds():
    x = np.repeat(np.arange(12.), 2)
    labels = np.tile(["A", "B"], 12)
    y = 2*x+np.tile([1., 8.], 12)
    scores = compare_joint_sharing(
        x, y, labels, standard_model(), [("a",)], folds=4, seed=42,
        bounds=([[0., 0.], [0., 7.]], [[4., 2.], [4., 9.]]),
        initial=[[2., 1.], [2., 8.]],
    )
    assert not scores[0].error
    assert scores[0].metrics["RMSE"] == pytest.approx(0, abs=1e-8)


@pytest.mark.parametrize("metric_failure", [False, True])
def test_joint_cv_discards_results_when_cancelled_during_final_metrics(monkeypatch, metric_failure):
    from autopyro.curves import joint_validation

    cancel = Event()
    original_metrics = joint_validation.prediction_metrics

    def cancel_during_metrics(y, prediction):
        cancel.set()
        if metric_failure:
            raise ValueError("Численное переполнение; измените масштаб данных или начальные параметры")
        return original_metrics(y, prediction)

    monkeypatch.setattr(joint_validation, "prediction_metrics", cancel_during_metrics)
    x = np.tile(np.arange(12.), 2)
    y = 2*x+np.repeat([1., 8.], 12)
    labels = np.repeat(["A", "B"], 12)
    assert compare_joint_sharing(
        x, y, labels, standard_model(), [("a",)], folds=2, cancelled=cancel.is_set,
    ) is None


def test_joint_source_status_tracks_saved_scope_and_legacy_snapshots():
    x = Column("X", "number")
    rows = [Row({x.id: value}) for value in (1, None, 3)]
    project = Project(columns=[x], rows=rows.copy())
    snapshot = project.view["joint_curve_report"] = {"row_ids": [rows[0].id]}
    update_joint_status(project)
    assert snapshot["source_status"] == "unverified"

    snapshot["run_id"] = "run"
    ids = [x.id]
    project.view["joint_curve_runs"] = {"run": {
        "version": 1, "source_ids": ids, "numeric_ids": ids, "row_ids": [r.id for r in rows[:2]],
        "digest": joint_source_digest(source_schema(project, ids),
                                      [(r.id, r.values[x.id]) for r in rows[:2]], []),
    }}
    project.rows.reverse()
    rows[2].values[x.id] = 5
    update_joint_status(project)
    assert snapshot["source_status"] == "current"
    rows[1].values[x.id] = 2
    update_joint_status(project, changed_rows=[rows[1]])
    assert snapshot["source_status"] == "stale"
    project.rows.remove(rows[1])
    update_joint_status(project)
    assert snapshot["source_status"] == "stale"


@pytest.mark.parametrize("shared", [("c1",), ("c0", "c2")])
def test_joint_wls_matches_dense_design(shared):
    rng = np.random.default_rng(6)
    series = np.repeat(["A", "B", "C"], [9, 22, 13])
    order = rng.permutation(len(series))
    series = series[order]
    x = rng.uniform(0, 3, len(series))
    sy = rng.uniform(.1, 1, len(x))
    y = 1+2*x+x*x+np.where(series == "A", .8, 0)+rng.normal(0, .1, len(x))
    model = standard_model("polynomial", degree=2)
    result = fit_joint_curve(x, y, series, model, shared=shared, sy=sy)
    basis = np.polynomial.polynomial.polyvander(x, 2)
    common = np.array([name in shared for name in model.parameters])
    design = np.column_stack([basis[:, common], *[basis[:, ~common]*(series == label)[:, None] for label in result.labels]])
    expected = np.linalg.lstsq(design/sy[:, None], y/sy, rcond=None)[0]
    assert_allclose(result.fitted, design @ expected, atol=1e-12)
    for i in range(len(result.labels)):
        indices = result.indices(i)
        assert_allclose(result.coefficients[i, common], expected[:len(shared)])
        assert_allclose(result.predict(i, x[indices]), result.fitted[indices])


def test_joint_fit_centers_large_x_with_series_intercepts():
    x = 1e9+np.tile(np.arange(8.), 3)
    series = np.repeat(["A", "B", "C"], 8)
    offsets = np.repeat([3., -4., 9.], 8)
    y = 2*(x-1e9)+offsets
    result = fit_joint_curve(x, y, series, standard_model(), shared=("a",))
    assert_allclose(result.coefficients[:, 0], 2.)
    assert_allclose(result.fitted, y, atol=1e-6)


def test_joint_fit_rejects_confounded_shared_coefficient():
    x = np.repeat([1., 2., 3.], 5)
    with pytest.raises(ValueError, match="Общие коэффициенты"):
        fit_joint_curve(x, 2*x+1, x, standard_model(), shared=("a",))


@pytest.mark.parametrize(("formula", "shared", "message"), [
    ("a*exp(b*x)+d+0*c", ("a", "b"), "Отдельные коэффициенты"),
    ("a*exp(b*x)+d+c", ("a", "b", "c"), "Общие коэффициенты"),
])
def test_joint_nonlinear_fit_rejects_unidentifiable_coefficients(formula, shared, message):
    x = np.tile(np.linspace(0, 1, 16), 2)
    series = np.repeat(["A", "B"], 16)
    y = 2*np.exp(.4*x)+np.repeat([1., 4.], 16)
    model = custom_model(formula, ("a", "b", "d", "c"))
    with pytest.raises(JointIdentifiabilityError, match=message):
        fit_joint_curve(x, y, series, model, shared=shared, initial=(2, .4, 1, 1))


def test_joint_nonlinear_bootstrap_rejects_replicates_that_lose_identifiability():
    x = np.r_[1., 2., np.zeros(30)]
    series = np.repeat(["A", "B"], 16)
    offsets = np.repeat([1., 4.], 16)
    y = 2*np.exp(.4*x)+offsets
    result = fit_joint_curve(x, y, series, standard_model("exponential"), shared=("a", "b"),
                             initial=[[2., .4, 1.], [2., .4, 4.]])
    with pytest.raises(ValueError, match="Недостаточно успешных повторов") as error:
        bootstrap_joint_fit(result, resamples=25, seed=42)
    assert isinstance(error.value.__cause__, JointIdentifiabilityError)


@pytest.mark.parametrize("model", [standard_model("exponential"), custom_model("a*exp(b*x)+c", ("a", "b", "c"))])
def test_bounded_nonlinear_joint_fit_matches_dense_reference(model):
    x = np.tile(np.linspace(0, 3, 21), 2)
    series = np.repeat(["A", "B"], 21)
    y = np.repeat([2., 4.], 21)*np.exp(.5*x) + np.repeat([.1, -.2], 21)
    sy = .3+x/10
    result = fit_joint_curve(x, y, series, model, shared=("b",), sy=sy,
                             initial=[1., .3, 0.], bounds=([0, 0, -np.inf], [np.inf, .36, np.inf]),
                             series_bounds={"A": {"a": [0, 1]}})

    def residual(parameters):
        rate, a, c, b, d = parameters
        amplitudes = np.where(series == "A", a, b)
        offsets = np.where(series == "A", c, d)
        return (amplitudes*np.exp(rate*x)+offsets-y)/sy

    dense = least_squares(residual, [.3, .9, 0, 1, 0], bounds=([0, 0, -np.inf, 0, -np.inf],
                                                              [.36, 1, np.inf, np.inf, np.inf]))
    rate, a, c, b, d = dense.x
    assert_allclose(result.coefficients, [[a, rate, c], [b, rate, d]], atol=1e-6)
    assert result.active_bounds[0, 0] == 1 and np.all(result.active_bounds[:, 1] == 1)


def test_bounded_linear_fit_applies_local_override_without_constraining_other_series():
    x = np.tile(np.arange(10.), 2)
    series = np.repeat(["A", "B"], 10)
    y = 2*x + np.repeat([1., 5.], 10)
    fit = fit_joint_curve(x, y, series, standard_model(), shared=("a",),
                          series_bounds={"B": {"b": [None, 3.]}})
    assert fit.coefficients[1, 1] == pytest.approx(3.)
    assert fit.coefficients[0, 1] < 1
    assert fit.coefficients[0, 0] == fit.coefficients[1, 0]
    with pytest.raises(ValueError, match="отдельного коэффициента"):
        fit_joint_curve(x, y, series, standard_model(), shared=("a",), series_bounds={"A": {"a": [0, 1]}})


@pytest.mark.parametrize("loss", ["huber", "soft_l1"])
def test_robust_joint_fit_matches_dense_weighted_reference(loss):
    x = np.tile(np.arange(14.), 2)
    codes = np.repeat([0, 1], 14)
    y = 2*x + np.where(codes == 0, 1., 4.)
    y[13] += 50
    sy = .4+x/10
    fit = fit_joint_curve(x, y, codes, standard_model(), shared=("a",), sy=sy, loss=loss, f_scale=.5)
    design = np.column_stack((x, codes == 0, codes == 1))
    dense = least_squares(lambda beta: (design @ beta-y)/sy, [2, 1, 4], loss=loss, f_scale=.5)
    assert_allclose(fit.fitted, design @ dense.x, atol=1e-6)
    assert abs(fit.coefficients[0, 0]-2) < .03


def test_joint_bootstrap_matches_dense_stratified_replicates():
    x = np.r_[np.arange(12.), np.arange(16.)]
    codes = np.repeat([0, 1], [12, 16])
    y = 2*x + np.where(codes == 0, 1., 4.) + np.sin(x)
    sy = .4+x/10
    fit = fit_joint_curve(x, y, codes, standard_model(), shared=("a",), sy=sy)
    bootstrap = bootstrap_joint_fit(fit, resamples=20, seed=9)
    rng = np.random.default_rng(np.random.SeedSequence(9).spawn(2)[0])
    design = np.column_stack((x, codes == 0, codes == 1))/sy[:, None]
    expected = []
    for _ in range(20):
        indices = np.r_[rng.integers(0, 12, 12), 12+rng.integers(0, 16, 16)]
        slope, a, b = np.linalg.lstsq(design[indices], (y/sy)[indices], rcond=None)[0]
        expected.append([[slope, a], [slope, b]])
    assert_allclose(bootstrap.coefficients, expected, atol=1e-12)
    assert bootstrap.prediction_draws.shape == (20, 2)


def test_joint_robust_bootstrap_preserves_bounds_and_builds_bands():
    x = np.tile(np.arange(16.), 2)
    codes = np.repeat([0, 1], 16)
    y = 2*x + np.where(codes == 0, 1., 4.) + .1*np.sin(x)
    y[15] += 20
    fit = fit_joint_curve(x, y, codes, standard_model(), shared=("a",), loss="huber", f_scale=.5,
                          bounds=([0, -np.inf], [1.8, np.inf]), series_bounds={1: {"b": [None, 5.]}})
    bootstrap = bootstrap_joint_fit(fit, resamples=25, seed=4)
    assert_allclose(bootstrap.coefficients[:, 0, 0], bootstrap.coefficients[:, 1, 0])
    assert np.all(bootstrap.coefficients[:, :, 0] <= 1.8)
    assert np.all(bootstrap.coefficients[:, 1, 1] <= 5.)
    assert np.all(np.isfinite(bootstrap.bands(fit, 0, [0., 5., 10.])))
    with pytest.raises(BootstrapCancelled):
        bootstrap_joint_fit(fit, resamples=20, cancelled=lambda: True)
