import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import least_squares
from scipy.stats import t

from autopyro.core import Column, Project, Row
from autopyro.curve_fitting import CurveModel, custom_model, fit_curve, standard_model
from autopyro.curve_bootstrap import bootstrap_bands
from autopyro.curve_numerics import residual_autocorrelation
from autopyro.curve_outputs import output_columns, prepare_output_columns, source_digest, invalidate_curve_outputs
from autopyro.curve_validation import validation_plan
from autopyro.model_comparison import ComparisonCancelled, FitCandidate, compare_models, _prediction_metrics


@pytest.mark.parametrize("allow_nonfinite", [False, True])
def test_custom_model_rejects_complex_values(allow_nonfinite):
    model = custom_model("a*x+b+(-1)**0.5", ("a", "b"))
    with pytest.raises(ValueError, match="Модель должна возвращать вещественные значения Y"):
        model.evaluate(np.arange(3.), [1., 2.], allow_nonfinite=allow_nonfinite)


@pytest.mark.parametrize("expression", ["a*x+b+1/0", "a*x+b+1e308**2"])
def test_custom_model_reports_scalar_arithmetic_errors_as_invalid_values(expression):
    model = custom_model(expression, ("a", "b"))
    with pytest.raises(ValueError, match="Модель вернула NaN/Inf"):
        model.evaluate(np.arange(3.), [1., 2.])


def test_residual_autocorrelation_matches_direct_products_in_x_order():
    x = np.random.default_rng(30).permutation(115)
    residuals = np.sin(x/4)+.2*np.cos(x*2)
    ordered = residuals[np.argsort(x)] - residuals.mean()
    expected = [np.dot(ordered[:-lag], ordered[lag:])/np.dot(ordered, ordered) for lag in range(1, 41)]
    assert_allclose(residual_autocorrelation(x, residuals*1e200), expected, atol=1e-14)
    assert residual_autocorrelation(x[:3], residuals[:3]).shape == (2,)


def test_residual_autocorrelation_rejects_ties_and_constant_residuals():
    with pytest.raises(ValueError, match="должны быть различны"):
        residual_autocorrelation(np.array([0, 0, 2]), np.array([1, 0, 3]))
    with pytest.raises(ValueError, match="постоянных остатков"):
        residual_autocorrelation(np.arange(5), np.ones(5))


def test_weighted_influence_matches_explicit_leave_one_out_refits():
    x = np.array([0., 1., 3., 4., 5., 11.])
    y = np.array([1., 1., 2., 4., 4., 12.])
    sy = np.array([1., .4, .8, 1., .6, .5])
    fit = fit_curve(x, y, sy=sy)
    leverage, cook = fit.influence()
    design = np.column_stack((x, np.ones(len(x))))/sy[:, None]
    projection = design @ np.linalg.pinv(design)
    assert_allclose(leverage, projection.diagonal(), atol=1e-13)
    mse = np.sum(((y-fit.fitted)/sy)**2)/(len(x)-2)
    expected = []
    predicted = []
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        beta = np.linalg.lstsq(design[keep], (y/sy)[keep], rcond=None)[0]
        expected.append(np.sum((design @ beta-fit.fitted/sy)**2)/(2*mse))
        predicted.append((design[i] @ beta)*sy[i])
    assert_allclose(cook, expected, atol=1e-12)
    assert_allclose(fit.leave_one_out(), predicted, atol=1e-12)
    assert_allclose(fit.leave_one_out(leverage=leverage), predicted, atol=1e-12)


def test_influence_leaves_undefined_cook_distance_empty_and_rejects_constraints():
    x = np.array([0., 0., 0., 2.])
    fit = fit_curve(x, [0, 1, 2, 4])
    leverage, cook = fit.influence()
    assert_allclose(leverage, [1/3, 1/3, 1/3, 1])
    assert np.isnan(cook[-1])
    assert_allclose(fit.leave_one_out(), [1.5, 1., .5, np.nan], atol=1e-12)
    assert np.all(np.isnan(fit_curve(x, 2*x+3).influence()[1]))
    with pytest.raises(ValueError, match="без ограничений"):
        fit_curve(x, [0, 1, 2, 4], bounds=([0, 0], [10, 10])).influence()


def test_large_offset_weighted_line_and_bands_match_centered_closed_form():
    x = 1e9+np.linspace(-1, 1, 40)
    y = 2*x+3+.01*np.sin(np.arange(len(x)))
    sigma = np.linspace(.2, .5, len(x))
    origin = x.mean()
    design = np.column_stack([x-origin, np.ones(len(x))])/sigma[:, None]
    weights = 1/sigma**2
    dx, dy = x-origin, y-y[0]
    mx, my = np.average(dx, weights=weights), np.average(dy, weights=weights)
    slope = np.sum(weights*(dx-mx)*(dy-my))/np.sum(weights*(dx-mx)**2)
    gamma = np.array([slope, y[0]+my-slope*mx])
    residual = (y-(gamma[0]*(x-origin)+gamma[1]))/sigma
    covariance = np.linalg.inv(design.T@design)*(residual@residual)/(len(x)-2)
    result = fit_curve(x, y, sy=sigma)
    grid = origin+np.array([-.8, 0, .8])
    centered = np.column_stack([grid-origin, np.ones(len(grid))])
    expected = t.ppf(.975, len(x)-2)*np.sqrt(np.einsum("ij,jk,ik->i", centered, covariance, centered))
    mean, low, high, _, _ = result.bands(grid)
    assert_allclose(result.parameters, [gamma[0], gamma[1]-gamma[0]*origin], rtol=1e-5)
    assert_allclose(high-mean, expected, rtol=3e-4)
    assert_allclose(mean-low, expected, rtol=3e-4)
    assert_allclose(mean, centered@gamma, atol=1e-5, rtol=0)


@pytest.mark.parametrize("custom", [False, True])
def test_small_rate_exponential_is_invariant_to_x_units(custom):
    model = custom_model("a*exp(b*x)+c", ("a", "b", "c")) if custom else standard_model("exponential")
    x = np.linspace(0, 1, 40)
    y = 2*np.exp(.8*x)+1+.001*np.sin(np.arange(len(x)))
    reference = fit_curve(x, y, model, initial=[2, .8, 1])
    result = fit_curve(x*1e9, y, model, initial=[2, .8e-9, 1])
    assert_allclose(result.parameters*[1, 1e9, 1], reference.parameters, rtol=2e-5)
    assert_allclose(result.standard_errors*[1, 1e9, 1], reference.standard_errors, rtol=2e-4)
    assert not any("плохо обусловлена" in w for w in result.warnings)


def test_linear_api_includes_fixed_offset_and_single_parameter_error_is_positive():
    model = CurveModel("affine", ("a",), lambda x, a: a*x+42, "a*x+42", linear=True)
    x = np.arange(12.)
    y = 2*x+42+.1*np.sin(x)
    result = fit_curve(x, y, model)
    expected = x@(y-42)/(x@x)
    assert_allclose(result.parameters, [expected])
    assert result.standard_errors[0] > 0
    constant = fit_curve(x, y, standard_model("constant"))
    assert constant.standard_errors[0] > 0
    exact = fit_curve(x, 2*x+42, model)
    assert not any("плохо обусловлена" in w for w in exact.warnings)


@pytest.mark.parametrize("loss", ["linear", "huber", "soft_l1"])
def test_nonlinear_odr_matches_independent_dense_solver(loss):
    x = np.linspace(.2, 2, 25)
    y = 2*np.exp(.4*x)+1+.02*np.sin(4*x)
    y[-1] += .15
    sx, sy = .02, .08
    model = standard_model("exponential")
    result = fit_curve(x, y, model, method="odr", initial=[2, .4, 1], sx=sx, sy=sy, loss=loss)

    def residual(z):
        a, b, c = z[:3]
        corrected = z[3:]
        return np.r_[(a*np.exp(b*corrected)+c-y)/sy, (corrected-x)/sx]

    reference = least_squares(residual, np.r_[result.parameters, x], jac="3-point", loss=loss,
                              x_scale="jac", max_nfev=1000, ftol=1e-11, xtol=1e-11, gtol=1e-11)
    assert reference.success
    assert_allclose(result.parameters, reference.x[:3], rtol=2e-4, atol=2e-4)
    assert result.objective == pytest.approx(2*reference.cost, rel=1e-6)


def test_prediction_samples_do_not_depend_on_grid_density_or_order():
    x = np.arange(30.)
    result = fit_curve(x, x+np.sin(x), loss="huber")
    boot = bootstrap_bands(result, [5, 5, 10], resamples=80, seed=42)
    bands = boot.bands()
    for band in bands:
        assert_allclose(band[0], band[1], atol=1e-13)
    reordered = boot.bands_at(result, [10, 5])
    for original, reordered_band in zip(bands, reordered):
        assert_allclose(original[[2, 0]], reordered_band, atol=1e-12)
    assert boot.diagnostics()["algorithm_version"] == 2


def test_bootstrap_output_chunks_reuse_noise_and_match_graph(monkeypatch):
    x = np.arange(20.)
    result = fit_curve(x, 2*x+1+.1*np.sin(x), loss="huber")
    boot = bootstrap_bands(result, x, resamples=40, seed=0)
    cx, cy = Column(name="X", kind="number"), Column(name="Y", kind="number")
    rows = [Row(values={cx.id: float(i % 20), cy.id: None}) for i in range(600)]
    import autopyro.curve_bootstrap as module
    monkeypatch.setattr(module, "_prediction_residuals", lambda *a, **k: pytest.fail("Noise pool was recomputed"))
    output = output_columns(result, rows, {"x": cx.id, "y": cy.id}, bootstrap=boot)
    expected = boot.bands()
    for key, band in zip(("confidence_lower", "confidence_upper", "prediction_lower", "prediction_upper"), expected[1:]):
        for i in (0, 19, 512, 599):
            assert output[key][rows[i].id] == pytest.approx(band[i % 20], abs=1e-12)


def test_simultaneous_bootstrap_reuses_grid_calibration_for_table_output():
    x = np.linspace(1, 10, 60)
    fit = fit_curve(x, 2*x + np.sin(x))
    grid = np.linspace(1, 10, 40)
    options = dict(resamples=40, seed=20)
    simultaneous = bootstrap_bands(fit, grid, simultaneous=True, **options)
    pointwise = bootstrap_bands(fit, grid, **options)
    mean, lo, hi, plo, phi = simultaneous.bands(confidence=.9)
    expected = np.quantile([max(abs(curve-mean)) for curve in pointwise.curve_samples], .9)
    assert_allclose(mean-lo, expected)
    assert_allclose(hi-mean, expected)
    assert_allclose([plo, phi], pointwise.bands(confidence=.9)[3:])
    rows = [Row(values={"x": float(value)}) for value in [1, 4.5, 10, 12]]
    values = output_columns(fit, rows, {"x": "x", "y": "y", "band_method": "bootstrap_simultaneous"},
                            bootstrap=simultaneous, confidence=.9)
    for row in rows[:3]:
        assert_allclose(values["prediction"][row.id]-values["confidence_lower"][row.id], expected)
    assert values["confidence_lower"][rows[-1].id] is None
    assert values["prediction_lower"][rows[-1].id] is not None


def test_explicit_bootstrap_failure_does_not_fall_back_in_output_columns():
    result = fit_curve(np.arange(10.), np.arange(10.)+np.sin(np.arange(10.)))
    cx, cy = Column(name="X", kind="number"), Column(name="Y", kind="number")
    rows = [Row(values={cx.id: 1., cy.id: 2.})]
    project = Project(columns=[cx, cy], rows=rows)
    record, columns = prepare_output_columns(project, rows, result,
                                             {"x": cx.id, "y": cy.id, "band_method": "bootstrap"}, [], confidence=.9)
    assert len(columns) == 2
    assert record["band_method_actual"] is None
    assert record["confidence"] == .9


def test_canonical_source_signature_ignores_layout_but_detects_data_changes():
    cx, cy = Column(name="X", kind="number"), Column(name="Y", kind="number")
    rows = [Row(values={cx.id: float(i), cy.id: 2*i+1}) for i in range(5)]
    project = Project(columns=[cx, cy], rows=rows)
    ids = [r.id for r in rows]
    signature = source_digest(project, [cx.id, cy.id], ids)
    project.columns.reverse()
    project.rows.reverse()
    assert source_digest(project, [cy.id, cx.id], list(reversed(ids))) == signature
    project.rows[0].values[cx.id] += 1
    assert source_digest(project, [cx.id, cy.id], ids) != signature


def test_legacy_signature_is_still_accepted():
    cx = Column(name="X", kind="number")
    project = Project(columns=[cx], rows=[Row(values={cx.id: 1.})])
    record = {"source_ids": [cx.id], "source_rows": [project.rows[0].id], "stale": False}
    record["source_digest"] = source_digest(project, record["source_ids"], record["source_rows"], version=1)
    project.view["curve_fit_results"] = [record]
    invalidate_curve_outputs(project, None, True)
    assert not record["stale"]


def test_time_validation_keeps_integer_nanoseconds_distinct():
    times = np.array([10**18+i for i in range(20)], dtype=object)
    plan = validation_plan(20, 3, strategy="time", times=times)
    assert len(np.unique(plan.time_rank)) == 20


def test_nonfinite_cv_metrics_are_rejected_and_comparison_can_be_cancelled():
    with pytest.raises(ValueError):
        _prediction_metrics(np.array([1., 2.]), np.array([1e200, 1e200]))
    with pytest.raises(ComparisonCancelled):
        compare_models(np.arange(10.), np.arange(10.), [FitCandidate("line", standard_model())], cancelled=lambda: True)


def test_prediction_and_bands_validate_domains_consistently():
    x = np.arange(1., 6.)
    result = fit_curve(x, x*x, standard_model("power"), initial=[1, 2])
    with pytest.raises(ValueError):
        result.predict([-1])
    constant = fit_curve(x, np.ones(5), standard_model("constant"))
    with pytest.raises(ValueError):
        constant.predict([np.nan])
    with pytest.raises(ValueError):
        constant.bands([])
