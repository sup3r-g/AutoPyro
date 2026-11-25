import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import linregress, t

from autopyro.curve_fitting import custom_model, fit_curve, standard_model


def test_ols_and_bands_match_closed_form():
    x = np.arange(15.)
    y = 2*x+3+np.sin(x)
    r = fit_curve(x, y)
    reference = linregress(x, y)
    assert_allclose(r.parameters, [reference.slope, reference.intercept])
    assert_allclose(r.standard_errors, [reference.stderr, reference.intercept_stderr], rtol=1e-7)
    grid = np.array([0., 7., 14.])
    mean, lo, hi, plo, phi = r.bands(grid)
    s2 = np.sum((y-r.fitted)**2)/(len(x)-2)
    se2 = s2*(1/len(x)+(grid-x.mean())**2/np.sum((x-x.mean())**2))
    critical = t.ppf(.975, len(x)-2)
    assert_allclose(hi-mean, critical*np.sqrt(se2), rtol=1e-7)
    assert_allclose(phi-mean, critical*np.sqrt(se2+s2), rtol=1e-7)
    assert np.all(plo <= lo)
    assert np.all(phi >= hi)
    assert r.metrics["R2"] == pytest.approx(reference.rvalue**2)
    assert r.metrics["RMSE"]**2 == pytest.approx(r.metrics["MSE"])
    assert r.metrics["MAE"] == pytest.approx(np.mean(np.abs(y-r.fitted)))


def test_tls_matches_svd_and_covariance_schur_complement():
    rng = np.random.default_rng(42)
    x = np.linspace(0, 10, 120)+rng.normal(0, .7, 120)
    y = 2*np.linspace(0, 10, 120)+1+rng.normal(0, .7, 120)
    r = fit_curve(x, y, method="odr")
    _, _, vt = np.linalg.svd(np.column_stack([x-x.mean(), y-y.mean()]))
    slope = -vt[-1, 0]/vt[-1, 1]
    assert_allclose(r.parameters, [slope, y.mean()-slope*x.mean()], rtol=2e-5)
    assert np.max(np.abs(r.adjusted_x-x)) > .1
    n = len(x)
    jac = np.zeros((2*n, n+2))
    jac[:n, :2] = np.column_stack([r.adjusted_x, np.ones(n)])
    jac[np.arange(n), np.arange(n)+2] = r.parameters[0]
    jac[np.arange(n)+n, np.arange(n)+2] = 1
    expected = np.linalg.inv(jac.T@jac)[:2, :2]*r.objective/(n-2)
    assert_allclose(r.covariance, expected, rtol=1e-6)


def test_weighted_tls_and_ols_absolute_uncertainties():
    x = np.arange(12.)
    y = 1.5*x+2+np.cos(x)
    sx, sy = .2, .8
    r = fit_curve(x, y, method="odr", sx=sx, sy=sy, absolute_sigma=True)
    transformed = fit_curve(x/sx, y/sy, method="odr")
    assert_allclose(r.parameters, transformed.parameters*[sy/sx, sy], rtol=1e-5)
    weighted = fit_curve(x, y, sy=np.linspace(.1, 2, len(x)), absolute_sigma=True)
    design = np.column_stack([x, np.ones(len(x))])/np.linspace(.1, 2, len(x))[:, None]
    assert_allclose(weighted.covariance, np.linalg.inv(design.T@design), rtol=1e-7)
    assert_allclose(weighted.parameters, np.linalg.lstsq(design, y/np.linspace(.1, 2, len(x)), rcond=None)[0])


@pytest.mark.parametrize("name,beta", [("linear", [2, 3]), ("polynomial", [1, 2, -.3]),
                                      ("exponential", [2, .3, 1]), ("power", [2, 1.2]),
                                      ("logarithmic", [2, 3]), ("constant", [3])])
def test_standard_models(name, beta):
    model = standard_model(name)
    x = np.linspace(1, 5, 40)
    r = fit_curve(x, model.evaluate(x, beta), model, initial=beta)
    assert_allclose(r.parameters, beta, atol=1e-7)


def test_custom_model_and_nonlinear_odr():
    model = custom_model("a*exp(b*x) + c", ("a", "b", "c"))
    x = np.linspace(0, 3, 35)
    y = model.evaluate(x, [2, .4, 1])+.02*np.sin(5*x)
    r = fit_curve(x, y, model, method="odr", initial=[2, .4, 1], sx=.01, sy=.1)
    assert_allclose(r.parameters, [2, .4, 1], atol=.12)
    assert np.all(np.isfinite(r.covariance))


@pytest.mark.parametrize("expression", ["__import__('os')", "x.__class__", "x[0]", "[a for a in x]", "np.exp(x)", "a + unknown"])
def test_custom_model_rejects_python(expression):
    with pytest.raises(ValueError):
        custom_model(expression, ("a",))


@pytest.mark.parametrize("kwargs", [{"sx": 1}, {"sy": 0}, {"sy": -1}, {"sy": np.nan},
                                    {"method": "other"}, {"initial": [1]}, {"absolute_sigma": True}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        fit_curve(np.arange(5.), np.arange(5.), **kwargs)


def test_degenerate_and_invalid_data():
    for x, y in [([1, 1, 1], [1, 2, 3]), ([1, 2], [1, 2]), ([1, 2, 3], [1, np.nan, 3])]:
        with pytest.raises(ValueError):
            fit_curve(x, y)
    with pytest.raises(ValueError):
        fit_curve([-1, 0, 1], [1, 2, 3], standard_model("power"))
    r = fit_curve(np.arange(5.), np.ones(5))
    assert np.isnan(r.metrics["R2"])
    with pytest.raises(ValueError):
        r.bands([1, 2], confidence=1)


def test_prediction_noise_and_confidence_level():
    r = fit_curve(np.arange(10.), np.arange(10.)+np.sin(np.arange(10.)))
    ordinary = r.bands([1, 5])
    wider = r.bands([1, 5], confidence=.99, prediction_sigma=5)
    assert np.all(wider[2]-wider[0] > ordinary[2]-ordinary[0])
    assert np.all(wider[4]-wider[0] > ordinary[4]-ordinary[0])


@pytest.mark.parametrize("loss", ["huber", "soft_l1"])
@pytest.mark.parametrize("method", ["ols", "odr"])
def test_robust_regression_resists_outliers(loss, method):
    x = np.linspace(0, 10, 60)
    y = 2*x+3+.05*np.sin(x)
    y[-4:] += 80
    ordinary = fit_curve(x, y, method=method, sx=.01 if method == "odr" else None)
    robust = fit_curve(x, y, method=method, loss=loss, f_scale=.2,
                       sx=.01 if method == "odr" else None)
    assert abs(robust.parameters[0]-2) < .05
    assert abs(robust.parameters[0]-2) < abs(ordinary.parameters[0]-2)/10
    assert robust.covariance is None
    assert robust.standard_errors is None
    assert robust.loss == loss
    with pytest.raises(ValueError, match="Полосы"):
        robust.bands([1, 2])
    assert len(robust.x) == len(x)
    assert robust.objective < ordinary.objective


@pytest.mark.parametrize("method", ["ols", "odr"])
def test_parameter_bounds_are_enforced(method):
    x = np.linspace(0, 5, 40)
    r = fit_curve(x, 3*x+1, method=method, bounds=([0, -10], [2, 10]))
    assert_allclose(r.parameters[0], 2, atol=1e-6)
    assert r.active_bounds == ("a",)
    assert r.covariance is None
    interior = fit_curve(x, 3*x+1+.1*np.sin(x), method=method, bounds=([0, -10], [4, 10]))
    assert interior.covariance is not None
    assert not interior.active_bounds
    assert np.all(np.isfinite(interior.bands([1, 2])))


def test_nonlinear_bounds_and_derivatives_stay_in_domain():
    model = custom_model("sqrt(a)*x + b", ("a", "b"))
    x = np.linspace(0, 5, 30)
    result = fit_curve(x, -x+2, model, initial=[.1, 1], bounds=([0, -10], [10, 10]), method="odr")
    assert result.parameters[0] < 1e-6
    assert result.covariance is None
    assert "a" in result.active_bounds


@pytest.mark.parametrize("kwargs", [
    {"loss": "unknown"}, {"f_scale": 0}, {"f_scale": np.inf},
    {"bounds": ([1, 0], [0, 1])}, {"bounds": ([0, 0], [0, 1])},
    {"bounds": ([0, np.nan], [1, 2])}, {"bounds": ([0, 1, 2], [3, 4, 5])},
    {"bounds": ([0, 0], [1, 1]), "initial": [2, .5]},
])
def test_invalid_robust_and_bounded_options(kwargs):
    with pytest.raises(ValueError):
        fit_curve(np.arange(8.), np.arange(8.), **kwargs)
