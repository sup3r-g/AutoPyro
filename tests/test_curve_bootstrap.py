from threading import Event

import numpy as np
import pytest
from numpy.testing import assert_allclose

from autopyro.curves.curve_bootstrap import BootstrapCancelled, bootstrap_bands, paired_mean_interval
from autopyro.curve_fitting import fit_curve, standard_model


@pytest.mark.parametrize("options, method", [({}, "rows"), ({"groups": ["A", "A", "B", "C", "D"]}, "groups"),
                                            ({"block_length": 2}, "moving_blocks")])
def test_paired_interval_mirrors_when_model_order_is_reversed(options, method):
    delta = np.array([1., -2., np.nan, 4., 3.])
    x = np.array([3., 2., 1., 0., 4.])
    ci = paired_mean_interval(delta, x, resamples=80, seed=12, **options)
    reversed_ci = paired_mean_interval(-delta, x, resamples=80, seed=12, **options)
    assert ci["method"] == method and ci["N_common"] == 4
    assert_allclose([reversed_ci["lower"], reversed_ci["upper"]], [-ci["upper"], -ci["lower"]], atol=1e-14)


@pytest.mark.parametrize("delta, expected", [([2., 2.], "a"), ([-2., -2.], "b"),
                                              ([-1., 1.], "within_margin"), ([.5, 1.5], "uncertain")])
def test_practical_margin_classifies_the_full_paired_interval(delta, expected):
    options = dict(resamples=80, seed=4)
    baseline = paired_mean_interval(delta, [0, 1], **options)
    interval = paired_mean_interval(delta, [0, 1], margin=1., **options)
    assert interval["practical"] == {"margin": 1., "conclusion": expected}
    assert {key: value for key, value in interval.items() if key != "practical"} == baseline
    with pytest.raises(ValueError, match="неотрицательным"):
        paired_mean_interval(delta, [0, 1], margin=-1., **options)


def test_paired_interval_uses_whole_groups_and_detects_zero_crossing():
    delta = np.array([-3., -1., 6., 10.])
    groups = ["a", "a", "b", "c"]
    ci = paired_mean_interval(delta, np.arange(4), groups=groups, resamples=80, seed=4)
    rng = np.random.default_rng(4)
    blocks = [delta[:2], delta[2:3], delta[3:]]
    means = [np.concatenate([blocks[i] for i in rng.integers(0, 3, 3)]).mean() for _ in range(80)]
    assert_allclose([ci["lower"], ci["upper"]], np.quantile(means, [.025, .975]))
    assert ci["direction"] == "uncertain"
    with pytest.raises(ValueError, match="две общие пробы"):
        paired_mean_interval([2., np.nan], [0., 1.])


def test_pairs_bootstrap_matches_independent_weighted_fits():
    x = np.linspace(0, 10, 35)
    y = 2*x+3+np.sin(x)
    sy = np.linspace(.2, 2, len(x))
    result = fit_curve(x, y, sy=sy, absolute_sigma=True)
    grid = np.array([0., 5., 10.])
    bootstrap = bootstrap_bands(result, grid, resamples=80, seed=42)
    rng = np.random.default_rng(np.random.SeedSequence(42).spawn(2)[0])
    parameters = []
    for _ in range(80):
        indices = rng.integers(0, len(x), size=len(x))
        design = np.column_stack([x[indices], np.ones(len(x))])/sy[indices, None]
        parameters.append(np.linalg.lstsq(design, y[indices]/sy[indices], rcond=None)[0])
    parameters = np.asarray(parameters)
    assert_allclose(bootstrap.parameters, parameters, atol=1e-10)
    curves = parameters[:, :1]*grid+parameters[:, 1:]
    assert_allclose(bootstrap.curve_samples, curves, atol=1e-10)
    _, lo, hi, _, _ = bootstrap.bands(confidence=.9)
    assert_allclose([lo, hi], np.quantile(curves, [.05, .95], axis=0))
    assert bootstrap.prediction_noise == "empirical_residuals"
    assert bootstrap.successful == 80
    assert bootstrap.failures == 0


def test_cluster_bootstrap_matches_weighted_fits_of_whole_unequal_groups():
    groups = np.array(["A", "B", "C", "B", "C", "C", "A", "B", "C"])
    x = np.arange(len(groups), dtype=float)
    y = 2*x+np.sin(x)+np.where(groups == "B", 3, 0)
    sy = np.linspace(.2, 2, len(x))
    result = fit_curve(x, y, sy=sy)
    bootstrap = bootstrap_bands(result, [0, 4, 8], groups=groups, resamples=40, seed=8)
    rng = np.random.default_rng(np.random.SeedSequence(8).spawn(2)[0])
    expected = []
    for _ in range(40):
        selected = rng.choice(["A", "B", "C"], size=3, replace=True)
        indices = np.concatenate([np.flatnonzero(groups == label) for label in selected])
        design = np.column_stack([x[indices], np.ones(len(indices))])/sy[indices, None]
        expected.append(np.linalg.lstsq(design, y[indices]/sy[indices], rcond=None)[0])
    assert_allclose(bootstrap.parameters, expected, atol=1e-10)
    assert bootstrap.diagnostics()["method"] == "cluster_percentile"
    assert bootstrap.diagnostics()["group_count"] == 3


@pytest.mark.parametrize("groups, error", [(["A"]*6, "хотя бы две"), (["A", "B"], "непустые группы"),
                                            (["A", "B", "", "B", "A", "B"], "непустые группы")])
def test_cluster_bootstrap_rejects_invalid_groups(groups, error):
    x = np.arange(6.)
    result = fit_curve(x, 2*x+1)
    with pytest.raises(ValueError, match=error):
        bootstrap_bands(result, [1, 2], groups=groups, resamples=20)


def test_moving_block_bootstrap_matches_independent_weighted_refits():
    x = np.random.default_rng(10).permutation(29).astype(float)
    y = 2*x+3*np.sin(x/3)
    sy = .2+x/20
    fit = fit_curve(x, y, sy=sy)
    block_length = 6
    bootstrap = bootstrap_bands(fit, [1, 10, 20], block_length=block_length, resamples=40, seed=4)
    rng = np.random.default_rng(np.random.SeedSequence(4).spawn(2)[0])
    sorted_rows = np.argsort(x)
    blocks = [sorted_rows[start:start+block_length] for start in range(len(x)-block_length+1)]
    expected = []
    for _ in range(40):
        indices = np.concatenate([blocks[start] for start in rng.choice(len(blocks), size=5)])[:len(x)]
        design = np.column_stack([x[indices], np.ones(len(indices))])/sy[indices, None]
        expected.append(np.linalg.lstsq(design, y[indices]/sy[indices], rcond=None)[0])
    assert_allclose(bootstrap.parameters, expected, atol=1e-10)
    assert bootstrap.diagnostics()["method"] == "moving_block_percentile"
    assert bootstrap.diagnostics()["block_length"] == block_length


def test_moving_blocks_reject_duplicate_x_and_invalid_configuration():
    x = np.array([0., 0., 1., 2.])
    fit = fit_curve(x, x+1)
    for options, error in (({"block_length": 2}, "должны быть различны"),
                           ({"block_length": 4}, "Длина блока"),
                           ({"block_length": 2, "groups": [1, 1, 2, 2]}, "либо группы")):
        with pytest.raises(ValueError, match=error):
            bootstrap_bands(fit, [0, 1], resamples=20, **options)


@pytest.mark.parametrize("loss", ["huber", "soft_l1"])
def test_robust_active_bounds_produce_finite_reproducible_bands(loss):
    x = np.linspace(0, 8, 45)
    y = 3*x+1+.2*np.sin(x)
    y[-3:] += 20
    result = fit_curve(x, y, loss=loss, f_scale=.5, bounds=([0, -10], [2, 10]))
    assert result.covariance is None
    assert "a" in result.active_bounds
    before = result.parameters.copy()
    grid = np.linspace(0, 8, 21)
    bootstrap = bootstrap_bands(result, grid, resamples=40, seed=0)
    repeated = bootstrap_bands(result, grid, resamples=40, seed=0)
    assert_allclose(bootstrap.parameters, repeated.parameters)
    assert_allclose(bootstrap.prediction_samples, repeated.prediction_samples)
    assert np.all((bootstrap.parameters[:, 0] >= 0) & (bootstrap.parameters[:, 0] <= 2))
    assert np.all(np.isfinite(bootstrap.bands()))
    assert_allclose(result.parameters, before)
    assert_allclose(bootstrap.bands()[0], result.predict(grid))
    # Prediction retains large empirical outliers instead of trimming them as if
    # robust fitting implied that future observations could never be outliers.
    assert np.max(bootstrap.prediction_samples-bootstrap.curve_samples) > 10


def test_odr_resamples_coordinate_uncertainties_and_bounds():
    x = np.linspace(1, 5, 25)
    sy = np.linspace(.1, .4, len(x))
    result = fit_curve(x, 2*x+1+.03*np.sin(x), method="odr", loss="huber", f_scale=.5,
                       sx=.05, sy=sy, absolute_sigma=True, bounds=([0, 0], [2, 4]), max_evaluations=300)
    sy[:] = 999  # The fitting result owns its uncertainty snapshot.
    assert result.sy.max() < 1
    assert_allclose(result.sx, .05)
    bootstrap = bootstrap_bands(result, [1, 3, 5], resamples=25, seed=2)
    assert bootstrap.successful >= 23
    assert np.all(bootstrap.parameters[:, 0] <= 2)
    assert np.all(np.isfinite(bootstrap.bands()))
    assert result.absolute_sigma
    assert result.max_evaluations == 300


def test_bootstrap_can_leave_zero_boundary_when_resampled_optimum_is_interior():
    x = np.arange(8.)
    y = np.array([-1., -1., -1., -1., 1., 1., 1., .8])
    result = fit_curve(x, y, standard_model("constant"), bounds=(0, 10))
    assert "c" in result.active_bounds
    bootstrap = bootstrap_bands(result, [0], resamples=40, seed=42)
    rng = np.random.default_rng(np.random.SeedSequence(42).spawn(2)[0])
    # A bounded constant's exact least-squares solution is its clamped mean.
    expected = [max(0, y[rng.integers(0, len(y), size=len(y))].mean()) for _ in range(40)]
    assert_allclose(bootstrap.parameters[:, 0], expected, atol=1e-4)  # Optimizer tolerance near zero.


def test_prediction_scale_reuses_same_confidence_distribution():
    x = np.linspace(0, 5, 40)
    result = fit_curve(x, 2*x+1+np.sin(x), sy=np.linspace(.5, 2, len(x)))
    one = bootstrap_bands(result, [1, 2, 3], resamples=30, seed=17, prediction_sigma=1)
    two = bootstrap_bands(result, [1, 2, 3], resamples=30, seed=17, prediction_sigma=2)
    assert_allclose(one.curve_samples, two.curve_samples)
    assert_allclose(two.prediction_samples-two.curve_samples,
                    2*(one.prediction_samples-one.curve_samples), atol=1e-12)
    assert two.prediction_noise == "standardized_residuals"


def test_degenerate_repeats_are_counted_without_silent_retries():
    x = np.array([0., 0., 1., 1.])
    result = fit_curve(x, [1., 1.2, 2.9, 3.1])
    updates = []
    bootstrap = bootstrap_bands(result, [0, 1], resamples=100, seed=11, min_success_fraction=.5,
                                progress=lambda done, total: updates.append((done, total)))
    rng = np.random.default_rng(np.random.SeedSequence(11).spawn(2)[0])
    failures = sum(np.ptp(x[rng.integers(0, len(x), size=len(x))]) == 0 for _ in range(100))
    assert 0 < failures < 50
    assert bootstrap.failures == failures
    assert bootstrap.successful == 100-failures
    assert bootstrap.failure_examples
    assert updates[-1] == (100, 100)
    with pytest.raises(ValueError, match="Недостаточно"):
        bootstrap_bands(result, [0, 1], resamples=100, seed=11, min_success_fraction=1)


def test_cancellation_does_not_publish_partial_results():
    result = fit_curve(np.arange(10.), 2*np.arange(10.)+np.sin(np.arange(10.)))
    with pytest.raises(BootstrapCancelled):
        bootstrap_bands(result, [1, 2], cancelled=lambda: True)
    event = Event()
    updates = []

    def progress(done, total):
        updates.append(done)
        if done == 3:
            event.set()

    with pytest.raises(BootstrapCancelled):
        bootstrap_bands(result, [1, 2], resamples=30, progress=progress, cancelled=event.is_set)
    assert updates == [1, 2, 3]


def test_confidence_changes_are_cached_and_noiseless_data_collapse():
    result = fit_curve(np.arange(20.), np.ones(20)*5)
    bootstrap = bootstrap_bands(result, [0, 10, 19], resamples=30)
    assert_allclose(bootstrap.bands(), np.ones((5, 3))*5, atol=1e-12)
    assert bootstrap.diagnostics(confidence=.99)["warnings"]
    with pytest.raises(ValueError):
        bootstrap.bands(confidence=1)


@pytest.mark.parametrize("kwargs", [{"resamples": 19}, {"resamples": True}, {"seed": -1},
                                    {"min_success_fraction": .4}, {"min_success_fraction": np.nan},
                                    {"prediction_sigma": 0}, {"prediction_sigma": [1, 2, 3]}])
def test_invalid_settings(kwargs):
    result = fit_curve(np.arange(8.), np.arange(8.)+np.sin(np.arange(8.)))
    with pytest.raises(ValueError):
        bootstrap_bands(result, [1, 2], **kwargs)
