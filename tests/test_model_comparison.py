import numpy as np
import pytest
from numpy.testing import assert_allclose

from autopyro.curve_fitting import custom_model, standard_model
from autopyro.model_comparison import FitCandidate, compare_models


def test_oof_predictions_match_independent_training_fits():
    x = np.linspace(0, 10, 53)
    y = 2*x+1+np.sin(3*x)
    result = compare_models(x, y, [FitCandidate("line", standard_model())], seed=123)
    score = result.ranking[0]
    assert np.bincount(result.fold_ids).tolist() == [11, 11, 11, 10, 10]
    for fold in range(5):
        test = result.fold_ids == fold
        slope, intercept = np.polyfit(x[~test], y[~test], 1)
        assert_allclose(score.predictions[test], slope*x[test]+intercept)
    error = y-score.predictions
    assert score.metrics["RMSE"] == pytest.approx(np.sqrt(np.mean(error**2)))
    assert score.metrics["MAE"] == pytest.approx(np.mean(np.abs(error)))
    assert score.metrics["R2"] == pytest.approx(1-np.sum(error**2)/np.sum((y-y.mean())**2))
    repeated = compare_models(x, y, [FitCandidate("line", standard_model())], seed=123)
    assert_allclose(repeated.fold_ids, result.fold_ids)
    assert_allclose(repeated.ranking[0].predictions, score.predictions)


def test_weighted_training_uses_only_training_uncertainties():
    x = np.arange(23.)
    y = 2*x+1+np.cos(x)
    sy = np.linspace(.1, 2, len(x))
    result = compare_models(x, y, [FitCandidate("line", standard_model())], sy=sy, folds=4)
    for fold in range(4):
        test = result.fold_ids == fold
        design = np.column_stack([x[~test], np.ones(np.sum(~test))])/sy[~test, None]
        beta = np.linalg.lstsq(design, y[~test]/sy[~test], rcond=None)[0]
        assert_allclose(result.scores[0].predictions[test], beta[0]*x[test]+beta[1])


def test_nested_group_cv_keeps_groups_whole_at_both_levels(monkeypatch):
    from autopyro.curves import model_comparison as module

    groups = np.repeat(np.arange(9), np.arange(2, 11))
    x = np.linspace(1, 10, len(groups))
    y = 1+x+.2*x*x
    candidates = [FitCandidate("line", standard_model()),
                  FitCandidate("quadratic", standard_model("polynomial", degree=2))]
    plans = []
    make_plan = module.validation_plan

    def capture_plan(size, folds, **options):
        plan = make_plan(size, folds, **options)
        plans.append((plan, options["groups"]))
        for fold in range(folds):
            train, test = plan.split(fold)
            assert not set(options["groups"][train]) & set(options["groups"][test])
        return plan

    monkeypatch.setattr(module, "validation_plan", capture_plan)
    options = dict(strategy="nested_group", groups=groups, folds=3, inner_folds=3, seed=6)
    result = compare_models(x, y, candidates, **options)
    assert len(plans) == 4 and result.nested.metrics["RMSE"] < 1e-10
    for fold, ((_, inner_groups), record) in enumerate(zip(plans[1:], result.nested.folds)):
        train, test = result.plan.split(fold)
        assert_allclose(inner_groups, result.plan.group_codes[train])
        assert record.train_groups == len(set(groups[train]))
        assert record.test_groups == len(set(groups[test]))
    changed = y.copy()
    changed[result.fold_ids == 0] += 100
    other = compare_models(x, changed, candidates, **options)
    first, repeated = result.nested.folds[0], other.nested.folds[0]
    assert first.winner == repeated.winner
    assert_allclose(first.parameters, repeated.parameters)
    assert repeated.metrics["RMSE"] == pytest.approx(100)


def test_nested_time_cv_enforces_chronology_and_gap_at_both_levels(monkeypatch):
    from autopyro.curves import model_comparison as module

    x = np.random.default_rng(5).permutation(144).astype(float)
    periods = x.astype(int)//2
    times = (np.datetime64("2020-01-01") + periods.astype("timedelta64[D]")).astype(str)
    y = 1+x+.02*x*x
    candidates = [FitCandidate("line", standard_model()),
                  FitCandidate("quadratic", standard_model("polynomial", degree=2))]
    plans = []
    make_plan = module.validation_plan

    def capture_plan(size, folds, **options):
        plan = make_plan(size, folds, **options)
        plans.append((plan, options["times"]))
        for fold in range(folds):
            train, test = plan.split(fold)
            assert plan.time_rank[test].min() - plan.time_rank[train].max() > options["gap"]
        return plan

    monkeypatch.setattr(module, "validation_plan", capture_plan)
    options = dict(strategy="nested_time", times=times, folds=3, inner_folds=2, gap=2)
    result = compare_models(x, y, candidates, **options)
    evaluated = result.fold_ids >= 0
    assert len(plans) == 4 and not result.shuffle
    assert result.nested.metrics["N_evaluated"] == 108
    assert np.all(np.isnan(result.nested.predictions[~evaluated]))
    assert_allclose(result.nested.predictions[evaluated], y[evaluated], atol=1e-9)
    for fold, ((_, inner_times), record) in enumerate(zip(plans[1:], result.nested.folds)):
        train, test = result.plan.split(fold)
        assert_allclose(inner_times, periods[train])
        assert record.train_periods == len(set(periods[train]))
        assert record.test_periods == len(set(periods[test]))
        assert record.gap_rows == 4
    changed = y.copy()
    first_test = result.fold_ids == 0
    changed[first_test] += 100
    other = compare_models(x, changed, candidates, **options)
    first, repeated = result.nested.folds[0], other.nested.folds[0]
    assert first.winner == repeated.winner == "quadratic"
    assert_allclose(first.parameters, repeated.parameters)
    assert repeated.metrics["RMSE"] == pytest.approx(100)


def test_nested_cv_keeps_outer_y_out_of_selection_and_fitting():
    x = np.linspace(1, 10, 36)
    y = 1+2*x+.2*x*x+.03*np.sin(x)
    sy = .2+x/20
    candidates = [FitCandidate("line", standard_model()),
                  FitCandidate("quadratic", standard_model("polynomial", degree=2))]
    options = dict(strategy="nested", folds=3, inner_folds=3, shuffle=False, sy=sy)
    baseline = compare_models(x, y, candidates, **options)
    outer_test = baseline.fold_ids == 0
    changed = y.copy()
    changed[outer_test] += 100
    other = compare_models(x, changed, candidates, **options)
    first, repeated = baseline.nested.folds[0], other.nested.folds[0]
    assert first.winner == repeated.winner == "quadratic"
    assert first.inner_rmse == repeated.inner_rmse
    assert_allclose(first.parameters, repeated.parameters)
    assert_allclose(baseline.nested.predictions[outer_test], other.nested.predictions[outer_test])
    expected = np.polynomial.polynomial.polyfit(x[~outer_test], y[~outer_test], 2, w=1/sy[~outer_test])
    assert_allclose(first.parameters, expected)
    predicted = np.polynomial.polynomial.polyval(x[outer_test], expected)
    assert first.metrics["RMSE"] == pytest.approx(np.sqrt(np.mean((y[outer_test]-predicted)**2)))
    assert repeated.metrics["RMSE"] == pytest.approx(np.sqrt(np.mean((changed[outer_test]-predicted)**2)))
    assert baseline.nested.metrics["RMSE"] == pytest.approx(np.sqrt(np.mean((y-baseline.nested.predictions)**2)))


def test_nested_cv_preserves_outer_errors_and_rejects_independent_test():
    x = np.r_[-np.arange(1., 7.), np.arange(1., 13.)]
    candidate = FitCandidate("log", standard_model("logarithmic"))
    options = dict(strategy="nested", folds=3, inner_folds=3, shuffle=False)
    result = compare_models(x, x, [candidate], **options)
    assert result.nested.folds[0].winner == "log"
    assert result.nested.folds[0].error
    assert np.all(np.isnan(result.nested.predictions[:6]))
    assert result.nested.metrics == {}
    with pytest.raises(ValueError, match="без отдельной тестовой выборки"):
        compare_models(x, x, [candidate], holdout=([2.], [2.]), **options)


def test_holdout_y_does_not_change_model_selection_or_training_coefficients():
    x = np.linspace(0, 10, 30)
    y = 1+2*x+.2*x**2+.1*np.sin(x)
    test_x = np.array([2., 5., 8.])
    test_y = 1+2*test_x+.2*test_x**2
    candidates = [FitCandidate("line", standard_model()),
                  FitCandidate("quadratic", standard_model("polynomial", degree=2))]
    first = compare_models(x, y, candidates, strategy="loo", holdout=(test_x, test_y))
    second = compare_models(x, y, candidates, strategy="loo", holdout=(test_x, test_y+10))
    assert first.holdout.winner == second.holdout.winner == "quadratic"
    assert [score.metrics for score in first.ranking] == [score.metrics for score in second.ranking]
    assert_allclose(first.holdout.parameters, np.polynomial.polynomial.polyfit(x, y, 2))
    assert_allclose(first.holdout.parameters, second.holdout.parameters)
    assert_allclose(first.holdout.predictions, second.holdout.predictions)
    assert first.holdout.metrics["RMSE"] < .1 and second.holdout.metrics["RMSE"] > 9


def test_paired_holdout_uses_common_predictions_and_reuses_the_winner_fit(monkeypatch):
    import autopyro.model_comparison as module

    x = np.arange(1., 20.)
    y = np.log(x)
    test_x = np.array([-2., 1., 4., 8.])
    test_y = np.array([12., 0., 1., 3.])
    models = [FitCandidate("log", standard_model("logarithmic")), FitCandidate("line", standard_model())]
    fit_curve = module.fit_curve
    calls = []

    def counted_fit(x, y, model, **options):
        calls.append(model.name)
        return fit_curve(x, y, model, **options)

    monkeypatch.setattr(module, "fit_curve", counted_fit)
    result = compare_models(x, y, models, strategy="loo", holdout=(test_x, test_y), holdout_pair=("log", "line"),
                            pair_bootstrap={"resamples": 40})
    assert calls.count("logarithmic") == calls.count("linear") == 2
    assert result.ranking[0].name == result.holdout.winner == "log"
    assert result.holdout.error  # The separate winner test still requires all predictions.
    pair = result.holdout.pair
    first, second = pair.models
    assert_allclose(first.predictions, np.r_[np.nan, np.log(test_x[1:])], atol=1e-12)
    slope, intercept = np.polyfit(x, y, 1)
    expected_b = slope*test_x+intercept
    assert_allclose(second.predictions, expected_b)
    expected_delta = np.abs(test_y[1:]-expected_b[1:])-np.abs(test_y[1:]-np.log(test_x[1:]))
    assert_allclose(pair.difference, np.r_[np.nan, expected_delta], atol=1e-12)
    assert pair.metrics["N_common"] == 3
    assert pair.interval["N_common"] == 3 and pair.interval["resamples"] == 40
    assert pair.metrics["mean_difference"] == pytest.approx(expected_delta.mean())


def test_holdout_prediction_failure_keeps_cv_ranking():
    x = np.arange(1., 20.)
    model = FitCandidate("log", standard_model("logarithmic"))
    result = compare_models(x, np.log(x), [model], holdout=([-1., 2.], [0., 1.]))
    assert result.ranking[0].name == result.holdout.winner == "log"
    assert result.holdout.error and result.holdout.metrics is None
    assert result.holdout.parameters is not None


def test_loo_comparison_matches_weighted_refits_and_isolates_ineligible_models(monkeypatch):
    import autopyro.model_comparison as module

    x = np.array([0., 0., 1., 1., 2.])
    y = np.array([1., 2., 4., 5., 6.])
    sy = np.array([.8, .5, .6, 1., 1.2])
    calls = []
    fit_curve = module.fit_curve

    def counted_fit(x, y, model, **options):
        calls.append(len(x))
        return fit_curve(x, y, model, **options)

    monkeypatch.setattr(module, "fit_curve", counted_fit)
    candidates = [FitCandidate("line", standard_model()),
                  FitCandidate("quadratic", standard_model("polynomial", degree=2)),
                  FitCandidate("bounded", standard_model(), bounds=(0., np.inf))]
    result = compare_models(x, y, candidates, strategy="loo", sy=sy, folds=20)
    assert calls == [len(x), len(x)]
    assert result.folds == len(x) and not result.shuffle
    assert_allclose(result.fold_ids, np.arange(len(x)))
    assert [score.name for score in result.ranking] == ["line"]
    assert all(score.errors and score.predictions is None for score in result.scores[1:])
    score = result.ranking[0]
    expected = []
    for i in range(len(x)):
        train = np.arange(len(x)) != i
        design = np.column_stack([x[train], np.ones(train.sum())])/sy[train, None]
        slope, intercept = np.linalg.lstsq(design, y[train]/sy[train], rcond=None)[0]
        expected.append(slope*x[i]+intercept)
    assert_allclose(score.predictions, expected)
    assert score.metrics["PRESS"] == pytest.approx(np.sum((y-expected)**2))
    assert score.fold_metrics == []
    with pytest.raises(ValueError, match="квадратичную"):
        compare_models(x, y, candidates, strategy="loo", loss="huber")


def test_ranking_prefers_correct_quadratic_and_common_folds():
    x = np.linspace(-3, 3, 100)
    y = 2*x*x+x+1+.1*np.random.default_rng(42).normal(size=len(x))
    candidates = [FitCandidate("line", standard_model()),
                  FitCandidate("quadratic", standard_model("polynomial", degree=2)),
                  FitCandidate("constant", standard_model("constant"))]
    result = compare_models(x, y, candidates)
    assert result.ranking[0].name == "quadratic"
    assert all(len(s.fold_metrics) == 5 for s in result.scores)
    assert all(len(s.predictions) == len(x) for s in result.scores)


def test_failed_model_cannot_win_and_other_models_continue():
    x = np.linspace(-2, 2, 40)
    candidates = [FitCandidate("line", standard_model()), FitCandidate("invalid", standard_model("power"))]
    result = compare_models(x, 2*x+1, candidates)
    assert [s.name for s in result.ranking] == ["line"]
    assert len(result.scores[1].errors) == 5
    assert result.scores[1].predictions is None
    assert result.scores[1].metrics == {}


def test_arithmetic_failure_is_isolated_to_its_candidate():
    candidates = [FitCandidate("bad formula", custom_model("a*x + 1/0", ("a",)), [1]),
                  FitCandidate("line", standard_model())]
    result = compare_models(np.arange(20.), 2*np.arange(20.)+1, candidates)
    assert len(result.scores[0].errors) == 5
    assert [score.name for score in result.ranking] == ["line"]


def test_custom_bounded_robust_and_odr_comparison():
    x = np.linspace(1, 5, 30)
    model = custom_model("a*x+b", ("a", "b"))
    candidates = [FitCandidate("custom", model, [1, 1], ([0, 0], [4, 4])),
                  FitCandidate("line", standard_model())]
    result = compare_models(x, 2*x+1+.01*np.sin(x), candidates, folds=3,
                            method="odr", loss="huber", f_scale=.2, sx=.1, sy=.1)
    assert len(result.ranking) == 2
    assert all(s.metrics["RMSE"] < .1 for s in result.ranking)


def test_ordered_folds_and_undefined_r2():
    result = compare_models(np.arange(8.), np.ones(8), [FitCandidate("constant", standard_model("constant"))],
                            folds=4, shuffle=False)
    assert_allclose(result.fold_ids, [0, 0, 1, 1, 2, 2, 3, 3])
    assert np.isnan(result.ranking[0].metrics["R2"])


@pytest.mark.parametrize("options", [{"folds": 1}, {"folds": 30}, {"folds": 2.5}, {"seed": -1}, {"sy": 0}])
def test_invalid_cv_settings(options):
    with pytest.raises(ValueError):
        compare_models(np.arange(8.), np.arange(8.), [FitCandidate("line", standard_model())], **options)


def test_underdetermined_training_rejected_before_fitting():
    with pytest.raises(ValueError):
        compare_models(np.arange(8.), np.arange(8.),
                       [FitCandidate("degree6", standard_model("polynomial", degree=6))])
