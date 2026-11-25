"""Reproducible K-fold comparisons on identical held-out observations.

Each fit sees only its training fold, including automatic initialization and
uncertainties. Scores measure unweighted vertical prediction error at original X,
also for ODR. Group splits and forward-in-time validation use the same CV plan.
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .curve_fitting import CurveModel, fit_curve
from .curve_bootstrap import BootstrapCancelled, paired_mean_interval
from .curve_validation import NESTED_STRATEGIES, validation_plan, ValidationPlan
from .curve_validation import prediction_metrics as _prediction_metrics


class ComparisonCancelled(RuntimeError):
    """Discard an obsolete comparison between fits, without publishing partial ranks."""


@dataclass(frozen=True)
class FitCandidate:
    name: str
    model: CurveModel
    initial: Sequence[float] | None = None
    bounds: tuple = (-np.inf, np.inf)


@dataclass
class ModelScore:
    name: str
    metrics: dict[str, float]
    fold_metrics: list[dict[str, float]]
    predictions: np.ndarray | None
    errors: tuple[str, ...]


@dataclass
class TestPrediction:
    name: str
    parameters: np.ndarray | None = None
    predictions: np.ndarray | None = None
    error: str | None = None


@dataclass
class HoldoutPair:
    models: tuple[TestPrediction, TestPrediction]
    absolute_errors: np.ndarray  # Two rows, in A/B order.
    difference: np.ndarray       # B minus A; positive favors A.
    metrics: dict[str, float | int]
    interval: dict | None = None
    interval_error: str | None = None


@dataclass
class HoldoutResult:
    x: np.ndarray
    y: np.ndarray
    winner: str | None = None
    parameters: np.ndarray | None = None
    predictions: np.ndarray | None = None
    metrics: dict[str, float] | None = None
    error: str | None = None
    pair: HoldoutPair | None = None


@dataclass
class NestedFold:
    fold: int
    train_n: int
    test_n: int
    inner_seed: int
    winner: str | None = None
    inner_rmse: float | None = None
    parameters: list[float] | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    train_groups: int | None = None
    test_groups: int | None = None
    train_periods: int | None = None
    test_periods: int | None = None
    gap_rows: int | None = None


@dataclass
class NestedResult:
    inner_folds: int
    folds: list[NestedFold]
    predictions: np.ndarray
    metrics: dict[str, float]


@dataclass
class ComparisonResult:
    scores: list[ModelScore]
    fold_ids: np.ndarray
    folds: int
    shuffle: bool
    seed: int
    plan: ValidationPlan | None = None
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    holdout: HoldoutResult | None = None
    nested: NestedResult | None = None

    @property
    def ranking(self):
        """Only candidates evaluated successfully on every fold can be ranked."""
        return sorted((s for s in self.scores if not s.errors), key=lambda s: s.metrics["RMSE"])


def _compare_leave_one_out(x, y, candidates, plan, seed, cancelled, **fit_options):
    if fit_options["method"] != "ols" or fit_options["loss"] != "linear":
        raise ValueError("Для сравнения LOO выберите OLS/WLS и квадратичную функцию потерь")
    scores = []
    for candidate in candidates:
        if cancelled is not None and cancelled():
            raise ComparisonCancelled("Сравнение отменено")
        try:
            if not candidate.model.linear or any(np.any(np.isfinite(bound)) for bound in candidate.bounds):
                raise ValueError("Для LOO нужна линейная по параметрам модель без ограничений")
            result = fit_curve(x, y, candidate.model, initial=candidate.initial, bounds=candidate.bounds, **fit_options)
            predicted = result.leave_one_out()
            valid = np.isfinite(predicted)
            if not np.all(valid):
                raise ValueError(f"Прогнозы LOO: {int(valid.sum())} из {len(y)}. Для сводки нужны все прогнозы.")
            metrics = _prediction_metrics(y, predicted)
            errors = y-predicted
            absolute_errors = np.abs(errors)
            metrics.update(PRESS=float(errors @ errors), N_evaluated=len(y),
                           RMSE_fold_mean=float(absolute_errors.mean()),
                           RMSE_fold_std=float(absolute_errors.std(ddof=1)))
            # Predictions and fold IDs describe all N folds without N dictionaries.
            scores.append(ModelScore(candidate.name, metrics, [], predicted, ()))
        except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
            scores.append(ModelScore(candidate.name, {}, [], None, (str(exc),)))
    if cancelled is not None and cancelled():
        raise ComparisonCancelled("Сравнение отменено")
    return ComparisonResult(scores, plan.fold_ids, len(y), False, seed, plan, x.copy(), y.copy())


def _compare_test_predictions(models, observed):
    errors = np.full((2, len(observed)), np.nan)
    for error, model in zip(errors, models):
        if model.predictions is not None:
            with np.errstate(over="ignore"):
                np.abs(observed-model.predictions, out=error)
            error[~np.isfinite(error)] = np.nan
    difference = errors[1]-errors[0]
    common = np.isfinite(difference)
    count = int(common.sum())
    metrics = {"N_common": count, "N_test": len(observed)}
    if count:
        delta = difference[common]
        metrics.update(MAE_A=float(np.sum(errors[0, common]/count)),
                       MAE_B=float(np.sum(errors[1, common]/count)),
                       mean_difference=float(np.sum(delta/count)),
                       wins_A=int(np.count_nonzero(delta > 0)),
                       wins_B=int(np.count_nonzero(delta < 0)),
                       ties=int(np.count_nonzero(delta == 0)))
    return HoldoutPair(models, errors, difference, metrics)


def _predict_test_rows(fitted, x):
    valid = x > 0 if fitted.model.positive_x else np.ones(len(x), dtype=bool)
    values = np.full_like(x, np.nan)
    if np.any(valid):
        values[valid] = fitted.model.evaluate(x[valid], fitted.parameters, allow_nonfinite=True)
    values[~np.isfinite(values)] = np.nan
    return values


def _score_test_winner(holdout, fitted):
    holdout.parameters = fitted.parameters
    try:
        holdout.predictions = fitted.predict(holdout.x)
        holdout.metrics = _prediction_metrics(holdout.y, holdout.predictions)
    except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
        holdout.error = str(exc)


def _evaluate_holdout(comparison, candidates, test, pair, pair_bootstrap, cancelled, **fit_options):
    if test is None:
        return comparison
    holdout = HoldoutResult(*test)
    comparison.holdout = holdout
    ranking = comparison.ranking
    holdout.winner = ranking[0].name if ranking else None
    names = tuple(dict.fromkeys(name for name in (holdout.winner, *(pair or ())) if name is not None))
    by_name = {candidate.name: candidate for candidate in candidates}
    paired_models = {}
    for name in names:
        if cancelled is not None and cancelled():
            raise ComparisonCancelled("Сравнение отменено")
        candidate = by_name[name]
        prediction = TestPrediction(name)
        is_winner = name == holdout.winner
        in_pair = pair is not None and name in pair
        try:
            fitted = fit_curve(comparison.x, comparison.y, candidate.model, initial=candidate.initial,
                               bounds=candidate.bounds, **fit_options)
        except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
            prediction.error = str(exc)
            if is_winner:
                holdout.error = prediction.error
        else:
            prediction.parameters = fitted.parameters
            if is_winner:
                _score_test_winner(holdout, fitted)
            if in_pair:
                try:
                    prediction.predictions = (holdout.predictions if is_winner and holdout.predictions is not None
                                              else _predict_test_rows(fitted, holdout.x))
                except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
                    prediction.error = str(exc)
        if in_pair:
            paired_models[name] = prediction
    if cancelled is not None and cancelled():
        raise ComparisonCancelled("Сравнение отменено")
    if pair is not None:
        holdout.pair = _compare_test_predictions((paired_models[pair[0]], paired_models[pair[1]]), holdout.y)
        if pair_bootstrap is not None:
            try:
                holdout.pair.interval = paired_mean_interval(holdout.pair.difference, holdout.x,
                                                            cancelled=cancelled, **pair_bootstrap)
            except BootstrapCancelled as exc:
                raise ComparisonCancelled("Сравнение отменено") from exc
            except ValueError as exc:
                holdout.pair.interval_error = str(exc)
    return comparison


def _compare_nested(x, y, candidates, plan, folds, inner_folds, shuffle, seed, sx, sy, cancelled, **fit_options):
    strategy = NESTED_STRATEGIES[plan.strategy]
    shuffle = bool(shuffle and strategy != "time")
    if type(inner_folds) is not int or inner_folds < 2:
        raise ValueError("Число внутренних блоков должно быть от 2 до размера внешней обучающей выборки")
    if strategy == "kfold":
        smallest_train = len(x) - (len(x)+folds-1)//folds
        if inner_folds > smallest_train:
            raise ValueError("Число внутренних блоков должно быть от 2 до размера внешней обучающей выборки")
        smallest_inner_train = smallest_train - (smallest_train+inner_folds-1)//inner_folds
        if smallest_inner_train <= max(len(candidate.model.parameters) for candidate in candidates):
            raise ValueError("В обучающем блоке CV должно быть больше точек, чем параметров каждой модели")
    by_name = {candidate.name: candidate for candidate in candidates}
    predictions = np.full_like(y, np.nan)
    records = []
    for fold in range(folds):
        if cancelled is not None and cancelled():
            raise ComparisonCancelled("Сравнение отменено")
        train, test = plan.split(fold)
        record = NestedFold(fold+1, int(train.sum()), int(test.sum()), seed+fold+1)
        records.append(record)
        train_x, train_y = x[train], y[train]
        options = dict(fit_options, sx=sx[train] if sx is not None else None,
                       sy=sy[train] if sy is not None else None)
        split_options = {}
        if plan.group_codes is not None:
            train_groups = plan.group_codes[train]
            record.train_groups = int(np.count_nonzero(np.bincount(train_groups)))
            record.test_groups = int(np.count_nonzero(np.bincount(plan.group_codes[test])))
            split_options["groups"] = train_groups
        elif plan.time_rank is not None:
            first_test = int(plan.time_starts[fold])
            train_periods = first_test-plan.gap
            train_times = plan.time_rank[train]
            record.train_periods = train_periods
            record.test_periods = int(plan.time_rank[test].max())-first_test+1
            record.gap_rows = int(np.count_nonzero((plan.time_rank >= train_periods) & (plan.time_rank < first_test)))
            split_options.update(times=train_times, gap=plan.gap)
        try:
            inner = compare_models(train_x, train_y, candidates, folds=inner_folds, shuffle=shuffle,
                                   seed=record.inner_seed, strategy=strategy, cancelled=cancelled,
                                   **split_options, **options)
            ranking = inner.ranking
            if not ranking:
                raise ValueError("Ни одна модель не прошла внутреннюю CV")
            record.winner = ranking[0].name
            record.inner_rmse = ranking[0].metrics["RMSE"]
            winner = by_name[record.winner]
            fitted = fit_curve(train_x, train_y, winner.model, initial=winner.initial, bounds=winner.bounds, **options)
            record.parameters = fitted.parameters.tolist()
            predicted = fitted.predict(x[test])
            record.metrics = _prediction_metrics(y[test], predicted)
            predictions[test] = predicted
        except ComparisonCancelled:
            raise
        except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
            record.error = str(exc)
    if cancelled is not None and cancelled():
        raise ComparisonCancelled("Сравнение отменено")
    metrics = {}
    if not any(record.error for record in records):
        evaluated = plan.fold_ids >= 0
        metrics = _prediction_metrics(y[evaluated], predictions[evaluated])
        metrics["N_evaluated"] = int(evaluated.sum())
    nested = NestedResult(inner_folds, records, predictions, metrics)
    return ComparisonResult([], plan.fold_ids, folds, bool(shuffle), seed, plan, x.copy(), y.copy(), nested=nested)


def compare_models(x, y, candidates, *, folds=5, shuffle=True, seed=0,
                   method="ols", loss="linear", f_scale=1., sx=None, sy=None,
                   absolute_sigma=False, max_evaluations=1000, strategy="kfold", groups=None,
                   times=None, gap=0, starts=1, start_seed=0, cancelled=None, holdout=None,
                   holdout_pair=None, pair_bootstrap=None, inner_folds=4) -> ComparisonResult:
    """Evaluate candidates on common validation splits and rank by pooled RMSE.

    Failed folds are reported explicitly; a partly successful model gets neither
    aggregate scores nor ranking. predictions are in original observation order.
    Fold RMSE dispersion describes split variability, not a confidence interval.
    With holdout=(test_x, test_y), the CV winner is refitted and tested. An optional
    holdout_pair=(name_a, name_b) compares two candidates on the same test rows.
    Pass pair_bootstrap={} or resampling options to calculate a paired ΔMAE interval.
    Nested strategies select a model by inner CV in each outer training split.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if (x.ndim != 1 or y.shape != x.shape or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(y))):
        raise ValueError("Для сравнения нужны конечные пары X/Y")
    if strategy != "loo" and (not isinstance(folds, int) or not 2 <= folds <= len(x)):
        raise ValueError("Число блоков CV должно быть от 2 до числа точек")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed должен быть неотрицательным целым числом")
    candidates = list(candidates)
    if not candidates or len({c.name for c in candidates}) != len(candidates):
        raise ValueError("Добавьте модели с уникальными названиями для сравнения")
    if strategy == "nested" and any(value is not None for value in (holdout, holdout_pair, groups, times)):
        raise ValueError("Вложенная CV доступна для независимых строк без отдельной тестовой выборки")
    if strategy == "nested_group" and any(value is not None for value in (holdout, holdout_pair, times)):
        raise ValueError("Для вложенной CV по группам укажите группы без отдельной тестовой выборки")
    if strategy == "nested_time" and any(value is not None for value in (holdout, holdout_pair, groups)):
        raise ValueError("Для вложенной временной CV укажите время без отдельной тестовой выборки")
    if holdout_pair is not None:
        names = {candidate.name for candidate in candidates}
        if (holdout is None or len(holdout_pair) != 2 or holdout_pair[0] == holdout_pair[1]
                or any(name not in names for name in holdout_pair)):
            raise ValueError("Для попарного теста выберите две разные модели и тестовую выборку")
    if holdout is not None:
        test_x, test_y = (np.array(values, dtype=float, copy=True) for values in holdout)
        if (test_x.ndim != 1 or not len(test_x) or test_x.shape != test_y.shape
                or not np.all(np.isfinite(test_x)) or not np.all(np.isfinite(test_y))):
            raise ValueError("Для независимого теста нужны конечные пары X/Y")
        holdout = test_x, test_y
    plan = validation_plan(len(x), folds, strategy=strategy, shuffle=shuffle, seed=seed,
                           groups=groups, times=times, gap=gap)
    if strategy != "loo" and strategy not in NESTED_STRATEGIES:
        max_parameters = max(len(candidate.model.parameters) for candidate in candidates)
        if any(np.count_nonzero(plan.split(fold)[0]) <= max_parameters for fold in range(folds)):
            raise ValueError("В обучающем блоке CV должно быть больше точек, чем параметров каждой модели")

    def uncertainties(value):
        if value is None:
            return None
        values = np.broadcast_to(np.asarray(value, float), x.shape)
        if np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("Погрешности должны быть положительными и конечными")
        return values

    sx, sy = uncertainties(sx), uncertainties(sy)
    fit_options = dict(method=method, loss=loss, f_scale=f_scale, absolute_sigma=absolute_sigma,
                       max_evaluations=max_evaluations, starts=starts, start_seed=start_seed)
    if strategy in NESTED_STRATEGIES:
        return _compare_nested(x, y, candidates, plan, folds, inner_folds, shuffle, seed, sx, sy, cancelled, **fit_options)
    if strategy == "loo":
        comparison = _compare_leave_one_out(x, y, candidates, plan, seed, cancelled, sx=sx, sy=sy, **fit_options)
        return _evaluate_holdout(comparison, candidates, holdout, holdout_pair, pair_bootstrap, cancelled,
                                 sx=sx, sy=sy, **fit_options)
    fold_ids = plan.fold_ids
    scores = []
    for candidate in candidates:
        predicted = np.full_like(y, np.nan)
        fold_metrics, errors = [], []
        for fold in range(folds):
            if cancelled is not None and cancelled():
                raise ComparisonCancelled("Сравнение отменено")
            train, test = plan.split(fold)
            try:
                result = fit_curve(x[train], y[train], candidate.model, initial=candidate.initial, bounds=candidate.bounds,
                                   sx=sx[train] if sx is not None else None,
                                   sy=sy[train] if sy is not None else None, **fit_options)
                predicted[test] = result.predict(x[test])
                fold_metrics.append({"fold": fold+1, "train_n": int(np.sum(train)), "test_n": int(np.sum(test)),
                                     **_prediction_metrics(y[test], predicted[test])})
            except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
                errors.append(f"Fold {fold+1}: {exc}")
        if errors:
            scores.append(ModelScore(candidate.name, {}, fold_metrics, None, tuple(errors)))
        else:
            evaluated = fold_ids >= 0
            metrics = _prediction_metrics(y[evaluated], predicted[evaluated])
            metrics["N_evaluated"] = int(np.sum(evaluated))
            metrics["RMSE_fold_mean"] = float(np.mean([f["RMSE"] for f in fold_metrics]))
            metrics["RMSE_fold_std"] = float(np.std([f["RMSE"] for f in fold_metrics], ddof=1))
            scores.append(ModelScore(candidate.name, metrics, fold_metrics, predicted, ()))
    if cancelled is not None and cancelled():
        raise ComparisonCancelled("Сравнение отменено")
    comparison = ComparisonResult(scores, fold_ids, folds, bool(shuffle and strategy != "time"), seed, plan, x.copy(), y.copy())
    return _evaluate_holdout(comparison, candidates, holdout, holdout_pair, pair_bootstrap, cancelled,
                             sx=sx, sy=sy, **fit_options)
