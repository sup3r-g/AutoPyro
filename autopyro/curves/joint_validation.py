"""Cross-validation of sharing schemes for series present in training."""

from dataclasses import dataclass

import numpy as np

from .curve_validation import factor_groups, prediction_metrics
from .joint_fitting import fit_joint_curve


@dataclass
class JointCVScore:
    shared: tuple[str, ...]
    metrics: dict | None
    error: str = ""


def compare_joint_sharing(x, y, series, model, candidates, *, folds=5, seed=0, sy=None,
                          cancelled=None, progress=None, **fit_options):
    """Score pooled out-of-fold predictions on the same within-series splits."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    series = np.asarray(series, dtype=object)
    if (x.ndim != 1 or y.shape != x.shape or not len(x)
            or not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)))):
        raise ValueError("Для сравнения нужны конечные пары X/Y")
    if model.positive_x and np.any(x <= 0):
        raise ValueError("Некорректные координаты прогноза")
    if sy is not None:
        try:
            sy = np.broadcast_to(np.asarray(sy, dtype=float), x.shape)
        except (TypeError, ValueError) as exc:
            raise ValueError("Погрешности должны быть положительными и конечными") from exc
        if np.any(sy <= 0) or not np.all(np.isfinite(sy)):
            raise ValueError("Погрешности должны быть положительными и конечными")
    codes, count = factor_groups(series, len(x), error="Для совместной подгонки укажите серию для каждой точки")
    if type(folds) is not int or folds < 2 or count < 2 or np.any(np.bincount(codes) < folds):
        raise ValueError("Для сравнения нужно не меньше двух серий и не меньше точек в каждой серии, чем блоков CV")
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed должен быть неотрицательным целым числом")
    order = np.argsort(codes, kind="stable")
    blocks = np.split(order, np.cumsum(np.bincount(codes))[:-1])
    rng = np.random.default_rng(seed)
    assignments = np.empty(len(x), dtype=int)
    for block in blocks:
        assignments[rng.permutation(block)] = np.arange(len(block)) % folds
    scores = []
    for candidate_index, shared in enumerate(candidates):
        prediction = np.empty(len(x))
        try:
            for fold in range(folds):
                if cancelled is not None and cancelled():
                    return None
                test = assignments == fold
                # Keep the original series order for per-series bounds and starts.
                train = order[~test[order]]
                result = fit_joint_curve(x[train], y[train], series[train], model, shared=shared,
                                         sy=sy[train] if sy is not None else None, cancelled=cancelled, **fit_options)
                if result is None:
                    return None
                by_label = {label: index for index, label in enumerate(result.labels)}
                for block in blocks:
                    holdout = block[test[block]]
                    prediction[holdout] = result.predict(by_label[series[holdout[0]]], x[holdout])
                if progress is not None:
                    progress(candidate_index*folds+fold+1)
            metrics = prediction_metrics(y, prediction)
        except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
            scores.append(JointCVScore(shared, None, str(exc)))
            if progress is not None:
                progress((candidate_index+1)*folds)
        else:
            scores.append(JointCVScore(shared, metrics))
    if cancelled is not None and cancelled():
        return None
    return sorted(scores, key=lambda score: score.metrics["RMSE"] if score.metrics is not None else np.inf)
