"""Compact shared CV plans: independent rows, disjoint groups, or past-to-future."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

NESTED_STRATEGIES = {"nested": "kfold", "nested_group": "group", "nested_time": "time"}


@dataclass
class ValidationPlan:
    fold_ids: np.ndarray
    strategy: str
    time_rank: np.ndarray | None = None
    gap: int = 0
    group_codes: np.ndarray | None = None
    time_starts: np.ndarray | None = None

    def split(self, fold):
        test = self.fold_ids == fold
        if NESTED_STRATEGIES.get(self.strategy, self.strategy) == "time":
            first = self.time_starts[fold] if self.time_starts is not None else np.min(self.time_rank[test])
            train = self.time_rank < first-self.gap
        else:
            train = ~test
        return train, test


def prediction_metrics(y, predicted):
    """Unweighted vertical prediction errors in the original Y units."""
    residual = y-predicted
    with np.errstate(over="ignore", invalid="ignore"):
        mse = float(np.mean(residual**2))
        total = float(np.sum((y-y.mean())**2))
    if not np.isfinite(mse) or not np.isfinite(total):
        raise ValueError("Численное переполнение; измените масштаб данных или начальные параметры")
    with np.errstate(over="ignore", invalid="ignore"):
        r2 = 1-float(residual@residual)/total if total > 0 else float("nan")
    return {"RMSE": float(np.sqrt(mse)), "MAE": float(np.mean(np.abs(residual))),
            "MedAE": float(np.median(np.abs(residual))),
            "R2": r2 if np.isfinite(r2) else float("nan")}


def holdout_masks(values):
    """Parse explicit roles. Empty cells leave observations out of both sets."""
    labels = np.asarray([value.strip().casefold() if isinstance(value, str) else value
                         for value in values], dtype=object)
    labels[pd.isna(labels)] = ""
    train = np.isin(labels, ("train", "обучение"))
    test = np.isin(labels, ("test", "тест"))
    empty = labels == ""
    if not np.all(train | test | empty):
        raise ValueError("В колонке ролей используйте «Обучение» / Train или «Тест» / Test")
    return train, test


def factor_groups(groups, size, *, error):
    values = np.asarray(groups, dtype=object)
    if (values.shape != (size,) or np.any(pd.isna(values))
            or any(isinstance(value, str) and not value.strip() for value in values)):
        raise ValueError(error)
    try:
        codes, names = pd.factorize(values, sort=False)
    except TypeError as exc:
        raise ValueError(error) from exc
    return codes, len(names)


def validation_plan(size, folds, *, strategy="kfold", shuffle=True, seed=0, groups=None, times=None, gap=0):
    if strategy == "loo":
        if size < 2:
            raise ValueError("Для LOO нужны хотя бы две точки")
        return ValidationPlan(np.arange(size), strategy)
    if type(folds) is not int or not 2 <= folds <= size:
        raise ValueError("Число блоков CV должно быть от 2 до числа точек")
    if type(gap) is not int or gap < 0:
        raise ValueError("Разрыв времени должен быть неотрицательным целым")
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed должен быть неотрицательным целым числом")
    rng = np.random.default_rng(seed)
    base_strategy = NESTED_STRATEGIES.get(strategy, strategy)
    if base_strategy == "kfold":
        ids = np.empty(size, dtype=int)
        indices = rng.permutation(size) if shuffle else np.arange(size)
        for fold, part in enumerate(np.array_split(indices, folds)):
            ids[part] = fold
        return ValidationPlan(ids, strategy)
    if base_strategy == "group":
        codes, group_count = factor_groups(groups, size, error="Для групповой CV нужны непустые группы для каждой строки")
        if group_count < folds:
            raise ValueError("Число групп должно быть не меньше числа блоков CV")
        counts = np.bincount(codes)
        order = rng.permutation(group_count) if shuffle else np.arange(group_count)
        order = order[np.argsort(-counts[order], kind="stable")]
        loads, assignments = np.zeros(folds, int), np.empty(group_count, int)
        for group in order:
            fold = int(np.argmin(loads))
            assignments[group] = fold
            loads[fold] += counts[group]
        return ValidationPlan(assignments[codes], strategy, group_codes=codes if strategy in NESTED_STRATEGIES else None)
    if base_strategy == "time":
        values = np.asarray(times)
        if values.shape != (size,) or np.any(pd.isna(values)):
            raise ValueError("Для временной CV нужно время каждой строки")
        if values.dtype.kind not in "iufM":
            numeric = pd.to_numeric(values, errors="coerce")
            values = (np.asarray(numeric) if np.all(np.isfinite(numeric)) else
                      pd.to_datetime(values, format="mixed", errors="raise", utc=True).to_numpy(dtype="datetime64[ns]"))
        if values.dtype.kind in "iuf" and not np.all(np.isfinite(values)):
            raise ValueError("Временные координаты должны быть конечными")
        unique, ranks = np.unique(values, return_inverse=True)
        if len(unique) < folds+1:
            raise ValueError("Для временной CV нужно не меньше K+1 различных моментов времени")
        fold_by_rank = np.full(len(unique), -1, dtype=int)
        parts = np.array_split(np.arange(len(unique)), folds+1)[1:]
        starts = np.array([part[0] for part in parts])
        if starts[0] <= gap:
            raise ValueError("Разрыв времени оставляет пустой обучающий блок")
        for fold, part in enumerate(parts):
            fold_by_rank[part] = fold
        return ValidationPlan(fold_by_rank[ranks], strategy, ranks, gap, time_starts=starts)
    raise ValueError("Неизвестная стратегия кросс-валидации")
