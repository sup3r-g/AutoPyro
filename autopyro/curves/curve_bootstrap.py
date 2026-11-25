"""Paired percentile bootstrap for fitted curves, including robust/bounded fits.

Resample observations and their uncertainties together, refit, and evaluate on a
fixed grid. Prediction adds independently sampled, centered empirical Y residuals.
The intervals are pointwise approximations, conditional on exact future X.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .curve_fitting import FitResult, fit_curve

BOOTSTRAP_BAND_METHODS = ("bootstrap", "bootstrap_simultaneous")
BOOTSTRAP_TAIL_WARNING = "Мало повторов в хвостах распределения; увеличьте число bootstrap-повторов."


def _band_quantiles(samples, probabilities):
    with np.errstate(over="ignore", invalid="ignore"):
        bounds = np.quantile(samples, probabilities, axis=0)
        overflow = ~np.isfinite(bounds).all(axis=0)
        if np.any(overflow):
            # Halving prevents overflow in the difference between adjacent samples.
            bounds[:, overflow] = 2*np.quantile(samples[:, overflow]/2, probabilities, axis=0)
    if not np.all(np.isfinite(bounds)) or np.any(bounds[0] > bounds[1]):
        raise ValueError("Численное переполнение доверительной полосы bootstrap")
    return bounds


def _percentile_bands(mean, curves, predictions, confidence, max_deviations=None):
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("Уровень доверия должен быть между 0 и 1")
    alpha = (1-confidence)/2
    if max_deviations is None:
        lo, hi = _band_quantiles(curves, [alpha, 1-alpha])
    else:
        radius = np.quantile(max_deviations, confidence)
        with np.errstate(over="ignore", invalid="ignore"):
            lo, hi = mean-radius, mean+radius
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            raise ValueError("Численное переполнение доверительной полосы bootstrap")
    plo, phi = _band_quantiles(predictions, [alpha, 1-alpha])
    return mean.copy(), lo, hi, plo, phi


class BootstrapCancelled(RuntimeError):
    """Cooperative cancellation between refits; partial samples are not published."""


@dataclass
class BootstrapResult:
    grid: np.ndarray
    fitted: np.ndarray
    parameters: np.ndarray
    curve_samples: np.ndarray
    prediction_samples: np.ndarray
    resamples: int
    seed: int
    min_success_fraction: float
    failures: int
    failure_examples: tuple[str, ...]
    prediction_noise: str
    prediction_sigma: np.ndarray | None
    warnings: tuple[str, ...]
    prediction_draws: np.ndarray | None = None
    group_count: int | None = None
    block_length: int | None = None
    max_deviations: np.ndarray | None = None

    @property
    def successful(self):
        return len(self.parameters)

    def bands(self, *, confidence=.95):
        """Original fitted curve and percentile CI/PI; no model refits."""
        return _percentile_bands(self.fitted, self.curve_samples, self.prediction_samples, confidence, self.max_deviations)

    def bands_at(self, result, x, *, confidence=.95, prediction_sigma=None):
        """Evaluate stored coefficient/noise draws on another grid without refits.

        Common noise draws make duplicate X and chunked table output agree with
        the plotted bands. For a nonconstant future sigma, new X require new sigma.
        """
        grid = np.atleast_1d(np.asarray(x, float))
        if (grid.ndim != 1 or not len(grid) or not np.all(np.isfinite(grid))
                or result.model.positive_x and np.any(grid <= 0)):
            raise ValueError("Некорректные координаты прогноза")
        if prediction_sigma is not None:
            pool, scale = _prediction_residuals(result, grid, prediction_sigma)
            rng = np.random.default_rng(np.random.SeedSequence(self.seed).spawn(2)[1])
            draws = pool[rng.integers(0, len(pool), size=self.successful)]
        else:
            scale = self.prediction_sigma
            if scale is not None:
                if np.all(scale == scale[0]):
                    scale = scale[0]
                elif not np.array_equal(grid, self.grid):
                    raise ValueError("Для новой сетки задайте будущую σY в каждой точке")
            draws = self.prediction_draws
            if draws is None:
                pool, _ = _prediction_residuals(result, grid, scale)
                rng = np.random.default_rng(np.random.SeedSequence(self.seed).spawn(2)[1])
                draws = pool[rng.integers(0, len(pool), size=self.successful)]
        curves = np.array([result.model.evaluate(grid, beta) for beta in self.parameters])
        predictions = curves+draws[:, None]*(1 if scale is None else scale)
        if not np.all(np.isfinite(predictions)):
            raise ValueError("Численное переполнение bootstrap Prediction; измените масштаб данных")
        return _percentile_bands(result.predict(grid), curves, predictions, confidence, self.max_deviations)

    def diagnostics(self, *, confidence=.95):
        """Small, JSON-compatible record; sample arrays remain separate."""
        if not np.isfinite(confidence) or not 0 < confidence < 1:
            raise ValueError("Уровень доверия должен быть между 0 и 1")
        warnings = list(self.warnings)
        if self.successful*(1-confidence)/2 < 10:
            warnings.append(BOOTSTRAP_TAIL_WARNING)
        method = "paired_percentile" if self.group_count is None else "cluster_percentile"
        if self.block_length is not None:
            method = "moving_block_percentile"
        return {"method": method, "group_count": self.group_count, "block_length": self.block_length,
                "confidence_method": "grid_max_deviation" if self.max_deviations is not None else "pointwise_percentile",
                "resamples": self.resamples, "seed": self.seed,
                "algorithm_version": 2, "prediction_sampling": "common_residual_per_resample",
                "min_success_fraction": self.min_success_fraction,
                "successful": self.successful, "failures": self.failures,
                "failure_examples": list(self.failure_examples), "prediction_noise": self.prediction_noise,
                "prediction_sigma": self.prediction_sigma.tolist() if self.prediction_sigma is not None else None,
                "warnings": warnings}

    def parameter_intervals(self, model, *, confidence=.95, bounds=None):
        from .curve_constraints import parameter_samples
        self.diagnostics(confidence=confidence)
        alpha = (1-confidence)/2
        expanded = parameter_samples(model, self.parameters)
        records = []
        for name, samples in expanded.items():
            low, high = np.quantile(samples, [alpha, 1-alpha])
            frequency = None
            if bounds is not None and name in model.parameters:
                j = model.parameters.index(name)
                at_bound = np.isclose(samples, bounds[0][j], rtol=1e-5, atol=1e-8) | np.isclose(samples, bounds[1][j], rtol=1e-5, atol=1e-8)
                frequency = float(np.mean(at_bound))
            records.append({"parameter": name, "lower": float(low), "upper": float(high),
                            "median": float(np.median(samples)), "sd": float(np.std(samples, ddof=1)),
                            "bound_frequency": frequency})
        return records


def _prediction_residuals(result, grid, prediction_sigma):
    """Return a centered noise pool and an optional future Y scale per grid point."""
    scale = None
    if prediction_sigma is not None:
        scale = np.broadcast_to(np.asarray(prediction_sigma, float), grid.shape).copy()
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("Погрешность прогноза должна быть положительной и конечной")
    residuals = _center_prediction_residuals(result.y-result.fitted, result.sy, result.loss, scale is not None)
    return residuals, scale


def _center_prediction_residuals(residuals, sigma, loss, standardized):
    if standardized:
        if sigma is not None:
            residuals = residuals/sigma
        else:
            spread = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.
            if spread == 0 or not np.isfinite(spread):
                raise ValueError("Для масштаба Prediction нужен ненулевой конечный разброс остатков")
            residuals = residuals/spread
    # Robust fits estimate a resistant location; retain empirical tails, including
    # outliers, while centering at the median. Ordinary fits use mean centering.
    center = np.median(residuals) if loss != "linear" else np.mean(residuals)
    residuals = residuals-center
    if np.any(~np.isfinite(residuals)):
        raise ValueError("Нечисловые остатки при расчёте Prediction")
    return residuals


def _index_sampler(x, groups, block_length):
    """Prepare a sampler once and retain the count of resampling groups."""
    size = len(x)
    if block_length is not None:
        if groups is not None:
            raise ValueError("Выберите либо группы, либо блоки bootstrap")
        if type(block_length) is not int or not 2 <= block_length < size:
            raise ValueError("Длина блока bootstrap должна быть от 2 до числа точек минус 1")
        x_order = np.argsort(x)
        sorted_x = x[x_order]
        if np.any(sorted_x[1:] == sorted_x[:-1]):
            raise ValueError("Для блочного bootstrap значения X должны быть различны")
        offsets = np.arange(block_length)
        blocks = (size+block_length-1)//block_length

        def sample(rng):
            starts = rng.integers(0, size-block_length+1, size=blocks)
            return x_order[(starts[:, None]+offsets).ravel()[:size]]

        return sample, None
    if groups is not None:
        from .curve_validation import factor_groups

        codes, count = factor_groups(groups, size, error="Для группового bootstrap нужны непустые группы для каждой точки")
        if count < 2:
            raise ValueError("Для группового bootstrap нужны хотя бы две группы")
        rows = np.arange(size)

        def sample(rng):
            multiplicities = np.bincount(rng.integers(0, count, size=count), minlength=count)
            return np.repeat(rows, multiplicities[codes])

        return sample, count
    return lambda rng: rng.integers(0, size, size=size), None


def _validate_resampling(resamples, seed):
    if type(resamples) is not int or resamples < 20:
        raise ValueError("Bootstrap требует не менее 20 повторов")
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed должен быть неотрицательным целым числом")


def paired_mean_interval(difference, x, *, confidence=.95, resamples=500, seed=0,
                         groups=None, block_length=None, margin=0., cancelled=None):
    """Percentile interval for mean paired loss difference with fixed predictions."""
    _validate_resampling(resamples, seed)
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("Уровень доверия должен быть между 0 и 1")
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("Порог ΔMAE должен быть конечным и неотрицательным")
    difference, x = np.asarray(difference, float), np.asarray(x, float)
    if difference.ndim != 1 or x.shape != difference.shape or not np.all(np.isfinite(x)):
        raise ValueError("Некорректные координаты прогноза")
    common = np.isfinite(difference)
    values = difference[common]
    if len(values) < 2:
        raise ValueError("Для интервала ΔMAE нужны хотя бы две общие пробы")
    if groups is not None:
        groups = np.asarray(groups, dtype=object)
        if groups.shape != difference.shape:
            raise ValueError("Для группового bootstrap нужны непустые группы для каждой точки")
        groups = groups[common]
    sample, group_count = _index_sampler(x[common], groups, block_length)
    rng = np.random.default_rng(seed)
    scale = np.max(np.abs(values)) or 1.
    normalized = values/scale
    means = np.empty(resamples)
    for i in range(resamples):
        if cancelled is not None and cancelled():
            raise BootstrapCancelled("Bootstrap отменён")
        means[i] = normalized[sample(rng)].mean()
    if cancelled is not None and cancelled():
        raise BootstrapCancelled("Bootstrap отменён")
    alpha = (1-confidence)/2
    lower, upper = np.quantile(means, [alpha, 1-alpha])*scale
    method = "groups" if groups is not None else "moving_blocks" if block_length is not None else "rows"
    direction = "a" if lower > 0 else "b" if upper < 0 else "uncertain"
    interval = {"lower": float(lower), "upper": float(upper), "confidence": float(confidence),
                "method": method, "direction": direction, "resamples": resamples, "seed": seed,
                "group_count": group_count, "block_length": block_length, "N_common": len(values)}
    if margin > 0:
        if lower > margin:
            conclusion = "a"
        elif upper < -margin:
            conclusion = "b"
        elif -margin <= lower and upper <= margin:
            conclusion = "within_margin"
        else:
            conclusion = "uncertain"
        interval["practical"] = {"margin": float(margin), "conclusion": conclusion}
    return interval


def bootstrap_bands(result: FitResult, x, *, resamples=500, seed=0,
                    prediction_sigma=None, min_success_fraction=.9, groups=None, block_length=None,
                    simultaneous=False,
                    progress: Callable[[int, int], None] | None = None,
                    cancelled: Callable[[], bool] | None = None) -> BootstrapResult:
    """Refit paired samples using the original loss, bounds, sx/sy and solver limit.

    With groups, draw as many whole groups as observed, with replacement. Group
    labels align with the fitted rows. Prediction still targets one observation.
    With block_length, draw overlapping blocks from rows sorted by distinct X.
    With simultaneous=True, calibrate a confidence radius over the given grid.

    Memory is O(N + B*(P+M)), not O(N*B): only one resampled dataset is held at a
    time. Retained curves allow cheap confidence changes. Failed samples are not
    retried; at least max(20, ceil(B*min_success_fraction)) must succeed.

    Without prediction_sigma, Prediction assumes exchangeable raw Y residuals.
    With it, standardized residuals are multiplied by the future Y scale. If sy
    was absent, standardization uses the empirical residual sample SD.
    """
    _validate_resampling(resamples, seed)
    if not np.isfinite(min_success_fraction) or not .5 <= min_success_fraction <= 1:
        raise ValueError("Минимальная доля успешных повторов должна быть от 0.5 до 1")
    sample, group_count = _index_sampler(result.x, groups, block_length)
    grid = np.atleast_1d(np.asarray(x, float)).copy()
    if (grid.ndim != 1 or not len(grid) or np.any(~np.isfinite(grid))
            or (result.model.positive_x and np.any(grid <= 0))):
        raise ValueError("Некорректные координаты прогноза")
    fitted = result.predict(grid)
    residuals, future_scale = _prediction_residuals(result, grid, prediction_sigma)
    sample_seed, noise_seed = np.random.SeedSequence(seed).spawn(2)
    sample_rng, noise_rng = np.random.default_rng(sample_seed), np.random.default_rng(noise_seed)
    curves = np.empty((resamples, len(grid)))
    parameters = np.empty((resamples, len(result.parameters)))
    required = max(20, int(np.ceil(resamples*min_success_fraction)))
    failures = successful = 0
    examples = []
    bounds = result.bounds if result.bounds is not None else (-np.inf, np.inf)
    report_every = max(1, resamples//100)
    for repeat in range(resamples):
        if cancelled is not None and cancelled():
            raise BootstrapCancelled("Bootstrap отменён")
        indices = sample(sample_rng)
        try:
            refit = fit_curve(result.x[indices], result.y[indices], result.model,
                              method=result.method, loss=result.loss, f_scale=result.f_scale,
                              initial=result.parameters, bounds=bounds,
                              sx=result.sx[indices] if result.sx is not None else None,
                              sy=result.sy[indices] if result.sy is not None else None,
                              absolute_sigma=result.absolute_sigma, max_evaluations=result.max_evaluations,
                              starts=result.start_count, start_seed=result.start_seed)
            curves[successful] = refit.predict(grid)
            parameters[successful] = refit.parameters
            successful += 1
        except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
            failures += 1
            if len(examples) < 3:
                examples.append(f"Repeat {repeat+1}: {exc}")
        if progress is not None and ((repeat+1) % report_every == 0 or repeat+1 == resamples):
            progress(repeat+1, resamples)
        if failures > resamples-required:
            raise ValueError(f"Недостаточно успешных bootstrap-повторов: {successful} из {repeat+1}; "
                             f"требуется минимум {required} из {resamples}. " + "; ".join(examples))
    if cancelled is not None and cancelled():
        raise BootstrapCancelled("Bootstrap отменён")
    curves, parameters = curves[:successful].copy(), parameters[:successful].copy()
    # A separate RNG keeps confidence samples unchanged by prediction settings.
    # Reuse one independent residual per replicate across X. These are pointwise
    # distributions, not simultaneous bands; shared draws avoid artificial jitter
    # and make predictions invariant to grid density, ordering and chunk size.
    draws = residuals[noise_rng.integers(0, len(residuals), size=successful)]
    predictions = curves+draws[:, None]*(1 if future_scale is None else future_scale)
    if cancelled is not None and cancelled():
        raise BootstrapCancelled("Bootstrap отменён")
    if np.any(~np.isfinite(predictions)):
        raise ValueError("Численное переполнение bootstrap Prediction; измените масштаб данных")
    max_deviations = None
    if simultaneous:
        with np.errstate(over="ignore"):
            deviations = curves-fitted
            np.abs(deviations, out=deviations)
            max_deviations = np.max(deviations, axis=1)
        if not np.all(np.isfinite(max_deviations)):
            raise ValueError("Численное переполнение доверительной полосы bootstrap")
    confidence_note = ("Одновременная доверительная полоса откалибрована на расчётной сетке. Покрытие приближённое."
                       if simultaneous else "Bootstrap: точечные percentile-полосы; номинальное покрытие не гарантируется.")
    warnings = [confidence_note,
                "Prediction условна по точному будущему X; эмпирические хвосты включают выбросы."]
    if result.bounds is not None and (np.any(np.isfinite(bounds[0])) or np.any(np.isfinite(bounds[1]))):
        warnings.append("Границы могут давать вырожденные или смещённые bootstrap-интервалы.")
    if result.method == "odr":
        warnings.append("Prediction использует вертикальные остатки ODR, без отдельного моделирования будущей ошибки X.")
    if result.sy is not None and future_scale is None:
        warnings.append("Prediction использует смесь сырых остатков; для неодинаковых ошибок задайте prediction_sigma.")
    if failures:
        warnings.append("Неудачные повторы исключены; это может смещать оценку неопределённости.")
    return BootstrapResult(grid, fitted, parameters, curves, predictions, resamples, seed,
                           min_success_fraction, failures, tuple(examples), "empirical_residuals" if future_scale is None else
                           "standardized_residuals", future_scale, tuple(warnings), draws,
                           group_count=group_count, block_length=block_length, max_deviations=max_deviations)
