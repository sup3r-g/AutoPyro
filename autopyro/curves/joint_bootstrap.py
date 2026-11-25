"""Bootstrap joint fits on a fixed set of series, with lazy bands for each series."""
from dataclasses import dataclass

import numpy as np

from .curve_bootstrap import (BOOTSTRAP_TAIL_WARNING, BootstrapCancelled, _center_prediction_residuals, _percentile_bands,
                              _validate_resampling)
from .curve_fitting import _linear_components
from .joint_fitting import _solve_joint


@dataclass
class JointBootstrapResult:
    coefficients: np.ndarray
    prediction_draws: np.ndarray
    resamples: int
    seed: int
    prediction_sigma: float | None

    def diagnostics(self, *, confidence=.95):
        successful = len(self.coefficients)
        warnings = [BOOTSTRAP_TAIL_WARNING] if successful*(1-confidence)/2 < 10 else []
        return {"method": "within_series_pairs_percentile", "resamples": self.resamples, "seed": self.seed,
                "successful": successful, "failures": self.resamples-successful, "prediction_sigma": self.prediction_sigma,
                "warnings": warnings}

    def bands(self, result, series, grid, *, confidence=.95, simultaneous=False):
        mean = result.predict(series, grid)
        curves = np.array([result.model.evaluate(grid, beta) for beta in self.coefficients[:, series]])
        scale = 1. if self.prediction_sigma is None else self.prediction_sigma
        with np.errstate(over="ignore"):
            predictions = curves+self.prediction_draws[:, series, None]*scale
            deviations = np.max(np.abs(curves-mean), axis=1) if simultaneous else None
        if not np.all(np.isfinite(predictions)) or deviations is not None and not np.all(np.isfinite(deviations)):
            raise ValueError("Численное переполнение bootstrap Prediction; измените масштаб данных")
        return _percentile_bands(mean, curves, predictions, confidence, deviations)


def bootstrap_joint_fit(result, *, resamples=500, seed=0, prediction_sigma=None, progress=None, cancelled=None):
    """Refit paired rows within each series, preserving each series' sample size."""
    _validate_resampling(resamples, seed)
    if prediction_sigma is not None and (not np.isfinite(prediction_sigma) or prediction_sigma <= 0):
        raise ValueError("Погрешность прогноза должна быть положительной и конечной")
    count, parameters = result.coefficients.shape
    if resamples*count*(parameters+1)*np.dtype(float).itemsize > 256*1024**2:
        raise ValueError("Слишком много серий и bootstrap-повторов. Уменьшите число повторов")
    order, boundaries = result.order, result.boundaries
    sizes = np.diff(boundaries)
    starts = boundaries[:-1]
    row_starts, row_sizes = np.repeat(starts, sizes), np.repeat(sizes, sizes)
    codes = result.codes[order]
    positions = np.arange(len(order))
    common = np.array([name in result.shared for name in result.model.parameters])
    limits = result.bounds if result.bounds is not None else (np.full_like(result.coefficients, -np.inf),
                                                             np.full_like(result.coefficients, np.inf))
    sigma = np.ones_like(result.x) if result.sy is None else result.sy
    residuals = (result.y-result.fitted)[order]
    noise_sigma = result.sy[order] if result.sy is not None else None
    for start, end in zip(starts, boundaries[1:]):
        block = slice(start, end)
        residuals[block] = _center_prediction_residuals(residuals[block], noise_sigma[block] if noise_sigma is not None else None,
                                                       result.loss, prediction_sigma is not None)
    linear = _linear_components(result.model, result.x) if result.model.linear else None
    sample_seed, noise_seed = np.random.SeedSequence(seed).spawn(2)
    rng, noise_rng = np.random.default_rng(sample_seed), np.random.default_rng(noise_seed)
    coefficients = np.empty((resamples, count, parameters))
    draws = np.empty((resamples, count))
    required = max(20, (9*resamples+9)//10)
    successful = 0
    report_every = max(1, resamples//100)
    for repeat in range(resamples):
        if cancelled is not None and cancelled():
            raise BootstrapCancelled("Bootstrap отменён")
        sample = order[row_starts+rng.integers(0, row_sizes)]
        design = tuple(array[sample] for array in linear) if linear is not None else None
        try:
            beta, _, _ = _solve_joint(result.model, result.x[sample], result.y[sample], sigma[sample], codes,
                                      positions, boundaries, common, limits, result.coefficients, design,
                                      result.max_evaluations, result.loss, result.f_scale, cancelled)
        except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
            if successful+resamples-repeat-1 < required:
                raise ValueError("Недостаточно успешных повторов совместного bootstrap") from exc
        else:
            if beta is None:
                raise BootstrapCancelled("Bootstrap отменён")
            coefficients[successful] = beta
            draws[successful] = residuals[starts+noise_rng.integers(0, sizes)]
            successful += 1
        if progress is not None and ((repeat+1) % report_every == 0 or repeat+1 == resamples):
            progress(repeat+1, resamples)
    if cancelled is not None and cancelled():
        raise BootstrapCancelled("Bootstrap отменён")
    return JointBootstrapResult(coefficients[:successful], draws[:successful], resamples, seed, prediction_sigma)
