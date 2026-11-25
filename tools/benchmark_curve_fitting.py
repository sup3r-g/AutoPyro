"""Reproducible fitting-only benchmark; excludes imports, Qt and rendering.

Run from the repository root: python -m tools.benchmark_curve_fitting
"""

from time import perf_counter
import json

import numpy as np

from autopyro.curve_fitting import fit_curve, standard_model
from autopyro.model_comparison import FitCandidate, compare_models
from autopyro.curve_bootstrap import bootstrap_bands


def main():
    measurements = []
    for size, method, loss in [(100_000, "ols", "linear"), (100_000, "odr", "linear"),
                               (100_000, "ols", "huber"), (10_000, "odr", "huber")]:
        x = np.linspace(0, 10, size)
        y = 2*x+1+np.random.default_rng(123).normal(0, .3, size)
        start = perf_counter()
        result = fit_curve(x, y, method=method, loss=loss)
        measurements.append({"n": size, "method": method, "loss": loss, "seconds": perf_counter()-start,
                             "parameters": result.parameters.tolist()})
    candidates = [FitCandidate("line", standard_model()),
                  FitCandidate("quadratic", standard_model("polynomial", degree=2)),
                  FitCandidate("cubic", standard_model("polynomial", degree=3))]
    start = perf_counter()
    comparison = compare_models(x, y, candidates, folds=5, seed=42)
    measurements.append({"n": len(x), "operation": "5-fold CV, 3 models", "seconds": perf_counter()-start,
                         "ranking": [score.name for score in comparison.ranking]})
    for size, method in [(10_000, "ols"), (1_000, "odr")]:
        x = np.linspace(0, 10, size)
        y = 2*x+1+np.random.default_rng(123).normal(0, .3, size)
        fit = fit_curve(x, y, method=method, loss="huber", f_scale=.5, bounds=([0, -10], [2, 10]))
        start = perf_counter()
        bootstrap = bootstrap_bands(fit, np.linspace(0, 10, 400), resamples=200, seed=42)
        measurements.append({"n": size, "operation": f"Huber {method}, bounded, bootstrap",
                             "resamples": bootstrap.resamples, "successful": bootstrap.successful,
                             "seconds": perf_counter()-start,
                             "retained_sample_bytes": bootstrap.parameters.nbytes+bootstrap.curve_samples.nbytes
                             +bootstrap.prediction_samples.nbytes})
    print(json.dumps(measurements, indent=2))


if __name__ == "__main__":
    main()
