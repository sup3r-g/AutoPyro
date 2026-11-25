"""Run with python -m tools.benchmark_curve_workflow; numerical work only."""
import json
from time import perf_counter

import numpy as np

from autopyro.core import Column, Project, Row
from autopyro.curve_fitting import fit_curve, standard_model
from autopyro.curve_outputs import prepare_output_columns
from autopyro.model_comparison import FitCandidate, compare_models


def main():
    n = 100_000
    x = np.linspace(0, 10, n)
    y = 2*x+1+np.random.default_rng(42).normal(0, .3, n)
    candidates = [FitCandidate("line", standard_model()),
                  FitCandidate("quadratic", standard_model("polynomial", degree=2)),
                  FitCandidate("cubic", standard_model("polynomial", degree=3))]
    measurements = []
    for strategy in ("kfold", "group", "time"):
        start = perf_counter()
        result = compare_models(x, y, candidates, folds=5, strategy=strategy,
                                groups=np.arange(n)//100, times=np.arange(n))
        measurements.append({"operation": strategy+" CV", "n": n, "models": 3,
                             "seconds": perf_counter()-start, "best": result.ranking[0].name})
    cx, cy = Column(name="X", kind="number"), Column(name="Y", kind="number")
    rows = [Row(values={cx.id: float(a), cy.id: float(b)}) for a, b in zip(x[:50_000], y[:50_000])]
    project = Project(columns=[cx, cy], rows=rows)
    fitted = fit_curve(x[:50_000], y[:50_000])
    start = perf_counter()
    _, columns = prepare_output_columns(project, rows, fitted, {"x": cx.id, "y": cy.id}, [r.id for r in rows])
    measurements.append({"operation": "apply fitted model and bands", "n": len(rows), "columns": len(columns),
                         "seconds": perf_counter()-start})
    print(json.dumps(measurements, indent=2))


if __name__ == "__main__":
    main()
