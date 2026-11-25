"""Bounded, reproducible multi-start search using the existing single-fit solver."""
from dataclasses import replace

import numpy as np

from .curve_fitting import fit_curve, _initial


def multistart_fit(x, y, model, *, starts, seed, options):
    rng = np.random.default_rng(seed)
    p = len(model.parameters)
    lower, upper = (np.broadcast_to(np.asarray(v, float), (p,)) for v in options["bounds"])
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(lower >= upper):
        raise ValueError("Некорректные границы параметров")
    base = (np.asarray(options["initial"], float) if options["initial"] is not None else
            _initial(model, np.asarray(x, float), np.asarray(y, float)))
    if base.shape != (p,) or not np.all(np.isfinite(base)):
        raise ValueError("Некорректное начальное приближение")
    if options["initial"] is not None and (np.any(base < lower) or np.any(base > upper)):
        raise ValueError("Начальные параметры должны лежать внутри заданных границ")
    direct = model.linear and options["method"] == "ols" and options["loss"] == "linear" and not np.any(np.isfinite([lower, upper]))
    if direct:
        starts = 1
    best = None
    diagnostics = []
    for start in range(starts):
        initial = options["initial"] if start == 0 else base+rng.normal(size=p)*(1+np.abs(base))
        if start:
            finite = np.isfinite(lower) & np.isfinite(upper)
            initial[finite] = rng.uniform(lower[finite], upper[finite])
            initial = np.clip(initial, lower, upper)
        try:
            result = fit_curve(x, y, model, **(options | {"initial": initial}))
            diagnostics.append({"start": start+1, "initial": None if initial is None else np.asarray(initial).tolist(),
                                "objective": result.objective, "error": None})
            if best is None or result.objective < best.objective:
                best, base = result, result.parameters.copy()
        except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
            diagnostics.append({"start": start+1, "initial": None if initial is None else np.asarray(initial).tolist(),
                                "objective": None, "error": str(exc)})
    if best is None:
        raise ValueError("Ни один старт не сошёлся: " + str(diagnostics[0]["error"]))
    return replace(best, start_count=starts, start_seed=seed, start_diagnostics=tuple(diagnostics))
