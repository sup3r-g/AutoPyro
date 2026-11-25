"""JSON-compatible fit snapshots and self-contained HTML reports, without Qt."""
import base64
from copy import deepcopy
from html import escape
from io import BytesIO
import json
from math import isfinite

import numpy as np
from matplotlib.figure import Figure
from scipy.stats import probplot

from .curve_constraints import parameter_samples


def fit_payload(result, settings, row_ids, grid, bands, *, bootstrap=None, bootstrap_error=None, confidence=.95):
    return {"settings": deepcopy(settings), "formula": result.model.formula,
            "parameters": dict(zip(result.model.parameters, result.parameters.tolist())),
            "all_parameters": {k: float(v[0]) for k, v in parameter_samples(result.model, result.parameters).items()},
            "covariance": result.covariance.tolist() if result.covariance is not None else None,
            "covariance_factor": result.covariance_factor.tolist() if result.covariance_factor is not None else None,
            "metrics": {k: v if np.isfinite(v) else None for k, v in result.metrics.items()},
            "loss": result.loss, "f_scale": result.f_scale, "active_bounds": result.active_bounds,
            "row_ids": list(row_ids), "x": result.x.tolist(), "y": result.y.tolist(),
            "adjusted_x": result.adjusted_x.tolist(), "fitted": result.fitted.tolist(),
            "sx": result.sx.tolist() if result.sx is not None else None,
            "sy": result.sy.tolist() if result.sy is not None else None,
            "objective": result.objective, "warnings": result.warnings, "confidence": confidence,
            "grid": grid.tolist(), "curve": result.predict(grid).tolist(),
            "bands": {k: v.tolist() for k, v in zip(("mean", "confidence_lower", "confidence_upper",
                                                    "prediction_lower", "prediction_upper"), bands)} if bands is not None else None,
            "bootstrap": bootstrap.diagnostics(confidence=confidence) if bootstrap else None,
            "bootstrap_parameters": bootstrap.parameters.tolist() if bootstrap else None,
            "bootstrap_prediction_draws": bootstrap.prediction_draws.tolist() if bootstrap and bootstrap.prediction_draws is not None else None,
            "bootstrap_error": bootstrap_error,
            "parameter_intervals": bootstrap.parameter_intervals(result.model, confidence=confidence, bounds=result.bounds) if bootstrap else None,
            "start_diagnostics": result.start_diagnostics}


def compare_archived_fit(record, result, row_ids):
    """Compare archived inputs by UUID and evaluate the new curve on the saved grid."""
    saved_ids = record.get("row_ids") or record["source_rows"]
    saved_positions = None
    if list(row_ids) != list(saved_ids):
        positions = {row_id: i for i, row_id in enumerate(saved_ids)}
        saved_positions = np.array([positions[row_id] for row_id in row_ids])
    changes = []
    changed = np.zeros(len(row_ids), dtype=bool)
    for field in ("x", "y", "sx", "sy"):
        old, new = record.get(field), getattr(result, field)
        if old is None or new is None:
            continue
        old = np.asarray(old, dtype=float)
        if saved_positions is not None:
            old = old[saved_positions]
        different = old != new
        changed |= different
        changes.extend([row_ids[i], field, float(old[i]), float(new[i])] for i in np.flatnonzero(different))

    def difference(old, new):
        if old is None or new is None:
            return None
        delta = new-old
        return delta if isfinite(delta) else None

    previous = record.get("all_parameters") or record["parameters"]
    current = {name: float(values[0]) for name, values in parameter_samples(result.model, result.parameters).items()}
    parameters = [[name, previous.get(name), current.get(name), difference(previous.get(name), current.get(name))]
                  for name in dict.fromkeys((*previous, *current))]
    grid, old_curve = np.asarray(record["grid"], float), np.asarray(record["curve"], float)
    overlap = (grid >= result.x.min()) & (grid <= result.x.max())
    curve, curve_error = [], None
    if not overlap.any():
        curve_error = "На сохранённой сетке нет точек внутри текущего диапазона X"
    else:
        grid, old_curve = grid[overlap], old_curve[overlap]
        try:
            current_curve = result.predict(grid)
            curve = [[x, old, new, difference(old, new)]
                     for x, old, new in zip(grid.tolist(), old_curve.tolist(), current_curve.tolist())]
        except ValueError as exc:
            curve_error = str(exc)
    peak = max((row for row in curve if row[3] is not None), key=lambda row: abs(row[3]), default=None)
    return {"archive_id": record["id"], "archive_name": record["name"], "changed_rows": int(changed.sum()),
            "changes": changes, "parameters": parameters, "curve": curve, "curve_error": curve_error,
            "peak": {"x": peak[0], "delta": peak[3]} if peak is not None else None}


def _table(headers, rows):
    def display(value):
        if value is None or isinstance(value, float) and not np.isfinite(value):
            return "—"
        return f"{value:.7g}" if isinstance(value, float) else str(value)
    return "<table><thead><tr>"+"".join(f"<th>{escape(str(h))}</th>" for h in headers)+"</tr></thead><tbody>"+"".join(
        "<tr>"+"".join(f"<td>{escape(display(v))}</td>" for v in row)+"</tr>" for row in rows)+"</tbody></table>"


def _image(figure):
    stream = BytesIO()
    figure.savefig(stream, format="svg")
    return '<img alt="Diagnostics" src="data:image/svg+xml;base64,'+base64.b64encode(stream.getvalue()).decode()+ '">'


def report_html(records, *, comparison=None, labels=None):
    """Render saved numbers and curves; never refit models while exporting."""
    labels = labels or {}
    label = lambda key, default: labels.get(key, default)
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>AutoPyro</title>",
             "<style>body{font:15px system-ui;max-width:1100px;margin:2rem auto;padding:1rem}table{border-collapse:collapse;margin:1rem 0}td,th{border:1px solid #aaa;padding:.4rem}img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}section{break-before:page}</style></head><body>",
             "<h1>"+escape(label("title", "Curve fitting"))+"</h1>"]
    for record in records:
        x, y, fitted = (np.asarray(record[k], float) for k in ("x", "y", "fitted"))
        residual = y-fitted
        figure = Figure(figsize=(10, 7), layout="constrained")
        fit_axis, residual_axis, qq_axis, histogram = figure.subplots(2, 2).ravel()
        fit_axis.scatter(x, y, s=6, rasterized=True)
        fit_axis.plot(record["grid"], record["curve"], color="tab:red")
        if record.get("bands"):
            b = record["bands"]
            fit_axis.fill_between(record["grid"], b["prediction_lower"], b["prediction_upper"], alpha=.15)
            fit_axis.fill_between(record["grid"], b["confidence_lower"], b["confidence_upper"], alpha=.3)
        axis_labels = record.get("axis_labels", ["X", "Y"])
        fit_axis.set(xlabel=axis_labels[0], ylabel=axis_labels[1])
        residual_axis.scatter(fitted, residual, s=6, rasterized=True)
        residual_axis.axhline(0, color="gray")
        residual_axis.set(xlabel=label("prediction", "Prediction"), ylabel=label("residual", "Residual"))
        (theory, ordered), (slope, intercept, _) = probplot(residual)
        qq_axis.scatter(theory, ordered, s=6, rasterized=True)
        qq_axis.plot(theory, slope*theory+intercept, color="gray")
        qq_axis.set(xlabel=label("normal", "Normal quantiles"), ylabel=label("residual", "Residual"))
        histogram.hist(residual, bins=30)
        histogram.set_xlabel(label("residual", "Residual"))
        parts.extend(["<section><h2>"+escape(record["name"])+"</h2>",
                      "<p>"+escape(record["at"])+" · "+escape(label("stale", "Stale") if record.get("stale") else label("current", "Current"))+"</p>",
                      "<p>"+escape(record["formula"])+"</p>",
                      _table([label("parameter", "Parameter"), label("estimate", "Estimate")], record["all_parameters"].items()),
                      _table([label("metric", "Metric"), label("value", "Value")], record["metrics"].items()), _image(figure)])
        intervals = record.get("parameter_intervals")
        if intervals:
            parts.append(_table([label("parameter", "Parameter"), label("lower", "CI lower"), label("upper", "CI upper"), "Bootstrap SD", label("bound", "Bound frequency")],
                                [[r[k] for k in ("parameter", "lower", "upper", "sd", "bound_frequency")] for r in intervals]))
        conditions = {k: record.get(k) for k in ("settings", "confidence", "warnings", "bootstrap", "bootstrap_error", "start_diagnostics", "source_ids", "source_digest")}
        parts.append("<pre>"+escape(json.dumps(conditions, ensure_ascii=False, indent=2))+"</pre></section>")
    if comparison is not None and comparison.nested is not None:
        nested = comparison.nested
        parts.append("<section><h2>"+escape(label("nested_cv", "Nested CV"))+"</h2>")
        strategy = comparison.plan.strategy
        configuration = dict(strategy=strategy, outer_folds=comparison.folds, inner_folds=nested.inner_folds,
                             seed=comparison.seed, shuffle=comparison.shuffle, gap=comparison.plan.gap,
                             warmup_rows=int(np.count_nonzero(comparison.fold_ids < 0)))
        parts.append("<pre>"+escape(json.dumps(configuration, indent=2))+"</pre>")
        if nested.metrics:
            parts.append(_table([label("outer_rmse", "Outer RMSE"), label("outer_mae", "Outer MAE"), "OOF R²"],
                                [[nested.metrics[key] for key in ("RMSE", "MAE", "R2")]]))
        headers = [label("outer_fold", "Outer fold"), label("selected_model", "Selected model"), "N train", "N test"]
        extra_fields = ()
        if strategy == "nested_group":
            headers.extend([label("train_groups", "Training groups"), label("test_groups", "Test groups")])
            extra_fields = ("train_groups", "test_groups")
        elif strategy == "nested_time":
            headers.extend([label("train_periods", "Training periods"), label("test_periods", "Test periods"),
                            label("gap_rows", "Rows in gap")])
            extra_fields = ("train_periods", "test_periods", "gap_rows")
        headers.extend([label("inner_rmse", "Inner RMSE"), label("outer_rmse", "Outer RMSE"),
                        label("outer_mae", "Outer MAE"), label("fold_errors", "Fold errors")])
        parts.append(_table(headers, [[fold.fold, fold.winner, fold.train_n, fold.test_n,
                                      *(getattr(fold, key) for key in extra_fields), fold.inner_rmse,
                                      fold.metrics.get("RMSE"), fold.metrics.get("MAE"), fold.error]
                                     for fold in nested.folds]))
        parts.append("</section>")
    elif comparison is not None:
        parts.append("<section><h2>"+escape(label("cv", "Cross-validation"))+"</h2>")
        configuration = dict(strategy=comparison.plan.strategy, folds=comparison.folds, seed=comparison.seed,
                             gap=comparison.plan.gap, shuffle=comparison.shuffle,
                             fold_metrics={s.name: s.fold_metrics for s in comparison.scores})
        parts.append("<pre>"+escape(json.dumps(configuration, ensure_ascii=False, indent=2))+"</pre>")
        headers = [label("model", "Model"), "OOF RMSE", "OOF MAE", "OOF R²"]
        metric_keys = ["RMSE", "MAE", "R2"]
        if comparison.plan.strategy == "loo":
            headers.append("PRESS")
            metric_keys.append("PRESS")
        parts.append(_table(headers, [[score.name, *(score.metrics[key] for key in metric_keys)]
                                      for score in comparison.ranking]))
        for score in comparison.scores:
            if score.errors:
                parts.append("<pre>"+escape(score.name+"\n"+"\n".join(score.errors))+"</pre>")
        if comparison.ranking:
            predicted = comparison.ranking[0].predictions
            valid = np.isfinite(predicted)
            figure = Figure(figsize=(8, 4), layout="constrained")
            axis = figure.subplots()
            axis.scatter(comparison.y[valid], predicted[valid], s=8, rasterized=True)
            axis.set(xlabel=label("observed", "Observed"), ylabel=label("oof", "Held-out prediction"))
            parts.append(_image(figure))
        parts.append("</section>")
        test = comparison.holdout
        if test is not None:
            parts.append("<section><h2>"+escape(label("holdout", "Independent test"))+"</h2>")
            if test.metrics is not None:
                parts.append(_table([label("model", "Model"), "N", "RMSE", "MAE", "R²"],
                                    [[test.winner, len(test.y), *(test.metrics[key] for key in ("RMSE", "MAE", "R2"))]]))
            if test.error:
                parts.append("<pre>"+escape(test.error)+"</pre>")
            if test.pair is not None:
                first, second = test.pair.models
                metrics = test.pair.metrics
                parts.append("<h3>"+escape(label("holdout_pair", "Paired test"))+"</h3>")
                parts.append(_table(["A", "B", "N", "MAE A", "MAE B", "Δ (B − A)"],
                                    [[first.name, second.name, f"{metrics['N_common']}/{metrics['N_test']}",
                                      *(metrics.get(key) for key in ("MAE_A", "MAE_B", "mean_difference"))]]))
                note = labels.get("holdout_interval_note")
                if note:
                    parts.extend("<p>"+escape(line)+"</p>" for line in note.splitlines())
                elif test.pair.interval is not None:
                    ci = test.pair.interval
                    direction = {"a": "Lower MAE for A", "b": "Lower MAE for B",
                                 "uncertain": "Undetermined (interval includes zero)"}[ci["direction"]]
                    parts.append(_table(["ΔMAE CI %", "Lower", "Upper", "Resampling", "Direction"],
                                        [[100*ci["confidence"], ci["lower"], ci["upper"], ci["method"], direction]]))
                    practical = ci.get("practical")
                    if practical is not None:
                        conclusion = {"a": "Advantage of A exceeds margin", "b": "Advantage of B exceeds margin",
                                      "within_margin": "Interval within margin", "uncertain": "Undetermined"}
                        parts.append(_table(["ΔMAE margin", "Practical conclusion"],
                                            [[practical["margin"], conclusion[practical["conclusion"]]]]))
                elif test.pair.interval_error:
                    parts.append("<p>"+escape(test.pair.interval_error)+"</p>")
            parts.append("</section>")
    return "".join(parts)+"</body></html>"
