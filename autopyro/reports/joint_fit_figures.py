"""Capture and render joint fit figures without running another optimization."""

import base64
import io

import numpy as np
from matplotlib.figure import Figure
from autopyro.curves.curve_numerics import curve_axis_scale


_OBSERVATION_LIMIT = 2000
_REPORT_OBSERVATION_TARGET = 20000


def _sample_observations(result, indices, x, limit):
    """Keep the X endpoints and Y and residual extrema of each X-ordered block."""
    order = np.argsort(x, kind="stable")
    y = result.y[indices]
    residuals = y-result.fitted[indices]
    selected = []
    # Each block contributes at most six distinct points.
    for block in np.array_split(order, limit // 6):
        observed = y[block]
        errors = residuals[block]
        selected.extend((block[0], block[-1], block[observed.argmin()], block[observed.argmax()],
                         block[errors.argmin()], block[errors.argmax()]))
    return indices[np.unique(selected)]


def capture_joint_plots(result, bootstrap, *, confidence, simultaneous):
    plots = []
    series_count = len(result.labels)
    limit = min(_OBSERVATION_LIMIT, max(6, _REPORT_OBSERVATION_TARGET // max(1, series_count)))
    for series in range(series_count):
        indices = result.indices(series)
        x = result.x[indices]
        grid = np.linspace(x.min(), x.max(), 200)
        sampled = len(indices) > limit
        if sampled:
            indices = _sample_observations(result, indices, x, limit)
        plot = {"x": result.x[indices].tolist(), "y": result.y[indices].tolist(), "grid": grid.tolist()}
        if sampled:
            plot["sampling"] = "x_block_extrema"
        if bootstrap is not None:
            try:
                bands = bootstrap.bands(result, series, grid, confidence=confidence, simultaneous=simultaneous)
            except ValueError:
                plot["bands_unavailable"] = True
            else:
                mean, ci_low, ci_high, pi_low, pi_high = bands
                plot.update(mean=mean.tolist(), confidence_band=[ci_low.tolist(), ci_high.tolist()],
                            prediction_band=[pi_low.tolist(), pi_high.tolist()])
        if "mean" not in plot:
            try:
                plot["mean"] = result.predict(series, grid).tolist()
            except ValueError:
                plot["mean"] = None
        plots.append(plot)
    return {"confidence": confidence, "simultaneous": simultaneous, "series": plots}


def render_joint_figures(fit, language, progress):
    plots = fit.get("plots")
    if not plots:
        return []
    english = language == "en"
    series_label = "Series" if english else "Серия"
    observed = "Observations" if english else "Наблюдения"
    trend = "Fitted curve" if english else "Кривая подгонки"
    prediction = "Prediction band" if english else "Предсказательная полоса"
    if plots["simultaneous"]:
        confidence = "Simultaneous confidence band" if english else "Одновременная доверительная полоса"
    else:
        confidence = "Confidence band" if english else "Доверительная полоса"
    level = f"{plots['confidence']*100:g}%"
    figures = []
    figure = Figure(figsize=(7.2, 3.8), layout="constrained")
    for index, (label, count, plot) in enumerate(zip(fit["series"], fit["counts"], plots["series"]), 1):
        name = f"{series_label} {index} · {label}"
        progress(name, index-1, len(plots["series"]))
        axis = figure.subplots()
        axis.set_xlabel(fit["axes"][0], parse_math=False)
        y_label = fit["axes"][1]
        y_values = {"observed": np.asarray(plot["y"])}
        if plot["mean"] is not None:
            y_values["mean"] = np.asarray(plot["mean"])
        for key in ("prediction_band", "confidence_band"):
            if key in plot:
                y_values[key] = np.asarray(plot[key])
        # Keep the renderer's axis span and tick calculations in range.
        scale = curve_axis_scale(*y_values.values())
        if scale != 1.:
            y_values = {key: values/scale for key, values in y_values.items()}
            y_label = f"{y_label} (× {scale:.0e})"
        axis.set_ylabel(y_label, parse_math=False)
        grid = plot["grid"]
        if "prediction_band" in plot:
            axis.fill_between(grid, *y_values["prediction_band"], color="tab:orange", alpha=.24, label=f"{prediction} {level}")
            axis.fill_between(grid, *y_values["confidence_band"], color="tab:blue", alpha=.32, label=f"{confidence} {level}")
        axis.scatter(plot["x"], y_values["observed"], s=9, linewidths=0, color="tab:gray", label=observed)
        if plot["mean"] is not None:
            axis.plot(grid, y_values["mean"], color="tab:red", label=trend)
        axis.legend(loc="upper left")
        caption = f"{observed} · {len(plot['x'])}/{count}"
        if plot.get("sampling") == "x_block_extrema":
            caption += " · " + ("Y and residual extremes preserved within X-ordered blocks" if english
                                 else "Сохранены крайние значения Y и остатков в блоках по X")
        if plot.get("bands_unavailable") or fit["bootstrap_failed"]:
            caption += " · " + ("Bootstrap bands unavailable" if english else "Полосы bootstrap недоступны")
        if plot["mean"] is None:
            caption += " · " + ("Fitted curve unavailable on the grid" if english else "Кривая на сетке недоступна")
        with io.BytesIO() as stream:
            figure.savefig(stream, format="png", dpi=130)
            figures.append({"name": name, "caption": caption, "png": base64.b64encode(stream.getvalue()).decode("ascii")})
        figure.clear()
    return figures
