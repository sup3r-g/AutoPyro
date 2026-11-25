"""Present fit coefficients, uncertainty bands and residuals and export the fit."""

from __future__ import annotations

import json

from PySide6.QtWidgets import QFileDialog, QMessageBox
import numpy as np

from autopyro.curves.curve_bootstrap import BOOTSTRAP_BAND_METHODS
from autopyro.curves.curve_constraints import parameter_samples
from autopyro.curves.curve_numerics import curve_axis_scale

from ..i18n import tr, message


def current_bands_available(editor):
    return editor.bootstrap_result is not None or (editor.result is not None and editor.result.covariance is not None
                                                 and editor.fit_settings["band_method"] not in BOOTSTRAP_BAND_METHODS)


def band_data(editor):
    """Shared cached bands for plotting and export; never starts a refit."""
    if editor.bootstrap_result is not None:
        bootstrap = editor.bootstrap_result
        return bootstrap.grid, bootstrap.bands(confidence=editor.confidence.value()/100)
    r = editor.result
    grid = np.linspace(np.min(r.x), np.max(r.x), 400)
    if editor.fit_settings["band_method"] not in BOOTSTRAP_BAND_METHODS and r.covariance is not None:
        return grid, r.bands(grid, confidence=editor.confidence.value()/100)
    return grid, None


def show_summary(editor):
    r = editor.result
    lines = [r.model.formula, f"{r.method.upper()} / {r.loss}", "", tr("Параметр = оценка ± стандартная ошибка")]
    errors = r.standard_errors
    lines.extend(f"{name} = {value:.8g}" + (f" ± {errors[i]:.5g}" if errors is not None else "")
                 for i, (name, value) in enumerate(zip(r.model.parameters, r.parameters)))
    lines.extend(["", *(f"{name}: {value:.8g}" for name, value in r.metrics.items()),
                  f"Objective: {r.objective:.8g}", "", *[message(w) for w in r.warnings]])
    if r.model.parameter_expressions:
        lines.extend(f"{name} = {values[0]:.8g}" for name, values in parameter_samples(r.model, r.parameters).items())
    if r.start_diagnostics:
        lines.extend(["", tr("Приближений: {0}; успешных: {1}", r.start_count,
                                sum(d["error"] is None for d in r.start_diagnostics))])
    if editor.bootstrap_result is not None:
        b = editor.bootstrap_result
        lines.extend(["", f"Bootstrap: {b.successful}/{b.resamples}; Seed={b.seed}; failures={b.failures}",
                      *[message(w) for w in b.diagnostics(confidence=editor.confidence.value()/100)["warnings"]]])
    if editor.bootstrap_error:
        lines.extend(["", editor.bootstrap_error])
    if selection := editor.fit_settings.get("selection"):
        lines.extend(["", tr("Модель из CV: {0} · Обучающих строк: {1}", selection["candidate"], len(editor.row_ids))])
    if replay := editor.fit_settings.get("replay"):
        lines.extend(["", tr("Пересчёт по архиву: {0}", replay["archive_name"])])
        if replay["source_was_stale"]:
            lines.append(tr("Архив помечен как устаревший. Использованы текущие значения."))
    editor.summary.setPlainText("\n".join(lines))


def redraw(editor, *args):
    if editor._archive_records is not None:
        editor.draw_saved_curves(editor._archive_records)
        return
    if editor.result is None:
        return
    r = editor.result
    try:
        grid, bands = editor.band_data()
        mean = r.predict(grid)
    except ValueError as exc:
        editor.status.setText(message(exc))
        return
    editor.figure.clear()
    axes, residuals = editor.figure.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    if bands is not None:
        _, lo, hi, plo, phi = bands
    show_prediction = bands is not None and editor.show_prediction.isChecked()
    show_confidence = bands is not None and editor.show_confidence.isChecked()
    visible_bands = (bands[3:] if show_prediction else ()) + (bands[1:3] if show_confidence else ())
    scale = curve_axis_scale(*visible_bands)
    y, label_y = r.y, editor.axis_labels[1]
    if scale != 1.:
        y, mean = y/scale, mean/scale
        lo, hi, plo, phi = lo/scale, hi/scale, plo/scale, phi/scale
        label_y = f"{label_y} (× {scale:.0e})"
    suffix = " (bootstrap)" if editor.bootstrap_result is not None else ""
    if show_prediction:
        axes.fill_between(grid, plo, phi, alpha=.12, label=f"{tr('Предсказательная полоса')} {editor.confidence.value():g}%{suffix}")
    if show_confidence:
        label = (tr("Одновременная доверительная полоса") if editor.bootstrap_result is not None
                 and editor.bootstrap_result.max_deviations is not None else tr("Доверительная полоса"))
        axes.fill_between(grid, lo, hi, alpha=.25, label=f"{label} {editor.confidence.value():g}%{suffix}")
    axes.scatter(r.x, y, s=12, alpha=.6, linewidths=0, rasterized=True, label=tr("Данные"))
    axes.plot(grid, mean, color="tab:red", label=r.method.upper())
    axes.set_ylabel(label_y, parse_math=False)
    # Automatic legend placement repeatedly scans every point on large plots.
    axes.legend(loc="upper left")
    residuals.scatter(r.x, r.y-r.fitted, s=8, linewidths=0, rasterized=True)
    residuals.axhline(0, color="gray", linewidth=.8)
    residuals.set_ylabel(tr("Остаток Y"))
    residuals.set_xlabel(editor.axis_labels[0])
    editor.canvas.draw_idle()
    if bands is None:
        editor.status.setText(tr("Точек: {0}; исключено: {1}", len(editor.row_ids), editor.excluded_count)
                            + "\n" + (editor.bootstrap_error or tr("Для этих полос выберите метод Bootstrap и пересчитайте")))
    elif editor.bootstrap_result is not None:
        b = editor.bootstrap_result
        status = tr("Bootstrap: успешно {0} из {1}; неудачных: {2}", b.successful, b.resamples, b.failures)
        if b.group_count is not None:
            status += "\n" + tr("Групповой bootstrap. Число групп: {0}", b.group_count)
        if b.block_length is not None:
            status += "\n" + tr("Блочный bootstrap. Точек в блоке: {0}", b.block_length)
        editor.status.setText(status)
        editor.show_summary()
        editor.display_bootstrap_parameters()


def export_result(editor):
    if editor.result is None or editor._archive_records is not None:
        return
    path, _ = QFileDialog.getSaveFileName(editor, tr("Экспорт подгонки"), "curve_fit.json", "JSON (*.json)")
    if not path:
        return
    r = editor.result
    from autopyro.curves.curve_archive import fit_payload
    from autopyro.data_io.file_output import atomic_output

    try:
        grid, band = editor.band_data()
        payload = fit_payload(r, editor.fit_settings, editor.row_ids, grid, band, bootstrap=editor.bootstrap_result,
                              bootstrap_error=editor.bootstrap_error, confidence=editor.confidence.value()/100)
        if editor.archive_diff is not None:
            payload["archive_comparison"] = editor.archive_diff
        with atomic_output(path) as temporary:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    except (OSError, ValueError) as exc:
        QMessageBox.warning(editor, tr("Ошибка"), message(exc))
