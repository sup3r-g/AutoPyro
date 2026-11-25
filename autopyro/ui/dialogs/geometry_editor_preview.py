"""Render geometry, coverage and sample classifications in analysis coordinates."""

from __future__ import annotations

from collections import Counter

from PySide6.QtWidgets import QDialogButtonBox
from matplotlib.collections import LineCollection
from shapely import get_parts
import numpy as np

from autopyro.model.core import parse_number
from autopyro.schemes.interpretation import GeometricEngine, STATUSES, transform
from autopyro.schemes.scheme_geometry import (
    coverage_domain,
    geometry_diagnostics,
    geometry_paths,
    vertices,
)

from ..i18n import tr, message


def scope_samples_changed(sheet):
    try:
        row_ids = sheet.project_owner.analysis_scope.rows()
        parent = sheet.parentWidget()
        if hasattr(parent, 'geometry_samples') and parent.scheme is not None:
            sheet.samples = parent.geometry_samples(row_ids=row_ids[:2000])
        else:
            selected = set(row_ids)
            sheet.samples = [sample for sample in sheet._source_samples if sample.row_id in selected]
        sheet.preview_results = []
        sheet.redraw()
    except (ValueError, KeyError, TypeError) as exc:
        sheet.samples = []
        sheet.preview_results = []
        sheet.redraw()
        sheet.diagnostics.setText(message(exc))


def restore_parameters(sheet):
    sheet.show_samples.setChecked(bool(sheet.samples))
    sheet.fit_view()


def redraw(sheet, *args, preview_only=False):
    if not hasattr(sheet, "ax"):
        return
    xlim, ylim = sheet.ax.get_xlim(), sheet.ax.get_ylim()
    sheet.ax.clear()
    for collection in ("regions", "lines"):
        for entry in sheet.draft.get(collection, []):
            for path in geometry_paths(entry):
                values = (
                    sheet.drag["values"]
                    if sheet.drag
                    and sheet.selected == (collection, entry["id"])
                    and path == sheet.selected_path
                    else vertices(entry, path)
                )
                displayed = values + [values[0]] if path.closed else values
                selected = (
                    sheet.selected == (collection, entry["id"])
                    and path == sheet.selected_path
                )
                color = entry.get("color", "#0072B2")
                try:
                    sheet.ax.plot(
                        [p[0] for p in displayed],
                        [p[1] for p in displayed],
                        color=color,
                        linewidth=2 if selected else 1,
                        marker="o" if selected else None,
                        markersize=5,
                        linestyle="--"
                        if path.closed and path.indices[-1] > 0
                        else "-",
                    )
                except ValueError:
                    sheet.ax.plot(
                        [p[0] for p in displayed],
                        [p[1] for p in displayed],
                        color="#777777",
                    )
                if selected:
                    for index, point in enumerate(values):
                        sheet.ax.annotate(
                            str(index + 1),
                            point,
                            xytext=(4, 4),
                            textcoords="offset points",
                            fontsize=8,
                        )
                    index = sheet.vertex_table.currentRow()
                    if 0 <= index < len(values):
                        sheet.ax.scatter(
                            *values[index],
                            s=90,
                            facecolors="none",
                            edgecolors="#d62728",
                            zorder=5,
                        )
    if sheet.drawing_points:
        sheet.ax.plot(
            [p[0] for p in sheet.drawing_points],
            [p[1] for p in sheet.drawing_points],
            "o--",
            color="#d62728",
        )
    diagnostics = None
    if not preview_only:
        if sheet._diagnostics is None:
            sheet._diagnostics = geometry_diagnostics(sheet.draft)
        diagnostics = sheet._diagnostics
        if sheet.show_coverage.isChecked() and diagnostics.get("coverage"):
            sheet.draw_coverage(diagnostics["coverage"])
    preview_note = ""
    if sheet.show_samples.isChecked():
        subset = sheet.samples[:2000]
        if not preview_only:
            sheet.preview_results = (
                GeometricEngine().evaluate(subset, sheet.draft)
                if not diagnostics["errors"]
                else []
            )
        status_by_id = (
            {}
            if preview_only
            else {item.row_id: item.status for item in sheet.preview_results}
        )
        colors = {
            "classified": "#009E73",
            "boundary": "#E69F00",
            "ambiguous": "#CC79A7",
            "outside": "#888888",
        }
        points = []
        for sample in subset:
            try:
                points.append(
                    (
                        transform(sample.x, sheet.draft["x"]),
                        transform(sample.y, sheet.draft["y"]),
                        colors.get(status_by_id.get(sample.row_id), "#777777"),
                    )
                )
            except (ValueError, TypeError, OverflowError):
                continue
        if points:
            sheet.ax.scatter(
                [p[0] for p in points],
                [p[1] for p in points],
                c=[p[2] for p in points],
                s=20,
                zorder=3,
            )
        preview_note = (
            tr(' · Предпросмотр: показано {0} из {1} проб', len(points), len(sheet.samples))
        )
        if not preview_only and sheet.preview_results:
            counts = Counter(item.status for item in sheet.preview_results)
            preview_note += " · " + "; ".join(
                f"{tr(STATUSES[status])}: {count}" for status, count in counts.items()
            )
    for dimension in ("x", "y"):
        axis = sheet.draft[dimension]
        (sheet.ax.set_xlabel if dimension == "x" else sheet.ax.set_ylabel)(
            f"{axis['property']} · analysis ({axis.get('transform', 'identity')})"
        )
    sheet.ax.grid(True, alpha=0.25)
    sheet.ax.set_xlim(xlim)
    sheet.ax.set_ylim(ylim)
    sheet.canvas.draw_idle()
    if preview_only:
        sheet.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        sheet.diagnostics.setText(
            tr('Незавершённый контур: {0} вершин. Проверка после завершения переноса / контура.', len(sheet.drawing_points))
        )
    else:
        sheet.validation = diagnostics
        warning_text = "; ".join(message(warning) for warning in diagnostics["warnings"][:5])
        if len(diagnostics["warnings"]) > 5:
            warning_text += (
                tr('; ещё предупреждений: {0}', len(diagnostics['warnings']) - 5)
            )
        sheet.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            not diagnostics["errors"] and not sheet.drawing_points
        )
        sheet.diagnostics.setText(
            (
                tr('Ошибка: ') + "; ".join(message(error) for error in diagnostics["errors"])
                if diagnostics["errors"]
                else tr('Геометрия корректна')
            )
            + ("\n" + warning_text if warning_text else "")
            + sheet.coverage_note(diagnostics)
            + preview_note
        )


def draw_coverage(sheet, coverage):
    for key, color, style in (
        ("domain", "#0072b2", "--"),
        ("gaps", "#d55e00", "-"),
        ("outside", "#cc3311", ":"),
    ):
        geometry = coverage[key]
        if not geometry.is_empty:
            segments = [np.asarray(part.coords) for part in get_parts(geometry.boundary)]
            sheet.ax.add_collection(LineCollection(segments, colors=color, linestyles=style,
                                                  linewidths=1.6, zorder=2), autolim=False)


def coverage_note(sheet, diagnostics):
    coverage = diagnostics.get("coverage")
    if coverage:
        domain, gaps, outside = (coverage[key] for key in ("domain", "gaps", "outside"))
        covered = 100 * (1 - gaps.area / domain.area)
        return "\n" + tr(
            'Покрытие домена {0} %. Разрывов {1}, площадь {2} analysis². Вне домена {3} analysis².',
            f"{covered:.6g}", len(get_parts(gaps)) if not gaps.is_empty else 0,
            f"{gaps.area:.6g}", f"{outside.area:.6g}",
        )
    if not diagnostics["errors"] and sheet.draft.get("regions") and sheet.draft.get("strategy", "regions") == "regions":
        return "\n" + tr('Для проверки покрытия задайте конечные границы обеих осей.')
    return ""


def fit_view(sheet):
    points = []
    if sheet.show_coverage.isChecked():
        domain = coverage_domain(sheet.draft)
        if domain is not None:
            points.extend(domain.exterior.coords)
    for collection in ("regions", "lines"):
        for entry in sheet.draft.get(collection, []):
            for path in geometry_paths(entry):
                points.extend(vertices(entry, path))
    for sample in sheet.samples[:2000]:
        try:
            points.append(
                [
                    transform(sample.x, sheet.draft["x"]),
                    transform(sample.y, sheet.draft["y"]),
                ]
            )
        except (ValueError, TypeError, OverflowError):
            pass
    if points:
        values = np.array(points, dtype=float)
        low, high = values.min(axis=0), values.max(axis=0)
        pad = np.maximum((high - low) * 0.08, np.maximum(np.abs(high), 1) * 0.01)
        sheet.ax.set_xlim(low[0] - pad[0], high[0] + pad[0])
        sheet.ax.set_ylim(low[1] - pad[1], high[1] + pad[1])
    else:
        sheet.ax.set_xlim(0, 1)
        sheet.ax.set_ylim(0, 1)
    sheet.update_limit_fields()
    sheet.canvas.draw_idle()


def update_limit_fields(sheet):
    for widget, value in zip(
        sheet.limits, (*sheet.ax.get_xlim(), *sheet.ax.get_ylim()), strict=True
    ):
        widget.setText(f"{value:.12g}")


def apply_limits(sheet):
    try:
        xmin, xmax, ymin, ymax = [
            parse_number(widget.text()) for widget in sheet.limits
        ]
        if (
            any(value is None for value in (xmin, xmax, ymin, ymax))
            or xmin >= xmax
            or ymin >= ymax
        ):
            raise ValueError(tr('Задайте конечные пределы: минимум < максимума'))
        sheet.ax.set_xlim(xmin, xmax)
        sheet.ax.set_ylim(ymin, ymax)
        sheet.canvas.draw_idle()
    except (ValueError, TypeError) as exc:
        sheet.diagnostics.setText(message(exc))
