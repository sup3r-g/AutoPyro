"""Frozen, self-contained analytical reports; no Qt and no network resources."""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import os
import tempfile
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from itertools import zip_longest
from pathlib import Path

from autopyro import __version__
from autopyro.analysis.calculations import recompute
from autopyro.model.core import Project, timestamp
from autopyro.model.sample_labels import sample_label, sample_label_fields
from autopyro.model.source_location import source_location_parts
from autopyro.schemes.interpretation import STATUSES, evaluate_project
from autopyro.visualization.plotting import PlotSpec, ensure_style_maps, render_plot
from autopyro.analysis.statistics import StatisticsOptions, calculate_statistics, report_table
from .correlations import correlation_blocks
from .interpretation_text import ENGLISH_NOTES, ENGLISH_STATUSES, english_reason
from .plot_notices import english_plot_notice


@dataclass(frozen=True)
class ReportOptions:
    column_ids: tuple[str, ...]
    group_ids: tuple[str, ...] = ()
    title: str = "Отчёт по данным пиролиза"
    author: str = ""
    conclusions: str = ""
    scope: str = "Текущая таблица"
    text_numbers: bool = False
    include_interpolated: bool = False
    include_interpretation: bool = True
    data_limit: int = 100
    include_stratigraphy: bool = True
    include_nonpyro: bool = True
    sections: tuple[str, ...] | None = None
    profile_depth_id: str | None = None
    logo_image: str | None = None
    section_titles: dict[str, str] = field(default_factory=dict)
    block_comments: dict[str, str] = field(default_factory=dict)
    language: str = "ru"
    joint_snapshot_id: str | None = None
    qc_open_only: bool = False
    qc_limit: int = 100
    protocol_limit: int = 100
    map_view_name: str | None = None
    detailed_correlations: bool = False


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def digest(report):
    return hashlib.sha256(canonical(report)).hexdigest()


def _well_scope_scheme(scheme, rows, well_id):
    wells = {row.values.get(well_id) for row in rows}
    return {**scheme, "intervals": [
        item for item in scheme["intervals"] if item["wellbore"] in wells
    ]}


def _release_checks(report, columns):
    """Check the frozen inputs and the sections actually prepared for release."""
    from autopyro.analysis.qc import role

    options = report["options"]
    enabled = options["sections"]
    total = len(report["row_ids"])
    english = options.get("language") == "en"
    warnings = []

    def add(code, message, *, severity="warning", hint=None, **details):
        check = {"code": code, "message": message, "severity": severity, **details}
        if hint:
            check["hint"] = hint
        warnings.append(check)

    if enabled is not None and (not enabled or not any(
        block[0] in enabled for block in _section_blocks(report)
    )):
        add("no_report_sections", "The report has no sections with content." if english
            else "В отчёте нет содержательных разделов.", severity="error",
            hint=("Open Settings → Sections and enable a section with data." if english
                  else "Откройте «Настройки → Разделы…» и включите раздел с данными."))
    omission = report.get("joint_curve_omission")
    if omission and omission["reason"] == "outside_scope":
        missing = omission["missing_rows"]
        add("joint_fit_outside_scope",
            f"The joint fit was omitted because {missing} fitted rows are outside the report selection." if english
            else f"Совместная подгонка пропущена: строк расчёта вне выборки отчёта — {missing}.",
            missing_rows=missing)
    elif omission:
        add("joint_fit_snapshot_missing", "The selected joint fit snapshot is unavailable." if english
            else "Выбранный снимок совместной подгонки недоступен.")
    if not total:
        add("empty_selection", "The selection is empty." if english else "Выборка пуста.",
            hint=("Check the selection, filters, and selected rows." if english
                  else "Проверьте поле «Выборка», фильтры и выделенные строки."))
        return warnings

    status = report.get("joint_curve_source_status")
    if status == "stale":
        add("stale_joint_fit", "The joint fit snapshot is stale. Source data or QC exclusions have changed." if english
            else "Снимок совместной подгонки устарел. Исходные данные или исключения QC изменились.")
    elif status == "unverified":
        add("unverified_joint_fit", "The joint fit snapshot could not be verified." if english
            else "Актуальность снимка совместной подгонки не подтверждена.")

    figures = report["figures"] if enabled is None or "figures" in enabled else []
    used = set(options["column_ids"])
    for figure in figures:
        used.update(figure["plotted"])
        used.update(figure["spec"].get(key) for key in ("x_id", "y_id", "z_id", "depth_bottom_id"))
        if not any(figure["plotted"].values()):
            name = figure["name"]
            add("empty_figure", f"The chart “{name}” contains no data." if english
                else f"На графике «{name}» нет данных.", figure=name,
                hint=("Check the selection and axis assignments in the chart settings, or deselect the chart." if english
                      else "Проверьте выборку и привязку осей в настройках графика или снимите отметку с графика."))
    available = Counter()
    for row in report["statistics"]["availability"]:
        available[row["column_id"]] += row["n"]
    unit_samples = {}
    for issue in report["qc"]["issues"]:
        if issue["rule_id"] == "unit" and issue["property_id"] in used:
            unit_samples.setdefault(issue["property_id"], set()).add(issue["sample_id"])
    selected = set(options["column_ids"])
    for cid, column in columns.items():
        name = column.name
        if cid not in used:
            continue
        if column.metadata.get("joint_curve_fit"):
            status = column.metadata.get("source_status", "unverified")
            if status == "stale":
                add("stale_joint_column", f"“{name}” is a stale joint fit column." if english
                    else f"«{name}» — устаревшая колонка совместной подгонки.", column_id=cid)
            elif status == "unverified":
                add("unverified_joint_column", f"Joint fit column “{name}” could not be verified." if english
                    else f"Актуальность колонки совместной подгонки «{name}» не подтверждена.", column_id=cid)
        if column.metadata.get("method") and column.metadata.get("stale"):
            add("stale_interpolation", f"Interpolation for “{name}” is stale and was excluded from analysis." if english
                else f"«{name}»: интерполяция устарела и исключена из анализа.", column_id=cid,
                hint=("Recalculate the interpolation in the project or remove this column from the report." if english
                      else "Пересчитайте интерполяцию в проекте или исключите эту колонку из отчёта."))
        elif cid in selected and not available[cid] and (
            column.kind == "number" or role(column)
        ) and not (column.metadata.get("method") and not options["include_interpolated"]):
            if column.kind != "number" and not options["text_numbers"]:
                hint = ("If this column contains numbers, enable numbers in text columns and rebuild the report."
                        if english else "Если колонка содержит числа, включите «Использовать числа в текстовых колонках» "
                        "и пересоберите отчёт.")
            else:
                hint = ("Check source values, selection filters, and QC exclusions for statistics." if english
                        else "Проверьте исходные значения, фильтры выборки и исключения QC для статистики.")
            add("no_numeric_data", f"No numeric values from “{name}” were included in the statistics." if english
                else f"«{name}»: в статистику не вошло ни одного числового значения.",
                column_id=cid, hint=hint)
        if cid in unit_samples:
            count = len(unit_samples[cid])
            add("unit_check", f"Check the units for “{name}”. Affected samples: {count}." if english
                else f"«{name}»: единицы измерения требуют проверки у {count} проб.",
                column_id=cid, sample_count=count,
                hint=("Check the source column units and the Quality control section." if english
                      else "Проверьте единицы исходной колонки и результаты в разделе «Контроль качества»."))
        if column.metadata.get("calculation"):
            invalid = sum(entry.get("status") in {"invalid_definition", "cyclic_dependency"}
                          for entry in column.metadata.get("diagnostics", {}).values())
            if invalid:
                add("calculation_binding",
                    (f"Check the calculation definition, inputs, and units for “{name}”. Affected samples: {invalid}."
                     if english else f"«{name}»: проверьте определение расчёта, его входы и единицы. "
                     f"Затронуто проб: {invalid}."), column_id=cid, sample_count=invalid)
    if enabled is not None and "profile" in enabled and not report.get("well_profiles"):
        add("empty_profile", "The depth profile could not be generated." if english else "Глубинный профиль не построен.",
            hint=("Check the profile depth, well assignment, and selected indicators." if english
                  else "Проверьте поле «Глубина профиля», привязку скважины и выбранные показатели."))
    if (enabled is not None and "comparison_figures" in enabled
        and len(report["statistics"]["group_row_ids"]) > 1 and not report.get("comparison_figures")):
        add("empty_comparison",
            "Comparison charts require numeric data for the same indicator in at least two groups." if english
            else "Графики сопоставления требуют числовых данных одного показателя в двух или более группах.",
            hint=("Check the grouping levels and indicator coverage in the comparison table." if english
                  else "Проверьте уровни группировки и покрытие показателей в таблице сопоставления."))
    limit = options["data_limit"]
    if (enabled is None or "data" in enabled) and limit < total:
        add("truncated_appendix", f"The report appendix shows {limit} of {total} samples." if english
            else f"В приложении отчёта показано {limit} из {total} проб.",
            severity="info", shown=limit, total=total,
            hint=("Increase the appendix row limit to show more samples." if english
                  else "Чтобы расширить приложение, увеличьте параметр «Строк в приложении»."))
    if (enabled is None or "quality" in enabled) and report.get("qc"):
        qc = report["qc"]
        qc_limit = options.get("qc_limit", limit)
        eligible = sum(not options.get("qc_open_only") or issue["decision"] == "open" for issue in qc["issues"])
        exclusions = len(qc.get("active_exclusions", ()))
        omitted = max(0, eligible - qc_limit) + max(0, exclusions - qc_limit)
        if omitted:
            add("truncated_qc", f"QC tables omit {omitted} rows." if english else
                f"В таблицах QC пропущено строк: {omitted}.",
                hint=("Increase the QC row limit." if english else "Увеличьте параметр «Строк в таблицах QC»."),
                omitted=omitted)
    if enabled is None or "nonpyro" in enabled:
        protocol_limit = options.get("protocol_limit", limit)
        omitted = sum(max(0, len(protocol["results"]) - protocol_limit) for protocol in report.get("nonpyro", ()))
        if omitted:
            add("truncated_protocol", f"Geochemistry protocol tables omit {omitted} rows." if english else
                f"В таблицах протоколов геохимии пропущено строк: {omitted}.",
                hint=("Increase the geochemistry protocol row limit." if english else
                      "Увеличьте параметр «Строк в протоколах геохимии»."),
                omitted=omitted)
    return warnings


def release_blockers(report):
    return [check for check in report.get("release_checks", ()) if check.get("severity") == "error"]


def release_check_text(check, language="ru"):
    label = ({"error": "Blocks release", "warning": "Warning", "info": "Note"} if language == "en"
             else {"error": "Блокирует выпуск", "warning": "Замечание", "info": "Справка"})
    text = f"{label[check.get('severity', 'warning')]} · {check['message']}"
    return f"{text}\n{check['hint']}" if check.get("hint") else text


@dataclass
class ReportSourceIndex:
    rows: dict
    measurements: dict


def index_report_source(project, scopes):
    """Index the requested scopes once on an isolated package snapshot."""
    by_sample = {}
    measurements = {}
    for ids in scopes:
        key = frozenset(ids)
        if key in measurements:
            continue
        measurements[key] = []
        for rid in key:
            by_sample.setdefault(rid, []).append(measurements[key])
    for mid, measurement in project.science.measurements.items():
        for bucket in by_sample.get(measurement.sample_id, ()):
            bucket.append(mid)
    return ReportSourceIndex(project.row_map(), measurements)


def _copy_report_rows(project, ids, rows, view, *, measurement_ids=None):
    return Project(
        id=project.id,
        columns=deepcopy(project.columns),
        rows=deepcopy([rows[rid] for rid in ids]),
        source=deepcopy(project.source),
        view=deepcopy(view),
        science=project.science.subset(
            ids, [column.id for column in project.columns],
            measurement_ids=measurement_ids,
        ),
    )


def freeze_report_source(project, ids, options, plots):
    """Keep an independent source for the preview and its later grouped exports."""
    from autopyro.model.snapshots import copy_project

    # Plot palettes and joint-fit provenance can depend on rows outside the selection.
    figures_enabled = options.sections is None or "figures" in options.sections
    if (len(ids) >= len(project.rows) or (plots and figures_enabled)
            or project.view.get("joint_curve_runs")):
        return copy_project(project)
    rows = project.row_map()
    if len(set(ids)) != len(ids) or any(rid not in rows for rid in ids):
        raise ValueError("Выборка содержит повторяющиеся или отсутствующие UUID")
    return _copy_report_rows(project, ids, rows, project.view)


def build_report(
    project, row_ids, options: ReportOptions, plots=(), *, selection_context=None, progress=None,
    source_index=None,
):
    """Recalculate on a private copy. Every figure uses the report's row scope.

    The JSON snapshot contains ALL columns for the selected rows, including
    original cells and calculation definitions. Rows outside the scope are omitted.
    Stored plot style maps are resolved against the full project before slicing.
    """
    english = options.language == "en"

    def update(russian, english_text, percent):
        if progress is not None:
            progress(english_text if english else russian, percent)

    update("Подготовка снимка данных…", "Preparing the data snapshot…", 0)
    ids = list(row_ids)
    rows = project.row_map() if source_index is None else source_index.rows
    selected_ids = frozenset(ids)
    if len(selected_ids) != len(ids) or not selected_ids.issubset(rows):
        raise ValueError("Выборка содержит повторяющиеся или отсутствующие UUID")
    enabled = set(REPORT_SECTIONS if options.sections is None else options.sections)
    from autopyro.curves.joint_sources import joint_report_snapshot, update_joint_status

    selected_fit = joint_report_snapshot(project, options.joint_snapshot_id) if "joint_curve_fit" in enabled else None
    joint_omission = None
    if selected_fit is not None:
        missing = sum(rid not in selected_ids for rid in selected_fit["row_ids"])
        if missing:
            joint_omission = {"reason": "outside_scope", "missing_rows": missing,
                              "snapshot_id": selected_fit.get("id") or selected_fit.get("run_id") or "legacy"}
            selected_fit = None
    elif "joint_curve_fit" in enabled and options.joint_snapshot_id is not None:
        joint_omission = {"reason": "missing_snapshot"}
    view = {key: value for key, value in project.view.items()
            if key not in ("joint_curve_report", "joint_curve_reports", "joint_curve_runs")}
    if selected_fit is not None:
        view["joint_curve_report"] = selected_fit
    plots = [(name, PlotSpec.from_dict(deepcopy(config))) for name, config in plots] if "figures" in enabled else []
    from autopyro.model.measurements import derived

    style_column_ids = set()
    for _, spec in plots:
        style_column_ids.update((spec.color_by, spec.shape_by, spec.size_by))
    # Derived categories must first be recalculated for every sample. Raw
    # categories can be read directly without copying their scientific records.
    needs_full_palette = any(
        derived(column) for column in project.columns
        if column.id in style_column_ids
    )
    if needs_full_palette:
        frozen = deepcopy(project, {id(project.view): deepcopy(view)})
    else:
        for _, spec in plots:
            ensure_style_maps(project, spec)
        frozen = _copy_report_rows(
            project, ids, rows, view,
            measurement_ids=source_index.measurements.get(selected_ids) if source_index else None,
        )
    update_joint_status(project, artifacts_project=frozen, rows_by_id=rows)
    update("Пересчёт показателей…", "Recalculating indicators…", 10)
    frozen.sync_measurements()
    columns = {c.id: c for c in frozen.columns}
    if not options.column_ids or set(options.column_ids + options.group_ids) - set(
        columns
    ):
        raise ValueError("Выберите существующие колонки отчёта")
    if any(not 0 <= limit <= 10000 for limit in (options.data_limit, options.qc_limit, options.protocol_limit)):
        raise ValueError("Число строк приложения должно быть от 0 до 10 000")
    recompute(frozen)
    from autopyro.analysis.nonpyro import recompute as recompute_nonpyro

    recompute_nonpyro(frozen)
    from autopyro.geology.stratigraphy import recompute as recompute_stratigraphy

    recompute_stratigraphy(frozen)
    update("Расчёт статистики…", "Calculating statistics…", 25)
    stats = calculate_statistics(
        frozen,
        ids,
        StatisticsOptions(
            options.column_ids,
            options.group_ids,
            options.text_numbers,
            options.include_interpolated,
            correlations="statistics" in enabled and len(options.column_ids) <= 30,
        ),
        language=options.language,
    )
    figures = []
    for index, (name, spec) in enumerate(plots):
        update(f"Построение графика · {name}", f"Building chart · {name}", 40 + 20 * index // len(plots))
        original_scope = spec.scope
        spec.scope = "report"
        # Preserve individual plotting policies and explicitly record them.
        ensure_style_maps(frozen, spec)
        if spec.width_mm * spec.height_mm * (spec.dpi / 25.4) ** 2 > 64_000_000:
            raise ValueError(f"{name}: размер изображения превышает 64 мегапикселя")
        figure, result = render_plot(frozen, ids, spec, language=options.language)
        stream = io.BytesIO()
        figure.savefig(stream, format="png", dpi=spec.dpi, transparent=spec.transparent)
        figure.clear()
        figure_spec = spec.to_dict()
        if layer := figure_spec.get("stratigraphy"):
            from autopyro.geology.stratigraphy import scheme_hash

            layer["source_scheme_hash"] = scheme_hash(layer["scheme"])
            layer["scheme"] = _well_scope_scheme(
                layer["scheme"], (rows[rid] for rid in ids), layer["binding"]["well_id"],
            )
        figures.append(
            {
                "name": name,
                "spec": figure_spec,
                "original_scope": original_scope,
                "row_ids": ids,
                "plotted": result["plotted"],
                "excluded": result["excluded"],
                "notices": result["notices"],
                "png": base64.b64encode(stream.getvalue()).decode("ascii"),
            }
        )
        if "curve_archive" in result:
            figures[-1]["curve_archive"] = deepcopy(result["curve_archive"])
    update("Интерпретация и стратиграфия…", "Preparing interpretation and stratigraphy…", 60)
    interpretation, interpretation_note = None, "Интерпретация не включена."
    if options.include_interpretation and "interpretation" in enabled:
        def check_interpretation_progress():
            update("Интерпретация и стратиграфия…", "Preparing interpretation and stratigraphy…", 60)

        settings = frozen.view.get("interpretation_current", {})
        scheme = frozen.view.get("interpretation_schemes", {}).get(
            settings.get("scheme_id")
        )
        if scheme and all(
            key in settings for key in ("x_id", "y_id", "x_unit", "y_unit")
        ):
            interpretation = evaluate_project(
                frozen,
                ids,
                settings["x_id"],
                settings["y_id"],
                settings["x_unit"],
                settings["y_unit"],
                scheme,
                text_numbers=settings.get("text_numbers", False),
                checkpoint=check_interpretation_progress,
            )
            if frozen.view.get("interpretation_comparison"):
                from autopyro.schemes.interpretation_comparison import comparison_panels, comparison_reports, type_agreement
                from autopyro.schemes.standard_interpretation import standard_schemes

                binding = dict(interpretation["binding"], scheme=scheme)
                panels = comparison_panels(frozen, standard_schemes(), binding, binding["text_numbers"])
                comparisons = comparison_reports(frozen, ids, panels, primary_report=interpretation,
                                                 checkpoint=check_interpretation_progress)
                interpretation.update(comparison_panels=panels, comparisons=comparisons,
                                      type_agreement=type_agreement(comparisons, len(interpretation["results"])))
            from autopyro.schemes.interpretation_export import review_export

            interpretation["expert_reviews"] = [
                review_export(frozen, interpretation, i) for i in range(len(interpretation["results"]))
            ]
            interpretation_note = "Пересчитано геометрическим ядром для выборки отчёта."
        else:
            interpretation_note = "Нет сохранённой схемы с полной привязкой к колонкам; интерпретация не выполнена."
    row_map = frozen.row_map()
    nonpyro = []
    if options.include_nonpyro and "nonpyro" in enabled:
        from autopyro.analysis.nonpyro import evaluate_project as evaluate_nonpyro

        for config in frozen.view.get("nonpyro_configs", {}).values():
            nonpyro.append(evaluate_nonpyro(frozen, ids, config))
    selected = [row_map[rid] for rid in ids]
    stratigraphy = None
    if options.include_stratigraphy and "stratigraphy" in enabled and frozen.view.get("stratigraphy"):
        from autopyro.geology.stratigraphy import evaluate_project as evaluate_stratigraphy, scheme_hash
        from .well_profile import datum_matches, depth_datum_column

        settings = frozen.view["stratigraphy"]
        binding = settings["binding"]
        source_scheme = settings["scheme"]
        datum_id = depth_datum_column(columns, binding.get("depth_id"))
        eligible = [
            row for row in selected if not datum_id
            or datum_matches(row.values.get(datum_id), binding.get("depth_reference"))
        ]
        scoped_scheme = _well_scope_scheme(source_scheme, eligible, binding["well_id"])
        stratigraphy = evaluate_stratigraphy(frozen, [row.id for row in eligible], source_scheme, binding)
        stratigraphy["source_scheme_hash"] = stratigraphy["scheme_hash"]
        stratigraphy["scheme"] = scoped_scheme
        stratigraphy["scheme_hash"] = scheme_hash(scoped_scheme)
        stratigraphy["datum_skipped"] = len(selected) - len(eligible)
    # Diagnostics may contain identifiers from the whole project; limit these too.
    for column in frozen.columns:
        metadata = column.metadata
        diagnostics = metadata.get("diagnostics")
        if isinstance(diagnostics, dict):
            metadata["diagnostics"] = {
                rid: val for rid, val in diagnostics.items() if rid in selected_ids
            }
        if metadata.get("method"):
            scope = metadata.get("scope_row_ids")
            if scope is not None:
                metadata["scope_row_ids"] = [rid for rid in scope if rid in selected_ids]
            if metadata.get("interpolation_protocol") and (
                set(scope if scope is not None else rows) - selected_ids
            ):
                del metadata["interpolation_protocol"]
                metadata["interpolation_protocol_omitted"] = True
            if isinstance(metadata.get("proposals"), list):
                metadata["proposals"] = [
                    proposal for proposal in metadata["proposals"] if proposal["row_id"] in selected_ids
                ]
    update("Контроль качества…", "Checking data quality…", 75)
    from autopyro.analysis.qc import PROFILE_KEY, RULES, SCOPES, active_exclusions, pyrolysis_profile, run_checks, selected_exclusions
    science = frozen.science.subset(ids, columns)
    exclusions = selected_exclusions(frozen, science.measurements)
    current_exclusions = [
        {"scope": scope, "analysis": SCOPES.get(scope, scope), "sample_id": m.sample_id,
         "property": columns[m.property_id].name, "measurement_id": m.id, **deepcopy(record),
         "is_working": science.working.get(m.sample_id, {}).get(m.property_id) == m.id}
        for scope, m, record in active_exclusions(frozen, science.measurements)
    ]
    def qc_checkpoint(_stage, percent):
        update("Контроль качества…", "Checking data quality…", 75 + percent * 9 // 100)

    qc_issues = [issue for issue in run_checks(
        frozen, language=options.language, checkpoint=qc_checkpoint if progress is not None else None,
    ) if issue["sample_id"] in selected_ids]
    qc_profile = deepcopy(pyrolysis_profile(frozen))
    report_view = {"qc_exclusions": exclusions}
    if (sample_scope := qc_profile.get("sample_scope")) is not None:
        sample_scope["row_ids"] = [rid for rid in sample_scope["row_ids"] if rid in selected_ids]
        report_view[PROFILE_KEY[0]] = {PROFILE_KEY[1]: qc_profile}
    from autopyro.database.refresh import snapshot_status
    database_status = snapshot_status(frozen, ids)
    profiles = {}
    if options.sections and "profile" in options.sections:
        from autopyro.reports.well_profile import build_profiles

        images, notices = build_profiles(
            frozen, ids, options,
            checkpoint=lambda: update("Построение глубинных профилей…", "Building depth profiles…", 85),
        )
        profiles = {"well_profiles": images, "well_profile_notices": notices}
    update("Подготовка разделов отчёта…", "Preparing report sections…", 92)
    from .overview import comparison_figures, overview

    summary = overview(frozen, columns, selected, stats, options) if "summary" in enabled else {}
    comparison_images = []
    if options.sections and "comparison_figures" in options.sections:
        comparison_images = comparison_figures(
            stats, display, lambda: update("Построение графиков сопоставления…", "Building comparison charts…", 93),
            language=options.language,
        )
    map_figures = []
    if options.map_view_name is not None and "figures" in enabled:
        from autopyro.spatial.mapping import build_map, render_map, saved_map_layers

        name = options.map_view_name
        snapshot = frozen.view.get("map_views", {}).get(name)
        if snapshot is None:
            raise ValueError(f"Saved map is unavailable: {name}" if english else f"Сохранённая карта недоступна: {name}")
        spec, layers, missing = saved_map_layers(snapshot, frozen.view.get("map_layers", ()))
        if missing:
            raise ValueError(f"Saved map has missing layers: {name}" if english else f"В сохранённой карте отсутствуют слои: {name}")
        map_report = build_map(frozen, ids, spec, layers)
        figure, _ = render_map(map_report)
        stream = io.BytesIO()
        try:
            dpi = min(spec.dpi, 200)
            figure.savefig(stream, format="png", dpi=dpi)
        finally:
            figure.clear()
        map_figures.append({
            "name": f"Map · {name}" if english else f"Карта · {name}",
            "caption": ("Saved view rendered for the report selection" if english
                        else "Сохранённый вид на выборке отчёта"),
            "spec": map_report["spec"],
            "row_ids": map_report["row_ids"],
            "png": base64.b64encode(stream.getvalue()).decode("ascii"),
        })
    report = {
        "format": "autopyro.report",
        "version": 1,
        "created_at": timestamp(),
        "autopyro_version": __version__,
        "options": asdict(options),
        "selection_context": deepcopy(selection_context or {}),
        "project": {
            "id": frozen.id,
            "source": frozen.source,
            "columns": [asdict(c) for c in frozen.columns],
            "rows": [asdict(r) for r in selected],
            "science": asdict(science),
            "view": report_view,
        },
        "row_ids": ids,
        "statistics": stats,
        "overview": summary,
        "comparison_figures": comparison_images,
        "figures": figures,
        "map_figures": map_figures,
        **profiles,
        "stratigraphy": stratigraphy,
        "nonpyro": nonpyro,
        "interpretation": interpretation,
        "interpretation_note": interpretation_note,
        "qc": {"rules": deepcopy(RULES), "issues": qc_issues, "exclusions": exclusions,
               "active_exclusions": current_exclusions,
               "pyrolysis_profile": qc_profile},
        "database_snapshot": database_status,
    }
    if joint_omission:
        report["joint_curve_omission"] = joint_omission
    if joint_fit := frozen.view.get("joint_curve_report"):
        from .joint_fit_figures import render_joint_figures

        report["joint_curve_source_status"] = joint_fit["source_status"]
        report["joint_curve_fit"] = joint_fit
        report["joint_curve_figures"] = render_joint_figures(
            joint_fit, options.language,
            lambda name, index, total: update(f"Построение графика · {name}", f"Building chart · {name}",
                                               94 + 4 * index // total),
        )
    report["release_checks"] = _release_checks(report, columns)
    return report


def display(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


REPORT_SECTIONS = {
    "summary": "Краткая сводка",
    "source": "Выборка и происхождение",
    "method": "Методика",
    "conclusions": "Выводы автора",
    "quality": "Контроль качества",
    "nonpyro": "Геохимия вне пиролиза",
    "stratigraphy": "Стратиграфия",
    "profile": "Глубинный профиль скважины",
    "comparison": "Сопоставление объектов",
    "comparison_figures": "Графики сопоставления",
    "statistics": "Расчёты и статистика",
    "figures": "Графики",
    "joint_curve_fit": "Совместная подгонка",
    "interpretation": "Геометрическая интерпретация",
    "data": "Приложение: данные",
}

_ENGLISH_SECTION_TITLES = {
    "summary": "Summary", "source": "Selection and provenance", "method": "Methods",
    "conclusions": "Author conclusions", "quality": "Quality control", "nonpyro": "Non-pyrolysis geochemistry",
    "stratigraphy": "Stratigraphy", "profile": "Well depth profile", "comparison": "Object comparison",
    "comparison_figures": "Comparison charts", "statistics": "Calculations and statistics",
    "figures": "Figures", "joint_curve_fit": "Joint fit", "interpretation": "Geometric interpretation",
    "data": "Data appendix",
}

_ENGLISH_QUALITY_TITLES = {
    "Состояние снимка БД": "Database snapshot status",
    "Замечания QC": "QC findings",
    "Действующие исключения QC": "Active QC exclusions",
    "Лимит строк QC": "QC row limit",
    "Замечания к рассмотрению": "Findings to review",
    "Область порогового профиля": "Threshold profile scope",
}

_ENGLISH_STATISTICS_TITLES = {
    "Описательная статистика": "Descriptive statistics", "Качество данных": "Data availability",
    "Корреляции": "Correlations", "Выбросы IQR": "IQR outliers",
    "Пары без коэффициентов": "Pairs without coefficients",
}

_ENGLISH_QC_SCOPES = {
    "statistics": "Statistics", "calculations": "Pyrolysis calculations", "plots": "Charts and maps",
    "interpretation": "Geometric interpretation", "interpolation": "Interpolation",
}

_ENGLISH_STRAT_STATUSES = {
    "matched": "Matched", "ambiguous": "Overlapping intervals",
    "cross_boundary": "Crosses a unit boundary", "partial": "Partial coverage",
    "invalid_interval": "Invalid sample interval", "outside": "Outside intervals",
    "unknown_wellbore": "No intervals for this wellbore", "insufficient_data": "Insufficient data",
    "interpolation_excluded": "Interpolation excluded",
}

_ENGLISH_NONPYRO_STATUSES = {
    "interpreted": "Interpreted", "profile": "Profile calculated", "not_applicable": "Not applicable",
    "boundary": "On the boundary",
    "outside_scope": "Outside the saved selection", "interpolation_excluded": "Interpolated inputs excluded",
    "qc_excluded": "Excluded by QC", "missing": "Missing value", "non_detect": "Below detection limit",
    "censored": "Censored value", "other": "Non-numeric value", "invalid_range": "Outside the allowed range",
    "numerical_error": "Numerical error", "zero_denominator": "Zero denominator",
    "invalid_definition": "Invalid definition",
}


def _nonpyro_interpretation(item):
    text, status = item["interpretation"], item["status"]
    if status in ("outside_scope", "interpolation_excluded"):
        return _ENGLISH_NONPYRO_STATUSES[status]
    if status == "numerical_error" and text == "Численное переполнение":
        return "Numerical overflow"
    if status == "invalid_range" and text == "Количество измерений должно быть целым и положительным":
        return "The measurement count must be a positive integer"
    return _nonpyro_input_diagnostic(text)


def _nonpyro_input_diagnostic(text):
    field, separator, reason = text.rpartition(": ")
    if separator:
        if reason == "вне допустимого диапазона":
            reason = "invalid_range"
        if reason in _ENGLISH_NONPYRO_STATUSES:
            return f"{field}: {_ENGLISH_NONPYRO_STATUSES[reason]}"
    from .nonpyro_text import english_finding

    return english_finding(text)


def report_language(report):
    return "en" if report.get("options", {}).get("language") == "en" else "ru"


def sections(report, *, paged=False):
    for _, kind, title, content in _section_entries(report, paged=paged):
        yield kind, title, content


def _section_entries(report, *, paged=False):
    language = report_language(report)
    english = language == "en"
    if report.get("release_checks"):
        yield "release", "text", "Pre-release checks" if english else "Проверка перед выпуском", "\n".join(
            f"• {release_check_text(check, language)}" for check in report["release_checks"]
        )
    options = report["options"]
    titles = options.get("section_titles", {})
    comments = options.get("block_comments", {})
    seen = set()
    for section, key, kind, title, content in report_blocks(report, localized=True):
        if paged and kind == "table":
            # Use the original block type before applying the author's heading.
            identifiers = 1
            if section == "comparison":
                identifiers = 3  # object, indicator and unit
            elif section == "statistics":
                identifiers = len(report["statistics"]["groups"]) + 2
            elif title in ("Замечания QC", "QC findings"):
                identifiers = 4  # severity, sample, indicator and source
            elif title in ("Действующие исключения QC", "Active QC exclusions"):
                identifiers = 4  # sample, indicator, analysis and measurement
            content = (*content, identifiers)
        first = section not in seen
        if first:
            seen.add(section)
        if first and section in titles:
            title = titles[section]
        yield section if first else None, kind, title, content
        if comments.get(key):
            yield None, "text", "Author comment" if english else "Комментарий автора", comments[key]


def table_panels(headers, rows, identifiers, width):
    """Keep the complete row identity together in each horizontal panel."""
    if len(headers) <= width:
        yield headers, _paged_table_rows(rows, identifiers)
        return
    if identifiers > 1:
        labels = headers[:identifiers]
        headers = [" / ".join(labels), *headers[identifiers:]]
        rows = [
            ["\n".join(f"{label}: {display(value)}" for label, value in zip(labels, row)), *row[identifiers:]]
            for row in rows
        ]
    for start in range(1, max(2, len(headers)), width - 1):
        columns = [0, *range(start, min(start + width - 1, len(headers)))]
        panel_rows = ([row[i] for i in columns] for row in rows)
        yield [headers[i] for i in columns], _paged_table_rows(panel_rows, 1)


def _paged_table_rows(rows, identifiers):
    # Split long remarks before pagination so every continuation keeps its row identity.
    for row in rows:
        identity = list(row[:identifiers])
        chunks = []
        for value in row[identifiers:]:
            if not isinstance(value, str) or len(value) <= 360 and value.count("\n") <= 10:
                chunks.append([value])
                continue
            parts = []
            while len(value) > 360 or value.count("\n") > 10:
                end = min(360, len(value))
                lines = value[:end].splitlines(keepends=True)
                if len(lines) > 11:
                    end = sum(map(len, lines[:11]))
                elif end < len(value):
                    boundary = max(value.rfind(" ", end // 2, end), value.rfind("\n", end // 2, end))
                    if boundary >= 0:
                        end = boundary + 1
                parts.append(value[:end])
                value = value[end:]
            chunks.append(parts + [value])
        for values in zip_longest(*chunks, fillvalue="") if chunks else [()]:
            yield identity + list(values)


def _english_block_title(section, kind, title, nonpyro_names):
    if title == REPORT_SECTIONS.get(section):
        title = _ENGLISH_SECTION_TITLES[section]
    elif section == "quality":
        title = _ENGLISH_QUALITY_TITLES.get(title, title)
    elif section == "nonpyro":
        prefix, translated = (("Геохимия вне пиролиза · ", "Non-pyrolysis geochemistry · ") if kind == "text"
                              else ("Результаты · ", "Results · "))
        name = title.removeprefix(prefix)
        title = translated + nonpyro_names.get(name, name)
    elif section == "stratigraphy":
        if title == "Стратиграфическая привязка":
            title = "Stratigraphic assignment"
        elif title.startswith("Стратиграфия: "):
            title = "Stratigraphy · " + title.removeprefix("Стратиграфия: ").title()
        elif title.startswith("Пересечения проб: "):
            title = "Sample intersections · " + title.removeprefix("Пересечения проб: ").title()
    elif section == "statistics":
        if title.startswith("Расчётная колонка: "):
            title = "Calculated column: " + title.removeprefix("Расчётная колонка: ")
        else:
            title = _ENGLISH_STATISTICS_TITLES.get(title, title)
    elif section == "comparison" and title == "Порядок группировки":
        title = "Grouping order"
    elif section == "comparison_figures" and title.startswith("Сопоставление медиан: "):
        title = "Median comparison: " + title.removeprefix("Сопоставление медиан: ")
    elif section == "profile" and title.startswith("Глубинный профиль: "):
        title = "Depth profile: " + title.removeprefix("Глубинный профиль: ")
    elif section == "interpretation":
        title = {"Схема интерпретации": "Interpretation scheme",
                 "Результаты интерпретации": "Interpretation results"}.get(title, title)
    return title


def report_blocks(report, *, localized=False):
    """Enumerate visible blocks using identifiers independent of custom titles."""
    english = localized and report_language(report) == "en"
    nonpyro_names = {}
    if english and report.get("nonpyro"):
        from .nonpyro_text import english_method_text

        nonpyro_names = {item["definition"]["name"]: english_method_text(item)[0] for item in report["nonpyro"]}
    order = report["options"].get("sections")
    if order is None:
        blocks = _section_blocks(report)
    else:
        grouped = {key: [] for key in order}
        for block in _section_blocks(report):
            if block[0] in grouped:
                grouped[block[0]].append(block)
        blocks = (block for group in grouped.values() for block in group)
    seen = Counter()
    for section, kind, title, content in blocks:
        base = section, kind, title
        key = canonical((*base, seen[base])).decode("utf-8") if kind in ("table", "figure") else None
        seen[base] += 1
        if english:
            title = _english_block_title(section, kind, title, nonpyro_names)
        yield section, key, kind, title, content


def _summary_blocks(report, total, options, english):
    available = Counter()
    for row in report["statistics"]["availability"]:
        available[row["column_id"]] += row["n"]
    issues = {issue["sample_id"] for issue in report.get("qc", {}).get("issues", [])}
    overview = report.get("overview", {})
    detail = []
    if depth := overview.get("depth"):
        unit = f" {depth['unit']}" if depth["unit"] else ""
        span = f"{display(depth['minimum'])}–{display(depth['maximum'])}{unit}"
        detail.append(f"Depth ({depth['column']}): {span}. Samples with depth: {depth['samples']} of {total}."
                      if english else f"Глубина ({depth['column']}): {span}. Проб с глубиной: {depth['samples']} из {total}.")
    elif overview.get("mixed_depth_datums"):
        detail.append("The selection uses different depth datums. See the individual profiles for their ranges."
                      if english else "В выборке разные точки отсчёта глубины. Диапазоны показаны на отдельных профилях.")
    for indicator in overview.get("indicators", []):
        mean, low, high = (display(indicator[key]) for key in ("mean", "minimum", "maximum"))
        prefix = f"{indicator['name']} [{indicator['unit']}]"
        detail.append(
            f"{prefix}: mean {mean}, range {low}–{high}." if english
            else f"{prefix}: среднее {mean}, диапазон {low}–{high}."
        )
    qc_count = f"{len(issues)} {'of' if english else 'из'} {total} ({len(issues) / total:.1%})"
    intro = (f"Samples in selection: {total}. Samples with QC findings: {qc_count}." if english
             else f"Проб в выборке: {total}. Проб с замечаниями QC: {qc_count}.")
    counts = "Numeric values used in the statistics:" if english else "Числовые значения, использованные в статистике:"
    yield (
        "summary", "text", "Краткая сводка",
        "\n".join([intro, *detail, counts, *(
            f"{column['name']} [{column['unit']}]: {available[column['id']]} {'of' if english else 'из'} {total}"
            for column in report["project"]["columns"] if column["id"] in options["column_ids"]
        )]),
    )


def _quality_blocks(report, options, english):
    section = "quality"
    if report.get("database_snapshot"):
        status = report["database_snapshot"]
        removed = [record["sample_id"] for record in status["records"] if not record["present"]]
        source = display(status["source"])
        yield section, "text", "Состояние снимка БД", (
            f"Checked: {status['checked_at']}. Applied: {status['applied_at']}.\n"
            "This reflects the source at the time it was read. The source is not monitored continuously.\n"
            f"Missing from the source and retained locally: {', '.join(removed) or 'none'}.\n"
            f"Source: {source}" if english else
            f"Проверен: {status['checked_at']}; применён: {status['applied_at']}.\n"
            "Это состояние на момент чтения, без постоянного наблюдения за источником.\n"
            f"Отсутствуют в источнике и сохранены локально: {', '.join(removed) or 'нет'}.\n"
            f"Источник: {source}")
    if report.get("qc"):
        from autopyro.analysis.qc import QC_SEVERITY_ORDER, exclusion_time, rule_description

        issues = report["qc"]["issues"]
        open_only = options.get("qc_open_only", False)
        eligible_issues = [item for item in issues if item["decision"] == "open"] if open_only else issues
        current_exclusions = report["qc"].get("active_exclusions", [])
        limit = options.get("qc_limit", options["data_limit"])
        shown_issues = []
        if limit:
            shown_issues = sorted(
                eligible_issues, key=lambda item: QC_SEVERITY_ORDER.get(item["severity"], len(QC_SEVERITY_ORDER)),
            )[:limit]
        shown_exclusions = current_exclusions[:limit]
        yield section, "text", "Контроль качества", (
            f"Findings: {len(issues)}. Source values are not corrected automatically. "
            "Exclusions apply to the specified analyses and measurement revisions. QC decisions are saved in JSON."
            if english else f"Замечаний: {len(issues)}. Исходные значения не исправляются автоматически. "
            "Исключения относятся только к указанным видам анализа и версиям измерений; решения сохранены в JSON.")
        if open_only:
            pending, accepted = len(eligible_issues), len(issues) - len(eligible_issues)
            yield section, "text", "Замечания к рассмотрению", (
                f"To review: {pending}. Accepted findings omitted from the table: {accepted}. JSON contains all findings."
                if english else f"К рассмотрению: {pending}. Принятых замечаний вне таблицы: {accepted}. Полный список доступен в JSON.")
        profile = report["qc"].get("pyrolysis_profile", {})
        if profile.get("enabled") and (scope := profile.get("sample_scope")) is not None:
            count = len(scope["row_ids"])
            yield section, "text", "Область порогового профиля", (
                f"Fixed sample set: {scope['name']}. Samples in this report within the profile scope: {count}."
                if english else f"Зафиксированный набор: {scope['name']}. Проб отчёта в области профиля: {count}.")
        omitted = []
        if len(shown_issues) < len(eligible_issues):
            shown, total = len(shown_issues), len(eligible_issues)
            omitted.append(f"QC findings: {shown} of {total} shown, {total - shown} omitted."
                           if english else f"Замечания QC: показано {shown} из {total}, пропущено {total - shown}.")
        if len(shown_exclusions) < len(current_exclusions):
            shown, total = len(shown_exclusions), len(current_exclusions)
            omitted.append(f"Active QC exclusions: {shown} of {total} shown, {total - shown} omitted."
                           if english else f"Действующие исключения QC: показано {shown} из {total}, пропущено {total - shown}.")
        if omitted:
            omitted.append("The JSON report contains the complete lists."
                           if english else "Полные списки доступны в JSON-отчёте.")
            yield section, "text", "Лимит строк QC", "\n".join(omitted)
        labels = {}
        if shown_issues or shown_exclusions:
            snapshot = report["project"]
            fields = sample_label_fields((column["id"], column["name"]) for column in snapshot["columns"])
            labels = {row["id"]: sample_label(row["id"], row["values"], fields) for row in snapshot["rows"]}
            duplicates = Counter(labels.values())
            labels = {rid: f"{label} · {rid}" if label != rid and duplicates[label] > 1 else label
                      for rid, label in labels.items()}
        if shown_issues:
            saved_rules = report["qc"].get("rules", {})
            headers = (["Severity", "Sample", "Indicator", "Source", "Check", "Description", "Decision time", "Decision", "Comment"]
                       if english else ["Уровень", "Проба", "Показатель", "Источник", "Проверка", "Описание",
                                        "Время решения", "Решение", "Комментарий"])
            source_labels = ({"file": "File", "database": "Database", "source": "Source", "sheet": "Sheet",
                              "table": "Table", "row": "Row", "record": "Source record"} if english else
                             {"file": "Файл", "database": "База", "source": "Источник", "sheet": "Лист",
                              "table": "Таблица", "row": "Строка", "record": "Запись источника"})
            indicator_names = {column["id"]: column["name"] for column in report["project"]["columns"]}
            severity = ({"warning": "Warning", "error": "Error", "critical": "Critical"} if english else
                        {"warning": "Предупреждение", "error": "Ошибка", "critical": "Критическое предупреждение"})
            decisions = ({"open": "Open", "acknowledged": "Acknowledged"} if english else
                         {"open": "Не рассмотрено", "acknowledged": "Принято"})
            yield section, "table", "Замечания QC", (headers,
                [[severity.get(i["severity"], i["severity"]), labels.get(i["sample_id"], i["sample_id"]),
                  indicator_names.get(i["property_id"], i["property_id"]),
                  " · ".join(f"{source_labels[key]}: {value}"
                             for key, value in source_location_parts(i.get("source") or {})) or "—",
                  f"{rule_description(i['rule_id'], saved_rules, english=english)} · v{i['rule_version']}",
                  i["message"], exclusion_time(i.get("decision_at")),
                  decisions.get(i["decision"], i["decision"]), i["comment"]]
                 for i in shown_issues])
        if shown_exclusions:
            headers = (["Sample", "Indicator", "Analysis", "Measurement", "Working result", "Revision", "Set at", "Reason"]
                       if english else ["Проба", "Показатель", "Вид анализа", "Измерение", "Рабочий результат",
                                        "Версия", "Установлено", "Причина"])
            working_labels = ({True: "Yes", False: "No"} if english else {True: "Да", False: "Нет"})
            yield section, "table", "Действующие исключения QC", (
                headers,
                [[labels.get(item["sample_id"], item["sample_id"]), item["property"],
                  _ENGLISH_QC_SCOPES.get(item.get("scope"), item["analysis"]) if english else item["analysis"],
                  item["measurement_id"],
                  working_labels.get(item.get("is_working"), "—"),
                  item["revision"], exclusion_time(item.get("at")), item.get("comment", "")]
                 for item in shown_exclusions],
            )


def _section_blocks(report):
    """Shared content for HTML and PDF. Wide tables are split by the PDF writer."""
    options = report["options"]
    language = report_language(report)
    english = language == "en"
    order = options.get("sections")
    enabled = set(REPORT_SECTIONS if order is None else order)
    total = len(report["row_ids"])
    if total and "summary" in enabled:
        yield from _summary_blocks(report, total, options, english)
    section = "source"
    yield (
        section,
        "text",
        "Выборка и происхождение",
        (
            (f"Project: {report['project']['id']}\nDate (UTC): {report['created_at']}\n"
             f"Author: {options['author']}\nSelection: {options['scope']}\nSamples: {total}\n"
             f"Filters and sorting at build time: {display(report.get('selection_context', {}))}\n"
             f"Source: {display(report['project']['source'])}\nAutoPyro: {report['autopyro_version']}\n"
             f"JSON snapshot SHA-256: {digest(report)}") if english else
            f"Проект: {report['project']['id']}\nДата UTC: {report['created_at']}\n"
            f"Автор: {options['author']}\nОбласть: {options['scope']}\nПроб: {len(report['row_ids'])}\n"
            f"Фильтры и сортировка на момент сборки: {display(report.get('selection_context', {}))}\n"
            f"Источник: {display(report['project']['source'])}\nAutoPyro: {report['autopyro_version']}\n"
            f"SHA-256 снимка JSON: {digest(report)}"
        ),
    )
    section = "method"
    if english:
        method = [
            f"Numbers in text columns: {options['text_numbers']}. Interpolated values in statistics: {options['include_interpolated']}.",
            "Values marked <, >, or ND are tracked separately and are not replaced by numbers. Stale interpolations are excluded.",
            "SD uses n−1. Quantiles use linear interpolation. Outliers are flagged at 1.5 IQR and retained. Correlations use pairwise complete observations.",
            "Standard charts are rebuilt for the report selection with their numeric and interpolation settings. Joint fit charts use the saved snapshot.",
            f"Statistics engine versions: {display(report['statistics']['versions'])}",
        ]
    else:
        method = [
            f"Числа в текстовых колонках: {options['text_numbers']}; интерполяции в статистике: {options['include_interpolated']}.",
            "Значения < / > и ND учитываются отдельно и не подменяются числами. Устаревшие интерполяции исключены.",
            "SD: выборочное (n-1); квантили: linear; выбросы: 1.5 IQR, без удаления; корреляции: попарно полные наблюдения.",
            "Обычные графики перестроены для выборки отчёта с сохранением настроек чисел и интерполяций. "
            "Графики совместной подгонки построены по сохранённому снимку.",
            f"Версии статистического ядра: {display(report['statistics']['versions'])}",
        ]
    if "statistics" not in enabled:
        method.append("Correlations were not calculated because the statistics section is disabled." if english
                      else "Корреляции не рассчитаны: раздел статистики выключен.")
    elif not report["statistics"]["options"]["correlations"]:
        method.append("Correlations were disabled because more than 30 columns were selected." if english
                      else "Корреляции отключены: выбрано более 30 колонок.")
    method.append("JSON includes all columns, original cells for the selected samples, calculation definitions, settings, and results."
                  if english else "Экспорт JSON содержит все колонки, исходные ячейки выбранных проб, определения расчётов, настройки и результаты.")
    method.extend(report["statistics"]["notices"])
    if any(column["metadata"].get("interpolation_protocol_omitted") for column in report["project"]["columns"]):
        method.append("The full interpolation protocol is available in the project. It includes samples outside the report selection."
                      if english else "Полный протокол интерполяции доступен в проекте: он включает пробы за пределами отчёта.")
    yield section, "text", "Методика", "\n".join(method)
    section = "conclusions"
    if options["conclusions"].strip():
        yield section, "text", "Выводы автора", options["conclusions"]
    yield from _quality_blocks(report, options, english)
    section = "nonpyro"
    for protocol in report.get("nonpyro", []):
        definition = protocol["definition"]
        summary = protocol["summary"]
        protocol_limit = options.get("protocol_limit", options["data_limit"])
        note = definition["note"]
        if english:
            from .nonpyro_text import english_method_text

            _, note = english_method_text(protocol)
            summary = {_ENGLISH_NONPYRO_STATUSES.get(status, status): count for status, count in summary.items()}
        yield (
            section,
            "text",
            "Геохимия вне пиролиза · " + definition["name"],
            (
                f"Context: {protocol['config']['context']}\n{note}\n"
                f"Sources: {display(definition['sources'])}\nMethod SHA-256: {protocol['definition_hash']}\n"
                f"Results: {display(summary)}\n"
                "Recalculated for the report selection using the method's applicability rules. JSON contains the full protocol and its inputs.\n"
                f"The table shows up to {protocol_limit} protocol rows."
                if english else f"Контекст: {protocol['config']['context']}\n{definition['note']}\n"
                f"Источники: {display(definition['sources'])}\nSHA-256 методики: {protocol['definition_hash']}\n"
                f"Результаты: {display(summary)}\n"
                "Пересчитано для выборки отчёта с сохранением области применимости. Полный протокол и входы находятся в JSON.\n"
                f"Ниже не более {protocol_limit} строк протокола."
            ),
        )
        yield (
            section,
            "table",
            "Результаты · " + definition["name"],
            (
                ["UUID", "Status", "Indicators", "Interpretation", "QC"] if english
                else ["UUID", "Статус", "Показатели", "Интерпретация", "QC"],
                [
                    [
                        item["row_id"],
                        _ENGLISH_NONPYRO_STATUSES.get(item["status"], item["status"]) if english else item["status"],
                        display(item["values"]),
                        _nonpyro_interpretation(item) if english else item["interpretation"],
                        display([_nonpyro_input_diagnostic(warning) for warning in item["warnings"]]
                                if english else item["warnings"]),
                    ]
                    for item in protocol["results"][:protocol_limit]
                ],
            ),
        )
    section = "stratigraphy"
    if report.get("stratigraphy"):
        from autopyro.geology.stratigraphy import LEVELS, STATUSES as STRAT_STATUSES

        strat = report["stratigraphy"]
        scheme = strat["scheme"]
        interval = strat.get("sample_mode", "point") == "interval"
        unit = strat["binding"]["depth_unit"]
        status_labels = _ENGLISH_STRAT_STATUSES if english else STRAT_STATUSES
        if english:
            notes = [
                f"{scheme['name']} | ID: {scheme['id']} | Revision: {scheme['revision']}",
                f"Interval depths: {scheme['depth_unit']}, {scheme['depth_reference']}",
                f"Sample depths: {unit}, {strat['binding']['depth_reference']}",
                f"Source: {display(scheme.get('source', ''))}",
                f"Report scheme SHA-256: {strat['scheme_hash']}",
            ]
            if strat.get("source_scheme_hash"):
                notes.append(f"Source scheme SHA-256: {strat['source_scheme_hash']}")
            if strat.get("datum_skipped"):
                notes.append(f"Samples without a compatible depth datum: {strat['datum_skipped']}.")
            notes.append(
                "An interval sample [top, base) is assigned only when fully and unambiguously contained. Lengths and coverage use the sample depth unit."
                if interval else "A point sample includes the top boundary and excludes the base boundary."
            )
            notes.append("Wellbores are matched by exact name. MD, TVD, and depth datums are not converted. Overlaps at the same level are not resolved automatically.")
        yield (
            section,
            "text",
            "Стратиграфическая привязка",
            "\n".join(notes) if english else (
                f"{scheme['name']} | ID: {scheme['id']} | редакция: {scheme['revision']}\n"
                f"Глубины интервалов: {scheme['depth_unit']}; {scheme['depth_reference']}\n"
                f"Глубины проб: {strat['binding']['depth_unit']}; {strat['binding']['depth_reference']}\n"
                f"Источник: {display(scheme.get('source', ''))}\n"
                f"SHA-256 схемы в отчёте: {strat['scheme_hash']}\n"
                + (f"SHA-256 исходной схемы: {strat['source_scheme_hash']}\n"
                   if strat.get("source_scheme_hash") else "")
                + (f"Проб без совместимого datum: {strat['datum_skipped']}.\n"
                   if strat.get("datum_skipped") else "")
                + (
                    "Интервальные пробы [от, до): попадание только при полном однозначном включении; длины и покрытие — в единице проб.\n"
                    if strat.get("sample_mode", "point") == "interval"
                    else "Точечная глубина; кровля включается, подошва исключается.\n"
                )
                + "Стволы сопоставляются по точному имени.\n"
                "MD/TVD и начало отсчёта не преобразуются. Перекрытия одного уровня не разрешаются автоматически."
            ),
        )
        for level in LEVELS:
            headers = ["Sample UUID", "Wellbore", f"Depth / top [{unit}]", "Units", "Status"] if english else [
                "UUID пробы",
                "Ствол",
                f"Глубина / верх [{unit}]",
                "Подразделения",
                "Статус",
            ]
            if interval:
                headers += ([
                    f"Base [{unit}]", "Coverage fraction (0–1)", "Covered length", "Uncovered length", "Ambiguous length",
                ] if english else [
                    f"Низ [{unit}]",
                    "Доля покрытия (0–1)",
                    "Покрытая длина",
                    "Не покрыто",
                    "Длина неоднозначности",
                ])
            rows = []
            for row in strat["results"]:
                result = row["levels"][level]
                values = [
                    row["row_id"],
                    row["wellbore"],
                    row["depth"],
                    result["names"],
                    status_labels[result["status"]],
                ]
                if interval:
                    values += [
                        row["bottom"],
                        result["coverage_fraction"],
                        result["covered_length"],
                        result["uncovered_length"],
                        result["ambiguous_length"],
                    ]
                rows.append(values)
            yield section, "table", f"Стратиграфия: {level}", (headers, rows)
            if interval:
                intersections = [
                    [
                        row["row_id"],
                        item["name"],
                        item["interval_id"],
                        item["length"],
                        item["fraction"],
                    ]
                    for row in strat["results"]
                    for item in row["levels"][level]["overlaps"]
                ]
                if intersections:
                    yield (
                        section,
                        "table",
                        f"Пересечения проб: {level}",
                        (
                            ["Sample UUID", "Unit", "Interval ID", f"Length [{unit}]", "Fraction of sample length"]
                            if english else [
                                "UUID пробы",
                                "Подразделение",
                                "ID интервала",
                                f"Длина [{unit}]",
                                "Доля длины пробы",
                            ],
                            intersections,
                        ),
                    )
    if report.get("well_profile_notices"):
        yield "profile", "text", REPORT_SECTIONS["profile"], "\n".join(report["well_profile_notices"])
    for profile in report.get("well_profiles", ()):
        yield "profile", "figure", profile["name"], profile
    statistics = report["statistics"]
    if "comparison" in enabled and statistics["groups"] and statistics["summary"]:
        yield (
            "comparison",
            "table",
            REPORT_SECTIONS["comparison"],
            (
                ["Object", "Indicator", "Unit", "Samples", "Numeric N", "Mean",
                 "Median", "Min", "Max", "Coverage, %"] if english else
                ["Объект", "Показатель", "Единица", "Проб", "N чисел", "Среднее",
                 "Медиана", "Мин.", "Макс.", "Покрытие, %"],
                [
                    [" / ".join(display(value) if value != "" else "—" for value in row["group"])]
                    + [row[key] for key in (
                        "property", "unit", "rows", "n", "mean", "median", "min", "max", "coverage_pct",
                    )]
                    for row in statistics["summary"]
                ],
            ),
        )
        yield (
            "comparison", "text", "Порядок группировки",
            " → ".join(statistics["groups"]),
        )
    if options.get("sections") and "comparison_figures" in options["sections"]:
        figures = report.get("comparison_figures", [])
        if not figures and len(statistics["group_row_ids"]) > 1:
            yield "comparison_figures", "text", REPORT_SECTIONS["comparison_figures"], (
                "The chart requires numeric data for the same indicator in at least two groups."
                if english else "Для графика нужны числовые данные одного показателя хотя бы в двух группах."
            )
        for figure in figures:
            yield "comparison_figures", "figure", figure["name"], figure
    section = "statistics"
    calculated = [
        c for c in report["project"]["columns"]
        if "statistics" in enabled and c["metadata"].get("calculation")
    ]
    for column in calculated:
        metadata = column["metadata"]
        counts = Counter(
            entry["status"] for entry in metadata.get("diagnostics", {}).values()
        )
        config = deepcopy(metadata["calculation"])
        scope_ids = config.pop("scope_row_ids", None)
        scope = len(scope_ids) if scope_ids is not None else ("all samples" if english else "все пробы")
        yield (
            section,
            "text",
            f"Расчётная колонка: {column['name']}",
            (
                f"Formula: {metadata.get('equation', '')}\nSource: {metadata.get('scientific_source', '')}\n"
                f"Definition: {display(config)}\nCalculation scope: {scope}\n"
                f"Diagnostics for the report selection: {display(dict(counts))}" if english else
                f"Формула: {metadata.get('equation', '')}\nИсточник: {metadata.get('scientific_source', '')}\n"
                f"Определение: {display(config)}\n"
                f"Область расчёта: {scope}\n"
                f"Диагностика выборки отчёта: {display(dict(counts))}"
            ),
        )
    for key, title in [
        ("summary", "Описательная статистика"),
        ("availability", "Качество данных"),
    ]:
        if "statistics" in enabled:
            yield section, "table", title, report_table(report["statistics"], key, language=language)
    if "statistics" in enabled:
        statistics = report["statistics"]
        # Existing snapshots recorded every pair in the document.
        for kind, title, content in correlation_blocks(
                statistics, language, detailed=options.get("detailed_correlations", True)):
            yield section, kind, title, content
        yield section, "table", "Выбросы IQR", report_table(statistics, "outliers", language=language)
    section = "figures"
    for figure in report["figures"]:
        named_figure = {
            **figure,
            "column_names": {c["id"]: c["name"] for c in report["project"]["columns"]},
        }
        yield section, "figure", figure["name"], named_figure
    for figure in report.get("map_figures", ()):
        yield section, "figure", figure["name"], figure
    if joint_fit := report.get("joint_curve_fit"):
        yield from _joint_fit_blocks(joint_fit, options.get("language", "ru"))
        for figure in report.get("joint_curve_figures", ()):
            yield "joint_curve_fit", "figure", figure["name"], figure
    section = "interpretation"
    note = report["interpretation_note"]
    yield section, "text", "Геометрическая интерпретация", ENGLISH_NOTES.get(note, note) if english else note
    interpretation = report["interpretation"]
    if interpretation:
        scheme = interpretation["scheme"]
        status_labels = ENGLISH_STATUSES if english else STATUSES
        reviews = interpretation.get("expert_reviews", ())
        review_labels = {
            "pending": ("Требует проверки", "Review required"),
            "open": ("Вопрос открыт", "Open"),
            "accepted": ("Автоматическая оценка принята", "Automatic result accepted"),
            "expert": ("Экспертный вывод", "Expert conclusion"),
            "stale": ("Предыдущее решение. Требуется повторная проверка", "Previous decision. Review required"),
        }
        panels = interpretation.get("comparison_panels", ())
        comparisons = interpretation.get("comparisons", {})
        yield (
            section,
            "text",
            "Схема интерпретации",
            (
                f"{scheme['name']} | ID: {scheme['id']} | Revision: {scheme['revision']}\n"
                f"Source: {scheme['source']}\nScientific status declared by the author: {scheme['scientific_status']}\n"
                f"SHA-256: {interpretation['scheme_hash']}\nEngine: {interpretation['engine']}. Shapely: {interpretation['shapely_version']}\n"
                "The demonstration status marks an educational scheme. The validated status is a claim made by the scheme author.\n"
                "Results apply to the geometry of the selected scheme. Interpolated columns are excluded."
                if english else f"{scheme['name']} | ID: {scheme['id']} | версия: {scheme['revision']}\n"
                f"Источник: {scheme['source']}\nСтатус, заявленный автором: {scheme['scientific_status']}\n"
                f"SHA-256: {interpretation['scheme_hash']}\nЯдро: {interpretation['engine']}; Shapely: {interpretation['shapely_version']}\n"
                "Статус demonstration означает учебную схему. Статус validated является заявлением автора схемы.\n"
                "Результаты относятся к геометрии выбранной схемы; интерполированные колонки исключены."
            ),
        )
        yield (
            section,
            "table",
            "Результаты интерпретации",
            (
                (["Sample UUID", "Status", "Regions", "Boundaries", "Lines", "Reason"] if english
                 else ["UUID пробы", "Статус", "Области", "Границы", "Линии", "Причина"])
                + [panel["name"] for panel in panels],
                [
                    [
                        r["row_id"],
                        status_labels[r["status"]],
                        r["regions"],
                        r["boundary_regions"],
                        r["line_hits"],
                        english_reason(r["reason"]) if english else r["reason"],
                    ] + [
                        status_labels[comparisons[panel["name"]]["results"][i]["status"]]
                        if panel["name"] in comparisons else panel.get("error", "")
                        for panel in panels
                    ]
                    for i, r in enumerate(interpretation["results"])
                ],
            ),
        )
        review_rows = []
        for item in reviews:
            decision = item["decision"] or {}
            if not item["state"] and not item["pyrogram_observation"]:
                continue
            state = review_labels.get(item["state"], ("", ""))[int(english)]
            review_rows.append([item["row_id"], state, decision.get("author", ""),
                                decision.get("conclusion", ""), decision.get("note", ""),
                                item["pyrogram_observation"]])
        if review_rows:
            yield section, "table", ("Expert review" if english else "Экспертная проверка"), (
                ["Sample UUID", "State", "Reviewer", "Conclusion", "Comment", "Pyrogram observation"] if english
                else ["UUID пробы", "Состояние", "Автор", "Вывод", "Комментарий", "Наблюдение по пирограмме"],
                review_rows,
            )
    if "data" not in enabled:
        return
    section = "data"
    columns = {c["id"]: c for c in report["project"]["columns"]}
    limit = options["data_limit"]
    total = len(report["row_ids"])
    shown = min(limit, total)
    yield (
        section,
        "text",
        "Приложение: данные",
        (
            f"Showing {shown} of {total} samples in selection order. "
            "JSON contains all sample UUIDs and values. An em dash indicates an empty cell."
            if english else
            f"Показано {shown} из {total} проб в порядке выборки. "
            "Полный список UUID и значения выборки доступны в JSON. Пустая ячейка обозначена символом —."
        ),
    )
    if limit:
        yield (
            section,
            "table",
            "Selected columns" if english else "Таблица выбранных колонок",
            (
                ["Sample UUID" if english else "UUID пробы"]
                + [
                    f"{columns[cid]['name']} [{columns[cid]['unit']}]"
                    for cid in options["column_ids"]
                ],
                [
                    [r["id"]] + [r["values"][cid] for cid in options["column_ids"]]
                    for r in report["project"]["rows"][:limit]
                ],
            ),
        )


def _joint_fit_blocks(fit, language):
    section = "joint_curve_fit"
    english = language == "en"
    title = "Joint fit" if english else REPORT_SECTIONS[section]
    settings = fit["settings"]
    method = "WLS" if settings.get("sy") else "OLS"
    loss = settings["loss"]
    model = f"{method} · {loss}"
    if loss != "linear":
        model += f" (f_scale={display(settings['f_scale'])})"
    saved = "Saved fit" if english else "Сохранённый расчёт"
    shared = "Shared coefficients" if english else "Общие коэффициенты"
    summary = [f"{saved} · {fit['saved_at']}", f"X = {fit['axes'][0]} · Y = {fit['axes'][1]}",
               f"{fit['formula']} · {model}", f"{shared} · {', '.join(settings['shared'])}"]
    if bootstrap := fit["bootstrap"]:
        simultaneous = fit.get("band_method_actual", settings.get("band_method")) == "bootstrap_simultaneous"
        if simultaneous:
            summary.append("Bootstrap with simultaneous confidence band" if english else "Bootstrap с одновременной доверительной полосой")
        else:
            summary.append("Percentile bootstrap" if english else "Перцентильный bootstrap")
        label = "Successful bootstrap replicates" if english else "Успешные повторы bootstrap"
        summary.append(f"{label} · {bootstrap['successful']}/{bootstrap['resamples']}")
        if bootstrap.get("warnings"):
            summary.append("Few replicates in the distribution tails. Increase the number of bootstrap replicates."
                           if english else "Мало повторов в хвостах распределения; увеличьте число bootstrap-повторов.")
    elif fit["bootstrap_failed"]:
        summary.append("Bootstrap bands unavailable" if english else "Полосы bootstrap недоступны")
    else:
        summary.append("Bands were not calculated" if english else "Полосы не рассчитаны")
    yield section, "text", title, "\n".join(summary)
    metrics = fit["metrics"]
    yield section, "table", "Fit metrics" if english else "Метрики подгонки", (list(metrics), [list(metrics.values())])
    parameters = fit["parameters"]
    headers = ["Series" if english else "Серия", "N", *parameters, "RMSE", "MAE"]
    rows = [[label, count, *beta, rmse, mae] for label, count, beta, rmse, mae in
            zip(fit["series"], fit["counts"], fit["coefficients"], fit["rmse"], fit["mae"])]
    if fit["active_bounds"] is not None:
        headers.append("Active bounds" if english else "Активные границы")
        for row, mask in zip(rows, fit["active_bounds"]):
            row.append(", ".join(f"{name} {'↑' if side > 0 else '↓'}"
                                 for name, side in zip(parameters, mask) if side))
    yield section, "table", "Fit by series" if english else "Подгонка по сериям", (headers, rows)


def figure_caption(figure, language="ru"):
    if "caption" in figure:
        return figure["caption"]
    if "status_counts" in figure:
        return " ".join(figure["notices"])
    spec = figure["spec"]
    names = figure.get("column_names", {})
    plotted = {names.get(cid, cid): count for cid, count in figure["plotted"].items()}
    excluded = {
        names.get(cid, cid): counts for cid, counts in figure["excluded"].items()
    }
    if language == "en":
        interval = ""
        if depth_bottom := spec.get("depth_bottom_id"):
            interval = (f"Sampling intervals: {names.get(spec['y_id'], spec['y_id'])} to "
                        f"{names.get(depth_bottom, depth_bottom)}. Midpoints are not used. ")
        return (f"{interval}Plotted: {display(plotted)}. Excluded: {display(excluded)}. "
                f"Numbers in text: {spec['text_numbers']}. Interpolated values: {spec['include_interpolated']}. "
                + " ".join(english_plot_notice(notice) for notice in figure["notices"]))
    interval_note = (
        f"Интервалы отбора: {names.get(spec['y_id'], spec['y_id'])} — {names.get(spec['depth_bottom_id'], spec['depth_bottom_id'])}; середина не используется. "
        if spec.get("depth_bottom_id")
        else ""
    )
    return (
        interval_note
        + f"Построено: {display(plotted)}; исключено: {display(excluded)}. "
        f"Числа в тексте: {spec['text_numbers']}; интерполяции: {spec['include_interpolated']}. "
        + " ".join(figure["notices"])
    )


def render_html(report, *, navigation=None):
    esc = lambda value: html.escape(display(value), quote=True)
    language = report_language(report)
    defaults = _ENGLISH_SECTION_TITLES if language == "en" else REPORT_SECTIONS
    parts = [
        f'<!doctype html><html lang="{language}"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{esc(report['options']['title'])}</title>",
        "<style>body{font:14px sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#152b40}"
        "table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccd5df;padding:6px;text-align:left}"
        "th{background:#eaf0f5}p{white-space:pre-wrap;overflow-wrap:anywhere}.table{overflow-x:auto}"
        "img{max-width:100%;height:auto}h2{margin-top:28px}@media print{body{margin:0}.table{overflow:visible}"
        "thead{display:table-header-group}tr,img{break-inside:avoid}}</style></head><body>",
        f"<h1>{'<a name="report-top"></a>' if navigation is not None else ''}"
        f"{esc(report['options']['title'])}</h1>",
    ]
    for index, (section, kind, title, content) in enumerate(_section_entries(report)):
        anchor = ""
        if navigation is not None and section is not None:
            name = f"report-{section}"
            label = report["options"].get("section_titles", {}).get(section, defaults.get(section, title))
            navigation.append((name, label))
            anchor = f'<a name="{name}"></a>'
            if kind in ("figure", "table") and title != label:
                navigation.append((name, title))
        elif navigation is not None and kind in ("figure", "table"):
            name = f"report-block-{index}"
            navigation.append((name, title))
            anchor = f'<a name="{name}"></a>'
        parts.append(f"<h2>{anchor}{esc(title)}</h2>")
        if kind == "text":
            parts.append(f"<p>{esc(content)}</p>")
        elif kind == "figure":
            parts.append(
                f'<img alt="{esc(title)}" src="data:image/png;base64,{content["png"]}">'
                f'<p>{esc(figure_caption(content, language))}</p>'
            )
        else:
            headers, rows = content
            parts.append(
                '<div class="table"><table><thead><tr>'
                + "".join(f"<th>{esc(h)}</th>" for h in headers)
                + "</tr></thead><tbody>"
            )
            parts.extend(
                "<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>"
                for row in rows
            )
            parts.append("</tbody></table></div>")
    return "".join(parts) + "</body></html>"


def render_pdf(report):
    from matplotlib import get_data_path
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Image,
        LongTable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    fonts = Path(get_data_path()) / "fonts" / "ttf"
    for face, file in [
        ("AutoPyro", "DejaVuSans.ttf"),
        ("AutoPyroBold", "DejaVuSans-Bold.ttf"),
    ]:
        if face not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(face, str(fonts / file)))
    body = ParagraphStyle(
        "body",
        fontName="AutoPyro",
        fontSize=9,
        leading=13,
        spaceAfter=7,
        splitLongWords=True,
    )
    cell = ParagraphStyle(
        "cell", parent=body, fontSize=7, leading=10, spaceAfter=0, alignment=TA_LEFT
    )
    heading = ParagraphStyle(
        "heading",
        parent=body,
        fontName="AutoPyroBold",
        fontSize=13,
        leading=17,
        spaceBefore=12,
        keepWithNext=True,
    )
    title_style = ParagraphStyle(
        "report_title", parent=heading, fontSize=19, leading=24
    )

    class TableParagraph(Paragraph):
        _wrapped_width = None

        def wrap(self, availWidth, availHeight):
            # ReportLab measures each cell for layout and drawing. Reuse the
            # layout at the same width unless splitting has discarded it.
            if self._wrapped_width != availWidth or not hasattr(self, "blPara"):
                self._wrapped_size = super().wrap(availWidth, availHeight)
                self._wrapped_width = availWidth
            return self._wrapped_size

    # All caller strings are literal text, including ReportLab markup-like values.
    def paragraph(value, style=body):
        text = display(value).replace("—", "-").replace("−", "-")
        paragraph_type = TableParagraph if style is cell else Paragraph
        return paragraph_type(html.escape(text).replace("\n", "<br/>"), style)

    stream = io.BytesIO()
    language = report_language(report)
    class ReportDocument(SimpleDocTemplate):
        def afterFlowable(self, flowable):
            if hasattr(flowable, "outline_key"):
                self.canv.bookmarkPage(flowable.outline_key)
                self.canv.addOutlineEntry(flowable.getPlainText(), flowable.outline_key)

    doc = ReportDocument(
        stream,
        pagesize=A4,
        leftMargin=17 * mm,
        rightMargin=17 * mm,
        topMargin=17 * mm,
        bottomMargin=18 * mm,
        title=report["options"]["title"],
        author=report["options"]["author"],
        lang=language,
    )
    story = [paragraph(report["options"]["title"], title_style)]
    for index, (kind, title, content) in enumerate(sections(report, paged=True)):
        section_heading = paragraph(title, heading)
        section_heading.outline_key = f"report-block-{index}"
        story.append(section_heading)
        if kind == "text":
            # Short blocks let even long user conclusions span pages safely.
            for line in content.splitlines():
                story.append(paragraph(line))
        elif kind == "figure":
            image = Image(io.BytesIO(base64.b64decode(content["png"])))
            ratio = min(doc.width / image.imageWidth, 155 * mm / image.imageHeight)
            image.drawWidth, image.drawHeight = (
                image.imageWidth * ratio,
                image.imageHeight * ratio,
            )
            story.extend([image, paragraph(figure_caption(content, language))])
        else:
            headers, rows, identifiers = content
            if not rows:
                story.append(paragraph("No records." if language == "en" else "Нет записей."))
                continue
            for panel_headers, panel_rows in table_panels(headers, rows, identifiers, 5):
                matrix = [[paragraph(value, cell) for value in panel_headers]]
                matrix += [[paragraph(value, cell) for value in row] for row in panel_rows]
                widths = [doc.width / len(panel_headers)] * len(panel_headers)
                if identifiers > 1 and len(headers) > 5 and len(panel_headers) > 2:
                    widths = [doc.width * 0.34] + [doc.width * 0.66 / (len(panel_headers) - 1)] * (len(panel_headers) - 1)
                table = LongTable(
                    matrix,
                    colWidths=widths,
                    repeatRows=1,
                    hAlign="LEFT",
                )
                table.setStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf0f5")),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#ccd5df")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
                story.extend([table, Spacer(1, 3 * mm)])

    short_hash = digest(report)[:16]

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("AutoPyro", 7)
        canvas.drawString(17 * mm, 10 * mm, f"AutoPyro | {short_hash}")
        canvas.drawRightString(A4[0] - 17 * mm, 10 * mm, str(document.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return stream.getvalue()


def encode_report(report, format):
    if blockers := release_blockers(report):
        raise ValueError("\n".join(release_check_text(check, report_language(report)) for check in blockers))
    if format == "html":
        return render_html(report).encode("utf-8")
    if format == "pdf":
        return render_pdf(report)
    if format == "json":
        return canonical(report)
    if format == "docx":
        from .word import render_docx

        return render_docx(report)
    raise ValueError("Неизвестный формат отчёта")


def export_report(path, report, format):
    """Atomic single-file export."""
    if format == "json" and report.get("format") == "autopyro.nonpyro":
        if blockers := release_blockers(report):
            raise ValueError("\n".join(release_check_text(check) for check in blockers))
        from autopyro.model.protocol_json import protocol_json_chunks

        write_report_file(path, protocol_json_chunks(report, sort_keys=True))
        return
    write_report_file(path, encode_report(report, format))


def write_report_file(path, payload):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            if isinstance(payload, bytes):
                stream.write(payload)
            else:
                stream.writelines(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
