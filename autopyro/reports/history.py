"""Read and compare immutable report snapshots."""

import hashlib
import json
from pathlib import Path

from .reporting import canonical, digest, display, report_language


_JOINT_FIT_LABEL = "Снимок совместной подгонки"
_REPORT_LANGUAGE_LABEL = "Язык отчёта"
_CORRELATIONS_LABEL = "Все пары корреляций"
_LANGUAGE_NAMES = {
    "ru": {"ru": "Русский", "en": "Английский"},
    "en": {"ru": "Russian", "en": "English"},
}


def snapshot_location(path, project_path=None):
    path = Path(path).resolve()
    location = {"path": str(path)}
    if project_path:
        try:
            location["relative_path"] = path.relative_to(Path(project_path).resolve().parent).as_posix()
        except ValueError:
            pass
    return location


def snapshot_path(entry, project_path=None):
    if project_path and entry.get("relative_path"):
        relative = Path(entry["relative_path"])
        if not relative.is_absolute() and not relative.drive and ".." not in relative.parts:
            candidate = Path(project_path).resolve().parent / relative
            if candidate.exists():
                return candidate
    return Path(entry["path"])


def load_snapshot(entry, project_id, project_path=None):
    report = json.loads(snapshot_path(entry, project_path).read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not isinstance(report.get("project"), dict):
        raise ValueError("Файл не является поддерживаемым снимком этого проекта.")
    if (report.get("format"), report.get("version"), report["project"].get("id")) != (
        "autopyro.report", 1, project_id,
    ):
        raise ValueError("Файл не является поддерживаемым снимком этого проекта.")
    if digest(report) != entry["sha256"]:
        raise ValueError("Контрольная сумма не совпадает с записью в истории отчётов.")
    return report


def compare_snapshots(before, after):
    """Compare by stable sample and column IDs, and by the ordered group key."""
    if before["project"]["id"] != after["project"]["id"]:
        raise ValueError("Для сравнения нужны отчёты одного проекта.")

    def samples(report):
        project = report["project"]
        science = project.get("science", {})
        working = science.get("working", {})
        measurements = science.get("measurements", {})
        return {
            row["id"]: (row, {cid: measurements.get(mid) for cid, mid in working.get(row["id"], {}).items()})
            for row in project["rows"]
        }

    def statistics(report):
        group_ids = report["options"]["group_ids"]
        return {
            canonical([group_ids, row["group"], row["column_id"]]): row
            for row in report["statistics"]["summary"]
        }

    def decisions(report):
        return {issue["id"]: (issue["sample_id"], issue["rule_id"], issue.get("decision", "open"),
                              issue.get("comment", ""), issue.get("decision_at"))
                for issue in report.get("qc", {}).get("issues", ())}

    def exclusions(report):
        return {(scope, mid): record for scope, records in report.get("qc", {}).get("exclusions", {}).items()
                for mid, record in records.items()}

    def changes(old, new):
        return [(key, old.get(key), new.get(key))
                for key in dict.fromkeys((*old, *new)) if old.get(key) != new.get(key)]

    old_samples, new_samples = samples(before), samples(after)
    return {
        "added": [rid for rid in new_samples if rid not in old_samples],
        "removed": [rid for rid in old_samples if rid not in new_samples],
        "changed": [rid for rid in new_samples if rid in old_samples and old_samples[rid] != new_samples[rid]],
        "statistics": [(old, new) for _, old, new in changes(statistics(before), statistics(after))],
        "qc_decisions": changes(decisions(before), decisions(after)),
        "exclusions": changes(exclusions(before), exclusions(after)),
        "schemes": changes(
            {key: _scheme(before.get(key)) for key in ("stratigraphy", "interpretation")},
            {key: _scheme(after.get(key)) for key in ("stratigraphy", "interpretation")},
        ),
        "policies": changes(
            _comparison_policy(before), _comparison_policy(after),
        ),
        "composition": changes(_composition(before), _composition(after)),
    }


def _scheme(section):
    if section:
        scheme = section["scheme"]
        return {"name": scheme["name"], "revision": scheme["revision"],
                "sha256": section.get("source_scheme_hash", section["scheme_hash"])}
    return None


def _comparison_policy(report):
    options = report["options"]
    return {
        "group_ids": list(options["group_ids"]),
        "column_ids": list(options["column_ids"]),
        "text_numbers": options["text_numbers"],
        "include_interpolated": options["include_interpolated"],
        "selection_context": report.get("selection_context", {}),
    }


def _composition(report):
    options = report["options"]
    columns = {column["id"]: column["name"] for column in report["project"].get("columns", [])}
    logo = options.get("logo_image")
    depth_id = options.get("profile_depth_id")
    sections = options.get("sections")
    return {
        "Заголовок": options.get("title"),
        "Автор": options.get("author"),
        _REPORT_LANGUAGE_LABEL: report_language(report),
        "Разделы и порядок": list(sections) if sections is not None else None,
        "Заголовки разделов": options.get("section_titles", {}),
        "Комментарии к таблицам и рисункам": options.get("block_comments", {}),
        "Выводы автора": options.get("conclusions", ""),
        "Лимит строк приложения": options.get("data_limit"),
        _CORRELATIONS_LABEL: options.get("detailed_correlations", True),
        "Колонка глубины профиля": {"id": depth_id, "name": columns.get(depth_id)} if depth_id else None,
        "Привязка стратиграфии": (report.get("stratigraphy") or {}).get("binding"),
        "Стратиграфия": options.get("include_stratigraphy", True),
        "Интерпретация": options.get("include_interpretation", True),
        "Геохимия вне пиролиза": options.get("include_nonpyro", True),
        "Логотип (SHA-256)": hashlib.sha256(logo.encode("ascii")).hexdigest() if logo else None,
        "Графики и их настройки": [_plot_signature(figure) for figure in report.get("figures", ())],
        _JOINT_FIT_LABEL: _joint_fit_signature(report),
    }


def _joint_fit_signature(report):
    """Summarize the selected fit without comparing plot images or raw arrays."""
    if fit := report.get("joint_curve_fit"):
        return {
            "id": fit.get("id") or fit.get("run_id"),
            "saved_at": fit.get("saved_at"),
            "formula": fit.get("formula"),
            "confidence": (fit.get("plots") or {}).get("confidence") if fit.get("bootstrap") else None,
            "source_status": fit.get("source_status") or report.get("joint_curve_source_status"),
        }
    if omission := report.get("joint_curve_omission"):
        return {
            "id": omission.get("snapshot_id") or report["options"].get("joint_snapshot_id"),
            "omission": omission["reason"],
            "missing_rows": omission.get("missing_rows"),
        }
    return None


def _joint_fit_text(signature, english):
    if signature is None:
        return display(None)
    parts = [signature["id"]] if signature["id"] else []
    if reason := signature.get("omission"):
        if reason == "outside_scope":
            count = signature["missing_rows"]
            parts.append(f"Omitted. Fitted rows outside the selection: {count}" if english
                         else f"Пропущен. Строк подгонки вне выборки: {count}")
        else:
            parts.append("Selected snapshot unavailable" if english else "Выбранный снимок недоступен")
    else:
        parts.extend(value for key in ("saved_at", "formula") if (value := signature.get(key)))
        if (confidence := signature.get("confidence")) is not None:
            parts.append(f"{'Confidence' if english else 'Доверие'} {confidence*100:g}%")
        statuses = ({"current": "Current", "stale": "Stale", "unverified": "Unverified"} if english
                    else {"current": "Актуален", "stale": "Устарел", "unverified": "Актуальность не подтверждена"})
        if status := signature.get("source_status"):
            parts.append(statuses.get(status, status))
    return " · ".join(parts) or display(None)


def _plot_signature(figure):
    spec = dict(figure["spec"])
    layer = spec.get("stratigraphy")
    if layer and layer.get("source_scheme_hash"):
        spec["stratigraphy"] = {key: value for key, value in layer.items() if key != "scheme"}
    if "curve_archive" in figure:
        spec["curve_archive"] = figure["curve_archive"]
    return {
        "name": figure["name"],
        "sha256": hashlib.sha256(canonical([figure.get("original_scope"), spec])).hexdigest()[:12],
    }


_ENGLISH_POLICY_LABELS = {
    "group_ids": "Grouping columns", "column_ids": "Selected columns",
    "text_numbers": "Numbers in text columns", "include_interpolated": "Include interpolated values",
    "selection_context": "Selection context",
}

_ENGLISH_COMPOSITION_LABELS = {
    _JOINT_FIT_LABEL: "Joint fit snapshot",
    _REPORT_LANGUAGE_LABEL: "Report language",
    _CORRELATIONS_LABEL: "All correlation pairs",
    "Заголовок": "Title", "Автор": "Author", "Разделы и порядок": "Sections and order",
    "Заголовки разделов": "Section headings", "Комментарии к таблицам и рисункам": "Table and figure comments",
    "Выводы автора": "Author conclusions", "Лимит строк приложения": "Appendix row limit",
    "Колонка глубины профиля": "Profile depth column", "Привязка стратиграфии": "Stratigraphic assignment",
    "Стратиграфия": "Stratigraphy", "Интерпретация": "Interpretation",
    "Геохимия вне пиролиза": "Non-pyrolysis geochemistry", "Логотип (SHA-256)": "Logo (SHA-256)",
    "Графики и их настройки": "Charts and settings",
}

_ENGLISH_STATISTICS_LABELS = {
    "rows": "Samples", "n": "Numeric N", "mean": "Mean", "median": "Median",
    "min": "Min", "max": "Max", "std": "Standard deviation", "outliers": "Outliers",
    "missing": "Missing", "censored": "Censored", "other": "Text and other values",
    "excluded_qc": "Excluded by QC", "excluded_interpolation": "Interpolations excluded",
    "interpolated_used": "Interpolated values used", "coverage_pct": "Coverage, %",
    "unit": "Unit", "property": "Indicator",
}


def comparison_text(diff, *, language="ru"):
    english = language == "en"
    lines = []
    for key, russian, translated in (
        ("added", "Добавлены в выборку", "Added to the selection"),
        ("removed", "Исключены из выборки", "Removed from the selection"),
        ("changed", "Изменены значения или рабочие измерения", "Values or working measurements changed"),
    ):
        label = translated if english else russian
        lines.append(f"{label} ({len(diff[key])})")
        lines.extend(diff[key])

    def heading(russian, translated, entries):
        lines.append(f"{translated if english else russian} ({len(entries)})")

    heading("Изменения настроек", "Settings changes", diff["policies"])
    for key, old, new in diff["policies"]:
        label = _ENGLISH_POLICY_LABELS.get(key, key) if english else key
        lines.append(f"{label} · {display(old)} → {display(new)}")
    composition = diff.get("composition", ())
    heading("Изменения состава и оформления", "Content and layout changes", composition)
    for key, old, new in composition:
        label = _ENGLISH_COMPOSITION_LABELS.get(key, key) if english else key
        if key == _JOINT_FIT_LABEL:
            before, after = _joint_fit_text(old, english), _joint_fit_text(new, english)
        elif key == _REPORT_LANGUAGE_LABEL:
            names = _LANGUAGE_NAMES["en" if english else "ru"]
            before, after = names.get(old, display(old)), names.get(new, display(new))
        else:
            before, after = display(old), display(new)
        lines.append(f"{label} · {before} → {after}")
    heading("Изменения статистики", "Statistics changes", diff["statistics"])
    for old, new in diff["statistics"]:
        row = new if new is not None else old
        group = " / ".join(map(display, row["group"]))
        lines.append(f"{row['property']} [{row['unit']}] · {group or ('entire selection' if english else 'вся выборка')}")
        if old is None or new is None:
            action = ("Added" if old is None else "Removed") if english else ("Добавлено" if old is None else "Удалено")
            lines.append(f"  {action}")
        for key, label in (
            ("rows", "Проб"), ("n", "N чисел"), ("mean", "Среднее"), ("median", "Медиана"),
            ("min", "Мин."), ("max", "Макс."), ("std", "Ст. откл."), ("p10", "P10"),
            ("q1", "Q1"), ("q3", "Q3"), ("p90", "P90"), ("iqr", "IQR"), ("outliers", "Выбросы"),
            ("missing", "Пропуски"), ("censored", "С порогом"), ("non_detect", "ND"),
            ("other", "Текст и прочее"), ("excluded_qc", "Исключено QC"),
            ("excluded_interpolation", "Исключено интерполяций"),
            ("interpolated_used", "Использовано интерполяций"), ("coverage_pct", "Покрытие, %"),
            ("unit", "Единица"), ("property", "Показатель"),
        ):
            a, b = (old or {}).get(key), (new or {}).get(key)
            if a != b:
                if english:
                    label = _ENGLISH_STATISTICS_LABELS.get(key, label)
                lines.append(f"  {label} · {display(a)} → {display(b)}")
    decisions = diff.get("qc_decisions", ())
    heading("Изменения решений QC", "QC decision changes", decisions)
    def decision_text(record):
        if record is None:
            return "—"
        state, comment, at = record[2:]
        if not english:
            state = {"open": "Открыто", "acknowledged": "Принято"}.get(state, state)
        return " · ".join(filter(None, (state, comment, at)))
    for _, old, new in decisions:
        sample, rule = (new or old)[:2]
        lines.append(f"{sample} · {rule} · {decision_text(old)} → {decision_text(new)}")
    heading("Изменения исключений QC", "QC exclusion changes", diff["exclusions"])
    for (scope, mid), old, new in diff["exclusions"]:
        lines.append(f"{scope} · {mid} · {display(old)} → {display(new)}")
    heading("Изменения схем", "Scheme changes", diff["schemes"])
    scheme_labels = ({"stratigraphy": "Stratigraphy", "interpretation": "Interpretation"} if english
                     else {"stratigraphy": "Стратиграфия", "interpretation": "Интерпретация"})
    for key, old, new in diff["schemes"]:
        label = scheme_labels[key]
        lines.append(f"{label} · {display(old)} → {display(new)}")
    return "\n".join(lines)
