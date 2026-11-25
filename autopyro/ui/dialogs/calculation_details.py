"""Format calculation and laboratory comparison diagnostics."""

from __future__ import annotations

from ..i18n import tr, message
from ..table.model import context_issue_text, format_censored_range, range_issue_text


def comparison_status_text(status, incomplete=False):
    label = tr('Численно в допуске') if status == "match" else tr('Расхождение')
    if incomplete:
        return label + " · " + tr('Метод или лаборатория указаны не для всех измерений')
    return label


def calculation_statuses():
    return {
        "ok": tr('рассчитано'),
        "missing": tr('нет значения'),
        "censored": tr('цензурировано'),
        "censored_range": tr('рассчитан диапазон'),
        "non_detect": "ND",
        "other": tr('нечисловые данные'),
        "invalid_range": tr('недопустимый диапазон'),
        "zero_denominator": tr('нулевой знаменатель'),
        "interpolation_input": tr('вход-интерполяция'),
        "qc_excluded": tr('Исключено решением QC'),
        "numerical_error": tr('численная ошибка'),
        "context_conflict": tr('различается аналитический контекст'),
    }


def calculation_detail(info, unit, labels):
    if "range_issue" in info:
        return range_issue_text(info)
    if info["status"] == "censored_range":
        return tr('Диапазон по пределам измерений: {0} [{1}]', format_censored_range(info["range"]), unit)
    if info["status"] == "context_conflict":
        return context_issue_text(info["fields"])
    if reason := info.get("reason"):
        prefix, separator, code = reason.partition(": ")
        return f"{prefix}: {labels[code]}" if separator and code in labels else message(reason)
    return labels.get(info["status"], info["status"])
