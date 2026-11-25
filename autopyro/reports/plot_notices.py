"""English display text for notices stored with ordinary report figures."""

import re


_EXACT = {
    "Нет числовых значений цвета: использован цвет пропусков": "No numeric color values. The missing-value color was used.",
    "Слой показывает сохранённую редакцию схемы, отличающуюся от текущей":
        "The layer shows a saved scheme revision that differs from the current revision.",
    "Пустая выборка: стратиграфический фон не показан": "The selection is empty. The stratigraphic layer is hidden.",
    "Для выбранного ствола и уровня нет интервалов": "No intervals for the selected wellbore and level.",
    "Стратиграфический слой содержит перекрытия на одном уровне":
        "The stratigraphic layer has overlapping intervals at the same level.",
    "Стратиграфический слой показывает сохранённую редакцию, отличающуюся от текущей":
        "The stratigraphic layer shows a saved revision that differs from the current revision.",
}

_COUNTED = {
    "Единица неопределённости не совпадает с единицей оси": "Uncertainty and axis units differ",
    "Тип неопределённости не указан или не поддерживается; интервал не использован":
        "The uncertainty type is missing or unsupported. The interval was skipped",
    "Интервал не использован: нет допустимого числового значения": "The interval was skipped because no valid numeric value is available",
    "Неопределённость должна быть конечной и неотрицательной": "Uncertainty must be finite and non-negative",
    "Неконечные границы неопределённости; интервал не использован": "The interval was skipped because its uncertainty bounds are not finite",
    "Интервал неопределённости имеет неположительную границу на логарифмической оси":
        "The uncertainty interval has a non-positive bound on a logarithmic axis",
}

_PATTERNS = tuple((re.compile(pattern), translation) for pattern, translation in (
    (r"Аннотация №(\d+): выбранный трек отсутствует на графике", "Annotation {}: The selected track is missing from the chart."),
    (r"Аннотация №(\d+): тип или координаты несовместимы с осями графика",
     "Annotation {}: The type or coordinates are incompatible with the chart axes."),
    (r"Интервалы неопределённости с неположительной границей на логарифмической оси: (\d+)",
     "Uncertainty intervals with a non-positive bound on a logarithmic axis: {}."),
    (r"Интервальная интерпретация неоднозначна: (\d+); отмечено оранжевым ромбом",
     "Ambiguous interval interpretations: {}. Orange diamonds mark these points."),
    (r"(.+): для violin недостаточно разброса; показаны точки",
     "{}: Insufficient spread for a violin plot. Individual points are shown."),
    (r"(.+) · кривая из архива, точки из текущей области",
     "{} · Archived curve with points from the current selection."),
))
_COUNT_PATTERN = re.compile(r"(.+): (" + "|".join(map(re.escape, _COUNTED)) + r") \((\d+)\)")


def english_plot_notice(notice):
    if notice in _EXACT:
        return _EXACT[notice]
    for pattern, translation in _PATTERNS:
        if match := pattern.fullmatch(notice):
            return translation.format(*match.groups())
    if match := _COUNT_PATTERN.fullmatch(notice):
        label, warning, count = match.groups()
        return f"{label}: {_COUNTED[warning]} ({count})"
    return notice
