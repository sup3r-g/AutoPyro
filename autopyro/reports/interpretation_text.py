"""English presentation text for geometric interpretation reports."""

ENGLISH_STATUSES = {
    "classified": "Within a region", "boundary": "On a boundary", "ambiguous": "Ambiguous",
    "outside": "Outside regions", "insufficient_data": "Insufficient data",
    "not_applicable": "Outside the applicability domain",
}

ENGLISH_NOTES = {
    "Интерпретация не включена.": "Interpretation is disabled.",
    "Пересчитано геометрическим ядром для выборки отчёта.": "Recalculated by the geometric engine for the report selection.",
    "Нет сохранённой схемы с полной привязкой к колонкам; интерпретация не выполнена.":
        "Interpretation was skipped because no saved scheme has complete column bindings.",
}

_REASONS = {
    "Отсутствует допустимое числовое значение X или Y": "No valid numeric X or Y value",
    "Координаты вне области применимости или преобразования": "Coordinates are outside the applicability or transformation domain",
    "Недопустимый интервал неопределённости": "Invalid uncertainty interval",
    "Неконечная координата": "Coordinate must be finite",
    "Координата ниже области применимости": "Coordinate is below the applicability range",
    "Координата выше области применимости": "Coordinate is above the applicability range",
    "Неположительная координата для log10": "The log10 transform requires a positive coordinate",
    "Интервал должен быть конечным, упорядоченным и содержать измерение":
        "The interval must have finite, ordered bounds and contain the measured value",
    "Интерпретация заблокирована: значение оси вне абсолютного диапазона":
        "Interpretation blocked because an axis value is outside the absolute range",
    "интервал допускает несколько исходов порогового правила": "The interval allows multiple threshold-rule outcomes",
    "Точка или часть интервала вне семейства траекторий": "The point or part of the interval is outside the curve family",
    "экстраполяция HIo запрещена": "HIo extrapolation is disabled",
    "Точка вне допустимого расстояния от кривых": "The point exceeds the allowed distance from the curves",
    "Часть интервала дальше допустимого расстояния от семейства кривых":
        "Part of the interval exceeds the allowed distance from the curve family",
    "Внутри интервала есть координаты вне допустимого расстояния":
        "The interval contains coordinates beyond the allowed distance",
    "Для всего интервала не подтверждён допустимый предел расстояния":
        "The distance limit could not be confirmed across the whole interval",
    "бюджет уточнения исчерпан": "The refinement budget was exhausted",
    "экстраполяция запрещена": "Extrapolation is disabled",
    "неопределённость HIo или типа вещества": "Uncertain HIo or organic matter type",
    "интерпретация всего интервала в нормированном пространстве": "The full interval is interpreted in normalized space",
    "консервативная оболочка возможных результатов": "Conservative envelope of possible results",
    "бюджет уточнения исчерпан, исключить дополнительные варианты не удалось":
        "The refinement budget was exhausted and additional outcomes could not be ruled out",
    "интервал допускает несколько типов или исходных HI": "The interval allows multiple types or initial HI values",
    "правило границ: отмечать отдельно": "Boundary policy: flag separately",
    "правило границ: включать": "Boundary policy: include",
    "правило границ: исключать": "Boundary policy: exclude",
}

_PREFIXES = {
    "Внутри: ": "Inside: ", "границы: ": "Boundaries: ", "линии: ": "Lines: ",
    "области диапазона неопределённости: ": "Uncertainty regions: ",
    "Возможные ближайшие кривые: ": "Possible nearest curves: ", "Ближайшая кривая: ": "Nearest curve: ",
    "Родительские кривые семейства HIo: ": "Parent curves in the HIo family: ",
    "модельная оценка HIo = ": "Model estimate HIo = ",
}


def english_reason(text):
    wrapper = "Недопустимые координаты или интервал: "
    if text.startswith(wrapper):
        detail = text.removeprefix(wrapper)
        return "Invalid coordinates or interval: " + _REASONS.get(detail, detail)
    parts = text.split("; ")
    for index, part in enumerate(parts):
        if part in _REASONS:
            parts[index] = _REASONS[part]
        else:
            for source, translation in _PREFIXES.items():
                if part.startswith(source):
                    parts[index] = translation + part.removeprefix(source)
                    break
    return "; ".join(parts)
