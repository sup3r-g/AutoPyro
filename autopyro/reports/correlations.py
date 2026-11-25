"""Present correlation pairs consistently in the human-readable report formats."""

from collections import Counter

from autopyro.analysis.statistics import report_table


_STATUS_LABELS = {
    "ok": ("Рассчитано", "Calculated"),
    "insufficient_pairs": ("Недостаточно пар наблюдений", "Insufficient observation pairs"),
    "constant": ("Один из показателей постоянен", "One indicator is constant"),
    "numerical_warning": ("Численное предупреждение", "Numerical warning"),
}


def correlation_blocks(statistics, language, *, detailed):
    pairs = statistics["correlations"]
    if not pairs:
        return

    english = language == "en"
    label_index = int(english)
    unknown = ("Причина не указана", "Reason unavailable")[label_index]
    no_coefficient = ("Коэффициенты не рассчитаны", "Coefficients unavailable")[label_index]
    labels = {status: names[label_index] for status, names in _STATUS_LABELS.items()}
    usable, omitted = [], Counter()
    for pair in pairs:
        if pair["pearson"] is not None or pair["spearman"] is not None:
            usable.append(pair)
        else:
            status = pair["status"]
            omitted[unknown if status == "ok" else labels.get(status, unknown)] += 1

    shown = pairs if detailed else usable
    if shown:
        headers, rows = report_table({**statistics, "correlations": shown}, "correlations", language=language)
        for pair, row in zip(shown, rows, strict=True):
            status = pair["status"]
            missing = pair["pearson"] is None and pair["spearman"] is None
            row[-1] = no_coefficient if missing and status == "ok" else labels.get(status, unknown)
        yield "table", "Корреляции", (headers, rows)

    if omitted:
        count, total = omitted.total(), len(pairs)
        text = [f"No coefficients for {count} of {total} indicator pairs." if english
                else f"Без коэффициентов — {count} из {total} пар показателей."]
        text.extend(f"{reason} — {number}." for reason, number in omitted.items())
        if labels["insufficient_pairs"] in omitted:
            minimum = statistics["options"].get("min_pairs")
            if minimum is not None:
                text.append(f"Minimum observations per pair — {minimum}." if english
                            else f"Минимум совместных наблюдений для пары — {minimum}.")
        yield "text", "Пары без коэффициентов", " ".join(text)
