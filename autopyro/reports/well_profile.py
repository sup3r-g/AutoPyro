"""Depth profiles for a frozen report selection."""

import base64
import io
import re
from collections import Counter

from matplotlib.figure import Figure

from autopyro.analysis.qc import is_excluded
from autopyro.analysis.statistics import interpolation_pairs
from autopyro.geology.stratigraphy import LEVELS, draw_layer
from autopyro.model.tabular import classify
from autopyro.visualization.plotting import PlotSpec


_ENGLISH_LAYER_ERRORS = {
    "Неизвестный формат стратиграфической схемы": "Unknown stratigraphic scheme format",
    "Единицы глубины: m или ft": "Scheme depth units must be m or ft",
    "Добавьте интервалы": "Add intervals to the scheme",
    "Каждому интервалу нужен уникальный ID": "Each interval requires a unique ID",
    "Укажите название ствола и подразделения": "Enter a wellbore and stratigraphic unit name",
    "Границы должны быть конечными числами": "Interval boundaries must be finite numbers",
    "NaN и бесконечность недопустимы": "NaN and infinity are not allowed",
    "Кровля должна быть меньше подошвы": "The top depth must be less than the base depth",
    "Некорректный цвет интервала": "Invalid interval color",
    "Литология stratapy должна быть строкой": "The stratapy lithology must be text",
    "Режим проб: point или interval": "Sample mode must be point or interval",
    "Выберите разные существующие колонки ствола и глубин проб": "Select distinct existing wellbore and sample depth columns",
    "Объявите единицу глубины проб: m или ft": "Declare the sample depth unit as m or ft",
    "Объявленная единица противоречит единице колонки глубины": "The declared unit differs from the depth column unit",
    "Тип глубины и начало отсчёта проб и интервалов должны совпадать": "Samples and intervals must use the same depth type and datum",
    "Стратиграфические результаты нельзя использовать как вход своей привязки": "Stratigraphic results cannot be inputs to their own assignment",
    "Стратиграфический фон не привязывается к интерполированным стволам или глубинам": "The stratigraphic layer requires source wellbores and depths",
    "Стратиграфический слой требует глубинного графика с привязанной осью Y": "The stratigraphic layer requires a depth chart with a bound Y axis",
    "Для интервальной привязки на графике нужна соответствующая колонка низа отбора": "Select the corresponding sample base column for interval assignment",
    "Укажите уровень стратиграфического слоя": "Select a stratigraphic level",
    "Для стратиграфического фона выберите пробы ровно одного ствола": "Select samples from exactly one wellbore",
    "Не задан ствол проб": "Sample wellbore is missing",
    "Стратиграфические границы логарифмической оси должны быть положительными": "Stratigraphic boundaries on a log axis must be positive",
}


def _english_layer_error(error):
    detail = str(error)
    for source, target in (("Неизвестная литология stratapy: ", "Unknown stratapy lithology: "),
                           ("Допустимые уровни: ", "Supported levels: ")):
        if detail.startswith(source):
            return target + detail.removeprefix(source)
    field_error = re.fullmatch(r"Укажите (id|revision|name|depth_reference); тип глубины и начало отсчёта должны быть заданы явно", detail)
    if field_error:
        return f"Specify {field_error[1]}. The depth type and datum must be explicit."
    return _ENGLISH_LAYER_ERRORS.get(detail, detail)


WELL_NAMES = {"well", "well name", "well_name", "wellbore", "wellbore name", "wellbore_name",
              "скважина", "название скважины", "имя скважины"}
FORMATION_NAMES = {"formation", "formation name", "formation_name", "reservoir", "alt top lstrat(fm)",
                   "пласт", "название пласта", "свита", "формация"}
TOP_DEPTH_NAMES = {"top md"}


def field_name(column):
    return re.sub(r"\s*\[[^\]]+\]\s*$", "", column.name.rsplit(" / ", 1)[-1]).strip()


def depth_unit(column):
    suffix = re.search(r"\[(m|ft)\]\s*$", column.name, re.IGNORECASE)
    return column.unit.strip().casefold() or (suffix[1].casefold() if suffix else "")


def find_columns(columns, names):
    return [column.id for column in columns if not column.metadata.get("method")
            and field_name(column).casefold() in names]


def unique_column(columns, names):
    matches = find_columns(columns.values(), names)
    return matches[0] if len(matches) == 1 else None


def depth_columns(columns, binding, explicit):
    depth_id = explicit or binding.get("depth_id") or unique_column(
        columns, TOP_DEPTH_NAMES | {"md", "depth", "depth_md", "глубина", "глубина отбора"}
    )
    if depth_id is not None and depth_id == binding.get("depth_id"):
        return depth_id, binding.get("bottom_id") if binding.get("sample_mode") == "interval" else None
    if depth_id in columns and field_name(columns[depth_id]).casefold() in TOP_DEPTH_NAMES:
        bottom_id = unique_column(columns, {"base md", "bottom md"})
        unit = depth_unit(columns[depth_id])
        if bottom_id and unit and depth_unit(columns[bottom_id]) == unit:
            return depth_id, bottom_id
    return depth_id, None


def depth_datum_column(columns, depth_id):
    if depth_id in columns:
        name = field_name(columns[depth_id]).casefold()
        if name in TOP_DEPTH_NAMES | {"md", "depth_md"}:
            return unique_column(columns, {"md datum"})
        if name in {"tvd", "top tvd", "depth_tvd"}:
            return unique_column(columns, {"tvd datum"})
    return None


def datum_matches(datum, reference):
    def normalize(value):
        return " ".join(value.casefold().split()) if isinstance(value, str) else ""

    actual, expected = normalize(datum), normalize(reference)
    if not actual or not expected:
        return False
    if actual == expected:
        return True
    return actual == re.sub(r"^(?:md|tvd)\s+", "", expected) and not re.match(r"^(?:md|tvd)\s+", actual)


def build_profiles(project, ids, options, *, checkpoint):
    english = options.language == "en"
    checkpoint()
    columns = {column.id: column for column in project.columns}
    configured = project.view.get("stratigraphy") or {}
    binding = configured.get("binding") or {}
    scheme = configured.get("scheme")

    depth_id, bottom_id = depth_columns(columns, binding, options.profile_depth_id)
    well_id = binding.get("well_id") or unique_column(columns, WELL_NAMES) or next(iter(options.group_ids), None)
    if depth_id not in columns or well_id not in columns or depth_id == well_id:
        return [], ["Select a depth column and a well column (first grouping level)." if english
                    else "Укажите колонку глубины и колонку скважины (первый уровень группировки)."]
    if columns[depth_id].metadata.get("method"):
        return [], ["The profile depth axis requires a source depth column." if english
                    else "Для шкалы профиля нужна исходная колонка глубины."]
    compatible = bool(scheme and depth_id == binding.get("depth_id"))
    strict_intervals = compatible and binding.get("sample_mode") == "interval"
    datum_id = depth_datum_column(columns, depth_id)
    lithology_id = unique_column(columns, {"lithology", "литология"})
    formation_id = unique_column(columns, FORMATION_NAMES)
    base_formation_id = unique_column(columns, {"alt base lstrat(fm)"})
    categories = [(cid, label, base) for cid, label, base in (
        (formation_id, "stratigraphy", base_formation_id), (lithology_id, "lithology", None),
    ) if cid]
    track_ids = list(dict.fromkeys((*options.column_ids, *(
        estimate for _, estimate in interpolation_pairs(project, options.column_ids)
    ))))
    tracks = [columns[cid] for cid in track_ids
              if cid not in {depth_id, well_id, bottom_id, lithology_id, formation_id}
              and not columns[cid].metadata.get("stale")]
    rows = project.row_map()
    wells = {}
    skipped = 0
    for rid in ids:
        row = rows[rid]
        well = row.values.get(well_id)
        if not isinstance(well, str) or not well.strip():
            skipped += 1
            continue
        top = classify(row.values.get(depth_id), numeric=True)[1]
        bottom = classify(row.values.get(bottom_id), numeric=True)[1] if bottom_id else None
        if top is None or (strict_intervals and bottom is None) or (bottom is not None and bottom < top):
            skipped += 1
            continue
        datum = row.values.get(datum_id) if datum_id else None
        wells.setdefault((well, datum), []).append((row, top, bottom))

    profiles = []
    notices = []
    if skipped:
        notices.append(f"Samples without a wellbore or valid depth: {skipped}." if english
                       else f"Проб без ствола или допустимой глубины: {skipped}.")
    if not wells:
        return [], notices + ["No samples have a valid depth." if english else "Нет проб с допустимой глубиной."]
    if not compatible:
        notices.append("The interval scheme is missing or uses a different depth column. Sample categories are shown at their sampling depths when available."
                       if english else "Интервальная схема отсутствует или привязана к другой колонке глубины. "
                       "При наличии подписей проб показаны их категории на глубинах отбора.")
    for (well, datum), samples in wells.items():
        checkpoint()
        series = {}
        for column in tracks:
            points = []
            for row, top, bottom in samples:
                value = classify(row.values.get(column.id),
                                 numeric=column.kind == "number" or options.text_numbers)[1]
                if value is None:
                    continue
                excluded = any(is_excluded(project, row.id, column.id, scope)
                               for scope in ("plots", "statistics"))
                if column.metadata.get("method"):
                    status = "qc_interpolated" if excluded else "interpolated"
                else:
                    status = "qc" if excluded else "measured"
                points.append((value, top, bottom, status))
            if points:
                series[column.id] = points
        if not series:
            notices.append(f"{well}: no numeric values for the selected indicators." if english
                           else f"{well}: нет числовых значений выбранных показателей.")
            continue
        datum_ok = not datum_id or datum_matches(datum, binding.get("depth_reference"))
        if compatible and not datum_ok:
            notices.append(f"{well} · {datum or 'datum not specified'}: interval scheme not shown. "
                           f"Expected depth reference: {binding.get('depth_reference', '—')}." if english
                           else f"{well} · {datum or 'datum не указан'}: интервальная схема не наложена. "
                           f"Ожидаемый отсчёт: {binding.get('depth_reference', '—')}.")
        intervals = [item for item in scheme["intervals"] if item["wellbore"] == well] if compatible and datum_ok else []
        levels = {item["level"] for item in intervals}
        level = next((rank for rank in ("FORMATION", *reversed(LEVELS)) if rank in levels), None)
        has_lithology = any(item.get("lithology", "").strip() for item in intervals if item["level"] == level)
        fallback = [(cid, label, base) for cid, label, base in categories
                    if (label == "stratigraphy" and not level)
                    or (label == "lithology" and not has_lithology)]
        keys = list(series)
        for start in range(0, len(keys), 3):
            checkpoint()
            part = keys[start:start + 3]
            profiles.append(_render_profile(
                project, well, datum, samples, part, series, columns, depth_id, bottom_id,
                binding, scheme, level, has_lithology, fallback, options.text_numbers, start // 3 + 1,
                language=options.language,
            ))
    return profiles, notices


def _render_profile(project, well, datum, samples, tracks, series, columns, depth_id, bottom_id,
                    binding, scheme, level, has_lithology, fallback, text_numbers, page, *, language):
    english = language == "en"
    figure = Figure(figsize=(8, 6))
    figure.subplots_adjust(left=.11, right=.96, top=.85, bottom=.20 if fallback else .12, wspace=.33)
    axes = figure.subplots(1, len(tracks) + len(fallback), sharey=True, squeeze=False)[0]
    suffix = f"{well}" + (f" · {datum}" if datum else "") + (f" ({page})" if page > 1 else "")
    name = f"Глубинный профиль: {suffix}"
    figure.suptitle(f"Depth profile: {suffix}" if english else name, fontsize=12)
    styles = {
        "measured": ("o", "#235789", "Measurement" if english else "Измерение"),
        "qc": ("x", "#ad2525", "Excluded by QC" if english else "Исключено по QC"),
        "interpolated": ("^", "#a05400", "Interpolation" if english else "Интерполяция"),
        "qc_interpolated": ("X", "#ad2525", "Interpolation excluded by QC" if english else "Интерполяция, исключённая по QC"),
    }
    handles = {}
    counts = {}
    for axis, cid in zip(axes, tracks):
        column = columns[cid]
        counts[cid] = dict(Counter(point[3] for point in series[cid]))
        axis.set_title(f"{column.name}\n[{column.unit}]" if column.unit else column.name,
                       fontsize=8, wrap=True)
        for status, (marker, color, label) in styles.items():
            points = [(value, top, bottom) for value, top, bottom, kind in series[cid] if kind == status]
            if points:
                values, tops, bottoms = zip(*points)
                handle = axis.scatter(values, tops, marker=marker, color=color, s=17, label=label)
                handles.setdefault(status, handle)
                intervals = [(value, top, bottom) for value, top, bottom in points if bottom is not None]
                if intervals:
                    xs, ys, ends = zip(*intervals)
                    axis.vlines(xs, ys, ends, color=color, linewidth=1)
        axis.grid(alpha=.25)
        axis.tick_params(labelsize=8)
    category_labels = ({"stratigraphy": "Sample stratigraphy", "lithology": "Sample lithology"} if english
                       else {"stratigraphy": "Стратиграфия проб", "lithology": "Литология проб"})
    for axis, (cid, label, base_id) in zip(axes[len(tracks):], fallback):
        values = {}
        for row, top, bottom in samples:
            value = row.values.get(cid)
            if value is not None and value != "":
                position = values.setdefault(str(value), len(values))
                axis.scatter(position, top, s=10, color="#526c57")
                if bottom is not None and label == "lithology":
                    axis.vlines(position, top, bottom, color="#526c57", linewidth=1)
            base_value = row.values.get(base_id) if base_id else None
            if bottom is not None and base_value is not None and base_value != "":
                position = values.setdefault(str(base_value), len(values))
                axis.scatter(position, bottom, s=10, marker="s", color="#526c57")
        axis.set_xticks(list(values.values()), list(values), rotation=35, ha="right", fontsize=7)
        axis.set_title(category_labels[label], fontsize=8)
    depths = [depth for _, top, bottom in samples for depth in (top, bottom) if depth is not None]
    lo, hi = min(depths), max(depths)
    pad = (hi - lo) * .025 or 1
    axes[0].set_ylim(hi + pad, lo - pad)
    depth = columns[depth_id]
    unit = depth.unit or (binding.get("depth_unit") if depth_id == binding.get("depth_id") else "")
    axes[0].set_ylabel(f"{depth.name} [{unit}]" if unit else depth.name)
    figure.legend(handles=list(handles.values()), loc="lower center", ncol=2, fontsize=8)
    notices = ["QC marks active exclusions for charts or statistics. Current interpolations are shown separately from measurements."
               if english else "QC: действующие исключения для графиков или статистики. "
               "Актуальные интерполяции показаны отдельно от измерений."]
    if any(base for _, _, base in fallback):
        notices.append("On the stratigraphy track, circles mark the unit at the sample top and squares mark it at the base."
                       if english else "На треке стратиграфии круг отмечает пласт на кровле пробы, квадрат — на подошве.")
    if level:
        spec = PlotSpec(kind="multi_depth", y_id=depth_id, depth_bottom_id=bottom_id or "",
                        stratigraphy={"scheme": scheme, "binding": binding, "level": level},
                        stratigraphy_info=False, stratigraphy_alpha=.36)
        try:
            notices.extend(draw_layer(axes, project, [row.id for row, _, _ in samples], spec, language=language))
            for track in figure.axes[len(axes):]:
                track.grid(axis="y", color="#637684", linewidth=.5, alpha=.32)
            revision = "rev." if english else "ред."
            notices.append(f"{scheme['name']} · {revision} {scheme['revision']} · {level} · {scheme['depth_reference']}.")
            if has_lithology:
                notices.append("Lithology is shown for the scheme intervals." if english else "Литология показана по интервалам схемы.")
        except ValueError as exc:
            has_lithology = False
            notices.append(f"Stratigraphic layer unavailable: {_english_layer_error(exc)}"
                           if english else f"Стратиграфический слой недоступен: {exc}")
    if not level and not any(label == "stratigraphy" for _, label, _ in fallback):
        notices.append("Stratigraphic data are unavailable." if english else "Стратиграфические данные недоступны.")
    if not has_lithology and not any(label == "lithology" for _, label, _ in fallback):
        notices.append("Lithology data are unavailable." if english else "Литологические данные недоступны.")
    stream = io.BytesIO()
    figure.savefig(stream, format="png", dpi=150)
    figure.clear()
    return {
        "name": name, "row_ids": [row.id for row, _, _ in samples], "status_counts": counts,
        "depth_range": [lo, hi],
        "interval_samples": sum(bottom is not None for _, _, bottom in samples),
        "spec": {"y_id": depth_id, "depth_bottom_id": bottom_id, "text_numbers": text_numbers,
                 "include_interpolated": True, "track_ids": tracks},
        "notices": notices,
        "png": base64.b64encode(stream.getvalue()).decode("ascii"),
    }
