"""Display distributions and manage correlation and scatter matrices."""

from __future__ import annotations

from pathlib import Path
from textwrap import fill
import json

from PySide6.QtWidgets import QFileDialog, QMessageBox

from autopyro.analysis.statistics import StatisticsOptions, TABLES
from autopyro.data_io.file_output import write_json
from autopyro.visualization.statistics_matrix import (
    MATRIX_STYLE_FIELDS,
    SCATTER_STYLE_FIELDS,
    correlation_matrix_figure,
    scatter_matrix_figure,
)

from ..i18n import tr, message
from ..tasks import run_with_progress


SCATTER_MATRIX_MAX_COLUMNS = 6


def clear_chart(sheet):
    sheet.matrix_artifact = None
    sheet.save_matrix_button.hide()
    sheet.load_matrix_style_button.hide()
    sheet.distribution_view.hide()
    canvas = sheet.distribution_view.takeWidget()
    if canvas is not None:
        from ..plot_preview import release_figure

        release_figure(canvas.figure)
        canvas.deleteLater()


def show_distribution(sheet, index):
    if not sheet.report or not index.isValid():
        return
    from matplotlib.cbook import boxplot_stats
    from matplotlib.figure import Figure
    from numpy import linspace

    selected = sheet.report["summary"][sheet.result_row(index)]
    cid = selected["column_id"]
    entries = [entry for entry in sheet.report["summary"] if entry["column_id"] == cid]
    groups = list(zip(entries, sheet.report["group_row_ids"], strict=True))
    selected_group = sheet.selected_result_group()
    if selected_group is not None:
        groups = [(entry, ids) for entry, ids in groups if entry["group"] == selected_group]
    page_note = ""
    if len(groups) > 100:
        count = len(sheet.report["options"]["column_ids"])
        start = (sheet.result_row(index) // count // 100) * 100
        end = min(start + 100, len(groups))
        page_note = tr('Группы {0}–{1} из {2}. Выберите строку сводки или отдельную группу.', start + 1, end, len(groups))
        groups = groups[start:end]
    contexts = {(entry["unit"], entry["basis"]) for entry, _ in groups if entry["n"]}
    if len(contexts) > 1:
        QMessageBox.information(sheet, tr('Распределение'), tr('Группы имеют разные единицы или основания измерения. Выберите одну группу.'))
        return
    report, data = sheet.report, sheet.input_snapshot
    options = StatisticsOptions(**report["options"])

    def prepare():
        all_ids = [rid for entry, ids in groups if entry["n"] for rid in ids]
        _, classified = sheet.classified_snapshot_values(data, options, cid, all_ids)
        distributions = []
        for position, (entry, ids) in enumerate(groups, start=1):
            if not entry["n"]:
                continue
            numbers = classified.loc[ids, "number"].dropna().to_numpy(dtype=float)
            points = numbers[linspace(0, len(numbers) - 1, 500, dtype=int)] if len(numbers) > 500 else numbers
            stats, = boxplot_stats(numbers)
            stats.pop("fliers")
            distributions.append((position, stats, points))
        return distributions

    distributions = (run_with_progress(sheet, tr('Подготовка распределения…'), prepare)
                     if sum(len(ids) for _, ids in groups) > 4096 else prepare())
    if report is not sheet.report:
        return

    figure = Figure(layout="constrained")
    axes = figure.add_subplot()
    labels = [fill(f"{sheet.chart_group_label(entry['group'])} (N={entry['n']})", width=48) for entry, _ in groups]
    if distributions:
        positions, stats, _ = zip(*distributions, strict=True)
        axes.bxp(stats, positions=positions, orientation="horizontal", manage_ticks=False, showfliers=False)
        for position, _, points in distributions:
            offsets = linspace(-0.15, 0.15, len(points)) if len(points) > 1 else 0
            axes.scatter(points, position + offsets, s=14, alpha=0.55, zorder=3)
    axes.set_yticks(range(1, len(labels) + 1), labels)
    axes.set_ylim(len(labels) + 0.5, 0.5)
    unit = next(iter(contexts), ("", ""))[0]
    xlabel = f"{selected['property']} [{unit}]" if unit else selected["property"]
    if any(entry["n"] > 500 for entry, _ in groups):
        xlabel += "\n" + tr('До 500 точек на группу. Квартили по всем значениям.')
    if page_note:
        xlabel += "\n" + page_note
    axes.set_xlabel(xlabel)
    axes.grid(axis="x", alpha=0.25)
    label_height = 20 + 14 * max(label.count("\n") + 1 for label in labels)
    sheet.show_chart(figure, height=max(260, 50 + label_height * len(labels)))


def show_correlation_matrix(sheet, index):
    if not sheet.report or not index.isValid():
        return
    column = index.column() - len(sheet.report["groups"])
    if column < 0:
        return
    method, label = TABLES["correlations"][column]
    if method not in ("pearson", "spearman"):
        return
    group, entries = sheet.correlation_group(sheet.result_row(index))
    artifact = sheet.matrix_payload("heatmap", group, entries)
    artifact["matrix"]["method"] = method
    sheet.open_matrix(artifact)


def correlation_group(sheet, row):
    size = len(sheet.report["options"]["column_ids"])
    pairs_per_group = size * (size - 1) // 2
    group = row // pairs_per_group
    start = group * pairs_per_group
    return group, sheet.report["correlations"][start:start + pairs_per_group]


def show_scatter_matrix(sheet):
    if not sheet.scatter_matrix.isEnabled():
        return
    import numpy as np

    index = sheet.tables["correlations"].currentIndex()
    if not index.isValid():
        return
    group, entries = sheet.correlation_group(sheet.result_row(index))
    ids = sheet.report["group_row_ids"][group]
    column_ids = sheet.report["options"]["column_ids"]
    values = []
    for cid in column_ids:
        _, classified = sheet.classified_values(cid, ids, group_index=group)
        values.append([float(value) if np.isfinite(value) else None
                       for value in classified["number"].to_numpy(dtype=float)])
    artifact = sheet.matrix_payload("scatter", group, entries)
    artifact["matrix"]["values"] = values
    sheet.open_matrix(artifact)


def matrix_payload(sheet, kind, group, entries):
    count = len(sheet.report["options"]["column_ids"])
    indicators = sheet.report["summary"][group * count:(group + 1) * count]
    return {"format": "autopyro.plot", "version": 1, "matrix": {
        "kind": kind, "scope": sheet.report_scope,
        "scope_label": sheet.report["scope"], "project_id": sheet.report["project_id"],
        "created_at": sheet.report["created_at"],
        "group_label": sheet.chart_group_label(entries[0]["group"]),
        "group_fields": sheet.report["groups"], "group": entries[0]["group"],
        "row_ids": sheet.report["group_row_ids"][group],
        "options": sheet.report["options"], "methods": sheet.report["methods"],
        "columns": [{"id": entry["column_id"], "name": entry["property"], "unit": entry["unit"]}
                    for entry in indicators],
        "pairs": entries,
    }}


def open_matrix(sheet, payload):
    if payload.get("format") != "autopyro.plot" or payload.get("version") != 1:
        raise ValueError(tr('Неизвестный формат шаблона'))
    spec = payload["matrix"]
    kind, size = spec["kind"], len(spec["columns"])
    row_ids = spec["row_ids"]
    min_pairs = spec["options"]["min_pairs"]
    if (not 2 <= size <= 30 or not isinstance(row_ids, (list, tuple))
            or any(not isinstance(rid, str) or not rid for rid in row_ids)
            or len(set(row_ids)) != len(row_ids)
            or type(min_pairs) is not int or min_pairs < 3):
        raise ValueError(tr('Неизвестный формат шаблона'))
    columns = {column["id"] for column in spec["columns"]}
    pairs = {frozenset((pair["x_id"], pair["y_id"])) for pair in spec["pairs"]}
    if (len(columns) != size or len(pairs) != size * (size - 1) // 2
            or len(pairs) != len(spec["pairs"])
            or any(len(pair) != 2 or not pair <= columns for pair in pairs)
            or any(type(pair["n_pairs"]) is not int or not 0 <= pair["n_pairs"] <= len(row_ids)
                   for pair in spec["pairs"])):
        raise ValueError(tr('Неизвестный формат шаблона'))
    for pair in spec["pairs"]:
        status = pair["status"]
        if status not in ("ok", "constant", "insufficient_pairs", "numerical_warning"):
            raise ValueError(tr('Неизвестный формат шаблона'))
        if (status == "insufficient_pairs") != (pair["n_pairs"] < min_pairs):
            raise ValueError(tr('Некорректное число полных пар для {0} и {1}.', pair["x"], pair["y"]))
        for method in ("pearson", "spearman"):
            value = pair[method]
            if value is not None and (type(value) not in (int, float) or not -1.000000000001 <= value <= 1.000000000001):
                raise ValueError(tr('Неизвестный формат шаблона'))
            if value is not None and pair["n_pairs"] < min_pairs:
                raise ValueError(tr('Некорректное число полных пар для {0} и {1}.', pair["x"], pair["y"]))
            if ((status == "constant" and value is not None)
                    or (status == "ok" and value is None)):
                raise ValueError(tr('Статус пары {0} и {1} не соответствует коэффициентам.', pair["x"], pair["y"]))
    if kind == "heatmap" and spec["method"] in ("pearson", "spearman"):
        figure = correlation_matrix_figure(spec, dict(TABLES["correlations"])[spec["method"]])
    elif kind == "scatter" and size <= SCATTER_MATRIX_MAX_COLUMNS and len(spec["values"]) == size and all(
        len(values) == len(row_ids) for values in spec["values"]
    ):
        import numpy as np

        values = np.asarray(spec["values"], dtype=float)
        if values.ndim != 2:
            raise ValueError(tr('Неизвестный формат шаблона'))
        present = np.isfinite(values)
        positions = {column["id"]: index for index, column in enumerate(spec["columns"])}
        for pair in spec["pairs"]:
            x, y = positions[pair["x_id"]], positions[pair["y_id"]]
            if pair["n_pairs"] != np.count_nonzero(present[x] & present[y]):
                raise ValueError(tr('Некорректное число полных пар для {0} и {1}.', pair["x"], pair["y"]))
        figure = scatter_matrix_figure(spec, values, tr("Матрица диаграмм рассеяния"))
    else:
        raise ValueError(tr('Неизвестный формат шаблона'))
    width, height = (round(dimension) for dimension in figure.bbox.size)
    sheet.show_chart(figure, width=width, height=height)
    sheet.matrix_artifact = payload
    sheet.save_matrix_button.show()
    sheet.load_matrix_style_button.show()


def load_matrix_style(sheet):
    current = sheet.matrix_artifact
    if current is None:
        return
    path, _ = QFileDialog.getOpenFileName(sheet, tr('Загрузить оформление'), "", "JSON (*.json)")
    if not path or current is not sheet.matrix_artifact:
        return
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "autopyro.plot" or payload.get("version") != 1:
            raise ValueError(tr('Неизвестный формат шаблона'))
        template = payload["spec"]
        if not isinstance(template, dict):
            raise ValueError(tr('Неизвестный формат шаблона'))
        matrix = current["matrix"]
        fields = MATRIX_STYLE_FIELDS
        if matrix["kind"] == "scatter":
            fields += SCATTER_STYLE_FIELDS
        style = {key: template[key] for key in fields if key in template}
        sheet.open_matrix({**current, "matrix": {**matrix, "style": style}})
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        QMessageBox.warning(sheet, tr('Ошибка'), message(exc))


def load_matrix(sheet):
    path, _ = QFileDialog.getOpenFileName(sheet, tr('Открыть матрицу…'), "", "JSON (*.json)")
    if path:
        try:
            sheet.open_matrix(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            QMessageBox.warning(sheet, tr('Ошибка'), message(exc))


def save_matrix(sheet):
    payload = sheet.matrix_artifact
    if payload is None or sheet.matrix_export.worker is not None:
        return
    path, selected_filter = QFileDialog.getSaveFileName(
        sheet, tr('Сохранить матрицу…'), "matrix.json",
        "JSON (*.json);;PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
    )
    if not path or payload is not sheet.matrix_artifact:
        return
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix({"PNG (*.png)": ".png", "SVG (*.svg)": ".svg",
                                 "PDF (*.pdf)": ".pdf"}.get(selected_filter, ".json"))
    try:
        if path.suffix.lower() == ".json":
            write_json(path, payload)
        else:
            spec = payload["matrix"]
            label = (dict(TABLES["correlations"])[spec["method"]] if spec["kind"] == "heatmap"
                     else tr('Матрица диаграмм рассеяния'))
            sheet.matrix_export.start([{"name": label, "path": str(path), "matrix": spec}])
    except (OSError, ValueError) as exc:
        QMessageBox.warning(sheet, tr('Экспорт'), message(exc))


def matrix_export_busy_changed(sheet, busy):
    sheet.save_matrix_button.setEnabled(not busy)
    if busy:
        sheet.matrix_export_progress.show()
    else:
        sheet.matrix_export_progress.reset()


def matrix_export_completed(sheet, outcome):
    if outcome["errors"]:
        sheet.matrix_export_failed("\n".join(outcome["errors"]))


def matrix_export_failed(sheet, error):
    if not sheet.closed:
        QMessageBox.warning(sheet, tr('Экспорт'), message(error))


def chart_group_label(sheet, group):
    categories = [
        f"{name} = {value if value not in (None, '') else tr('Пусто')}"
        for name, value in zip(sheet.report["groups"], group)
    ]
    return " · ".join(categories) if categories else tr('Все пробы')


def show_chart(sheet, figure, *, height, width=0):
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    sheet.clear_chart()
    canvas = FigureCanvasQTAgg(figure)
    canvas.setMinimumSize(width, height)
    sheet.distribution_view.setWidget(canvas)
    sheet.distribution_view.show()
    canvas.show()
