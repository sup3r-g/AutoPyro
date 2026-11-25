"""Inspect samples behind statistical results and navigate to their source rows."""

from __future__ import annotations

from copy import deepcopy

from PySide6.QtWidgets import QMessageBox
from pandas import Series

from autopyro.analysis.statistics import StatisticsOptions, TABLES, classify_property
from autopyro.model.sample_labels import sample_label, sample_label_fields

from ..i18n import tr
from ..widgets.results import SampleResultsModel


def show_comparison_samples(sheet, index):
    if not sheet.report or "comparison" not in sheet.report or not index.isValid():
        return
    key = TABLES["comparison"][index.column()][0]
    if key not in ("a_used", "a_excluded", "b_used", "b_excluded"):
        return
    side, kind = key.split("_")
    entry = sheet.report["comparison"][sheet.result_row(index)]
    context = sheet.report["comparison_context"]
    samples = [(rid, status) for rid, status in entry["samples"][side]
               if (status == "numeric") == (kind == "used")]
    sheet.show_samples(entry, samples, context[side], context[f"{side}_scope"])


def show_availability_samples(sheet, index):
    if not sheet.report or not index.isValid():
        return
    column_index = index.column() - len(sheet.report["groups"])
    if column_index < 0:
        return
    key = TABLES["availability"][column_index][0]
    if key in ("property", "unit"):
        return
    row = sheet.result_row(index)
    entry = sheet.report["availability"][row]
    column_count = len(sheet.report["options"]["column_ids"])
    ids = sheet.report["group_row_ids"][row // column_count]
    cid = entry["column_id"]
    column, classified = sheet.classified_values(cid, ids, group_index=row // column_count)
    statuses = classified["status"]
    if key in ("n", "coverage_pct"):
        statuses = statuses[statuses.eq("numeric")]
    elif key == "interpolated_used":
        statuses = statuses[statuses.eq("numeric")] if column.metadata.get("method") else statuses.iloc[:0]
    elif key != "rows":
        statuses = statuses[statuses.eq(key)]
    sheet.show_samples(entry, statuses.items(), sheet.report["scope"], sheet.report_scope)


def classified_values(sheet, cid, ids, *, group_index=None):
    options = StatisticsOptions(**sheet.report["options"])
    column, classified = sheet.classified_snapshot_values(sheet.input_snapshot, options, cid, ids)
    if group_index is not None:
        position = group_index * len(options.column_ids) + options.column_ids.index(cid)
        if sheet.report["summary"][position]["excluded_context"]:
            classified.loc[classified["status"].eq("numeric"), ["status", "number"]] = ["excluded_context", None]
    return column, classified


def classified_snapshot_values(project, options, cid, ids):
    column = next(column for column in project.columns if column.id == cid)
    source_rows = project.row_map()
    values = Series([source_rows[rid].values[cid] for rid in ids], index=ids, dtype=object)
    return column, classify_property(project, values, column, options)


def show_samples(sheet, entry, samples, label, scope):
    cid = entry["column_id"]
    statuses = {
        "numeric": tr('Использовано'),
        "missing": tr('Пусто'),
        "censored": tr('Пороговое значение'),
        "non_detect": "ND",
        "other": tr('Прочие / текст'),
        "excluded_interpolation": tr('Искл. интерполяции'),
        "excluded_qc": tr('Исключено QC'),
        "excluded_context": tr('Искл. несовместимый контекст'),
    }
    source_rows = sheet.input_snapshot.row_map()
    rows = [
        [rid, source_rows[rid].values[cid], statuses[status]]
        for rid, status in samples
    ]
    sheet.display_samples(["UUID пробы", "Значение", "Статус"], rows, label, entry["property"], scope)


def show_repeat_samples(sheet, index):
    if not sheet.report or not index.isValid():
        return
    entry = sheet.report["repeats"][sheet.result_row(index)]
    headers = ["UUID пробы", "Медиана A", "Анализов A", "Медиана B", "Анализов B", entry["kind"]]
    rows = [[sample[key] for key in ("row_id", "a", "a_count", "b", "b_count", "difference")]
            for sample in entry["samples"]]
    if entry["method_b"] is None:
        del headers[3:5]
        for row in rows:
            del row[3:5]
    sheet.display_samples(headers, rows, sheet.report["scope"], entry["property"], sheet.report_scope)


def show_threshold_samples(sheet, index):
    if not sheet.report or not index.isValid():
        return
    entry = sheet.report["thresholds"][sheet.result_row(index)]
    statuses = {"below": tr('Ниже порога'), "above": tr('Выше порога'), "non_detect": "ND",
                "other": tr('Без порога')}
    key = TABLES["thresholds"][index.column()][0].removesuffix("_pct")
    samples = entry["samples"]
    if key == "excluded_qc":
        samples = [sample for sample in samples if sample["excluded_qc"]]
    elif key == "no_numeric_limit":
        samples = [sample for sample in samples if sample["status"] != "non_detect" and sample["limit"] is None]
    else:
        for status in ("below", "above", "non_detect"):
            if key == status or key.startswith(status + "_"):
                samples = [sample for sample in samples if sample["status"] == status
                           and (not key.endswith(("_min", "_max")) or sample["limit"] == entry[key]
                                and sample["limit"] is not None)]
                break
    headers = ["UUID пробы", "Значение", "Статус", "Порог из записи", "LOD", "LOQ", "Исключено QC"]
    rows = [
        [sample["row_id"], sample["value"], statuses[sample["status"]], sample["limit"],
         sample["lod"], sample["loq"], tr('Да') if sample["excluded_qc"] else tr('Нет')]
        for sample in samples
    ]
    if key in ("n", "excluded_qc"):
        project = sheet.input_snapshot
        source_rows = project.row_map()
        cid = entry["column_id"]
        qc_rows = set(entry["other_qc_row_ids"])
        ids = entry["other_qc_row_ids"] if key == "excluded_qc" else entry["other_row_ids"]
        for rid in ids:
            measurement = project.science.current(rid, cid)
            rows.append([rid, source_rows[rid].values[cid], statuses["other"], None,
                         getattr(measurement, "lod", None), getattr(measurement, "loq", None),
                         tr('Да') if rid in qc_rows else tr('Нет')])
    group = sheet.selected_result_group()
    label = sheet.report["scope"] if group is None else f"{sheet.report['scope']} · {sheet.chart_group_label(group)}"
    sheet.display_samples(headers, rows, label, entry["property"], sheet.report_scope)


def show_interpolation_samples(sheet, index):
    if not sheet.report or not index.isValid():
        return
    entry = sheet.report["interpolation"][sheet.result_row(index)]
    statuses = {
        "added": tr('Использовано'),
        "stale": tr('Устаревшие оценки'),
        "unit_mismatch": tr('Разные единицы'),
        "invalid": tr('Недоступные оценки'),
        "source_numeric": tr('Есть исходное число'),
        "source_censored": tr('Есть исходный порог'),
        "source_non_detect": tr('Исходное ND'),
        "source_other": tr('Прочая исходная запись'),
        "source_excluded_qc": tr('Исходное исключено QC'),
        "no_estimate": tr('Пусто без оценки'),
    }
    from .interpolation import diagnostic_labels
    from autopyro.geology.stratigraphy import STATUSES

    labels = diagnostic_labels()
    labels["outside_scope"] = tr('Проба вне исходной выборки интерполяции')
    unknown = tr('В протоколе нет сведений')
    headers = ["UUID пробы", "Исходное значение", "Оценка интерполяции", "Статус",
               "Диагностика интерполяции", "UUID левой опоры", "UUID правой опоры"]
    column = index.column() - len(sheet.report["groups"])
    key = TABLES["interpolation"][column][0] if column >= 0 else None
    measured = key in ("measured", "before_pct", "before_median", "after_pct", "after_median", "delta")
    augmented = key in ("after_pct", "after_median", "delta")
    samples = entry["samples"]
    if measured:
        originals = set(entry["measured_ids"])
        samples = [[rid, raw, estimate, "source_numeric" if rid in originals else status]
                   for rid, raw, estimate, status in samples
                   if rid in originals or (augmented and status == "added")]
        present = {sample[0] for sample in samples}
        if originals - present:
            source_rows = sheet.input_snapshot.row_map()
            samples.extend([rid, source_rows[rid].values[entry["column_id"]], None, "source_numeric"]
                           for rid in entry["measured_ids"] if rid not in present)
    elif key == "rows":
        present = {sample[0] for sample in samples}
        ids = [rid for rid in sheet.report["group_row_ids"][entry["group_index"]] if rid not in present]
        source_rows = sheet.input_snapshot.row_map()
        _, classified = sheet.classified_values(entry["column_id"], ids)
        samples = samples + [[rid, source_rows[rid].values[entry["column_id"]], None, f"source_{status}"]
                             for rid, status in classified["status"].items()]
    else:
        status = "added" if key == "estimate_median" else key
        if status in statuses:
            samples = [sample for sample in samples if sample[3] == status]
    rows = []
    provenance = entry["sample_provenance"]
    for rid, original, estimate, status in samples:
        role, reason, left_id, right_id = provenance.get(rid, (None, None, None, None))
        explanation = labels.get(role, unknown)
        if reason in STATUSES:
            explanation += f" · {tr(STATUSES[reason])}"
        rows.append([rid, original, estimate, statuses[status], explanation, left_id, right_id])
    sheet.display_samples(headers, rows, sheet.report["scope"], entry["estimate"], sheet.report_scope,
                         anchor_columns=(5, 6))


def display_samples(sheet, headers, rows, label, property_name, scope, *, anchor_columns=()):
    sheet.sample_scope = scope
    sheet.sample_anchor_columns = anchor_columns
    old = sheet.sample_table.model()
    project = sheet.owner.model.project
    fields = sample_label_fields((column.id, column.name) for column in project.columns)
    positions = sheet.owner.model.resolve_row_positions(row[0] for row in rows) if fields else {}
    labels = {rid: sample_label(rid, project.rows[position].values, fields) for rid, position in positions.items()}
    sheet.sample_table.setModel(SampleResultsModel([tr('Проба'), *headers[1:]], rows, labels, sheet.sample_table))
    sheet.sample_table.selectionModel().currentChanged.connect(sheet.update_sample_action)
    sheet.update_sample_action(sheet.sample_table.currentIndex())
    sheet.sample_action_button.show()
    if old:
        old.deleteLater()
    sheet.sample_table.horizontalHeader().setResizeContentsPrecision(20)
    sheet.sample_table.resizeColumnsToContents()
    hint = tr(
        '{0} · {1} · проб: {2}. Двойной щелчок открывает пробу в таблице. Для скрытой пробы меняется фильтр таблицы.',
        label, property_name, len(rows),
    )
    if anchor_columns:
        hint += " " + tr('Выберите ячейку с UUID опоры, чтобы открыть опорную пробу.')
    sheet.sample_label.setText(hint)
    sheet.sample_label.show()
    sheet.sample_table.show()


def update_sample_action(sheet, index, *_):
    anchor = index.column() in sheet.sample_anchor_columns
    sheet.sample_action_button.setText(
        tr('Показать опорную пробу в таблице') if anchor else tr('Показать пробу в таблице')
    )
    enabled = index.isValid()
    if enabled and anchor:
        rid = sheet.sample_table.model().rows[index.row()][index.column()]
        enabled = isinstance(rid, str) and bool(sheet.owner.model.resolve_row_positions([rid]))
    sheet.sample_action_button.setEnabled(enabled)


def go_to_listed_sample(sheet, index):
    if not sheet.report or not index.isValid():
        return
    row = sheet.sample_table.model().rows[index.row()]
    if index.column() in sheet.sample_anchor_columns:
        rid = row[index.column()]
        if isinstance(rid, str):
            sheet.go_to_sample(rid, None)
    else:
        sheet.go_to_sample(row[0], sheet.sample_scope)


def go_to_outlier(sheet, index):
    if sheet.report and index.isValid():
        sheet.go_to_sample(sheet.report["outliers"][sheet.result_row(index)]["row_id"], sheet.report_scope)


def go_to_sample(sheet, rid, scope):
    owner = sheet.owner
    position = owner.model.resolve_row_positions([rid]).get(rid)
    if position is None:
        return
    column = next((i for i in range(owner.model.columnCount()) if not owner.table.isColumnHidden(i)), 0)
    source = owner.model.index(position, column)
    target = owner.proxy.mapFromSource(source)
    if not target.isValid():
        generation = sheet.generation
        answer = QMessageBox.question(
            sheet, tr('Проба скрыта фильтром'),
            tr('Временно изменить фильтр таблицы, чтобы показать пробу? Прежний фильтр можно вернуть из строки состояния.'),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes or sheet.closed or generation != sheet.generation:
            return
        previous = sheet.previous_sample_filter
        if previous is None:
            previous = (deepcopy(owner.proxy.rules), owner.proxy.combine,
                        owner.proxy.row_id_filter, owner.active_sample_set)
        view = owner.model.project.view.get("named", {}).get(scope[5:]) if scope and scope.startswith("view:") else None
        set_name = scope[4:] if scope and scope.startswith("set:") else None
        if view is not None:
            set_name = view.get("sample_set")
        sets = owner.model.project.view.get("sample_sets", {})
        saved_rows = sets.get(set_name, ()) if set_name is not None else ({rid} if scope is None else None)
        owner.active_sample_set = set_name
        if view is None:
            owner.proxy.set_rules([], row_ids=saved_rows)
        else:
            owner.proxy.set_rules(view.get("rules", []), view.get("combine", "AND"), row_ids=saved_rows)
        owner.view_changed()
        sheet.previous_sample_filter = previous
        sheet.restore_sample_filter_button.show()
        target = owner.proxy.mapFromSource(source)
    if target.isValid():
        owner.desktop.open_table()
        owner.table.selectRow(target.row())
        owner.table.scrollTo(target)


def clear_sample_filter_restore(sheet):
    sheet.previous_sample_filter = None
    sheet.restore_sample_filter_button.hide()


def restore_sample_filter(sheet):
    if sheet.previous_sample_filter is None:
        return
    rules, combine, row_ids, set_name = sheet.previous_sample_filter
    sheet.clear_sample_filter_restore()
    if set_name is not None and set_name not in sheet.owner.model.project.view.get("sample_sets", {}):
        set_name, row_ids = None, None
    sheet.owner.active_sample_set = set_name
    sheet.owner.proxy.set_rules(rules, combine, row_ids=row_ids)
    sheet.owner.view_changed()
