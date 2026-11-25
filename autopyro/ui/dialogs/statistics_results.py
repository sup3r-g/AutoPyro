"""Display, filter and navigate statistical result tables."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker

from autopyro.analysis.statistics import TABLES, report_table

from ..i18n import tr
from ..theme import icon
from ..widgets.results import SortableResultsModel
from .statistics_calculation import DIAGNOSTIC_TABS
from .statistics_charts import SCATTER_MATRIX_MAX_COLUMNS


def clear_tables(sheet):
    sheet.deferred_tables.clear()
    sheet.cancel_calculation_button.setEnabled(False)
    sheet.stale_results.hide()
    sheet.comparison_status.setText(sheet.status.text())
    for table in sheet.tables.values():
        old = table.model()
        table.setModel(SortableResultsModel([], [], table))
        if old:
            old.deleteLater()


def display_report(sheet, report, table_names=None):
    sheet.report = report
    sheet.stale_results.hide()
    if table_names is None:
        sheet.fill_result_groups(report)
    for name in sheet.tables if table_names is None else table_names:
        if name != sheet.current_name() and len(report.get(name, ())) > 2048:
            sheet.deferred_tables[name] = report
        else:
            sheet.display_result_table(report, name)
    sheet.update_navigation_hint()
    sheet.update_export_buttons()


def display_result_table(sheet, report, name):
    sheet.deferred_tables.pop(name, None)
    table = sheet.tables[name]
    headers, rows = report_table(report, name)
    if name == "comparison":
        status = next(index for index, (key, _) in enumerate(TABLES[name]) if key == "comparison_status")
        for row in rows:
            row[status] = tr(row[status])
    elif name == "repeats":
        kind = len(report["groups"]) + 1
        for row in rows:
            row[kind] = tr(row[kind])
    row_order = None
    selected_group = sheet.selected_result_group()
    if name == "thresholds":
        group = sheet.result_group.currentIndex() - 1 if selected_group is not None else None
        row_order = [index for index, entry in enumerate(report.get(name, [])) if entry["group_index"] == group]
    elif selected_group is not None and name != "comparison":
        row_order = [index for index, row in enumerate(rows) if row[:len(selected_group)] == selected_group]
    if row_order is not None:
        rows = [rows[index] for index in row_order]
    old = table.model()
    model = SortableResultsModel(headers, rows, table, row_order=row_order)
    table.setModel(model)
    header = table.horizontalHeader()
    if header.sortIndicatorSection() >= 0:
        model.sort(header.sortIndicatorSection(), header.sortIndicatorOrder())
    table.selectionModel().currentChanged.connect(sheet.update_result_action)
    if name == "correlations":
        table.selectionModel().selectionChanged.connect(sheet.update_result_action)
    if old:
        old.deleteLater()
    header.setResizeContentsPrecision(20)
    table.resizeColumnsToContents()
    for column in range(model.columnCount()):
        if table.columnWidth(column) > 280:
            table.setColumnWidth(column, 280)
    sheet.pin_result_columns(name)


def selected_result_group(sheet):
    index = sheet.result_group.currentIndex()
    return sheet.result_group_values[index - 1] if index > 0 else None


def fill_result_groups(sheet, report):
    fields = tuple(report["groups"])
    selected = sheet.selected_result_group() if fields == sheet.result_group_fields else None
    sheet.result_group_fields = fields
    count = len(report["options"]["column_ids"])
    sheet.result_group_values = [entry["group"] for entry in report["summary"][::count]] if fields else []
    with QSignalBlocker(sheet.result_group):
        sheet.result_group.clear()
        sheet.result_group.addItem(tr('Все группы'))
        sheet.result_group.addItems([sheet.chart_group_label(group) for group in sheet.result_group_values])
        if selected in sheet.result_group_values:
            sheet.result_group.setCurrentIndex(sheet.result_group_values.index(selected) + 1)


def filter_result_groups(sheet):
    if sheet.report is not None:
        sheet.clear_details()
        sheet.display_report(sheet.report, tuple(name for name in sheet.tables if name != "comparison"))


def current_name(sheet):
    return list(sheet.tables)[sheet.tabs.currentIndex()]


def update_result_pins(sheet):
    for name in sheet.tables:
        sheet.pin_result_columns(name)


def pin_result_columns(sheet, name):
    table = sheet.tables[name]
    model = table.model()
    if model is None or not model.headers:
        return
    header = table.horizontalHeader()
    if not sheet.pin_identifiers.isChecked():
        table.set_pinned_columns(())
        for column in range(header.count()):
            header.moveSection(header.visualIndex(column), column)
        return
    group_count = sum(heading.startswith("Группа: ") for heading in model.headers)
    columns = (0, group_count) if group_count else (0, 1) if name == "correlations" else (0,)
    for column in columns:
        table.setColumnWidth(column, min(table.columnWidth(column), 140))
    table.set_pinned_columns(columns)


def result_row(index):
    return index.model().row_order[index.row()]


def clear_details(sheet):
    sheet.clear_chart()
    sheet.sample_table.hide()
    sheet.sample_label.hide()
    sheet.sample_action_button.hide()
    model = sheet.sample_table.model()
    if model is not None:
        sheet.sample_table.setModel(None)
        model.set_rows([])
        model.deleteLater()


def tab_changed(sheet):
    sheet.clear_details()
    tab = sheet.current_name()
    if tab in sheet.deferred_tables:
        sheet.display_result_table(sheet.deferred_tables[tab], tab)
    if tab in DIAGNOSTIC_TABS and tab not in sheet.requested_diagnostics:
        sheet.requested_diagnostics.add(tab)
        if not (sheet.task_running or sheet.pending or sheet.snapshot_iterator is not None):
            sheet.calculate_diagnostic(tab)
    sheet.update_navigation_hint()
    sheet.update_export_buttons()


def update_navigation_hint(sheet):
    sheet.update_calculation_context()
    correlations = sheet.current_name() == "correlations"
    sheet.load_matrix_button.setVisible(correlations)
    sheet.scatter_matrix.setVisible(correlations)
    hint = {
        "summary": tr('Двойной щелчок по строке открывает распределение показателя по группам.'),
        "availability": tr('Двойной щелчок по счётчику или проценту открывает список проб.'),
        "correlations": tr('Двойной щелчок по коэффициенту Pearson или Spearman открывает тепловую карту группы. Для матрицы рассеяния выберите строку и от 2 до {0} показателей. N — число полных пар.', SCATTER_MATRIX_MAX_COLUMNS),
        "outliers": tr('Двойной щелчок по выбросу открывает пробу в таблице. Для скрытой пробы меняется фильтр таблицы.'),
        "repeats": tr('Повторы: размах по пробе. Сравнение методов или лабораторий: медианы по пробе, затем разность B − A. Двойной щелчок по строке открывает пробы.'),
        "thresholds": tr('Двойной щелчок по строке открывает список проб.'),
        "interpolation": tr('Двойной щелчок по строке открывает список проб.'),
    }.get(sheet.current_name(), "")
    if correlations and sheet.report and not sheet.report["options"]["correlations"]:
        hint = (tr('Корреляции пропущены: выбрано более 30 показателей.') if sheet.correlations.isChecked()
                else tr('Расчёт корреляций отключён.'))
    elif sheet.current_name() == "repeats" and sheet.report and sheet.report.get("repeats") == []:
        hint = tr('В выбранной области нет сопоставимых повторных анализов с числовыми результатами и указанным методом.')
    elif sheet.current_name() == "thresholds" and sheet.report and sheet.report.get("thresholds") == []:
        hint = tr('В выбранной области нет непустых рабочих записей по выбранным показателям.')
    elif sheet.current_name() == "interpolation" and sheet.report and sheet.report.get("interpolation") == []:
        hint = tr('Нет строк для сравнения. Выберите исходный показатель со связанной колонкой интерполяции и непустую выборку.')
    elif (sheet.cancel_requested and sheet.report is not None and sheet.current_name() in DIAGNOSTIC_TABS
          and sheet.current_name() not in sheet.report):
        hint = tr('Чтобы повторить расчёт, переключитесь на другой раздел и вернитесь.')
    sheet.navigation_hint.setText(hint)
    sheet.navigation_hint.setVisible(bool(hint))


def update_calculation_context(sheet):
    report = sheet.report
    sheet.policy.setVisible(report is not None)
    if report is None:
        return
    tab = sheet.current_name()
    options = report["options"]
    comparison = report.get("comparison_context")
    if tab == "comparison":
        status = tr('Расчёт UTC: {0}', report["created_at"])
        context = ""
        if comparison:
            scopes = [
                tr('Набор проб: {0}', comparison[side]) if comparison[f"{side}_scope"].startswith("set:")
                else tr('Представление: {0}', comparison[side])
                for side in ("a", "b")
            ]
            context = tr('A: {0} · B: {1}', *scopes) + "\n"
            context += tr('Группировка сводки не применяется. Для разницы медиан требуются известный метод и согласованный аналитический контекст.')
    else:
        if tab == "thresholds":
            group = sheet.result_group.currentIndex() - 1
            ids = report["group_row_ids"][group] if group >= 0 else report["row_ids"]
            status = tr('{0} · строк: {1} · показателей: {2} · расчёт UTC: {3}',
                        report["scope"], len(ids), len(options["column_ids"]), report["created_at"])
        else:
            status = tr('{0} · строк: {1} · показателей: {2} · групп: {3} · расчёт UTC: {4}',
                        report["scope"], len(report["row_ids"]), len(options["column_ids"]),
                        len(report["group_row_ids"]), report["created_at"])
        if not report["row_ids"]:
            status += tr(' · Выборка пуста, подстановка всех проб не выполняется.')
        if tab == "thresholds":
            group = sheet.selected_result_group()
            context = sheet.chart_group_label(group) + "\n" if group is not None else ""
            context += tr('Группы: показатель, метод, лаборатория, единица и основание. Доли считаются от непустых рабочих записей, включая исключённые QC.')
        else:
            grouping = ", ".join(report["groups"]) or tr('Нет группировки')
            context = tr('Группировка: {0}', grouping) + "\n"
            if tab == "repeats":
                context += tr('Используются рабочие и нерабочие результаты с указанным методом. QC-исключения и записи с квалификаторами пропускаются.')
            elif tab == "interpolation":
                context += tr('Актуальные оценки применяются только к пустым исходным ячейкам. Флажок включения интерполяций в сводку здесь не влияет на расчёт.')
            else:
                context += tr('Пропуски, пороги, ND и QC-исключения не используются.')
                context += " " + tr('Значения групп с разными единицами или основаниями измерения исключаются.')
                if tab == "summary":
                    context += " " + tr('SD: n−1. Квантили: linear.')
                elif tab == "outliers":
                    context += " " + tr('Границы IQR: 1,5 × IQR за квартилями при N ≥ 4.')
                elif tab == "correlations" and options["correlations"]:
                    context += " " + tr('Корреляции используют полные пары. Минимум пар: {0}.', options["min_pairs"])
    regular = tab in ("summary", "availability", "correlations", "outliers") or (tab == "comparison" and comparison)
    if regular or tab in ("repeats", "interpolation"):
        text_policy = tr('Числа из текста включены') if options["text_numbers"] else tr('Числа из текста исключены')
        context += "\n" + text_policy
    if regular:
        interpolation_policy = tr('Актуальные интерполяции включены отдельно') if options["include_interpolated"] else tr('Интерполяции исключены')
        context += " · " + interpolation_policy
    if report["notices"]:
        status += "\n" + "; ".join(report["notices"])
    if report["numerical_warning_count"]:
        status += "\n" + tr('Численная неустойчивость в строках сводки: {0}', report["numerical_warning_count"])
    if not sheet.task_running and not sheet.cancel_requested:
        sheet.status.setText(status)
    sheet.policy.setText(context)
    sheet.policy.setVisible(bool(context))


def result_action_index(sheet):
    tab = sheet.current_name()
    index = sheet.tables[tab].currentIndex()
    if not sheet.report or not index.isValid():
        return None
    column = index.column() - (0 if tab == "comparison" else len(sheet.report["groups"]))
    if tab == "availability" and column < 2:
        return None
    if tab == "comparison" and TABLES[tab][column][0] not in ("a_used", "a_excluded", "b_used", "b_excluded"):
        return None
    if tab == "correlations":
        # Pearson is the default when the selected cell is not a coefficient.
        keys = [key for key, _ in TABLES[tab]]
        if column < 0 or keys[column] not in ("pearson", "spearman"):
            index = index.siblingAtColumn(len(sheet.report["groups"]) + keys.index("pearson"))
    return index


def update_result_action(sheet, *args):
    tab = sheet.current_name()
    context = ""
    if tab == "comparison" and sheet.report is not None:
        current = sheet.tables[tab].currentIndex()
        if current.isValid():
            context = sheet.report[tab][sheet.result_row(current)]["context_explanation"]
    sheet.comparison_context_hint.setText(context)
    sheet.comparison_context_hint.setVisible(bool(context))
    label, _, image = sheet.result_actions[tab]
    index = sheet.result_action_index()
    if tab == "correlations" and index is not None:
        label += " · " + TABLES[tab][index.column() - len(sheet.report["groups"])][1]
    sheet.result_action_button.setText(label)
    sheet.result_action_button.setIcon(icon(image))
    sheet.result_action_button.setEnabled(index is not None)
    sheet.scatter_matrix.setEnabled(
        tab == "correlations" and index is not None
        and sheet.tables[tab].selectionModel().isSelected(sheet.tables[tab].currentIndex())
        and len(sheet.report["options"]["column_ids"]) <= SCATTER_MATRIX_MAX_COLUMNS
    )


def activate_result_action(sheet):
    index = sheet.result_action_index()
    if index is not None:
        sheet.result_actions[sheet.current_name()][1](index)


def update_export_buttons(sheet, *args):
    sheet.update_result_action()
    ready = sheet.report is not None
    tab = sheet.current_name()
    sheet.result_group.setVisible(bool(sheet.result_group_values) and tab != "comparison")
    sheet.result_group.setEnabled(ready)
    tab_ready = ready and (tab not in ("comparison", *DIAGNOSTIC_TABS) or tab in sheet.report)
    sheet.copy.setEnabled(tab_ready)
    sheet.export.setEnabled(tab_ready)
    sheet.export_json.setEnabled(ready)
