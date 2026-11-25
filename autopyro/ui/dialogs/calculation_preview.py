"""Display, filter and navigate the calculation preview."""

from __future__ import annotations

from math import isnan

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import QDialogButtonBox, QHeaderView, QTableWidgetItem

from autopyro.analysis.calculations import FORMULAS

from ..i18n import tr
from ..table.model import context_issue_text, format_censored_range, uncertainty_tooltip
from ..table.selection import select_row_ids
from ..widgets.documents import source_token
from .calculation_comparison import COMPARISON_KEY
from .calculation_details import calculation_detail, calculation_statuses, comparison_status_text


def invalidate(editor, *args, announce=True):
    editor._preview_revision += 1
    stale_message = ""
    if editor._preview_source is not None:
        stale_message = tr("Данные или выборка изменились. Рассчитайте предпросмотр заново.")
        editor.set_status(stale_message, announce=announce)
    editor.prepared = []
    editor._preview_source = None
    editor._preview_outputs = []
    editor._comparison_rows = {}
    editor._sample_labels = {}
    editor._page = 0
    editor._row_filter = None
    editor.show_input_errors({})
    if hasattr(editor, "preview"):
        editor.preview.setRowCount(0)
        editor.preview.setAccessibleDescription(stale_message)
        editor.previous_page.setEnabled(False)
        editor.next_page.setEnabled(False)
        editor.export_button.setEnabled(False)
        editor.save_comparison_button.setEnabled(False)
        editor.saved_comparisons_button.setEnabled(bool(editor.owner.model.project.view.get(COMPARISON_KEY)))
        editor.sample_search.setEnabled(False)
        editor.open_sample.setEnabled(False)
        editor.reset_preview_filters.setEnabled(False)
        editor.filter_hint.hide()
        editor.preview_detail.clear()
        editor.page_label.clear()
        editor.unavailable_only.setEnabled(False)
    if hasattr(editor, "mismatches_only"):
        editor.mismatches_only.setEnabled(False)
    if hasattr(editor, "buttons"):
        editor.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)


def update_preview_summary(editor):
    editor.summary.setText(editor._preview_summary or tr('Выберите показатели или лабораторные колонки.'))


def change_page(editor, step):
    current, opposite = ((editor.next_page, editor.previous_page) if step > 0
                         else (editor.previous_page, editor.next_page))
    pager_had_focus = current.hasFocus()
    editor._page += step
    editor.render_preview()
    if pager_had_focus and not current.isEnabled() and opposite.isEnabled():
        opposite.setFocus(Qt.FocusReason.OtherFocusReason)
    editor.announce(editor.page_label, editor.page_label.text())


def render_preview(editor):
    if editor._preview_source is None:
        return
    reasons = calculation_statuses()
    headers = [tr('Проба')]
    for column, _ in editor._preview_outputs:
        formula = column.metadata["calculation"]["formula"]
        headers.append(column.name)
        if formula in editor._comparison_rows:
            unit = FORMULAS[formula]["unit"]
            headers.extend([
                tr('{0} лаборатория [{1}]', formula, unit),
                tr('{0} расчёт − лаборатория [{1}]', formula, unit),
                tr('{0} статус сверки', formula),
            ])
    filters = (editor.mismatches_only.isEnabled() and editor.mismatches_only.isChecked(),
               editor.unavailable_only.isEnabled() and editor.unavailable_only.isChecked(),
               editor.sample_search.text().strip().casefold())
    filter_changed = editor._row_filter is not None and filters != editor._row_filter
    if filters != editor._row_filter:
        editor._row_filter = filters
        editor._page = 0
        allowed = (editor._mismatch_rows if filters[0] else set()) | (editor._unavailable_rows if filters[1] else set())
        candidates = [rid for rid in editor._preview_rows if rid in allowed] if any(filters[:2]) else editor._preview_rows
        query = filters[2]
        editor._visible_rows = ([rid for rid in candidates if query in rid.casefold()
                               or query in editor._sample_labels.get(rid, "").casefold()] if query else candidates)
    total = len(editor._visible_rows)
    start = editor._page * 200
    rows = editor._visible_rows[start:start + 200]
    editor.reset_preview_filters.setEnabled(bool(editor._preview_rows) and any(filters))
    editor.filter_hint.setVisible(bool(filters[0] and filters[1]))
    if rows:
        page = tr('{0}–{1} из {2}', start + 1, start + len(rows), total)
        if any(filters):
            page = tr('По фильтрам: {0}', page)
    elif editor._preview_rows:
        page = tr('По текущим фильтрам пробы не найдены.')
    else:
        page = ""
    editor.page_label.setText(page)
    editor.preview.setAccessibleDescription(editor.page_label.text())
    editor.previous_page.setEnabled(editor._page > 0)
    editor.next_page.setEnabled(start + len(rows) < total)
    editor.preview.setCurrentCell(-1, -1)
    editor.preview.clearContents()
    editor.open_sample.setEnabled(False)
    new_headers = headers != getattr(editor, "_preview_headers", None)
    if new_headers:
        editor._preview_headers = headers
        editor.preview.setColumnCount(len(headers))
        editor.preview.setHorizontalHeaderLabels(headers)
        editor.preview.setColumnHidden(0, True)
    editor.preview.setRowCount(len(rows))
    for i, rid in enumerate(rows):
        label = editor._sample_labels.get(rid, rid)
        sample_header = QTableWidgetItem(label)
        sample_header.setToolTip(f"{label}\n{rid}" if label != rid else rid)
        editor.preview.setVerticalHeaderItem(i, sample_header)
        cells = [label]
        tooltips = {0: rid} if label != rid else {}
        for column, values in editor._preview_outputs:
            value = values[rid]
            info = column.metadata["diagnostics"][rid]
            reason = calculation_detail(info, column.unit, reasons)
            text = str(value) if value is not None else reason
            if info["status"] == "censored_range":
                text = format_censored_range(info["range"])
                tooltips[len(cells)] = reason
            uncertainty = info.get("uncertainty")
            if uncertainty:
                tooltips[len(cells)] = uncertainty_tooltip(uncertainty, column.unit)
                if uncertainty["status"] == "ok":
                    warning = " ⚠" if uncertainty.get("unstable_denominator") else ""
                    text = f"{value:.6g} ± {uncertainty['standard']:.6g}{warning}"
            cells.append(text)
            formula = column.metadata["calculation"]["formula"]
            if formula not in editor._comparison_rows:
                continue
            entry = editor._comparison_rows[formula].loc[rid]
            lab, difference, status = entry["number"], entry["difference"], entry["status"]
            if status in ("match", "mismatch"):
                label = comparison_status_text(status, entry["context_incomplete"])
            elif entry["context_fields"]:
                label = tr('Сравнение недоступно: {0}', context_issue_text(entry["context_fields"]))
            else:
                label = tr('Сравнение недоступно: {0}', reasons.get(status, reason))
            cells.extend([
                str(lab) if not isnan(lab) else "—",
                f"{difference:+.10g}" if status in ("match", "mismatch") else "—",
                label,
            ])
        for j, text in enumerate(cells):
            item = QTableWidgetItem(text)
            if j == 0:
                item.setData(Qt.ItemDataRole.UserRole, rid)
            item.setToolTip(f"{text}\n{tooltips[j]}" if j in tooltips else text)
            editor.preview.setItem(i, j, item)
    if new_headers:
        header = editor.preview.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for j in range(1, len(headers)):
            width = max(85, editor.preview.sizeHintForColumn(j), header.sectionSizeHint(j))
            editor.preview.setColumnWidth(j, min(width, 260))
    editor.update_preview_summary()
    if filter_changed:
        editor.announce(editor.page_label, editor.page_label.text())


def clear_preview_filters(editor):
    with QSignalBlocker(editor.sample_search), QSignalBlocker(editor.mismatches_only), QSignalBlocker(editor.unavailable_only):
        editor.sample_search.clear()
        editor.mismatches_only.setChecked(False)
        editor.unavailable_only.setChecked(False)
    editor.sample_search.setFocus(Qt.FocusReason.OtherFocusReason)
    editor.render_preview()


def show_preview_detail(editor, item, _previous):
    editor.open_sample.setEnabled(editor._preview_source is not None and item is not None)
    if item is None:
        editor.preview_detail.clear()
        return
    sample = editor.preview.verticalHeaderItem(item.row())
    field = editor.preview.horizontalHeaderItem(item.column())
    if sample is not None and field is not None:
        editor.preview_detail.setText(f"{sample.text()} · {field.text()}\n{item.toolTip()}")


def jump_to_sample(editor, row, _column):
    if editor._preview_source is None or editor._preview_source != source_token(editor.owner):
        return
    item = editor.preview.item(row, 0)
    if item is None:
        return
    owner = editor.owner
    rid = item.data(Qt.ItemDataRole.UserRole)
    position = owner.model.resolve_row_positions([rid]).get(rid)
    if position is None:
        editor.set_status(tr('Строка пробы отсутствует в проекте.'))
        return
    columns = owner.model.project.columns
    column = next((i for i in range(len(columns)) if not owner.table.isColumnHidden(i)), 0)
    target = owner.proxy.mapFromSource(owner.model.index(position, column))
    if not target.isValid():
        if owner.open_source_cell(rid, columns[column].id, editor):
            select_row_ids(owner, [rid])
            owner.table.setFocus()
        return
    select_row_ids(owner, [rid])
    owner.desktop.open_table()
    owner.table.setFocus()
    owner.table.scrollTo(target)
