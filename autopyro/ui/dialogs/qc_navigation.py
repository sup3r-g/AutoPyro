"""Open QC measurements and sources and restore the original table view."""

from copy import deepcopy
import json

from PySide6.QtCore import QItemSelectionModel, QSignalBlocker, Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QMessageBox, QTextEdit, QVBoxLayout

from ..i18n import tr


def source_details_text(source):
    labels = {
        "path": 'Полный путь', "sheet": 'Лист', "row": 'Строка импорта',
        "row_number": 'Строка источника', "source_name": 'Название источника',
        "dbname": 'База данных', "table": 'Таблица', "relation": 'Объект базы данных',
        "source_record_id": 'Запись источника', "import_batch_id": 'Партия импорта',
        "analysis_run_id": 'Аналитический запуск', "imported_at": 'Время импорта',
        "sha256": 'Контрольная сумма SHA-256', "endpoint": 'Адрес базы данных',
        "type": 'Тип источника', "host": 'Хост', "port": 'Порт',
    }
    order = (*labels, *(key for key in source if key not in labels))
    lines = []
    for key in order:
        value = source.get(key)
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        lines.append(f"{tr(labels[key]) if key in labels else key}\n{value}")
    return "\n\n".join(lines)


def open_measurement(editor):
    issue = editor.action_issue()
    if not issue or not issue["measurement_id"]:
        return
    model = editor.owner.model
    measurement = model.project.science.measurements.get(issue["measurement_id"])
    if (measurement is None or measurement.revision != issue["revision"]
            or (measurement.sample_id, measurement.property_id) != (issue["sample_id"], issue["property_id"])):
        editor.status.setText(tr('Данные изменились. Проверьте замечание перед применением действия.'))
        return
    editor.show_measurement_card(measurement, editor)


def show_measurement_card(editor, measurement, parent):
    from .measurement import MeasurementDialog

    dialog = MeasurementDialog(editor.owner.model, measurement.sample_id, measurement.property_id, parent,
                               measurement_id=measurement.id)
    try:
        dialog.exec()
    finally:
        dialog.deleteLater()


def show_source(editor):
    issue = editor.action_issue()
    if not issue or not editor.source_button.isEnabled():
        return
    dialog = QDialog(editor)
    dialog.setWindowTitle(tr('Источник замечания QC'))
    dialog.resize(620, 420)
    layout = QVBoxLayout(dialog)
    details = QTextEdit()
    details.setReadOnly(True)
    details.setAccessibleName(tr('Источник замечания QC'))
    details.setPlainText(source_details_text(issue["source"]))
    layout.addWidget(details)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.button(QDialogButtonBox.StandardButton.Close).setText(tr('Закрыть'))
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.exec()


def go_to_sample(editor):
    issue = editor.action_issue()
    if not issue:
        return
    owner = editor.owner
    model = owner.model
    sid = issue["sample_id"]
    position = model.resolve_row_positions([sid]).get(sid)
    if position is None:
        return
    property_id = issue.get("property_id")
    columns = owner.model.project.columns
    column = next((n for n, col in enumerate(columns) if col.id == property_id), None)
    if column is None and property_id is not None:
        editor.status.setText(tr('Колонка замечания больше не доступна. Обновите проверки.'))
        return
    if column is None:
        column = next((n for n in range(owner.model.columnCount()) if not owner.table.isColumnHidden(n)), 0)
    source = owner.model.index(position, column)
    target = owner.proxy.mapFromSource(source)
    hidden_row = not target.isValid()
    hidden_column = owner.table.isColumnHidden(column)
    if hidden_row or hidden_column:
        revision = model.data_revision
        if hidden_row and hidden_column:
            question = tr('Проба скрыта фильтром, колонка «{0}» тоже скрыта. Временно показать обе?', columns[column].name)
        elif hidden_column:
            question = tr('Колонка «{0}» скрыта. Временно показать её?', columns[column].name)
        else:
            question = tr('Проба скрыта фильтром. Временно показать её?')
        answer = QMessageBox.question(
            editor, tr('Показать ячейку в таблице'),
            question + "\n" + tr('Прежний вид можно вернуть из панели контроля качества.'),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes or model is not owner.model or revision != model.data_revision:
            return
    if hidden_row:
        if not editor.temporary_row_filter_active():
            editor._previous_filter = (deepcopy(owner.proxy.rules), owner.proxy.combine,
                                     owner.proxy.row_id_filter, owner.active_sample_set)
        editor._revealed_row = sid
        owner.active_sample_set = None
        owner.proxy.set_rules([], row_ids=[sid])
        target = owner.proxy.mapFromSource(source)
    if hidden_column:
        editor._revealed_columns[columns[column].id] = owner.table.columnWidth(column)
        editor.set_temporary_column_visibility(column, True)
    if hidden_row or hidden_column:
        editor.restore_filter_button.show()
    if target.isValid():
        if editor.property("workspaceTracked"):
            owner.desktop.open_table()
        owner.table.setFocus()
        selection = QItemSelectionModel.SelectionFlag
        owner.table.selectionModel().setCurrentIndex(target, selection.ClearAndSelect | selection.Rows)
        owner.table.scrollTo(target)


def set_temporary_column_visibility(editor, column, visible):
    owner = editor.owner
    with QSignalBlocker(owner.columns):
        owner.columns.item(column).setCheckState(Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked)
    owner.table.setColumnHidden(column, not visible)


def forget_temporary_column(editor, column_id):
    editor._revealed_columns.pop(column_id, None)
    editor.restore_filter_button.setVisible(bool(editor._revealed_columns or editor.temporary_row_filter_active()))


def temporary_row_filter_active(editor):
    owner = editor.owner
    return (editor._previous_filter is not None and owner.active_sample_set is None and not owner.proxy.rules
            and owner.proxy.row_id_filter == frozenset((editor._revealed_row,)))


def persistent_table_view(editor, view, *, include_sample_set):
    if editor._revealed_columns:
        view["visible"] = [cid for cid in view["visible"] if cid not in editor._revealed_columns]
        view["widths"].update({cid: width for cid, width in editor._revealed_columns.items() if cid in view["widths"]})
    if editor.temporary_row_filter_active():
        rules, combine, _, set_name = editor._previous_filter
        view.update(rules=rules, combine=combine)
        if include_sample_set and set_name in editor.owner.model.project.view.get("sample_sets", {}):
            view["sample_set"] = set_name
            if editor.owner._sample_set_return_filter is not None:
                view["sample_set_return_filter"] = editor.owner._sample_set_return_filter
    return view


def restore_table_filter(editor):
    owner = editor.owner
    if editor.temporary_row_filter_active():
        rules, combine, row_ids, set_name = editor._previous_filter
        if set_name is not None and set_name not in owner.model.project.view.get("sample_sets", {}):
            set_name, row_ids = None, None
        owner.active_sample_set = set_name
        owner.proxy.set_rules(rules, combine, row_ids=row_ids)
    for index, column in enumerate(owner.model.project.columns):
        if column.id in editor._revealed_columns:
            editor.set_temporary_column_visibility(index, False)
    editor._revealed_columns.clear()
    editor._previous_filter = None
    editor._revealed_row = None
    editor.restore_filter_button.hide()
