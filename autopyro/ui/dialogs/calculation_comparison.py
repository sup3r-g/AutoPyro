"""Save, open and export comparisons with laboratory results."""

from __future__ import annotations

from copy import deepcopy
from math import isnan
import json

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QUndoCommand
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QPushButton,
    QTableView,
    QVBoxLayout,
)

from autopyro.data_io.file_output import write_excel_csv
from autopyro.model.audit import archive_protocol, read_protocol
from autopyro.model.core import new_id, timestamp

from ..i18n import tr, message
from ..table.model import context_issue_text, large_project
from ..tasks import run_with_progress
from ..widgets.documents import source_token
from ..widgets.results import ResultsModel
from .calculation_details import calculation_detail, calculation_statuses, comparison_status_text
from ..theme import icon


COMPARISON_KEY = "pyrolysis_comparisons"


COMPARISON_HEADERS = (
    "sample_id", "sample_label", "formula", "calculated", "laboratory", "difference", "status",
    "calculation_status", "laboratory_column_id", "laboratory_column_name", "laboratory_input_unit",
    "comparison_unit", "absolute_tolerance", "context_conflict_fields", "context_incomplete",
    "calculation_diagnostics_json",
)


def comparison_rows(outputs, comparisons, labels):
    for column, values in outputs:
        formula = column.metadata["calculation"]["formula"]
        if formula not in comparisons:
            continue
        lab = column.metadata["lab_comparison"]
        diagnostics = column.metadata["diagnostics"]
        for rid, status, number, difference, fields, incomplete in comparisons[formula].itertuples(name=None):
            info = diagnostics[rid]
            fields = fields or (info.get("fields", ()) if info["status"] == "context_conflict" else ())
            yield (rid, labels.get(rid, rid), formula, values[rid], None if isnan(number) else number,
                   None if isnan(difference) else difference, status, info["status"],
                   lab["column_id"], lab["column_name"], lab["unit"], lab["tolerance_unit"], lab["tolerance"],
                   ",".join(fields), "yes" if incomplete else "",
                   json.dumps(info, ensure_ascii=False, allow_nan=False))


class SaveComparisonCommand(QUndoCommand):
    def __init__(self, model, run_id, entry):
        super().__init__(tr('Сохранить сверку в проекте'))
        self.model, self.run_id, self.entry = model, run_id, entry

    def redo(self):
        self.model.project.view.setdefault(COMPARISON_KEY, {})[self.run_id] = self.entry
        self.model.notify_changed(data_changed=False)

    def undo(self):
        runs = self.model.project.view[COMPARISON_KEY]
        runs.pop(self.run_id)
        if not runs:
            del self.model.project.view[COMPARISON_KEY]
        self.model.notify_changed(data_changed=False)


class SavedComparisonModel(ResultsModel):
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and index.column() in (6, 7) and role in (
            Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole
        ):
            labels = calculation_statuses()
            row = self.rows[index.row()]
            detail = ""
            if len(row) > 15 and row[15]:
                try:
                    detail = calculation_detail(json.loads(row[15]), row[11], labels)
                except (ValueError, TypeError, KeyError, AttributeError, IndexError):
                    pass
            if index.column() == 7:
                return detail or labels.get(row[7], row[7])
            status = row[6]
            if status in ("match", "mismatch"):
                return comparison_status_text(status, len(row) > 14 and row[14] == "yes")
            if len(row) > 13 and row[13]:
                reason = context_issue_text(row[13].split(","))
            else:
                code = row[7] if status == "calculation_unavailable" else status
                reason = (detail if status == "calculation_unavailable" and detail
                          else labels.get(code, code))
            return tr('Сравнение недоступно: {0}', reason)
        if index.isValid() and index.column() == 1 and role == Qt.ItemDataRole.ToolTipRole:
            row = self.rows[index.row()]
            return f"{row[1]}\n{row[0]}" if row[1] != row[0] else row[0]
        return super().data(index, role)


def valid_comparison_row(row):
    if not isinstance(row, (list, tuple)) or not 13 <= len(row) <= len(COMPARISON_HEADERS):
        return False
    numeric = (3, 4, 5, 12)
    return all(value is None or type(value) in (int, float) if index in numeric else isinstance(value, str)
               for index, value in enumerate(row))


def export_comparison(editor):
    if not editor._comparison_rows or editor._preview_source != source_token(editor.owner):
        return
    editor.export_comparison_rows(comparison_rows(editor._preview_outputs, editor._comparison_rows, editor._sample_labels))


def export_comparison_rows(editor, rows, *, historical=False):
    path, _ = QFileDialog.getSaveFileName(editor, tr('Выгрузить сверку…'), "pyrolysis-comparison.csv",
                                         tr('Таблица CSV (*.csv)'))
    if not path or (not historical and editor._preview_source != source_token(editor.owner)):
        return

    def save():
        # Older snapshots may lack one or both trailing context fields.
        records = (tuple(row) + ("",) * (len(COMPARISON_HEADERS) - len(row)) for row in rows)
        write_excel_csv(path, COMPARISON_HEADERS, records, ".")

    try:
        if large_project(editor.owner.model.project):
            run_with_progress(editor, tr('Выгрузить сверку…'), save)
        else:
            save()
    except (OSError, ValueError) as exc:
        editor.set_status(tr('Ошибка: {0}', exc))


def save_comparison(editor):
    token = editor._preview_source
    if not editor._comparison_rows or token != source_token(editor.owner):
        return
    outputs, comparisons, labels = editor._preview_outputs, editor._comparison_rows, editor._sample_labels
    created = timestamp()

    def capture():
        settings = {}
        for column, _ in outputs:
            config = column.metadata["calculation"]
            formula = config["formula"]
            if formula in comparisons:
                settings[formula] = {
                    "calculation": deepcopy({k: v for k, v in config.items() if k != "scope_row_ids"}),
                    "laboratory": deepcopy(column.metadata["lab_comparison"]),
                }
        rows = [list(row) for row in comparison_rows(outputs, comparisons, labels)]
        return archive_protocol({"format": "autopyro.pyrolysis-comparison", "version": 1,
                                 "settings": settings, "rows": rows})

    title = tr('Сохранить сверку в проекте')
    try:
        protocol = (run_with_progress(editor, title, capture)
                    if large_project(editor.owner.model.project) else capture())
    except (ValueError, TypeError) as exc:
        editor.set_status(tr('Ошибка: {0}', message(exc)))
        return
    if editor._preview_source != token or source_token(editor.owner) != token:
        return
    entry = {"created_at": created, "formulas": list(comparisons), "protocol": protocol}
    editor.owner.model.undo.push(SaveComparisonCommand(editor.owner.model, new_id(), entry))
    editor.saved_comparisons_button.setEnabled(True)
    editor.set_status(tr('Сверка сохранена в проекте.'))


def open_saved_comparisons(editor):
    runs = editor.owner.model.project.view.get(COMPARISON_KEY, {})
    if not runs:
        return
    if not isinstance(runs, dict):
        editor.set_status(tr('Не удалось открыть снимок сверки: данные повреждены или формат неизвестен.'))
        return
    dialog = QDialog(editor)
    dialog.setWindowTitle(tr('Сохранённые сверки…'))
    dialog.resize(editor.size().boundedTo(QSize(1000, 520)))
    layout = QVBoxLayout(dialog)
    selector = QComboBox()
    for run_id, run in reversed(list(runs.items())):
        if (isinstance(run, dict) and isinstance(run.get("created_at"), str)
                and isinstance(run.get("formulas"), list)
                and all(isinstance(formula, str) for formula in run["formulas"])):
            caption = f"{run['created_at'][:19].replace('T', ' ')} UTC · {', '.join(run['formulas'])}"
        else:
            caption = tr('Повреждённый снимок сверки')
        selector.addItem(caption, run_id)
    layout.addWidget(selector)
    details = QLabel()
    details.setTextFormat(Qt.TextFormat.PlainText)
    details.setWordWrap(True)
    layout.addWidget(details)
    table = QTableView()
    headers = ("UUID пробы", "Проба", "Показатель", "Расчёт", "Лаборатория",
               "Разница", "Статус", "Статус расчёта", "UUID колонки", "Колонка",
               "Единица входа", "Единица сравнения", "Абсолютный допуск")
    model = SavedComparisonModel(headers, [], table)
    table.setModel(model)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    for column in (0, *range(8, len(headers))):
        table.setColumnHidden(column, True)
    table.setColumnWidth(1, 175)
    table.setColumnWidth(6, 165)
    table.horizontalHeader().setStretchLastSection(True)
    layout.addWidget(table)
    export = QPushButton(icon('log-out'), tr('Выгрузить сверку…'))
    export.clicked.connect(lambda: editor.export_comparison_rows(model.rows, historical=True))
    layout.addWidget(export)
    close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    close.rejected.connect(dialog.reject)
    layout.addWidget(close)

    def load():
        model.set_rows([])
        export.setEnabled(False)
        try:
            payload = runs[selector.currentData()]["protocol"]
            snapshot = (run_with_progress(dialog, tr('Открытие сохранённой сверки…'), lambda: read_protocol(payload))
                        if isinstance(payload, str) else payload)
            if (snapshot.get("format") != "autopyro.pyrolysis-comparison" or snapshot.get("version") != 1
                    or not isinstance(snapshot.get("rows"), list) or not isinstance(snapshot.get("settings"), dict)):
                raise ValueError
            if not all(valid_comparison_row(row) for row in snapshot["rows"]):
                raise ValueError
            captions = [tr('{0}: {1} · абсолютный допуск {2:g} [{3}]', formula, lab["column_name"],
                           lab["tolerance"], lab["tolerance_unit"])
                        for formula, entry in snapshot["settings"].items() if (lab := entry["laboratory"])]
        except (ValueError, AttributeError, KeyError, TypeError):
            details.setText(tr('Не удалось открыть снимок сверки: данные повреждены или формат неизвестен.'))
            return
        model.set_rows(snapshot["rows"])
        export.setEnabled(bool(model.rows))
        details.setText("\n".join([tr('Снимок результатов · строк: {0}', len(model.rows)), *captions]))

    selector.currentIndexChanged.connect(load)
    load()
    dialog.exec()
