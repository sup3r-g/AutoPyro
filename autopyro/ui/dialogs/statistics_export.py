"""Copy or export statistics tables and reproducible input snapshots."""

from __future__ import annotations

from dataclasses import asdict
import csv
import io
import json

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from autopyro.analysis.statistics import TABLES, export_statistics_csv, report_table
from autopyro.data_io.file_output import atomic_output
from autopyro.data_io.io import spreadsheet_literal

from ..i18n import language, tr, message
from ..tasks import run_with_progress
from .statistics_calculation import DIAGNOSTIC_TABS


def copy_current(sheet):
    if not sheet.copy.isEnabled():
        return
    tab = sheet.current_name()
    headers, rows = report_table(sheet.report, tab)
    group = sheet.selected_result_group() if tab == "thresholds" else None
    if group is not None:
        headers = [f"Группа: {field}" for field in sheet.report["groups"]] + headers
    else:
        group = []
    stream = io.StringIO()
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(map(spreadsheet_literal, headers))
    writer.writerows(map(spreadsheet_literal, group + rows[index]) for index in sheet.tables[tab].model().row_order)
    QApplication.clipboard().setText(stream.getvalue())


def export_current(sheet):
    if not sheet.export.isEnabled():
        return
    report, table_name = sheet.report, sheet.current_name()
    row_order = sheet.tables[table_name].model().row_order
    group = sheet.result_group.currentIndex() - 1 if sheet.selected_result_group() is not None else None
    path, _ = QFileDialog.getSaveFileName(
        sheet, tr('Экспорт текущей вкладки'), "statistics.csv", "CSV (*.csv)"
    )
    if path and report is sheet.report:
        try:
            export_statistics_csv(path, report, table_name, row_order=row_order, group_index=group, language=language())
        except OSError as exc:
            QMessageBox.warning(sheet, tr('Экспорт'), message(exc))


def export_snapshot(sheet):
    report, data = sheet.report, sheet.input_snapshot
    if not report:
        return
    path, _ = QFileDialog.getSaveFileName(
        sheet, tr('Снимок рассчитанных разделов JSON…'), "statistics.json", "JSON (*.json)"
    )
    if path and report is sheet.report:
        def save():
            payload = {
                "format": "autopyro.statistics",
                "version": 1,
                "sections": {
                    "calculated": [name for name in TABLES if name in report
                                   and (name != "correlations" or report["options"]["correlations"])],
                    "pending": [name for name in DIAGNOSTIC_TABS if name not in report],
                    "comparison_configured": "comparison_context" in report,
                    "correlations_enabled": report["options"]["correlations"],
                },
                "report": report,
                "input": {
                    "columns": [asdict(c) for c in data.columns],
                    "source": data.source,
                    "science": asdict(data.science),
                    "view": data.view,
                    "rows": [
                        {"id": r.id, "values": r.values}
                        for r in data.rows
                    ],
                },
            }
            with atomic_output(path) as temporary, temporary.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
        try:
            run_with_progress(sheet, tr('Сохранение снимка статистики…'), save)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(sheet, tr('Экспорт'), message(exc))
