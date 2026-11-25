"""Open historical runs and export non-pyrolysis protocols."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableView,
    QVBoxLayout,
)

from autopyro.data_io.file_output import atomic_output
from autopyro.data_io.io import export_table_csv

from ..geochemistry import status_label
from ..i18n import tr, message
from ..tasks import run_with_progress
from ..widgets.results import ResultsModel


def history_reports(editor):
    view = editor.owner.model.project.view
    key = editor.method.currentData()
    previous = view.get("nonpyro_history", {}).get(key, [])
    current = view.get("nonpyro_results", {}).get(key)
    return [*previous, current] if current else previous


def open_history(editor):
    reports = list(reversed(editor.history_reports()))
    labels = [f"{i}. {item['created_at']} · " + tr('строк: {0}', len(item['row_ids']))
              for i, item in enumerate(reports, 1)]
    if not labels:
        return
    selected, accepted = QInputDialog.getItem(editor, tr('История расчётов'), tr('Запуск'), labels, 0, False)
    if not accepted:
        return
    report = reports[labels.index(selected)]
    dialog = QDialog(editor)
    dialog.setWindowTitle(tr('Архивный расчёт') + ' · ' + selected)
    dialog.resize(950, 560)
    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel(report["config"]["context"]))
    outputs = report["definition"]["outputs"]
    table = QTableView(dialog)
    headers = ["UUID", tr('Статус'), *(f"{label} [{unit}]" for label, unit in outputs.values())]
    rows = [(result["row_id"], status_label(result["status"]),
             *(result["values"].get(key) for key in outputs)) for result in report["results"]]
    table.setModel(ResultsModel(headers, rows, table))
    layout.addWidget(table)
    export = QPushButton(tr('Экспорт протокола JSON…'))
    layout.addWidget(export)

    def save():
        path, _ = QFileDialog.getSaveFileName(dialog, tr('Экспорт протокола'), 'nonpyro-history.json', 'JSON (*.json)')
        if path:
            try:
                editor.write_profile_export(dialog, path, report)
            except (ValueError, OSError) as exc:
                QMessageBox.warning(dialog, tr('Экспорт протокола'), message(exc))

    export.clicked.connect(save)
    dialog.exec()


def export(editor):
    if not editor.current_report():
        editor.status.setText(tr('Сначала выполните актуальный расчёт.'))
        return
    path, _ = QFileDialog.getSaveFileName(
        editor,
        tr('Протокол интерпретации'),
        "nonpyro.json",
        tr('Протокол JSON (*.json);;Таблица CSV (*.csv)'),
    )
    if not path:
        return
    try:
        editor.write_profile_export(editor, path, editor.report)
        editor.status.setText(tr('Сохранено: {0}', path))
    except Exception as exc:
        QMessageBox.warning(editor, tr('Протокол'), message(exc))


def write_profile_export(parent, path, report):
    report = dict(report)
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        def write():
            from autopyro.reports.reporting import export_report

            export_report(path, report, "json")
    elif suffix == ".csv":
        outputs = report["definition"]["outputs"]
        headers = ["UUID", tr('Статус'), *[label + f" [{unit}]" for label, unit in outputs.values()],
                   tr('Интерпретация'), tr('Оговорки')]

        def write():
            translate = lru_cache(maxsize=512)(message)
            with atomic_output(path) as temporary:
                export_table_csv(temporary, headers, (
                    (item["row_id"], item["status"], *(item["values"].get(key) for key in outputs),
                     translate(item["interpretation"]), "; ".join(translate(warning) for warning in item["warnings"]))
                    for item in report["results"]
                ))
    else:
        raise ValueError(tr('Выберите JSON или CSV'))
    if len(report["results"]) >= 256:
        run_with_progress(parent, tr('Сохранение протокола геохимии…'), write)
    else:
        write()
