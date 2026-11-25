"""Prepare sample coverage and calculate non-pyrolysis profiles."""

from __future__ import annotations

from copy import deepcopy
from time import sleep

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QTableView, QVBoxLayout

from autopyro.analysis.nonpyro import METHODS, evaluate_project, preview_profile_coverage

from ..geochemistry import status_label
from ..i18n import tr, message
from ..tasks import TaskCancelled, run_with_progress
from ..widgets.results import ResultsModel


def show_profile_coverage(editor):
    project = editor.owner.model.project
    try:
        config = editor.config()
        context = editor.report_context()
        row_ids = context[1]

        def check():
            return preview_profile_coverage(project, row_ids, config)

        coverage = (run_with_progress(editor, tr('Проверка пригодности проб…'), check)
                    if len(row_ids) >= 256 else check())
        if project is not editor.owner.model.project or context != editor.report_context():
            raise ValueError(tr('Выборка изменилась. Повторите проверку.'))
    except ValueError as exc:
        editor.status.setText(message(exc))
        return

    definition = METHODS[config["method"]]
    sample_column = next((column.id for column in project.columns if column.name == 'Sample / Name'), None)
    examples = {rid for item in coverage["failures"] for rid in item["examples"]}
    positions = editor.owner.model.resolve_row_positions(examples)
    names = {rid: project.rows[index].values.get(sample_column) or tr('Строка {0}', index + 1)
             for rid, index in positions.items()}
    rows = [(tr('Показатель'), tr(definition["outputs"][key][0]), count, '')
            for key, count in coverage["outputs"].items()]
    for item in coverage["failures"]:
        reason = item["reason"]
        role_key, _, detail = reason.partition(': ')
        if role_key in definition["roles"]:
            reason = f"{tr(definition['roles'][role_key]['label'])}: {status_label(detail)}"
        else:
            reason = message(reason)
        rows.append((status_label(item["status"]), reason, item["count"],
                     ', '.join(str(names.get(rid, rid)) for rid in item["examples"])))
    dialog = QDialog(editor)
    dialog.setWindowTitle(tr('Пригодность выбранных проб'))
    dialog.resize(850, 480)
    layout = QVBoxLayout(dialog)
    summary = QLabel(tr('Хотя бы один результат доступен для {0} из {1} строк. '
                        'Число строк для каждого показателя и причины полного пропуска приведены ниже.',
                        coverage["with_values"], coverage["total"]))
    summary.setWordWrap(True)
    layout.addWidget(summary)
    table = QTableView(dialog)
    table.setModel(ResultsModel([tr('Тип'), tr('Показатель или причина'), tr('Строк'),
                                tr('Примеры проб')], rows, table))
    table.setColumnWidth(1, 320)
    table.horizontalHeader().setStretchLastSection(True)
    layout.addWidget(table)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.exec()


def calculate(editor):
    editor.refresh_columns()
    editor.invalidate()
    try:
        config = editor.config()
        project = editor.owner.model.project
        row_ids = editor.row_ids()
        context = editor.report_context()
        row_label = tr('Строка {0}')

        title = tr('Расчёт геохимического профиля…')

        def prepare(update=None):
            def checkpoint(percent):
                update(title, percent)
                # Give the Qt thread time to process progress and cancellation.
                sleep(0.001)

            report = evaluate_project(project, row_ids, config, checkpoint=checkpoint if update else None)
            if update:
                checkpoint(90)
            sample_column = next((c.id for c in project.columns if c.name == 'Sample / Name'), None)
            requested = set(row_ids)
            samples = {row.id: row.values.get(sample_column) or row_label.format(index + 1)
                       for index, row in enumerate(project.rows) if row.id in requested}
            rows = [
                (item["row_id"], samples.get(item["row_id"]) or item["row_id"], item["status"],
                 *(item["values"].get(key) for key in report["definition"]["outputs"]),
                 item["interpretation"], item["warnings"])
                for item in report["results"]
            ]
            return report, rows

        if len(row_ids) >= 256:
            report, rows = run_with_progress(editor, title, prepare, cancellable=True)
        else:
            report, rows = prepare()
        if project is not editor.owner.model.project or context != editor.report_context():
            raise ValueError(tr('Входы изменились во время расчёта. Повторите расчёт.'))
        editor.report = report
        editor.results_model.set_rows(rows)
        if editor.results_model.rowCount() <= 500:
            editor.results.resizeColumnsToContents()
        for column in range(1, editor.results_model.columnCount()):
            editor.results.setColumnWidth(column, min(190, max(100, editor.results.columnWidth(column))))
        editor.results.setColumnWidth(editor.results_model.columnCount() - 2, 310)
        if editor.results_model.rowCount() <= 500:
            editor.results.resizeRowsToContents()
        project.view.setdefault("nonpyro_configs", {})[config["method"]] = deepcopy(
            config
        )
        current = project.view.setdefault("nonpyro_results", {})
        if (previous := current.get(config["method"])) is not None:
            project.view.setdefault("nonpyro_history", {}).setdefault(config["method"], []).append(previous)
        # Protocol contents are read-only. Keep the top-level stale flag independent.
        current[config["method"]] = dict(report)
        project.view["nonpyro_current"] = {"method": config["method"]}
        if editor.module_id:
            project.view.setdefault("geochemistry_current", {})[editor.module_id] = {"method": config["method"]}
        project.event(
            "nonpyro_interpretation",
            method=config["method"],
            definition_hash=report["definition_hash"],
            row_ids=report["row_ids"],
        )
        editor.owner.view_changed()
        editor._report_context = editor.report_context()
        editor.update_result_actions()
        editor.result_tabs.setCurrentWidget(editor.results)
        editor.status.setText(
            tr('Проб: {0}. {1}.', len(report['results']), '; '.join(
                f'{status_label(key)}: {count}' for key, count in report['summary'].items()))
        )
    except TaskCancelled:
        editor.status.setText(tr('Расчёт отменён.'))
    except Exception as exc:
        editor.status.setText(tr('Расчёт не выполнен: {0}', message(exc)))
