"""Read previous interpretation results and export the current report."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QMessageBox,
    QTableView,
    QVBoxLayout,
)

from autopyro.schemes.interpretation import STATUSES
from autopyro.schemes.interpretation_export import write_result_csv, write_result_snapshot

from ..i18n import tr, message
from ..widgets.results import SampleResultsModel, sample_labels


def show_previous_result(sheet):
    project = sheet.owner.model.project
    report = project.view.get('interpretation_result')
    if not report or 'results' not in report:
        return
    dialog = QDialog(sheet)
    dialog.setWindowTitle(tr('Предыдущий расчёт интерпретации'))
    dialog.resize(760, 480)
    layout = QVBoxLayout(dialog)
    note = QLabel(tr('Предыдущий расчёт. Актуальность для текущих настроек не подтверждена.'))
    note.setWordWrap(True)
    layout.addWidget(note)
    table = QTableView(dialog)
    rows = [[item['row_id'], tr(STATUSES[item['status']]), message(item['reason'])]
            for item in report['results']]
    table.setModel(SampleResultsModel([tr('Проба'), tr('Статус'), tr('Обоснование')],
                                     rows, sample_labels(project, (row[0] for row in rows)), table))
    table.horizontalHeader().setStretchLastSection(True)
    layout.addWidget(table)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=dialog)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.exec()
    dialog.deleteLater()


def update_export_labels(sheet):
    visible = sheet.export_scope.currentData() == 'visible'
    sheet.export_json.setText(tr('Видимый снимок JSON…') if visible else tr('Полный снимок JSON…'))
    sheet.export_csv.setText(tr('Видимые результаты CSV…') if visible else tr('Полные результаты CSV…'))


def export_positions(sheet, report):
    if sheet.export_scope.currentData() == 'all':
        return range(len(report['results']))
    return sheet.visible_positions


def save_report(sheet):
    report = sheet.report
    if report is None:
        return
    project = sheet.owner.model.project
    scope = sheet.export_scope.currentData()
    positions = sheet.export_positions(report)
    visible_count = len(sheet.visible_positions)
    mode = sheet.result_filter.currentData()
    extra = tuple(key for key, action in sheet.extra_filter_actions.items() if action.isChecked())
    path, _ = QFileDialog.getSaveFileName(
        sheet, tr('Снимок интерпретации'), "interpretation.json", "JSON (*.json)"
    )
    if path:
        if sheet.report is not report or sheet.owner.model.project is not project:
            QMessageBox.warning(sheet, tr('Экспорт'), tr('Результаты изменились. Повторите экспорт.'))
            return
        try:
            write_result_snapshot(
                path, project, report, positions, scope=scope, filter_mode=mode,
                visible_count=visible_count, additional_filters=extra,
            )
        except (OSError, ValueError) as exc:
            QMessageBox.warning(sheet, tr('Экспорт'), message(exc))


def save_csv(sheet):
    report = sheet.report
    if report is None:
        return
    project = sheet.owner.model.project
    positions = sheet.export_positions(report)
    path, _ = QFileDialog.getSaveFileName(
        sheet, tr('Результаты интерпретации'), "interpretation.csv", "CSV (*.csv)"
    )
    if path:
        if sheet.report is not report or sheet.owner.model.project is not project:
            QMessageBox.warning(sheet, tr('Экспорт'), tr('Результаты изменились. Повторите экспорт.'))
            return

        try:
            write_result_csv(path, project, report, positions,
                             [panel['name'] for panel in sheet.comparison_panels], sheet.comparison_cells,
                             sample_labels=sheet.table.model().labels)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(sheet, tr('Экспорт'), message(exc))
