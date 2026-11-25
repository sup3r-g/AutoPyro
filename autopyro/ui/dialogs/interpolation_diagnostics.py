"""Show interpolation constraints, compare methods and export diagnostics."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHeaderView,
    QLabel,
    QMessageBox,
    QTableView,
    QVBoxLayout,
)

from autopyro.analysis.interpolation import validation_scores
from autopyro.analysis.statistics import classify
from autopyro.data_io.file_output import atomic_output
from autopyro.data_io.io import export_table_csv

from ..i18n import tr, message
from ..tasks import TaskCancelled, run_with_progress
from ..widgets.documents import refill_combo, source_token
from ..widgets.results import ResultsModel
from .interpolation_results import BACKGROUND_ROWS


def diagnostic_labels():
    return {
        "eligible": tr('Допущена'),
        "missing_wellbore": tr('Не задан ствол'),
        "stratigraphy_excluded": tr('Исключена по стратиграфии'),
        "insufficient_anchors": tr('Недостаточно опор в группе'),
        "duplicate_coordinate": tr('Повторяющаяся координата X'),
        "interval_sample_excluded": tr('Исключена интервальная проба'),
        "insufficient_spline_anchors": tr('Недостаточно опор для сплайна'),
        "insufficient_segment_anchors": tr('Недостаточно опор на непрерывном участке'),
        "invalid_x": tr('Нет числовой координаты X'),
        "qualified_y": tr('Y содержит ND, предел или нечисловую запись'),
        "y_non_detect": tr('Y не обнаружен (ND)'),
        "y_censored": tr('Y указан с порогом'),
        "y_other": tr('Нечисловая запись Y'),
        "excluded_qc": tr('Исключена по QC'),
        "anchor": tr('Опорное измерение'),
        "proposed": tr('Предложена интерполяция'),
        "extrapolated": tr('Предложена экстраполяция'),
        "missing_group": tr('Не заполнена группировка'),
        "outside_range": tr('Вне диапазона опор'),
        "extrapolation_distance": tr('Слишком далеко от крайней опоры'),
        "extrapolation_gap": tr('Слишком большой разрыв между крайними опорами'),
        "gap_too_large": tr('Превышен максимальный разрыв'),
    }


def duplicate_y_role(value, labels):
    status, _ = classify(value, numeric=True)
    if status == "numeric":
        return tr('Числовая опора')
    if status == "missing":
        return tr('Пустое значение Y')
    return labels[f"y_{status}"]


def diagnostic_rows(diagnostics, labels, reasons, coordinates, sample_labels, anchors, proposals, *,
                    duplicate_details=None, checkpoint=None):
    rows, excluded, unknown = [], [], 0
    fields = (("method", tr('Метод')), ("laboratory", tr('Лаборатория')), ("instrument", tr('Прибор')),
              ("heating_program", tr('Программа нагрева')), ("preparation", tr('Подготовка')),
              ("fraction", tr('Фракция')), ("unit", tr('Единица')), ("basis", tr('База')))
    for index, (rid, item) in enumerate(diagnostics.items()):
        if checkpoint and index % 256 == 0:
            checkpoint()
        status = item["status"]
        x = coordinates.get(rid)
        sample, well = sample_labels.get(rid, (rid, ""))
        role = proposals.get(rid, "anchor" if rid in anchors else status) if status == "eligible" else status
        reason = reasons.get(item.get("reason"), "")
        if "actual" in item and "limit" in item:
            reason = tr('Фактически {0} при пределе {1} (в единицах X)', item["actual"], item["limit"])
        elif "anchors_found" in item and "anchors_required" in item:
            reason = tr('Найдено опор: {0}. Требуется: {1}.', item["anchors_found"], item["anchors_required"])
            if item.get("method_known") is False:
                reason += " " + tr('Метод пробы неизвестен. Опоры с известным методом считаются отдельно.')
            context = item.get("analytical_context", {})
            reason += "".join(f" · {label}: {context[field]}" for field, label in fields if context.get(field))
        if status == "duplicate_coordinate" and duplicate_details:
            reason = duplicate_details.get(rid, reason)
        row = [sample, well, "" if x is None else str(x), labels[role], reason, rid]
        rows.append(row)
        if status != "eligible":
            excluded.append(row)
        unknown += item.get("method_known") is False
    return rows, excluded, unknown


def set_diagnostics(sheet, diagnostics, column_id):
    sheet._diagnostic_rows, sheet._excluded_rows, _ = diagnostics
    sheet._diagnostic_column_id = column_id
    sheet._diagnostics_current = True
    if sheet.property("workspaceTracked"):
        sheet._preview_source = source_token(sheet.owner)
    sheet.update_result_accessibility()
    statuses = sorted({row[3] for row in sheet._diagnostic_rows})
    refill_combo(sheet.diagnostic_status, [(tr('Все статусы'), None), *((label, label) for label in statuses)])
    sheet.show_diagnostics()


def show_diagnostics(sheet):
    rows = sheet._excluded_rows if sheet.only_excluded.isChecked() else sheet._diagnostic_rows
    if status := sheet.diagnostic_status.currentData():
        rows = [row for row in rows if row[3] == status]
    sheet.diagnostic_model.set_rows(rows)
    total = len(sheet._diagnostic_rows)
    text = tr('Показано проб: {0} из {1}', len(rows), total)
    if total and not rows:
        text += "\n" + tr('Фильтры скрыли все строки диагностики.')
    sheet.diagnostic_count.setText(text)
    sheet.export_diagnostics_button.setEnabled(bool(rows) and sheet._diagnostics_current)
    if sheet.diagnostic_hint is not None:
        if sheet._diagnostics_current:
            sheet.diagnostic_hint.setText(tr('Enter или двойной щелчок открывает исходную ячейку пробы.'))
        sheet.diagnostic_hint.setVisible(bool(rows))


def validate_methods(sheet):
    if not sheet.current_diagnostics() or sheet.protocol is None:
        return
    sheet.validation_status.hide()
    contexts = {sheet.measurement_context[rid][1:] for rid in sheet._support_spans}
    if len(contexts) > 1:
        text = tr('Для сравнения методов выберите область с одной единицей и базой Y.')
        sheet.validation_status.setText(text)
        sheet.validation_status.show()
        sheet.announce_summary(text)
        return
    title = tr('Сравнение методов на опорах')
    snapshot = sheet.protocol
    try:
        sampled, common, scores = run_with_progress(
            sheet, title, lambda progress: sheet.run_snapshot_task(validation_scores, snapshot, title, progress),
            cancellable=True,
        )
    except TaskCancelled:
        text = tr('Сравнение отменено')
        sheet.validation_status.setText(text)
        sheet.validation_status.show()
        sheet.announce_summary(text)
        return
    except ValueError as exc:
        QMessageBox.warning(sheet, title, message(exc))
        return
    if not sheet.current_diagnostics() or snapshot is not sheet.protocol:
        return
    dialog = QDialog(sheet)
    dialog.setWindowTitle(title)
    dialog.resize(640, 320)
    layout = QVBoxLayout(dialog)
    note = QLabel(tr('Проверено опор: {0}. Все методы дали прогноз на {1} из них. '
                     'MAE и RMSE рассчитаны только по этим общим опорам, в единицах Y. '
                     'N показывает число прогнозов каждого метода. '
                     'Для интерполяции проверены внутренние опоры с учётом заданного минимума опор и предела разрыва.',
                     sampled, common))
    note.setWordWrap(True)
    layout.addWidget(note)
    view = QTableView(dialog)
    rows = []
    for i in range(sheet.method.count()):
        n, mae, rmse = scores[sheet.method.itemData(i)]
        rows.append((sheet.method.itemText(i), n, "—" if mae is None else f"{mae:.6g}",
                     "—" if rmse is None else f"{rmse:.6g}"))
    if "extrapolation" in scores:
        n, mae, rmse = scores["extrapolation"]
        rows.append((tr('Линейная экстраполяция'), n, "—" if mae is None else f"{mae:.6g}",
                     "—" if rmse is None else f"{rmse:.6g}"))
        note.setText(note.text() + "\n" + tr('Экстраполяция проверена отдельно на крайних опорах с учётом предела расстояния.'))
    view.setModel(ResultsModel([tr('Метод'), "N", "MAE", "RMSE"], rows, view))
    view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    layout.addWidget(view)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.exec()


def export_diagnostics(sheet):
    rows = sheet.diagnostic_model.rows
    if not rows or not sheet.export_diagnostics_button.isEnabled() or not sheet.current_diagnostics():
        return
    title = tr('Экспорт диагностики CSV…')
    path, _ = QFileDialog.getSaveFileName(sheet, title, "interpolation-diagnostics.csv", "CSV (*.csv)")
    if not path:
        return
    if not sheet.current_diagnostics():
        return
    headers = sheet.diagnostic_model.headers
    sheet.export_status.hide()

    committed = False

    def write(progress=None):
        nonlocal committed

        def update(written, total):
            progress(title, min(99, 100 * written // total))

        def before_replace():
            progress(title, 99)

        with atomic_output(path, before_replace=before_replace if progress else None) as temporary:
            export_table_csv(temporary, headers, rows, progress=update if progress else None)
        committed = True

    try:
        if len(rows) >= BACKGROUND_ROWS:
            run_with_progress(sheet, title, write, cancellable=True)
        else:
            write()
    except TaskCancelled:
        if not committed:
            sheet.set_export_status(tr('Экспорт отменён.'))
    except (OSError, ValueError, TypeError) as exc:
        QMessageBox.warning(sheet, title, message(exc))
    if committed:
        sheet.set_export_status(tr('CSV сохранён в файл «{0}». Строк: {1}.', Path(path).name, len(rows)))


def set_export_status(sheet, text):
    sheet.export_status.setText(text)
    sheet.export_status.show()
    sheet.announce_summary(text)


def current_diagnostics(sheet):
    if not sheet._diagnostics_current:
        return False
    if sheet.property("workspaceTracked") and sheet._preview_source != source_token(sheet.owner):
        sheet.invalidate()
        return False
    return True
