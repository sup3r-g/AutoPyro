"""Schedule interpretation, capture project inputs and manage cancellable calculations."""

from __future__ import annotations

from copy import deepcopy
from threading import Event

from PySide6.QtCore import QTimer

from autopyro.schemes.interpretation import check_saved_binding, evaluate_project
from autopyro.schemes.interpretation_input import capture_interpretation_steps
from autopyro.schemes.interpretation_revisions import retain_previous_revision, same_snapshot_metadata

from ..i18n import tr, message
from ..tasks import TaskCancelled, Worker


def initialize(sheet):
    sheet.timer = QTimer(sheet)
    sheet.timer.setSingleShot(True)
    sheet.timer.setInterval(200)
    sheet.timer.timeout.connect(sheet.start_calculation)
    sheet.calculation_worker = None
    sheet.calculation_cancelled = None
    sheet.calculation_pending = False
    sheet.calculation_revision = 0
    sheet.calculation_steps = None
    sheet.calculation_steps_timer = QTimer(sheet)
    sheet.calculation_steps_timer.timeout.connect(sheet.advance_calculation_steps)


def selection_changed(sheet, *args):
    if sheet.scope.currentData() == "selected":
        sheet.schedule()


def show_stale_reason(sheet):
    project = sheet.owner.model.project
    saved = project.view.get("interpretation_result") or {}
    if not saved.get("stale"):
        return
    reason = saved.get("stale_reason") or {}
    code = reason.get("code")
    row_id = reason.get("row_id")
    column_id = reason.get("column_id")
    if code in ("input_changed", "measurement_changed", "uncertainty_changed"):
        labels = {
            "input_changed": "Изменилось значение",
            "measurement_changed": "Изменилось рабочее измерение или его версия",
            "uncertainty_changed": "Изменилась неопределённость измерения",
        }
        column = next((c.name for c in project.columns if c.id == column_id), column_id)
        detail = tr('{0}. Показатель {1}, проба {2}.', tr(labels[code]), column, row_id)
    elif code == "scheme_changed":
        detail = tr('Схема «{0}» обновлена с редакции {1} до {2}.',
                    reason["scheme_name"], reason["old_revision"], reason["new_revision"])
    elif code == "sample_removed":
        detail = tr('Проба {0} удалена.', row_id)
    elif code == "binding_changed":
        fields = {
            "x_id": "колонка X", "y_id": "колонка Y",
            "x_unit": "единица X", "y_unit": "единица Y",
            "text_numbers": "чтение чисел из текста",
            "scheme_id": "выбранная схема", "scope": "область данных",
            "comparison": "Показать стандартные схемы сравнения",
        }
        names = ", ".join(tr(fields.get(key, key)) for key in reason["fields"])
        detail = tr('Изменились параметры расчёта ({0}).', names)
    else:
        messages = {
            "axis_definition_changed": "Изменились единицы, тип или формула показателя на оси либо колонка удалена.",
            "qc_changed": "Изменились предупреждения или исключения контроля качества.",
            "import_changed": "Таблица изменилась после импорта или его отмены.",
            "database_changed": "Изменился снимок базы данных.",
            "project_refreshed": "Проект обновлён. Требуется повторная проверка входов.",
            "snapshot_unverifiable": "Проверить входы сохранённого снимка не удалось.",
        }
        detail = tr(messages.get(code, "Причина не записана в сохранённом снимке."))
    if reason.get("diagram"):
        detail = tr('{0} · {1}', reason["diagram"], detail)
    sheet.previous_stale_notice = tr('Сохранённый снимок устарел. {0}', detail)
    sheet.stale_notice.setText(sheet.previous_stale_notice)
    sheet.stale_notice.show()


def schedule(sheet, *args):
    if sheet.closed:
        return
    sheet.invalidate_calculation()
    if hasattr(sheet, 'plot') and sheet.scheme:
        check_saved_binding(sheet.owner.model.project, {
            "scheme_id": sheet.scheme["id"], "x_id": sheet.x.currentData(),
            "y_id": sheet.y.currentData(), "x_unit": sheet.x_unit.text().strip(),
            "y_unit": sheet.y_unit.text().strip(), "scope": sheet.scope.currentData(),
            "text_numbers": sheet.text_numbers.isChecked(),
            "comparison": sheet.comparison.isChecked(),
        })
    sheet.show_stale_reason()
    sheet.clear_result_view()
    if sheet.scheme and not sheet.editor_dirty:
        sheet.status.setText(tr('Ожидается расчёт интерпретации…'))
    sheet.timer.start()


def calculation_request(sheet, project):
    from autopyro.schemes.interpretation_comparison import comparison_panels

    scope = sheet.scope.currentData()
    row_ids = tuple(sheet.owner.analysis_scope.rows())
    check_saved_binding(project, {"scope": scope}, scope_rows=row_ids)
    sheet.show_stale_reason()
    binding = dict(
        scheme=deepcopy(sheet.scheme), x_id=sheet.x.currentData(), y_id=sheet.y.currentData(),
        x_unit=sheet.x_unit.text().strip(), y_unit=sheet.y_unit.text().strip(),
        text_numbers=sheet.text_numbers.isChecked(),
    )
    panels = (comparison_panels(project, sheet.standard_schemes, binding, binding['text_numbers'])
              if sheet.comparison.isChecked() else [])
    return row_ids, binding, panels, scope


def calculate_report(project, request, checkpoint=None):
    from autopyro.schemes.interpretation_comparison import comparison_reports, type_agreement

    ids, binding, panels, scope = request
    report = evaluate_project(project, ids, **binding, checkpoint=checkpoint)
    report['scope'] = scope
    report['comparisons'] = comparison_reports(project, ids, panels, primary_report=report, checkpoint=checkpoint)
    report['comparison_panels'] = panels
    if panels:
        report['type_agreement'] = type_agreement(report['comparisons'], len(report['results']))
    return report


def invalidate_calculation(sheet):
    sheet.calculation_revision += 1
    sheet.calculation_pending = False
    sheet.calculation_steps_timer.stop()
    sheet.calculation_steps = None
    if sheet.calculation_cancelled is not None:
        sheet.calculation_cancelled.set()


def cancel_calculation(sheet):
    sheet.invalidate_calculation()
    sheet.timer.stop()
    sheet.cancel_calculation_button.hide()
    text = tr('Расчёт интерпретации отменён.')
    sheet.status.setText(text)
    sheet.explanation.setPlainText(text)
    sheet.previous_result_button.setVisible(bool(sheet.owner.model.project.view.get('interpretation_result')))


def start_calculation(sheet):
    if sheet.closed or not sheet.scheme or sheet.editor_dirty:
        return
    if sheet.calculation_worker is not None:
        sheet.calculation_pending = True
        return
    project = sheet.owner.model.project
    try:
        request = sheet.calculation_request(project)
        _, binding, panels, _ = request
        steps = capture_interpretation_steps(project, request[0], [
            binding, *(panel['binding'] for panel in panels if 'binding' in panel),
        ])
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        sheet.calculation_error(exc)
        return
    sheet.begin_calculation_steps(
        project, steps, lambda snapshot: sheet.run_calculation(project, request, snapshot),
        tr('Подготовка данных интерпретации…'),
    )


def begin_calculation_steps(sheet, project, steps, finished, status):
    sheet.calculation_steps = (sheet.calculation_revision, project, steps, finished)
    sheet.status.setText(status)
    sheet.cancel_calculation_button.show()
    sheet.calculation_steps_timer.start(0)


def advance_calculation_steps(sheet):
    if sheet.calculation_steps is None:
        return
    revision, project, steps, finished = sheet.calculation_steps
    if sheet.closed or revision != sheet.calculation_revision or sheet.owner.model.project is not project:
        sheet.invalidate_calculation()
        sheet.cancel_calculation_button.hide()
        return
    try:
        next(steps)
    except StopIteration as done:
        sheet.calculation_steps_timer.stop()
        sheet.calculation_steps = None
        finished(done.value)
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        sheet.invalidate_calculation()
        sheet.cancel_calculation_button.hide()
        sheet.calculation_error(exc)


def run_calculation(sheet, project, request, snapshot):
    revision = sheet.calculation_revision
    cancelled = sheet.calculation_cancelled = Event()

    def checkpoint():
        if cancelled.is_set():
            raise TaskCancelled()

    calculate_report = sheet.calculate_report
    report_storage = sheet.report_storage
    previous = project.view.get('interpretation_result')

    def calculate():
        report = calculate_report(snapshot, request, checkpoint)
        checkpoint()
        saved, predecessor = report_storage(previous, report)
        checkpoint()
        return report, saved, predecessor

    worker = sheet.calculation_worker = Worker(calculate, sheet)
    outcome = {}
    worker.succeeded.connect(lambda report: outcome.update(report=report))
    worker.failed.connect(lambda error: outcome.update(error=error))

    def finished():
        sheet.calculation_worker = sheet.calculation_cancelled = None
        worker.function = None
        worker.deleteLater()
        sheet.cancel_calculation_button.hide()
        if (not sheet.closed and revision == sheet.calculation_revision
                and sheet.owner.model.project is project):
            if 'report' in outcome:
                report, saved, predecessor = outcome['report']
                sheet.begin_calculation_steps(
                    project, sheet.prepare_report_steps(project, report),
                    lambda prepared: sheet.install_report(report, prepared, saved, predecessor),
                    tr('Подготовка таблицы результатов…'),
                )
            elif 'error' in outcome:
                sheet.calculation_error(outcome['error'])
        if sheet.calculation_pending and not sheet.timer.isActive():
            sheet.calculation_pending = False
            sheet.start_calculation()

    worker.finished.connect(finished)
    sheet.cancel_calculation_button.show()
    sheet.status.setText(tr('Выполняется расчёт интерпретации…'))
    worker.start()


def calculate(sheet):
    """Immediate calculation for callers that require the result on return."""
    if not sheet.scheme or sheet.editor_dirty:
        return
    sheet.timer.stop()
    sheet.invalidate_calculation()
    project = sheet.owner.model.project
    try:
        report = sheet.calculate_report(project, sheet.calculation_request(project))
        steps = sheet.prepare_report_steps(project, report)
        while True:
            try:
                next(steps)
            except StopIteration as done:
                previous = project.view.get('interpretation_result')
                saved, predecessor = sheet.report_storage(previous, report)
                sheet.install_report(report, done.value, saved, predecessor)
                break
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        sheet.calculation_error(exc)


def calculation_error(sheet, error):
    sheet.cancel_calculation_button.hide()
    sheet.clear_result_view()
    text = tr('Интерпретация не выполнена: {0}', message(error))
    sheet.status.setText(text)
    sheet.explanation.setPlainText(text)


def report_storage(cls, previous, report):
    predecessor = cls.report_predecessor(previous, report)
    if predecessor is None and same_snapshot_metadata(previous, report):
        return None, None
    return deepcopy(report), predecessor


def report_predecessor(previous, report):
    view = {'interpretation_result': previous}
    retain_previous_revision(view, report)
    return view.get('interpretation_previous_result')


def stop_calculation(sheet):
    sheet.invalidate_calculation()
    if sheet.calculation_worker is not None:
        sheet.calculation_worker.wait()
    sheet.cancel_calculation_button.hide()
