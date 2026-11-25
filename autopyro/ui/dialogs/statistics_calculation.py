"""Capture isolated statistics inputs, run background calculations and handle cancellation."""

from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import asdict, replace
from threading import Event
from time import perf_counter

from PySide6.QtCore import QTimer

from autopyro.analysis.statistics import (
    StatisticsOptions,
    calculate_statistics,
    compare_statistics,
    interpolation_comparison,
    interpolation_pairs,
    repeat_statistics,
    snapshot_steps,
    threshold_diagnostics,
)

from ..i18n import language, tr, message


SNAPSHOT_BATCH_SIZE = 2048


SNAPSHOT_TICK_BUDGET = 0.012


DIAGNOSTIC_TABS = ("repeats", "thresholds", "interpolation")


def initialize(sheet):
    sheet.report = None
    sheet.requested_diagnostics = set()
    sheet.deferred_tables = {}
    sheet.input_snapshot = None
    sheet.generation = 0
    sheet.pending = False
    sheet.closed = False
    sheet.work_cancelled = None
    sheet.cancel_requested = False
    sheet.running_diagnostic = None
    sheet.timer = QTimer(sheet)
    sheet.timer.setSingleShot(True)
    sheet.timer.setInterval(180)
    sheet.timer.timeout.connect(sheet.calculate)
    sheet.snapshot_timer = QTimer(sheet)
    sheet.snapshot_timer.setSingleShot(True)
    sheet.snapshot_timer.setInterval(1)
    sheet.snapshot_timer.timeout.connect(sheet.advance_snapshot)
    sheet.snapshot_iterator = None
    sheet.snapshot_callback = None


def request_policy_refresh(sheet, *args):
    sheet.request_refresh(reuse_snapshot=True)


def request_refresh(sheet, *args, reuse_snapshot=False):
    if sheet.closed:
        return
    sheet.cancel_snapshot()
    if sheet.work_cancelled is not None:
        sheet.work_cancelled.set()
    sheet.cancel_requested = False
    sheet.generation += 1
    sheet.pending = True
    sheet.report = None
    sheet.policy.hide()
    sheet.stale_results.setVisible(bool(sheet.deferred_tables) or any(
        table.model() is not None and table.model().rowCount() for table in sheet.tables.values()
    ))
    sheet.cancel_calculation_button.setEnabled(True)
    if not reuse_snapshot:
        sheet.input_snapshot = None
    sheet.update_export_buttons()
    sheet.scatter_matrix.setEnabled(False)
    sheet.clear_details()
    sheet.comparison_status.setText(tr('Данные или настройки изменились; ожидается пересчёт…'))
    sheet.status.setText(tr('Данные или настройки изменились; ожидается пересчёт…'))
    sheet.timer.start()


def calculate(sheet):
    if sheet.task_running or sheet.snapshot_iterator is not None:
        return  # completion schedules the pending generation
    options = StatisticsOptions(
        column_ids=sheet.checked(sheet.properties),
        group_ids=sheet.checked(sheet.groups),
        text_numbers=sheet.text_numbers.isChecked(),
        include_interpolated=sheet.interpolated.isChecked(),
        correlations=sheet.correlations.isChecked(),
        min_pairs=sheet.min_pairs.value(),
    )
    sheet.pending = False
    if not options.column_ids:
        sheet.status.setText(tr('Выберите хотя бы один показатель'))
        sheet.clear_tables()
        return
    a_scope, b_scope = (combo.currentData() for combo in sheet.comparison_views)
    diagnostics = sheet.requested_diagnostics.copy()
    project = sheet.owner.model.project
    pairs = interpolation_pairs(project, options.column_ids) if "interpolation" in diagnostics else ()
    correlation_limit = options.correlations and len(options.column_ids) > 30
    correlation_notice = tr('Корреляции пропущены: выбрано более 30 показателей.')
    try:
        row_ids = sheet.selected_row_ids()
        comparison_context = None
        comparison_status = tr('Выберите две сохранённые выборки.')
        a_ids, b_ids = [], []
        if a_scope is not None and b_scope is not None:
            try:
                a_ids = sheet.owner.analysis_scope.rows(a_scope)
                b_ids = sheet.owner.analysis_scope.rows(b_scope)
            except ValueError as exc:
                comparison_status = message(exc)
            else:
                comparison_context = {"a": a_scope.partition(":")[2], "b": b_scope.partition(":")[2],
                                      "a_scope": a_scope, "b_scope": b_scope, "row_ids_a": a_ids,
                                      "row_ids_b": b_ids, "overlap": len(set(a_ids).intersection(b_ids))}
    except (ValueError, KeyError) as exc:
        sheet.status.setText(message(exc))
        sheet.clear_tables()
        return
    settings = {**asdict(options), "scope": sheet.scope.currentData(),
                "comparison_scope_a": a_scope, "comparison_scope_b": b_scope}
    if sheet.owner.model.project.view.get("statistics") != settings:
        sheet.owner.model.project.view["statistics"] = settings
        sheet.owner.model_changed()
    generation = sheet.generation
    scope_label = sheet.scope.currentText()
    locale = language()
    sheet.status.setText(tr('Расчёт: {0}; строк: {1}…', scope_label, len(row_ids)))
    data = None

    def compute(check_cancelled):
        report = calculate_statistics(data, row_ids, replace(options, correlations=False) if correlation_limit else options,
                                      check_cancelled=check_cancelled)
        if comparison_context:
            report["comparison_context"] = comparison_context
            report["comparison"] = compare_statistics(
                data, a_ids, b_ids, options, check_cancelled=check_cancelled, language=locale,
            )
        for tab in DIAGNOSTIC_TABS:
            if tab in diagnostics:
                report[tab] = sheet.diagnostic_rows(tab, data, report, options, pairs, check_cancelled)
        return report

    def result(report):
        if generation != sheet.generation:
            return
        report["scope"] = scope_label
        sheet.report_scope = settings["scope"]
        if correlation_limit:
            report["notices"].append(correlation_notice)
        sheet.input_snapshot = data
        if comparison_context:
            sheet.comparison_status.setText(tr(
                'A: {0} проб · B: {1} проб · общих: {2}. Двойной щелчок по счётчику открывает список проб.',
                len(a_ids), len(b_ids), comparison_context["overlap"],
            ))
        else:
            sheet.comparison_status.setText(comparison_status)
        sheet.display_report(report)

    def start_worker(completed_snapshot):
        nonlocal data
        data = completed_snapshot
        sheet.run_statistics_task(compute, result)

    snapshot_ids = set(row_ids).union(a_ids, b_ids)
    column_ids = options.column_ids + options.group_ids + tuple(cid for pair in pairs for cid in pair)
    cached = sheet.input_snapshot
    if (cached is not None and set(column_ids) == {column.id for column in cached.columns}
            and snapshot_ids == {row.id for row in cached.rows}):
        start_worker(cached)
        return
    sheet.snapshot_iterator = snapshot_steps(project, snapshot_ids, column_ids, batch_size=SNAPSHOT_BATCH_SIZE)
    sheet.snapshot_callback = start_worker
    sheet.advance_snapshot()


def cancel_snapshot(sheet):
    sheet.snapshot_timer.stop()
    sheet.snapshot_iterator = None
    sheet.snapshot_callback = None


def advance_snapshot(sheet):
    steps = sheet.snapshot_iterator
    if steps is None:
        return
    deadline = perf_counter() + SNAPSHOT_TICK_BUDGET
    while True:
        try:
            data = next(steps)
        except (ValueError, KeyError) as exc:
            sheet.cancel_snapshot()
            sheet.status.setText(message(exc))
            sheet.clear_tables()
            return
        if steps is not sheet.snapshot_iterator:
            return
        if data is not None:
            callback = sheet.snapshot_callback
            sheet.cancel_snapshot()
            callback(data)
            return
        if perf_counter() >= deadline:
            sheet.snapshot_timer.start()
            return


def diagnostic_rows(tab, data, report, options, pairs, check_cancelled):
    if tab == "repeats":
        return repeat_statistics(data, report, check_cancelled=check_cancelled)
    if tab == "thresholds":
        return threshold_diagnostics(data, report["row_ids"], options, group_row_ids=report["group_row_ids"],
                                     check_cancelled=check_cancelled)
    return interpolation_comparison(data, report, pairs, options, check_cancelled=check_cancelled)


def run_statistics_task(sheet, compute, callback, *, diagnostic=None):
    cancelled = sheet.work_cancelled = Event()
    sheet.running_diagnostic = diagnostic
    sheet.cancel_calculation_button.setEnabled(True)

    def check_cancelled():
        if cancelled.is_set():
            raise CancelledError()

    sheet.run_task(lambda: compute(check_cancelled), callback, on_finished=sheet.after_work,
                  disable_dialog=False, cancelled=cancelled)


def cancel_calculation(sheet):
    sheet.cancel_requested = True
    sheet.pending = False
    sheet.generation += 1
    sheet.timer.stop()
    sheet.cancel_snapshot()
    if sheet.work_cancelled is not None:
        sheet.work_cancelled.set()
    if sheet.report is not None:
        sheet.requested_diagnostics.intersection_update(sheet.report)
    sheet.cancel_calculation_button.setEnabled(False)
    sheet.status.setText(tr('Расчёт отменён.'))
    sheet.update_navigation_hint()


def calculate_diagnostic(sheet, tab):
    report, data = sheet.report, sheet.input_snapshot
    if report is None or data is None:
        sheet.request_refresh()
        return
    options = StatisticsOptions(**report["options"])
    pairs = ()
    if tab == "interpolation":
        pairs = interpolation_pairs(sheet.owner.model.project, options.column_ids)
        available = {column.id for column in data.columns}
        if any(cid not in available for pair in pairs for cid in pair):
            sheet.request_refresh()
            return

    generation = sheet.generation
    sheet.cancel_requested = False
    sheet.status.setText(tr('Расчёт: {0}; строк: {1}…', report["scope"], len(report["row_ids"])))

    def result(rows):
        if generation != sheet.generation:
            return
        sheet.display_report({**report, tab: rows}, (tab,))

    sheet.run_statistics_task(lambda check: sheet.diagnostic_rows(tab, data, report, options, pairs, check),
                             result, diagnostic=tab)


def after_work(sheet):
    sheet.work_cancelled = None
    diagnostic, sheet.running_diagnostic = sheet.running_diagnostic, None
    sheet.cancel_calculation_button.setEnabled(sheet.pending)
    if sheet.pending:
        sheet.timer.start()
    elif sheet.cancel_requested:
        # A user can request a diagnostic again before its cancelled worker finishes.
        if sheet.report is not None:
            tab = next((name for name in DIAGNOSTIC_TABS
                        if name in sheet.requested_diagnostics and name not in sheet.report), None)
            if tab:
                sheet.calculate_diagnostic(tab)
    elif sheet.report is None:
        sheet.status.setText(tr('Расчёт не выполнен; проверьте данные и параметры'))
    else:
        if diagnostic is not None and diagnostic not in sheet.report:
            sheet.requested_diagnostics.discard(diagnostic)
        next_tab = next((tab for tab in DIAGNOSTIC_TABS
                         if tab in sheet.requested_diagnostics and tab not in sheet.report), None)
        if next_tab:
            sheet.calculate_diagnostic(next_tab)
        else:
            sheet.update_navigation_hint()
