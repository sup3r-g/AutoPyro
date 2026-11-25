"""Statistics worksheet and its public dialog interface."""

from __future__ import annotations

from ..i18n import tr
from ..tasks import TaskDialog
from ..widgets.results import ResultsModel as ResultsModel
from . import (
    statistics_calculation,
    statistics_charts,
    statistics_export,
    statistics_forms,
    statistics_inputs,
    statistics_results,
    statistics_samples,
)
from .statistics_calculation import (
    DIAGNOSTIC_TABS as DIAGNOSTIC_TABS,
    SNAPSHOT_BATCH_SIZE as SNAPSHOT_BATCH_SIZE,
    SNAPSHOT_TICK_BUDGET as SNAPSHOT_TICK_BUDGET,
)
from .statistics_charts import SCATTER_MATRIX_MAX_COLUMNS as SCATTER_MATRIX_MAX_COLUMNS


class StatisticsDialog(TaskDialog):
    """Compose the modeless statistics workspace and its analysis tools."""

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.setWindowTitle(tr('Статистика · текущие данные и явная политика исключений'))
        self.resize(1200, 780)
        statistics_calculation.initialize(self)
        statistics_forms.build(self)

    def restore_parameters(self):
        return statistics_inputs.restore_parameters(self)

    def add_comparison_tab(self):
        return statistics_forms.add_comparison_tab(self)

    def fill_comparison_views(self, selected):
        return statistics_inputs.fill_comparison_views(self, selected)

    def sync_comparison_views(self):
        return statistics_inputs.sync_comparison_views(self)

    def rename_comparison_scope(self, old_scope, new_scope):
        return statistics_inputs.rename_comparison_scope(self, old_scope, new_scope)

    def search_properties(self, text):
        return statistics_inputs.search_properties(self, text)

    @staticmethod
    def checked(widget):
        return statistics_inputs.checked(widget)

    def restore_settings(self, settings):
        return statistics_inputs.restore_settings(self, settings)

    def selected_row_ids(self):
        return statistics_inputs.selected_row_ids(self)

    def selection_changed(self, *args):
        return statistics_inputs.selection_changed(self, *args)

    def sync_columns(self, *args):
        return statistics_inputs.sync_columns(self, *args)

    def request_policy_refresh(self, *args):
        return statistics_calculation.request_policy_refresh(self, *args)

    def request_refresh(self, *args, reuse_snapshot=False):
        return statistics_calculation.request_refresh(self, *args, reuse_snapshot=reuse_snapshot)

    def calculate(self):
        return statistics_calculation.calculate(self)

    def cancel_snapshot(self):
        return statistics_calculation.cancel_snapshot(self)

    def advance_snapshot(self):
        return statistics_calculation.advance_snapshot(self)

    @staticmethod
    def diagnostic_rows(tab, data, report, options, pairs, check_cancelled):
        return statistics_calculation.diagnostic_rows(tab, data, report, options, pairs, check_cancelled)

    def run_statistics_task(self, compute, callback, *, diagnostic=None):
        return statistics_calculation.run_statistics_task(self, compute, callback, diagnostic=diagnostic)

    def cancel_calculation(self):
        return statistics_calculation.cancel_calculation(self)

    def calculate_diagnostic(self, tab):
        return statistics_calculation.calculate_diagnostic(self, tab)

    def after_work(self):
        return statistics_calculation.after_work(self)

    def clear_tables(self):
        return statistics_results.clear_tables(self)

    def display_report(self, report, table_names=None):
        return statistics_results.display_report(self, report, table_names)

    def display_result_table(self, report, name):
        return statistics_results.display_result_table(self, report, name)

    def selected_result_group(self):
        return statistics_results.selected_result_group(self)

    def fill_result_groups(self, report):
        return statistics_results.fill_result_groups(self, report)

    def filter_result_groups(self):
        return statistics_results.filter_result_groups(self)

    def current_name(self):
        return statistics_results.current_name(self)

    def update_result_pins(self):
        return statistics_results.update_result_pins(self)

    def pin_result_columns(self, name):
        return statistics_results.pin_result_columns(self, name)

    @staticmethod
    def result_row(index):
        return statistics_results.result_row(index)

    def clear_chart(self):
        return statistics_charts.clear_chart(self)

    def clear_details(self):
        return statistics_results.clear_details(self)

    def tab_changed(self):
        return statistics_results.tab_changed(self)

    def update_navigation_hint(self):
        return statistics_results.update_navigation_hint(self)

    def update_calculation_context(self):
        return statistics_results.update_calculation_context(self)

    def result_action_index(self):
        return statistics_results.result_action_index(self)

    def update_result_action(self, *args):
        return statistics_results.update_result_action(self, *args)

    def activate_result_action(self):
        return statistics_results.activate_result_action(self)

    def update_export_buttons(self, *args):
        return statistics_results.update_export_buttons(self, *args)

    def show_distribution(self, index):
        return statistics_charts.show_distribution(self, index)

    def show_correlation_matrix(self, index):
        return statistics_charts.show_correlation_matrix(self, index)

    def correlation_group(self, row):
        return statistics_charts.correlation_group(self, row)

    def show_scatter_matrix(self):
        return statistics_charts.show_scatter_matrix(self)

    def matrix_payload(self, kind, group, entries):
        return statistics_charts.matrix_payload(self, kind, group, entries)

    def open_matrix(self, payload):
        return statistics_charts.open_matrix(self, payload)

    def load_matrix_style(self):
        return statistics_charts.load_matrix_style(self)

    def load_matrix(self):
        return statistics_charts.load_matrix(self)

    def save_matrix(self):
        return statistics_charts.save_matrix(self)

    def matrix_export_busy_changed(self, busy):
        return statistics_charts.matrix_export_busy_changed(self, busy)

    def matrix_export_completed(self, outcome):
        return statistics_charts.matrix_export_completed(self, outcome)

    def matrix_export_failed(self, error):
        return statistics_charts.matrix_export_failed(self, error)

    def chart_group_label(self, group):
        return statistics_charts.chart_group_label(self, group)

    def show_chart(self, figure, *, height, width=0):
        return statistics_charts.show_chart(self, figure, height=height, width=width)

    def show_comparison_samples(self, index):
        return statistics_samples.show_comparison_samples(self, index)

    def show_availability_samples(self, index):
        return statistics_samples.show_availability_samples(self, index)

    def classified_values(self, cid, ids, *, group_index=None):
        return statistics_samples.classified_values(self, cid, ids, group_index=group_index)

    @staticmethod
    def classified_snapshot_values(project, options, cid, ids):
        return statistics_samples.classified_snapshot_values(project, options, cid, ids)

    def show_samples(self, entry, samples, label, scope):
        return statistics_samples.show_samples(self, entry, samples, label, scope)

    def show_repeat_samples(self, index):
        return statistics_samples.show_repeat_samples(self, index)

    def show_threshold_samples(self, index):
        return statistics_samples.show_threshold_samples(self, index)

    def show_interpolation_samples(self, index):
        return statistics_samples.show_interpolation_samples(self, index)

    def display_samples(self, headers, rows, label, property_name, scope, *, anchor_columns=()):
        return statistics_samples.display_samples(self, headers, rows, label, property_name, scope, anchor_columns=anchor_columns)

    def update_sample_action(self, index, *_):
        return statistics_samples.update_sample_action(self, index, *_)

    def go_to_listed_sample(self, index):
        return statistics_samples.go_to_listed_sample(self, index)

    def go_to_outlier(self, index):
        return statistics_samples.go_to_outlier(self, index)

    def go_to_sample(self, rid, scope):
        return statistics_samples.go_to_sample(self, rid, scope)

    def clear_sample_filter_restore(self):
        return statistics_samples.clear_sample_filter_restore(self)

    def restore_sample_filter(self):
        return statistics_samples.restore_sample_filter(self)

    def copy_current(self):
        return statistics_export.copy_current(self)

    def export_current(self):
        return statistics_export.export_current(self)

    def export_snapshot(self):
        return statistics_export.export_snapshot(self)

    def prepare_project_close(self):
        super().prepare_project_close()
        self.closed = True
        self.timer.stop()
        self.cancel_snapshot()
        self.matrix_export.cancel()
        self.clear_sample_filter_restore()

    def release_closed_report(self):
        self.closed = True
        self.matrix_export.cancel()
        self.matrix_export_progress.reset()
        self.restore_sample_filter()
        self.timer.stop()
        self.cancel_snapshot()
        self.report = None
        self.input_snapshot = None
        self.deferred_tables.clear()
        self.clear_details()
        for table in self.tables.values():
            model = table.model()
            if model is not None:
                table.setModel(None)
                model.set_rows([])
                model.deleteLater()

    def reject(self):
        if self.task_running:
            return
        self.release_closed_report()
        super().reject()

    def closeEvent(self, event):
        if not self.task_running:
            self.release_closed_report()
        super().closeEvent(event)
