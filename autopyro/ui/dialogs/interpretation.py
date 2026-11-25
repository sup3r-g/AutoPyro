"""Interpretation worksheet and its public dialog interface."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDialog

from ..i18n import tr
from . import (
    interpretation_calculation,
    interpretation_chart,
    interpretation_forms,
    interpretation_reports,
    interpretation_results,
    interpretation_review,
    interpretation_schemes,
)
from .interpretation_chart import SPLITTER_SETTING as SPLITTER_SETTING


class InterpretationDialog(QDialog):
    """Compose the scheme editor, interpretation results and embedded plot."""

    scheme_saved = Signal()

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.closed = False
        self.scheme = None
        interpretation_results.initialize(self)
        self.editor_dirty = False
        self.setWindowTitle(tr('Интерпретация · геометрические схемы Shapely'))
        self.resize(1200, 800)
        interpretation_calculation.initialize(self)
        interpretation_forms.build(self)

    def sync_columns(self, *args):
        return interpretation_schemes.sync_columns(self, *args)

    def scheme_columns(self, axis):
        return interpretation_schemes.scheme_columns(self, axis)

    def scheme_column(self, axis):
        return interpretation_schemes.scheme_column(self, axis)

    def set_scheme(self, scheme, auto_bind=True, *, binding=None):
        return interpretation_schemes.set_scheme(self, scheme, auto_bind, binding=binding)

    def remember_current_scheme(self):
        return interpretation_schemes.remember_current_scheme(self)

    def fill_standard_schemes(self):
        return interpretation_schemes.fill_standard_schemes(self)

    def update_catalog_preview(self, *_):
        return interpretation_schemes.update_catalog_preview(self, *_)

    def load_standard_scheme(self):
        return interpretation_schemes.load_standard_scheme(self)

    def load_scheme(self):
        return interpretation_schemes.load_scheme(self)

    def validate_revision(self, scheme):
        return interpretation_schemes.validate_revision(self, scheme)

    def store_scheme(self):
        return interpretation_schemes.store_scheme(self)

    def geometry_samples(self, row_ids=None):
        return interpretation_schemes.geometry_samples(self, row_ids)

    def commit_graphical_scheme(self, scheme, *, auto_bind=False):
        return interpretation_schemes.commit_graphical_scheme(self, scheme, auto_bind=auto_bind)

    @staticmethod
    def require_reference_cases(scheme, previous=None):
        return interpretation_schemes.require_reference_cases(scheme, previous)

    def check_reference_cases(self):
        return interpretation_schemes.check_reference_cases(self)

    def revision_binding(self, scheme, preferred):
        return interpretation_schemes.revision_binding(self, scheme, preferred)

    def preview_revision(self, before, after, auto_bind):
        return interpretation_schemes.preview_revision(self, before, after, auto_bind)

    def open_geometry_editor(self):
        return interpretation_schemes.open_geometry_editor(self)

    def new_graphical_scheme(self):
        return interpretation_schemes.new_graphical_scheme(self)

    def editor_changed(self):
        return interpretation_schemes.editor_changed(self)

    def clear_result_view(self):
        return interpretation_results.clear_result_view(self)

    def apply_editor(self):
        return interpretation_schemes.apply_editor(self)

    def confirm_replace_draft(self):
        return interpretation_schemes.confirm_replace_draft(self)

    def save_scheme(self):
        return interpretation_schemes.save_scheme(self)

    def select_result_rows(self, row_ids, *, scroll=True):
        """Select report UUIDs without changing the source-table calculation scope."""
        return interpretation_results.select_result_rows(self, row_ids, scroll=scroll)

    def results_selection_changed(self, *args):
        return interpretation_results.results_selection_changed(self, *args)

    def highlight_result_selection(self):
        return interpretation_results.highlight_result_selection(self)

    def update_explanation(self, *args):
        return interpretation_review.update_explanation(self, *args)

    def show_pyrograms(self):
        return interpretation_review.show_pyrograms(self)

    def edit_review(self):
        return interpretation_review.edit_review(self)

    def show_summary(self):
        return interpretation_review.show_summary(self)

    def update_review_queue_count(self):
        return interpretation_review.update_review_queue_count(self)

    def update_comparison_actions(self):
        return interpretation_review.update_comparison_actions(self)

    def set_comparison_base(self, checked):
        return interpretation_review.set_comparison_base(self, checked)

    def show_revisions(self):
        return interpretation_review.show_revisions(self)

    def selection_changed(self, *args):
        return interpretation_calculation.selection_changed(self, *args)

    def show_stale_reason(self):
        return interpretation_calculation.show_stale_reason(self)

    def schedule(self, *args):
        return interpretation_calculation.schedule(self, *args)

    def comparison_cells(self, position):
        return interpretation_results.comparison_cells(self, position)

    @staticmethod
    def format_comparison_cells(position, panels, reports, labels_by_panel):
        return interpretation_results.format_comparison_cells(position, panels, reports, labels_by_panel)

    def sync_agreement_filter(self):
        return interpretation_results.sync_agreement_filter(self)

    def calculation_request(self, project):
        return interpretation_calculation.calculation_request(self, project)

    @staticmethod
    def calculate_report(project, request, checkpoint=None):
        return interpretation_calculation.calculate_report(project, request, checkpoint)

    def invalidate_calculation(self):
        return interpretation_calculation.invalidate_calculation(self)

    def cancel_calculation(self):
        return interpretation_calculation.cancel_calculation(self)

    def show_previous_result(self):
        return interpretation_reports.show_previous_result(self)

    def start_calculation(self):
        return interpretation_calculation.start_calculation(self)

    def begin_calculation_steps(self, project, steps, finished, status):
        return interpretation_calculation.begin_calculation_steps(self, project, steps, finished, status)

    def advance_calculation_steps(self):
        return interpretation_calculation.advance_calculation_steps(self)

    def run_calculation(self, project, request, snapshot):
        return interpretation_calculation.run_calculation(self, project, request, snapshot)

    def calculate(self):
        """Immediate calculation for callers that require the result on return."""
        return interpretation_calculation.calculate(self)

    def calculation_error(self, error):
        return interpretation_calculation.calculation_error(self, error)

    @classmethod
    def report_storage(cls, previous, report):
        return interpretation_calculation.report_storage(cls, previous, report)

    @staticmethod
    def report_predecessor(previous, report):
        return interpretation_calculation.report_predecessor(previous, report)

    def prepare_report_steps(self, project, report):
        return interpretation_results.prepare_report_steps(self, project, report)

    def install_report(self, report, prepared, saved, predecessor):
        return interpretation_results.install_report(self, report, prepared, saved, predecessor)

    def update_result_columns(self):
        return interpretation_results.update_result_columns(self)

    def active_filters(self):
        return interpretation_results.active_filters(self)

    def clear_result_filters(self):
        return interpretation_results.clear_result_filters(self)

    def filter_diagrams(self):
        return interpretation_results.filter_diagrams(self)

    @staticmethod
    def matches_filter(report, position, mode):
        return interpretation_results.matches_filter(report, position, mode)

    @staticmethod
    def matches_review_filter(state, mode):
        return interpretation_results.matches_review_filter(state, mode)

    def apply_result_filter(self):
        return interpretation_results.apply_result_filter(self)

    def filter_results(self, *_):
        return interpretation_results.filter_results(self, *_)

    def create_plot(self):
        return interpretation_chart.create_plot(self)

    def configure_plot_axes(self):
        return interpretation_chart.configure_plot_axes(self)

    def save_splitter(self, *_):
        return interpretation_chart.save_splitter(self, *_)

    def update_preview_context(self, axes):
        return interpretation_chart.update_preview_context(self, axes)

    def show_on_plot(self):
        """Compatibility entry point; visualization stays in this document."""
        return interpretation_chart.show_on_plot(self)

    def on_activated(self):
        return interpretation_chart.on_activated(self)

    def update_plot(self):
        return interpretation_chart.update_plot(self)

    def update_export_labels(self):
        return interpretation_reports.update_export_labels(self)

    def export_positions(self, report):
        return interpretation_reports.export_positions(self, report)

    def save_report(self):
        return interpretation_reports.save_report(self)

    def save_csv(self):
        return interpretation_reports.save_csv(self)

    def reject(self):
        self.closed = True
        self.timer.stop()
        self.stop_calculation()
        self.plot.release_plot()
        super().reject()

    def closeEvent(self, event):
        self.closed = True
        self.timer.stop()
        self.stop_calculation()
        self.plot.release_plot()
        super().closeEvent(event)

    def stop_calculation(self):
        return interpretation_calculation.stop_calculation(self)
