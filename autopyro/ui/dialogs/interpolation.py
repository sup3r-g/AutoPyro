"""Depth interpolation worksheet and its public dialog interface."""

from __future__ import annotations

from PySide6.QtGui import QAccessible, QAccessibleAnnouncementEvent
from PySide6.QtWidgets import QDialog

from ..i18n import tr
from . import (
    interpolation_calculation,
    interpolation_charts,
    interpolation_diagnostics,
    interpolation_forms,
    interpolation_inputs,
    interpolation_navigation,
    interpolation_results,
    interpolation_runs,
)
from .interpolation_calculation import (
    describe_calculation_groups as describe_calculation_groups,
)
from .interpolation_charts import (
    CHART_SERIES_LIMIT as CHART_SERIES_LIMIT,
    sample_chart_points as sample_chart_points,
    set_chart_series_points as set_chart_series_points,
)
from .interpolation_diagnostics import (
    diagnostic_labels as diagnostic_labels,
    diagnostic_rows as diagnostic_rows,
    duplicate_y_role as duplicate_y_role,
)
from .interpolation_inputs import (
    WELL_NAMES as WELL_NAMES,
    column_label as column_label,
)
from .interpolation_navigation import (
    SampleNavigationTable as SampleNavigationTable,
)
from .interpolation_results import (
    BACKGROUND_ROWS as BACKGROUND_ROWS,
    InterpolationDiagnosticsModel as InterpolationDiagnosticsModel,
    InterpolationPreviewModel as InterpolationPreviewModel,
)


class InterpolationDialog(QDialog):
    """Compose depth interpolation, its preview and saved calculations."""

    def __init__(self, project, row_ids, parent=None):
        super().__init__(parent)
        self.project, self.row_ids = project, row_ids
        self.owner = parent
        self._preview_source = None
        self._pending_chart = None
        self.measurement_context = {}
        self.chart_scope_label = ""
        self.chart_axis_names = ("", "")
        self._chart_dirty = False
        self._chart_group_by_anchor = {}
        self._support_spans = {}
        self.protocol = None
        self.setWindowTitle(tr('Интерполяция'))
        self.resize(850, 620)
        interpolation_forms.build(self)

    def default_well(self):
        return interpolation_inputs.default_well(self)

    def bound_well(self):
        return interpolation_inputs.bound_well(self)

    def default_well_label(self):
        return interpolation_inputs.default_well_label(self)

    def selected_well(self):
        return interpolation_inputs.selected_well(self)

    def numeric_choices(self):
        return interpolation_inputs.numeric_choices(self)

    def selected_group_ids(self):
        return interpolation_inputs.selected_group_ids(self)

    def input_error(self):
        return interpolation_inputs.input_error(self)

    def restore_parameters(self):
        return interpolation_inputs.restore_parameters(self)

    def invalidate(self, *args, announce=True):
        return interpolation_results.invalidate(self, *args, announce=announce)

    def announce_summary(self, text=None):
        if QAccessible.isActive():
            QAccessible.updateAccessibility(QAccessibleAnnouncementEvent(
                self.summary, self.summary.text() if text is None else text,
            ))

    def set_announced_summary(self, text):
        self.summary.setText(text)
        self.announce_summary()

    def set_name_error(self, text=""):
        return interpolation_inputs.set_name_error(self, text)

    def update_result_accessibility(self):
        return interpolation_results.update_result_accessibility(self)

    def update_extrapolation_controls(self, *args):
        return interpolation_inputs.update_extrapolation_controls(self, *args)

    def update_method_controls(self):
        return interpolation_inputs.update_method_controls(self)

    def calculate(self):
        return interpolation_calculation.calculate(self)

    def set_diagnostics(self, diagnostics, column_id):
        return interpolation_diagnostics.set_diagnostics(self, diagnostics, column_id)

    def show_diagnostics(self):
        return interpolation_diagnostics.show_diagnostics(self)

    def validate_methods(self):
        return interpolation_diagnostics.validate_methods(self)

    def export_diagnostics(self):
        return interpolation_diagnostics.export_diagnostics(self)

    def set_export_status(self, text):
        return interpolation_diagnostics.set_export_status(self, text)

    def current_diagnostics(self):
        return interpolation_diagnostics.current_diagnostics(self)

    def can_open_sample(self):
        return interpolation_navigation.can_open_sample(self)

    def open_diagnostic_sample(self, index):
        return interpolation_navigation.open_diagnostic_sample(self, index)

    def open_proposal_sample(self, index):
        return interpolation_navigation.open_proposal_sample(self, index)

    def open_sample_choices(self, ids, title):
        return interpolation_navigation.open_sample_choices(self, ids, title)

    def open_chart_sample(self, chart, samples_at, point):
        return interpolation_charts.open_chart_sample(self, chart, samples_at, point)

    def update_chart_filter(self):
        return interpolation_charts.update_chart_filter(self)

    def chart_data(self):
        return interpolation_charts.chart_data(self)

    def show_chart_data(self):
        return interpolation_charts.show_chart_data(self)

    def chart_scope_text(self):
        return interpolation_charts.chart_scope_text(self)

    def render_pending_chart(self):
        return interpolation_charts.render_pending_chart(self)

    def replace_chart(self, chart):
        return interpolation_charts.replace_chart(self, chart)

    def output_contexts(self):
        return interpolation_runs.output_contexts(self)

    def apply(self):
        return interpolation_runs.apply(self)

    def refresh_columns(self):
        return interpolation_inputs.refresh_columns(self)

    def refresh_boundaries(self):
        return interpolation_inputs.refresh_boundaries(self)

    def refresh_saved_runs(self):
        return interpolation_runs.refresh_saved_runs(self)

    def load_saved_parameters(self):
        return interpolation_runs.load_saved_parameters(self)

    @staticmethod
    def run_snapshot_task(function, protocol, title, update):
        return interpolation_runs.run_snapshot_task(function, protocol, title, update)

    def replay_saved(self):
        return interpolation_runs.replay_saved(self)
