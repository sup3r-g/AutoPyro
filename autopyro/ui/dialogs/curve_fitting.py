"""Curve fitting worksheet and its public dialog interface."""

from __future__ import annotations

from PySide6.QtCore import Signal

from ..i18n import tr
from ..tasks import TaskDialog
from .curve_analysis import CurveAnalysisTools
from .curve_joint_analysis import JointCurveTools
from . import (
    curve_fitting_calculation,
    curve_fitting_comparison,
    curve_fitting_forms,
    curve_fitting_inputs,
    curve_fitting_results,
)


class CurveFittingDialog(CurveAnalysisTools, JointCurveTools, TaskDialog):
    """Compose curve fitting, diagnostics and joint analysis."""

    bootstrap_progress = Signal(int, int, int)
    joint_cv_progress = Signal(object, int, int)

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        curve_fitting_calculation.initialize(self)
        self.setWindowTitle(tr("Подгонка кривых"))
        curve_fitting_forms.build(self)

    def reset_parameter_inputs(self, *args):
        # Bounds and starts are positional: never reuse them for a different model.
        return curve_fitting_inputs.reset_parameter_inputs(self, *args)

    def sync_columns(self, *args):
        return curve_fitting_inputs.sync_columns(self, *args)

    def build_comparison_controls(self):
        return curve_fitting_comparison.build_comparison_controls(self)

    def refill_holdout_columns(self, selected):
        return curve_fitting_comparison.refill_holdout_columns(self, selected)

    def update_cv_controls(self, *args):
        return curve_fitting_comparison.update_cv_controls(self, *args)

    def comparison_settings(self):
        return curve_fitting_comparison.comparison_settings(self)

    def save_comparison_settings(self):
        return curve_fitting_comparison.save_comparison_settings(self)

    def refill_test_models(self):
        return curve_fitting_comparison.refill_test_models(self)

    def add_candidate(self):
        return curve_fitting_comparison.add_candidate(self)

    def remove_candidate(self):
        return curve_fitting_comparison.remove_candidate(self)

    def invalidate_comparison(self, *args):
        return curve_fitting_comparison.invalidate_comparison(self, *args)

    def show_comparison(self):
        return curve_fitting_comparison.show_comparison(self)

    def show_nested_comparison(self, nested):
        return curve_fitting_comparison.show_nested_comparison(self, nested)

    def set_comparison_table(self, headers, rows):
        return curve_fitting_comparison.set_comparison_table(self, headers, rows)

    def fit_selected_model(self):
        return curve_fitting_comparison.fit_selected_model(self)

    def export_comparison(self):
        return curve_fitting_comparison.export_comparison(self)

    def update_controls(self, *args):
        return curve_fitting_inputs.update_controls(self, *args)

    def update_bootstrap_controls(self, *args):
        return curve_fitting_inputs.update_bootstrap_controls(self, *args)

    def refresh_saved_models(self):
        return curve_fitting_inputs.refresh_saved_models(self)

    def load_custom_model(self, *args):
        return curve_fitting_inputs.load_custom_model(self, *args)

    def save_custom_model(self):
        return curve_fitting_inputs.save_custom_model(self)

    def invalidate(self, *args):
        return curve_fitting_calculation.invalidate(self, *args)

    def prepare_project_close(self):
        super().prepare_project_close()
        self._bootstrap_cancel.set()
        self._comparison_cancel.set()
        self._joint_cancel.set()
        self._joint_band_cache.clear()

    def settings(self):
        return curve_fitting_inputs.settings(self)

    def restore_settings(self, settings):
        return curve_fitting_inputs.restore_settings(self, settings)

    def calculate(self, *, comparison=False, selection=None, replay=None):
        return curve_fitting_calculation.calculate(self, comparison=comparison, selection=selection, replay=replay)

    def current_bands_available(self):
        return curve_fitting_results.current_bands_available(self)

    def show_bootstrap_progress(self, generation, done, total):
        return curve_fitting_calculation.show_bootstrap_progress(self, generation, done, total)

    def band_data(self):
        """Shared cached bands for plotting and export; never starts a refit."""
        return curve_fitting_results.band_data(self)

    def show_summary(self):
        return curve_fitting_results.show_summary(self)

    def redraw(self, *args):
        return curve_fitting_results.redraw(self, *args)

    def export_result(self):
        return curve_fitting_results.export_result(self)
