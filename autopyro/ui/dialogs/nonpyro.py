"""Non-pyrolysis geochemistry worksheet and its public dialog interface."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog

from autopyro.analysis.nonpyro import METHODS

from ..i18n import tr
from . import (
    nonpyro_calculation,
    nonpyro_charts,
    nonpyro_forms,
    nonpyro_inputs,
    nonpyro_reports,
    nonpyro_results,
)
from .nonpyro_results import ProfileResultsModel as ProfileResultsModel


def __getattr__(name):
    # Preserve the former public import without loading the database UI as part
    # of an analytical module. Database loading is owned by MainWindow.
    if name == 'NonPyroDatabaseDialog':
        from ...database.ui.nonpyro import NonPyroDatabaseDialog
        return NonPyroDatabaseDialog
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


class NonPyroDialog(QDialog):
    """Compose non-pyrolysis methods, bindings and result workflows."""

    def __init__(self, owner, module_id=None):
        super().__init__(owner)
        self.owner = owner
        self.module_id = module_id
        from ..geochemistry import modules
        title, self.method_ids = modules()[module_id] if module_id else (tr("Геохимия"), tuple(METHODS))
        self.report = None
        self.loading = True
        self.promoting = False
        self.roles = {}
        self._schema = None
        self._report_context = None
        self.alkane_canvas = None
        self.sara_canvas = None
        self.sara_profiles_canvas = None
        self.homohopane_canvas = None
        self.chung_canvas = None
        self.setWindowTitle(title)
        self.resize(1320, 850)
        nonpyro_forms.build(self)

    def restore_parameters(self):
        return nonpyro_inputs.restore_parameters(self)

    def connect_model(self):
        return nonpyro_inputs.connect_model(self)

    def input_changed(self):
        return nonpyro_inputs.input_changed(self)

    def source_schema(self):
        return nonpyro_inputs.source_schema(self)

    def refresh_columns(self):
        return nonpyro_inputs.refresh_columns(self)

    def report_context(self):
        return nonpyro_results.report_context(self)

    def current_report(self):
        return nonpyro_results.current_report(self)

    def invalidate(self, *_):
        return nonpyro_results.invalidate(self, *_)

    def update_result_actions(self):
        return nonpyro_results.update_result_actions(self)

    def show_component_profiles(self):
        return nonpyro_charts.show_component_profiles(self)

    def show_ternary(self):
        return nonpyro_charts.show_ternary(self)

    def select_sara_point(self, event):
        return nonpyro_charts.select_sara_point(self, event)

    def selected_alkane_distribution(self):
        return nonpyro_charts.selected_alkane_distribution(self)

    def show_alkane_distribution(self):
        return nonpyro_charts.show_alkane_distribution(self)

    def update_workflow(self):
        return nonpyro_inputs.update_workflow(self)

    def method_changed(self, *_, use_saved=True):
        return nonpyro_inputs.method_changed(self, *_, use_saved=use_saved)

    def update_applicability(self, method_id):
        return nonpyro_inputs.update_applicability(self, method_id)

    def hints_changed(self, *_):
        return nonpyro_inputs.hints_changed(self, *_)

    def rebuild_bindings(self, preferred=None):
        return nonpyro_inputs.rebuild_bindings(self, preferred)

    def update_unit(self, key):
        return nonpyro_inputs.update_unit(self, key)

    def role_changed(self, key):
        return nonpyro_inputs.role_changed(self, key)

    def row_ids(self):
        return nonpyro_inputs.row_ids(self)

    def config(self):
        return nonpyro_inputs.config(self)

    def show_profile_coverage(self):
        return nonpyro_calculation.show_profile_coverage(self)

    def calculate(self):
        return nonpyro_calculation.calculate(self)

    def history_reports(self):
        return nonpyro_reports.history_reports(self)

    def open_history(self):
        return nonpyro_reports.open_history(self)

    def promote(self):
        return nonpyro_results.promote(self)

    def plot(self):
        return nonpyro_charts.plot(self)

    def select_samples(self):
        return nonpyro_results.select_samples(self)

    def export(self):
        return nonpyro_reports.export(self)

    @staticmethod
    def write_profile_export(parent, path, report):
        return nonpyro_reports.write_profile_export(parent, path, report)
