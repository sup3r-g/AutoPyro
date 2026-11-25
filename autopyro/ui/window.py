from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication, QMainWindow

from autopyro.model.core import Project

from . import window_actions, window_preferences, window_status
from .analysis_documents import AnalysisDocuments
from .analysis_scope import AnalysisScope
from .data_io import database_actions, table_export, table_import
from .data_io.database_actions import DatabaseDialog as DatabaseDialog
from .project_session import ProjectSession
from .settings import load_preferences
from .table import cell_actions, context_menus, project_table, source_navigation
from .table.filter_sort import TableFilterSort
from .table.sample_sets import SampleSetActions
from .table.view import DataTable
from .table.view_state import TableViewState
from .theme import apply_theme
from .widgets.parameters import annotate_parameters
from .workspace import DesktopWorkspace


class MainWindow(QMainWindow):
    """Assemble the desktop and coordinate its project and table components."""

    def __init__(self):
        super().__init__()
        apply_theme(QApplication.instance(), load_preferences().theme)
        self.resize(1480, 900)
        self.session = ProjectSession(self)
        self.analysis_documents = AnalysisDocuments(self)
        self.analysis_revision = 0
        self.sample_sets = SampleSetActions(self)
        self.analysis_scope = AnalysisScope(self)
        self.table = DataTable()
        self.table_filter_sort = TableFilterSort(self)
        self.table_view = TableViewState(self)
        project_table.install(self)
        window_actions.install(self)
        window_status.install(self)
        self.desktop = DesktopWorkspace(self)
        self.set_project(Project())
        self.apply_preferences()
        annotate_parameters(self)

    # Keep existing window callers bound to the single session state.
    @property
    def _autosave_deferred(self):
        return self.session._autosave_deferred

    @_autosave_deferred.setter
    def _autosave_deferred(self, value):
        self.session._autosave_deferred = value

    @property
    def _autosave_wait(self):
        return self.session._autosave_wait

    @_autosave_wait.setter
    def _autosave_wait(self, value):
        self.session._autosave_wait = value

    @property
    def _autosaved_revision(self):
        return self.session._autosaved_revision

    @_autosaved_revision.setter
    def _autosaved_revision(self, value):
        self.session._autosaved_revision = value

    @property
    def _background_autosave(self):
        return self.session._background_autosave

    @_background_autosave.setter
    def _background_autosave(self, value):
        self.session._background_autosave = value

    @property
    def _save_cancelled(self):
        return self.session._save_cancelled

    @_save_cancelled.setter
    def _save_cancelled(self, value):
        self.session._save_cancelled = value

    @property
    def autosave_timer(self):
        return self.session.autosave_timer

    @autosave_timer.setter
    def autosave_timer(self, value):
        self.session.autosave_timer = value

    @property
    def dirty(self):
        return self.session.dirty

    @dirty.setter
    def dirty(self, value):
        self.session.dirty = value

    @property
    def path(self):
        return self.session.path

    @path.setter
    def path(self, value):
        self.session.path = value

    @property
    def recovered_from(self):
        return self.session.recovered_from

    @recovered_from.setter
    def recovered_from(self, value):
        self.session.recovered_from = value

    @property
    def session_id(self):
        return self.session.session_id

    @session_id.setter
    def session_id(self, value):
        self.session.session_id = value

    # Analysis windows remain available to the workspace and existing callers.
    @property
    def statistics_window(self):
        return self.analysis_documents.statistics_window

    @property
    def curve_fitting_window(self):
        return self.analysis_documents.curve_fitting_window

    @property
    def plot_window(self):
        return self.analysis_documents.plot_window

    @property
    def interpretation_window(self):
        return self.analysis_documents.interpretation_window

    @property
    def map_window(self):
        return self.analysis_documents.map_window

    @property
    def qc_window(self):
        return self.analysis_documents.qc_window

    @property
    def nonpyro_window(self):
        return self.analysis_documents.nonpyro_window

    @property
    def geochemistry_windows(self):
        return self.analysis_documents.geochemistry_windows

    @property
    def geochemistry_workspace_window(self):
        return self.analysis_documents.geochemistry_workspace_window

    @property
    def interpolation_window(self):
        return self.analysis_documents.interpolation_window

    @property
    def calculation_window(self):
        return self.analysis_documents.calculation_window

    @property
    def stratigraphy_window(self):
        return self.analysis_documents.stratigraphy_window

    @stratigraphy_window.setter
    def stratigraphy_window(self, document):
        self.analysis_documents.stratigraphy_window = document

    def restore_view_defaults(self):
        self.table_view.restore_defaults()

    def apply_preferences(self):
        window_preferences.apply_preferences(self)

    def show_settings(self):
        window_preferences.show_settings(self)

    def set_project(self, project, path=None, *, prepared=False, recovery_path=None):
        prepared = project_table.prepare(self, project, prepared=prepared)
        # Untitled projects derive the recovery filename from the old project ID.
        previous_recovery = self.session.recovery_files() if self.dirty else ()
        self.session.prepare_replacement()
        self.analysis_documents.close_for_project_replacement()
        self.sample_sets.reset()
        self.table_filter_sort.missing_filter_column = False
        self.table_view.column_precision = {}
        old_model, old_proxy = project_table.bind(self, project, prepared=prepared)
        self.path = Path(path) if path else None
        project_table.fit_columns(self)
        self.analysis_scope.reset()
        self.restore_view(project.view.get("current", {}))
        self.analysis_scope.bind(self.analysis_scope_combo)
        self.dirty = False
        self.update_status()
        self.desktop.bind_project()
        self.recovered_from = recovery_path
        if previous_recovery:
            self.session.remove_recovery_files(previous_recovery, preserve=(path, recovery_path))
        if old_proxy:
            old_proxy.deleteLater()
            old_model.deleteLater()

    def model_changed(self):
        self.analysis_revision += 1
        self.session.mark_changed()
        self.analysis_documents.invalidate_after_edit()
        self.analysis_scope.refresh()
        self.update_status()

    def view_changed(self):
        """Persist display changes without invalidating analysis results."""
        self.session.mark_changed()
        self.analysis_scope.refresh()
        self.update_status()

    def pyrograms_changed(self):
        self.view_changed()
        if self.interpretation_window is not None:
            self.interpretation_window.update_explanation()

    def update_status(self):
        window_status.update_status(self)

    def update_selection_status(self):
        window_status.update_selection_status(self)

    def toggle_all(self, visible):
        return self.table_view.toggle_all(visible)

    @property
    def column_precision(self):
        return self.table_view.column_precision

    @column_precision.setter
    def column_precision(self, value):
        self.table_view.column_precision = value

    def visible_ids(self):
        return self.table_view.visible_ids()

    def row_ids(self):
        return list(self.proxy.ordered_row_ids())

    def capture_view(self, *, include_sample_set=False):
        return self.table_view.capture_view(include_sample_set=include_sample_set)

    def restore_view(self, view):
        return self.table_view.restore_view(view)

    def save_view(self):
        self.table_view.show_saved_views(focus_name=True)

    def load_view(self):
        self.table_view.show_saved_views()

    @property
    def active_sample_set(self):
        return self.sample_sets.active_name

    @active_sample_set.setter
    def active_sample_set(self, name):
        self.sample_sets.active_name = name

    @property
    def _sample_set_return_filter(self):
        return self.sample_sets.return_filter

    def save_sample_set(self):
        return self.sample_sets.save_sample_set()

    def choose_sample_set(self, title):
        return self.sample_sets.choose_sample_set(title)

    def open_sample_set(self):
        return self.sample_sets.open_sample_set()

    def close_sample_set(self):
        return self.sample_sets.close_sample_set()

    def compare_sample_sets(self):
        return self.sample_sets.compare_sample_sets()

    def rename_sample_set(self):
        return self.sample_sets.rename_sample_set()

    def delete_sample_set(self):
        return self.sample_sets.delete_sample_set()

    def may_replace(self):
        return self.session.may_replace()

    def new_project(self):
        return self.session.new_project()

    def open_project(self):
        return self.session.open_project()

    def save(self, checked=False, as_new=False):
        return self.session.save(checked, as_new)

    def autosave_path(self):
        return self.session.autosave_path()

    def _stop_autosave_deferral(self):
        return self.session._stop_autosave_deferral()

    def autosave(self):
        return self.session.autosave()

    def autosave_failed(self, error):
        return self.session.autosave_failed(error)

    def autosave_saved(self, revision):
        return self.session.autosave_saved(revision)

    def clear_autosave(self, *, preserve=None):
        return self.session.clear_autosave(preserve=preserve)

    def load_autosave(self, path):
        return self.session.load_autosave(path)

    def restore_autosave(self):
        return self.session.restore_autosave()

    def cell_menu(self, position, table=None):
        context_menus.cell_menu(self, position, table)

    def show_measurements(self):
        source_navigation.show_measurements(self)

    def open_source_cell(self, row_id, column_id, parent):
        return source_navigation.open_source_cell(self, row_id, column_id, parent)

    def import_file(self):
        table_import.import_table(self)

    def append_file(self):
        table_import.import_table(self, append=True)

    def import_log(self):
        table_import.import_log(self)

    def import_template_library(self):
        table_import.import_template_library(self)

    def show_qc(self):
        self.analysis_documents.open('qc')

    def refresh_database(self):
        database_actions.refresh_database(self)

    def database_refresh_log(self):
        database_actions.database_refresh_log(self)

    def import_geochemistry(self, *, append=False):
        database_actions.import_geochemistry(self, append=append)

    def import_database(self):
        database_actions.import_database(self)

    def demo(self):
        self.session.load_demo()

    def apply_filter(self, rules, combine):
        self.table_filter_sort.apply_filter(rules, combine)

    def filter_dialog(self):
        self.table_filter_sort.show_filter()

    def multi_sort(self):
        self.table_filter_sort.show_sort()

    def single_sort(self, section):
        self.table_filter_sort.single_sort(section)

    def header_menu(self, position, header=None):
        context_menus.header_menu(self, position, header)

    def pin_column(self, index, pinned):
        return self.table_view.pin_column(index, pinned)

    def choose_column_precision(self, index):
        return self.table_view.choose_column_precision(index)

    def copy_cells(self):
        cell_actions.copy_cells(self)

    def paste_cells(self):
        cell_actions.paste_cells(self)

    def parsed_change(self, index, value):
        return cell_actions.parsed_change(self, index, value)

    def fill_cells(self):
        cell_actions.fill_cells(self)

    def preview_batch_edit(self, title, targets, *, cell_count=0):
        cell_actions.preview_batch_edit(self, title, targets, cell_count=cell_count)

    def add_column(self):
        project_table.add_column(self)

    def interpolate(self):
        self.analysis_documents.open('interpolation')

    def show_curve_fitting(self):
        self.analysis_documents.open('curve_fitting')

    def show_statistics(self):
        self.analysis_documents.open('statistics')

    def show_report(self):
        self.analysis_documents.show_report()

    def show_stratigraphy(self):
        self.analysis_documents.open('stratigraphy')

    def show_maps(self):
        self.analysis_documents.open('map')

    def show_geochemistry_workspace(self):
        self.analysis_documents.open('geochemistry_workspace')

    def show_nonpyro(self, module_id=None):
        if module_id is None:
            self.analysis_documents.open('nonpyro')
        else:
            self.analysis_documents.open('geochemistry', module_id)

    def show_plots(self):
        self.analysis_documents.open('plot')

    def show_calculations(self):
        self.analysis_documents.open('calculation')

    def show_interpretation(self):
        self.analysis_documents.open('interpretation')

    def interpretation_scheme_saved(self):
        self.analysis_documents.interpretation_scheme_saved()

    def export(self):
        return table_export.export_visible_rows(self)

    def export_selected_samples(self):
        return table_export.export_selected_samples(self)

    def history(self):
        self.analysis_documents.show_history()

    def closeEvent(self, event):
        if self.session.confirm_documents_idle(self.analysis_documents.shutdown_blocker()) and self.may_replace():
            self.session.stop()
            if self.isVisible():
                self.desktop.save_layout()
            if self.dirty:
                self.clear_autosave()
            event.accept()
        else:
            event.ignore()
