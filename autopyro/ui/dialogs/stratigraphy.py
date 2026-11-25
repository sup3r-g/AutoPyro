"""Stratigraphy worksheet and public dialog interface."""

from __future__ import annotations

from ..i18n import tr
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QDialog
from . import (
    stratigraphy_appearance,
    stratigraphy_binding,
    stratigraphy_column,
    stratigraphy_column_export,
    stratigraphy_draft,
    stratigraphy_files,
    stratigraphy_forms,
    stratigraphy_review,
)
from .stratigraphy_binding import BindingPreviewModel as BindingPreviewModel
from .stratigraphy_binding import SetStratigraphyCommand as SetStratigraphyCommand


class StratigraphyDialog(QDialog):
    """Compose the interval editor, sample binding and column preview."""

    BACKGROUND_BINDING_ROWS = 1000
    BACKGROUND_PREVIEW_INTERVALS = 250

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        stratigraphy_column.initialize(self)
        self.setWindowTitle(tr('Стратиграфия · колонки и привязка проб'))
        self.resize(1100, 820)
        stratigraphy_draft.initialize(self)
        stratigraphy_forms.build(self)

    @staticmethod
    def interval_menu(toolbar, title, image):
        return stratigraphy_forms.interval_menu(toolbar, title, image)

    def create_column_style(self):
        return stratigraphy_appearance.create_column_style(self)

    def load_style(self, options):
        return stratigraphy_appearance.load_style(self, options)

    def update_style_controls(self):
        return stratigraphy_appearance.update_style_controls(self)

    def style_values(self):
        return stratigraphy_appearance.style_values(self)

    def render_options(self, template=None):
        return stratigraphy_appearance.render_options(self, template)

    def configure_correlation(self):
        return stratigraphy_appearance.configure_correlation(self)

    def show_geological_columns(self, visible):
        return stratigraphy_draft.show_geological_columns(self, visible)

    def invalidate_preview(self, *_):
        return stratigraphy_binding.invalidate_preview(self, *_)

    def update_binding_revision(self, preview=None):
        return stratigraphy_binding.update_binding_revision(self, preview)

    def has_unsaved_draft(self):
        return stratigraphy_draft.has_unsaved_draft(self)

    def mark_draft_saved(self, style):
        return stratigraphy_draft.mark_draft_saved(self, style)

    def invalidate_column(self, *_):
        return stratigraphy_column.invalidate_column(self, *_)

    def schedule_column_preview(self):
        return stratigraphy_column.schedule_column_preview(self)

    def resume_column_preview(self, *_):
        return stratigraphy_column.resume_column_preview(self, *_)

    def refresh_column_preview(self):
        return stratigraphy_column.refresh_column_preview(self)

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self.resume_column_preview)

    def hideEvent(self, event):
        self.column_preview_timer.stop()
        super().hideEvent(event)

    def load_draft(self):
        return stratigraphy_draft.load_draft(self)

    def populate_draft(self):
        return stratigraphy_draft.populate_draft(self)

    def add_interval(self, checked=False, interval=None):
        return stratigraphy_draft.add_interval(self, checked, interval)

    def insert_interval(self, row, item):
        return stratigraphy_draft.insert_interval(self, row, item)

    def refresh_interval_color_preview(self, cell):
        return stratigraphy_draft.refresh_interval_color_preview(self, cell)

    def write_interval(self, row, item):
        return stratigraphy_draft.write_interval(self, row, item)

    def remove_intervals(self):
        return stratigraphy_draft.remove_intervals(self)

    def choose_geological_cell(self, row, column):
        return stratigraphy_draft.choose_geological_cell(self, row, column)

    def choose_lithology(self):
        return stratigraphy_draft.choose_lithology(self)

    def read_scheme(self):
        return stratigraphy_draft.read_scheme(self)

    def read_scheme_steps(self):
        return stratigraphy_draft.read_scheme_steps(self)

    def read_render_scheme(self):
        return stratigraphy_draft.read_render_scheme(self)

    def refresh_wellbores(self, *_):
        return stratigraphy_draft.refresh_wellbores(self, *_)

    def show_column(self):
        return stratigraphy_column.show_column(self)

    def render_column_preview(self, *, automatic):
        return stratigraphy_column.render_column_preview(self, automatic=automatic)

    def render_large_column(self, scheme, well, level, options, samples, limits, revision):
        return stratigraphy_column.render_large_column(self, scheme, well, level, options, samples, limits, revision)

    @staticmethod
    def label_column_axes(figure, unit, level):
        return stratigraphy_column.label_column_axes(figure, unit, level)

    def preview_samples(self, scheme, options):
        return stratigraphy_column.preview_samples(self, scheme, options)

    @staticmethod
    def sample_track_wells(scheme, options, level, selected):
        return stratigraphy_column.sample_track_wells(scheme, options, level, selected)

    def column_samples(self, scheme, options, level, selected):
        return stratigraphy_column.column_samples(self, scheme, options, level, selected)

    def calculate_sample_positions(self, scheme, binding, wells):
        return stratigraphy_column.calculate_sample_positions(self, scheme, binding, wells)

    def export_sample_positions(self, scheme, options, level, selected):
        return stratigraphy_column_export.export_sample_positions(self, scheme, options, level, selected)

    def open_sample_row(self, event):
        return stratigraphy_column.open_sample_row(self, event)

    def column_notice(self, notice):
        return stratigraphy_column.column_notice(self, notice)

    def export_column(self):
        return stratigraphy_column_export.export_column(self)

    def save_column_figure(self, path, scheme, well, level, style, *, samples=None, check_cancelled=None):
        return stratigraphy_column_export.save_column_figure(self, path, scheme, well, level, style, samples=samples, check_cancelled=check_cancelled)

    def export_column_batch(self):
        return stratigraphy_column_export.export_column_batch(self)

    def write_column_batch(self, directory, extension, scheme, level, style, selected, individual, comparison,
                           *, samples=None):
        return stratigraphy_column_export.write_column_batch(self, directory, extension, scheme, level, style, selected, individual, comparison, samples=samples)

    def save_scheme(self):
        return stratigraphy_draft.save_scheme(self)

    def settings(self):
        return stratigraphy_binding.settings(self)

    def current_binding(self):
        return stratigraphy_binding.current_binding(self)

    def compare_revisions(self):
        return stratigraphy_review.compare_revisions(self)

    def show_revision_column(self, scheme, target, parent):
        return stratigraphy_column.show_revision_column(self, scheme, target, parent)

    def calculate_binding_report(self, row_ids, bindings, title, calculate, *, parent=None):
        return stratigraphy_binding.calculate_binding_report(self, row_ids, bindings, title, calculate, parent=parent)

    def preview(self):
        return stratigraphy_binding.preview(self)

    def commit(self, add_columns=False):
        return stratigraphy_binding.commit(self, add_columns)

    def new_binding_columns(self, existing):
        return stratigraphy_binding.new_binding_columns(self, existing)

    def prepare_binding_save(self, settings, columns):
        return stratigraphy_binding.prepare_binding_save(self, settings, columns)

    def has_raw_binding_inputs(self, binding):
        return stratigraphy_binding.has_raw_binding_inputs(self, binding)

    def review_binding_update(self, saved, settings, *, report=None):
        return stratigraphy_binding.review_binding_update(self, saved, settings, report=report)

    def apply(self):
        return stratigraphy_binding.apply(self)

    def add_columns(self):
        return stratigraphy_binding.add_columns(self)

    def show_plot(self):
        return stratigraphy_binding.show_plot(self)

    def show_thickness_summary(self):
        return stratigraphy_review.show_thickness_summary(self)

    def review_intervals(self):
        return stratigraphy_review.review_intervals(self)

    def review_interval_errors(self):
        return stratigraphy_review.review_interval_errors(self)

    def locate_finding(self, finding, show_plot):
        return stratigraphy_review.locate_finding(self, finding, show_plot)

    def review_merge(self, current, incoming):
        return stratigraphy_review.review_merge(self, current, incoming)

    def import_scheme(self, *, merge=False):
        return stratigraphy_files.import_scheme(self, merge=merge)

    def import_database_scheme(self):
        return stratigraphy_files.import_database_scheme(self)

    def stage_import(self, scheme, current):
        return stratigraphy_files.stage_import(self, scheme, current)

    def lithology_matcher(self):
        return stratigraphy_files.lithology_matcher(self)

    def import_lithologies(self, scheme):
        return stratigraphy_files.import_lithologies(self, scheme)

    def save_lithology_rules(self, learned):
        return stratigraphy_files.save_lithology_rules(self, learned)

    def manage_lithology_rules(self):
        return stratigraphy_files.manage_lithology_rules(self)

    def map_lithologies(self):
        return stratigraphy_files.map_lithologies(self)

    def export_scheme(self):
        return stratigraphy_files.export_scheme(self)

    def refresh_columns(self):
        return stratigraphy_binding.refresh_columns(self)
