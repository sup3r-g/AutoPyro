"""Qt plot editor and its public dialog interface."""

from __future__ import annotations

from ..i18n import tr
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QDialog
from autopyro.visualization.plotting import PlotSpec
from ..widgets.forms import Form as Form
from .plot_editor import PlotConfigCommand as PlotConfigCommand
from . import plot_editor, plot_export_workflow, plot_forms, plot_interaction, plot_presets, plot_rendering


class PlotDialog(QDialog):
    """Compose the plot editor, its preview and the export workflow."""

    def __init__(self, owner, *, interpretation_owner=None):
        super().__init__(owner)
        self.owner = owner
        self.interpretation_owner = interpretation_owner
        self.settings_key = "interpretation_plot_current" if interpretation_owner else "plot_current"
        self.closed = False
        self.loading = True
        self.spec = PlotSpec()
        self.result = None
        plot_interaction.install(self)
        plot_editor.install(self)
        plot_rendering.install(self)
        plot_export_workflow.install(self)
        self.setWindowTitle(tr('Пиролитические графики · данные, оформление и экспорт'))
        self.resize(1380, 900)
        plot_forms.build(self)
        plot_forms.initialize_editor(self)

    def refresh_scopes(self):
        return plot_editor.refresh_scopes(self)

    def rename_scope(self, old, new):
        return plot_editor.rename_scope(self, old, new)

    def analysis_scope_changed(self):
        return plot_editor.analysis_scope_changed(self)

    def table_scope_changed(self):
        return plot_editor.table_scope_changed(self)

    def mark_frame(self, text=None):
        return plot_rendering.mark_frame(self, text)

    def restore_parameters(self):
        return plot_editor.restore_parameters(self)

    def make_toolbar(self, canvas):
        return plot_rendering.make_toolbar(self, canvas)

    def replace_figure(self, figure, *, remote=False, navigation=False):
        return plot_rendering.replace_figure(self, figure, remote=remote, navigation=navigation)

    def sync_columns(self, *args):
        return plot_editor.sync_columns(self, *args)

    def selected_tracks(self):
        return plot_editor.selected_tracks(self)

    def tracks_changed(self, *args):
        return plot_editor.tracks_changed(self, *args)

    @staticmethod
    def show_field(widget, visible):
        return plot_editor.show_field(widget, visible)

    def selected_kind(self):
        return plot_editor.selected_kind(self)

    def sync_variant(self, saved_kind=None):
        return plot_editor.sync_variant(self, saved_kind)

    def load_kind_options(self, *_):
        return plot_editor.load_kind_options(self, *_)

    def read_kind_options(self, kind):
        return plot_editor.read_kind_options(self, kind)

    def update_enabled(self):
        return plot_editor.update_enabled(self)

    def kind_changed(self, *args):
        return plot_editor.kind_changed(self, *args)

    def update_x_targets(self):
        return plot_editor.update_x_targets(self)

    def switch_x_target(self):
        return plot_editor.switch_x_target(self)

    def apply_interpretation_binding(self):
        return plot_editor.apply_interpretation_binding(self)

    def change_size_unit(self):
        return plot_editor.change_size_unit(self)

    def read_controls(self):
        return plot_editor.read_controls(self)

    def sync_shared_appearance(self, *, publish=False):
        return plot_editor.sync_shared_appearance(self, publish=publish)

    def sync_peer_appearance(self):
        return plot_editor.sync_peer_appearance(self)

    def on_activated(self):
        return plot_editor.on_activated(self)

    def remove_stratigraphy(self):
        return plot_editor.remove_stratigraphy(self)

    def load_controls(self):
        return plot_editor.load_controls(self)

    def load_track_order(self):
        # Restore both check state and ordering.
        return plot_editor.load_track_order(self)

    def schedule(self, *args):
        return plot_rendering.schedule(self, *args)

    def edit_local_filter(self):
        return plot_editor.edit_local_filter(self)

    def scope_ids(self, spec=None, *, selected_ids=None):
        return plot_editor.scope_ids(self, spec, selected_ids=selected_ids)

    def selection_ids(self):
        return plot_interaction.selection_ids(self)

    def selection_changed(self, *args):
        return plot_interaction.selection_changed(self, *args)

    def capture_plot_input(self, ids, specs):
        return plot_rendering.capture_plot_input(self, ids, specs)

    def prepare_async(self):
        return plot_rendering.prepare_async(self)

    def input_captured(self, generation, source):
        return plot_rendering.input_captured(self, generation, source)

    def preparation_ready(self, generation, prepared):
        return plot_rendering.preparation_ready(self, generation, prepared)

    def eventFilter(self, watched, event):
        if (hasattr(self, 'canvas_scroll') and watched is self.canvas_scroll.viewport()
                and event.type() in (QEvent.Type.Resize, QEvent.Type.Show)):
            self.viewport_timer.start()
        return super().eventFilter(watched, event)

    def preview_spec(self):
        return plot_rendering.preview_spec(self)

    def request_preview(self):
        return plot_rendering.request_preview(self)

    def preview_ready(self, generation, frame):
        return plot_rendering.preview_ready(self, generation, frame)

    def present_preview(self, generation, serial, frame):
        return plot_rendering.present_preview(self, generation, serial, frame)

    def preparation_failed(self, generation, error):
        return plot_rendering.preparation_failed(self, generation, error)

    def preparation_idle(self):
        return plot_rendering.preparation_idle(self)

    def cancel_preparation(self):
        return plot_rendering.cancel_preparation(self)

    def redraw(self, prepared=None, *, preview=None, navigation=False):
        return plot_rendering.redraw(self, prepared, preview=preview, navigation=navigation)

    def request_navigation(self):
        return plot_rendering.request_navigation(self)

    def enable_brush(self, enabled):
        return plot_interaction.enable_brush(self, enabled)

    def select_rows(self, row_ids):
        return plot_interaction.select_rows(self, row_ids)

    def connect_canvas_events(self):
        return plot_interaction.connect_canvas_events(self)

    def annotation_context_menu(self, event):
        return plot_interaction.annotation_context_menu(self, event)

    def edit_annotations(self, position=None):
        return plot_interaction.edit_annotations(self, position)

    def hide_hover(self, *args):
        return plot_interaction.hide_hover(self, *args)

    def clear_hover(self, *args):
        return plot_interaction.clear_hover(self, *args)

    def on_hover(self, event):
        return plot_interaction.on_hover(self, event)

    def point_tooltip(self, row_id, coordinates, binding=None, *, track_id=None):
        return plot_interaction.point_tooltip(self, row_id, coordinates, binding, track_id=track_id)

    def show_hover(self):
        return plot_interaction.show_hover(self)

    def on_pick(self, event):
        return plot_interaction.on_pick(self, event)

    def on_rectangle(self, start, end):
        return plot_interaction.on_rectangle(self, start, end)

    def refresh_categories(self):
        return plot_editor.refresh_categories(self)

    def category_changed(self, item):
        return plot_editor.category_changed(self, item)

    def reset_category_parameter(self, row, column):
        return plot_editor.reset_category_parameter(self, row, column)

    def choose_category_color(self, row, column):
        return plot_editor.choose_category_color(self, row, column)

    def use_template(self, title):
        return plot_presets.use_template(self, title)

    def apply_template(self):
        return plot_presets.apply_template(self)

    def remove_passport(self):
        return plot_presets.remove_passport(self)

    def capture_limits(self):
        return plot_presets.capture_limits(self)

    def copy_visible_limits(self, spec):
        return plot_presets.copy_visible_limits(self, spec)

    def reset_limits(self):
        return plot_presets.reset_limits(self)

    def freeze_scope(self):
        return plot_presets.freeze_scope(self)

    def save_named(self):
        return plot_presets.save_named(self)

    def load_named(self):
        return plot_presets.load_named(self)

    def make_export_jobs(self, descriptions, *, scopes=None, source=None):
        return plot_export_workflow.make_export_jobs(self, descriptions, scopes=scopes, source=source)

    def export_busy(self):
        return plot_export_workflow.export_busy(self)

    def start_export(self, descriptions, manifest=None, *, report_rows=False, snapshot=False, panel_columns=None):
        return plot_export_workflow.start_export(self, descriptions, manifest, report_rows=report_rows, snapshot=snapshot, panel_columns=panel_columns)

    def export_input_captured(self, generation, source):
        return plot_export_workflow.export_input_captured(self, generation, source)

    def submit_export_jobs(self, jobs, manifest, snapshot, panel_columns):
        return plot_export_workflow.submit_export_jobs(self, jobs, manifest, snapshot, panel_columns)

    def export_capture_failed(self, generation, error):
        return plot_export_workflow.export_capture_failed(self, generation, error)

    def cancel_export(self):
        return plot_export_workflow.cancel_export(self)

    def export_busy_changed(self, busy):
        return plot_export_workflow.export_busy_changed(self, busy)

    def export_progress(self, done, total):
        return plot_export_workflow.export_progress(self, done, total)

    def export_completed(self, outcome):
        return plot_export_workflow.export_completed(self, outcome)

    def export_failed(self, error):
        return plot_export_workflow.export_failed(self, error)

    def export_plot(self):
        return plot_export_workflow.export_plot(self)

    def export_track(self):
        return plot_export_workflow.export_track(self)

    def export_visible_limits(self):
        return plot_export_workflow.export_visible_limits(self)

    def can_export_visible_limits(self):
        return plot_export_workflow.can_export_visible_limits(self)

    def save_current_plot_image(self, spec, default_path):
        return plot_export_workflow.save_current_plot_image(self, spec, default_path)

    def export_snapshot(self):
        return plot_export_workflow.export_snapshot(self)

    def replay_snapshot(self):
        return plot_export_workflow.replay_snapshot(self)

    def add_archived_curve(self):
        return plot_presets.add_archived_curve(self)

    def remove_archived_curve(self):
        return plot_presets.remove_archived_curve(self)

    def remove_interpretation(self):
        return plot_presets.remove_interpretation(self)

    def export_selected_plots(self, *, panels=False):
        return plot_export_workflow.export_selected_plots(self, panels=panels)

    def export_panels(self, names, *, columns, style=None):
        return plot_export_workflow.export_panels(self, names, columns=columns, style=style)

    def batch_export(self, *, names=None, style=None):
        return plot_export_workflow.batch_export(self, names=names, style=style)

    def export_spec(self):
        return plot_presets.export_spec(self)

    def import_spec(self):
        return plot_presets.import_spec(self)

    def release_plot(self):
        return plot_rendering.release_plot(self)

    def reject(self):
        self.release_plot()
        super().reject()

    def closeEvent(self, event):
        self.release_plot()
        super().closeEvent(event)
