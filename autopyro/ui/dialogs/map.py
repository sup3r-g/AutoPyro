"""Offline map worksheet and its public dialog interface."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDialog

from ..i18n import tr
from ..widgets.map_layers import MapLayerTools
from . import (
    map_files,
    map_forms,
    map_layers,
    map_parameters,
    map_regional,
    map_rendering,
    map_selection,
)
from .map_layers import LAYER_DEFAULTS as LAYER_DEFAULTS
from .spatial_interpolation import (
    FORECAST_REASONS as FORECAST_REASONS,
    IDW_REGION_MODES as IDW_REGION_MODES,
    INTERPOLATION_METHODS as INTERPOLATION_METHODS,
    PartitionResultsModel as PartitionResultsModel,
    PartitionSortProxy as PartitionSortProxy,
    SpatialInterpolationDialog as SpatialInterpolationDialog,
)


class MapDialog(MapLayerTools, QDialog):
    """Compose the offline map, layer controls and sample navigation."""

    project_tasks_finished = Signal()

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.closed = False
        self.loading = True
        map_rendering.initialize(self)
        self.regional_window = None
        self.spatial_window = None
        self.regional_highlights = []
        self.selector = None
        map_parameters.load_project_state(self)
        self.setWindowTitle(tr('Локальные карты · пробы, скважины и площадки'))
        self.resize(1380, 900)
        map_forms.build(self)

    def forms(self):
        return map_forms.forms(self)

    def connect_canvas_events(self):
        return map_selection.connect_canvas_events(self)

    def focus_map_coordinates(self):
        return map_parameters.focus_map_coordinates(self)

    def schedule_map_layout(self, *_):
        return map_rendering.schedule_map_layout(self, *_)

    def update_map_layout(self):
        return map_rendering.update_map_layout(self)

    def visualization_crs_changed(self, *_):
        return map_parameters.visualization_crs_changed(self, *_)

    def restore_parameters(self):
        return map_parameters.restore_parameters(self)

    def add_form(self, form, title, icon_name):
        return map_forms.add_form(self, form, title, icon_name)

    def sync_columns(self, *_):
        return map_parameters.sync_columns(self, *_)

    def sync_layers(self):
        return map_layers.sync_layers(self)

    def layer_color_attribute_changed(self, layer_id, field):
        return map_layers.layer_color_attribute_changed(self, layer_id, field)

    def sync_layer_categories(self, *_):
        return map_layers.sync_layer_categories(self, *_)

    def layer_category_edited(self, item):
        return map_layers.layer_category_edited(self, item)

    def reset_layer_categories(self):
        return map_layers.reset_layer_categories(self)

    def select_attribute_layer(self):
        return map_layers.select_attribute_layer(self)

    def sync_attributes(self, *_):
        return map_layers.sync_attributes(self, *_)

    def sync_coordinate_controls(self, *_):
        return map_parameters.sync_coordinate_controls(self, *_)

    def sync_layer_fields(self, *_):
        return map_layers.sync_layer_fields(self, *_)

    def highlight_attributes(self, *_):
        return map_selection.highlight_attributes(self, *_)

    def zoom_to_attribute(self, index):
        return map_selection.zoom_to_attribute(self, index)

    def zoom_to_layer(self, index):
        return map_selection.zoom_to_layer(self, index)

    def zoom_to_bounds(self, bounds, span=None):
        return map_selection.zoom_to_bounds(self, bounds, span)

    def feature_press(self, event):
        return map_selection.feature_press(self, event)

    def feature_release(self, event):
        return map_selection.feature_release(self, event)

    def schedule(self, *_):
        return map_rendering.schedule(self, *_)

    def selection_changed(self, *_):
        return map_selection.selection_changed(self, *_)

    def selected_ids(self):
        return map_selection.selected_ids(self)

    def scope_ids(self, selected_ids=None):
        return map_selection.scope_ids(self, selected_ids)

    def redraw(self):
        return map_rendering.redraw(self)

    def render_large_map(self, row_ids, previous_viewport):
        return map_rendering.render_large_map(self, row_ids, previous_viewport)

    def run_map_task(self, render):
        return map_rendering.run_map_task(self, render)

    def render_map_view(self):
        return map_rendering.render_map_view(self)

    def clear_map_display(self, status):
        return map_rendering.clear_map_display(self, status)

    def remember(self, stored_layers=None):
        return map_parameters.remember(self, stored_layers)

    def refresh_categories(self):
        return map_layers.refresh_categories(self)

    def category_changed(self, item):
        return map_layers.category_changed(self, item)

    def reset_category_parameter(self, row, column):
        return map_layers.reset_category_parameter(self, row, column)

    def reset_layer_parameter(self, row, column):
        return map_layers.reset_layer_parameter(self, row, column)

    def layer_changed(self, item):
        return map_layers.layer_changed(self, item)

    def import_layer(self):
        return map_files.import_layer(self)

    def remove_layer(self):
        return map_layers.remove_layer(self)

    def select_rows(self, ids):
        return map_selection.select_rows(self, ids)

    def on_pick(self, event):
        return map_selection.on_pick(self, event)

    def on_rectangle(self, start, end):
        return map_selection.on_rectangle(self, start, end)

    def capture_limits(self):
        return map_selection.capture_limits(self)

    def export(self):
        return map_files.export(self)

    def open_interpolation(self):
        if self.report:
            dialog = self.spatial_window = SpatialInterpolationDialog(self.report, self)
            dialog.project_tasks_finished.connect(self.project_tasks_finished)

            def dispose():
                self.spatial_window = None
                dialog.deleteLater()

            try:
                result = dialog.exec()
            finally:
                if dialog.task_running:
                    # Hiding a modal dialog can unwind exec() before its worker finishes.
                    dialog.prepare_project_close()
                    dialog.project_tasks_finished.connect(dispose)
                else:
                    dispose()
            if not self.closed and result == SpatialInterpolationDialog.CONFIGURE_MAP_CRS:
                self.visualization_crs.setFocus()
                self.visualization_crs.showPopup()

    def add_interpolation_layers(self, layers):
        return map_layers.add_interpolation_layers(self, layers)

    def open_regional_geochemistry(self):
        return map_regional.open_regional_geochemistry(self)

    def open_regional_sample(self, row_id):
        return map_regional.open_regional_sample(self, row_id)

    def clear_regional_highlights(self):
        return map_regional.clear_regional_highlights(self)

    def highlight_regional_selection(self, selection):
        return map_regional.highlight_regional_selection(self, selection)

    def draw_paired_map(self, ax, comparison):
        return map_regional.draw_paired_map(self, ax, comparison)

    def _analysis_children(self):
        return tuple(dialog for dialog in (self.regional_window, self.spatial_window) if dialog is not None)

    @property
    def task_running(self):
        return (self.map_render_running or self.map_export_running
                or any(dialog.task_running for dialog in self._analysis_children()))

    def prepare_project_close(self):
        self.closed = True
        self.map_generation += 1
        self.map_redraw_pending = False
        self.timer.stop()
        self.viewport_timer.stop()
        for dialog in self._analysis_children():
            dialog.prepare_project_close()
            dialog.hide()

    def _release_map(self):
        for dialog in self._analysis_children():
            dialog.close()
        self.clear_regional_highlights()
        self.viewport_timer.stop()
        if self.selection_overlay is not None:
            self.selection_overlay.dispose()
            self.selection_overlay = None
        self.closed = True
        self.timer.stop()

    def closeEvent(self, event):
        if self.task_running:
            event.ignore()
            return
        self._release_map()
        super().closeEvent(event)

    def reject(self):
        if self.task_running:
            return
        self._release_map()
        super().reject()
