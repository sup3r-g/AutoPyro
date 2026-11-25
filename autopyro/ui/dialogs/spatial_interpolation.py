"""Spatial interpolation worksheet and its public dialog interface."""

from __future__ import annotations

from copy import deepcopy

from PySide6.QtCore import QSize
from PySide6.QtGui import QAccessible, QAccessibleAnnouncementEvent

from ..i18n import tr, message
from ..tasks import TaskDialog
from . import (
    spatial_blocks,
    spatial_export,
    spatial_forms,
    spatial_grid,
    spatial_idw,
    spatial_input,
    spatial_kriging,
    spatial_spread,
)
from .spatial_labels import (
    FORECAST_REASONS as FORECAST_REASONS,
    IDW_REGION_MODES as IDW_REGION_MODES,
    INTERPOLATION_METHODS as INTERPOLATION_METHODS,
)
from .spatial_partition_comparison import (
    PartitionResultsModel as PartitionResultsModel,
    PartitionSortProxy as PartitionSortProxy,
    SpatialPartitionComparison,
)
from .spatial_settings_comparison import SpatialSettingsComparison


class SpatialInterpolationDialog(TaskDialog):
    """Compose interpolation, validation and map surface tools."""

    CONFIGURE_MAP_CRS = 2

    def __init__(self, report, parent=None):
        super().__init__(parent)
        self.snapshot = deepcopy(report)
        self.result = None
        self.block_result = None
        self.generation = 0
        self.partition_comparison = None
        self.owner = getattr(parent, 'owner', None)
        self.map_dialog = parent if hasattr(parent, 'add_interpolation_layers') else None
        self.setWindowTitle(tr('Пространственная интерполяция'))
        self.resize(1250, 850)
        spatial_forms.build(self, report, parent)

    def configure_map_crs(self):
        if not self.task_running:
            self.done(self.CONFIGURE_MAP_CRS)

    def add_interpolation_tab(self, content, label, icon_name):
        return spatial_forms.add_interpolation_tab(self, content, label, icon_name)

    def showEvent(self, event):
        super().showEvent(event)
        available = self.screen().availableGeometry().size() - QSize(24, 48)
        self.resize(self.size().boundedTo(available))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        spatial_forms.resize_layout(self, event)

    def layout_interpolation_actions(self):
        return spatial_forms.layout_interpolation_actions(self)

    def set_interpolation_status(self, text, *, announce=True):
        if self.status.text() == text:
            return
        self.status.setText(text)
        if announce and QAccessible.isActive():
            QAccessible.updateAccessibility(QAccessibleAnnouncementEvent(self.status, text))

    def run_interpolation_task(self, function, callback):
        generation = self.generation

        def failed(error):
            if generation == self.generation:
                self.set_interpolation_status(tr('Расчёт не выполнен: {0}', message(error)))

        self.run_task(function, callback, on_error=failed)

    def set_idw_preview_visible(self, visible):
        return spatial_idw.set_idw_preview_visible(self, visible)

    def build_sample_preview(self):
        return spatial_input.build_sample_preview(self)

    def clear_sample_preview(self):
        return spatial_input.clear_sample_preview(self)

    def check_sample(self):
        return spatial_input.check_sample(self)

    def show_sample_preview(self, result, options):
        return spatial_input.show_sample_preview(self, result, options)

    def build_settings_comparison(self):
        self.settings_comparison = SpatialSettingsComparison(self)
        self.settings_comparison.invalidated.connect(self.settings_comparison_invalidated)
        self.add_interpolation_tab(self.settings_comparison, tr('Сравнение настроек IDW'), 'files')

    def settings_comparison_invalidated(self):
        if self.partition_comparison is not None and self.partition_comparison.partition_nested.isChecked():
            self.partition_comparison.clear_partitions()

    def clear_settings_comparison(self):
        return self.settings_comparison.clear_settings_comparison()

    def set_settings_rows(self, rows):
        return self.settings_comparison.set_settings_rows(rows)

    def variant_settings(self, options):
        return self.settings_comparison.variant_settings(options)

    def calculate_settings_comparison(self, *, nested=False):
        return self.settings_comparison.calculate_settings_comparison(nested=nested)

    def run_settings_comparison(self, calculate):
        return self.settings_comparison.run_settings_comparison(calculate)

    def completed_nested(self, result):
        return self.settings_comparison.completed_nested(result)

    @staticmethod
    def nested_scores(score, *, coverage=True):
        return SpatialSettingsComparison.nested_scores(score, coverage=coverage)

    def show_nested_tables(self, result, summary_label, summary, fold_headers, folds):
        return self.settings_comparison.show_nested_tables(result, summary_label, summary, fold_headers, folds)

    def calculate_nested_methods(self):
        return self.settings_comparison.calculate_nested_methods()

    def kriging_ranges(self):
        return self.settings_comparison.kriging_ranges()

    @staticmethod
    def nested_method_labels():
        return SpatialSettingsComparison.nested_method_labels()

    def completed_nested_methods(self, result):
        return self.settings_comparison.completed_nested_methods(result)

    def completed_settings(self, result):
        return self.settings_comparison.completed_settings(result)

    def apply_idw_variant(self):
        return self.settings_comparison.apply_idw_variant()

    def export_settings_comparison(self):
        return self.settings_comparison.export_settings_comparison()

    def build_kriging(self):
        return spatial_forms.build_kriging(self)

    def focus_comparison_setup(self):
        return spatial_forms.focus_comparison_setup(self)

    def clear_kriging(self):
        return spatial_kriging.clear_kriging(self)

    def clear_nested_methods(self):
        if self.nested_result and self.nested_result['format'] == 'autopyro.nested-spatial-methods':
            self.clear_settings_comparison()

    def build_partition_comparison(self):
        self.partition_comparison = SpatialPartitionComparison(self)
        self.add_interpolation_tab(self.partition_comparison, tr('Устойчивость к разбиению'), 'shield-check')

    def reset_partition_notes(self):
        return self.partition_comparison.reset_partition_notes()

    def clear_partitions(self, *_):
        return self.partition_comparison.clear_partitions(*_)

    def describe_partition_setup(self, options, model, smoothing, variants, ranges):
        return self.partition_comparison.describe_partition_setup(options, model, smoothing, variants, ranges)

    def preview_partition_setup(self):
        return self.partition_comparison.preview_partition_setup()

    def calculate_partitions(self):
        return self.partition_comparison.calculate_partitions()

    def export_partitions(self):
        return self.partition_comparison.export_partitions()

    def completed_partitions(self, results, snapshot):
        return self.partition_comparison.completed_partitions(results, snapshot)

    def selected_partition_row(self):
        return self.partition_comparison.selected_partition_row()

    def show_partition_tuning(self):
        return self.partition_comparison.show_partition_tuning()

    def show_partition(self, snapshot, limit):
        return self.partition_comparison.show_partition(snapshot, limit)

    def clear_residual_locations(self):
        return self.partition_comparison.clear_residual_locations()

    @staticmethod
    def sample_location_labels(snapshot):
        return SpatialPartitionComparison.sample_location_labels(snapshot)

    def set_residual_locations(self, locations, snapshot, ax):
        return self.partition_comparison.set_residual_locations(locations, snapshot, ax)

    def highlight_residual_location(self, *_):
        return self.partition_comparison.highlight_residual_location(*_)

    def pick_residual_location(self, event):
        return self.partition_comparison.pick_residual_location(event)

    def calculate_kriging(self, *, preview):
        return spatial_kriging.calculate_kriging(self, preview=preview)

    def redraw_comparison_surface(self, *_):
        return spatial_kriging.redraw_comparison_surface(self, *_)

    def comparison_grid(self, result, method):
        return spatial_kriging.comparison_grid(self, result, method)

    def draw_variogram(self, ax, result, model):
        return spatial_kriging.draw_variogram(self, ax, result, model)

    @staticmethod
    def draw_surface_grid(ax, grid, values, **style):
        return spatial_kriging.draw_surface_grid(ax, grid, values, **style)

    def draw_idw_tin_difference(self, result):
        return spatial_grid.draw_idw_tin_difference(self, result)

    @staticmethod
    def add_grid_index_fields(layout):
        return spatial_grid.add_grid_index_fields(layout)

    @staticmethod
    def pick_grid_indices(event, grid, markers, fields):
        return spatial_grid.pick_grid_indices(event, grid, markers, fields)

    def pick_grid_node(self, event):
        return spatial_grid.pick_grid_node(self, event)

    def inspect_grid_node(self, *_, announce=True):
        return spatial_grid.inspect_grid_node(self, *_, announce=announce)

    def completed_kriging(self, result, model):
        return spatial_kriging.completed_kriging(self, result, model)

    def refresh_comparison_metrics(self, *_):
        return spatial_kriging.refresh_comparison_metrics(self, *_)

    @staticmethod
    def kriging_method_label(method, anisotropic=False):
        return spatial_kriging.kriging_method_label(method, anisotropic)

    def set_kriging_rows(self, headers, rows):
        return spatial_kriging.set_kriging_rows(self, headers, rows)

    def active_surface(self):
        return spatial_export.active_surface(self)

    def update_surface_actions(self, *_):
        return spatial_export.update_surface_actions(self, *_)

    def add_surface_to_map(self):
        return spatial_export.add_surface_to_map(self)

    def export_surface_tiff(self):
        return spatial_export.export_surface_tiff(self)

    def invalidate(self, *, keep_comparison=False):
        self.generation += 1
        self.input_summary.setText(tr('Выборка для проверки или расчёта ещё не подготовлена.'))
        self.clear_sample_preview()
        self.result = None
        self.set_idw_preview_visible(False)
        self.export_button.setEnabled(False)
        self.figure.clear()
        self.canvas.draw_idle()
        self.block_result = None
        self.block_patches = []
        self.block_buffer_outline = None
        self.block_figure.clear()
        self.block_canvas.draw_idle()
        self.set_block_rows([])
        if not keep_comparison:
            self.clear_settings_comparison()
        self.clear_kriging()
        self.set_interpolation_status(tr('Параметры изменены; выполните расчёт заново.'), announce=False)

    def selected_region_layer(self):
        return spatial_input.selected_region_layer(self)

    def update_region_fields(self):
        return spatial_input.update_region_fields(self)

    def update_region_keys(self):
        return spatial_input.update_region_keys(self)

    def update_region_controls(self):
        return spatial_input.update_region_controls(self)

    def calculation_input(self):
        return spatial_input.calculation_input(self)

    def calculate(self):
        return spatial_idw.calculate(self)

    def calculate_blocks(self, *, evaluate):
        return spatial_blocks.calculate_blocks(self, evaluate=evaluate)

    def calculate_spread(self):
        return spatial_spread.calculate_spread(self)

    def show_spread(self, result, snapshot, options, size):
        return spatial_spread.show_spread(self, result, snapshot, options, size)

    def set_block_rows(self, rows):
        return spatial_blocks.set_block_rows(self, rows)

    def highlight_block(self, *_):
        return spatial_blocks.highlight_block(self, *_)

    def draw_blocks(self, figure, result, snapshot, *, residual_limit=None):
        return spatial_blocks.draw_blocks(self, figure, result, snapshot, residual_limit=residual_limit)

    def completed_blocks(self, result, snapshot):
        return spatial_blocks.completed_blocks(self, result, snapshot)

    def completed(self, result):
        return spatial_idw.completed(self, result)

    def export(self):
        return spatial_export.export(self)

    # Existing callers can access the controls and results through the dialog.

    @property
    def apply_variant(self):
        return self.settings_comparison.apply_variant

    @property
    def export_variants(self):
        return self.settings_comparison.export_variants

    @property
    def nested_folds(self):
        return self.settings_comparison.nested_folds

    @property
    def nested_note(self):
        return self.settings_comparison.nested_note

    @property
    def nested_ranges(self):
        return self.settings_comparison.nested_ranges

    @property
    def nested_result(self):
        return self.settings_comparison.nested_result

    @nested_result.setter
    def nested_result(self, value):
        self.settings_comparison.nested_result = value

    @property
    def nested_shift(self):
        return self.settings_comparison.nested_shift

    @property
    def nested_summary(self):
        return self.settings_comparison.nested_summary

    @property
    def settings_result(self):
        return self.settings_comparison.settings_result

    @settings_result.setter
    def settings_result(self, value):
        self.settings_comparison.settings_result = value

    @property
    def settings_table(self):
        return self.settings_comparison.settings_table

    @property
    def variant_fields(self):
        return self.settings_comparison.variant_fields

    @property
    def export_partition_button(self):
        return self.partition_comparison.export_partition_button

    @property
    def partition_canvas(self):
        return self.partition_comparison.partition_canvas

    @property
    def partition_details(self):
        return self.partition_comparison.partition_details

    @property
    def partition_export(self):
        return self.partition_comparison.partition_export

    @partition_export.setter
    def partition_export(self, value):
        self.partition_comparison.partition_export = value

    @property
    def partition_figure(self):
        return self.partition_comparison.partition_figure

    @property
    def partition_nested(self):
        return self.partition_comparison.partition_nested

    @property
    def partition_results(self):
        return self.partition_comparison.partition_results

    @partition_results.setter
    def partition_results(self, value):
        self.partition_comparison.partition_results = value

    @property
    def partition_setup(self):
        return self.partition_comparison.partition_setup

    @property
    def partition_shift(self):
        return self.partition_comparison.partition_shift

    @property
    def partition_sizes(self):
        return self.partition_comparison.partition_sizes

    @property
    def partition_table(self):
        return self.partition_comparison.partition_table

    @property
    def partition_tuning(self):
        return self.partition_comparison.partition_tuning

    @property
    def residual_locations(self):
        return self.partition_comparison.residual_locations

    @residual_locations.setter
    def residual_locations(self, value):
        self.partition_comparison.residual_locations = value

    @property
    def residual_marker(self):
        return self.partition_comparison.residual_marker

    @residual_marker.setter
    def residual_marker(self, value):
        self.partition_comparison.residual_marker = value

    @property
    def residual_table(self):
        return self.partition_comparison.residual_table
