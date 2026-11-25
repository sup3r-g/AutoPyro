"""Regional geochemistry worksheet linked to the map."""

from copy import deepcopy

from PySide6.QtCore import Signal

from ..i18n import tr
from ..tasks import TaskDialog
from . import (
    regional_geochemistry_details,
    regional_geochemistry_forms,
    regional_geochemistry_inputs,
    regional_geochemistry_results,
    regional_geochemistry_scenarios,
)
from .regional_geochemistry_details import (
    SAMPLE_STATUS as SAMPLE_STATUS,
    SPATIAL_COLORS as SPATIAL_COLORS,
    SPATIAL_STATUS as SPATIAL_STATUS,
)


class RegionalGeochemistryDialog(TaskDialog):
    """Compose regional analysis, scenarios and the map-linked results."""

    map_selection_changed = Signal(object)
    sample_opened = Signal(str)
    scenarios_changed = Signal()

    def __init__(self, project, report, parent=None, *, data_revision=None):
        super().__init__(parent)
        self.input_revision = 0
        self.map_data_revision = data_revision
        self.statistics_exclusions = deepcopy(project.view.get('qc_exclusions', {}).get('statistics', {}))
        self.project, self.report, self.result = project, report, None
        self.setWindowTitle(tr('Геохимия по палеогеографическим областям'))
        self.resize(1250, 800)
        regional_geochemistry_forms.build(self, project, report)

    def build_detail_tabs(self):
        return regional_geochemistry_details.build_detail_tabs(self)

    def refresh_samples(self):
        return regional_geochemistry_details.refresh_samples(self)

    def replace_model(self, view, headers, rows):
        return regional_geochemistry_details.replace_model(self, view, headers, rows)

    def highlight_samples(self, *_):
        return regional_geochemistry_details.highlight_samples(self, *_)

    def refresh_profile(self, *_):
        return regional_geochemistry_details.refresh_profile(self, *_)

    def draw_profile(self, *_):
        return regional_geochemistry_details.draw_profile(self, *_)

    def refresh_spatial(self, *_):
        return regional_geochemistry_details.refresh_spatial(self, *_)

    def highlight_spatial(self):
        return regional_geochemistry_details.highlight_spatial(self)

    def detail_tab_changed(self, *_):
        return regional_geochemistry_details.detail_tab_changed(self, *_)

    def publish_comparison_selection(self, selection):
        return regional_geochemistry_scenarios.publish_comparison_selection(self, selection)

    def capture_comparison_snapshot(self):
        return regional_geochemistry_scenarios.capture_comparison_snapshot(self)

    def refresh_scenarios(self, name=''):
        return regional_geochemistry_scenarios.refresh_scenarios(self, name)

    def save_scenario(self):
        return regional_geochemistry_scenarios.save_scenario(self)

    def delete_scenario(self):
        return regional_geochemistry_scenarios.delete_scenario(self)

    def load_scenario(self):
        return regional_geochemistry_scenarios.load_scenario(self)

    def update_regions(self):
        return regional_geochemistry_inputs.update_regions(self)

    def populate(self, key, binding):
        return regional_geochemistry_inputs.populate(self, key, binding)

    def check_choices(self, listing, checked):
        return regional_geochemistry_inputs.check_choices(self, listing, checked)

    def set_map_input(self, project, report, *, data_revision):
        return regional_geochemistry_inputs.set_map_input(self, project, report, data_revision=data_revision)

    def invalidate(self, *_):
        self.input_revision += 1
        self.result = None
        if not hasattr(self, 'comparison') or self.tabs.currentWidget() is not self.comparison:
            self.map_selection_changed.emit(None)
        if not hasattr(self, 'exports'):
            return
        self.comparison.capture.setEnabled(False)
        for button in self.exports:
            button.setEnabled(False)
        self.table.clear()
        self.table.setRowCount(0)
        self.canvas.figure.clear()
        self.canvas.draw_idle()
        self.refresh_samples()
        self.refresh_spatial()
        self.diagnostics.setText(tr('Параметры изменены. Нажмите «Рассчитать сравнение».'))

    def options(self):
        return regional_geochemistry_inputs.options(self)

    def calculate(self):
        return regional_geochemistry_inputs.calculate(self)

    def show_result(self, result):
        return regional_geochemistry_results.show_result(self, result)

    def select_on_map(self, refresh_details=True):
        return regional_geochemistry_results.select_on_map(self, refresh_details)

    def select_map_region(self, layer_id, feature_id):
        return regional_geochemistry_results.select_map_region(self, layer_id, feature_id)

    def reject(self):
        if self.task_running:
            return
        self.map_selection_changed.emit(None)
        super().reject()

    def wait_for_worker(self):
        # The owning window can also be destroyed during application shutdown.
        if self.worker is not None:
            self.worker.wait()

    def closeEvent(self, event):
        super().closeEvent(event)
        if event.isAccepted():
            self.map_selection_changed.emit(None)

    def plot(self, *_):
        return regional_geochemistry_results.plot(self, *_)

    def export(self, kind):
        return regional_geochemistry_results.export(self, kind)
