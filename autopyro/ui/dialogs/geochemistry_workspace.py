"""Linked-sample passports and descriptive group comparison."""

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QDialog
from ..i18n import tr, message
from . import (
    geochemistry_workspace_archive,
    geochemistry_workspace_catalog,
    geochemistry_workspace_comparison,
    geochemistry_workspace_exports,
    geochemistry_workspace_forms,
    geochemistry_workspace_models,
)
from .geochemistry_workspace_catalog import MaterialLinkDialog as MaterialLinkDialog, WorkspaceCommand as WorkspaceCommand
from .geochemistry_workspace_forms import ComparisonChoicesModel as ComparisonChoicesModel


class GeochemistryWorkspaceDialog(QDialog):
    """Compose the catalog, comparison and model views for a project."""

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.index = None
        self._index_context = None
        self._index_measurements = None
        self._index_measurement_count = 0
        self._catalog_rows = None
        self._catalog_schema = None
        self._catalog_scope = None
        self.report = None
        self.profile_match = None
        self.prediction_run = None
        self.prediction_application = None
        self.pca_run = None
        self.clustering_run = None
        self.kmeans_run = None
        self.dbscan_run = None
        self._report_context = None
        self._loading = False
        self.setWindowTitle(tr('Геохимические паспорта и сравнение'))
        geochemistry_workspace_forms.build(self)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.refresh)
        self.refresh()
        owner.model.changed.connect(self.changed)
        owner.analysis_scope.bind(self.scope, self.changed)
        geochemistry_workspace_comparison.restore_saved_comparison(self)

    def selected_samples(self):
        return geochemistry_workspace_catalog.selected_samples(self)

    def invalidate_oil_correlation(self, *_):
        return geochemistry_workspace_comparison.invalidate_oil_correlation(self, *_)

    def changed(self, *_):
        return geochemistry_workspace_catalog.changed(self, *_)

    def invalidate_comparison(self, *_):
        return geochemistry_workspace_comparison.invalidate_comparison(self, *_)

    def clear_multivariate_results(self, *_):
        return geochemistry_workspace_comparison.clear_multivariate_results(self, *_)

    def preprocessing(self):
        return geochemistry_workspace_comparison.preprocessing(self)

    def change_preprocessing(self):
        return geochemistry_workspace_comparison.change_preprocessing(self)

    def save_model_run(self, report, result):
        return geochemistry_workspace_archive.save_model_run(self, report, result)

    def model_title(self, result):
        return geochemistry_workspace_archive.model_title(self, result)

    def show_model_archive(self):
        return geochemistry_workspace_archive.show_model_archive(self)

    def show_review_archive(self):
        return geochemistry_workspace_archive.show_review_archive(self)

    def refresh_review_archive(self):
        return geochemistry_workspace_archive.refresh_review_archive(self)

    def selected_archived_review(self):
        return geochemistry_workspace_archive.selected_archived_review(self)

    def display_archived_review(self):
        return geochemistry_workspace_archive.display_archived_review(self)

    def export_archived_review(self):
        return geochemistry_workspace_archive.export_archived_review(self)

    def refresh_model_archive(self):
        return geochemistry_workspace_archive.refresh_model_archive(self)

    def preprocessing_label(self, result):
        return geochemistry_workspace_archive.preprocessing_label(self, result)

    def display_model_passport(self):
        return geochemistry_workspace_archive.display_model_passport(self)

    def selected_model_run(self):
        return geochemistry_workspace_archive.selected_model_run(self)

    def apply_archived_prediction(self):
        return geochemistry_workspace_archive.apply_archived_prediction(self)

    def export_prediction_application(self):
        return geochemistry_workspace_archive.export_prediction_application(self)

    def open_model_run(self, *_):
        return geochemistry_workspace_archive.open_model_run(self, *_)

    def check_model_run(self):
        return geochemistry_workspace_archive.check_model_run(self)

    def export_model_run(self):
        return geochemistry_workspace_archive.export_model_run(self)

    def import_model_run(self):
        return geochemistry_workspace_archive.import_model_run(self)

    def refresh(self):
        return geochemistry_workspace_catalog.refresh(self)

    def filter_samples(self, text):
        return geochemistry_workspace_catalog.filter_samples(self, text)

    def filter_comparison_fields(self, text):
        return geochemistry_workspace_catalog.filter_comparison_fields(self, text)

    def select_field_preset(self):
        return geochemistry_workspace_catalog.select_field_preset(self)

    def reset_links(self):
        return geochemistry_workspace_catalog.reset_links(self)

    def update_sample_count(self):
        return geochemistry_workspace_catalog.update_sample_count(self)

    def show_passport(self):
        return geochemistry_workspace_catalog.show_passport(self)

    def update_measurement_actions(self):
        return geochemistry_workspace_catalog.update_measurement_actions(self)

    def commit(self, state, title):
        return geochemistry_workspace_catalog.commit(self, state, title)

    def link_selected(self):
        return geochemistry_workspace_catalog.link_selected(self)

    def unlink_selected(self):
        return geochemistry_workspace_catalog.unlink_selected(self)

    def repair_invalid_link(self):
        return geochemistry_workspace_catalog.repair_invalid_link(self)

    def link_materials(self):
        return geochemistry_workspace_catalog.link_materials(self)

    def remove_material_links(self):
        return geochemistry_workspace_catalog.remove_material_links(self)

    def save_selected_group(self):
        return geochemistry_workspace_catalog.save_selected_group(self)

    def delete_group(self):
        return geochemistry_workspace_catalog.delete_group(self)

    def measurement_record(self):
        return geochemistry_workspace_catalog.measurement_record(self)

    def choose_measurement(self):
        return geochemistry_workspace_catalog.choose_measurement(self)

    def reset_measurement(self):
        return geochemistry_workspace_catalog.reset_measurement(self)

    def open_method(self):
        return geochemistry_workspace_catalog.open_method(self)

    def resolve_group(self, combo):
        return geochemistry_workspace_comparison.resolve_group(self, combo)

    def comparison_selection(self):
        return geochemistry_workspace_comparison.comparison_selection(self)

    def preview_oil_correlation(self):
        return geochemistry_workspace_comparison.preview_oil_correlation(self)

    def show_correlation_coverage(self, match, names):
        return geochemistry_workspace_comparison.show_correlation_coverage(self, match, names)

    def calculate_comparison(self):
        return geochemistry_workspace_comparison.calculate_comparison(self)

    def display_comparison(self):
        return geochemistry_workspace_comparison.display_comparison(self)

    def comparison_measurement(self, row, column):
        return geochemistry_workspace_comparison.comparison_measurement(self, row, column)

    def open_comparison_measurement(self, row, column):
        return geochemistry_workspace_comparison.open_comparison_measurement(self, row, column)

    def update_correlation_readiness(self):
        return geochemistry_workspace_comparison.update_correlation_readiness(self)

    def show_profile_match(self, *, oil_correlation=False):
        return geochemistry_workspace_comparison.show_profile_match(self, oil_correlation=oil_correlation)

    def show_correlation_sensitivity(self):
        return geochemistry_workspace_comparison.show_correlation_sensitivity(self)

    def save_profile_review(self):
        return geochemistry_workspace_comparison.save_profile_review(self)

    def calculate_oil_mixing(self):
        return geochemistry_workspace_models.calculate_oil_mixing(self)

    def render_oil_mixing(self, report, run):
        return geochemistry_workspace_models.render_oil_mixing(self, report, run)

    def calculate_prediction(self):
        return geochemistry_workspace_models.calculate_prediction(self)

    def render_prediction(self, report, run):
        return geochemistry_workspace_models.render_prediction(self, report, run)

    def calculate_pca(self):
        return geochemistry_workspace_models.calculate_pca(self)

    def render_pca(self, report, run):
        return geochemistry_workspace_models.render_pca(self, report, run)

    def dbscan_parameters(self, title, sample_count):
        return geochemistry_workspace_models.dbscan_parameters(self, title, sample_count)

    def calculate_clustering(self, *, method='ward'):
        return geochemistry_workspace_models.calculate_clustering(self, method=method)

    def render_clustering(self, report, run, title):
        return geochemistry_workspace_models.render_clustering(self, report, run, title)

    def draw_comparison(self):
        return geochemistry_workspace_comparison.draw_comparison(self)

    def export_passport(self):
        return geochemistry_workspace_exports.export_passport(self)

    def export_correlation(self):
        return geochemistry_workspace_exports.export_correlation(self)

    def export_comparison(self):
        return geochemistry_workspace_exports.export_comparison(self)

    def check_comparison_export(self, report, *, match=None):
        return geochemistry_workspace_exports.check_comparison_export(self, report, match=match)

    def attempt(self, function):
        try:
            function()
        except (ValueError, KeyError, OSError) as exc:
            self.status.setText(message(exc))
