"""Quality-control worksheet and its public dialog interface."""

from PySide6.QtWidgets import QDialog

from ..i18n import tr
from . import (
    qc_checks,
    qc_decisions,
    qc_exclusions,
    qc_forms,
    qc_navigation,
    qc_results,
)
from .qc_navigation import source_details_text as source_details_text
from .qc_results import (
    QCIssuesModel as QCIssuesModel,
    source_location as source_location,
    source_location_labels as source_location_labels,
    source_search_text as source_search_text,
)


class QCDialog(QDialog):
    """Compose QC checks, review actions and navigation to project data."""

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.setWindowTitle(tr('Контроль качества · проверки и решения'))
        self.resize(1150, 720)
        self.issues = []
        self.profile_coverage = []
        self._checked_revision = None
        self._decision_revision = None
        self._checking = False
        self._check_error = ""
        self._scope_cache = None
        self._sort_cache = None
        self._review_cache = None
        self._source_search_cache = {}
        self._render_key = None
        self._draft_issue = None
        self._draft_exclusion = None
        self._decision_drafts = {}
        self._exclusion_drafts = {}
        self._previous_filter = None
        self._compact_table = False
        self._revealed_row = None
        self._revealed_columns = {}
        qc_forms.build(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        qc_forms.resize_layout(self, event)

    def adjust_compact_docks(self):
        return qc_forms.adjust_compact_docks(self)

    def on_activated(self):
        return qc_checks.on_activated(self)

    def refresh_checks(self):
        return qc_checks.refresh_checks(self)

    def schedule(self):
        return qc_checks.schedule(self)

    def edit_profile(self):
        return qc_checks.edit_profile(self)

    def reload(self):
        return qc_checks.reload(self)

    def current_issue(self):
        return qc_results.current_issue(self)

    def select_issue_row(self, row):
        # QTableView.selectRow follows the current keyboard modifiers.
        return qc_results.select_issue_row(self, row)

    def selected_spans(self):
        return qc_results.selected_spans(self)

    def selected_issue(self):
        return qc_results.selected_issue(self)

    def render(self, *_):
        return qc_results.render(self, *_)

    def show_coverage(self):
        return qc_checks.show_coverage(self)

    def show_details(self, *_):
        return qc_results.show_details(self, *_)

    @staticmethod
    def switch_draft(field, drafts, previous, current):
        return qc_results.switch_draft(field, drafts, previous, current)

    def open_measurement(self):
        return qc_navigation.open_measurement(self)

    def show_measurement_card(self, measurement, parent):
        return qc_navigation.show_measurement_card(self, measurement, parent)

    def show_source(self):
        return qc_navigation.show_source(self)

    def decide(self, action):
        return qc_decisions.decide(self, action)

    def batch_decision(self):
        return qc_decisions.batch_decision(self)

    def action_issue(self):
        return qc_decisions.action_issue(self)

    def exclude(self, excluded):
        return qc_exclusions.exclude(self, excluded)

    def update_exclusion(self, scope, mid, record, *, refresh=True):
        return qc_exclusions.update_exclusion(self, scope, mid, record, refresh=refresh)

    def manage_exclusions(self):
        return qc_exclusions.manage_exclusions(self)

    def go_to_sample(self):
        return qc_navigation.go_to_sample(self)

    def set_temporary_column_visibility(self, column, visible):
        return qc_navigation.set_temporary_column_visibility(self, column, visible)

    def forget_temporary_column(self, column_id):
        return qc_navigation.forget_temporary_column(self, column_id)

    def temporary_row_filter_active(self):
        return qc_navigation.temporary_row_filter_active(self)

    def persistent_table_view(self, view, *, include_sample_set):
        return qc_navigation.persistent_table_view(self, view, include_sample_set=include_sample_set)

    def restore_table_filter(self):
        return qc_navigation.restore_table_filter(self)
