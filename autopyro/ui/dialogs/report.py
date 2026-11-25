"""Report worksheet and its public dialog interface."""

from __future__ import annotations

from PySide6.QtGui import QAccessible, QAccessibleAnnouncementEvent
from PySide6.QtWidgets import QDialog

from ..i18n import tr
from . import (
    report_batch,
    report_composition,
    report_export,
    report_forms,
    report_preview,
    report_templates,
)
from .report_preview import ReportPreview as ReportPreview, resize_to_screen
from .report_templates import (
    BUILTIN_REPORTS as BUILTIN_REPORTS,
    GROUP_ALIASES as GROUP_ALIASES,
    PLOT_KEY_ROLE as PLOT_KEY_ROLE,
    legacy_plot_key as legacy_plot_key,
)


class ReportDialog(QDialog):
    """Compose reports, frozen previews and export workflows."""

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.report = None
        self.report_source = None
        self.preview_report = None
        self.section_order = None
        self.section_titles = {}
        self.block_comments = {}
        self.logo_image = None
        self.setWindowTitle(tr('Отчёт · HTML / PDF / DOCX / снимок JSON'))
        resize_to_screen(self, 1250, 850)
        report_forms.build(self)

    @staticmethod
    def report_shortcut(parent, keys, callback):
        return report_forms.report_shortcut(parent, keys, callback)

    def focus_preview_sections(self):
        return report_preview.focus_preview_sections(self)

    def announce(self, text):
        if QAccessible.isActive():
            QAccessible.updateAccessibility(QAccessibleAnnouncementEvent(self.status, text))

    def announce_status(self, text):
        self.status.setText(text)
        self.announce(text)

    def scope_changed(self):
        return report_composition.scope_changed(self)

    def update_scope_summary(self):
        return report_composition.update_scope_summary(self)

    def add_list_controls(self, layout, items, search_label):
        return report_forms.add_list_controls(self, layout, items, search_label)

    def invalidate(self, *_):
        return report_preview.invalidate(self, *_)

    def composition(self):
        return report_composition.composition(self)

    def selection_context(self, scope, resolved_ids=None):
        return report_composition.selection_context(self, scope, resolved_ids)

    def edit_sections(self):
        return report_composition.edit_sections(self)

    def edit_comments(self):
        return report_composition.edit_comments(self)

    def save_template(self):
        return report_templates.save_template(self)

    def template_record(self, options, plots):
        return report_templates.template_record(self, options, plots)

    def load_template(self):
        return report_templates.load_template(self)

    @staticmethod
    def plot_references_exist(config, columns):
        return report_templates.plot_references_exist(config, columns)

    def repair_template(self, options, plot_refs, lost):
        return report_templates.repair_template(self, options, plot_refs, lost)

    def apply_builtin_template(self, name):
        return report_templates.apply_builtin_template(self, name)

    def export_batch(self):
        return report_batch.export_batch(self)

    def choose_batch_formats(self):
        return report_batch.choose_batch_formats(self)

    def write_batch(self, prepared, parent):
        return report_batch.write_batch(self, prepared, parent)

    def retry_batch(self, failed, folder):
        return report_batch.retry_batch(self, failed, folder)

    def review_batch(self, prepared):
        return report_batch.review_batch(self, prepared)

    def build(self):
        return report_preview.build(self)

    def update_check_actions(self, report):
        return report_preview.update_check_actions(self, report)

    def preview_pages(self, kind):
        return report_preview.preview_pages(self, kind)

    def show_pdf_pages(self, payload):
        return report_preview.show_pdf_pages(self, payload)

    def archive_report(self):
        return report_export.archive_report(self)

    def choose_logo(self):
        return report_composition.choose_logo(self)

    def set_logo(self, image):
        return report_composition.set_logo(self, image)

    def show_history(self):
        return report_export.show_history(self)

    def export(self, kind, *, archive=False):
        return report_export.export(self, kind, archive=archive)
