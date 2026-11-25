"""Formula calculation worksheet and its public dialog interface."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtGui import QAccessible, QAccessibleAnnouncementEvent
from PySide6.QtWidgets import QDialog

from ..i18n import tr
from . import (
    calculation_comparison,
    calculation_forms,
    calculation_inputs,
    calculation_preview,
    calculation_workflow,
)
from .calculation_comparison import (
    COMPARISON_KEY as COMPARISON_KEY,
    COMPARISON_HEADERS as COMPARISON_HEADERS,
    SaveComparisonCommand as SaveComparisonCommand,
    SavedComparisonModel as SavedComparisonModel,
    comparison_rows as comparison_rows,
    valid_comparison_row as valid_comparison_row,
)
from .calculation_details import (
    calculation_detail as calculation_detail,
    calculation_statuses as calculation_statuses,
    comparison_status_text as comparison_status_text,
)


class CalculationDialog(QDialog):
    """Compose formula controls, result preview and laboratory comparison."""

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.prepared = []
        self._preview_source = None
        self._preview_revision = 0
        self.setWindowTitle(tr('Пиролиз · вычисляемые показатели'))
        self.resize(950, 700)
        calculation_forms.build(self)

    def eventFilter(self, watched, event):
        if (watched is self.sample_search and event.type() == QEvent.Type.KeyPress
                and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)):
            if self.preview.rowCount():
                if self.preview.currentRow() < 0:
                    self.preview.setCurrentCell(0, 1)
                self.preview.setFocus(Qt.FocusReason.TabFocusReason)
            return True
        return super().eventFilter(watched, event)

    def showEvent(self, event):
        if self.isWindow():
            available = self.screen().availableGeometry().size() - QSize(24, 48)
            size = self.size().boundedTo(available)
            if size != self.size():
                self._adapt_layout(size.width())
                self.layout().activate()
                self.resize(size)
        super().showEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if getattr(self, "_layout_ready", False):
            self._adapt_layout()

    def _adapt_layout(self, width=None):
        return calculation_forms._adapt_layout(self, width)

    def restore_parameters(self):
        return calculation_inputs.restore_parameters(self)

    def invalidate(self, *args, announce=True):
        return calculation_preview.invalidate(self, *args, announce=announce)

    def show_input_errors(self, errors):
        return calculation_inputs.show_input_errors(self, errors)

    def calculate(self):
        return calculation_workflow.calculate(self)

    @staticmethod
    def announce(widget, text):
        if text and QAccessible.isActive():
            QAccessible.updateAccessibility(QAccessibleAnnouncementEvent(widget, text))

    def set_status(self, text, *, announce=True):
        self.summary.setText(text)
        if announce:
            self.announce(self.summary, text)

    def update_preview_summary(self):
        return calculation_preview.update_preview_summary(self)

    def change_page(self, step):
        return calculation_preview.change_page(self, step)

    def render_preview(self):
        return calculation_preview.render_preview(self)

    def clear_preview_filters(self):
        return calculation_preview.clear_preview_filters(self)

    def show_preview_detail(self, item, _previous):
        return calculation_preview.show_preview_detail(self, item, _previous)

    def jump_to_sample(self, row, _column):
        return calculation_preview.jump_to_sample(self, row, _column)

    def export_comparison(self):
        return calculation_comparison.export_comparison(self)

    def export_comparison_rows(self, rows, *, historical=False):
        return calculation_comparison.export_comparison_rows(self, rows, historical=historical)

    def save_comparison(self):
        return calculation_comparison.save_comparison(self)

    def open_saved_comparisons(self):
        return calculation_comparison.open_saved_comparisons(self)

    def input_column_changed(self, role):
        return calculation_inputs.input_column_changed(self, role)

    @staticmethod
    def lab_unit(formula, column):
        return calculation_inputs.lab_unit(formula, column)

    def lab_column_changed(self, formula):
        return calculation_inputs.lab_column_changed(self, formula)

    def apply(self):
        return calculation_workflow.apply(self)

    def refresh_columns(self):
        return calculation_inputs.refresh_columns(self)

    def update_lab_choices(self):
        return calculation_inputs.update_lab_choices(self)
