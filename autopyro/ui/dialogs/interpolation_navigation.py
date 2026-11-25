"""Open source samples and interpolation supports from preview results."""

from __future__ import annotations

from PySide6.QtCore import QModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QListWidget,
    QListWidgetItem,
    QTableView,
    QVBoxLayout,
)

from ..i18n import tr


class SampleNavigationTable(QTableView):
    sample_requested = Signal(QModelIndex)

    def __init__(self):
        super().__init__()
        self.doubleClicked.connect(self.sample_requested)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and event.modifiers() in (
            Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.KeypadModifier
        ):
            if not event.isAutoRepeat():
                self.sample_requested.emit(self.currentIndex())
            event.accept()
            return
        super().keyPressEvent(event)


def can_open_sample(sheet):
    return (hasattr(sheet.owner, "open_source_cell") and sheet.owner.model.project is sheet.project
            and sheet._diagnostic_column_id is not None and sheet.current_diagnostics())


def open_diagnostic_sample(sheet, index):
    if not index.isValid() or not sheet.can_open_sample():
        return
    row_id = sheet.diagnostic_model.rows[index.row()][-1]
    sheet.owner.open_source_cell(row_id, sheet._diagnostic_column_id, sheet)


def open_proposal_sample(sheet, index):
    if not index.isValid() or not sheet.can_open_sample():
        return
    proposal = sheet.proposals[index.row()]
    if index.column() != 3:
        sheet.owner.open_source_cell(proposal.row_id, sheet._diagnostic_column_id, sheet)
        return
    ids = (sheet._support_spans[proposal.left_id] if sheet.preview_model.segment_method
           else (proposal.left_id, proposal.right_id))
    sheet.open_sample_choices(ids, tr('Опорные пробы'))


def open_sample_choices(sheet, ids, title):
    dialog = QDialog(sheet)
    dialog.setWindowTitle(title)
    layout = QVBoxLayout(dialog)
    samples = QListWidget(dialog)
    for rid in ids:
        label, well = sheet.preview_model.sample_labels[rid]
        item = QListWidgetItem(f"{label} · {well}" if well else label)
        item.setData(Qt.ItemDataRole.UserRole, rid)
        item.setToolTip(rid)
        samples.addItem(item)
    layout.addWidget(samples)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Close)
    layout.addWidget(buttons)

    def open_selected():
        if (item := samples.currentItem()) and sheet.can_open_sample():
            dialog.accept()
            sheet.owner.open_source_cell(item.data(Qt.ItemDataRole.UserRole), sheet._diagnostic_column_id, sheet)

    samples.setCurrentRow(0)
    samples.itemDoubleClicked.connect(open_selected)
    buttons.accepted.connect(open_selected)
    buttons.rejected.connect(dialog.reject)
    dialog.exec()
