"""Manage scheme revisions, draft state and undo commands."""

from __future__ import annotations

from copy import deepcopy

from PySide6.QtCore import Qt
from PySide6.QtGui import QUndoCommand
from PySide6.QtWidgets import QListWidgetItem

from autopyro.schemes.interpretation import validate_scheme

from ..i18n import tr, message


class DraftCommand(QUndoCommand):
    def __init__(self, editor, before, after, label):
        super().__init__(label)
        self.editor, self.before, self.after = editor, deepcopy(before), deepcopy(after)

    def redo(self):
        self.editor.set_draft(self.after)

    def undo(self):
        self.editor.set_draft(self.before)


def push(sheet, after, label):
    if after != sheet.draft:
        sheet.undo_stack.push(DraftCommand(sheet, sheet.draft, after, label))


def set_draft(sheet, scheme):
    sheet.loading = True
    sheet.draft = deepcopy(scheme)
    sheet._diagnostics = None
    for widget, key in (
        (sheet.name, "name"),
        (sheet.revision, "revision"),
        (sheet.source, "source"),
    ):
        widget.setText(sheet.draft[key])
        widget.setToolTip(sheet.draft[key])
        widget.setCursorPosition(0)
    sheet.author_status.setCurrentIndex(
        sheet.author_status.findData(sheet.draft["scientific_status"])
    )
    sheet.objects.blockSignals(True)
    sheet.objects.clear()
    selected_index = 0
    for collection, prefix in (("regions", tr('Область')), ("lines", tr('Линия'))):
        for entry in sheet.draft.get(collection, []):
            item = QListWidgetItem(f"{prefix}: {entry['label']}")
            item.setData(Qt.ItemDataRole.UserRole, (collection, entry["id"]))
            if sheet.selected == (collection, entry["id"]):
                selected_index = sheet.objects.count()
            sheet.objects.addItem(item)
    sheet.objects.blockSignals(False)
    sheet.loading = False
    sheet.objects.setCurrentRow(selected_index if sheet.objects.count() else -1)
    sheet.select_object()


def metadata_changed(sheet, *args):
    if sheet.loading:
        return
    after = deepcopy(sheet.draft)
    after.update(
        name=sheet.name.text().strip(),
        revision=sheet.revision.text().strip(),
        source=sheet.source.text().strip(),
        scientific_status=sheet.author_status.currentData(),
    )
    sheet.push(after, tr('Изменить описание схемы'))


def apply(sheet):
    if sheet.drawing_points or sheet.drag:
        sheet.diagnostics.setText(tr('Завершите или сбросьте текущий контур / перенос'))
        return
    sheet.metadata_changed()
    try:
        validate_scheme(sheet.draft)
        if (
            not sheet.is_new
            and sheet.draft != sheet.original
            and sheet.draft["revision"] == sheet.original["revision"]
        ):
            raise ValueError(tr('Изменённая схема должна иметь новую редакцию'))
        sheet.edited_scheme = deepcopy(sheet.draft)
        sheet.accept()
    except (ValueError, KeyError, TypeError) as exc:
        sheet.diagnostics.setText(message(exc))
