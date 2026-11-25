"""Local interval draft undo: cell deltas and only the rows actually inserted/deleted."""

from contextlib import contextmanager
from copy import deepcopy
from time import monotonic

from PySide6.QtCore import QEvent, QObject, QSignalBlocker, Qt
from PySide6.QtGui import QUndoCommand, QUndoStack
from PySide6.QtWidgets import QApplication, QLineEdit

from ..i18n import tr

PREVIOUS_TEXT = int(Qt.ItemDataRole.UserRole) + 1


class CellChanges(QUndoCommand):
    def __init__(self, history, changes, text, already_applied=False):
        super().__init__(text)
        self.history, self.changes = history, changes
        self.already_applied = already_applied

    def redo(self):
        if self.already_applied:
            self.already_applied = False
        else:
            self.history.apply_cells(self.changes, after=True)

    def undo(self):
        self.history.apply_cells(self.changes, after=False)


class InsertInterval(QUndoCommand):
    def __init__(self, history, row, interval):
        super().__init__(tr('Добавление интервала'))
        self.history, self.row, self.interval = history, row, deepcopy(interval)

    def redo(self):
        with self.history.suspended():
            self.history.owner.insert_interval(self.row, self.interval)
        self.history.structure_changed()

    def undo(self):
        with self.history.suspended():
            self.history.table.removeRow(self.row)
        self.history.structure_changed()


class DeleteIntervals(QUndoCommand):
    def __init__(self, history, rows):
        super().__init__(tr('Удаление интервалов'))
        self.history = history
        self.rows = [(row, [history.table.item(row, col).clone()
                           for col in range(history.table.columnCount())]) for row in rows]

    def redo(self):
        with self.history.suspended():
            for row, cells in reversed(self.rows):
                self.history.table.removeRow(row)
        self.history.structure_changed()

    def undo(self):
        with self.history.suspended():
            for row, cells in self.rows:
                self.history.table.insertRow(row)
                for col, cell in enumerate(cells):
                    self.history.table.setItem(row, col, cell.clone())
        self.history.structure_changed()


class ReplaceDraft(QUndoCommand):
    def __init__(self, history, scheme):
        super().__init__(tr('Замена черновика интервалов'))
        self.history, self.scheme = history, scheme
        self.previous = history.owner.draft
        self.metadata = dict(history.metadata)
        self.rows = [[history.table.item(row, col).clone() for col in range(history.table.columnCount())]
                     for row in range(history.table.rowCount())]

    def redo(self):
        owner = self.history.owner
        with self.history.suspended():
            owner.draft = self.scheme
            owner.populate_draft()
        self.history.sync_metadata()
        self.history.structure_changed()

    def undo(self):
        owner = self.history.owner
        with self.history.suspended():
            owner.draft = self.previous
            self.history.table.setRowCount(len(self.rows))
            for row, cells in enumerate(self.rows):
                for col, cell in enumerate(cells):
                    self.history.table.setItem(row, col, cell.clone())
            for widget, value in self.metadata.items():
                with QSignalBlocker(widget):
                    if isinstance(widget, QLineEdit):
                        widget.setText(value)
                    else:
                        widget.setCurrentText(value)
        self.history.sync_metadata()
        self.history.structure_changed()


class MergeIntervals(QUndoCommand):
    def __init__(self, history, entries):
        super().__init__(tr('Объединение интервалов'))
        self.history = history
        self.start = history.table.rowCount()
        self.additions = [entry.item for entry in entries if not entry.target]
        self.updates = []
        for entry in entries:
            if entry.target:
                row = history.rows_by_id[entry.target]
                cells = [history.table.item(row, col).clone()
                         for col in range(history.table.columnCount())]
                self.updates.append((row, cells, entry.item))

    def redo(self):
        with self.history.suspended():
            for row, _, item in self.updates:
                self.history.owner.write_interval(row, item)
            for offset, item in enumerate(self.additions):
                self.history.owner.insert_interval(self.start + offset, item)
        self.history.structure_changed()

    def undo(self):
        with self.history.suspended():
            self.history.table.setRowCount(self.start)
            for row, cells, _ in self.updates:
                for col, cell in enumerate(cells):
                    self.history.table.setItem(row, col, cell.clone())
        self.history.structure_changed()


class MetadataChange(QUndoCommand):
    def __init__(self, history, widget, before, after):
        super().__init__(tr('Свойства схемы'))
        self.history, self.widget = history, widget
        self.before, self.after = before, after
        self.timestamp = monotonic()
        self.already_applied = True

    def id(self):
        return 1

    def mergeWith(self, other):
        if not isinstance(other, MetadataChange) or other.widget is not self.widget:
            return False
        if other.timestamp - self.timestamp > 1:
            return False
        self.after, self.timestamp = other.after, other.timestamp
        return True

    def apply(self, value):
        with QSignalBlocker(self.widget):
            if isinstance(self.widget, QLineEdit):
                self.widget.setText(value)
            else:
                self.widget.setCurrentText(value)
        self.history.metadata[self.widget] = value
        self.history.owner.invalidate_column()

    def redo(self):
        if self.already_applied:
            self.already_applied = False
        else:
            self.apply(self.after)

    def undo(self):
        self.apply(self.before)


class IntervalHistory(QObject):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner, self.table = owner, owner.intervals
        self.stack = QUndoStack(self)
        self.loading = False
        self.rows_by_id = {}
        self.metadata = {}
        self.reset()
        self.table.itemChanged.connect(self.record_cell)
        for widget in (owner.name, owner.revision, owner.reference, owner.unit):
            signal = widget.textChanged if isinstance(widget, QLineEdit) else widget.currentTextChanged
            signal.connect(lambda value, field=widget: self.record_metadata(field, value))
        QApplication.instance().installEventFilter(self)

    @contextmanager
    def suspended(self):
        previous, self.loading = self.loading, True
        try:
            with QSignalBlocker(self.table):
                yield
        finally:
            self.loading = previous

    def reset(self):
        self.stack.clear()
        with self.suspended():
            self.reindex()
        self.sync_metadata()

    def sync_metadata(self):
        self.metadata = {widget: widget.text() if isinstance(widget, QLineEdit) else widget.currentText()
                         for widget in (self.owner.name, self.owner.revision, self.owner.reference, self.owner.unit)}

    def reindex(self):
        self.rows_by_id = {}
        for row in range(self.table.rowCount()):
            self.rows_by_id[self.table.item(row, 6).text()] = row
            for col in range(self.table.columnCount()):
                cell = self.table.item(row, col)
                cell.setData(PREVIOUS_TEXT, cell.text())

    def structure_changed(self):
        with self.suspended():
            self.reindex()
        self.owner.refresh_wellbores()
        self.owner.invalidate_column()

    def record_cell(self, cell):
        if self.loading:
            return
        before, after = cell.data(PREVIOUS_TEXT), cell.text()
        if before == after:
            return
        with self.suspended():
            cell.setData(PREVIOUS_TEXT, after)
        interval_id = self.table.item(cell.row(), 6).text()
        self.stack.push(CellChanges(self, [(interval_id, cell.column(), before, after)],
                                    tr('Изменение интервала'), already_applied=True))

    def record_metadata(self, widget, value):
        if self.loading:
            return
        before = self.metadata[widget]
        self.metadata[widget] = value
        if before != value:
            self.stack.push(MetadataChange(self, widget, before, value))

    def change_cells(self, changes, text):
        changes = [(key, col, self.table.item(self.rows_by_id[key], col).text(), value)
                   for key, col, value in changes
                   if self.table.item(self.rows_by_id[key], col).text() != value]
        if changes:
            self.stack.push(CellChanges(self, changes, text))
        return bool(changes)

    def apply_cells(self, changes, *, after):
        with self.suspended():
            for key, col, old, new in changes:
                cell = self.table.item(self.rows_by_id[key], col)
                value = new if after else old
                cell.setText(value)
                cell.setData(PREVIOUS_TEXT, value)
        if any(col == 0 for key, col, old, new in changes):
            self.owner.refresh_wellbores()
        self.owner.invalidate_column()

    def eventFilter(self, watched, event):
        if event.type() not in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress):
            return False
        modifiers = event.modifiers()
        if event.key() != Qt.Key.Key_Z or not modifiers & Qt.KeyboardModifier.ControlModifier:
            return False
        if modifiers & (Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier):
            return False
        focus = QApplication.focusWidget()
        if focus is None or not self.owner.isVisible() or focus.window() is not self.owner.window():
            return False
        in_document = focus is self.owner or self.owner.isAncestorOf(focus)
        in_properties = self.owner.parameter_tabs.isAncestorOf(focus)
        if not (in_document or in_properties):
            return False
        # Uncommitted text in an editor keeps Qt's native text undo.
        if isinstance(focus, QLineEdit) and focus not in self.metadata:
            return False
        event.accept()
        if event.type() == QEvent.Type.KeyPress:
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                self.stack.redo()
            else:
                self.stack.undo()
        return True
