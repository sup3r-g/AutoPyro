"""Capture and restore editable Qt parameter controls."""

from copy import deepcopy

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QAbstractSpinBox, QCheckBox, QComboBox, QDoubleSpinBox,
    QLineEdit, QListWidget, QPlainTextEdit, QSpinBox, QTextEdit, QWidget,
)
from shiboken6 import isValid


def _eligible(widget, dialog, tables, excluded):
    parent = widget
    while parent is not None and parent is not dialog:
        if parent in excluded:
            return False
        if parent is not widget and isinstance(parent, (QComboBox, QAbstractSpinBox)):
            return False
        if isinstance(parent, QAbstractItemView) and parent not in tables and not isinstance(widget, QListWidget):
            return False
        parent = parent.parentWidget()
    return parent is dialog


class ParameterDefaults:
    """Snapshot only editable parameter controls, never result/data tables."""
    def __init__(self, dialog, *, tables=(), exclude=()):
        self.excluded = exclude
        self.entries = []
        self.table_items = []
        types = (QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QListWidget, QTextEdit, QPlainTextEdit)
        for widget in dialog.findChildren(QWidget):
            if not isinstance(widget, types):
                continue
            if not _eligible(widget, dialog, tables, exclude):
                continue
            if isinstance(widget, (QLineEdit, QTextEdit, QPlainTextEdit)) and widget.isReadOnly():
                continue
            if isinstance(widget, QComboBox):
                state = ("combo", deepcopy(widget.currentData()), widget.currentText())
            elif isinstance(widget, QCheckBox):
                state = ("check", widget.isChecked())
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                state = ("number", widget.value())
            elif isinstance(widget, QListWidget):
                state = ("list", [(deepcopy(widget.item(i).data(Qt.ItemDataRole.UserRole)), widget.item(i).text(), widget.item(i).checkState())
                                  for i in range(widget.count()) if widget.item(i).data(Qt.ItemDataRole.CheckStateRole) is not None])
            elif isinstance(widget, (QTextEdit, QPlainTextEdit)):
                state = ("multiline", widget.toPlainText())
            else:
                state = ("text", widget.text())
            self.entries.append((widget, state))
        for table in tables:
            self.table_items.append((table, [(r, c, table.item(r, c).text()) for r in range(table.rowCount())
                                              for c in range(table.columnCount()) if table.item(r, c) is not None]))

    def restore(self):
        widgets = [widget for widget, state in self.entries if isValid(widget)]
        tables = [table for table, items in self.table_items if isValid(table)]
        blockers = [QSignalBlocker(widget) for widget in [*widgets, *tables]]
        for widget, state in self.entries:
            if not isValid(widget):
                continue
            kind, *value = state
            if kind == "combo":
                index = widget.findData(value[0]) if value[0] is not None else widget.findText(value[1])
                if widget.isEditable():
                    if index >= 0 and widget.itemText(index) == value[1]:
                        widget.setCurrentIndex(index)
                        widget.setEditText(value[1])
                    else:
                        widget.setCurrentText(value[1])
                else:
                    widget.setCurrentIndex(max(0, index))
            elif kind == "check":
                widget.setChecked(value[0])
            elif kind == "number":
                widget.setValue(value[0])
            elif kind == "multiline":
                widget.setPlainText(value[0])
            elif kind == "list":
                for i in range(widget.count()):
                    item = widget.item(i)
                    if item.data(Qt.ItemDataRole.CheckStateRole) is None:
                        continue
                    default_checked = Qt.CheckState.Unchecked
                    for identifier, text, checked in value[0]:
                        if (identifier is not None and item.data(Qt.ItemDataRole.UserRole) == identifier) or (identifier is None and item.text() == text):
                            default_checked = checked
                            break
                    item.setCheckState(default_checked)
            else:
                widget.setText(value[0])
        for table, items in self.table_items:
            if isValid(table):
                for row, column, text in items:
                    if table.item(row, column) is not None:
                        table.item(row, column).setText(text)
        del blockers
        for widget, state in self.entries:
            if isValid(widget) and getattr(widget, '_analysis_scope', None) is not None:
                widget._analysis_scope.set_mode(widget.currentData())
