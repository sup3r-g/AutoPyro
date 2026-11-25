"""Table editor for the local bounds of one series at a time."""

import math
from copy import deepcopy

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QHeaderView, QLabel, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout)

from ..i18n import tr
from ..theme import icon
from ..theme.controls import apply_control_icons


class JointBoundsDialog(QDialog):
    def __init__(self, parent, labels, defaults, overrides, notice=""):
        super().__init__(parent)
        self.setWindowTitle(tr("Границы по сериям"))
        self.setWindowIcon(icon("settings-2"))
        self.resize(600, 320)
        self.overrides = deepcopy(overrides)
        self.defaults = defaults
        layout = QVBoxLayout(self)
        self.series = QComboBox()
        self.series.setAccessibleName(tr("Серия"))
        self.series.addItems(list(dict.fromkeys([*labels, *overrides])))
        layout.addWidget(self.series)
        note = QLabel(tr("Снимите флажок, чтобы использовать общие границы. Пустая ячейка при переопределении означает отсутствие границы."))
        note.setWordWrap(True)
        layout.addWidget(note)
        if notice:
            warning = QLabel(notice)
            warning.setWordWrap(True)
            layout.addWidget(warning)
        self.table = QTableWidget(len(defaults), 4)
        self.table.setAccessibleName(tr("Границы по сериям"))
        self.table.setHorizontalHeaderLabels([tr("Параметр"), tr("Переопределить"),
                                              tr("Нижняя граница"), tr("Верхняя граница")])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        self.error = QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        reset = QPushButton(icon("rotate-ccw"), tr("Сбросить границы всех серий"))
        reset.clicked.connect(self.reset)
        layout.addWidget(reset)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.index = self.series.currentIndex()
        self.load_series()
        self.series.currentIndexChanged.connect(self.change_series)
        self.table.itemChanged.connect(self.toggle_override)
        apply_control_icons(self)

    def load_series(self):
        overrides = self.overrides.get(self.series.itemText(self.index), {})
        with QSignalBlocker(self.table):
            for row, (name, default) in enumerate(self.defaults.items()):
                item = QTableWidgetItem(name)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, 0, item)
                enabled = name in overrides
                check = QTableWidgetItem()
                check.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable)
                check.setCheckState(Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked)
                self.table.setItem(row, 1, check)
                for col, value in enumerate(overrides.get(name, default), 2):
                    text = ("" if enabled else "∞") if value is None else str(value)
                    item = QTableWidgetItem(text)
                    item.setToolTip(tr("По умолчанию: {0}", default[col-2] if default[col-2] is not None else "∞"))
                    if not enabled:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.table.setItem(row, col, item)

    def toggle_override(self, item):
        if item.column() != 1:
            return
        enabled = item.checkState() == Qt.CheckState.Checked
        with QSignalBlocker(self.table):
            for col, default in enumerate(self.defaults[self.table.item(item.row(), 0).text()], 2):
                cell = self.table.item(item.row(), col)
                if enabled:
                    if default is None:
                        cell.setText("")
                    cell.setFlags(cell.flags() | Qt.ItemFlag.ItemIsEditable)
                else:
                    cell.setText("∞" if default is None else str(default))
                    cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)

    def save_series(self):
        changes = {}
        for row, name in enumerate(self.defaults):
            if self.table.item(row, 1).checkState() != Qt.CheckState.Checked:
                continue
            try:
                texts = [self.table.item(row, col).text().strip() for col in (2, 3)]
                lower, upper = (float(text) if text else None for text in texts)
                if (any(value is not None and not math.isfinite(value) for value in (lower, upper))
                        or lower is not None and upper is not None and lower >= upper):
                    raise ValueError
            except ValueError:
                self.error.setText(tr("Некорректные границы коэффициента {0}.", name))
                return False
            changes[name] = [lower, upper]
        label = self.series.itemText(self.index)
        if changes:
            self.overrides[label] = changes
        else:
            self.overrides.pop(label, None)
        self.error.clear()
        return True

    def change_series(self, index):
        if not self.save_series():
            with QSignalBlocker(self.series):
                self.series.setCurrentIndex(self.index)
            return
        self.index = index
        self.load_series()

    def reset(self):
        self.overrides.clear()
        self.error.clear()
        self.load_series()

    def accept(self):
        if self.save_series():
            super().accept()
