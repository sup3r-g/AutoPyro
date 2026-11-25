"""Inspect saved sample-set membership against the current project table."""

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QTableView, QVBoxLayout,
)

from ..i18n import tr
from ..table import FilterProxy


class SampleSetComparisonDialog(QDialog):
    def __init__(self, owner, sample_sets):
        super().__init__(owner)
        self.setWindowTitle(tr('Сравнить наборы проб…'))
        self.resize(940, 570)
        self.sample_sets = sample_sets

        layout = QVBoxLayout(self)
        controls = QFormLayout()
        layout.addLayout(controls)
        names = sorted(sample_sets)
        self.first, self.second = QComboBox(), QComboBox()
        for label, combo in ((tr('Первый набор'), self.first), (tr('Второй набор'), self.second)):
            combo.addItems(names)
            controls.addRow(label, combo)
        self.second.setCurrentIndex(1)
        self.group = QComboBox()
        controls.addRow(tr('Группа проб'), self.group)

        self.table = QTableView()
        self.table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)
        self.status = QLabel()
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.proxy = FilterProxy(self)
        self.proxy.setSourceModel(owner.model)
        self.first.currentIndexChanged.connect(self.compare)
        self.second.currentIndexChanged.connect(self.compare)
        self.group.currentIndexChanged.connect(self.show_group)
        self.compare()
        self.table.setModel(self.proxy)

        source_header = owner.table.horizontalHeader()
        header = self.table.horizontalHeader()
        for visual in range(source_header.count()):
            column = source_header.logicalIndex(visual)
            header.moveSection(header.visualIndex(column), visual)
            self.table.setColumnHidden(column, owner.table.isColumnHidden(column))
            if not self.table.isColumnHidden(column):
                self.table.setColumnWidth(column, owner.table.columnWidth(column))

    def compare(self):
        first = frozenset(self.sample_sets[self.first.currentText()])
        second = frozenset(self.sample_sets[self.second.currentText()])
        self.groups = (first & second, first - second, second - first)
        current = max(0, self.group.currentIndex())
        with QSignalBlocker(self.group):
            self.group.clear()
            common, only_first, only_second = self.groups
            self.group.addItems([
                tr('Общие пробы ({0:,})', len(common)),
                tr('Только в первом наборе ({0:,})', len(only_first)),
                tr('Только во втором наборе ({0:,})', len(only_second)),
            ])
            self.group.setCurrentIndex(current)
        self.show_group()

    def show_group(self):
        members = self.groups[self.group.currentIndex()]
        self.proxy.set_rules([], row_ids=members)
        visible = self.proxy.rowCount()
        self.status.setText(tr('Проб в таблице: {0:,}. Отсутствуют в проекте: {1:,}.',
                               visible, len(members) - visible))
