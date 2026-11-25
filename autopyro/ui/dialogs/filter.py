from __future__ import annotations
from copy import deepcopy

from ..i18n import tr, message

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..table.filter_rules import OPERATORS, VALUELESS_OPERATORS, validated_rules
from ..widgets.parameters import add_parameter_tools
from ..theme import icon


class FilterDialog(QDialog):
    def __init__(self, project, rules, combine, parent=None):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle(
            tr('Фильтр · условия применяются к значениям, не к форматированию')
        )
        self.resize(760, 400)
        layout = QVBoxLayout(self)
        self.mode = QComboBox()
        self.mode.addItems(["AND", "OR"])
        self.mode.setCurrentText(combine)
        layout.addWidget(self.mode)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels([tr('Колонка'), tr('Условие'), tr('Значение'), "NOT"])
        layout.addWidget(self.table)
        add = QPushButton(icon('plus'), tr('Добавить условие'))
        add.clicked.connect(lambda: self.add_rule())
        layout.addWidget(add)
        add_group = QPushButton(icon('list-plus'), tr('Добавить группу условий'))
        add_group.clicked.connect(lambda: self.add_rule({"rules": [], "combine": "AND"}))
        layout.addWidget(add_group)
        remove = QPushButton(icon('trash'), tr('Удалить выбранное условие'))
        remove.clicked.connect(
            lambda: (
                self.table.removeRow(self.table.currentRow())
                if self.table.currentRow() >= 0
                else None
            )
        )
        layout.addWidget(remove)
        for rule in rules:
            self.add_rule(rule)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        add_parameter_tools(self, reset=self.restore_parameters)

    def restore_parameters(self):
        self.table.setRowCount(0)
        self.mode.setCurrentText("AND")

    def add_rule(self, rule=None):
        rule = rule or {}
        r = self.table.rowCount()
        self.table.insertRow(r)
        if "rules" in rule:
            group = QPushButton(icon('sliders-horizontal'), tr('Изменить группу условий…'))
            group.rules = deepcopy(rule["rules"])
            group.clicked.connect(lambda: self.edit_group(group))
            self.table.setCellWidget(r, 0, group)
            combine = QComboBox()
            combine.addItems(["AND", "OR"])
            combine.setCurrentText(rule.get("combine", "AND"))
            self.table.setCellWidget(r, 1, combine)
            negate = QCheckBox()
            negate.setChecked(rule.get("negate", False))
            self.table.setCellWidget(r, 3, negate)
            return
        columns = QComboBox()
        for column in self.project.columns:
            columns.addItem(column.name, column.id)
        if rule.get("column"):
            columns.setCurrentIndex(columns.findData(rule["column"]))
        operator = QComboBox()
        for label, value in OPERATORS:
            operator.addItem(tr(label), value)
        operator.setCurrentIndex(max(0, operator.findData(rule.get("operator"))))
        self.table.setCellWidget(r, 0, columns)
        self.table.setCellWidget(r, 1, operator)
        self.table.setItem(r, 2, QTableWidgetItem(str(rule.get("value", ""))))
        negate = QCheckBox()
        negate.setChecked(rule.get("negate", False))
        self.table.setCellWidget(r, 3, negate)

    def edit_group(self, button):
        row = next(index for index in range(self.table.rowCount())
                   if self.table.cellWidget(index, 0) is button)
        mode = self.table.cellWidget(row, 1)
        editor = FilterDialog(self.project, button.rules, mode.currentText(), self)
        if editor.exec():
            button.rules = deepcopy(editor.rules)
            mode.setCurrentText(editor.mode.currentText())
        editor.deleteLater()

    def validate(self):
        try:
            drafts = []
            for r in range(self.table.rowCount()):
                field = self.table.cellWidget(r, 0)
                negate = self.table.cellWidget(r, 3).isChecked()
                if isinstance(field, QPushButton):
                    drafts.append({"rules": deepcopy(field.rules),
                                   "combine": self.table.cellWidget(r, 1).currentText(), "negate": negate})
                else:
                    drafts.append({
                        "column": self.table.cellWidget(r, 0).currentData(),
                        "operator": self.table.cellWidget(r, 1).currentData(),
                        "value": self.table.item(r, 2).text(),
                        "negate": negate,
                    })
            rules = validated_rules(self.project, drafts)
        except ValueError as exc:
            QMessageBox.warning(self, tr('Фильтр'), message(exc))
            return
        self.rules = rules
        self.accept()
