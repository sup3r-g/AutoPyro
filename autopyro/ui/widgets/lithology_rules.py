"""Edit the project's saved rock name rules without loading interval data."""

import json
from pathlib import Path

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSize, QSortFilterProxyModel, Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QHeaderView, QHBoxLayout,
    QFileDialog, QLabel, QLineEdit, QPushButton, QTableView, QVBoxLayout,
)

from autopyro.data_io.file_output import write_json
from autopyro.geology.lithology_matching import RULES_FORMAT, decode_rules
from ..i18n import message, tr
from ..theme import icon
from ..theme.controls import apply_control_icons
from .lithology import LithologyPicker, standard_lithologies
from .interval_review import IntervalReviewDialog
from .results import ResultsModel
from ..theme.controls import add_search_icon


class RuleImportModel(ResultsModel):
    def __init__(self, current, incoming, catalog, parent):
        self.incoming = incoming
        self.names = list(incoming)
        self.additions = set()
        self.review_rows = set()
        self.accepted_rows = set()
        standard = {'', *(entry.code for entry in standard_lithologies())}
        rows = []
        for index, (name, code) in enumerate(incoming.items()):
            if code not in catalog.by_code:
                status = tr('Литология недоступна: {0}', code)
            elif current.get(name) == code:
                status = tr('Дубль')
            elif code not in standard:
                status = tr('Проверьте свою породу')
                self.review_rows.add(index)
            elif name not in current:
                status = tr('Добавление')
                self.additions.add(name)
            else:
                status = tr('Конфликт')
                self.review_rows.add(index)
            old = catalog.by_code.get(current.get(name))
            new = catalog.by_code.get(code)
            rows.append((name, old.label if old else current.get(name),
                         new.label if new else code, status, ''))
        super().__init__([tr('Исходная порода'), tr('В проекте'), tr('В файле'),
                          tr('Статус'), tr('Принять из файла')], rows, parent)

    def flags(self, index):
        flags = super().flags(index)
        if index.column() == 4 and index.row() in self.review_rows:
            flags |= Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.CheckStateRole and index.column() == 4 and index.row() in self.review_rows:
            return Qt.CheckState.Checked if index.row() in self.accepted_rows else Qt.CheckState.Unchecked
        return super().data(index, role)

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if role != Qt.ItemDataRole.CheckStateRole or not self.flags(index) & Qt.ItemFlag.ItemIsUserCheckable:
            return False
        if Qt.CheckState(value) == Qt.CheckState.Checked:
            self.accepted_rows.add(index.row())
        else:
            self.accepted_rows.discard(index.row())
        self.dataChanged.emit(index, index, [role])
        return True

    def changes(self):
        names = self.additions | {self.names[row] for row in self.accepted_rows}
        return {name: self.incoming[name] for name in names}


def unique_json_keys(pairs):
    values = {}
    for key, value in pairs:
        if key in values:
            raise ValueError(tr('В файле словаря повторяется ключ: {0}', key))
        values[key] = value
    return values


class LithologyRulesModel(QAbstractTableModel):
    def __init__(self, rules, catalog, parent):
        super().__init__(parent)
        self.rows = sorted(rules.items())
        self.catalog = catalog

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 2

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return (tr('Исходная порода'), tr('Литология stratapy'))[section]
        return super().headerData(section, orientation, role)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        name, code = self.rows[index.row()]
        if index.column() == 0:
            return name if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole) else None
        entry = self.catalog.by_code.get(code)
        if role == Qt.ItemDataRole.DecorationRole and entry:
            return self.catalog.icon(entry)
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            return entry.label if entry else tr('Литология недоступна: {0}', code)

    def set_code(self, row, code):
        name, previous = self.rows[row]
        if code != previous:
            self.rows[row] = (name, code)
            index = self.index(row, 1)
            self.dataChanged.emit(index, index)

    def remove_rule(self, row):
        self.beginRemoveRows(QModelIndex(), row, row)
        del self.rows[row]
        self.endRemoveRows()


class LithologyRulesDialog(QDialog):
    def __init__(self, rules, catalog, parent=None):
        super().__init__(parent)
        self.catalog = catalog
        self.setWindowTitle(tr('Правила сопоставления пород'))
        self.resize(760, 500)
        layout = QVBoxLayout(self)
        hint = QLabel(tr('Сохранённые правила используются при следующем сопоставлении пород. Названия показаны без учёта регистра и лишних пробелов.'))
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.search = QLineEdit()
        self.search.setPlaceholderText(tr('Поиск по названию породы или коду…'))
        add_search_icon(self.search)
        layout.addWidget(self.search)
        self.model = LithologyRulesModel(rules, catalog, self)
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterKeyColumn(-1)
        self.proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.search.textChanged.connect(self.proxy.setFilterFixedString)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setIconSize(QSize(64, 40))
        self.table.verticalHeader().setDefaultSectionSize(46)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        self.summary = QLabel()
        layout.addWidget(self.summary)
        self.feedback = QLabel()
        self.feedback.setWordWrap(True)
        layout.addWidget(self.feedback)
        actions = QHBoxLayout()
        self.edit_button = QPushButton(icon('layers'), tr('Изменить литологию…'))
        self.delete_button = QPushButton(icon('trash'), tr('Удалить правило'))
        for button in (self.edit_button, self.delete_button):
            button.setAutoDefault(False)
            actions.addWidget(button)
        layout.addLayout(actions)
        exchange = QHBoxLayout()
        import_button = QPushButton(icon('import'), tr('Импорт словаря…'))
        export_button = QPushButton(icon('log-out'), tr('Экспорт словаря…'))
        for button in (import_button, export_button):
            button.setAutoDefault(False)
            exchange.addWidget(button)
        layout.addLayout(exchange)
        import_button.clicked.connect(self.import_rules)
        export_button.clicked.connect(self.export_rules)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr('Применить'))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.edit_button.clicked.connect(self.edit_rule)
        self.delete_button.clicked.connect(self.delete_rule)
        self.table.doubleClicked.connect(self.edit_rule)
        self.table.selectionModel().selectionChanged.connect(self.update_actions)
        self.proxy.rowsInserted.connect(self.update_actions)
        self.proxy.rowsRemoved.connect(self.update_actions)
        self.update_actions()
        apply_control_icons(self)

    def selected_row(self):
        rows = self.table.selectionModel().selectedRows()
        return self.proxy.mapToSource(rows[0]).row() if rows else -1

    def update_actions(self):
        selected = self.selected_row() >= 0
        self.edit_button.setEnabled(selected)
        self.delete_button.setEnabled(selected)
        self.summary.setText(tr('Показано правил: {0} из {1}', self.proxy.rowCount(), self.model.rowCount()))

    def edit_rule(self):
        row = self.selected_row()
        if row >= 0:
            _, code = self.model.rows[row]
            picker = LithologyPicker(self.catalog, current=code, parent=self)
            if picker.exec() == picker.DialogCode.Accepted:
                self.model.set_code(row, picker.selected_code)

    def delete_rule(self):
        row = self.selected_row()
        if row >= 0:
            self.model.remove_rule(row)
            self.update_actions()

    def rules(self):
        return dict(self.model.rows)

    def import_rules(self):
        path, _ = QFileDialog.getOpenFileName(self, tr('Импорт словаря…'), '', 'JSON (*.json)')
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique_json_keys)
            incoming = decode_rules(payload)
            current = self.rules()
            dialog = IntervalReviewDialog(
                self, tr('Проверка словаря пород'),
                tr('Новые правила для стандартных пород добавятся автоматически. Отметьте нужные замены '
                   'и проверьте свои породы: используются определения текущего проекта. '
                   'Дубли и недоступные литологии пропускаются.'), [], [])
            model = RuleImportModel(current, incoming, self.catalog, dialog.table)
            dialog.table.setModel(model)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr('Добавить в список'))
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            dialog.layout_box.addWidget(buttons)
            apply_control_icons(dialog)
            accepted = dialog.exec() == dialog.DialogCode.Accepted
            changes = model.changes() if accepted else {}
            dialog.deleteLater()
            if accepted:
                if changes:
                    self.model.beginResetModel()
                    self.model.rows = sorted({**current, **changes}.items())
                    self.model.endResetModel()
                    self.update_actions()
                self.feedback.setText(tr('Правил добавлено или обновлено: {0}', len(changes)))
        except (OSError, ValueError) as exc:
            self.feedback.setText(tr('Ошибка: {0}', message(exc)))

    def export_rules(self):
        path, _ = QFileDialog.getSaveFileName(self, tr('Экспорт словаря…'), 'rock-rules.json', 'JSON (*.json)')
        if not path:
            return
        try:
            write_json(path, {'format': RULES_FORMAT, 'version': 1, 'rules': self.rules()})
            self.feedback.setText(tr('Словарь пород экспортирован.'))
        except (OSError, ValueError) as exc:
            self.feedback.setText(tr('Ошибка: {0}', message(exc)))
