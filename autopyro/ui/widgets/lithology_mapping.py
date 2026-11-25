"""Review unique imported names once, using the existing lazy pattern catalog."""

from copy import deepcopy

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSize, Qt
from PySide6.QtGui import QUndoCommand
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QHeaderView, QLabel, QPushButton,
    QTableView, QVBoxLayout, QAbstractItemView,
)

from autopyro.geology.lithology_matching import ORIGINAL_SOURCE, LithologyMapping, default_source, source_columns
from ..i18n import tr
from ..theme import icon
from .interval_history import PREVIOUS_TEXT
from .lithology import LithologyPicker


class MappingModel(QAbstractTableModel):
    def __init__(self, mapping, catalog, parent=None):
        super().__init__(parent)
        self.mapping, self.catalog = mapping, catalog
        self.rows = sorted(mapping.groups.values(), key=lambda group: group.code is not None)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 4

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Vertical:
            return str(section + 1)
        return [tr('Исходная порода'), tr('Интервалов'), tr('Литология stratapy'), tr('Сопоставление')][section]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        group = self.rows[index.row()]
        entry = self.catalog.resolve(group.code) if group.code is not None else None
        if role == Qt.ItemDataRole.DecorationRole and index.column() == 2 and entry:
            return self.catalog.icon(entry)
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            reason = {'code': tr('Код stratapy'), 'name': tr('Название из каталога'),
                      'alias': tr('Известное название'), 'saved': tr('Правило проекта'),
                      'ambiguous': tr('Несколько вариантов'), 'unknown': tr('Не распознано'),
                      'manual': tr('Выбрано вручную')}[group.reason]
            return [group.value, group.count, entry.label if entry else tr('Не назначена'), reason][index.column()]


class LithologyMappingDialog(QDialog):
    def __init__(self, scheme, catalog, matcher, parent=None):
        super().__init__(parent)
        self.scheme, self.catalog, self.matcher = scheme, catalog, matcher
        self.setWindowTitle(tr('Сопоставление названий пород'))
        self.resize(980, 620)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(tr('Колонка с исходными названиями пород')))
        self.source = QComboBox()
        self.source.addItem(tr('Текущая литология интервалов'), None)
        if any('lithology_source' in item for item in scheme['intervals']):
            self.source.addItem(tr('Исходные названия до сопоставления'), ORIGINAL_SOURCE)
        columns = source_columns(scheme)
        for column in columns:
            self.source.addItem(column, column)
        self.source.setCurrentIndex(max(0, self.source.findData(default_source(scheme, columns))))
        layout.addWidget(self.source)
        hint = QLabel(tr('Одно соответствие применяется ко всем интервалам с таким названием. Проверьте автоматические назначения. Неопознанные значения останутся без литологии; исходные названия сохранятся.'))
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.table = QTableView()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setIconSize(QSize(64, 40))
        self.table.verticalHeader().setDefaultSectionSize(46)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)
        self.choose_button = QPushButton(icon('layers'), tr('Выбрать литологию для названия…'))
        self.choose_button.clicked.connect(self.choose_lithology)
        self.choose_button.setAutoDefault(False)
        layout.addWidget(self.choose_button)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr('Применить'))
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.source.currentIndexChanged.connect(self.rebuild)
        self.table.doubleClicked.connect(lambda index: self.choose_lithology())
        self.rebuild()
        from ..theme.controls import apply_control_icons
        apply_control_icons(self)

    def rebuild(self):
        self.mapping = LithologyMapping(self.scheme, self.matcher, self.source.currentData())
        old = self.table.model()
        self.model = MappingModel(self.mapping, self.catalog, self)
        self.table.setModel(self.model)
        if old is not None:
            old.deleteLater()
        for column, width in enumerate((235, 90, 380, 190)):
            self.table.setColumnWidth(column, width)
        self.table.selectionModel().currentChanged.connect(
            lambda *args: self.choose_button.setEnabled(self.table.currentIndex().isValid()))
        self.update_summary()

    def update_summary(self):
        missing = sum(group.count for group in self.model.rows if group.code is None)
        self.summary.setText(tr('Названий: {0}; интервалов без соответствия: {1}', len(self.model.rows), missing))
        self.choose_button.setEnabled(self.table.currentIndex().isValid())

    def choose_lithology(self):
        row = self.table.currentIndex().row()
        if row < 0:
            return
        group = self.model.rows[row]
        picker = LithologyPicker(self.catalog, current=group.code or '', parent=self)
        if picker.exec() == picker.DialogCode.Accepted:
            group.code, group.reason = picker.selected_code, 'manual'
            self.model.dataChanged.emit(self.model.index(row, 2), self.model.index(row, 3))
            self.update_summary()


class MapDraftLithologies(QUndoCommand):
    """One undo step for codes and their small provenance fields, without row snapshots."""
    SOURCE_KEYS = ('lithology_source', 'lithology_source_field')

    def __init__(self, history, before, after):
        super().__init__(tr('Сопоставление названий пород'))
        self.history = history
        self.changes = []
        for old, new in zip(before['intervals'], after['intervals']):
            old_source = {key: old[key] for key in self.SOURCE_KEYS if key in old}
            new_source = {key: new[key] for key in self.SOURCE_KEYS if key in new}
            if old.get('lithology', '') != new.get('lithology', '') or old_source != new_source:
                self.changes.append((old['id'], old.get('lithology', ''), new.get('lithology', ''),
                                     old_source, new_source))

    def apply(self, after):
        with self.history.suspended():
            for key, old, new, old_source, new_source in self.changes:
                row = self.history.rows_by_id[key]
                value = new if after else old
                cell = self.history.table.item(row, 7)
                cell.setText(value)
                cell.setData(PREVIOUS_TEXT, value)
                identity = self.history.table.item(row, 6)
                metadata = identity.data(Qt.ItemDataRole.UserRole)
                for source_key in self.SOURCE_KEYS:
                    metadata.pop(source_key, None)
                metadata.update(deepcopy(new_source if after else old_source))
                identity.setData(Qt.ItemDataRole.UserRole, metadata)
        self.history.owner.invalidate_column()

    def redo(self):
        self.apply(True)

    def undo(self):
        self.apply(False)
