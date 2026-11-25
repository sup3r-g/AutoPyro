"""Read-only preview of the exact changes held by an undo command."""

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Slot
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHeaderView, QHBoxLayout, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QTableView, QVBoxLayout,
)
from shiboken6 import isValid

from ..i18n import tr
from ..table.selection import sample_label_column
from ..tasks import TaskCancelled, run_with_progress


class BatchEditPreviewModel(QAbstractTableModel):
    def __init__(self, command, parent=None):
        super().__init__(parent)
        project = command.model.project
        self.changes = command.changes
        self.positions = command.row_positions
        self.rows = project.rows
        self.columns = {column.id: column.name for column in project.columns}
        self.sample_column = sample_label_column(project.columns)
        self.headers = (tr('Проба'), tr('Строка проекта'), tr('Колонка'), tr('До'), tr('После'))
        self.empty = tr('Пусто')

    def sample_name(self, identifier):
        return self.rows[self.positions[identifier]].values.get(self.sample_column)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.changes)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.headers)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if (orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole
                and 0 <= section < len(self.headers)):
            return self.headers[section]
        return super().headerData(section, orientation, role)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            return None
        rid, cid, old, new = self.changes[index.row()]
        column = index.column()
        if column == 0:
            value = self.sample_name(rid)
            if role == Qt.ItemDataRole.ToolTipRole:
                return f"{value}\n{rid}" if value not in (None, "") else rid
        elif column == 1:
            value = self.positions[rid] + 1
        elif column == 2:
            value = self.columns[cid]
        else:
            value = old if column == 3 else new
        if value is None or value == "":
            return self.empty if role == Qt.ItemDataRole.ToolTipRole else "—"
        return str(value)


class BatchEditPreview(QDialog):
    def __init__(self, title, summary, command, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(860, 480)
        layout = QVBoxLayout(self)
        self.summary = QLabel(summary)
        layout.addWidget(self.summary)
        search_layout = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText(tr('Имя пробы или номер строки'))
        self.search.setAccessibleName(self.search.placeholderText())
        self.search.returnPressed.connect(self.find_next)
        self.search_button = QPushButton(tr('Найти далее'))
        self.search_button.setAutoDefault(False)
        self.search_button.clicked.connect(self.find_next)
        search_layout.addWidget(self.search)
        search_layout.addWidget(self.search_button)
        layout.addLayout(search_layout)
        self.search_status = QLabel()
        layout.addWidget(self.search_status)
        self.search.textChanged.connect(self.search_status.clear)
        self.table = QTableView()
        self.table.setAccessibleName(tr('Изменения'))
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.model = BatchEditPreviewModel(command, self.table)
        self.table.setModel(self.model)
        for column, width in enumerate((145, 115, 170, 180)):
            self.table.setColumnWidth(column, width)
        layout.addWidget(self.table)
        self.cell_value = QPlainTextEdit()
        self.cell_value.setReadOnly(True)
        self.cell_value.setMaximumHeight(75)
        self.cell_value.setAccessibleName(tr('Полное значение выбранной ячейки'))
        self.cell_value.setPlaceholderText(self.cell_value.accessibleName())
        self.table.selectionModel().currentChanged.connect(self.show_cell_value)
        layout.addWidget(self.cell_value)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        self.apply_button = buttons.button(QDialogButtonBox.StandardButton.Apply)
        self.apply_button.setText(tr('Применить'))
        self.apply_button.clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        cancel = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        cancel.setDefault(True)
        cancel.setFocus()
        self.source_model = command.model
        self.source_revision = self.source_model.data_revision
        self.stale = False
        self.source_model.changed.connect(self.check_source)
        self.source_model.modelAboutToBeReset.connect(self.invalidate_preview)
        self.source_model.columnsAboutToBeRemoved.connect(self.invalidate_preview)
        self.source_model.destroyed.connect(self.invalidate_preview)

    def show_cell_value(self, current, _previous):
        self.cell_value.setPlainText(current.data() or "")

    def find_next(self):
        query = self.search.text().strip().casefold()
        if not query or self.stale:
            return
        count = self.model.rowCount()
        start = self.table.currentIndex().row() + 1

        def scan():
            for offset in range(count):
                if self.stale:
                    return None
                row = (start + offset) % count
                identifier = self.model.changes[row][0]
                name = self.model.sample_name(identifier)
                if ((name is not None and query in str(name).casefold())
                        or query == str(self.model.positions[identifier] + 1)):
                    return row
                if count >= 10_000 and offset % 2048 == 2047:
                    yield
            return None

        steps = scan()
        try:
            next(steps)
            row = run_with_progress(self, tr('Поиск в предпросмотре…'), lambda update, result: result,
                                    cancellable=True, prepare=steps)
        except StopIteration as done:
            row = done.value
        except TaskCancelled:
            return
        if self.stale:
            return
        if row is None:
            self.search_status.setText(tr('Совпадений нет.'))
            return
        self.search_status.clear()
        index = self.model.index(row, 0)
        self.table.setCurrentIndex(index)
        self.table.scrollTo(index)

    def check_source(self):
        if self.stale:
            return
        owner = self.parentWidget()
        replaced = owner is not None and getattr(owner, "model", self.source_model) is not self.source_model
        if replaced or self.source_model.data_revision != self.source_revision:
            self.invalidate_preview()

    @Slot()
    def invalidate_preview(self):
        if self.stale or not isValid(self.apply_button):
            return
        self.stale = True
        self.apply_button.setEnabled(False)
        self.search.setEnabled(False)
        self.search_button.setEnabled(False)
        self.search_status.clear()
        self.cell_value.clear()
        self.summary.setText(tr('Таблица изменилась. Повторите операцию.'))
        self.summary.setWordWrap(True)
        self.model.beginResetModel()
        self.model.changes = []
        self.model.positions = {}
        self.model.rows = ()
        self.model.columns = {}
        self.model.endResetModel()

    def accept(self):
        self.check_source()
        if not self.stale:
            super().accept()
