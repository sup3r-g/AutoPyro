"""Read-only previews for interval checks and imports."""

from PySide6.QtCore import QSortFilterProxyModel, Qt
from PySide6.QtWidgets import QAbstractItemView, QDialog, QHeaderView, QLabel, QLineEdit, QTableView, QVBoxLayout

from ..i18n import tr
from .results import ResultsModel
from ..theme.controls import add_search_icon


class IntervalReviewDialog(QDialog):
    def __init__(self, parent, title, summary, headers, rows):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(850, 490)
        self.layout_box = QVBoxLayout(self)
        self.summary = QLabel(summary)
        self.summary.setWordWrap(True)
        self.layout_box.addWidget(self.summary)
        self.table = QTableView()
        self.table.setModel(ResultsModel(headers, rows, self.table))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.layout_box.addWidget(self.table, 1)

    def add_search(self):
        self.search = QLineEdit()
        self.search.setPlaceholderText(tr('Поиск…'))
        add_search_icon(self.search)
        self.layout_box.insertWidget(1, self.search)
        self.proxy = QSortFilterProxyModel(self.table)
        self.proxy.setSourceModel(self.table.model())
        self.proxy.setFilterKeyColumn(-1)
        self.proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.table.setModel(self.proxy)
        self.search.textChanged.connect(self.proxy.setFilterFixedString)

    def source_row(self):
        index = self.table.currentIndex()
        return self.proxy.mapToSource(index).row() if hasattr(self, 'proxy') else index.row()
