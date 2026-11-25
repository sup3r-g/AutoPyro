from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QComboBox, QStyledItemDelegate, QTableView


class _FieldDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        source = index
        while hasattr(source.model(), "mapToSource"):
            source = source.model().mapToSource(source)
        model = source.model()
        if model.project.columns[source.column()].kind == "category":
            column_id = model.project.columns[source.column()].id
            editor = QComboBox(parent)
            editor.setEditable(True)
            editor.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            values = {row.values[column_id] for row in model.project.rows if row.values[column_id] is not None}
            editor.addItem("")
            editor.addItems(sorted(values, key=str.casefold))
            return editor
        return super().createEditor(parent, option, index)

    def setEditorData(self, editor, index):
        if isinstance(editor, QComboBox):
            editor.setEditText(index.data(Qt.ItemDataRole.EditRole))
        else:
            super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if isinstance(editor, QComboBox):
            model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)
        else:
            super().setModelData(editor, model, index)


class _PrecisionDelegate(_FieldDelegate):
    def __init__(self, parent):
        super().__init__(parent)
        self.precision = {}

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        digits = self.precision.get(index.column())
        if digits is None:
            return
        value = index.data(Qt.ItemDataRole.UserRole)
        if type(value) is float:
            option.text = f"{value:.{digits}f}"
        elif type(value) is int and digits:
            # Fixed-point float formatting would round large integer values.
            option.text = f"{value}." + "0" * digits


class PinnedTable(QTableView):
    def __init__(self):
        super().__init__()
        self.pinned_columns = ()
        self._arranging_pins = False
        self.frozen = QTableView(self)
        self.setHorizontalScrollMode(QTableView.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollMode(QTableView.ScrollMode.ScrollPerPixel)

        self.frozen.setFrameShape(QTableView.Shape.NoFrame)
        self.frozen.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.frozen.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.frozen.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.frozen.setVerticalScrollMode(self.verticalScrollMode())
        self.frozen.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self.frozen.setAlternatingRowColors(True)
        self.frozen.verticalHeader().hide()
        self.frozen.hide()
        self.viewport().stackUnder(self.frozen)

        header = self.horizontalHeader()
        header.sectionMoved.connect(self._sync_frozen_columns)
        header.sectionResized.connect(self._sync_frozen_width)
        self.frozen.horizontalHeader().sectionResized.connect(self._resize_pinned_column)
        header.geometriesChanged.connect(self._position_frozen)
        header.sortIndicatorChanged.connect(
            lambda section, order: self.frozen.horizontalHeader().setSortIndicator(section, order)
        )
        self.frozen.horizontalHeader().setSortIndicatorShown(True)
        self.frozen.horizontalHeader().setSectionsClickable(True)
        self.frozen.horizontalHeader().sectionClicked.connect(self._sort_pinned_section)
        self.verticalHeader().sectionResized.connect(
            lambda row, _, height: self.frozen.setRowHeight(row, height)
        )
        self.verticalScrollBar().valueChanged.connect(self.frozen.verticalScrollBar().setValue)
        self.frozen.verticalScrollBar().valueChanged.connect(self._scroll_from_frozen)

    def setModel(self, model):
        self.set_pinned_columns(())
        super().setModel(model)

    def _sort_pinned_section(self, section):
        if not self.isSortingEnabled():
            self.horizontalHeader().sectionClicked.emit(section)
            return
        header = self.horizontalHeader()
        descending = (header.sortIndicatorSection() == section
                      and header.sortIndicatorOrder() == Qt.SortOrder.AscendingOrder)
        order = Qt.SortOrder.DescendingOrder if descending else Qt.SortOrder.AscendingOrder
        self.sortByColumn(section, order)

    def setColumnHidden(self, column, hide):
        super().setColumnHidden(column, hide)
        if column in self.pinned_columns:
            self._sync_frozen_columns()

    def set_pinned_columns(self, columns):
        self.pinned_columns = tuple(columns)
        self._sync_frozen_columns()

    def _sync_frozen_columns(self, *_):
        """Keep pinned columns at the leading edge and mirror them in the overlay."""
        if self._arranging_pins:
            return
        header = self.horizontalHeader()
        pins = [column for column in self.pinned_columns if column < header.count()]
        visible = {column for column in pins if not self.isColumnHidden(column)}
        if not visible:
            self.frozen.hide()
            if not pins and self.frozen.model() is not None:
                self.frozen.setModel(None)
            return
        self._arranging_pins = True
        try:
            if self.frozen.model() is not self.model():
                self.frozen.setModel(self.model())
                own_selection = self.frozen.selectionModel()
                self.frozen.setSelectionModel(self.selectionModel())
                own_selection.deleteLater()
                self.frozen.verticalHeader().restoreState(self.verticalHeader().saveState())
            frozen_header = self.frozen.horizontalHeader()
            frozen_header.setSortIndicator(header.sortIndicatorSection(), header.sortIndicatorOrder())
            for target, column in enumerate(pins):
                header.moveSection(header.visualIndex(column), target)
                frozen_header.moveSection(frozen_header.visualIndex(column), target)
            for column in range(header.count()):
                self.frozen.setColumnHidden(column, column not in visible)
            for column in visible:
                self.frozen.setColumnWidth(column, self.columnWidth(column))
        finally:
            self._arranging_pins = False
        self._position_frozen()
        self.frozen.show()
        self.frozen.verticalScrollBar().setValue(self.verticalScrollBar().value())

    def _sync_frozen_width(self, column, _, width):
        if column in self.pinned_columns and width > 0:
            self.frozen.setColumnWidth(column, width)
            self._position_frozen()

    def _resize_pinned_column(self, column, _, width):
        if not self._arranging_pins and column in self.pinned_columns and width > 0:
            self.setColumnWidth(column, width)

    def _position_frozen(self):
        if not self.pinned_columns:
            return
        viewport = self.viewport().geometry()
        header = self.horizontalHeader().geometry()
        visible = [column for column in self.pinned_columns if not self.isColumnHidden(column)]
        width = sum(self.columnWidth(column) for column in visible)
        unpinned = len(visible) < self.horizontalHeader().count() - self.horizontalHeader().hiddenSectionCount()
        width = min(width, max(1, viewport.width() - (48 if unpinned else 0)))
        x = viewport.right() + 1 - width if self.isRightToLeft() else viewport.x()
        self.frozen.setGeometry(x, header.y(), width,
                                viewport.height() + header.height())

    def _scroll_from_frozen(self, value):
        if not self.frozen.isHidden():
            self.verticalScrollBar().setValue(value)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_frozen()

    def scrollTo(self, index, hint=QTableView.ScrollHint.EnsureVisible):
        if not self.frozen.isHidden() and index.column() in self.pinned_columns:
            self.frozen.scrollTo(index, hint)
            return
        super().scrollTo(index, hint)
        if self.frozen.isVisible() and index.isValid():
            left = self.columnViewportPosition(index.column())
            if self.isRightToLeft():
                distance = self.viewport().width() - left - self.columnWidth(index.column())
            else:
                distance = left
            if distance < self.frozen.width():
                scrollbar = self.horizontalScrollBar()
                scrollbar.setValue(scrollbar.value() + distance - self.frozen.width())


class DataTable(PinnedTable):
    def __init__(self):
        super().__init__()
        self.horizontalHeader().setSectionsClickable(True)
        self.horizontalHeader().setSortIndicatorShown(True)
        self.setItemDelegate(_FieldDelegate(self))
        self.frozen.setItemDelegate(_FieldDelegate(self.frozen))
        self._precision_delegate = _PrecisionDelegate(self)
        self._frozen_precision_delegate = _PrecisionDelegate(self.frozen)

    def _sort_pinned_section(self, section):
        self.horizontalHeader().sectionClicked.emit(section)

    def sortByColumn(self, column, order):
        self.model().set_sort_rules([], column, order)

    def clearSelection(self):
        self.window().sample_selection.clear()

    def update_sort_indicator(self):
        proxy = self.model()
        column, order = proxy.sortColumn(), proxy.sortOrder()
        if proxy.sort_rules:
            primary, descending = proxy.sort_rules[0]
            columns = proxy.sourceModel().project.columns
            column = next((i for i, item in enumerate(columns) if item.id == primary), -1)
            descending = bool(descending) != (order == Qt.SortOrder.DescendingOrder)
            order = Qt.SortOrder.DescendingOrder if descending else Qt.SortOrder.AscendingOrder
        self.horizontalHeader().setSortIndicator(column, order)

    def set_display_precision(self, precision):
        for view, delegate in ((self, self._precision_delegate), (self.frozen, self._frozen_precision_delegate)):
            old = delegate.precision
            if old == precision:
                continue
            delegate.precision = precision
            for column in old.keys() - precision.keys():
                view.setItemDelegateForColumn(column, None)
            for column in precision.keys() - old.keys():
                view.setItemDelegateForColumn(column, delegate)
            view.viewport().update()

    def edit(self, index, trigger=None, event=None):
        # Keyboard editing starts in the main view even if the cell is drawn
        # by the overlay. Open the editor in the view that displays the cell.
        if not self.frozen.isHidden() and index.column() in self.pinned_columns:
            view = self.frozen
        else:
            view = super()
        if trigger is None:
            return view.edit(index)
        return view.edit(index, trigger, event)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Copy):
            self.window().copy_cells()
        elif event.matches(QKeySequence.StandardKey.Paste):
            self.window().paste_cells()
        elif event.key() == Qt.Key.Key_Escape:
            self.clearSelection()
            event.accept()
        else:
            super().keyPressEvent(event)
