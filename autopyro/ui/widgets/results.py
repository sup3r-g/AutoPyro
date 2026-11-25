from __future__ import annotations

from numbers import Real

from ..i18n import tr
from autopyro.model.sample_labels import sample_label, sample_label_fields
from ..theme import icon

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt


STATUS_LABELS = {
    "ok": "OK",
    "constant": "Постоянный показатель",
    "insufficient_pairs": "Недостаточно пар",
    "numerical_warning": "Численная неустойчивость",
    "incompatible_context": "Разные единицы или основания измерения",
}


def sample_labels(project, row_ids=None):
    steps = sample_label_steps(project, row_ids)
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value


def sample_label_steps(project, row_ids=None, *, chunk_size=512):
    fields = sample_label_fields((column.id, column.name) for column in project.columns)
    selected = None if row_ids is None else set(row_ids)
    labels = {}
    for index, row in enumerate(project.rows):
        if index and index % chunk_size == 0:
            yield
        if selected is None or row.id in selected:
            labels[row.id] = sample_label(row.id, row.values, fields)
    return labels


class ResultsModel(QAbstractTableModel):
    def __init__(self, headers, rows, parent=None):
        super().__init__(parent)
        self.headers, self.rows = headers, rows

    def set_rows(self, rows, *, headers=None):
        self.beginResetModel()
        self.rows = rows
        if headers is not None:
            self.headers = headers
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.headers)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self.rows[index.row()][index.column()]
        if role == Qt.ItemDataRole.DecorationRole and self.headers[index.column()] in ('Статус', tr('Статус')):
            names = {'ok': 'check', 'constant': 'circle-question-mark',
                     'insufficient_pairs': 'circle-question-mark', 'numerical_warning': 'circle-question-mark'}
            if isinstance(value, str) and value in names:
                return icon(names[value])
        if role == Qt.ItemDataRole.DisplayRole:
            if value is None:
                return "—"
            if isinstance(value, float):
                return f"{value:.6g}"
            if isinstance(value, str):
                return tr(STATUS_LABELS[value]) if value in STATUS_LABELS else value
            return str(value)
        if role == Qt.ItemDataRole.ToolTipRole:
            return tr('Не определено') if value is None else str(value)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole or (
            role == Qt.ItemDataRole.ToolTipRole and orientation == Qt.Orientation.Horizontal
        ):
            if orientation == Qt.Orientation.Vertical:
                return str(section + 1)
            header = self.headers[section]
            return tr('Группа: {0}', header[8:]) if header.startswith('Группа: ') else tr(header)
        return None


class SampleResultsModel(ResultsModel):
    """Show human-readable sample labels while retaining UUIDs as row keys."""

    def __init__(self, headers, rows, labels, parent=None):
        super().__init__(headers, rows, parent)
        self.labels = labels

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and index.column() == 0:
            row_id = self.rows[index.row()][0]
            label = self.labels.get(row_id, row_id)
            if role == Qt.ItemDataRole.DisplayRole:
                return label
            if role == Qt.ItemDataRole.UserRole:
                return row_id
            if role == Qt.ItemDataRole.ToolTipRole:
                return f'{label}\n{row_id}' if label != row_id else row_id
        return super().data(index, role)


class SortableResultsModel(ResultsModel):
    """Keep the result's original row indices while reordering its display rows."""

    def __init__(self, headers, rows, parent=None, *, row_order=None):
        super().__init__(headers, rows, parent)
        self.row_order = list(range(len(rows))) if row_order is None else row_order

    def set_rows(self, rows):
        self.row_order = list(range(len(rows)))
        super().set_rows(rows)

    def sort(self, column, order=Qt.SortOrder.AscendingOrder):
        if column >= len(self.headers) or len(self.rows) < 2:
            return
        positions = sorted(range(len(self.rows)), key=self.row_order.__getitem__)
        if column >= 0:
            def key(position):
                value = self.rows[position][column]
                return (0, value) if isinstance(value, Real) else (1, str(value).casefold())

            present = [position for position in positions if self.rows[position][column] is not None]
            missing = [position for position in positions if self.rows[position][column] is None]
            positions = sorted(present, key=key, reverse=order == Qt.SortOrder.DescendingOrder) + missing
        if all(position == index for index, position in enumerate(positions)):
            return
        self.layoutAboutToBeChanged.emit()
        persistent = self.persistentIndexList()
        new_positions = {old: new for new, old in enumerate(positions)}
        self.rows = [self.rows[position] for position in positions]
        self.row_order = [self.row_order[position] for position in positions]
        self.changePersistentIndexList(persistent, [
            self.index(new_positions[index.row()], index.column()) for index in persistent
        ])
        self.layoutChanged.emit()
