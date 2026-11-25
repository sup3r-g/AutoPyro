"""Row UUID selection without expanding every selected cell into a QModelIndex."""
from PySide6.QtCore import QObject, QItemSelection, QItemSelectionModel, Signal

from autopyro.model.sample_labels import sample_label_column


def _uses_source_order(proxy):
    return proxy.row_id_filter is None and not proxy.rules and proxy.sortColumn() < 0


def _selected_intervals(owner):
    intervals = sorted((area.top(), area.bottom())
                       for area in owner.table.selectionModel().selection())
    start, end = 0, -1
    for top, bottom in intervals:
        if top > end + 1:
            if start <= end:
                yield range(start, end + 1)
            start, end = top, bottom
        else:
            end = max(end, bottom)
    if start <= end:
        yield range(start, end + 1)


def selected_rows(owner, *, ordered=False):
    rows = (row for interval in tuple(_selected_intervals(owner)) for row in interval)
    project_rows = owner.model.project.rows
    proxy = owner.proxy
    if _uses_source_order(proxy):
        return (project_rows[row] for row in rows)
    else:
        return (project_rows[proxy.mapToSource(proxy.index(row, 0)).row()] for row in rows)


def selected_row_ids(owner, *, ordered=False):
    identifiers = (row.id for row in selected_rows(owner, ordered=ordered))
    return list(identifiers) if ordered else set(identifiers)


class SampleSelection(QObject):
    """Keep sample identities across filters, including the table's cell ranges."""

    changed = Signal()

    def __init__(self, owner):
        super().__init__(owner.proxy)
        self.owner = owner
        self._positions = {}
        self._table_ranges = ()
        self._syncing = False
        self._filter_depth = 0
        self._data_filtering = False
        self._column_snapshot = None
        owner.table.selectionModel().selectionChanged.connect(self._table_changed)
        owner.proxy.filteringChanged.connect(self._filtering_changed)
        owner.model.updatingData.connect(self._data_updating)
        owner.model.modelReset.connect(self._reset)
        owner.model.updatingColumns.connect(self._columns_updating)

    def _columns_updating(self, started):
        if started:
            # Capture before Qt's proxy removes selected cells from its selection.
            self._column_snapshot = self.capture_structure()
            self._syncing = True
        else:
            snapshot, self._column_snapshot = self._column_snapshot, None
            self._syncing = False
            self.restore_structure(snapshot)

    def rows(self, *, ordered=False):
        visible = selected_rows(self.owner, ordered=ordered)
        if not self._positions:
            return visible
        visible = list(visible)
        visible_ids = {row.id for row in visible}
        project_rows = self.owner.model.project.rows
        return iter(visible + [project_rows[pos] for rid, pos in self._positions.items()
                               if rid not in visible_ids])

    def ids(self, *, ordered=False):
        if not ordered and self._positions:
            return set(self._positions)
        identifiers = (row.id for row in self.rows(ordered=ordered))
        return list(identifiers) if ordered else set(identifiers)

    def counts(self):
        visible = sum(map(len, _selected_intervals(self.owner)))
        return visible, len(self._positions) - visible if self._positions else 0

    def clear(self):
        selection = self.owner.table.selectionModel()
        if not self._positions and not selection.hasSelection():
            return
        self._positions = {}
        self._table_ranges = ()
        self._syncing = True
        try:
            selection.clearSelection()
        finally:
            self._syncing = False
        self.changed.emit()

    def export_snapshot(self):
        owner = self.owner
        intervals = tuple(_selected_intervals(owner))
        positions = self._positions  # Selection changes replace the mapping.
        project_rows, proxy = owner.model.project.rows, owner.proxy
        direct = _uses_source_order(proxy)

        def resolve():
            seen = set()
            for interval in intervals:
                for row in interval:
                    source = row if direct else proxy.mapToSource(proxy.index(row, 0)).row()
                    sample = project_rows[source]
                    if positions:
                        seen.add(sample.id)
                    yield sample
            for rid, source in positions.items():
                if rid not in seen:
                    yield project_rows[source]

        # Stored UUID positions already include the visible part of the selection.
        count = len(positions) if positions else sum(map(len, intervals))
        return count, resolve()

    def select(self, row_ids):
        # The model resolves positions in source order.
        self._positions = self.owner.model.resolve_row_positions(row_ids)
        self._table_ranges = ()
        return len(self._positions) - self._sync_table()

    def capture_structure(self):
        if not self._positions:
            self._remember_table()
        if not self._positions:
            return None
        columns = self.owner.model.project.columns
        ranges = [(tuple(column.id for column in columns[left:right + 1]), positions)
                  for left, right, positions in self._table_ranges]
        return self._positions, ranges

    def restore_structure(self, snapshot):
        if snapshot is None:
            return
        previous, ranges = snapshot
        model = self.owner.model
        self._positions = model.resolve_row_positions(previous, previous)
        self._table_ranges = []
        if ranges:
            columns = {column.id: i for i, column in enumerate(model.project.columns)}
            by_old_position = {previous[rid]: pos for rid, pos in self._positions.items()}
            surviving_rows = set()
            for column_ids, old_positions in ranges:
                indices = sorted(columns[cid] for cid in column_ids if cid in columns)
                if not indices:
                    continue
                positions = [by_old_position[pos] for pos in old_positions if pos in by_old_position]
                if not positions:
                    continue
                surviving_rows.update(positions)
                left = right = indices[0]
                for column in indices[1:]:
                    if column != right + 1:
                        self._table_ranges.append((left, right, positions))
                        left = column
                    right = column
                self._table_ranges.append((left, right, positions))
            self._positions = {rid: pos for rid, pos in self._positions.items() if pos in surviving_rows}
        self._sync_table()

    def _remember_table(self):
        owner = self.owner
        proxy = owner.proxy
        ranges = owner.table.selectionModel().selection()
        if not ranges:
            return
        direct = _uses_source_order(proxy)
        source_rows = {}
        table_ranges = []
        for area in ranges:
            positions = []
            for row in range(area.top(), area.bottom() + 1):
                if row not in source_rows:
                    source_rows[row] = row if direct else proxy.mapToSource(proxy.index(row, 0)).row()
                positions.append(source_rows[row])
            table_ranges.append((area.left(), area.right(), positions))
        project_rows = owner.model.project.rows
        self._positions = {project_rows[pos].id: pos for pos in sorted(source_rows.values())}
        self._table_ranges = table_ranges

    def _restore_table_ranges(self):
        owner = self.owner
        proxy = owner.proxy
        if _uses_source_order(proxy):
            visible = {pos: pos for pos in self._positions.values()}
        elif len(self._positions) > proxy.rowCount():
            visible = {self._positions[rid]: row for row, rid in enumerate(proxy.ordered_row_ids())
                       if rid in self._positions}
        else:
            visible = {}
            for pos in self._positions.values():
                index = proxy.mapFromSource(owner.model.index(pos, 0))
                if index.isValid():
                    visible[pos] = index.row()
        selection = QItemSelection()
        for left, right, positions in self._table_ranges:
            rows = sorted(visible[pos] for pos in positions if pos in visible)
            add_row_ranges(selection, proxy, rows, left, right)
        owner.table.selectionModel().select(selection, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        return len(visible)

    def _sync_table(self):
        self._syncing = True
        try:
            if self._table_ranges:
                visible = self._restore_table_ranges()
                if visible == len(self._positions):
                    self._positions = {}
                    self._table_ranges = ()
            else:
                visible = select_row_ids(self.owner, self._positions, source_positions=self._positions)
        finally:
            self._syncing = False
        self.changed.emit()
        return visible

    def _table_changed(self, *_):
        if not self._syncing and not (self._filter_depth and self._positions):
            self._positions = {}
            self._table_ranges = ()
            self.changed.emit()

    def _filtering_changed(self, started):
        if started and not self._filter_depth and not self._positions:
            self._remember_table()
        self._filter_depth += 1 if started else -1
        if not self._filter_depth and self._positions:
            self._sync_table()

    def _data_updating(self, started):
        if started:
            proxy = self.owner.proxy
            self._data_filtering = bool(proxy.rules) and proxy.dynamicSortFilter()
        if self._data_filtering:
            self._filtering_changed(started)
        if not started:
            self._data_filtering = False

    def _reset(self):
        if self._positions:
            self._positions = {}
            self._table_ranges = ()
            self.changed.emit()


def select_row_ids(owner, row_ids, *, source_positions=None, preserve_if_empty=False):
    wanted = set(row_ids)
    selection = QItemSelection()
    if len(wanted) <= 1 or (source_positions is not None and len(wanted) * 64 < owner.proxy.rowCount()):
        positions = owner.model.resolve_row_positions(wanted) if source_positions is None else source_positions
        rows = sorted(index.row() for position in positions.values()
                      if (index := owner.proxy.mapFromSource(owner.model.index(position, 0))).isValid())
    else:
        # For dense selections, use the cached proxy order instead of mapping each UUID.
        rows = (row for row, rid in enumerate(owner.proxy.ordered_row_ids()) if rid in wanted)
    visible = add_row_ranges(selection, owner.proxy, rows, 0, 0)
    if preserve_if_empty and not visible:
        return 0
    # One notification, including clearing the previous selection or an empty scope.
    owner.table.selectionModel().select(selection,
        QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows)
    return visible


def add_row_ranges(selection, model, rows, left, right):
    start = end = None
    visible = 0
    for row in rows:
        if end is None:
            start = row
        elif row != end + 1:
            selection.select(model.index(start, left), model.index(end, right))
            start = row
        end = row
        visible += 1
    if start is not None:
        selection.select(model.index(start, left), model.index(end, right))
    return visible
