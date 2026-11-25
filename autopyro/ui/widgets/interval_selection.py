"""Link interval IDs to a compact table and lightweight depth highlights."""

from matplotlib.patches import Rectangle
from PySide6.QtCore import QItemSelection, QItemSelectionModel, QModelIndex, QObject, QPersistentModelIndex, QTimer, Qt
from PySide6.QtWidgets import QAbstractItemView, QTableView

from autopyro.geology.stratigraphy import LEVELS, _DepthIndex
from ..table.selection import add_row_ranges
from .stratigraphy_labels import GeologicalDelegate


class VisibleIntervalTable(QTableView):
    def setSelection(self, rect, flags):
        model = self.model()
        if model is None:
            return
        rect = rect.normalized()
        first = self.rowAt(max(0, rect.top()))
        if first < 0:
            self.selectionModel().select(QItemSelection(), flags)
            return
        last = self.rowAt(min(self.viewport().height() - 1, rect.bottom()))
        if last < 0:
            last = model.rowCount() - 1
        rows = (row for row in range(first, last + 1) if not self.isRowHidden(row))
        selection = QItemSelection()
        add_row_ranges(selection, model, rows, 0, model.columnCount() - 1)
        self.selectionModel().select(selection, flags)

    def selectAll(self):
        model = self.model()
        if model is None:
            return
        rows = (row for row in range(model.rowCount()) if not self.isRowHidden(row))
        selection = QItemSelection()
        add_row_ranges(selection, model, rows, 0, model.columnCount() - 1)
        self.selectionModel().select(selection, QItemSelectionModel.SelectionFlag.ClearAndSelect)


class IntervalSelection(QObject):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.table = owner.intervals
        self.view = VisibleIntervalTable()
        self.view.setModel(self.table.model())
        self.view.setSelectionModel(self.table.selectionModel())
        self.view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        for column in range(self.table.columnCount()):
            self.view.setColumnHidden(column, column not in (1, 3, 4))
        self.view.setColumnWidth(1, 140)
        self.view.setColumnWidth(0, 100)
        self.view.setColumnWidth(2, 110)
        self.view.setItemDelegateForColumn(2, GeologicalDelegate(self.view, LEVELS))
        header = self.view.horizontalHeader()
        for position, column in enumerate((3, 4)):
            header.moveSection(header.visualIndex(column), position)
            self.view.setColumnWidth(column, 80)
        self.view.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.view.horizontalHeader().setTextElideMode(Qt.TextElideMode.ElideRight)
        self.rendered = {}
        self.depth_indexes = {}
        self._indexed_source = None
        self.highlights = []
        self.background = None
        self.table.itemSelectionChanged.connect(self.update_highlights)
        self.connect_canvas()

    def connect_canvas(self):
        canvas = self.owner.column_canvas
        self.background = None
        canvas.mpl_connect('button_press_event', self.on_click)
        canvas.mpl_connect('draw_event', self.cache_background)
        canvas.mpl_connect('resize_event', self.invalidate_background)

    def invalidate_background(self, *_):
        self.background = None

    def cache_background(self, event):
        canvas = event.canvas
        self.background = canvas.copy_from_bbox(canvas.figure.bbox)
        self.paint_highlights()

    def paint_highlights(self):
        axes = set(self.owner.column_figure.axes)
        for artist in self.highlights:
            if artist.axes in axes:
                artist.axes.draw_artist(artist)

    def selected_ids(self):
        rows = {row for selection in self.table.selectedRanges()
                for row in range(selection.topRow(), selection.bottomRow() + 1)}
        return {cell.text() for row in rows if (cell := self.table.item(row, 6)) is not None}

    def set_rendered(self, scheme, well, level):
        self.highlights = []  # The renderer has already cleared the old axes.
        self.primary_level = level
        self.primary_well = well
        tracks = {self.track_key(axis) for axis in self.owner.column_figure.axes
                  if getattr(axis, '_autopyro_kind', '') != 'samples'}
        self.origins = {self.track_key(axis)[0]: getattr(axis, '_autopyro_depth_origin', 0)
                        for axis in self.owner.column_figure.axes}
        source = self._indexed_source
        if source is None or source[0] is not scheme or source[1] != tracks:
            self.set_rendered_tracks(scheme, tracks)
        self.update_highlights()

    def set_rendered_tracks(self, scheme, tracks):
        intervals_by_track = {track: [] for track in tracks}
        self.rendered = {}
        for item in scheme['intervals']:
            track = (item['wellbore'], item['level'])
            if track in intervals_by_track:
                intervals_by_track[track].append(item)
                self.rendered[item['id']] = item
        self.depth_indexes = {track: _DepthIndex(items) for track, items in intervals_by_track.items()}
        self.view.setColumnHidden(0, len({well for well, level in tracks}) < 2)
        self.view.setColumnHidden(2, len({level for well, level in tracks}) < 2)
        for key, row in self.owner.interval_history.rows_by_id.items():
            self.view.setRowHidden(row, key not in self.rendered)
        self._indexed_source = scheme, tracks

    def track_key(self, axis):
        return (getattr(axis, '_autopyro_well', self.primary_well),
                getattr(axis, '_autopyro_level', self.primary_level))

    def clear(self):
        self.rendered = {}
        self.depth_indexes = {}
        self._indexed_source = None
        self.origins = {}
        self.highlights = []
        self.background = None
        action = getattr(self.owner, 'zoom_intervals_action', None)
        if action is not None:
            action.setEnabled(False)

    def update_highlights(self):
        selected = self.selected_ids() & self.rendered.keys()
        action = getattr(self.owner, 'zoom_intervals_action', None)
        if action is not None:
            action.setEnabled(bool(selected) and self.owner._has_column_preview)
        if not self.rendered or not self.owner._has_column_preview:
            return
        if not selected and not self.highlights:
            return
        canvas = self.owner.column_canvas
        can_blit = self.background is not None and not canvas.figure.stale
        for artist in self.highlights:
            artist.remove()
        self.highlights = []
        # Group once by well and rank, then merge adjacent bands for each track.
        intervals_by_track = {}
        for key in selected:
            item = self.rendered[key]
            origin = self.origins[item['wellbore']]
            intervals_by_track.setdefault((item['wellbore'], item['level']), []).append(
                (item['top'] - origin, item['bottom'] - origin))
        spans_by_track = {}
        for track, intervals in intervals_by_track.items():
            spans = spans_by_track[track] = []
            for top, bottom in sorted(intervals):
                if spans and top <= spans[-1][1]:
                    spans[-1] = (spans[-1][0], max(bottom, spans[-1][1]))
                else:
                    spans.append((top, bottom))
        color = self.table.palette().highlight().color().name()
        for ax in self.owner.column_figure.axes:
            if getattr(ax, '_autopyro_kind', '') == 'samples':
                continue
            for top, bottom in spans_by_track.get(self.track_key(ax), []):
                highlight = Rectangle((0, top), 1, bottom - top, transform=ax.get_yaxis_transform(),
                                      facecolor=color, edgecolor=color, alpha=.25, animated=True)
                ax.add_artist(highlight)
                self.highlights.append(highlight)
        if self.owner.column_page.isVisible():
            if can_blit:
                canvas.restore_region(self.background)
                self.paint_highlights()
                canvas.figure.stale = False
                canvas.blit(canvas.figure.bbox)
            else:
                canvas.draw_idle()

    def select_ids(self, ids):
        selection = QItemSelection()
        positions = self.owner.interval_history.rows_by_id
        rows = sorted({positions[key] for key in ids if key in positions})
        model = self.table.model()
        add_row_ranges(selection, model, rows, 0, model.columnCount() - 1)
        self.table.selectionModel().select(selection, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        if rows:
            index = model.index(rows[0], 1)
            self.table.selectionModel().setCurrentIndex(index, QItemSelectionModel.SelectionFlag.NoUpdate)
            self.table.scrollTo(index)
            depth = model.index(rows[0], 3)
            self.view.scrollTo(depth)
            if self.view.isVisible() and not self.view.viewport().rect().intersects(self.view.visualRect(depth)):
                # Switching tracks changes hidden rows before Qt lays out the scroll range.
                pending = QPersistentModelIndex(depth)
                QTimer.singleShot(0, self, lambda: self.scroll_to_current_interval(pending))

    def scroll_to_current_interval(self, index):
        if index.isValid() and self.view.isVisible() and index.row() == self.view.currentIndex().row():
            cell = QModelIndex(index)
            if self.view.selectionModel().isSelected(cell):
                self.view.scrollTo(cell)

    def on_click(self, event):
        if (not self.owner._has_column_preview or event.button != 1 or event.ydata is None
                or event.inaxes not in self.owner.column_figure.axes
                or self.owner.column_canvas.navigation_draw_pending
                or self.owner.column_toolbar.mode):
            return
        if getattr(event.inaxes, '_autopyro_kind', '') == 'samples':
            self.owner.open_sample_row(event)
            return
        well, level = self.track_key(event.inaxes)
        depth = event.ydata + self.origins.get(well, 0)
        index = self.depth_indexes.get((well, level))
        hits = {item['id'] for item in index.hits(depth)} if index is not None else set()
        if not hits:
            return
        key = event.key or ''
        if 'control' in key or 'ctrl' in key:
            hits ^= self.selected_ids()
        elif 'shift' in key:
            hits |= self.selected_ids()
        self.select_ids(hits)

    def zoom_to_selected(self):
        selected = self.selected_ids() & self.rendered.keys()
        if not selected or not self.owner._has_column_preview:
            return
        items = [self.rendered[key] for key in selected]
        top = min(item['top'] - self.origins[item['wellbore']] for item in items)
        bottom = max(item['bottom'] - self.origins[item['wellbore']] for item in items)
        self.zoom_to_depth(top, bottom)

    def zoom_to_depth(self, top, bottom):
        padding = (bottom - top) * .04
        self.owner.column_toolbar.push_current()
        axis = self.owner.column_figure.axes[0]
        limits = (bottom + padding, top - padding)
        axis.set_ylim(limits if axis.yaxis_inverted() else limits[::-1])
        self.owner.column_toolbar.push_current()
        self.owner.column_canvas.draw_idle()
