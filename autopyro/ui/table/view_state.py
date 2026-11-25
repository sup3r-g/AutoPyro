"""Capture and restore the project table's presentation and selection."""

from PySide6.QtCore import QObject, Qt
from PySide6.QtWidgets import QInputDialog, QListWidgetItem

from ..i18n import tr


class TableViewState(QObject):
    """Keep table layout and view snapshots with the widgets they describe."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.column_precision = {}

    def capture_structure_view(self):
        window = self.window
        self.structure_view = self.capture_view(include_sample_set=True)
        self.structure_columns = {c.id for c in window.model.project.columns}
        self.structure_selection = window.sample_selection.capture_structure()

    def restore_structure_view(self):
        window = self.window
        view = self.structure_view
        view["visible"] += [c.id for c in window.model.project.columns if c.id not in self.structure_columns]
        self.rebuild_columns()
        self.restore_view(view)
        selection, self.structure_selection = self.structure_selection, None
        window.sample_selection.restore_structure(selection)

    def rebuild_columns(self, *args):
        window = self.window
        hidden = {
            window.columns.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(window.columns.count())
            if window.columns.item(i).checkState() == Qt.CheckState.Unchecked
        }
        window.columns.blockSignals(True)
        window.columns.clear()
        for i, column in enumerate(window.model.project.columns):
            item = QListWidgetItem(column.name)
            item.setData(Qt.ItemDataRole.UserRole, column.id)
            item.setCheckState(
                Qt.CheckState.Unchecked
                if column.id in hidden
                else Qt.CheckState.Checked
            )
            window.columns.addItem(item)
            window.table.setColumnHidden(i, column.id in hidden)
        window.columns.blockSignals(False)
        self.update_column_precision()

    def toggle_all(self, visible):
        window = self.window
        for i in range(window.columns.count()):
            window.columns.item(i).setCheckState(
                Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked
            )

    def column_toggled(self, item):
        window = self.window
        cid = item.data(Qt.ItemDataRole.UserRole)
        index = next(i for i, c in enumerate(window.model.project.columns) if c.id == cid)
        window.table.setColumnHidden(index, item.checkState() != Qt.CheckState.Checked)
        if window.qc_window is not None:
            window.qc_window.forget_temporary_column(cid)
        window.view_changed()

    def visible_ids(self):
        window = self.window
        header = window.table.horizontalHeader()
        return [
            window.model.project.columns[header.logicalIndex(i)].id
            for i in range(header.count())
            if not window.table.isColumnHidden(header.logicalIndex(i))
        ]

    def capture_view(self, *, include_sample_set=False):
        window = self.window
        view = {
            "visible": self.visible_ids(),
            **window.table_filter_sort.capture(),
            "pinned": [window.model.project.columns[i].id for i in window.table.pinned_columns],
            "column_precision": self.column_precision.copy(),
            "widths": {
                c.id: window.table.columnWidth(i)
                for i, c in enumerate(window.model.project.columns)
            },
        }
        if include_sample_set:
            window.sample_sets.capture_into(view)
        if window.qc_window is not None:
            return window.qc_window.persistent_table_view(view, include_sample_set=include_sample_set)
        return view

    def restore_view(self, view):
        window = self.window
        window.table.set_pinned_columns(())
        precision = view.get("column_precision", {})
        self.column_precision = {
            cid: digits for cid, digits in precision.items()
            if type(digits) is int and 0 <= digits <= 15
        } if isinstance(precision, dict) else {}
        self.update_column_precision()
        ids = {c.id for c in window.model.project.columns}
        row_ids = window.sample_sets.restore_from(view)
        window.table_filter_sort.restore_filter(
            view.get("rules", []), view.get("combine", "AND"), row_ids=row_ids,
        )
        visible = view.get("visible", [c.id for c in window.model.project.columns])
        for i in range(window.columns.count()):
            window.columns.item(i).setCheckState(
                Qt.CheckState.Checked
                if window.columns.item(i).data(Qt.ItemDataRole.UserRole) in visible
                else Qt.CheckState.Unchecked
            )
        header = window.table.horizontalHeader()
        for target, cid in enumerate(visible):
            if cid in ids:
                logical = next(
                    i for i, c in enumerate(window.model.project.columns) if c.id == cid
                )
                header.moveSection(header.visualIndex(logical), target)
        for i, column in enumerate(window.model.project.columns):
            if column.id in view.get("widths", {}):
                window.table.setColumnWidth(i, max(30, int(view["widths"][column.id])))
        positions = {column.id: i for i, column in enumerate(window.model.project.columns)}
        window.table.set_pinned_columns([positions[cid] for cid in view.get("pinned", ()) if cid in positions])
        window.table_filter_sort.restore_sort(view, positions)
        window.update_status()

    def show_saved_views(self, *, focus_name=False):
        desktop = self.window.desktop
        desktop.show_table_properties("views")
        if focus_name:
            desktop.table_properties.view_name.setFocus()

    def restore_defaults(self):
        window = self.window
        self.restore_view({})
        for index in range(window.model.columnCount()):
            window.table.setColumnHidden(index, False)
        window.table.resizeColumnsToContents()
        window.view_changed()
        window.desktop.table_properties.reload_view()

    def pin_column(self, index, pinned):
        window = self.window
        columns = window.table.pinned_columns
        columns = (*columns, index) if pinned else tuple(i for i in columns if i != index)
        window.table.set_pinned_columns(columns)
        window.view_changed()

    def update_column_precision(self):
        window = self.window
        window.table.set_display_precision({
            i: self.column_precision[column.id]
            for i, column in enumerate(window.model.project.columns)
            if column.kind == "number" and column.id in self.column_precision
        })

    def choose_column_precision(self, index):
        window = self.window
        cid = window.model.project.columns[index].id
        digits, accepted = QInputDialog.getInt(
            window, tr('Точность отображения…'), tr('Знаков после запятой (−1 — исходный вид)'),
            self.column_precision.get(cid, -1), -1, 15,
        )
        if not accepted or digits == self.column_precision.get(cid, -1):
            return
        if digits == -1:
            self.column_precision.pop(cid, None)
        else:
            self.column_precision[cid] = digits
        self.update_column_precision()
        window.view_changed()
