"""Open measurement cards and navigate from analysis results to source cells."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from ..i18n import tr


def show_measurements(window):
    from ..dialogs.measurement import DerivedCellDialog, MeasurementDialog
    from autopyro.model.measurements import derived

    index = window.table.currentIndex()
    if not index.isValid():
        return
    source = window.proxy.mapToSource(index)
    project = window.model.project
    column = project.columns[source.column()]
    row = project.rows[source.row()]
    if derived(column):
        dialog = DerivedCellDialog(project, row, column, window, open_support=window.open_source_cell)
    else:
        dialog = MeasurementDialog(window.model, row.id, column.id, window)
    try:
        dialog.exec()
    finally:
        dialog.deleteLater()


def open_source_cell(window, row_id, column_id, parent):
    title = tr('Исходная ячейка')
    position = window.model.resolve_row_positions([row_id]).get(row_id)
    if position is None:
        QMessageBox.information(parent, title, tr('Строка пробы отсутствует в проекте.'))
        return False
    column = next((i for i, column in enumerate(window.model.project.columns) if column.id == column_id), None)
    if column is None:
        QMessageBox.information(parent, title, tr('Исходная колонка отсутствует в проекте.'))
        return False

    source = window.model.index(position, column)
    target = window.proxy.mapFromSource(source)
    if not target.isValid():
        membership = window.proxy.row_id_filter
        close_set = window.active_sample_set is not None and membership is not None and row_id not in membership
        prompt = (
            tr('Проба вне открытого набора. Закрыть набор и сбросить фильтр строк для перехода?')
            if close_set else tr('Проба скрыта. Сбросить фильтр строк для перехода?')
        )
        answer = QMessageBox.question(
            parent, title, prompt,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False
        if close_set:
            window.sample_sets.reset()
            window.table_filter_sort.restore_filter([], "AND", row_ids=None)
        elif window.table_filter_sort.missing_filter_column:
            window.apply_filter([], "AND")
        else:
            window.proxy.set_rules([])
        window.view_changed()
        target = window.proxy.mapFromSource(source)
    window.desktop.open_table()
    if window.table.isColumnHidden(column):
        window.columns.item(column).setCheckState(Qt.CheckState.Checked)
    window.table.setCurrentIndex(target)
    window.table.scrollTo(target)
    return True
