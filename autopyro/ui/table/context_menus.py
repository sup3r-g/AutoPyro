"""Context menus for cells and column headers in the project table."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMenu

from autopyro.model.measurements import derived

from ..i18n import tr
from ..theme import icon


def cell_menu(window, position, table=None):
    table = window.table if table is None else table
    index = table.indexAt(position)
    if not index.isValid():
        return
    window.table.setCurrentIndex(index)
    menu = QMenu(window)
    column = window.model.project.columns[window.proxy.mapToSource(index).column()]
    label = tr('Карточка производного показателя…') if derived(column) else tr('Измерения, источник и история…')
    menu.addAction(icon('logs'), label, window.show_measurements)
    try:
        menu.exec(table.viewport().mapToGlobal(position))
    finally:
        menu.deleteLater()


def header_menu(window, position, header=None):
    header = window.table.horizontalHeader() if header is None else header
    index = header.logicalIndexAt(position)
    if index < 0:
        return
    menu = QMenu(window)
    menu.addAction(
        icon('eye-off'), tr('Скрыть колонку'),
        lambda: window.columns.item(index).setCheckState(Qt.CheckState.Unchecked),
    )
    menu.addAction(
        icon('panel-left'), tr('Закрепить первой'),
        lambda: window.table.horizontalHeader().moveSection(
            window.table.horizontalHeader().visualIndex(index), 0
        ),
    )
    pin = menu.addAction(icon('panel-left'), tr('Закрепить при прокрутке'))
    pin.setCheckable(True)
    pin.setChecked(index in window.table.pinned_columns)
    pin.triggered.connect(lambda checked: window.pin_column(index, checked))
    if window.model.project.columns[index].kind == "number":
        menu.addAction(icon('sliders-horizontal'), tr('Точность отображения…'),
                       lambda: window.choose_column_precision(index))
    try:
        menu.exec(header.mapToGlobal(position))
    finally:
        menu.deleteLater()
