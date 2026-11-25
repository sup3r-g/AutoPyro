"""Set up and bind the main project's table and column controls."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDockWidget, QInputDialog, QLineEdit, QListWidget, QMessageBox,
    QPushButton, QTableView, QVBoxLayout, QWidget,
)

from autopyro.model.core import Column
from autopyro.model.field_types import FIELD_KINDS

from ..field_types import field_kind_label
from ..i18n import tr
from ..tasks import run_with_progress
from ..theme import icon
from .commands import AddColumnCommand
from .model import TableModel, large_project, prepare_project
from .proxy import FilterProxy
from .selection import SampleSelection
from ..theme.controls import add_search_icon


def install(window):
    window.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    window.table.customContextMenuRequested.connect(window.cell_menu)
    window.table.frozen.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    window.table.frozen.customContextMenuRequested.connect(
        lambda position: window.cell_menu(position, window.table.frozen)
    )
    window.table.setAlternatingRowColors(True)
    window.table.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
    window.table.horizontalHeader().setSectionsMovable(True)
    window.table.horizontalHeader().setContextMenuPolicy(
        Qt.ContextMenuPolicy.CustomContextMenu
    )
    window.table.horizontalHeader().customContextMenuRequested.connect(
        window.header_menu
    )
    frozen_header = window.table.frozen.horizontalHeader()
    frozen_header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    frozen_header.customContextMenuRequested.connect(
        lambda position: window.header_menu(position, frozen_header)
    )
    window.table.horizontalHeader().sectionClicked.connect(window.single_sort)
    window.table.horizontalHeader().sectionMoved.connect(
        lambda *_: window.view_changed()
    )
    window.setCentralWidget(window.table)
    window.columns = QListWidget()
    window.columns.itemChanged.connect(window.table_view.column_toggled)
    dock = QDockWidget(tr('Видимые колонки'), window)
    holder = QWidget()
    layout = QVBoxLayout(holder)
    search = QLineEdit()
    search.setPlaceholderText(tr('Поиск колонки…'))
    add_search_icon(search)
    search.textChanged.connect(lambda text: _filter_columns(window.columns, text))
    layout.addWidget(search)
    layout.addWidget(window.columns)
    show_all = QPushButton(icon('eye'), tr('Показать все'))
    show_all.clicked.connect(lambda: window.toggle_all(True))
    hide_all = QPushButton(icon('eye-off'), tr('Скрыть все'))
    hide_all.clicked.connect(lambda: window.toggle_all(False))
    layout.addWidget(show_all)
    layout.addWidget(hide_all)
    dock.setWidget(holder)
    window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    window.column_dock = dock


def _filter_columns(columns, text):
    query = text.casefold()
    for index in range(columns.count()):
        item = columns.item(index)
        item.setHidden(query not in item.text().casefold())


def prepare(window, project, *, prepared):
    if not prepared and large_project(project):
        run_with_progress(window, tr("Подготовка таблицы…"), lambda: prepare_project(project))
        prepared = True
    return prepared


def bind(window, project, *, prepared):
    old_model = getattr(window, "model", None)
    old_proxy = getattr(window, "proxy", None)
    window.model = TableModel(project, window, prepared=prepared)
    window.model.preserve_recovery_edit = window.session.preserve_recovery_edit
    window.proxy = FilterProxy(window)
    window.proxy.setSourceModel(window.model)
    window.table.setModel(window.proxy)
    window.proxy.sortingChanged.connect(window.table.update_sort_indicator)
    window.table.update_sort_indicator()
    window.sample_selection = SampleSelection(window)
    window.sample_selection.changed.connect(window.update_selection_status)
    window.update_selection_status()
    window.model.error.connect(
        lambda text: QMessageBox.warning(window, tr('Значение'), text)
    )
    window.model.changed.connect(window.model_changed)
    window.model.columnsInserted.connect(window.table_view.rebuild_columns)
    window.model.columnsRemoved.connect(window.table_view.rebuild_columns)
    window.model.modelAboutToBeReset.connect(window.table_view.capture_structure_view)
    window.model.structureChanged.connect(window.table_view.restore_structure_view)
    window.model.undo.canUndoChanged.connect(window.undo_action.setEnabled)
    window.model.undo.canRedoChanged.connect(window.redo_action.setEnabled)
    window.undo_action.setEnabled(False)
    window.redo_action.setEnabled(False)
    return old_model, old_proxy


def fit_columns(window):
    window.table_view.rebuild_columns()
    window.table.horizontalHeader().setResizeContentsPrecision(100)
    window.table.resizeColumnsToContents()


def add_column(window):
    name, ok = QInputDialog.getText(window, tr('Новая колонка'), tr('Уникальное название'))
    if not ok:
        return
    if not name.strip() or name.strip() in {
        c.name for c in window.model.project.columns
    }:
        QMessageBox.warning(
            window, tr('Колонка'), tr('Название должно быть непустым и уникальным')
        )
        return
    kinds = {field_kind_label(kind): kind for kind in FIELD_KINDS}
    label, ok = QInputDialog.getItem(
        window, tr('Тип'), tr('Тип колонки'), list(kinds), editable=False
    )
    if ok:
        window.model.undo.push(
            AddColumnCommand(
                window.model,
                Column(name.strip(), kinds[label]),
                {},
            )
        )
