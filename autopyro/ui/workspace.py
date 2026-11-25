"""LabPlot-style document workspace, project tree and docked properties."""

from PySide6.QtCore import QByteArray, QObject, QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QDockWidget, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QMenu, QPushButton, QScrollArea, QSizePolicy, QSplitter, QStackedWidget, QTabWidget,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget, QToolBar,
    QHeaderView, QGroupBox,
)
from shiboken6 import isValid

from . import i18n
from .document_descriptors import DOCUMENTS, document_descriptor, geochemistry_documents
from .i18n import tr
from .field_types import field_kind_label
from .theme import icon
from .table.properties import TableProperties
from .table.view_properties import SavedViewProperties
from .table.column_type import ColumnTypeProperties
from .theme.controls import add_search_icon


# Version 1 included a tabified column dock. Restoring it after moving the
# controls into table properties can crash native Qt while showing the window.
LAYOUT_VERSION = 2
DOCUMENT_ICONS = {
    "project": "folder-kanban", "folder": "folder", "table": "table-2",
    "column": "columns-3", "view": "layout-template",
    "saved_plot": DOCUMENTS["plot"].icon, "map_layer": "layers",
    **{kind: descriptor.icon for kind, descriptor in DOCUMENTS.items()},
}


class ObjectProperties(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.kind = "project"
        self.column_id = None
        layout = QVBoxLayout(self)
        self.heading = QLabel(tr("Свойства объекта"))
        layout.addWidget(self.heading)
        group = QGroupBox(tr('Общие сведения'))
        layout.addWidget(group)
        form = self.form = QFormLayout(group)
        self.fields = {}
        for key, label in (("name", tr("Название")), ("type", tr("Тип")), ("rows", tr("Строк")),
                           ("columns", tr("Колонок")), ("unit", tr("Единица")), ("source", tr("Источник"))):
            field = QLineEdit()
            field.setReadOnly(True)
            field.setFrame(False)
            self.fields[key] = field
            form.addRow(label, field)
        self.visible_column = QCheckBox(tr("Показывать колонку"))
        self.visible_column.toggled.connect(self.toggle_column)
        layout.addWidget(self.visible_column)
        self.column_type = ColumnTypeProperties(window)
        self.column_type.hide()
        layout.addWidget(self.column_type)
        hint = QLabel(tr("Параметры графика и карты появляются здесь при выборе соответствующего листа."))
        hint.setWordWrap(True)
        hint.setObjectName("workspaceHint")
        layout.addWidget(hint)
        layout.addStretch()

    def show_object(self, kind="project", key=""):
        project = self.window.model.project
        column = next((c for c in project.columns if c.id == key), None) if kind == "column" else None
        if kind == "column" and column is None:
            kind = "table"
        self.kind = kind
        self.column_id = column.id if column else None
        self.column_type.show_column(column)
        source = project.source
        values = {
            "name": column.name if column else tr("Таблица") if kind == "table" else self.window.path.name if self.window.path else tr("Новый проект"),
            "type": field_kind_label(column.kind) if column else tr("Таблица") if kind == "table" else tr("Проект"),
            "rows": str(len(project.rows)), "columns": str(len(project.columns)),
            "unit": column.unit if column else "",
            "source": str(source.get("path") or source.get("host") or source.get("type") or ""),
        }
        for name, value in values.items():
            self.fields[name].setText(value)
        self.heading.setText(tr("Свойства колонки") if column else tr("Таблица") if kind == "table" else tr("Свойства проекта"))
        self.visible_column.setVisible(column is not None)
        for name in ("source", "unit"):
            visible = kind == "project" if name == "source" else column is not None
            self.fields[name].setVisible(visible)
            self.form.labelForField(self.fields[name]).setVisible(visible)
        blocker = QSignalBlocker(self.visible_column)
        if column:
            index = project.columns.index(column)
            self.visible_column.setChecked(not self.window.table.isColumnHidden(index))
        del blocker

    def toggle_column(self, checked):
        for index in range(self.window.columns.count()):
            item = self.window.columns.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == self.column_id:
                item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
                return


class DesktopWorkspace(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self._signature = None
        self._borrowed = None
        self._documents = {}
        self._closing_documents = False
        self.welcome_buttons = []
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(80)
        self._refresh_timer.timeout.connect(self.refresh_project)
        self.documents = QTabWidget()
        self.documents.setObjectName("documentWorkspace")
        self.documents.setDocumentMode(True)
        self.documents.setMovable(True)
        self.documents.setTabsClosable(True)
        self.documents.setUsesScrollButtons(True)
        self.documents.tabBar().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.documents.tabBar().customContextMenuRequested.connect(self.document_menu)
        self.table_page = QStackedWidget()
        window.takeCentralWidget()
        self.table_page.addWidget(window.table)
        self.table_page.addWidget(self.welcome_page())
        self.documents.addTab(self.table_page, icon("table-2"), tr("Таблица"))
        self.documents.tabCloseRequested.connect(self.close_document)
        self.documents.currentChanged.connect(self.document_changed)
        window.setCentralWidget(self.documents)

        self.project_dock = QDockWidget(tr("Обозреватель проекта"), window)
        self.project_dock.setMinimumWidth(235)
        self.project_dock.setObjectName("projectExplorerDock")
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        self.search = QLineEdit()
        self.search.setClearButtonEnabled(True)
        self.search.setPlaceholderText(tr("Поиск объектов проекта…"))
        add_search_icon(self.search)
        self.search.setWhatsThis(tr("Ищет таблицы, колонки, листы и сохранённые представления по имени. Исходные данные не фильтруются."))
        layout.addWidget(self.search)
        self.tree = QTreeWidget()
        self.tree.setObjectName("projectExplorer")
        self.tree.setHeaderLabels([tr("Объект"), tr("Тип")])
        self.tree.setColumnWidth(0, 190)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setUniformRowHeights(True)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.project_menu)
        self.tree.currentItemChanged.connect(self.object_selected)
        self.search.textChanged.connect(self.filter_tree)
        layout.addWidget(self.tree)
        self.project_dock.setWidget(panel)

        self.properties_dock = QDockWidget(tr("Свойства"), window)
        self.properties_dock.setMinimumWidth(365)
        self.properties_dock.setObjectName("propertiesExplorerDock")
        self.properties_stack = QStackedWidget()
        self.properties = ObjectProperties(window)
        self.properties_stack.addWidget(self.properties)
        self.table_properties = TableProperties(window, ObjectProperties(window))
        self.properties_stack.addWidget(self.table_properties)
        self.view_properties = SavedViewProperties(window)
        self.properties_stack.addWidget(self.view_properties)
        properties_scroll = QScrollArea()
        properties_scroll.setWidgetResizable(True)
        properties_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        properties_scroll.setWidget(self.properties_stack)
        self.properties_dock.setWidget(properties_scroll)
        window.column_dock.setObjectName("columnVisibilityDock")
        self.default_layout()
        self.install_menus()
        self.restore_layout()

    def welcome_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addStretch()
        heading = QLabel(tr("Рабочее пространство AutoPyro"))
        heading.setObjectName("workspaceHeading")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heading.setWordWrap(True)
        layout.addWidget(heading)
        hint = QLabel(tr("Импортируйте данные, затем создайте график, карту или расчёт статистики."))
        hint.setObjectName("workspaceHint")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        layout.addWidget(hint)
        for source, callback, name in (
            (tr("Открыть таблицу как новый проект…"), self.window.import_file, "document-import"),
            (tr("Открыть проект…"), self.window.open_project, "folder-open"),
            (tr("Импорт геохимии из БД…"), self.window.import_geochemistry, "database"),
            (tr("Демонстрационные данные"), self.window.demo, "flask-conical"),
        ):
            button = QPushButton(icon(name), source)
            button.clicked.connect(callback)
            self.welcome_buttons.append(button)
            layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(page)
        return scroll

    def default_layout(self):
        window = self.window
        window.setDockOptions(window.DockOption.AnimatedDocks | window.DockOption.AllowNestedDocks | window.DockOption.AllowTabbedDocks)
        window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.project_dock)
        # Keep the legacy dock attached (and hidden): removing it leaves a
        # placeholder in saveState(), which can crash native Qt on restore.
        window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, window.column_dock)
        window.column_dock.hide()
        window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.properties_dock)
        for dock in (self.project_dock, self.properties_dock):
            dock.setFloating(False)
            dock.show()
        self.project_dock.raise_()
        window.resizeDocks([self.project_dock, self.properties_dock], [270, 390], Qt.Orientation.Horizontal)

    def install_menus(self):
        window = self.window
        view = window.menuBar().addMenu(tr("Вид"))
        for dock, icon_name in ((self.project_dock, 'folder-kanban'), (self.properties_dock, 'sliders-horizontal')):
            action = dock.toggleViewAction()
            action.setIcon(icon(icon_name))
            view.addAction(action)
        focus = view.addAction(icon("panel-left"), tr("Скрыть / вернуть боковые панели"))
        focus.setShortcut("Ctrl+Shift+F11")
        def toggle_focus():
            if self.project_dock.isVisible() or self.properties_dock.isVisible():
                self._focus_docks = (self.project_dock.isVisible(), self.properties_dock.isVisible())
                self.project_dock.hide()
                self.properties_dock.hide()
            else:
                project, properties = getattr(self, "_focus_docks", (True, True))
                self.project_dock.setVisible(project)
                self.properties_dock.setVisible(properties)
        focus.triggered.connect(toggle_focus)
        view.addSeparator()
        for toolbar in window.findChildren(QToolBar):
            view.addAction(toolbar.toggleViewAction())
        view.addSeparator()
        view.addAction(icon('layout-template'), tr("Восстановить расположение панелей"), self.default_layout)
        self.window_menu = window.menuBar().addMenu(tr("Окно"))
        self.window_menu.aboutToShow.connect(self.rebuild_window_menu)
        settings_action = next(action for action in window.menuBar().actions() if action.text() == tr("Настройки"))
        for menu in (view, self.window_menu):
            window.menuBar().removeAction(menu.menuAction())
            window.menuBar().insertAction(settings_action, menu.menuAction())
        close = QAction(icon("x"), tr("Закрыть текущий лист"), window)
        close.setShortcut(QKeySequence("Ctrl+W"))
        close.triggered.connect(lambda: self.close_document(self.documents.currentIndex()))
        close.setEnabled(False)
        window.addAction(close)
        self.close_action = close

    def rebuild_window_menu(self):
        self.window_menu.clear()
        self.window_menu.addAction(self.close_action)
        self.window_menu.addSeparator()
        for index in range(self.documents.count()):
            action = self.window_menu.addAction(self.documents.tabText(index))
            action.setIcon(self.documents.tabIcon(index))
            action.setCheckable(True)
            action.setChecked(index == self.documents.currentIndex())
            action.triggered.connect(lambda checked=False, selected=index: self.documents.setCurrentIndex(selected))

    def open_document(self, key, widget, title, *, profile="", icon_name=None):
        index = self.documents.indexOf(widget)
        if index < 0:
            widget.setParent(self.documents)
            widget.setWindowFlags(Qt.WindowType.Widget)
            widget.setModal(False)
            self._ignore_document_size(widget)
            if hasattr(widget, "whats_this_action"):
                # The main window supplies the common Shift+F1 shortcut; the
                # document's help button still invokes its native Qt action.
                widget.whats_this_action.setShortcut(QKeySequence())
            self._documents[key, profile] = widget
            if not widget.property("workspaceTracked"):
                widget.finished.connect(lambda result, document=widget: self.finish_document(document))
                widget.setProperty("workspaceTracked", True)
            widget.setProperty("workspaceIcon", icon_name or DOCUMENT_ICONS.get(key, "file-text"))
            index = self.documents.addTab(widget, icon(widget.property("workspaceIcon")), title)
        if self.documents.currentIndex() == index:
            self.document_changed(index)
        else:
            # currentChanged already updates the parameter panel.
            self.documents.setCurrentIndex(index)
        widget.show()
        self.schedule_refresh()

    def close_document(self, index):
        widget = self.documents.widget(index)
        if widget is None:
            return
        if widget is self.table_page:
            self.finish_document(widget)
            return
        if getattr(widget, "worker", None) and widget.worker.isRunning():
            return
        self.restore_properties()
        if widget.close():
            self.finish_document(widget)
        elif not self._closing_documents:
            self.document_changed(self.documents.currentIndex())

    def close_documents(self, keep=None):
        """Close a snapshot of worksheets, respecting each document's veto."""
        if self._closing_documents:
            return
        if keep is not None and (not isValid(keep) or self.documents.indexOf(keep) < 0):
            return
        widgets = [self.documents.widget(index) for index in range(self.documents.count())]
        self.restore_properties()
        blocker = QSignalBlocker(self.documents)
        self._closing_documents = True
        try:
            for widget in reversed(widgets):
                if widget is not keep and isValid(widget):
                    self.close_document(self.documents.indexOf(widget))
            if keep is not None and self.documents.indexOf(keep) >= 0:
                self.documents.setCurrentWidget(keep)
        finally:
            self._closing_documents = False
            del blocker
            self.document_changed(self.documents.currentIndex())

    def create_document_menu(self, widget):
        menu = QMenu(self.documents.tabBar())
        close = menu.addAction(icon('x'), tr("Закрыть"))
        close.setEnabled(widget is not None)
        def close_selected():
            if widget is not None and isValid(widget):
                self.close_document(self.documents.indexOf(widget))
        close.triggered.connect(close_selected)
        others = menu.addAction(icon('files'), tr("Закрыть остальные"))
        others.setEnabled(widget is not None and self.documents.count() > 1)
        others.triggered.connect(lambda: self.close_documents(keep=widget))
        all_documents = menu.addAction(icon('x'), tr("Закрыть все"))
        all_documents.setEnabled(self.documents.count() > 0)
        all_documents.triggered.connect(lambda: self.close_documents())
        return menu

    def document_menu(self, position):
        bar = self.documents.tabBar()
        widget = self.documents.widget(bar.tabAt(position)) or self.documents.currentWidget()
        menu = self.create_document_menu(widget)
        try:
            menu.exec(bar.mapToGlobal(position))
        finally:
            menu.deleteLater()

    def finish_document(self, widget):
        if not isValid(widget):
            return
        index = self.documents.indexOf(widget)
        if index >= 0:
            self.restore_properties()
            blocker = QSignalBlocker(self.documents)
            self.documents.removeTab(index)
            widget.hide()
            del blocker
            if not self._closing_documents:
                self.document_changed(self.documents.currentIndex())

    def open_table(self):
        """Restore the worksheet view while retaining its model and selection."""
        blocker = QSignalBlocker(self.documents)
        if self.documents.indexOf(self.table_page) < 0:
            self.documents.insertTab(0, self.table_page, icon("table-2"), tr("Таблица"))
        self.documents.setCurrentWidget(self.table_page)
        self.table_page.show()
        del blocker
        self.document_changed(self.documents.currentIndex())

    def clear_documents(self):
        self.restore_properties()
        blocker = QSignalBlocker(self.documents)
        for index in range(self.documents.count() - 1, -1, -1):
            if self.documents.widget(index) is not self.table_page:
                self.documents.removeTab(index)
        self._documents.clear()
        self._signature = None
        del blocker

    def restore_properties(self):
        if self._borrowed is not None:
            tabs, splitter, document = self._borrowed
            if isValid(tabs) and isValid(splitter):
                self._ignore_document_size(document)
                self.properties_stack.removeWidget(tabs)
                splitter.insertWidget(0, tabs)
            self._borrowed = None
        self.properties_stack.setCurrentWidget(self.properties)

    def _set_document_dock_widths(self, widget):
        is_plot = widget is not None and widget is self.window.plot_window
        is_interpretation = widget is not None and widget is self.window.interpretation_window
        is_qc = widget is not None and widget is self.window.qc_window
        is_interpolation = widget is not None and widget is self.window.interpolation_window
        compact_document = widget is not None and any(widget is getattr(self.window, name, None) for name in (
            "map_window", "statistics_window", "geochemistry_workspace_window", "stratigraphy_window",
            "curve_fitting_window",
        ))
        if compact_document:
            self.project_dock.setMinimumWidth(165)
            self.properties_dock.setMinimumWidth(270)
            return
        # Leave room for the plot and results on a 1024 px workspace with
        # both docks open. Keep wider default widths for other documents.
        self.project_dock.setMinimumWidth(205 if is_interpretation or is_qc or is_interpolation else 235)
        self.properties_dock.setMinimumWidth(315 if is_plot or is_interpretation or is_qc or is_interpolation else 365)

    @staticmethod
    def _ignore_document_size(page):
        # The tab stack would otherwise keep the minimum size of hidden sheets,
        # including their parameter panels when those return from the dock.
        if not hasattr(page, "_workspace_size_policy"):
            page._workspace_size_policy = page.sizePolicy()
            page.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def document_changed(self, index):
        if not hasattr(self, "properties_stack"):
            return
        widget = self.documents.widget(index)
        for position in range(self.documents.count()):
            page = self.documents.widget(position)
            if page is not widget:
                self._ignore_document_size(page)
        self.restore_properties()
        self._set_document_dock_widths(widget)
        if hasattr(widget, "refresh_columns"):
            widget.refresh_columns()
        if hasattr(widget, "on_activated"):
            widget.on_activated()
        tabs = getattr(widget, "parameter_tabs", None)
        if tabs is None and widget is not None and widget in (self.window.plot_window, self.window.map_window):
            tabs = widget.tabs
        if tabs is not None:
            splitter = tabs.parentWidget()
            if isinstance(splitter, QSplitter):
                self._borrowed = tabs, splitter, widget
                self.properties_stack.addWidget(tabs)
                self.properties_stack.setCurrentWidget(tabs)
                tabs.show()
        if widget is not None and hasattr(widget, "_workspace_size_policy"):
            widget.setSizePolicy(widget._workspace_size_policy)
            del widget._workspace_size_policy
        if hasattr(self.window, "model"):
            if widget is self.table_page:
                self.show_object_properties("table")
            else:
                self.properties.show_object()
        self.close_action.setEnabled(widget is not None)
        self.sync_selection()

    def active_payload(self, selected):
        widget = self.documents.currentWidget()
        if widget is None:
            return ("project", "")
        if widget is self.table_page:
            return selected if selected[0] in ("column", "table", "project", "view") else ("table", "")
        payload = next((key for key, document in self._documents.items() if document is widget), None)
        if payload is None:
            return selected
        kind, _ = payload
        if (kind == "plot" and selected[0] == "saved_plot"
                and selected[1] in self.window.model.project.view.get('plots', {})):
            return selected
        if (kind == 'map' and selected[0] == 'map_layer'
                and any(layer['id'] == selected[1] for layer in self.window.model.project.view.get('map_layers', []))):
            return selected
        return payload

    def sync_selection(self):
        if not hasattr(self, "tree"):
            return
        current = self.tree.currentItem()
        payload = self.active_payload(current.data(0, Qt.ItemDataRole.UserRole) if current else ("project", ""))
        def visit(item):
            if item.data(0, Qt.ItemDataRole.UserRole) == payload:
                blocker = QSignalBlocker(self.tree)
                self.tree.setCurrentItem(item)
                del blocker
                return True
            return any(visit(item.child(i)) for i in range(item.childCount()))
        for index in range(self.tree.topLevelItemCount()):
            if visit(self.tree.topLevelItem(index)):
                break

    def refresh_theme(self):
        for index in range(self.documents.count()):
            widget = self.documents.widget(index)
            name = "table-2" if widget is self.table_page else widget.property("workspaceIcon") or "file-text"
            self.documents.setTabIcon(index, icon(name))
        for button, name in zip(self.welcome_buttons, ("document-import", "folder-open", "database", "flask-conical")):
            button.setIcon(icon(name))
        self._signature = None
        self.refresh_project()

    def bind_project(self):
        self.view_properties.clear_target()
        self.window.table.selectionModel().currentColumnChanged.connect(self.column_selected)
        self.open_table()
        self.table_properties._schema = None
        self.show_object_properties("table")
        self.refresh_project()

    def column_selected(self, current, previous):
        if self.documents.currentWidget() is self.table_page and current.isValid():
            column = self.window.model.project.columns[current.column()]
            self.show_object_properties("column", column.id)

    def show_object_properties(self, kind="project", key=""):
        if kind in ("table", "column", "view"):
            self.table_properties.summary.show_object("column" if kind == "column" else "table", key)
            self.table_properties.refresh()
            self.properties_stack.setCurrentWidget(self.table_properties)
        else:
            self.properties.show_object()
            self.properties_stack.setCurrentWidget(self.properties)

    def show_table_properties(self, page):
        self.open_table()
        self.show_object_properties("table")
        self.table_properties.select_page(page)
        self.properties_dock.show()
        self.properties_dock.raise_()
        self.tree.setCurrentItem(self.tree.topLevelItem(0).child(0))

    def show_view_properties(self, name):
        self.view_properties.edit(name)
        self.properties_stack.setCurrentWidget(self.view_properties)
        self.properties_dock.show()
        self.properties_dock.raise_()

    def finish_view_properties(self):
        self.document_changed(self.documents.currentIndex())

    def schedule_refresh(self):
        self._refresh_timer.start()

    def refresh_project(self):
        project = self.window.model.project
        self.properties.show_object(self.properties.kind, self.properties.column_id or "")
        summary = self.table_properties.summary
        summary.show_object(summary.kind, summary.column_id or "")
        self.table_properties.refresh()
        self.view_properties.refresh()
        self.table_page.setCurrentIndex(0 if project.columns else 1)
        documents = self.window.analysis_documents
        visible = [
            descriptor for descriptor in DOCUMENTS.values()
            if documents.cached(descriptor) is not None or any(project.view.get(key) for key in descriptor.view_keys)
        ]
        configured = set(project.view.get("nonpyro_configs", {}))
        legacy = project.view.get("nonpyro_current", {}).get("method")
        visible.extend(
            descriptor for descriptor in geochemistry_documents()
            if (documents.cached(descriptor) is not None
                or configured.intersection(descriptor.methods) or legacy in descriptor.methods)
        )
        signature = (project.id, str(self.window.path), tuple((c.id, c.name, c.kind, c.unit) for c in project.columns),
                     tuple(project.view.get("plots", {})), tuple(project.view.get("named", {})),
                     tuple(descriptor.payload for descriptor in visible),
                     tuple((layer["id"], layer["name"]) for layer in project.view.get("map_layers", [])))
        if signature == self._signature:
            return
        self._signature = signature
        current = self.tree.currentItem()
        selected = self.active_payload(current.data(0, Qt.ItemDataRole.UserRole) if current else ("project", ""))
        blocker = QSignalBlocker(self.tree)
        self.tree.clear()

        def item(parent, name, kind, key="", type_name="", *, icon_name=None):
            node = QTreeWidgetItem(parent, [name, type_name])
            node.setToolTip(0, name)
            node.setData(0, Qt.ItemDataRole.UserRole, (kind, key))
            node.setIcon(0, icon(icon_name or DOCUMENT_ICONS.get(kind, "file-text")))
            if (kind, key) == selected:
                self.tree.setCurrentItem(node)
            return node

        root = item(self.tree, self.window.path.name if self.window.path else tr("Новый проект"), "project", type_name=tr("Проект"))
        table = item(root, tr("Таблица"), "table", type_name=tr("Таблица"))
        for column in project.columns:
            item(table, column.name, "column", column.id, field_kind_label(column.kind))
        sheets = item(root, tr("Листы"), "folder")
        for descriptor in visible:
            node = item(sheets, descriptor.title(), *descriptor.payload, icon_name=descriptor.icon)
            if descriptor.kind == 'map':
                for layer in project.view.get('map_layers', []):
                    item(node, layer['name'], 'map_layer', layer['id'], tr('Слой'))
        for name in project.view.get("plots", {}):
            item(sheets, name, "saved_plot", name, tr("График"))
        if project.view.get("named"):
            views = item(root, tr("Представления"), "folder", "views")
            for name in project.view["named"]:
                item(views, name, "view", name)
            views.setExpanded(True)
        root.setExpanded(True)
        table.setExpanded(True)
        sheets.setExpanded(True)
        del blocker
        self.filter_tree(self.search.text())

    def filter_tree(self, text):
        needle = text.casefold()

        def visit(item):
            children = [visit(item.child(i)) for i in range(item.childCount())]
            match = not needle or needle in item.text(0).casefold() or any(children)
            item.setHidden(not match)
            if needle and match:
                item.setExpanded(True)
            return match

        for index in range(self.tree.topLevelItemCount()):
            visit(self.tree.topLevelItem(index))

    def object_selected(self, current, previous):
        if current is None:
            return
        kind, key = current.data(0, Qt.ItemDataRole.UserRole)
        self.open_object(kind, key)

    def open_object(self, kind, key=''):
        window = self.window
        if kind == 'column' and not any(column.id == key for column in window.model.project.columns):
            return
        if kind in ('saved_plot', 'view') and key not in window.model.project.view.get('plots' if kind == 'saved_plot' else 'named', {}):
            return
        if kind in ("column", "table"):
            self.open_table()
            self.show_object_properties(kind, key)
            if kind == "column" and window.proxy.rowCount():
                column = next(i for i, c in enumerate(window.model.project.columns) if c.id == key)
                window.table.setCurrentIndex(window.proxy.index(0, column))
        elif kind == 'saved_plot':
            from autopyro.visualization.plotting import PlotSpec
            window.show_plots()
            window.plot_window.spec = PlotSpec.from_dict(window.model.project.view["plots"][key])
            window.plot_window.load_controls()
            window.plot_window.schedule()
        elif kind == 'map_layer':
            from .project_menu import open_map_layer
            open_map_layer(self, key)
        elif descriptor := document_descriptor(kind, key):
            descriptor.command(window)()
        elif kind == "view":
            self.open_table()
            window.restore_view(window.model.project.view["named"][key])
            window.model_changed()
            self.show_object_properties("view")
            self.table_properties.select_page("views")
            self.table_properties.saved_views.setCurrentText(key)
        else:
            self.properties_stack.setCurrentWidget(self.properties)
            self.properties.show_object()

    def create_project_menu(self, payload):
        from .project_menu import ExplorerMenu
        return ExplorerMenu(self, payload)

    def project_menu(self, position):
        item = self.tree.itemAt(position)
        payload = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else ('project', '')
        if item is not None:
            blocker = QSignalBlocker(self.tree)
            self.tree.setCurrentItem(item)
            del blocker
        menu = self.create_project_menu(payload)
        try:
            menu.exec(self.tree.viewport().mapToGlobal(position))
        finally:
            menu.deleteLater()

    def save_layout(self):
        store = i18n.settings()
        store.setValue("workspace/geometry", self.window.saveGeometry())
        store.setValue("workspace/state", self.window.saveState(LAYOUT_VERSION))

    def restore_layout(self):
        store = i18n.settings()
        geometry, state = store.value("workspace/geometry"), store.value("workspace/state")
        if isinstance(geometry, QByteArray):
            self.window.restoreGeometry(geometry)
        if isinstance(state, QByteArray):
            self.window.restoreState(state, LAYOUT_VERSION)
        # Older layouts may contain the visibility dock, now embedded in the
        # table properties. Keep its model owner alive without a second panel.
        self.window.column_dock.hide()
