"""Non-modal table properties for the scientific desktop workspace."""

from copy import deepcopy

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLayout, QLineEdit, QListView, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QScrollArea, QTabWidget, QVBoxLayout, QWidget,
)

from autopyro.model.saved_views import create_view
from .filter_rules import OPERATORS, VALUELESS_OPERATORS, validated_rules
from ..i18n import message, tr
from ..widgets.parameters import annotate_parameters
from ..widgets.parameter_reset import install_parameter_resets
from ..theme.controls import add_search_icon
from ..theme import icon


class FilterRule(QGroupBox):
    def __init__(self, columns, rule, remove):
        super().__init__(tr("Условие"))
        form = QFormLayout(self)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.column = QComboBox()
        for column in columns:
            self.column.addItem(column.name, column.id)
        self.column.parameter_default_state = ('combo', self.column.itemData(0), self.column.itemText(0))
        if rule.get("column"):
            index = self.column.findData(rule["column"])
            if index < 0:
                index = self.column.count()
                label = rule.get("column_name", rule["column"])
                self.column.addItem(tr('Колонка недоступна: {0}', label), rule["column"])
                self.column.setItemData(index, rule["column"], Qt.ItemDataRole.ToolTipRole)
                self.column.setItemData(index, label, Qt.ItemDataRole.UserRole + 1)
            self.column.setCurrentIndex(index)
        self.operator = QComboBox()
        for label, value in OPERATORS:
            self.operator.addItem(tr(label), value)
        self.operator.parameter_default_state = ('combo', self.operator.itemData(0), self.operator.itemText(0))
        self.operator.setCurrentIndex(max(0, self.operator.findData(rule.get("operator"))))
        self.value = QLineEdit(str(rule.get("value", "")))
        self.value.parameter_default_state = ('text', '')
        self.negate = QCheckBox("NOT")
        self.negate.parameter_default_state = ('check', False)
        self.negate.setChecked(rule.get("negate", False))
        form.addRow(tr("Колонка"), self.column)
        form.addRow(tr("Условие"), self.operator)
        form.addRow(tr("Значение"), self.value)
        form.addRow(self.negate)
        button = QPushButton(icon('trash'), tr("Удалить условие"))
        button.clicked.connect(lambda: remove(self))
        form.addRow(button)
        self.operator.currentIndexChanged.connect(self.update_enabled)
        self.update_enabled()
        annotate_parameters(self)
        install_parameter_resets(self)
        annotate_parameters(self)

    def update_enabled(self):
        self.value.setEnabled(self.operator.currentData() not in VALUELESS_OPERATORS)

    def draft(self, *, include_names=False):
        draft = {"column": self.column.currentData(), "operator": self.operator.currentData(),
                 "value": self.value.text(), "negate": self.negate.isChecked()}
        if include_names:
            draft["column_name"] = self.column.currentData(Qt.ItemDataRole.UserRole + 1) or self.column.currentText()
        return draft


class FilterGroup(QGroupBox):
    def __init__(self, columns, rule, remove):
        super().__init__(tr("Группа условий"))
        self.columns = columns
        self.rules = []
        layout = QVBoxLayout(self)
        self.combine = QComboBox()
        self.combine.addItems(["AND", "OR"])
        self.combine.setCurrentText(rule.get("combine", "AND"))
        self.negate = QCheckBox("NOT")
        self.negate.setChecked(rule.get("negate", False))
        controls = QHBoxLayout()
        controls.addWidget(self.combine)
        controls.addWidget(self.negate)
        layout.addLayout(controls)
        self.rules_layout = QVBoxLayout()
        layout.addLayout(self.rules_layout)
        for child in rule["rules"]:
            self.add_rule(child)
        for image, label, action in (
            ('plus', tr("Добавить условие"), lambda: self.add_rule({})),
            ('list-plus', tr("Добавить группу"), lambda: self.add_rule({"rules": []})),
            ('trash', tr("Удалить группу"), lambda: remove(self)),
        ):
            button = QPushButton(icon(image), label)
            button.clicked.connect(action)
            layout.addWidget(button)

    def add_rule(self, rule):
        widget = filter_widget(self.columns, rule, self.remove_rule)
        self.rules.append(widget)
        self.rules_layout.addWidget(widget)

    def remove_rule(self, widget):
        self.rules.remove(widget)
        self.rules_layout.removeWidget(widget)
        widget.hide()
        widget.deleteLater()

    def draft(self, *, include_names=False):
        return {"combine": self.combine.currentText(), "negate": self.negate.isChecked(),
                "rules": [rule.draft(include_names=include_names) for rule in self.rules]}


def filter_widget(columns, rule, remove):
    editor = FilterGroup if "rules" in rule else FilterRule
    return editor(columns, rule, remove)


class TableProperties(QTabWidget):
    def __init__(self, window, summary):
        super().__init__()
        self.window = window
        self.summary = summary
        self._schema = None
        self._filter_state = None
        self._sort_state = None
        self._view_names = None
        self.rules = []
        self.pages = {}
        general = self.page("table", tr("Таблица"), 'table-2')
        general.addWidget(summary)
        self.reset_button = self.button(general, tr("Сбросить параметры представления"), self.reset_view, 'rotate-ccw')

        filters = self.page("filter", tr("Фильтр"), 'funnel')
        self.combine = QComboBox()
        self.combine.addItems(["AND", "OR"])
        self.combine.setWhatsThis(tr("AND требует выполнения всех условий, OR — хотя бы одного. NOT инвертирует условие или группу."))
        filters.addWidget(self.combine)
        self.rules_layout = QVBoxLayout()
        filters.addLayout(self.rules_layout)
        self.add_rule_button = self.button(filters, tr("Добавить условие"), self.add_rule, 'plus')
        self.add_group_button = self.button(filters, tr("Добавить группу"), lambda: self.add_rule({"rules": []}), 'list-plus')
        self.filter_error = QLabel()
        self.filter_error.setWordWrap(True)
        filters.addWidget(self.filter_error)
        self.apply_filter_button = self.button(filters, tr("Применить фильтр"), self.apply_filter, 'funnel')
        self.button(filters, tr("Сбросить фильтр"), self.reset_filter, 'funnel-x')

        sorting = self.page("sort", tr("Сортировка"), 'arrow-down-wide-narrow')
        hint = QLabel(tr('Выберите колонки по приоритету; порядок меняется перетаскиванием.\nДвойной щелчок меняет направление сортировки колонки.'))
        hint.setWordWrap(True)
        sorting.addWidget(hint)
        self.sort_columns = QListWidget()
        self.sort_columns.setMinimumHeight(180)
        self.sort_columns.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.sort_columns.itemDoubleClicked.connect(self.reverse_sort)
        self.sort_columns.setWhatsThis(hint.text())
        sorting.addWidget(self.sort_columns)
        order = QHBoxLayout()
        for text, offset, image in (("↑", -1, 'upload'), ("↓", 1, 'download')):
            button = QPushButton(icon(image), text)
            button.clicked.connect(lambda checked=False, step=offset: self.move_sort(step))
            order.addWidget(button)
        sorting.addLayout(order)
        self.apply_sort_button = self.button(sorting, tr("Применить сортировку"), self.apply_sort, 'arrow-down-wide-narrow')
        self.button(sorting, tr("Сбросить сортировку"), self.reset_sort, 'rotate-ccw')

        columns = self.page("columns", tr("Видимые колонки"), 'columns-3')
        self.column_search = QLineEdit()
        self.column_search.setPlaceholderText(tr("Поиск колонки…"))
        add_search_icon(self.column_search)
        self.column_search.textChanged.connect(self.filter_columns)
        columns.addWidget(self.column_search)
        self.columns = QListView()
        # Share check states with the existing table visibility controller.
        self.columns.setModel(window.columns.model())
        self.columns.setMinimumHeight(180)
        columns.addWidget(self.columns)
        self.button(columns, tr("Показать все"), lambda: window.toggle_all(True), 'eye')
        self.button(columns, tr("Скрыть все"), lambda: window.toggle_all(False), 'eye-off')

        views = self.page("views", tr("Представления"), 'layout-template')
        views.addWidget(QLabel(tr("Название")))
        self.view_name = QLineEdit()
        views.addWidget(self.view_name)
        self.save_view_button = self.button(views, tr("Сохранить представление"), self.save_view, 'save')
        self.saved_views = QComboBox()
        views.addWidget(self.saved_views)
        self.load_view_button = self.button(views, tr("Применить представление"), self.load_view, 'layout-template')
        self.view_status = QLabel()
        self.view_status.setWordWrap(True)
        views.addWidget(self.view_status)
        for key, page in self.pages.items():
            page.widget().layout().addStretch()
        annotate_parameters(self)
        install_parameter_resets(self)
        annotate_parameters(self)

    def page(self, key, title, icon_name):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        group = QGroupBox(title)
        layout.addWidget(group)
        parameters = QVBoxLayout(group)
        scroll.setWidget(content)
        self.pages[key] = scroll
        self.addTab(scroll, icon(icon_name), title)
        return parameters

    def button(self, layout, text, callback, icon_name):
        button = QPushButton(icon(icon_name), text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def select_page(self, key):
        self.setCurrentWidget(self.pages[key])

    def filter_source(self):
        return self.window.proxy.rules, self.window.proxy.combine

    def sort_source(self):
        proxy = self.window.proxy
        return proxy.sort_rules, proxy.sortColumn(), proxy.sortOrder()

    def refresh(self):
        project = self.window.model.project
        schema = (self.window.model, tuple((c.id, c.name, c.kind) for c in project.columns))
        changed = schema != self._schema
        same_model = self._schema is not None and schema[0] is self._schema[0]
        if changed:
            if not same_model:
                self.view_name.clear()
                self.view_status.clear()
                self.column_search.clear()
            self._schema = schema
        filter_state = self.filter_source()
        if changed or filter_state != self._filter_state:
            rules, combine = filter_state
            if same_model and changed and filter_state == self._filter_state:
                rules = [rule.draft(include_names=True) for rule in self.rules]
                combine = self.combine.currentText()
            self.clear_rules()
            self.combine.setCurrentText(combine)
            for rule in rules:
                self.add_rule(rule)
            self.filter_error.clear()
            self._filter_state = deepcopy(filter_state)
        sort_state = self.sort_source()
        if changed or sort_state != self._sort_state:
            self.load_sort()
            self._sort_state = deepcopy(sort_state)
        self.add_rule_button.setEnabled(bool(project.columns))
        self.add_group_button.setEnabled(bool(project.columns))
        self.apply_filter_button.setEnabled(bool(project.columns))
        names = tuple(project.view.get("named", {}))
        if changed or names != self._view_names:
            selected = self.saved_views.currentText()
            blocker = QSignalBlocker(self.saved_views)
            self.saved_views.clear()
            self.saved_views.addItems(names)
            if selected in names:
                self.saved_views.setCurrentText(selected)
            del blocker
            self._view_names = names
        self.load_view_button.setEnabled(bool(names))
        self.filter_columns(self.column_search.text())

    def add_rule(self, rule=None):
        if not self.window.model.project.columns and not rule:
            return
        widget = filter_widget(self.window.model.project.columns, rule if isinstance(rule, dict) else {}, self.remove_rule)
        self.rules.append(widget)
        self.rules_layout.addWidget(widget)

    def remove_rule(self, widget):
        self.rules.remove(widget)
        self.rules_layout.removeWidget(widget)
        widget.hide()
        widget.deleteLater()

    def clear_rules(self):
        for widget in list(self.rules):
            self.remove_rule(widget)

    def apply_filter(self):
        self.refresh()
        try:
            rules = validated_rules(self.window.model.project, [rule.draft(include_names=True) for rule in self.rules])
        except ValueError as exc:
            self.filter_error.setText(message(exc))
            return
        self.window.apply_filter(rules, self.combine.currentText())
        self.filter_error.clear()
        self.refresh()

    def reset_filter(self):
        self.clear_rules()
        self.combine.setCurrentText("AND")
        self.window.apply_filter([], "AND")
        self.filter_error.clear()
        self.refresh()

    def load_sort(self):
        self.sort_columns.clear()
        project = self.window.model.project
        rules, column, order = self.sort_source()
        selected = list(rules)
        if not selected and 0 <= column < len(project.columns):
            selected = [(project.columns[column].id, order == Qt.SortOrder.DescendingOrder)]
        directions = dict(selected)
        ids = [cid for cid, descending in selected]
        ids += [column.id for column in project.columns if column.id not in ids]
        by_id = {column.id: column for column in project.columns}
        for cid in ids:
            if cid not in by_id:
                continue
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, cid)
            item.setData(Qt.ItemDataRole.UserRole + 1, directions.get(cid, False))
            item.setData(Qt.ItemDataRole.UserRole + 2, by_id[cid].name)
            self.sort_label(item)
            item.setCheckState(Qt.CheckState.Checked if cid in directions else Qt.CheckState.Unchecked)
            self.sort_columns.addItem(item)

    def sort_label(self, item):
        item.setText(item.data(Qt.ItemDataRole.UserRole + 2) + (" ↓" if item.data(Qt.ItemDataRole.UserRole + 1) else " ↑"))

    def reverse_sort(self, item):
        item.setData(Qt.ItemDataRole.UserRole + 1, not item.data(Qt.ItemDataRole.UserRole + 1))
        self.sort_label(item)

    def move_sort(self, step):
        row = self.sort_columns.currentRow()
        target = row + step
        if row >= 0 and 0 <= target < self.sort_columns.count():
            item = self.sort_columns.takeItem(row)
            self.sort_columns.insertItem(target, item)
            self.sort_columns.setCurrentItem(item)

    def apply_sort(self):
        self.refresh()
        rules = [(item.data(Qt.ItemDataRole.UserRole), item.data(Qt.ItemDataRole.UserRole + 1))
                            for i in range(self.sort_columns.count()) if (item := self.sort_columns.item(i)).checkState() == Qt.CheckState.Checked]
        self.window.table_filter_sort.apply_sort(rules)
        self.refresh()

    def reset_sort(self):
        self.window.table_filter_sort.apply_sort([])
        self.load_sort()
        self.refresh()

    def filter_columns(self, text):
        for index in range(self.window.columns.count()):
            self.columns.setRowHidden(index, text.casefold() not in self.window.columns.item(index).text().casefold())

    def save_view(self):
        self.view_status.clear()
        name = self.view_name.text().strip()
        project = self.window.model.project
        settings = self.window.capture_view(include_sample_set=True)
        views = project.view.get("named", {})
        if name in views:
            answer = QMessageBox.question(
                self, tr('Сохранить представление'), tr('Заменить представление «{0}»?', name),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes or self.window.model.project is not project:
                return
            views[name] = deepcopy(settings)
        else:
            try:
                create_view(project.view, name, settings)
            except ValueError as exc:
                self.view_status.setText(message(exc))
                return
        self.window.view_changed()
        self.refresh()
        self.saved_views.setCurrentText(name)
        self.view_status.setText(tr("Представление сохранено"))

    def load_view(self):
        name = self.saved_views.currentText()
        views = self.window.model.project.view.get("named", {})
        if name in views:
            self.window.restore_view(views[name])
            self.window.view_changed()
            self.refresh()
            self.view_status.clear()

    def reset_view(self):
        self.window.restore_view_defaults()

    def reload_view(self):
        self._filter_state = None
        self._sort_state = None
        self.refresh()
