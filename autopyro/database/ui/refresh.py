"""Compare a new read-only snapshot with its baseline and local edits."""


from ...ui.i18n import tr, message
import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..sources import descriptor, read_snapshot, field_key
from ..refresh import preview_refresh, prepare_refresh, source_row
from ...ui.tasks import TaskDialog
from ..geochem import local_parameters
from ...ui.table import RefreshDatabaseCommand
from ...ui.table.model import snapshot_project
from ...ui.settings import database_parameters
from ...ui.widgets.parameters import add_parameter_tools
from ...ui.theme import icon


STATUSES = {"source_changed": "Изменён источник", "local_only": "Локальная правка", "conflict": "Конфликт",
            "converged": "Изменения совпали", "invalid_source": "Некорректное значение БД"}


ACTIONS = {"unresolved": "Выберите решение", "keep": "Сохранить локальное", "source": "Использовать источник",
           "add": "Добавить пробу", "skip": "Пропустить новую пробу"}


PAGE_SIZE = 100


def key_label(value):
    return " / ".join(str(item) for _, item in json.loads(value))


def display(value):
    return tr('пусто') if value is None else str(value)


class DatabaseRefreshDialog(TaskDialog):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.plan, self.incoming, self.decisions, self.page = None, None, {}, 0
        self.setWindowTitle(tr('Обновление снимка БД · исходный снимок / локальные данные / источник'))
        self.resize(1200, 780)
        layout = QVBoxLayout(self)
        source = descriptor(owner.model.project.source)
        endpoint = source if source["type"] != "postgresql_nonpyro" else source["endpoint"]
        source_label = f"{source['path']} · {source['table']}" if source["type"] == "sqlite" else (
            f"{endpoint['host']}:{endpoint['port']} / {endpoint.get('dbname', endpoint.get('database'))}")
        label = QLabel(tr('Источник только для чтения: {0}\nЛокальные данные не удаляются при исчезновении записи в БД. Пароль остаётся только в памяти подключения.', source_label))
        label.setWordWrap(True)
        layout.addWidget(label)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        settings = QWidget()
        settings_layout = QVBoxLayout(settings)
        form = QFormLayout()
        parameters = database_parameters(local_parameters()) if source["type"] != "sqlite" else {"user": "", "password": ""}
        if source["type"] != "sqlite" and (str(endpoint["host"]), str(endpoint["port"]), endpoint.get("dbname", endpoint.get("database"))) != (
                str(parameters["host"]), str(parameters["port"]), parameters["dbname"]):
            parameters["password"] = ""
        self.user, self.password = QLineEdit(parameters["user"]), QLineEdit(parameters["password"])
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.user.textChanged.connect(self.password.clear)
        for name, widget in ((tr('Пользователь PostgreSQL'), self.user), (tr('Пароль'), self.password)):
            if source["type"] != "sqlite":
                form.addRow(name, widget)
                widget.textChanged.connect(self.invalidate)
            else:
                widget.hide()
        settings_layout.addLayout(form)
        self.keys = QListWidget()
        snapshot = owner.model.project.view.get("database_snapshot", {})
        saved_keys = snapshot.get("keys", [])
        source_ids = {cid for row in owner.model.project.rows if source_row(owner.model.project, row) for cid in row.original}
        for column in owner.model.project.columns:
            if column.id in source_ids:
                item = QListWidgetItem(f"{column.name} [{column.unit or tr('без единицы')}]")
                item.setData(Qt.ItemDataRole.UserRole, column.id)
                item.setCheckState(Qt.CheckState.Checked if column.id in saved_keys else Qt.CheckState.Unchecked)
                self.keys.addItem(item)
        self.keys.itemChanged.connect(self.invalidate)
        if source["type"] == "sqlite":
            settings_layout.addWidget(QLabel(tr('Отметьте поля уникального ключа SQLite. Порядок строк не является ключом.\nПосле применения ключ фиксируется для последующих обновлений.')))
            settings_layout.addWidget(self.keys)
            self.keys.setEnabled(not bool(snapshot))
        else:
            self.keys.hide()
            settings_layout.addWidget(QLabel(tr('Ключ PostgreSQL: source_record_id. Аналитическая группа, фильтры и номер повтора сохраняются.')))
        settings_layout.addStretch()
        tabs.addTab(settings, icon('database'), tr('Источник и ключ'))
        review = QWidget()
        review_layout = QVBoxLayout(review)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([tr('Ключ пробы'), tr('Показатель'), tr('Исходный снимок'), tr('Локально'), tr('В источнике'), tr('Состояние'), tr('Решение')])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        review_layout.addWidget(self.table)
        paging = QHBoxLayout()
        previous, following = QPushButton(icon('undo-2'), tr('Предыдущие 100')), QPushButton(icon('redo-2'), tr('Следующие 100'))
        self.previous, self.following = previous, following
        previous.clicked.connect(lambda: self.change_page(-1))
        following.clicked.connect(lambda: self.change_page(1))
        self.page_label = QLabel()
        for widget in (previous, self.page_label, following):
            paging.addWidget(widget)
        review_layout.addLayout(paging)
        tabs.addTab(review, icon('files'), tr('Трёхстороннее сравнение'))
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        tabs.addTab(self.details, icon('circle-question-mark'), tr('Структура и ошибки'))
        self.schema = QCheckBox(tr('Подтверждаю изменение полей: исчезнувшие сохраняются локально, новые поля пока не импортируются'))
        self.schema.toggled.connect(self.update_ready)
        self.schema.setEnabled(False)
        layout.addWidget(self.schema)
        self.status = QLabel(tr('Снимок ещё не проверен.'))
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        read = QPushButton(icon('files'), tr('Прочитать и сравнить'))
        read.clicked.connect(self.read)
        self.apply_button = QPushButton(icon('refresh-cw'), tr('Применить обновление'))
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.apply_refresh)
        close = QPushButton(icon('x'), tr('Отмена'))
        close.clicked.connect(self.reject)
        for button in (read, self.apply_button, close):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        add_parameter_tools(self, changed=self.invalidate)

    def invalidate(self, *_):
        self.plan = None
        if hasattr(self, "apply_button"):
            self.apply_button.setEnabled(False)
            self.status.setText(tr('Параметры изменены. Выполните сравнение заново.'))

    def read(self):
        self.invalidate()
        project = snapshot_project(self.owner.model.project, self)
        keys = [self.keys.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.keys.count())
                if self.keys.item(i).checkState() == Qt.CheckState.Checked]
        parameters = {"user": self.user.text(), "password": self.password.text()}
        def read_and_compare():
            incoming = read_snapshot(project.source, parameters)
            return incoming, preview_refresh(project, incoming, keys)
        self.run_task(read_and_compare, self.loaded)

    def loaded(self, outcome):
        self.incoming, self.plan = outcome
        self.decisions, self.page = {}, 0
        self.entries = self.plan.cells + self.plan.new_rows + self.plan.missing_rows
        self.schema.setChecked(False)
        self.schema.setEnabled(any(self.plan.schema_changes.values()))
        labels = {field_key(c): c.name for c in self.owner.model.project.columns + self.incoming.columns}
        self.details.setPlainText("\n".join([message(error) for error in self.plan.errors] + [
            tr('Новые поля (не импортируются автоматически): ') + ", ".join(labels.get(key, key) for key in self.plan.schema_changes["added"]),
            tr('Исчезнувшие поля (локальные значения сохраняются): ') + ", ".join(labels.get(key, key) for key in self.plan.schema_changes["removed"])]) )
        self.render_page()
        self.update_ready()

    def update_ready(self, *_):
        if not self.plan:
            return
        pending = sum(self.decisions.get(e["id"], e["action"]) == "unresolved" for e in self.entries)
        self.apply_button.setEnabled(not self.plan.errors and not pending and
            (not any(self.plan.schema_changes.values()) or self.schema.isChecked()))
        self.status.setText(tr('Получено: {0} UTC. Изменений: {1}; новых проб: {2}; исчезнувших: {3}; решений требуется: {4}; ошибок: {5}. Это состояние на момент чтения снимка.', self.plan.checked_at, len(self.plan.cells), len(self.plan.new_rows), len(self.plan.missing_rows), pending, len(self.plan.errors)))

    def render_page(self):
        if not self.plan:
            return
        items = self.entries[self.page * PAGE_SIZE:(self.page + 1) * PAGE_SIZE]
        columns = {c.id: c for c in self.owner.model.project.columns}
        self.table.setRowCount(len(items))
        for i, entry in enumerate(items):
            if "property_id" in entry:
                values = [key_label(entry["key"]), columns[entry["property_id"]].name,
                          display(entry["base"]), display(entry["local"]), display(entry["source"]), tr(STATUSES[entry["status"]])]
                options = ("unresolved", "keep") if entry["status"] == "invalid_source" else ("unresolved", "keep", "source")
            elif entry in self.plan.new_rows:
                summary = "; ".join(f"{columns[cid].name}={display(value)}" for cid, value in list(entry["values"].items())[:4])
                values = [key_label(entry["key"]), tr('Новая проба'), "—", "—", summary,
                          tr('Некорректная строка: ') + ", ".join(entry["invalid"]) if entry["invalid"] else tr('Новая запись')]
                options = ("unresolved", "skip") if entry["invalid"] else ("add", "skip")
            else:
                values = [key_label(entry["key"]), tr('Проба целиком'), tr('Была в снимке'), tr('Сохраняется'), tr('Отсутствует'), tr('Запись исчезла из источника')]
                options = ("unresolved", "keep")
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.table.setItem(i, column, item)
            choice = QComboBox()
            for action in options:
                choice.addItem(tr(ACTIONS[action]), action)
            choice.setCurrentIndex(choice.findData(self.decisions.get(entry["id"], entry["action"])))
            def changed(*_, eid=entry["id"], combo=choice):
                self.decisions[eid] = combo.currentData()
                self.update_ready()
            choice.currentIndexChanged.connect(changed)
            self.table.setCellWidget(i, 6, choice)
        self.table.resizeColumnsToContents()
        for column in (0, 1, 2, 3, 4, 5):
            self.table.setColumnWidth(column, min(self.table.columnWidth(column), 220))
        self.page_label.setText(tr('Страница {0}; изменений всего: {1}', self.page + 1, len(self.entries)))
        self.previous.setEnabled(self.page > 0)
        self.following.setEnabled((self.page + 1) * PAGE_SIZE < len(self.entries))

    def change_page(self, delta):
        self.page += delta
        self.render_page()

    def apply_refresh(self):
        if not self.plan:
            return
        project, incoming, plan = snapshot_project(self.owner.model.project, self), self.incoming, self.plan
        decisions, schema = dict(self.decisions), self.schema.isChecked()
        self.run_task(lambda: prepare_refresh(project, incoming, plan, decisions, accept_schema_changes=schema), self.prepared)

    def prepared(self, outcome):
        result, protocol = outcome
        try:
            self.owner.model.undo.push(RefreshDatabaseCommand(self.owner.model, result, protocol, self.plan.project_digest))
            self.accept()
        except ValueError as exc:
            self.status.setText(message(exc))
            self.apply_button.setEnabled(False)
