from __future__ import annotations

from ...ui.i18n import tr

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from autopyro.model.core import timestamp
from ...ui.tasks import TaskDialog
from ..readers import read_postgres_snapshot, read_sqlite
from ...ui.settings import database_parameters, load_preferences
from ...ui.widgets.parameters import add_parameter_tools
from ...ui.widgets.parameter_reset import parameter_field
from ...ui.theme import icon


class DatabaseDialog(TaskDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr('Снимок источника · только чтение'))
        self.resize(620, 400)
        layout = QVBoxLayout(self)
        self.kind = QComboBox()
        self.kind.addItems(
            [tr('SQLite: таблица / представление'), "PostgreSQL: geochem.source_record"]
        )
        layout.addWidget(self.kind)
        self.form = QFormLayout()
        self.fields = {}
        parameters = database_parameters()
        preferences = load_preferences()
        for key, label, default in [
            ("path", tr('Путь SQLite'), preferences.sqlite_path),
            ("table", tr('Таблица SQLite'), preferences.sqlite_table),
            ("host", tr("Сервер"), parameters["host"]),
            ("port", tr('Порт'), str(parameters["port"])),
            ("dbname", tr('База'), parameters["dbname"]),
            ("user", tr('Пользователь'), parameters["user"]),
            ("password", tr('Пароль'), parameters.get("password", "")),
        ]:
            field = QLineEdit(default)
            if key == "password":
                field.setEchoMode(QLineEdit.EchoMode.Password)
            self.fields[key] = field
            self.form.addRow(label, field)
        for key in ("host", "port", "dbname", "user"):
            self.fields[key].textChanged.connect(self.fields["password"].clear)
        layout.addLayout(self.form)
        layout.addWidget(
            QLabel(
                tr('PostgreSQL: используйте учётную запись только с SELECT (readonly).\nПриложение открывает read-only транзакцию. Пароль в проект не записывается.\nЛимит первой версии — 100 000 строк, без усечения данных.')
            )
        )
        self.load = QPushButton(icon('database'), tr('Прочитать снимок'))
        self.load.clicked.connect(self.read)
        layout.addWidget(self.load)
        self.kind.currentIndexChanged.connect(self.update_fields)
        self.update_fields()
        self.snapshot = None
        add_parameter_tools(self, reset=self.restore_parameters)

    def restore_parameters(self):
        self.parameter_defaults.restore()
        preferences = load_preferences()
        parameters = database_parameters()
        for key, value in {**parameters, "path": preferences.sqlite_path, "table": preferences.sqlite_table}.items():
            if key in self.fields and key != "password":
                self.fields[key].setText(str(value))
        self.fields["password"].setText(parameters.get("password", ""))
        self.snapshot = None
        self.update_fields()

    def update_fields(self):
        sqlite = self.kind.currentIndex() == 0
        for key, widget in self.fields.items():
            visible = sqlite == (key in ("path", "table"))
            widget.setVisible(visible)
            field = parameter_field(widget)
            field.setVisible(visible)
            label = self.form.labelForField(field)
            if label is None:
                label = field.parentWidget().layout().labelForField(field)
            label.setVisible(visible)

    def read(self):
        fields = {k: v.text() for k, v in self.fields.items()}
        if self.kind.currentIndex() == 0:
            source = {
                "type": "sqlite",
                "path": str(Path(fields["path"]).resolve()),
                "table": fields["table"],
                "read_only": True,
                "imported_at": timestamp(),
            }
            function = lambda: (*read_sqlite(fields["path"], fields["table"]), source)
        else:
            from psycopg.conninfo import make_conninfo

            dsn = make_conninfo(
                **{k: v for k, v in fields.items() if k not in ("path", "table")}
            )
            source = {k: fields[k] for k in ("host", "port", "dbname")}
            source.update(
                type="postgresql",
                read_only=True,
                imported_at=timestamp(),
                relation="geochem.source_record.raw_cells",
            )
            function = lambda: (*read_postgres_snapshot(dsn), source)
        self.run_task(function, self.loaded)

    def loaded(self, snapshot):
        headers, rows, numbers, source = snapshot
        self.snapshot = (headers, rows, numbers, {**source, "source_columns": list(headers)})
        self.accept()
