"""Database connection section of application settings."""

from PySide6.QtWidgets import (
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from ...ui.i18n import tr
from ...ui.theme import icon


class DatabaseSettingsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        box = QVBoxLayout(self)
        group = QGroupBox(tr("PostgreSQL по умолчанию"))
        form = QFormLayout(group)
        self.host, self.database, self.user, self.password = QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        for field in (self.host, self.database, self.user):
            field.textChanged.connect(self.password.clear)
        self.port.valueChanged.connect(self.password.clear)
        for label, field in ((tr("Сервер"), self.host), (tr("Порт"), self.port), (tr("База"), self.database),
                             (tr("Пользователь"), self.user), (tr("Пароль"), self.password)):
            form.addRow(label, field)
        note = QLabel(tr("Пароль хранится только в памяти текущего сеанса приложения."))
        note.setWordWrap(True)
        form.addRow(note)
        box.addWidget(group)
        group = QGroupBox(tr("SQLite по умолчанию"))
        form = QFormLayout(group)
        self.sqlite_path, self.sqlite_table = QLineEdit(), QLineEdit()
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.sqlite_path)
        choose = QPushButton(icon("folder-open"), tr("Выбрать файл…"))
        choose.clicked.connect(self.choose_sqlite)
        row.addWidget(choose)
        form.addRow(tr("Путь SQLite"), holder)
        form.addRow(tr("Таблица SQLite"), self.sqlite_table)
        box.addWidget(group)
        note = QLabel(tr("Параметры подставляются в новые диалоги. Источники открытых проектов не меняются."))
        note.setWordWrap(True)
        box.addWidget(note)
        box.addStretch()

    def choose_sqlite(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("Файл SQLite"), self.sqlite_path.text(),
                                             tr("SQLite (*.sqlite *.sqlite3 *.db);;Все файлы (*)"))
        if path:
            self.sqlite_path.setText(path)

    def populate(self, preferences, password):
        self.host.setText(preferences.database_host)
        self.port.setValue(preferences.database_port)
        self.database.setText(preferences.database_name)
        self.user.setText(preferences.database_user)
        self.password.setText(password)
        self.sqlite_path.setText(preferences.sqlite_path)
        self.sqlite_table.setText(preferences.sqlite_table)

    def values(self):
        return {
            "database_host": self.host.text(), "database_port": self.port.value(),
            "database_name": self.database.text(), "database_user": self.user.text(),
            "sqlite_path": self.sqlite_path.text(), "sqlite_table": self.sqlite_table.text(),
        }
