"""Application-wide preferences, edited as a draft until accepted."""

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...database.ui.settings import DatabaseSettingsWidget
from .. import i18n
from ..i18n import tr, message
from ..settings import (
    CSV_DELIMITERS,
    CSV_ENCODINGS,
    Preferences,
    database_parameters,
    load_preferences,
    save_preferences,
)
from ..widgets.parameters import add_parameter_tools, ParameterDefaults
from ..theme import icon


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Настройки приложения"))
        self.resize(660, 560)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        general = QWidget()
        form = QFormLayout(general)
        self.language = QComboBox()
        for code, name in i18n.available_languages().items():
            self.language.addItem(name, code)
        form.addRow(tr("Язык интерфейса"), self.language)
        self.theme = QComboBox()
        self.theme.addItem(tr("Светлая — KDE Breeze"), "breeze_light")
        self.theme.addItem(tr("Тёмная — KDE Breeze"), "breeze_dark")
        form.addRow(tr("Оформление"), self.theme)
        note = QLabel(tr("Изменение языка применяется после перезапуска приложения."))
        note.setWordWrap(True)
        form.addRow(note)
        self.autosave_enabled = QCheckBox(tr("Автосохранение проекта"))
        form.addRow(self.autosave_enabled)
        self.autosave_minutes = QSpinBox()
        self.autosave_minutes.setRange(1, 120)
        form.addRow(tr("Интервал автосохранения, мин"), self.autosave_minutes)
        self.autosave_enabled.toggled.connect(self.autosave_minutes.setEnabled)
        tabs.addTab(general, icon('settings-2'), tr("Общие"))

        imports = QWidget()
        form = QFormLayout(imports)
        self.delimiter = QComboBox()
        for delimiter in CSV_DELIMITERS:
            self.delimiter.addItem(tr("Табуляция") if delimiter == "\t" else delimiter, delimiter)
        self.encoding = QComboBox()
        self.encoding.addItems(CSV_ENCODINGS)
        self.decimal = QComboBox()
        for label, value in ((tr("Точка или запятая"), "auto"), (tr("Точка"), "."), (tr("Запятая"), ",")):
            self.decimal.addItem(label, value)
        form.addRow(tr("Разделитель"), self.delimiter)
        form.addRow(tr("Кодировка"), self.encoding)
        form.addRow(tr("Десятичный знак"), self.decimal)
        note = QLabel(tr("Параметры применяются к новым диалогам импорта; шаблон может их переопределить."))
        note.setWordWrap(True)
        form.addRow(note)
        tabs.addTab(imports, icon('import'), tr("Импорт по умолчанию"))

        self.database_settings = DatabaseSettingsWidget()
        # Preserve field access for callers of the application settings dialog.
        for name in ("host", "port", "database", "user", "password", "sqlite_path", "sqlite_table"):
            setattr(self, name, getattr(self.database_settings, name))
        tabs.addTab(self.database_settings, icon('database'), tr("Подключение к БД"))
        self.error = QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
                                        | QDialogButtonBox.StandardButton.RestoreDefaults)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr("Сохранить"))
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr("Отмена"))
        self.buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).setText(tr("По умолчанию"))
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(self.restore_defaults)
        layout.addWidget(self.buttons)
        self.populate(load_preferences(use_saved=False), "")
        defaults = ParameterDefaults(self)
        self.populate(load_preferences(), database_parameters().get("password", ""))
        add_parameter_tools(self, defaults=defaults, reset=self.restore_defaults)

    def populate(self, preferences, password):
        self.language.setCurrentIndex(self.language.findData(preferences.language))
        self.theme.setCurrentIndex(self.theme.findData(preferences.theme))
        self.autosave_enabled.setChecked(preferences.autosave_enabled)
        self.autosave_minutes.setValue(preferences.autosave_minutes)
        self.autosave_minutes.setEnabled(preferences.autosave_enabled)
        self.delimiter.setCurrentIndex(self.delimiter.findData(preferences.csv_delimiter))
        self.encoding.setCurrentText(preferences.csv_encoding)
        self.decimal.setCurrentIndex(self.decimal.findData(preferences.decimal_separator))
        self.database_settings.populate(preferences, password)
        self.error.clear()

    def restore_defaults(self):
        self.populate(load_preferences(use_saved=False), "")

    def accept(self):
        preferences = Preferences(
            language=self.language.currentData(), theme=self.theme.currentData(), autosave_enabled=self.autosave_enabled.isChecked(),
            autosave_minutes=self.autosave_minutes.value(), csv_delimiter=self.delimiter.currentData(),
            csv_encoding=self.encoding.currentText(), decimal_separator=self.decimal.currentData(),
            **self.database_settings.values(),
        )
        try:
            save_preferences(preferences, password=self.password.text())
        except (ValueError, OSError) as exc:
            self.error.setText(message(exc))
            return
        super().accept()
