"""Global defaults are independent of projects and committed only on acceptance."""
from copy import deepcopy
from dataclasses import replace

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QDialog

from autopyro.ui import i18n, settings
from autopyro.database import preferences as database_preferences
from autopyro.ui.dialogs.settings import SettingsDialog
from autopyro.ui.window import MainWindow

BASE = {"host": "localhost", "port": "15432", "dbname": "geochemistry", "user": "readonly", "password": "environment-secret"}


@pytest.fixture
def store(tmp_path, monkeypatch, qapp):
    result = QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(i18n, "settings", lambda: result)
    monkeypatch.setattr(settings, "_local_parameters", lambda: dict(BASE))
    monkeypatch.setattr(database_preferences, "_session_credentials", None)
    i18n.initialize()
    yield result
    i18n.save_language("ru")
    i18n.initialize()


def test_preferences_round_trip_and_password_is_session_only(store):
    preferences = replace(settings.load_preferences(), language="en", autosave_enabled=False, autosave_minutes=9,
                          csv_delimiter="\t", csv_encoding="utf-16", decimal_separator=",",
                          database_host="remote.example", database_port=5432, database_name="samples",
                          database_user="reader", sqlite_path="source.sqlite", sqlite_table="samples")
    settings.save_preferences(preferences, password="session-secret")
    fresh = QSettings(store.fileName(), QSettings.Format.IniFormat)
    assert settings.load_preferences(fresh) == preferences
    assert not any("password" in key for key in fresh.allKeys())
    assert "session-secret" not in open(store.fileName(), encoding="utf-8").read()
    assert settings.database_parameters()["password"] == "session-secret"
    database_preferences._session_credentials = None  # Simulate a new process.
    assert settings.database_parameters()["password"] == ""


def test_defaults_do_not_forward_password_to_other_connection(store):
    assert settings.database_parameters()["password"] == "environment-secret"
    preferences = replace(settings.load_preferences(), database_host="other.example")
    settings.save_preferences(preferences)
    assert settings.database_parameters()["password"] == ""
    preferences = replace(preferences, database_host=BASE["host"], database_user="different-user")
    settings.save_preferences(preferences)
    assert settings.database_parameters()["password"] == ""


def test_settings_cancel_and_restore_defaults_do_not_write(store, qtbot):
    preferences = replace(settings.load_preferences(), autosave_minutes=17, database_name="custom")
    settings.save_preferences(preferences)
    before = {key: store.value(key) for key in store.allKeys()}
    dialog = SettingsDialog()
    qtbot.addWidget(dialog)
    dialog.host.setText("cancelled.example")
    dialog.restore_defaults()
    assert dialog.autosave_minutes.value() == 1
    assert dialog.database.text() == BASE["dbname"]
    dialog.reject()
    assert settings.load_preferences() == preferences
    assert {key: store.value(key) for key in store.allKeys()} == before


def test_invalid_draft_does_not_save_any_values(store, qtbot):
    dialog = SettingsDialog()
    qtbot.addWidget(dialog)
    dialog.host.clear()
    dialog.autosave_minutes.setValue(12)
    dialog.accept()
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert dialog.error.text()
    assert store.allKeys() == []


def test_save_defaults_populates_new_import_and_database_dialogs(store, qtbot, tmp_path):
    from autopyro.ui.data_io.import_file import ImportDialog
    from autopyro.ui.database.snapshot import DatabaseDialog
    from autopyro.ui.database.nonpyro import NonPyroDatabaseDialog

    dialog = SettingsDialog()
    qtbot.addWidget(dialog)
    dialog.host.setText("database.example")
    dialog.port.setValue(5433)
    dialog.database.setText("research")
    dialog.user.setText("research_reader")
    dialog.password.setText("session-only")
    dialog.sqlite_path.setText("data.sqlite")
    dialog.sqlite_table.setText("samples")
    dialog.delimiter.setCurrentIndex(dialog.delimiter.findData("\t"))
    dialog.encoding.setCurrentText("utf-16")
    dialog.decimal.setCurrentIndex(dialog.decimal.findData(","))
    dialog.accept()
    assert dialog.result() == QDialog.DialogCode.Accepted
    snapshot = DatabaseDialog()
    qtbot.addWidget(snapshot)
    assert snapshot.fields["host"].text() == "database.example"
    assert snapshot.fields["port"].text() == "5433"
    assert snapshot.fields["dbname"].text() == "research"
    assert snapshot.fields["path"].text() == "data.sqlite"
    assert snapshot.fields["password"].text() == "session-only"
    snapshot.fields["user"].setText("another-reader")
    assert snapshot.fields["password"].text() == ""
    profile = NonPyroDatabaseDialog("atomic_ratios")
    qtbot.addWidget(profile)
    assert profile.connection_form.values()["host"] == "database.example"
    assert profile.connection_form.values()["password"] == "session-only"
    imported = ImportDialog(path=tmp_path / "new.csv")
    qtbot.addWidget(imported)
    assert imported.separator.currentText() == "TAB"
    assert imported.encoding.currentText() == "utf-16"
    assert imported.decimal.currentData() == ","


def test_connection_edits_clear_password_in_settings(store, qtbot):
    dialog = SettingsDialog()
    qtbot.addWidget(dialog)
    assert dialog.password.text() == "environment-secret"
    dialog.host.setText("new.example")
    assert dialog.password.text() == ""


def test_saved_preferences_apply_to_timer_without_changing_project(store, qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.may_replace = lambda: True
    project = deepcopy(window.model.project)
    assert window.autosave_timer.isActive()
    assert window.autosave_timer.interval() == 60_000

    def accept(dialog):
        dialog.autosave_minutes.setValue(7)
        dialog.autosave_enabled.setChecked(False)
        dialog.accept()
        return dialog.result()

    monkeypatch.setattr(SettingsDialog, "exec", accept)
    window.actions[i18n.tr("Настройки приложения…")].trigger()
    assert not window.autosave_timer.isActive()
    assert window.autosave_timer.interval() == 7 * 60_000
    assert window.model.project == project
    assert not window.dirty
    settings.save_preferences(replace(settings.load_preferences(), autosave_enabled=True))
    window.apply_preferences()
    assert window.autosave_timer.isActive()


def test_corrupted_settings_fall_back_to_defaults(store):
    store.setValue("general/autosave_minutes", -100)
    store.setValue("database/port", "not a port")
    store.setValue("import/delimiter", "invalid")
    store.setValue("database/host", "")
    preferences = settings.load_preferences()
    assert preferences.autosave_minutes == 1
    assert preferences.database_port == int(BASE["port"])
    assert preferences.csv_delimiter == ";"
    assert preferences.database_host == BASE["host"]
