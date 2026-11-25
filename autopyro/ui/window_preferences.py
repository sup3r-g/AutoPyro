"""Apply application preferences to the desktop window."""

from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from .i18n import language, preferred_language, tr
from .settings import load_preferences
from .theme import apply_theme
from .window_actions import refresh_icons


def apply_preferences(window):
    preferences = load_preferences()
    apply_theme(QApplication.instance(), preferences.theme)
    refresh_icons(window)
    if hasattr(window, "desktop"):
        window.desktop.refresh_theme()
    window.session.configure_autosave(preferences.autosave_enabled, preferences.autosave_minutes)


def show_settings(window):
    from .dialogs.settings import SettingsDialog

    dialog = SettingsDialog(window)
    try:
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
    finally:
        dialog.deleteLater()
    if not accepted:
        return
    window.apply_preferences()
    if preferred_language() != language():
        QMessageBox.information(
            window,
            tr("Язык интерфейса"),
            tr("Язык сохранён. Перезапустите AutoPyro, чтобы применить изменение."),
        )
