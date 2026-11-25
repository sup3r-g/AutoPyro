"""Database workflows initiated from the desktop."""

from copy import deepcopy

from PySide6.QtWidgets import QDialog, QMessageBox, QTextEdit, QVBoxLayout

from autopyro.database.ui.snapshot import DatabaseDialog
from autopyro.model.core import new_id

from ..i18n import message, tr
from ..tasks import run_with_progress
from .import_file import ImportDialog


def refresh_database(window):
    from autopyro.database.sources import descriptor
    from autopyro.database.ui.refresh import DatabaseRefreshDialog
    try:
        descriptor(window.model.project.source)
    except (ValueError, KeyError):
        QMessageBox.information(window, tr('Обновление БД'), tr('Текущий проект не содержит поддерживаемого снимка БД. Сначала импортируйте источник.'))
        return
    dialog = DatabaseRefreshDialog(window)
    try:
        dialog.exec()
    finally:
        dialog.deleteLater()

def database_refresh_log(window):
    from autopyro.database.ui.refresh import key_label, ACTIONS
    dialog = QDialog(window)
    dialog.setWindowTitle(tr('Протоколы обновления снимков БД'))
    dialog.resize(1100, 700)
    layout = QVBoxLayout(dialog)
    text = QTextEdit()
    text.setReadOnly(True)
    names = {c.id: c.name for c in window.model.project.columns}
    lines = []
    for run in window.model.project.view.get("database_refresh_runs", {}).values():
        protocol = run["protocol"]
        lines.append(tr('{0} · {1}; снимок получен {2}', protocol['at'], tr('применено') if run['status'] == 'applied' else tr('отменено'), protocol['checked_at']))
        for change in protocol["changes"]:
            lines.append(f"{key_label(change['key'])} · {tr(ACTIONS[change['action']])}")
            if "property_id" in change:
                lines.append(tr('  {0}: снимок {1!r}; локально {2!r}; источник {3!r}', names.get(change['property_id'], change['property_id']), change['base'], change['local'], change['source']))
        if not protocol["changes"]:
            lines.append(tr('Изменений данных нет.'))
        lines.append("")
    text.setPlainText("\n".join(lines) or tr('Обновлений снимка ещё не было.'))
    layout.addWidget(text)
    try:
        dialog.exec()
    finally:
        dialog.deleteLater()

def import_geochemistry(window, *, append=False):
    from autopyro.database.ui.nonpyro import NonPyroDatabaseDialog

    dialog = NonPyroDatabaseDialog(parent=window, append=True) if append else NonPyroDatabaseDialog(parent=window)
    try:
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.project is None:
            return
        if append:
            from autopyro.database.geochem import append_geochemistry_snapshot
            from autopyro.data_io.importing import digest
            from ..table import ImportCommand

            project = window.model.project
            incoming = dialog.project
            expected, prepared = run_with_progress(window, tr('Подготовка добавления анализов…'),
                lambda: (digest(project), append_geochemistry_snapshot(project, incoming)))
            if window.model.project is not project:
                raise ValueError(tr('Проект изменился. Повторите импорт.'))
            protocol = {"id": new_id(), "kind": "geochemistry_snapshot_append",
                        "rows": len(incoming.rows), "source": deepcopy(incoming.source)}
            window.model.undo.push(ImportCommand(window.model, prepared, protocol, expected))
            window.desktop.open_table()
            window.statusBar().showMessage(tr('Записи геохимии добавлены в текущий проект.'))
        elif window.may_replace():
            window.set_project(dialog.project)
            window.model_changed()
            window.desktop.open_table()
            window.statusBar().showMessage(tr('Данные геохимии загружены. Выберите раздел в меню «Обработка → Геохимия».'))
    except (ValueError, KeyError) as exc:
        QMessageBox.warning(window, tr('Импорт геохимии'), message(exc))
    finally:
        dialog.deleteLater()

def import_database(window):
    if not window.may_replace():
        return
    dialog = DatabaseDialog(window)
    mapping = None
    try:
        if dialog.exec() == QDialog.DialogCode.Accepted:
            mapping = ImportDialog(snapshot=dialog.snapshot, parent=window, template_view=window.model.project.view)
            if mapping.exec() == QDialog.DialogCode.Accepted:
                window.set_project(mapping.project, prepared=mapping.project_prepared)
                window.model_changed()
    finally:
        if mapping is not None:
            mapping.deleteLater()
        dialog.deleteLater()
