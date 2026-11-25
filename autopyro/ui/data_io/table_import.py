"""Open a table as a project or append its data to the current project."""

from pathlib import Path

from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox, QTextEdit, QVBoxLayout

from ..i18n import message, tr
from .import_file import ImportDialog


def import_table(window, *, append=False):
    if not append and not window.may_replace():
        return

    caption = (tr('Добавить измерения в текущий проект…') if append
               else tr('Открыть таблицу как новый проект…'))
    path, _ = QFileDialog.getOpenFileName(
        window, caption, "", tr('Таблицы (*.csv *.tsv *.txt *.xlsx)')
    )
    if not path:
        return

    mapping = review = None
    try:
        mapping = ImportDialog(path=path, parent=window, template_view=window.model.project.view)
        if Path(path).suffix.lower() == ".tsv":
            mapping.separator.setCurrentText("TAB")
        mapping.load()
        if mapping.exec() != QDialog.DialogCode.Accepted:
            return

        if append:
            from .append_import import AppendImportDialog

            review = AppendImportDialog(window, mapping.project)
            review.exec()
        else:
            window.set_project(mapping.project, prepared=mapping.project_prepared)
            window.model_changed()
    except Exception as exc:
        title = tr('Добавление данных') if append else tr('Импорт')
        QMessageBox.warning(window, title, message(exc))
    finally:
        # Closed forms retain the source table and import plan until disposed.
        if review is not None:
            review.deleteLater()
        if mapping is not None:
            mapping.deleteLater()


def import_log(window):
    dialog = QDialog(window)
    dialog.setWindowTitle(tr('Протоколы импорта'))
    dialog.resize(1000, 650)
    layout = QVBoxLayout(dialog)
    text = QTextEdit()
    text.setReadOnly(True)
    lines = []
    actions = {"new": tr('Новая проба'), "append": tr('Добавлен анализ'), "skip": tr('Пропущено')}
    for run in window.model.project.view.get("import_runs", {}).values():
        protocol = run["protocol"]
        lines.extend([tr('Импорт {0} · {1}', protocol['at'], tr('применён') if run['status'] == 'applied' else tr('отменён')),
                      tr('Источник: {0}', protocol['source'].get('path', tr('не указан'))),
                      tr('Новых проб: {0}; анализов: {1}; измерений: {2}; пропущено строк: {3}', protocol['new_samples'], protocol['analyses'], protocol['measurements'], protocol['skipped'])])
        lines.extend(tr('Строка {0}: {1}; проба {2}', item['source_row'], actions[item['action']], item.get('target_id') or '—') for item in protocol["rows"])
        lines.append("")
    text.setPlainText("\n".join(lines) or tr('Дополнительных импортов пока нет.'))
    layout.addWidget(text)
    try:
        dialog.exec()
    finally:
        dialog.deleteLater()


def import_template_library(window):
    from .templates import TemplateLibraryDialog
    dialog = TemplateLibraryDialog(window)
    try:
        dialog.exec()
    finally:
        dialog.deleteLater()
