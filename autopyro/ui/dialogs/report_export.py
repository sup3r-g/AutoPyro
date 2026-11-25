"""Export complete reports and manage history snapshots and comparisons."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from autopyro.reports.history import (
    compare_snapshots,
    comparison_text,
    load_snapshot,
    snapshot_location,
    snapshot_path,
)
from autopyro.reports.reporting import digest, encode_report, release_blockers, write_report_file

from ..i18n import language, message, tr
from ..tasks import TaskCancelled, run_with_progress
from .report_preview import resize_to_screen


def archive_report(sheet):
    sheet.export("json", archive=True)


def show_history(sheet):
    project = sheet.owner.model.project
    project_path = sheet.owner.path
    entries = project.view.get("report_history", [])
    if not entries:
        sheet.announce_status(tr('История пуста. Соберите отчёт и сохраните его в историю.'))
        return
    dialog = QDialog(sheet)
    dialog.setWindowTitle(tr('История отчётов'))
    resize_to_screen(dialog, 900, 650)
    layout = QVBoxLayout(dialog)
    form = QFormLayout()
    before, after = QComboBox(), QComboBox()
    for entry in entries:
        label = f"{entry['created_at']} · {entry['title']} · {entry['sha256'][:12]}"
        for combo in (before, after):
            combo.addItem(label, entry)
    before.setCurrentIndex(max(0, len(entries) - 2))
    after.setCurrentIndex(len(entries) - 1)
    for title, combo in ((tr('Исходный снимок'), before), (tr('Новый снимок'), after)):
        controls = QHBoxLayout()
        controls.addWidget(combo, 1)
        locate = QPushButton(tr('Указать файл…'))
        locate.clicked.connect(lambda checked=False, target=combo: relocate(target))
        controls.addWidget(locate)
        form.addRow(title, controls)
    layout.addLayout(form)
    result = QTextEdit()
    result.setReadOnly(True)
    result.setPlainText(tr('Выберите два снимка для сравнения. Если файл перемещён, укажите его новое место.'))
    layout.addWidget(result)

    def compare():
        old_entry, new_entry = before.currentData(), after.currentData()
        interface_language = language()

        def compare_files():
            old = load_snapshot(old_entry, project.id, project_path)
            new = load_snapshot(new_entry, project.id, project_path)
            return comparison_text(compare_snapshots(old, new), language=interface_language)

        try:
            text = run_with_progress(dialog, tr('Сравнение снимков…'), compare_files)
        except Exception as exc:
            result.setPlainText(tr('Не удалось прочитать или сравнить снимки: {0}', message(exc)))
        else:
            result.setPlainText(text)

    def relocate(combo):
        index = combo.currentIndex()
        entry = entries[index]
        path, _ = QFileDialog.getOpenFileName(
            dialog, tr('Указать новое место снимка'), str(snapshot_path(entry, project_path).parent),
            "JSON (*.json)",
        )
        if not path:
            return
        replacement = {key: value for key, value in entry.items() if key != "relative_path"}
        replacement.update(snapshot_location(path, project_path))
        try:
            run_with_progress(dialog, tr('Проверка снимка…'), lambda: load_snapshot(replacement, project.id))
        except Exception as exc:
            result.setPlainText(tr('Снимок не перепривязан: {0}', message(exc)))
            return
        entries[index] = replacement
        for selector in (before, after):
            selector.setItemData(index, replacement)
        sheet.owner.view_changed()
        result.setPlainText(tr('Новое место снимка сохранено: {0}', replacement["path"]))

    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    compare_button = buttons.addButton(tr('Сравнить'), QDialogButtonBox.ButtonRole.ActionRole)
    compare_button.setEnabled(len(entries) > 1)
    compare_button.clicked.connect(compare)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.exec()


def export(sheet, kind, *, archive=False):
    snapshot = sheet.report
    if snapshot is None:
        return
    if release_blockers(snapshot):
        sheet.announce_status(tr('Исправьте блокирующие проверки перед экспортом.'))
        return
    project = sheet.owner.model.project
    entries = project.view.get("report_history", [])
    sha256 = digest(sheet.report) if archive else None
    if archive and any(entry["sha256"] == sha256 for entry in entries):
        sheet.announce_status(tr('Этот снимок уже есть в истории.'))
        return
    path, _ = QFileDialog.getSaveFileName(
        sheet,
        tr('Сохранить в историю') if archive else tr('Экспорт {0}', kind.upper()),
        f"report-{sha256[:12]}.json" if archive else f"report.{kind}",
        f"{kind.upper()} (*.{kind})",
    )
    if path:
        if sheet.report is not snapshot:
            return
        try:
            path = Path(path).resolve()
            if any(path == snapshot_path(entry, sheet.owner.path).resolve() for entry in entries):
                sheet.announce_status(tr('Этот файл уже связан с историей отчётов. Выберите другой путь.'))
                return
            def prepare(update):
                update(tr('Формирование файла {0}…', kind.upper()), 0)
                return encode_report(snapshot, kind)

            payload = run_with_progress(sheet, tr('Экспорт {0}', kind.upper()), prepare, cancellable=True)
            run_with_progress(sheet, tr('Запись отчёта'), lambda: write_report_file(path, payload))
            if archive:
                project.view.setdefault("report_history", []).append({
                    **snapshot_location(path, sheet.owner.path),
                    "sha256": sha256, "created_at": snapshot["created_at"],
                    "title": snapshot["options"]["title"],
                })
                sheet.owner.view_changed()
            sheet.announce_status(tr('Снимок в истории: {0}', str(path)) if archive
                                 else tr('Сохранено: {0}', str(path)))
        except TaskCancelled:
            sheet.announce_status(tr('Экспорт отменён. Файл не записан.'))
        except Exception as exc:
            QMessageBox.warning(sheet, tr('Отчёт'), message(exc))
