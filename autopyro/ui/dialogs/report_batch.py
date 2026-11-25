"""Prepare, validate, stage and write grouped report batches."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
import re

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QInputDialog,
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QTextEdit,
    QVBoxLayout,
)

from autopyro.reports.reporting import (
    build_report,
    export_report,
    index_report_source,
    release_blockers,
    release_check_text,
    write_report_file,
)

from ..i18n import message, tr
from ..tasks import TaskCancelled, run_with_progress
from .report_preview import resize_to_screen


def export_batch(sheet):
    snapshot = sheet.report
    if snapshot is None:
        sheet.announce_status(tr('Соберите актуальный отчёт перед экспортом.'))
        return
    if release_blockers(snapshot):
        sheet.announce_status(tr('Исправьте блокирующие проверки перед экспортом.'))
        return
    project, options, plots = sheet.report_source
    ids = snapshot["row_ids"]
    if not ids or not options.column_ids:
        sheet.announce_status(tr('Для пакета выберите пробы и колонки отчёта.'))
        return
    columns = {
        f"{number}. {column.name}": column
        for number, column in enumerate(project.columns, 1)
    }
    label, accepted = QInputDialog.getItem(
        sheet, tr('Пакет отчётов'), tr('Один отчёт на каждое значение колонки'),
        list(columns), editable=False
    )
    if not accepted:
        return
    if sheet.report is not snapshot:
        sheet.announce_status(tr('Соберите актуальный отчёт перед экспортом.'))
        return
    column = columns[label]
    formats = sheet.choose_batch_formats()
    if not formats:
        return
    if sheet.report is not snapshot:
        sheet.announce_status(tr('Соберите актуальный отчёт перед экспортом.'))
        return
    rows = project.row_map()
    groups = {}
    for row_id in ids:
        value = rows[row_id].values.get(column.id)
        if isinstance(value, str) and not value.strip():
            value = None
        groups.setdefault(value, []).append(row_id)
    similar_labels = Counter(str(value).strip().casefold() for value in groups if isinstance(value, str))
    dialog = QDialog(sheet)
    dialog.setWindowTitle(tr('Файлы пакета'))
    layout = QVBoxLayout(dialog)
    instructions = tr('Отметьте файлы для экспорта. Пакет будет записан в новую папку.')
    layout.addWidget(QLabel(instructions))
    items = QListWidget()
    items.setAccessibleName(tr('Файлы для экспорта'))
    items.setAccessibleDescription(instructions)
    jobs = []
    for number, (value, group_ids) in enumerate(groups.items(), 1):
        label = tr('Нет значения') if value is None else str(value)
        slug = re.sub(r'[^\w.-]+', '_', label).strip('._')[:70]
        files = {}
        for kind in formats:
            filename = f"{number:03d}-{slug or 'report'}.{kind}"
            item = QListWidgetItem(tr('{0} · проб: {1}', filename, len(group_ids)), items)
            if isinstance(value, str) and similar_labels[value.strip().casefold()] > 1:
                item.setText(tr('{0} · значение группы: {1}', item.text(), repr(value)))
            item.setCheckState(Qt.CheckState.Checked)
            files[kind] = (filename, item)
        jobs.append((label, value, group_ids, files))
    layout.addWidget(items)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
    items.itemChanged.connect(lambda: ok.setEnabled(any(
        items.item(index).checkState() == Qt.CheckState.Checked for index in range(items.count())
    )))
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    if not dialog.exec():
        return
    if sheet.report is not snapshot:
        sheet.announce_status(tr('Соберите актуальный отчёт перед экспортом.'))
        return
    jobs = [
        (label, value, group_ids, selected)
        for label, value, group_ids, files in jobs
        if (selected := {kind: filename for kind, (filename, item) in files.items()
                         if item.checkState() == Qt.CheckState.Checked})
    ]
    if not jobs:
        return
    context = snapshot["selection_context"]

    def prepare(update, staging):
        update(tr('Индексирование измерений пакета…'), 0)
        source_index = index_report_source(project, (job[2] for job in jobs))
        prepared = []
        for index, (label, value, group_ids, files) in enumerate(jobs):
            def file_progress(stage, percent):
                update(tr('{0} из {1} · {2}', index + 1, len(jobs), stage),
                       (index * 100 + percent) // len(jobs))

            try:
                report = build_report(
                    project, group_ids, replace(options, title=f"{options.title} — {label}"), plots,
                    selection_context={**context, "batch_group": {"column_id": column.id, "value": value}},
                    progress=file_progress,
                    source_index=source_index,
                )
                notices = [release_check_text(item, options.language) for item in report["release_checks"]]
                notices.extend(report.get("well_profile_notices", ()))
            except TaskCancelled:
                raise
            except Exception as exc:
                prepared.extend((filename, label, None, [], str(exc) or type(exc).__name__)
                                for filename in files.values())
                continue
            for kind, filename in files.items():
                file_progress(tr('Проверка формата {0}…', kind.upper()), 98)
                staged = staging / filename
                try:
                    export_report(staged, report, kind)
                except Exception as exc:
                    prepared.append((filename, label, None, notices, str(exc) or type(exc).__name__))
                else:
                    prepared.append((filename, label, staged, notices, None))
        update(tr('Проверка пакета завершена.'), 100)
        return prepared

    try:
        with TemporaryDirectory(prefix="autopyro-report-batch-") as staging:
            prepared = run_with_progress(
                sheet, tr('Проверка готовности пакета'),
                lambda update: prepare(update, Path(staging)), cancellable=True,
            )
            prepared = sheet.review_batch(prepared)
            if not prepared:
                sheet.announce_status(tr('Пакет не записан.'))
                return
            parent = QFileDialog.getExistingDirectory(sheet, tr('Папка для пакета отчётов'))
            if not parent:
                return
            if sheet.report is not snapshot:
                sheet.announce_status(tr('Соберите актуальный отчёт перед экспортом.'))
                return
            sheet.write_batch(prepared, parent)
    except TaskCancelled:
        sheet.announce_status(tr('Проверка пакета отменена. Файлы выпуска не записаны.'))
    except Exception as exc:
        sheet.announce_status(tr('Не удалось подготовить пакет: {0}', message(exc)))


def choose_batch_formats(sheet):
    dialog = QDialog(sheet)
    dialog.setWindowTitle(tr('Форматы пакета'))
    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel(tr('Выберите один или несколько форматов.')))
    checks = {kind: QCheckBox(kind.upper()) for kind in ("pdf", "html", "docx", "json")}
    checks["json"].setText(tr('JSON — полный снимок выборки'))
    checks["json"].setAccessibleDescription(tr('Содержит все колонки и исходные ячейки выбранных проб.'))
    checks["pdf"].setChecked(True)
    for check in checks.values():
        layout.addWidget(check)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    layout.addWidget(buttons)
    ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
    for check in checks.values():
        check.toggled.connect(lambda: ok.setEnabled(any(c.isChecked() for c in checks.values())))
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    return tuple(kind for kind, check in checks.items() if check.isChecked()) if dialog.exec() else ()


def write_batch(sheet, prepared, parent):
    written = set()
    warnings = {filename for filename, _, _, notices, _ in prepared if notices}
    pending = {filename: staged for filename, _, staged, _, _ in prepared}
    folder = None

    def generate(update):
        nonlocal folder
        if folder is None:
            update(tr('Подготовка папки пакета…'), 0)
            folder = Path(mkdtemp(prefix="reports-", dir=parent))
        failed = {}
        for index, (filename, staged) in enumerate(pending.items()):
            update(tr('Сохранение {0}…', filename), 100 * index // len(pending))
            try:
                with staged.open("rb") as source:
                    write_report_file(folder / filename, iter(lambda: source.read(1024 * 1024), b""))
            except OSError as exc:
                failed[filename] = str(exc)
            else:
                written.add(filename)
        update(tr('Попытка записи завершена.'), 100)
        return failed

    while pending:
        try:
            failed = run_with_progress(sheet, tr('Запись пакета отчётов'), generate, cancellable=True)
        except TaskCancelled:
            sheet.announce_status(tr('Запись отменена. Сохранено файлов: {0} из {1}.', len(written), len(prepared))
                              + (f"\n{folder}" if folder else ""))
            return
        except Exception as exc:
            sheet.announce_status(tr('Сохранено файлов: {0} из {1}. Папка: {2}. Ошибка: {3}',
                                 len(written), len(prepared), str(folder or parent), message(exc)))
            return
        if not failed:
            break
        if not sheet.retry_batch(failed, folder):
            sheet.announce_status(tr('Сохранено файлов: {0} из {1}. Не записано: {2}. Папка: {3}',
                                 len(written), len(prepared), len(failed), str(folder)))
            return
        pending = {filename: pending[filename] for filename in failed}
    sheet.announce_status(tr('Сохранено файлов: {0}. С замечаниями: {1}. Папка: {2}',
                         len(written), len(written & warnings), str(folder)))


def retry_batch(sheet, failed, folder):
    dialog = QDialog(sheet)
    dialog.setWindowTitle(tr('Не удалось записать файлы'))
    resize_to_screen(dialog, 680, 360)
    layout = QVBoxLayout(dialog)
    label = QLabel(tr('Повторная попытка запишет только указанные файлы в ту же папку.'))
    label.setWordWrap(True)
    layout.addWidget(label)
    details = QTextEdit()
    details.setReadOnly(True)
    details.setAccessibleName(tr('Файлы с ошибками записи'))
    details.setAccessibleDescription(label.text())
    details.setPlainText(str(folder) + "\n\n" + "\n".join(
        f"{filename}: {error}" for filename, error in failed.items()
    ))
    layout.addWidget(details)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.addButton(tr('Повторить запись'), QDialogButtonBox.ButtonRole.AcceptRole)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    return bool(dialog.exec())


def review_batch(sheet, prepared):
    dialog = QDialog(sheet)
    dialog.setWindowTitle(tr('Готовность пакета'))
    resize_to_screen(dialog, 720, 450)
    layout = QVBoxLayout(dialog)
    failed = sum(error is not None for _, _, _, _, error in prepared)
    warnings = sum(bool(notices) and error is None for _, _, _, notices, error in prepared)
    summary = QLabel(tr('Файлов: {0}. С замечаниями: {1}. С ошибками: {2}.',
                        len(prepared), warnings, failed))
    summary.setWordWrap(True)
    layout.addWidget(summary)
    details = QTextEdit()
    details.setReadOnly(True)
    details.setAccessibleName(tr('Результаты проверки файлов пакета'))
    details.setAccessibleDescription(summary.text())
    errors_only = QCheckBox(tr('Показать только ошибки'))
    errors_only.setEnabled(bool(failed))
    layout.addWidget(errors_only)

    def update_details():
        lines = []
        shared_notices = {}
        for filename, label, _, notices, error in prepared:
            if errors_only.isChecked() and not error:
                continue
            lines.append(f"{label} · {filename}")
            if error:
                lines.append(tr('Ошибка: {0}', error))
            elif notices:
                shared_notices.setdefault((label, tuple(notices)), []).append(filename)
                lines.append(tr('Готов к записи с замечаниями ниже'))
            else:
                lines.append(tr('Готов к записи'))
            lines.append("")
        for (label, notices), filenames in shared_notices.items():
            lines.append(f"{label} · {', '.join(filenames)}")
            lines.extend(f"• {notice}" for notice in notices)
            lines.append("")
        details.setPlainText("\n".join(lines))

    errors_only.toggled.connect(update_details)
    update_details()
    layout.addWidget(details)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
    save = buttons.addButton(tr('Записать пакет…'), QDialogButtonBox.ButtonRole.AcceptRole)
    save.setEnabled(not failed)
    ready_only = QCheckBox(tr('Исключить файлы с ошибками и выпустить оставшиеся'))
    if 0 < failed < len(prepared):
        layout.insertWidget(layout.count() - 1, ready_only)
        ready_only.toggled.connect(save.setEnabled)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    if not dialog.exec() or (failed and not ready_only.isChecked()):
        return []
    return [entry for entry in prepared if entry[-1] is None]
