"""Import and export stratigraphy schemes and manage lithology mappings."""

from __future__ import annotations

from ..i18n import tr, message
import json
from pathlib import Path
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox
from autopyro.data_io.file_output import write_json
from autopyro.data_io.io import file_source, read_file
from autopyro.geology.lithology_matching import (
    RULES_KEY,
    SOURCE_FIELDS,
    LithologyMatcher,
    LithologyMapping,
    default_source,
    normalize_rock,
    source_columns,
)
from autopyro.geology.stratigraphy import from_records, validate_scheme
from ..widgets.lithology import LithologyCatalog
from ..widgets.lithology_mapping import LithologyMappingDialog, MapDraftLithologies
from ..widgets.lithology_rules import LithologyRulesDialog
from ..widgets.interval_history import ReplaceDraft


def import_scheme(editor, *, merge=False):
    path, _ = QFileDialog.getOpenFileName(
        editor, tr('Интервалы'), "", tr('Интервалы (*.json *.csv *.xlsx)')
    )
    if not path:
        return
    try:
        current = editor.read_scheme() if merge else None
        if Path(path).suffix.lower() == ".json":
            scheme = validate_scheme(
                json.loads(Path(path).read_text(encoding="utf-8-sig"))
            )
        else:
            if not editor.reference.text().strip():
                raise ValueError(
                    tr('Перед импортом задайте тип глубины и начало отсчёта интервалов; файл не считается источником этой информации')
                )
            options = {}
            if Path(path).suffix.lower() == ".csv":
                delimiter, ok = QInputDialog.getItem(
                    editor,
                    "CSV",
                    tr('Разделитель (UTF-8)'),
                    [";", ",", tr('Табуляция')],
                    editable=False,
                )
                if not ok:
                    return
                options["delimiter"] = (
                    "\t" if delimiter == tr('Табуляция') else delimiter
                )
            else:
                from autopyro.data_io.io import excel_sheet_names

                sheet, ok = QInputDialog.getItem(
                    editor, "Excel", tr('Лист интервалов'),
                    excel_sheet_names(path), editable=False,
                )
                if not ok:
                    return
                options["sheet"] = sheet
            scheme = from_records(
                *read_file(path, **options),
                name=Path(path).stem,
                depth_reference=editor.reference.text(),
                source=file_source(path, **options),
            )
        editor.stage_import(scheme, current)
    except Exception as exc:
        editor.status.setText(tr('Ошибка импорта: {0}', message(exc)))


def import_database_scheme(editor):
    from ...database.ui.stratigraphy import StratigraphyDatabaseDialog

    try:
        dialog = StratigraphyDatabaseDialog(editor.reference.text(), editor)
        accepted = dialog.exec() == dialog.DialogCode.Accepted
        scheme = dialog.scheme
        dialog.deleteLater()
        if accepted and scheme is not None:
            current = None
            if editor.intervals.rowCount():
                try:
                    current = editor.read_scheme()
                except ValueError as exc:
                    answer = QMessageBox.question(
                        editor, tr('Заменить черновик интервалов'),
                        tr('Черновик нельзя объединить с импортом. {0}\nЗаменить его выбранной редакцией из БД?',
                           message(exc)),
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                        QMessageBox.StandardButton.Cancel)
                    if answer != QMessageBox.StandardButton.Yes:
                        return
            editor.stage_import(scheme, current)
    except Exception as exc:
        editor.status.setText(tr('Ошибка импорта: {0}', message(exc)))


def stage_import(editor, scheme, current):
    mapping, learned = editor.import_lithologies(scheme)
    if mapping is None:
        return
    if current is not None:
        if editor.review_merge(current, mapping):
            editor.save_lithology_rules(learned)
        return
    editor.draft_undo.push(ReplaceDraft(editor.interval_history, mapping))
    editor.save_lithology_rules(learned)
    editor.invalidate_preview()
    editor.status.setText(
        tr('Интервалы импортированы. Выберите ствол и уровень, затем постройте колонку. Привязка замеров необязательна.')
    )


def lithology_matcher(editor):
    custom = editor.column_style_form.rocks_editor.values()
    catalog = LithologyCatalog(custom) if custom else editor.lithology_catalog
    matcher = LithologyMatcher(((entry.code, entry.name) for entry in catalog.entries),
                              editor.owner.model.project.view.get(RULES_KEY, {}))
    return catalog, matcher


def import_lithologies(editor, scheme):
    catalog, matcher = editor.lithology_matcher()
    columns = source_columns(scheme)
    mapping = LithologyMapping(scheme, matcher, default_source(scheme, columns))
    ambiguous_source = (not mapping.groups and
                        sum(normalize_rock(column) in SOURCE_FIELDS for column in columns) > 1)
    if ambiguous_source or any(group.code is None for group in mapping.groups.values()):
        picker = LithologyMappingDialog(scheme, catalog, matcher, editor)
        if picker.exec() != picker.DialogCode.Accepted:
            return None, {}
        mapping = picker.mapping
    return mapping.apply(), mapping.learned_rules()


def save_lithology_rules(editor, learned):
    rules = editor.owner.model.project.view.get(RULES_KEY, {})
    if any(key not in rules or rules[key] != code for key, code in learned.items()):
        editor.owner.model.project.view[RULES_KEY] = {**rules, **learned}
        editor.owner.view_changed()


def manage_lithology_rules(editor):
    try:
        project = editor.owner.model.project
        rules = project.view.get(RULES_KEY, {})
        catalog, _ = editor.lithology_matcher()
        dialog = LithologyRulesDialog(rules, catalog, editor)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        updated = dialog.rules()
        if updated != rules:
            project.view[RULES_KEY] = updated
            editor.owner.view_changed()
            editor.status.setText(tr('Правила пород обновлены. Сохраните проект, чтобы записать изменения на диск.'))
    except Exception as exc:
        editor.status.setText(tr('Ошибка: {0}', message(exc)))


def map_lithologies(editor):
    try:
        scheme = editor.read_scheme()
        catalog, matcher = editor.lithology_matcher()
        picker = LithologyMappingDialog(scheme, catalog, matcher, editor)
        if picker.exec() != picker.DialogCode.Accepted:
            return
        mapped = picker.mapping.apply()
        command = MapDraftLithologies(editor.interval_history, scheme, mapped)
        if command.changes:
            editor.draft_undo.push(command)
        editor.save_lithology_rules(picker.mapping.learned_rules())
        editor.lithology_delegate.catalog = catalog
        editor.intervals.viewport().update()
        editor.status.setText(tr('Названия пород сопоставлены. Постройте колонку заново.'))
    except Exception as exc:
        editor.status.setText(tr('Ошибка: {0}', message(exc)))


def export_scheme(editor):
    try:
        scheme = editor.read_scheme()
        path, _ = QFileDialog.getSaveFileName(
            editor, tr('Схема интервалов'), "stratigraphy.json", "JSON (*.json)"
        )
        if path:
            write_json(path, scheme)
            editor.status.setText(tr('Схема экспортирована: {0}', path))
    except Exception as exc:
        editor.status.setText(tr('Ошибка экспорта: {0}', message(exc)))
