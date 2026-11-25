"""Load, bind and edit interpretation schemes and validate their revisions."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

from autopyro.schemes.interpretation import (
    evaluate_project,
    scheme_hash,
    validate_axis_binding,
    validate_scheme,
)
from autopyro.schemes.interpretation_input import capture_interpretation_steps
from autopyro.schemes.interpretation_quality import column_property_name, property_name
from autopyro.schemes.interpretation_revisions import store_scheme_revision, validate_scheme_revision
from autopyro.schemes.reference_cases import missing_reference_checks, reference_failures

from ..i18n import tr, message, settings
from ..tasks import TaskCancelled, run_with_progress


def restore_selection(sheet):
    owner = sheet.owner
    settings = owner.model.project.view.get("interpretation_current", {})
    for widget, key in ((sheet.x, "x_id"), (sheet.y, "y_id"), (sheet.scope, "scope")):
        index = widget.findData(settings.get(key))
        if index >= 0:
            widget.setCurrentIndex(index)
    sheet.x_unit.setText(settings.get("x_unit", ""))
    sheet.y_unit.setText(settings.get("y_unit", ""))
    sheet.text_numbers.setChecked(settings.get("text_numbers", False))
    scheme = owner.model.project.view.get("interpretation_schemes", {}).get(
        settings.get("scheme_id")
    )
    if scheme:
        sheet.set_scheme(scheme, auto_bind=False)


def sync_columns(sheet, *args):
    for combo in (sheet.x, sheet.y):
        selected = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(tr('Выберите показатель'), "")
        for column in sheet.owner.model.project.columns:
            combo.addItem(column.name, column.id)
        if selected and combo.findData(selected) < 0:
            combo.addItem(tr('Удалённая колонка — выберите замену'), selected)
        combo.setCurrentIndex(max(0, combo.findData(selected)))
        combo.blockSignals(False)
    sheet.update_catalog_preview()
    if hasattr(sheet, "status"):
        sheet.schedule()


def scheme_columns(sheet, axis):
    matches = []
    for column in sheet.owner.model.project.columns:
        try:
            validate_axis_binding(column, column.unit, axis)
        except ValueError:
            continue
        matches.append(column)
    return matches


def scheme_column(sheet, axis):
    matches = sheet.scheme_columns(axis)
    return matches[0].id if len(matches) == 1 else ""


def set_scheme(sheet, scheme, auto_bind=True, *, binding=None):
    validate_scheme(scheme)
    previous = sheet.scheme
    sheet.scheme = deepcopy(scheme)
    if sheet.standard_schemes.get(scheme['name']) == scheme:
        group_index = sheet.scheme_group.findData(scheme.get('group'))
        sheet.scheme_group.setCurrentIndex(max(0, group_index))
        sheet.standard_choice.setCurrentIndex(max(0, sheet.standard_choice.findData(scheme['name'])))
    else:
        sheet.standard_choice.setCurrentIndex(0)
    sheet.editor.blockSignals(True)
    sheet.editor.setPlainText(json.dumps(scheme, ensure_ascii=False, indent=2))
    sheet.editor.blockSignals(False)
    sheet.editor_dirty = False
    status = {
        "demonstration": tr('демонстрационная'),
        "user_supplied": tr('пользовательская'),
        "validated": tr('проверена автором'),
    }[scheme["scientific_status"]]
    bundled = sheet.standard_schemes.get(scheme['name'])
    acceptance = (tr('Ожидает независимого решения') if bundled == scheme
                  else tr('Для этой редакции нет записи'))
    sheet.scheme_info.setText(
        tr('{0} · редакция {1}\nСтатус, заявленный автором: {2}\nНезависимая приёмка: {3}\nИсточник: {4}',
           scheme['name'], scheme['revision'], status, acceptance, scheme['source'])
        + ('\n' + scheme['limitations'] if scheme.get('limitations') else '')
    )
    if (bundled and bundled['id'] == scheme['id'] and bundled.get('max_distance') is not None
            and scheme.get('strategy') == 'nearest_curve' and scheme.get('max_distance') is None):
        sheet.scheme_info.setText(sheet.scheme_info.text() + '\n' + tr(
            'Доступна редакция {0} с пределом расстояния. Выберите эту схему в каталоге для обновления.',
            bundled['revision']))
    if auto_bind or binding is not None:
        for combo, unit, key in (
            (sheet.x, sheet.x_unit, "x"),
            (sheet.y, sheet.y_unit, "y"),
        ):
            column_id = sheet.scheme_column(scheme[key]) if binding is None else binding[f'{key}_id']
            combo.setCurrentIndex(max(0, combo.findData(column_id)))
            if binding is not None:
                unit.setText(binding[f'{key}_unit'])
            else:
                unit.setText(scheme[key]['unit'] if column_id else '')
    if (
        sheet.owner.model.project.view.get("interpretation_schemes", {}).get(
            scheme["id"]
        )
        == scheme
    ):
        sheet.remember_current_scheme()
    if hasattr(sheet, 'plot') and (previous is None or any(previous[key] != scheme[key] for key in ('x', 'y'))):
        sheet.configure_plot_axes()
    sheet.schedule()


def remember_current_scheme(sheet):
    settings = {
        "scheme_id": sheet.scheme["id"],
        "x_id": sheet.x.currentData(),
        "y_id": sheet.y.currentData(),
        "x_unit": sheet.x_unit.text().strip(),
        "y_unit": sheet.y_unit.text().strip(),
        "text_numbers": sheet.text_numbers.isChecked(),
        "scope": sheet.scope.currentData(),
    }
    if sheet.owner.model.project.view.get("interpretation_current") != settings:
        sheet.owner.model.project.view["interpretation_current"] = settings
        sheet.owner.model_changed()


def fill_standard_schemes(sheet):
    group = sheet.scheme_group.currentData()
    with QSignalBlocker(sheet.standard_choice):
        sheet.standard_choice.clear()
        sheet.standard_choice.addItem(tr('Выберите схему…'), None)
        for name, scheme in sheet.standard_schemes.items():
            if scheme.get('group') == group or name == group:
                title = name.removeprefix(group + ' · ') if group else name
                if scheme['scientific_status'] == 'demonstration':
                    title += ' · ' + tr('демонстрационная')
                sheet.standard_choice.addItem(title, name)
                sheet.standard_choice.setItemData(sheet.standard_choice.count() - 1, name, Qt.ItemDataRole.ToolTipRole)
    sheet.update_catalog_preview()


def update_catalog_preview(sheet, *_):
    scheme = sheet.standard_schemes.get(sheet.standard_choice.currentData())
    sheet.catalog_preview.setVisible(scheme is not None)
    sheet.apply_catalog_scheme.setEnabled(scheme is not None)
    if scheme is None:
        sheet.catalog_preview.clear()
        return
    strategy = scheme.get('strategy', 'regions')
    if strategy == 'nearest_curve':
        mode = tr('Оценка по ближайшей кривой')
    elif strategy == 'initial_hi':
        mode = tr('Модельная оценка HIo')
    elif scheme.get('regions') or any('threshold' in line for line in scheme.get('lines', ())):
        mode = tr('Классификация по областям или порогам')
    else:
        mode = tr('Справочная диаграмма')
    status = {
        'demonstration': tr('демонстрационная'),
        'user_supplied': tr('пользовательская'),
        'validated': tr('проверена автором'),
    }[scheme['scientific_status']]
    lines = [mode, tr('Редакция {0}', scheme['revision']),
             tr('Статус, заявленный автором: {0}', status), tr('Источник: {0}', scheme['source'])]
    lines.append(tr('Независимая приёмка: ожидает решения'))
    if scheme.get('limitations'):
        lines.append(scheme['limitations'])
    columns = sheet.owner.model.project.columns
    for key in ('x', 'y'):
        axis = scheme[key]
        matches = sheet.scheme_columns(axis)
        if len(matches) == 1:
            column = matches[0]
            state = column.name
        elif matches:
            state = tr('Подходящих колонок: {0}. Выберите нужную вручную.', len(matches))
        elif any(not column.unit and column_property_name(column) == property_name(axis['property'])
                 for column in columns):
            state = tr('Единица исходной колонки не указана. Выберите колонку и укажите единицу вручную.')
        elif any(column_property_name(column) == property_name(axis['property']) for column in columns):
            state = tr('Есть показатель, но единица отличается')
        else:
            state = tr('Колонка показателя отсутствует')
        lines.append(tr('{0} · {1} [{2}] — {3}', key.upper(), axis['property'], axis['unit'], state))
    sheet.catalog_preview.setText('\n'.join(lines))


def load_standard_scheme(sheet):
    name = sheet.standard_choice.currentData()
    if name and sheet.confirm_replace_draft():
        try:
            sheet.commit_graphical_scheme(sheet.standard_schemes[name], auto_bind=True)
        except (ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(sheet, tr('Схема'), message(exc))


def load_scheme(sheet):
    path, _ = QFileDialog.getOpenFileName(
        sheet, tr('Геометрическая схема'), "", "JSON (*.json)"
    )
    if path and sheet.confirm_replace_draft():
        try:
            sheet.commit_graphical_scheme(json.loads(Path(path).read_text(encoding="utf-8")), auto_bind=True)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            QMessageBox.warning(sheet, tr('Схема'), message(exc))


def validate_revision(sheet, scheme):
    return validate_scheme_revision(sheet.owner.model.project, scheme)


def store_scheme(sheet):
    if store_scheme_revision(sheet.owner.model.project, sheet.scheme):
        sheet.scheme_saved.emit()
    sheet.remember_current_scheme()


def geometry_samples(sheet, row_ids=None):
    from autopyro.schemes.interpretation import InterpretationInput
    from autopyro.analysis.statistics import classify

    report = sheet.report
    if row_ids is not None and sheet.scheme is not None:
        report = evaluate_project(sheet.owner.model.project, row_ids,
            sheet.x.currentData(), sheet.y.currentData(), sheet.x_unit.text().strip(),
            sheet.y_unit.text().strip(), sheet.scheme, text_numbers=sheet.text_numbers.isChecked())
    if not report:
        return []
    result = []
    for entry in report["inputs"]:
        values = [
            classify(entry[key], numeric=True)[1] if status == "numeric" else None
            for key, status in zip(("x", "y"), entry["statuses"], strict=True)
        ]
        result.append(InterpretationInput(entry["row_id"], *values,
            x_bounds=entry.get("x_bounds"), y_bounds=entry.get("y_bounds")))
    return result


def commit_graphical_scheme(sheet, scheme, *, auto_bind=False):
    validate_scheme(scheme)
    stored = sheet.validate_revision(scheme)
    sheet.require_reference_cases(scheme, stored)
    binding = None
    if stored is not None and stored != scheme:
        binding = sheet.preview_revision(stored, scheme, auto_bind)
        if binding is None:
            return
    previous = sheet.scheme
    sheet.scheme = deepcopy(scheme)
    try:
        sheet.store_scheme()
    finally:
        sheet.scheme = previous
    sheet.set_scheme(scheme, auto_bind=auto_bind, binding=binding)


def require_reference_cases(scheme, previous=None):
    missing = missing_reference_checks(previous, scheme) if previous is not None else []
    if missing:
        details = ', '.join(f'{case_id} · {field}' if field else case_id
                            for case_id, field in missing)
        raise ValueError(tr('В reference_cases новой редакции отсутствуют эталоны или проверки: {0}.',
                            details))
    failures = reference_failures(scheme)
    if failures:
        details = [
            tr('{0} · {1}. Ожидалось {2}, получено {3}.', case_id, field,
               json.dumps(expected, ensure_ascii=False), json.dumps(actual, ensure_ascii=False))
            for case_id, field, expected, actual in failures
        ]
        raise ValueError(tr('Редакция не прошла проверку эталонов.') + '\n' + '\n'.join(details))


def check_reference_cases(sheet):
    title = tr('Эталонные пробы')
    try:
        scheme = json.loads(sheet.editor.toPlainText())
        validate_scheme(scheme)
        stored = sheet.owner.model.project.view.get('interpretation_schemes', {}).get(scheme['id'])
        sheet.require_reference_cases(scheme, stored)
    except (ValueError, KeyError, TypeError) as exc:
        QMessageBox.warning(sheet, title, message(exc))
        return
    count = len(scheme.get('reference_cases', []))
    text = tr('Проверку прошли все эталоны ({0}).', count) if count else tr('В схеме пока нет эталонных проб.')
    QMessageBox.information(sheet, title, text)


def revision_binding(sheet, scheme, preferred):
    columns = {column.id: column for column in sheet.owner.model.project.columns}
    binding = {'text_numbers': preferred.get('text_numbers', sheet.text_numbers.isChecked())}
    for key in ('x', 'y'):
        axis = scheme[key]
        column_id = preferred.get(f'{key}_id')
        unit = preferred.get(f'{key}_unit', '')
        try:
            validate_axis_binding(columns[column_id], unit, axis)
        except (KeyError, ValueError):
            column_id, unit = sheet.scheme_column(axis), axis['unit']
        if not column_id:
            raise ValueError(tr('Для редакции {0} выберите колонку показателя {1}.',
                                scheme['revision'], axis['property']))
        binding.update({f'{key}_id': column_id, f'{key}_unit': unit})
    return binding


def preview_revision(sheet, before, after, auto_bind):
    from .scheme_revision import SchemeRevisionDialog

    model = sheet.owner.model
    project = model.project
    ids = tuple(sheet.owner.analysis_scope.rows())
    current = dict(x_id=sheet.x.currentData(), y_id=sheet.y.currentData(),
                   x_unit=sheet.x_unit.text().strip(), y_unit=sheet.y_unit.text().strip(),
                   text_numbers=sheet.text_numbers.isChecked())
    saved = project.view.get('interpretation_result') or {}
    old_preferred = saved['binding'] if saved.get('scheme_hash') == scheme_hash(before) else current
    old_binding = sheet.revision_binding(before, old_preferred)
    same_scheme = sheet.scheme is not None and sheet.scheme['id'] == after['id']
    new_binding = sheet.revision_binding(after, current if same_scheme or not auto_bind else {})
    revision = sheet.calculation_revision
    active = sheet.report
    reusable = (active is not None and not active.get('stale') and active['scheme'] == before
                and active['binding'] == old_binding
                and {item['row_id'] for item in active['inputs']} == set(ids))
    bindings = [dict(new_binding, scheme=after)]
    if not reusable:
        bindings.append(dict(old_binding, scheme=before))
    def still_current():
        return not sheet.closed and model.project is project and revision == sheet.calculation_revision

    def capture():
        steps = capture_interpretation_steps(project, ids, bindings)
        while still_current():
            try:
                next(steps)
            except StopIteration as done:
                return done.value
            yield
        raise TaskCancelled()

    def compare(update, snapshot):
        old_stage = tr('Расчёт исходной редакции…')
        new_stage = tr('Расчёт предложенной редакции…')
        old = active if reusable else evaluate_project(
            snapshot, ids, scheme=before, **old_binding,
            checkpoint=lambda: update(old_stage, 0),
        )
        new = evaluate_project(
            snapshot, ids, scheme=after, **new_binding,
            checkpoint=lambda: update(new_stage, 50),
        )
        difference_stage = tr('Подготовка различий между редакциями…')
        update(difference_stage, 90)
        prepared = SchemeRevisionDialog.prepare(old, new, lambda: update(difference_stage, 90))
        return old, new, prepared

    try:
        old, new, prepared = run_with_progress(
            sheet, tr('Сравнение редакций схемы'), compare, cancellable=True,
            prepare=capture(), preparation_label=tr('Подготовка снимка для сравнения…'),
        )
    except TaskCancelled:
        return None
    if not still_current():
        return None
    dialog = SchemeRevisionDialog(old, new, prepared, sheet)
    model.changed.connect(dialog.reject)
    try:
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
    finally:
        model.changed.disconnect(dialog.reject)
        dialog.deleteLater()
    if (accepted and not sheet.closed and model.project is project and revision == sheet.calculation_revision
            and project.view.get('interpretation_schemes', {}).get(before['id']) == before
            and set(sheet.owner.analysis_scope.rows()) == set(ids)):
        return new_binding


def open_geometry_editor(sheet):
    from .geometry_editor import GeometryEditor

    try:
        draft = (
            json.loads(sheet.editor.toPlainText())
            if sheet.editor_dirty
            else sheet.scheme
        )
        if draft is None:
            raise ValueError(tr('Загрузите схему или нажмите «Новая схема»'))
        validate_scheme(draft)
        dialog = GeometryEditor(draft, sheet, samples=sheet.geometry_samples())
        if dialog.exec() == QDialog.DialogCode.Accepted:
            sheet.commit_graphical_scheme(dialog.edited_scheme)
    except (ValueError, TypeError, KeyError) as exc:
        QMessageBox.warning(sheet, tr('Графический редактор'), message(exc))


def new_graphical_scheme(sheet):
    from .geometry_editor import GeometryEditor, NewSchemeDialog

    if not sheet.confirm_replace_draft():
        return
    properties = NewSchemeDialog(sheet)
    if properties.exec() != QDialog.DialogCode.Accepted:
        return
    dialog = GeometryEditor(properties.scheme, sheet, new=True)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        try:
            sheet.commit_graphical_scheme(dialog.edited_scheme, auto_bind=True)
        except (ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(sheet, tr('Новая схема'), message(exc))


def editor_changed(sheet):
    sheet.invalidate_calculation()
    sheet.editor_dirty = True
    sheet.clear_result_view()
    text = tr('Редактор изменён. Проверьте revision и примените редакцию перед расчётом.')
    sheet.status.setText(text)
    sheet.explanation.setPlainText(text)


def apply_editor(sheet):
    try:
        scheme = json.loads(sheet.editor.toPlainText())
        sheet.commit_graphical_scheme(scheme)
    except (ValueError, KeyError, TypeError) as exc:
        QMessageBox.warning(sheet, tr('Редакция схемы'), message(exc))


def confirm_replace_draft(sheet):
    if not sheet.editor_dirty:
        return True
    choice = QMessageBox.question(
        sheet, tr('Черновик схемы'), tr('Сохранить неприменённый черновик перед сменой схемы?'),
        QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.Cancel,
    )
    return choice == QMessageBox.StandardButton.Discard or (
        choice == QMessageBox.StandardButton.Save and sheet.save_scheme()
    )


def save_scheme(sheet):
    from autopyro.data_io.file_output import atomic_output

    if not sheet.scheme and not sheet.editor_dirty:
        return False
    draft = sheet.editor_dirty
    content = sheet.editor.toPlainText()
    path, _ = QFileDialog.getSaveFileName(
        sheet, tr('Сохранить черновик схемы') if draft else tr('Сохранить схему'),
        "scheme-draft.json" if draft else "scheme.json", "JSON (*.json)"
    )
    if path:
        try:
            with atomic_output(path) as temporary:
                temporary.write_text(content, encoding="utf-8")
            return True
        except (OSError, ValueError) as exc:
            QMessageBox.warning(sheet, tr('Схема'), message(exc))
    return False
