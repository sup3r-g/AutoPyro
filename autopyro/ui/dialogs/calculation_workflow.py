"""Prepare formula calculations and apply a current preview to the project."""

from __future__ import annotations

from copy import deepcopy

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QDialogButtonBox

from autopyro.analysis.calculations import (
    FORMULAS,
    InputConfigurationError,
    compare_lab_results,
    diagnostic_counts,
    make_calculated_column,
    outside_scope_diagnostic,
)

from ..i18n import tr, message
from ..table import AddColumnCommand
from ..table.model import large_project
from ..tasks import TaskCancelled, run_with_progress
from ..widgets.documents import source_token
from ..widgets.results import sample_label_steps
from .calculation_details import calculation_statuses


def calculate(editor):
    editor.invalidate(announce=False)
    editor.refresh_columns()
    project = editor.owner.model.project
    token = source_token(editor.owner)
    revision = editor._preview_revision
    jobs, names = [], {c.name for c in project.columns}
    columns = {c.id: c for c in project.columns}
    comparisons = {}
    errors = {}
    text_numbers = editor.text_numbers.isChecked()
    try:
        scope = editor.owner.analysis_scope.rows()
        if not scope:
            editor.set_status(tr('В выбранной области нет проб.'))
            return
        for i, (formula, definition) in enumerate(FORMULAS.items()):
            create = editor.outputs.cellWidget(i, 0).isChecked()
            if formula in editor.lab_inputs:
                column, unit, tolerance = editor.lab_inputs[formula]
                lab = columns.get(column.currentData())
                if lab is not None:
                    declared = editor.lab_unit(formula, lab)
                    if lab.id in {editor.inputs[role][0].currentData() for role in definition["roles"]}:
                        errors[formula] = tr('Лабораторная колонка уже используется как вход формулы')
                    elif declared and declared != unit.currentData():
                        errors[formula] = tr('{0}: несовместимая единица лабораторной колонки ({1})', formula, lab.unit)
                    else:
                        comparisons[formula] = lab, unit.currentData(), tolerance.value()
            if not create and formula not in comparisons:
                continue
            name = editor.outputs.item(i, 2).text().strip() if create else formula
            if create and (not name or name in names):
                errors[formula] = tr('Названия новых колонок должны быть непустыми и уникальными')
                if formula not in comparisons:
                    continue
                create, name = False, formula
            if create:
                names.add(name)
            config = {
                "formula": formula,
                "version": 1,
                "analysis_context": editor.context.text().strip(),
                "compatible_confirmed": editor.confirmed.isChecked(),
                "check_context": True,
                "text_numbers": text_numbers,
                "scope_row_ids": scope,
                "inputs": {
                    role: {
                        "column_id": editor.inputs[role][0].currentData(),
                        "unit": editor.inputs[role][1].currentText(),
                    }
                    for role in definition["roles"]
                },
            }
            if editor.propagate_uncertainty.isChecked():
                config["propagate_uncertainty"] = True
            if editor.show_censored_ranges.isChecked():
                config["show_censored_ranges"] = True
            jobs.append((name, config, create))

        def calculate(progress=None):
            outputs, prepared, comparisons_by_formula = [], [], {}
            counts, mismatches, unavailable = {}, set(), set()
            failures = {}
            input_errors = {}
            for index, (name, config, create) in enumerate(jobs):
                formula = config["formula"]
                checkpoint = None
                if progress:
                    stage = f"{tr('Рассчитать предпросмотр')} · {formula}"
                    percent = index * 100 // len(jobs)

                    def checkpoint():
                        progress(stage, percent)

                    checkpoint()
                try:
                    column, values = make_calculated_column(project, name, deepcopy(config),
                                                           row_ids=scope, checkpoint=checkpoint)
                except InputConfigurationError as exc:
                    failures[formula] = str(exc)
                    input_errors.update(exc.fields)
                    continue
                except (ValueError, KeyError, TypeError) as exc:
                    failures[formula] = str(exc)
                    continue
                outputs.append((column, values))
                if create and any(info["status"] in ("ok", "censored_range")
                                  for info in column.metadata["diagnostics"].values()):
                    prepared.append((column, values))
                if formula in comparisons:
                    lab, unit, tolerance = comparisons[formula]
                    result = compare_lab_results(
                        project, lab, formula, values, scope, unit, tolerance,
                        text_numbers=text_numbers,
                        inputs=tuple(item["column_id"] for item in config["inputs"].values()),
                        checkpoint=checkpoint,
                    )
                    column.metadata["lab_comparison"] = {
                        "column_id": lab.id, "column_name": lab.name, "unit": unit,
                        "tolerance": tolerance, "tolerance_unit": FORMULAS[formula]["unit"],
                    }
                    comparisons_by_formula[formula] = result
                    counts[formula] = result["status"].value_counts().to_dict()
                    mismatches.update(result.index[result["status"].eq("mismatch")])
                    unavailable.update(result.index[~result["status"].isin(("match", "mismatch"))])
            labels = {}
            if outputs:
                steps = sample_label_steps(project, scope)
                try:
                    while True:
                        next(steps)
                        if progress:
                            progress(tr('Рассчитать предпросмотр'), 99)
                except StopIteration as done:
                    labels = done.value
            return (outputs, prepared, comparisons_by_formula, counts, mismatches, unavailable,
                    failures, labels, input_errors)

        result = (run_with_progress(editor, tr('Рассчитать предпросмотр'), calculate, cancellable=True)
                  if jobs and large_project(project) else calculate())
    except TaskCancelled:
        editor.set_status(tr('Расчёт отменён.'))
        return
    except (ValueError, KeyError, TypeError) as exc:
        editor.set_status(tr('Ошибка: {0}', message(exc)))
        return
    if token != source_token(editor.owner) or revision != editor._preview_revision:
        editor.invalidate()
        editor.set_status(tr("Данные или выборка изменились. Рассчитайте предпросмотр заново."))
        return
    (editor._preview_outputs, editor.prepared, editor._comparison_rows, counts,
     editor._mismatch_rows, editor._unavailable_rows, failures, editor._sample_labels, input_errors) = result
    editor.show_input_errors(input_errors)
    editor._preview_rows = scope if editor._preview_outputs else []
    errors.update({formula: message(reason) for formula, reason in failures.items()})
    with QSignalBlocker(editor.outputs):
        for i, (formula, definition) in enumerate(FORMULAS.items()):
            tooltip = errors.get(formula, f"{definition['equation']} [{definition['unit']}]")
            editor.outputs.item(i, 1).setToolTip(tooltip)
    editor._preview_source = token
    editor.sample_search.setEnabled(bool(editor._preview_rows))
    editor.mismatches_only.setEnabled(bool(editor._comparison_rows))
    editor.unavailable_only.setEnabled(bool(editor._comparison_rows))
    editor.export_button.setEnabled(bool(editor._comparison_rows))
    editor.save_comparison_button.setEnabled(bool(editor._comparison_rows))
    statuses = calculation_statuses()
    scope_summary = [tr('Проб в области расчёта: {0}', len(scope))] if editor._preview_outputs else []
    editor._preview_summary = " · ".join(scope_summary + [
            f"{c.name}: "
            + ", ".join(
                f"{statuses.get(k, k)}: {n}"
                for k, n in diagnostic_counts(c).items()
                if k != "outside_scope"
            )
            for c, _ in editor._preview_outputs
        ] + [
            tr('{0}: в допуске {1}, расхождений {2}, сравнение недоступно {3}',
               formula, tally.get("match", 0), tally.get("mismatch", 0),
               sum(tally.values()) - tally.get("match", 0) - tally.get("mismatch", 0))
            for formula, tally in counts.items()
        ] + [f"{formula}: {reason}" for formula, reason in errors.items()])
    editor.render_preview()
    if editor.parameter_tabs.indexOf(editor._compact_results) >= 0:
        editor.parameter_tabs.setCurrentWidget(editor._compact_results)
        editor._compact_results.ensureWidgetVisible(editor.preview, 0, 4)
    editor.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(editor.prepared))
    editor.announce(editor.summary, editor.summary.text())


def apply(editor):
    if not editor.prepared:
        return
    if editor._preview_source != source_token(editor.owner):
        editor.invalidate()
        editor.set_status(tr("Данные или выборка изменились. Рассчитайте предпросмотр заново."))
        return
    editor.owner.model.undo.beginMacro(tr('Вычисляемые показатели пиролиза'))
    try:
        for column, values in editor.prepared:
            diagnostics = column.metadata["diagnostics"]
            if len(diagnostics) < len(editor.owner.model.project.rows):
                column.metadata["diagnostics"] = {
                    row.id: diagnostics[row.id] if row.id in diagnostics else outside_scope_diagnostic()
                    for row in editor.owner.model.project.rows
                }
            editor.owner.model.undo.push(
                AddColumnCommand(editor.owner.model, column, values)
            )
    finally:
        editor.owner.model.undo.endMacro()
    if editor.property("workspaceTracked"):
        editor.invalidate()
        editor.set_status(tr("Вычисляемые колонки созданы. Доступна отмена."))
    else:
        editor.accept()
