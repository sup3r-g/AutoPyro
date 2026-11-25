"""Manage formula bindings, laboratory units and validation feedback."""

from __future__ import annotations

from autopyro.analysis.calculations import FORMULAS, input_unit

from ..i18n import tr, message
from ..widgets.documents import refill_combo


def restore_parameters(editor):
    editor.parameter_defaults.restore()
    editor.invalidate()
    editor.sample_search.clear()
    editor.preview.setRowCount(0)
    editor.summary.clear()


def show_input_errors(editor, errors):
    for role, label in editor.input_feedback.items():
        reason = message(errors[role]) if role in errors else ""
        label.setText(f"⚠ {reason}" if reason else "")
        label.setVisible(bool(reason))
        for widget in editor.inputs[role]:
            widget.setAccessibleDescription(reason)


def input_column_changed(editor, role):
    combo, unit = editor.inputs[role]
    column = next((c for c in editor.owner.model.project.columns if c.id == combo.currentData()), None)
    if column is not None:
        index = unit.findText(input_unit(role, column.unit))
        if index >= 0:
            unit.setCurrentIndex(index)
    if hasattr(editor, "lab_inputs"):
        editor.update_lab_choices()
    editor.invalidate()


def lab_unit(formula, column):
    unit = (column.unit or "").strip()
    if formula == "HI":
        return "mg HC/g TOC" if unit.casefold().replace(" ", "") in {"mghc/gtoc", "mg/gtoc"} else unit
    return {"доля": "1", "fraction": "1"}.get(unit.casefold(), unit)


def lab_column_changed(editor, formula):
    combo, unit, _ = editor.lab_inputs[formula]
    column = next((c for c in editor.owner.model.project.columns if c.id == combo.currentData()), None)
    if column is not None:
        index = unit.findData(editor.lab_unit(formula, column))
        if index >= 0:
            unit.setCurrentIndex(index)
    editor.invalidate()


def refresh_columns(editor):
    choices = [(tr('Не выбрано'), "")] + [
        (column.name + (f" [{column.unit}]" if column.unit else ""), column.id)
        for column in editor.owner.model.project.columns
        if not column.metadata.get("calculation") and not column.metadata.get("method")]
    for role, (combo, _) in editor.inputs.items():
        if refill_combo(combo, choices):
            editor.input_column_changed(role)
    for formula, (combo, _, _) in editor.lab_inputs.items():
        if refill_combo(combo, choices):
            editor.lab_column_changed(formula)
    editor.update_lab_choices()


def update_lab_choices(editor):
    for formula, (combo, _, _) in editor.lab_inputs.items():
        blocked = {editor.inputs[role][0].currentData() for role in FORMULAS[formula]["roles"]}
        warning = tr('Лабораторная колонка уже используется как вход формулы')
        for index in range(combo.count()):
            cid = combo.itemData(index)
            available = not cid or cid not in blocked
            item = combo.model().item(index)
            item.setEnabled(available)
            item.setToolTip("" if available else warning)
        selected = combo.currentData()
        combo.setAccessibleDescription(warning if selected and selected in blocked else "")
