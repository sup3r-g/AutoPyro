"""Read and restore curve settings, column bindings and model definitions."""

from __future__ import annotations

from PySide6.QtWidgets import QInputDialog
import numpy as np

from autopyro.curves.curve_constraints import parameter_model
from autopyro.curves.curve_fitting import custom_model, standard_model
from autopyro.curves.curve_validation import NESTED_STRATEGIES

from ..i18n import tr, message
from ..theme import icon
from ..widgets.documents import refill_combo


_FIT_DEFAULTS = {"sx": None, "sy": None, "model": "linear", "method": "ols", "degree": 2,
                 "expression": "a*x + b", "parameters": "a, b", "initial": "", "absolute_sigma": False,
                 "loss": "linear", "f_scale": 1., "lower_bounds": "", "upper_bounds": "", "band_method": "auto",
                 "bootstrap_resamples": 500, "bootstrap_seed": 0, "prediction_sigma": 0., "bootstrap_group": None,
                 "bootstrap_block_length": 1, "parameter_rules": "", "starts": 1, "start_seed": 0}


def _column_label(column):
    return f"{column.name} [{column.unit}]" if column.unit else column.name


def _model_options(settings):
    """Read one persisted model definition; independent of widgets and workers."""
    model = (custom_model(settings["expression"], tuple(p.strip() for p in settings["parameters"].split(",")))
             if settings["model"] == "custom" else standard_model(settings["model"], degree=settings["degree"]))
    model = parameter_model(model, settings.get("parameter_rules", ""))
    initial = ([float(v.strip()) for v in settings["initial"].split(",")]
               if settings["initial"].strip() else None)
    if settings["model"] == "custom" and initial is None:
        raise ValueError("Для пользовательской модели задайте начальные значения параметров")
    bounds = tuple([float(v.strip()) for v in settings[key].split(",")]
                   if settings.get(key, "").strip() else default
                   for key, default in (("lower_bounds", -np.inf), ("upper_bounds", np.inf)))
    return model, initial, bounds


def reset_parameter_inputs(editor, *args):
    # Bounds and starts are positional: never reuse them for a different model.
    editor.initial.clear()
    editor.lower_bounds.clear()
    editor.upper_bounds.clear()
    editor.parameter_rules.clear()


def sync_columns(editor, *args):
    choices = [(_column_label(c), c.id)
               for c in editor.owner.model.project.columns
               if c.kind == "number" and not c.metadata.get("method") and not c.metadata.get("curve_fit")]
    for widget in (editor.x_column, editor.y_column):
        refill_combo(widget, choices)
    for widget in (editor.sx_column, editor.sy_column):
        refill_combo(widget, [(tr("Не задана (σ = 1)"), None), *choices])
    if hasattr(editor, "cv_key"):
        refill_combo(editor.cv_key, [(c.name, c.id) for c in editor.owner.model.project.columns])
        editor.refill_holdout_columns(editor.cv_holdout.currentData())
    editor.sync_joint_columns()
    group_choices = [(tr("Независимые строки"), None),
                     *((c.name, c.id) for c in editor.owner.model.project.columns)]
    group_id = editor.bootstrap_group.currentData()
    if group_id and group_id not in {cid for _, cid in group_choices}:
        group_choices.append((tr("Колонка групп отсутствует"), group_id))
    if refill_combo(editor.bootstrap_group, group_choices):
        editor.bootstrap_group.setItemIcon(0, icon("table-2"))
        group_icon = icon("layers")
        for index in range(1, editor.bootstrap_group.count()):
            editor.bootstrap_group.setItemIcon(index, group_icon)
    editor.invalidate()


def update_controls(editor, *args):
    custom = editor.model_choice.currentData() == "custom"
    editor.expression.setEnabled(custom)
    editor.parameter_names.setEnabled(custom)
    editor.degree.setEnabled(editor.model_choice.currentData() == "polynomial")
    editor.sx_column.setEnabled(editor.method.currentData() == "odr")
    editor.f_scale.setEnabled(editor.loss.currentData() != "linear")
    editor.update_bootstrap_controls()
    editor.model_formula.setText(editor.expression.text() if custom else
                               standard_model(editor.model_choice.currentData(), degree=editor.degree.value()).formula)
    model = None
    try:
        model = (custom_model(editor.expression.text(), tuple(p.strip() for p in editor.parameter_names.text().split(",")))
                 if custom else standard_model(editor.model_choice.currentData(), degree=editor.degree.value()))
        editor.free_parameter_label.setText(", ".join(parameter_model(model, editor.parameter_rules.text()).parameters))
    except ValueError as exc:
        editor.free_parameter_label.setText(message(exc))
    editor.refresh_joint_parameters(model)


def update_bootstrap_controls(editor, *args):
    paired_test = (editor.cv_strategy.currentData() not in NESTED_STRATEGIES and editor.cv_holdout.currentData() is not None
                   and all(combo.currentData() for combo in editor.cv_pair))
    bootstrap_bands_enabled = editor.band_method.currentData() != "analytic"
    bootstrap_enabled = bootstrap_bands_enabled or paired_test
    for widget in (editor.bootstrap_resamples, editor.bootstrap_seed, editor.bootstrap_group):
        widget.setEnabled(bootstrap_enabled)
    editor.cv_pair_margin.setEnabled(paired_test)
    editor.prediction_sigma.setEnabled(bootstrap_bands_enabled)
    editor.bootstrap_block_length.setEnabled(bootstrap_enabled and editor.bootstrap_group.currentData() is None)


def refresh_saved_models(editor):
    choices = [(tr("Выберите сохранённую модель"), None)]
    choices.extend((name, name) for name in editor.owner.model.project.view.get("curve_fitting_models", {}))
    refill_combo(editor.saved_models, choices)


def load_custom_model(editor, *args):
    definition = editor.owner.model.project.view.get("curve_fitting_models", {}).get(editor.saved_models.currentData())
    if definition is None:
        return
    editor.model_choice.setCurrentIndex(editor.model_choice.findData("custom"))
    editor.expression.setText(definition["expression"])
    editor.parameter_names.setText(definition["parameters"])
    editor.initial.setText(definition["initial"])
    editor.lower_bounds.setText(definition.get("lower_bounds", ""))
    editor.upper_bounds.setText(definition.get("upper_bounds", ""))
    editor.parameter_rules.setText(definition.get("parameter_rules", ""))


def save_custom_model(editor):
    try:
        model = custom_model(editor.expression.text(), tuple(p.strip() for p in editor.parameter_names.text().split(",")))
        model = parameter_model(model, editor.parameter_rules.text())
        initial = np.array([float(v.strip()) for v in editor.initial.text().split(",")])
        if initial.shape != (len(model.parameters),) or not np.all(np.isfinite(initial)):
            raise ValueError("Укажите конечные начальные значения для каждого параметра")
    except ValueError as exc:
        editor.status.setText(message(exc))
        return
    name, accepted = QInputDialog.getText(editor, tr("Моя модель"), tr("Название (существующее будет обновлено)"))
    name = name.strip()
    if not accepted or not name:
        return
    models = editor.owner.model.project.view.setdefault("curve_fitting_models", {})
    models[name] = {"expression": editor.expression.text(), "parameters": editor.parameter_names.text(),
                    "initial": editor.initial.text(), "lower_bounds": editor.lower_bounds.text(),
                    "upper_bounds": editor.upper_bounds.text(), "parameter_rules": editor.parameter_rules.text()}
    editor.refresh_saved_models()
    editor.saved_models.setCurrentIndex(editor.saved_models.findData(name))
    editor.owner.view_changed()


def settings(editor):
    return {key: widget.currentData() for key, widget in
            (("x", editor.x_column), ("y", editor.y_column), ("sx", editor.sx_column),
             ("sy", editor.sy_column), ("model", editor.model_choice), ("method", editor.method))} | {
                "degree": editor.degree.value(), "expression": editor.expression.text(),
                "parameters": editor.parameter_names.text(), "initial": editor.initial.text(),
                "absolute_sigma": editor.absolute_sigma.isChecked(),
                "loss": editor.loss.currentData(), "f_scale": editor.f_scale.value(),
                "lower_bounds": editor.lower_bounds.text(), "upper_bounds": editor.upper_bounds.text(),
                "band_method": editor.band_method.currentData(), "bootstrap_resamples": editor.bootstrap_resamples.value(),
                "bootstrap_seed": editor.bootstrap_seed.value(), "prediction_sigma": editor.prediction_sigma.value(),
                "bootstrap_group": editor.bootstrap_group.currentData(),
                "bootstrap_block_length": editor.bootstrap_block_length.value(),
                "parameter_rules": editor.parameter_rules.text(), "starts": editor.starts.value(), "start_seed": editor.start_seed.value()}


def restore_settings(editor, settings):
    settings = _FIT_DEFAULTS | settings
    for key, widget in (("x", editor.x_column), ("y", editor.y_column), ("sx", editor.sx_column),
                        ("sy", editor.sy_column), ("model", editor.model_choice), ("method", editor.method)):
        index = widget.findData(settings.get(key))
        if index >= 0:
            widget.setCurrentIndex(index)
    editor.degree.setValue(settings["degree"])
    for key, widget in (("expression", editor.expression), ("parameters", editor.parameter_names), ("initial", editor.initial)):
        widget.setText(settings[key])
    editor.absolute_sigma.setChecked(settings["absolute_sigma"])
    editor.loss.setCurrentIndex(max(0, editor.loss.findData(settings["loss"])))
    editor.f_scale.setValue(settings["f_scale"])
    editor.lower_bounds.setText(settings["lower_bounds"])
    editor.upper_bounds.setText(settings["upper_bounds"])
    editor.band_method.setCurrentIndex(max(0, editor.band_method.findData(settings["band_method"])))
    editor.bootstrap_resamples.setValue(settings["bootstrap_resamples"])
    editor.bootstrap_seed.setValue(settings["bootstrap_seed"])
    group_id = settings["bootstrap_group"]
    group_index = editor.bootstrap_group.findData(group_id)
    if group_id and group_index < 0:
        editor.bootstrap_group.addItem(icon("layers"), tr("Колонка групп отсутствует"), group_id)
        group_index = editor.bootstrap_group.count()-1
    editor.bootstrap_group.setCurrentIndex(max(0, group_index))
    editor.bootstrap_block_length.setValue(settings["bootstrap_block_length"])
    editor.prediction_sigma.setValue(settings["prediction_sigma"])
    editor.parameter_rules.setText(settings["parameter_rules"])
    editor.starts.setValue(settings["starts"])
    editor.start_seed.setValue(settings["start_seed"])
