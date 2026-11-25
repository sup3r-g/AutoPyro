"""Manage method options, analytical scope and bindings to source columns."""

from __future__ import annotations

import html

from PySide6.QtWidgets import QComboBox, QFormLayout, QGridLayout, QGroupBox, QLabel, QWidget

from autopyro.analysis.nonpyro import (
    METHODS,
    REPORTED_BIOMARKER_METHODS,
    UNITS,
    field_matches,
    source_identity,
    suggest_bindings,
    validate_config,
    variant_tag,
)

from ..geochemistry import binding_title, group_label
from ..i18n import tr, message
from ..widgets.parameter_reset import install_parameter_resets
from ..widgets.parameters import annotate_parameters


def restore_parameters(editor):
    editor.loading = True
    editor.form.reset_defaults()
    editor.loading = False
    editor.method_changed(use_saved=False)


def connect_model(editor):
    editor.owner.model.changed.connect(editor.input_changed)


def input_changed(editor):
    if not editor.promoting:
        editor.refresh_columns()
        editor.current_report()
        editor.update_workflow()


def source_schema(editor):
    return tuple((column.id, column.name, column.unit, source_identity(column))
                 for column in editor.owner.model.project.columns
                 if not column.metadata.get("nonpyro") and not source_identity(column)[0].startswith("Py"))


def refresh_columns(editor):
    if editor._schema != editor.source_schema():
        preferred = {key: {"column_id": combo.currentData(), "unit": unit.currentText()}
                     for key, (combo, unit, _) in editor.roles.items()}
        editor.rebuild_bindings(preferred)


def update_workflow(editor):
    project = editor.owner.model.project
    editor.input_hint.setText(
        tr('Используются данные основной таблицы: проб — {0}, колонок — {1}. Выберите метод и выборку.', len(project.rows), len(project.columns))
        if project.columns else tr('Сначала загрузите измерения через «Импорт геохимии из БД…» или «Открыть таблицу как новый проект…» в меню «Файл». Затем выберите метод и выборку.'))
    definition = METHODS[editor.method.currentData()]
    required = [key for key, role in definition['roles'].items() if not role['optional']]
    missing = [tr(definition['roles'][key]['label']) for key in required
               if key not in editor.roles or not editor.roles[key][0].currentData()]
    editor.bindings_summary.setText(
        tr('Привяжите хотя бы одну полную пару пиков для расчёта отношения.')
        if editor.method.currentData() == 'terpane_hopane_profile'
        else tr('Для каждого C31–C35 привяжите сумму 22S+22R или оба эпимера.')
        if editor.method.currentData() == 'homohopane_distribution'
        else tr('Привяжите пики 22S и 22R хотя бы для одного гомолога.')
        if editor.method.currentData() == 'homohopane_epimers'
        else tr('Привяжите хотя бы один исходный показатель.')
        if editor.method.currentData() in REPORTED_BIOMARKER_METHODS
        else tr('Привяжите полный набор пиков хотя бы для одного ароматического индекса.')
        if editor.method.currentData() == 'aromatic_indices_from_peaks'
        else tr('Привяжите полный набор пиков хотя бы для одного показателя Rallakis.')
        if 'ratio_formulas' in definition
        else tr('Обязательные показатели: привязано {0} из {1}.', len(required) - len(missing), len(required))
    )
    scope_error = None
    try:
        rows = editor.row_ids()
    except ValueError as exc:
        rows = []
        scope_error = message(exc)
    if not editor.owner.model.project.columns:
        hint = tr('Сначала импортируйте данные в основную таблицу через меню «Файл».')
    elif scope_error:
        hint = tr('Шаг 1: проверьте выборку: {0}', scope_error)
    elif not rows:
        hint = tr('Шаг 1: выбранная выборка пуста. Измените область данных.')
    elif missing:
        hint = tr('Шаг 2: выберите колонки для показателей: {0}.', ', '.join(missing))
    else:
        try:
            validate_config(project, editor.config(), require_confirmation=False)
        except ValueError as exc:
            hint = tr('Проверьте привязки и условия: {0}', message(exc))
        else:
            hint = ''
    editor.preview_button.setEnabled(not hint)
    editor.preview_button.setToolTip(hint or tr('Показывает, для скольких проб доступны результаты с учётом QC.'))
    if not hint and (not editor.form.fields['context'][0].text().strip()
                     or not editor.form.fields['confirmed'][0].isChecked()):
        hint = tr('Шаг 3: укажите материал и условия анализа, затем подтвердите применимость методики.')
    editor.calculate_button.setEnabled(not hint)
    editor.calculate_button.setToolTip(hint or tr('Выполняет расчёт для текущей выборки. Исходные колонки сохраняются.'))
    editor.workflow_hint.setText(hint or tr('Данные и условия заданы. Можно рассчитать результаты.'))
    editor.update_result_actions()


def method_changed(editor, *_, use_saved=True):
    if editor.loading:
        return
    editor.loading = True
    key = editor.method.currentData()
    definition = METHODS[key]
    editor.results_model.set_rows([], headers=[
        'UUID', tr('Проба'), tr('Статус'),
        *[tr(label) + f' [{unit}]' for label, unit in definition['outputs'].values()],
        tr('Интерпретация'), tr('Оговорки / QC'),
    ])
    editor.results.setColumnHidden(0, True)
    editor.results.setColumnWidth(editor.results_model.columnCount() - 2, 310)
    editor.group.clear()
    for group in definition["groups"]:
        editor.group.addItem(group_label(group), group)
    if not definition["same_group"]:
        editor.group.insertItem(0, tr('Совместный профиль'), "")
        editor.group.setCurrentIndex(0)
    variant = (
        "vol"
        if key == "gas_ratios"
        else "h"
        if key in ("stereo_profile", "aromatic_profile", "sterane_distribution",
                   "terpane_hopane_profile", "homohopane_distribution", "homohopane_epimers",
                   "c27_diasterane_ratio", "aromatic_indices_from_peaks", "ster_hop_from_peaks"
                   ) or key in REPORTED_BIOMARKER_METHODS or key in (
                       "rallakis_aromatics", "rallakis_steroids", "rallakis_steranes", "rallakis_saturated")
        else "a"
        if key in ("gc_profile", "alkane_indices", "rallakis_gc")
        else ""
    )
    saved = editor.owner.model.project.view.get("nonpyro_configs", {}).get(key, {}) if use_saved else {}
    if saved and "variant" not in saved:
        selected_ids = {item["column_id"] for item in saved.get("inputs", {}).values()}
        tags = {variant_tag(source_identity(column)[1])
                for column in editor.owner.model.project.columns if column.id in selected_ids}
        tags.discard("")
        if len(tags) == 1:
            variant = tags.pop()
    editor.form.load(
        {
            "variant": variant,
            "context": "",
            "confirmed": False,
            "isotope_reference": "",
            "material": "",
            "population_confirmed": False,
            **{
                k: v
                for k, v in saved.items()
                if k not in ("inputs", "scope_row_ids", "method")
            },
            "confirmed": False,
        }
    )
    editor.notes.setHtml(
        "<p>"
        + html.escape(tr(definition["note"]))
        + "</p>"
        + "".join(
            f'<p><a href="{html.escape(source, quote=True)}">{html.escape(source)}</a></p>'
            for source in definition["sources"]
        )
    )
    editor.update_applicability(key)
    editor.loading = False
    editor.rebuild_bindings(saved.get("inputs"))
    editor.invalidate()


def update_applicability(editor, method_id):
    relevant = {
        "isotope_reference": method_id in ("sofer", "gas_isotopes"),
        "material": method_id == "sofer",
        "population_confirmed": method_id == "vitrinite",
        "closure_tolerance": method_id in ("sara", "microscopy"),
        "comparison_tolerance": method_id == "sofer",
    }
    sections = set()
    for key, visible in relevant.items():
        widget = editor.form.fields[key][0]
        holder = widget
        while holder.parentWidget() is not None:
            layout = holder.parentWidget().layout()
            if isinstance(layout, QFormLayout) and layout.indexOf(holder) >= 0:
                break
            holder = holder.parentWidget()
        layout.setRowVisible(holder, visible)
        widget.setVisible(visible)
        sections.add(holder.parentWidget())
    for section in sections:
        layout = section.layout()
        section.setVisible(any(layout.isRowVisible(row) for row in range(layout.rowCount())))


def hints_changed(editor, *_):
    if not editor.loading:
        variant = editor.variant.currentData()
        suggested = suggest_bindings(editor.owner.model.project, editor.method.currentData(),
                                     editor.group.currentData(), variant)
        columns = {column.id: column for column in editor.owner.model.project.columns}
        for key, (combo, unit, _) in editor.roles.items():
            column = columns.get(combo.currentData())
            tag = variant_tag(source_identity(column)[1]) if column else ""
            if column and (not variant or not tag or tag == variant):
                suggested[key] = {'column_id': combo.currentData(), 'unit': unit.currentText()}
        editor.rebuild_bindings(suggested)
        editor.invalidate()


def rebuild_bindings(editor, preferred=None):
    editor.loading = True
    key = editor.method.currentData()
    definition = METHODS[key]
    proposed = preferred if preferred is not None else suggest_bindings(
        editor.owner.model.project,
        key,
        editor.group.currentData(),
        editor.variant.currentData(),
    )
    editor.roles = {}
    while editor.bindings_layout.count():
        item = editor.bindings_layout.takeAt(0)
        if item.widget() is not None:
            item.widget().hide()
            item.widget().deleteLater()
    compact = len(definition["roles"]) > 12
    if compact:
        grid_panel = QWidget()
        grid = QGridLayout(grid_panel)
        editor.bindings_layout.addWidget(grid_panel)
        for column, title in enumerate((tr('Пик'), tr('Колонка'), tr('Единица'))):
            grid.addWidget(QLabel(title), 0, column)
        grid.setColumnStretch(1, 1)
    for role_key, role_def in definition["roles"].items():
        group_box = QGroupBox(binding_title(key, role_key, role_def))
        group_box.setObjectName('geochemistryBinding_' + role_key)
        form = QFormLayout(group_box)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        label = QLabel(tr(role_def["label"]) + '\n' + (tr('Необязательный показатель') if role_def["optional"] else tr('Обязательный показатель')))
        label.setWordWrap(True)
        if not compact:
            form.addRow(label)
        combo = QComboBox()
        combo.addItem(tr('Выберите колонку…'), "")
        for column in editor.owner.model.project.columns:
            group, name = source_identity(column)
            if column.metadata.get("nonpyro") or group.startswith("Py"):
                continue
            variant = editor.variant.currentData()
            tag = variant_tag(name)
            if (not group or field_matches(group, name, role_def)) and (not variant or not tag or tag == variant):
                combo.addItem(column.name, column.id)
        units = QComboBox()
        units.addItems(list(UNITS[role_def["unit_family"]]))
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(8 if compact else 12)
        source_unit = QLabel()
        units.setToolTip(tr('Укажите единицу чисел в выбранной колонке. Это не команда преобразования единиц.'))
        if compact:
            row = len(editor.roles) + 1
            title = QLabel(binding_title(key, role_key, role_def))
            title.setToolTip(label.text())
            grid.addWidget(title, row, 0)
            grid.addWidget(combo, row, 1)
            grid.addWidget(units, row, 2)
            source_unit.setParent(grid_panel)
            source_unit.hide()
            group_box.deleteLater()
        else:
            form.addRow(tr('Исходная колонка'), combo)
            form.addRow(tr('Единица значений'), units)
            form.addRow(tr('В исходных данных'), source_unit)
            editor.bindings_layout.addWidget(group_box)
        editor.roles[role_key] = (combo, units, source_unit)
        item = proposed.get(role_key, {})
        combo.setCurrentIndex(max(0, combo.findData(item.get("column_id", ""))))
        editor.update_unit(role_key)
        if item.get("unit") in UNITS[role_def["unit_family"]]:
            units.setCurrentText(item["unit"])
        combo.currentIndexChanged.connect(
            lambda *args, key=role_key: editor.role_changed(key)
        )
        units.currentIndexChanged.connect(editor.invalidate)
        if not compact:
            annotate_parameters(group_box)
            install_parameter_resets(group_box, changed=editor.invalidate)
            annotate_parameters(group_box)
    editor.bindings_layout.addStretch()
    editor._schema = editor.source_schema()
    editor.loading = False


def update_unit(editor, key):
    combo, units, source_unit = editor.roles[key]
    column = next(
        (
            c
            for c in editor.owner.model.project.columns
            if c.id == combo.currentData()
        ),
        None,
    )
    source_unit.setText(column.unit or tr('Единица не указана') if column else tr('Колонка не выбрана'))
    if column and units.findText(column.unit) >= 0:
        units.setCurrentText(column.unit)


def role_changed(editor, key):
    editor.update_unit(key)
    editor.invalidate()


def row_ids(editor):
    return editor.owner.analysis_scope.rows()


def config(editor):
    options = editor.form.values()
    for key in ("group", "scope"):
        options.pop(key, None)
    options.update(
        version=1,
        inputs={
            key: {"column_id": combo.currentData(), "unit": units.currentText()}
            for key, (combo, units, row) in editor.roles.items()
            if combo.currentData()
        },
        scope_row_ids=editor.row_ids(),
    )
    return options
