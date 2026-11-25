"""Save, load and repair report templates and apply built-in compositions."""

from __future__ import annotations

from dataclasses import asdict, replace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QInputDialog,
    QFormLayout,
    QLabel,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from autopyro.reports.reporting import ReportOptions
from autopyro.reports.well_profile import FORMATION_NAMES, WELL_NAMES, find_columns

from ..i18n import tr
from .report_preview import resize_to_screen


BUILTIN_REPORTS = {
    "Паспорт скважины": (
        ("well", "formation"),
        ("summary", "source", "method", "profile", "comparison", "comparison_figures", "stratigraphy", "figures",
         "interpretation", "statistics", "joint_curve_fit", "quality", "conclusions", "data"),
    ),
    "Характеристика пласта": (
        ("formation", "well"),
        ("summary", "source", "method", "stratigraphy", "comparison", "comparison_figures", "statistics",
         "figures", "joint_curve_fit", "interpretation", "nonpyro", "quality", "conclusions", "data"),
    ),
    "Контроль качества": (
        (),
        ("summary", "source", "quality", "method", "statistics", "conclusions", "data"),
    ),
}


GROUP_ALIASES = {
    "well": WELL_NAMES,
    "formation": FORMATION_NAMES,
}


PLOT_KEY_ROLE = Qt.ItemDataRole.UserRole + 1


def legacy_plot_key(label):
    if label in ("Текущий график", "Current plot"):
        return "current"
    for prefix in ("Сохранённый: ", "Saved: "):
        if label.startswith(prefix):
            return "saved:" + label[len(prefix):]
    return "saved:" + label


def save_template(sheet):
    options, _ = sheet.composition()
    if not options.column_ids:
        sheet.announce_status(tr('Выберите колонки отчёта.'))
        return
    name, accepted = QInputDialog.getText(sheet, tr('Шаблон отчёта'), tr('Название шаблона'))
    name = name.strip()
    if not accepted or not name:
        return
    templates = sheet.owner.model.project.view.setdefault("report_templates", {})
    if name in templates and QMessageBox.question(
        sheet, tr('Шаблон отчёта'), tr('Заменить шаблон «{0}»?', name)
    ) != QMessageBox.StandardButton.Yes:
        return
    plots = [sheet.plots.item(i) for i in range(sheet.plots.count())]
    templates[name] = sheet.template_record(options, {
        item.data(PLOT_KEY_ROLE): item.text() for item in plots
        if item.checkState() == Qt.CheckState.Checked
    })
    sheet.owner.view_changed()
    sheet.announce_status(tr('Шаблон «{0}» сохранён в проекте.', name))


def template_record(sheet, options, plots):
    referenced = {*options.column_ids, *options.group_ids, options.profile_depth_id}
    return {
        "options": asdict(replace(options, conclusions="", scope="", joint_snapshot_id=None)),
        "column_labels": {
            column.id: column.name for column in sheet.owner.model.project.columns
            if column.id in referenced
        },
        "plot_refs": plots,
    }


def load_template(sheet):
    templates = sheet.owner.model.project.view.get("report_templates", {})
    choices = sorted(templates)
    builtins = {}
    for name in BUILTIN_REPORTS:
        label = tr('Встроенный: {0}', tr(name))
        while label in choices:
            label += " ·"
        builtins[label] = name
        choices.append(label)
    name, accepted = QInputDialog.getItem(
        sheet, tr('Шаблон отчёта'), tr('Выберите шаблон'), choices, editable=False
    )
    if not accepted:
        return
    if name in builtins:
        sheet.apply_builtin_template(builtins[name])
        return
    saved = templates[name]
    values = dict(saved["options"])
    values.setdefault("detailed_correlations", True)
    for name_of_limit in ("qc_limit", "protocol_limit"):
        values.setdefault(name_of_limit, values.get("data_limit", 100))
    options = ReportOptions(**values)
    plot_refs = saved.get("plot_refs")
    if plot_refs is None:
        plot_refs = {legacy_plot_key(label): label for label in saved.get("plots", ())}
    column_items = [sheet.columns.item(i) for i in range(sheet.columns.count())]
    plot_items = [sheet.plots.item(i) for i in range(sheet.plots.count())]
    columns = {item.data(Qt.ItemDataRole.UserRole): item.text() for item in column_items}
    plots = {item.data(PLOT_KEY_ROLE): item.text() for item in plot_items
             if sheet.plot_references_exist(item.data(Qt.ItemDataRole.UserRole), columns)}
    roles = {}
    for ids, role in (
        (options.column_ids, tr('показатель')),
        (options.group_ids, tr('группировка')),
        ((options.profile_depth_id,) if options.profile_depth_id else (), tr('глубина профиля')),
    ):
        for cid in ids:
            roles.setdefault(cid, []).append(role)
    lost = []
    for cid, uses in roles.items():
        depth = cid == options.profile_depth_id
        if cid not in columns or (depth and sheet.profile_depth.findData(cid) < 0):
            label = saved.get("column_labels", {}).get(cid) or cid
            candidates = {key: value for key, value in columns.items()
                          if not depth or sheet.profile_depth.findData(key) >= 0}
            lost.append(("column", cid, f"{label} · {', '.join(uses)}", candidates))
    for key, label in plot_refs.items():
        if key not in plots:
            lost.append(("plot", key, tr('График: {0}', label), plots))
    if options.map_view_name and sheet.map_view.findData(options.map_view_name) < 0:
        candidates = {sheet.map_view.itemData(i): sheet.map_view.itemText(i)
                      for i in range(1, sheet.map_view.count())}
        lost.append(("map", options.map_view_name, f"{tr('Карта')} · {options.map_view_name}", candidates))
    if lost:
        repaired = sheet.repair_template(options, plot_refs, lost)
        if repaired is None:
            return
        options, plot_refs = repaired
        templates[name] = sheet.template_record(options, {key: plots[key] for key in plot_refs})
        sheet.owner.view_changed()
    sheet.title.setText(options.title)
    sheet.author.setText(options.author)
    sheet.group.setCurrentIndex(sheet.group.findData(next(iter(options.group_ids), None)))
    sheet.subgroup.setCurrentIndex(sheet.subgroup.findData(
        options.group_ids[1] if len(options.group_ids) > 1 else None
    ))
    sheet.profile_depth.setCurrentIndex(sheet.profile_depth.findData(options.profile_depth_id))
    sheet.map_view.setCurrentIndex(max(0, sheet.map_view.findData(options.map_view_name)))
    sheet.set_logo(options.logo_image)
    sheet.limit.setValue(options.data_limit)
    sheet.qc_limit.setValue(options.qc_limit)
    sheet.protocol_limit.setValue(options.protocol_limit)
    sheet.qc_open_only.setChecked(options.qc_open_only)
    sheet.text_numbers.setChecked(options.text_numbers)
    sheet.interpolated.setChecked(options.include_interpolated)
    sheet.detailed_correlations.setChecked(options.detailed_correlations)
    sheet.interpretation.setChecked(options.include_interpretation)
    sheet.stratigraphy.setChecked(options.include_stratigraphy)
    sheet.nonpyro.setChecked(options.include_nonpyro)
    sheet.section_order = options.sections
    sheet.section_titles = options.section_titles
    sheet.block_comments = options.block_comments
    for items, key, selected in (
        (column_items, lambda item: item.data(Qt.ItemDataRole.UserRole), options.column_ids),
        (plot_items, lambda item: item.data(PLOT_KEY_ROLE), plot_refs),
    ):
        for item in items:
            item.setCheckState(Qt.CheckState.Checked if key(item) in selected else Qt.CheckState.Unchecked)
    sheet.invalidate()
    if lost:
        sheet.announce_status(tr('Шаблон «{0}» восстановлен в проекте. Соберите отчёт.', name))


def plot_references_exist(config, columns):
    kind = config.get("kind", "scatter")
    if kind == "multi_depth":
        required = [*config.get("track_ids", ()), config.get("y_id")]
    elif ":" in kind:
        from autopyro.visualization.plot_catalog import library_parameters
        from autopyro.visualization.plot_controls import controls_for

        try:
            parameters = library_parameters(kind)
        except KeyError:
            return False
        if parameters == ["matrix"]:
            required = list(config.get("track_ids", ()))
        else:
            bindings = {arg: config.get(axis + "_id") for axis, arg in controls_for(kind).axes}
            bindings.update(config.get("library_bindings", {}))
            optional = {"hue", "size", "style", "weights", "xerr", "yerr", "zerr"}
            if kind.startswith("seaborn:"):
                optional.add("y")
            if any(not bindings.get(key) for key in parameters if key not in optional):
                return False
            required = [bindings[key] for key in parameters if bindings.get(key)]
    elif config.get("interpretation_panels"):
        required = [panel["binding"].get(axis) for panel in config["interpretation_panels"]
                    if panel.get("binding") for axis in ("x_id", "y_id")]
    else:
        required = [config.get("x_id")]
        if kind in {"scatter", "depth"}:
            required.append(config.get("y_id"))
    style_keys = ("color_by", "shape_by", "size_by", "label_by") if kind in {
        "scatter", "depth", "multi_depth",
    } else ("color_by",)
    required.extend(config[key] for key in style_keys if config.get(key))
    if config.get("depth_bottom_id"):
        required.append(config["depth_bottom_id"])
    if config.get("stratigraphy"):
        binding = config["stratigraphy"].get("binding", {})
        required.extend(binding.get(key) for key in ("well_id", "depth_id"))
        if binding.get("sample_mode") == "interval":
            required.append(binding.get("bottom_id"))
    return bool(required) and all(cid in columns for cid in required)


def repair_template(sheet, options, plot_refs, lost):
    dialog = QDialog(sheet)
    dialog.setWindowTitle(tr('Восстановление ссылок шаблона'))
    resize_to_screen(dialog, 720, 450)
    layout = QVBoxLayout(dialog)
    help_text = QLabel(tr('Для каждой потерянной ссылки выберите замену или явно исключите её из шаблона.'))
    help_text.setWordWrap(True)
    layout.addWidget(help_text)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    fields = QWidget()
    form = QFormLayout(fields)
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    scroll.setWidget(fields)
    layout.addWidget(scroll)
    replacements = {}
    for kind, key, label, candidates in lost:
        combo = QComboBox()
        combo.addItem(tr('Выберите замену…'))
        combo.addItem(tr('Исключить из шаблона'))
        for cid, name in candidates.items():
            combo.addItem(name, cid)
        form.addRow(label, combo)
        replacements[kind, key] = combo
    note = QLabel()
    note.setWordWrap(True)
    layout.addWidget(note)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
    apply = buttons.addButton(tr('Восстановить и применить'), QDialogButtonBox.ButtonRole.AcceptRole)
    layout.addWidget(buttons)
    result = None

    def update():
        nonlocal result
        result = None
        apply.setEnabled(False)
        if any(combo.currentIndex() == 0 for combo in replacements.values()):
            note.setText(tr('Обработайте все потерянные ссылки.'))
            return

        def resolve(kind, key):
            combo = replacements.get((kind, key))
            return combo.currentData() if combo is not None else key

        def column_ids(keys):
            return tuple(dict.fromkeys(cid for key in keys if (cid := resolve("column", key)) is not None))

        selected = column_ids(options.column_ids)
        if not selected:
            note.setText(tr('В шаблоне должна остаться хотя бы одна колонка отчёта.'))
            return
        result = (
            replace(options, column_ids=selected, group_ids=column_ids(options.group_ids),
                    profile_depth_id=resolve("column", options.profile_depth_id),
                    map_view_name=resolve("map", options.map_view_name)),
            tuple(dict.fromkeys(key for old in plot_refs if (key := resolve("plot", old)) is not None)),
        )
        note.clear()
        apply.setEnabled(True)

    for combo in replacements.values():
        combo.currentIndexChanged.connect(update)
    update()
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    return result if dialog.exec() else None


def apply_builtin_template(sheet, name):
    roles, sheet.section_order = BUILTIN_REPORTS[name]
    sheet.section_titles = {}
    sheet.block_comments = {}
    sheet.profile_depth.setCurrentIndex(0)
    sheet.title.setText(tr(name))
    columns = sheet.owner.model.project.columns
    missing = []
    for control, role in zip((sheet.group, sheet.subgroup), (*roles, None, None)):
        matches = find_columns(columns, GROUP_ALIASES[role]) if role else []
        control.setCurrentIndex(control.findData(matches[0]) if len(matches) == 1 else 0)
        if role and len(matches) != 1:
            missing.append(tr('скважина') if role == "well" else tr('пласт'))
    sheet.interpretation.setChecked("interpretation" in sheet.section_order)
    sheet.stratigraphy.setChecked("stratigraphy" in sheet.section_order)
    sheet.nonpyro.setChecked("nonpyro" in sheet.section_order)
    sheet.invalidate()
    if missing:
        sheet.announce_status(tr('Шаблон применён. Укажите колонки группировки вручную: {0}.', ", ".join(missing)))
    else:
        sheet.announce_status(tr('Шаблон «{0}» применён. Проверьте текущие показатели, интерполяции и лимиты перед сборкой.', tr(name)))
