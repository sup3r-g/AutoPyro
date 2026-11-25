"""Manage saved plots, axes, archived curves and portable templates."""

from __future__ import annotations

from ..i18n import tr, message
from copy import deepcopy
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox
from ..plot_templates import bind_plot_template, read_plot_template, write_plot_template
from autopyro.visualization.plot_passports import (
    INTERVAL_PASSPORTS, PASSPORTS, axis_requirements, matches_axis, template_kind,
)
from autopyro.visualization.plotting import AxisSpec, PlotSpec


def use_template(dialog, title):
    dialog.template.setCurrentIndex(dialog.template.findData(title))
    dialog.apply_template()


def apply_template(dialog):
    title = dialog.template.currentData()
    if title not in PASSPORTS:
        return

    ids = []
    for role, unit in axis_requirements(title):
        matches = [column.id for column in dialog.owner.model.project.columns
                   if matches_axis(column, role, unit)]
        if len(matches) != 1:
            QMessageBox.information(
                dialog, tr('Выбор показателей'),
                tr('Для {0} нужна одна колонка {1} [{2}]. Проверьте единицы или выберите показатели свободного графика.', title, role, unit),
            )
            return
        ids.append(matches[0])
    bottom = ids.pop() if title in INTERVAL_PASSPORTS else ""
    *tracks, y = ids
    x = tracks[0]
    dialog.spec.passport_id = PASSPORTS[title][0]
    kind = template_kind(title)
    dialog.data.load({"kind": kind, "x_id": x, "y_id": y, "depth_bottom_id": bottom})
    with QSignalBlocker(dialog.plot_mode):
        dialog.plot_mode.setCurrentIndex(0)
    if bottom:
        dialog.style.load({'shape_by': ''})
        dialog.data.load({'connect': False})
    dialog.kind_changed()
    dialog.previous_x_target = ""
    dialog.x_target.setCurrentIndex(0)
    dialog.x_form.load(vars(AxisSpec()))
    dialog.y_form.load(vars(AxisSpec(inverted=kind in ('depth', 'multi_depth'))))
    if kind == 'multi_depth':
        dialog.spec.track_ids = tracks
        dialog.spec.track_axes.clear()
        dialog.load_track_order()
        dialog.update_x_targets()
    label = tr('{0} · интервальный профиль MD', PASSPORTS[title][1]) if bottom else title
    dialog.layout_form.load({"title": label})
    dialog.schedule()


def remove_passport(dialog):
    dialog.spec.passport_id = ''
    dialog.schedule()


def capture_limits(dialog):
    if not dialog.result:
        return
    dialog.read_controls()
    dialog.copy_visible_limits(dialog.spec)
    dialog.load_controls()
    dialog.schedule()


def copy_visible_limits(dialog, spec):
    for i, ax in enumerate(dialog.result["axes"]):
        limits = sorted(ax.get_xlim())
        axis = (
            spec.track_axes.setdefault(
                spec.track_ids[i], deepcopy(spec.x_axis)
            )
            if spec.kind == "multi_depth"
            else spec.x_axis
        )
        axis.minimum, axis.maximum = limits
    limits = sorted(dialog.result["axes"][0].get_ylim())
    spec.y_axis.minimum, spec.y_axis.maximum = limits
    ax = dialog.result["axes"][0]
    if ax.name == '3d':
        spec.z_axis.minimum, spec.z_axis.maximum = sorted(ax.get_zlim())
        spec.camera_3d = (ax.elev, ax.azim, ax.roll)


def reset_limits(dialog):
    for form in (dialog.x_form, dialog.y_form, dialog.z_form):
        form.load({'minimum': None, 'maximum': None})
    dialog.spec.camera_3d = None
    for axis in (
        dialog.spec.x_axis,
        dialog.spec.y_axis,
        dialog.spec.z_axis,
        *dialog.spec.track_axes.values(),
    ):
        axis.minimum = axis.maximum = None
    dialog.schedule()


def freeze_scope(dialog):
    if not dialog.result or not dialog.rendered_ids:
        QMessageBox.information(dialog, tr('Наборы проб'), tr('Сначала постройте график по непустой области.'))
        return
    row_ids = list(dialog.rendered_ids)
    project = dialog.owner.model.project
    name, accepted = QInputDialog.getText(dialog, tr('Наборы проб'),
        tr('Название набора (сохраняется состав проб, значения остаются в проекте)'))
    if not accepted or dialog.closed or project is not dialog.owner.model.project:
        return
    name = name.strip()
    sets = project.view.setdefault('sample_sets', {})
    if not name or name in sets:
        QMessageBox.warning(dialog, tr('Наборы проб'), tr('Введите уникальное название набора.'))
        return
    sets[name] = row_ids
    dialog.owner.analysis_scope.refresh()
    dialog.scope.setCurrentIndex(dialog.scope.findData(f'set:{name}'))
    dialog.owner.view_changed()


def save_named(dialog):
    if dialog.closed:
        return
    if not dialog.result:
        QMessageBox.information(dialog, tr('Сохранить график'), tr('Дождитесь построения графика.'))
        return
    project = dialog.owner.model.project
    config = dialog.spec.to_dict()
    if config['scope'] == 'global':
        config['scope'] = dialog.owner.analysis_scope.mode
    name, ok = QInputDialog.getText(dialog, tr('Сохранить график'), tr('Название'))
    name = name.strip()
    if not ok or not name or dialog.closed or project is not dialog.owner.model.project:
        return
    plots = project.view.setdefault("plots", {})
    if name in plots and plots[name] != config and QMessageBox.question(
        dialog, tr('Сохранить график'), tr('Заменить график «{0}»?', name),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
    ) != QMessageBox.StandardButton.Yes:
        return
    if dialog.closed or project is not dialog.owner.model.project:
        return
    plots[name] = config
    dialog.owner.view_changed()


def load_named(dialog):
    if dialog.closed:
        return
    project = dialog.owner.model.project
    plots = project.view.get("plots", {})
    if not plots:
        QMessageBox.information(dialog, tr('Графики проекта'), tr('В проекте пока нет сохранённых графиков.'))
        return
    name, ok = QInputDialog.getItem(
        dialog, tr('Графики проекта'), tr('Выберите'), list(plots), editable=False
    )
    if ok and not dialog.closed and project is dialog.owner.model.project:
        try:
            dialog.spec = PlotSpec.from_dict(plots[name])
            dialog.load_controls()
            dialog.schedule()
        except (ValueError, TypeError, KeyError) as exc:
            QMessageBox.warning(dialog, tr('График'), message(exc))


def add_archived_curve(dialog):
    from autopyro.visualization.curve_layer import curve_axes

    project = dialog.owner.model.project
    records = project.view.get('curve_fit_results', [])
    if not records:
        QMessageBox.information(dialog, tr('Сохранённая кривая'),
                                tr('Сначала сохраните результат в модуле подгонки кривых.'))
        return
    labels = [f"{record['name']} · {record['id']}" for record in records]
    label, accepted = QInputDialog.getItem(dialog, tr('Сохранённая кривая'), tr('Выберите'), labels, editable=False)
    if not accepted or dialog.closed or project is not dialog.owner.model.project:
        return
    record = records[labels.index(label)]
    try:
        x, y = curve_axes(record, {column.id: column for column in project.columns})
        dialog.read_controls()
        if dialog.spec.interpretation or dialog.spec.interpretation_panels or dialog.spec.stratigraphy:
            raise ValueError(tr('Перед добавлением кривой уберите интерпретационный и стратиграфический слои.'))
    except ValueError as exc:
        QMessageBox.warning(dialog, tr('Сохранённая кривая'), message(exc))
        return
    if dialog.spec.x_id != x:
        dialog.spec.x_axis = AxisSpec()
    if dialog.spec.y_id != y or dialog.spec.kind in ('depth', 'multi_depth'):
        dialog.spec.y_axis = AxisSpec()
    dialog.spec.kind = 'scatter'
    dialog.spec.x_id, dialog.spec.y_id = x, y
    dialog.spec.depth_bottom_id = ''
    dialog.spec.curve_archive_id = record['id']
    dialog.load_controls()
    dialog.schedule()


def remove_archived_curve(dialog):
    dialog.spec.curve_archive_id = ''
    dialog.schedule()


def remove_interpretation(dialog):
    dialog.spec.interpretation_panels = []
    dialog.spec.interpretation = None
    dialog.schedule()


def export_spec(dialog):
    if not dialog.result:
        return
    path, _ = QFileDialog.getSaveFileName(
        dialog, tr('Оформление графика'), "plot.json", "JSON (*.json)"
    )
    if path:
        try:
            write_plot_template(path, dialog.spec, dialog.owner.model.project.columns)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(dialog, "JSON", message(exc))


def import_spec(dialog):
    path, _ = QFileDialog.getOpenFileName(
        dialog, tr('Загрузить оформление'), "", "JSON (*.json)"
    )
    if path:
        try:
            payload = read_plot_template(path)
            if "matrix" in payload:
                dialog.owner.show_statistics()
                statistics = dialog.owner.statistics_window
                statistics.tabs.setCurrentWidget(statistics.tables["correlations"])
                statistics.open_matrix(payload)
                return
            dialog.spec = bind_plot_template(payload, dialog.owner.model.project.columns)
            dialog.load_controls()
            dialog.schedule()
        except (OSError, ValueError, TypeError, KeyError) as exc:
            QMessageBox.warning(dialog, tr('Шаблон'), message(exc))
