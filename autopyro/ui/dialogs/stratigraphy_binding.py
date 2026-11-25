"""Preview, calculate and commit stratigraphic binding of samples."""

from __future__ import annotations

from ..i18n import tr, message
from copy import deepcopy
from weakref import ref
from PySide6.QtCore import Qt
from PySide6.QtGui import QUndoCommand
from PySide6.QtWidgets import QDialog, QDialogButtonBox
from autopyro.model.core import Project, Row
from autopyro.model.measurements import derived
from autopyro.geology.stratigraphy import (
    column_style,
    LEVELS,
    STATUSES,
    evaluate_project,
    capture_result_columns,
    prepare_result_columns,
    overlap_count,
    result_column,
    scheme_hash,
    validate_binding,
    validate_scheme,
)
from autopyro.geology.stratigraphy_revisions import interval_changes, sample_changes
from ..table import AddColumnCommand
from ..tasks import TaskCancelled, run_with_progress
from ..widgets.documents import refill_combo
from ..theme import icon
from ..widgets.stratigraphy_revisions import membership_label
from ..widgets.stratigraphy_labels import geological_label
from ..widgets.interval_review import IntervalReviewDialog
from ..widgets.results import ResultsModel


class BindingPreviewModel(ResultsModel):
    def __init__(self, parent):
        super().__init__(
            [tr('UUID пробы'), tr('Ствол'), tr('Глубина'), *(geological_label(level) for level in LEVELS)],
            [], parent,
        )
        self.interval_mode = False

    def set_report(self, report):
        self.interval_mode = report['sample_mode'] == 'interval'
        self.set_rows(report['results'])

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            return None
        item = self.rows[index.row()]
        column = index.column()
        if column < 3:
            value = item[('row_id', 'wellbore', 'depth')[column]]
            if column == 2 and self.interval_mode:
                return f"[{value}, {item['bottom']})"
            return '' if value is None else str(value)
        result = item['levels'][LEVELS[column - 3]]
        status = tr(STATUSES[result['status']])
        text = ' · '.join(part for part in (', '.join(result['names']), status) if part)
        if result.get('coverage_fraction') is not None:
            text += tr(' · покрытие {0:.1%}', result['coverage_fraction'])
        return text


class SetStratigraphyCommand(QUndoCommand):
    def __init__(self, model, settings, *, scheme_only=False, style=None, columns=()):
        super().__init__(tr('Стратиграфическая схема и привязка'))
        self.scheme_only = scheme_only
        if scheme_only:
            validate_scheme(settings["scheme"])
        else:
            validate_binding(model.project, settings["scheme"], settings["binding"])
        self.model = model
        self.column_commands = [AddColumnCommand(model, column, {}) for column in columns]
        self.prepared_report = None
        self.cached_before = self.cached_after = None
        self.cache_revision = None
        self.before = {
            key: deepcopy(model.project.view.get(key))
            for key in ("stratigraphy", "stratigraphy_scheme", "stratigraphy_revisions", "stratigraphy_style")
        }
        self.after = deepcopy(self.before)
        scheme = settings["scheme"]
        history = self.after["stratigraphy_revisions"] or {}
        current = self.before["stratigraphy"]
        if current:
            old = current["scheme"]
            history.setdefault(old["id"], {})[old["revision"]] = deepcopy(old)
        old = self.before["stratigraphy_scheme"]
        if old:
            history.setdefault(old["id"], {})[old["revision"]] = deepcopy(old)
        previous = history.get(scheme["id"], {}).get(scheme["revision"])
        if previous and scheme_hash(previous) != scheme_hash(scheme):
            raise ValueError(
                tr('Эта редакция уже использовалась с другим содержимым; измените revision')
            )
        history.setdefault(scheme["id"], {})[scheme["revision"]] = deepcopy(scheme)
        self.after.update(stratigraphy_scheme=deepcopy(scheme), stratigraphy_revisions=history)
        if style is not None:
            self.after["stratigraphy_style"] = column_style(style)
        if not scheme_only:
            self.after["stratigraphy"] = deepcopy(settings)

    def apply(self, values, action, *, report=None, insert_columns=False):
        view = self.model.project.view
        binding_changed = not self.scheme_only and view.get("stratigraphy") != values["stratigraphy"]
        for key, value in values.items():
            if key == "stratigraphy" and not binding_changed:
                continue
            if value is None:
                view.pop(key, None)
            else:
                view[key] = deepcopy(value)
        self.model.project.event(
            action,
            settings={"scheme": values["stratigraphy_scheme"]}
            if self.scheme_only else values["stratigraphy"],
            scheme_hash=scheme_hash(self.after["stratigraphy_scheme"]),
        )
        if insert_columns:
            for command in self.column_commands:
                command.insert_column()
        if binding_changed or self.column_commands:
            self.model.refresh(invalidate_interpolation=False, stratigraphy_report=report)
        else:
            self.model.notify_changed(data_changed=False)

    def redo(self):
        report, self.prepared_report = self.prepared_report, None
        self.check_cache_revision()
        if report is None:
            report = self.cached_after
        self.apply(self.after, "stratigraphy", report=report, insert_columns=True)
        self.cache_revision = self.model.data_revision

    def undo(self):
        self.check_cache_revision()
        for command in reversed(self.column_commands):
            command.remove_column()
        self.apply(self.before, "undo_stratigraphy", report=self.cached_before)
        self.cache_revision = self.model.data_revision

    def check_cache_revision(self):
        # Only reuse results across adjacent undo/redo of this command.
        if self.cache_revision is not None and self.cache_revision != self.model.data_revision:
            self.cached_before = self.cached_after = None

    def cache_results(self, report, *, cache_before):
        if cache_before:
            self.cached_before = capture_result_columns(self.model.project, self.before["stratigraphy"])
        previous = getattr(self.model, '_stratigraphy_cache_owner', None)
        if previous is not None and (command := previous()) is not None:
            command.cached_before = command.cached_after = None
        self.model._stratigraphy_cache_owner = ref(self)
        self.cached_after = {key: report[key] for key in ("row_ids", "binding", "scheme_hash", "_column_results")}
        self.cached_after["scheme"] = {"revision": report["scheme"]["revision"]}


def invalidate_preview(editor, *_):
    editor._sample_track_cache = None
    editor.results.model().set_rows([])
    editor.update_binding_revision()
    editor.status.setText(tr('Параметры привязки изменены; проверьте привязку заново.'))
    if editor.sample_track.isChecked():
        editor.schedule_column_preview()


def update_binding_revision(editor, preview=None):
    view = editor.owner.model.project.view
    active = (view.get('stratigraphy') or {}).get('scheme')
    if active:
        text = tr('Сохранённая привязка проб: {0} · редакция {1}', active['name'], active['revision'])
        saved = view.get('stratigraphy_scheme')
        if saved and (saved['id'], saved['revision']) != (active['id'], active['revision']):
            text += '\n' + tr('Сохранённая схема: {0} · редакция {1}. Для этой редакции привязку нужно сохранить отдельно.',
                              saved['name'], saved['revision'])
    else:
        text = tr('Сохранённой привязки проб нет.')
    if preview:
        text += '\n' + tr('Результаты предпросмотра рассчитаны по черновику: {0} · редакция {1}',
                          preview['name'], preview['revision'])
    editor.binding_revision.setText(text)


def settings(editor):
    editor.refresh_columns()
    scheme = editor.read_scheme()
    binding = editor.current_binding()
    validate_binding(editor.owner.model.project, scheme, binding)
    return {"scheme": scheme, "binding": binding}


def current_binding(editor):
    binding = {
        "well_id": editor.well.currentData(),
        "depth_id": editor.depth.currentData(),
        "depth_unit": editor.sample_unit.currentText(),
        "depth_reference": editor.sample_reference.text(),
        "text_numbers": editor.text_numbers.isChecked(),
    }
    if editor.mode.currentData() == "interval":
        binding.update(sample_mode="interval", bottom_id=editor.bottom.currentData())
    return binding


def calculate_binding_report(editor, row_ids, bindings, title, calculate, *, parent=None):
    project = editor.owner.model.project
    if len(row_ids) < editor.BACKGROUND_BINDING_ROWS:
        return calculate(project, None)

    revision = editor.owner.analysis_revision
    column_ids = {binding[key] for binding in bindings for key in ('well_id', 'depth_id', 'bottom_id')
                  if binding.get(key)}
    selected = set(row_ids)

    def check_project():
        if editor.owner.model.project is not project or editor.owner.analysis_revision != revision:
            raise TaskCancelled()

    def prepare():
        snapshot = Project(columns=[deepcopy(column) for column in project.columns if column.id in column_ids])
        for index, row in enumerate(project.rows):
            if index % 512 == 0:
                check_project()
                yield
            if row.id in selected:
                snapshot.rows.append(Row({key: deepcopy(row.values.get(key)) for key in column_ids}, id=row.id))
        return snapshot

    def work(update, snapshot):
        def checkpoint():
            update(title, 0)

        checkpoint()
        return calculate(snapshot, checkpoint)

    result = run_with_progress(parent if parent is not None else editor, title, work,
                               cancellable=True, prepare=prepare())
    check_project()
    return result


def preview(editor):
    editor.results.model().set_rows([])
    editor.update_binding_revision()
    try:
        settings = editor.settings()
        scope = editor.owner.analysis_scope
        scope_revision = scope.revision
        row_ids = scope.rows()

        def calculate(project, checkpoint):
            report = evaluate_project(project, row_ids, **settings, checkpoint=checkpoint)
            if checkpoint:
                checkpoint()
            return report, overlap_count(settings['scheme'])

        report, overlaps = editor.calculate_binding_report(
            row_ids, (settings['binding'],), tr('Проверить привязку'), calculate)
        if scope.revision != scope_revision:
            raise TaskCancelled()
        editor.results.model().set_report(report)
        editor.status.setText(
            tr('Проб: {0}; пересечений интервалов одного уровня: {1}.',
               len(report['results']), overlaps)
        )
        editor.result_tabs.setCurrentWidget(editor.results)
        editor.update_binding_revision(settings['scheme'])
    except TaskCancelled:
        editor.status.setText(tr('Подготовка отменена.'))
    except Exception as exc:
        editor.status.setText(tr('Ошибка: {0}', message(exc)))


def commit(editor, add_columns=False):
    settings = editor.settings()
    style = editor.style_values()
    model = editor.owner.model
    saved = model.project.view.get('stratigraphy')
    new_columns = editor.new_binding_columns(model.project.columns) if add_columns else []
    command = SetStratigraphyCommand(model, settings, style=style, columns=new_columns)
    report = (editor.prepare_binding_save(settings, [*model.project.columns, *new_columns])
              if saved != settings or new_columns else None)
    if saved and saved != settings:
        revision = editor.owner.analysis_revision
        if not editor.review_binding_update(saved, settings, report=report):
            return None
        if model is not editor.owner.model or revision != editor.owner.analysis_revision:
            raise TaskCancelled()
    command.prepared_report = report
    if report is not None:
        command.cache_results(report, cache_before=bool(saved) and editor.has_raw_binding_inputs(saved['binding']))
    model.undo.push(command)
    editor.mark_draft_saved(style)
    editor.status.setText(
        tr('Схема и привязка сохранены. Колонки всех проб пересчитываются при редактировании; доступна отмена.')
    )
    return settings


def new_binding_columns(editor, existing):
    names = {column.name for column in existing}
    columns = []
    for status in (False, True):
        column = result_column(editor.level.currentData(), status=status)
        if any(item.metadata.get('stratigraphy') == column.metadata['stratigraphy'] for item in existing):
            continue
        original, count = column.name, 2
        while column.name in names:
            column.name = f"{original} ({count})"
            count += 1
        names.add(column.name)
        columns.append(column)
    return columns


def prepare_binding_save(editor, settings, columns):
    project = editor.owner.model.project
    if len(project.rows) < editor.BACKGROUND_BINDING_ROWS:
        return None
    definitions = {column.id: deepcopy(column.metadata['stratigraphy'])
                   for column in columns if column.metadata.get('stratigraphy')}
    if not definitions:
        return None
    binding = settings['binding']
    # Refresh can recalculate derived inputs before writing the assignments.
    if not editor.has_raw_binding_inputs(binding):
        return None
    row_ids = [row.id for row in project.rows]

    def calculate(snapshot, checkpoint):
        report = evaluate_project(snapshot, row_ids, **settings, checkpoint=checkpoint)
        report['_column_results'] = prepare_result_columns(report, definitions, checkpoint=checkpoint)
        return report

    return editor.calculate_binding_report(row_ids, (binding,), tr('Пересчёт привязки проб'), calculate)


def has_raw_binding_inputs(editor, binding):
    inputs = {binding.get(key) for key in ('well_id', 'depth_id', 'bottom_id')}
    return not any(column.id in inputs and derived(column) for column in editor.owner.model.project.columns)


def review_binding_update(editor, saved, settings, *, report=None):
    project = editor.owner.model.project
    before, after = saved['scheme'], settings['scheme']
    old_binding, new_binding = saved['binding'], settings['binding']
    rows = []
    try:
        validate_binding(project, before, old_binding)
    except ValueError as exc:
        summary = tr('Не удалось проверить сохранённую привязку: {0}', message(exc))
    else:
        row_ids = [row.id for row in project.rows]

        def compare(snapshot, checkpoint):
            identities = interval_changes(before, after)[1] if before['id'] == after['id'] else {}
            changes = sample_changes(snapshot, row_ids, before, after, new_binding, identities,
                                     before_binding=old_binding, checkpoint=checkpoint, after_report=report)
            changed_ids, formatted = set(), []
            for number, (item, level, left, right) in enumerate(changes):
                if checkpoint and number % 256 == 0:
                    checkpoint()
                changed_ids.add(item['row_id'])
                formatted.append((item['row_id'], geological_label(level),
                                  membership_label(left, old_binding['depth_unit']),
                                  membership_label(right, new_binding['depth_unit'])))
            return len(changed_ids), formatted

        changed_count, rows = editor.calculate_binding_report(
            row_ids, (old_binding, new_binding), tr('Пересчёт привязки проб'), compare)
        summary = tr('Изменится принадлежность проб: {0}. Изменений по уровням: {1}.',
                     changed_count, len(rows))
    dialog = IntervalReviewDialog(
        editor, tr('Пересчёт привязки проб'),
        tr('Сохранённая редакция: {0}. Новая редакция: {1}. Область проверки: все пробы проекта.',
           before['revision'], after['revision']) + '\n' + summary,
        [tr('UUID пробы'), tr('Уровень'), tr('До'), tr('После')], rows)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
    save = buttons.button(QDialogButtonBox.StandardButton.Save)
    save.setText(tr('Пересчитать и сохранить'))
    save.setIcon(icon('save'))
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    dialog.layout_box.addWidget(buttons)
    accepted = dialog.exec() == QDialog.DialogCode.Accepted
    dialog.deleteLater()
    return accepted


def apply(editor):
    try:
        editor.commit()
    except TaskCancelled:
        editor.status.setText(tr('Подготовка отменена.'))
    except Exception as exc:
        editor.status.setText(tr('Ошибка: {0}', message(exc)))


def add_columns(editor):
    try:
        editor.commit(add_columns=True)
    except TaskCancelled:
        editor.status.setText(tr('Подготовка отменена.'))
    except Exception as exc:
        editor.status.setText(tr('Ошибка: {0}', message(exc)))


def show_plot(editor):
    try:
        settings = editor.settings()
        if settings != editor.owner.model.project.view.get("stratigraphy"):
            raise ValueError(tr('Сначала сохраните схему и привязку'))
        ids = set(editor.owner.analysis_scope.rows())
        wells = {
            r.values[settings["binding"]["well_id"]]
            for r in editor.owner.model.project.rows
            if r.id in ids
        }
        if len(wells) != 1:
            raise ValueError(tr('Отфильтруйте таблицу до одного ствола'))
        candidates = [
            c
            for c in editor.owner.model.project.columns
            if c.kind == "number"
            and c.id
            not in (
                settings["binding"]["depth_id"],
                settings["binding"].get("bottom_id"),
            )
        ]
        if not candidates:
            raise ValueError(tr('Нужна числовая колонка показателя для оси X'))
        editor.owner.show_plots()
        dialog = editor.owner.plot_window
        dialog.read_controls()
        dialog.spec.kind = "depth"
        dialog.spec.x_id = candidates[0].id
        dialog.spec.y_id = settings["binding"]["depth_id"]
        dialog.spec.depth_bottom_id = settings["binding"].get("bottom_id", "")
        if dialog.spec.depth_bottom_id:
            dialog.spec.shape_by = ""
        dialog.spec.scope = "global"
        dialog.spec.connect = False
        dialog.spec.title = tr('Глубинный профиль')
        dialog.spec.text_numbers = settings["binding"]["text_numbers"]
        dialog.spec.interpretation = None
        dialog.spec.interpretation_panels = []
        dialog.spec.comparison_preview_panel = None
        dialog.spec.curve_archive_id = ""
        dialog.spec.annotations = []
        dialog.spec.stratigraphy = {
            **deepcopy(settings),
            "level": editor.level.currentData(),
        }
        from autopyro.visualization.plotting import AxisSpec

        dialog.spec.x_axis = AxisSpec()
        dialog.spec.y_axis = AxisSpec(
            inverted=True,
            label=tr('Глубина отбора [{0}]', settings['binding']['depth_unit']),
        )
        dialog.load_controls()
        dialog.redraw()
        if not editor.property("workspaceTracked"):
            editor.accept()
    except Exception as exc:
        editor.status.setText(tr('Ошибка: {0}', message(exc)))


def refresh_columns(editor):
    columns = [column for column in editor.owner.model.project.columns if not column.metadata.get("stratigraphy")]
    for combo, prompt, depth in ((editor.well, tr('Выберите колонку ствола'), False),
                                 (editor.depth, tr('Выберите глубину / верх отбора'), True),
                                 (editor.bottom, tr('Выберите низ отбора'), True)):
        choices = [(prompt, None)] + [(f"{column.name} [{column.unit}]" if depth else column.name, column.id)
                                     for column in columns]
        if refill_combo(combo, choices):
            editor.invalidate_preview()
