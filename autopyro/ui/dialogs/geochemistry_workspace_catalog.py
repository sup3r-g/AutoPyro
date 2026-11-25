"""Refresh the geochemistry catalog, display passports and edit sample links."""

from copy import deepcopy
import html
import json
from time import sleep
from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtGui import QUndoCommand
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QInputDialog,
    QListWidgetItem,
    QMessageBox,
    QTableWidgetItem,
    QVBoxLayout,
)
from autopyro.analysis.geochemistry_workspace import (
    CATALOG_COLUMNS,
    STATE_KEY,
    GeochemistryIndex,
    comparison_is_current,
    link_samples,
    relate_materials,
    save_group,
    select_measurement,
    settings,
    unlink_sample,
)
from autopyro.model.measurements import derived
from autopyro.analysis.nonpyro import METHODS, field_matches, source_identity, variant_tag
from ..geochemistry import modules
from ..i18n import tr, message
from ..tasks import run_with_progress
from .geochemistry_workspace_display import _number, _status


class WorkspaceCommand(QUndoCommand):
    def __init__(self, model, state, title):
        super().__init__(title)
        self.model = model
        self.before, self.after = settings(model.project), deepcopy(state)

    def apply(self, state):
        project = self.model.project
        project.view[STATE_KEY] = deepcopy(state)
        project.event('geochemistry_workspace', operation=self.text())
        self.model.notify_changed(data_changed=False)

    def undo(self):
        self.apply(self.before)

    def redo(self):
        self.apply(self.after)


class MaterialLinkDialog(QDialog):
    def __init__(self, samples, parent):
        super().__init__(parent)
        self.setWindowTitle(tr('Связь исходного и производного материала'))
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)
        self.parent_sample, self.child_sample, self.kind = QComboBox(), QComboBox(), QComboBox()
        for sample in samples:
            label = sample['name'] + ' [' + sample['id'][-8:] + ']'
            self.parent_sample.addItem(label, sample['id'])
            self.child_sample.addItem(label, sample['id'])
        self.child_sample.setCurrentIndex(min(1, len(samples) - 1))
        self.kind.addItem(tr('Экстракт из исходной пробы'), 'extract')
        self.kind.addItem(tr('Остаток после экстракции'), 'residue')
        form.addRow(tr('Исходная проба'), self.parent_sample)
        form.addRow(tr('Производный материал'), self.child_sample)
        form.addRow(tr('Тип связи'), self.kind)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def selected_samples(workspace):
    return [item.data(Qt.ItemDataRole.UserRole) for item in
            sorted(workspace.samples.selectedItems(), key=workspace.samples.row)]


def changed(workspace, *_):
    model = workspace.owner.model
    previous = workspace._index_context
    state = settings(model.project) if workspace.report is not None and previous is not None else {}
    groups_only = (workspace.report is not None and previous is not None and previous[:2] == (model, model.data_revision)
                   and state.get('groups') != previous[2].get('groups')
                   and {key: value for key, value in state.items() if key != 'groups'}
                   == {key: value for key, value in previous[2].items() if key != 'groups'})
    if not groups_only or not comparison_is_current(model.project, workspace.report):
        workspace.invalidate_comparison()
    workspace.timer.start(0)


def refresh(workspace):
    if workspace._loading:
        return
    workspace._loading = True
    workspace.refresh_button.setEnabled(False)
    try:
        selected = set(workspace.selected_samples())
        project = workspace.owner.model.project
        scope = workspace.owner.analysis_scope.rows()
        model = workspace.owner.model
        context = model.data_revision, workspace.owner.analysis_scope.revision
        index_context = (model, model.data_revision, settings(project))
        reusable_index = workspace.index if workspace._index_context == index_context else None
        row_records = None
        if (workspace.index is not None and workspace._index_context is not None
                and workspace._index_context[0] is model and workspace.index.project is project
                and workspace._index_context[1] + 1 == model.data_revision == model.cell_edit_revision
                and workspace._index_measurements is project.science.measurements
                and workspace._index_measurement_count == len(project.science.measurements)):
            row_records = workspace.index.row_records
            catalog_columns = {workspace.index.by_name[name] for name in CATALOG_COLUMNS if name in workspace.index.by_name}
            if (not model.cell_edit_columns.intersection(catalog_columns)
                    and workspace._index_context[2] == index_context[2]):
                reusable_index = workspace.index
        catalog_scope = tuple(scope)

        def check_context():
            if (workspace.owner.model is not model or model.project is not project
                    or context != (model.data_revision, workspace.owner.analysis_scope.revision)):
                raise ValueError(tr('Данные изменились во время расчёта. Повторите сравнение.'))

        def prepare(checkpoint=None):
            index = reusable_index or GeochemistryIndex(project, checkpoint=checkpoint, row_records=row_records)
            if reusable_index is not None and catalog_scope == workspace._catalog_scope:
                return index, workspace._catalog_rows
            catalog = []
            for position, sample in enumerate(index.iter_catalog(scope)):
                if checkpoint is not None and position % 1024 == 0:
                    checkpoint()
                label = sample['name'] + ' [' + sample['id'][-8:] + ']'
                catalog.append((sample['id'], label, json.dumps(sample, ensure_ascii=False)))
            return index, catalog

        if ((reusable_index is None or catalog_scope != workspace._catalog_scope)
                and (len(project.rows) >= 10000 or len(project.science.measurements) >= 100000)):
            workspace.index, catalog = run_with_progress(
                workspace.owner, tr('Подготовка измерений…'), lambda: prepare(lambda: sleep(0.001)),
            )
            check_context()
        else:
            workspace.index, catalog = prepare()
        schema = (tuple((column.id, column.name, column.kind, column.unit, derived(column),
                         source_identity(column), bool(column.metadata.get('geochem_field')))
                        for column in workspace.index.columns.values()),
                  tuple(workspace.index.state.get('groups', {})))
        def populate():
            workspace.available_ids = {sample_id for sample_id, _, _ in catalog}
            blocker = QSignalBlocker(workspace.samples)
            workspace.samples.clear()
            labels = [label for _, label, _ in catalog]
            workspace.samples.addItems(labels)
            for i, (sample_id, _, tooltip) in enumerate(catalog):
                item = workspace.samples.item(i)
                item.setData(Qt.ItemDataRole.UserRole, sample_id)
                item.setToolTip(tooltip)
                if sample_id in selected:
                    item.setSelected(True)
                if i % 512 == 511:
                    yield
                    check_context()
            if not workspace.samples.selectedItems() and workspace.samples.count():
                workspace.samples.item(0).setSelected(True)
            del blocker
            workspace.count.setText(tr('Проб: {0}. Исходных строк: {1}.', len(catalog), len(workspace.owner.model.project.rows)))
            old_choices = [combo.currentData() for combo in (workspace.a, workspace.b)]
            blockers = [QSignalBlocker(combo) for combo in (workspace.a, workspace.b)]
            groups = workspace.index.state.get('groups', {})
            identifiers = ['sample:' + sample_id for sample_id, _, _ in catalog]
            identifiers.extend('group:' + name for name in groups)
            positions = {identifier: i for i, identifier in enumerate(identifiers)}
            workspace.comparison_choices.set_choices(
                labels + [tr('Группа: {0}', name) for name in groups], identifiers,
            )
            for combo, old in zip((workspace.a, workspace.b), old_choices):
                combo.setCurrentIndex(positions.get(old, 0) if identifiers else -1)
                combo.setToolTip(combo.currentText())
            del blockers
            if workspace.a.currentData() == workspace.b.currentData() and workspace.b.count() > 1:
                workspace.b.setCurrentIndex(1)
            checked = {workspace.fields.item(i).data(Qt.ItemDataRole.UserRole) for i in range(workspace.fields.count())
                       if workspace.fields.item(i).checkState() == Qt.CheckState.Checked}
            blocker = QSignalBlocker(workspace.fields)
            workspace.fields.clear()
            columns = [column for column in workspace.index.columns.values() if column.kind == 'number' and not derived(column)]
            columns.sort(key=lambda column: not bool(column.metadata.get('geochem_field')))
            for i, column in enumerate(columns):
                item = QListWidgetItem(column.name + f' [{column.unit}]', workspace.fields)
                item.setData(Qt.ItemDataRole.UserRole, column.id)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if column.id in checked else Qt.CheckState.Unchecked)
                if i % 256 == 255:
                    yield
                    check_context()
            del blocker

        catalog_changed = catalog != workspace._catalog_rows or schema != workspace._catalog_schema
        if catalog_changed:
            if len(catalog) >= 10000:
                run_with_progress(workspace.owner, tr('Подготовка измерений…'), lambda _: None, prepare=populate())
            else:
                for _ in populate():
                    pass
        check_context()
        workspace._index_context = index_context
        workspace._index_measurements = project.science.measurements
        workspace._index_measurement_count = len(project.science.measurements)
        workspace._catalog_rows, workspace._catalog_schema = catalog, schema
        workspace._catalog_scope = catalog_scope
        if catalog_changed:
            workspace.filter_comparison_fields(workspace.field_search.text())
            workspace.filter_samples(workspace.search.text())
        workspace.show_passport()
        workspace.repair_links_button.setVisible(bool(workspace.index.invalid_links))
    except (ValueError, KeyError) as exc:
        workspace.status.setText(message(exc))
        workspace.index = None
        workspace._index_context = None
        workspace._index_measurements = None
        workspace._index_measurement_count = 0
        workspace._catalog_rows = workspace._catalog_schema = None
        workspace._catalog_scope = None
        workspace.available_ids = set()
        workspace.samples.clear()
        workspace.fields.clear()
        workspace.comparison_choices.set_choices([], [])
        workspace.measurements.setRowCount(0)
        workspace.methods.setRowCount(0)
        workspace.export_passport_button.setEnabled(False)
        workspace.details.clear()
        workspace.repair_links_button.hide()
    finally:
        workspace._loading = False
        workspace.refresh_button.setEnabled(True)
    workspace.export_comparison_button.setEnabled(bool(workspace.report) and
        comparison_is_current(workspace.owner.model.project, workspace.report))
    if workspace.passport_tabs.currentWidget() is workspace.review_archive_page:
        workspace.display_archived_review()


def filter_samples(workspace, text):
    blocker = QSignalBlocker(workspace.samples)
    query = text.strip().casefold()
    exact_wells = {
        sample_id for sample_id, sample in workspace.index.samples.items()
        if sample_id in workspace.available_ids and query and any(well.casefold() == query for well in sample['well'])
    } if workspace.index else set()
    for i in range(workspace.samples.count()):
        item = workspace.samples.item(i)
        hidden = (item.data(Qt.ItemDataRole.UserRole) not in exact_wells if exact_wells
                  else bool(query) and query not in (item.text() + item.toolTip()).casefold())
        if item.isHidden() != hidden:
            item.setHidden(hidden)
    del blocker
    if not workspace._loading:
        workspace.update_sample_count()


def filter_comparison_fields(workspace, text):
    for row in range(workspace.fields.count()):
        item = workspace.fields.item(row)
        item.setHidden(text.casefold() not in item.text().casefold())


def select_field_preset(workspace):
    method = workspace.field_preset.currentData()
    basis = workspace.field_basis.currentData()
    roles = METHODS[method]['roles']
    candidates = {}
    for row in range(workspace.fields.count()):
        item = workspace.fields.item(row)
        column = workspace.index.columns[item.data(Qt.ItemDataRole.UserRole)]
        group, label = source_identity(column)
        if variant_tag(label) != basis:
            continue
        for key, role in roles.items():
            if field_matches(group, label, role):
                candidates.setdefault(key, []).append(item)
    with QSignalBlocker(workspace.fields):
        for items in candidates.values():
            if len(items) == 1:
                items[0].setCheckState(Qt.CheckState.Checked)
    workspace.invalidate_comparison()
    if method == 'homohopane_distribution':
        missing = [f'C{carbon}' for carbon in range(31, 36)
                   if not (len(candidates.get(f'h{carbon}_total', ())) == 1 or
                           all(len(candidates.get(f'h{carbon}_{epimer}', ())) == 1 for epimer in ('s', 'r')))]
    else:
        missing = [tr(role['label']) for key, role in roles.items() if len(candidates.get(key, ())) != 1]
    workspace.status.setText(
        tr('Недостающие или неоднозначные колонки: {0}', ', '.join(missing)) if missing else
        tr('Показатели добавлены к выбору. Нажмите «Сравнить».')
    )


def reset_links(workspace):
    state = settings(workspace.owner.model.project)
    for key in ('links', 'relationships', 'selections'):
        state.pop(key, None)
    workspace.commit(state, tr('Сбросить ручные связи и выбор измерений'))


def update_sample_count(workspace):
    selected = workspace.samples.selectedItems()
    hidden = sum(item.isHidden() for item in selected)
    workspace.count.setText(tr('Проб: {0}. Выбрано: {1}, из них скрыто поиском: {2}.',
                          workspace.samples.count(), len(selected), hidden))


def show_passport(workspace):
    ids = workspace.selected_samples()
    workspace.update_sample_count()
    workspace.measurements.setRowCount(0)
    workspace.methods.setRowCount(0)
    workspace.export_passport_button.setEnabled(bool(ids))
    if not ids or workspace.index is None:
        workspace.details.clear()
        workspace.update_measurement_actions()
        return
    workspace.passport = workspace.index.passport(ids[0])
    sample = workspace.passport['sample']
    esc = lambda value: html.escape(str(value))
    metadata = [(tr('Название'), sample['name']), (tr('ID паспорта'), sample['id']),
                (tr('ID пробы источника'), ', '.join(sample['source_sample_ids'])),
                (tr('Скважина'), ', '.join(sample['well'])),
                (tr('Интервалы MD, м'), '; '.join(
                    f"{_number(interval['top_md'])} – {_number(interval['base_md'])} ({interval['datum'] or '—'})"
                    for interval in sample['intervals'])),
                (tr('Состояние материала'), ', '.join(value for value in sample['material'].values() if value)),
                (tr('Исходные записи'), ', '.join(str(item) for item in sample['source_record_ids'] if item is not None)),
                (tr('Исходные строки проекта'), ', '.join(sample['row_ids']))]
    text = '<h2>' + esc(sample['name']) + '</h2><dl>' + ''.join(
        '<dt><b>' + esc(label) + '</b></dt><dd>' + esc(value or '—') + '</dd>' for label, value in metadata) + '</dl>'
    text += '<h3>' + tr('Связанные материалы') + '</h3>'
    for relation in workspace.passport['relationships']:
        parent = workspace.index.samples.get(relation['parent'], {'name': relation['parent']})['name']
        child = workspace.index.samples.get(relation['child'], {'name': relation['child']})['name']
        kind = tr('Экстракт') if relation['kind'] == 'extract' else tr('Остаток')
        text += '<p>' + esc(parent) + ' → ' + esc(child) + ': ' + kind + '</p>'
        if relation.get('missing_sample_ids'):
            text += '<p><b>' + esc(tr('Связанная проба отсутствует в текущем каталоге. Проверьте изменения реквизитов или удаление исходных записей.')) + '</b></p>'
    text += '<h3>' + tr('Сохранённые результаты методик') + '</h3>'
    for result in workspace.passport['results']:
        text += '<p><b>' + esc(tr(METHODS.get(result['method_id'], {'name': result['method_id']})['name'])) + '</b>: ' + esc(message(result.get('interpretation', '')))
        text += ' · ' + (tr('Требуется пересчёт') if result['stale'] else tr('Сохранённый протокол')) + '</p>'
    workspace.details.setHtml(text)
    records = workspace.passport['measurements']
    workspace.measurements.setRowCount(len(records))
    for row, record in enumerate(records):
        values = [record['name'], record['raw'], record['unit'], record.get('analysis_run_id') or record.get('method', ''),
                  record.get('repeat_number'), tr('Да') if record['working'] else tr('Нет'),
                  tr('Да') if record['chosen'] else tr('Нет'), _status(record['comparison_status']),
                  record.get('source_record_id') or record['row_id']]
        for col, value in enumerate(values):
            cell = QTableWidgetItem(_number(value))
            cell.setData(Qt.ItemDataRole.UserRole, record)
            comment = record.get('statistics_exclusion', {}).get('comment')
            if col == 7 and comment:
                cell.setToolTip(f"<qt>{html.escape(_status(record['comparison_status']))}<br>{html.escape(comment)}</qt>")
            else:
                cell.setToolTip(json.dumps(record, ensure_ascii=False, default=str, indent=2))
            workspace.measurements.setItem(row, col, cell)
    workspace.methods.setRowCount(len(workspace.passport['availability']))
    for row, item in enumerate(workspace.passport['availability']):
        missing = [tr(role['label']) + ': ' + _status(role['status'])
                   for role in item['roles'].values() if role['status'] != 'numeric']
        values = [tr(item['name']), tr('Данные есть; подтвердите условия в модуле') if not missing else tr('Нужны данные / выбор измерений'),
                  '; '.join(missing)]
        for col, value in enumerate(values):
            cell = QTableWidgetItem(value)
            cell.setData(Qt.ItemDataRole.UserRole, item['method_id'])
            workspace.methods.setItem(row, col, cell)
    for table in (workspace.measurements, workspace.methods):
        table.resizeColumnsToContents()
        for col in range(table.columnCount()):
            table.setColumnWidth(col, min(330, max(90, table.columnWidth(col))))
    workspace.update_measurement_actions()


def update_measurement_actions(workspace):
    workspace.use_measurement.setEnabled(bool(workspace.measurements.selectedIndexes()))
    if hasattr(workspace, 'reset_button'):
        workspace.reset_button.setEnabled(bool(workspace.measurements.selectedIndexes()))


def commit(workspace, state, title):
    workspace.owner.model.undo.push(WorkspaceCommand(workspace.owner.model, state, title))
    workspace.timer.stop()
    workspace.refresh()


def link_selected(workspace):
    ids = workspace.selected_samples()
    name, ok = QInputDialog.getText(workspace, tr('Подтвердите объединение одной пробы'),
        tr('Выбрано проб: {0}. Все их исходные строки будут связаны как одна проба. Укажите название:', len(ids)))
    if ok:
        workspace.attempt(lambda: workspace.commit(link_samples(workspace.owner.model.project, ids, name), tr('Объединить анализы пробы')))


def unlink_selected(workspace):
    ids = workspace.selected_samples()
    if len(ids) == 1:
        workspace.attempt(lambda: workspace.commit(unlink_sample(workspace.owner.model.project, ids[0]), tr('Разъединить анализы пробы')))


def repair_invalid_link(workspace):
    if workspace.index is None or not workspace.index.invalid_links:
        return
    links = {f"{link.get('name') or '—'} [{link['id']}]": link['id'] for link in workspace.index.invalid_links}
    label, ok = QInputDialog.getItem(workspace, tr('Исправить объединение проб…'),
                                    tr('Выберите объединение для разъединения'), list(links), 0, False)
    if ok:
        state = settings(workspace.owner.model.project)
        state['links'] = [link for link in state.get('links', []) if link['id'] != links[label]]
        workspace.commit(state, tr('Разъединить анализы пробы'))


def link_materials(workspace):
    ids = workspace.selected_samples()
    if len(ids) < 2:
        workspace.status.setText(tr('Выберите исходный и производный материалы.'))
        return
    dialog = MaterialLinkDialog([workspace.index.samples[sid] for sid in ids], workspace)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        workspace.attempt(lambda: workspace.commit(relate_materials(workspace.owner.model.project,
            dialog.parent_sample.currentData(), dialog.child_sample.currentData(), dialog.kind.currentData()),
            tr('Связать материалы')))
    dialog.deleteLater()


def remove_material_links(workspace):
    selected = set(workspace.selected_samples())
    state = settings(workspace.owner.model.project)
    state['relationships'] = [link for link in state.get('relationships', [])
                              if not selected.intersection((link['parent'], link['child']))]
    workspace.commit(state, tr('Удалить связи материалов'))


def save_selected_group(workspace):
    ids = workspace.selected_samples()
    name, ok = QInputDialog.getText(workspace, tr('Группа сравнения'), tr('Название группы'))
    if not ok:
        return
    if name.strip() in workspace.index.state.get('groups', {}):
        if QMessageBox.question(workspace, tr('Изменить группу'), tr('Заменить состав существующей группы?')) != QMessageBox.StandardButton.Yes:
            return
    workspace.attempt(lambda: workspace.commit(save_group(workspace.owner.model.project, name, ids), tr('Сохранить группу проб')))


def delete_group(workspace):
    names = list(workspace.index.state.get('groups', {}))
    if not names:
        return
    name, ok = QInputDialog.getItem(workspace, tr('Удалить группу'), tr('Группа'), names, 0, False)
    if ok:
        state = settings(workspace.owner.model.project)
        state['groups'].pop(name)
        workspace.commit(state, tr('Удалить группу проб'))


def measurement_record(workspace):
    rows = workspace.measurements.selectionModel().selectedRows()
    return workspace.measurements.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole) if rows else None


def choose_measurement(workspace):
    record = workspace.measurement_record()
    ids = workspace.selected_samples()
    if record and ids:
        workspace.attempt(lambda: workspace.commit(select_measurement(workspace.owner.model.project, ids[0], record['column_id'], record['id']),
                                        tr('Выбрать измерение паспорта')))


def reset_measurement(workspace):
    record = workspace.measurement_record()
    ids = workspace.selected_samples()
    if record and ids:
        workspace.commit(select_measurement(workspace.owner.model.project, ids[0], record['column_id'], None),
                    tr('Сбросить выбор измерения паспорта'))


def open_method(workspace):
    rows = workspace.methods.selectionModel().selectedRows()
    if not rows:
        return
    key = workspace.methods.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
    module_id = next(module for module, (_, methods) in modules().items() if key in methods)
    workspace.owner.show_nonpyro(module_id)
    dialog = workspace.owner.geochemistry_windows[module_id]
    dialog.method.setCurrentIndex(dialog.method.findData(key))
