"""Shared UI selection of the rows used by analysis documents."""

from copy import deepcopy
import weakref

from PySide6.QtCore import QObject, QSignalBlocker, Signal, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QInputDialog, QMenu, QMessageBox
from shiboken6 import isValid

from autopyro.model.sample_sets import check_sample_set_deletion, delete_sample_set, rename_sample_set
from autopyro.model import saved_views

from .i18n import message, tr
from .theme import icon
from .table import FilterProxy
from .table.filter_rules import leaf_rules, saved_sort_column


class AnalysisScope(QObject):
    changed = Signal()
    saved_scopes_changed = Signal()
    saved_scope_renamed = Signal(str, str)

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.mode = 'filtered'
        self.revision = 0
        self._bindings = []
        self._sources = []
        self._named = {}
        self._sample_set_names = ()
        self._sample_sets = {}
        self._proxies = {}
        self._action_groups = []

    def reset(self):
        for reference, listener, callback in self._bindings:
            combo = reference()
            if combo is not None and isValid(combo):
                combo.currentIndexChanged.disconnect(listener)
                combo._analysis_scope = None
            if callback is not None:
                try:
                    self.changed.disconnect(callback)
                except (RuntimeError, TypeError):
                    pass
        self._bindings.clear()
        for signal, callback in self._sources:
            try:
                signal.disconnect(callback)
            except (RuntimeError, TypeError):
                pass
        self._sources.clear()
        self._clear_proxies()
        self._named = deepcopy(self.owner.model.project.view.get('named', {}))
        self._sample_set_names = tuple(self.owner.model.project.view.get('sample_sets', {}))
        self._sample_sets = {name: tuple(ids) for name, ids in
                             self.owner.model.project.view.get('sample_sets', {}).items()}
        saved = self.owner.model.project.view.get('analysis_scope', 'filtered')
        self.mode = saved if isinstance(saved, str) and saved in {value for _, value in self.choices()} else 'filtered'
        self.revision += 1
        for signal, callback in (
            (self.owner.sample_selection.changed, self._selection_changed),
            (self.owner.proxy.filtersChanged, self._filter_changed),
            (self.owner.proxy.layoutChanged, self._filter_changed),
            (self.owner.model.changed, self._notify),
        ):
            signal.connect(callback)
            self._sources.append((signal, callback))

    def choices(self):
        return [(tr('Текущая таблица'), 'filtered'), (tr('Все пробы'), 'all'),
                (tr('Выделенные строки'), 'selected')] + [
                    (tr('Представление: {0}', name), f'view:{name}') for name in self._named] + [
                    (tr('Набор проб: {0}', name), f'set:{name}') for name in self._sample_set_names]

    def bind(self, combo, callback=None):
        self.refresh()
        self._fill(combo)
        combo._analysis_scope = weakref.proxy(self)
        reference = weakref.ref(combo)
        manager_reference = weakref.ref(self)
        def selected(*args):
            widget = reference()
            manager = manager_reference()
            if widget is not None and isValid(widget) and manager is not None and isValid(manager):
                manager.set_mode(widget.currentData())
        combo.currentIndexChanged.connect(selected)
        if callback is not None:
            self.changed.connect(callback)
        self._bindings.append((reference, selected, callback))
        if not hasattr(combo, '_analysis_view_actions'):
            combo._analysis_view_actions = self._make_actions(combo)
            combo.addActions(combo._analysis_view_actions)
            combo.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        self._update_actions()

    def unbind(self, combo):
        for index, (reference, listener, callback) in enumerate(self._bindings):
            if reference() is combo:
                combo.currentIndexChanged.disconnect(listener)
                if callback is not None:
                    self.changed.disconnect(callback)
                del self._bindings[index]
                combo._analysis_scope = None
                return

    def _make_actions(self, parent):
        actions = []
        for icon_name, label, callback in (
            ('plus', tr('Создать представление…'), self.prompt_create),
            ('sliders-horizontal', tr('Редактировать представление'), self.edit_view),
            ('tags', tr('Переименовать представление…'), self.prompt_rename),
            ('trash', tr('Удалить представление…'), self.prompt_delete),
        ):
            action = QAction(icon(icon_name), label, parent)
            action.triggered.connect(callback)
            actions.append(action)
        self._action_groups.append(tuple(weakref.ref(action) for action in actions))
        self._update_actions()
        return actions

    def create_menu(self, parent):
        menu = QMenu(parent)
        menu._view_actions = self._make_actions(menu)
        menu.addActions(menu._view_actions)
        menu.aboutToShow.connect(self._update_actions)
        return menu

    def _update_actions(self):
        selected_view = self.mode.startswith('view:') and self.mode[5:] in self._named
        valid_groups = []
        for group in self._action_groups:
            actions = [reference() for reference in group]
            if all(action is not None and isValid(action) for action in actions):
                for action in actions[1:]:
                    action.setEnabled(selected_view)
                valid_groups.append(group)
        self._action_groups = valid_groups

    def _dialog_parent(self):
        return QApplication.activeModalWidget() or self.owner

    def edit_view(self):
        if self.mode.startswith('view:'):
            self.owner.desktop.show_view_properties(self.mode[5:])

    def prompt_create(self):
        name, accepted = QInputDialog.getText(self._dialog_parent(), tr('Создать представление…'),
            tr('Название представления (сохраняется текущий вид таблицы):'))
        if accepted:
            try:
                self.create_view(name)
            except ValueError as exc:
                QMessageBox.warning(self._dialog_parent(), tr('Представления'), message(exc))

    def prompt_rename(self, *, name=None):
        if name is None and not self.mode.startswith('view:'):
            return
        old_name = self.mode[5:] if name is None else name
        name, accepted = QInputDialog.getText(self._dialog_parent(), tr('Переименовать представление…'),
                                             tr('Новое название представления:'), text=old_name)
        if accepted:
            try:
                self.rename_view(old_name, name)
            except ValueError as exc:
                QMessageBox.warning(self._dialog_parent(), tr('Представления'), message(exc))

    def prompt_delete(self, *, name=None):
        if name is None and not self.mode.startswith('view:'):
            return
        name = self.mode[5:] if name is None else name
        if QMessageBox.question(self._dialog_parent(), tr('Удалить представление…'),
            tr('Удалить представление «{0}»?', name),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            try:
                self.delete_view(name)
            except ValueError as exc:
                QMessageBox.warning(self._dialog_parent(), tr('Представления'), message(exc))

    def _saved_scope_settings(self):
        view = self.owner.model.project.view
        for key, label in (('plot_current', tr('График')), ('interpretation_plot_current', tr('График интерпретации')),
                           ('statistics', tr('Статистика')), ('interpretation_current', tr('Интерпретация')),
                           ('map_current', tr('Карта'))):
            yield label, view.get(key)
        yield from view.get('plots', {}).items()

    def sample_set_dependencies(self, name):
        scope = f'set:{name}'
        names = [label for label, settings in self._saved_scope_settings()
                 if isinstance(settings, dict) and settings.get('scope') == scope]
        statistics = self.owner.model.project.view.get("statistics") or {}
        if scope in (statistics.get("comparison_scope_a"), statistics.get("comparison_scope_b")):
            names.append(tr('Сравнение в статистике'))
        return list(dict.fromkeys(names))

    def _retarget_scopes(self, old_scope, mode):
        view = self.owner.model.project.view
        statistics = view.get("statistics")
        if isinstance(statistics, dict):
            for key in ("comparison_scope_a", "comparison_scope_b"):
                if statistics.get(key) == old_scope:
                    statistics[key] = mode if mode.startswith(("view:", "set:")) else None
            for key in ("comparison_a", "comparison_b"):
                if old_scope.startswith("view:") and statistics.get(key) == old_scope[5:]:
                    statistics[key] = mode[5:] if mode.startswith("view:") else None
        settings = [settings for _, settings in self._saved_scope_settings()]
        map_window = self.owner.map_window
        if map_window is not None and map_window.spec.scope == old_scope:
            map_window.spec.scope = mode
        plots = [self.owner.plot_window]
        if self.owner.interpretation_window is not None:
            plots.append(self.owner.interpretation_window.plot)
        for plot in plots:
            if plot is None:
                continue
            if plot.spec.scope == old_scope:
                plot.spec.scope = mode
            settings.append(plot.last_config)
            for index in range(plot.undo_stack.count()):
                command = plot.undo_stack.command(index)
                settings.extend((getattr(command, 'before', None), getattr(command, 'after', None)))
        for entry in settings:
            if isinstance(entry, dict) and entry.get('scope') == old_scope:
                entry['scope'] = mode

    def _commit_views(self, select_name=None):
        self.refresh()
        self.owner.view_changed()
        properties = self.owner.desktop.table_properties
        properties.refresh()
        if select_name is not None:
            properties.saved_views.setCurrentText(select_name)
            properties.view_name.setText(select_name)

    def create_view(self, name):
        name = saved_views.create_view(
            self.owner.model.project.view, name, self.owner.capture_view(include_sample_set=True),
        )
        self._commit_views(name)
        self.set_mode(f'view:{name}')

    def update_view(self, name, changes):
        if saved_views.update_view(self.owner.model.project.view, name, changes):
            self._commit_views()

    def rename_view(self, old_name, name):
        project = self.owner.model.project
        name = saved_views.rename_view(project.view, old_name, name)
        if name == old_name:
            return
        self._retarget_scopes(f'view:{old_name}', f'view:{name}')
        self.saved_scope_renamed.emit(f'view:{old_name}', f'view:{name}')
        self.owner.desktop.view_properties.rename_target(old_name, name)
        if self.mode == f'view:{old_name}':
            self.mode = f'view:{name}'
            project.view['analysis_scope'] = self.mode
        properties = self.owner.desktop.table_properties
        selected = properties.saved_views.currentText() == old_name
        if properties.view_name.text() == old_name:
            properties.view_name.setText(name)
        self._commit_views(name if selected else None)

    def delete_view(self, name):
        saved_views.delete_view(self.owner.model.project.view, name)
        self._retarget_scopes(f'view:{name}', 'filtered')
        if self.owner.desktop.table_properties.view_name.text() == name:
            self.owner.desktop.table_properties.view_name.clear()
        self._commit_views()

    def rename_sample_set(self, old_name, name):
        view = self.owner.model.project.view
        name = rename_sample_set(view, old_name, name)
        if name == old_name:
            return
        if self.owner.active_sample_set == old_name:
            self.owner.active_sample_set = name
        old_scope, new_scope = f'set:{old_name}', f'set:{name}'
        self._retarget_scopes(old_scope, new_scope)
        self.saved_scope_renamed.emit(old_scope, new_scope)
        active = self.mode == old_scope
        if active:
            self.mode = view['analysis_scope'] = new_scope
        self.refresh()
        self.owner.view_changed()
        if active:
            self._notify()

    def check_sample_set_deletion(self, name):
        check_sample_set_deletion(self.owner.model.project.view, name)

    def delete_sample_set(self, name):
        view = self.owner.model.project.view
        delete_sample_set(view, name)
        self._retarget_scopes(f'set:{name}', 'filtered')
        if self.owner.active_sample_set == name:
            self.owner.close_sample_set()
        self.refresh()
        self.owner.view_changed()

    def _fill(self, combo):
        blocker = QSignalBlocker(combo)
        combo.clear()
        for label, value in self.choices():
            combo.addItem(label, value)
            combo.setItemData(combo.count() - 1, label, Qt.ItemDataRole.ToolTipRole)
        combo.setCurrentIndex(combo.findData(self.mode))
        del blocker

    def _update_combos(self):
        for reference, _, _ in self._bindings:
            combo = reference()
            if combo is not None and isValid(combo):
                self._fill(combo)
        self._update_actions()

    def set_mode(self, mode):
        if mode not in {value for _, value in self.choices()}:
            mode = 'filtered'
        if mode == self.mode:
            self._update_combos()
            return
        self.mode = mode
        self.owner.model.project.view['analysis_scope'] = mode
        self._update_combos()
        self.owner.view_changed()
        self._notify()

    def refresh(self):
        named = self.owner.model.project.view.get('named', {})
        sample_sets = {name: tuple(ids) for name, ids in
                       self.owner.model.project.view.get('sample_sets', {}).items()}
        set_names = tuple(sample_sets)
        named_changed = named != self._named
        sets_changed = sample_sets != self._sample_sets
        if not named_changed and not sets_changed and set_names == self._sample_set_names:
            return
        self._sample_set_names = set_names
        self._sample_sets = sample_sets
        if named_changed:
            self._named = deepcopy(named)
        if named_changed or sets_changed:
            self._clear_proxies()
        self.saved_scopes_changed.emit()
        if self.mode not in {value for _, value in self.choices()}:
            self.set_mode('filtered')
        else:
            self._update_combos()
            if ((named_changed or sets_changed) and self.mode.startswith('view:')
                    or sets_changed and self.mode.startswith('set:')):
                self._notify()

    def _clear_proxies(self):
        for proxy in self._proxies.values():
            proxy.deleteLater()
        self._proxies.clear()

    def _notify(self, *args):
        self.revision += 1
        self.changed.emit()

    def _selection_changed(self, *args):
        if self.mode == 'selected':
            self._notify()

    def _filter_changed(self, *args):
        if self.mode == 'filtered':
            self._notify()

    def rows(self, mode=None, *, selected_ids=None):
        mode = self.mode if mode is None else mode
        project = self.owner.model.project
        if mode == 'all':
            return [row.id for row in project.rows]
        if mode == 'selected':
            if selected_ids is None:
                return self.owner.sample_selection.ids(ordered=True)
            selected = set(selected_ids)
            return [rid for rid in self.owner.row_ids() if rid in selected] if selected else []
        if mode.startswith('set:'):
            sets = project.view.get('sample_sets', {})
            if mode[4:] not in sets:
                raise ValueError(tr('Сохранённый набор проб отсутствует.'))
            saved_ids = sets[mode[4:]]
            positions = self.owner.model.resolve_row_positions(saved_ids)
            missing = len(saved_ids) - len(positions)
            if missing:
                self.owner.statusBar().showMessage(
                    tr('Проб в наборе, отсутствующих в проекте: {0}', missing), 10_000,
                )
            return [rid for rid in saved_ids if rid in positions]
        if mode.startswith('view:'):
            view = project.view.get('named', {}).get(mode[5:])
            if view is None:
                raise ValueError(tr('Сохранённое представление отсутствует'))
            sample_set = view.get('sample_set')
            row_ids = None
            if sample_set is not None:
                sets = project.view.get('sample_sets', {})
                if sample_set not in sets:
                    raise ValueError(tr('Сохранённый набор проб отсутствует.'))
                row_ids = sets[sample_set]
            columns = {column.id for column in project.columns}
            if any(rule['column'] not in columns for rule in leaf_rules(view.get('rules', []))):
                raise ValueError(tr('Колонка сохранённого фильтра удалена; обновите представление'))
            if mode not in self._proxies:
                proxy = FilterProxy(self)
                proxy.setSourceModel(self.owner.model)
                proxy.set_rules(view.get('rules', []), view.get('combine', 'AND'), row_ids=row_ids)
                sort = [rule for rule in view.get('sort', []) if rule[0] in columns]
                proxy.set_sort_rules(sort, saved_sort_column(view, project.columns),
                    Qt.SortOrder.DescendingOrder if view.get('descending') else Qt.SortOrder.AscendingOrder)
                self._proxies[mode] = proxy
            return list(self._proxies[mode].ordered_row_ids())
        proxy = self.owner.proxy
        if proxy.row_id_filter is None and not proxy.rules and not proxy.sort_rules and proxy.sortColumn() < 0:
            return [row.id for row in project.rows]
        return self.owner.row_ids()
