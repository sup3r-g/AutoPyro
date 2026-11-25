"""Context actions for the stable object identifiers in the project explorer."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialogButtonBox, QFileDialog, QInputDialog, QMenu, QMessageBox
from shiboken6 import isValid

from .document_descriptors import DOCUMENTS, document_descriptor, geochemistry_documents
from .i18n import tr, message
from .theme import icon


def open_map_layer(desktop, identifier, attributes=False):
    existing = desktop.window.map_window
    if existing is not None and desktop.documents.indexOf(existing) >= 0:
        desktop.documents.setCurrentWidget(existing)
    else:
        desktop.window.show_maps()
    dialog = desktop.window.map_window
    row = next((i for i, layer in enumerate(dialog.layers) if layer['id'] == identifier), None)
    if row is None:
        return None
    desktop.document_changed(desktop.documents.indexOf(dialog))
    dialog.layer_table.clearSelection()
    dialog.layer_table.selectRow(row)
    if attributes:
        dialog.attribute_layer.setCurrentIndex(dialog.attribute_layer.findData(identifier))
        for index in range(dialog.tabs.count()):
            page = dialog.tabs.widget(index)
            if page.isAncestorOf(dialog.attribute_table):
                dialog.tabs.setCurrentIndex(index)
                break
    else:
        dialog.tabs.setCurrentWidget(dialog.layer_table)
    desktop.properties_dock.show()
    desktop.properties_dock.raise_()
    return dialog


class ExplorerMenu(QMenu):
    def __init__(self, desktop, payload):
        super().__init__(desktop.tree)
        self.desktop, self.window = desktop, desktop.window
        self.kind, self.key = payload
        self.build()

    def command(self, menu, icon_name, text, callback, enabled=True):
        action = menu.addAction(icon(icon_name), text)
        action.setEnabled(enabled)
        action.triggered.connect(lambda checked=False: callback())
        return action

    def open(self):
        self.desktop.open_object(self.kind, self.key)

    def properties(self):
        self.activate_document()
        if self.document() is not None:
            self.desktop.document_changed(self.desktop.documents.indexOf(self.document()))
        self.desktop.properties_dock.show()
        self.desktop.properties_dock.raise_()

    def document(self):
        if self.kind == 'table':
            return self.desktop.table_page
        kind = 'plot' if self.kind == 'saved_plot' else self.kind
        descriptor = document_descriptor(kind, self.key)
        return self.window.analysis_documents.cached(descriptor) if descriptor else None

    def call_document(self, method):
        self.activate_document()
        document = self.document()
        if document is not None and isValid(document):
            getattr(document, method)()

    def activate_document(self):
        if self.kind == 'saved_plot':
            self.open()
            return
        document = self.document()
        if document is not None and self.desktop.documents.indexOf(document) >= 0:
            self.desktop.documents.setCurrentWidget(document)
        else:
            self.open()

    def modules(self):
        menu = self.addMenu(icon('layout-template'), tr('Открыть модуль'))
        for kind in ('plot', 'map', 'statistics', 'curve_fitting', 'interpretation', 'interpolation',
                     'calculation', 'stratigraphy', 'qc'):
            self.module_command(menu, DOCUMENTS[kind])
        nonpyro = DOCUMENTS['nonpyro']
        chemistry = menu.addMenu(icon(nonpyro.icon), nonpyro.title())
        self.module_command(chemistry, nonpyro)
        chemistry.addSeparator()
        for descriptor in geochemistry_documents():
            self.module_command(chemistry, descriptor)
        menu.addSeparator()
        self.command(menu, 'file-text', tr('Отчёт…'), lambda: self.in_scope(self.window.show_report))

    def module_command(self, menu, descriptor):
        callback = descriptor.command(self.window)
        self.command(menu, descriptor.icon, descriptor.action_title(), lambda: self.in_scope(callback))

    def in_scope(self, callback):
        if self.kind == 'view':
            if self.key not in self.window.model.project.view.get('named', {}):
                return
            self.window.analysis_scope.set_mode(f'view:{self.key}')
        callback()

    def build(self):
        kind = self.kind
        if kind == 'project':
            self.command(self, 'settings-2', tr('Свойства проекта'), self.properties)
            self.command(self, 'table-2', tr('Открыть таблицу'), self.desktop.open_table)
            self.command(self, 'file-plus', tr('Новый проект'), self.window.new_project)
            self.command(self, 'folder-open', tr('Открыть проект…'), self.window.open_project)
            self.command(self, 'save', tr('Сохранить'), self.window.save)
            self.command(self, 'save-all', tr('Сохранить как…'), lambda: self.window.save(as_new=True))
            imports = self.addMenu(icon('import'), tr('Импорт данных'))
            for icon_name, label, method in (('import', tr('Открыть таблицу как новый проект…'), 'import_file'),
                    ('list-plus', tr('Добавить измерения в текущий проект…'), 'append_file'),
                    ('database', tr('Снимок базы данных…'), 'import_database'),
                    ('layout-template', tr('Шаблоны импорта…'), 'import_template_library')):
                self.command(imports, icon_name, label, getattr(self.window, method))
            database = bool(self.window.model.project.view.get('database_snapshot'))
            self.command(imports, 'refresh-cw', tr('Обновить снимок БД…'), self.window.refresh_database, database)
            self.addSeparator()
            self.command(self, 'file-text', tr('Отчёт…'), self.window.show_report)
            self.command(self, 'log-out', tr('Экспорт текущего представления…'), self.window.export)
            self.command(self, 'logs', tr('История изменений…'), self.window.history)
            self.command(self, 'logs', tr('Протоколы импорта…'), self.window.import_log)
            self.command(self, 'logs', tr('Протоколы обновления БД…'), self.window.database_refresh_log, database)
            self.command(self, 'settings-2', tr('Настройки приложения…'), self.window.show_settings)
        elif kind == 'folder':
            if self.key == 'views':
                self.command(self, 'plus', tr('Создать представление…'), self.window.analysis_scope.prompt_create)
            else:
                self.command(self, 'x', tr('Закрыть все'), self.desktop.close_documents,
                             self.desktop.documents.count() > 0)
        elif kind == 'view':
            scope = self.window.analysis_scope
            self.command(self, 'check', tr('Использовать для анализа'),
                         lambda: self.in_scope(lambda: None))
            self.command(self, 'sliders-horizontal', tr('Редактировать представление'),
                         lambda: self.in_scope(lambda: self.desktop.show_view_properties(self.key)))
            self.command(self, 'layout-template', tr('Применить представление к таблице'), self.open)
            self.command(self, 'tags', tr('Переименовать представление…'), lambda: scope.prompt_rename(name=self.key))
            self.command(self, 'trash', tr('Удалить представление…'), lambda: scope.prompt_delete(name=self.key))
            self.addSeparator()
            self.command(self, 'plus', tr('Создать представление…'), scope.prompt_create)
        elif kind == 'column':
            self.column_actions()
        elif kind == 'map_layer':
            self.layer_actions()
        else:
            self.command(self, 'folder-open', tr('Открыть'), self.open)
            self.command(self, 'sliders-horizontal', tr('Свойства'), self.properties)
            if kind == 'table':
                self.table_actions()
            elif kind == 'saved_plot':
                self.command(self, 'tags', tr('Переименовать график…'), self.rename_plot)
                self.command(self, 'trash', tr('Удалить сохранённый график…'), self.delete_plot)
                self.command(self, 'log-out', tr('Экспорт графика…'), self.export_saved_plot,
                             not (self.document() is not None and self.document().export_busy()))
            else:
                self.document_actions()
            self.close_actions()
        self.addSeparator()
        self.modules()
        if kind in ('project', 'folder'):
            self.addSeparator()
            self.command(self, 'maximize', tr('Развернуть всё'), self.desktop.tree.expandAll)
            self.command(self, 'panel-left', tr('Свернуть всё'), self.desktop.tree.collapseAll)

    def table_actions(self):
        self.addSeparator()
        for icon_name, label, page in (('funnel', tr('Фильтр…'), 'filter'), ('arrow-down-wide-narrow', tr('Сортировка…'), 'sort'),
                            ('columns-3', tr('Видимые колонки…'), 'columns'), ('layout-template', tr('Представления…'), 'views')):
            self.command(self, icon_name, label, lambda tab=page: self.desktop.show_table_properties(tab))
        self.command(self, 'plus', tr('Создать представление…'), self.window.analysis_scope.prompt_create)
        self.command(self, 'columns-3', tr('Добавить колонку…'), self.window.add_column)
        self.command(self, 'list-plus', tr('Добавить измерения в текущий проект…'), self.window.append_file)
        self.command(self, 'rotate-ccw', tr('Сбросить параметры представления'), self.window.restore_view_defaults)
        self.command(self, 'copy', tr('Копировать выделение'), self.window.copy_cells,
                     self.window.table.selectionModel().hasSelection())
        self.command(self, 'log-out', tr('Экспорт выделенных проб…'), self.window.export_selected_samples,
                     self.window.table.selectionModel().hasSelection())
        self.command(self, 'log-out', tr('Экспорт текущего представления…'), self.window.export)

    def column_index(self):
        return next((index for index, column in enumerate(self.window.model.project.columns)
                     if column.id == self.key), None)

    def column_actions(self):
        index = self.column_index()
        if index is None:
            return
        column = self.window.model.project.columns[index]
        self.command(self, 'columns-3', tr('Свойства колонки'), self.properties)
        visible = self.command(self, 'eye', tr('Показывать колонку'), self.toggle_column)
        visible.setCheckable(True)
        visible.setChecked(not self.window.table.isColumnHidden(index))
        header = self.window.table.horizontalHeader()
        self.command(self, 'panel-left', tr('Закрепить первой'), lambda: header.moveSection(header.visualIndex(index), 0))
        pinned = index in self.window.table.pinned_columns
        pin = self.command(self, 'panel-left', tr('Закрепить при прокрутке'),
                           lambda: self.window.pin_column(index, not pinned))
        pin.setCheckable(True)
        pin.setChecked(pinned)
        if column.kind == "number":
            self.command(self, 'sliders-horizontal', tr('Точность отображения…'), lambda: self.window.choose_column_precision(index))
        self.command(self, 'copy', tr('Копировать колонку'), self.copy_column, visible.isChecked())
        self.addSeparator()
        self.command(self, 'funnel', tr('Фильтр по колонке…'), self.filter_column)
        self.command(self, 'arrow-down-wide-narrow', tr('По возрастанию'), lambda: self.sort_column(Qt.SortOrder.AscendingOrder))
        self.command(self, 'arrow-down-wide-narrow', tr('По убыванию'), lambda: self.sort_column(Qt.SortOrder.DescendingOrder))
        self.addSeparator()
        self.command(self, 'chart-no-axes-combined', tr('График по колонке…') if column.kind == 'number' else tr('График с группировкой по колонке…'), self.plot_column)
        self.command(self, 'chart-column', tr('Статистика колонки…'), self.statistics_column)
        self.command(self, 'spline', tr('Интерполяция колонки…'), self.interpolate_column,
                     column.kind == 'number' and not column.metadata.get('method'))

    def toggle_column(self):
        index = self.column_index()
        if index is not None:
            item = self.window.columns.item(index)
            item.setCheckState(Qt.CheckState.Unchecked if not self.window.table.isColumnHidden(index) else Qt.CheckState.Checked)

    def copy_column(self):
        index = self.column_index()
        if index is not None:
            self.desktop.open_table()
            self.window.table.selectColumn(index)
            self.window.copy_cells()

    def sort_column(self, order):
        index = self.column_index()
        if index is not None:
            self.desktop.open_table()
            self.window.proxy.set_sort_rules([], index, order)
            self.window.view_changed()

    def filter_column(self):
        if self.column_index() is not None:
            self.desktop.show_table_properties('filter')
            properties = self.desktop.table_properties
            rule = next((rule for rule in properties.rules
                         if hasattr(rule, "column") and rule.column.currentData() == self.key), None)
            if rule is None:
                properties.add_rule({'column': self.key})
                rule = properties.rules[-1]
            rule.value.setFocus()

    def plot_column(self):
        index = self.column_index()
        if index is None:
            return
        self.window.show_plots()
        plot = self.window.plot_window
        if self.window.model.project.columns[index].kind == 'number':
            plot.data.load({'x_id': self.key})
        else:
            plot.style.load({'color_by': self.key, 'color_mode': 'category'})
        plot.schedule()

    def statistics_column(self):
        if self.column_index() is not None:
            self.window.show_statistics()
            dialog = self.window.statistics_window
            for index in range(dialog.properties.count()):
                item = dialog.properties.item(index)
                item.setCheckState(Qt.CheckState.Checked if item.data(Qt.ItemDataRole.UserRole) == self.key else Qt.CheckState.Unchecked)
            dialog.request_refresh()

    def interpolate_column(self):
        if self.column_index() is not None:
            self.window.interpolate()
            dialog = self.window.interpolation_window
            index = dialog.y.findData(self.key)
            if index >= 0:
                dialog.y.setCurrentIndex(index)

    def layer_actions(self):
        layers = self.window.map_window.layers if self.window.map_window is not None else self.window.model.project.view.get('map_layers', [])
        layer = next((layer for layer in layers if layer['id'] == self.key), None)
        if layer is None:
            return
        self.command(self, 'sliders-horizontal', tr('Свойства слоя'), self.open)
        self.command(self, 'table-2', tr('Атрибуты слоя'), lambda: open_map_layer(self.desktop, self.key, attributes=True))
        visible = self.command(self, 'eye', tr('Показывать слой'), self.toggle_layer)
        visible.setCheckable(True)
        visible.setChecked(layer.get('visible', True))
        self.command(self, 'trash', tr('Удалить слой'), self.remove_layer)

    def toggle_layer(self):
        dialog = open_map_layer(self.desktop, self.key)
        if dialog is not None:
            item = dialog.layer_table.item(dialog.layer_table.currentRow(), 0)
            item.setCheckState(Qt.CheckState.Unchecked if item.checkState() == Qt.CheckState.Checked else Qt.CheckState.Checked)
            dialog.remember()

    def remove_layer(self):
        dialog = open_map_layer(self.desktop, self.key)
        if dialog is not None:
            dialog.remove_layer()

    def close_actions(self):
        document = self.document()
        opened = document is not None and self.desktop.documents.indexOf(document) >= 0
        busy = document is not None and getattr(document, 'worker', None) is not None and document.worker.isRunning()
        self.addSeparator()
        self.command(self, 'x', tr('Закрыть'),
            lambda: self.desktop.close_document(self.desktop.documents.indexOf(document)) if document is not None and isValid(document) else None, opened and not busy)
        self.command(self, 'files', tr('Закрыть остальные'), lambda: self.desktop.close_documents(keep=document),
                     opened and self.desktop.documents.count() > 1)
        self.command(self, 'x', tr('Закрыть все'), self.desktop.close_documents, self.desktop.documents.count() > 0)

    def document_actions(self):
        document = self.document()
        if document is None:
            return
        busy = getattr(document, 'worker', None) is not None and document.worker.isRunning()
        result = getattr(document, 'report', None) is not None or getattr(document, 'result', None) is not None
        buttons = getattr(document, 'buttons', None)
        can_apply = (isinstance(buttons, QDialogButtonBox)
                     and buttons.button(QDialogButtonBox.StandardButton.Ok) is not None
                     and buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled())
        operations = {
            'plot': [('refresh-cw', tr('Обновить график'), 'schedule', True), ('save', tr('Сохранить график…'), 'save_named', result),
                     ('log-out', tr('Экспорт графика…'), 'export_plot', result and not document.export_busy()),
                     ('file-text', tr('JSON оформления…'), 'export_spec', True), ('import', tr('Загрузить JSON…'), 'import_spec', True)],
        } if self.kind == 'plot' else {}
        operations.update({
            'map': [('refresh-cw', tr('Обновить карту'), 'redraw', True), ('import', tr('Импорт слоя…'), 'import_layer', True),
                    ('log-out', tr('Экспорт карты…'), 'export', result), ('spline', tr('Пространственная интерполяция…'), 'open_interpolation', result)],
            'statistics': [('refresh-cw', tr('Пересчитать статистику'), 'request_refresh', True),
                    ('log-out', tr('Экспорт текущей вкладки'), 'export_current', result), ('file-text', tr('Воспроизводимый снимок'), 'export_snapshot', result)],
            'interpretation': [('refresh-cw', tr('Пересчитать интерпретацию'), 'calculate', bool(getattr(document, 'scheme', None)) and not getattr(document, 'editor_dirty', False)),
                    ('folder-open', tr('Загрузить схему…'), 'load_scheme', True), ('mouse-pointer-2', tr('Графический редактор схемы…'), 'open_geometry_editor', True),
                    ('file-plus', tr('Новая схема…'), 'new_graphical_scheme', True),
                    ('save', tr('Сохранить схему…'), 'save_scheme', bool(getattr(document, 'scheme', None))),
                    ('log-out', tr('Экспорт результатов JSON…'), 'save_report', result), ('log-out', tr('Экспорт результатов CSV…'), 'save_csv', result)],
            'interpolation': [('calculator', tr('Рассчитать предпросмотр'), 'calculate', True),
                    ('columns-3', tr('Создать колонку'), 'apply', can_apply),
                    ('shield-check', tr('Проверить сохранённый расчёт'), 'replay_saved', getattr(document, 'saved_runs', None) is not None and document.saved_runs.count() > 0)],
            'calculation': [('calculator', tr('Рассчитать предпросмотр'), 'calculate', True),
                    ('columns-3', tr('Создать колонки'), 'apply', can_apply)],
            'stratigraphy': [('layers', tr('Построить колонку'), 'show_column', True), ('shield-check', tr('Проверить привязку'), 'preview', True),
                    ('save', tr('Сохранить схему'), 'save_scheme', True), ('import', tr('Импорт схемы…'), 'import_scheme', True),
                    ('columns-3', tr('Создать колонки привязки'), 'add_columns', True), ('chart-no-axes-combined', tr('Построить профиль'), 'show_plot', True),
                    ('log-out', tr('Экспорт схемы…'), 'export_scheme', True),
                    ('log-out', tr('Экспорт колонки…'), 'export_column', getattr(document, 'export_column_button', None) is not None and document.export_column_button.isEnabled())],
            'nonpyro': [('calculator', tr('Рассчитать'), 'calculate', True), ('chart-no-axes-combined', tr('Построить график'), 'plot', result),
                    ('columns-3', tr('Создать колонки'), 'promote', result), ('table-2', tr('Выбрать пробы в таблице'), 'select_samples', result),
                    ('database', tr('Снимок базы данных…'), 'open_database', True), ('log-out', tr('Экспорт протокола…'), 'export', result)],
            'qc': [('refresh-cw', tr('Обновить проверки'), 'refresh_checks', True)],
        })
        self.addSeparator()
        for icon_name, label, method, enabled in operations.get('nonpyro' if self.kind == 'geochemistry' else self.kind, []):
            self.command(self, icon_name, label, lambda name=method: self.call_document(name), enabled and not busy)
        if hasattr(document, 'restore_parameters'):
            self.command(self, 'rotate-ccw', tr('Сбросить параметры'), lambda: self.call_document('restore_parameters'), not busy)
        view = getattr(document, 'canvas_view', None)
        if self.kind == 'interpretation':
            view = document.plot.canvas_view
        if view is not None:
            scale = self.addMenu(icon('maximize'), tr('Масштаб просмотра'))
            modes = (('maximize', tr('Подогнать под окно')), ('zoom-in', tr('Реальный размер')))
            for index, (icon_name, label) in enumerate(modes):
                action = self.command(scale, icon_name, label,
                    lambda mode=index: self.set_view_mode(mode))
                action.setCheckable(True)
                action.setChecked(view.fit == (index == 0))

    def set_view_mode(self, mode):
        self.activate_document()
        document = self.document()
        if document is not None:
            view = document.plot.canvas_view if self.kind == 'interpretation' else document.canvas_view
            view.mode_combo.setCurrentIndex(mode)

    def rename_plot(self):
        plots = self.window.model.project.view.get('plots', {})
        if self.key not in plots:
            return
        name, accepted = QInputDialog.getText(self.window, tr('Переименовать график…'), tr('Название'), text=self.key)
        if not accepted:
            return
        name = name.strip()
        if not name or (name != self.key and name in plots):
            QMessageBox.warning(self.window, tr('График'), tr('Название должно быть непустым и уникальным'))
            return
        if name != self.key:
            self.window.model.project.view['plots'] = {name if key == self.key else key: value for key, value in plots.items()}
            self.window.view_changed()

    def delete_plot(self):
        if self.key in self.window.model.project.view.get('plots', {}) and QMessageBox.question(self.window,
                tr('Удалить сохранённый график…'), tr('Удалить сохранённый график «{0}»?', self.key),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            del self.window.model.project.view['plots'][self.key]
            self.window.view_changed()

    def export_saved_plot(self):
        from autopyro.visualization.plotting import PlotSpec
        saved = self.window.model.project.view.get('plots', {}).get(self.key)
        if saved is None:
            return
        path, _ = QFileDialog.getSaveFileName(self.window, tr('Экспорт графика'), 'plot.png',
            'PNG (*.png);;TIFF (*.tif);;SVG (*.svg);;PDF (*.pdf)')
        if path:
            try:
                self.open()
                self.window.plot_window.start_export([(self.key, path, PlotSpec.from_dict(saved))])
            except (OSError, ValueError, TypeError, KeyError) as exc:
                QMessageBox.warning(self.window, tr('Экспорт'), message(exc))
