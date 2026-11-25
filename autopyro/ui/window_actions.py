"""Register the main window's menus, toolbars and keyboard shortcuts."""

from functools import partial

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QComboBox, QMessageBox, QToolButton, QWhatsThis

from .document_descriptors import DOCUMENTS, geochemistry_documents
from .i18n import tr
from .theme import icon
from .theme.controls import apply_control_icons


def install(window):
    window.actions = {}
    add = partial(_register_action, window)
    data_actions = _file_menu(window, add)
    _edit_menu(window, add)
    filter_action = _table_menu(window, add)
    analysis_actions = ((filter_action, tr('Фильтр')), *_analysis_menu(window, add))
    _settings_and_help_menus(window, add)
    _toolbars(window, data_actions, analysis_actions)
    refresh_icons(window)


def refresh_icons(window):
    apply_control_icons(window)


def _register_action(window, menu, icon_name, text, handler, shortcut=None):
    action = QAction(icon(icon_name), text, window)
    action.triggered.connect(handler)
    if shortcut:
        action.setShortcut(shortcut)
    menu.addAction(action)
    window.actions[text] = action
    return action


def _file_menu(window, add):
    file = window.menuBar().addMenu(tr('Файл'))
    new_project = add(file, 'file-plus', tr('Новый проект'), window.new_project, "Ctrl+N")
    open_project = add(file, 'folder-open', tr('Открыть проект…'), window.open_project, "Ctrl+O")
    add(file, 'database-backup', tr('Восстановить автосохранение…'), window.restore_autosave)
    save = add(file, 'save', tr('Сохранить'), window.save, "Ctrl+S")
    add(
        file, 'save-all', tr('Сохранить как…'), lambda: window.save(as_new=True), "Ctrl+Shift+S"
    )
    file.addSeparator()
    import_file = add(file, 'import', tr('Открыть таблицу как новый проект…'), window.import_file, "Ctrl+I")
    add(file, 'list-plus', tr('Добавить измерения в текущий проект…'), window.append_file)
    add(file, 'logs', tr('Протоколы импорта…'), window.import_log)
    add(file, 'layout-template', tr('Шаблоны импорта…'), window.import_template_library)
    import_database = add(file, 'database', tr('Снимок базы данных…'), window.import_database)
    add(file, 'database', tr('Импорт геохимии из БД…'), window.import_geochemistry)
    add(file, 'database', tr('Добавить геохимию из БД в проект…'), lambda: window.import_geochemistry(append=True))
    add(file, 'refresh-cw', tr('Обновить снимок БД…'), window.refresh_database)
    add(file, 'logs', tr('Протоколы обновления БД…'), window.database_refresh_log)
    add(file, 'log-out', tr('Экспорт текущего представления…'), window.export)
    add(file, 'log-out', tr('Экспорт выделенных проб…'), window.export_selected_samples)
    add(file, 'file-text', tr('Отчёт…'), window.show_report)
    add(file, 'flask-conical', tr('Демонстрационные данные'), window.demo)
    return (
        (new_project, tr('Новый проект')), (open_project, tr('Открыть проект')),
        (save, tr('Сохранить')), (import_file, tr('Новый из таблицы')), (import_database, tr('База')),
    )


def _edit_menu(window, add):
    edit = window.menuBar().addMenu(tr('Правка'))
    window.undo_action = add(
        edit, 'undo-2', tr('Отменить'), lambda: window.model.undo.undo(), "Ctrl+Z"
    )
    window.redo_action = add(
        edit, 'redo-2', tr('Повторить'), lambda: window.model.undo.redo(), "Ctrl+Y"
    )
    add(edit, 'copy', tr('Копировать'), window.copy_cells)
    add(edit, 'clipboard-paste', tr('Вставить'), window.paste_cells)
    window.clear_selection_action = add(edit, 'x', tr('Снять выделение'), lambda: window.sample_selection.clear())
    add(edit, 'table-2', tr('Заполнить выделенные ячейки…'), window.fill_cells)
    add(edit, 'columns-3', tr('Добавить колонку…'), window.add_column)
    add(edit, 'flask-conical', tr('Измерения выбранной ячейки…'), window.show_measurements)


def _table_menu(window, add):
    table = window.menuBar().addMenu(tr('Таблица'))
    filter_action = add(table, 'funnel', tr('Условия фильтрации…'), window.filter_dialog)
    window.table_actions = [
        filter_action,
        add(table, 'funnel-x', tr('Сбросить фильтр'), lambda: window.apply_filter([], "AND")),
        add(table, 'arrow-down-wide-narrow', tr('Сортировка по нескольким колонкам…'), window.multi_sort),
        add(table, 'save', tr('Сохранить представление…'), window.save_view),
        add(table, 'layout-template', tr('Открыть представление…'), window.load_view),
        add(table, 'panel-left', tr('Панель колонок'), lambda: window.desktop.show_table_properties("columns")),
        add(table, 'rotate-ccw', tr("Сбросить параметры представления"), window.restore_view_defaults),
    ]
    table.addSeparator()
    add(table, 'save', tr('Сохранить выделенные пробы как набор…'), window.save_sample_set)
    add(table, 'folder-open', tr('Открыть набор проб…'), window.open_sample_set)
    window.close_sample_set_action = add(table, 'x', tr('Закрыть набор проб'), window.close_sample_set)
    add(table, 'files', tr('Сравнить наборы проб…'), window.compare_sample_sets)
    add(table, 'tags', tr('Переименовать набор проб…'), window.rename_sample_set)
    add(table, 'trash', tr('Удалить набор проб…'), window.delete_sample_set)
    return filter_action


def _analysis_menu(window, add):
    analysis = window.menuBar().addMenu(tr('Обработка'))
    actions = {}
    for kind in ('qc', 'interpolation', 'statistics', 'curve_fitting', 'plot', 'calculation',
                 'interpretation', 'stratigraphy', 'map'):
        descriptor = DOCUMENTS[kind]
        actions[kind] = add(analysis, descriptor.icon, descriptor.action_title(),
                            descriptor.command(window), descriptor.shortcut)
    chemistry = DOCUMENTS['nonpyro']
    geochemistry = analysis.addMenu(icon(chemistry.icon), chemistry.title())
    workspace = DOCUMENTS['geochemistry_workspace']
    add(geochemistry, workspace.icon, workspace.action_title(), workspace.command(window))
    geochemistry.addSeparator()
    for descriptor in geochemistry_documents():
        add(geochemistry, descriptor.icon, descriptor.action_title(), descriptor.command(window))
    add(analysis, 'logs', tr('История изменений…'), window.history)
    return tuple((actions[kind], DOCUMENTS[kind].title()) for kind in ('plot', 'map', 'statistics'))


def _settings_and_help_menus(window, add):
    settings_menu = window.menuBar().addMenu(tr("Настройки"))
    add(settings_menu, 'settings-2', tr("Настройки приложения…"), window.show_settings)
    help_menu = window.menuBar().addMenu(tr("Помощь"))
    window.whats_this_action = QWhatsThis.createAction(window)
    window.whats_this_action.setText(tr("Что это?"))
    window.whats_this_action.setIcon(icon('circle-question-mark'))
    window.whats_this_action.setShortcut(QKeySequence("Shift+F1"))
    window.addAction(window.whats_this_action)
    help_menu.addAction(window.whats_this_action)
    from autopyro import __version__
    add(help_menu, 'circle-question-mark', tr('О программе'),
        lambda: QMessageBox.about(window, tr('О программе'), f"AutoPyro {__version__}"))



def _toolbar(window, title, name, icon_name, actions):
    toolbar = window.addToolBar(title)
    toolbar.setObjectName(name)
    toolbar.toggleViewAction().setIcon(icon(icon_name))
    toolbar.setIconSize(QSize(22, 22))
    toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    for action, short in actions:
        action.setIconText(short)
        toolbar.addAction(action)
    return toolbar


def _toolbars(window, data_actions, analysis_actions):
    _toolbar(window, tr('Данные'), 'dataToolbar', 'table-2', data_actions)
    analysis_toolbar = _toolbar(window, tr('Обработка'), 'analysisToolbar', 'calculator', analysis_actions)
    filter_action = analysis_actions[0][0]
    window.analysis_scope_combo = QComboBox()
    window.analysis_scope_combo.setMinimumContentsLength(12)
    window.analysis_scope_combo.setMaximumWidth(280)
    window.analysis_scope_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    window.analysis_scope_combo.setToolTip(tr('Область данных для всех модулей анализа'))
    window.analysis_scope_combo.setWhatsThis(window.analysis_scope_combo.toolTip())
    window.analysis_scope_combo.setAccessibleName(tr('Область'))
    analysis_toolbar.insertWidget(filter_action, window.analysis_scope_combo)
    window.analysis_views_button = QToolButton()
    window.analysis_views_button.setIcon(icon('ellipsis'))
    window.analysis_views_button.setText('⋯')
    window.analysis_views_button.setToolTip(tr('Управление представлениями'))
    window.analysis_views_button.setAccessibleName(tr('Управление представлениями'))
    window.analysis_views_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    window.analysis_views_button.setMenu(window.analysis_scope.create_menu(window.analysis_views_button))
    analysis_toolbar.insertWidget(filter_action, window.analysis_views_button)
