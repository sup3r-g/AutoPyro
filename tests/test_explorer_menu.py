"""Explorer context commands target the clicked object without activating it first."""

from copy import deepcopy

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox

from autopyro.io import build_project
from autopyro.plotting import PlotSpec
from autopyro.ui.geochemistry import modules
from autopyro.ui.i18n import tr
from autopyro.ui.window import MainWindow


@pytest.fixture
def owner(qtbot):
    project = build_project(['Well', 'X', 'Y'], [['A', 1, 3], ['B', 2, 4], ['A', 3, 5]], [2, 3, 4],
        [(0, 'Well', 'text', ''), (1, 'X', 'number', 'm'), (2, 'Y', 'number', 'm')], {})
    well = project.columns[0].id
    project.view['named'] = {name: {'rules': [{'column': well, 'operator': 'eq', 'value': name}],
                                   'combine': 'AND'} for name in ('A', 'B')}
    window = MainWindow()
    qtbot.addWidget(window, before_close_func=lambda w: setattr(w, 'dirty', False))
    window.set_project(project)
    window.may_replace = lambda: True
    window.autosave_timer.stop()
    yield window
    window.desktop.close_documents()
    window.dirty = False


def action(menu, source):
    return next(entry for entry in menu.actions() if entry.text() == tr(source))


def node(tree, payload):
    def find(item):
        if item.data(0, Qt.ItemDataRole.UserRole) == payload:
            return item
        return next((result for index in range(item.childCount()) if (result := find(item.child(index)))), None)
    return next((result for index in range(tree.topLevelItemCount()) if (result := find(tree.topLevelItem(index)))), None)


def test_popup_uses_clicked_view_without_changing_filters_or_opening_modules(owner, monkeypatch):
    owner.apply_filter([{'column': owner.model.project.columns[0].id, 'operator': 'eq', 'value': 'A'}], 'AND')
    owner.analysis_scope.set_mode('all')
    owner.show()
    owner.desktop.refresh_project()
    QApplication.processEvents()
    tree = owner.desktop.tree
    clicked = node(tree, ('view', 'B'))
    before = deepcopy(owner.capture_view())
    opened = owner.desktop.documents.count()
    observed = []
    def popup(menu, position):
        observed.append(menu.key)
        assert action(menu, 'Редактировать представление').isEnabled()
    monkeypatch.setattr('autopyro.ui.project_menu.ExplorerMenu.exec', popup)
    owner.desktop.project_menu(tree.visualItemRect(clicked).center())
    assert observed == ['B']
    assert owner.capture_view() == before
    assert owner.analysis_scope.mode == 'all'
    assert owner.desktop.documents.count() == opened
    assert owner.plot_window is None and owner.map_window is None


def test_all_modules_and_geochemistry_profiles_are_available(owner, qtbot, monkeypatch):
    called = []
    methods = [('Графики…', 'show_plots'), ('Карты…', 'show_maps'), ('Статистика…', 'show_statistics'),
        ('Интерпретация…', 'show_interpretation'), ('Интерполяция…', 'interpolate'),
        ('Расчёты пиролиза…', 'show_calculations'), ('Стратиграфия…', 'show_stratigraphy'),
        ('Контроль качества…', 'show_qc'), ('Отчёт…', 'show_report')]
    for _, method in methods:
        monkeypatch.setattr(owner, method, lambda name=method: called.append(name))
    chemistry = []
    monkeypatch.setattr(owner, 'show_nonpyro', lambda key=None: chemistry.append(key))
    menu = owner.desktop.create_project_menu(('view', 'B'))
    qtbot.addWidget(menu)
    modules_menu = action(menu, 'Открыть модуль').menu()
    for title, method in methods:
        action(modules_menu, title).trigger()
        assert called[-1] == method
        assert owner.analysis_scope.mode == 'view:B'
    chemistry_menu = action(modules_menu, 'Геохимия').menu()
    action(chemistry_menu, 'Все методы').trigger()
    for key, (title, _) in modules().items():
        next(entry for entry in chemistry_menu.actions() if entry.text() == title + '…').trigger()
    assert chemistry == [None, *modules()]


def test_column_actions_target_uuid_visibility_sorting_and_filter(owner, qtbot):
    project = owner.model.project
    cid = project.columns[1].id
    before = deepcopy(project.rows)
    menu = owner.desktop.create_project_menu(('column', cid))
    qtbot.addWidget(menu)
    action(menu, 'Показывать колонку').trigger()
    assert owner.table.isColumnHidden(1)
    action(menu, 'Показывать колонку').trigger()
    assert not owner.table.isColumnHidden(1)
    action(menu, 'Закрепить первой').trigger()
    assert owner.table.horizontalHeader().visualIndex(1) == 0
    action(menu, 'По убыванию').trigger()
    assert owner.row_ids() == [row.id for row in reversed(project.rows)]
    action(menu, 'Фильтр по колонке…').trigger()
    assert owner.desktop.table_properties.rules[-1].column.currentData() == cid
    count = len(owner.desktop.table_properties.rules)
    action(menu, 'Фильтр по колонке…').trigger()
    assert len(owner.desktop.table_properties.rules) == count
    action(menu, 'Копировать колонку').trigger()
    assert QApplication.clipboard().text().split() == ['3', '2', '1']
    assert project.rows == before


def test_view_commands_apply_to_target_instead_of_current_scope(owner, qtbot, monkeypatch):
    owner.analysis_scope.set_mode('view:B')
    source = deepcopy(owner.capture_view())
    menu = owner.desktop.create_project_menu(('view', 'A'))
    qtbot.addWidget(menu)
    monkeypatch.setattr(QInputDialog, 'getText', lambda *a, **kw: ('Renamed A', True))
    action(menu, 'Переименовать представление…').trigger()
    assert owner.analysis_scope.mode == 'view:B'
    assert 'Renamed A' in owner.model.project.view['named']
    assert owner.capture_view() == source
    renamed = owner.desktop.create_project_menu(('view', 'Renamed A'))
    qtbot.addWidget(renamed)
    action(renamed, 'Редактировать представление').trigger()
    assert owner.desktop.view_properties.name == 'Renamed A'
    assert owner.analysis_scope.mode == 'view:Renamed A'
    monkeypatch.setattr(QMessageBox, 'question', lambda *a: QMessageBox.StandardButton.Yes)
    owner.analysis_scope.set_mode('view:B')
    action(renamed, 'Удалить представление…').trigger()
    assert owner.analysis_scope.mode == 'view:B'
    assert 'Renamed A' not in owner.model.project.view['named']
    target = owner.desktop.create_project_menu(('view', 'B'))
    qtbot.addWidget(target)
    action(target, 'Применить представление к таблице').trigger()
    assert owner.row_ids() == [owner.model.project.rows[1].id]


def test_export_and_close_background_document_do_not_reset_its_result(owner, qtbot, monkeypatch):
    owner.show_maps()
    dialog = owner.map_window
    dialog.timer.stop()
    report = {'snapshot': True}
    dialog.report = report
    owner.desktop.open_table()
    exported = []
    monkeypatch.setattr(dialog, 'export', lambda: exported.append(dialog.report))
    menu = owner.desktop.create_project_menu(('map', ''))
    qtbot.addWidget(menu)
    assert action(menu, 'Экспорт карты…').isEnabled()
    action(menu, 'Экспорт карты…').trigger()
    assert exported == [report] and dialog.report is report
    owner.desktop.open_table()
    action(menu, 'Закрыть').trigger()
    assert owner.desktop.documents.indexOf(dialog) == -1
    assert owner.desktop.documents.currentWidget() is owner.desktop.table_page


def test_saved_plot_operations_preserve_values_and_export_clicked_spec(owner, qtbot, monkeypatch, tmp_path):
    project = owner.model.project
    spec = PlotSpec(x_id=project.columns[1].id, y_id=project.columns[2].id, marker_size=19)
    project.view['plots'] = {'Preset': spec.to_dict()}
    rows = deepcopy(project.rows)
    menu = owner.desktop.create_project_menu(('saved_plot', 'Preset'))
    qtbot.addWidget(menu)
    path = tmp_path / 'preset.png'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a: (str(path), 'PNG'))
    owner.show_plots()
    owner.plot_window.timer.stop()
    jobs = []
    monkeypatch.setattr(owner.plot_window, 'start_export', lambda descriptions: jobs.extend(descriptions))
    action(menu, 'Экспорт графика…').trigger()
    assert jobs[0][0] == 'Preset' and jobs[0][1] == str(path)
    assert jobs[0][2].to_dict() == spec.to_dict()
    monkeypatch.setattr(QInputDialog, 'getText', lambda *a, **kw: ('Renamed', True))
    action(menu, 'Переименовать график…').trigger()
    assert project.view['plots'] == {'Renamed': spec.to_dict()}
    renamed = owner.desktop.create_project_menu(('saved_plot', 'Renamed'))
    qtbot.addWidget(renamed)
    monkeypatch.setattr(QMessageBox, 'question', lambda *a: QMessageBox.StandardButton.Yes)
    action(renamed, 'Удалить сохранённый график…').trigger()
    assert project.view['plots'] == {}
    assert project.rows == rows
    owner.plot_window.timer.stop()


@pytest.mark.parametrize('method,kind,attribute,command', [
    ('show_plots', 'plot', 'plot_window', 'Обновить график'),
    ('show_statistics', 'statistics', 'statistics_window', 'Пересчитать статистику'),
    ('show_interpretation', 'interpretation', 'interpretation_window', 'Загрузить схему…'),
    ('interpolate', 'interpolation', 'interpolation_window', 'Рассчитать предпросмотр'),
    ('show_calculations', 'calculation', 'calculation_window', 'Рассчитать предпросмотр'),
    ('show_stratigraphy', 'stratigraphy', 'stratigraphy_window', 'Построить колонку'),
    ('show_nonpyro', 'nonpyro', 'nonpyro_window', 'Рассчитать'),
    ('show_qc', 'qc', 'qc_window', 'Обновить проверки'),
])
def test_document_menus_offer_existing_operations_without_executing_them(owner, qtbot, method, kind, attribute, command):
    getattr(owner, method)()
    document = getattr(owner, attribute)
    if hasattr(document, 'timer'):
        document.timer.stop()
    if kind == 'interpretation':
        document.plot.timer.stop()
    menu = owner.desktop.create_project_menu((kind, ''))
    qtbot.addWidget(menu)
    assert action(menu, command).isEnabled()
    assert action(menu, 'Закрыть').isEnabled()
    assert action(menu, 'Открыть модуль').menu() is not None
    if kind in ('interpolation', 'calculation'):
        assert not action(menu, 'Создать колонку' if kind == 'interpolation' else 'Создать колонки').isEnabled()
    if kind in ('plot', 'interpretation', 'stratigraphy'):
        scale = action(menu, 'Масштаб просмотра').menu()
        assert action(scale, 'Подогнать под окно').isChecked()
        action(scale, 'Реальный размер').trigger()
        view = document.plot.canvas_view if kind == 'interpretation' else document.canvas_view
        assert not view.fit


def test_map_layer_nodes_and_commands_affect_only_clicked_layer(owner, qtbot):
    layers = [dict(id=f'layer-{index}', name=f'Layer {index}', visible=True, color='#778899',
                   fill='#c7ddeb', alpha=.35, line_width=1, features=[], crs='EPSG:32631', source={})
              for index in range(2)]
    owner.model.project.view['map_layers'] = deepcopy(layers)
    owner.desktop.refresh_project()
    assert node(owner.desktop.tree, ('map_layer', 'layer-1')) is not None
    menu = owner.desktop.create_project_menu(('map_layer', 'layer-1'))
    qtbot.addWidget(menu)
    assert owner.map_window is None
    action(menu, 'Показывать слой').trigger()
    dialog = owner.map_window
    dialog.timer.stop()
    assert dialog.layers[0]['visible']
    assert not dialog.layers[1]['visible']
    assert not owner.model.project.view['map_layers'][1]['visible']
    dialog.layer_table.selectRow(0)
    action(menu, 'Удалить слой').trigger()
    dialog.timer.stop()
    assert [layer['id'] for layer in dialog.layers] == ['layer-0']
    assert [layer['id'] for layer in owner.model.project.view['map_layers']] == ['layer-0']


def test_context_menu_does_not_close_running_document(owner, qtbot):
    owner.show_statistics()
    dialog = owner.statistics_window
    dialog.timer.stop()
    class Running:
        def isRunning(self):
            return True
    original = dialog.worker
    dialog.worker = Running()
    try:
        menu = owner.desktop.create_project_menu(('statistics', ''))
        qtbot.addWidget(menu)
        assert not action(menu, 'Пересчитать статистику').isEnabled()
        assert not action(menu, 'Закрыть').isEnabled()
    finally:
        dialog.worker = original
