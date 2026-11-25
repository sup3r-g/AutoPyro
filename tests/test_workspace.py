"""Docked scientific desktop navigation and portable KDE Breeze styling."""
from copy import deepcopy

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QDialog, QTabBar

from autopyro.core import Project
from autopyro.io import build_project
from autopyro.plotting import PlotSpec
from autopyro.ui import i18n, settings
from autopyro.ui.theme import ASSETS, apply_theme, breeze_palette, icon
from autopyro.ui.window import MainWindow


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch, qapp):
    store = QSettings(str(tmp_path / "workspace.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(i18n, "settings", lambda: store)
    i18n.initialize()
    project = build_project(["Well", "X", "Y"], [["A", "1", "3"], ["B", "2", "4"]], [2, 3],
                            [(0, "Well", "text", ""), (1, "X", "number", "m"), (2, "Y", "number", "m")], {})
    widget = MainWindow()
    qtbot.addWidget(widget)
    widget.set_project(project)
    widget.may_replace = lambda: True
    yield widget
    widget.dirty = False
    apply_theme(qapp, "breeze_light")


def find_item(tree, payload):
    def visit(item):
        if item.data(0, Qt.ItemDataRole.UserRole) == payload:
            return item
        for index in range(item.childCount()):
            result = visit(item.child(index))
            if result is not None:
                return result
    return visit(tree.topLevelItem(0))


def test_table_close_button_preserves_data_view_selection_and_undo(window):
    desktop = window.desktop
    model, selection = window.model, window.table.selectionModel()
    column = model.project.columns[1].id
    model.setData(model.index(0, 1), '9')
    window.apply_filter([{'column': column, 'operator': 'gt', 'value': 3}], 'AND')
    window.table.selectRow(0)
    selected = selection.selectedIndexes()
    view = window.capture_view()
    before = deepcopy(model.project.rows)
    bar = desktop.documents.tabBar()
    close = next(button for side in (QTabBar.ButtonPosition.LeftSide, QTabBar.ButtonPosition.RightSide)
                 if (button := bar.tabButton(desktop.documents.indexOf(desktop.table_page), side)) is not None)
    close.click()
    assert desktop.documents.count() == 0
    assert desktop.documents.currentWidget() is None
    assert not desktop.close_action.isEnabled()
    assert desktop.properties_stack.currentWidget() is desktop.properties
    assert window.model is model
    assert model.project.rows == before
    assert window.capture_view() == view
    desktop.tree.setCurrentItem(find_item(desktop.tree, ('table', '')))
    assert desktop.documents.currentWidget() is desktop.table_page
    assert desktop.documents.count() == 1
    assert desktop.close_action.isEnabled()
    assert window.table.selectionModel() is selection
    assert selection.selectedIndexes() == selected
    assert window.capture_view() == view
    model.undo.undo()
    assert model.project.rows[0].values[column] == 1


def test_analysis_at_index_zero_can_close_after_table_is_closed(window):
    desktop = window.desktop
    window.show_plots()
    plot = window.plot_window
    plot.timer.stop()
    desktop.close_document(desktop.documents.indexOf(desktop.table_page))
    assert desktop.documents.currentWidget() is plot
    assert desktop.documents.indexOf(plot) == 0
    assert desktop.properties_stack.currentWidget() is plot.tabs
    assert desktop.close_action.isEnabled()
    desktop.close_action.trigger()
    assert desktop.documents.count() == 0
    assert desktop._borrowed is None
    desktop.show_table_properties('filter')
    assert desktop.documents.currentWidget() is desktop.table_page
    assert desktop.properties_stack.currentWidget() is desktop.table_properties


def test_new_project_reopens_closed_table_and_removes_first_analysis(window):
    desktop = window.desktop
    desktop.close_document(desktop.documents.indexOf(desktop.table_page))
    window.show_plots()
    window.plot_window.timer.stop()
    assert desktop.documents.count() == 1
    window.set_project(Project())
    assert desktop.documents.count() == 1
    assert desktop.documents.currentWidget() is desktop.table_page
    assert desktop.table_page.currentIndex() == 1
    assert window.plot_window is None
    assert desktop._borrowed is None
    desktop.open_table()
    assert desktop.documents.count() == 1


def test_qc_navigation_reopens_closed_table(window):
    project = build_project(["Well", "TOC"], [["A", "101"], ["B", "2"]], [2, 3],
                            [(0, "Well", "text", ""), (1, "TOC", "number", "%wt")], {})
    window.set_project(project)
    desktop = window.desktop
    desktop.close_document(desktop.documents.indexOf(desktop.table_page))
    window.show_qc()
    dialog = window.qc_window
    dialog.timer.stop()
    issue_row = next(i for i, issue in enumerate(dialog.results_model.issues)
                     if issue["rule_id"] == "range" and issue["sample_id"] == project.rows[0].id)
    dialog.table.selectRow(issue_row)
    dialog.go_to_sample()
    assert desktop.documents.currentWidget() is desktop.table_page
    assert desktop.documents.count() == 2
    assert window.table.selectionModel().selectedRows()[0].row() == 0
    assert window.proxy.mapToSource(window.table.currentIndex()).column() == 1


def test_reordered_tabs_keep_table_navigation_and_document_properties(window):
    desktop = window.desktop
    window.show_plots()
    plot = window.plot_window
    plot.timer.stop()
    assert desktop.documents.isMovable()
    desktop.documents.tabBar().moveTab(0, 1)
    assert desktop.documents.widget(0) is plot
    assert desktop.documents.widget(1) is desktop.table_page
    assert desktop.documents.currentWidget() is plot
    assert desktop.properties_stack.currentWidget() is plot.tabs
    desktop.open_table()
    assert desktop.documents.currentIndex() == 1
    assert desktop.properties_stack.currentWidget() is desktop.table_properties
    desktop.close_action.trigger()
    assert desktop.documents.currentWidget() is plot
    assert desktop.properties_stack.currentWidget() is plot.tabs


@pytest.mark.parametrize('keep_table', [True, False])
def test_tab_context_menu_closes_others_relative_to_clicked_tab(window, keep_table):
    desktop = window.desktop
    window.show_plots()
    plot = window.plot_window
    plot.timer.stop()
    desktop.documents.tabBar().moveTab(0, 1)
    keep = desktop.table_page if keep_table else plot
    desktop.documents.setCurrentWidget(plot if keep_table else desktop.table_page)
    before = deepcopy(window.model.project.rows)
    menu = desktop.create_document_menu(keep)
    next(action for action in menu.actions() if action.text() == i18n.tr('Закрыть остальные')).trigger()
    assert desktop.documents.count() == 1
    assert desktop.documents.currentWidget() is keep
    assert window.model.project.rows == before
    assert desktop.properties_stack.currentWidget() is (desktop.table_properties if keep_table else plot.tabs)
    menu.deleteLater()


def test_tab_context_menu_closes_all_and_cached_documents_can_reopen(window):
    desktop = window.desktop
    window.show_plots()
    plot = window.plot_window
    plot.timer.stop()
    desktop.documents.tabBar().moveTab(0, 1)
    before = deepcopy(window.model.project.rows)
    menu = desktop.create_document_menu(desktop.table_page)
    next(action for action in menu.actions() if action.text() == i18n.tr('Закрыть все')).trigger()
    assert desktop.documents.count() == 0
    assert desktop._borrowed is None
    assert not desktop.close_action.isEnabled()
    assert window.model.project.rows == before
    window.show_plots()
    plot.timer.stop()
    assert window.plot_window is plot
    assert desktop.documents.currentWidget() is plot
    desktop.open_table()
    assert desktop.documents.count() == 2
    menu.deleteLater()


def test_bulk_close_respects_document_refusal(window, monkeypatch):
    desktop = window.desktop
    window.show_interpretation()
    dialog = window.interpretation_window
    dialog.timer.stop()
    monkeypatch.setattr(dialog, 'close', lambda: False)
    desktop.close_documents()
    assert desktop.documents.count() == 1
    assert desktop.documents.currentWidget() is dialog
    assert desktop.properties_stack.currentWidget() is dialog.parameter_tabs


def test_project_explorer_search_and_column_properties(window):
    desktop = window.desktop
    column = window.model.project.columns[1]
    item = find_item(desktop.tree, ("column", column.id))
    assert item.text(0) == "X"
    desktop.tree.setCurrentItem(item)
    properties = desktop.table_properties.summary
    assert properties.fields["name"].text() == "X"
    assert properties.fields["unit"].text() == "m"
    before = deepcopy(window.model.project.rows)
    properties.visible_column.setChecked(False)
    assert window.table.isColumnHidden(1)
    desktop.search.setText("Well")
    assert item.isHidden()
    assert window.proxy.rowCount() == 2
    assert window.model.project.rows == before
    desktop.search.clear()
    assert not item.isHidden()


def test_table_properties_are_separate_from_project_and_visualization(window, qtbot):
    desktop = window.desktop
    properties = desktop.table_properties
    table_menu = next(action.menu() for action in window.menuBar().actions()
                      if action.text() == i18n.tr("Таблица"))
    assert set(window.table_actions) <= set(table_menu.actions())
    assert properties is not desktop.properties
    assert desktop.properties_stack.currentWidget() is properties
    assert set(properties.pages) == {"table", "filter", "sort", "columns", "views"}
    desktop.refresh_project()
    assert desktop.properties_stack.currentWidget() is properties

    # Use the actual reset action from the panel after changing table state.
    before = deepcopy(window.model.project.rows)
    window.toggle_all(False)
    assert window.table.isColumnHidden(1)
    qtbot.mouseClick(properties.reset_button, Qt.MouseButton.LeftButton)
    assert all(not window.table.isColumnHidden(i) for i in range(window.model.columnCount()))
    assert window.model.project.rows == before

    window.show_plots()
    window.plot_window.timer.stop()
    assert desktop.properties_stack.currentWidget() is window.plot_window.tabs
    desktop.tree.setCurrentItem(find_item(desktop.tree, ("table", "")))
    assert desktop.documents.currentWidget() is desktop.table_page
    assert desktop.properties_stack.currentWidget() is properties
    desktop.tree.setCurrentItem(find_item(desktop.tree, ("project", "")))
    assert desktop.properties_stack.currentWidget() is desktop.properties
    assert desktop.properties.heading.text() == i18n.tr("Свойства проекта")
    desktop.tree.setCurrentItem(find_item(desktop.tree, ("column", window.model.project.columns[1].id)))
    assert desktop.properties_stack.currentWidget() is properties


def test_table_actions_open_embedded_pages_without_modal_windows(window, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    def unexpected(*args, **kwargs):
        pytest.fail("Table properties must not open a modal dialog")
    monkeypatch.setattr(QDialog, "exec", unexpected)
    monkeypatch.setattr(QInputDialog, "getText", unexpected)
    monkeypatch.setattr(QInputDialog, "getItem", unexpected)
    for label, key in (("Условия фильтрации…", "filter"), ("Сортировка по нескольким колонкам…", "sort"),
                       ("Сохранить представление…", "views"), ("Открыть представление…", "views"),
                       ("Панель колонок", "columns")):
        window.actions[i18n.tr(label)].trigger()
        editor = window.desktop.table_properties
        assert window.desktop.properties_stack.currentWidget() is editor
        assert editor.currentWidget() is editor.pages[key]


def test_undo_and_redo_shortcuts_follow_the_current_project(window, qtbot):
    window.show()
    window.activateWindow()
    qtbot.waitUntil(window.isActiveWindow)
    window.table.setFocus()

    for project in (window.model.project, deepcopy(window.model.project)):
        if project is not window.model.project:
            window.set_project(project)
        column = project.columns[0].id
        original = project.rows[0].values[column]
        assert window.model.setData(window.model.index(0, 0), 'Changed')
        qtbot.keyClick(window.table, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        assert project.rows[0].values[column] == original
        qtbot.keyClick(window.table, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
        assert project.rows[0].values[column] == 'Changed'


def test_cell_and_header_context_menus_dispatch_to_the_table(window, qtbot, monkeypatch):
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QMenu
    from autopyro.ui.table import context_menus

    window.show()
    qtbot.waitUntil(window.table.isVisible)
    index = window.proxy.index(0, 1)
    selected = []
    monkeypatch.setattr(window, 'show_measurements', lambda: selected.append(window.table.currentIndex()))

    class CommandMenu(QMenu):
        def exec(self, position):
            action = next(action for action in self.actions() if action.text() == i18n.tr(command))
            assert not action.icon().isNull()
            action.trigger()

    monkeypatch.setattr(context_menus, 'QMenu', CommandMenu)
    command = 'Измерения, источник и история…'
    window.cell_menu(window.table.visualRect(index).center())
    assert selected == [index]

    command = 'Закрепить при прокрутке'
    header = window.table.horizontalHeader()
    window.header_menu(QPoint(header.sectionViewportPosition(1) + 1, 1), header)
    assert 1 in window.table.pinned_columns


def test_add_column_from_main_menu_can_be_undone(window, monkeypatch):
    from PySide6.QtWidgets import QInputDialog
    from autopyro.ui.field_types import field_kind_label

    monkeypatch.setattr(QInputDialog, 'getText', lambda *args: (' New column ', True))
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args, **kwargs: (field_kind_label('number'), True))
    count = window.model.columnCount()
    window.actions[i18n.tr('Добавить колонку…')].trigger()

    column = window.model.project.columns[-1]
    assert (column.name, column.kind) == ('New column', 'number')
    assert window.model.columnCount() == count + 1
    window.undo_action.trigger()
    assert window.model.columnCount() == count


def test_inline_filter_validation_application_and_draft_preservation(window):
    editor = window.desktop.table_properties
    editor.add_rule()
    rule = editor.rules[0]
    column = window.model.project.columns[1]
    rule.column.setCurrentIndex(rule.column.findData(column.id))
    rule.operator.setCurrentIndex(rule.operator.findData("ge"))
    rule.value.setText("invalid")
    before = deepcopy(window.model.project.rows)
    editor.apply_filter()
    assert editor.filter_error.text()
    assert window.proxy.rules == []
    rule.value.setText("2")
    window.desktop.refresh_project()
    assert editor.rules[0] is rule and rule.value.text() == "2"
    editor.apply_filter()
    assert not editor.filter_error.text()
    assert window.proxy.rowCount() == 1
    assert window.proxy.rules[0]["value"] == 2
    assert window.model.project.rows == before
    editor.reset_filter()
    assert window.proxy.rowCount() == 2 and not editor.rules
    assert editor.combine.currentText() == "AND"
    editor.add_rule()
    editor.rules[0].value.setText("pending draft")
    editor.sort_columns.item(0).setCheckState(Qt.CheckState.Checked)
    window.actions[i18n.tr("Сбросить параметры представления")].trigger()
    assert not editor.rules
    assert all(editor.sort_columns.item(i).checkState() == Qt.CheckState.Unchecked for i in range(editor.sort_columns.count()))


def test_inline_sorting_priority_direction_and_reset(window):
    editor = window.desktop.table_properties
    editor.sort_columns.setCurrentRow(2)
    editor.move_sort(-1)
    editor.move_sort(-1)
    first = editor.sort_columns.item(0)
    first.setCheckState(Qt.CheckState.Checked)
    editor.reverse_sort(first)
    editor.sort_columns.item(1).setCheckState(Qt.CheckState.Checked)
    before = deepcopy(window.model.project.rows)
    editor.apply_sort()
    assert window.proxy.sort_rules[0] == (window.model.project.columns[2].id, True)
    assert window.row_ids() == [before[1].id, before[0].id]
    assert window.model.project.rows == before
    window.apply_filter([{"column": window.model.project.columns[0].id, "operator": "eq", "value": "B", "negate": False}], "AND")
    editor.reset_sort()
    assert not window.proxy.sort_rules and window.proxy.sortColumn() == -1
    assert window.proxy.rowCount() == 1


def test_restoring_rules_resorts_even_with_same_native_column_and_order(window):
    project = window.model.project
    window.proxy.set_sort_rules([(project.columns[1].id, False)])
    ascending = window.capture_view()
    descending = deepcopy(ascending)
    descending["sort"] = [(project.columns[1].id, True)]
    window.restore_view(descending)
    assert window.row_ids() == [project.rows[1].id, project.rows[0].id]
    window.restore_view(ascending)
    assert window.row_ids() == [r.id for r in project.rows]


def test_inline_columns_and_saved_views_round_trip(window):
    editor = window.desktop.table_properties
    assert editor.columns.model() is window.columns.model()
    index = editor.columns.model().index(1, 0)
    editor.columns.model().setData(index, Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
    assert window.table.isColumnHidden(1)
    editor.column_search.setText("Well")
    assert editor.columns.isRowHidden(1)
    before = deepcopy(window.model.project.rows)
    window.apply_filter([{"column": window.model.project.columns[0].id, "operator": "eq", "value": "B", "negate": False}], "AND")
    editor.view_name.setText("Test view")
    editor.save_view()
    saved = deepcopy(window.model.project.view["named"]["Test view"])
    editor.reset_view()
    assert window.proxy.rowCount() == 2 and not window.table.isColumnHidden(1)
    editor.load_view()
    assert window.proxy.rowCount() == 1 and window.table.isColumnHidden(1)
    assert window.proxy.rules == saved["rules"]
    assert window.model.project.rows == before
    window.set_project(Project())
    assert not editor.rules and editor.sort_columns.count() == 0
    assert editor.saved_views.count() == 0 and not editor.view_name.text()
    assert not editor.add_rule_button.isEnabled()


def test_visual_worksheets_use_tabs_and_properties_dock(window):
    desktop = window.desktop
    window.show_plots()
    plot = window.plot_window
    plot.timer.stop()
    assert not plot.isWindow()
    assert desktop.documents.currentWidget() is plot
    assert plot.whats_this_action.shortcut().isEmpty()
    assert window.whats_this_action.shortcut().toString() == "Shift+F1"
    assert desktop.properties_stack.currentWidget() is plot.tabs
    window.show_maps()
    mapping = window.map_window
    mapping.timer.stop()
    assert desktop.documents.count() == 3
    assert desktop.properties_stack.currentWidget() is mapping.tabs
    assert plot.tabs.parentWidget() is not desktop.properties_stack
    desktop.documents.setCurrentIndex(desktop.documents.indexOf(plot))
    assert desktop.properties_stack.currentWidget() is plot.tabs
    desktop.close_document(desktop.documents.indexOf(mapping))
    assert desktop.documents.count() == 2
    assert desktop.documents.currentWidget() is plot
    plot.reject()  # The dialog's own close action must also remove its tab.
    assert desktop.documents.count() == 1
    window.show_plots()
    plot.timer.stop()
    assert window.plot_window is plot
    assert desktop.documents.count() == 2


def test_project_replacement_removes_documents_without_stale_properties(window, qtbot):
    from shiboken6 import isValid

    window.show_plots()
    plot = window.plot_window
    plot.timer.stop()
    window.desktop.close_document(window.desktop.documents.indexOf(plot))
    assert window.plot_window is plot  # A closed tab remains cached for this project.
    window.show_maps()
    mapping = window.map_window
    mapping.timer.stop()
    window.set_project(Project())
    assert window.desktop.documents.count() == 1
    assert window.desktop.table_page.currentIndex() == 1
    assert window.desktop._borrowed is None
    assert window.plot_window is None and window.map_window is None
    qtbot.waitUntil(lambda: not isValid(plot) and not isValid(mapping))
    window.show_plots()
    assert window.plot_window is not plot
    assert window.desktop.documents.currentWidget() is window.plot_window
    window.plot_window.timer.stop()


@pytest.mark.parametrize("method, attribute, key", [
    ("interpolate", "interpolation_window", "interpolation"),
    ("show_calculations", "calculation_window", "calculation"),
    ("show_stratigraphy", "stratigraphy_window", "stratigraphy"),
    ("show_nonpyro", "nonpyro_window", "nonpyro"),
    ("show_qc", "qc_window", "qc"),
    ("show_statistics", "statistics_window", "statistics"),
    ("show_interpretation", "interpretation_window", "interpretation"),
])
def test_analysis_documents_use_docked_parameters(window, monkeypatch, method, attribute, key):
    monkeypatch.setattr(QDialog, "exec", lambda *args: pytest.fail("Analysis must be non-modal"))
    before = deepcopy(window.model.project)
    getattr(window, method)()
    document = getattr(window, attribute)
    if key in ("statistics", "interpretation"):
        document.timer.stop()  # This test verifies the layout without starting a calculation.
    desktop = window.desktop
    assert desktop.documents.currentWidget() is document
    assert not document.isWindow() and not document.isModal()
    assert desktop.properties_stack.currentWidget() is document.parameter_tabs
    assert document.whats_this_action.shortcut().isEmpty()
    if key == "nonpyro":
        field = document.form.fields["context"][0]
    elif key == "qc":
        field = document.comment
    elif key == "statistics":
        field = document.search
    elif key == "interpretation":
        field = document.x_unit
    else:
        field = document.context if key == "calculation" else document.sample_reference if key == "stratigraphy" else document.name
    default = field.text()
    field.setText("Private parameter draft")
    desktop.refresh_project()
    assert find_item(desktop.tree, (key, "")) is not None
    desktop.documents.setCurrentIndex(0)
    assert document.parameter_tabs.parentWidget() is not desktop.properties_stack
    desktop.tree.setCurrentItem(find_item(desktop.tree, (key, "")))
    assert desktop.properties_stack.currentWidget() is document.parameter_tabs
    assert field.text() == "Private parameter draft"
    document.reset_defaults_button.click()
    assert field.text() == default
    if key == "statistics":
        assert all(not document.properties.item(i).isHidden() for i in range(document.properties.count()))
    if key in ("statistics", "interpretation"):
        document.timer.stop()
    desktop.close_document(desktop.documents.indexOf(document))
    assert desktop.documents.count() == 1
    getattr(window, method)()
    if key in ("statistics", "interpretation"):
        document.timer.stop()
    assert getattr(window, attribute) is document
    assert desktop.properties_stack.currentWidget() is document.parameter_tabs
    assert window.model.project == before


def test_interpretation_json_draft_stays_in_docked_schema_page(window):
    window.show_interpretation()
    interpretation = window.interpretation_window
    interpretation.timer.stop()
    draft = '{"private_draft": true}'
    interpretation.editor.setPlainText(draft)
    interpretation.parameter_tabs.setCurrentIndex(1)
    interpretation.timer.stop()
    window.show_statistics()
    statistics = window.statistics_window
    statistics.timer.stop()
    assert window.desktop.properties_stack.currentWidget() is statistics.parameter_tabs
    window.desktop.documents.setCurrentWidget(interpretation)
    assert window.desktop.properties_stack.currentWidget() is interpretation.parameter_tabs
    assert interpretation.parameter_tabs.currentIndex() == 1
    assert interpretation.parameter_tabs.isAncestorOf(interpretation.editor)
    assert not interpretation.parameter_tabs.isAncestorOf(interpretation.table)
    interpretation.reset_defaults_button.click()
    interpretation.timer.stop()
    assert interpretation.editor.toPlainText() == draft
    assert interpretation.editor_dirty
    window.set_project(Project())
    assert window.statistics_window is None and window.interpretation_window is None
    assert window.desktop._borrowed is None


def test_embedded_interpolation_creation_stale_preview_and_undo(window):
    project = build_project(["Well", "X", "Y"], [["A", "0", "2"], ["A", "1", ""], ["A", "2", "6"]],
                            [2, 3, 4], [(0, "Well", "text", ""), (1, "X", "number", "m"), (2, "Y", "number", "")], {})
    window.set_project(project)
    window.interpolate()
    dialog = window.interpolation_window
    dialog.x.setCurrentIndex(dialog.x.findData(project.columns[1].id))
    dialog.y.setCurrentIndex(dialog.y.findData(project.columns[2].id))
    dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
    dialog.calculate()
    assert len(dialog.proposals) == 1
    window.model.setData(window.model.index(2, 2), "8")
    dialog.apply()
    assert window.model.columnCount() == 3
    dialog.calculate()
    dialog.apply()
    assert window.model.columnCount() == 4
    created = project.columns[-1]
    assert project.rows[1].values[created.id] == 5
    assert project.rows[1].values[project.columns[2].id] is None
    assert window.desktop.documents.currentWidget() is dialog
    dialog.apply()
    assert window.model.columnCount() == 4
    dialog.restore_parameters()
    assert not dialog.preview_button.isEnabled()
    dialog.load_saved_parameters()
    assert (dialog.x.currentData(), dialog.y.currentData()) == (project.columns[1].id, project.columns[2].id)
    assert dialog.preview_button.isEnabled()
    window.model.undo.undo()
    assert window.model.columnCount() == 3
    window.model.undo.redo()
    assert window.model.columnCount() == 4


def test_embedded_calculations_keep_document_and_macro_undo(window):
    project = build_project(["TOC", "S1", "S2", "S3"], [["2", ".5", "6", "1.2"]], [2],
                            [(0, "TOC", "number", "%wt"), (1, "S1", "number", "mg HC/g rock"),
                             (2, "S2", "number", "mg HC/g rock"), (3, "S3", "number", "mg CO2/g rock")], {})
    window.set_project(project)
    window.show_calculations()
    dialog = window.calculation_window
    dialog.context.setText("Test series")
    dialog.confirmed.setChecked(True)
    dialog.calculate()
    assert dialog.prepared
    window.apply_filter([], "AND")
    dialog.apply()
    assert window.model.columnCount() == 4
    dialog.calculate()
    count = len(dialog.prepared)
    dialog.apply()
    assert window.model.columnCount() == 4 + count
    hi = next(c for c in project.columns if c.name == "HI")
    assert project.rows[0].values[hi.id] == 300
    assert window.desktop.documents.currentWidget() is dialog
    dialog.apply()
    assert window.model.columnCount() == 4 + count
    window.model.undo.undo()
    assert window.model.columnCount() == 4
    window.model.undo.redo()
    assert window.model.columnCount() == 4 + count


def test_project_replacement_cleans_analysis_parameters_and_drafts(window):
    window.interpolate()
    window.show_calculations()
    window.show_stratigraphy()
    window.show_nonpyro()
    window.show_qc()
    window.stratigraphy_window.add_interval()
    window.set_project(Project())
    assert window.desktop.documents.count() == 1
    assert window.desktop._borrowed is None
    assert window.interpolation_window is None
    assert window.calculation_window is None
    assert window.stratigraphy_window is None
    assert window.nonpyro_window is None and window.qc_window is None


def test_embedded_stratigraphy_preview_commit_and_undo(window):
    project = build_project(["Well", "MD", "TOC"], [["A", "10", "1"], ["A", "25", "2"]], [2, 3],
                            [(0, "Well", "text", ""), (1, "MD", "number", "m"), (2, "TOC", "number", "%wt")], {})
    window.set_project(project)
    window.show_stratigraphy()
    dialog = window.stratigraphy_window
    dialog.reference.setText("MD RKB")
    dialog.sample_reference.setText("MD RKB")
    dialog.well.setCurrentIndex(1)
    dialog.depth.setCurrentIndex(2)
    dialog.add_interval(interval={"id": "Upper", "name": "Upper", "wellbore": "A", "level": "FORMATION",
                                  "top": 0, "bottom": 30, "color": "#c7ddeb"})
    dialog.preview()
    assert dialog.results.model().rowCount() == 2
    assert dialog.result_tabs.currentWidget() is dialog.results
    assert "stratigraphy" not in project.view
    dialog.add_columns()
    assert window.model.columnCount() == 5
    assert window.desktop.documents.currentWidget() is dialog
    window.model.undo.undo()
    assert window.model.columnCount() == 3 and "stratigraphy" not in project.view
    window.model.undo.redo()
    assert window.model.columnCount() == 5


def test_saved_plot_tree_navigation_keeps_source_data(window):
    project = window.model.project
    spec = PlotSpec(x_id=project.columns[1].id, y_id=project.columns[2].id, marker_size=25)
    project.view["plots"] = {"Named plot": spec.to_dict()}
    before = deepcopy(project.rows)
    window.desktop.refresh_project()
    item = find_item(window.desktop.tree, ("saved_plot", "Named plot"))
    window.desktop.tree.setCurrentItem(item)
    window.plot_window.timer.stop()
    assert window.plot_window.style.values()["marker_size"] == 25
    assert project.rows == before
    assert project.view["plots"]["Named plot"] == spec.to_dict()


def test_geochemistry_workspace_reopens_from_project_explorer(window):
    desktop = window.desktop
    window.actions["Паспорта и сравнение проб…"].trigger()
    document = window.geochemistry_workspace_window
    assert desktop.documents.currentWidget() is document
    desktop.refresh_project()

    payload = ("geochemistry_workspace", "")
    item = find_item(desktop.tree, payload)
    index = desktop.documents.indexOf(document)
    assert item.text(0) == desktop.documents.tabText(index)
    menu = desktop.create_project_menu(payload)
    close = next(action for action in menu.actions() if action.text() == "Закрыть")
    assert close.isEnabled()
    close.trigger()
    assert desktop.documents.indexOf(document) == -1

    desktop.tree.setCurrentItem(item)
    assert desktop.documents.currentWidget() is document
    assert desktop.active_payload(("project", "")) == payload


def test_open_document_updates_parameter_panel_once(window, monkeypatch):
    desktop = window.desktop
    restore = desktop.restore_properties
    updates = []

    def restore_properties():
        updates.append(True)
        return restore()

    monkeypatch.setattr(desktop, "restore_properties", restore_properties)
    window.show_plots()
    plot = window.plot_window
    plot.timer.stop()
    assert len(updates) == 1
    assert desktop.properties_stack.currentWidget() is plot.tabs
    updates.clear()
    window.show_plots()  # Refresh the already active document too.
    plot.timer.stop()
    assert len(updates) == 1
    assert desktop.properties_stack.currentWidget() is plot.tabs
    desktop.documents.setCurrentWidget(desktop.table_page)
    updates.clear()
    window.show_plots()
    plot.timer.stop()
    assert len(updates) == 1
    assert desktop.properties_stack.currentWidget() is plot.tabs


def test_layout_is_stored_outside_project(window, qtbot):
    before = deepcopy(window.model.project)
    window.desktop.properties_dock.hide()
    window.desktop.save_layout()
    assert window.model.project == before
    other = MainWindow()
    qtbot.addWidget(other)
    other.may_replace = lambda: True
    assert other.desktop.properties_dock.isHidden()
    other.desktop.default_layout()
    assert not other.desktop.properties_dock.isHidden()


def test_legacy_tabified_column_layout_is_not_restored(window, qtbot):
    # This layout predates moving column controls into table properties.
    # Native Windows Qt can crash on show() if it is restored after that move.
    window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, window.column_dock)
    window.tabifyDockWidget(window.desktop.project_dock, window.column_dock)
    window.desktop.properties_dock.hide()
    store = i18n.settings()
    store.setValue("workspace/state", window.saveState(1))
    other = MainWindow()
    qtbot.addWidget(other)
    other.may_replace = lambda: True
    assert not other.desktop.properties_dock.isHidden()
    assert other.column_dock.isHidden()
    assert other.desktop.properties_stack.currentWidget() is other.desktop.table_properties
    # A layout saved by this version must still restore normally.
    other.desktop.properties_dock.hide()
    other.desktop.save_layout()
    reopened = MainWindow()
    qtbot.addWidget(reopened)
    reopened.may_replace = lambda: True
    assert reopened.desktop.properties_dock.isHidden()


@pytest.mark.parametrize("theme", ["breeze_light", "breeze_dark"])
def test_official_breeze_palettes_and_icons(window, qapp, theme):
    apply_theme(qapp, theme)
    expected = breeze_palette(theme)
    for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Text, QPalette.ColorRole.Highlight):
        assert qapp.palette().color(role) == expected.color(role)
    assert qapp.property("autopyroStyleEngine") in ("Fusion", "breeze", "Breeze")
    assert not icon("document-open").isNull()
    assert (ASSETS / "COPYING-ICONS").is_file()


@pytest.mark.parametrize("theme", ["breeze_light", "breeze_dark"])
def test_selection_text_contrast_in_workspace(window, qapp, theme):
    apply_theme(qapp, theme)
    window.show_qc()

    def luminance(color):
        def linear(value):
            return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

        return (0.2126 * linear(color.redF()) + 0.7152 * linear(color.greenF())
                + 0.0722 * linear(color.blueF()))

    for widget in (window.desktop.tree, window.table, window.qc_window.table):
        palette = widget.palette()
        for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
            darker, lighter = sorted(luminance(palette.color(group, role)) for role in
                                     (QPalette.ColorRole.Highlight, QPalette.ColorRole.HighlightedText))
            assert (lighter + 0.05) / (darker + 0.05) >= 4.5


def test_theme_setting_applies_without_restart_or_project_changes(window, monkeypatch, qapp):
    from autopyro.ui.dialogs.settings import SettingsDialog

    before = deepcopy(window.model.project)
    def accept(dialog):
        dialog.theme.setCurrentIndex(dialog.theme.findData("breeze_dark"))
        dialog.accept()
        return dialog.result()
    monkeypatch.setattr(SettingsDialog, "exec", accept)
    window.show_settings()
    assert qapp.property("autopyroTheme") == "breeze_dark"
    assert settings.load_preferences().theme == "breeze_dark"
    assert window.model.project == before


def test_focus_mode_restores_visible_docks(window):
    window.show()
    docks = window.desktop.project_dock, window.desktop.properties_dock
    docks[1].hide()
    action = next(action for menu in window.menuBar().actions() for action in menu.menu().actions()
                  if action.shortcut().toString() == "Ctrl+Shift+F11")
    action.trigger()
    assert all(dock.isHidden() for dock in docks)
    action.trigger()
    assert not docks[0].isHidden() and docks[1].isHidden()
