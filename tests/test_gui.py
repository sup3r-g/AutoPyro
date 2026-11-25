import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog

from autopyro.dialogs import ImportDialog, InterpolationDialog
from autopyro.table import AddColumnCommand, FilterProxy, TableModel
from autopyro.window import MainWindow
from autopyro.core import Column, Project
from autopyro.io import build_project
from autopyro.ui.dialogs.batch_edit import BatchEditPreview


@pytest.fixture
def project():
    return build_project(["Well", "Depth", "S2"], [["B", "30", "6"], ["A", "20", "4"], ["A", "10", "2"]],
                         [2, 3, 4], [(0, "Well", "text", ""), (1, "Depth", "number", "m"),
                                      (2, "S2", "number", "mg HC/g rock")], {"type": "test"})


def test_filter_dialog_preserves_nested_groups_and_cancelled_edits(qtbot, project, monkeypatch):
    from copy import deepcopy
    from autopyro.ui.dialogs.filter import FilterDialog
    from autopyro.model.filtering import matches_rules

    well, depth, _ = project.columns
    rules = [{"combine": "OR", "negate": True, "rules": [
        {"column": well.id, "operator": "eq", "value": "B", "negate": False},
        {"combine": "AND", "negate": False, "rules": [
            {"column": depth.id, "operator": "ge", "value": 20, "negate": False},
            {"column": well.id, "operator": "eq", "value": "A", "negate": False},
        ]},
    ]}]
    original = deepcopy(rules)
    dialog = FilterDialog(project, rules, "AND")
    qtbot.addWidget(dialog)
    dialog.validate()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.rules == rules
    assert [row.id for row in project.rows if matches_rules(project, row, dialog.rules)] == [project.rows[2].id]
    group = dialog.table.cellWidget(0, 0)

    def cancel(editor):
        editor.table.item(0, 2).setText("changed")
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(FilterDialog, "exec", cancel)
    dialog.edit_group(group)
    dialog.validate()
    assert dialog.rules == original
    assert rules == original


def test_sort_edit_undo_preserves_identity(qtbot, project):
    model = TableModel(project)
    proxy = FilterProxy()
    proxy.setSourceModel(model)
    proxy.sort(1, Qt.SortOrder.AscendingOrder)
    target = proxy.mapToSource(proxy.index(0, 2))
    row_id = project.rows[2].id
    assert model.setData(target, "7,5")
    assert project.row_map()[row_id].values[project.columns[2].id] == 7.5
    assert project.rows[2].original[project.columns[2].id] == "2"
    model.undo.undo()
    assert project.rows[2].values[project.columns[2].id] == 2
    model.undo.redo()
    assert project.rows[2].values[project.columns[2].id] == 7.5


def test_filter_and_multisort(qtbot, project):
    model = TableModel(project)
    proxy = FilterProxy()
    proxy.setSourceModel(model)
    proxy.sort_rules = [(project.columns[0].id, False), (project.columns[1].id, True)]
    proxy.sort(0)
    assert [proxy.index(i, 1).data() for i in range(3)] == ["20", "10", "30"]
    proxy.set_rules([{"column": project.columns[0].id, "operator": "eq", "value": "A"},
                     {"column": project.columns[1].id, "operator": "ge", "value": 15}])
    assert proxy.rowCount() == 1
    assert proxy.index(0, 1).data() == "20"


def test_window_paste_transaction_and_restore_view(qtbot, project, tmp_path, monkeypatch):
    monkeypatch.setattr(BatchEditPreview, "exec", lambda self: QDialog.DialogCode.Accepted)
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.table.setCurrentIndex(window.proxy.index(0, 1))
    QApplication.clipboard().setText("11\t12\n21\t22")
    window.paste_cells()
    assert project.rows[0].values[project.columns[1].id] == 11
    assert project.rows[1].values[project.columns[2].id] == 22
    window.model.undo.undo()
    assert project.rows[0].values[project.columns[1].id] == 30
    window.columns.item(0).setCheckState(Qt.CheckState.Unchecked)
    window.apply_filter([{"column": project.columns[0].id, "operator": "eq", "value": "A"}], "AND")
    window.path = tmp_path / "test.apyro"
    assert window.save()
    window.set_project(Project.load(window.path), window.path)
    assert window.proxy.rowCount() == 2
    assert window.table.isColumnHidden(0)
    window.dirty = False


def test_import_dialog_background_preview(qtbot, tmp_path):
    path = tmp_path / "test.csv"
    path.write_text("Well;S2\n001;<0.1\n002;3.2\n", encoding="utf-8-sig")
    dialog = ImportDialog(path=path)
    qtbot.addWidget(dialog)
    dialog.load()
    qtbot.waitUntil(lambda: dialog.raw is not None and not dialog.worker.isRunning(), timeout=10000)
    assert dialog.mapping.cellWidget(0, 3).currentData() == "text"
    assert dialog.mapping.cellWidget(1, 3).currentData() == "text"
    dialog.import_data()
    qtbot.waitUntil(lambda: dialog.result() == QDialog.DialogCode.Accepted, timeout=10000)
    assert dialog.project.rows[0].values[dialog.project.columns[1].id] == "<0.1"


def test_add_interpolation_undo_and_staleness(qtbot, project):
    model = TableModel(project)
    column = Column("S2 interpolated", "number", metadata={"method": "linear", "stale": False})
    model.undo.push(AddColumnCommand(model, column, {project.rows[0].id: 7.0}))
    assert model.columnCount() == 4
    assert not model.flags(model.index(0, 3)) & Qt.ItemFlag.ItemIsEditable
    model.setData(model.index(0, 2), "8")
    assert column.metadata["stale"] is True
    model.undo.undo()
    model.undo.undo()
    assert model.columnCount() == 3
    model.undo.redo()
    assert model.columnCount() == 4


def test_interpolation_dialog_preview_chart(qtbot, project):
    project.rows[1].values[project.columns[2].id] = None
    dialog = InterpolationDialog(project, set(project.row_map()))
    qtbot.addWidget(dialog)
    dialog.x.setCurrentIndex(dialog.x.findData(project.columns[1].id))
    dialog.y.setCurrentIndex(dialog.y.findData(project.columns[2].id))
    # All rows need the same explicit group for this synthetic fixture.
    for row in project.rows:
        row.values[project.columns[0].id] = "A"
    dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
    dialog.calculate()
    assert len(dialog.proposals) == 1
    assert dialog.proposals[0].value == 4
    assert not dialog.chart_view.chart().series()
    y_id = project.columns[2].id
    project.rows[0].values[y_id] = 600
    dialog.preview_tabs.setCurrentWidget(dialog.chart_view)
    chart = dialog.chart_view.chart()
    measured, predicted = chart.series()
    assert [(point.x(), point.y()) for point in measured.points()] == [(30, 6), (10, 2)]
    assert [(point.x(), point.y()) for point in predicted.points()] == [(20, 4)]
    dialog.preview_tabs.setCurrentWidget(dialog.preview)
    dialog.preview_tabs.setCurrentWidget(dialog.chart_view)
    assert dialog.chart_view.chart() is chart
    project.rows[0].values[y_id] = 6
    dialog.apply()
    assert dialog.column.metadata["method"] == "linear"
    assert dialog.values[project.rows[1].id] == 4


def test_interpolation_invalid_column_name_returns_focus_and_announces_error(qtbot, project, monkeypatch):
    from autopyro.ui.i18n import tr

    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    assert window.model.setData(window.model.index(1, 2), "")
    assert window.model.setData(window.model.index(0, 0), "A")
    window.show()
    window.interpolate()
    window.activateWindow()
    dialog = window.interpolation_window
    dialog.x.setCurrentIndex(dialog.x.findData(project.columns[1].id))
    dialog.y.setCurrentIndex(dialog.y.findData(project.columns[2].id))
    dialog.calculate()
    assert dialog.preview_model.rowCount() == 1
    summary = dialog.summary.text()
    announcements = []
    monkeypatch.setattr(dialog, "announce_summary", lambda text=None: announcements.append(text))
    try:
        for name, error in (("  ", tr('Введите название новой колонки.')),
                            ("S2", tr('Колонка с таким названием уже есть.'))):
            dialog.name.setText(name)
            dialog.parameter_tabs.setCurrentIndex(1)
            dialog.apply()
            assert dialog.name_error.isVisible()
            assert dialog.name.accessibleDescription() == announcements[-1] == error
            assert dialog.summary.text() == summary
            qtbot.waitUntil(dialog.name.hasFocus)
            assert dialog.name.selectedText() == name
            assert len(project.columns) == 3
        dialog.name.setText("S2 interpolated")
        assert not dialog.name_error.isVisible() and not dialog.name.accessibleDescription()
        dialog.apply()
        assert project.columns[-1].name == "S2 interpolated"
    finally:
        window.dirty = False


def test_interpolation_keyboard_opens_samples_and_supports(qtbot, project, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    assert window.model.setData(window.model.index(1, 2), "")
    assert window.model.setData(window.model.index(0, 0), "A")
    window.show()
    window.activateWindow()
    window.interpolate()
    dialog = window.interpolation_window
    dialog.x.setCurrentIndex(dialog.x.findData(project.columns[1].id))
    y_id = project.columns[2].id
    dialog.y.setCurrentIndex(dialog.y.findData(y_id))
    opened, choices = [], []
    monkeypatch.setattr(window, "open_source_cell", lambda rid, cid, parent: opened.append((rid, cid)))
    monkeypatch.setattr(dialog, "open_sample_choices", lambda ids, title: choices.append(ids))
    try:
        dialog.calculate()
        assert dialog.preview_model.rowCount() == 1, dialog.summary.text()
        dialog.preview.setFocus()
        dialog.preview.setCurrentIndex(dialog.preview_model.index(0, 0))
        qtbot.keyClick(dialog.preview, Qt.Key.Key_Return)
        assert opened == [(project.rows[1].id, y_id)]
        dialog.preview.setCurrentIndex(dialog.preview_model.index(0, 3))
        qtbot.keyClick(dialog.preview, Qt.Key.Key_Enter)
        assert choices == [(project.rows[2].id, project.rows[0].id)]
        dialog.preview_tabs.setCurrentWidget(dialog.diagnostics.parentWidget())
        dialog.diagnostics.setFocus()
        dialog.diagnostics.setCurrentIndex(dialog.diagnostic_model.index(0, 0))
        row_id = dialog.diagnostic_model.rows[0][-1]
        qtbot.keyClick(dialog.diagnostics, Qt.Key.Key_Return)
        assert opened[-1] == (row_id, y_id)
        dialog.invalidate()
        qtbot.keyClick(dialog.diagnostics, Qt.Key.Key_Return)
        assert len(opened) == 2
        assert len(project.columns) == 3
    finally:
        window.dirty = False


def test_interpolation_chart_opens_visible_sample_and_resolves_overlap(qtbot, monkeypatch):
    from PySide6.QtCore import QPointF

    project = build_project(
        ["Well", "Depth", "S2"],
        [["A", "0", "2"], ["A", "5", ""], ["A", "10", "4"], ["B", "0", "2"], ["B", "10", "8"]],
        list(range(2, 7)), [(0, "Well", "text", ""), (1, "Depth", "number", "m"), (2, "S2", "number", "")], {},
    )
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.interpolate()
    dialog = window.interpolation_window
    dialog.x.setCurrentIndex(dialog.x.findData(project.columns[1].id))
    y_id = project.columns[2].id
    dialog.y.setCurrentIndex(dialog.y.findData(y_id))
    opened, choices = [], []
    monkeypatch.setattr(window, "open_source_cell", lambda rid, cid, parent: opened.append((rid, cid)))
    monkeypatch.setattr(dialog, "open_sample_choices", lambda ids, title: choices.append(ids))
    dialog.calculate()
    dialog.preview_tabs.setCurrentWidget(dialog.chart_view)
    measured, proposed = dialog.chart_view.chart().series()
    proposed.clicked.emit(QPointF(5, 3))
    assert opened == [(project.rows[1].id, y_id)]
    measured.clicked.emit(QPointF(0, 2))
    assert choices == [[project.rows[0].id, project.rows[3].id]]
    dialog.chart_well.setCurrentIndex(dialog.chart_well.findData("well:B"))
    filtered = dialog.chart_view.chart().series()[0]
    filtered.clicked.emit(QPointF(0, 2))
    assert opened[-1] == (project.rows[3].id, y_id)
    dialog.invalidate()
    filtered.clicked.emit(QPointF(0, 2))
    assert len(opened) == 2
    window.dirty = False


@pytest.mark.parametrize("split_by", ["formation", "laboratory"])
def test_interpolation_proposals_show_same_groups_as_chart(qtbot, split_by):
    from autopyro.ui.i18n import tr

    project = build_project(
        ["Well", "Formation", "Depth", "S2"],
        [["A", group, str(x), y] for group, endpoint in (("F1", "4"), ("F2", "8"))
         for x, y in ((0, "2"), (5, ""), (10, endpoint))],
        list(range(2, 8)), [(0, "Well", "text", ""), (1, "Formation", "text", ""),
                            (2, "Depth", "number", "m"), (3, "S2", "number", "")], {},
    )
    dialog = InterpolationDialog(project, set(project.row_map()))
    qtbot.addWidget(dialog)
    dialog.x.setCurrentIndex(dialog.x.findData(project.columns[2].id))
    dialog.y.setCurrentIndex(dialog.y.findData(project.columns[3].id))
    if split_by == "formation":
        dialog.groups.item(1).setCheckState(Qt.CheckState.Checked)
        expected = ("F1", "F2")
    else:
        project.sync_measurements()
        for row in project.rows[3:]:
            measurement = project.science.current(row.id, project.columns[3].id)
            project.science.analyses[measurement.analysis_id].laboratory = "Lab B"
        expected = (f"{tr('Лаборатория')}: {tr('не указано')}", f"{tr('Лаборатория')}: Lab B")
    dialog.calculate()

    model = dialog.preview_model
    assert model.rowCount() == 2
    assert [model.index(i, 4).data() for i in range(2)] == ["A", "A"]
    for i, group_detail in enumerate(expected):
        cell = model.index(i, 5)
        label = dialog.chart_well.itemText(dialog.chart_well.findData(f"group:{i + 1}"))
        assert group_detail in label
        assert cell.data() == cell.data(Qt.ItemDataRole.ToolTipRole) == label


def test_interpolation_mixed_units_require_separate_output_columns(qtbot, monkeypatch):
    from PySide6.QtGui import QAccessible
    from PySide6.QtWidgets import QTableView
    from autopyro.ui.i18n import tr

    project = build_project(
        ["Well", "Depth", "Y"],
        [[well, str(x), y] for well, endpoints in (("A", ("1", "3")), ("B", ("3000", "5000")))
         for x, y in zip((0, 5, 10), (endpoints[0], "", endpoints[1]))],
        list(range(2, 8)), [(0, "Well", "text", ""), (1, "Depth", "number", "m"),
                            (2, "Y", "number", "mg/g")], {},
    )
    well, x, y = project.columns
    project.sync_measurements()
    for row in project.rows[3:]:
        measurement = project.science.current(row.id, y.id)
        measurement.unit, measurement.basis = "ug/g", "dry"

    def preview(rows):
        dialog = InterpolationDialog(project, {row.id for row in rows})
        qtbot.addWidget(dialog)
        for combo, column in ((dialog.x, x), (dialog.y, y), (dialog.well, well)):
            combo.setCurrentIndex(combo.findData(column.id))
        dialog.name.setText("Predicted Y")
        dialog.calculate()
        return dialog

    dialog = preview(project.rows)
    assert [p.value for p in dialog.proposals] == [2, 4000]
    dialog.apply()
    assert dialog.result() != QDialog.DialogCode.Accepted
    dialog.preview_tabs.setCurrentWidget(dialog.chart_view)
    assert not dialog.chart_view.chart().series()
    dialog.chart_well.setCurrentIndex(dialog.chart_well.findData("well:B"))
    assert "ug/g" in dialog.chart_view.chart().title()

    def inspect_data(modal):
        table = modal.findChild(QTableView)
        model = table.model()
        accessible = QAccessible.queryAccessibleInterface(table)
        assert accessible.text(QAccessible.Text.Name) == tr('Данные графика…')
        assert model.rowCount() == 3
        assert [model.index(i, 6).data() for i in range(3)] == ["ug/g"] * 3
        assert model.index(2, 4).data() == "4000.0"

    monkeypatch.setattr(QDialog, "exec", inspect_data)
    dialog.show_chart_data()
    dialog.x.setCurrentIndex(dialog.x.findData(None))
    title = dialog.chart_view.chart().title()
    assert "ug/g" in title and tr('Показан предыдущий предпросмотр.') in title
    single = preview(project.rows[3:])
    single.apply()
    assert single.result() == QDialog.DialogCode.Accepted
    assert (single.column.unit, single.column.metadata["basis"]) == ("ug/g", "dry")


def test_interpolation_loading_invalid_parameters_preserves_form(qtbot, project):
    from autopyro.ui.i18n import tr

    _, x, y = project.columns
    saved = Column("Saved", "number", metadata={
        "x_id": x.id, "y_id": y.id, "method": "nearest", "max_gap": None,
        "interpolation_protocol": {"old": True},
    })
    project.columns.append(saved)
    dialog = InterpolationDialog(project, set(project.row_map()))
    qtbot.addWidget(dialog)
    dialog.saved_runs.setCurrentIndex(dialog.saved_runs.findData(saved.id))
    before = dialog.x.currentData(), dialog.y.currentData(), dialog.method.currentData(), dialog.gap.value()
    dialog.load_saved_parameters()
    assert (dialog.x.currentData(), dialog.y.currentData(), dialog.method.currentData(), dialog.gap.value()) == before
    assert dialog.summary.text() == tr('Сохранённые параметры повреждены или выходят за допустимые пределы.')


def test_interpolation_rejects_axes_as_grouping_or_well_columns(qtbot, project):
    from autopyro.ui.i18n import tr

    dialog = InterpolationDialog(project, set(project.row_map()))
    qtbot.addWidget(dialog)
    well, x, y = project.columns
    dialog.x.setCurrentIndex(dialog.x.findData(x.id))
    dialog.y.setCurrentIndex(dialog.y.findData(y.id))
    assert dialog.preview_button.isEnabled()
    for index in (1, 2):
        item = dialog.groups.item(index)
        item.setCheckState(Qt.CheckState.Checked)
        assert not dialog.preview_button.isEnabled()
        assert dialog.summary.text() == tr('Исключите колонки X и Y из группировки.')
        item.setCheckState(Qt.CheckState.Unchecked)
    dialog.well.setCurrentIndex(dialog.well.findData(x.id))
    assert not dialog.preview_button.isEnabled()
    assert dialog.summary.text() == tr('Для ствола выберите колонку, отличную от X и Y.')
    dialog.well.setCurrentIndex(dialog.well.findData(well.id))
    assert dialog.preview_button.isEnabled()


def test_interpolation_rejects_axes_as_bottom_column_before_calculation(qtbot, project):
    from autopyro.ui.i18n import tr

    _, x, y = project.columns
    bottom = Column("Bottom", "number", unit="m")
    project.columns.append(bottom)
    for row in project.rows:
        row.values[bottom.id] = None
    dialog = InterpolationDialog(project, set(project.row_map()))
    qtbot.addWidget(dialog)
    dialog.x.setCurrentIndex(dialog.x.findData(x.id))
    dialog.y.setCurrentIndex(dialog.y.findData(y.id))

    for axis in (x, y):
        dialog.interval_bottom.setCurrentIndex(dialog.interval_bottom.findData(axis.id))
        assert not dialog.preview_button.isEnabled()
        assert dialog.summary.text() == tr('Выберите отдельную числовую колонку подошвы пробы')
        dialog.calculate()
        assert dialog.protocol is None
    dialog.interval_bottom.setCurrentIndex(dialog.interval_bottom.findData(bottom.id))
    assert dialog.preview_button.isEnabled()


def test_interpolation_requires_bound_bottom_for_interval_mode(qtbot, project, monkeypatch):
    from autopyro.ui.dialogs import interpolation_calculation as interpolation
    from autopyro.ui.i18n import tr

    well, depth, y = project.columns
    bottom, other = Column("Bottom", "number"), Column("Other", "number")
    project.columns.extend((bottom, other))
    for row in project.rows:
        row.values.update({bottom.id: None, other.id: None})
    project.view["stratigraphy"] = {"binding": {
        "well_id": well.id, "depth_id": depth.id, "bottom_id": bottom.id, "sample_mode": "interval",
    }}
    dialog = InterpolationDialog(project, set(project.row_map()))
    qtbot.addWidget(dialog)
    dialog.x.setCurrentIndex(dialog.x.findData(depth.id))
    dialog.y.setCurrentIndex(dialog.y.findData(y.id))
    with monkeypatch.context() as patch:
        patch.setattr(interpolation, "prepare", lambda *args, **kwargs: pytest.fail("Started with invalid bottom"))
        for choice in (None, other.id):
            dialog.interval_bottom.setCurrentIndex(dialog.interval_bottom.findData(choice))
            assert not dialog.preview_button.isEnabled()
            assert dialog.summary.text() == tr('Выберите колонку подошвы из привязки стратиграфии')
            dialog.calculate()
    dialog.interval_bottom.setCurrentIndex(dialog.interval_bottom.findData(bottom.id))
    assert dialog.preview_button.isEnabled()
    dialog.calculate()
    assert dialog.protocol is not None


def test_interpolation_requires_explicit_axes_and_clears_removed_column(qtbot, project):
    from autopyro.model.core import Column
    from autopyro.ui.window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.interpolate()
    dialog = window.interpolation_window
    assert dialog.x.currentData() is None and dialog.y.currentData() is None
    assert not dialog.preview_button.isEnabled()
    from autopyro.ui.table import AddColumnCommand

    extra = Column("Other number", "number")
    window.model.undo.push(AddColumnCommand(window.model, extra, {}))
    dialog.x.setCurrentIndex(dialog.x.findData(extra.id))
    assert not dialog.preview_button.isEnabled()
    y_id = project.columns[2].id
    dialog.y.setCurrentIndex(dialog.y.findData(y_id))
    assert dialog.preview_button.isEnabled()
    window.model.undo.undo()
    assert dialog.x.currentData() is None
    assert dialog.y.currentData() == y_id
    assert not dialog.preview_button.isEnabled()
    dialog.restore_parameters()
    assert dialog.y.currentData() is None
    window.dirty = False


@pytest.mark.parametrize("without_well", [False, True])
def test_interpolation_tab_follows_basic_form_order(qtbot, project, without_well):
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.show()
    window.interpolate()
    window.activateWindow()
    dialog = window.interpolation_window
    controls = [dialog.scope, dialog.x, dialog.y, dialog.well]
    if without_well:
        dialog.well.setCurrentIndex(dialog.well.findData(None))
        controls.append(dialog.without_well)
    controls.extend((dialog.method, dialog.gap, dialog.name, dialog.groups))
    dialog.scope.setFocus()
    qtbot.waitUntil(dialog.scope.hasFocus)

    visited = [dialog.scope]
    for _ in range(2 * len(controls)):
        qtbot.keyClick(QApplication.focusWidget(), Qt.Key.Key_Tab)
        focused = QApplication.focusWidget()
        if focused in controls:
            visited.append(focused)
            if focused is dialog.groups:
                break
    assert visited == controls
    visited = [dialog.groups]
    for _ in range(2 * len(controls)):
        # Send the reverse navigation key without leaving a synthetic Shift pressed.
        qtbot.keyClick(QApplication.focusWidget(), Qt.Key.Key_Backtab)
        focused = QApplication.focusWidget()
        if focused in controls:
            visited.append(focused)
            if focused is dialog.scope:
                break
    assert visited == controls[::-1]
    assert QApplication.keyboardModifiers() == Qt.KeyboardModifier.NoModifier


def test_interpolation_workspace_fits_compact_width_with_both_docks(qtbot, project):
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.show()
    window.desktop.project_dock.show()
    window.desktop.properties_dock.show()
    QApplication.processEvents()
    window.resize(960, 540)
    QApplication.processEvents()
    assert window.width() == 960
    window.interpolate()
    QApplication.processEvents()

    dialog = window.interpolation_window
    assert window.minimumSizeHint().width() <= 960
    assert window.width() == 960
    assert dialog.preview_button.isVisible()
    assert dialog.preview_button.geometry().right() < dialog.preview_button.parentWidget().width()


def test_interpolation_long_column_names_fit_compact_workspace(qtbot, project):
    for column in project.columns:
        column.name = column.name + " " + "very long laboratory property name " * 12
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.show()
    window.resize(960, 540)
    window.interpolate()
    QApplication.processEvents()
    dialog = window.interpolation_window
    assert window.minimumSizeHint().width() <= 960
    for combo in (dialog.x, dialog.y, dialog.well, dialog.interval_bottom):
        assert combo.minimumSizeHint().width() < 300
        for i in range(combo.count()):
            cid = combo.itemData(i)
            if cid is not None:
                column = next(column for column in project.columns if column.id == cid)
                suffix = f" [{column.unit}]" if column.unit and combo is not dialog.well else ""
                assert combo.itemText(i) == column.name + suffix
    assert dialog.preview_button.isVisible()
    window.dirty = False


def test_interpolation_basic_fields_scroll_in_compact_layout(qtbot, project):
    from PySide6.QtWidgets import QScrollArea

    dialog = InterpolationDialog(project, set(project.row_map()))
    qtbot.addWidget(dialog)
    basic = dialog.parameter_tabs.widget(0)
    advanced = dialog.parameter_tabs.widget(1)
    assert dialog.parameter_tabs.currentWidget() is basic
    assert isinstance(basic, QScrollArea)
    assert isinstance(advanced, QScrollArea)
    for control in (dialog.scope, dialog.x, dialog.y, dialog.method, dialog.gap, dialog.groups, dialog.name):
        assert basic.isAncestorOf(control)
    assert not dialog.parameter_tabs.isAncestorOf(dialog.preview_button)
    dialog.resize(860, 400)
    dialog.show()
    qtbot.waitUntil(lambda: basic.verticalScrollBar().maximum() > 0)
    basic.ensureWidgetVisible(dialog.groups)
    assert basic.verticalScrollBar().value() > 0
    assert dialog.preview_button.isVisible()

    dialog.parameter_tabs.setCurrentWidget(advanced)
    dialog.min_anchors.setValue(7)
    dialog.parameter_tabs.setCurrentWidget(basic)
    dialog.parameter_tabs.setCurrentWidget(advanced)
    assert dialog.min_anchors.value() == 7


def test_interpolation_exposes_accessibility_names_and_announces_status(qtbot, project, monkeypatch):
    from types import SimpleNamespace
    from PySide6.QtGui import QAccessible
    from autopyro.ui.dialogs import interpolation
    from autopyro.ui.i18n import tr

    project.rows[0].values[project.columns[0].id] = "A"
    project.rows[1].values[project.columns[2].id] = None
    dialog = InterpolationDialog(project, set(project.row_map()))
    qtbot.addWidget(dialog)
    for widget, name in ((dialog.preview, tr('Предлагаемые значения')), (dialog.diagnostics, tr('Ограничения')),
                         (dialog.groups, tr('Группировка')), (dialog.saved_runs, tr('Сохранённые расчёты'))):
        interface = QAccessible.queryAccessibleInterface(widget)
        assert interface.text(QAccessible.Text.Name) == name
    announcements = []
    monkeypatch.setattr(interpolation, "QAccessible", SimpleNamespace(
        isActive=lambda: True, updateAccessibility=lambda event: announcements.append(event.message()),
    ))
    dialog.x.setCurrentIndex(dialog.x.findData(project.columns[1].id))
    dialog.y.setCurrentIndex(dialog.y.findData(project.columns[2].id))
    dialog.calculate()
    assert dialog.preview_model.rowCount() == 1
    assert announcements == [dialog.summary.text()]
    dialog.gap.setValue(50)
    assert announcements[-1] == dialog.summary.text()
    assert len(announcements) == 2
    description = QAccessible.Text.Description
    stale = tr('Предыдущие результаты доступны только для просмотра.')
    assert QAccessible.queryAccessibleInterface(dialog.preview).text(description) == stale
    assert QAccessible.queryAccessibleInterface(dialog.diagnostics).text(description) == stale
    assert stale in QAccessible.queryAccessibleInterface(dialog.chart_view).text(description)
    dialog.calculate()
    assert QAccessible.queryAccessibleInterface(dialog.preview).text(description) == ""
    assert QAccessible.queryAccessibleInterface(dialog.diagnostics).text(description) == ""
    assert stale not in QAccessible.queryAccessibleInterface(dialog.chart_view).text(description)


def test_large_preview_chart_preserves_extrema_and_counts(qtbot, project, monkeypatch):
    from autopyro.analysis.interpolation import Proposal
    from autopyro.ui.dialogs import interpolation_charts as interpolation
    from autopyro.ui.i18n import tr

    monkeypatch.setattr(interpolation, "CHART_SERIES_LIMIT", 6)
    measured = [(i, y) for i, y in enumerate([0, 0, 100, -100, 0, 0, 70, -80, 0, 0])]
    proposals = [Proposal(str(x), x, y, "left", "right") for x, y in measured]
    dialog = InterpolationDialog(project, set())
    qtbot.addWidget(dialog)
    dialog.preview_model.set_rows(proposals)
    anchors = [("A", x, y) for x, y in reversed(measured)]
    labels = {proposal.row_id: (proposal.row_id, "A") for proposal in proposals}
    dialog._pending_chart = ("Depth", "S2"), anchors, [], proposals, labels, {}
    dialog._chart_dirty = True
    dialog.preview_tabs.setCurrentWidget(dialog.chart_view)
    expected = {(0, 0), (2, 100), (3, -100), (6, 70), (7, -80), (9, 0)}
    for series, label in zip(dialog.chart_view.chart().series(),
                             (tr('Опорные измерения'), tr('Предлагаемые значения')), strict=True):
        assert {(point.x(), point.y()) for point in series.points()} == expected
        assert series.name() == tr('{0} · {1:,} из {2:,}', label, 6, 10)
    assert dialog.preview_model.rowCount() == 10
