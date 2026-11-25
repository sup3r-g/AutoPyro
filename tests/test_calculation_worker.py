"""Large pyrolysis previews run outside Qt and reject changed inputs/scope."""
from copy import deepcopy
from threading import Event

import pytest
from PySide6.QtCore import QThread, QTimer, Qt
from PySide6.QtWidgets import QDialogButtonBox, QFileDialog

from autopyro.io import build_project
from autopyro.ui.dialogs import calculation as gui
from autopyro.ui.dialogs import calculation_comparison, calculation_workflow
from autopyro.ui.table.model import context_issue_text, range_issue_text
from autopyro.ui.tasks import TaskCancelled
from autopyro.ui.window import MainWindow


@pytest.fixture
def owner(qtbot, monkeypatch):
    project = build_project(["TOC", "S2"], [["2", "6"], ["2", "ND"]], [2, 3],
                            [(0, "TOC", "number", "%wt"), (1, "S2", "number", "mg HC/g rock")], {})
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.may_replace = lambda: True
    window.show_calculations()
    dialog = window.calculation_window
    dialog.context.setText("Synthetic whole rock")
    dialog.confirmed.setChecked(True)
    for i in range(dialog.outputs.rowCount()):
        dialog.outputs.cellWidget(i, 0).setChecked(i == 0)
    monkeypatch.setattr(calculation_workflow, "large_project", lambda _: True)
    monkeypatch.setattr(calculation_comparison, "large_project", lambda _: True)
    yield window
    window.dirty = False
    window.close()


def test_large_preview_is_isolated_and_apply_undo_stay_in_model(owner, monkeypatch):
    threads = []
    original = calculation_workflow.make_calculated_column

    def tracked(*args, **kwargs):
        threads.append(QThread.currentThread())
        return original(*args, **kwargs)

    monkeypatch.setattr(calculation_workflow, "make_calculated_column", tracked)
    project = owner.model.project
    before = deepcopy(project)
    dialog = owner.calculation_window
    dialog.calculate()
    assert project == before
    assert len(threads) == 1 and threads[0] != QThread.currentThread()
    assert dialog.prepared[0][1][project.rows[0].id] == 300
    assert dialog.prepared[0][1][project.rows[1].id] is None
    dialog.apply()
    hi = project.columns[-1]
    assert project.rows[0].values[hi.id] == 300
    assert project.science == before.science
    owner.model.undo.undo()
    assert len(project.columns) == 2
    owner.model.undo.redo()
    assert project.rows[0].values[hi.id] == 300


def test_range_only_preview_can_create_a_column_without_numeric_values(owner):
    project = owner.model.project
    for i in range(len(project.rows)):
        owner.model.setData(owner.model.index(i, 1), "<0.2")
    dialog = owner.calculation_window
    dialog.show_censored_ranges.setChecked(True)
    dialog.calculate()
    assert dialog.preview.item(0, 1).text() == "[0, 10)"
    assert dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    dialog.apply()
    hi = project.columns[-1]
    assert all(row.values[hi.id] is None for row in project.rows)
    index = owner.model.index(0, len(project.columns) - 1)
    assert "[0, 10)" in index.data(Qt.ItemDataRole.ToolTipRole)
    owner.model.undo.undo()
    assert len(project.columns) == 2
    owner.model.undo.redo()
    assert "[0, 10)" in index.data(Qt.ItemDataRole.ToolTipRole)


def test_uncertainty_preview_and_working_measurement_change(owner):
    from autopyro.ui.table import SelectMeasurementCommand

    project = owner.model.project
    row = project.rows[0]
    toc, s2 = project.columns
    for column, amount in ((toc, .1), (s2, .2)):
        measurement = project.science.current(row.id, column.id)
        measurement.uncertainty, measurement.uncertainty_type = amount, "standard"
    dialog = owner.calculation_window
    dialog.propagate_uncertainty.setChecked(True)
    dialog.calculate()
    assert dialog.preview.item(0, 1).text() == "300 ± 18.0278"
    assert "TOC: 15" in dialog.preview.item(0, 1).toolTip()
    dialog.apply()
    hi = project.columns[-1]
    assert row.values[hi.id] == 300
    assert hi.metadata["calculation"]["propagate_uncertainty"]

    repeat = project.science.add_result(row.id, toc.id, "2")
    owner.model.undo.push(SelectMeasurementCommand(owner.model, row.id, toc.id, repeat.id))
    assert hi.metadata["diagnostics"][row.id]["uncertainty"]["inputs"] == {"TOC": "missing"}
    owner.model.undo.undo()
    assert hi.metadata["diagnostics"][row.id]["uncertainty"]["status"] == "ok"


def test_lab_comparison_checks_full_scope_and_keeps_lab_columns(owner, monkeypatch, tmp_path):
    import csv
    rows = [["2", "6", "2", "300", "25"]] * 200 + [
        ["2", "6", "2", "ND", "20"],
        ["2", "6", "2", "280", "25"],
        ["2", "ND", "2", "300", "25"],
    ]
    project = build_project(
        ["TOC", "S2", "S1", "HI lab", "PI lab"], rows, list(range(2, 2 + len(rows))),
        [(0, "TOC", "number", "%wt"), (1, "S2", "number", "mg HC/g rock"),
         (2, "S1", "number", "mg HC/g rock"), (3, "HI lab", "number", "mg HC/g TOC"),
         (4, "PI lab", "number", "%")], {},
    )
    owner.set_project(project)
    owner.show_calculations()
    dialog = owner.calculation_window
    dialog.context.setText("Synthetic whole rock")
    dialog.confirmed.setChecked(True)
    for i in range(dialog.outputs.rowCount()):
        dialog.outputs.cellWidget(i, 0).setChecked(False)
    for formula, source in (("HI", project.columns[3]), ("PI", project.columns[4])):
        combo, unit, tolerance = dialog.lab_inputs[formula]
        combo.setCurrentIndex(combo.findData(source.id))
        tolerance.setValue(1 if formula == "HI" else 0.05)
    assert dialog.lab_inputs["PI"][1].currentData() == "%"

    dialog.calculate()
    assert dialog.preview.rowCount() == 200
    assert float(dialog.preview.item(0, 6).text()) == 0.25
    assert not dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    assert dialog._comparison_rows["HI"].loc[project.rows[201].id, "difference"] == 20
    dialog.next_page.click()
    assert dialog.preview.rowCount() == 3
    dialog.sample_search.setText(project.rows[201].id)
    assert dialog.preview.rowCount() == 1
    assert dialog.preview.item(0, 0).text() == project.rows[201].id
    dialog.sample_search.clear()
    path = tmp_path / "comparison.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    dialog.export_button.click()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 406
    assert float(exported[201]["difference"]) == 20
    assert float(exported[201]["absolute_tolerance"]) == 1
    dialog.mismatches_only.setChecked(True)
    assert dialog.preview.rowCount() == 1
    assert dialog.preview.item(0, 0).text() == project.rows[201].id
    assert dialog._comparison_rows["PI"].loc[project.rows[201].id, "status"] == "match"
    assert not dialog.prepared
    dialog.unavailable_only.setChecked(True)
    assert dialog.preview.rowCount() == 3
    assert not dialog.filter_hint.isHidden()
    dialog.unavailable_only.setChecked(False)

    dialog.lab_inputs["PI"][2].setValue(0.01)
    assert not dialog.mismatches_only.isEnabled()
    pi_unit = dialog.lab_inputs["PI"][1]
    pi_unit.setCurrentIndex(pi_unit.findData("1"))
    dialog.calculate()
    assert set(dialog._comparison_rows) == {"HI"}
    assert dialog.mismatches_only.isEnabled()
    pi_unit.setCurrentIndex(pi_unit.findData("%"))
    dialog.outputs.cellWidget(0, 0).setChecked(True)
    dialog.calculate()
    assert dialog.preview.rowCount() == 2
    assert "ND" in dialog.preview.item(0, 4).text()
    opened = []
    monkeypatch.setattr(owner.desktop, "open_table", lambda: opened.append(True))
    dialog.jump_to_sample(1, 0)
    assert opened == [True]
    selected = owner.table.selectionModel().selectedRows()[0]
    assert owner.proxy.mapToSource(selected).row() == 201
    dialog.apply()
    assert len(project.columns) == 6
    assert project.rows[201].values[project.columns[3].id] == 280
    assert project.rows[201].values[project.columns[-1].id] == 300


def test_save_comparison_roundtrips_and_opens_historical_results(owner, tmp_path, monkeypatch):
    from autopyro.model.audit import read_protocol
    from autopyro.model.core import Project
    from PySide6.QtWidgets import QDialog, QTableView

    project = build_project(
        ["TOC", "S2", "HI lab"], [["2", "6", "280"], ["2", "ND", "300"], ["2", "6", "300"],
                                 ["<LOD", "ND", "300"]],
        [2, 3, 4, 5], [(0, "TOC", "number", "%wt"), (1, "S2", "number", "mg HC/g rock"),
                 (2, "HI lab", "number", "mg HC/g TOC")], {},
    )
    owner.set_project(project)
    row = project.rows[2]
    toc, _, lab = project.columns
    source = project.science.current(row.id, toc.id)
    project.science.analyses[source.analysis_id].method = "Method A"
    repeat = project.science.add_result(row.id, lab.id, 300, analysis_fields={"method": "Method B"})
    project.select_measurement(row.id, lab.id, repeat.id)
    owner.show_calculations()
    dialog = owner.calculation_window
    dialog.context.setText("Synthetic whole rock")
    dialog.confirmed.setChecked(True)
    dialog.outputs.cellWidget(0, 0).setChecked(False)
    combo, _, tolerance = dialog.lab_inputs["HI"]
    combo.setCurrentIndex(combo.findData(project.columns[2].id))
    tolerance.setValue(1)
    dialog.show_censored_ranges.setChecked(True)
    dialog.calculate()
    conflict = context_issue_text(["method"])
    assert conflict in dialog.preview.item(2, 4).text()
    incomplete = gui.tr('Метод или лаборатория указаны не для всех измерений')
    assert incomplete in dialog.preview.item(0, 4).text()
    assert dialog.save_comparison_button.isEnabled() and not dialog.prepared
    dialog.save_comparison_button.click()
    entry, = project.view[gui.COMPARISON_KEY].values()
    original = read_protocol(entry["protocol"])
    assert [row[6] for row in original["rows"]] == [
        "mismatch", "calculation_unavailable", "context_conflict", "calculation_unavailable",
    ]
    assert original["rows"][0][14] == "yes"
    assert original["rows"][2][13:15] == ["method", ""]
    assert original["settings"]["HI"]["laboratory"]["tolerance"] == 1
    import json
    diagnostic = json.loads(original["rows"][3][15])
    assert diagnostic["range_issues"] == [
        {"range_issue": "missing_lod", "role": "TOC"},
        {"range_issue": "non_detect", "role": "S2"},
    ]
    assert len(project.columns) == 3

    owner.model.undo.undo()
    assert gui.COMPARISON_KEY not in project.view
    owner.model.undo.redo()
    assert project.view[gui.COMPARISON_KEY]
    assert owner.model.setData(owner.model.index(0, 1), "8")
    path = tmp_path / "comparison.apyro"
    project.save(path)
    owner.set_project(Project.load(path))
    owner.show_calculations()
    dialog = owner.calculation_window

    opened = []
    def inspect(viewer):
        model = viewer.findChild(QTableView).model()
        assert conflict in model.index(2, 6).data(Qt.ItemDataRole.ToolTipRole)
        assert incomplete in model.index(0, 6).data(Qt.ItemDataRole.DisplayRole)
        assert model.index(1, 6).data() == gui.tr('Сравнение недоступно: {0}', "S2: ND")
        assert model.index(1, 7).data() == "S2: ND"
        assert model.index(3, 7).data() == range_issue_text(diagnostic)
        assert range_issue_text(diagnostic) in model.index(3, 6).data()
        opened.extend(model.rows)
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", inspect)
    assert dialog.saved_comparisons_button.isEnabled()
    dialog.saved_comparisons_button.click()
    assert opened == original["rows"]
    assert opened[0][3:6] == [300, 280, 20]


@pytest.mark.parametrize("length", [13, 14, 15])
def test_saved_comparison_explains_unavailable_calculation_in_older_snapshots(length):
    row = ["sample", "Sample A", "HI", None, 300, None, "calculation_unavailable",
           "zero_denominator", "lab", "HI lab", "mg HC/g TOC", "mg HC/g TOC", 1, "", ""][:length]
    model = gui.SavedComparisonModel([""] * 13, [row])
    reason = gui.tr('нулевой знаменатель')
    assert model.index(0, 6).data() == gui.tr('Сравнение недоступно: {0}', reason)
    assert model.index(0, 6).data(Qt.ItemDataRole.ToolTipRole) == model.index(0, 6).data()
    assert model.index(0, 7).data() == reason


def test_preview_announces_completion_and_invalidation_to_accessibility_clients(owner, monkeypatch):
    from types import SimpleNamespace
    from PySide6.QtGui import QAccessible

    events = []
    monkeypatch.setattr(gui, "QAccessible", SimpleNamespace(
        isActive=lambda: True,
        updateAccessibility=lambda event: events.append((event.object(), event.message())),
    ))
    dialog = owner.calculation_window
    dialog.calculate()
    assert events == [(dialog.summary, dialog.summary.text())]
    table = QAccessible.queryAccessibleInterface(dialog.preview)
    assert table.text(QAccessible.Text.Description) == dialog.page_label.text()

    dialog.calculate()
    assert events == [(dialog.summary, dialog.summary.text())] * 2
    dialog.context.setText("Changed")
    assert events[-1] == (dialog.summary, dialog.summary.text())
    assert len(events) == 3
    assert table.text(QAccessible.Text.Description) == dialog.summary.text()


def test_keyboard_pagination_keeps_focus_on_pager_and_announces_each_page(owner, qtbot, monkeypatch):
    from types import SimpleNamespace

    project = build_project(
        ["TOC", "S2"], [["2", "6"]] * 201, list(range(2, 203)),
        [(0, "TOC", "number", "%wt"), (1, "S2", "number", "mg HC/g rock")], {},
    )
    owner.set_project(project)
    monkeypatch.setattr(calculation_workflow, "large_project", lambda _: False)
    messages = []
    monkeypatch.setattr(gui, "QAccessible", SimpleNamespace(
        isActive=lambda: True, updateAccessibility=lambda event: messages.append(event.message()),
    ))
    dialog = gui.CalculationDialog(owner)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.activateWindow()
    qtbot.waitUntil(dialog.isActiveWindow)
    dialog.context.setText("Synthetic whole rock")
    dialog.confirmed.setChecked(True)
    dialog.calculate()
    messages.clear()

    dialog.next_page.setFocus()
    qtbot.keyClick(dialog.next_page, Qt.Key.Key_Space)
    assert dialog.preview.rowCount() == 1
    assert dialog.previous_page.hasFocus() and not dialog.next_page.isEnabled()
    assert messages == [dialog.page_label.text()]
    assert "201–201" in messages[0]

    qtbot.keyClick(dialog.previous_page, Qt.Key.Key_Space)
    assert dialog.preview.rowCount() == 200
    assert dialog.next_page.hasFocus() and not dialog.previous_page.isEnabled()
    assert len(messages) == 2 and "1–200" in messages[1]


def test_compact_calculation_layout_shows_results_and_labels_input_controls(owner, qtbot):
    dialog = gui.CalculationDialog(owner)
    qtbot.addWidget(dialog)
    dialog.resize(540, 520)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.parameter_tabs.indexOf(dialog._compact_results) >= 0)
    assert dialog.preview_button.isVisible()
    assert dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isVisible()
    assert dialog.inputs["TOC"][0].accessibleName() != dialog.inputs["TOC"][1].accessibleName()
    assert all(widget.accessibleName() for fields in dialog.inputs.values() for widget in fields)
    assert all(widget.accessibleName() for fields in dialog.lab_inputs.values() for widget in fields)
    dialog.context.setText("Synthetic whole rock")
    dialog.confirmed.setChecked(True)
    dialog.preview_button.click()
    assert dialog.parameter_tabs.currentWidget() is dialog._compact_results
    assert dialog.preview.isVisible() and dialog.preview.rowCount() == 2
    dialog.resize(1000, 700)
    qtbot.waitUntil(lambda: dialog.parameter_tabs.indexOf(dialog._compact_results) == -1)
    assert dialog.preview.isVisible() and dialog.preview.rowCount() == 2


def test_enter_in_search_never_creates_columns_and_moves_to_matching_rows(owner, qtbot):
    dialog = gui.CalculationDialog(owner)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.activateWindow()
    qtbot.waitUntil(dialog.isActiveWindow)
    dialog.context.setText("Synthetic whole rock")
    dialog.confirmed.setChecked(True)
    dialog.calculate()
    dialog.activateWindow()
    qtbot.waitUntil(dialog.isActiveWindow)
    project = owner.model.project
    count = len(project.columns)
    apply = dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert apply.isEnabled()

    dialog.sample_search.setText("no such sample")
    qtbot.mouseClick(dialog.sample_search, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(dialog.sample_search.hasFocus)
    qtbot.keyClick(dialog.sample_search, Qt.Key.Key_Return)
    assert dialog.sample_search.hasFocus()
    assert len(project.columns) == count and dialog.isVisible()

    dialog.sample_search.setText(project.rows[0].id)
    qtbot.keyClick(dialog.sample_search, Qt.Key.Key_Enter)
    assert dialog.preview.hasFocus() and dialog.preview.currentRow() == 0
    assert len(project.columns) == count and dialog.isVisible()

    apply.setFocus()
    qtbot.keyClick(apply, Qt.Key.Key_Return)
    assert len(project.columns) == count + 1


def test_preview_search_shows_sample_labels_and_opens_the_exact_row_from_keyboard(owner, qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    project = build_project(
        ["Well", "Sample", "Depth", "TOC", "S2"],
        [["W-1", "Sample A", "1200", "2", "6"], ["W-2", "Sample A", "1350", "2", "8"]],
        [2, 3],
        [(0, "Well", "text", ""), (1, "Sample", "text", ""), (2, "Depth", "number", "m"),
         (3, "TOC", "number", "%wt"), (4, "S2", "number", "mg HC/g rock")],
        {},
    )
    owner.set_project(project)
    owner.show_calculations()
    dialog = owner.calculation_window
    dialog.context.setText("Synthetic whole rock")
    dialog.confirmed.setChecked(True)
    dialog.calculate()
    dialog.sample_search.setText("w-2")
    assert dialog.preview.rowCount() == 1
    label = dialog.preview.item(0, 0)
    assert label.text() == "W-2 · Sample A · 1350"
    assert project.rows[1].id in label.toolTip()
    dialog.sample_search.setText("1350")
    assert dialog.preview.rowCount() == 1
    opened = []
    monkeypatch.setattr(owner.desktop, "open_table", lambda: opened.append(True))
    dialog.preview.setCurrentCell(0, 1)
    assert dialog.open_sample.isEnabled()
    qtbot.keyClick(dialog.preview, Qt.Key.Key_Return)
    assert opened == [True]
    selected = owner.table.selectionModel().selectedRows()[0]
    assert owner.proxy.mapToSource(selected).row() == 1

    owner.apply_filter([{"column": project.columns[0].id, "operator": "eq", "value": "W-1"}], "AND")
    owner.analysis_scope.set_mode("all")
    owner.table.selectRow(0)
    dialog.calculate()
    assert dialog.preview.rowCount() == 1
    answers = iter([QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes])
    monkeypatch.setattr(QMessageBox, "question", lambda *args: next(answers))

    dialog.jump_to_sample(0, 0)
    assert owner.proxy.rowCount() == 1
    selected = owner.table.selectionModel().selectedRows()[0]
    assert owner.proxy.mapToSource(selected).row() == 0

    dialog.jump_to_sample(0, 0)
    assert owner.proxy.rowCount() == 2
    selected = owner.table.selectionModel().selectedRows()[0]
    assert owner.proxy.mapToSource(selected).row() == 1


def test_preview_summary_counts_only_scoped_samples_and_skips_empty_scope(owner, monkeypatch):
    project = owner.model.project
    dialog = owner.calculation_window
    owner.proxy.set_rules([], row_ids=[project.rows[0].id])
    owner.analysis_scope.set_mode("filtered")
    dialog.calculate()
    assert dialog.preview.rowCount() == 1
    column, values = dialog.prepared[0]
    assert set(values) == set(column.metadata["diagnostics"]) == {project.rows[0].id}
    assert dialog.summary.text() == " · ".join([
        gui.tr('Проб в области расчёта: {0}', 1),
        "HI: " + gui.tr('рассчитано') + ": 1",
    ])

    owner.proxy.set_rules([], row_ids=[])
    monkeypatch.setattr(calculation_workflow, "make_calculated_column", lambda *args: pytest.fail("empty scope started a calculation"))
    dialog.calculate()
    assert dialog.preview.rowCount() == 0
    assert not dialog.prepared
    assert dialog.summary.text() == gui.tr('В выбранной области нет проб.')


def test_lab_selector_disables_formula_inputs_and_rejects_a_stale_binding(owner):
    project = owner.model.project
    s2 = project.columns[1]
    dialog = owner.calculation_window
    lab, _, tolerance = dialog.lab_inputs["HI"]
    index = lab.findData(s2.id)
    assert index >= 0 and not lab.model().item(index).isEnabled()
    lab.setCurrentIndex(index)
    tolerance.setValue(300)
    dialog.calculate()
    assert not dialog._comparison_rows
    assert not dialog.save_comparison_button.isEnabled()
    assert gui.tr('Лабораторная колонка уже используется как вход формулы') in dialog.summary.text()
    assert dialog.prepared[0][1][project.rows[0].id] == 300


def test_preview_empty_filter_is_announced_and_reset_preserves_width(owner, monkeypatch):
    from PySide6.QtGui import QAccessible

    dialog = owner.calculation_window
    messages = []
    monkeypatch.setattr(dialog, "announce", lambda widget, text: messages.append(text))
    dialog.calculate()
    summary = dialog.summary.text()
    expected_sample = dialog.preview.item(0, 0).text()
    assert dialog.preview.isColumnHidden(0)
    assert dialog.preview.verticalHeaderItem(0).text() == expected_sample
    table = QAccessible.queryAccessibleInterface(dialog.preview)
    headers = [child.text(QAccessible.Text.Name) for i in range(table.childCount())
               if (child := table.child(i)).role() == QAccessible.Role.RowHeader]
    assert headers[0] == expected_sample
    dialog.preview.setColumnWidth(1, 95)
    dialog.preview.setCurrentCell(1, 1)
    assert dialog.preview_detail.text().endswith("S2: ND")

    dialog.sample_search.setText("missing sample")
    assert dialog.preview.rowCount() == 0
    assert dialog.summary.text() == summary
    assert dialog.page_label.text() == messages[-1] == gui.tr('По текущим фильтрам пробы не найдены.')
    assert dialog.reset_preview_filters.isEnabled()
    dialog.reset_preview_filters.click()
    assert dialog.preview.rowCount() == 2
    assert dialog.preview.columnWidth(1) == 95
    assert dialog.preview.verticalHeaderItem(0).text() == expected_sample
    assert dialog.page_label.text() == messages[-1]


def test_open_damaged_comparison_clears_rows_and_disables_export(owner, monkeypatch):
    from PySide6.QtWidgets import QComboBox, QDialog, QLabel, QPushButton, QTableView

    project = owner.model.project
    good = ["uuid", "sample", "HI", 300, 300, 0, "match", "ok",
            "lab-id", "Lab", "mg HC/g TOC", "mg HC/g TOC", 0]
    def snapshot(row):
        return {"created_at": "2026-11-17T00:00:00", "formulas": ["HI"], "protocol": {
            "format": "autopyro.pyrolysis-comparison", "version": 1, "settings": {}, "rows": [row],
        }}

    project.view[gui.COMPARISON_KEY] = {"damaged": snapshot(["bad"]), "valid": snapshot(good)}
    def inspect(dialog):
        table = dialog.findChild(QTableView)
        selector = dialog.findChild(QComboBox)
        export = next(button for button in dialog.findChildren(QPushButton) if "Выгрузить" in button.text())
        assert table.model().rowCount() == 1 and export.isEnabled()
        selector.setCurrentIndex(selector.findData("damaged"))
        assert table.model().rowCount() == 0 and not export.isEnabled()
        assert any("данные повреждены" in label.text() for label in dialog.findChildren(QLabel))
        selector.setCurrentIndex(selector.findData("valid"))
        assert table.model().index(0, 6).data() == gui.tr('Численно в допуске')
        assert export.isEnabled()

    monkeypatch.setattr(QDialog, "exec", inspect)
    owner.calculation_window.open_saved_comparisons()


def test_cancel_during_single_formula_discards_partial_preview(owner, monkeypatch):
    dialog = owner.calculation_window
    stages = []

    def run(_parent, _title, function, *, cancellable):
        assert cancellable

        def cancel_on_engine_checkpoint(stage, _percent):
            stages.append(stage)
            if len(stages) == 4:
                raise TaskCancelled()

        return function(cancel_on_engine_checkpoint)

    monkeypatch.setattr(calculation_workflow, "run_with_progress", run)
    dialog.calculate()
    assert len(stages) == 4 and all("HI" in stage for stage in stages)
    assert not dialog.prepared
    assert dialog.preview.rowCount() == 0
    assert dialog.summary.text() == gui.tr('Расчёт отменён.')


def test_applying_scoped_preview_keeps_outside_scope_diagnostics(owner):
    project = owner.model.project
    included, outside = project.rows
    owner.proxy.set_rules([], row_ids=[included.id])
    owner.analysis_scope.set_mode("filtered")
    dialog = owner.calculation_window
    dialog.calculate()
    dialog.apply()
    column = project.columns[-1]
    assert included.values[column.id] == 300 and outside.values[column.id] is None
    assert column.metadata["diagnostics"][outside.id]["status"] == "outside_scope"


def test_invalid_formula_preserves_other_results_and_edit_clears_preview(owner):
    dialog = owner.calculation_window
    dialog.outputs.cellWidget(1, 0).setChecked(True)
    dialog.calculate()
    assert [column.name for column, _ in dialog.prepared] == ["HI"]
    assert "S3" in dialog.outputs.item(1, 1).toolTip()
    assert not dialog.input_feedback["S3"].isHidden()
    assert "S3" in dialog.inputs["S3"][0].accessibleDescription()
    assert dialog.input_feedback["TOC"].isHidden()
    assert dialog.preview.rowCount() == 2
    dialog.context.setText("Updated context")
    assert dialog.preview.rowCount() == 0
    assert dialog.input_feedback["S3"].isHidden()

    toc_unit = dialog.inputs["TOC"][1]
    toc_unit.setCurrentText("g/g rock")
    dialog.calculate()
    assert not dialog.prepared
    assert not dialog.input_feedback["TOC"].isHidden()
    assert "g/g rock" in toc_unit.accessibleDescription()
    assert not dialog.input_feedback["S3"].isHidden()


@pytest.mark.parametrize("change", ["data", "filter", "proxy_filter", "proxy_sort"])
def test_changed_inputs_during_worker_cannot_be_applied(owner, monkeypatch, change):
    changed = Event()
    original = calculation_workflow.make_calculated_column

    def delayed(*args, **kwargs):
        assert changed.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(calculation_workflow, "make_calculated_column", delayed)

    def mutate():
        if change == "data":
            owner.model.setData(owner.model.index(0, 1), "8")
        elif change == "filter":
            owner.apply_filter([{"column": owner.model.project.columns[1].id,
                                 "operator": "empty"}], "AND")
        elif change == "proxy_filter":
            owner.proxy.set_rules([{"column": owner.model.project.columns[1].id, "operator": "empty"}])
        else:
            owner.proxy.sort(1, Qt.SortOrder.DescendingOrder)
        changed.set()

    QTimer.singleShot(0, mutate)
    dialog = owner.calculation_window
    dialog.calculate()
    assert dialog.prepared == []
    assert not dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    dialog.apply()
    assert len(owner.model.project.columns) == 2
