import csv
import json

import pytest
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QPushButton

from autopyro.core import Column, Project, Row
from autopyro.io import build_project
from autopyro.statistics import StatisticsOptions, calculate_statistics
from autopyro.window import MainWindow


def wait_report(qtbot, dialog):
    qtbot.waitUntil(lambda: dialog.report is not None and dialog.worker is not None
                   and not dialog.task_running and not dialog.pending
                   and not dialog.timer.isActive(), timeout=10000)


def test_statistics_follows_filter_edit_and_selection(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    window.show_statistics()
    dialog = window.statistics_window
    wait_report(qtbot, dialog)
    assert window.desktop.properties_stack.currentWidget() is dialog.parameter_tabs
    assert dialog.parameter_tabs.isAncestorOf(dialog.properties)
    assert dialog.parameter_tabs.isAncestorOf(dialog.scope)
    assert not dialog.parameter_tabs.isAncestorOf(dialog.tabs)
    window.show_interpretation()
    interpretation = window.interpretation_window
    interpretation.timer.stop()
    assert window.desktop.properties_stack.currentWidget() is interpretation.parameter_tabs
    window.desktop.documents.setCurrentWidget(dialog)
    assert window.desktop.properties_stack.currentWidget() is dialog.parameter_tabs
    assert len(dialog.report["row_ids"]) == 5
    well, depth, s2 = window.model.project.columns[0].id, window.model.project.columns[2].id, window.model.project.columns[4].id
    window.apply_filter([{"column": well, "operator": "eq", "value": "A"}], "AND")
    wait_report(qtbot, dialog)
    assert len(dialog.report["row_ids"]) == 3
    assert next(e for e in dialog.report["summary"] if e["column_id"] == depth)["mean"] == 1010
    assert window.model.setData(window.model.index(0, 2), "970")
    wait_report(qtbot, dialog)
    assert next(e for e in dialog.report["summary"] if e["column_id"] == depth)["mean"] == 1000
    selection = window.table.selectionModel()
    selection.select(window.proxy.index(0, 2), QItemSelectionModel.SelectionFlag.ClearAndSelect)
    dialog.scope.setCurrentIndex(dialog.scope.findData("selected"))
    wait_report(qtbot, dialog)
    assert len(dialog.report["row_ids"]) == 1
    assert next(e for e in dialog.report["summary"] if e["column_id"] == s2)["std"] is None
    selection.clearSelection()
    wait_report(qtbot, dialog)
    assert dialog.report["row_ids"] == []
    assert dialog.report["summary"] == []
    dialog.close()
    interpretation.close()
    window.dirty = False


def test_grouped_statistics_settings_and_reproducible_export(qtbot, tmp_path, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    well = window.model.project.columns[0].id
    window.model.project.view["named"] = {"A only": {"rules": [{"column": well, "operator": "eq", "value": "A"}], "combine": "AND"}}
    window.show_statistics()
    dialog = window.statistics_window
    wait_report(qtbot, dialog)
    dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
    dialog.scope.setCurrentIndex(dialog.scope.findData("view:A only"))
    wait_report(qtbot, dialog)
    assert len(dialog.report["row_ids"]) == 3
    assert dialog.report["summary"][0]["group"] == ["A"]
    assert window.proxy.rowCount() == 5  # saved filter is evaluated independently
    export_path = tmp_path / "statistics.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(export_path), "JSON (*.json)"))
    dialog.export_snapshot()
    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert exported["sections"]["pending"] == ["repeats", "thresholds", "interpolation"]
    assert "thresholds" not in exported["sections"]["calculated"]
    assert len(exported["input"]["rows"]) == 3
    report = exported["report"]
    options = StatisticsOptions(**{**report["options"], "column_ids": tuple(report["options"]["column_ids"]),
                                  "group_ids": tuple(report["options"]["group_ids"])})
    reconstructed = Project(id=report["project_id"],
                            columns=[Column(**c) for c in exported["input"]["columns"]],
                            rows=[Row(**r) for r in exported["input"]["rows"]])
    recalculated = calculate_statistics(reconstructed, report["row_ids"], options)
    assert recalculated["summary"] == report["summary"]
    assert "numpy" in report["versions"]
    window.path = tmp_path / "project.apyro"
    assert window.save()
    saved_settings = Project.load(window.path).view["statistics"]
    assert saved_settings["group_ids"] == [well]
    assert saved_settings["scope"] == "view:A only"
    dialog.close()
    window.set_project(Project.load(window.path), window.path)
    window.show_statistics()
    wait_report(qtbot, window.statistics_window)
    assert len(window.statistics_window.report["row_ids"]) == 3
    window.statistics_window.close()
    window.dirty = False


def test_extreme_summary_exports_atomically(qtbot, tmp_path, monkeypatch):
    from autopyro.data_io import file_output

    window = MainWindow()
    qtbot.addWidget(window)
    project = build_project(["X"], [[-1.7e308], [1.7e308]], [2, 3], [(0, "X", "number", "")], {})
    window.set_project(project)
    window.show_statistics()
    dialog = window.statistics_window
    wait_report(qtbot, dialog)
    assert "Численная неустойчивость" in dialog.status.text()
    output = tmp_path / "statistics.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(output), "JSON (*.json)"))
    dialog.export_snapshot()
    saved = output.read_bytes()
    assert json.loads(saved)["report"]["summary"][0]["std"] is None

    def fail_replace(*args):
        raise OSError("replace failed")

    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    monkeypatch.setattr(file_output.os, "replace", fail_replace)
    dialog.export_snapshot()
    assert warnings
    assert output.read_bytes() == saved
    assert list(tmp_path.iterdir()) == [output]


def test_result_actions_work_from_keyboard_and_clipboard_keeps_text_literal(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    project = build_project(["Well", "Value"], [["  =1+1", -2], ["  =1+1", None]], [2, 3],
                            [(0, "Well", "text", ""), (1, "Value", "number", "")], {})
    window.set_project(project)
    window.show()
    window.show_statistics()
    dialog = window.statistics_window
    dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
    wait_report(qtbot, dialog)
    dialog.copy_current()
    clipboard = list(csv.DictReader(QApplication.clipboard().text().splitlines(), delimiter="\t"))
    assert clipboard[0]["Группа: Well"].startswith("'")
    assert clipboard[0]["Группа: Well"].lstrip("'").strip() == "=1+1"
    assert clipboard[0]["Минимум"] == "-2"

    table = dialog.tables["availability"]
    dialog.tabs.setCurrentWidget(table)
    model = table.model()
    table.setCurrentIndex(model.index(0, model.headers.index("Показатель")))
    assert not dialog.result_action_button.isEnabled()
    table.setCurrentIndex(model.index(0, model.headers.index("Пусто")))
    assert dialog.result_action_button.isEnabled()
    dialog.result_action_button.setFocus()
    qtbot.keyClick(dialog.result_action_button, Qt.Key.Key_Space)
    samples = dialog.sample_table.model()
    assert samples.rows[0][0] == project.rows[1].id
    dialog.sample_table.setCurrentIndex(samples.index(0, 0))
    assert dialog.sample_action_button.isEnabled()
    dialog.sample_action_button.setFocus()
    qtbot.keyClick(dialog.sample_action_button, Qt.Key.Key_Space)
    selected = window.table.selectionModel().selectedRows()
    assert [window.row_ids()[index.row()] for index in selected] == [project.rows[1].id]


def test_sorted_statistics_keep_source_rows_and_export_visible_order(qtbot, tmp_path, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    project = build_project(["Well", "Value"], [["A", 2], ["B", 10], ["C", None]], [2, 3, 4],
                            [(0, "Well", "text", ""), (1, "Value", "number", "")], {})
    window.set_project(project)
    window.show_statistics()
    dialog = window.statistics_window
    dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
    wait_report(qtbot, dialog)
    summary = dialog.tables["summary"]
    model = summary.model()
    median_column = model.headers.index("Медиана")
    summary.setCurrentIndex(model.index(0, median_column))
    summary.sortByColumn(median_column, Qt.SortOrder.DescendingOrder)
    assert [row[0] for row in model.rows] == ["B", "A", "C"]
    assert summary.currentIndex().row() == 1
    assert model.row_order == [1, 0, 2]
    dialog.copy_current()
    assert [row["Группа: Well"] for row in csv.DictReader(
        QApplication.clipboard().text().splitlines(), delimiter="\t",
    )] == ["B", "A", "C"]
    path = tmp_path / "sorted.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), "CSV (*.csv)"))
    dialog.export_current()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        assert [row["Группа: Well"] for row in csv.DictReader(stream, delimiter=";")] == ["B", "A", "C"]

    availability = dialog.tables["availability"]
    dialog.tabs.setCurrentWidget(availability)
    model = availability.model()
    missing_column = model.headers.index("Пусто")
    availability.sortByColumn(missing_column, Qt.SortOrder.DescendingOrder)
    availability.setCurrentIndex(model.index(0, missing_column))
    dialog.result_action_button.click()
    assert dialog.sample_table.model().rows[0][0] == project.rows[2].id


def test_statistics_identifiers_stay_visible_and_can_be_unpinned(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    names = ["Well", "Formation", "Long property name " * 8]
    project = build_project(names, [["A", "F1", 2], ["B", "F2", 10]], [2, 3], [
        (i, name, "number" if i == 2 else "text", "") for i, name in enumerate(names)
    ], {})
    window.set_project(project)
    window.resize(1024, 720)
    window.show()
    window.show_statistics()
    dialog = window.statistics_window
    dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
    dialog.groups.item(1).setCheckState(Qt.CheckState.Checked)
    wait_report(qtbot, dialog)

    table = dialog.tables["summary"]
    qtbot.waitUntil(lambda: table.frozen.isVisible() and table.horizontalScrollBar().maximum() > 0)
    assert table.pinned_columns == (0, 2)
    assert table.frozen.selectionModel() is table.selectionModel()
    assert table.frozen.isColumnHidden(1)
    before = table.frozen.geometry()
    table.horizontalScrollBar().setValue(table.horizontalScrollBar().maximum())
    assert table.frozen.geometry() == before
    assert table.frozen.columnViewportPosition(0) == 0
    assert table.frozen.columnViewportPosition(2) == table.columnWidth(0)

    model = table.model()
    table.frozen.setCurrentIndex(model.index(0, 2))
    table.frozen.doubleClicked.emit(model.index(0, 2))
    assert dialog.distribution_view.widget() is not None
    table.frozen.horizontalHeader().sectionClicked.emit(0)
    table.frozen.horizontalHeader().sectionClicked.emit(0)
    assert [row[0] for row in model.rows] == ["B", "A"]

    dialog.pin_identifiers.setChecked(False)
    assert table.frozen.isHidden()
    assert table.horizontalHeader().visualIndex(1) == 1
    assert table.horizontalHeader().visualIndex(2) == 2


def test_calculation_context_follows_effective_tab_policies(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    window.show_statistics()
    dialog = window.statistics_window
    dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
    dialog.text_numbers.setChecked(True)
    wait_report(qtbot, dialog)
    assert "Группировка: Скважина" in dialog.policy.text()
    assert "Числа из текста включены" in dialog.policy.text()

    dialog.tabs.setCurrentWidget(dialog.tables["thresholds"])
    wait_report(qtbot, dialog)
    assert "показатель, метод, лаборатория" in dialog.policy.text()
    assert "от непустых рабочих записей, включая исключённые QC" in dialog.policy.text()
    assert "Группировка: Скважина" not in dialog.policy.text()

    dialog.tabs.setCurrentWidget(dialog.tables["interpolation"])
    wait_report(qtbot, dialog)
    assert "пустым исходным ячейкам" in dialog.policy.text()
    assert "флажок" in dialog.policy.text().casefold()
    dialog.text_numbers.setChecked(False)
    assert dialog.policy.isHidden()
    wait_report(qtbot, dialog)
    dialog.tabs.setCurrentWidget(dialog.tables["summary"])
    assert "Числа из текста исключены" in dialog.policy.text()


def test_hidden_dialog_does_not_trigger_jobs(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    window.show_statistics()
    dialog = window.statistics_window
    wait_report(qtbot, dialog)
    dialog.close()
    assert dialog.closed
    generation = dialog.generation
    window.model.setData(window.model.index(0, 2), "999")
    assert dialog.generation == generation
    window.show_statistics()
    wait_report(qtbot, dialog)
    assert next(e for e in dialog.report["summary"] if e["property"] == "MD")["min"] == 999
    dialog.close()
    window.dirty = False


def test_comparison_drills_into_excluded_sample_hidden_by_table_filter(qtbot, tmp_path, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    project = window.model.project
    for analysis in project.science.analyses.values():
        analysis.method = "Demo method"
    well, _, depth, toc = project.columns[:4]

    def rules(value):
        return [{"column": well.id, "operator": "eq", "value": value}]

    project.view["named"] = {"A only": {"rules": rules("A")}, "B only": {"rules": rules("B")}}
    window.apply_filter(rules("A"), "AND")
    window.show_statistics()
    dialog = window.statistics_window
    dialog.tabs.setCurrentWidget(dialog.tables["comparison"].parentWidget())
    dialog.text_numbers.setChecked(True)
    dialog.properties.item(3).setCheckState(Qt.CheckState.Checked)
    for combo, name in zip(dialog.comparison_views, project.view["named"]):
        combo.setCurrentIndex(combo.findData(f"view:{name}"))
    wait_report(qtbot, dialog)
    entries = {entry["column_id"]: entry for entry in dialog.report["comparison"]}
    assert entries[depth.id]["delta"] == -5
    assert entries[toc.id]["b_used"] == entries[toc.id]["b_excluded"] == 1
    assert dialog.report["comparison_context"]["overlap"] == 0
    assert project.view["statistics"]["comparison_scope_b"] == "view:B only"
    assert window.row_ids() == [row.id for row in project.rows[:3]]

    path = tmp_path / "comparison.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), "CSV (*.csv)"))
    dialog.export_current()
    with path.open(encoding="utf-8-sig", newline="") as exported:
        by_property = {record["Показатель"]: record for record in csv.DictReader(exported, delimiter=";")}
    assert float(by_property["MD"]["Медиана B − A"]) == -5
    assert by_property["TOC"]["Представление B"] == "B only"

    table = dialog.tables["comparison"]
    model = table.model()
    row = next(i for i, values in enumerate(model.rows) if values[0] == "TOC")
    table.doubleClicked.emit(model.index(row, model.headers.index("B · исключено")))
    samples = dialog.sample_table.model()
    assert samples.rows == [[project.rows[4].id, "<0.1", "Пороговое значение"]]
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.No)
    dialog.sample_table.doubleClicked.emit(samples.index(0, 0))
    assert window.row_ids() == [row.id for row in project.rows[:3]]
    assert dialog.restore_sample_filter_button.isHidden()
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    dialog.sample_table.doubleClicked.emit(samples.index(0, 0))
    assert window.desktop.documents.currentWidget() is not dialog
    assert window.row_ids() == [row.id for row in project.rows[3:]]
    selected = window.table.selectionModel().selectedRows()
    assert [window.row_ids()[index.row()] for index in selected] == [project.rows[4].id]
    assert not dialog.restore_sample_filter_button.isHidden()
    dialog.restore_sample_filter_button.click()
    assert window.row_ids() == [row.id for row in project.rows[:3]]
    assert dialog.restore_sample_filter_button.isHidden()
    wait_report(qtbot, dialog)

    window.analysis_scope.rename_view("B only", "B renamed")
    wait_report(qtbot, dialog)
    assert dialog.comparison_views[1].currentData() == "view:B renamed"
    assert project.view["statistics"]["comparison_scope_b"] == "view:B renamed"
    assert dialog.report["comparison_context"]["b"] == "B renamed"
    window.analysis_scope.delete_view("B renamed")
    wait_report(qtbot, dialog)
    assert dialog.comparison_views[1].currentData() is None
    assert "comparison" not in dialog.report
    assert not dialog.export.isEnabled()


def test_comparison_can_mix_legacy_view_and_same_named_sample_set(qtbot, tmp_path, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    project = window.model.project
    for analysis in project.science.analyses.values():
        analysis.method = "Demo method"
    well, _, depth = project.columns[:3]
    rules = [{"column": well.id, "operator": "eq", "value": "A"}]
    set_ids = [project.rows[0].id, project.rows[4].id]
    project.view["named"] = {"A": {"rules": rules}}
    project.view["sample_sets"] = {"A": set_ids}
    project.view["statistics"] = {"comparison_a": "A"}
    window.apply_filter(rules, "AND")
    window.show_statistics()
    dialog = window.statistics_window
    a, b = dialog.comparison_views
    assert a.currentData() == "view:A"
    b.setCurrentIndex(b.findData("set:A"))
    table = dialog.tables["comparison"]
    dialog.tabs.setCurrentWidget(table.parentWidget())
    wait_report(qtbot, dialog)
    context = dialog.report["comparison_context"]
    assert (context["a_scope"], context["b_scope"], context["overlap"]) == ("view:A", "set:A", 1)
    assert "A: Представление: A · B: Набор проб: A" in dialog.policy.text()
    assert "Группировка сводки не применяется" in dialog.policy.text()
    assert next(entry for entry in dialog.report["comparison"] if entry["column_id"] == depth.id)["delta"] == -5
    assert window.row_ids() == [row.id for row in project.rows[:3]]

    output = tmp_path / "mixed.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(output), "CSV (*.csv)"))
    dialog.export_current()
    with output.open(encoding="utf-8-sig", newline="") as stream:
        exported = next(csv.DictReader(stream, delimiter=";"))
    assert (exported["Идентификатор A"], exported["Идентификатор B"], exported["Общие пробы"]) == ("view:A", "set:A", "1")

    model = table.model()
    row = next(i for i, values in enumerate(model.rows) if values[0] == "MD")
    table.doubleClicked.emit(model.index(row, model.headers.index("B · использовано")))
    samples = dialog.sample_table.model()
    assert [item[0] for item in samples.rows] == set_ids
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    dialog.sample_table.doubleClicked.emit(samples.index(1, 0))
    assert window.active_sample_set == "A"
    assert window.row_ids() == set_ids
    wait_report(qtbot, dialog)

    window.analysis_scope.rename_sample_set("A", "Review")
    wait_report(qtbot, dialog)
    assert a.currentData() == "view:A" and b.currentData() == "set:Review"
    assert project.view["statistics"]["comparison_scope_b"] == "set:Review"
    window.analysis_scope.delete_sample_set("Review")
    wait_report(qtbot, dialog)
    assert a.currentData() == "view:A" and b.currentData() is None
    assert not dialog.export.isEnabled()


def test_refresh_releases_obsolete_snapshots_and_completed_workers(qtbot):
    from weakref import ref
    from autopyro.ui.tasks import Worker

    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    window.show_statistics()
    dialog = window.statistics_window
    wait_report(qtbot, dialog)
    for value in ("900", "800", "700"):
        old_snapshot = ref(dialog.input_snapshot)
        assert window.model.setData(window.model.index(0, 2), value)
        assert old_snapshot() is None
        wait_report(qtbot, dialog)
        qtbot.waitUntil(lambda: len(dialog.findChildren(Worker)) == 1)
    last_snapshot = ref(dialog.input_snapshot)
    dialog.close()
    assert last_snapshot() is None
    assert all(table.model() is None for table in dialog.tables.values())


def test_statistical_policy_changes_reuse_snapshot_but_data_edits_do_not(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    window.show_statistics()
    dialog = window.statistics_window
    dialog.properties.item(3).setCheckState(Qt.CheckState.Checked)
    wait_report(qtbot, dialog)
    original = dialog.input_snapshot
    dialog.min_pairs.setValue(5)
    dialog.text_numbers.setChecked(True)
    wait_report(qtbot, dialog)
    assert dialog.input_snapshot is original
    assert dialog.report["options"]["min_pairs"] == 5
    assert next(row for row in dialog.report["summary"] if row["property"] == "TOC")["n"] == 4
    assert window.model.setData(window.model.index(0, 3), "9")
    wait_report(qtbot, dialog)
    assert dialog.input_snapshot is not original
    assert next(row for row in dialog.report["summary"] if row["property"] == "TOC")["max"] == 9


def test_edit_and_close_cancel_batched_snapshot_capture(qtbot, monkeypatch):
    from autopyro.ui.dialogs import statistics_calculation

    monkeypatch.setattr(statistics_calculation, "SNAPSHOT_BATCH_SIZE", 1)
    monkeypatch.setattr(statistics_calculation, "SNAPSHOT_TICK_BUDGET", 0)
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    window.show_statistics()
    dialog = window.statistics_window
    dialog.timer.stop()
    dialog.calculate()
    assert dialog.snapshot_timer.isActive()
    assert dialog.worker is None
    assert window.model.setData(window.model.index(0, 2), "900")
    assert dialog.snapshot_iterator is None
    assert not dialog.snapshot_timer.isActive()
    wait_report(qtbot, dialog)
    assert next(row for row in dialog.report["summary"] if row["property"] == "MD")["min"] == 900

    dialog.request_refresh()
    dialog.timer.stop()
    dialog.calculate()
    assert dialog.snapshot_timer.isActive()
    dialog.close()
    assert dialog.snapshot_iterator is None
    assert not dialog.snapshot_timer.isActive()


def test_snapshot_tick_copies_multiple_chunks_before_yielding(qtbot, monkeypatch):
    from autopyro.ui.dialogs import statistics_calculation

    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    window.show_statistics()
    dialog = window.statistics_window
    dialog.timer.stop()
    now = 0.0
    copied, completed = [], []
    result = object()

    def steps():
        nonlocal now
        for index in range(4):
            copied.append(index)
            now += 0.005
            yield None
        yield result

    monkeypatch.setattr(statistics_calculation, "perf_counter", lambda: now)
    monkeypatch.setattr(statistics_calculation, "SNAPSHOT_TICK_BUDGET", 0.012)
    dialog.snapshot_iterator = steps()
    dialog.snapshot_callback = completed.append
    dialog.advance_snapshot()
    assert copied == [0, 1, 2]
    assert not completed and dialog.snapshot_timer.isActive()
    dialog.snapshot_timer.stop()
    dialog.advance_snapshot()
    assert copied == [0, 1, 2, 3]
    assert completed == [result] and not dialog.snapshot_timer.isActive()


def test_changes_during_worker_discard_obsolete_result(qtbot, monkeypatch):
    from threading import Event
    from autopyro.ui.dialogs import statistics_calculation

    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    window.show_statistics()
    dialog = window.statistics_window
    wait_report(qtbot, dialog)
    release = Event()
    original = statistics_calculation.calculate_statistics
    def controlled_calculation(*args, **kwargs):
        assert release.wait(timeout=5)
        return original(*args, **kwargs)
    monkeypatch.setattr(statistics_calculation, "calculate_statistics", controlled_calculation)
    dialog.request_refresh()
    dialog.timer.stop()
    dialog.calculate()
    assert dialog.worker.isRunning()
    try:
        window.model.setData(window.model.index(0, 2), "900")
        window.model.setData(window.model.index(0, 2), "800")
    finally:
        release.set()
    wait_report(qtbot, dialog)
    depth = window.model.project.columns[2].id
    entry = next(e for e in dialog.report["summary"] if e["column_id"] == depth)
    assert entry["min"] == 800
    assert entry["mean"] == 968
    dialog.close()
    window.dirty = False


def test_cancel_keeps_ready_tabs_and_drops_late_worker_result(qtbot, monkeypatch):
    from threading import Event
    from autopyro.ui.dialogs import statistics_calculation

    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    window.show()
    window.show_statistics()
    dialog = window.statistics_window
    wait_report(qtbot, dialog)
    old_summary = dialog.tables["summary"].model()
    old_availability = dialog.tables["availability"].model()
    entered, release, returned = Event(), Event(), Event()
    original = statistics_calculation.calculate_statistics

    def late_result(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        result = original(*args)  # Simulate a task that returns despite cancellation.
        returned.set()
        return result

    monkeypatch.setattr(statistics_calculation, "calculate_statistics", late_result)
    assert window.model.setData(window.model.index(0, 2), "900")
    try:
        qtbot.waitUntil(entered.is_set)
        assert dialog.tabs.isEnabled()
        dialog.tabs.setCurrentWidget(dialog.tables["availability"])
        assert dialog.tables["availability"].model() is old_availability
        assert old_availability.rowCount() > 0
        assert not dialog.export.isEnabled()
        dialog.cancel_calculation_button.click()
        assert not dialog.cancel_calculation_button.isEnabled()
    finally:
        release.set()
    qtbot.waitUntil(lambda: returned.is_set() and not dialog.task_running)
    assert dialog.report is None
    assert dialog.tables["summary"].model() is old_summary
    assert "Расчёт отменён" in dialog.status.text()
    assert not dialog.timer.isActive()

    dialog.request_refresh()
    wait_report(qtbot, dialog)
    assert next(row for row in dialog.report["summary"] if row["property"] == "MD")["min"] == 900


def test_availability_and_outlier_navigation_use_the_calculated_group(qtbot, monkeypatch, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    data = [["A", value] for value in [1, 1, 1, 1, 100, "<0.1"]] + [["B", "ND"], ["B", "<0.2"]]
    project = build_project(["Well", "TOC"], data, list(range(2, 10)),
                            [(0, "Well", "text", ""), (1, "TOC", "text", "%")], {})
    window.set_project(project)
    window.show_statistics()
    dialog = window.statistics_window
    dialog.scope.setCurrentIndex(dialog.scope.findData("all"))
    dialog.properties.item(1).setCheckState(Qt.CheckState.Checked)
    dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
    dialog.text_numbers.setChecked(True)
    wait_report(qtbot, dialog)

    dialog.tabs.setCurrentWidget(dialog.tables["availability"])
    availability = dialog.tables["availability"]
    report, worker = dialog.report, dialog.worker
    dialog.result_group.setCurrentIndex(dialog.result_group.findText("Well = B"))
    assert dialog.report is report and dialog.worker is worker
    summary = dialog.tables["summary"]
    dialog.tabs.setCurrentWidget(summary)
    summary.doubleClicked.emit(summary.model().index(0, 0))
    axes = dialog.distribution_view.widget().figure.axes[0]
    assert [label.get_text() for label in axes.get_yticklabels()] == ["Well = B (N=0)"]
    dialog.tabs.setCurrentWidget(availability)
    model = availability.model()
    assert [values[0] for values in model.rows] == ["B"]
    assert model.row_order == [1]
    path = tmp_path / "filtered.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), "CSV (*.csv)"))
    dialog.export_current()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        assert [row["Группа: Well"] for row in csv.DictReader(stream, delimiter=";")] == ["B"]
    availability.doubleClicked.emit(model.index(0, model.headers.index("< / >")))
    sample_model = dialog.sample_table.model()
    assert sample_model.rows == [[project.rows[7].id, "<0.2", "Пороговое значение"]]
    dialog.sample_table.doubleClicked.emit(sample_model.index(0, 0))
    selected = window.table.selectionModel().selectedRows()
    assert [window.row_ids()[index.row()] for index in selected] == [project.rows[7].id]

    well = project.columns[0].id
    window.apply_filter([{"column": well, "operator": "eq", "value": "B"}], "AND")
    assert dialog.sample_table.model() is None
    assert sample_model.rows == []
    wait_report(qtbot, dialog)
    assert dialog.result_group.currentText() == "Well = B"
    dialog.result_group.setCurrentIndex(0)
    outliers = dialog.tables["outliers"]
    dialog.tabs.setCurrentWidget(outliers)
    assert dialog.sample_table.isHidden()
    assert outliers.model().rowCount() == 1
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    outliers.doubleClicked.emit(outliers.model().index(0, 0))
    assert window.row_ids() == [row.id for row in project.rows]
    selected = window.table.selectionModel().selectedRows()
    assert [window.row_ids()[index.row()] for index in selected] == [project.rows[4].id]
    window.apply_filter([{"column": well, "operator": "eq", "value": "B"}], "AND")
    assert window.row_ids() == [row.id for row in project.rows[6:]]
    assert dialog.restore_sample_filter_button.isHidden()
    wait_report(qtbot, dialog)


def test_summary_distribution_matches_used_samples_and_keeps_empty_groups(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    data = [["A", "1.2"], ["A", "2"], ["A", "<0.1"], ["B", "3"], ["B", "ND"], [None, "ND"]]
    project = build_project(["Well", "TOC"], data, list(range(2, 8)),
                            [(0, "Well", "text", ""), (1, "TOC", "text", "%")], {})
    toc = project.columns[1].id
    project.sync_measurements()
    excluded = project.science.current(project.rows[1].id, toc)
    project.view["qc_exclusions"] = {"statistics": {excluded.id: {"revision": excluded.revision}}}
    window.set_project(project)
    window.show_statistics()
    dialog = window.statistics_window
    dialog.properties.item(1).setCheckState(Qt.CheckState.Checked)
    dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
    dialog.text_numbers.setChecked(True)
    wait_report(qtbot, dialog)

    summary = dialog.tables["summary"]
    summary.doubleClicked.emit(summary.model().index(1, 0))
    assert not dialog.distribution_view.isHidden()
    canvas = dialog.distribution_view.widget()
    canvas.draw()
    figure = canvas.figure
    axes = canvas.figure.axes[0]
    assert axes.get_xlabel() == "TOC [%]"
    assert [label.get_text() for label in axes.get_yticklabels()] == [
        "Well = A (N=1)", "Well = B (N=1)", "Well = Пусто (N=0)",
    ]
    assert [points.get_offsets().tolist() for points in axes.collections] == [[[1.2, 1.0]], [[3.0, 2.0]]]
    dialog.text_numbers.setChecked(False)
    assert dialog.distribution_view.isHidden()
    assert dialog.distribution_view.widget() is None
    assert not figure.axes
    wait_report(qtbot, dialog)


def test_wide_summary_is_available_before_and_after_visiting_diagnostics(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    columns = [Column(f"P{i}", "number") for i in range(31)]
    window.set_project(Project(columns=columns, rows=[Row({column.id: 1 for column in columns})]))
    window.show_statistics()
    dialog = window.statistics_window
    wait_report(qtbot, dialog)
    assert len(dialog.report["summary"]) == 31
    assert dialog.correlations.isChecked()
    assert not dialog.report["options"]["correlations"]
    assert "Корреляции пропущены" in dialog.status.text()
    dialog.tabs.setCurrentWidget(dialog.tables["correlations"])
    assert "Корреляции пропущены" in dialog.navigation_hint.text()
    dialog.tabs.setCurrentWidget(dialog.tables["thresholds"])
    wait_report(qtbot, dialog)
    assert len(dialog.report["summary"]) == 31 and not dialog.report["correlations"]


def test_correlation_matrix_uses_selected_method_group_and_pair_counts(qtbot, tmp_path, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    data = [["A", 1, 1, None], ["A", 2, 8, 7], ["A", 3, 4, 7], ["B", 1, 1, 7]]
    project = build_project(["Well", "X", "Y", "Z"], data, list(range(2, 6)),
                            [(0, "Well", "text", ""), (1, "X", "number", ""),
                             (2, "Y", "number", ""), (3, "Z", "number", "")], {})
    window.set_project(project)
    window.show_statistics()
    dialog = window.statistics_window
    dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
    table = dialog.tables["correlations"]
    dialog.tabs.setCurrentWidget(table)
    wait_report(qtbot, dialog)
    model = table.model()
    pair = dialog.report["correlations"][0]
    assert pair["pearson"] != pair["spearman"]
    for method, header in (("pearson", "Pearson r"), ("spearman", "Spearman ρ")):
        table.setCurrentIndex(model.index(0, model.headers.index(header)))
        assert dialog.result_action_button.text().endswith(header)
        dialog.result_action_button.click()
        canvas = dialog.distribution_view.widget()
        canvas.draw()
        axes = canvas.figure.axes[0]
        matrix = axes.images[0].get_array()
        assert matrix[0, 1] == matrix[1, 0] == pair[method]
        assert matrix.mask[0, 0] and matrix.mask[0, 2]
        assert f"{pair[method]:.2f}\nN=3" in [label.get_text() for label in axes.texts]
        assert "—\nN=2" in [label.get_text() for label in axes.texts]
        assert axes.get_title() == f"{header} · Well = A"
    table.doubleClicked.emit(model.index(3, model.headers.index("Spearman ρ")))
    axes = dialog.distribution_view.widget().figure.axes[0]
    assert axes.get_title() == "Spearman ρ · Well = B"
    assert {label.get_text() for label in axes.texts} == {"—\nN=1"}
    dialog.result_group.setCurrentIndex(dialog.result_group.findText("Well = B"))
    model = table.model()
    model.sort(model.headers.index("Y"), Qt.SortOrder.DescendingOrder)
    assert sorted(model.row_order) == [3, 4, 5]
    table.doubleClicked.emit(model.index(0, model.headers.index("Spearman ρ")))
    assert dialog.distribution_view.widget().figure.axes[0].get_title() == "Spearman ρ · Well = B"
    path = tmp_path / "matrix.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), "JSON (*.json)"))
    dialog.save_matrix()
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = payload["matrix"]
    assert matrix["group"] == ["B"]
    assert matrix["options"]["min_pairs"] == 3
    assert all(pair["spearman"] is None and pair["n_pairs"] == 1 for pair in matrix["pairs"])
    dialog.clear_chart()
    dialog.open_matrix(payload)
    assert dialog.distribution_view.widget().figure.axes[0].images[0].get_array().mask.all()
    invalid = json.loads(json.dumps(payload))
    invalid["matrix"]["pairs"][0]["spearman"] = 0.5
    with pytest.raises(ValueError, match="Некорректное число полных пар"):
        dialog.open_matrix(invalid)
    invalid["matrix"]["options"]["min_pairs"] = 1
    with pytest.raises(ValueError, match="Неизвестный формат шаблона"):
        dialog.open_matrix(invalid)
    invalid = json.loads(json.dumps(payload))
    invalid["matrix"]["row_ids"].extend(["another row", "third row"])
    for entry in invalid["matrix"]["pairs"]:
        entry.update(n_pairs=3, status="constant")
    invalid["matrix"]["pairs"][0]["spearman"] = 0.81
    with pytest.raises(ValueError, match="не соответствует коэффициентам"):
        dialog.open_matrix(invalid)
    assert dialog.matrix_artifact == payload
    template = tmp_path / "plot-style.json"
    style = {"background": "#f0ead6", "font_family": "DejaVu Serif", "title_size": 16,
             "cmap": "viridis", "scope": "all", "x_id": "unrelated"}
    template.write_text(json.dumps({"format": "autopyro.plot", "version": 1, "spec": style}), encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(template), "JSON (*.json)"))
    dialog.load_matrix_style_button.click()
    expected_style = {key: style[key] for key in ("background", "font_family", "title_size")}
    payload = {"format": "autopyro.plot", "version": 1, "matrix": {**matrix, "style": expected_style}}
    assert dialog.matrix_artifact == payload
    axes = dialog.distribution_view.widget().figure.axes[0]
    assert axes.title.get_fontfamily() == ["DejaVu Serif"]
    assert axes.title.get_fontsize() == 16
    assert axes.images[0].get_cmap().name == "RdBu_r"
    assert axes.images[0].get_array().mask.all()
    dialog.save_matrix()
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    bad_style = {"format": "autopyro.plot", "version": 1, "spec": {"background": "invalid color"}}
    template.write_text(json.dumps(bad_style), encoding="utf-8")
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    dialog.load_matrix_style_button.click()
    assert warnings and dialog.matrix_artifact == payload
    assert dialog.distribution_view.widget().figure.axes[0] is axes
    image = tmp_path / "matrix.png"
    preview = dialog.distribution_view.widget().figure
    preview.set_size_inches(12, 8, forward=False)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(image), "PNG (*.png)"))
    with qtbot.waitSignal(dialog.matrix_export.completed, timeout=15000) as exported_job:
        dialog.save_matrix()
        assert not dialog.save_matrix_button.isEnabled()
    assert exported_job.args[0]["errors"] == []
    assert dialog.save_matrix_button.isEnabled()
    exported = QImage(str(image))
    assert (exported.width(), exported.height()) == (669, 519)
    assert exported.pixelColor(0, 0).name() == "#f0ead6"
    assert tuple(preview.get_size_inches()) == (12, 8)
    previous = image.read_bytes()
    with qtbot.waitSignal(dialog.matrix_export.completed, timeout=15000) as cancelled_job:
        dialog.save_matrix()
        assert dialog.matrix_export_progress.isVisible()
        qtbot.mouseClick(dialog.matrix_export_progress.findChild(QPushButton), Qt.MouseButton.LeftButton)
    assert cancelled_job.args[0] == {"plots": [], "errors": [], "cancelled": True}
    assert image.read_bytes() == previous
    assert set(tmp_path.iterdir()) == {image, path, template}
    assert not dialog.matrix_export_progress.isVisible()
    dialog.min_pairs.setValue(4)
    assert dialog.distribution_view.isHidden()
    wait_report(qtbot, dialog)
    window.show_plots()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(path), "JSON (*.json)"))
    window.plot_window.import_spec()
    assert dialog.matrix_artifact == payload
    assert dialog.distribution_view.widget().figure.axes[0].images[0].get_array().mask.all()
    wait_report(qtbot, dialog)


def test_scatter_matrix_uses_pairwise_rows_and_limits_panel_count(qtbot, tmp_path, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    names = ["Well", "X", "Y", "Z", "P", "Q", "R", "S"]
    data = [
        ["A", 100, 100, 100, 1, 1, 1, 1],
        ["B", 1, 2, None, 1, 1, 1, 1],
        ["B", 3, None, 4, 1, 1, 1, 1],
        ["B", 5, 6, 7, 1, 1, 1, 1],
    ]
    project = build_project(names, data, [2, 3, 4, 5], [
        (i, name, "number" if i else "text", "") for i, name in enumerate(names)
    ], {})
    window.set_project(project)
    window.show_statistics()
    dialog = window.statistics_window
    table = dialog.tables["correlations"]
    dialog.tabs.setCurrentWidget(table)
    dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
    wait_report(qtbot, dialog)
    assert not dialog.scatter_matrix.isEnabled()

    for i in range(4, len(names)):
        dialog.properties.item(i).setCheckState(Qt.CheckState.Unchecked)
    wait_report(qtbot, dialog)
    assert not dialog.scatter_matrix.isEnabled()
    dialog.scatter_matrix.click()
    assert dialog.matrix_artifact is None
    table.setCurrentIndex(table.model().index(3, 0))
    assert dialog.scatter_matrix.isEnabled()
    table.clearSelection()
    assert not dialog.scatter_matrix.isEnabled()
    table.setCurrentIndex(table.model().index(3, 1))
    assert dialog.scatter_matrix.isEnabled()
    dialog.scatter_matrix.click()
    canvas = dialog.distribution_view.widget()
    canvas.draw()
    assert canvas.figure.texts[0].get_text().endswith("Well = B")
    panels = [axes for axes in canvas.figure.axes if axes.get_visible()]
    assert len(panels) == 6
    assert [axes.collections[0].get_offsets().tolist() for axes in panels if axes.collections] == [
        [[1, 2], [5, 6]], [[3, 4], [5, 7]], [[6, 7]],
    ]
    assert [axes.texts[0].get_text() for axes in panels] == ["N=3", "N=2", "N=2", "N=2", "N=1", "N=2"]
    payload = json.loads(json.dumps(dialog.matrix_artifact, allow_nan=False))
    assert payload["matrix"]["values"][1] == [2.0, None, 6.0]
    dialog.clear_chart()
    dialog.open_matrix(payload)
    assert dialog.distribution_view.widget().figure.axes[3].collections[0].get_offsets().tolist() == [[1, 2], [5, 6]]
    invalid = json.loads(json.dumps(payload))
    invalid["matrix"]["pairs"][0]["n_pairs"] = 0
    with pytest.raises(ValueError, match="Некорректное число полных пар для X и Y"):
        dialog.open_matrix(invalid)
    duplicate = json.loads(json.dumps(payload))
    duplicate["matrix"]["row_ids"][-1] = duplicate["matrix"]["row_ids"][0]
    with pytest.raises(ValueError, match="Неизвестный формат шаблона"):
        dialog.open_matrix(duplicate)
    assert dialog.matrix_artifact == payload
    template = tmp_path / "plot-style.json"
    style = {"color": "#ff0000", "alpha": 0.25}
    template.write_text(json.dumps({"format": "autopyro.plot", "version": 1, "spec": style}), encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(template), "JSON (*.json)"))
    dialog.load_matrix_style_button.click()
    assert dialog.matrix_artifact["matrix"] == {**payload["matrix"], "style": style}
    preview = dialog.distribution_view.widget().figure
    assert preview.axes[3].collections[0].get_facecolors()[0] == pytest.approx([1, 0, 0, 0.25])
    preview.set_size_inches(12, 8, forward=False)
    image = tmp_path / "scatter.svg"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(image), "SVG (*.svg)"))
    with qtbot.waitSignal(dialog.matrix_export.completed, timeout=15000) as exported_job:
        dialog.save_matrix()
    assert exported_job.args[0]["errors"] == []
    assert 'width="442.8pt" height="385.2pt"' in image.read_text(encoding="utf-8")
    assert tuple(preview.get_size_inches()) == (12, 8)
    dialog.min_pairs.setValue(4)
    assert dialog.distribution_view.isHidden()
    assert not dialog.scatter_matrix.isEnabled()
    wait_report(qtbot, dialog)


def test_repeat_analysis_tab_calculates_and_opens_paired_samples(qtbot, tmp_path, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    project = window.model.project
    rid = project.rows[0].id
    cid = project.columns[4].id
    for value, lab in [(2, "A"), (4, "A"), (8, "B")]:
        project.science.add_result(rid, cid, value, analysis_fields={"method": "M", "laboratory": lab})
    window.show_statistics()
    dialog = window.statistics_window
    wait_report(qtbot, dialog)
    assert "repeats" not in dialog.report
    table = dialog.tables["repeats"]
    dialog.tabs.setCurrentWidget(table)
    wait_report(qtbot, dialog)
    model = table.model()
    row = next(i for i, values in enumerate(model.rows) if values[1] == "Парные разности")
    table.doubleClicked.emit(model.index(row, 0))
    samples = dialog.sample_table.model()
    assert samples.rows == [[rid, 3, 2, 8, 1, 5]]
    samples_index = samples.index(0, 0)
    dialog.sample_table.doubleClicked.emit(samples_index)
    assert window.row_ids()[window.table.selectionModel().selectedRows()[0].row()] == rid

    path = tmp_path / "repeats.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), "CSV (*.csv)"))
    dialog.export_current()
    with path.open(encoding="utf-8-sig", newline="") as exported:
        records = list(csv.DictReader(exported, delimiter=";"))
    assert {row["Вид сравнения"]: float(row["Медиана"]) for row in records} == {
        "Парные разности": 5, "Размах повторов": 2,
    }


def test_threshold_tab_exports_grouped_diagnostics_and_opens_samples(qtbot, tmp_path, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    project = window.model.project
    rid = project.rows[4].id
    cid = project.columns[3].id
    result = project.science.add_result(
        rid, cid, "<0.1", analysis_fields={"method": "M", "laboratory": "Lab"},
        details={"lod": 0.05},
    )
    project.select_measurement(rid, cid, result.id)
    window.show_statistics()
    dialog = window.statistics_window
    dialog.properties.item(3).setCheckState(Qt.CheckState.Checked)
    dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
    table = dialog.tables["thresholds"]
    dialog.tabs.setCurrentWidget(table)
    wait_report(qtbot, dialog)
    model = table.model()
    row = next(i for i, values in enumerate(model.rows) if values[:3] == ["TOC", "M", "Lab"])
    assert model.headers[0] == "Показатель"
    table.doubleClicked.emit(model.index(row, 0))
    samples = dialog.sample_table.model()
    assert samples.rows == [[rid, "<0.1", "Ниже порога", 0.1, 0.05, None, "Нет"]]
    dialog.sample_table.doubleClicked.emit(samples.index(0, 0))
    assert window.row_ids()[window.table.selectionModel().selectedRows()[0].row()] == rid

    path = tmp_path / "thresholds.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), "CSV (*.csv)"))
    dialog.export_current()
    with path.open(encoding="utf-8-sig", newline="") as exported:
        lab = next(record for record in csv.DictReader(exported, delimiter=";") if record["Лаборатория"] == "Lab")
    assert float(lab["Ниже порога, %"]) == 100


def test_threshold_counts_open_matching_samples_and_keep_selected_group(qtbot, tmp_path, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    project = build_project(
        ["Well", "Sample", "X"],
        [["A", "A-1", "ND"], ["A", "A-2", "<1"], ["B", "B-1", ">6"], ["A", "A-3", 3]],
        [1, 2, 3, 4], [(0, "Well", "text", ""), (1, "Sample", "text", ""), (2, "X", "number", "%")], {},
    )
    window.set_project(project)
    excluded = project.science.current(project.rows[-1].id, project.columns[2].id)
    project.view["qc_exclusions"] = {"statistics": {excluded.id: {"revision": excluded.revision}}}
    window.show_statistics()
    dialog = window.statistics_window
    dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
    table = dialog.tables["thresholds"]
    dialog.tab_picker.setCurrentIndex(dialog.tabs.indexOf(table))
    wait_report(qtbot, dialog)
    dialog.result_group.setCurrentIndex(dialog.result_group.findText("Well = A"))
    model = table.model()
    assert model.rows[0][model.headers.index("Выше порога")] == 0
    assert model.rows[0][model.headers.index("ND")] == 1
    table.doubleClicked.emit(model.index(0, model.headers.index("ND")))
    samples = dialog.sample_table.model()
    assert [row[0] for row in samples.rows] == [project.rows[0].id]
    assert samples.data(samples.index(0, 0)) == "A · A-1"
    assert samples.data(samples.index(0, 0), Qt.ItemDataRole.UserRole) == project.rows[0].id
    table.doubleClicked.emit(model.index(0, model.headers.index("Исключено QC")))
    assert [row[0] for row in dialog.sample_table.model().rows] == [project.rows[-1].id]
    path = tmp_path / "thresholds.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), "CSV (*.csv)"))
    dialog.export_current()
    with path.open(encoding="utf-8-sig", newline="") as exported:
        result, = csv.DictReader(exported, delimiter=";")
    assert result["Группа: Well"] == "A"
    assert result["Строк в области"] == "3"


def test_mixed_units_hide_summary_and_explain_ab_difference(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    project = build_project(
        ["Sample", "X", "Y"], [["A-1", 1, 2], ["B-1", 1000, 4], ["B-2", 2000, 6]], [1, 2, 3],
        [(0, "Sample", "text", ""), (1, "X", "number", "%"), (2, "Y", "number", "")], {},
    )
    window.set_project(project)
    cid = project.columns[1].id
    for index, row in enumerate(project.rows):
        measurement = project.science.current(row.id, cid)
        project.science.analyses[measurement.analysis_id].method = "M"
        if index:
            measurement.unit = "ppm"
    project.view["sample_sets"] = {"A": [project.rows[0].id], "B": [row.id for row in project.rows[1:]]}
    window.show_statistics()
    dialog = window.statistics_window
    table = dialog.tables["comparison"]
    for combo, scope in zip(dialog.comparison_views, ("set:A", "set:B"), strict=True):
        combo.setCurrentIndex(combo.findData(scope))
    dialog.tabs.setCurrentWidget(table.parentWidget())
    wait_report(qtbot, dialog)
    model = table.model()
    table.setCurrentIndex(model.index(0, model.headers.index("Статус")))
    assert dialog.comparison_context_hint.text() == "Единица: A = % · B = ppm"
    assert dialog.report["summary"][0]["median"] is None
    assert dialog.report["comparison"][0]["delta"] is None
    assert dialog.report["correlations"][0]["n_pairs"] == 0

    availability = dialog.tables["availability"]
    dialog.tabs.setCurrentWidget(availability)
    model = availability.model()
    column = model.headers.index("Искл. несовместимый контекст")
    availability.doubleClicked.emit(model.index(0, column))
    assert [row[0] for row in dialog.sample_table.model().rows] == [row.id for row in project.rows]


def test_interpolation_tab_includes_linked_column_and_recalculates_when_stale(qtbot, monkeypatch):
    from autopyro.analysis.interpolation import prepare
    from autopyro.model import audit

    window = MainWindow()
    qtbot.addWidget(window)
    project = build_project(["S2", "Depth", "Well"],
                            [[2, 0, "A"], [None, 2, "A"], [8, 6, "A"], [None, 10, "A"], [None, 11, "A"]],
                            [2, 3, 4, 5, 6],
                            [(0, "S2", "number", "mg/g"), (1, "Depth", "number", "m"), (2, "Well", "text", "")], {})
    source, depth, well = project.columns
    first, target, last, outside_range, outside_scope = [row.id for row in project.rows]
    scope = [first, target, last, outside_range]
    _, protocol = prepare(project, depth.id, source.id, [well.id], set(scope), constraints={})
    monkeypatch.setattr(audit, "MIN_PROTOCOL_BYTES", 0)
    archived = audit.archive_protocol(protocol)
    assert isinstance(archived, str)
    estimate = Column("S2 estimate", "number", unit=source.unit,
                      metadata={"method": "linear", "y_id": source.id, "stale": False,
                                "scope_row_ids": scope, "interpolation_protocol": archived})
    project.columns.append(estimate)
    for row, value in zip(project.rows, [None, 4, None, None, None], strict=True):
        row.values[estimate.id] = value
    window.set_project(project)
    window.show_statistics()
    dialog = window.statistics_window
    table = dialog.tables["interpolation"]
    wait_report(qtbot, dialog)
    original_snapshot = dialog.input_snapshot
    dialog.tabs.setCurrentWidget(table)
    wait_report(qtbot, dialog)
    assert dialog.input_snapshot is not original_snapshot
    assert estimate.id not in dialog.report["options"]["column_ids"]
    assert dialog.report["interpolation"][0]["added"] == 1
    table.doubleClicked.emit(table.model().index(0, 0))
    missing = [
        [outside_range, None, None, "Пусто без оценки", "Вне диапазона опор", None, None],
        [outside_scope, None, None, "Пусто без оценки", "Проба вне исходной выборки интерполяции", None, None],
    ]
    assert dialog.sample_table.model().rows == [
        [target, None, 4, "Использовано", "Предложена интерполяция", first, last],
        *missing,
    ]
    model = table.model()
    table.doubleClicked.emit(model.index(0, model.headers.index("Всего")))
    assert {row[0] for row in dialog.sample_table.model().rows} == {row.id for row in project.rows}
    table.doubleClicked.emit(model.index(0, model.headers.index("Добавлено оценок")))
    assert [row[0] for row in dialog.sample_table.model().rows] == [target]
    table.doubleClicked.emit(model.index(0, model.headers.index("Пусто без оценки")))
    assert dialog.sample_table.model().rows == missing
    empty_anchor = dialog.sample_table.model().index(0, 5)
    dialog.sample_table.setCurrentIndex(empty_anchor)
    assert not dialog.sample_action_button.isEnabled()

    dialog.scope.setCurrentIndex(dialog.scope.findData("all"))
    window.proxy.set_rules([], row_ids={target})
    window.view_changed()
    wait_report(qtbot, dialog)
    assert window.row_ids() == [target]
    table.doubleClicked.emit(table.model().index(0, 0))
    left = dialog.sample_table.model().index(0, 5)
    dialog.sample_table.setCurrentIndex(left)
    assert dialog.sample_action_button.isEnabled()
    assert dialog.sample_action_button.text() == "Показать опорную пробу в таблице"
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.No)
    dialog.sample_action_button.click()
    assert window.row_ids() == [target]
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    dialog.sample_action_button.click()
    assert window.row_ids() == [first]
    assert [index.row() for index in window.table.selectionModel().selectedRows()] == [0]
    dialog.restore_sample_filter_button.click()
    assert window.row_ids() == [target]
    wait_report(qtbot, dialog)
    assert window.model.setData(window.model.index(0, 0), "3")
    wait_report(qtbot, dialog)
    assert dialog.report["interpolation"][0]["added"] == 0
    assert dialog.report["interpolation"][0]["stale"] == 1


def test_cancel_diagnostic_keeps_current_report_usable(qtbot, monkeypatch):
    from threading import Event
    from autopyro.ui.dialogs import statistics_calculation

    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    window.show_statistics()
    dialog = window.statistics_window
    wait_report(qtbot, dialog)
    report = dialog.report
    entered, release = Event(), Event()
    original = statistics_calculation.threshold_diagnostics

    def late_result(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return original(*args)

    monkeypatch.setattr(statistics_calculation, "threshold_diagnostics", late_result)
    dialog.tabs.setCurrentWidget(dialog.tables["thresholds"])
    try:
        qtbot.waitUntil(entered.is_set)
        dialog.tabs.setCurrentWidget(dialog.tables["summary"])
        assert dialog.report is report
        assert dialog.export.isEnabled()
        dialog.cancel_calculation_button.click()
    finally:
        release.set()
    qtbot.waitUntil(lambda: not dialog.task_running)
    assert dialog.report is report and "thresholds" not in report
    assert not dialog.timer.isActive()


def test_diagnostic_discards_reused_snapshot_if_data_changes_during_calculation(qtbot, monkeypatch):
    from threading import Event
    from autopyro.ui.dialogs import statistics_calculation

    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    entered, release = Event(), Event()
    original = statistics_calculation.threshold_diagnostics

    def controlled(data, row_ids, options, **kwargs):
        if not entered.is_set():
            entered.set()
            assert release.wait(timeout=5)
        return original(data, row_ids, options, **kwargs)

    monkeypatch.setattr(statistics_calculation, "threshold_diagnostics", controlled)
    window.show_statistics()
    dialog = window.statistics_window
    dialog.properties.item(3).setCheckState(Qt.CheckState.Checked)
    wait_report(qtbot, dialog)
    original_snapshot = dialog.input_snapshot
    dialog.tabs.setCurrentWidget(dialog.tables["thresholds"])
    try:
        qtbot.waitUntil(entered.is_set)
        assert window.model.setData(window.model.index(4, 3), "ND")
    finally:
        release.set()
    wait_report(qtbot, dialog)
    assert dialog.input_snapshot is not original_snapshot
    toc = next(row for row in dialog.report["thresholds"] if row["property"] == "TOC")
    assert (toc["below"], toc["non_detect"]) == (0, 1)


def test_reentering_cancelled_diagnostic_restarts_after_worker_finishes(qtbot, monkeypatch):
    from threading import Event
    from autopyro.ui.dialogs import statistics_calculation

    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    window.show_statistics()
    dialog = window.statistics_window
    wait_report(qtbot, dialog)
    entered, release = Event(), Event()
    original = statistics_calculation.threshold_diagnostics
    calls = []

    def controlled(*args, **kwargs):
        calls.append(len(calls))
        if len(calls) == 1:
            entered.set()
            assert release.wait(timeout=10)
            return [{"obsolete": True}]
        return original(*args, **kwargs)

    monkeypatch.setattr(statistics_calculation, "threshold_diagnostics", controlled)
    summary, thresholds = (dialog.tables[name] for name in ("summary", "thresholds"))
    dialog.tabs.setCurrentWidget(thresholds)
    try:
        qtbot.waitUntil(entered.is_set)
        dialog.cancel_calculation()
        dialog.tabs.setCurrentWidget(summary)
        assert dialog.export.isEnabled()
        dialog.tabs.setCurrentWidget(thresholds)
    finally:
        release.set()
    qtbot.waitUntil(lambda: not dialog.task_running and "thresholds" in dialog.report, timeout=10000)
    assert calls == [0, 1]
    assert all("obsolete" not in row for row in dialog.report["thresholds"])


def test_statistics_tabs_share_fresh_snapshot_and_reject_stale_export(qtbot, tmp_path, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    project = window.model.project
    for analysis in project.science.analyses.values():
        analysis.method = "Demo method"
    well, _, depth, toc, s2, _ = project.columns
    project.view["named"] = {
        name: {"rules": [{"column": well.id, "operator": "eq", "value": name}]}
        for name in ("A", "B")
    }
    for value in (2, 4):
        project.science.add_result(project.rows[0].id, s2.id, value, analysis_fields={"method": "M"})
    window.show_statistics()
    dialog = window.statistics_window
    dialog.properties.item(3).setCheckState(Qt.CheckState.Checked)
    for combo, name in zip(dialog.comparison_views, ("A", "B")):
        combo.setCurrentIndex(combo.findData(f"view:{name}"))
    wait_report(qtbot, dialog)
    original_snapshot = dialog.input_snapshot
    summary_model = dialog.tables["summary"].model()
    for name in ("repeats", "thresholds"):
        dialog.tabs.setCurrentWidget(dialog.tables[name])
        wait_report(qtbot, dialog)
        assert dialog.input_snapshot is original_snapshot
        assert dialog.tables["summary"].model() is summary_model

    report = dialog.report
    assert report["comparison"] and report["repeats"] and report["thresholds"]
    dialog.tabs.setCurrentWidget(dialog.tables["repeats"])
    assert dialog.report is report
    assert dialog.export.isEnabled()

    output = tmp_path / "statistics.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(output), "JSON (*.json)"))
    dialog.export_snapshot()
    exported = json.loads(output.read_text(encoding="utf-8"))
    assert all(exported["report"][name] for name in ("comparison", "repeats", "thresholds"))
    assert {"comparison", "repeats", "thresholds"} <= set(exported["sections"]["calculated"])
    assert exported["sections"]["pending"] == ["interpolation"]

    stale_csv = tmp_path / "stale.csv"
    def edit_during_export(*args):
        assert window.model.setData(window.model.index(1, 2), "980")
        return str(stale_csv), "CSV (*.csv)"

    monkeypatch.setattr(QFileDialog, "getSaveFileName", edit_during_export)
    dialog.export_current()
    assert not stale_csv.exists()
    wait_report(qtbot, dialog)
    report = dialog.report
    assert next(row for row in report["comparison"] if row["column_id"] == depth.id)["delta"] == 5
    assert any(row["column_id"] == s2.id for row in report["repeats"])
    assert any(row["column_id"] == toc.id and row["below"] == 1 for row in report["thresholds"])
