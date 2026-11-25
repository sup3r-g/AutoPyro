import json
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QInputDialog, QFileDialog

from autopyro.core import Column, Project, Row
from autopyro.ui.window import MainWindow


def fitted_window(qtbot, *, with_notes_and_weights=False):
    window = MainWindow()
    qtbot.addWidget(window)
    x, y = Column(name="X", kind="number"), Column(name="Y", kind="number")
    rows = [Row(values={x.id: float(i), y.id: float(2*i+1+np.sin(i)*.1)}) for i in range(20)]
    columns = [x, y]
    if with_notes_and_weights:
        note, sigma = Column(name="Note"), Column(name="Sigma", kind="number")
        columns.extend((note, sigma))
        for row in rows:
            row.values.update({note.id: "Original note", sigma.id: 1.0})
    window.set_project(Project(columns=columns, rows=rows))
    window.show_curve_fitting()
    dialog = window.curve_fitting_window
    if with_notes_and_weights:
        dialog.sy_column.setCurrentIndex(dialog.sy_column.findData(sigma.id))
    dialog.calculate()
    qtbot.waitUntil(lambda: dialog.result is not None and not dialog.worker.isRunning(), timeout=10000)
    return window, dialog


@pytest.mark.parametrize("changed_input", ["X", "Sigma", "unit"])
def test_saved_fit_and_output_columns_ignore_notes(qtbot, monkeypatch, tmp_path, changed_input):
    from autopyro.ui.i18n import tr

    window, dialog = fitted_window(qtbot, with_notes_and_weights=True)
    project = window.model.project
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Saved fit", True))
    dialog.save_current_result()
    qtbot.waitUntil(lambda: not dialog.task_running and dialog.saved_results.count() == 1)
    dialog.apply_current_fit()
    qtbot.waitUntil(lambda: not dialog.task_running and len(project.columns) > 4)
    run = next(iter(project.view["curve_fit_runs"].values()))
    archive = project.view["curve_fit_results"][0]
    outputs = project.columns[4:]
    assert project.columns[3].id in run["source_ids"]

    assert window.model.setData(window.model.index(0, 2), "Edited note")
    window.model.undo.undo()
    window.model.undo.redo()
    assert not run["stale"] and not archive["stale"]
    assert all(not column.metadata["stale"] for column in outputs)

    path = tmp_path / "fit.apyro"
    project.save(path)
    reopened = Project.load(path)
    assert not next(iter(reopened.view["curve_fit_runs"].values()))["stale"]
    assert not reopened.view["curve_fit_results"][0]["stale"]
    assert all(not column.metadata["stale"] for column in reopened.columns[4:])

    if changed_input == "unit":
        project.columns[0].unit = "m"
        window.model.refresh()
    else:
        index = 0 if changed_input == "X" else 3
        assert window.model.setData(window.model.index(0, index), "3")
    assert run["stale"] and archive["stale"]
    assert all(column.metadata["stale"] for column in outputs)
    status = window.statusBar().currentMessage()
    assert tr('Устаревших колонок подгонки: {0}', len(outputs)) in status
    assert tr('Строк: {0:,} / {1:,}   Колонок: {2}   Условия: {3}   Устаревших интерполяций: {4}{5}',
              20, 20, len(project.columns), 0, 0, "") in status


def test_diagnostic_navigation_and_undoable_output_columns(qtbot, tmp_path):
    window, dialog = fitted_window(qtbot)
    dialog.tabs.setCurrentWidget(dialog.diagnostic_page)
    assert dialog.residual_table.model().rowCount() == 20
    dialog.residual_table.setCurrentIndex(dialog.residual_table.model().index(0, 0))
    wanted = dialog.diagnostic_rows[0]
    dialog.jump_to_sample()
    from autopyro.ui.table.selection import selected_row_ids
    assert selected_row_ids(window) == {wanted}
    dialog.apply_current_fit()
    qtbot.waitUntil(lambda: not dialog.worker.isRunning() and len(window.model.project.columns) == 8, timeout=15000)
    assert not (window.model.flags(window.model.index(0, 2)) & Qt.ItemFlag.ItemIsEditable)
    assert not window.model.setData(window.model.index(0, 2), "999")
    path = tmp_path / "outputs.autopyro"
    window.model.project.save(path)
    reloaded = Project.load(path)
    output_ids = {c.id for c in reloaded.columns if c.metadata.get("curve_fit")}
    assert all(not output_ids.intersection(values) for values in reloaded.science.working.values())
    assert not any(c.metadata.get("stale") for c in reloaded.columns[2:])
    window.model.setData(window.model.index(0, 0), "10")
    assert all(c.metadata["stale"] for c in window.model.project.columns[2:])
    window.model.undo.undo()  # Source edit.
    window.model.undo.undo()  # All six output columns, one macro.
    assert len(window.model.project.columns) == 2
    window.dirty = False


def test_save_overlay_restore_and_html_report(qtbot, monkeypatch, tmp_path):
    window, dialog = fitted_window(qtbot)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Saved line", True))
    dialog.save_current_result()
    qtbot.waitUntil(lambda: not dialog.worker.isRunning() and dialog.saved_results.count() == 1, timeout=10000)
    assert dialog.selected_records()[0]["name"] == "Saved line"
    project_path = tmp_path / "saved.autopyro"
    window.model.project.save(project_path)
    restored = Project.load(project_path)
    assert restored.view["curve_fit_results"][0]["curve"] == dialog.selected_records()[0]["curve"]
    dialog.show_saved_curves()
    assert len(dialog.figure.axes) == 1
    assert len(dialog.figure.axes[0].lines) == 1
    output = tmp_path / "report.html"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(output), ""))
    dialog.export_fit_report()
    qtbot.waitUntil(lambda: not dialog.worker.isRunning() and output.exists(), timeout=15000)
    assert "Saved line" in output.read_text(encoding="utf-8")
    dialog.restore_result_settings()
    assert dialog.result is None
    assert dialog.x_column.currentData() == window.model.project.columns[0].id
    window.model.setData(window.model.index(0, 0), "77")
    assert window.model.project.view["curve_fit_results"][0]["stale"]
    dialog.saved_results.item(0).setSelected(True)
    dialog.replay_saved_result()
    qtbot.waitUntil(lambda: dialog.result is not None and not dialog.task_running, timeout=10000)
    assert dialog.result.x[0] == 77
    assert dialog.fit_settings["replay"]["source_was_stale"]
    diff = dialog.archive_diff
    assert diff["changed_rows"] == 1
    assert diff["changes"] == [[restored.rows[0].id, "x", 0., 77.]]
    grid, archived_y, _, delta = np.asarray(diff["curve"]).T
    slope, intercept = np.polyfit(dialog.result.x, dialog.result.y, 1)
    np.testing.assert_allclose(delta, slope*grid+intercept-archived_y, atol=1e-12)
    dialog.tabs.setCurrentWidget(dialog.archive_diff_page)
    assert not dialog.tabs.tabIcon(dialog.archive_diff_tab).isNull()
    assert dialog.archive_diff_table.model().rowCount() == 1
    selected = []
    monkeypatch.setattr(dialog, "select_source_sample", selected.append)
    dialog.archive_diff_table.doubleClicked.emit(dialog.archive_diff_table.model().index(0, 0))
    assert selected == [restored.rows[0].id]
    dialog.archive_diff_choice.setCurrentIndex(dialog.archive_diff_choice.findData("curve"))
    assert dialog.archive_diff_table.model().rowCount() == len(grid)
    output = tmp_path / "archive-diff.json"
    dialog.export_result()
    assert json.loads(output.read_text(encoding="utf-8"))["archive_comparison"] == diff
    restored.rows.pop()
    window.set_project(restored)
    window.show_curve_fitting()
    dialog = window.curve_fitting_window
    dialog.saved_results.item(0).setSelected(True)
    dialog.replay_saved_result()
    from autopyro.ui.i18n import tr
    assert dialog.status.text() == tr("В проекте отсутствуют сохранённые обучающие строки: {0}", 1)
    assert not dialog.task_running
    window.dirty = False


def test_group_time_controls_and_oof_export(qtbot, monkeypatch, tmp_path):
    window, dialog = fitted_window(qtbot)
    groups = Column(name="Group", kind="text")
    time = Column(name="Time", kind="text")
    project = window.model.project
    project.columns.extend([groups, time])
    for i, row in enumerate(project.rows):
        row.values[groups.id] = str(i//5)
        row.values[time.id] = f"2026-11-{i+1:02d}"
    # Load the extended project through the normal model lifecycle.
    window.set_project(project)
    window.show_curve_fitting()
    dialog = window.curve_fitting_window
    dialog.add_candidate()
    dialog.cv_folds.setValue(3)
    dialog.cv_strategy.setCurrentIndex(dialog.cv_strategy.findData("group"))
    dialog.cv_key.setCurrentIndex(dialog.cv_key.findData(groups.id))
    dialog.calculate(comparison=True)
    qtbot.waitUntil(lambda: dialog.comparison_result is not None and not dialog.worker.isRunning(), timeout=10000)
    assert dialog.comparison_result.plan.strategy == "group"
    dialog.cv_strategy.setCurrentIndex(dialog.cv_strategy.findData("time"))
    dialog.cv_key.setCurrentIndex(dialog.cv_key.findData(time.id))
    dialog.calculate(comparison=True)
    qtbot.waitUntil(lambda: dialog.comparison_result is not None and not dialog.worker.isRunning(), timeout=10000)
    result = dialog.comparison_result
    assert result.plan.strategy == "time"
    assert np.any(result.fold_ids < 0)
    output = tmp_path / "time.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(output), ""))
    dialog.export_comparison()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert any(v is None for v in data["scores"][0]["predictions"])
    dialog.tabs.setCurrentWidget(dialog.diagnostic_page)
    assert dialog.residual_table.model().rowCount() == np.sum(result.fold_ids >= 0)
    window.dirty = False


def test_linked_bootstrap_parameter_view_and_multistart(qtbot):
    window, dialog = fitted_window(qtbot)
    dialog.parameter_rules.setText("b=0")
    dialog.lower_bounds.setText("0")
    dialog.upper_bounds.setText("4")
    dialog.starts.setValue(3)
    dialog.bootstrap_resamples.setValue(100)
    dialog.band_method.setCurrentIndex(dialog.band_method.findData("bootstrap"))
    dialog.calculate()
    qtbot.waitUntil(lambda: dialog.result is not None and not dialog.worker.isRunning(), timeout=15000)
    assert dialog.result.model.parameters == ("a",)
    assert dialog.free_parameter_label.text() == "a"
    assert dialog.result.start_count == 3
    assert dialog.parameter_table.model().rowCount() == 2
    assert dialog.bootstrap_result.parameter_intervals(dialog.result.model)[1]["lower"] == 0
    window.dirty = False
