import json
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from docx import Document
from PySide6.QtCore import Qt
from PySide6.QtGui import QAccessible
from PySide6.QtWidgets import QDialog, QFileDialog, QHeaderView, QInputDialog, QMessageBox

from autopyro.core import Column, Project, Row
from autopyro.reports.reporting import export_report
from autopyro.ui.dialogs.joint_bounds import JointBoundsDialog
from autopyro.ui.dialogs.joint_band_cache import JointBandCache
from autopyro.ui.dialogs.report import ReportDialog
from autopyro.ui.table.commands import QCCommand
from autopyro.ui.table.selection import selected_row_ids
from autopyro.window import MainWindow


def test_joint_band_cache_keeps_latest_request_and_reuses_results(qtbot):
    cache = JointBandCache()
    release = Event()
    results = []
    cache.ready.connect(lambda key, value: results.append((key, value)))

    def first():
        assert release.wait(5)
        return "first"

    cache.request((0, .95), first)
    cache.request((1, .95), lambda: "obsolete")
    cache.request((1, .90), lambda: "latest")
    release.set()
    qtbot.waitUntil(lambda: len(results) == 2)
    assert results == [((0, .95), "first"), ((1, .90), "latest")]
    assert cache.request((0, .95), lambda: "unexpected") == "first"
    assert cache.request((1, .90), lambda: "unexpected") == "latest"
    release.clear()
    cache.request((1, .95), first)
    cache.clear()
    cache.request((1, .95), lambda: "new fit")
    release.set()
    qtbot.waitUntil(lambda: len(results) == 3)
    assert results[-1] == ((1, .95), "new fit")


def test_bootstrap_robust_bands_export_and_cached_confidence(qtbot, monkeypatch, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window, before_close_func=lambda widget: setattr(widget, 'dirty', False))
    x, y = Column(name="X", kind="number"), Column(name="Y", kind="number")
    window.set_project(Project(columns=[x, y], rows=[Row(values={x.id: i, y.id: 2*i+3+(i % 3)*.2}) for i in range(25)]))
    window.show_curve_fitting()
    dialog = window.curve_fitting_window
    dialog.loss.setCurrentIndex(dialog.loss.findData("huber"))
    dialog.upper_bounds.setText("1.8, inf")
    dialog.bootstrap_resamples.setValue(100)
    dialog.bootstrap_seed.setValue(31)
    dialog.calculate()
    qtbot.waitUntil(lambda: dialog.result is not None and not dialog.worker.isRunning(), timeout=15000)
    bootstrap = dialog.bootstrap_result
    assert bootstrap is not None
    assert dialog.result.covariance is None
    assert dialog.show_confidence.isEnabled()
    assert len(dialog.figure.axes[0].collections) == 3  # CI, PI, data.
    dialog.confidence.setValue(90)
    assert dialog.bootstrap_result is bootstrap
    assert not dialog.worker.isRunning()
    output = tmp_path / "bootstrap.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(output), ""))
    dialog.export_result()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["bootstrap"]["method"] == "paired_percentile"
    assert payload["bootstrap"]["seed"] == 31
    assert len(payload["bootstrap_parameters"]) == payload["bootstrap"]["successful"]
    assert payload["bands"] is not None
    assert payload["confidence"] == .9
    with monkeypatch.context() as patch:
        grid = bootstrap.grid
        lower = np.full_like(grid, -9.5e307)
        upper = np.full_like(grid, 9.5e307)
        patch.setattr(dialog, "band_data", lambda: (grid, (dialog.result.predict(grid), lower, upper, lower, upper)))
        dialog.redraw()
        dialog.canvas.draw()
        assert "× 1e+308" in dialog.figure.axes[0].get_ylabel()
        dialog.export_result()
        assert json.loads(output.read_text(encoding="utf-8"))["bands"]["prediction_upper"][0] == upper[0]
    window.model.setData(window.model.index(0, 1), "5")
    assert dialog.bootstrap_result is None
    assert not dialog.export_button.isEnabled()
    window.dirty = False


def test_bootstrap_cancellation_and_stale_result(qtbot):
    window = MainWindow()
    qtbot.addWidget(window, before_close_func=lambda widget: setattr(widget, 'dirty', False))
    x, y = Column(name="X", kind="number"), Column(name="Y", kind="number")
    window.set_project(Project(columns=[x, y], rows=[Row(values={x.id: i, y.id: 2*i+3+(i % 3)*.2}) for i in range(30)]))
    window.show_curve_fitting()
    dialog = window.curve_fitting_window
    dialog.band_method.setCurrentIndex(dialog.band_method.findData("bootstrap"))
    dialog.bootstrap_resamples.setValue(10000)
    dialog.calculate()
    qtbot.waitUntil(lambda: dialog.cancel_bootstrap.isEnabled(), timeout=10000)
    dialog.cancel_bootstrap.click()
    qtbot.waitUntil(lambda: not dialog.worker.isRunning() and not dialog.cancel_bootstrap.isEnabled(), timeout=10000)
    assert dialog.result is None
    assert dialog.bootstrap_result is None
    dialog.bootstrap_resamples.setValue(100)
    dialog.calculate()
    dialog.invalidate()
    qtbot.waitUntil(lambda: not dialog.worker.isRunning(), timeout=10000)
    qtbot.wait(20)
    assert dialog.bootstrap_result is None
    window.dirty = False


def test_failed_bootstrap_keeps_fit_without_partial_bands(qtbot):
    window = MainWindow()
    qtbot.addWidget(window, before_close_func=lambda widget: setattr(widget, 'dirty', False))
    x, y = Column(name="X", kind="number"), Column(name="Y", kind="number")
    pairs = [(0, .9), (0, 1), (0, 1.1), (1, 3)]
    window.set_project(Project(columns=[x, y], rows=[Row(values={x.id: a, y.id: b}) for a, b in pairs]))
    window.show_curve_fitting()
    dialog = window.curve_fitting_window
    dialog.loss.setCurrentIndex(dialog.loss.findData("huber"))
    dialog.bootstrap_resamples.setValue(100)
    dialog.calculate()
    qtbot.waitUntil(lambda: dialog.result is not None and not dialog.worker.isRunning(), timeout=10000)
    # Many paired samples contain only X=0, so a slope cannot be identified.
    assert dialog.bootstrap_error
    assert dialog.bootstrap_result is None
    assert dialog.result is not None
    assert dialog.export_button.isEnabled()
    assert not dialog.show_confidence.isEnabled()
    assert len(dialog.figure.axes[0].collections) == 1  # Data only; no partial CI/PI.
    window.dirty = False


def test_robust_bounds_export_without_invalid_bands(qtbot, monkeypatch, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window, before_close_func=lambda widget: setattr(widget, 'dirty', False))
    x, y = Column(name="X", kind="number"), Column(name="Y", kind="number")
    window.set_project(Project(columns=[x, y], rows=[Row(values={x.id: i, y.id: 2*i+3}) for i in range(20)]))
    window.show_curve_fitting()
    dialog = window.curve_fitting_window
    dialog.loss.setCurrentIndex(dialog.loss.findData("huber"))
    dialog.band_method.setCurrentIndex(dialog.band_method.findData("analytic"))
    dialog.lower_bounds.setText("0, -inf")
    dialog.upper_bounds.setText("4, inf")
    dialog.calculate_button.click()
    qtbot.waitUntil(lambda: dialog.result is not None and not dialog.worker.isRunning(), timeout=10000)
    assert dialog.result.covariance is None
    assert len(dialog.figure.axes) == 2
    assert not dialog.show_confidence.isEnabled()
    path = tmp_path / "robust.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
    dialog.export_result()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["covariance"] is None
    assert payload["bands"] is None
    assert payload["settings"]["lower_bounds"] == "0, -inf"
    assert payload["loss"] == "huber"
    assert "nan" not in dialog.summary.toPlainText().lower()
    window.dirty = False


@pytest.mark.parametrize("strategy", ["kfold", "loo"])
def test_comparison_export_settings_and_invalidation(qtbot, monkeypatch, tmp_path, strategy):
    window = MainWindow()
    qtbot.addWidget(window, before_close_func=lambda widget: setattr(widget, 'dirty', False))
    x, y = Column(name="X", kind="number"), Column(name="Y", kind="number")
    project = Project(columns=[x, y], rows=[Row(values={x.id: i, y.id: i*i+2*i+1}) for i in range(30)])
    window.set_project(project)
    window.show_curve_fitting()
    dialog = window.curve_fitting_window
    dialog.add_candidate()
    dialog.model_choice.setCurrentIndex(dialog.model_choice.findData("polynomial"))
    dialog.degree.setValue(2)
    dialog.add_candidate()
    strategy_index = dialog.cv_strategy.findData(strategy)
    dialog.cv_strategy.setCurrentIndex(strategy_index)
    if strategy == "loo":
        assert not dialog.cv_strategy.itemIcon(strategy_index).isNull()
        assert not dialog.cv_folds.isEnabled() and not dialog.cv_key.isEnabled()
    dialog.calculate(comparison=True)
    qtbot.waitUntil(lambda: dialog.comparison_result is not None and not dialog.worker.isRunning(), timeout=10000)
    assert len(dialog.comparison_result.ranking) == 2
    assert "polynomial" in dialog.comparison_result.ranking[0].name
    assert dialog.comparison_table.model().rowCount() == 2
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(output), ""))
    dialog.export_comparison()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["row_ids"]) == len(payload["fold_ids"]) == 30
    assert len(payload["scores"][0]["predictions"]) == 30
    assert len(project.view["curve_fitting_comparison"]["candidates"]) == 2
    assert payload["cross_validation"]["strategy"] == strategy
    if strategy == "loo":
        assert payload["cross_validation"]["folds"] == 30
        assert "PRESS" in payload["scores"][0]["metrics"]
        assert payload["scores"][0]["fold_metrics"] == []
        dialog.cv_strategy.setCurrentIndex(dialog.cv_strategy.findData("kfold"))
    else:
        dialog.cv_seed.setValue(123)
    assert dialog.comparison_result is None
    assert not dialog.export_comparison_button.isEnabled()
    dialog.calculate(comparison=True)
    dialog.invalidate()
    qtbot.waitUntil(lambda: not dialog.worker.isRunning(), timeout=10000)
    qtbot.wait(20)
    assert dialog.comparison_result is None
    window.dirty = False


@pytest.mark.parametrize("strategy", ["nested", "nested_group", "nested_time"])
def test_nested_comparison_shows_outer_folds_and_exports_predictions(qtbot, monkeypatch, tmp_path, strategy):
    window = MainWindow()
    qtbot.addWidget(window, before_close_func=lambda widget: setattr(widget, "dirty", False))
    x, y = Column(name="X", kind="number"), Column(name="Y", kind="number")
    context = Column(name="Context", kind="text" if strategy == "nested_group" else "number")
    rows = [Row(values={x.id: i, y.id: i*i+2*i+1,
                        context.id: f"G{i//8}" if strategy == "nested_group" else i//2}) for i in range(80)]
    window.set_project(Project(columns=[x, y, context], rows=rows))
    window.show_curve_fitting()
    dialog = window.curve_fitting_window
    dialog.add_candidate()
    dialog.model_choice.setCurrentIndex(dialog.model_choice.findData("polynomial"))
    dialog.add_candidate()
    index = dialog.cv_strategy.findData(strategy)
    assert not dialog.cv_strategy.itemIcon(index).isNull()
    dialog.cv_strategy.setCurrentIndex(index)
    assert dialog.cv_inner_folds.isEnabled()
    dialog.cv_folds.setValue(3)
    dialog.cv_inner_folds.setValue(3)
    if strategy != "nested":
        assert dialog.cv_key.isEnabled()
        dialog.cv_key.setCurrentIndex(dialog.cv_key.findData(context.id))
    if strategy == "nested_time":
        assert dialog.cv_gap.isEnabled() and not dialog.cv_shuffle.isEnabled()
        dialog.cv_gap.setValue(1)
    dialog.calculate(comparison=True)
    qtbot.waitUntil(lambda: dialog.comparison_result is not None and not dialog.worker.isRunning(), timeout=10000)
    assert dialog.comparison_table.model().rowCount() == 3
    assert dialog.comparison_result.nested.metrics["RMSE"] < 1e-9
    dialog.tabs.setCurrentWidget(dialog.diagnostic_page)
    expected_count = 60 if strategy == "nested_time" else 80
    assert dialog.residual_table.model().rowCount() == expected_count
    output = tmp_path / "nested.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(output), ""))
    dialog.export_comparison()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["cross_validation"]["strategy"] == strategy
    assert payload["nested"]["inner_folds"] == 3
    assert all("polynomial" in fold["winner"] for fold in payload["nested"]["folds"])
    expected = [row.values[y.id] for row in rows]
    predictions = payload["nested"]["predictions"]
    if strategy == "nested_time":
        assert predictions[:20] == [None]*20
        assert predictions[20:] == pytest.approx(expected[20:])
        assert all(fold["gap_rows"] == 2 for fold in payload["nested"]["folds"])
    else:
        assert predictions == pytest.approx(expected)
    if strategy == "nested_group":
        assert all(fold["train_groups"]+fold["test_groups"] == 10 for fold in payload["nested"]["folds"])
    output = tmp_path / "nested.html"
    dialog.export_fit_report()
    qtbot.waitUntil(lambda: output.exists() and not dialog.worker.isRunning(), timeout=10000)
    assert "outer_folds" in output.read_text(encoding="utf-8")
    from autopyro.ui.dialogs import curve_fitting_calculation as module

    compare = module.compare_models
    calls = []

    def capture_comparison(x, y, candidates, **options):
        calls.append((len(x), options["strategy"], options["folds"]))
        return compare(x, y, candidates, **options)

    monkeypatch.setattr(module, "compare_models", capture_comparison)
    assert dialog.fit_selected_button.isEnabled() and not dialog.fit_selected_button.icon().isNull()
    dialog.fit_selected_button.click()
    qtbot.waitUntil(lambda: dialog.result is not None and not dialog.task_running, timeout=10000)
    base = {"nested": "kfold", "nested_group": "group", "nested_time": "time"}[strategy]
    assert calls == [(80, base, 3)]
    assert dialog.result.parameters == pytest.approx([1, 2, 1])
    assert dialog.fit_settings["selection"]["source_strategy"] == strategy
    assert dialog.model_choice.currentData() == "polynomial"
    window.dirty = False


def test_independent_test_roles_and_json_export(qtbot, monkeypatch, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window, before_close_func=lambda widget: setattr(widget, "dirty", False))
    x, y = Column(name="X", kind="number"), Column(name="Y", kind="number")
    sigma, role = Column(name="σY", kind="number"), Column(name="Role", kind="text")
    rows = [Row(values={x.id: i, y.id: 2*i+1, sigma.id: .5, role.id: "Train"}) for i in range(18)]
    tests = [Row(values={x.id: i, y.id: 2*i+3, sigma.id: None, role.id: "тест"}) for i in (2, 5)]
    ignored = Row(values={x.id: 100, y.id: 900, sigma.id: .5, role.id: None})
    window.set_project(Project(columns=[x, y, sigma, role], rows=[*rows, *tests, ignored]))
    window.show_curve_fitting()
    dialog = window.curve_fitting_window
    dialog.sy_column.setCurrentIndex(dialog.sy_column.findData(sigma.id))
    dialog.add_candidate()
    dialog.model_choice.setCurrentIndex(dialog.model_choice.findData("constant"))
    dialog.add_candidate()
    index = dialog.cv_holdout.findData(role.id)
    assert index > 0 and not dialog.cv_holdout.itemIcon(index).isNull()
    dialog.cv_holdout.setCurrentIndex(index)
    dialog.cv_pair[0].setCurrentIndex(1)
    dialog.cv_pair[1].setCurrentIndex(2)
    assert dialog.cv_pair_margin.isEnabled()
    dialog.cv_pair_margin.setValue(10.)
    dialog.bootstrap_resamples.setValue(100)
    dialog.confidence.setValue(90)
    dialog.calculate(comparison=True)
    qtbot.waitUntil(lambda: dialog.comparison_result is not None and not dialog.worker.isRunning(), timeout=10000)
    assert dialog.excluded_count == 1
    assert dialog.comparison_result.holdout.metrics["RMSE"] == pytest.approx(2)
    assert "Тест RMSE" in dialog.comparison_table.model().headers or "Test RMSE" in dialog.comparison_table.model().headers
    output = tmp_path / "holdout.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(output), ""))
    dialog.export_comparison()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["row_ids"] == [row.id for row in rows]
    assert payload["holdout"]["row_ids"] == [row.id for row in tests]
    assert payload["holdout"]["parameters"] == pytest.approx([2, 1])
    assert payload["holdout"]["predictions"] == pytest.approx([5, 11])
    assert payload["cross_validation"]["holdout_key"] == role.id
    assert payload["holdout"]["pair"]["difference_B_minus_A"] == pytest.approx([9, 3])
    assert payload["holdout"]["pair"]["metrics"]["N_common"] == 2
    assert payload["holdout"]["pair"]["interval"]["confidence"] == .9
    assert payload["holdout"]["pair"]["interval"]["direction"] == "a"
    assert payload["holdout"]["pair"]["interval"]["practical"] == {"margin": 10., "conclusion": "within_margin"}
    assert payload["cross_validation"]["holdout_margin"] == 10.
    assert dialog.tabs.isTabEnabled(dialog.holdout_pair_tab)
    dialog.tabs.setCurrentWidget(dialog.holdout_pair_page)
    table = dialog.holdout_pair_table
    table.sortByColumn(7, Qt.SortOrder.AscendingOrder)
    model = table.model()
    assert [row[0] for row in model.rows] == [tests[1].id, tests[0].id]
    selected = []
    monkeypatch.setattr(dialog, "select_source_sample", selected.append)
    table.doubleClicked.emit(model.index(0, 0))
    assert selected == [tests[1].id]
    assert "90%" in dialog.holdout_pair_summary.text()
    assert "±10" in dialog.holdout_pair_summary.text()
    output = tmp_path / "holdout.html"
    dialog.export_fit_report()
    qtbot.waitUntil(lambda: output.exists() and not dialog.worker.isRunning(), timeout=10000)
    assert "±10" in output.read_text(encoding="utf-8")
    dialog.tabs.setCurrentWidget(dialog.comparison_table)
    dialog.comparison_table.setCurrentIndex(dialog.comparison_table.model().index(0, 0))
    dialog.fit_selected_button.click()
    qtbot.waitUntil(lambda: dialog.result is not None and not dialog.task_running, timeout=10000)
    assert dialog.row_ids == [row.id for row in rows]
    assert dialog.result.parameters == pytest.approx([2, 1])
    assert dialog.model_choice.currentData() == "linear"
    assert dialog.fit_settings["selection"]["holdout_column"] == role.id
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("CV fit", True))
    dialog.save_current_result()
    qtbot.waitUntil(lambda: not dialog.task_running and dialog.saved_results.count() == 1, timeout=10000)
    archive = window.model.project.view["curve_fit_results"][0]
    project_path = tmp_path / "cv-fit.autopyro"
    window.model.project.save(project_path)
    dialog.apply_current_fit()
    qtbot.waitUntil(lambda: not dialog.task_running and len(window.model.project.columns) == 10, timeout=10000)
    run = next(iter(window.model.project.view["curve_fit_runs"].values()))
    assert run["training_rows"] == [row.id for row in rows]
    assert run["settings"]["selection"] == dialog.fit_settings["selection"]
    assert role.id in run["source_ids"]
    predicted_column = next(c for c in window.model.project.columns
                            if c.metadata.get("curve_fit", {}).get("output") == "prediction")
    assert [row.values[predicted_column.id] for row in window.model.project.rows if row.id in {test.id for test in tests}] == pytest.approx([5, 11])
    reopened = Project.load(project_path)
    reopened.rows.reverse()
    window.set_project(reopened)
    window.show_curve_fitting()
    dialog = window.curve_fitting_window
    dialog.saved_results.item(0).setSelected(True)
    dialog.replay_saved_result()
    qtbot.waitUntil(lambda: dialog.result is not None and not dialog.task_running, timeout=10000)
    assert dialog.row_ids == [row.id for row in rows]
    assert dialog.result.parameters == pytest.approx([2, 1])
    assert dialog.fit_settings["replay"]["archive_id"] == archive["id"]
    assert dialog.archive_diff["changes"] == []
    assert dialog.archive_diff["peak"]["delta"] == pytest.approx(0)
    training_position = next(i for i, row in enumerate(window.model.project.rows) if row.id == rows[0].id)
    assert window.model.setData(window.model.index(training_position, 3), "Test")
    dialog.saved_results.item(0).setSelected(True)
    dialog.replay_saved_result()
    from autopyro.ui.i18n import tr
    assert dialog.status.text() == tr("У сохранённых обучающих строк изменились роли выборки")
    assert not dialog.task_running
    window.dirty = False


@pytest.mark.parametrize("loss", ["linear", "huber"])
def test_joint_fit_series_plot_and_export(qtbot, tmp_path, monkeypatch, loss):
    window = MainWindow()
    qtbot.addWidget(window, before_close_func=lambda widget: setattr(widget, "dirty", False))
    x = Column(name=r"X $\unknowncommand$", kind="number")
    y = Column(name=r"Y $\unknowncommand$", kind="number")
    series = Column(name="Well", kind="text")
    rows = [Row(values={x.id: i, y.id: 2*i+offset, series.id: name})
            for name, offset in (("A", 1), ("B", 5)) for i in range(12)]
    missing_x = Row(values={x.id: None, y.id: 99, series.id: "B"})
    window.set_project(Project(columns=[x, y, series], rows=[*rows, missing_x]))
    window.show_curve_fitting()
    dialog = window.curve_fitting_window
    dialog.joint_column.setCurrentIndex(dialog.joint_column.findData(series.id))
    dialog.loss.setCurrentIndex(dialog.loss.findData(loss))
    dialog.bootstrap_resamples.setValue(100)
    dialog.method.setCurrentIndex(dialog.method.findData("odr"))
    assert not dialog.joint_run_button.isEnabled()
    assert "OLS/WLS" in dialog.joint_compatibility.text()
    dialog.method.setCurrentIndex(dialog.method.findData("ols"))
    dialog.parameter_rules.setText("b=0")
    assert not dialog.joint_run_button.isEnabled()
    dialog.parameter_rules.clear()
    assert dialog.joint_run_button.isEnabled()
    dialog.band_method.setCurrentIndex(dialog.band_method.findData("analytic"))
    assert "недоступны" in dialog.joint_band_note.text()
    dialog.band_method.setCurrentIndex(dialog.band_method.findData("auto"))
    dialog.starts.setValue(3)
    assert not dialog.joint_run_button.icon().isNull()
    from autopyro.ui.dialogs import curve_analysis

    announcements = []
    monkeypatch.setattr(curve_analysis, "QAccessible", SimpleNamespace(
        isActive=lambda: True,
        updateAccessibility=lambda event: announcements.append((event.object(), event.message())),
    ))
    dialog.joint_run_button.click()
    qtbot.waitUntil(lambda: dialog.joint_result is not None and not dialog.task_running, timeout=10000)
    assert announcements == [(dialog.status, "Серий: 2 · Точек: 24 · Исключено: 1")]
    assert dialog.joint_result.coefficients.tolist() == [pytest.approx([2, 1]), pytest.approx([2, 5])]
    assert dialog.joint_exclusion_table.model() is None
    assert dialog.joint_exclusion_button.isEnabled()
    dialog.joint_exclusion_button.click()
    exclusions = dialog.joint_exclusion_table.model()
    assert dialog.joint_exclusion_table.currentIndex().row() == 0
    assert exclusions.index(0, 0).data() == missing_x.id
    assert exclusions.index(0, 1).data() == "X · Пропуск"
    dialog.tabs.setCurrentIndex(dialog.joint_tab)
    assert dialog.joint_table.model().rowCount() == 2
    for action in (dialog.joint_columns_button, dialog.joint_report_button, dialog.joint_exclusion_button):
        assert action.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonIconOnly
        assert action.accessibleName() == action.toolTip() == action.text()
        assert action.minimumSizeHint().width() < action.fontMetrics().horizontalAdvance(action.text())
    fit = dialog.joint_result
    dialog.joint_cv_button.click()
    qtbot.waitUntil(lambda: dialog.joint_cv_table.model() is not None and not dialog.task_running)
    assert announcements[1:] == [
        (dialog.status, "Вариантов совместной подгонки проверено: 2\nУспешно: 2 · С ошибками: 0"),
    ]
    assert dialog.joint_cv_table.model().index(0, 0).data() == "a"
    assert QAccessible.queryAccessibleInterface(dialog.joint_cv_table).text(QAccessible.Text.Name) == "CV совместной подгонки"
    assert dialog.joint_result is fit
    if loss == "linear":
        from time import monotonic, sleep
        from autopyro.ui.dialogs import curve_joint_analysis

        def wait_for_cancel(*args, progress, cancelled, **kwargs):
            progress(1)
            deadline = monotonic()+5
            while not cancelled() and monotonic() < deadline:
                sleep(.01)
            return None

        with monkeypatch.context() as patch:
            patch.setattr(curve_joint_analysis, "compare_joint_sharing", wait_for_cancel)
            dialog.joint_cv_button.click()
            qtbot.waitUntil(lambda: dialog.status.text() == "Совместная CV: блок 1 из 10")
            assert dialog.joint_cv_cancel_button.isEnabled()
            assert not dialog.joint_cv_cancel_button.icon().isNull()
            dialog.joint_cv_cancel_button.click()
            qtbot.waitUntil(lambda: not dialog.task_running)
            assert dialog.status.text() == "Сравнение отменено"
            assert dialog.joint_cv_table.model() is None
            assert dialog.joint_result is fit
    dialog.tabs.setCurrentIndex(dialog.joint_tab)
    dialog.joint_table.setCurrentIndex(dialog.joint_table.model().index(1, 0))
    assert dialog.joint_series.currentData() == 1
    assert dialog.joint_figure.axes[0].lines[0].get_ydata()[0] == pytest.approx(5)
    plots = dialog.joint_plots
    assert not plots.tabIcon(plots.overview_tab).isNull()
    assert not plots.overview.figure.axes and not plots.residuals.figure.axes
    plots.setCurrentIndex(plots.overview_tab)
    plots.overview.draw()
    assert plots.overview.figure.axes[0].get_xlabel() == x.name
    assert plots.overview.figure.axes[0].get_ylabel() == y.name
    if loss == "linear":
        highlighted = plots.series
        segment = plots.segments[highlighted]
        plots.segments[highlighted] = segment[:0]
        plots.draw_overview()
        assert str(fit.labels[highlighted]) in plots.overview_warning.text()
        assert plots.overview.accessibleDescription() == plots.overview_warning.text()
        plots.segments[highlighted] = segment
        plots.draw_overview()
        assert not plots.overview.accessibleDescription()
    segments = plots.segments
    assert len(segments) == 2
    assert plots.highlight.get_ydata()[0] == pytest.approx(5)
    dialog.joint_series.setCurrentIndex(0)
    assert dialog.joint_table.currentIndex().row() == 0
    assert plots.segments is segments
    assert plots.highlight.get_ydata()[0] == pytest.approx(1)
    dialog.joint_series.setCurrentIndex(1)
    plots.setCurrentIndex(plots.residual_tab)
    assert plots.residual_points.get_offsets()[:, 1].tolist() == pytest.approx([0]*12, abs=1e-8)
    plots.pick_residual(SimpleNamespace(artist=plots.residual_points, ind=[1]))
    assert selected_row_ids(window) == {rows[13].id}
    residual_table = plots.residual_table
    residual_model = residual_table.model()
    assert residual_table.accessibleName() == "Остатки по серии"
    accessible = QAccessible.queryAccessibleInterface(residual_table)
    assert accessible.role() == QAccessible.Role.Table
    assert "Enter" in accessible.text(QAccessible.Text.Description)
    assert QAccessible.queryAccessibleInterface(plots.residuals).text(QAccessible.Text.Name) == "График остатков по серии"
    assert residual_model.rowCount() == 12
    assert residual_model.index(0, 0).data() == rows[12].id
    assert residual_model.index(0, 0).data(Qt.ItemDataRole.ToolTipRole) == rows[12].id
    assert all(residual_table.horizontalHeader().sectionResizeMode(i) == QHeaderView.ResizeMode.Stretch for i in range(3))
    residual_table.setCurrentIndex(residual_model.index(1, 0))
    qtbot.keyClick(residual_table, Qt.Key.Key_Up)
    qtbot.keyClick(residual_table, Qt.Key.Key_Return)
    assert residual_table.currentIndex().row() == 0
    assert selected_row_ids(window) == {rows[12].id}
    assert window.proxy.mapToSource(window.table.currentIndex()).row() == 12
    plots.setCurrentIndex(0)
    dialog.joint_canvas.draw()
    assert dialog.joint_figure.axes[0].get_xlabel() == x.name
    assert dialog.joint_figure.axes[0].get_ylabel() == y.name
    prediction_warning = "Некорректные координаты прогноза"
    status_before_prediction_error = dialog.status.text()

    def fail_prediction(*args, **kwargs):
        raise ValueError(prediction_warning)

    with monkeypatch.context() as patch:
        patch.setattr(type(dialog.joint_result), "predict", fail_prediction)
        dialog.draw_joint_series()
        assert dialog.status.text() == prediction_warning
        assert announcements[-1] == (dialog.status, prediction_warning)
        assert not dialog.joint_figure.axes[0].lines
        dialog.draw_joint_series()
        assert dialog.status.text() == prediction_warning

    dialog.draw_joint_series()
    assert len(dialog.joint_figure.axes[0].lines) == 1
    assert dialog.status.text() == status_before_prediction_error
    assert announcements[-1] == (dialog.status, status_before_prediction_error)
    if loss == "huber":
        qtbot.waitUntil(lambda: len(dialog.joint_figure.axes[0].collections) == 3)
        bootstrap = dialog.joint_bootstrap
        warning = "Численное переполнение bootstrap Prediction; измените масштаб данных"
        previous_status = dialog.status.text()

        def fail_bands(*args, **kwargs):
            raise ValueError(warning)

        with monkeypatch.context() as patch:
            patch.setattr(type(bootstrap), "bands", fail_bands)
            dialog.confidence.setValue(91)
            qtbot.waitUntil(lambda: dialog.status.text() == warning)
            assert announcements[-1] == (dialog.status, warning)
            assert not dialog.joint_figure.axes[0].texts

        dialog.show_confidence.setChecked(False)
        dialog.show_prediction.setChecked(False)
        assert dialog.status.text() == previous_status
        assert announcements[-1] == (dialog.status, previous_status)
        assert len(dialog.joint_figure.axes[0].collections) == 1
        dialog.show_confidence.setChecked(True)
        assert dialog.status.text() == warning
        dialog.show_prediction.setChecked(True)

        def extreme_bands(_bootstrap, result, series, grid, **kwargs):
            lower = np.full_like(grid, -9.5e307)
            upper = np.full_like(grid, 9.5e307)
            return result.predict(series, grid), lower, upper, lower, upper

        with monkeypatch.context() as patch:
            patch.setattr(type(bootstrap), "bands", extreme_bands)
            dialog.confidence.setValue(92)
            qtbot.waitUntil(lambda: len(dialog.joint_figure.axes[0].collections) == 3)
            dialog.joint_canvas.draw()
            assert "× 1e+308" in dialog.joint_figure.axes[0].get_ylabel()
            dialog.draw_joint_series()
            assert "× 1e+308" in dialog.joint_figure.axes[0].get_ylabel()

        dialog.confidence.setValue(90)
        qtbot.waitUntil(lambda: len(dialog.joint_figure.axes[0].collections) == 3)
        assert dialog.status.text() == previous_status
        assert announcements[-1] == (dialog.status, previous_status)
        assert dialog.joint_bootstrap is bootstrap
    output = tmp_path / "joint.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(output), ""))
    dialog.joint_export_button.click()
    qtbot.waitUntil(lambda: output.exists() and not dialog.task_running, timeout=10000)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["settings"]["shared"] == ["a"]
    assert "starts" not in payload["settings"] and "bootstrap_group" not in payload["settings"]
    assert payload["band_method_actual"] == ("bootstrap" if loss == "huber" else None)
    assert ("Полосы не рассчитаны" in dialog.joint_summary.text()) == (loss == "linear")
    assert payload["row_ids"] == [row.id for row in rows]
    assert payload["series_ids"] == [0]*12 + [1]*12
    if loss == "huber":
        assert payload["bootstrap"]["successful"] == 100
        assert payload["bootstrap"]["warnings"]
        assert payload["bands"]["series"] == 1 and payload["bands"]["confidence"] == .9
        assert not dialog.joint_report_button.icon().isNull()
        dialog.joint_report_button.click()
        qtbot.waitUntil(lambda: "joint_curve_report" in window.model.project.view and not dialog.task_running)
        first_snapshot = window.model.project.view["joint_curve_report"]["id"]
        dialog.confidence.setValue(95)
        dialog.joint_report_button.click()
        qtbot.waitUntil(lambda: first_snapshot in window.model.project.view.get("joint_curve_reports", {})
                        and not dialog.task_running)
    assert not dialog.joint_columns_button.icon().isNull()
    dialog.joint_columns_button.click()
    qtbot.waitUntil(lambda: len(window.model.project.columns) == 5 and not dialog.task_running)
    prediction, residual = window.model.project.columns[-2:]
    assert prediction.metadata["joint_curve_fit"]["settings"]["column"] == series.id
    assert prediction.metadata["joint_curve_fit"]["settings"]["loss"] == loss
    assert [row.values[prediction.id] for row in rows] == pytest.approx([row.values[y.id] for row in rows])
    assert [row.values[residual.id] for row in rows] == pytest.approx([0]*len(rows), abs=1e-8)
    assert missing_x.values[prediction.id] is None and missing_x.values[residual.id] is None
    assert prediction.metadata["source_status"] == residual.metadata["source_status"] == "current"
    project = window.model.project
    measurement = project.science.measurements[project.science.working[rows[0].id][y.id]]
    window.model.undo.push(QCCommand(window.model, ("qc_exclusions", "statistics"),
                                    {measurement.id: {"revision": measurement.revision}}))
    assert prediction.metadata["source_status"] == "stale"
    window.model.undo.undo()
    assert prediction.metadata["source_status"] == "current"
    assert window.model.setData(window.model.index(0, 2), "C")
    assert prediction.metadata["source_status"] == "stale"
    window.model.undo.undo()
    assert prediction.metadata["source_status"] == "current"
    assert window.model.setData(window.model.index(0, 1), "10")
    assert prediction.metadata["source_status"] == residual.metadata["source_status"] == "stale"
    assert "устарел" in window.model.headerData(3, Qt.Orientation.Horizontal)
    assert not window.model.headerData(3, Qt.Orientation.Horizontal, Qt.ItemDataRole.DecorationRole).isNull()
    assert dialog.joint_result is None and not dialog.joint_export_button.isEnabled()
    assert not dialog.joint_columns_button.isEnabled()
    if loss == "huber":
        report = ReportDialog(window)
        qtbot.addWidget(report)
        assert report.joint_snapshot.count() == 2
        assert report.joint_snapshot.currentData() == project.view["joint_curve_report"]["id"]
        assert not report.joint_snapshot.itemIcon(0).isNull()
        report.joint_snapshot.setCurrentIndex(report.joint_snapshot.findData(first_snapshot))
        report.scope.setCurrentIndex(report.scope.findData("all"))
        report.build()
        assert report.report is not None, report.status.text()
        fit = report.report["joint_curve_fit"]
        assert fit["id"] == first_snapshot
        assert fit["source_status"] == "stale"
        assert "устарел" in dialog.joint_snapshot_warning.text()
        assert {"stale_joint_fit", "stale_joint_column"} <= {check["code"] for check in report.report["release_checks"]}
        assert fit["series"] == ["A", "B"] and fit["counts"] == [12, 12]
        assert fit["bootstrap"]["successful"] == 100
        assert fit["plots"]["confidence"] == .9
        first, second = fit["plots"]["series"]
        for plot in (first, second):
            assert len(plot["grid"]) == 200
            assert len(plot["confidence_band"][0]) == len(plot["prediction_band"][1]) == 200
        assert second["confidence_band"][0] == payload["bands"]["confidence_lower"]
        assert second["prediction_band"][1] == payload["bands"]["prediction_upper"]
        assert [figure["name"] for figure in report.report["joint_curve_figures"]] == ["Серия 1 · A", "Серия 2 · B"]
        assert "Серия 2 · B" in report.preview.toPlainText()
        assert "Подгонка по сериям" in report.preview.toPlainText()
        assert "Мало повторов в хвостах распределения" in report.preview.toPlainText()
        report.report["options"]["language"] = "en"
        docx = tmp_path / "joint.docx"
        export_report(docx, report.report, "docx")
        document = Document(docx)
        assert len(document.inline_shapes) == 2
        assert any(paragraph.text.endswith("Fit by series") for paragraph in document.paragraphs)
        assert any(table.cell(0, 0).text == "Series" for table in document.tables)
        report.scope.setCurrentIndex(report.scope.findData("selected"))
        report.build()
        assert "joint_curve_fit" not in report.report
        assert "joint_curve_figures" not in report.report
        assert report.report["row_ids"] == [rows[12].id]
        assert report.report["joint_curve_omission"] == {
            "reason": "outside_scope", "missing_rows": 23, "snapshot_id": first_snapshot,
        }
        assert any(check["code"] == "joint_fit_outside_scope" for check in report.report["release_checks"])
    window.model.undo.undo()
    assert prediction.metadata["source_status"] == "current"
    window.model.undo.undo()
    assert len(window.model.project.columns) == 3


def test_joint_fit_announces_bootstrap_failure_with_available_coefficients(qtbot, monkeypatch):
    from autopyro.ui.dialogs import curve_analysis, curve_joint_analysis

    window = MainWindow()
    qtbot.addWidget(window, before_close_func=lambda widget: setattr(widget, "dirty", False))
    x, y, series = Column("X", "number"), Column("Y", "number"), Column("Series", "text")
    rows = [Row({x.id: i, y.id: 2*i+offset, series.id: name})
            for name, offset in (("A", 1), ("B", 5)) for i in range(6)]
    window.set_project(Project(columns=[x, y, series], rows=rows))
    window.show_curve_fitting()
    dialog = window.curve_fitting_window
    dialog.joint_column.setCurrentIndex(dialog.joint_column.findData(series.id))
    dialog.band_method.setCurrentIndex(dialog.band_method.findData("bootstrap"))
    warning = "Недостаточно успешных повторов совместного bootstrap"

    def fail_bootstrap(*args, **kwargs):
        raise ValueError(warning)

    announcements = []
    monkeypatch.setattr(curve_joint_analysis, "bootstrap_joint_fit", fail_bootstrap)
    monkeypatch.setattr(curve_analysis, "QAccessible", SimpleNamespace(
        isActive=lambda: True, updateAccessibility=lambda event: announcements.append(event.message()),
    ))
    dialog.joint_run_button.click()
    qtbot.waitUntil(lambda: dialog.joint_result is not None and not dialog.task_running)
    assert dialog.joint_table.model().rowCount() == 2
    assert not dialog.show_prediction.isEnabled()
    assert announcements == [f"Серий: 2 · Точек: 12 · Исключено: 0\n{warning}"]
    assert dialog.status.text() == announcements[0]


def test_joint_exclusions_explain_failures_and_open_original_row(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window, before_close_func=lambda widget: setattr(widget, "dirty", False))
    x, y, sy = (Column(name, "number") for name in ("X", "Y", "sigma"))
    series = Column("Series", "text")
    rows = [Row({x.id: None, y.id: "ND", sy.id: 0, series.id: " "}),
            Row({x.id: 1, y.id: "<3", sy.id: 1, series.id: "B"})]
    window.set_project(Project(columns=[x, y, sy, series], rows=rows))
    window.show()
    window.show_curve_fitting()
    dialog = window.curve_fitting_window
    dialog.sy_column.setCurrentIndex(dialog.sy_column.findData(sy.id))
    dialog.joint_column.setCurrentIndex(dialog.joint_column.findData(series.id))
    measurement = window.model.project.science.measurements[window.model.project.science.working[rows[1].id][x.id]]
    window.model.undo.push(QCCommand(window.model, ("qc_exclusions", "statistics"),
                                    {measurement.id: {"revision": measurement.revision}}))
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: warnings.append(text))
    dialog.joint_run_button.click()
    qtbot.waitUntil(lambda: bool(warnings) and not dialog.task_running)
    assert dialog.joint_result is None
    table = dialog.joint_exclusion_table
    model = table.model()
    assert model.rowCount() == 2
    assert table.columnWidth(0) == 120
    assert model.index(0, 0).data(Qt.ItemDataRole.ToolTipRole) == rows[0].id
    assert model.index(0, 1).data() == (
        "X · Пропуск / Y · Ниже предела обнаружения (ND) / "
        "σY · Требуется положительное значение / Серия · Пропуск"
    )
    assert model.index(1, 1).data() == "Y · Цензурированное значение / Исключено QC"
    assert not dialog.joint_exclusion_open.icon().isNull()
    table.setCurrentIndex(model.index(1, 0))
    assert table.accessibleName() == "Исключённые строки"
    qtbot.keyClick(table, Qt.Key.Key_Return)
    assert selected_row_ids(window) == {rows[1].id}
    assert window.proxy.mapToSource(window.table.currentIndex()).row() == 1
    assert window.focusWidget() is window.table
    window.proxy.set_rules([], row_ids=[rows[1].id])
    assert selected_row_ids(window) == {rows[1].id}
    from autopyro.ui.dialogs import curve_analysis

    announcements = []
    monkeypatch.setattr(curve_analysis, "QAccessible", SimpleNamespace(
        isActive=lambda: True,
        updateAccessibility=lambda event: announcements.append((event.object(), event.message())),
    ))
    dialog.select_source_sample(rows[0].id)
    assert selected_row_ids(window) == {rows[1].id}
    assert dialog.status.text() == "Проба скрыта фильтром исходной таблицы"
    assert announcements == [(dialog.status, dialog.status.text())]
    dialog.sy_column.setCurrentIndex(0)
    assert not dialog.tabs.isTabEnabled(dialog.joint_exclusion_tab)


def test_joint_nonlinear_bounds_controls_and_export(qtbot, tmp_path, monkeypatch):
    import math

    window = MainWindow()
    qtbot.addWidget(window, before_close_func=lambda widget: setattr(widget, "dirty", False))
    x, y = Column(name="X", kind="number"), Column(name="Y", kind="number")
    series = Column(name="Series", kind="text")
    rows = [Row(values={x.id: i/5, y.id: amplitude*math.exp(.4*i/5)+1, series.id: label})
            for label, amplitude in (("A", 2.), ("B", 3.)) for i in range(16)]
    project = Project(columns=[x, y, series], rows=rows)
    project.view["joint_curve_fitting"] = {"column": series.id, "shared": ["b"], "series_bounds": '{"A":{"a":[0,2.6]}}'}
    window.set_project(project)
    window.show_curve_fitting()
    dialog = window.curve_fitting_window
    dialog.model_choice.setCurrentIndex(dialog.model_choice.findData("exponential"))
    dialog.joint_parameters.item(0).setCheckState(Qt.CheckState.Unchecked)
    dialog.joint_parameters.item(1).setCheckState(Qt.CheckState.Checked)
    dialog.joint_column.setCurrentIndex(dialog.joint_column.findData(series.id))
    dialog.initial.setText("2, 0.3, 1")
    dialog.lower_bounds.setText("0, 0, -inf")
    dialog.upper_bounds.setText("inf, 0.6, inf")
    def edit_bounds(editor):
        assert QAccessible.queryAccessibleInterface(editor.table).text(QAccessible.Text.Name) == "Границы по сериям"
        assert editor.series.currentText() == "A"
        assert editor.table.item(0, 3).text() == "2.6"
        editor.table.item(0, 3).setText("-1")
        editor.accept()
        assert editor.error.text()
        editor.table.item(0, 3).setText("2.5")
        editor.series.setCurrentIndex(1)
        assert editor.table.item(0, 1).checkState() == Qt.CheckState.Unchecked
        assert editor.table.item(0, 2).text() == "0.0"
        assert not editor.table.item(0, 2).flags() & Qt.ItemFlag.ItemIsEditable
        editor.accept()
        return editor.result()

    monkeypatch.setattr(JointBoundsDialog, "exec", edit_bounds)
    assert not dialog.joint_bounds_button.icon().isNull()
    dialog.joint_bounds_button.click()
    assert json.loads(project.view["joint_curve_fitting"]["series_bounds"]) == {"A": {"a": [0, 2.5]}}
    dialog.joint_run_button.click()
    qtbot.waitUntil(lambda: dialog.joint_result is not None and not dialog.task_running, timeout=10000)
    assert dialog.joint_result.coefficients.tolist() == [pytest.approx([2, .4, 1]), pytest.approx([3, .4, 1])]
    output = tmp_path / "bounded-joint.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(output), ""))
    dialog.joint_export_button.click()
    qtbot.waitUntil(lambda: output.exists() and not dialog.task_running, timeout=10000)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["settings"]["shared"] == ["b"]
    assert payload["bounds"][1] == [[2.5, .6, None], [None, .6, None]]
    assert payload["evaluations"] > 0

    dialog.model_choice.setCurrentIndex(dialog.model_choice.findData("custom"))
    dialog.expression.setText("a*exp(b*x)+c+0*d")
    dialog.parameter_names.setText("a, b, c, d")
    dialog.initial.setText("2, 0.4, 1, 10")
    dialog.lower_bounds.clear()
    dialog.upper_bounds.clear()
    def reset_bounds(editor):
        editor.reset()
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(JointBoundsDialog, "exec", reset_bounds)
    dialog.joint_bounds_button.click()
    assert json.loads(project.view["joint_curve_fitting"]["series_bounds"]) == {"A": {"a": [0, 2.5]}}
    monkeypatch.setattr(JointBoundsDialog, "exec", lambda editor: QDialog.DialogCode.Accepted)
    dialog.joint_bounds_button.click()
    assert not project.view["joint_curve_fitting"]["series_bounds"]
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: warnings.append(text))
    dialog.joint_run_button.click()
    qtbot.waitUntil(lambda: bool(warnings) and not dialog.task_running)
    assert "Отдельные коэффициенты" in warnings[0]
    assert dialog.status.text() == warnings[0]
    assert dialog.joint_result is None and not dialog.joint_report_button.isEnabled()


def test_fit_document_scope_export_and_invalidation(qtbot, tmp_path, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window, before_close_func=lambda widget: setattr(widget, 'dirty', False))
    x, y = Column(name="X", kind="number"), Column(name="Y", kind="number")
    rows = [Row(values={x.id: i, y.id: 2*i+3+(.1 if i % 2 else -.1)}) for i in range(12)]
    rows.append(Row(values={x.id: 13, y.id: "<LOD"}))
    window.set_project(Project(columns=[x, y], rows=rows))
    window.show_curve_fitting()
    dialog = window.curve_fitting_window
    assert window.desktop.properties_stack.currentWidget() is dialog.parameter_tabs
    dialog.calculate()
    qtbot.waitUntil(lambda: dialog.result is not None and not dialog.worker.isRunning(), timeout=15000)
    assert len(dialog.row_ids) == 12
    assert "1" in dialog.status.text()
    assert len(dialog.figure.axes) == 2
    output = tmp_path / "fit.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(output), ""))
    dialog.export_result()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metrics"]["N"] == 12
    assert len(payload["bands"]["mean"]) == 400
    assert payload["row_ids"] == dialog.row_ids
    window.model.setData(window.model.index(0, 1), "4")
    assert dialog.result is None
    assert not dialog.export_button.isEnabled()
    assert not dialog.figure.axes
    window.dirty = False


def test_custom_model_and_stale_worker_result(qtbot):
    window = MainWindow()
    qtbot.addWidget(window, before_close_func=lambda widget: setattr(widget, 'dirty', False))
    x, y = Column(name="X", kind="number"), Column(name="Y", kind="number")
    window.set_project(Project(columns=[x, y], rows=[Row(values={x.id: i, y.id: 3*i+2}) for i in range(8)]))
    window.show_curve_fitting()
    dialog = window.curve_fitting_window
    dialog.model_choice.setCurrentIndex(dialog.model_choice.findData("custom"))
    dialog.initial.setText("1, 1")
    dialog.calculate()
    dialog.invalidate()
    qtbot.waitUntil(lambda: not dialog.worker.isRunning(), timeout=10000)
    qtbot.wait(20)
    assert dialog.result is None
    dialog.calculate()
    qtbot.waitUntil(lambda: dialog.result is not None and not dialog.worker.isRunning(), timeout=10000)
    assert abs(dialog.result.parameters[0]-3) < 1e-6
    window.set_project(Project())
    assert window.curve_fitting_window is None
    window.dirty = False


def test_model_library_and_filtered_scope(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window, before_close_func=lambda widget: setattr(widget, 'dirty', False))
    x, y = Column(name="X", kind="number"), Column(name="Y", kind="number")
    window.set_project(Project(columns=[x, y], rows=[Row(values={x.id: i, y.id: 3*i+2}) for i in range(10)]))
    window.show_curve_fitting()
    dialog = window.curve_fitting_window
    dialog.initial.setText("1, 1")
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("My line", True))
    dialog.save_custom_model()
    assert "My line" in window.model.project.view["curve_fitting_models"]
    dialog.expression.setText("a*x-b")
    dialog.saved_models.setCurrentIndex(0)
    dialog.saved_models.setCurrentIndex(dialog.saved_models.findData("My line"))
    assert dialog.expression.text() == "a*x + b"
    window.apply_filter([{"column": x.id, "operator": "ge", "value": 5}], "AND")
    dialog.calculate()
    qtbot.waitUntil(lambda: dialog.result is not None and not dialog.worker.isRunning(), timeout=10000)
    assert len(dialog.row_ids) == 5
    window.analysis_scope.set_mode("all")
    assert dialog.result is None
    dialog.calculate()
    qtbot.waitUntil(lambda: dialog.result is not None and not dialog.worker.isRunning(), timeout=10000)
    assert len(dialog.row_ids) == 10
    window.dirty = False
