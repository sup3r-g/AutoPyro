import re

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QInputDialog, QPushButton

from autopyro.core import Column, Project, Row
from autopyro.ui.window import MainWindow


def window_with_data(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    x, y = Column(name="X", kind="number"), Column(name="Y", kind="number")
    window.set_project(Project(columns=[x, y], rows=[Row(values={x.id: i, y.id: 2*i+1+.1*(i % 3)}) for i in range(25)]))
    window.show_curve_fitting()
    return window, window.curve_fitting_window


def wait_fit(qtbot, dialog):
    qtbot.waitUntil(lambda: dialog.result is not None and not dialog.worker.isRunning(), timeout=15000)


def test_archived_view_cannot_export_or_apply_a_hidden_current_fit(qtbot, monkeypatch):
    window, dialog = window_with_data(qtbot)
    dialog.calculate()
    wait_fit(qtbot, dialog)
    current = dialog.result
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Archived", True))
    dialog.save_current_result()
    qtbot.waitUntil(lambda: dialog.saved_results.count() == 1 and not dialog.worker.isRunning(), timeout=10000)
    dialog.show_saved_curves()
    assert not dialog.export_button.isEnabled()
    assert not dialog.apply_fit_button.isEnabled()
    assert not dialog.confidence.isEnabled()
    dialog.show_confidence.setChecked(False)
    assert len(dialog.figure.axes) == 1
    dialog.apply_current_fit()
    assert len(window.model.project.columns) == 2
    dialog.show_current_fit()
    assert dialog.result is current
    assert dialog.export_button.isEnabled() and dialog.confidence.isEnabled()
    assert len(dialog.figure.axes) == 2
    dialog.show_saved_curves()
    dialog.remove_saved_results()
    assert dialog._archive_records is None
    assert dialog.result is current and dialog.export_button.isEnabled()
    window.dirty = False


def test_influence_ranks_points_and_links_to_source_rows(qtbot, monkeypatch):
    window, dialog = window_with_data(qtbot)
    dialog.calculate()
    wait_fit(qtbot, dialog)
    assert dialog.tabs.isTabEnabled(dialog.influence_tab)
    assert dialog.influence_table.model() is None
    dialog.tabs.setCurrentWidget(dialog.influence_page)
    model = dialog.influence_table.model()
    distances = [row[-1] for row in model.rows]
    assert distances == sorted(distances, reverse=True)
    press = sum(row[4]**2 for row in model.rows)
    assert f"PRESS {press:.6g}" in dialog.influence_summary.text()
    selected = []
    monkeypatch.setattr(dialog, "select_source_sample", selected.append)
    dialog.influence_table.doubleClicked.emit(model.index(0, 0))
    assert selected == [model.rows[0][0]]
    dialog.loss.setCurrentIndex(dialog.loss.findData("huber"))
    dialog.calculate()
    wait_fit(qtbot, dialog)
    assert not dialog.tabs.isTabEnabled(dialog.influence_tab)
    assert dialog.influence_table.model() is None
    window.dirty = False


def test_autocorrelation_is_drawn_on_demand_and_invalidated_with_fit(qtbot):
    window, dialog = window_with_data(qtbot)
    assert not dialog.tabs.isTabEnabled(dialog.autocorrelation_tab)
    dialog.calculate()
    wait_fit(qtbot, dialog)
    assert dialog.tabs.isTabEnabled(dialog.autocorrelation_tab)
    assert not dialog.autocorrelation_figure.axes
    dialog.tabs.setCurrentWidget(dialog.autocorrelation_page)
    assert len(dialog.autocorrelation_figure.axes[0].patches) == 24
    dialog.invalidate()
    assert not dialog.tabs.isTabEnabled(dialog.autocorrelation_tab)
    assert not dialog.autocorrelation_figure.axes
    window.dirty = False


def test_cv_control_changes_do_not_invalidate_a_running_bootstrap(qtbot):
    window, dialog = window_with_data(qtbot)
    mode = dialog.band_method.findData("bootstrap_simultaneous")
    assert mode >= 0 and not dialog.band_method.itemIcon(mode).isNull()
    dialog.band_method.setCurrentIndex(mode)
    dialog.bootstrap_resamples.setValue(100)
    dialog.bootstrap_block_length.setValue(4)
    dialog.calculate()
    generation = dialog.generation
    dialog.cv_seed.setValue(123)
    assert dialog.generation == generation
    assert not dialog._bootstrap_cancel.is_set()
    wait_fit(qtbot, dialog)
    assert dialog.bootstrap_result is not None
    assert dialog.bootstrap_result.block_length == 4
    assert dialog.bootstrap_result.diagnostics()["confidence_method"] == "grid_max_deviation"
    assert dialog.fit_settings["bootstrap_block_length"] == 4
    window.dirty = False


def test_invalidated_cv_clears_its_diagnostics(qtbot):
    window, dialog = window_with_data(qtbot)
    dialog.add_candidate()
    dialog.calculate(comparison=True)
    qtbot.waitUntil(lambda: dialog.comparison_result is not None and not dialog.worker.isRunning(), timeout=10000)
    dialog.tabs.setCurrentWidget(dialog.diagnostic_page)
    assert dialog.residual_table.model() is not None
    dialog.cv_seed.setValue(42)
    assert dialog.comparison_result is None
    assert dialog.residual_table.model() is None
    window.dirty = False


@pytest.mark.parametrize("method", ["bootstrap", "bootstrap_simultaneous"])
def test_forced_bootstrap_failure_keeps_analytic_bands_disabled(qtbot, monkeypatch, method):
    window, dialog = window_with_data(qtbot)
    from autopyro.ui.dialogs import curve_fitting_calculation as module

    def fail_bootstrap(*args, **kwargs):
        raise ValueError("Bootstrap unavailable")

    monkeypatch.setattr(module, "bootstrap_bands", fail_bootstrap)
    dialog.band_method.setCurrentIndex(dialog.band_method.findData(method))
    dialog.calculate()
    wait_fit(qtbot, dialog)
    assert dialog.result.covariance is not None
    assert not dialog.show_confidence.isEnabled()
    dialog.apply_current_fit()
    qtbot.waitUntil(lambda: not dialog.worker.isRunning() and len(window.model.project.columns) == 4, timeout=10000)
    record = next(iter(window.model.project.view["curve_fit_runs"].values()))
    assert record["band_method_actual"] is None
    window.dirty = False


def test_grouped_bootstrap_tracks_group_column_as_an_output_source(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    x, y = Column(name="X", kind="number"), Column(name="Y", kind="number")
    group = Column(name="Sample group", kind="text")
    rows = [Row(values={x.id: i, y.id: 2*i+1+.1*(i % 3), group.id: f"G{i % 5}"}) for i in range(25)]
    rows.append(Row(values={x.id: None, y.id: 4, group.id: None}))  # Excluded points need no group label.
    window.set_project(Project(columns=[x, y, group], rows=rows))
    window.show_curve_fitting()
    dialog = window.curve_fitting_window
    dialog.band_method.setCurrentIndex(dialog.band_method.findData("bootstrap"))
    dialog.bootstrap_resamples.setValue(100)
    index = dialog.bootstrap_group.findData(group.id)
    assert index > 0 and not dialog.bootstrap_group.itemIcon(index).isNull()
    dialog.bootstrap_group.setCurrentIndex(index)
    assert not dialog.bootstrap_block_length.isEnabled()
    dialog.calculate()
    wait_fit(qtbot, dialog)
    assert dialog.bootstrap_result.group_count == 5
    monkeypatch.setattr(QInputDialog, "getText", lambda *args: ("Grouped fit", True))
    dialog.save_current_result()
    qtbot.waitUntil(lambda: dialog.saved_results.count() == 1 and not dialog.worker.isRunning(), timeout=10000)
    snapshot = window.model.project.view["curve_fit_results"][0]
    assert snapshot["settings"]["bootstrap_group"] == group.id
    assert group.id in snapshot["source_ids"]
    dialog.apply_current_fit()
    qtbot.waitUntil(lambda: len(window.model.project.columns) == 9 and not dialog.worker.isRunning(), timeout=10000)
    record = next(iter(window.model.project.view["curve_fit_runs"].values()))
    assert record["bootstrap"]["method"] == "cluster_percentile"
    assert group.id in record["source_ids"]
    assert window.model.setData(window.model.index(0, 2), "Changed group")
    assert snapshot["stale"] and record["stale"]
    window.dirty = False


@pytest.mark.parametrize("language", ["ru", "en"])
@pytest.mark.parametrize("theme", ["breeze_light", "breeze_dark"])
def test_curve_controls_have_localized_text_and_renderable_icons(qtbot, qapp, monkeypatch, tmp_path, language, theme):
    from autopyro.ui import i18n
    from autopyro.ui.theme import apply_theme

    store = QSettings(str(tmp_path / "interface.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(i18n, "settings", lambda: store)
    i18n.save_language(language, store)
    i18n.initialize(store)
    previous_theme = qapp.property("autopyroTheme") or "breeze_light"
    try:
        window, dialog = window_with_data(qtbot)
        apply_theme(qapp, theme)
        expected = (("Доверительная полоса", "Предсказательная полоса") if language == "ru"
                    else ("Confidence band", "Prediction band"))
        assert (dialog.show_confidence.text(), dialog.show_prediction.text()) == expected
        assert dialog.bootstrap_block_length.specialValueText() == ("Без блоков" if language == "ru" else "No blocks")
        assert dialog.tabs.tabText(dialog.autocorrelation_tab) == (
            "Автокорреляция остатков" if language == "ru" else "Residual autocorrelation")
        assert dialog.tabs.tabText(dialog.influence_tab) == ("Влияние точек" if language == "ru" else "Point influence")
        loo_index = dialog.cv_strategy.findData("loo")
        assert dialog.cv_strategy.itemText(loo_index) == ("По одной точке (LOO)" if language == "ru" else "Leave one out (LOO)")
        assert not dialog.cv_strategy.itemIcon(loo_index).isNull()
        for strategy, ru, en in (("nested_group", "Вложенная CV по группам", "Nested group CV"),
                                 ("nested_time", "Вложенная временная CV", "Nested time-series CV")):
            index = dialog.cv_strategy.findData(strategy)
            assert dialog.cv_strategy.itemText(index) == (ru if language == "ru" else en)
            assert not dialog.cv_strategy.itemIcon(index).pixmap(20, 20).isNull()
        assert dialog.cv_holdout.itemText(0) == ("Без тестовой выборки" if language == "ru" else "No independent test set")
        for root in (dialog, dialog.parameter_tabs):
            for button in root.findChildren(QPushButton):
                assert not button.icon().isNull(), button.text()
                assert not button.icon().pixmap(20, 20).isNull(), button.text()
                if language == "en":
                    assert not re.search("[А-Яа-яЁё]", button.text()), button.text()
        for tabs in (dialog.tabs, dialog.parameter_tabs):
            for index in range(tabs.count()):
                assert not tabs.tabIcon(index).isNull(), tabs.tabText(index)
        window.dirty = False
    finally:
        i18n.save_language("ru", store)
        i18n.initialize(store)
        apply_theme(qapp, previous_theme)
