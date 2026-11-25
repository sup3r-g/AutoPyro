"""Real Qt jobs and spawned exports stay responsive and reject stale input."""
from threading import Event, get_ident

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFileDialog, QMessageBox

from autopyro.core import Project
from autopyro.io import build_project
from autopyro.plot_data import PlotDataCache
from autopyro.ui.window import MainWindow


@pytest.fixture
def window(qtbot):
    owner = MainWindow()
    owner.may_replace = lambda: True
    qtbot.addWidget(owner)
    owner.set_project(build_project(['X', 'Y'], [[1, 2], [2, 3], [3, 4]], [2, 3, 4],
        [(0, 'X', 'number', ''), (1, 'Y', 'number', '')], {}))
    owner.show_plots()
    owner.plot_window.timer.stop()
    owner.plot_window.redraw()
    yield owner
    if owner.plot_window:
        dialog = owner.plot_window
        dialog.close()
        qtbot.waitUntil(lambda: dialog.preparation.worker is None, timeout=10000)
        qtbot.waitUntil(lambda: dialog.export_service.worker is None, timeout=30000)
    owner.dirty = False


def test_preparation_runs_off_gui_and_latest_change_wins(window, qtbot, monkeypatch):
    dialog = window.plot_window
    original = PlotDataCache.prepare
    entered, release = Event(), Event()
    threads = []
    def delayed(cache, *args, **kwargs):
        threads.append(get_ident())
        entered.set()
        release.wait(2)
        return original(cache, *args, **kwargs)
    monkeypatch.setattr(PlotDataCache, 'prepare', delayed)
    dialog.schedule()
    dialog.timer.stop()
    dialog.prepare_async()
    qtbot.waitUntil(entered.is_set)
    window.model.setData(window.model.index(0, 1), '20')
    dialog.timer.stop()
    dialog.prepare_async()
    release.set()
    qtbot.waitUntil(lambda: dialog.result is not None and dialog.preparation.worker is None, timeout=10000)
    assert threads and all(thread != get_ident() for thread in threads)
    assert max(dialog.result['artists'][0].get_offsets()[:, 1]) == 20
    assert dialog.prepared.source.token[1] == window.model.data_revision


def test_cancel_preparation_does_not_publish_old_result(window, qtbot, monkeypatch):
    dialog = window.plot_window
    original = PlotDataCache.prepare
    entered, release = Event(), Event()
    def delayed(cache, *args, **kwargs):
        entered.set()
        release.wait(2)
        return original(cache, *args, **kwargs)
    monkeypatch.setattr(PlotDataCache, 'prepare', delayed)
    dialog.schedule()
    dialog.timer.stop()
    dialog.prepare_async()
    qtbot.waitUntil(entered.is_set)
    dialog.cancel_preparation()
    release.set()
    qtbot.waitUntil(lambda: dialog.preparation.worker is None, timeout=10000)
    assert dialog.result is None and not dialog.export_button.isEnabled()


def test_edited_preview_and_background_recovery_use_same_revision(window, qtbot, tmp_path, monkeypatch):
    from autopyro.ui import autosave

    path = tmp_path / 'recovery.apyro'
    monkeypatch.setattr('autopyro.ui.project_session.large_snapshot', lambda project: True)
    monkeypatch.setattr(window, 'autosave_path', lambda: path)
    entered, release = Event(), Event()
    write_recovery = autosave.write_recovery

    def delayed_save(*args, **options):
        entered.set()
        assert release.wait(30)
        write_recovery(*args, **options)

    monkeypatch.setattr(autosave, 'write_recovery', delayed_save)
    window.autosave_timer.stop()
    dialog = window.plot_window
    column_id = window.model.project.columns[1].id
    assert window.model.setData(window.model.index(0, 1), '20')
    dialog.timer.stop()
    dialog.prepare_async()
    window.autosave()
    job = window._background_autosave
    assert job is not None
    try:
        qtbot.waitUntil(entered.is_set, timeout=10000)
        qtbot.waitUntil(lambda: dialog.result is not None and not dialog.preview.busy, timeout=20000)
        assert job.worker is not None and not path.exists()
        assert max(dialog.result['artists'][0].get_offsets()[:, 1]) == 20
        assert dialog.prepared.source.token[1] == window.model.data_revision
        release.set()
        qtbot.waitUntil(lambda: job.worker is None, timeout=10000)
        assert window._autosaved_revision == window.analysis_revision
        assert Project.load(path).rows[0].values[column_id] == 20
    finally:
        release.set()
        job.shutdown()


def test_spawned_export_keeps_input_snapshot_and_gui_heartbeat(window, qtbot, tmp_path, monkeypatch):
    dialog = window.plot_window
    dialog.style.load({'color': '#ff0000'})
    dialog.redraw()
    path = tmp_path / 'background.svg'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), 'SVG'))
    failures, results, ticks = [], [], []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: failures.append(args[-1]))
    dialog.export_service.completed.connect(results.append)
    heartbeat = QTimer()
    heartbeat.setInterval(10)
    heartbeat.timeout.connect(lambda: ticks.append(1))
    heartbeat.start()
    revision = window.model.data_revision
    dialog.export_plot()
    assert dialog.export_service.worker is not None
    window.model.setData(window.model.index(0, 1), '999')
    dialog.style.load({'color': '#008000'})
    qtbot.waitUntil(lambda: dialog.export_service.worker is None, timeout=30000)
    heartbeat.stop()
    assert not failures and results and len(ticks) >= 5
    text = path.read_text(encoding='utf-8')
    assert '#ff0000' in text and '#008000' not in text
    assert results[0]['plots'][0]['data_revision'] == revision
    assert not any(p.name.startswith('.') for p in tmp_path.iterdir())


def test_cancel_export_preserves_existing_file(window, qtbot, tmp_path, monkeypatch):
    dialog = window.plot_window
    path = tmp_path / 'cancelled.svg'
    path.write_bytes(b'previous')
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), 'SVG'))
    results = []
    dialog.export_service.completed.connect(results.append)
    dialog.export_plot()
    dialog.export_service.cancel()
    qtbot.waitUntil(lambda: dialog.export_service.worker is None, timeout=30000)
    assert results and results[0]['cancelled']
    assert path.read_bytes() == b'previous'
    assert list(tmp_path.iterdir()) == [path]


def test_project_replacement_cancels_owned_jobs(window, qtbot):
    dialog = window.plot_window
    service = dialog.preparation
    dialog.schedule()
    dialog.timer.stop()
    dialog.prepare_async()
    window.set_project(Project())
    qtbot.waitUntil(lambda: service.worker is None, timeout=10000)
    assert window.plot_window is None


def test_cold_export_capture_rejects_data_changed_between_chunks(window, qtbot, tmp_path, monkeypatch):
    dialog = window.plot_window
    dialog.captured_input = None
    path = tmp_path / 'changed.svg'
    path.write_bytes(b'previous')
    failures = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: failures.append(args[-1]))
    dialog.start_export([('Cold', path, dialog.spec)])
    assert dialog.export_request is not None
    window.model.setData(window.model.index(0, 1), '999')
    qtbot.waitUntil(lambda: not dialog.export_busy(), timeout=10000)
    assert failures and path.read_bytes() == b'previous'
    assert dialog.export_service.worker is None


def test_cold_capture_can_be_cancelled_before_a_process_starts(window, tmp_path):
    dialog = window.plot_window
    dialog.captured_input = None
    path = tmp_path / 'not-started.svg'
    dialog.start_export([('Cold', path, dialog.spec)])
    dialog.cancel_export()
    assert not dialog.export_busy() and dialog.export_capture.steps is None
    assert not path.exists()


def test_batch_does_not_reuse_preview_after_parsing_policy_changed(window, tmp_path):
    dialog = window.plot_window
    previous = dialog.prepared
    dialog.spec.text_numbers = not dialog.spec.text_numbers
    jobs = dialog.make_export_jobs([('New policy', tmp_path / 'new.svg', dialog.spec)])
    assert jobs[0]['prepared'] is None
    dialog.spec.text_numbers = not dialog.spec.text_numbers
    dialog.schedule()
    jobs = dialog.make_export_jobs([('Pending preview', tmp_path / 'pending.svg', dialog.spec)])
    assert dialog.prepared is previous and jobs[0]['prepared'] is None


def test_chunked_capture_is_restarted_after_an_edit(window, qtbot, monkeypatch):
    from autopyro.ui import plot_preparation

    window.set_project(build_project(['X', 'Y'], [[i, i + 1] for i in range(3000)], list(range(2, 3002)),
        [(0, 'X', 'number', ''), (1, 'Y', 'number', '')], {}))
    window.show_plots()
    dialog = window.plot_window
    dialog.timer.stop()
    dialog.prepare_async()
    with monkeypatch.context() as patch:
        clock = iter((0, .007))
        patch.setattr(plot_preparation, "perf_counter", lambda: next(clock))
        dialog.preparation.capture._step()
    assert dialog.preparation.capture.steps is not None
    window.model.setData(window.model.index(0, 1), '9876')
    dialog.timer.stop()
    dialog.prepare_async()
    qtbot.waitUntil(lambda: dialog.result is not None and dialog.preparation.worker is None, timeout=15000)
    assert max(dialog.result['artists'][0].get_offsets()[:, 1]) == 9876


def test_runtime_palette_is_frozen_for_the_export_process(window, qtbot, tmp_path, monkeypatch):
    import matplotlib
    from matplotlib.colors import ListedColormap
    name = 'autopyro-private-export-palette'
    matplotlib.colormaps.register(ListedColormap(['#ff00ff', '#00ffff'], name=name))
    try:
        dialog = window.plot_window
        dialog.style.load({'color_by': window.model.project.columns[0].id, 'color_mode': 'numeric', 'cmap': name})
        dialog.redraw()
        path = tmp_path / 'palette.svg'
        monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), 'SVG'))
        failures = []
        monkeypatch.setattr(QMessageBox, 'warning', lambda *args: failures.append(args[-1]))
        dialog.export_plot()
        qtbot.waitUntil(lambda: not dialog.export_busy(), timeout=30000)
        assert not failures
        content = path.read_text(encoding='utf-8')
        assert '#ff00ff' in content and '#00ffff' in content
    finally:
        matplotlib.colormaps.unregister(name)



@pytest.mark.parametrize('expired', [False, True])
def test_autosave_gives_visible_plot_a_bounded_head_start(window, qtbot, tmp_path, monkeypatch, expired):
    from types import SimpleNamespace

    path = tmp_path / 'recovery.apyro'
    monkeypatch.setattr('autopyro.ui.project_session.large_snapshot', lambda project: True)
    monkeypatch.setattr(window, 'autosave_path', lambda: path)
    window.show()
    plot = window.plot_window
    assert plot.isVisible()
    plot.timer.start(30000)
    window.dirty = True
    if expired:
        window._autosave_wait = SimpleNamespace(isValid=lambda: True, elapsed=lambda: 5000, invalidate=lambda: None)
    try:
        window.autosave()
        job = window._background_autosave
        if expired:
            assert job.worker is not None
        else:
            assert window._autosave_deferred.isActive()
            assert job.worker is None
        plot.timer.stop()
        qtbot.waitUntil(lambda: path.exists() and job.worker is None, timeout=10000)
        assert not window._autosave_deferred.isActive()
        assert Project.load(path).rows == window.model.project.rows
    finally:
        plot.timer.stop()
        window._stop_autosave_deferral()
        window._background_autosave.shutdown()


def test_autosave_defers_for_large_measurement_store_with_small_table(window, tmp_path, monkeypatch):
    project = window.model.project
    # Only dispatch is under test. No placeholder record is read or saved.
    monkeypatch.setattr(project.science, 'measurements', dict.fromkeys(range(100_000)))
    monkeypatch.setattr(window, 'autosave_path', lambda: tmp_path / 'recovery.apyro')
    window.show()
    plot = window.plot_window
    plot.timer.start(30000)
    window.dirty = True
    try:
        window.autosave()
        assert window._autosave_deferred.isActive()
        assert window._background_autosave.worker is None
    finally:
        plot.timer.stop()
        window._stop_autosave_deferral()
