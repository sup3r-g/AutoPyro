"""Operation-count and responsiveness regressions, without machine-specific timings."""

from copy import deepcopy
from threading import Event, get_ident
import time

import pytest
from PySide6.QtCore import QTimer, Qt

from autopyro.database import recovery_writer
from autopyro.ui import autosave
from autopyro.core import Project
from autopyro.io import build_project
from autopyro.ui.autosave import BackgroundAutosave
from autopyro.ui.table import FilterProxy, TableModel
from autopyro.ui.tasks import run_with_progress


def data(rows=1000):
    return build_project(["Well", "Depth", "S2"], [["A", str(i), str(i * 2)] for i in range(rows)],
                         list(range(2, rows + 2)), [(0, "Well", "text", ""), (1, "Depth", "number", "m"),
                                                  (2, "S2", "number", "mg/g")], {})


def test_single_edit_does_not_scan_measurements_or_resort_entire_table(qtbot, monkeypatch):
    comparisons = []
    original_compare = FilterProxy.lessThan
    def compare(self, left, right):
        comparisons.append(1)
        return original_compare(self, left, right)
    monkeypatch.setattr(FilterProxy, "lessThan", compare)
    project = data()
    model = TableModel(project)
    proxy = FilterProxy()
    proxy.setSourceModel(model)
    proxy.sort(1)
    comparisons.clear()
    calls = []
    original_current = project.science.current
    def current(*args):
        calls.append(args)
        return original_current(*args)
    monkeypatch.setattr(project.science, "current", current)
    changed = []
    model.dataChanged.connect(lambda start, end, roles: changed.append((start.row(), end.row())))
    assert model.setData(model.index(500, 1), "-1")
    assert len(calls) <= 2 * len(project.columns)
    assert len(comparisons) < 100
    assert changed == [(500, 500)]
    assert proxy.mapToSource(proxy.index(0, 0)).row() == 500
    model.undo.undo()
    assert project.rows[500].values[project.columns[1].id] == 500
    model.undo.redo()
    assert project.rows[500].values[project.columns[1].id] == -1


def test_pyro_edit_recalculates_and_notifies_only_changed_row(qtbot, monkeypatch):
    from autopyro import calculations
    from tools.benchmark_dependent_edit import input_project

    project = input_project(1000)
    model = TableModel(project, prepared=True)
    by_name = {column.name: column for column in project.columns}
    lengths = []
    original = calculations.project_frame

    def frame(*args, **kwargs):
        result = original(*args, **kwargs)
        lengths.append(len(result))
        return result

    monkeypatch.setattr(calculations, "project_frame", frame)
    changed = []
    model.dataChanged.connect(lambda first, last, roles: changed.append((first.row(), last.row())))
    assert model.setData(model.index(500, 3), "8")
    # Editing S2 only recalculates formulas that depend on S2, on this row.
    dependent_count = sum("S2" in definition["roles"] for definition in calculations.FORMULAS.values())
    assert lengths == [1] * dependent_count
    assert changed == [(500, 500)]
    assert project.rows[500].values[by_name["HI"].id] == 400
    model.undo.undo()
    assert project.rows[500].values[by_name["HI"].id] == 300
    model.undo.redo()
    assert project.rows[500].values[by_name["HI"].id] == 400


def test_optimized_sort_preserves_nulls_ties_and_dynamic_filter_updates(qtbot):
    project = build_project(["Name", "Value"], [["B", "2"], ["A", "2"], ["C", ""], ["D", "10"]], [2, 3, 4, 5],
                            [(0, "Name", "text", ""), (1, "Value", "number", "")], {})
    model = TableModel(project)
    proxy = FilterProxy()
    proxy.setSourceModel(model)
    proxy.sort(0)
    proxy.sort(1)
    rows = lambda: [proxy.mapToSource(proxy.index(i, 0)).row() for i in range(proxy.rowCount())]
    assert rows() == [1, 0, 3, 2]
    proxy.sort(1, Qt.SortOrder.DescendingOrder)
    assert rows()[:2] == [2, 3]
    proxy.set_rules([{"column": project.columns[1].id, "operator": "gt", "value": 5}])
    assert rows() == [3]
    model.setData(model.index(0, 1), "20")
    assert rows() == [0, 3]
    proxy.sort_rules = [(project.columns[1].id, False), (project.columns[0].id, True)]
    proxy.sort(0)
    model.setData(model.index(3, 1), "30")
    assert rows() == [0, 3]


def test_filter_preserves_sorted_rows_without_sorting_them_again(qtbot, monkeypatch):
    project = data()
    model = TableModel(project)
    proxy = FilterProxy()
    proxy.setSourceModel(model)
    proxy.sort(1)
    depth = project.columns[1].id
    proxy.set_rules([{"column": depth, "operator": "gt", "value": 500}])
    comparisons = []
    original = proxy.lessThan

    def compare(left, right):
        comparisons.append(1)
        return original(left, right)

    monkeypatch.setattr(proxy, "lessThan", compare)
    proxy.set_rules([{"column": depth, "operator": "gt", "value": 600}])
    assert proxy.rowCount() == 399
    assert comparisons == []
    proxy.set_rules([{"column": depth, "operator": "gt", "value": 300}])
    positions = [proxy.mapToSource(proxy.index(i, 0)).row() for i in range(proxy.rowCount())]
    assert positions == list(range(301, 1000))
    assert model.setData(model.index(0, 1), "9999")
    assert proxy.mapToSource(proxy.index(proxy.rowCount() - 1, 0)).row() == 0


def test_large_numeric_filter_preserves_exact_integers_and_updates_after_edit(qtbot):
    from autopyro.core import Column, Project, Row

    column = Column("Value", "number")
    limit = 2 ** 53
    values = [limit, limit + 1, None, limit - 1] * 1100
    project = Project(columns=[column], rows=[Row({column.id: value}) for value in values])
    model = TableModel(project, prepared=True)
    proxy = FilterProxy()
    proxy.setSourceModel(model)
    proxy.rowCount()
    proxy.set_rules([{"column": column.id, "operator": "gt", "value": limit, "negate": True}])
    positions = lambda: [proxy.mapToSource(proxy.index(i, 0)).row() for i in range(proxy.rowCount())]
    expected = [i for i in range(len(values)) if i % 4 != 1]
    assert positions() == expected
    assert model.setData(model.index(2, 0), str(limit + 1))
    assert positions() == [i for i in expected if i != 2]
    model.undo.undo()
    assert positions() == expected


@pytest.mark.parametrize("combine", ["AND", "OR"])
def test_large_combined_filter_matches_scalar_rules(qtbot, monkeypatch, combine):
    from autopyro.core import Column, Project, Row

    name, depth = Column("Well"), Column("Depth", "number")
    values = [("Straße[1]", 10), ("STRASSE[1]", 2), ("Strasse1", 10), (None, 10), ("", None)] * 900
    project = Project(columns=[name, depth],
                      rows=[Row({name.id: text, depth.id: number}) for text, number in values])
    model = TableModel(project, prepared=True)
    proxy = FilterProxy()
    proxy.setSourceModel(model)
    proxy.rowCount()
    rules = [{"column": name.id, "operator": "contains", "value": "STRASSE[1]"},
             {"column": depth.id, "operator": "ge", "value": 5, "negate": True}]
    proxy.set_rules(rules, combine)
    batch = proxy.ordered_row_ids()
    monkeypatch.setattr(proxy, "_prepare_filter_mask", lambda: None)
    proxy.set_rules([], combine)
    proxy.set_rules(rules, combine)
    assert proxy.ordered_row_ids() == batch


def test_progress_work_runs_off_gui_thread_and_keeps_timers_alive(qapp):
    main_thread = get_ident()
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(5)
    worker_thread = run_with_progress(None, "Preparing data", lambda: (time.sleep(.08), get_ident())[1])
    timer.stop()
    assert worker_thread != main_thread
    assert ticks


def test_background_recovery_is_consistent_and_does_not_block(qtbot, tmp_path, monkeypatch):
    project = data(2)
    TableModel(project)
    entered, release = Event(), Event()
    write = autosave.write_recovery
    def delayed_save(*args, **options):
        entered.set()
        assert release.wait(5)
        write(*args, **options)
    monkeypatch.setattr(autosave, "write_recovery", delayed_save)
    job = BackgroundAutosave()
    path = tmp_path / "recovery.apyro"
    view = {"visible": [c.id for c in project.columns]}
    try:
        assert job.start(project, path, view, {"target": None}, 1)
        qtbot.waitUntil(entered.is_set)
        ticks = []
        QTimer.singleShot(0, lambda: ticks.append(1))
        qtbot.waitUntil(lambda: bool(ticks))
        assert not path.exists()
        release.set()
        qtbot.waitUntil(lambda: job.worker is None, timeout=10000)
        restored = Project.load(path)
        assert restored.rows == project.rows
        assert restored.view["current"] == view
        assert restored.science.current(project.rows[0].id, project.columns[1].id).id == project.science.current(project.rows[0].id, project.columns[1].id).id
    finally:
        release.set()
        job.shutdown()
        job.deleteLater()


def test_cancelled_background_recovery_cannot_publish_after_clear(qtbot, tmp_path, monkeypatch):
    project = data(2)
    TableModel(project)
    entered, release = Event(), Event()
    write = autosave.write_recovery
    def delayed_save(*args, **options):
        write(*args, **options)
        entered.set()
        assert release.wait(5)
    monkeypatch.setattr(autosave, "write_recovery", delayed_save)
    job = BackgroundAutosave()
    path = tmp_path / "recovery.apyro"
    path.write_bytes(b"previous recovery")
    try:
        job.start(project, path, {}, {}, 1)
        qtbot.waitUntil(entered.is_set, timeout=10000)
        assert job.staging.exists()
        job.cancel()
        release.set()
        qtbot.waitUntil(lambda: job.worker is None)
        assert path.read_bytes() == b"previous recovery"
        assert list(tmp_path.glob("*.pending-*")) == []
    finally:
        release.set()
        job.shutdown()
        job.deleteLater()


def test_large_qc_uses_virtual_rows_and_reuses_checks_for_decisions(qtbot, monkeypatch):
    from autopyro.ui.dialogs import qc_checks as qc
    from autopyro.ui.window import MainWindow
    window = MainWindow()
    qtbot.addWidget(window)
    window.may_replace = lambda: True
    project = data(2)
    project.columns[2].name = "TOC"
    project.columns[2].unit = "%wt"
    window.set_project(project)
    checks = []
    original = qc.run_checks
    def checked(project, **kwargs):
        checks.append(1)
        return original(project, **kwargs)
    monkeypatch.setattr(qc, "run_checks", checked)
    window.show_qc()
    dialog = window.qc_window
    count = len(checks)
    assert dialog.current_issue() is not None
    dialog.comment.setText("Reviewed")
    dialog.decide("acknowledged")
    dialog.reload()
    assert len(checks) == count
    window.model.undo.undo()
    dialog.reload()
    assert len(checks) == count
    issue = deepcopy(dialog.issues[0])
    dialog.issues = [{**issue, "id": str(i)} for i in range(32_000)]
    dialog.render()
    assert dialog.table.model().rowCount() == 32_000
    assert dialog.table.model().index(31_999, 0).data(Qt.ItemDataRole.UserRole)["id"] == "31999"
    window.dirty = False


def test_small_window_autosave_queues_another_copy_after_edit(qtbot, tmp_path, monkeypatch):
    from autopyro.ui import window as module
    window = module.MainWindow()
    qtbot.addWidget(window)
    window.may_replace = lambda: True
    window.session.configure_autosave(True, 1)
    window.autosave_timer.stop()
    project = data(2)
    window.set_project(project)
    path = tmp_path / "automatic.apyro"
    monkeypatch.setattr(window, "autosave_path", lambda: path)
    entered, release = Event(), Event()
    write = autosave.write_recovery
    gui_thread = get_ident()
    writer_threads = []
    def delayed(*args, **options):
        writer_threads.append(get_ident())
        entered.set()
        assert release.wait(5)
        write(*args, **options)
    monkeypatch.setattr(autosave, "write_recovery", delayed)
    try:
        window.model.setData(window.model.index(0, 1), "10")
        window.autosave()
        qtbot.waitUntil(entered.is_set)
        assert len(writer_threads) == 1 and writer_threads[0] != gui_thread
        window.model.setData(window.model.index(0, 1), "20")
        release.set()
        qtbot.waitUntil(lambda: window._autosaved_revision == window.analysis_revision)
        assert Project.load(path).rows[0].values[project.columns[1].id] == 20
        assert len(writer_threads) == 2
        window.clear_autosave()
        assert not path.exists()
    finally:
        release.set()
        if window._background_autosave is not None:
            window._background_autosave.shutdown()
        window.dirty = False


@pytest.mark.parametrize('phase', ['transfer', 'writer'])
def test_recovery_publishes_during_edits_and_catches_up(qtbot, tmp_path, monkeypatch, phase):
    from itertools import count
    from autopyro.ui import window as module

    window = module.MainWindow()
    qtbot.addWidget(window)
    window.may_replace = lambda: True
    window.session.configure_autosave(True, 1)
    window.autosave_timer.stop()
    project = data(1000)
    initial_history = len(project.history)
    empty_row = project.rows[950]
    empty_column = project.columns[2].id
    empty_row.original.pop(empty_column)
    empty_row.values[empty_column] = None
    window.set_project(project)
    path = tmp_path / "recovery.apyro"
    project.save_recovery(path)
    previous_files = set(tmp_path.iterdir())
    monkeypatch.setattr(window, "autosave_path", lambda: path)
    entered, release = [Event(), Event()], [Event(), Event()]
    processes = []
    popen = recovery_writer.Popen
    transfer_count = 0

    def pause_between_batches(_seconds):
        # Stop after the first row batch, before reading a later edited row.
        if transfer_count < 2 and not entered[transfer_count].is_set():
            entered[transfer_count].set()
            assert release[transfer_count].wait(10)

    def pause_after_transfer(*args, **options):
        process = popen(*args, **options)
        cycle = len(processes)
        processes.append(process)
        poll = process.poll

        def wait_for_edit():
            if cycle < 2 and not entered[cycle].is_set():
                entered[cycle].set()
                assert release[cycle].wait(10)
            return poll()
        process.poll = wait_for_edit
        return process

    if phase == 'transfer':
        monkeypatch.setattr(recovery_writer, "perf_counter", count().__next__)
        monkeypatch.setattr(recovery_writer, "sleep", pause_between_batches)
        send = recovery_writer._send_project

        def count_transfers(*args):
            nonlocal transfer_count
            try:
                send(*args)
            finally:
                transfer_count += 1

        monkeypatch.setattr(recovery_writer, "_send_project", count_transfers)
    else:
        monkeypatch.setattr(recovery_writer, "Popen", pause_after_transfer)
    try:
        index = window.model.index(900, 1)
        row_id = project.rows[900].id
        column_id = project.columns[1].id
        assert window.model.setData(index, "10")
        window.autosave()
        job = window._background_autosave
        errors, published = [], []
        job.failed.connect(errors.append)
        job.saved.connect(published.append)
        for cycle, value in enumerate((20, 30)):
            qtbot.waitUntil(entered[cycle].is_set, timeout=10000)
            assert window.model.setData(index, str(value))
            if cycle == 0:
                # Creating a measurement appends a key while the earlier
                # revision still needs to enumerate the original store.
                assert window.model.setData(window.model.index(950, 2), "40")
            release[cycle].set()
            qtbot.waitUntil(lambda: len(published) == cycle + 1, timeout=10000)
            copy = Project.load(path)
            measurement = copy.science.current(row_id, column_id)
            assert copy.rows[900].values[column_id] == measurement.value == value - 10
            assert len(measurement.history) == cycle + 1
            assert copy.rows[950].values[empty_column] == (40 if cycle else None)
            assert len(copy.history) == initial_history + (3 if cycle else 1)

        qtbot.waitUntil(lambda: window._autosaved_revision == window.analysis_revision, timeout=10000)
        assert not errors
        assert all(process.poll() is not None for process in processes)
        assert set(tmp_path.iterdir()) == previous_files
        assert len(published) == 3 and published == sorted(set(published))
        window.load_autosave(path)
        restored = window.model.project
        measurement = restored.science.current(row_id, column_id)
        assert restored.rows[900].values[column_id] == measurement.value == 30
        assert len(measurement.history) == 3 and len(restored.history) == initial_history + 4
        assert restored.science.current(empty_row.id, empty_column).value == 40
    finally:
        for event in release:
            event.set()
        if window._background_autosave is not None:
            window._background_autosave.shutdown()
        window.dirty = False


def test_background_save_failure_keeps_previous_recovery(qtbot, tmp_path):
    project = data(2)
    project.sync_measurements()
    path = tmp_path / "recovery.apyro"
    path.write_bytes(b"previous recovery")
    measurement = next(iter(project.science.measurements.values()))
    measurement.confidence_level = 2
    job = BackgroundAutosave()
    errors = []
    job.failed.connect(errors.append)
    try:
        job.start(project, path, {}, {}, 1)
        qtbot.waitUntil(lambda: job.worker is None, timeout=10000)
        assert len(errors) == 1 and "(0, 1]" in errors[0]
        assert path.read_bytes() == b"previous recovery"
        assert set(tmp_path.iterdir()) == {path}
    finally:
        job.shutdown()
        job.deleteLater()


def test_finished_import_dialog_releases_its_source_data(qtbot, monkeypatch):
    import weakref
    from shiboken6 import isValid
    from PySide6.QtWidgets import QDialog
    from autopyro.ui.data_io import table_import
    from autopyro.ui.window import MainWindow

    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.may_replace = lambda: True
    created = []
    class ReadyImport(QDialog):
        def __init__(self, *, parent, **kwargs):
            super().__init__(parent)
            self.project = data(2)
            self.project_prepared = False
            self.preview_loaded = False
            created.append(weakref.ref(self))
        def load(self):
            self.preview_loaded = True
        def exec(self):
            assert self.preview_loaded
            return QDialog.DialogCode.Accepted
    monkeypatch.setattr(table_import, "ImportDialog", ReadyImport)
    monkeypatch.setattr(table_import.QFileDialog, "getOpenFileName", lambda *args: ("npd.xlsx", ""))
    owner.import_file()
    assert owner.model.rowCount() == 2
    qtbot.waitUntil(lambda: created[0]() is None or not isValid(created[0]()))
    assert owner.model.rowCount() == 2  # The accepted data belongs to the table now.
    owner.dirty = False
