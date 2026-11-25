"""Explicit saves isolate their inputs and keep recovery/file ownership intact."""
from dataclasses import asdict
from threading import Event, get_ident

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox

from autopyro.ui import autosave, project_session as session_module
from autopyro.core import Project
from autopyro.database.storage import file_digest
from autopyro.io import build_project
from autopyro.ui import window as module


@pytest.fixture
def owner(qtbot, tmp_path, monkeypatch):
    project = build_project(["Sample", "TOC", "S2"],
        [["001", "2.00", "6"], ["001", "3", "ND"]], [2, 3],
        [(0, "Sample", "text", ""), (1, "TOC", "number", "%wt"),
         (2, "S2", "number", "mg HC/g rock")], {})
    path = tmp_path / "original.apyro"
    project.save(path)
    window = module.MainWindow()
    qtbot.addWidget(window, before_close_func=lambda w: setattr(w, "dirty", False))
    window.set_project(project, path)
    window.autosave_timer.stop()
    window.save_errors = []
    monkeypatch.setattr(session_module.QMessageBox, "warning", lambda *args: window.save_errors.append(str(args[-1])))
    assert window.model.setData(window.model.index(0, 2), "7")
    try:
        yield window
    finally:
        if window._background_autosave is not None:
            window._background_autosave.shutdown()
        window.dirty = False


def run_when_entered(owner, entered, release, action):
    """Run one GUI callback while a worker holds its independent snapshot."""
    timer = QTimer(owner)
    def tick():
        if entered.is_set():
            timer.stop()
            try:
                action()
            finally:
                release.set()
    timer.timeout.connect(tick)
    timer.start(5)
    return timer


@pytest.mark.parametrize("typed_name", ["other", "other.csv", "lab.v2.csv"])
@pytest.mark.parametrize("approve", [False, True])
def test_save_as_confirms_final_path_before_replacing(owner, qtbot, monkeypatch, typed_name, approve):
    source = owner.path
    original = source.read_bytes()
    owner.autosave()
    recovery = owner.autosave_path()
    qtbot.waitUntil(lambda: recovery.exists() and owner._background_autosave.worker is None, timeout=10000)
    recovery_bytes = recovery.read_bytes()
    typed_path = source.with_name(typed_name)
    destination = typed_path.with_suffix(".apyro")
    existing = Project()
    existing.save(destination)
    destination_bytes = destination.read_bytes()
    prompts = []

    def confirm(parent, title, message, buttons, default):
        prompts.append(message)
        assert str(destination) in message
        assert default == QMessageBox.StandardButton.No
        assert destination.read_bytes() == destination_bytes
        return QMessageBox.StandardButton.Yes if approve else QMessageBox.StandardButton.No

    monkeypatch.setattr(session_module.QFileDialog, "getSaveFileName", lambda *a: (str(typed_path), ""))
    monkeypatch.setattr(QMessageBox, "question", confirm)

    assert owner.save(as_new=True) is approve

    assert len(prompts) == 1
    assert source.read_bytes() == original and not typed_path.exists()
    if approve:
        assert owner.path == destination and not owner.dirty
        assert not recovery.exists()
        assert Project.load(destination).id == owner.model.project.id
    else:
        assert owner.path == source and owner.dirty
        assert destination.read_bytes() == destination_bytes and recovery.read_bytes() == recovery_bytes
        assert Project.load(destination).id == existing.id


def test_save_keeps_gui_timers_alive_and_preserves_live_objects_and_undo(owner, monkeypatch):
    project, model = owner.model.project, owner.model
    row, column = project.rows[0], project.columns[2]
    measurement = project.science.current(row.id, column.id)
    before = asdict(project.science)
    entered, release = Event(), Event()
    threads, ticks = [], []
    save = Project.save
    def delayed(snapshot, path):
        threads.append(get_ident())
        assert snapshot is not project and snapshot.science is not project.science
        entered.set()
        assert release.wait(10)
        save(snapshot, path)
    monkeypatch.setattr(Project, "save", delayed)
    timer = run_when_entered(owner, entered, release, lambda: ticks.append(1))
    try:
        assert owner.save()
        assert ticks and threads[0] != get_ident()
        assert owner.model is model and model.project is project
        assert project.science.current(row.id, column.id) is measurement
        assert asdict(project.science) == before
        assert project._storage_digest == file_digest(owner.path)
        loaded = Project.load(owner.path)
        assert loaded.rows == project.rows and loaded.science == project.science
        assert not owner.dirty and not owner.save_errors
        model.undo.undo()
        assert project.science.current(row.id, column.id) is measurement and measurement.value == 6
    finally:
        release.set()
        timer.stop()


@pytest.mark.parametrize("change_during_wait", [False, True])
def test_explicit_save_waits_for_cancelled_autosave_without_late_publication(owner, qtbot, monkeypatch, change_during_wait):
    entered, release = Event(), Event()
    write_recovery, save = autosave.write_recovery, Project.save
    recovery = owner.autosave_path()
    recovery.write_bytes(b"previous recovery")
    before = owner.path.read_bytes()
    def delayed(*args, **options):
        entered.set()
        assert release.wait(10)
        write_recovery(*args, **options)
    monkeypatch.setattr(autosave, "write_recovery", delayed)
    owner.autosave()
    qtbot.waitUntil(entered.is_set)
    done = owner._background_autosave._write_done
    writes = []
    def checked_save(snapshot, path):
        assert done.is_set()
        writes.append(1)
        save(snapshot, path)
    monkeypatch.setattr(Project, "save", checked_save)
    checks = []
    def during_wait():
        checks.append(owner._save_cancelled is not None)
        owner.autosave()  # The timer cannot start a new generation while saving.
        assert owner._background_autosave._write_done is done
        if change_during_wait:
            assert owner.model.setData(owner.model.index(0, 2), "8")
    timer = run_when_entered(owner, entered, release, during_wait)
    try:
        saved = owner.save()
        qtbot.waitUntil(lambda: owner._background_autosave.worker is None)
        assert checks == [True]
        assert not list(owner.path.parent.glob("*.pending-*"))
        if change_during_wait:
            assert not saved and not writes and owner.dirty and owner.save_errors
            assert owner.path.read_bytes() == before and recovery.read_bytes() == b"previous recovery"
        else:
            assert saved and writes == [1]
            assert not recovery.exists() and not owner.save_errors
            assert Project.load(owner.path).rows[0].values[owner.model.project.columns[2].id] == 7
    finally:
        release.set()
        timer.stop()


def test_autosave_synchronizes_only_its_private_snapshot(owner, qtbot):
    original = owner.path.read_bytes()
    project = owner.model.project
    row, column = project.rows[0], project.columns[2]
    live = project.science.current(row.id, column.id)
    old_state = asdict(live)
    row.values[column.id] = 8  # Legacy API edit without TableModel synchronization.
    owner.model_changed()

    owner.autosave()
    qtbot.waitUntil(lambda: owner._background_autosave.worker is None, timeout=10000)

    recovered = Project.load(owner.autosave_path())
    result = recovered.science.current(row.id, column.id)
    assert result.id == live.id and result.value == 8
    assert result.history[-1]["previous"]["value"] == old_state["value"]
    assert asdict(live) == old_state
    assert owner.path.read_bytes() == original


@pytest.mark.parametrize(("raw", "value"), [(8, 8), ("ND", None)])
def test_save_preserves_legacy_direct_edits_and_exact_measurement_history(owner, raw, value):
    project = owner.model.project
    row, column = project.rows[0], project.columns[2]
    measurement = project.science.current(row.id, column.id)
    row.values[column.id] = raw  # Legacy API: not synchronized through TableModel.
    assert owner.save()
    assert project.science.current(row.id, column.id) is measurement
    assert measurement.value == value and measurement.table_value() == raw
    assert Project.load(owner.path).science == project.science
    assert row.original[column.id] == "6"


@pytest.mark.parametrize("failure", ["io", "conflict"])
def test_failed_save_keeps_file_recovery_binding_and_dirty_state(owner, monkeypatch, failure):
    from autopyro.database import storage
    path = owner.path
    project = owner.model.project
    old_binding = project._storage_path, project._storage_digest
    recovery = owner.autosave_path()
    recovery.write_bytes(b"previous recovery")
    if failure == "io":
        replace = storage.os.replace
        def fail_commit(source, destination):
            if str(destination) == str(path.resolve()):
                raise OSError("QA commit failure")
            return replace(source, destination)
        monkeypatch.setattr(storage.os, "replace", fail_commit)
    else:
        external = Project.load(path)
        external.rows[0].values[external.columns[2].id] = 9
        external.refresh()
        external.save(path)
    before = path.read_bytes()
    assert not owner.save()
    assert path.read_bytes() == before and recovery.read_bytes() == b"previous recovery"
    assert (project._storage_path, project._storage_digest) == old_binding
    assert owner.dirty and owner._save_cancelled is None and owner.save_errors
    assert not list(path.parent.glob("*.tmp"))


def test_external_change_after_successful_save_is_rejected_on_next_save(owner):
    project = owner.model.project
    old_digest = project._storage_digest
    assert owner.save()
    saved_digest = project._storage_digest
    assert saved_digest != old_digest and saved_digest == file_digest(owner.path)
    assert owner.model.setData(owner.model.index(0, 2), "8")
    external = Project.load(owner.path)
    external.rows[0].values[external.columns[2].id] = 9
    external.refresh()
    external.save(owner.path)
    external_digest = file_digest(owner.path)
    assert not owner.save()
    assert file_digest(owner.path) == external_digest
    assert project._storage_digest == saved_digest and owner.dirty
    assert project.rows[0].values[project.columns[2].id] == 8
    assert owner.save_errors


def test_change_during_snapshot_copy_aborts_before_writing(owner, monkeypatch):
    copy = session_module.copy_project
    entered, release = Event(), Event()
    before = owner.path.read_bytes()
    def delayed(project):
        entered.set()
        assert release.wait(10)
        return copy(project)
    monkeypatch.setattr(session_module, "copy_project", delayed)
    timer = run_when_entered(owner, entered, release,
        lambda: owner.model.setData(owner.model.index(0, 2), "8"))
    try:
        assert not owner.save()
        assert owner.path.read_bytes() == before
        assert owner.dirty and owner._save_cancelled is None and owner.save_errors
    finally:
        release.set()
        timer.stop()


@pytest.mark.parametrize("replace_project", [False, True])
def test_changes_during_write_are_not_marked_saved(owner, tmp_path, monkeypatch, replace_project):
    entered, release = Event(), Event()
    save = Project.save
    project = owner.model.project
    column_id = project.columns[2].id
    destination = tmp_path / "save-as.apyro"
    monkeypatch.setattr(session_module.QFileDialog, "getSaveFileName", lambda *args: (str(destination), ""))
    def delayed(snapshot, path):
        entered.set()
        assert release.wait(10)
        save(snapshot, path)
    monkeypatch.setattr(Project, "save", delayed)
    replacements = []
    def edit():
        assert not owner.save()  # Reentrant Save cannot start another write.
        owner.autosave()
        if replace_project:
            other = Project()
            replacements.append(other)
            owner.set_project(other)
        else:
            assert owner.model.setData(owner.model.index(0, 2), "8")
    timer = run_when_entered(owner, entered, release, edit)
    try:
        assert not owner.save(as_new=True)
        assert Project.load(destination).rows[0].values[column_id] == 7
        if replace_project:
            assert owner.model.project is replacements[0] and owner.path is None
            assert not owner.dirty
        else:
            assert owner.path == destination and owner.dirty
            assert project.rows[0].values[column_id] == 8
            assert project._storage_path == destination.resolve()
            assert project._storage_digest == file_digest(destination)
        assert owner._save_cancelled is None and not owner.save_errors
    finally:
        release.set()
        timer.stop()
