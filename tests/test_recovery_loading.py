"""Recovery reads run off-thread and commit only after successful preparation."""
from dataclasses import asdict
from threading import Event, get_ident
import time

import pytest
from PySide6.QtCore import QTimer

from autopyro.ui import project_session as session_module
from autopyro.core import Project
from autopyro.database.storage import file_digest
from autopyro.io import build_project
from autopyro.ui.window import MainWindow


@pytest.fixture
def recovery_files(tmp_path):
    project = build_project(["Sample", "TOC", "S2"], [["001", "2", "6"]], [2],
        [(0, "Sample", "text", ""), (1, "TOC", "number", "%wt"),
         (2, "S2", "number", "mg HC/g rock")], {})
    target = tmp_path / "source.apyro"
    project.save(target)
    original_bytes = target.read_bytes()
    project.rows[0].values[project.columns[2].id] = 9
    project.refresh()
    project.view["recovery"] = {"target": str(target), "source_digest": file_digest(target)}
    recovery = tmp_path / "recovery.apyro"
    project.save_recovery(recovery)
    return target, recovery, original_bytes


def test_recovery_load_and_preparation_leave_gui_timers_running(qtbot, monkeypatch, recovery_files):
    target, recovery, original_bytes = recovery_files
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(Project.load(target), target)
    main_thread = get_ident()
    entered, finished = Event(), Event()
    calls, ticks = [], []
    load, prepare = Project.load, session_module.prepare_project

    def delayed_load(path):
        calls.append(("load", get_ident()))
        entered.set()
        time.sleep(.08)
        try:
            return load(path)
        finally:
            finished.set()

    def prepare_once(project):
        calls.append(("prepare", get_ident()))
        return prepare(project)

    monkeypatch.setattr(Project, "load", staticmethod(delayed_load))
    monkeypatch.setattr(session_module, "prepare_project", prepare_once)
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1) if entered.is_set() and not finished.is_set() else None)
    timer.start(5)
    try:
        owner.load_autosave(recovery)
        assert ticks
        assert [name for name, _ in calls] == ["load", "prepare"]
        assert all(thread != main_thread for _, thread in calls)
        assert owner.path == target and owner.dirty
        restored = owner.model.project
        assert restored.rows[0].values[restored.columns[2].id] == 9
        assert restored.rows[0].original[restored.columns[2].id] == "6"
        assert restored._storage_path == target.resolve()
        assert restored._storage_digest == file_digest(target)
        assert "recovery" not in restored.view and target.read_bytes() == original_bytes
    finally:
        timer.stop()
        owner.dirty = False


@pytest.mark.parametrize("damage", ["invalid_json", "not_recovery"])
def test_failed_recovery_keeps_current_project_and_files(qtbot, recovery_files, damage):
    target, recovery, original_bytes = recovery_files
    if damage == "invalid_json":
        recovery.write_bytes(b"invalid JSON")
    else:
        recovery.write_bytes(original_bytes)
    recovery_bytes = recovery.read_bytes()
    owner = MainWindow()
    qtbot.addWidget(owner)
    original = Project.load(target)
    owner.set_project(original, target)
    before = asdict(original)
    with pytest.raises(ValueError):
        owner.load_autosave(recovery)
    assert owner.model.project is original and asdict(original) == before
    assert owner.path == target and not owner.dirty
    assert target.read_bytes() == original_bytes and recovery.read_bytes() == recovery_bytes
