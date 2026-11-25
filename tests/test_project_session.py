"""Project replacement commits only after the user's file/recovery choice."""
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime
from threading import Event

import pytest
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QDialog, QMessageBox
from shiboken6 import isValid

from autopyro.model.core import Project
from autopyro.ui import project_session as module
from autopyro.ui.autosave import recovery_metadata
from autopyro.ui.i18n import tr
from autopyro.ui.window import MainWindow


@pytest.fixture
def owner(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.autosave_timer.stop()
    return window


@pytest.mark.parametrize("choice", ["Yes", "No", "Cancel"])
def test_open_with_newer_recovery(owner, tmp_path, monkeypatch, choice):
    project = Project()
    target = tmp_path / "source.apyro"
    project.save(target)
    original = target.read_bytes()
    project.view["marker"] = "recovered"
    project.view["recovery"] = recovery_metadata(project, target)
    recovery = target.with_name(target.name + ".autosave-other.apyro")
    project.save_recovery(recovery)
    later = target.stat().st_mtime + 10
    os.utime(recovery, (later, later))
    previous = owner.model.project
    monkeypatch.setattr(module.QFileDialog, "getOpenFileName", lambda *a: (str(target), ""))
    monkeypatch.setattr(module.QMessageBox, "question", lambda *a: getattr(QMessageBox.StandardButton, choice))
    errors = []
    monkeypatch.setattr(module.QMessageBox, "warning", lambda *a: errors.append(a))

    owner.open_project()

    assert not errors
    assert target.read_bytes() == original and recovery.exists()
    if choice == "Cancel":
        assert owner.model.project is previous
        assert owner.path is None and not owner.dirty
    else:
        assert owner.path == target
        assert owner.model.project.id == project.id
        assert owner.dirty == (choice == "Yes")
        assert owner.model.project.view.get("marker") == ("recovered" if choice == "Yes" else None)
        assert owner.recovered_from == (recovery if choice == "Yes" else None)


@pytest.mark.parametrize("valid_copy", [False, True])
def test_open_only_offers_recovery_for_selected_project(owner, tmp_path, monkeypatch, valid_copy):
    project, other = Project(), Project()
    target, neighbor = tmp_path / "well[12].apyro", tmp_path / "well1.apyro"
    project.save(target)
    other.save(neighbor)
    original_files = {target: target.read_bytes(), neighbor: neighbor.read_bytes()}
    modified = target.stat().st_mtime

    def write_copy(project, bound_path, name, age):
        project.view["recovery"] = recovery_metadata(project, bound_path)
        path = tmp_path / name
        project.save_recovery(path)
        os.utime(path, (modified + age, modified + age))
        original_files[path] = path.read_bytes()
        return path

    write_copy(other, neighbor, "well1.apyro.autosave-neighbor.apyro", 50)
    write_copy(other, target, "well[12].apyro.autosave-wrong-id.apyro", 40)
    write_copy(project, neighbor, "well[12].apyro.autosave-wrong-path.apyro", 30)
    broken = tmp_path / "well[12].apyro.autosave-broken.apyro"
    broken.write_bytes(b"SQLite format 3\x00" + b"damaged page" * 10)
    os.utime(broken, (modified + 20, modified + 20))
    original_files[broken] = broken.read_bytes()
    recovery = None
    if valid_copy:
        project.view["marker"] = "own recovery"
        recovery = write_copy(project, target, "well[12].apyro.autosave-own.apyro", 10)
    offers, warnings = [], []

    def accept(*args):
        offers.append(args)
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(module.QFileDialog, "getOpenFileName", lambda *a: (str(target), ""))
    monkeypatch.setattr(QMessageBox, "question", accept)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: warnings.append(a))

    owner.open_project()

    assert not warnings and len(offers) == int(valid_copy)
    assert owner.path == target and owner.model.project.id == project.id
    assert owner.recovered_from == recovery
    assert owner.model.project.view.get("marker") == ("own recovery" if valid_copy else None)
    for path, expected in original_files.items():
        assert path.read_bytes() == expected


@pytest.mark.parametrize("invalid", [False, True])
def test_open_cancel_or_invalid_file_keeps_current_project(owner, tmp_path, monkeypatch, invalid):
    target = tmp_path / "invalid.apyro"
    target.write_text("not json", encoding="utf-8")
    previous = owner.model.project
    errors = []
    monkeypatch.setattr(module.QFileDialog, "getOpenFileName", lambda *a: (str(target) if invalid else "", ""))
    monkeypatch.setattr(module.QMessageBox, "warning", lambda *a: errors.append(a))
    owner.open_project()
    assert owner.model.project is previous
    assert owner.path is None and not owner.dirty
    assert bool(errors) == invalid


@pytest.mark.parametrize("source_format", ["unreadable", "sqlite", "legacy"])
@pytest.mark.parametrize("accept", [False, True])
def test_open_damaged_project_offers_older_copy_without_replacing_files(
        owner, tmp_path, monkeypatch, source_format, accept):
    target = tmp_path / "well[12].apyro"
    neighbor = tmp_path / "well1.apyro"
    damaged, other = Project(), Project()
    damaged.save(target)
    other.save(neighbor)
    damaged.view["marker"] = "last edit"
    damaged.view["recovery"] = recovery_metadata(damaged, target)
    copy = target.with_name(target.name + ".autosave-own.apyro")
    damaged.save_recovery(copy)
    if source_format == "sqlite":
        with closing(sqlite3.connect(target)) as database:
            database.execute("UPDATE metadata SET payload='damaged' WHERE key='source'")
            database.commit()
    elif source_format == "legacy":
        target.write_text(json.dumps({"format": "autopyro.table", "version": 1, "id": damaged.id,
                                      "columns": [], "rows": ["damaged"]}), encoding="utf-8")
    else:
        target.write_bytes(b"damaged header")
    now = target.stat().st_mtime
    os.utime(copy, (now - 30, now - 30))

    other.view["recovery"] = recovery_metadata(other, neighbor)
    wrong_path = target.with_name(target.name + ".autosave-other.apyro")
    other.save_recovery(wrong_path)
    os.utime(wrong_path, (now, now))
    originals = {p: p.read_bytes() for p in (target, neighbor, copy, wrong_path)}
    if source_format != "unreadable":
        other.view["recovery"] = recovery_metadata(other, target)
        wrong_id = target.with_name(target.name + ".autosave-wrong-id.apyro")
        other.save_recovery(wrong_id)
        os.utime(wrong_id, (now, now))
        originals[wrong_id] = wrong_id.read_bytes()

    current = owner.model.project
    current_path = tmp_path / "current.apyro"
    current.save(current_path)
    owner.path = current_path
    owner.model_changed()
    current_copy = owner.autosave_path()
    current_copy.write_bytes(b"previous session copy")
    originals[current_path] = current_path.read_bytes()
    warnings, offers = [], []
    replies = iter((QMessageBox.StandardButton.Discard,
                    QMessageBox.StandardButton.Yes if accept else QMessageBox.StandardButton.Cancel))

    def answer(*args):
        offers.append(args)
        return next(replies)

    monkeypatch.setattr(module.QFileDialog, "getOpenFileName", lambda *a: (str(target), ""))
    monkeypatch.setattr(QMessageBox, "question", answer)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: warnings.append(a))
    owner.actions[tr('Открыть проект…')].trigger()

    assert not warnings and len(offers) == 2
    text = offers[-1][2]
    assert str(copy) in text and str(wrong_path) not in text
    assert datetime.fromtimestamp(copy.stat().st_mtime_ns // 1_000_000_000).isoformat(sep=" ") in text
    uuid_warning = tr('Не удалось прочитать UUID исходного проекта. Найденная копия совпадает по имени и записанному пути.')
    assert (uuid_warning in text) == (source_format == "unreadable")
    assert offers[-1][-1] == QMessageBox.StandardButton.Cancel
    if accept:
        assert owner.model.project.id == damaged.id
        assert owner.model.project.view["marker"] == "last edit"
        assert owner.dirty and owner.path is None and owner.recovered_from == copy
        alias = tmp_path / "source-alias.apyro"
        os.link(target, alias)
        destinations = iter((target, copy, alias, tmp_path / "restored.apyro"))
        suggested = target.with_stem(target.stem + "-recovered")

        def save_path(*args):
            assert args[2] == str(suggested)
            return str(next(destinations)), ""

        monkeypatch.setattr(module.QFileDialog, "getSaveFileName", save_path)
        for _ in range(3):
            assert not owner.save() and owner.dirty and owner.path is None
        assert len(warnings) == 3
        assert owner.save()
        assert owner.path == tmp_path / "restored.apyro" and not owner.dirty
        saved = Project.load(owner.path)
        assert saved.id == damaged.id and saved.view["marker"] == "last edit"
        assert "recovery" not in saved.view
    else:
        assert owner.model.project is current and owner.path == current_path and owner.dirty
        assert current_copy.read_bytes() == b"previous session copy"
    assert all(path.read_bytes() == expected for path, expected in originals.items())


@pytest.mark.parametrize("outcome", ["cancel_picker", "invalid_file", "cancel_recovery", "open"])
def test_discard_clears_autosave_only_after_project_replacement(owner, qtbot, tmp_path, monkeypatch, outcome):
    current_path = tmp_path / "current.apyro"
    project = owner.model.project
    project.save(current_path)
    owner.path = current_path
    original = current_path.read_bytes()
    project.view["unsaved"] = True
    owner.model_changed()
    owner.autosave()
    recovery = owner.autosave_path()
    qtbot.waitUntil(lambda: recovery.exists() and owner._background_autosave.worker is None, timeout=10000)
    recovery_bytes = recovery.read_bytes()
    target = tmp_path / "next.apyro"
    incoming = Project()
    incoming.save(target)
    if outcome == "invalid_file":
        target.write_bytes(b"invalid project")
    if outcome == "cancel_recovery":
        incoming.view["recovery"] = recovery_metadata(incoming, target)
        incoming_copy = target.with_name(target.name + ".autosave-test.apyro")
        incoming.save_recovery(incoming_copy)
        later = target.stat().st_mtime + 10
        os.utime(incoming_copy, (later, later))
    target_bytes = target.read_bytes()
    warnings = []
    replies = iter((QMessageBox.StandardButton.Discard, QMessageBox.StandardButton.Cancel))
    monkeypatch.setattr(QMessageBox, "question", lambda *a: next(replies))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: warnings.append(a))
    monkeypatch.setattr(module.QFileDialog, "getOpenFileName",
                        lambda *a: ("" if outcome == "cancel_picker" else str(target), ""))

    owner.open_project()

    assert bool(warnings) == (outcome == "invalid_file")
    assert current_path.read_bytes() == original and target.read_bytes() == target_bytes
    if outcome == "open":
        assert not recovery.exists()
        assert owner.path == target and owner.model.project.id == incoming.id and not owner.dirty
    else:
        assert recovery.read_bytes() == recovery_bytes
        assert owner.path == current_path and owner.model.project is project and owner.dirty
        assert owner.model.project.view["unsaved"]


def test_discard_and_restore_same_session_keeps_selected_copy(owner, qtbot, tmp_path, monkeypatch):
    current = owner.model.project
    target = tmp_path / "current.apyro"
    current.save(target)
    owner.path = target
    current.view["saved_by_autosave"] = True
    owner.model_changed()
    owner.autosave()
    recovery = owner.autosave_path()
    qtbot.waitUntil(lambda: recovery.exists() and owner._background_autosave.worker is None, timeout=10000)
    later = target.stat().st_mtime + 10
    os.utime(recovery, (later, later))
    original, recovery_bytes = target.read_bytes(), recovery.read_bytes()
    replies = iter((QMessageBox.StandardButton.Discard, QMessageBox.StandardButton.Yes))
    monkeypatch.setattr(QMessageBox, "question", lambda *a: next(replies))
    monkeypatch.setattr(module.QFileDialog, "getOpenFileName", lambda *a: (str(target), ""))

    owner.open_project()

    assert owner.model.project is not current and owner.model.project.id == current.id
    assert owner.recovered_from == recovery and owner.model.project.view["saved_by_autosave"]
    assert target.read_bytes() == original and recovery.read_bytes() == recovery_bytes


@pytest.mark.parametrize("choice", ["Cancel", "Discard", "Save"])
def test_new_project_respects_unsaved_decision(owner, qtbot, tmp_path, monkeypatch, choice):
    previous = owner.model.project
    owner.path = tmp_path / "current.apyro"
    owner.dirty = True
    owner.autosave()
    recovery = owner.autosave_path()
    qtbot.waitUntil(lambda: recovery.exists() and owner._background_autosave.worker is None, timeout=10000)
    monkeypatch.setattr(module.QMessageBox, "question", lambda *a: getattr(QMessageBox.StandardButton, choice))
    owner.new_project()
    assert (owner.model.project is previous) == (choice == "Cancel")
    assert recovery.exists() == (choice == "Cancel")
    if choice == "Save":
        assert Project.load(tmp_path / "current.apyro").id == previous.id
    if choice != "Cancel":
        assert owner.path is None and not owner.dirty


def test_replacement_waits_for_worker_before_deleting_old_statistics(owner, monkeypatch, qtbot):
    owner.show_statistics()
    statistics = owner.statistics_window
    statistics.timer.stop()
    cancelled, release = Event(), Event()
    results, notifications = [], []

    def calculate():
        assert release.wait(5)
        return 'old report'

    statistics.run_task(calculate, results.append, cancelled=cancelled)
    monkeypatch.setattr(QMessageBox, 'information', lambda *args: notifications.append(args))
    previous = owner.model.project
    try:
        owner.new_project()
        assert notifications and owner.model.project is previous
        assert owner.statistics_window is statistics and not cancelled.is_set()

        replacement = Project()
        owner.set_project(replacement)
        assert owner.model.project is replacement and owner.statistics_window is None
        assert cancelled.is_set() and isValid(statistics) and statistics.isHidden()
        closing = QCloseEvent()
        owner.closeEvent(closing)
        assert not closing.isAccepted() and len(notifications) == 2
    finally:
        release.set()
    qtbot.waitUntil(lambda: not isValid(statistics))
    assert not results


def test_replacement_does_not_delete_document_that_refuses_close(owner, qtbot):
    class Document(QDialog):
        allow_close = False

        def closeEvent(self, event):
            if self.allow_close:
                super().closeEvent(event)
            else:
                event.ignore()

    document = owner.analysis_documents.qc_window = Document(owner)
    owner.desktop.open_document('test-document', document, 'QC')
    owner.set_project(Project())
    qtbot.wait(1)
    assert isValid(document) and document.isHidden()
    assert owner.qc_window is None and owner.desktop.documents.indexOf(document) == -1

    document.allow_close = True
    assert document.close()
    qtbot.waitUntil(lambda: not isValid(document))
