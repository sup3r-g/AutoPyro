"""Compressed interpolation snapshots retain replay, provenance and undo behavior."""
from copy import deepcopy
from dataclasses import asdict
import json

import pytest
from PySide6.QtCore import Qt, QThread

from autopyro import audit
from autopyro.core import Project
from autopyro.interpolation import prepare, replay
from autopyro.io import build_project
from autopyro.ui.window import MainWindow


@pytest.fixture
def data():
    project = build_project(["Well", "X", "Y"], [["A", "0", "2"], ["A", "1", ""], ["A", "2", "6"],
                                                 ["A", ".25", "ND"], ["A", ".5", "<0.1"]],
                            [2, 3, 4, 5, 6], [(0, "Well", "text", ""), (1, "X", "number", "m"),
                                        (2, "Y", "number", "mg/g")], {})
    project.sync_measurements()
    return project


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("method", ["linear", "nearest", "pchip"])
def test_archived_v1_v2_all_methods_replay_after_json_and_sqlite(data, monkeypatch, version, method, tmp_path):
    monkeypatch.setattr(audit, "MIN_PROTOCOL_BYTES", 0)
    well, x, y = data.columns
    options = dict(x_id=x.id, y_id=y.id, group_ids=[well.id], row_ids=set(data.row_map()), method=method)
    if version == 2:
        options["constraints"] = {"well_id": well.id, "analytical_context": True}
    proposals, protocol = prepare(data, **options)
    expected = deepcopy(protocol)
    archived = audit.archive_protocol(protocol)
    assert isinstance(archived, str)
    assert len(archived) < len(json.dumps(protocol))
    assert audit.read_protocol(archived) == expected
    assert [row["values"][y.id] for row in audit.read_protocol(archived)["rows"]][-2:] == ["ND", "<0.1"]
    assert replay(json.loads(json.dumps(archived))) == proposals
    data.columns[-1].metadata["interpolation_protocol"] = archived
    path = tmp_path / "packed.apyro"
    data.save(path)
    restored = Project.load(path)
    assert restored.columns[-1].metadata["interpolation_protocol"] == archived
    assert replay(archived) == proposals
    assert protocol == expected


@pytest.mark.parametrize("damage", ["checksum", "truncated", "version", "size"])
def test_corrupted_archive_fails_without_touching_project(data, monkeypatch, damage):
    monkeypatch.setattr(audit, "MIN_PROTOCOL_BYTES", 0)
    well, x, y = data.columns
    _, protocol = prepare(data, x.id, y.id, [well.id], set(data.row_map()))
    archived = audit.archive_protocol(protocol)
    if damage == "checksum":
        fields = archived.split(":")
        fields[5] = "0" * 64
        archived = ":".join(fields)
    elif damage == "truncated":
        archived = archived[:-12]
    elif damage == "version":
        archived = archived.replace("zlib-json:1:", "zlib-json:999:", 1)
    else:
        archived = archived.replace(f":{len(json.dumps(protocol, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))}:",
                                    f":{audit.MAX_PROTOCOL_BYTES + 1}:", 1)
    before = deepcopy(data)
    with pytest.raises(ValueError, match="целостность протокола интерполяции"):
        replay(archived)
    assert data == before


@pytest.mark.parametrize("compact", [False, True])
def test_gui_archive_worker_undo_reopen_and_lazy_replay(data, qtbot, monkeypatch, tmp_path, compact):
    from autopyro.ui.dialogs import interpolation_calculation as gui
    if compact:
        from autopyro import interpolation
        monkeypatch.setattr(interpolation, "COMPACT_MEASUREMENT_THRESHOLD", 0)
    monkeypatch.setattr(audit, "MIN_PROTOCOL_BYTES", 0)
    monkeypatch.setattr(gui, "BACKGROUND_ROWS", 1)
    threads = []
    archive = gui.archive_protocol

    def tracked_archive(protocol, **kwargs):
        threads.append(QThread.currentThread())
        return archive(protocol, **kwargs)

    monkeypatch.setattr(gui, "archive_protocol", tracked_archive)
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(data)
    owner.may_replace = lambda: True
    try:
        owner.interpolate()
        dialog = owner.interpolation_window
        dialog.x.setCurrentIndex(dialog.x.findData(data.columns[1].id))
        dialog.y.setCurrentIndex(dialog.y.findData(data.columns[2].id))
        dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
        dialog.calculate()
        assert dialog.diagnostic_model.rowCount() == 5
        assert threads and threads[0] != QThread.currentThread()
        assert isinstance(dialog.protocol, str)
        science = deepcopy(data.science)
        dialog.apply()
        column = data.columns[-1]
        assert column.metadata["interpolation_protocol"] is dialog.protocol
        event = data.history[-1]
        assert event["metadata"]["interpolation_protocol"] is dialog.protocol
        owner.model.undo.undo()
        assert len(data.columns) == 3
        owner.model.undo.redo()
        assert data.columns[-1].metadata["interpolation_protocol"] is dialog.protocol
        assert data.science == science
        path = tmp_path / "gui-packed.apyro"
        data.save(path)
        # Opening and attaching a project must not inflate archived snapshots.
        def forbidden(_):
            raise AssertionError("archive decoded while opening")

        read = audit.read_protocol
        monkeypatch.setattr(audit, "read_protocol", forbidden)
        restored = Project.load(path)
        owner.set_project(restored, path)
        assert isinstance(restored.columns[-1].metadata["interpolation_protocol"], str)
        monkeypatch.setattr(audit, "read_protocol", read)
        owner.interpolate()
        before = deepcopy(restored)
        owner.interpolation_window.replay_button.click()
        assert restored == before
        assert len(replay(restored.columns[-1].metadata["interpolation_protocol"])) == 1
    finally:
        owner.dirty = False
        owner.close()


def test_history_expands_nested_archive_only_for_selected_action(data, qtbot, monkeypatch):
    from autopyro.ui.dialogs import history as gui
    monkeypatch.setattr(audit, "MIN_PROTOCOL_BYTES", 0)
    well, x, y = data.columns
    _, protocol = prepare(data, x.id, y.id, [well.id], set(data.row_map()))
    archived = audit.archive_protocol(protocol)
    data.history = [{"action": "add_column", "metadata": {"interpolation_protocol": archived}},
                    {"action": "other"}]
    before = deepcopy(data.history)
    dialog = gui.HistoryDialog(data)
    qtbot.addWidget(dialog)
    qtbot.waitUntil(lambda: dialog.isEnabled() and bool(dialog.details.toPlainText()), timeout=10000)
    assert '"other"' in dialog.details.toPlainText()
    dialog.actions.setCurrentRow(0)
    qtbot.waitUntil(lambda: dialog.isEnabled() and '"add_column"' in dialog.details.toPlainText(), timeout=10000)
    assert json.loads(dialog.details.toPlainText())["metadata"]["interpolation_protocol"] == protocol
    assert data.history == before


@pytest.mark.parametrize("version", [1, 2, 3, 4])
@pytest.mark.parametrize("archived", [False, True])
def test_project_versions_keep_inline_and_archived_snapshots(data, monkeypatch, tmp_path,
                                                           write_v2_project, write_v3_project, version, archived):
    monkeypatch.setattr(audit, "MIN_PROTOCOL_BYTES", 0)
    well, x, y = data.columns
    proposals, protocol = prepare(data, x.id, y.id, [well.id], set(data.row_map()),
                                   constraints={"well_id": well.id, "analytical_context": True})
    stored = audit.archive_protocol(protocol) if archived else protocol
    data.columns[-1].metadata["interpolation_protocol"] = stored
    path = tmp_path / "old.apyro"
    if version == 1:
        legacy = asdict(data)
        legacy.pop("science")  # Original table-only v1 reconstructs scientific records on load.
        payload = {"format": "autopyro.table", "version": 1, **legacy}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    elif version == 2:
        write_v2_project(data, path)
    elif version == 3:
        write_v3_project(data, path)
    else:
        data.save(path)
    before = path.read_bytes()
    restored = Project.load(path)
    assert replay(restored.columns[-1].metadata["interpolation_protocol"]) == proposals
    assert path.read_bytes() == before


def test_size_limit_keeps_inline_protocol_replayable(data, monkeypatch):
    monkeypatch.setattr(audit, "MAX_PROTOCOL_BYTES", 1)
    well, x, y = data.columns
    proposals, protocol = prepare(data, x.id, y.id, [well.id], set(data.row_map()))
    stored = audit.archive_protocol(protocol)
    assert stored is protocol
    assert replay(stored) == proposals


def test_inline_history_copies_use_worker_and_remain_independent(data, qtbot, monkeypatch, tmp_path):
    from autopyro.ui.table import commands
    monkeypatch.setattr(audit, "MAX_PROTOCOL_BYTES", 1)
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(data)
    owner.autosave_timer.stop()
    try:
        owner.interpolate()
        dialog = owner.interpolation_window
        dialog.x.setCurrentIndex(dialog.x.findData(data.columns[1].id))
        dialog.y.setCurrentIndex(dialog.y.findData(data.columns[2].id))
        dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
        dialog.calculate()
        assert isinstance(dialog.protocol, dict) and dialog.proposals
        science = deepcopy(data.science)
        threads = []
        copy = commands.deepcopy

        def tracked(value):
            if isinstance(value, dict) and "interpolation_protocol" in value:
                threads.append(QThread.currentThread())
            return copy(value)

        monkeypatch.setattr(commands, "deepcopy", tracked)
        monkeypatch.setattr(commands, "large_snapshot", lambda _: True)
        dialog.apply()
        column = data.columns[-1]
        first = data.history[-1]["metadata"]["interpolation_protocol"]
        assert first == dialog.protocol and first is not dialog.protocol
        owner.model.undo.undo()
        owner.model.undo.redo()
        second = data.history[-1]["metadata"]["interpolation_protocol"]
        assert second == first and second is not first and second is not dialog.protocol
        assert len(threads) == 2 and all(thread != QThread.currentThread() for thread in threads)
        assert data.science == science
        # Mutating one inline history event cannot alter the column or redo event.
        first["rows"][0]["values"][data.columns[2].id] = "changed history"
        assert second == column.metadata["interpolation_protocol"]
        path = tmp_path / "inline.apyro"
        data.save(path)
        restored = Project.load(path)
        assert replay(restored.columns[-1].metadata["interpolation_protocol"]) == dialog.proposals
    finally:
        owner.dirty = False
        owner.close()


@pytest.mark.parametrize("change", ["data", "proxy_filter", "proxy_sort", "error"])
def test_inline_history_preparation_rejects_stale_preview_before_push(data, qtbot, monkeypatch, change):
    from threading import Event
    from PySide6.QtCore import QTimer
    from autopyro.ui.table import commands
    monkeypatch.setattr(audit, "MAX_PROTOCOL_BYTES", 1)
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(data)
    owner.autosave_timer.stop()
    try:
        owner.interpolate()
        dialog = owner.interpolation_window
        dialog.x.setCurrentIndex(dialog.x.findData(data.columns[1].id))
        dialog.y.setCurrentIndex(dialog.y.findData(data.columns[2].id))
        dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
        dialog.calculate()
        before = len(data.columns), len(data.history), owner.model.undo.count()
        changed = Event()
        copy = commands.deepcopy

        def delayed(value):
            if isinstance(value, dict) and "interpolation_protocol" in value:
                if change == "error":
                    raise ValueError("Synthetic history preparation failure")
                assert changed.wait(5)
            return copy(value)

        def change_source():
            if change == "data":
                data.rows[0].values[data.columns[2].id] = 20
                owner.model.notify_changed()
            elif change == "proxy_filter":
                owner.proxy.set_rules([{"column": data.columns[1].id, "operator": "gt", "value": 0}])
            else:
                owner.proxy.sort(1, Qt.SortOrder.DescendingOrder)
            changed.set()

        monkeypatch.setattr(commands, "deepcopy", delayed)
        monkeypatch.setattr(commands, "large_snapshot", lambda _: True)
        if change != "error":
            QTimer.singleShot(0, change_source)
        dialog.apply()
        assert (len(data.columns), len(data.history), owner.model.undo.count()) == before
        assert dialog._preview_source is None
    finally:
        owner.dirty = False
        owner.close()
