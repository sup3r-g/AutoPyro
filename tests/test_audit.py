"""Versioned compact protocols, legacy compatibility and selected-event replay."""
from copy import deepcopy
from dataclasses import asdict
import json
import sqlite3

import pytest

from autopyro import audit
from autopyro.core import Project
from autopyro.importing import digest
from autopyro.io import build_project


@pytest.fixture
def protocol():
    return {"id": "lab-run-1", "source": {"path": "лаборатория.csv"},
            "rows": [{"sample": "001", "values": [2**60 + 1, -0.0, .03, None, True, "ND", "<0.2"]}
                     for _ in range(1500)]}


def test_archive_returns_independent_details_and_preserves_all_values(protocol):
    before = deepcopy(protocol)
    archived = audit.archive_protocol(protocol)
    assert isinstance(archived, str) and archived.startswith(audit.PREFIX)
    first, second = audit.read_protocol(archived), audit.read_protocol(archived)
    assert first == second == before
    assert list(first) == list(protocol)
    assert list(first["rows"][0]) == list(protocol["rows"][0])
    first["rows"][0]["values"].append("changed")
    assert audit.read_protocol(archived) == protocol == before


def test_small_and_non_json_extensions_keep_their_copy_semantics():
    value = {"rows": [(1, 2)]}
    assert audit.archive_protocol(value) is value
    result = audit.read_protocol(value)
    assert type(result["rows"][0]) is tuple
    result["rows"].append((3, 4))
    assert value == {"rows": [(1, 2)]}
    small = {"id": "small"}
    assert audit.archive_protocol(small) is small


def test_archive_read_can_cancel_between_decoding_stages(protocol):
    class Cancelled(Exception):
        pass

    archived = audit.archive_protocol(protocol)
    checkpoints = 0

    def checkpoint():
        nonlocal checkpoints
        checkpoints += 1
        if checkpoints == 3:
            raise Cancelled

    with pytest.raises(Cancelled):
        audit.read_protocol(archived, checkpoint=checkpoint)
    assert audit.read_protocol(archived) == protocol


def test_reload_shares_payloads_without_altering_legacy_events(protocol):
    archived = audit.archive_protocol(protocol)
    history = json.loads(json.dumps([{"protocol": archived}, {"protocol": archived}, {"protocol": {"id": "old"}}]))
    legacy = history[2]["protocol"]
    audit.share_history_protocols(history)
    assert history[0]["protocol"] is history[1]["protocol"]
    assert history[2]["protocol"] is legacy
    assert audit.read_protocol(history[0]["protocol"]) == protocol


@pytest.mark.parametrize("damage", ["checksum", "length", "truncated", "oversized", "version"])
def test_corrupt_or_unknown_archives_are_rejected(protocol, damage):
    archived = audit.archive_protocol(protocol)
    length, checksum, payload = archived[len(audit.PREFIX):].split(":", 2)
    if damage == "checksum":
        checksum = "0" * 64
    elif damage == "length":
        length = str(int(length) - 1)
    elif damage == "truncated":
        payload = payload[:-8]
    elif damage == "oversized":
        length = str(audit.MAX_PROTOCOL_BYTES + 1)
    else:
        archived = archived.replace("zlib-json:1:", "zlib-json:2:", 1)
    if damage != "version":
        archived = f"{audit.PREFIX}{length}:{checksum}:{payload}"
    with pytest.raises(ValueError):
        audit.read_protocol(archived)


@pytest.mark.parametrize("version", [1, 2, 3, 4])
@pytest.mark.parametrize("archived", [False, True])
def test_old_and_new_protocols_open_without_rewriting_sources(tmp_path, protocol, version, archived, write_v2_project, write_v3_project):
    project = build_project(["S2"], [["ND"]], [2], [(0, "S2", "number", "mg HC/g rock")], {"sha256": "initial"})
    project.sync_measurements()
    stored = audit.archive_protocol(protocol) if archived else protocol
    project.history = [{"at": "first", "action": "import", "protocol": stored},
                       {"at": "second", "action": "undo_import", "protocol": deepcopy(stored)}]
    path = tmp_path / "history.apyro"
    if version == 1:
        payload = asdict(project)
        payload.pop("science")
        path.write_text(json.dumps({"format": "autopyro.table", "version": 1, **payload}), encoding="utf-8")
    elif version == 2:
        write_v2_project(project, path)
    elif version == 3:
        write_v3_project(project, path)
    else:
        project.save(path)
    before = path.read_bytes()
    loaded = Project.load(path)
    assert path.read_bytes() == before
    assert [event["action"] for event in loaded.history] == ["import", "undo_import"]
    assert all(audit.read_protocol(event["protocol"]) == protocol for event in loaded.history)
    if archived:
        assert loaded.history[0]["protocol"] is loaded.history[1]["protocol"]
    new_path = tmp_path / "resaved.apyro"
    loaded.save(new_path)
    reopened = Project.load(new_path)
    assert reopened == loaded and digest(reopened) == digest(loaded)
    assert path.read_bytes() == before


def test_bad_archive_cannot_replace_a_valid_file(tmp_path, protocol):
    project = Project(history=[{"action": "import", "protocol": audit.archive_protocol(protocol)}])
    path = tmp_path / "valid.apyro"
    project.save(path)
    before = path.read_bytes()
    project.history[0]["protocol"] = audit.PREFIX + "1:" + "0" * 64 + ":invalid"
    with pytest.raises(ValueError):
        project.save(path)
    assert path.read_bytes() == before


def test_history_dialog_expands_only_the_selected_action(qtbot, protocol, monkeypatch):
    from autopyro.ui.dialogs import history as gui
    project = Project(history=[{"action": "first", "protocol": audit.archive_protocol(protocol)},
                               {"action": "second", "note": "no protocol"}])
    seen = []
    read = gui.read_protocol

    def tracked(value):
        seen.append(value)
        return read(value)

    monkeypatch.setattr(gui, "read_protocol", tracked)
    dialog = gui.HistoryDialog(project)
    qtbot.addWidget(dialog)
    qtbot.waitUntil(lambda: dialog.isEnabled() and bool(dialog.details.toPlainText()), timeout=10000)
    assert seen == [] and json.loads(dialog.details.toPlainText())["action"] == "second"
    dialog.actions.setCurrentRow(0)
    qtbot.waitUntil(lambda: dialog.isEnabled() and '"first"' in dialog.details.toPlainText(), timeout=10000)
    assert len(seen) == 1
    assert json.loads(dialog.details.toPlainText())["protocol"] == protocol
