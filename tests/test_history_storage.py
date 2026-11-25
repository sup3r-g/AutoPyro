"""Version 3 journal registry and safe migration from the metadata-only layout."""
from contextlib import closing
from copy import deepcopy
import hashlib
import json
import sqlite3

import pytest

from autopyro.audit import archive_protocol, read_protocol
from autopyro.core import Project
from autopyro.database import storage
from autopyro.importing import digest


@pytest.fixture
def journal():
    protocol = {"id": "run", "rows": [{"Sample": "001", "S2": "ND", "TOC": 2} for _ in range(2000)]}
    archived = archive_protocol(protocol)
    assert isinstance(archived, str)
    events = [{"protocol": archived, "action": "import" if i % 2 else "undo_import", "at": str(i)}
              for i in range(30)]
    events.extend([{"action": "legacy", "protocol": {"notes": ["original"]}}, "old annotation"])
    return Project(history=events), protocol


def test_registry_preserves_event_order_payloads_and_hash(tmp_path, journal):
    project, protocol = journal
    path = tmp_path / "indexed.apyro"
    before = digest(project)
    project.save(path)
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == storage.FORMAT_VERSION
        assert connection.execute("SELECT COUNT(*) FROM history_protocols").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM history_events").fetchone()[0] == 32
        assert connection.execute("SELECT COUNT(*) FROM metadata WHERE key='history'").fetchone()[0] == 0
    loaded = Project.load(path)
    assert loaded == project and digest(loaded) == before
    assert list(loaded.history[0]) == list(project.history[0])
    assert all(event["protocol"] is loaded.history[0]["protocol"] for event in loaded.history[:30])
    assert read_protocol(loaded.history[0]["protocol"]) == protocol
    details = read_protocol(loaded.history[0]["protocol"])
    details["rows"].clear()
    assert read_protocol(loaded.history[1]["protocol"]) == protocol


def test_history_accepts_utf8_blob_and_rejects_trailing_json(tmp_path):
    path = tmp_path / "history-payload.apyro"
    project = Project(history=[["annotation", "annotation"]])
    project.save(path)
    with closing(sqlite3.connect(path)) as connection:
        payload = json.dumps(project.history[0]).encode("utf-8-sig")
        connection.execute("UPDATE history_events SET payload=?", (payload,))
        connection.commit()
    restored = Project.load(path)
    assert restored.history == project.history
    assert restored.history[0][0] is restored.history[0][1]
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("UPDATE history_events SET payload='[] {}'")
        connection.commit()
    with pytest.raises(json.JSONDecodeError, match="Extra data"):
        Project.load(path)


def test_distinct_content_with_same_run_id_does_not_merge(tmp_path, journal):
    project, protocol = journal
    other = deepcopy(protocol)
    other["rows"][0]["S2"] = 9
    project.history.append({"action": "other", "protocol": archive_protocol(other)})
    path = tmp_path / "distinct.apyro"
    project.save(path)
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM history_protocols").fetchone()[0] == 2
    assert read_protocol(Project.load(path).history[-1]["protocol"]) == other


def test_v2_open_is_read_only_and_first_save_keeps_exact_backup(tmp_path, journal, write_v2_project):
    original, _ = journal
    path = write_v2_project(original, tmp_path / "old.apyro")
    before = path.read_bytes()
    loaded = Project.load(path)
    assert loaded == original and path.read_bytes() == before
    assert not list(tmp_path.glob("*.v2-*.bak"))
    loaded.save(path)
    backup, = tmp_path.glob("*.v2-*.bak")
    assert backup.read_bytes() == before and Project.load(backup) == original
    loaded.save(path)
    assert list(tmp_path.glob("*.v2-*.bak")) == [backup]
    assert Project.load(path) == original


def test_interrupted_v2_migration_can_be_retried(tmp_path, journal, write_v2_project, monkeypatch):
    project, _ = journal
    path = write_v2_project(project, tmp_path / "old.apyro")
    before = path.read_bytes()
    loaded = Project.load(path)
    replace = storage.os.replace

    def fail(source, target):
        if target == path:
            raise OSError("simulated replace failure")
        return replace(source, target)

    monkeypatch.setattr(storage.os, "replace", fail)
    with pytest.raises(OSError, match="simulated"):
        loaded.save(path)
    assert path.read_bytes() == before
    backup, = tmp_path.glob("*.v2-*.bak")
    assert backup.read_bytes() == before and not list(tmp_path.glob("*.tmp"))
    monkeypatch.setattr(storage.os, "replace", replace)
    loaded.save(path)
    assert Project.load(path) == project and backup.read_bytes() == before


def test_wrong_v2_backup_prevents_replacement(tmp_path, journal, write_v2_project):
    project, _ = journal
    path = write_v2_project(project, tmp_path / "old.apyro")
    before = path.read_bytes()
    backup = path.with_name(path.name + f".v2-{storage.file_digest(path)[:12]}.bak")
    backup.write_bytes(b"different data")
    with pytest.raises(ValueError, match="Резервная копия"):
        Project.load(path).save(path)
    assert path.read_bytes() == before and backup.read_bytes() == b"different data"
    assert not list(tmp_path.glob("*.tmp"))


def test_recovery_save_leaves_v2_primary_untouched(tmp_path, journal, write_v2_project):
    project, _ = journal
    path = write_v2_project(project, tmp_path / "old.apyro")
    before = path.read_bytes()
    loaded = Project.load(path)
    recovery = tmp_path / "recovery.apyro"
    loaded.save_recovery(recovery)
    assert path.read_bytes() == before and Project.load(recovery) == loaded
    assert not list(tmp_path.glob("*.v2-*.bak"))
    loaded.save(path)
    backup, = tmp_path.glob("*.v2-*.bak")
    assert backup.read_bytes() == before


@pytest.mark.parametrize("damage", ["count", "gap", "last", "checksum", "missing", "ambiguous", "unused"])
def test_corrupt_history_is_rejected(tmp_path, journal, damage):
    project, protocol = journal
    path = tmp_path / "corrupt.apyro"
    project.save(path)
    with closing(sqlite3.connect(path)) as connection:
        if damage == "count":
            connection.execute("UPDATE metadata SET payload='999' WHERE key='history_event_count'")
        elif damage == "gap":
            connection.execute("DELETE FROM history_events WHERE position=1")
        elif damage == "last":
            connection.execute("DELETE FROM history_events WHERE position=31")
        elif damage == "checksum":
            connection.execute("UPDATE history_protocols SET payload=payload || 'x'")
        elif damage == "missing":
            connection.execute("UPDATE history_events SET protocol_id='missing' WHERE position=0")
        elif damage == "ambiguous":
            connection.execute("UPDATE history_events SET payload=? WHERE position=0",
                               (json.dumps({"protocol": {"other": "data"}}),))
        else:
            other = archive_protocol({**protocol, "id": "unused"})
            connection.execute("INSERT INTO history_protocols VALUES (?,?)",
                               (hashlib.sha256(other.encode("utf-8")).hexdigest(), other))
        connection.commit()
    before = path.read_bytes()
    with pytest.raises(ValueError, match="целостность"):
        Project.load(path)
    assert path.read_bytes() == before


def test_future_format_is_not_overwritten(tmp_path, journal):
    project, _ = journal
    path = tmp_path / "future.apyro"
    project.save(path)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA user_version=999")
        connection.commit()
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Неизвестный формат"):
        Project().save(path)
    assert path.read_bytes() == before and not list(tmp_path.glob("*.tmp"))
