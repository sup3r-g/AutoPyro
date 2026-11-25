"""Cached UUID snapshots retain the previous source-token and Qt invalidation contract."""
from copy import deepcopy

import pytest
from PySide6.QtCore import QModelIndex, Qt

from autopyro.core import Row
from autopyro.io import build_project
from autopyro.ui.widgets.documents import source_token
from autopyro.ui.window import MainWindow


@pytest.fixture
def owner(qtbot):
    project = build_project(["Sample", "Depth", "S2"],
                            [["001", "30", "6"], ["001", "10", "ND"], ["001", "20", "8"]],
                            [2, 3, 4], [(0, "Sample", "text", ""), (1, "Depth", "number", "m"),
                                        (2, "S2", "number", "mg HC/g rock")], {})
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.autosave_timer.stop()
    window.may_replace = lambda: True
    yield window
    window.dirty = False
    window.close()


def test_repeated_tokens_do_not_walk_qt_and_returned_lists_are_independent(owner, monkeypatch):
    # A sorted view still needs Qt's row mapping on the first read.
    owner.proxy.sort(1)
    calls = []
    original = owner.proxy.mapToSource
    owner.proxy._invalidate_row_ids()

    def tracked(index):
        calls.append(1)
        return original(index)

    monkeypatch.setattr(owner.proxy, "mapToSource", tracked)
    token = source_token(owner)
    walked = len(calls)
    assert walked >= len(owner.model.project.rows)
    assert isinstance(token[2], tuple)
    assert source_token(owner) == token
    ids = owner.row_ids()
    ids.clear()
    assert owner.row_ids() == list(token[2])
    assert len(calls) == walked


def test_direct_proxy_changes_are_detected_without_owner_revision(owner):
    project = owner.model.project
    original = source_token(owner)
    revision = owner.analysis_revision
    owner.proxy.sort(1)
    sorted_token = source_token(owner)
    assert sorted_token[:2] == original[:2]
    assert sorted_token[2] == tuple(project.rows[i].id for i in (1, 2, 0))
    owner.proxy.set_rules([{"column": project.columns[1].id, "operator": "lt", "value": 25}])
    first = source_token(owner)
    assert first[2] == tuple(project.rows[i].id for i in (1, 2))
    owner.proxy.set_rules([{"column": project.columns[1].id, "operator": "ge", "value": 20}])
    second = source_token(owner)
    assert second[2] == tuple(project.rows[i].id for i in (2, 0))
    assert len(first[2]) == len(second[2]) and first != second
    owner.proxy.sort_rules = [(project.columns[1].id, True)]
    owner.proxy.invalidate()
    assert source_token(owner)[2] == tuple(project.rows[i].id for i in (0, 2))
    assert owner.analysis_revision == revision


def test_unsorted_vectorized_filter_reuses_scope_without_walking_qt(owner, monkeypatch):
    project = build_project(["Group"], [["A" if i % 2 == 0 else "B"] for i in range(4200)],
                            list(range(4200)), [(0, "Group", "text", "")], {})
    owner.set_project(project)
    proxy = owner.proxy
    proxy.sort(-1)
    column = project.columns[0].id
    proxy.set_rules([{"column": column, "operator": "eq", "value": "A"}])
    expected = tuple(row.id for row in project.rows[::2])
    assert tuple(project.rows[proxy.mapToSource(proxy.index(i, 0)).row()].id
                 for i in range(proxy.rowCount())) == expected

    def reject_walk(*args):
        pytest.fail("The vectorized filter already established the unsorted row order")

    with monkeypatch.context() as patch:
        patch.setattr(proxy, "mapToSource", reject_walk)
        assert proxy.ordered_row_ids() == expected
    assert owner.model.setData(owner.model.index(0, 0), "B")
    assert proxy.ordered_row_ids() == expected[1:]


@pytest.mark.parametrize("operation", ["insert", "remove", "move", "reset"])
def test_native_row_changes_invalidate_cached_scope(owner, operation):
    model = owner.model
    project = model.project
    before = source_token(owner)
    if operation == "insert":
        model.beginInsertRows(QModelIndex(), 1, 1)
        project.rows.insert(1, Row(dict(project.rows[0].values)))
        model.endInsertRows()
    elif operation == "remove":
        model.beginRemoveRows(QModelIndex(), 1, 1)
        project.rows.pop(1)
        model.endRemoveRows()
    elif operation == "move":
        assert model.beginMoveRows(QModelIndex(), 0, 0, QModelIndex(), 3)
        project.rows.append(project.rows.pop(0))
        model.endMoveRows()
    else:
        model.beginResetModel()
        project.rows.reverse()
        model.endResetModel()
    after = source_token(owner)
    assert after[:2] == before[:2]
    assert after[2] == tuple(row.id for row in project.rows)
    assert after != before


def test_data_notification_and_revision_guard_refresh_uuid_snapshot(owner):
    model = owner.model
    before = source_token(owner)
    replacement = Row(dict(model.project.rows[0].values))
    model.project.rows[0] = replacement
    model.dataChanged.emit(model.index(0, 0), model.index(0, 2))
    notified = source_token(owner)
    assert notified[2][0] == replacement.id and notified != before
    model.project.rows[0] = Row(dict(replacement.values))
    model.data_revision += 1  # Isolated derived updates can omit dataChanged.
    assert source_token(owner)[2][0] == model.project.rows[0].id


def test_source_and_project_replacement_do_not_reuse_former_ids(owner):
    before = source_token(owner)
    replacement = deepcopy(owner.model.project)
    replacement.rows[0] = Row(dict(replacement.rows[0].values))
    owner.model.project = replacement
    assert source_token(owner)[2][0] == replacement.rows[0].id
    # Normal GUI replacement changes model identity even when UUIDs are equal.
    owner.set_project(deepcopy(replacement))
    assert source_token(owner)[0] != before[0]
    owner.proxy.setSourceModel(None)
    assert owner.row_ids() == []
