"""Temporary source indexes preserve Qt identity and dynamic sorting semantics."""
from contextlib import contextmanager
from copy import deepcopy

import pytest
from PySide6.QtCore import QModelIndex, QItemSelectionModel, QPersistentModelIndex, Qt

from autopyro.core import Column, Project, Row
from autopyro.io import build_project
from autopyro.ui.table import FilterProxy, TableModel


class UncachedModel(TableModel):
    @contextmanager
    def cache_sort_indexes(self, column):
        yield


class ComparisonProxy(FilterProxy):
    def sort_ranks(self, column):
        return None


def test_large_nested_filter_uses_batch_mask(qtbot, monkeypatch):
    depth = Column("Depth", "number")
    project = Project([depth], [Row({depth.id: i}) for i in range(5000)])
    model = TableModel(project, prepared=True)
    proxy = FilterProxy()
    proxy.setSourceModel(model)
    assert proxy.rowCount() == 5000
    monkeypatch.setattr(proxy, "_matches_leaf", lambda *args: pytest.fail("Expected a batch mask"))
    proxy.set_rules([{"combine": "OR", "rules": [
        {"column": depth.id, "operator": "lt", "value": 2},
        {"combine": "AND", "negate": True, "rules": [
            {"column": depth.id, "operator": "lt", "value": 4998},
        ]},
    ]}])
    assert proxy.ordered_row_ids() == tuple(project.rows[i].id for i in (0, 1, 4998, 4999))


@pytest.mark.parametrize("primary", [
    [2**100, 2**53 + 1, 2**53, None, 2**53 + 1, 0, 0],
    [None] * 7,
    [2, "<0.1", "ND", None, 2, 0, "≈1.2"],
    ["A", "a", "B", None, "A", "b", "a"],
])
@pytest.mark.parametrize("directions", [(False, True), (True, False)])
@pytest.mark.parametrize("order", [Qt.SortOrder.AscendingOrder, Qt.SortOrder.DescendingOrder])
def test_multiple_key_ranks_match_comparisons_and_expire_after_sort(qtbot, primary, directions, order):
    columns = [Column("Primary"), Column("Secondary")]
    project = Project(columns=columns, rows=[
        Row({columns[0].id: value, columns[1].id: secondary})
        for value, secondary in zip(primary, [1, 3, 2, 0, 1, 2, 2], strict=True)])
    models = [TableModel(project, prepared=True), TableModel(deepcopy(project), prepared=True)]
    proxies = [FilterProxy(), ComparisonProxy()]
    for proxy, model in zip(proxies, models, strict=True):
        proxy.setSourceModel(model)
        proxy.sort_rules = [(c.id, descending) for c, descending in zip(columns, directions, strict=True)]
        proxy.sort(0, order)
    assert row_ids(proxies[0]) == row_ids(proxies[1])
    assert proxies[0]._sort_ranks is None and models[0]._sort_indexes == ()
    # A subsequent implicit Qt re-sort must read current values, not former ranks.
    for model in models:
        model.project.rows[0].values[columns[0].id] = primary[-1]
        model.notify_changed()
    assert row_ids(proxies[0]) == row_ids(proxies[1])


def row_ids(proxy):
    project = proxy.sourceModel().project
    return [project.rows[proxy.mapToSource(proxy.index(i, 0)).row()].id for i in range(proxy.rowCount())]


@pytest.mark.parametrize("column", [0, 1, 2])
def test_atomic_rule_changes_match_legacy_and_preserve_selection(qtbot, project, column, monkeypatch):
    model = TableModel(project)
    proxy = FilterProxy()
    reference = ComparisonProxy()
    proxy.setSourceModel(model)
    reference.setSourceModel(TableModel(deepcopy(project)))
    selection = QItemSelectionModel(proxy)
    index = proxy.mapFromSource(model.index(3, 1))
    selection.setCurrentIndex(index, QItemSelectionModel.SelectionFlag.Select)
    persistent = QPersistentModelIndex(index)
    calls = []
    compare = proxy.lessThan
    monkeypatch.setattr(proxy, "lessThan", lambda a, b: (calls.append(1), compare(a, b))[1])
    for rules, native, order in [
        ([(project.columns[column].id, False)], 0, Qt.SortOrder.AscendingOrder),
        ([(project.columns[column].id, True)], 0, Qt.SortOrder.AscendingOrder),
        ([(project.columns[1].id, False)], 0, Qt.SortOrder.AscendingOrder),
        ([], 2, Qt.SortOrder.DescendingOrder),
        ([], -1, Qt.SortOrder.AscendingOrder),
    ]:
        calls.clear()
        proxy.set_sort_rules(rules, native, order)
        reference.sort_rules = rules
        reference.invalidate()
        reference.sort(native, order)
        assert row_ids(proxy) == row_ids(reference)
        for selected in (selection.currentIndex(), selection.selectedIndexes()[0], QModelIndex(persistent)):
            assert project.rows[proxy.mapToSource(selected).row()].id == project.rows[3].id
        if native == -1:
            assert calls == []  # Reset must not sort by the old comparison first.
        calls.clear()
        proxy.set_sort_rules(rules, native, order)
        assert calls == []  # Reapplying identical settings is a no-op.
        assert proxy._sort_ranks is None and model._sort_indexes == ()


@pytest.fixture
def project():
    return build_project(["Name", "Depth", "S2"],
                         [["B", "2", "ND"], ["A", "", "<0.1"], ["A", "1", "3"],
                          ["b", "2", ""], ["B", "10", "2"], ["A", "0", "≈1.2"]],
                         list(range(2, 8)), [(0, "Name", "text", ""),
                                               (1, "Depth", "number", "m"),
                                               (2, "S2", "number", "mg HC/g rock")], {})


def test_cached_sort_matches_uncached_order_through_edits_filters_and_undo(qtbot, project):
    model = TableModel(project)
    reference = UncachedModel(deepcopy(project))
    proxies = [FilterProxy(), FilterProxy()]
    for proxy, source in zip(proxies, (model, reference), strict=True):
        proxy.setSourceModel(source)
    ids = [r.id for r in project.rows]
    science = deepcopy(project.science)
    for column, order in [(0, Qt.SortOrder.AscendingOrder), (1, Qt.SortOrder.AscendingOrder),
                          (1, Qt.SortOrder.DescendingOrder), (2, Qt.SortOrder.AscendingOrder),
                          (2, Qt.SortOrder.DescendingOrder), (-1, Qt.SortOrder.AscendingOrder)]:
        for proxy in proxies:
            proxy.sort(column, order)
        assert row_ids(proxies[0]) == row_ids(proxies[1])
        assert model._sort_indexes == ()
    rules = [{"column": project.columns[1].id, "operator": "gt", "value": 0}]
    for proxy in proxies:
        proxy.set_rules(rules)
        proxy.sort_rules = [(project.columns[0].id, False), (project.columns[1].id, True)]
        proxy.sort(0)
    assert row_ids(proxies[0]) == row_ids(proxies[1])
    assert project.science == science
    for source in (model, reference):
        assert source.setData(source.index(5, 1), "20")
    assert row_ids(proxies[0]) == row_ids(proxies[1])
    assert project.rows[5].id in row_ids(proxies[0])
    for source in (model, reference):
        source.undo.undo()
    assert row_ids(proxies[0]) == row_ids(proxies[1])
    assert project.rows[5].id not in row_ids(proxies[0])
    assert [r.id for r in project.rows] == ids


def test_selection_and_persistent_indexes_follow_sample_uuid(qtbot, project):
    model = TableModel(project)
    proxy = FilterProxy()
    proxy.setSourceModel(model)
    selection = QItemSelectionModel(proxy)
    selected = proxy.mapFromSource(model.index(3, 1))
    selection.select(selected, QItemSelectionModel.SelectionFlag.Select)
    persistent = QPersistentModelIndex(selected)
    expected = project.rows[3].id
    for column, order in [(0, Qt.SortOrder.AscendingOrder), (1, Qt.SortOrder.DescendingOrder),
                          (2, Qt.SortOrder.AscendingOrder), (-1, Qt.SortOrder.AscendingOrder)]:
        proxy.sort(column, order)
        for index in (selection.selectedIndexes()[0], QModelIndex(persistent)):
            assert project.rows[proxy.mapToSource(index).row()].id == expected
    proxy.set_rules([{"column": project.columns[1].id, "operator": "gt", "value": 0}])
    proxy.sort(1)
    assert project.rows[proxy.mapToSource(QModelIndex(persistent)).row()].id == expected


def test_sort_avoids_repeated_source_bounds_callbacks(qtbot, monkeypatch):
    project = build_project(["Depth"], [[str(i % 100)] for i in range(1000)],
                            list(range(2, 1002)), [(0, "Depth", "number", "m")], {})
    model = TableModel(project)
    proxy = FilterProxy()
    proxy.setSourceModel(model)
    calls = []
    row_count = model.rowCount

    def counted(parent=QModelIndex()):
        calls.append(1)
        return row_count(parent)

    monkeypatch.setattr(model, "rowCount", counted)
    proxy.sort(0)
    # Source rowCount must not cross the Qt/Python boundary for every comparison.
    assert len(calls) < len(project.rows)
    assert model._sort_indexes == ()
    assert [proxy.index(i, 0).data(Qt.ItemDataRole.UserRole) for i in range(proxy.rowCount())] == sorted(
        r.values[project.columns[0].id] for r in project.rows)


def test_index_bounds_and_exception_cleanup(qtbot, project):
    model = TableModel(project)
    with pytest.raises(RuntimeError, match="interrupted"):
        with model.cache_sort_indexes(1):
            assert model.index(3, 1).data(Qt.ItemDataRole.UserRole) == 2
            for row, column in [(-1, 1), (len(project.rows), 1), (0, -1), (0, len(project.columns))]:
                assert not model.index(row, column).isValid()
            assert not model.index(0, 1, model.index(0, 0)).isValid()
            raise RuntimeError("interrupted")
    assert model._sort_indexes == ()
    assert model._sort_index_column == -1
