"""A new independent column must refresh dependent views without a global resort."""
from copy import deepcopy

import pytest
from PySide6.QtCore import QItemSelectionModel, QModelIndex, QPersistentModelIndex

from autopyro.calculations import make_calculated_column
from autopyro.io import build_project
from autopyro.ui.table import AddColumnCommand, FilterProxy, TableModel


class FullRefreshModel(TableModel):
    def refresh_derived_structure(self, column_id):
        self.refresh(invalidate_interpolation=False, sync_measurements=False)


@pytest.fixture
def project():
    return build_project(["Name", "Depth", "TOC", "S2"],
                         [["001", "30", "2", "6"], ["001", "10", "2", "4"],
                          ["001", "20", "2", "8"], ["001", "40", "2", "ND"]], [2, 3, 4, 5],
                         [(0, "Name", "text", ""), (1, "Depth", "number", "m"),
                          (2, "TOC", "number", "%wt"), (3, "S2", "number", "mg HC/g rock")], {})


def hi(project):
    return make_calculated_column(project, "HI", {
        "formula": "HI", "version": 1, "analysis_context": "Synthetic whole rock",
        "compatible_confirmed": True,
        "inputs": {c.name: {"column_id": c.id, "unit": c.unit} for c in project.columns[2:]}})


def row_ids(proxy):
    project = proxy.sourceModel().project
    return [project.rows[proxy.mapToSource(proxy.index(i, 0)).row()].id for i in range(proxy.rowCount())]


def test_unrelated_structure_does_not_recheck_sort_or_filter(qtbot, project, monkeypatch):
    model = TableModel(project)
    proxy = FilterProxy()
    proxy.setSourceModel(model)
    proxy.set_rules([{"column": project.columns[1].id, "operator": "gt", "value": 0}])
    proxy.sort(1)
    expected = row_ids(proxy)
    selection = QItemSelectionModel(proxy)
    selected = proxy.mapFromSource(model.index(2, 1))
    selection.select(selected, QItemSelectionModel.SelectionFlag.Select)
    persistent = QPersistentModelIndex(selected)
    calls, cell_notifications, changes = [], [], []
    compare, accept = proxy.lessThan, proxy.filterAcceptsRow
    monkeypatch.setattr(proxy, "lessThan", lambda a, b: (calls.append("sort"), compare(a, b))[1])
    monkeypatch.setattr(proxy, "filterAcceptsRow", lambda r, p: (calls.append("filter"), accept(r, p))[1])
    model.dataChanged.connect(lambda *args: cell_notifications.append(1))
    model.changed.connect(lambda: changes.append(1))
    column, values = hi(project)
    revision = model.data_revision
    for action in (lambda: model.undo.push(AddColumnCommand(model, column, values)),
                   model.undo.undo, model.undo.redo):
        action()
        assert row_ids(proxy) == expected
        assert project.rows[proxy.mapToSource(QModelIndex(persistent)).row()].id == project.rows[2].id
        assert project.rows[proxy.mapToSource(selection.selectedIndexes()[0]).row()].id == project.rows[2].id
    assert calls == []
    assert cell_notifications == []
    assert len(changes) == 3 and model.data_revision == revision + 3


@pytest.mark.parametrize("mode", ["filter", "multiple_sort"])
def test_related_rules_match_full_refresh_through_add_undo_redo(qtbot, project, mode):
    model = TableModel(project)
    reference = FullRefreshModel(deepcopy(project))
    column, values = hi(project)
    proxies = [FilterProxy(), FilterProxy()]
    for proxy, source in zip(proxies, (model, reference), strict=True):
        proxy.setSourceModel(source)
        if mode == "filter":
            proxy.set_rules([{"column": column.id, "operator": "gt", "value": 250}])
            proxy.sort(1)
        else:
            proxy.sort_rules = [(column.id, True), (project.columns[1].id, False)]
            proxy.sort(0)
    science = deepcopy(project.science)
    model.undo.push(AddColumnCommand(model, column, values))
    reference.undo.push(AddColumnCommand(reference, deepcopy(column), values.copy()))
    for action in (None, "undo", "redo"):
        if action:
            getattr(model.undo, action)()
            getattr(reference.undo, action)()
        assert row_ids(proxies[0]) == row_ids(proxies[1])
        assert [r.values for r in project.rows] == [r.values for r in reference.project.rows]
        assert project.science == science
        if action != "undo":
            expected = [project.rows[2].id, project.rows[0].id]
            if mode == "multiple_sort":
                expected += [project.rows[1].id, project.rows[3].id]
            assert row_ids(proxies[0]) == expected


def test_switching_source_disconnects_structure_notifications(qtbot, project):
    first, second = TableModel(project), TableModel(deepcopy(project))
    column, values = hi(project)
    proxy = FilterProxy()
    proxy.setSourceModel(first)
    proxy.set_rules([{"column": column.id, "operator": "gt", "value": 250}])
    proxy.setSourceModel(second)
    calls = []
    proxy.filtersChanged.connect(lambda: calls.append(1))
    first.undo.push(AddColumnCommand(first, column, values))
    assert calls == [] and proxy.rowCount() == 0
    second.undo.push(AddColumnCommand(second, deepcopy(column), values.copy()))
    assert calls == [1] and proxy.rowCount() == 2
    proxy.setSourceModel(None)
    second.undo.undo()
    assert calls == [1]


def check_filter_cycle(project, changed_rules, combine, present_matches, missing_matches, monkeypatch):
    model = TableModel(project)
    column, values = hi(project)
    proxy = FilterProxy()
    proxy.setSourceModel(model)
    rules = [{"column": project.columns[1].id, "operator": "gt", "value": 15}]
    rules += [{"column": column.id, **rule} for rule in changed_rules]
    proxy.set_rules(rules, combine)
    proxy.sort(1)
    raw_matches = {0, 2, 3}

    def expected(changed_matches):
        accepted = raw_matches & changed_matches if combine == "AND" else raw_matches | changed_matches
        return [project.rows[i].id for i in (1, 2, 0, 3) if i in accepted]

    previous = expected(missing_matches)
    assert row_ids(proxy) == previous
    preparations = []
    ranks = proxy.sort_ranks

    def tracked(column):
        preparations.append(1)
        return ranks(column)

    monkeypatch.setattr(proxy, "sort_ranks", tracked)
    for action, matches in ((lambda: model.undo.push(AddColumnCommand(model, column, values)), present_matches),
                            (model.undo.undo, missing_matches), (model.undo.redo, present_matches)):
        preparations.clear()
        action()
        current = expected(matches)
        assert row_ids(proxy) == current
        if len(current) > len(previous):
            assert preparations == [1]  # Prepare ordering once for added members.
        elif len(current) < len(previous):
            assert preparations == []  # Removing members cannot need a re-sort.
        assert proxy._sort_ranks is None and model._sort_indexes == ()
        previous = current


@pytest.mark.parametrize("operator,target,present", [
    ("empty", None, {3}), ("not_empty", None, {0, 1, 2}),
    ("contains", "0", {0, 1, 2}), ("eq", 300, {0}), ("ne", 300, {1, 2}),
    ("gt", 300, {2}), ("ge", 300, {0, 2}), ("lt", 300, {1}), ("le", 300, {0, 1}),
    ("unsupported", None, set()),
])
@pytest.mark.parametrize("negate", [False, True])
@pytest.mark.parametrize("combine", ["AND", "OR"])
def test_filter_expansion_preserves_operators_negation_and_missing_values(
        qtbot, project, monkeypatch, operator, target, present, negate, combine):
    missing = {0, 1, 2, 3} if operator == "empty" else set()
    if negate:
        present = {0, 1, 2, 3} - present
        missing = {0, 1, 2, 3} - missing
    check_filter_cycle(project, [{"operator": operator, "value": target, "negate": negate}],
                       combine, present, missing, monkeypatch)


@pytest.mark.parametrize("first,combine,present,missing", [
    ("not_empty", "AND", {2}, set()),
    ("not_empty", "OR", {0, 1, 2}, set()),
    ("empty", "AND", set(), set()),
    ("empty", "OR", {2, 3}, {0, 1, 2, 3}),
])
def test_multiple_rules_on_the_changed_column(qtbot, project, monkeypatch, first, combine, present, missing):
    check_filter_cycle(project, [{"operator": first}, {"operator": "gt", "value": 300}],
                       combine, present, missing, monkeypatch)
