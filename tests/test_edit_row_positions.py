"""Cell edits resolve UUIDs without scanning the table on the usual path."""
from copy import deepcopy

import pytest

from autopyro.core import Column, Project, Row
from autopyro.ui.table import TableModel
from autopyro.ui.table.commands import EditCommand


def model():
    column = Column('X', 'number')
    return TableModel(Project(columns=[column], rows=[Row({column.id: n}) for n in range(5)]), prepared=True)


@pytest.mark.parametrize('position', [0, 2, 4])
def test_single_edit_undo_redo_do_not_iterate_the_source_rows(qtbot, position):
    table = model()
    class AddressedRows(list):
        def __iter__(self):
            raise AssertionError('A cell edit must not scan all source rows')
    table.project.rows = AddressedRows(table.project.rows)
    changed = []
    table.dataChanged.connect(lambda first, last, roles: changed.append((first.row(), last.row())))
    assert table.setData(table.index(position, 0), '12')
    table.undo.undo()
    assert table.project.rows[position].values[table.project.columns[0].id] == position
    table.undo.redo()
    assert table.project.rows[position].values[table.project.columns[0].id] == 12
    assert changed == [(position, position)] * 3
    assert [event['action'] for event in table.project.history] == ['edit', 'undo_edit', 'edit']


@pytest.mark.parametrize('reorder', [False, True])
def test_command_resolves_current_rows_after_reordering_and_replacement(qtbot, reorder):
    table = model()
    cid = table.project.columns[0].id
    assert table.setData(table.index(1, 0), '12')
    old = table.project.rows[1]
    table.beginResetModel()
    table.project.rows = deepcopy(list(reversed(table.project.rows)) if reorder else table.project.rows)
    table.endResetModel()
    position = 3 if reorder else 1
    changed = []
    table.dataChanged.connect(lambda first, last, roles: changed.append((first.row(), last.row())))
    table.undo.undo()
    assert table.project.rows[position].id == old.id
    assert table.project.rows[position].values[cid] == 1
    assert old.values[cid] == 12  # The command must not retain the retired Row.
    table.undo.redo()
    assert table.project.rows[position].values[cid] == 12
    assert changed == [(position, position)] * 2


def test_bulk_edit_keeps_uuid_identity_and_notification_range(qtbot):
    table = model()
    cid = table.project.columns[0].id
    original = [row.values[cid] for row in table.project.rows]
    changed = []
    table.dataChanged.connect(lambda first, last, roles: changed.append((first.row(), last.row())))
    changes = [(table.project.rows[n].id, cid, n + 10) for n in (4, 1)]
    table.undo.push(EditCommand(table, changes))
    assert [row.values[cid] for row in table.project.rows] == [0, 11, 2, 3, 14]
    table.undo.undo()
    assert [row.values[cid] for row in table.project.rows] == original
    assert changed == [(1, 4)] * 2
    applied, undone = table.project.history[-2:]
    assert [(entry["row"], entry["old"], entry["new"]) for entry in applied["changes"]] == [
        (table.project.rows[4].id, 4, 14), (table.project.rows[1].id, 1, 11),
    ]
    assert [(entry["old"], entry["new"]) for entry in undone["changes"]] == [(14, 4), (11, 1)]


@pytest.mark.parametrize('position', [-1, 99, 0])
def test_stale_or_invalid_position_hint_falls_back_to_uuid(qtbot, position):
    table = model()
    cid = table.project.columns[0].id
    rid = table.project.rows[2].id
    table.undo.push(EditCommand(table, [(rid, cid, 12)], row_positions={rid: position}))
    assert table.project.rows[2].values[cid] == 12
    assert table.project.rows[0].values[cid] == 0


@pytest.mark.parametrize('dependency', ['method', 'stratigraphy', 'nonpyro'])
def test_cross_row_dependencies_keep_full_notification(qtbot, dependency):
    table = model()
    table.project.columns[0].metadata[dependency] = True
    changed = []
    table.dataChanged.connect(lambda first, last, roles: changed.append((first.row(), last.row())))
    row = table.project.rows[2]
    table.notify_changed(rows=[row], row_positions={row.id: 2})
    assert changed == [(0, 4)]


def test_bulk_edit_syncs_changed_cells_and_initializes_new_samples(qtbot, monkeypatch):
    from autopyro.model.measurements import Measurement

    edited, untouched = Column("Edited"), Column("Untouched")
    first, second = [Row({edited.id: "old", untouched.id: "keep"}) for _ in range(2)]
    project = Project(columns=[edited, untouched], rows=[first, second])
    project.sync_measurements(rows=[first])
    table = TableModel(project, prepared=True)
    other = project.science.current(first.id, untouched.id)
    table_value = Measurement.table_value

    def read(measurement):
        assert measurement is not other
        return table_value(measurement)

    monkeypatch.setattr(Measurement, "table_value", read)
    table.undo.push(EditCommand(table, [(first.id, edited.id, "new"), (second.id, edited.id, "new")]))
    assert project.science.current(second.id, untouched.id).value == "keep"
    assert project.science.current(first.id, untouched.id) is other
    table.undo.undo()
    assert project.science.current(first.id, edited.id).value == "old"
    assert project.science.current(second.id, edited.id).value == "old"


def test_missing_uuid_fails_before_mutating_any_cell(qtbot):
    table = model()
    cid = table.project.columns[0].id
    with pytest.raises(KeyError):
        EditCommand(table, [(table.project.rows[0].id, cid, 12), ('missing', cid, 13)])
    assert table.project.rows[0].values[cid] == 0
    assert not table.project.history
