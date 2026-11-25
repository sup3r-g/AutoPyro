"""Compact rows retain scientific fields and ordinary copy/extension semantics."""
from copy import deepcopy
from dataclasses import asdict, fields, replace
import gc
import pickle
import weakref

import pytest

from autopyro.core import Column, Project, Row
from autopyro.database.storage import encode_record
from autopyro.snapshots import copy_project, copy_table_state


def row():
    return Row({'number': 2 ** 100, 'censored': '<0.1', 'missing': None}, id='sample-id',
               source={'sheet': 'Лаборатория'}, original={'raw': 'ND'})


def test_fields_stay_compact_writable_and_serializable():
    record = row()
    assert [item.name for item in fields(record)] == ['values', 'id', 'source', 'original']
    assert vars(record) == {}  # Inspecting extensions must not materialize the row fields.
    assert weakref.ref(record)() is record
    expected = asdict(record)
    assert 'sample-id' in encode_record(record)
    record.id = 'changed'
    assert replace(record, id='sample-id') == Row(**expected)


@pytest.mark.parametrize('copy', [deepcopy, lambda value: pickle.loads(pickle.dumps(value, protocol=5))])
def test_copy_retains_extensions_shared_cells_and_cycles(copy):
    record = row()
    shared = {'notes': ['original']}
    record.source['shared'] = shared
    record.values['shared'] = shared
    record.extra = {'shared': shared, 'owner': record}
    copied = copy(record)
    assert copied is not record and copied.extra['owner'] is copied
    assert copied.extra['shared'] is copied.source['shared'] is copied.values['shared']
    copied.extra['shared']['notes'].append('changed')
    assert shared == {'notes': ['original']}
    assert copied.id == record.id and copied.original == record.original


def test_old_dict_pickle_state_populates_slots(monkeypatch):
    record = row()
    record.extra = {'owner': record}
    # Produce the former Row payload shape using the real Row class/opcodes.
    def old_state(self):
        return {**{item.name: getattr(self, item.name) for item in fields(self)}, **vars(self)}
    monkeypatch.setattr(Row, '__getstate__', old_state)
    payload = pickle.dumps(record, protocol=4)
    copied = pickle.loads(payload)
    assert asdict(copied) == asdict(record)
    assert copied.extra['owner'] is copied
    assert vars(copied) == {'extra': copied.extra}


@pytest.mark.parametrize('table_only', [False, True])
def test_scientific_snapshots_keep_row_extensions_and_cross_layer_aliases(table_only):
    column = Column('S2', 'number')
    record = Row({column.id: '<0.1'})
    project = Project(columns=[column], rows=[record])
    project.sync_measurements()
    shared = {'items': ['original']}
    record.extra = shared
    record.source['shared'] = shared
    project.view['shared'] = shared
    if table_only:
        columns, rows, science, view = copy_table_state((project.columns, project.rows, project.science, project.view))
        copied = rows[0]
    else:
        frozen = copy_project(project)
        copied, view = frozen.rows[0], frozen.view
    assert copied.extra is copied.source['shared'] is view['shared']
    copied.extra['items'].append('changed')
    assert shared == {'items': ['original']}


def test_save_reopen_retains_row_schema_and_scientific_values(tmp_path):
    column = Column('S2', 'number')
    project = Project(columns=[column], rows=[Row({column.id: '<0.1'}, source={'sheet': 'Lab'},
                                                 original={column.id: '<0.1'})])
    project.sync_measurements()
    before = asdict(project.rows[0])
    target = tmp_path / 'compact.apyro'
    project.save(target)
    restored = Project.load(target)
    assert asdict(restored.rows[0]) == before
    assert restored.science.current(restored.rows[0].id, column.id).qualifier == 'below_limit'


def test_extended_row_cycles_are_still_collected():
    record = row()
    record.extra = {'owner': record}
    record.source['owner'] = record
    reference = weakref.ref(record)
    del record
    gc.collect()
    assert reference() is None
