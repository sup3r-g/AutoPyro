from decimal import Decimal
from types import SimpleNamespace

import pytest

from autopyro.database import stratigraphy as db
from autopyro.geology.stratigraphy_review import plan_merge


def row(**updates):
    return dict({
        'interpretation_id': 4, 'interpretation_interval_id': 92, 'interval_number': 1,
        'wellbore_id': 7, 'wellbore_name': 'A', 'version_tag': 'v2', 'status': 'published',
        'unit_name': 'Layer', 'unit_level': 'FORMATION', 'top_depth_m': Decimal('12.50'),
        'bottom_depth_m': Decimal('20.75'), 'depth_type': None, 'depth_datum': None,
        'depth_reference': 'not specified in source workbook', 'source_interval_id': 19,
        'interval_note': None,
    }, **updates)


def test_database_merge_validates_final_geometry_and_keeps_a_split_together():
    endpoint = {'host': 'local', 'port': 5432, 'database': 'geo'}
    original = row(top_depth_m=Decimal(0), bottom_depth_m=Decimal(20))
    neighbor = row(interpretation_interval_id=93, interval_number=2, source_interval_id=20,
                   top_depth_m=Decimal(20), bottom_depth_m=Decimal(30))
    current = db.scheme_from_interpretations([original, neighbor], endpoint, 'MD RKB')
    extended = db.scheme_from_interpretations(
        [{**original, 'bottom_depth_m': Decimal(25)}], endpoint, 'MD RKB')
    conflict, = plan_merge(current, extended)
    assert conflict.kind == 'conflict' and conflict.reason == 'overlap'
    assert conflict.conflict_with['id'] == current['intervals'][1]['id']

    split = db.scheme_from_interpretations([
        {**original, 'bottom_depth_m': Decimal(10)},
        row(interpretation_interval_id=94, interval_number=3, source_interval_id=21,
            top_depth_m=Decimal(10), bottom_depth_m=Decimal(20)),
    ], endpoint, 'MD RKB')
    assert [entry.kind for entry in plan_merge(current, split, update_database=False)] == ['update', 'conflict']
    assert [entry.kind for entry in plan_merge(current, split)] == ['update', 'add']


def test_db_read_selects_requested_revisions_and_keeps_json_safe_provenance(monkeypatch):
    records = [row()]
    calls = []

    class Connection:
        info = SimpleNamespace(host='localhost', port=15432, dbname='geo')

        def execute(self, sql, parameters):
            calls.append((sql, parameters))
            return SimpleNamespace(fetchall=lambda: records)

        def close(self):
            calls.append('closed')

    monkeypatch.setattr(db, 'connect', lambda params: Connection())
    parameters = {'password': 'not persisted'}
    scheme = db.read_interpretations(parameters, [4], 'MD RKB')
    assert 'ANY(%s)' in calls[0][0]
    assert calls[0][1] == ([4], 100001)
    assert calls[1] == 'closed'
    assert scheme == db.read_interpretations(parameters, [4], 'MD RKB')
    assert scheme['source']['interpretations'][0]['version_tag'] == 'v2'
    assert 'password' not in scheme['source']['endpoint']
    interval = scheme['intervals'][0]
    assert interval['top'] == 12.5
    assert interval['raw_cells']['top_depth_m'] == '12.50'
    assert interval['raw_cells']['depth_reference'] == 'not specified in source workbook'
    records.clear()
    with pytest.raises(ValueError, match='недоступны'):
        db.read_interpretations(parameters, [4], 'MD RKB')


def test_db_import_rejects_mixed_depth_systems_and_two_revisions_of_one_well():
    with pytest.raises(ValueError, match='Система глубин'):
        db.scheme_from_interpretations([row(depth_type='TVD')], {}, 'MD RKB')
    with pytest.raises(ValueError, match='Система глубин'):
        db.scheme_from_interpretations([row(depth_reference='MD RKB', depth_type='TVD')], {}, 'MD RKB')
    with pytest.raises(ValueError, match='одну редакцию'):
        db.scheme_from_interpretations([row(), row(interpretation_id=5, interpretation_interval_id=93)],
                                       {}, 'MD RKB')


@pytest.mark.parametrize(('changes', 'kind'), [
    ({}, 'update'),
    ({'source_interval_id': 22}, 'conflict'),
    ({'source_interval_id': None}, 'conflict'),
    ({'wellbore_id': 8}, 'conflict'),
])
def test_db_revision_merge_matches_stable_source_not_geometry(changes, kind):
    endpoint = {'host': 'localhost', 'port': 5432, 'database': 'geo'}
    current = db.scheme_from_interpretations([row()], endpoint, 'MD RKB')
    previous = current['intervals'][0]
    previous.update(lithology='607', color='#aaaaaa')
    revised = row(**{'interpretation_id': 5, 'interpretation_interval_id': 93,
                     'top_depth_m': Decimal('13.25'), 'bottom_depth_m': Decimal('21.75'), **changes})
    incoming = db.scheme_from_interpretations([revised], endpoint, 'MD RKB')
    incoming['intervals'][0]['lithology'] = '608'
    entry, = plan_merge(current, incoming)
    assert entry.kind == kind
    if kind == 'update':
        assert entry.item['id'] == entry.target == previous['id']
        assert (entry.previous['top'], entry.previous['bottom']) == (12.5, 20.75)
        assert (entry.item['top'], entry.item['bottom']) == (13.25, 21.75)
        assert entry.item['lithology'] == '607'
        assert entry.item['color'] == '#aaaaaa'
        assert entry.item['raw_cells']['interpretation_interval_id'] == 93
        updated = {**current, 'intervals': [entry.item]}
        assert plan_merge(updated, incoming)[0].kind == 'duplicate'


def test_db_revision_merge_rejects_ambiguous_ids_and_other_endpoints():
    endpoint = {'host': 'localhost', 'port': 5432, 'database': 'geo'}
    current = db.scheme_from_interpretations([row()], endpoint, 'MD RKB')
    revised = row(interpretation_id=5, interpretation_interval_id=93, top_depth_m=Decimal('13'))
    ambiguous = db.scheme_from_interpretations(
        [revised, {**revised, 'interpretation_interval_id': 94, 'interval_number': 2}], endpoint, 'MD RKB')
    assert [entry.kind for entry in plan_merge(current, ambiguous)] == ['conflict', 'conflict']
    other = db.scheme_from_interpretations([revised], {**endpoint, 'database': 'other'}, 'MD RKB')
    assert plan_merge(current, other)[0].kind == 'conflict'
