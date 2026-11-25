from autopyro.stratigraphy import _DepthIndex
from autopyro.stratigraphy_revisions import interval_changes, membership_key


def test_revision_matching_keeps_stable_identity_across_reimport():
    base = {'wellbore': 'A', 'level': 'FORMATION', 'name': 'Sand', 'top': 0, 'bottom': 10}
    before = {'depth_unit': 'm', 'depth_reference': 'MD', 'intervals': [
        {**base, 'id': 'old'},
        {**base, 'id': 'gone', 'name': 'Clay', 'top': 20, 'bottom': 30},
    ]}
    after = {**before, 'intervals': [
        {**base, 'id': 'new', 'bottom': 12},
        {**base, 'id': 'added', 'name': 'Shale', 'top': 30, 'bottom': 40},
    ]}
    changes, identities = interval_changes(before, after)
    assert [entry[0] for entry in changes] == [('bottom',), ('removed',), ('added',)]
    assert [entry[3] for entry in changes] == ['name_overlap', None, None]
    assert identities == {'new': 'old'}
    old = {'status': 'matched', 'interval_ids': ['old'], 'names': ['Sand']}
    new = {**old, 'interval_ids': ['new']}
    assert membership_key(old, {}) == membership_key(new, identities)


def test_revision_matching_distinguishes_lithology_change_from_disjoint_same_name():
    base = {'id': 'old', 'wellbore': 'A', 'level': 'FORMATION', 'name': 'Sand', 'top': 0, 'bottom': 10}
    before = {'depth_unit': 'm', 'depth_reference': 'MD', 'intervals': [base]}
    after = {**before, 'intervals': [{**base, 'id': 'new', 'lithology': '607'}]}
    changes, identities = interval_changes(before, after)
    assert [(fields, method) for fields, _, _, method in changes] == [(('lithology',), 'name_bounds')]
    assert identities == {'new': 'old'}

    after['intervals'][0].update(top=10, bottom=20)
    changes, identities = interval_changes(before, after)
    assert [entry[0] for entry in changes] == [('removed',), ('added',)]
    assert identities == {}


def test_depth_index_preserves_order_and_half_open_overlap():
    items = [{'top': 5, 'bottom': 15}, {'top': 0, 'bottom': 20}, {'top': 20, 'bottom': 30}]
    index = _DepthIndex(items)
    assert index.hits(5) == items[:2]
    assert index.hits(15) == [items[1]]
    assert index.hits(20) == [items[2]]
    assert index.hits(15, 20) == [items[1]]
    assert index.hits(19, 21) == items[1:]
