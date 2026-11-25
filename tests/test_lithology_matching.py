"""Name matching is conservative, preserves provenance, and scales with unique names."""

from copy import deepcopy

import pytest

from autopyro.lithology_matching import (
    LithologyMatcher, LithologyMapping, default_source, normalize_rock, source_columns,
)
from autopyro.stratigraphy import from_records


@pytest.fixture
def matcher():
    from stratapy.setup import formatting
    return LithologyMatcher((code, style[-1]) for code, style in formatting.lithologies.items())


def imported(values, field='lithology'):
    headers = ['wellbore_name', 'unit_name', 'unit_level', 'top_depth_m', 'bottom_depth_m', field]
    rows = [['A', f'Unit {i}', 'FORMATION', i * 10, (i + 1) * 10, value] for i, value in enumerate(values)]
    return from_records(headers, rows, list(range(2, len(rows) + 2)),
                        name='Names', depth_reference='MD RKB', source={'sha256': 'source'})


@pytest.mark.parametrize('name, code', [(' ГЛИНА ', '620'), ('Massive sandstone', '607'),
                                     ('  massive\tSANDSTONE ', '607'), ('Песчаник', '607'),
                                     ('Известняк', '627'), ('607*', '607*'), ('620B', '620b')])
def test_exact_names_codes_and_explicit_aliases(matcher, name, code):
    assert matcher.match(name)[0] == code


def test_ambiguous_mixed_and_unknown_descriptions_are_not_guessed(matcher):
    assert matcher.match('Crossbedded sandstone') == (None, 'ambiguous')
    assert matcher.match('песчаник с прослоями известняка') == (None, 'unknown')
    assert matcher.match('sandston') == (None, 'unknown')
    assert matcher.match('608') == ('608', 'code')
    assert normalize_rock('ТЁМНЫЙ\u00a0  сланец') == 'темный сланец'


def test_saved_rules_custom_names_and_stale_rules():
    entries = [('607', 'Massive sandstone'), ('609', 'Crossbedded sandstone'),
               ('610', 'Crossbedded sandstone'), ('my_rock', 'My rock')]
    matcher = LithologyMatcher(entries, {'crossbedded sandstone': '610', 'местная порода': 'my_rock',
                                        'нет данных': '', 'old': 'missing'})
    assert matcher.match('Crossbedded sandstone') == ('610', 'saved')
    assert matcher.match('Местная порода') == ('my_rock', 'saved')
    assert matcher.match('my rock') == ('my_rock', 'name')
    assert matcher.match('нет данных') == ('', 'saved')
    assert matcher.match('old') == (None, 'unknown')


def test_mapping_groups_names_preserves_ids_and_raw_cells(matcher):
    scheme = imported(['песчаник', ' ПЕСЧАНИК ', 'mystery', '607*', ''])
    before = deepcopy(scheme)
    mapping = LithologyMapping(scheme, matcher)
    assert len(mapping.groups) == 3
    assert mapping.groups['песчаник'].count == 2
    result = mapping.apply()
    assert scheme == before
    assert [item.get('lithology', '') for item in result['intervals']] == ['607', '607', '', '607*', '']
    for old, new in zip(before['intervals'], result['intervals']):
        assert old['raw_cells'] == new['raw_cells']
        assert (old['id'], old['source_row'], old['name']) == (new['id'], new['source_row'], new['name'])
    assert result['intervals'][2]['lithology_source'] == 'mystery'
    assert result['source'] == before['source']


def test_alternative_source_column_and_manual_rule(matcher):
    scheme = imported(['Известняк', 'Известняк', 'Local'], field='Порода')
    columns = source_columns(scheme)
    assert default_source(scheme, columns) == 'Порода'
    mapping = LithologyMapping(scheme, matcher, 'Порода')
    group = mapping.groups['local']
    group.code, group.reason = '607*', 'manual'
    assert mapping.learned_rules() == {'local': '607*'}
    result = mapping.apply()
    assert [item['lithology'] for item in result['intervals']] == ['627', '627', '607*']
    assert result['intervals'][2]['lithology_source_field'] == 'Порода'


def test_matcher_runs_once_per_unique_name_without_decoding_images(matcher, monkeypatch):
    import autopyro.ui.widgets.lithology as catalog
    def unexpected(*args, **kwargs):
        pytest.fail('Bulk matching must not decode pattern images')
    monkeypatch.setattr(catalog, 'pattern_thumbnail', unexpected)
    calls = []
    match = matcher.match
    def record(value):
        calls.append(value)
        return match(value)
    monkeypatch.setattr(matcher, 'match', record)
    scheme = {'intervals': [{'lithology': value} for value in ['Глина', 'Песчаник'] * 5000]}
    mapping = LithologyMapping(scheme, matcher)
    assert len(calls) == 2
    assert all(group.count == 5000 for group in mapping.groups.values())
