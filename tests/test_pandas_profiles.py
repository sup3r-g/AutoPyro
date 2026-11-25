"""Scalar-equivalence and literal-value contracts for the remaining batch paths."""


import numpy as np
import pandas as pd
import pytest
from dataclasses import replace
import json

from autopyro.core import Column, Project, Row
from autopyro.interpretation_quality import InterpretationQuality
from autopyro.nonpyro import _interpret_rows, interpret_cell
from autopyro.regional_geochemistry import RegionalOptions, analyze_regions
from autopyro.mapping import MapSpec, sample_points
from autopyro.plotting import category_key, category_keys
from autopyro.tabular import classify, classify_series, depth_filter_masks


@pytest.mark.parametrize('numeric', [False, True])
def test_bulk_classifier_matches_domain_parser_without_rounding_integers(numeric):
    values = [None, '', '  ', True, False, 0, 1, -1, 1.5, float('nan'), float('inf'),
              9007199254740993, 10 ** 400, '001', '1,5', 'ND', 'n.d.', '<0.01', '≥1',
              'NaN', 'bad', np.int64(12), np.float64(1.25)] * 20
    for index in [pd.RangeIndex(len(values)), pd.Index(['same'] * len(values))]:
        source = pd.Series(values, index=index, dtype=object)
        result = classify_series(source, numeric=numeric)
        assert list(result.itertuples(index=False, name=None)) == [classify(value, numeric=numeric) for value in values]
        pd.testing.assert_series_equal(source, pd.Series(values, index=index, dtype=object))


@pytest.mark.parametrize('values', [['A', 'a', ' A ', None, ''], [1, '1', 1., True, -0., 0., None]])
def test_repeated_category_keys_preserve_type_spelling_and_signed_zero(values):
    source = pd.Series(values * 100, dtype=object)
    assert category_keys(source).tolist() == [category_key(value) for value in source]


@pytest.mark.parametrize('bottom,lower,upper,expected', [
    (None, 10, 20, ([False, False, False, False, True], [True, False, False, True, True])),
    ([10, 20, 30, 40, 50], 20, 20, ([False, False, False, False, True], [True, True, False, True, False])),
    ([0, 10, 25, 30, None], 10, 25, ([True, True, False, True, True], [True, True, False, True, True])),
])
def test_depth_masks_keep_half_open_boundaries(bottom, lower, upper, expected):
    depth = pd.Series([0, 10, 20, 30, None], dtype=object)
    bottom = pd.Series(bottom, dtype=object) if bottom is not None else None
    invalid, outside = depth_filter_masks(depth, bottom, lower, upper)
    assert (invalid.tolist(), outside.tolist()) == expected


def test_quality_batch_preserves_each_message_order_and_axis_block():
    columns = [Column(name, 'text', id=name, unit=unit) for name, unit in
               [('TOC', '%wt'), ('HI', 'mg HC/g TOC'), ('S2', 'mg HC/g rock'), ('Tmax', '°C')]]
    values = [[1, 200, 5, 430], [101, 200, 5, 430], [.01, 'ND', '<1', 320],
              [None, 1300, -1, 700], [0, 0, 0, None], ['1,5', 'bad', 1, 450]]
    rows = [Row(dict(zip([c.id for c in columns], values[i % len(values)])), id=f'row-{i}') for i in range(300)]
    project = Project(columns=columns, rows=rows)
    project.sync_measurements(rows=rows[:1])
    measurement = project.science.current(rows[0].id, 'TOC')
    project.view['qc_exclusions'] = {'interpretation': {measurement.id: {'revision': measurement.revision}}}
    for mode in ['normal', 'interpolated', 'unknown_unit']:
        if mode == 'interpolated':
            columns[2].metadata['method'] = 'linear'
        elif mode == 'unknown_unit':
            columns[2].metadata.clear()
            columns[2].unit = 'unknown'
        checker = InterpretationQuality(project, 'TOC', 'HI', '%wt', 'mg HC/g TOC',
                                         {'x': {'property': 'TOC'}, 'y': {'property': 'HI'}}, True)
        assert checker.check_many(rows) == {row.id: checker.check(row) for row in rows}
    assert checker.check_many([]) == {}


def test_strict_projection_failure_does_not_discard_other_points(monkeypatch):
    import pyproj

    class Operation:
        description, accuracy = 'Test operation', 0

        def transform(self, x, y, *, errcheck):
            assert errcheck is True
            if np.any(np.asarray(x) < 0):
                raise pyproj.exceptions.ProjError('Outside domain')
            return x + 1, y + 2

    monkeypatch.setattr('autopyro.mapping.transformer', lambda *args: Operation())
    columns = [Column('X', 'number', id='x', unit='m'), Column('Y', 'number', id='y', unit='m')]
    project = Project(columns=columns, rows=[Row({'x': x, 'y': 5}, id=str(i)) for i, x in enumerate([1, -1, 3])])
    spec = MapSpec(x_id='x', y_id='y', source_crs='EPSG:32631', target_crs='EPSG:32631')
    points, excluded, _ = sample_points(project, project.rows, spec, [])
    assert [(point['id'], point['x'], point['y']) for point in points] == [('0', 2., 7.), ('2', 4., 7.)]
    assert excluded == {'coordinate_transform_error': 1}


def test_layer_matching_keeps_typed_keys_ambiguity_and_coordinate_errors():
    from shapely.geometry import Point, mapping

    columns = [Column('Key', 'text', id='key')]
    rows = [Row({'key': value}, id=str(i)) for i, value in enumerate([1, '1', 2, None, 'bad'])]
    project = Project(columns=columns, rows=rows)
    entries = [(1, Point(1, 2)), ('1', Point(3, 4)), (2, Point(5, 6)), (2, Point(7, 8)),
               ('bad', Point(float('nan'), 1)), ('unused', Point())]
    layer = dict(id='points', crs='EPSG:32631', features=[
        dict(properties={'key': key}, geometry=mapping(point)) for key, point in entries])
    spec = MapSpec(location_mode='layer', point_layer='points', sample_key='key', layer_key='key', target_crs='EPSG:32631')
    points, excluded, _ = sample_points(project, rows, spec, [layer])
    assert [(point['id'], point['x'], point['y']) for point in points] == [('0', 1., 2.), ('1', 3., 4.)]
    assert excluded == {'unmatched_or_ambiguous_location': 2, 'coordinate_transform_error': 1}


def test_regional_empty_groups_and_typed_methods_keep_total_row_ids():
    from shapely.geometry import box, mapping

    columns = [Column('TOC', 'number', id='toc'), Column('Method', 'text', id='method'),
               Column('Well', 'text', id='well')]
    rows = [Row(dict(toc=value, method=method, well=well), id=f'row-{i}')
            for i, (value, method, well) in enumerate([(1, 1, 1), (3, 1, '1'), ('ND', 1, 1), (5, '1', 1), (None, None, None)])]
    project = Project(columns=columns, rows=rows)
    layer = dict(id='areas', name='Areas', crs='EPSG:32631', fields=['Region'], features=[
        dict(id='first', geometry=mapping(box(0, 0, 10, 10)), properties={'Region': 'A'}),
        dict(id='empty', geometry=mapping(box(20, 0, 30, 10)), properties={'Region': 'B'})])
    report = dict(row_ids=[r.id for r in rows], layers=[layer], spec={'target_crs': 'EPSG:32631'},
                  points=[dict(id=rows[0].id, row_ids=[r.id for r in rows], x=5, y=5)])
    options = RegionalOptions('areas', 'Region', toc_id='toc', well_id='well', toc_method_id='method')
    result = analyze_regions(project, report, options)
    integer_method = next(row for row in result['summary'] if row['region'] == 'A' and type(row['method']) is int)
    assert integer_method['row_ids'] == ['row-0', 'row-1', 'row-2']
    assert (integer_method['n_total'], integer_method['n_valid'], integer_method['n_wells'], integer_method['mean']) == (3, 2, 2, 2.)
    empty = next(row for row in result['summary'] if row['region'] == 'B')
    assert empty['row_ids'] == [] and empty['n_total'] == 0 and empty['mean'] is None
    unbound = analyze_regions(project, report, replace(options, toc_method_id='', well_id=''))
    assert unbound['summary'][0]['method'] is None
    assert all(value['well'] is None for entry in unbound['distributions'] for value in entry['values'])
    json.dumps(unbound, allow_nan=False)
    report['row_ids'], report['points'] = [], []
    empty = analyze_regions(project, report, options)
    assert len(empty['summary']) == 2
    assert all(row['n_total'] == 0 and row['mean'] is None for row in empty['summary'])


PROFILE_INPUTS = {
    'gas_ratios': [('c1', 90, '%vol'), ('c2', 6, '%vol'), ('c3', 4, '%vol')],
    'sara': [('sat', 60, '%wt'), ('arom', 20, '%wt'), ('res', 15, '%wt'), ('asph', 5, '%wt')],
    'elemental': [('c', 80, '%wt'), ('h', 8, '%wt'), ('o', 8, '%wt')],
    'atomic_ratios': [('hc', 120, '%mol (a/b)'), ('oc', 7.5, '%mol (a/b)')],
    'gc_profile': [('prph', 2, '1'), ('pr17', .8, '1'), ('ph18', .5, '1')],
    'stereo_profile': [('st20', 52, '% (a/(a+b))'), ('stbb', 60, '% (a/(a+b))'), ('h31', 58, '% (a/(a+b))')],
    'aromatic_profile': [('mpi', .8, '1'), ('dbt', 1.5, '1')],
    'microscopy': [('vit', 50, '%area'), ('lipt', 30, '%area'), ('inert', 20, '%area')],
    'extract_toc': [('extract', 2, 'mg/g rock'), ('toc', 4, '%wt')],
}


@pytest.mark.parametrize('method', PROFILE_INPUTS)
def test_batch_profiles_match_scalar_rules_for_every_supported_method(method):
    definitions = PROFILE_INPUTS[method]
    columns = [Column(f'Bench / {key}[un]', 'text', id=key, unit=unit) for key, _, unit in definitions]
    columns.append(Column('Bench / Quality', 'text', id='quality'))
    variations = [None, 'ND', '<0.1', 'bad', 0, -1, True, '1,5', 1e308]
    rows = []
    for i in range(300):
        values = {key: value for key, value, _ in definitions}
        if i % 2:
            values[definitions[i % len(definitions)][0]] = variations[i % len(variations)]
        values['quality'] = 'check' if i % 11 == 0 else None
        rows.append(Row(values, id=f'row-{i}'))
    project = Project(columns=columns, rows=rows)
    config = dict(method=method, text_numbers=True, inputs={key: {'column_id': key, 'unit': unit} for key, _, unit in reversed(definitions)})
    by_id = {column.id: column for column in columns}
    for mode in ['all', 'scope', 'interpolation']:
        if mode == 'scope':
            config['scope_row_ids'] = {row.id for row in rows[::2]}
        elif mode == 'interpolation':
            config.pop('scope_row_ids')
            for column in columns:
                column.metadata['method'] = 'linear'
        expected = [{'row_id': row.id, **interpret_cell(project, row, config, by_id)} for row in rows]
        assert _interpret_rows(project, rows, config, by_id) == expected
