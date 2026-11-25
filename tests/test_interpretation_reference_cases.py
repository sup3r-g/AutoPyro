import json
from pathlib import Path

import pytest

from autopyro.schemes.interpretation import validate_scheme
from autopyro.schemes.reference_cases import missing_reference_checks, reference_failures
from test_standard_interpretation import curves


@pytest.fixture
def scheme():
    path = Path(__file__).parents[1] / 'examples' / 'interpretation-reference-cases.json'
    return json.loads(path.read_text(encoding='utf-8'))


def test_audit_cases_detect_lost_interval_and_reversed_threshold(scheme):
    assert reference_failures(scheme) == []
    scheme['regions'][0]['geometry']['coordinates'][0] = [[0, 0], [2.2, 0], [2.2, 500], [0, 500], [0, 0]]
    scheme['lines'][0]['geometry']['coordinates'] = [[2.5, 300], [4, 300]]
    assert reference_failures(scheme) == [
        ('interval-leaves-one-region', 'status', 'ambiguous', 'classified'),
        ('above-threshold', 'thresholds.cutoff', 'greater', 'less'),
    ]


def test_reference_detects_lost_threshold_relations_with_same_nominal_side(scheme):
    scheme['reference_cases'] = [{
        'id': 'crosses-cutoff', 'x': 3, 'y': 105, 'y_bounds': [95, 110],
        'expected': {'thresholds': {'cutoff': {
            'relation': 'greater', 'possible_relations': ['greater', 'less', 'boundary'],
        }}},
    }]
    validate_scheme(scheme)
    assert reference_failures(scheme) == []

    for point in scheme['lines'][0]['geometry']['coordinates']:
        point[1] = 90
    assert reference_failures(scheme) == [
        ('crosses-cutoff', 'thresholds.cutoff.possible_relations', ['greater', 'less', 'boundary'], ['greater']),
    ]


@pytest.mark.parametrize('relations', [[], ['greater', 'unknown'], ['greater', 'greater'], 'greater'])
def test_threshold_reference_rejects_invalid_possible_relations(scheme, relations):
    scheme['reference_cases'][1]['expected']['thresholds']['cutoff'] = {
        'relation': 'greater', 'possible_relations': relations,
    }
    with pytest.raises(ValueError, match='above-threshold'):
        validate_scheme(scheme)


def test_bundled_gp_reference_cases_detect_shifted_cutoff():
    from autopyro.schemes.interpretation_review import review_scheme_hash
    from autopyro.schemes.standard_interpretation import standard_schemes

    scheme = standard_schemes()['S1+S2–TOC']
    assert scheme['revision'] == '4'
    assert scheme['reference_cases']
    assert reference_failures(scheme) == []
    previous = {key: value for key, value in scheme.items() if key != 'reference_cases'}
    previous['revision'] = '3'
    assert review_scheme_hash(previous) == review_scheme_hash(scheme)
    for region in scheme['regions']:
        ring, = region['geometry']['coordinates']
        region['geometry']['coordinates'] = [[(x, 2.3 if y == 2 else y) for x, y in ring]]
    failed_checks = {(case, field) for case, field, _, _ in reference_failures(scheme)}
    assert ('gp-boundary-2', 'boundary_regions') in failed_checks
    assert ('gp-interval-crosses-2', 'uncertainty_regions') in failed_checks
    assert not any(case.endswith('-6') for case, _ in failed_checks)


@pytest.mark.parametrize(('name', 'old_cutoff', 'new_cutoff'), [
    ('S1–TOC', .5, .65),
    ('S2–TOC', 2.5, 3.1),
    ('HI–TOC', 50, 65),
    ('PI–Tmax', .1, .15),
])
def test_bundled_grid_reference_cases_detect_shifted_cutoff(name, old_cutoff, new_cutoff):
    from autopyro.schemes.interpretation_review import review_scheme_hash
    from autopyro.schemes.standard_interpretation import standard_schemes

    scheme = standard_schemes()[name]
    assert scheme['revision'] == '3'
    assert reference_failures(scheme) == []
    previous = {key: value for key, value in scheme.items() if key != 'reference_cases'}
    previous['revision'] = '2'
    assert review_scheme_hash(previous) == review_scheme_hash(scheme)

    for region in scheme['regions']:
        ring, = region['geometry']['coordinates']
        region['geometry']['coordinates'] = [[(x, new_cutoff if y == old_cutoff else y) for x, y in ring]]
    failed_checks = {(case, field) for case, field, _, _ in reference_failures(scheme)}
    assert ('first-y-boundary', 'boundary_regions') in failed_checks
    assert ('shared-corner', 'boundary_regions') in failed_checks
    assert ('interval-crosses-first-y-boundary', 'uncertainty_regions') in failed_checks


@pytest.mark.parametrize(('name', 'axis', 'cutoff', 'shifted', 'case_id'), [
    ('S1–TOC', 0, 4, 4.2, 'x-boundary-4'),
    ('S2–TOC', 1, 20, 21, 'y-boundary-4'),
    ('HI–TOC', 1, 600, 610, 'y-boundary-4'),
    ('PI–Tmax', 0, 470, 471, 'x-boundary-2'),
    ('PI–Tmax', 1, .4, .41, 'y-boundary-2'),
    ('S1+S2–TOC', 0, 2, 2.1, 'x-boundary-3'),
])
def test_bundled_cutoff_cases_detect_shift_beyond_first_cell(name, axis, cutoff, shifted, case_id):
    from autopyro.schemes.standard_interpretation import standard_schemes

    scheme = standard_schemes()[name]
    for region in scheme['regions']:
        ring, = region['geometry']['coordinates']
        shifted_ring = []
        for vertex in ring:
            coordinates = list(vertex)
            if coordinates[axis] == cutoff:
                coordinates[axis] = shifted
            shifted_ring.append(coordinates)
        region['geometry']['coordinates'] = [shifted_ring]
    failed_checks = {(case, field) for case, field, _, _ in reference_failures(scheme)}
    assert (case_id, 'boundary_regions') in failed_checks


@pytest.mark.parametrize(('author', 'property_name'), [
    ('Edress', 'GP'),
    ('Maky', 'GP'),
    ('Spasennykh', 'S0+S1+S2'),
])
def test_svg_potential_cases_detect_removed_polygon(author, property_name):
    from autopyro.schemes.interpretation_review import review_scheme_hash
    from autopyro.schemes.standard_interpretation import standard_schemes

    scheme = next(s for name, s in standard_schemes().items()
                  if name.startswith(f'S1+S2–TOC · {author}'))
    assert scheme['revision'] == '2'
    assert scheme['y']['property'] == property_name
    assert len(scheme['reference_cases']) == 8
    assert reference_failures(scheme) == []
    previous = {key: value for key, value in scheme.items() if key != 'reference_cases'}
    previous['revision'] = '1'
    assert review_scheme_hash(previous) == review_scheme_hash(scheme)

    poor_id, = scheme['reference_cases'][0]['expected']['regions']
    scheme['regions'] = [region for region in scheme['regions'] if region['id'] != poor_id]
    failed_checks = {(case, field) for case, field, _, _ in reference_failures(scheme)}
    assert ('poor', 'regions') in failed_checks
    assert ('interval-crosses-poor-fair', 'uncertainty_regions') in failed_checks


@pytest.mark.parametrize(('author', 'case_count'), [
    ('Chen', 7), ('Edress', 27), ('T.A. Ryazanova', 7), ('Zeb', 7),
])
def test_svg_s2_cases_cover_all_regions_and_detect_removed_polygon(author, case_count):
    from autopyro.schemes.interpretation_review import review_scheme_hash
    from autopyro.schemes.standard_interpretation import standard_schemes

    scheme = next(s for name, s in standard_schemes().items()
                  if name.startswith(f'S2–TOC · {author}'))
    assert scheme['revision'] == '2'
    cases = scheme['reference_cases']
    assert len(cases) == case_count
    assert {case['id'] for case in cases if case['id'].startswith('inside-')} == {
        f"inside-{region['id']}" for region in scheme['regions']
    }
    assert reference_failures(scheme) == []
    previous = {key: value for key, value in scheme.items() if key != 'reference_cases'}
    previous['revision'] = '1'
    assert review_scheme_hash(previous) == review_scheme_hash(scheme)

    interval = next(case for case in cases if case['id'] == 'interval-crosses-boundary')
    removed = interval['expected']['uncertainty_regions'][0]
    scheme['regions'] = [region for region in scheme['regions'] if region['id'] != removed]
    failed = {(case, field) for case, field, _, _ in reference_failures(scheme)}
    assert (f'inside-{removed}', 'regions') in failed
    assert ('interval-crosses-boundary', 'uncertainty_regions') in failed


@pytest.mark.parametrize(('author', 'case_count'), [
    ('Akram', 8), ('Edress', 9), ('Khatibi', 7),
])
def test_svg_pi_cases_cover_all_regions_and_detect_removed_geometry(author, case_count):
    from autopyro.schemes.interpretation_review import review_scheme_hash
    from autopyro.schemes.standard_interpretation import standard_schemes

    scheme = next(s for name, s in standard_schemes().items()
                  if name.startswith(f'PI–Tmax · {author}'))
    assert scheme['revision'] == '2'
    cases = scheme['reference_cases']
    assert len(cases) == case_count
    assert {case['id'] for case in cases if case['id'].startswith('inside-')} == {
        f"inside-{region['id']}" for region in scheme['regions']
    }
    assert reference_failures(scheme) == []
    previous = {key: value for key, value in scheme.items() if key != 'reference_cases'}
    previous['revision'] = '1'
    assert review_scheme_hash(previous) == review_scheme_hash(scheme)

    if author == 'Edress':
        scheme['lines'] = []
        assert reference_failures(scheme) == [('conversion-line', 'status', 'boundary', 'classified')]

    interval = next(case for case in cases if case['id'] == 'interval-crosses-boundary')
    removed = interval['expected']['uncertainty_regions'][0]
    scheme['regions'] = [region for region in scheme['regions'] if region['id'] != removed]
    failed = {(case, field) for case, field, _, _ in reference_failures(scheme)}
    assert (f'inside-{removed}', 'regions') in failed
    assert ('interval-crosses-boundary', 'uncertainty_regions') in failed


def test_svg_hi_oi_cases_cover_every_curve_and_detect_removed_type():
    from autopyro.schemes.interpretation_review import review_scheme_hash
    from autopyro.schemes.standard_interpretation import standard_schemes

    schemes = [scheme for scheme in standard_schemes().values() if scheme['id'].startswith('svg:HI-OI:')]
    assert len(schemes) == 6
    for scheme in schemes:
        assert scheme['revision'] == '3'
        cases = scheme['reference_cases']
        assert {case['id'] for case in cases if case['id'].startswith('near-')} == {
            f"near-{line['id']}" for line in scheme['lines'] if not line.get('reference_only')
        }
        assert reference_failures(scheme) == [], scheme['name']
        previous = {key: value for key, value in scheme.items() if key != 'reference_cases'}
        previous['revision'] = '2'
        assert review_scheme_hash(previous) == review_scheme_hash(scheme)

        interval = next(case for case in cases if case['id'] == 'interval-spans-types-i-ii')
        removed = interval['expected']['possible_types'][-1]
        scheme['lines'] = [line for line in scheme['lines'] if line['id'] != removed]
        failed = {(case, field) for case, field, _, _ in reference_failures(scheme)}
        assert (f'near-{removed}', 'possible_types') in failed
        assert ('interval-spans-types-i-ii', 'possible_types') in failed


def test_svg_hi_tmax_intervals_cover_each_classification_curve():
    from autopyro.schemes.interpretation_review import review_scheme_hash
    from autopyro.schemes.standard_interpretation import standard_schemes

    schemes = [s for s in standard_schemes().values()
               if s['id'].startswith('svg:HI-Tmax:') and s.get('strategy') == 'nearest_curve']
    assert len(schemes) == 12
    for scheme in schemes:
        assert scheme['revision'] == '3'
        cases = scheme['reference_cases']
        assert {case['id'] for case in cases if 'x_bounds' in case} == {
            f"near-{line['id']}" for line in scheme['lines'] if not line.get('reference_only')
        }
        assert reference_failures(scheme) == [], scheme['name']
        previous = {key: value for key, value in scheme.items() if key != 'reference_cases'}
        previous['revision'] = '2'
        assert review_scheme_hash(previous) == review_scheme_hash(scheme)

        removed, = cases[0]['expected']['possible_types']
        scheme['lines'] = [line for line in scheme['lines'] if line['id'] != removed]
        failed = {(case, field) for case, field, _, _ in reference_failures(scheme)}
        assert (f'near-{removed}', 'possible_types') in failed, scheme['name']


def test_every_bundled_scheme_has_passing_reference_cases():
    from autopyro.schemes.standard_interpretation import standard_schemes

    for scheme in standard_schemes().values():
        assert scheme.get('reference_cases'), scheme['id']
        assert reference_failures(scheme) == [], scheme['id']


@pytest.mark.parametrize(('scheme_id', 'geometry', 'removed', 'case_id', 'field'), [
    ('svg:HI-Tmax:7db83cea51ca', 'regions', 'path9:path11', 'maturity-interval', 'uncertainty_regions'),
    ('svg:HI-TOC:0361ff38ba45', 'regions', 'path41', 'source-quality-interval', 'uncertainty_regions'),
    ('svg:TOCo:4c82336c6d05', 'lines', 'path537', 'on-path537', 'line_hits'),
])
def test_last_svg_cases_detect_removed_geometry(scheme_id, geometry, removed, case_id, field):
    from autopyro.schemes.standard_interpretation import standard_schemes

    scheme = next(s for s in standard_schemes().values() if s['id'] == scheme_id)
    scheme[geometry] = [entry for entry in scheme[geometry] if entry['id'] != removed]
    failures = {(case, key) for case, key, _, _ in reference_failures(scheme)}
    assert (case_id, field) in failures


def test_hio_and_possible_types_are_checked():
    scheme = curves('initial_hi')
    scheme['reference_cases'] = [{
        'id': 'hio', 'x': 5, 'y': 75,
        'expected': {'status': 'classified', 'possible_types': ['1'], 'value_bounds': [150, 150]},
    }]
    assert reference_failures(scheme) == []
    scheme['reference_cases'][0]['expected'].update(possible_types=['missing'], value_bounds=[140, 150])
    assert {field for _, field, _, _ in reference_failures(scheme)} == {'possible_types', 'value_bounds'}


@pytest.mark.parametrize(('name', 'interval_case'), [
    ('HI–Tmax', 'crossing-interval'),
    ('HI–OI', 'interval-spans-types-i-ii'),
])
def test_bundled_type_cases_detect_removed_curve(name, interval_case):
    from autopyro.schemes.interpretation_review import review_scheme_hash
    from autopyro.schemes.standard_interpretation import standard_schemes

    scheme = standard_schemes()[name]
    assert scheme['revision'] == '4'
    assert reference_failures(scheme) == []
    previous = {key: value for key, value in scheme.items() if key != 'reference_cases'}
    previous['revision'] = '3'
    assert review_scheme_hash(previous) == review_scheme_hash(scheme)

    scheme['lines'] = [line for line in scheme['lines'] if line['id'] != 'I']
    failed_checks = {(case, field) for case, field, _, _ in reference_failures(scheme)}
    assert ('type-i', 'possible_types') in failed_checks
    assert (interval_case, 'possible_types') in failed_checks


def test_bundled_hio_cases_detect_changed_start_of_type_i_curve():
    from autopyro.schemes.interpretation_review import review_scheme_hash
    from autopyro.schemes.standard_interpretation import standard_schemes

    scheme = standard_schemes()['HIo · HI–Tmax']
    assert scheme['revision'] == '3'
    assert reference_failures(scheme) == []
    previous = {key: value for key, value in scheme.items() if key != 'reference_cases'}
    previous['revision'] = '2'
    assert review_scheme_hash(previous) == review_scheme_hash(scheme)

    type_i = next(line for line in scheme['lines'] if line['id'] == 'I')
    type_i['geometry']['coordinates'][0][1] += 0.001
    failures = {(case, field) for case, field, _, _ in reference_failures(scheme)}
    assert ('hio-type-i-anchor', 'value_bounds') in failures
    assert ('hio-crossing-interval', 'value_bounds') in failures


def test_reference_bounds_must_contain_the_sample(scheme):
    scheme['reference_cases'][0]['x_bounds'] = [1, 1.5]
    with pytest.raises(ValueError, match='interval-leaves-one-region'):
        validate_scheme(scheme)


def test_revision_keeps_individual_threshold_checks_but_can_change_expectations(scheme):
    from copy import deepcopy

    previous = scheme
    previous['reference_cases'][1]['expected']['thresholds']['other'] = 'boundary'
    draft = deepcopy(previous)
    checks = draft['reference_cases'][1]['expected']['thresholds']
    checks['cutoff'] = 'less'
    del checks['other']
    assert missing_reference_checks(previous, draft) == [('above-threshold', 'thresholds.other')]
    checks['other'] = 'greater'
    assert missing_reference_checks(previous, draft) == []


def test_revision_preserves_threshold_interval_assertion(scheme):
    from copy import deepcopy
    from autopyro.ui.dialogs.interpretation import InterpretationDialog

    scheme['reference_cases'][1]['expected']['thresholds']['cutoff'] = {
        'relation': 'greater', 'possible_relations': ['greater'],
    }
    draft = deepcopy(scheme)
    draft['reference_cases'][1]['expected']['thresholds']['cutoff'] = 'greater'
    validate_scheme(draft)
    assert reference_failures(draft) == []
    assert missing_reference_checks(scheme, draft) == [
        ('above-threshold', 'thresholds.cutoff.possible_relations'),
    ]
    with pytest.raises(ValueError, match='thresholds.cutoff.possible_relations'):
        InterpretationDialog.require_reference_cases(draft, scheme)

    draft['reference_cases'][1]['expected']['thresholds']['cutoff'] = {
        'relation': 'less', 'possible_relations': ['less', 'boundary'],
    }
    assert missing_reference_checks(scheme, draft) == []


def test_failed_draft_does_not_replace_stored_revision(scheme, qtbot, monkeypatch):
    from copy import deepcopy
    from PySide6.QtWidgets import QMessageBox
    from autopyro.ui.window import MainWindow
    from test_interpretation_comparison import data

    project, _, _ = data()
    scheme['reference_cases'][1]['expected']['thresholds']['cutoff'] = {
        'relation': 'greater', 'possible_relations': ['greater'],
    }
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.may_replace = lambda: True
    window.show_interpretation()
    dialog = window.interpretation_window
    dialog.commit_graphical_scheme(scheme, auto_bind=True)
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warnings.append(args[-1]))
    draft = deepcopy(scheme)
    draft['revision'] = '2'
    draft['lines'][0]['geometry']['coordinates'] = [[2.5, 300], [4, 300]]
    dialog.editor.setPlainText(json.dumps(draft))
    dialog.apply_editor()
    assert 'above-threshold' in warnings[0]
    assert project.view['interpretation_schemes'][scheme['id']]['revision'] == '1'
    assert '2' not in project.view['interpretation_scheme_revisions'][scheme['id']]
    draft.pop('reference_cases')
    dialog.editor.setPlainText(json.dumps(draft))
    dialog.apply_editor()
    assert 'reference_cases' in warnings[-1]

    for cases in ([], scheme['reference_cases'][:1]):
        draft['reference_cases'] = cases
        dialog.editor.setPlainText(json.dumps(draft))
        warning_count = len(warnings)
        dialog.check_reference_cases()
        assert len(warnings) == warning_count + 1
        assert 'above-threshold' in warnings[-1]
        dialog.apply_editor()
        assert len(warnings) == warning_count + 2
        assert 'above-threshold' in warnings[-1]
        assert project.view['interpretation_schemes'][scheme['id']]['revision'] == '1'
        assert '2' not in project.view['interpretation_scheme_revisions'][scheme['id']]

    draft['reference_cases'] = deepcopy(scheme['reference_cases'])
    del draft['reference_cases'][1]['expected']['thresholds']
    dialog.editor.setPlainText(json.dumps(draft))
    warning_count = len(warnings)
    dialog.check_reference_cases()
    dialog.apply_editor()
    assert len(warnings) == warning_count + 2
    assert all('above-threshold · thresholds' in warning for warning in warnings[-2:])
    assert project.view['interpretation_schemes'][scheme['id']]['revision'] == '1'
    assert '2' not in project.view['interpretation_scheme_revisions'][scheme['id']]
    draft['reference_cases'] = deepcopy(scheme['reference_cases'])
    draft['reference_cases'][1]['expected']['thresholds']['cutoff'] = 'greater'
    dialog.editor.setPlainText(json.dumps(draft))
    warning_count = len(warnings)
    dialog.check_reference_cases()
    dialog.apply_editor()
    assert len(warnings) == warning_count + 2
    assert all('thresholds.cutoff.possible_relations' in warning for warning in warnings[-2:])
    assert project.view['interpretation_schemes'][scheme['id']]['revision'] == '1'
    dialog.close()
    window.dirty = False
