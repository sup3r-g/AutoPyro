"""Scheme-owned regression cases evaluated in the original axis units."""

import math

from .interpretation import GeometricEngine, InterpretationInput, STATUSES


HIO_TOLERANCE = 1e-6
ID_FIELDS = {'possible_types', 'regions', 'boundary_regions', 'uncertainty_regions', 'line_hits'}
EXPECTED_FIELDS = ID_FIELDS | {'status', 'value_bounds', 'thresholds'}
THRESHOLD_RELATIONS = ('less', 'boundary', 'greater', 'outside_domain')


def _number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _bounds(value):
    return (isinstance(value, list) and len(value) == 2
            and all(_number(item) for item in value) and value[0] <= value[1])


def _ids(value):
    return isinstance(value, list) and all(isinstance(item, str) and item for item in value)


def _threshold_expectation(value):
    if not isinstance(value, dict):
        return value in THRESHOLD_RELATIONS
    if value.keys() != {'relation', 'possible_relations'} or value['relation'] not in THRESHOLD_RELATIONS:
        return False
    possible = value['possible_relations']
    return (bool(possible) and _ids(possible) and len(set(possible)) == len(possible)
            and all(relation in THRESHOLD_RELATIONS for relation in possible))


def validate_reference_cases(cases):
    if not isinstance(cases, list):
        raise ValueError('reference_cases должен быть списком')
    ids = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get('id'), str) or not case['id'].strip():
            raise ValueError('У каждого эталона должен быть непустой id')
        case_id = case['id']
        if case_id in ids:
            raise ValueError(f'Повторяется id эталона {case_id}')
        ids.add(case_id)
        for axis in ('x', 'y'):
            value = case.get(axis)
            bounds = case.get(f'{axis}_bounds')
            if not _number(value):
                raise ValueError(f'Эталон {case_id}: координата {axis} должна быть конечным числом')
            if bounds is not None and (not _bounds(bounds) or not bounds[0] <= value <= bounds[1]):
                raise ValueError(f'Эталон {case_id}: интервал {axis} должен содержать координату')
        expected = case.get('expected')
        if not isinstance(expected, dict) or not expected or expected.keys() - EXPECTED_FIELDS:
            raise ValueError(f'Эталон {case_id}: укажите поддерживаемые поля expected')
        if 'status' in expected and (not isinstance(expected['status'], str) or expected['status'] not in STATUSES):
            raise ValueError(f'Эталон {case_id}: неизвестный ожидаемый статус')
        for key in sorted(ID_FIELDS):
            if key in expected and not _ids(expected[key]):
                raise ValueError(f'Эталон {case_id}: {key} должен быть списком идентификаторов')
        if expected.get('value_bounds') is not None and not _bounds(expected['value_bounds']):
            raise ValueError(f'Эталон {case_id}: value_bounds должен быть диапазоном HIo или null')
        if 'thresholds' in expected:
            thresholds = expected['thresholds']
            if not isinstance(thresholds, dict) or not thresholds or any(
                not _threshold_expectation(relation) for relation in thresholds.values()
            ):
                raise ValueError(f'Эталон {case_id}: thresholds должен сопоставлять id порога и ожидаемую сторону')


def missing_reference_checks(previous, scheme):
    """List dropped cases and assertions, while allowing expectations to evolve."""
    current = {case['id']: case['expected'] for case in scheme.get('reference_cases', ())}
    missing = []
    for case in previous.get('reference_cases', ()):
        case_id = case['id']
        if case_id not in current:
            missing.append((case_id, None))
            continue
        old, new = case['expected'], current[case_id]
        missing.extend((case_id, key) for key in sorted(old.keys() - new.keys()))
        if 'thresholds' in old and 'thresholds' in new:
            missing.extend((case_id, f'thresholds.{key}')
                           for key in sorted(old['thresholds'].keys() - new['thresholds'].keys()))
            for key in sorted(old['thresholds'].keys() & new['thresholds'].keys()):
                if isinstance(old['thresholds'][key], dict) and not isinstance(new['thresholds'][key], dict):
                    missing.append((case_id, f'thresholds.{key}.possible_relations'))
    return missing


def reference_failures(scheme):
    """Return (case ID, field, expected, actual) for each failed assertion."""
    cases = scheme.get('reference_cases', [])
    if not cases:
        return []
    samples = [InterpretationInput(case['id'], case['x'], case['y'],
                                   case.get('x_bounds'), case.get('y_bounds')) for case in cases]
    results = GeometricEngine().evaluate(samples, scheme)
    failures = []
    for case, result in zip(cases, results, strict=True):
        for field, expected in case['expected'].items():
            actual = getattr(result, field)
            if field == 'thresholds':
                for key, rule in expected.items():
                    observed = actual.get(key, {})
                    path = f'thresholds.{key}'
                    relation = rule['relation'] if isinstance(rule, dict) else rule
                    if observed.get('relation') != relation:
                        failures.append((case['id'], path, relation, observed.get('relation')))
                    if isinstance(rule, dict):
                        possible = rule['possible_relations']
                        actual_possible = observed.get('possible_relations')
                        if set(actual_possible or ()) != set(possible):
                            failures.append((case['id'], f'{path}.possible_relations', possible, actual_possible))
                continue
            if field in ID_FIELDS:
                matches = set(actual) == set(expected)
            elif field == 'value_bounds' and actual is not None and expected is not None:
                matches = all(math.isclose(a, b, rel_tol=0, abs_tol=HIO_TOLERANCE)
                              for a, b in zip(actual, expected, strict=True))
            else:
                matches = actual == expected
            if not matches:
                failures.append((case['id'], field, expected, actual))
    return failures
