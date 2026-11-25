"""User screening rules are explicit, unit-aware and replayable."""
from pathlib import Path
import json

import pytest

from autopyro.io import build_project
from autopyro.interpretation import evaluate_project, replay_report
from autopyro.interpretation_quality import InterpretationQuality, RANGES, CANONICAL_UNITS
from autopyro.interpretation_explanation import explain_sample


def test_axis_binding_rejects_same_unit_with_wrong_role_and_accepts_metadata():
    from autopyro.schemes.interpretation import validate_axis_binding
    from autopyro.schemes.standard_interpretation import standard_schemes

    project = build_project(['TOC', 'S1', 'Лабораторный результат'], [['1.5', '12', '12']], [1],
                            [(0, 'TOC', 'number', '%wt'), (1, 'S1', 'number', 'mg HC/g rock'),
                             (2, 'Лабораторный результат', 'number', 'mg HC/g rock')], {})
    x, wrong, right = project.columns
    right.metadata['property_role'] = 'S-2'
    scheme = standard_schemes()['S2–TOC']
    with pytest.raises(ValueError, match='нужен показатель S2'):
        evaluate_project(project, project.row_map(), x.id, wrong.id, '%wt', wrong.unit, scheme)
    report = evaluate_project(project, project.row_map(), x.id, right.id, '%wt', right.unit, scheme)
    assert report['results'][0]['status'] == 'classified'
    with pytest.raises(ValueError, match='S1/S2'):
        validate_axis_binding(right, right.unit, {'property': 'S1/S2', 'unit': right.unit})


def test_all_bundled_axes_require_matching_declared_and_source_units():
    from types import SimpleNamespace

    from autopyro.schemes.interpretation import validate_axis_binding
    from autopyro.schemes.standard_interpretation import standard_schemes

    for scheme in standard_schemes().values():
        for axis in (scheme['x'], scheme['y']):
            unit = axis['unit']
            column = SimpleNamespace(name=axis['property'], unit=unit, metadata={})
            validate_axis_binding(column, unit, axis)
            incompatible = '%wt' if unit != '%wt' else '°C'
            with pytest.raises(ValueError, match='не соответствует схеме'):
                validate_axis_binding(column, incompatible, axis)
            column.unit = incompatible
            with pytest.raises(ValueError, match='противоречит декларации'):
                validate_axis_binding(column, unit, axis)
            column.unit = ''
            with pytest.raises(ValueError, match='Укажите единицу'):
                validate_axis_binding(column, '', axis)
            validate_axis_binding(column, unit, axis)


@pytest.mark.parametrize('role', list(RANGES))
def test_absolute_and_typical_boundaries(role):
    low, typical_low, typical_high, high = RANGES[role]
    values = [low - .01, low, typical_low, typical_high, high, high + .01]
    project = build_project([role, 'Y'], [[str(v), '1'] for v in values], list(range(6)),
                            [(0, role, 'number', CANONICAL_UNITS[role]), (1,'Y','number','1')], {})
    x,y = [c.id for c in project.columns]
    checker = InterpretationQuality(project,x,y,CANONICAL_UNITS[role],'1',
                                    {'x':{'property':role},'y':{'property':'Y'}},False)
    for i, row in enumerate(project.rows):
        issues, blocked = checker.check(row)
        assert bool(blocked) == (i in (0,5))
        assert [q['code'] for q in issues] == (['absolute_range'] if i in (0,5)
                                              else ['typical_range'] if i in (1,4) else [])


def test_alias_units_and_unchecked_unit():
    project = build_project(['SI','T-Max'], [['3000','450']], [1],
                            [(0,'SI','number','ug HC/g rock'), (1,'T-Max','number','°C')], {})
    x,y = [c.id for c in project.columns]
    checker = InterpretationQuality(project,x,y,'ug HC/g rock','°C',
                                    {'x':{'property':'SI'},'y':{'property':'T-Max'}},False)
    issues, blocked = checker.check(project.rows[0])
    assert not blocked
    assert issues[0]['code'] == 'typical_range'
    assert '3 mg HC/g rock' in issues[0]['message']


def test_qc_result_replays_and_explanation_preserves_source_values():
    scheme = json.loads((Path(__file__).parents[1]/'examples/interpretation-demo.json').read_text(encoding='utf-8'))
    project = build_project(['TOC','HI'], [['-1','200'],['.05','200'],['1','200']], [1,2,3],
                            [(0,'TOC','number','%wt'),(1,'HI','number','mg HC/g TOC')], {})
    x,y = [c.id for c in project.columns]
    original = project.rows[0].values[x]
    report = evaluate_project(project, project.row_map(),x,y,'%wt','mg HC/g TOC',scheme)
    assert report['results'][0]['status'] == 'not_applicable'
    assert report['inputs'][0]['statuses'][0] == 'range_excluded'
    assert report['inputs'][1]['quality'][0]['code'] == 'typical_range'
    assert report['results'][1]['status'] == 'classified'
    assert project.rows[0].values[x] == original
    assert replay_report(json.loads(json.dumps(report))) == report['results']
    text = explain_sample(report,0)
    assert 'вне абсолютного диапазона' in text
    assert scheme['source'] in text and scheme['revision'] in text
    # Keep an explicit reason for geometrically ambiguous results.
    report['results'][2].update(status='ambiguous', reason='Перекрытие двух областей', regions=['A','B'])
    assert 'Причина неоднозначности: Перекрытие двух областей' in explain_sample(report,2)


def test_ancillary_errors_do_not_block_axes_and_unknown_units_are_reported():
    scheme = {'x': {'property':'TOC'}, 'y': {'property':'HI'}}
    project = build_project(['TOC','HI','S1','OI'], [['1','200','101','900']], [1],
                            [(0,'TOC','number','%wt'), (1,'HI','number','mg HC/g TOC'),
                             (2,'S1','number','mg HC/g rock'), (3,'OI','number','unknown')], {})
    x,y = [c.id for c in project.columns[:2]]
    issues, blocked = InterpretationQuality(project,x,y,'%wt','mg HC/g TOC',scheme,False).check(project.rows[0])
    assert not blocked
    assert {q['code'] for q in issues} == {'absolute_range','unit_unchecked'}


def test_explicit_qc_exclusion_precedes_numeric_range_check():
    project = build_project(['TOC','HI'], [['101','200']], [1],
                            [(0,'TOC','number','%wt'),(1,'HI','number','mg HC/g TOC')], {})
    row = project.rows[0]
    x,y = [c.id for c in project.columns]
    project.sync_measurements()
    measurement = project.science.current(row.id,x)
    project.view['qc_exclusions'] = {'interpretation': {measurement.id: {'revision': measurement.revision}}}
    issues, blocked = InterpretationQuality(project,x,y,'%wt','mg HC/g TOC',
                            {'x':{'property':'TOC'},'y':{'property':'HI'}},False).check(row)
    assert not blocked
    assert issues[0]['code'] == 'qc_excluded'


def test_qc_reports_missing_and_ambiguous_ancillary_columns_in_both_paths():
    from autopyro.standard_interpretation import standard_schemes

    headers = ['Tmax', 'HI', 'TOC v1', 'TOC v2']
    units = ['°C', 'mg HC/g TOC', '%wt', '%wt']
    project = build_project(headers, [[430, 400, 1, 2]] * 256, range(256),
                            [(i, name, 'number', units[i]) for i, name in enumerate(headers)], {})
    x, y, *toc = project.columns
    for column in toc:
        column.metadata['property_role'] = 'TOC'
    scheme = standard_schemes()['HI–Tmax']
    qc = InterpretationQuality(project, x.id, y.id, x.unit, y.unit, scheme, False)
    direct, blocked = qc.check(project.rows[0])
    assert not blocked
    assert {issue['property']: issue['code'] for issue in direct} == {
        'S2': 'coverage_unchecked', 'TOC': 'coverage_unchecked',
    }
    assert 'несколько' in direct[1]['message']
    report = evaluate_project(project, project.row_map(), x.id, y.id, x.unit, y.unit, scheme)
    assert report['inputs'][0]['quality'] == direct
    assert report['results'][0]['status'] == 'classified'
    explanation = explain_sample(report, 0)
    assert 'Диапазоны проверены: Tmax, HI' in explanation
    assert 'S2: диапазон не проверен' in explanation
