"""Analytic and adversarial checks for complete-interval curve interpretation."""
from copy import deepcopy
import json

import numpy as np
import pytest
import shapely
from shapely.geometry import Point

from autopyro.interpretation import GeometricEngine, InterpretationInput, evaluate_project, replay_report
from autopyro.io import build_project
from test_standard_interpretation import curves


def parallel():
    scheme=curves()
    scheme['lines']=[dict(id=str(i),label='Тип '+str(i),geometry=dict(type='LineString',
                     coordinates=[[-20,y],[20,y]])) for i,y in enumerate((0,10,20))]
    return scheme


def result(s,x,y,**kwargs):
    return GeometricEngine().evaluate([InterpretationInput('row',x,y,**kwargs)],s)[0]


def test_parallel_curves_unique_and_two_possible_types():
    s=parallel()
    stable=result(s,0,4,x_bounds=(-2,2),y_bounds=(3,4.5))
    assert stable.status=='classified' and stable.possible_types==['0']
    ambiguous=result(s,0,5,y_bounds=(4,6))
    assert ambiguous.status=='ambiguous'
    assert ambiguous.possible_types==['0','1']
    assert not ambiguous.interval_details['conservative']


def test_interior_winner_is_not_lost_by_corner_sampling():
    s=parallel()
    s['lines']=[dict(id='A',label='A',geometry=dict(type='MultiLineString',
        coordinates=[[(0,0),(10,10)],[(0,10),(10,0)]])),
        dict(id='B',label='B',geometry=dict(type='LineString',coordinates=[(2,3),(2.1,3)]))]
    gs=[shapely.geometry.shape(e['geometry']) for e in s['lines']]
    for p in [(0,0),(0,10),(10,0),(10,10),(5,5)]:
        assert Point(p).distance(gs[0]) < Point(p).distance(gs[1])
    r=result(s,5,5,x_bounds=(0,10),y_bounds=(0,10))
    assert r.status=='ambiguous' and r.possible_types==['A','B']


def test_bounded_search_stays_conservative_and_validates_bounds():
    s=parallel();s['interval_max_evaluations']=5
    # Root bounds can include extra candidates; they must never be called certain.
    r=result(s,0,1,x_bounds=(-19,19),y_bounds=(.9,1.1))
    assert r.status=='ambiguous'
    assert r.interval_details['conservative'] and r.interval_details['budget_exhausted']
    for bounds in [(2,1),(-1,float('inf')),(2,3)]:
        assert result(s,0,1,x_bounds=bounds).status=='not_applicable'
    s['x']['transform']='log10'
    assert result(s,1,1,x_bounds=(0,2)).status=='not_applicable'


def test_hio_range_encloses_dense_grid_of_nominal_results():
    s=curves('initial_hi');s['intermediate_curves']=19
    r=result(s,5,75,x_bounds=(4.9,5.1),y_bounds=(70,80))
    assert r.status=='ambiguous' and r.value is None
    assert r.value_bounds[0] < 150 < r.value_bounds[1]
    samples=[InterpretationInput(str(i),x,y) for i,(x,y) in enumerate(
        ( (x,y) for x in np.linspace(4.9,5.1,9) for y in np.linspace(70,80,9)))]
    for nominal in GeometricEngine().evaluate(samples,s):
        assert r.value_bounds[0] <= nominal.value_bounds[0]
        assert nominal.value_bounds[1] <= r.value_bounds[1]


def test_hio_requires_unique_type_for_point_and_zero_width_interval():
    scheme = curves('initial_hi')
    scheme['hi_resolution'] = 10
    scheme['lines'][1]['geometry']['coordinates'][0] = (0, 108)
    for options in ({}, {'x_bounds': (10, 10)}):
        item = result(scheme, 10, 0, **options)
        assert item.status == 'ambiguous'
        assert item.possible_types == ['0', '1']
        assert item.value is None
        assert item.value_bounds == [100, 108]

    scheme['x']['property'], scheme['y']['property'] = 'X', 'Y'
    project = build_project(['X', 'Y'], [[10, 0]], [1],
                            [(0, 'X', 'number', '°C'), (1, 'Y', 'number', 'mg HC/g TOC')], {})
    x, y = [column.id for column in project.columns]
    snapshot = evaluate_project(project, project.row_map(), x, y, '°C', 'mg HC/g TOC', scheme)
    assert replay_report(snapshot) == snapshot['results']
    del snapshot['interval_policy']['hio_requires_unique_type']
    legacy, = replay_report(snapshot)
    assert legacy['status'] == 'classified' and legacy['value'] == 100

def test_domain_checks_entire_interval_and_replay_is_frozen():
    s=parallel();s['max_distance']=1
    assert result(s,0,1,y_bounds=(.5,2)).status=='not_applicable'
    s=curves('initial_hi');s['no_extrapolation']=True
    assert result(s,5,75,y_bounds=(25,100)).status=='not_applicable'
    s=parallel()
    p=build_project(['X','Y'],[['0','5']],[1],[(0,'X','number','°C'),(1,'Y','number','mg HC/g TOC')],{})
    # Use synthetic property names so screening does not reject the test coordinates.
    s['x']['property']='X';s['y']['property']='Y'
    p.sync_measurements();x,y=[c.id for c in p.columns]
    m=p.science.current(p.rows[0].id,y);m.uncertainty=1;m.uncertainty_type='absolute'
    report=evaluate_project(p,p.row_map(),x,y,'°C','mg HC/g TOC',s)
    assert report['results'][0]['possible_types']==['0','1']
    snapshot=json.loads(json.dumps(report))
    assert replay_report(snapshot)==report['results']
    assert snapshot['interval_policy']['method']=='adaptive_lipschitz_v1'


def test_replay_rejects_interval_policy_conflicting_with_saved_scheme():
    scheme = parallel()
    scheme['x']['property'] = 'X'
    scheme['y']['property'] = 'Y'
    scheme['max_distance'] = 6
    project = build_project(['X', 'Y'], [[0, 5]], [1],
                            [(0, 'X', 'number', '°C'), (1, 'Y', 'number', 'mg HC/g TOC')], {})
    x, y = [column.id for column in project.columns]
    report = evaluate_project(project, project.row_map(), x, y, '°C', 'mg HC/g TOC', scheme)
    assert replay_report(report) == report['results']
    assert replay_report({key: value for key, value in report.items() if key != 'interval_policy'}) == report['results']

    report['interval_policy']['parameters']['max_distance'] = 1
    with pytest.raises(ValueError, match='Параметры интервального воспроизведения противоречат схеме'):
        replay_report(report)


def test_renderer_reuses_interval_results_from_snapshot(monkeypatch):
    from autopyro.plot_data import capture_input,input_token,PlotDataCache
    from autopyro.plotting import PlotSpec,render_plot
    s=parallel();s['x']['property']='X';s['y']['property']='Y'
    p=build_project(['X','Y'],[['0','5']],[1],[(0,'X','number','°C'),(1,'Y','number','mg HC/g TOC')],{})
    p.sync_measurements();x,y=[c.id for c in p.columns]
    m=p.science.current(p.rows[0].id,y);m.uncertainty=1;m.uncertainty_type='absolute'
    report=evaluate_project(p,p.row_map(),x,y,'°C','mg HC/g TOC',s)
    p.view['interpretation_result']=report
    spec=PlotSpec(x_id=x,y_id=y,interpretation={'scheme':s,**report['binding']})
    source=capture_input(p,p.row_map(),[spec],input_token(p,0,[spec]))
    prepared=PlotDataCache().prepare(source,p.row_map(),spec)
    def forbidden(*args,**kwargs):
        raise AssertionError('Rendering must reuse the frozen interval interpretation')
    monkeypatch.setattr(GeometricEngine,'evaluate',forbidden)
    _,r=render_plot(p,p.row_map(),spec,prepared=prepared)
    assert r['uncertainty_crossings'][p.rows[0].id]==['0','1']
