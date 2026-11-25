"""Declared uncertainty is replayable, visible and mapped to polygon ambiguity."""
import json
from pathlib import Path
from io import BytesIO

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection, LineCollection

from autopyro.io import build_project
from autopyro.interpretation import evaluate_project, replay_report, check_saved_result
from autopyro.measurement_uncertainty import measurement_interval
from autopyro.plot_data import capture_input, input_token, PlotDataCache
from autopyro.plotting import PlotSpec, render_plot
from autopyro.selection_overlay import SelectionOverlay


def data(*, y_uncertainty=True):
    project=build_project(['TOC','HI'],[['1.9','200'],['1','100']],[1,2],
                          [(0,'TOC','number','%wt'),(1,'HI','number','mg HC/g TOC')],{})
    project.sync_measurements()
    x,y=[c.id for c in project.columns]
    m=project.science.current(project.rows[0].id,x)
    m.uncertainty=.2;m.uncertainty_type='absolute';m.confidence_level=.95
    if y_uncertainty:
        m=project.science.current(project.rows[0].id,y)
        m.uncertainty=10;m.uncertainty_type='standard'
    scheme=json.loads((Path(__file__).parents[1]/'examples/interpretation-demo.json').read_text(encoding='utf-8'))
    binding=dict(x_id=x,y_id=y,x_unit='%wt',y_unit='mg HC/g TOC',scheme=scheme,text_numbers=False)
    report=evaluate_project(project,project.row_map(),x,y,'%wt','mg HC/g TOC',scheme)
    return project,binding,report


@pytest.mark.parametrize('y_uncertainty',[False,True])
def test_intervals_classify_crossing_and_replay_exactly(y_uncertainty):
    project,binding,report=data(y_uncertainty=y_uncertainty)
    assert report['inputs'][0]['x_bounds'] == pytest.approx((1.7,2.1))
    assert report['results'][0]['status']=='ambiguous'
    assert report['results'][0]['uncertainty_regions']==['A','B']
    assert report['results'][1]['status']=='classified'
    frozen=json.loads(json.dumps(report))
    assert replay_report(frozen)==report['results']
    project.view['interpretation_result']=report
    m=project.science.current(project.rows[0].id,binding['x_id'])
    m.uncertainty=.3
    check_saved_result(project)
    assert report['stale']
    assert report['stale_reason']['code'] == 'uncertainty_changed'
    assert replay_report(frozen)==frozen['results']


def test_saved_result_records_measurement_revision_and_preserves_first_cause(tmp_path):
    from autopyro.core import Project

    project, binding, report = data(y_uncertainty=False)
    row_id = project.rows[0].id
    column_id = binding['y_id']
    measurement = project.science.current(row_id, column_id)
    assert report['inputs'][0]['measurements']['y'] == {
        'id': measurement.id, 'revision': measurement.revision,
    }
    project.view['interpretation_result'] = report
    measurement.revision += 1
    path = tmp_path / 'snapshot.apyro'
    project.save(path)

    loaded = Project.load(path)
    saved = loaded.view['interpretation_result']
    reason = saved['stale_reason']
    assert saved['stale']
    assert (reason['code'], reason['row_id'], reason['column_id']) == (
        'measurement_changed', row_id, column_id,
    )
    assert reason['current']['revision'] == reason['previous']['revision'] + 1
    loaded.save(path)
    assert Project.load(path).view['interpretation_result']['stale_reason'] == reason
    assert replay_report(saved) == saved['results']


def test_saved_result_identifies_scheme_revision():
    project, binding, report = data()
    scheme = binding['scheme']
    project.view['interpretation_result'] = report
    project.view['interpretation_schemes'] = {scheme['id']: dict(scheme, revision='new')}

    check_saved_result(project)

    reason = report['stale_reason']
    assert reason['code'] == 'scheme_changed'
    assert reason['scheme_id'] == scheme['id']
    assert reason['old_revision'] == scheme['revision']
    assert reason['new_revision'] == 'new'


@pytest.mark.parametrize('kind, amount, expected',[
    ('relative_percent',10,(1.71,2.09)), ('standard',.2,(1.7,2.1)),
    ('expanded',.2,(1.7,2.1)), ('absolute',0,None), ('',.2,None)])
def test_no_invented_confidence_multiplier(kind,amount,expected):
    project,binding,_=data()
    m=project.science.current(project.rows[0].id,binding['x_id'])
    m.uncertainty=amount;m.uncertainty_type=kind
    bounds,metadata,warning=measurement_interval(project,project.rows[0].id,binding['x_id'],1.9,'%wt')
    assert (bounds is None) if expected is None else bounds == pytest.approx(expected)
    assert bool(warning)==(kind=='')
    assert metadata['confidence_level']==.95


@pytest.mark.parametrize('invalid_type', [None, 12, ['absolute']])
def test_invalid_uncertainty_type_warns_and_report_replays(invalid_type):
    from autopyro.schemes.measurement_uncertainty import interval_from_metadata

    project, binding, _ = data()
    measurement = project.science.current(project.rows[0].id, binding['x_id'])
    measurement.uncertainty_type = invalid_type
    bounds, _, warning = measurement_interval(
        project, project.rows[0].id, binding['x_id'], 1.9, '%wt')
    assert bounds is None
    assert 'Тип неопределённости' in warning
    # Older exports may use the alternate key.
    assert interval_from_metadata(
        {'uncertainty_type': invalid_type, 'uncertainty': .2}, 1.9, '%wt')[0] is None
    report = evaluate_project(project, project.row_map(), **binding)
    assert len(report['results']) == len(project.rows)
    assert any(issue['code'] == 'uncertainty_unchecked' for issue in report['inputs'][0]['quality'])
    assert report['inputs'][0]['x_bounds'] is None
    assert replay_report(report) == report['results']


@pytest.mark.parametrize('y_uncertainty',[False,True])
def test_prepared_glyphs_export_and_uuid_selection(y_uncertainty):
    project,binding,_=data(y_uncertainty=y_uncertainty)
    spec=PlotSpec(x_id=binding['x_id'],y_id=binding['y_id'],interpretation=binding)
    source=capture_input(project,project.row_map(),[spec],input_token(project,0,[spec]))
    assert not source.project.science.measurements  # Workers only need the frozen metadata.
    prepared=PlotDataCache().prepare(source,project.row_map(),spec)
    figure,result=render_plot(project,project.row_map(),spec,prepared=prepared)
    assert result['uncertainty_crossings']=={project.rows[0].id:['A','B']}
    kind=PolyCollection if y_uncertainty else LineCollection
    assert any(isinstance(c,kind) for c in figure.axes[0].collections)
    assert len(result['artists'])==1
    overlay=SelectionOverlay(FigureCanvasAgg(figure),result)
    overlay.set_selected([project.rows[0].id])
    assert overlay.selected_ids=={project.rows[0].id}
    overlay.dispose()
    for fmt in ('png','svg','pdf'):
        output=BytesIO();figure.savefig(output,format=fmt);assert output.tell()>100
    spec.show_uncertainty=False
    _,hidden=render_plot(project,project.row_map(),spec,prepared=prepared)
    assert hidden['uncertainty_crossings']==result['uncertainty_crossings']


@pytest.mark.parametrize('amount', [None, 10 ** 400, True, -1, '0.2'])
def test_invalid_saved_uncertainty_amount_does_not_interrupt_plot(amount):
    from autopyro.schemes.measurement_uncertainty import interval_from_metadata
    from autopyro.visualization.uncertainty_plot import draw_measurement_error_bars
    from matplotlib.figure import Figure
    from types import SimpleNamespace

    metadata = {'type': 'absolute', 'unit': '%wt'}
    if amount is not None:
        metadata['uncertainty'] = amount
    bounds, warning = interval_from_metadata(metadata, 1.9, '%wt')
    assert bounds is None and 'Неопределённость должна' in warning
    project, binding, _ = data()
    row_id = project.rows[0].id
    column_id = binding['x_id']
    project.view['measurement_uncertainty'] = {column_id: {row_id: metadata}}
    axis = Figure().subplots()
    warnings = draw_measurement_error_bars(
        axis, project, [(row_id, 1.9)], [column_id], [SimpleNamespace(scale='linear')],
        {column.id: column for column in project.columns})
    assert len(warnings) == 1
    assert not axis.collections
