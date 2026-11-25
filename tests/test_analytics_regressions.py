from copy import deepcopy

import pytest

from autopyro.database.tools.compare_analytics import compare_reports


def report(times=(100,101,99)):
    return {'postgres':'PostgreSQL test','settings':{'jit':'on'},'counts':{'results':100},
            'sample_id':1,'well_id':2,'repeats':len(times),'queries':{'example':{
                'sql':'SELECT * FROM example WHERE id=%s','params':[1],'error':None,
                'execution_ms':list(times),'median_ms':0,
                'plans':[{'Execution Time':t,'Plan':{'Actual Rows':7}} for t in times]}}}


def test_median_detects_regression_despite_one_fast_run():
    result = compare_reports(report(),report((1,150,160)))
    assert not result['passed']
    assert result['queries']['example']['candidate_ms']==150


def test_single_slow_outlier_is_not_regression():
    assert compare_reports(report(),report((100,101,10000)))['passed']


def test_small_absolute_change_is_not_significant():
    assert compare_reports(report((1,1,1)),report((2,2,2)))['passed']


@pytest.mark.parametrize('key,value',[('counts',{'results':101}),('settings',{}),
    ('postgres','new version'),('sample_id',4),('well_id',9)])
def test_incomparable_conditions_are_rejected(key,value):
    candidate = report()
    candidate[key]=value
    with pytest.raises(ValueError,match='Incomparable'):
        compare_reports(report(),candidate)


@pytest.mark.parametrize('key,value',[('sql','SELECT 1'),('params',[2])])
def test_changed_query_is_rejected(key,value):
    candidate=report()
    candidate['queries']['example'][key]=value
    with pytest.raises(ValueError,match='query'):
        compare_reports(report(),candidate)


def test_changed_result_count_is_rejected():
    candidate=report()
    for plan in candidate['queries']['example']['plans']:
        plan['Plan']['Actual Rows']=8
    with pytest.raises(ValueError,match='row count changed'):
        compare_reports(report(),candidate)


def test_timeout_is_failure_even_with_partial_success():
    candidate=report()
    candidate['queries']['example']['error']='Timeout after 45s'
    candidate['queries']['example']['execution_ms']=[1]
    assert not compare_reports(report(),candidate)['passed']


def test_incomplete_success_is_rejected():
    candidate=report()
    candidate['queries']['example']['execution_ms'].pop()
    with pytest.raises(ValueError,match='incomplete'):
        compare_reports(report(),candidate)


def test_missing_query_is_rejected():
    candidate=report()
    candidate['queries']['other']=deepcopy(candidate['queries']['example'])
    with pytest.raises(ValueError,match='query sets'):
        compare_reports(report(),candidate)


@pytest.mark.parametrize('times',[(float('nan'),1,2),(float('inf'),1,2),(-1,1,2)])
def test_invalid_timings_are_rejected(times):
    with pytest.raises(ValueError,match='execution time'):
        compare_reports(report(),report(times))


def test_changed_timing_without_matching_explain_is_rejected():
    candidate=report()
    candidate['queries']['example']['execution_ms'][0]=1000
    with pytest.raises(ValueError,match='EXPLAIN'):
        compare_reports(report(),candidate)
