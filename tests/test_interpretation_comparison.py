"""Shared comparison input, three schemes and UUID selection across axes."""
from io import BytesIO

from matplotlib.backends.backend_agg import FigureCanvasAgg

from autopyro.io import build_project
from autopyro.plotting import PlotSpec, render_plot
from autopyro.plot_data import capture_input, input_token, PlotDataCache
from autopyro.interpretation_comparison import comparison_panels, comparison_reports, type_agreement
from autopyro.schemes.revision_comparison import changed_samples


def test_comparison_rejects_missing_or_duplicate_primary_axis():
    import pytest

    project, binding, _ = data()
    for invalid in ('', 'removed-column', binding['y_id']):
        with pytest.raises(ValueError, match='Выберите две разные существующие колонки'):
            comparison_panels(project, standard_schemes(), dict(binding, x_id=invalid), False)


def test_revision_preview_excludes_numeric_values_rejected_by_qc():
    from copy import deepcopy

    from autopyro.schemes.interpretation import evaluate_project
    from autopyro.ui.dialogs.scheme_revision import SchemeRevisionDialog

    project, binding, _ = data()
    after = evaluate_project(project, project.row_map(), **binding)
    before = deepcopy(after)
    for result in before['results']:
        result['status'] = 'outside'
    after['inputs'][0]['statuses'][0] = 'qc_excluded'
    rows, previous, coordinates = SchemeRevisionDialog.prepare(before, after, lambda: None)
    assert len(rows) == 2
    assert previous == {0: (1, 3), 1: (2, 4)}
    assert coordinates == {1: (2, 4)}
    after['scheme']['x']['transform'] = 'log10'
    after['inputs'][1]['x'] = 0
    assert SchemeRevisionDialog.prepare(before, after, lambda: None)[2] == {}


def test_revision_views_agree_on_same_label_replacement_and_ignore_order():
    from copy import deepcopy

    from autopyro.schemes.interpretation import evaluate_project
    from autopyro.schemes.interpretation_revisions import revision_changes

    project, binding, _ = data()
    before = evaluate_project(project, project.row_map(), **binding)
    first, second = before['results']
    assert len(second['boundary_regions']) > 1
    after = deepcopy(before)
    old_id = next(key for key in first['boundary_regions'] if key not in second['boundary_regions'])
    region = next(region for region in after['scheme']['regions'] if region['id'] == old_id)
    region['id'] = 'replacement'
    after['results'][0]['boundary_regions'] = [
        'replacement' if key == old_id else key for key in first['boundary_regions']]
    after['results'][1]['boundary_regions'].reverse()

    assert [row_id for row_id, _, _ in changed_samples(before, after)] == [first['row_id']]
    change, = revision_changes(project, before, after)
    assert change.row_id == first['row_id'] and change.flags == ['outcome']
    assert any(name.endswith(f'[{old_id}]') for name in change.before[None][1])
    assert any(name.endswith('[replacement]') for name in change.after[None][1])


def test_revision_input_details_distinguish_binding_measurement_and_uncertainty():
    from copy import deepcopy

    from autopyro.schemes.interpretation import evaluate_project
    from autopyro.schemes.interpretation_revisions import input_differences, revision_changes

    project, binding, _ = data()
    before = evaluate_project(project, project.row_map(), **binding)
    after = deepcopy(before)
    after['binding']['x_id'] = 'replacement-toc'
    first = before['inputs'][0]
    first['measurements']['y'] = {'id': 's2', 'revision': 1}
    first['uncertainty']['y'] = dict(measurement_id='s2', revision=1, uncertainty=.1)
    first['y_bounds'] = [2.9, 3.1]
    updated = after['inputs'][0]
    updated['measurements']['y'] = {'id': 's2', 'revision': 2}
    updated['uncertainty']['y'] = dict(measurement_id='s2', revision=2, uncertainty=.1)
    updated['y_bounds'] = [2.8, 3.2]
    updated['uncertainty']['x'] = {'uncertainty': .05}
    after['inputs'].reverse()
    after['results'].reverse()

    changes = {change.row_id: change for change in revision_changes(project, before, after)}
    details = list(input_differences(before, after, changes[first['row_id']]))
    assert [(axis, field) for _, axis, field, _, _ in details] == [
        ('x', 'binding'), ('x', 'uncertainty'), ('y', 'measurement'), ('y', 'bounds')]
    assert details[1][-2:] == (None, {'uncertainty': .05})
    assert details[2][-2:] == ({'id': 's2', 'revision': 1}, {'id': 's2', 'revision': 2})
    assert details[3][-2:] == ((2.9, 3.1), (2.8, 3.2))


def test_revision_comparison_uses_uuid_and_checks_types_status_and_hio_range():
    def row(row_id, types, status='classified', bounds=(100, 200)):
        return dict(row_id=row_id, possible_types=types, status=status, value_bounds=bounds)

    scheme = {'strategy': 'initial_hi'}
    before = {'scheme': scheme, 'results': [
        row('type', ['I']), row('status', ['I']), row('range', ['I']), row('same', ['I', 'II']),
    ]}
    after = {'scheme': scheme, 'results': [
        row('same', ['II', 'I']), row('range', ['I'], bounds=(100, 250)),
        row('status', ['I'], status='ambiguous'), row('type', ['II']),
    ]}
    assert [row_id for row_id, _, _ in changed_samples(before, after)] == ['range', 'status', 'type']


def test_revision_preview_detects_threshold_side_and_region_rename():
    from copy import deepcopy

    scheme = {'regions': [{'id': 'a', 'label': 'Before'}],
              'lines': [{'id': 'cut', 'label': 'Cut', 'threshold': {'less': 'Low', 'greater': 'High'}}]}
    before = {'scheme': scheme, 'results': [
        {'row_id': 'threshold', 'regions': [], 'boundary_regions': [], 'status': 'classified',
         'thresholds': {'cut': {'relation': 'less', 'possible_relations': ['less']}}},
        {'row_id': 'region', 'regions': ['a'], 'boundary_regions': [], 'status': 'classified'},
    ]}
    after = deepcopy(before)
    after['results'][0]['thresholds']['cut'] = {'relation': 'greater', 'possible_relations': ['greater']}
    after['scheme']['regions'][0]['label'] = 'After'
    assert [row_id for row_id, _, _ in changed_samples(before, after)] == ['threshold', 'region']
from autopyro.standard_interpretation import standard_schemes
from autopyro.selection_overlay import SelectionOverlay


def data():
    units = ['%wt','mg HC/g rock','mg HC/g TOC','mg CO2/g TOC','°C']
    roles = ['TOC','S2','HI','OI','Tmax']
    project = build_project(roles,[['1','3','300','30','430'],['2','4','200','20','440']],[1,2],
                            [(i,n,'number',u) for i,(n,u) in enumerate(zip(roles,units))],{})
    x,y = [c.id for c in project.columns[:2]]
    schemes = standard_schemes()
    binding = dict(scheme=schemes['S2–TOC'],x_id=x,y_id=y,x_unit=units[0],y_unit=units[1],text_numbers=False)
    panels = comparison_panels(project,schemes,binding,False)
    return project, binding, panels


def test_threshold_side_change_appears_in_both_revision_comparisons():
    from copy import deepcopy
    from autopyro.schemes.interpretation import evaluate_project
    from autopyro.schemes.interpretation_revisions import revision_changes

    project, binding, _ = data()
    scheme = binding['scheme']
    scheme['regions'] = []
    scheme['lines'] = [{
        'id': 'cutoff', 'label': 'Порог', 'threshold': {'axis': 'y', 'less': 'Низкий', 'greater': 'Высокий'},
        'geometry': {'type': 'LineString', 'coordinates': [[0, 1], [5, 1]]},
    }]
    ids = [project.rows[0].id]
    old = evaluate_project(project, ids, **binding)
    new_scheme = deepcopy(scheme)
    new_scheme['revision'] += '-test'
    new_scheme['lines'][0]['geometry']['coordinates'] = [[0, 5], [5, 5]]
    new = evaluate_project(project, ids, **dict(binding, scheme=new_scheme))
    assert old['results'][0]['status'] == new['results'][0]['status'] == 'classified'
    [(row_id, before, after)] = changed_samples(old, new)
    assert row_id == ids[0]
    assert before[4][1] == ('Порог: Высокий',)
    assert after[4][1] == ('Порог: Низкий',)
    assert [change.flags for change in revision_changes(project, old, new)] == [['outcome']]


def test_project_refresh_keeps_snapshot_current_until_a_calculation_input_changes():
    from autopyro.schemes.interpretation import evaluate_project

    project, binding, _ = data()
    project.sync_measurements()
    row, other = project.rows
    report = evaluate_project(project, [row.id], **binding)
    project.view['interpretation_result'] = report
    project.refresh()
    assert not report['stale']

    column_id = binding['x_id']
    other.values[column_id] = 2.5
    project.refresh(rows=[other])
    assert not report['stale']

    row.values[column_id] = 1.5
    project.refresh(rows=[row])
    assert report['stale_reason'] == {
        'code': 'input_changed', 'row_id': row.id, 'column_id': column_id, 'axis': 'x', 'diagram': None,
    }


def test_project_refresh_invalidates_all_rows_snapshot_when_a_sample_is_added():
    from autopyro.model.core import Row
    from autopyro.schemes.interpretation import evaluate_project

    project, binding, _ = data()
    project.sync_measurements()
    report = evaluate_project(project, project.row_map(), **binding)
    project.view['interpretation_result'] = dict(report, scope='all')
    project.rows.append(Row(dict(project.rows[0].values)))

    project.refresh()

    assert project.view['interpretation_result']['stale_reason'] == {
        'code': 'binding_changed', 'fields': ['scope'],
    }


def test_saved_snapshot_detects_qc_change_after_reopening(tmp_path):
    from autopyro.schemes.interpretation import evaluate_project, check_saved_result

    project, binding, _ = data()
    oi = next(column.id for column in project.columns if column.name == 'OI')
    row = project.rows[0]
    row.values[oi] = 3
    project.sync_measurements()
    report = evaluate_project(project, [row.id], **binding)
    project.view['interpretation_result'] = report
    check_saved_result(project)
    assert not report['stale']
    row.values[oi] = 400
    project.sync_measurements()
    path = tmp_path / 'qc-change.apyro'
    project.save(path)
    saved = type(project).load(path).view['interpretation_result']
    assert saved['stale_reason'] == {'code': 'qc_changed', 'row_id': row.id, 'diagram': None}


def test_review_mode_changes_preserve_decision_but_recheck_shared_evidence():
    from copy import deepcopy
    from autopyro.schemes.interpretation import evaluate_project
    from autopyro.schemes.interpretation_review import review_state, save_review

    project, binding, panels = data()
    ids = list(project.row_map())
    primary = evaluate_project(project, ids, **binding)
    combined = dict(primary, comparisons=comparison_reports(project, ids, panels, primary_report=primary))
    for reviewed in (primary, combined):
        decision = save_review(project, reviewed, 0, 'expert', 'Вывод', 'Комментарий')
        for shown in (primary, combined):
            assert review_state(project, shown, 0) == ('expert', decision)
    modified = deepcopy(combined)
    modified['comparisons']['HI–OI']['inputs'][0]['x'] = 40
    assert review_state(project, modified, 0) == ('stale', decision)


def test_saved_snapshot_detects_new_qc_exclusion_on_axis():
    from autopyro.schemes.interpretation import evaluate_project, check_saved_result

    project, binding, _ = data()
    project.sync_measurements()
    row = project.rows[0]
    report = evaluate_project(project, [row.id], **binding)
    project.view['interpretation_result'] = report
    column_id = binding['x_id']
    measurement = project.science.current(row.id, column_id)
    project.view['qc_exclusions'] = {'interpretation': {measurement.id: {'revision': measurement.revision}}}
    check_saved_result(project, changed_rows=[row])
    assert report['stale_reason'] == {
        'code': 'qc_changed', 'row_id': row.id, 'column_id': column_id, 'axis': 'x', 'diagram': None,
    }


def test_worker_input_preserves_qc_and_uncertainty_after_live_edits():
    from autopyro.schemes.interpretation import evaluate_project
    from autopyro.schemes.interpretation_input import capture_interpretation

    project, binding, _ = data()
    project.sync_measurements()
    row = project.rows[0]
    ids = [row.id]
    x = project.science.current(row.id, binding['x_id'])
    x.uncertainty, x.uncertainty_type = 0.1, 'absolute'
    oi = next(column.id for column in project.columns if column.name == 'OI')
    ancillary = project.science.current(row.id, oi)
    project.view['qc_exclusions'] = {'interpretation': {ancillary.id: {'revision': ancillary.revision}}}
    expected = evaluate_project(project, ids, **binding)
    snapshot = capture_interpretation(project, ids, [binding])
    row.values[binding['x_id']] = 99
    x.uncertainty = 1
    project.view['qc_exclusions'].clear()
    actual = evaluate_project(snapshot, ids, **binding)
    assert actual['inputs'] == expected['inputs']
    assert actual['results'] == expected['results']


def test_three_panels_share_prepared_data_and_selection():
    project,binding,panels = data()
    assert all('binding' in p for p in panels)
    spec = PlotSpec(x_id=binding['x_id'],y_id=binding['y_id'],interpretation=binding,interpretation_panels=panels)
    source = capture_input(project,project.row_map(),[spec],input_token(project,0,[spec]))
    assert source.local_ids == {c.id for c in project.columns}
    prepared = PlotDataCache().prepare(source,project.row_map(),spec)
    figure,result = render_plot(project,project.row_map(),spec,prepared=prepared)
    assert len(result['axes']) == 3
    assert [ax.get_title() for ax in result['axes']] == ['HI–Tmax','HI–OI','S2–TOC']
    assert all(len(a._autopyro_row_ids) == 2 for a in result['artists'])
    canvas = FigureCanvasAgg(figure)
    overlay = SelectionOverlay(canvas,result)
    overlay.set_selected([project.rows[0].id])
    assert overlay.selected_ids == {project.rows[0].id}
    assert len(overlay.layers) == 3
    assert all(len(layer.artist.get_offsets()) == 1 for layer in overlay.layers)
    overlay.dispose()
    figure.savefig(BytesIO(),format='svg')
    reports = comparison_reports(project,project.row_map(),panels)
    assert len(reports) == 3
    assert all(len(r['results']) == 2 for r in reports.values())
    assert PlotSpec.from_dict(spec.to_dict()).interpretation_panels == panels


def test_missing_or_duplicate_columns_are_explicit_placeholders():
    project,binding,panels = data()
    project.columns = [c for c in project.columns if c.name != 'OI']
    panels = comparison_panels(project,standard_schemes(),binding,False)
    assert 'error' in panels[1] and 'OI' in panels[1]['error']
    spec = PlotSpec(x_id=binding['x_id'],y_id=binding['y_id'],interpretation_panels=panels)
    figure,result = render_plot(project,project.row_map(),spec)
    assert len(result['axes']) == 3 and len(result['notices']) == 1
    assert len(result['artists']) == 2


def test_primary_report_is_reused_without_a_snapshot_cycle(monkeypatch):
    import json
    import autopyro.interpretation_comparison as module
    from autopyro.interpretation import evaluate_project
    project,binding,panels = data()
    primary = evaluate_project(project,project.row_map(),binding['x_id'],binding['y_id'],
                               binding['x_unit'],binding['y_unit'],binding['scheme'])
    calls=[]
    original=module.evaluate_project
    def evaluate(*args, **kwargs):
        calls.append(args[6]['name'])
        return original(*args, **kwargs)
    monkeypatch.setattr(module,'evaluate_project',evaluate)
    reports=module.comparison_reports(project,project.row_map(),panels,primary_report=primary)
    assert calls == ['HI–Tmax','HI–OI']
    assert reports['S2–TOC']['results'] is primary['results']
    primary['comparisons']=reports
    assert json.dumps(primary)


def test_ambiguous_columns_and_unknown_units_are_not_guessed():
    from autopyro.core import Column
    project,binding,_ = data()
    oi=next(c for c in project.columns if c.name=='OI')
    oi.unit=''
    panels=comparison_panels(project,standard_schemes(),binding,False)
    assert 'единицу' in panels[1]['error']
    oi.unit='mg CO2/g TOC'
    project.columns.append(Column('Alternative',kind='number',unit=oi.unit,metadata={'property_role':'OI'}))
    panels=comparison_panels(project,standard_schemes(),binding,False)
    assert 'несколько' in panels[1]['error']


def test_type_agreement_distinguishes_overlap_from_missing_results():
    reports = {
        'HI–Tmax': {'results': [
            {'status': 'ambiguous', 'possible_types': ['I', 'II']},
            {'status': 'ambiguous', 'possible_types': ['I', 'II']},
            {'status': 'classified', 'possible_types': ['III']},
            {'status': 'not_applicable', 'possible_types': ['II']},
        ]},
        'HI–OI': {'results': [
            {'status': 'ambiguous', 'possible_types': ['II', 'I']},
            {'status': 'classified', 'line_hits': ['II']},
            {'status': 'classified', 'possible_types': ['III/IV']},
            {'status': 'classified', 'possible_types': ['II']},
        ]},
    }
    assert type_agreement(reports, 4) == ['equal', 'partial', 'different', 'unavailable']
    assert type_agreement({'HI–Tmax': reports['HI–Tmax']}, 4) == ['unavailable'] * 4


def test_review_rechecks_only_the_changed_sample_or_scheme_revision():
    from copy import deepcopy

    from autopyro.schemes.interpretation import evaluate_project
    from autopyro.schemes.interpretation_review import review_changes, review_state, save_review

    project, binding, _ = data()
    project.sync_measurements()

    def evaluate():
        return evaluate_project(project, project.row_map(), **binding)

    report = evaluate()
    original = deepcopy(report)
    decision = save_review(project, report, 0, 'expert', 'Оценка эксперта', 'Обоснование')
    assert report == original

    other = project.science.current(project.rows[1].id, binding['x_id'])
    other.revision += 1
    assert review_state(project, evaluate(), 0) == ('expert', decision)

    measurement = project.science.current(project.rows[0].id, binding['x_id'])
    measurement.revision += 1
    revised = evaluate()
    assert review_state(project, revised, 0) == ('stale', decision)
    assert review_changes(revised, 0, decision) == [(report['scheme']['name'], ['input'])]
    save_review(project, revised, 0, 'accepted')

    binding['scheme']['revision'] += '-review-test'
    binding['scheme']['regions'][0]['color'] = '#123456'
    assert review_state(project, evaluate(), 0)[0] == 'accepted'
    binding['scheme']['tolerance'] = 0.01
    revised = evaluate()
    state, previous = review_state(project, revised, 0)
    assert state == 'stale'
    assert previous['status'] == 'accepted'
    assert 'scheme' in dict(review_changes(revised, 0, previous))[report['scheme']['name']]


def test_review_explains_quality_change_on_companion_diagram():
    from copy import deepcopy

    from autopyro.schemes.interpretation import evaluate_project
    from autopyro.schemes.interpretation_review import review_changes, review_state, save_review

    project, binding, panels = data()
    report = evaluate_project(project, project.row_map(), **binding)
    report['comparisons'] = comparison_reports(project, project.row_map(), panels, primary_report=report)
    decision = save_review(project, report, 0, 'accepted')
    changed = deepcopy(report)
    changed['comparisons']['HI–OI']['inputs'][0]['quality'] = [{'code': 'example'}]
    assert review_state(project, changed, 0) == ('stale', decision)
    assert review_changes(changed, 0, decision) == [('HI–OI', ['quality'])]
    decision.pop('evidence_parts')
    assert review_state(project, changed, 0) == ('stale', decision)
    assert review_changes(changed, 0, decision) == []


def test_legacy_review_uses_sample_uuid_after_scope_and_order_change():
    from copy import deepcopy
    from autopyro.schemes.interpretation import evaluate_project, scheme_hash
    from autopyro.schemes.interpretation_review import _evidence, review_state, save_review

    project, binding, panels = data()
    primary = evaluate_project(project, project.row_map(), **binding)
    compared = deepcopy(primary)
    compared['comparisons'] = comparison_reports(project, project.row_map(), panels, primary_report=primary)
    decision = save_review(project, compared, 0, 'accepted')
    decision.pop('evidence')
    decision.pop('evidence_version')
    decision['evidence_hash'] = scheme_hash(_evidence(compared, 0))
    project.view['interpretation_previous_result'] = compared

    reordered = deepcopy(primary)
    reordered['inputs'].reverse()
    reordered['results'].reverse()
    assert review_state(project, reordered, len(reordered['results']) - 1) == ('accepted', decision)
    narrowed = deepcopy(primary)
    narrowed['inputs'] = narrowed['inputs'][:1]
    narrowed['results'] = narrowed['results'][:1]
    assert review_state(project, narrowed, 0) == ('accepted', decision)
    reordered['inputs'][-1]['x'] = 'changed'
    assert review_state(project, reordered, len(reordered['results']) - 1) == ('stale', decision)


def test_review_comparison_visibility_and_legacy_evidence():
    from autopyro.schemes.interpretation import evaluate_project, scheme_hash
    from autopyro.schemes.interpretation_review import _evidence, migrate_review_evidence, review_state, save_review

    project, binding, panels = data()

    def evaluate(compare):
        report = evaluate_project(project, project.row_map(), **binding)
        if compare:
            report['comparisons'] = comparison_reports(project, project.row_map(), panels, primary_report=report)
        return report

    primary, compared = evaluate(False), evaluate(True)
    save_review(project, primary, 0, 'accepted')
    assert review_state(project, compared, 0)[0] == 'accepted'
    decision = save_review(project, compared, 0, 'accepted')
    decision.pop('evidence')
    decision.pop('evidence_version')
    decision['evidence_hash'] = scheme_hash(_evidence(compared, 0))
    migrate_review_evidence(project, compared)
    assert decision['evidence_version'] == 2
    assert review_state(project, primary, 0)[0] == 'accepted'

    oi_id = next(column.id for column in project.columns if column.name == 'OI')
    project.rows[0].values[oi_id] = '200'
    assert review_state(project, evaluate(True), 0)[0] == 'stale'
