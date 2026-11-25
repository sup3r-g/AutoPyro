"""Sample identity, provenance and one-observation-per-sample comparisons."""
from copy import deepcopy
import json

import pytest

from autopyro.core import Column, Project, Row
from autopyro.geochemistry_workspace import (
    STATE_KEY, GeochemistryIndex, compare_samples, comparison_clustering, comparison_dbscan, comparison_is_current, comparison_kmeans,
    comparison_oil_mixing, comparison_pca, correlate_oil_profiles,
    link_samples, relate_materials, save_group, select_measurement, unlink_sample,
)
from autopyro.database.geochem import project_with_repeats
from autopyro.geochemistry_workspace import MODEL_RUNS_KEY, archive_model_run, compare_alkane_profiles, comparison_prediction, replay_model_run
from autopyro.geochemistry_workspace import verify_model_run_package


@pytest.fixture
def dataset():
    name = Column('Sample / Name')
    value = Column('Concentration', 'number', unit='%wt')
    rows = [Row({name.id: 'Same name', value.id: number},
                source={'sample_id': sid, 'source_record_id': i,
                        'sample_type': 'rock', 'preparation': 'whole'})
            for i, (sid, number) in enumerate([(1, 2), (1, 3), (2, 4), (3, 8)])]
    project = Project(columns=[name, value], rows=rows,
                      source={'endpoint': {'host': 'db', 'port': 5432, 'database': 'geo'}})
    project.sync_measurements()
    return project, value


def ids(project):
    return list(GeochemistryIndex(project).samples)


def test_pca_standardizes_units_and_records_exclusions():
    x, y, constant = [Column(name, 'number') for name in ('X', 'Y', 'Constant')]
    project = Project(columns=[x, y, constant], rows=[
        Row({x.id: a, y.id: b, constant.id: 5}) for a, b in [(0, 0), (1, 0), (2, 1000), (3, 'ND')]
    ])
    samples = ids(project)
    report = compare_samples(project, samples[:2], samples[2:], [x.id, y.id, constant.id])
    run = comparison_pca(report)
    assert run['excluded'] == {samples[-1]: {y.id: 'non_detect'}}
    assert run['constant_fields'] == [constant.id]
    assert run['fields'] == [x.id, y.id]
    assert run['explained_variance_ratio'] == pytest.approx([(1 + 3**.5 / 2) / 2, (1 - 3**.5 / 2) / 2])
    assert run['input_digest'] == report['input_digest']
    assert json.loads(json.dumps(run, allow_nan=False)) == run


def test_pca_rejects_rank_one_data():
    x, y = [Column(name, 'number') for name in ('X', 'Y')]
    project = Project(columns=[x, y], rows=[Row({x.id: i, y.id: 2 * i}) for i in range(3)])
    samples = ids(project)
    with pytest.raises(ValueError, match='независимых'):
        comparison_pca(compare_samples(project, samples[:1], samples[1:], [x.id, y.id]))


def test_ward_clustering_uses_standardized_measurements_and_reports_exclusions():
    x, y = [Column(name, 'number') for name in ('X', 'Y')]
    project = Project(columns=[x, y], rows=[
        Row({x.id: a, y.id: b}) for a, b in [(0, 0), (1, 100), (10, 1000), (11, 1100), (12, 'ND')]
    ])
    samples = ids(project)
    report = compare_samples(project, samples[::2], samples[1::2], [x.id, y.id])
    run = comparison_clustering(report, 2)
    labels = {item['sample_id']: item['cluster'] for item in run['membership']}
    assert labels[samples[0]] == labels[samples[1]] != labels[samples[2]] == labels[samples[3]]
    assert run['excluded'] == {samples[4]: {y.id: 'non_detect'}}
    assert len(run['linkage']) == 3
    assert json.loads(json.dumps(run, allow_nan=False)) == run
    with pytest.raises(ValueError, match='Число кластеров'):
        comparison_clustering(report, 4)


def test_kmeans_reproduces_centers_and_keeps_groups_separate_from_clusters():
    x = Column('X', 'number')
    project = Project(columns=[x], rows=[Row({x.id: value}) for value in (0, 1, 10, 11, 'ND')])
    samples = ids(project)
    report = compare_samples(project, samples[::2], samples[1::2], [x.id])
    run = comparison_kmeans(report, 2)
    assert comparison_kmeans(report, 2) == run
    assert [item['cluster'] for item in run['membership']] == [1, 2, 1, 2]
    assert run['excluded'] == {samples[-1]: {x.id: 'non_detect'}}
    variance = 101 / 3
    assert [center[0] for center in run['centers']] == pytest.approx([-5 / variance**.5, 5 / variance**.5])
    assert run['inertia'] == pytest.approx(1 / variance)


def test_dbscan_expands_through_cores_and_distinguishes_noise_from_exclusions():
    x = Column('X', 'number')
    project = Project(columns=[x], rows=[Row({x.id: value}) for value in (0, .05, .25, .45, 1.5, 4, 'ND')])
    samples = ids(project)
    report = compare_samples(project, samples[:3], samples[3:], [x.id])
    run = comparison_dbscan(report, 0.15, 3)
    assert [item['role'] for item in run['membership']] == ['border', 'core', 'core', 'border', 'noise', 'noise']
    assert [item['cluster'] for item in run['membership']] == [1, 1, 1, 1, None, None]
    assert run['excluded'] == {samples[-1]: {x.id: 'non_detect'}}
    assert run['noise_count'] == 2 and run['cluster_count'] == 1
    all_noise = comparison_dbscan(report, 0.001, 3)
    assert all_noise['noise_count'] == 6 and all_noise['cluster_count'] == 0
    assert json.loads(json.dumps(run, allow_nan=False)) == run


def test_clr_ignores_sample_total_and_excludes_nonpositive_parts():
    columns = [Column(name, 'number', unit='%wt') for name in ('A', 'B', 'C')]
    rows = [(1, 2, 4), (2, 4, 8), (4, 2, 1), (2, 4, 1), (0, 2, 4), (-1, 2, 4)]
    project = Project(columns=columns, rows=[Row(dict(zip([c.id for c in columns], row))) for row in rows])
    samples = ids(project)
    report = compare_samples(project, samples[:3], samples[3:], [c.id for c in columns])
    run = comparison_pca(report, preprocessing='clr')
    assert run['scores'][0]['coordinates'] == pytest.approx(run['scores'][1]['coordinates'])
    assert run['excluded'] == {sample: {columns[0].id: 'nonpositive_composition'} for sample in samples[-2:]}
    assert run['preprocessing'] == 'clr'
    assert sum(run['composition']['pca_center']) == pytest.approx(0)
    ward = comparison_clustering(report, 2, preprocessing='clr')
    assert ward['linkage'][0][2] == pytest.approx(0, abs=1e-12)
    assert ward['membership'][0]['cluster'] == ward['membership'][1]['cluster']
    kmeans = comparison_kmeans(report, 2, preprocessing='clr')
    assert all(sum(center) == pytest.approx(0) for center in kmeans['centers'])
    dbscan = comparison_dbscan(report, .001, 2, preprocessing='clr')
    assert dbscan['noise_count'] == 2
    columns[-1].unit = 'ppm'
    mixed = compare_samples(project, samples[:3], samples[3:], [c.id for c in columns])
    with pytest.raises(ValueError, match='одинаковых единицах'):
        comparison_pca(mixed, preprocessing='clr')


def test_model_runs_survive_project_save_and_replay_without_live_inputs(tmp_path):
    columns = [Column(name, 'number') for name in ('X', 'Y')]
    project = Project(columns=columns, rows=[
        Row({columns[0].id: a, columns[1].id: b}) for a, b in [(1, 1), (2, 3), (3, 2), (5, 5)]
    ])
    samples = ids(project)
    report = compare_samples(project, samples[:2], samples[2:], [c.id for c in columns])
    for result in (comparison_pca(report), comparison_clustering(report, 2),
                   comparison_kmeans(report, 2), comparison_dbscan(report, .5, 2)):
        archive_model_run(project, report, result)
    project.rows[0].values[columns[0].id] = 99
    path = tmp_path / 'model-runs.apyro'
    project.save(path)
    restored = Project.load(path)
    archive = restored.view[MODEL_RUNS_KEY]
    assert len(archive['snapshots']) == 1 and len(archive['runs']) == 4
    assert len({record['id'] for record in archive['runs']}) == 4
    for record in archive['runs']:
        snapshot = archive['snapshots'][record['snapshot_id']]
        assert not comparison_is_current(restored, snapshot)
        assert replay_model_run(record, snapshot)
    corrupted = deepcopy(snapshot)
    corrupted['inputs']['points'][columns[0].id][samples[0]]['value'] = 100
    with pytest.raises(ValueError, match='снимок'):
        replay_model_run(record, corrupted)
    record['result']['noise_count'] += 1
    assert not replay_model_run(record, snapshot)
    with pytest.raises(ValueError, match='отличается'):
        verify_model_run_package({'run': record, 'snapshot': snapshot})
    with pytest.raises(ValueError, match='не содержит'):
        verify_model_run_package({'run': []})


def test_profile_overlap_preserves_proportions_and_checks_run_identity():
    columns = [Column(f'Sat-GC / nC{carbon}[a]', 'number', unit='count') for carbon in range(24, 35)]
    column_ids = [column.id for column in columns]
    project = Project(columns=columns, rows=[Row(dict(zip(column_ids, values))) for values in (
        [1, 3] + [0] * 9, [4, 4] + [0] * 9,
    )])
    project.sync_measurements()
    a, b = ids(project)
    report = compare_samples(project, [a], [b], column_ids)
    match = compare_alkane_profiles(report)
    assert match['overlap_percent'] == pytest.approx(75)
    assert [peak['difference_pp'] for peak in match['peaks'][:2]] == pytest.approx([25, -25])
    report['inputs']['points'][column_ids[0]][b]['measurement']['analysis_id'] = 'other-run'
    with pytest.raises(ValueError, match='одному анализу'):
        compare_alkane_profiles(report)


def test_oil_rock_correlation_preserves_missing_evidence_and_checks_material():
    peaks = [Column(f'Sat-GC / nC{carbon}[a]', 'number', unit='count') for carbon in range(24, 35)]
    ratio = Column('Sat-GC / Pr/Ph[a]', 'number', unit='1')
    censored = Column('Arom-GCMS / MPI-1[h]', 'number', unit='1')
    unrelated = Column('Porosity', 'number', unit='%')
    sat = Column('Isotopes / d13C Sat [‰]', 'number', unit='‰')
    arom = Column('Isotopes / d13C Arom [‰]', 'number', unit='‰')
    bulk = Column('Isotopes / d13C Bulk [‰]', 'number', unit='‰')
    columns = [*peaks, ratio, censored, unrelated, sat, arom, bulk]
    project = Project(columns=columns, rows=[
        Row({**{peak.id: i for i, peak in enumerate(peaks, 1)}, ratio.id: 1.5,
             censored.id: 0.8, unrelated.id: 10, sat.id: -29.5, arom.id: -28.2, bulk.id: -29},
            source={'sample_type': 'oil'}),
        Row({**{peak.id: i * 2 for i, peak in enumerate(peaks, 1)}, ratio.id: 2,
             censored.id: '<0.1', unrelated.id: 12, sat.id: -30, arom.id: '<-28', bulk.id: -30},
            source={'sample_type': 'rock', 'preparation': 'extract'}),
    ])

    def comparison():
        a, b = ids(project)
        return compare_samples(project, [a], [b], [column.id for column in columns], allow_mixed_material=True)

    report = comparison()
    match = correlate_oil_profiles(report)
    assert match['correlation']['materials'] == ['oil', 'rock_extract']
    assert match['correlation']['expert_review_required']
    assert match['overlap_percent'] == pytest.approx(100)
    prph, mpi, porosity, sat_unconfirmed, _, _ = match['evidence']
    assert prph['difference'] == pytest.approx(0.5)
    assert mpi['b']['status'] == 'censored' and mpi['difference'] is None
    assert porosity['a']['status'] == 'unsupported_correlation_indicator' and porosity['difference'] is None
    assert compare_alkane_profiles(report)['evidence'][2]['difference'] == 2
    assert sat_unconfirmed['a']['status'] == 'unconfirmed_isotope_basis'
    assert sat_unconfirmed['difference'] is None
    confirmed = correlate_oil_profiles(report, confirm_c15_vpdb=True)
    assert confirmed['review_key'] != match['review_key']
    assert confirmed['correlation']['isotope_basis'] == {'fraction': 'C15+', 'reference': 'VPDB'}
    sat_confirmed, arom_confirmed, bulk_confirmed = confirmed['evidence'][3:]
    assert sat_confirmed['a']['status'] == 'numeric' and sat_confirmed['difference'] == pytest.approx(-0.5)
    assert arom_confirmed['b']['status'] == 'censored' and arom_confirmed['difference'] is None
    assert bulk_confirmed['a']['status'] == 'unsupported_correlation_indicator'
    assert bulk_confirmed['difference'] is None
    b = report['request']['group_b'][0]
    report['inputs']['points'][peaks[0].id][b].update(status='missing', value=None)
    partial = correlate_oil_profiles(report)
    assert partial['overlap_percent'] is None
    assert 'полные профили' in partial['overlap_issue']
    assert partial['evidence']
    project.rows[1].source['preparation'] = 'whole'
    with pytest.raises(ValueError, match='экстракт'):
        correlate_oil_profiles(comparison())


@pytest.mark.parametrize('mixture,status,fraction,outside', [
    ((20, 30, 5, 'ND'), 'consistent', 0.25, False),
    ((20, 30, 9, 'ND'), 'inconsistent', 0.25, False),
    ((10, 40, 5, 'ND'), 'endpoint', 0, False),
    ((0, 50, 5, 'ND'), 'inconsistent', 0, True),
    ((20, '<30', 5, 'ND'), 'underdetermined', None, None),
])
def test_binary_oil_mixing_mass_balance_and_archive(mixture, status, fraction, outside):
    columns = [Column(name, 'number', unit=unit) for name, unit in
               zip(('X', 'Y', 'Control', 'Missing'), ('mg/kg oil', 'µg/g oil', 'mg/kg oil', 'mg/g oil'))]
    fields = [column.id for column in columns]
    project = Project(columns=columns, rows=[
        Row(dict(zip(fields, values)), source={'sample_type': 'oil'})
        for values in ((10, 40, 5, 1), (50, 0, 5, 2), mixture)
    ])
    a, b, mix = ids(project)
    report = compare_samples(project, [a, b], [mix], fields)
    tolerances = dict(zip(fields, (1, 2, 1, 0.1)))
    result = comparison_oil_mixing(report, tolerances)
    assert (result['status'], result['fraction_b'], result['outside_interval']) == (status, fraction, outside)
    assert result['markers'][3]['statuses'] == ['numeric', 'numeric', 'non_detect']
    assert not result['markers'][2]['used_for_fit']
    if fraction is not None:
        reverse = comparison_oil_mixing(compare_samples(project, [b, a], [mix], fields), tolerances)
        assert reverse['fraction_b'] == pytest.approx(1 - fraction)
        assert result['markers'][2]['predicted'] == 5
        assert result['markers'][2]['residual'] == mixture[2] - 5
    record = archive_model_run(project, report, result)
    snapshot = project.view[MODEL_RUNS_KEY]['snapshots'][record['snapshot_id']]
    record, snapshot = json.loads(json.dumps([record, snapshot]))
    assert replay_model_run(record, snapshot)
    record['result']['status'] = 'altered'
    assert not replay_model_run(record, snapshot)


def test_prediction_fits_training_only_and_rejects_shared_wells():
    well = Column('Well / Well Name')
    x, y = [Column(name, 'number') for name in ('X', 'Y')]
    project = Project(columns=[well, x, y], rows=[
        Row({well.id: name, x.id: a, y.id: b}) for name, a, b in (
            ('A-1', 0, 1), ('A-1', 1, 3), ('A-2', 2, 5), ('B-1', 10, 50), ('B-2', 20, 70),
        )
    ])
    samples = ids(project)
    report = compare_samples(project, samples[:3], samples[3:], [x.id, y.id])
    run = comparison_prediction(report, y.id)
    assert run['train_means'] == [1] and run['train_sample_sd'] == [1]
    assert run['coefficients'] == pytest.approx([3, 2])
    assert [row['predicted'] for row in run['predictions']] == pytest.approx([21, 41])
    assert all(row['outside_train_range'] == [x.id] for row in run['predictions'])
    assert run['metrics']['RMSE'] == pytest.approx(29)
    assert run['baseline']['value'] == 3
    assert run['baseline']['metrics']['RMSE'] == pytest.approx(3349**.5)
    record = archive_model_run(project, report, run)
    assert replay_model_run(record, project.view[MODEL_RUNS_KEY]['snapshots'][record['snapshot_id']])
    old = archive_model_run(project, report, comparison_prediction(report, y.id, version=1))
    assert 'baseline' not in old['result']
    assert replay_model_run(old, project.view[MODEL_RUNS_KEY]['snapshots'][old['snapshot_id']])
    project.rows[-1].values[well.id] = ' a-1 '
    with pytest.raises(ValueError, match='не должны пересекаться'):
        comparison_prediction(compare_samples(project, samples[:3], samples[3:], [x.id, y.id]), y.id)


def test_external_validation_reports_equal_well_weight_and_preserves_v2_replay():
    well = Column('Well / Well Name')
    x, y = [Column(name, 'number') for name in ('X', 'Y')]
    project = Project(columns=[well, x, y], rows=[
        Row({well.id: name, x.id: a, y.id: b}) for name, a, b in (
            ('A', 0, 1), ('A', 1, 3), ('A', 2, 5), ('B1', 1, 4), ('B1', 2, 6), ('B2', 3, 11),
        )
    ])
    samples = ids(project)
    report = compare_samples(project, samples[:3], samples[3:], [x.id, y.id])
    run = comparison_prediction(report, y.id)
    breakdown = run['validation_by_well']
    assert [item['sample_ids'] for item in breakdown['wells']] == [samples[3:5], samples[5:]]
    assert run['metrics']['RMSE'] == pytest.approx(6**.5)
    assert breakdown['equal_weight_metrics'] == pytest.approx({'RMSE': 8.5**.5, 'MAE': 2.5})
    assert breakdown['equal_weight_baseline_metrics']['RMSE'] == pytest.approx(34.5**.5)
    old = archive_model_run(project, report, comparison_prediction(report, y.id, version=2))
    assert 'validation_by_well' not in old['result']
    assert replay_model_run(old, project.view[MODEL_RUNS_KEY]['snapshots'][old['snapshot_id']])


def test_identity_uses_namespace_id_and_material_not_name(dataset):
    p, c = dataset
    index = GeochemistryIndex(p)
    assert len(index.samples) == 3
    first = next(iter(index.samples))
    assert index.catalog([p.rows[0].id])[0]['row_ids'] == [r.id for r in p.rows[:2]]
    assert index.value(first, c.id)['status'] == 'ambiguous'
    p.rows[1].source['preparation'] = 'extract'
    assert len(ids(p)) == 4
    p.rows[1].source['preparation'] = 'whole'
    p.rows[1].source['geochemistry_endpoint'] = {'host': 'another-db'}
    assert len(ids(p)) == 4
    p.source.clear()
    assert len(ids(p)) == 4


def test_manual_links_preserve_all_rows_measurements_and_raw_data(dataset):
    p, c = dataset
    before = deepcopy((p.rows, p.science))
    selected = ids(p)[1:]
    p.view[STATE_KEY] = link_samples(p, selected, 'Combined')
    index = GeochemistryIndex(p)
    linked = next(s for s in index.samples if s.startswith('link:'))
    assert index.samples[linked]['name'] == 'Combined'
    assert len(index.candidates(linked, c.id)) == 2
    assert (p.rows, p.science) == before
    p.view[STATE_KEY] = unlink_sample(p, linked)
    assert ids(p)[1:] == selected


def test_link_rejects_different_materials_and_invalid_duplicates(dataset):
    p, _ = dataset
    p.rows[-1].source['preparation'] = 'extract'
    with pytest.raises(ValueError, match='состояния материала'):
        link_samples(p, ids(p)[1:], 'bad')
    with pytest.raises(ValueError):
        link_samples(p, [ids(p)[0]] * 2, 'bad')


def test_explicit_selection_resolves_ambiguity_but_not_revision_changes(dataset):
    p, c = dataset
    sid = ids(p)[0]
    mid = GeochemistryIndex(p).candidates(sid, c.id)[1]['id']
    p.view[STATE_KEY] = select_measurement(p, sid, c.id, mid)
    assert GeochemistryIndex(p).value(sid, c.id)['value'] == 3
    p.science.measurements[mid].revision += 1
    assert GeochemistryIndex(p).value(sid, c.id)['status'] == 'stale_selection'
    p.view[STATE_KEY] = select_measurement(p, sid, c.id, None)
    assert GeochemistryIndex(p).value(sid, c.id)['status'] == 'ambiguous'
    with pytest.raises(ValueError):
        select_measurement(p, ids(p)[1], c.id, mid)


@pytest.mark.parametrize('raw,status', [('ND', 'non_detect'), ('<0.1', 'below_limit'),
                                     ('>10', 'above_limit'), ('~2', 'approximate'), (None, 'missing')])
def test_qualified_values_are_not_zeros(dataset, raw, status):
    p, c = dataset
    row = p.rows[-1]
    p.science.set_value(row.id, c.id, raw, c.kind)
    row.values[c.id] = raw
    report = compare_samples(p, [ids(p)[1]], [ids(p)[2]], [c.id])
    assert report['results'][0]['b']['n'] == 0
    assert report['results'][0]['b']['excluded'] == {status: 1}
    assert report['results'][0]['difference'] is None


def test_groups_statistics_qc_and_current_report(dataset, tmp_path):
    p, c = dataset
    a, b, d = ids(p)
    mid = GeochemistryIndex(p).candidates(a, c.id)[0]['id']
    p.view[STATE_KEY] = select_measurement(p, a, c.id, mid)
    p.view[STATE_KEY] = save_group(p, 'Reference', [a, b])
    report = compare_samples(p, [a, b], [d], [c.id])
    result = report['results'][0]
    assert result['a']['n'] == 2  # Three imported rows, two logical samples.
    assert result['a']['mean'] == result['a']['median'] == 3
    assert result['a']['sd'] == pytest.approx(2 ** .5)
    assert result['difference'] == 5
    assert comparison_is_current(p, report)
    p.view['geochemistry_comparison'] = report
    path = tmp_path / 'samples.apyro'
    p.save(path)
    loaded = Project.load(path)
    assert comparison_is_current(loaded, loaded.view['geochemistry_comparison'])
    assert loaded.view[STATE_KEY] == p.view[STATE_KEY]
    p.view['qc_exclusions'] = {'statistics': {mid: {'revision': 1}}}
    assert not comparison_is_current(p, report)
    result = compare_samples(p, [a, b], [d], [c.id])['results'][0]
    assert result['a']['mean'] == 4
    assert result['a']['excluded'] == {'qc_excluded': 1}


def test_clr_rejects_mixed_known_compositions_and_standalone_ratios():
    columns = [Column(name, 'number', unit='% (a/(a+b))') for name in (
        'Arom-GCMS / 27MAS[h]', 'Arom-GCMS / 28MAS[h]', 'Arom-GCMS / 26TAS[h]',
        'Sat-GCMS / H32 22S/R[h]', 'Arom-GCMS / 29MAS[a]',
    )]
    project = Project(columns=columns, rows=[
        Row({column.id: 20 + i * j for j, column in enumerate(columns, 1)}) for i in range(3)
    ])
    a, b, c = GeochemistryIndex(project).samples
    for other, reason in ((2, 'одного известного'), (3, 'Отдельные отношения'), (4, 'аналитическая основа')):
        report = compare_samples(project, [a, b], [c], [columns[0].id, columns[other].id])
        with pytest.raises(ValueError, match=reason):
            comparison_pca(report, preprocessing='clr')


def test_comparison_rejects_overlaps_duplicates_derived_and_mixed_material(dataset):
    p, c = dataset
    a, b, d = ids(p)
    for group_a, group_b in [([a], [a]), ([a, a], [b]), ([], [b]), ([a], ['unknown'])]:
        with pytest.raises(ValueError):
            compare_samples(p, group_a, group_b, [c.id])
    p.rows[-1].source['preparation'] = 'extract'
    d = ids(p)[-1]
    with pytest.raises(ValueError, match='Подтвердите'):
        compare_samples(p, [b], [d], [c.id])
    assert compare_samples(p, [b], [d], [c.id], allow_mixed_material=True)['results'][0]['b']['n'] == 1
    c.metadata['calculation'] = {'expression': '1 + 2'}
    with pytest.raises(ValueError, match='исходные'):
        compare_samples(p, [a], [b], [c.id])


def test_unknown_units_and_conflicting_basis_excluded(dataset):
    p, c = dataset
    a, b, d = ids(p)
    record = p.science.current(p.rows[-1].id, c.id)
    record.unit = 'unknown'
    assert GeochemistryIndex(p).value(d, c.id)['status'] == 'unit_mismatch'
    record.unit = c.unit
    record.basis = 'dry'
    p.science.current(p.rows[2].id, c.id).basis = 'wet'
    result = compare_samples(p, [b], [d], [c.id])['results'][0]
    assert result['a']['excluded'] == result['b']['excluded'] == {'basis_mismatch': 1}


def test_material_relationships_are_explicit_acyclic_and_prevent_merge(dataset):
    p, _ = dataset
    a, b, c = ids(p)
    p.view[STATE_KEY] = relate_materials(p, a, b, 'extract')
    p.view[STATE_KEY] = relate_materials(p, b, c, 'residue')
    with pytest.raises(ValueError, match='цикл'):
        relate_materials(p, c, a, 'extract')
    with pytest.raises(ValueError, match='уже задана'):
        relate_materials(p, a, c, 'extract')
    with pytest.raises(ValueError, match='Сначала'):
        link_samples(p, [a, b], 'bad')
    assert len(GeochemistryIndex(p).passport(b)['relationships']) == 2


def repeated_project():
    fields = [{'source_field_id': 11, 'group_name': 'Элементный анализ',
               'source_label': 'C', 'source_dtype': 'float', 'unit_label': '%wt'}]
    records = [{'source_record_id': 1, 'sample_id': 100, 'sample_name': 'Core', 'sample_type': 'rock'}]
    results = [{'source_record_id': 1, 'source_field_id': 11, 'group_name': 'Элементный анализ',
                'analysis_run_id': run, 'repeat_number': repeat, 'value_float': value}
               for run, repeat, value in [(21, 1, 80), (22, 2, 82)]]
    source = {'repeat_number': 1, 'endpoint': {'host': 'db', 'database': 'geo'}}
    return records, results, fields, source


def test_repeat_import_preserves_all_events_and_selected_projection(tmp_path):
    records, results, fields, source = repeated_project()
    p = project_with_repeats(records, results, fields, source=source)
    column = next(c for c in p.columns if c.metadata.get('geochem_field'))
    sid = ids(p)[0]
    index = GeochemistryIndex(p)
    candidates = index.candidates(sid, column.id)
    assert sorted((r['repeat_number'], r['raw']) for r in candidates) == [(1, 80), (2, 82)]
    assert index.value(sid, column.id)['value'] == 80
    assert p.rows[0].values[column.id] == 80
    assert {r['analysis_run_id'] for r in candidates} == {21, 22}
    assert 'alkane_indices' in {item['method_id'] for item in index.passport(sid)['availability']}
    p.save(tmp_path / 'repeats.apyro')
    loaded = Project.load(tmp_path / 'repeats.apyro')
    assert GeochemistryIndex(loaded).candidates(sid, column.id) == candidates
    assert 'password' not in json.dumps(loaded.source)


def test_repeat_import_does_not_fallback_and_rejects_duplicates():
    records, results, fields, source = repeated_project()
    source['repeat_number'] = 3
    p = project_with_repeats(records, results, fields, source=source)
    column = next(c for c in p.columns if c.metadata.get('geochem_field'))
    assert GeochemistryIndex(p).value(ids(p)[0], column.id)['status'] == 'missing'
    assert len(GeochemistryIndex(p).candidates(ids(p)[0], column.id)) == 2
    with pytest.raises(ValueError, match='Повторяющийся'):
        project_with_repeats(records, results + results[:1], fields, source=source)


def test_passport_and_comparison_are_read_only(dataset):
    p, c = dataset
    before = deepcopy(p)
    index = GeochemistryIndex(p)
    index.passport(ids(p)[0])
    compare_samples(p, [ids(p)[1]], [ids(p)[2]], [c.id])
    assert p == before


def test_passport_readiness_and_supported_unit_conversion():
    from autopyro.nonpyro import UNITS
    columns = [Column(f'Elem / {label} [%wt]', 'number', unit='%wt', metadata={
        'geochem_field': {'id': i, 'group': 'Elem', 'label': f'{label} [%wt]'}})
        for i, label in enumerate(('C', 'H', 'O'))]
    row = Row({column.id: value for column, value in zip(columns, (80, 8, 8))})
    p = Project(columns=columns, rows=[row])
    p.sync_measurements()
    sid = ids(p)[0]
    elemental = lambda: next(item for item in GeochemistryIndex(p).passport(sid)['availability']
                             if item['method_id'] == 'elemental')
    assert elemental()['status'] == 'inputs_ready'
    record = p.science.current(row.id, columns[0].id)
    units = UNITS['mass']
    alternative = next(unit for unit in units if unit != '%wt')
    record.unit = alternative
    assert GeochemistryIndex(p).value(sid, columns[0].id)['value'] == pytest.approx(
        80 * units[alternative] / units['%wt'])
    record.unit = 'unknown'
    assert elemental()['roles']['c']['status'] == 'unit_mismatch'


def test_multi_profile_read_union_repeats_and_refresh_descriptor(monkeypatch):
    from types import SimpleNamespace
    from autopyro.database import geochem, sources
    fields = [
        {'source_field_id': 11, 'group_name': 'Elem', 'source_label': 'C [%wt]',
         'unit_label': '%wt', 'source_dtype': 'float'},
        {'source_field_id': 12, 'group_name': 'VR', 'source_label': 'Indig pop Mean [%Ro]',
         'unit_label': '%Ro', 'source_dtype': 'float'},
    ]
    records = [{'source_record_id': 1, 'sample_id': 7, 'import_batch_id': 9, 'sample_name': 'Core'}]
    results = [{'source_record_id': 1, 'analysis_run_id': 100 + i,
                'group_name': field['group_name'], 'repeat_number': repeat,
                'source_field_id': field['source_field_id'], 'value_float': 2 + i}
               for i, (field, repeat) in enumerate([(fields[0], 1), (fields[0], 2), (fields[1], 1)])]
    queries = []
    class Connection:
        info = SimpleNamespace(host='db', port=5432, dbname='geo')
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def execute(self, sql, parameters=None):
            queries.append((sql, parameters))
            assert sql.lstrip().startswith(('SELECT', 'SET LOCAL', 'SHOW'))
            if 'FROM geochem.source_field' in sql:
                rows = fields
            elif 'FROM geochem.source_record' in sql:
                rows = records
                assert parameters[-3] is True
            elif 'FROM geochem.analysis_run a JOIN' in sql:
                rows = results
                assert parameters[1] is True
                assert set(parameters[-1]) == {11, 12}
            elif 'FROM geochem.import_batch' in sql:
                rows = []
            else:
                rows = []
            return SimpleNamespace(fetchall=lambda: rows, fetchone=lambda: {'transaction_read_only': 'on'})
    monkeypatch.setattr(geochem, 'connect', lambda parameters: Connection())
    p = geochem.read_profiles({'password': 'do-not-store'}, ['elemental', 'vitrinite'], include_repeats=True)
    assert p.source['method_profiles'] == ['elemental', 'vitrinite']
    assert p.source['include_repeats'] is True
    assert len([c for c in p.columns if c.metadata.get('geochem_field')]) == 2
    assert len([m for m in p.science.measurements.values()
                if m.property_id in {c.id for c in p.columns if c.metadata.get('geochem_field')}]) == 3
    descriptor = sources.descriptor(p.source)
    assert descriptor['include_repeats'] is True
    assert 'do-not-store' not in json.dumps(p.source)
    calls = []
    monkeypatch.setattr(geochem, 'read_profile', lambda params, method, **options:
                        calls.append((params, method, options)))
    sources.read_snapshot(descriptor, {'password': 'temporary'})
    assert calls[0][2]['method_ids'] == ['elemental', 'vitrinite']
    assert calls[0][2]['include_repeats'] is True
    assert 'password' not in json.dumps(descriptor)


def test_invalid_saved_links_can_be_reported_without_silent_partial_merge(dataset):
    p, _ = dataset
    p.view[STATE_KEY] = {'links': [{'id': 'bad', 'name': 'bad', 'row_ids': ['missing']}]}
    index = GeochemistryIndex(p)
    assert index.invalid_links == p.view[STATE_KEY]['links']
    assert set(index.samples) == set(GeochemistryIndex(Project(columns=p.columns, rows=p.rows, source=p.source)).samples)


def test_legacy_comparison_digest_survives_unrelated_group(dataset):
    from autopyro.analysis.geochemistry_workspace import _hash, freeze_comparison

    project, column = dataset
    a, b, c = ids(project)
    report = compare_samples(project, [b], [c], [column.id])
    report['input_digest'] = _hash(report['inputs'])
    snapshot = freeze_comparison(report)
    project.view[STATE_KEY] = save_group(project, 'Unrelated', [a])
    assert comparison_is_current(project, report)
    assert comparison_is_current(project, snapshot)
    project.rows[-1].values[column.id] = 42
    project.sync_measurements()
    assert not comparison_is_current(project, report)
    assert not comparison_is_current(project, snapshot)
