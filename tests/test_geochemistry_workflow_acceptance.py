"""Linked analytical records, well validation and offline model transfer."""
from copy import deepcopy
import json

import pytest

from autopyro.analysis.geochemistry_workspace import (
    MODEL_RUNS_KEY, GeochemistryIndex, apply_prediction, archive_model_run,
    compare_alkane_profiles, compare_samples, comparison_is_current,
    comparison_clustering, comparison_dbscan, comparison_kmeans, comparison_oil_mixing, comparison_pca,
    comparison_prediction, correlate_oil_profiles, correlation_sensitivity, freeze_comparison, install_model_run,
    replay_model_run, verify_model_run_package,
)
from autopyro.model.core import Column, Project, Row
from autopyro.analysis.nonpyro import check_saved_reports, evaluate_project, replay_report, sara_ternary_points, suggest_bindings
from autopyro.database.geochem import append_geochemistry_snapshot, profile_fields, project_from_records, project_with_repeats
from autopyro.data_io.io import build_project


def test_database_append_preserves_measurements_and_profile_qc_replays():
    group = 'Sat-GCMS'
    method = 'c27_diasterane_ratio'
    fields = [
        {'source_field_id': i, 'group_name': group, 'source_label': f'{peak}[h] [count]',
         'unit_label': 'count', 'source_dtype': 'Float64'}
        for i, peak in enumerate(('St27dbS', 'St27aaR'), 1)
    ]

    def snapshot(sample, repeats):
        observations = [
            {'source_record_id': sample, 'source_field_id': i, 'analysis_run_id': 100 * sample + repeat,
             'repeat_number': repeat, 'group_name': group, 'value_float': value}
            for repeat in range(1, repeats + 1)
            for i, value in enumerate((12, 4), 1)
        ]
        return project_with_repeats([{'source_record_id': sample, 'sample_name': str(sample), 'sample_id': sample}],
                                    observations, fields, source={'endpoint': {'host': 'fixture'}})

    original, incoming = snapshot(1, 1), snapshot(2, 2)
    project = append_geochemistry_snapshot(original, incoming)
    assert len(project.columns) == len(original.columns)
    assert len(project.rows) == 2 and len(original.rows) == 1
    config = {'method': method, 'version': 1, 'confirmed': True, 'context': 'fixture', 'variant': 'h',
              'inputs': suggest_bindings(project, method, group, 'h')}
    column_id = config['inputs']['dia']['column_id']
    index = GeochemistryIndex(project)
    a, b = index.samples
    assert len(index.candidates(b, column_id)) == 2
    measurement = index.value(a, column_id)['measurement']
    project.view['qc_exclusions'] = {'statistics': {
        measurement['id']: {'revision': measurement['revision'], 'comment': 'Peak interference'},
    }}
    comparison = compare_samples(project, [a], [b], [column_id])
    point = comparison['inputs']['points'][column_id][a]
    assert point['status'] == 'qc_excluded'
    assert point['measurement']['id'] == measurement['id']
    assert comparison['qc_exclusions'][measurement['id']] == {
        'revision': measurement['revision'], 'comment': 'Peak interference',
    }
    assert freeze_comparison(comparison)['qc_exclusions'] == comparison['qc_exclusions']
    passport = GeochemistryIndex(project).passport(a)
    excluded, = [record for record in passport['measurements'] if 'statistics_exclusion' in record]
    assert excluded['id'] == measurement['id']
    assert excluded['comparison_status'] == 'qc_excluded'
    assert excluded['statistics_exclusion']['comment'] == 'Peak interference'
    report = evaluate_project(project, [project.rows[0].id], config)
    assert report['results'][0]['status'] == 'qc_excluded'
    assert report['inputs'][0]['measurements'][column_id]['revision'] == measurement['revision']
    assert report['inputs'][0]['qc_exclusions'][column_id]['comment'] == 'Peak interference'
    replayed = replay_report(json.loads(json.dumps(report)))
    assert replayed['results'] == report['results']
    assert replayed['inputs'] == report['inputs']
    old_profile, old_comparison = deepcopy(report), deepcopy(comparison)
    old_profile['inputs'][0]['qc_exclusions'][column_id].pop('comment')
    old_comparison.pop('qc_exclusions')
    project.view['qc_exclusions']['statistics'][measurement['id']]['comment'] = 'Revised explanation'
    check_saved_reports(project, [old_profile, report])
    assert not old_profile['stale'] and not report['stale']
    assert comparison_is_current(project, old_comparison)
    assert comparison_is_current(project, comparison)
    assert comparison['qc_exclusions'][measurement['id']]['comment'] == 'Peak interference'

def test_published_lab_oil_blend_reports_mass_balance_disagreement():
    # Energies 2023, 16, 7387, Tables 3 and 5 and Figure 5.
    # https://doi.org/10.3390/en16217387
    # Convert the printed whole-oil mass percentages to mg/g oil.
    headers = ['Sample / Name', 'Sample / Type', 'Saturates', 'Aromatics']
    project = build_project(
        headers,
        [('Blend A', 'oil', 589.9, 164.0), ('Blend B', 'oil', 637.1, 205.6),
         ('AB 50/50 before deposition test', 'oil', 633.0, 183.5)],
        [2, 3, 4],
        [(i, name, 'number' if i >= 2 else 'text', 'mg/g oil' if i >= 2 else '')
         for i, name in enumerate(headers)],
        {'citation': 'doi:10.3390/en16217387, Tables 3 and 5'},
    )
    a, b, blend = GeochemistryIndex(project).samples
    fields = [column.id for column in project.columns[2:]]
    comparison = compare_samples(project, [a, b], [blend], fields)
    # Test tolerance of one mass percentage point per fraction, not lab uncertainty.
    run = comparison_oil_mixing(comparison, dict.fromkeys(fields, 10))
    # Separate equal-weight least-squares calculation from the published values.
    assert run['fraction_b'] == pytest.approx(0.71885610347615, abs=1e-12, rel=0)
    assert not run['outside_interval']
    assert [marker['residual'] for marker in run['markers']] == pytest.approx(
        [9.16999191593, -10.40441390461], abs=1e-9, rel=0,
    )
    assert [marker['within_tolerance'] for marker in run['markers']] == [True, False]
    assert run['status'] == 'inconsistent'

    record = archive_model_run(project, comparison, run)
    snapshot = project.view[MODEL_RUNS_KEY]['snapshots'][record['snapshot_id']]
    record, snapshot = json.loads(json.dumps([record, snapshot]))
    assert replay_model_run(record, snapshot)


def test_published_bulla_siyazan_lab_blend_recovers_mass_fraction():
    # Wang et al. (2025), Tables 1–2, https://doi.org/10.3390/pr13051500.
    # Published bulk composition measurements, converted from mass percent to mg/g.
    headers = ['Sample / Name', 'Sample / Type', 'Paraffins', 'Asphaltenes']
    project = build_project(
        headers,
        [('Siyazan', 'oil', 11.8, 8.1), ('Bulla', 'oil', 133.4, 2.3),
         ('5 mass percent Bulla', 'oil', 17.5, 7.8)],
        [2, 3, 4],
        [(i, name, 'number' if i >= 2 else 'text', 'mg/g oil' if i >= 2 else '')
         for i, name in enumerate(headers)],
        {'citation': 'doi:10.3390/pr13051500, Tables 1 and 2'},
    )
    siyazan, bulla, blend = GeochemistryIndex(project).samples
    fields = [column.id for column in project.columns[2:]]
    comparison = compare_samples(project, [siyazan, bulla], [blend], fields)
    # Regression tolerances, not uncertainty estimates from the paper.
    run = comparison_oil_mixing(comparison, dict.fromkeys(fields, 1))
    assert run['status'] == 'consistent'
    assert run['fraction_b'] == pytest.approx(0.05, abs=0.005, rel=0)
    assert run['fraction_b'] == pytest.approx(0.04688600693648, abs=1e-12, rel=0)
    assert all(marker['used_for_fit'] and marker['within_tolerance'] for marker in run['markers'])

    record = archive_model_run(project, comparison, run)
    snapshot = project.view[MODEL_RUNS_KEY]['snapshots'][record['snapshot_id']]
    record, snapshot = json.loads(json.dumps([record, snapshot]))
    assert replay_model_run(record, snapshot)


def test_published_gammacerane_alone_cannot_identify_binary_oil_fraction():
    # Zhao et al. (2010), Table 2. The test uses field oils and two candidate
    # endpoint pairs, with gammacerane as the sole quantified marker.
    # https://doi.org/10.11781/sysydz201002196
    headers = ['Sample / Name', 'Sample / Type', 'Gammacerane']
    project = build_project(
        headers,
        [('CFD12-1S-2', 'oil', 96), ('LD10-1-2', 'oil', 399),
         ('QHD35-4-1', 'oil', 146), ('BZ19-4N-2D', 'oil', 453), ('BZ19-4N-1', 'oil', 333)],
        range(2, 7),
        [(0, headers[0], 'text', ''), (1, headers[1], 'text', ''), (2, headers[2], 'number', 'µg/g oil')],
        {'citation': 'doi:10.11781/sysydz201002196, Table 2'},
    )
    a, b, degraded_a, degraded_b, mixture = GeochemistryIndex(project).samples
    field_id = project.columns[2].id
    for endpoints, values in (
        ([a, b], [96, 399, 333]),
        ([degraded_a, degraded_b], [146, 453, 333]),
    ):
        comparison = compare_samples(project, endpoints, [mixture], [field_id])
        run = comparison_oil_mixing(comparison, {field_id: 1})
        assert run['status'] == 'underdetermined'
        assert run['fraction_b'] is None and run['unbounded_fraction_b'] is None
        marker, = run['markers']
        assert marker['values'] == values
        assert marker['statuses'] == ['numeric'] * 3 and marker['used_for_fit']
        assert marker['residual'] is None


def test_usgs_published_gas_ratios_and_detection_limit():
    # USGS SIR 2018–5097, Table 5 (mole percent) and Table 7 (integer ratios).
    # https://pubs.usgs.gov/sir/2018/5097/sir20185097.pdf
    published = [
        ('SU-23', '73.21', '12.41', '2.49', 5),
        ('ST-141', '92.68', '0.0571', '0.0001', 1620),
        ('PE-59', '64.41', '0.266', '0.0372', 212),
        ('MT-1286', '5.46', '0.154', '0.0414', 28),
        ('HY-53', '93.31', '0.0042', '0.0002', 21207),
        ('AB-142', '71.79', '0.0116', '<0.0001', 6189),
    ]
    headers = ['Well / Well Name', 'Gas / C1[mol]', 'Gas / C2[mol]', 'Gas / C3[mol]']
    project = build_project(headers, [row[:4] for row in published], range(2, 8),
                            [(i, name, 'number' if i else 'text', '%mol' if i else '')
                             for i, name in enumerate(headers)],
                            {'citation': 'USGS SIR 2018–5097, Tables 5 and 7'})
    config = {'method': 'gas_ratios', 'version': 1, 'confirmed': True,
              'context': 'USGS SIR 2018–5097, headspace gas in molar percent',
              'inputs': {f'c{i}': {'column_id': project.columns[i].id, 'unit': '%mol'} for i in range(1, 4)}}
    report = evaluate_project(project, [row.id for row in project.rows], config)
    for expected, result in zip(published[:-1], report['results'][:-1]):
        assert result['status'] == 'profile', expected[0]
        assert result['values']['Bernard'] == pytest.approx(expected[4], abs=0.5), expected[0]
    limited = report['results'][-1]
    assert limited['status'] == 'censored'
    assert 'Bernard' not in limited['values']
    assert replay_report(json.loads(json.dumps(report)))['results'] == report['results']


def test_published_vitrinite_reflectance_requires_population_confirmation():
    # Liu et al., Table 1, https://www.osti.gov/servlets/purl/1801745
    # Maturity labels below follow the application's BGS-based bands.
    published = [
        ('NAS-2', '0.55', '94', 'Переход начала нефтеобразования'),
        ('Ohio-2', '0.62', '64', 'Ориентировочный нефтяной диапазон'),
    ]
    headers = ['Sample', 'VR / User pop Mean [%]', 'VR / User pop Num [Euc]']
    units = ['', '%Ro', 'Euc']
    project = build_project(
        headers, [row[:3] for row in published], range(2, 4),
        [(i, name, 'number' if i else 'text', units[i]) for i, name in enumerate(headers)],
        {'citation': 'Liu et al., Assessing the thermal maturity of black shales, Table 1'},
    )
    config = {
        'method': 'vitrinite', 'version': 1, 'confirmed': True,
        'context': 'Mean random reflectance of dispersed vitrinite, Liu et al., Table 1',
        'inputs': {
            role: {'column_id': project.columns[i].id, 'unit': units[i]}
            for i, role in enumerate(('ro', 'n'), start=1)
        },
    }
    row_ids = [row.id for row in project.rows]
    unconfirmed = evaluate_project(project, row_ids, config)
    confirmed = evaluate_project(project, row_ids, dict(config, population_confirmed=True))
    for (sample, ro, _, band), before, after in zip(
        published, unconfirmed['results'], confirmed['results'], strict=True
    ):
        assert before['status'] == 'not_applicable', sample
        assert after['status'] == 'interpreted', sample
        assert before['values'] == after['values'] == {'Ro': float(ro)}, sample
        assert after['interpretation'].startswith(band), sample
        assert all('40–50' not in warning for warning in after['warnings']), sample
    assert replay_report(json.loads(json.dumps(confirmed)))['results'] == confirmed['results']


@pytest.mark.parametrize('sample,sat,arom,published,precision,citation', [
    ('Valemon MPSR#166', -29.56, -27.62, 1.82, 0.01,
     'Geolab Nor, Valemon 34/11-5 S, Table 10, https://factpages.sodir.no/pbl/geochemical_pdfs/5248_2.pdf'),
    ('Dl-25', -27.3, -26.9, -2.34, 0.1, 'Kotarba et al. (2021), Table 4, doi:10.1007/s00531-021-02035-7'),
    ('So-9', -27.6, -26.5, -0.61, 0.1, 'Kotarba et al. (2021), Table 4, doi:10.1007/s00531-021-02035-7'),
    ('El-58', -28.7, -27.4, 0.10, 0.1, 'Kotarba et al. (2021), Table 4, doi:10.1007/s00531-021-02035-7'),
])
def test_published_sofer_cv_respects_printed_precision(sample, sat, arom, published, precision, citation):
    # This is a numerical reference. The scale and C15+ declarations in the
    # test configuration are analyst inputs, not inferred from the table.
    headers = ['Sample / Name', 'd13C saturated', 'd13C aromatic', 'Published CV']
    units = ('', 'per mil', 'per mil', 'Euc')
    project = build_project(headers, [(sample, sat, arom, published)], [2],
                            [(i, name, 'number' if i else 'text', units[i]) for i, name in enumerate(headers)],
                            {'citation': citation})
    config = {'method': 'sofer', 'version': 1, 'confirmed': True, 'context': 'Published numerical reference',
              'material': 'oil_C15plus', 'isotope_reference': 'VPDB',
              'inputs': {role: {'column_id': project.columns[i].id, 'unit': units[i]}
                         for i, role in enumerate(('sat', 'arom', 'cv_reported'), 1)}}
    # Propagate the maximum half-step rounding of the two printed isotope values
    # and the independently rounded CV. This is not analytical uncertainty.
    tolerance = (2.53 + 2.22) * precision / 2 + 0.005
    report = evaluate_project(project, [project.rows[0].id], config)
    result = report['results'][0]
    assert result['status'] == 'interpreted'
    assert result['values']['CV'] == pytest.approx(published, abs=tolerance)
    assert replay_report(json.loads(json.dumps(report)))['results'] == report['results']


@pytest.mark.parametrize('basis,table,ratios,peaks', [
    ('a', '9Ba', (1.70, 0.59, 0.40), (4742975, 2793552, 8028737, 7021832)),
    ('h', '9Bb', (1.59, 0.64, 0.46), (661034, 416706, 1026353, 905182)),
])
def test_valemon_gc_profile_preserves_published_ratios_by_basis(basis, table, ratios, peaks):
    # Valemon 34/11-5 S, sample 142/0001-0, Tables 9Aa/9Ab and 9Ba/9Bb.
    # https://factpages.sodir.no/pbl/geochemical_pdfs/5248_2.pdf
    pristane, phytane, nc17, nc18 = peaks
    assert ratios == pytest.approx(
        (pristane / phytane, pristane / nc17, phytane / nc18), abs=0.005
    )

    headers = [f'Sat-GC / {name}[{basis}] [Euc]' for name in ('Pr/Ph', 'Pr/nC17', 'Ph/nC18')]
    project = build_project(
        headers, [list(map(str, ratios))], [2],
        [(i, name, 'number', '1') for i, name in enumerate(headers)],
        {'citation': f'Geolab Nor, Valemon 142/0001-0, Table {table}'},
    )
    config = {
        'method': 'gc_profile', 'version': 1, 'confirmed': True,
        'context': 'Valemon saturated fraction, sample 142/0001-0',
        'inputs': {
            role: {'column_id': column.id, 'unit': '1'}
            for role, column in zip(('prph', 'pr17', 'ph18'), project.columns, strict=True)
        },
    }
    report = evaluate_project(project, [project.rows[0].id], config)
    result, = report['results']
    assert result['status'] == 'profile'
    assert result['values'] == dict(zip(('PrPh', 'Pr17', 'Ph18'), ratios, strict=True))
    assert replay_report(json.loads(json.dumps(report)))['results'] == report['results']


def test_published_extract_yields_normalize_to_original_organic_carbon():
    # Huai Hin Lat Formation, Table 3. Reference ratios are calculated from the printed inputs.
    # https://doi.org/10.3389/feart.2022.1085869
    published = [
        ('A1', '160', '2.1', 7.61905),
        ('A2', '120', '0.91', 13.18681),
        ('A3', '2939', '18.41', 15.96415),
        ('A4', '4127', '20.17', 20.46108),
    ]
    headers = ['Sample', 'Extract / Extr yld [ppm (µg/g)]', 'Elem / TOC [%wt]']
    units = ['', 'ppm (µg/g)', '%wt']
    project = build_project(
        headers, [row[:3] for row in published], range(2, 6),
        [(i, name, 'number' if i else 'text', units[i]) for i, name in enumerate(headers)],
        {'citation': 'doi:10.3389/feart.2022.1085869, Table 3'},
    )
    config = {
        'method': 'extract_toc', 'version': 1, 'confirmed': True,
        'context': 'Huai Hin Lat mudstones, solvent extract and independently measured original TOC',
        'inputs': {
            role: {'column_id': project.columns[i].id, 'unit': units[i]}
            for i, role in enumerate(('extract', 'toc'), start=1)
        },
    }
    report = evaluate_project(project, [row.id for row in project.rows], config)
    for (sample, _, _, expected), result in zip(published, report['results'], strict=True):
        assert result['status'] == 'profile', sample
        assert result['values'] == {'EOM_TOC': pytest.approx(expected, abs=5e-6, rel=0)}, sample
    assert replay_report(json.loads(json.dumps(report)))['results'] == report['results']


def test_usgs_microscopy_preserves_maceral_balance_by_counting_basis():
    # USGS OFR 97-137A, Table 8, area percentages from point counts.
    # https://pubs.usgs.gov/of/1997/0137a/report.pdf
    published = [
        ('K-1, without minerals or mixtinite', '81.6', '2.9', '15.4', 99.9, 'vit', False),
        ('K-2, without minerals or mixtinite', '22.1', '2.5', '75.4', 100.0, 'inert', False),
        ('K-1, without minerals', '75.1', '2.7', '14.2', 92.0, 'vit', True),
        ('K-2, without minerals', '17.8', '2.0', '60.8', 80.6, 'inert', True),
    ]
    headers = ['Sample', 'VisK-TL / Vit [%area]', 'VisK-TL / Lipt [%area]', 'VisK-TL / Inert [%area]']
    project = build_project(
        headers, [row[:4] for row in published], range(2, 6),
        [(i, name, 'number' if i else 'text', '%area' if i else '') for i, name in enumerate(headers)],
        {'citation': 'USGS OFR 97-137A, Table 8'},
    )
    config = {
        'method': 'microscopy', 'version': 1, 'confirmed': True,
        'context': 'Kyrgyz coal point counts, counting basis identified for each row',
        'inputs': {
            role: {'column_id': column.id, 'unit': '%area'}
            for role, column in zip(('vit', 'lipt', 'inert'), project.columns[1:], strict=True)
        },
    }
    report = evaluate_project(project, [row.id for row in project.rows], config)
    for expected, result in zip(published, report['results'], strict=True):
        sample, vit, lipt, inert, total, dominant, warns = expected
        assert result['status'] == 'interpreted', sample
        assert result['values'] == pytest.approx({
            'Vit': float(vit), 'Lipt': float(lipt), 'Inert': float(inert), 'total': total,
        }, abs=1e-12, rel=0), sample
        assert result['interpretation'].endswith(dominant), sample
        assert any('Сумма трёх макрогрупп' in warning for warning in result['warnings']) is warns, sample
    assert replay_report(json.loads(json.dumps(report)))['results'] == report['results']


def test_published_stereo_profile_converts_percentages_to_fractions():
    # Al-Arouri (1996), Tables 5.1 and 5.3(a), Snake Creek Mudstone.
    # https://digital.library.adelaide.edu.au/dspace/bitstream/2440/18891/8/02whole.pdf
    published = [
        ('35 Flinton-1', '37.3', '39', '60.3'),
        ('41 Glenhaughton-1', '49.5', '43', '62.5'),
        ('45 Cabawin-1', '40.3', '45', '58.5'),
    ]
    unit = '% (a/(a+b))'
    headers = ['Sample'] + [
        f'Sat-GCMS / {name}[un] [{unit}]'
        for name in ('St29 20S/R', 'St29 bb/aa', 'H31 22S/R')
    ]
    project = build_project(
        headers, published, range(2, 5),
        [(i, name, 'number' if i else 'text', unit if i else '') for i, name in enumerate(headers)],
        {'citation': 'Al-Arouri (1996), Tables 5.1 and 5.3(a)'},
    )
    config = {
        'method': 'stereo_profile', 'version': 1, 'confirmed': True,
        'context': 'Snake Creek Mudstone, published isomer percentages',
        'inputs': {
            role: {'column_id': column.id, 'unit': unit}
            for role, column in zip(('st20', 'stbb', 'h31'), project.columns[1:], strict=True)
        },
    }
    report = evaluate_project(project, [row.id for row in project.rows], config)
    for (sample, st20, stbb, h31), result in zip(published, report['results'], strict=True):
        assert result['status'] == 'profile', sample
        assert result['values'] == pytest.approx({
            'St20S': float(st20) / 100, 'StBB': float(stbb) / 100, 'H31S': float(h31) / 100,
        }, abs=1e-12, rel=0), sample
    assert replay_report(json.loads(json.dumps(report)))['results'] == report['results']


def test_published_aromatic_profile_preserves_indices_with_unspecified_basis():
    # Yang et al. (2025), Table 5. The table does not specify peak areas or heights.
    # https://www.cup.edu.cn/geosci/docs/2025-09/2b0e9037eb1a43989691e4d4c59e7b96.pdf
    published = [
        ('Q1–1', '0.61', '0.02'),
        ('Q1–11', '0.96', '0.20'),
        ('Q1–14', '0.86', '0.53'),
    ]
    headers = ['Sample', 'Arom-GCMS / MPI-1[un] [Euc]', 'Arom-GCMS / DBT/PHEN[un] [Euc]']
    project = build_project(
        headers, published, range(2, 5),
        [(0, headers[0], 'text', ''), (1, headers[1], 'number', '1'), (2, headers[2], 'number', '1')],
        {'citation': 'Yang et al. (2025), International Journal of Coal Geology 307, 104827, Table 5'},
    )
    config = {
        'method': 'aromatic_profile', 'version': 1, 'confirmed': True,
        'context': 'Qianjiang Formation, published GC–MS ratios',
        'inputs': {
            'mpi': {'column_id': project.columns[1].id, 'unit': '1'},
            'dbt': {'column_id': project.columns[2].id, 'unit': '1'},
        },
    }
    report = evaluate_project(project, [row.id for row in project.rows], config)
    for (sample, mpi, dbt), result in zip(published, report['results'], strict=True):
        assert result['status'] == 'profile', sample
        assert result['values'] == {'MPI1': float(mpi), 'DBTPhen': float(dbt)}, sample
        assert any('Основа измерения [un] не определена' in warning for warning in result['warnings']), sample
    assert replay_report(json.loads(json.dumps(report)))['results'] == report['results']


VALEMON_AREAS = (3040851, 2946384, 2211753, 1794788, 1414314, 1182227,
                 915492, 723006, 563522, 518453, 432218)
T92_0143_AREAS = (1496281, 1301564, 1025254, 960221, 673461, 634097,
                  576066, 409894, 302250, 405629, 434516)


def test_valemon_cpi_and_oep_from_saturated_fraction_peak_areas():
    # Geolab Nor, Valemon 34/11-5 S, sample 142/0001-0 (MPSR#166).
    # Table 9Aa lists peak areas in carbon-number order. Table 9Ba reports CPI 1.09.
    # https://factpages.sodir.no/pbl/geochemical_pdfs/5248_2.pdf
    headers = [f'Sat-GC / nC{carbon}[a]' for carbon in range(24, 35)]
    project = build_project(headers, [VALEMON_AREAS], [2],
                            [(i, name, 'number', 'count') for i, name in enumerate(headers)],
                            {'citation': 'Valemon 34/11-5 S, Tables 9Aa and 9Ba'})
    bindings = suggest_bindings(project, 'alkane_indices', 'Sat-GC', 'a')
    assert len(bindings) == 11
    config = {'method': 'alkane_indices', 'version': 1, 'confirmed': True,
              'context': 'Published saturated fraction peak areas, MPSR#166', 'inputs': bindings}
    report = evaluate_project(project, [project.rows[0].id], config)
    result = report['results'][0]
    assert result['status'] == 'profile'
    assert result['values']['CPI'] == pytest.approx(1.09, abs=0.005)
    # Separate reference arithmetic from the published USGS OA6 equation:
    # on the five printed C25–C29 areas the weighted sums are 14897339 and 14504268.
    # Valemon does not report OEP itself.
    assert result['values']['OEP27'] == pytest.approx(14897339 / 14504268, rel=1e-12)
    assert replay_report(json.loads(json.dumps(report)))['results'] == report['results']


def test_valemon_c32_epimer_fraction_from_database_fields(qtbot):
    from autopyro.ui.window import MainWindow

    # Geolab Nor, Valemon oil 142/0001-0, Tables 11a and 11i (Ratio 14).
    # https://factpages.sodir.no/pbl/geochemical_pdfs/5248_2.pdf
    method = 'homohopane_epimers'
    fields = [
        {'source_field_id': i, 'group_name': 'Sat-GCMS', 'source_label': label,
         'unit_label': unit, 'source_dtype': 'Float64'}
        for i, (label, unit) in enumerate((
            ('H32abS[h] [count]', 'count'),
            ('H32abR[h] [count]', 'count'),
            ('H32 22S/R[h] [% (a/(a+b))]', '% (a/(a+b))'),
        ), 1)
    ]
    assert profile_fields({'fields': fields}, method, 'Sat-GCMS', 'h') == fields[:2]
    observations = [
        {'source_record_id': 1, 'source_field_id': i, 'analysis_run_id': 1, 'repeat_number': 1,
         'group_name': 'Sat-GCMS', 'value_float': value}
        for i, value in enumerate((624.6, 185.4, 77.12), 1)
    ]
    project = project_from_records([{'source_record_id': 1, 'sample_name': 'Valemon'}],
                                   observations, fields, source={'citation': 'Valemon, Tables 11a and 11i'})
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_nonpyro('terpanes')
    dialog = owner.geochemistry_windows['terpanes']
    dialog.method.setCurrentIndex(dialog.method.findData(method))
    dialog.form.fields['context'][0].setText('Valemon, GC-MS peak heights')
    dialog.form.fields['confirmed'][0].setChecked(True)
    assert dialog.calculate_button.isEnabled(), dialog.workflow_hint.text()
    dialog.calculate_button.click()
    report = dialog.report
    assert report is not None, dialog.status.text()
    row, = report['results']
    assert row['values'] == pytest.approx({'C32_22S': 0.771111111111})
    # Printed heights and published percentage have separate rounding.
    assert row['values']['C32_22S'] * 100 == pytest.approx(77.12, abs=0.01)
    assert replay_report(json.loads(json.dumps(report)))['results'] == report['results']


def test_valemon_homohopane_distribution_from_database_fields(qtbot):
    from PySide6.QtCore import QItemSelectionModel

    from autopyro.ui.window import MainWindow

    # Geolab Nor, Valemon oil 142/0001-0, Table 11a, m/z 191 peak heights.
    # https://factpages.sodir.no/pbl/geochemical_pdfs/5248_2.pdf
    method = 'homohopane_distribution'
    peaks = {31: (354.3, 243.3), 32: (624.6, 185.4), 33: (528.9, 135),
             34: (380.7, 164), 35: (231.8, 182.6)}
    heights = {f'H{carbon}ab{epimer}': height
               for carbon, pair in peaks.items() for epimer, height in zip(('S', 'R'), pair)}
    heights['H31abS+R'] = sum(peaks[31])
    fields = [
        {'source_field_id': i, 'group_name': 'Sat-GCMS', 'source_label': f'{peak}[h] [count]',
         'unit_label': 'count', 'source_dtype': 'Float64'}
        for i, peak in enumerate(heights, 1)
    ]
    excluded = [
        {'group_name': 'Sat-GCMS', 'source_label': label}
        for label in ('H31ba[h] [count]', 'H31abS[un] [unknown]', '31HomoHop[h] [% (a/(a+b))]')
    ]
    assert profile_fields({'fields': [*fields, *excluded]}, method, 'Sat-GCMS') == fields
    omitted = {1: {'H31abS+R'}, 2: {'H31abR'}, 3: {'H35abR'}, 4: set(), 5: set()}
    names = ('Valemon', 'Valemon with C31 sum', 'Missing epimer', 'Zero sum', 'Synthetic C35 ×2')
    records = [{'source_record_id': sample, 'sample_name': name} for sample, name in enumerate(names, 1)]
    observations = [
        {'source_record_id': sample, 'source_field_id': i, 'analysis_run_id': sample,
         'repeat_number': 1, 'group_name': 'Sat-GCMS',
         'value_float': value * (0 if sample == 4 else 2 if sample == 5 and peak.startswith('H35') else 1)}
        for sample, missing in omitted.items()
        for i, (peak, value) in enumerate(heights.items(), 1) if peak not in missing
    ]
    project = project_from_records(records, observations, fields, source={'citation': 'Valemon, Table 11a'})
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_nonpyro('terpanes')
    dialog = owner.geochemistry_windows['terpanes']
    dialog.method.setCurrentIndex(dialog.method.findData(method))
    dialog.form.fields['context'][0].setText('Valemon, GC-MS peak heights')
    dialog.form.fields['confirmed'][0].setChecked(True)
    assert dialog.calculate_button.isEnabled(), dialog.workflow_hint.text()
    dialog.calculate_button.click()
    report = dialog.report
    assert report is not None, dialog.status.text()
    split, combined, incomplete, zero, _ = report['results']
    # Independently normalized against the total height 3030.6.
    expected = dict(zip(('C31', 'C32', 'C33', 'C34', 'C35'),
                        (19.71886755, 26.72738072, 21.90655316, 17.97333861, 13.67385996)))
    assert split['values'] == pytest.approx(expected)
    assert combined['values'] == pytest.approx(expected)
    assert sum(split['values'].values()) == pytest.approx(100)
    assert incomplete['status'] == 'not_applicable' and not incomplete['values']
    assert zero['status'] == 'zero_denominator'
    assert replay_report(json.loads(json.dumps(report)))['results'] == report['results']

    assert not dialog.homohopane_button.isEnabled()
    selection = dialog.results.selectionModel()
    flags = QItemSelectionModel.SelectionFlag
    for row in (0, 2, 3, 4):
        selection.select(dialog.results.model().index(row, 1), flags.Select | flags.Rows)
    dialog.homohopane_button.click()
    canvas = dialog.homohopane_canvas
    assert dialog.result_tabs.currentWidget() is canvas
    axes = canvas.figure.axes[0]
    assert [tick.get_text() for tick in axes.get_xticklabels()] == list(expected)
    valemon, comparison = axes.lines
    assert list(valemon.get_xdata()) == list(range(31, 36))
    assert list(valemon.get_ydata()) == pytest.approx(list(expected.values()))
    assert list(comparison.get_ydata()) == pytest.approx(
        [100 * signal / 3445 for signal in (597.6, 810, 663.9, 544.7, 828.8)])
    assert [label.get_text() for label in axes.get_legend().get_texts()] == [
        '1. Valemon', '5. Synthetic C35 ×2']
    assert '2 из 4' in dialog.status.text()

    dialog.result_tabs.setCurrentWidget(dialog.results)
    selection.select(dialog.results.model().index(2, 0), flags.ClearAndSelect | flags.Rows)
    dialog.homohopane_button.click()
    assert 'нет полного ряда C31–C35' in dialog.status.text()
    tab = dialog.result_tabs.indexOf(canvas)
    assert not dialog.result_tabs.isTabVisible(tab)
    selection.select(dialog.results.model().index(0, 0), flags.ClearAndSelect | flags.Rows)
    dialog.homohopane_button.click()
    assert dialog.result_tabs.isTabVisible(tab)
    dialog.form.fields['context'][0].setText('Changed context')
    assert not dialog.result_tabs.isTabVisible(tab)


def test_sr1_terpane_hopane_profile_from_database_fields(qtbot):
    from autopyro.ui.window import MainWindow

    # Geolab Nor, Appendix F, SR-1, run MPDMIO00855201, GC-MS peak height data.
    # https://factpages.sodir.no/pbl/geochemical_pdfs/3314_6.pdf
    method = 'terpane_hopane_profile'
    peaks = {'t19': 741666, 't23': 9281260, 'tet24': 1393110, 'Gamm': 0,
             'H30ab': 42681100, 'H27Ts': 6486050, 'H27Tm': 4232460}
    fields = [
        {'source_field_id': i, 'group_name': 'Sat-GCMS', 'source_label': f'{peak}[h] [count]',
         'unit_label': 'count', 'source_dtype': 'Float64'}
        for i, peak in enumerate(peaks, 1)
    ]
    ratio_field = {'source_field_id': 8, 'group_name': 'Sat-GCMS',
                   'source_label': 't19/t23[h] [Euc (a/b)]', 'unit_label': 'Euc (a/b)'}
    unknown_field = {'source_field_id': 9, 'group_name': 'Sat-GCMS',
                     'source_label': 't19[un] [unknown]', 'unit_label': 'unknown'}
    assert profile_fields({'fields': [*fields, ratio_field, unknown_field]}, method, 'Sat-GCMS') == fields
    records = [{'source_record_id': 1, 'sample_name': 'SR-1'},
               {'source_record_id': 2, 'sample_name': 'Partial profile'}]
    observations = [
        {'source_record_id': sample, 'source_field_id': i, 'analysis_run_id': sample,
         'repeat_number': 1, 'group_name': 'Sat-GCMS', 'value_float': value}
        for sample in (1, 2)
        for i, (peak, value) in enumerate(peaks.items(), 1)
        if sample == 1 or peak != 't19'
    ]
    project = project_from_records(records, observations, fields, source={'citation': 'Geolab Nor, SR-1'})
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_nonpyro('terpanes')
    dialog = owner.geochemistry_windows['terpanes']
    dialog.form.fields['context'][0].setText('SR-1, GC-MS peak heights')
    dialog.form.fields['confirmed'][0].setChecked(True)
    assert dialog.calculate_button.isEnabled(), dialog.workflow_hint.text()
    dialog.calculate_button.click()
    report = dialog.report
    assert report is not None, dialog.status.text()
    full, partial = report['results']
    assert full['values'] == pytest.approx({
        'C19_C23': 0.0799100553, 'Tet24_C23': 0.1500992322,
        'Gamm_C30': 0, 'Ts_fraction': 0.6051260856,
    })
    # Published H6, and quotient of independently rounded H12 and H11.
    assert round(full['values']['Ts_fraction'], 2) == 0.61
    assert full['values']['Tet24_C23'] == pytest.approx(3.26 / 21.75, abs=0.0003)
    assert partial['status'] == 'profile'
    assert partial['values'] == {key: value for key, value in full['values'].items() if key != 'C19_C23'}
    assert replay_report(json.loads(json.dumps(report)))['results'] == report['results']


def test_valemon_regular_sterane_fractions_from_database_fields(qtbot):
    from autopyro.ui.window import MainWindow

    # Geolab Nor, Valemon 142/0001-0, Table 11g (m/z 217, ng/g oil).
    # https://factpages.sodir.no/pbl/geochemical_pdfs/5248_2.pdf
    unit = 'ppm (µg/g)'
    fields = [
        {'source_field_id': carbon, 'group_name': 'Sat-GCMS',
         'source_label': f'St{carbon}aaR[ch] [{unit}]', 'unit_label': unit, 'source_dtype': 'Float64'}
        for carbon in (27, 28, 29)
    ]
    assert profile_fields({'fields': fields}, 'sterane_distribution', 'Sat-GCMS', 'ch') == fields
    records = [{'source_record_id': i, 'sample_name': name}
               for i, name in enumerate(('Valemon', 'Missing peak', 'Zero sum'), 1)]
    observations = [
        {'source_record_id': sample, 'source_field_id': carbon, 'analysis_run_id': sample,
         'repeat_number': 1, 'group_name': 'Sat-GCMS', 'value_float': value}
        for sample, values in enumerate(((9.3274, 2.3754, 2.4805), (9.3274, None, 2.4805), (0, 0, 0)), 1)
        for carbon, value in zip((27, 28, 29), values, strict=True) if value is not None
    ]
    project = project_from_records(records, observations, fields, source={'citation': 'Valemon, Table 11g'})
    config = {
        'method': 'sterane_distribution', 'version': 1, 'confirmed': True,
        'context': 'Valemon, quantified ααα 20R peaks', 'variant': 'ch',
        'inputs': suggest_bindings(project, 'sterane_distribution', 'Sat-GCMS', 'ch'),
    }
    project.view['nonpyro_configs'] = {'sterane_distribution': config}
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_nonpyro('steranes')
    dialog = owner.geochemistry_windows['steranes']
    dialog.form.fields['confirmed'][0].setChecked(True)
    dialog.calculate_button.click()
    report = dialog.report
    assert report is not None, dialog.status.text()
    valid, missing, zero = report['results']
    # Independent normalization of the three printed concentrations by 14183.3 ng/g.
    assert valid['values'] == pytest.approx({'C27': 65.76325679, 'C28': 16.74786545, 'C29': 17.48887776})
    assert missing['status'] == 'missing'
    assert zero['status'] == 'zero_denominator'
    assert replay_report(json.loads(json.dumps(report)))['results'] == report['results']
    dialog.sara_plot_button.click()
    assert dialog.result_tabs.currentWidget() is dialog.sara_canvas
    axes = dialog.sara_canvas.figure.axes[0]
    assert [label.get_text() for label in axes.texts] == [f'C{carbon} ααα 20R' for carbon in (27, 28, 29)]
    offsets = axes.collections[0].get_offsets()
    assert len(offsets) == 1
    assert tuple(offsets[0]) == pytest.approx((0.2549230433, 0.1514581647))
    owner.dirty = False


def test_published_alkane_profiles_match_reference_overlap():
    # Geolab Nor, Table 9a1, saturated fraction peak areas in nC24–nC34 order.
    # https://factpages.sodir.no/pbl/geochemical_pdfs/3260_2.pdf
    areas = [
        ('T92/0143-0', *T92_0143_AREAS),
        ('T92/0144-0', 1354683, 1152354, 898410, 819250, 567058, 503251,
         474377, 302128, 235018, 329451, 395507),
    ]
    headers = ['Sample'] + [f'Sat-GC / nC{carbon}[a]' for carbon in range(24, 35)]
    project = build_project(
        headers, areas, [2, 3],
        [(i, name, 'number' if i else 'text', 'count' if i else '') for i, name in enumerate(headers)],
        {'citation': 'Geolab Nor, Norwegian Offshore Directorate report 3260_2, Table 9a1'},
    )
    project.sync_measurements()
    a, b = GeochemistryIndex(project).samples
    column_ids = [column.id for column in project.columns[1:]]
    report = compare_samples(project, [a], [b], column_ids)
    match = compare_alkane_profiles(report)

    # Reference from the printed areas: 100 minus half the L1 distance in percent.
    assert match['overlap_percent'] == pytest.approx(97.74444599496, abs=1e-10, rel=0)
    assert [peak['carbon'] for peak in match['peaks']] == list(range(24, 35))
    assert sum(peak['a_percent'] for peak in match['peaks']) == pytest.approx(100)
    assert sum(peak['b_percent'] for peak in match['peaks']) == pytest.approx(100)
    reverse = compare_alkane_profiles(compare_samples(project, [b], [a], column_ids))
    assert reverse['overlap_percent'] == pytest.approx(match['overlap_percent'])
    assert [peak['difference_pp'] for peak in reverse['peaks']] == pytest.approx(
        [-peak['difference_pp'] for peak in match['peaks']]
    )


def test_published_oil_and_rock_extract_reproduce_descriptive_correlation():
    # Oil: Valemon 142/0001-0, Tables 9Aa and 9Ba, https://factpages.sodir.no/pbl/geochemical_pdfs/5248_2.pdf
    # Core extract: T92/0143-0, Table 9a1, https://factpages.sodir.no/pbl/geochemical_pdfs/3260_6.pdf
    # Samples from separate reports form a numerical reference pair.
    headers = ['Sample', 'Sample / Type', 'Sample / Preparation', *[
        f'Sat-GC / nC{carbon}[a]' for carbon in range(24, 35)
    ], 'Sat-GC / Pr/Ph[a]']
    units = ['', '', '', *(['count'] * 11), '1']
    project = build_project(
        headers,
        [('142/0001-0', 'oil', '', *VALEMON_AREAS, 1.70),
         ('T92/0143-0', 'rock', 'extract', *T92_0143_AREAS, '')],
        [2, 3],
        [(i, name, 'number' if units[i] else 'text', units[i]) for i, name in enumerate(headers)],
        {'citation': 'Geolab Nor reports 5248_2 (9Aa, 9Ba) and 3260_6 (9a1)'},
    )
    oil, extract = GeochemistryIndex(project).samples
    report = compare_samples(
        project, [oil], [extract], [column.id for column in project.columns[3:]],
        allow_mixed_material=True,
    )
    result = correlate_oil_profiles(report)
    assert result['correlation']['materials'] == ['oil', 'rock_extract']
    assert result['correlation']['expert_review_required']
    # Separate calculation: 100 minus half the L1 distance of the normalized profiles in percent.
    assert result['overlap_percent'] == pytest.approx(93.64383451705, abs=1e-10, rel=0)
    last_peak = result['peaks'][-1]
    assert last_peak['carbon'] == 34
    assert last_peak['difference_pp'] == pytest.approx(2.54111593039, abs=1e-10, rel=0)
    evidence, = result['evidence']
    assert evidence['a'] == {'status': 'numeric', 'value': 1.70}
    assert evidence['b']['status'] == 'missing' and evidence['difference'] is None
    assert correlation_sensitivity(report, result) == [
        {'group': 'Sat-GC', 'comparable': 0, 'lost': [], 'overlap_percent': None},
    ]
    before = deepcopy(report)
    trial, = correlation_sensitivity(report, result, exclude_point=(project.columns[3].id, oil))
    assert trial['overlap_percent'] is None
    assert report == before
    snapshot = json.loads(json.dumps(freeze_comparison(report)))
    assert correlate_oil_profiles(snapshot) == result


@pytest.mark.parametrize('basis, signals, reported', [
    ('h', (592, 423, 512, 351, 481, 195, 599, 310, 232, 110, 206, 264, 101, 214, 382, 318, 279,
           1310, 369, 1891, 1258, 1093, 5160, 1594, 1248, 1097, 758, 883, 606, 662, 475, 404, 269),
     0.2919222099910887),
    ('a', (3985, 2664, 4116, 2410, 5910, 1934, 7515, 2581, 2000, 914, 1531, 2052, 871, 1792, 3191, 2546, 2612,
           8837, 2625, 12618, 8821, 7569, 33668, 10528, 9553, 7385, 4860, 5887, 4306, 4423, 3124, 2544, 1801),
     0.3782526507401846),
])
def test_barents_ster_hop_profile_and_correlation(basis, signals, reported):
    # Barents Sea Update 2025, Stratum Reservoir, 7122/9-1 T2, sample 6133415181.
    # Both oil rows are synthetic controls using scaled values from that sample.
    peaks = ('St27dbS St27dbR St28dbS St28dbR St29dbR St27aaS St27bbR+ St27bbS St27aaR '
             'St28aaS St28bbR St28bbS St28aaR St29aaS St29bbR St29bbS St29aaR '
             'H27Ts H27Tm H29ab H29Ts H30D H30ab H31abS H31abR H32abS H32abR H33abS H33abR '
             'H34abS H34abR H35abS H35abR').split()
    headers = ['Sample / Name', 'Sample / Type', 'Sample / Preparation',
               *(f'Sat-GCMS / {peak}[{basis}]' for peak in peaks)]
    scaled = tuple(2 * value if i < 17 else value for i, value in enumerate(signals))
    project = build_project(
        headers, [('6133415181', 'rock', 'extract', *signals), ('Oil A', 'oil', '', *scaled),
                  ('Oil B', 'oil', '', *signals)],
        [2, 3, 4],
        [(i, name, 'text' if i < 3 else 'number', '' if i < 3 else 'count')
         for i, name in enumerate(headers)],
        {'citation': 'Barents Sea Update 2025, sample 6133415181'},
    )
    method = 'ster_hop_from_peaks'
    config = {'method': method, 'version': 1, 'confirmed': True, 'context': 'Stratum Reservoir',
              'inputs': suggest_bindings(project, method, 'Sat-GCMS', basis)}
    profile = evaluate_project(project, [row.id for row in project.rows], config)
    assert [row['values']['ster_hop'] for row in profile['results']] == pytest.approx(
        [reported, 2 * reported, reported], rel=1e-14)
    assert replay_report(json.loads(json.dumps(profile)))['results'] == profile['results']

    extract, oil, second_oil = GeochemistryIndex(project).samples
    fields = [column.id for column in project.columns[3:]]
    for target in (extract, second_oil):
        comparison = compare_samples(project, [oil], [target], fields, allow_mixed_material=True)
        match = correlate_oil_profiles(comparison)
        evidence, = match['evidence']
        assert evidence['input_field_ids'] == fields
        assert evidence['difference'] == pytest.approx(-reported)
        assert correlate_oil_profiles(json.loads(json.dumps(freeze_comparison(comparison)))) == match

    project.rows[-1].values[fields[-1]] = None
    comparison = compare_samples(project, [oil], [second_oil], fields)
    evidence, = correlate_oil_profiles(comparison)['evidence']
    assert evidence['b']['status'] == 'missing' and evidence['difference'] is None
    for column_id in fields[17:]:
        project.rows[-1].values[column_id] = 0
    comparison = compare_samples(project, [oil], [second_oil], fields)
    evidence, = correlate_oil_profiles(comparison)['evidence']
    assert evidence['b']['status'] == 'zero_denominator' and evidence['difference'] is None


def test_berea_diasterane_profile_and_correlation_replay():
    # Hackley et al., BLTN19077, Table 3: C27 Dia/ααα Ster.
    # Only the ratios are published. The peaks below are scaled test inputs.
    headers = ['Sample', 'Sample / Type', 'Sample / Preparation',
               'Sat-GCMS / St27dbS[h]', 'Sat-GCMS / St27aaR[h]',
               'Sat-GCMS / St27dbS/St27aaR[h]']
    units = ('', '', '', 'count', 'count', 'Euc (a/b)')
    project = build_project(
        headers, [('NYTIS-ALC 20', 'oil', '', 721, 100, 7.21),
                  ('NYTIS-Torchlight 8', 'oil', '', 725, 100, 7.25),
                  ('Aristech 4 863.8 ft', 'rock', 'extract', 50, 100, 0.50)],
        [2, 3, 4],
        [(i, name, 'text' if i < 3 else 'number', unit)
         for i, (name, unit) in enumerate(zip(headers, units, strict=True))],
        {'citation': 'Hackley et al., BLTN19077, Table 3'},
    )
    method = 'c27_diasterane_ratio'
    config = {'method': method, 'version': 1, 'confirmed': True, 'context': 'Berea',
              'inputs': suggest_bindings(project, method, 'Sat-GCMS', 'h')}
    profile = evaluate_project(project, [row.id for row in project.rows], config)
    assert [row['values']['dia27_sr'] for row in profile['results']] == pytest.approx([7.21, 7.25, 0.5])
    assert replay_report(json.loads(json.dumps(profile)))['results'] == profile['results']

    oil, second_oil, extract = GeochemistryIndex(project).samples
    for target, expected in ((second_oil, 0.04), (extract, -6.71)):
        comparison = compare_samples(project, [oil], [target], [project.columns[-1].id],
                                     allow_mixed_material=True)
        correlation = correlate_oil_profiles(comparison)
        assert correlation['evidence'][0]['difference'] == pytest.approx(expected)
        assert correlate_oil_profiles(json.loads(json.dumps(freeze_comparison(comparison)))) == correlation


def test_valemon_aromatic_indices_from_published_peaks():
    # Geolab Nor, Valemon, Tables 12c and 12d.
    peaks = {'Phen': 1480203, '2MP': 791845, '9MP': 898032.1, '1MP': 526248.7,
             '4MDBT': 143517.8, '1MDBT': 6095.2}
    headers = ['Sample', *(f'Arom-GCMS / {peak}[h]' for peak in peaks)]
    project = build_project(
        headers, [('Valemon', *peaks.values())], [2],
        [(i, name, 'number' if i else 'text', 'count' if i else '') for i, name in enumerate(headers)],
        {'citation': 'Geolab Nor, Valemon, Tables 12c and 12d'},
    )
    method = 'aromatic_indices_from_peaks'
    config = {'method': method, 'version': 1, 'confirmed': True, 'context': 'Valemon',
              'inputs': suggest_bindings(project, method, 'Arom-GCMS', 'h')}
    profile = evaluate_project(project, [row.id for row in project.rows], config)
    row, = profile['results']
    assert row['values'] == pytest.approx({'mpi2': 0.8178854363, 'mpr': 1.5046973038, 'mdbtr': 23.5460362252})
    assert replay_report(json.loads(json.dumps(profile)))['results'] == profile['results']


def test_berea_correlation_compares_homohopane_abundances_and_ratios():
    # Hackley et al., BLTN19077, Table 4: S+R percentages and C35/C34 (22S).
    # The [h] tag encodes a common basis for exercising the database contract.
    headers = ['Sample', 'Sample / Type', 'Sample / Preparation',
               *(f'Sat-GCMS / {carbon}HomoHop[h]' for carbon in range(31, 36)),
               'Sat-GCMS / H35ab/H31ab[h]', 'Sat-GCMS / H35ab/H34ab[h]',
               'Sat-GCMS / H35abS/H34abS[h]',
               'Sat-GCMS / 35HomoHop[un]']
    project = build_project(
        headers,
        [('NYTIS-ALC 20', 'oil', '', 34, 26, 21, 12, 7, 7/34, 7/12, 0.57, 7),
         ('NYTIS-Torchlight 8', 'oil', '', 35, 27, 20, 11, 7, 7/35, 7/11, 0.63, 7),
         ('Aristech 4 863.8 ft', 'rock', 'extract', 49, 21, 16, 9, 5, 5/49, 5/9, 0.48, 5)],
        [2, 3, 4],
        [(i, name, 'text' if i < 3 else 'number',
          '' if i < 3 else 'Euc (a/b)' if 8 <= i <= 10 else '% (a/(a+b))')
         for i, name in enumerate(headers)],
        {'citation': 'Hackley et al., BLTN19077, Table 4'},
    )
    oil, second_oil, extract = GeochemistryIndex(project).samples
    fields = [column.id for column in project.columns[3:]]
    for target, expected in (
        (second_oil, [1, 1, -1, -1, 0, -0.005882353, 0.053030303, 0.06]),
        (extract, [15, -5, -5, -3, -2, -0.103841537, -0.027777778, -0.09]),
    ):
        report = compare_samples(project, [oil], [target], fields, allow_mixed_material=True)
        result = correlate_oil_profiles(report)
        assert [item['difference'] for item in result['evidence'][:-1]] == pytest.approx(expected, abs=1e-8)
        unknown = result['evidence'][-1]
        assert unknown['a']['status'] == 'unsupported_correlation_indicator'
        assert unknown['difference'] is None
        assert result['overlap_percent'] is None
        assert result['correlation']['expert_review_required']
        assert correlate_oil_profiles(json.loads(json.dumps(freeze_comparison(report)))) == result


def test_hongche_oil_correlation_compares_homohopane_epimers():
    # He et al., doi:10.3389/feart.2022.953245, Table 3, columns r and s.
    # Encode published C31/C32 fractions as percentages with a common test basis.
    # The C33–C35 columns below are synthetic controls.
    headers = ['Sample', 'Sample / Type', *(
        f'Sat-GCMS / H{carbon} 22S/R[{tag}]'
        for carbon, tag in zip(range(31, 36), ('h', 'h', 'h', 'h', 'un'))
    )]
    project = build_project(
        headers, [('CF6', 'oil', 71, 60, 50, 45, 60), ('CF7', 'oil', 60, 40, None, 101, 50)],
        [2, 3],
        [(i, name, 'text' if i < 2 else 'number', '' if i < 2 else '% (a/(a+b))')
         for i, name in enumerate(headers)],
        {'citation': 'doi:10.3389/feart.2022.953245, Table 3'},
    )
    oil_a, oil_b = GeochemistryIndex(project).samples
    report = compare_samples(project, [oil_a], [oil_b], [column.id for column in project.columns[2:]])
    result = correlate_oil_profiles(report)
    c31, c32, c33, c34, c35 = result['evidence']
    assert [c31['difference'], c32['difference']] == pytest.approx([-11, -20])
    assert c33['a']['status'] == 'numeric' and c33['b']['status'] == 'missing'
    assert c34['b']['status'] == 'invalid_range'
    assert c35['a']['status'] == 'unsupported_correlation_indicator'
    assert all(item['difference'] is None for item in (c33, c34, c35))
    assert result['correlation']['expert_review_required']
    assert correlate_oil_profiles(json.loads(json.dumps(freeze_comparison(report)))) == result


def test_hongche_oil_correlation_compares_hopane_and_moretane_fields():
    # He et al., doi:10.3389/feart.2022.953245, Table 3, columns k, o, p.
    # Convert the published ratios to the database definitions on a shared test basis.
    headers = ['Sample', 'Sample / Type', 'Sat-GCMS / H30ab/H29ab[h]',
               'Sat-GCMS / H29ab/H29ba[h]', 'Sat-GCMS / H30ab/H30ba[h]',
               'Sat-GCMS / H30ab/H30ba[a]']
    units = ('', '', 'Euc (a/b)', '% (a/(a+b))', '% (a/(a+b))', 'Euc (a/b)')
    project = build_project(
        headers,
        [('CF6', 'oil', 1 / 1.23, 100 / 1.22, 100 / 1.33, 0.33),
         ('CF7', 'oil', 1 / 1.00, 100 / 1.08, 100 / 1.10, 0.10)],
        [2, 3],
        [(i, name, 'text' if i < 2 else 'number', unit)
         for i, (name, unit) in enumerate(zip(headers, units, strict=True))],
        {'citation': 'doi:10.3389/feart.2022.953245, Table 3'},
    )
    oil_a, oil_b = GeochemistryIndex(project).samples
    report = compare_samples(project, [oil_a], [oil_b], [column.id for column in project.columns[2:]])
    result = correlate_oil_profiles(report)
    hopane, c29, c30, wrong_unit = result['evidence']
    assert [item['difference'] for item in (hopane, c29, c30)] == pytest.approx(
        [0.18699187, 10.62537948, 15.72112098])
    assert wrong_unit['a']['status'] == 'unsupported_correlation_indicator'
    assert wrong_unit['difference'] is None
    assert correlate_oil_profiles(json.loads(json.dumps(freeze_comparison(report)))) == result


@pytest.mark.parametrize('field, oils, values, difference, citation', [
    ('t20/t21', ('CF6', 'CF7'), (0.74, 0.73), -0.01, 'He et al., 10.3389/feart.2022.953245, Table 3'),
    ('t22/t21', ('NYTIS-ALC 20', 'NYTIS-Torchlight 8'), (0.37, 0.43), 0.06, 'Hackley et al., BLTN19077, Table 4'),
])
def test_oil_correlation_compares_published_tricyclic_ratios(field, oils, values, difference, citation):
    # Encode the published ratios on a common test basis in the database format.
    headers = ['Sample', 'Sample / Type', f'Sat-GCMS / {field}[h]', f'Sat-GCMS / {field}[un]']
    project = build_project(
        headers, [(name, 'oil', value, value) for name, value in zip(oils, values, strict=True)],
        [2, 3],
        [(i, name, 'text' if i < 2 else 'number', '' if i < 2 else 'Euc (a/b)')
         for i, name in enumerate(headers)],
        {'citation': citation},
    )
    oil_a, oil_b = GeochemistryIndex(project).samples
    report = compare_samples(project, [oil_a], [oil_b], [column.id for column in project.columns[2:]])
    result = correlate_oil_profiles(report)
    supported, unknown = result['evidence']
    assert supported['difference'] == pytest.approx(difference)
    assert unknown['a']['status'] == 'unsupported_correlation_indicator'
    assert unknown['difference'] is None
    assert correlate_oil_profiles(json.loads(json.dumps(freeze_comparison(report)))) == result


def test_hongche_oil_correlation_compares_terpane_ratios():
    # He et al., doi:10.3389/feart.2022.953245, Table 3, columns q, m, l.
    # The [h] tag is a shared test basis. The tricyclic and tetracyclic ratios are synthetic controls.
    headers = ['Sample', 'Sample / Type', 'Sat-GCMS / H27Ts/H27Tm[h]',
               'Sat-GCMS / H31abR/H30ab[h]', 'Sat-GCMS / Gamm/31abR[h]',
               'Sat-GCMS / t19/t23[h]', 'Sat-GCMS / t23/tet24[h]',
               'Sat-GCMS / t19+20/t23[h]', 'Sat-GCMS / t19/t23[un]', 'Sat-GCMS / H31ab/H30ab[h]']
    project = build_project(
        headers,
        [('CF6', 'oil', 0.53, 0.20, 1.63, 0.2, 3.0, 0.4, 0.2, 0.50),
         ('CF7', 'oil', 0.25, 0.06, 1.04, 0.3, 2.0, 0.6, 0.3, 0.15)],
        [2, 3],
        [(i, name, 'text' if i < 2 else 'number', '' if i < 2 else 'Euc (a/b)')
         for i, name in enumerate(headers)],
        {'citation': 'doi:10.3389/feart.2022.953245, Table 3'},
    )
    oil_a, oil_b = GeochemistryIndex(project).samples
    report = compare_samples(project, [oil_a], [oil_b], [column.id for column in project.columns[2:]])
    result = correlate_oil_profiles(report)
    assert [item['difference'] for item in result['evidence'][:5]] == pytest.approx(
        [-0.28, -0.14, -0.59, 0.1, -1])
    for item in result['evidence'][5:]:
        assert item['a']['status'] == 'unsupported_correlation_indicator'
        assert item['difference'] is None
    assert correlate_oil_profiles(json.loads(json.dumps(freeze_comparison(report)))) == result


def test_lucaogou_oil_extract_correlation_reproduces_published_biomarker_differences():
    # doi:10.1021/acsomega.0c05224, Table 3 and Figures 11–12.
    # Oil from fractures in sample 4, the matched U1 extract, and a nearby siltstone U1 extract.
    headers = ['Sample', 'Sample / Type', 'Sample / Preparation',
               'Sat-GC / iC18/nC18[a]', 'Sat-GC / Pr/nC19[a]']
    project = build_project(
        headers, [('J174 oil in shale', 'oil', '', 0.44, 0.47),
                  ('4 U1', 'rock', 'extract', 0.40, 0.42),
                  ('2 U1', 'rock', 'extract', 0.72, 1.14)],
        [2, 3, 4],
        [(i, name, 'text' if i < 3 else 'number', '' if i < 3 else '1') for i, name in enumerate(headers)],
        {'citation': 'doi:10.1021/acsomega.0c05224, Table 3 and Figures 11–12'},
    )
    oil, shale, siltstone = GeochemistryIndex(project).samples
    fields = [column.id for column in project.columns[3:]]
    for extract, expected in ((shale, [-0.04, -0.05]), (siltstone, [0.28, 0.67])):
        report = compare_samples(project, [oil], [extract], fields, allow_mixed_material=True)
        result = correlate_oil_profiles(report)
        assert [item['difference'] for item in result['evidence']] == pytest.approx(expected)
        assert result['overlap_percent'] is None and result['peaks'] == []
        assert result['correlation']['expert_review_required']
        assert correlate_oil_profiles(json.loads(json.dumps(freeze_comparison(report)))) == result


@pytest.fixture
def published_sara():
    # Kotarba et al. (2021), Table 4, https://doi.org/10.1007/s00531-021-02035-7.
    # Fractions are printed to 0.1 mass percent and Sat/Arom to 0.1.
    published = [
        ('Dl-25', 54.1, 35.4, 9.8, 0.7, 1.5),
        ('So-9', 79.5, 13.3, 5.3, 1.9, 6.0),
        ('Wl-15', 66.5, 22.1, 7.8, 3.6, 3.0),
        ('El-58', 59.3, 23.5, 12.0, 5.2, 2.5),
        ('Le-2', 96.1, 3.6, 0.3, 0.0, 26.7),
        ('S-Sa/1', 63.4, 23.1, 8.0, 5.4, 2.7),
    ]
    headers = ['Sample / Name', 'Saturates', 'Aromatics', 'Resins', 'Asphaltenes']
    project = build_project(headers, [row[:5] for row in published], range(2, 8),
                            [(i, name, 'number' if i else 'text', '%wt' if i else '')
                             for i, name in enumerate(headers)],
                            {'citation': 'Kotarba et al. (2021), Table 4, doi:10.1007/s00531-021-02035-7'})
    return project, published


def test_published_sara_fractions_preserve_balance_and_match_sat_arom(published_sara):
    project, published = published_sara
    config = {'method': 'sara', 'version': 1, 'confirmed': True, 'context': 'Published oil fractions, mass percent',
              'closure_tolerance': 0.2,
              'inputs': {role: {'column_id': project.columns[i].id, 'unit': '%wt'}
                         for i, role in enumerate(('sat', 'arom', 'res', 'asph'), 1)}}
    report = evaluate_project(project, [row.id for row in project.rows], config)
    for (sample, sat, arom, _, _, reference), result in zip(published, report['results']):
        assert result['status'] == 'interpreted', sample
        assert not result['warnings'], sample
        assert result['values']['sat_arom'] == pytest.approx(reference, abs=0.05), sample
    assert report['results'][4]['canonical_inputs']['asph'] == 0
    assert report['results'][5]['values']['sum'] == pytest.approx(99.9)
    assert len(sara_ternary_points(report)) == len(published)
    assert replay_report(json.loads(json.dumps(report)))['results'] == report['results']


def test_published_sara_clr_pca_matches_covariance_reference(published_sara):
    project, _ = published_sara
    sample_ids = list(GeochemistryIndex(project).samples)
    fields = [column.id for column in project.columns[1:]]
    comparison = compare_samples(project, sample_ids[:3], sample_ids[3:], fields)
    result = comparison_pca(comparison, preprocessing='clr')

    # Reference: eigendecomposition of the sample covariance of log-centered
    # compositions. PC1 is oriented with a positive asphaltene coefficient.
    assert result['excluded'] == {sample_ids[4]: {fields[3]: 'nonpositive_composition'}}
    assert result['explained_variance_ratio'] == pytest.approx([0.825648051, 0.156960708], abs=1e-8, rel=0)
    assert [point['sample_id'] for point in result['scores']] == sample_ids[:4] + sample_ids[5:]
    assert [point['coordinates'][0] for point in result['scores']] == pytest.approx(
        [-1.35634770, 0.04049980, 0.26996796, 0.45516510, 0.59071484], abs=1e-8, rel=0,
    )
    record = archive_model_run(project, comparison, result)
    snapshot = project.view[MODEL_RUNS_KEY]['snapshots'][record['snapshot_id']]
    record, snapshot = json.loads(json.dumps([record, snapshot]))
    assert replay_model_run(record, snapshot)


def test_published_sara_clr_ward_matches_merge_cost_reference(published_sara):
    project, _ = published_sara
    dl, so, wl, el, zero_asphaltenes, ss = GeochemistryIndex(project).samples
    fields = [column.id for column in project.columns[1:]]
    comparison = compare_samples(project, [dl, so, wl], [el, zero_asphaltenes, ss], fields)
    result = comparison_clustering(comparison, 3, preprocessing='clr')

    # Reference from comparing increases in within-cluster squared deviations.
    # The linkage distance is sqrt(2 * increase).
    increases = [row[2] ** 2 / 2 for row in result['linkage']]
    assert increases[:3] == pytest.approx([0.06181704396, 0.09410133219, 0.51105898642], abs=1e-10, rel=0)
    assert result['excluded'] == {zero_asphaltenes: {fields[3]: 'nonpositive_composition'}}
    labels = {point['sample_id']: point['cluster'] for point in result['membership']}
    assert set(labels) == {dl, so, wl, el, ss}
    assert labels[wl] == labels[el] == labels[ss]
    assert len({labels[dl], labels[so], labels[wl]}) == 3

    record = archive_model_run(project, comparison, result)
    snapshot = project.view[MODEL_RUNS_KEY]['snapshots'][record['snapshot_id']]
    record, snapshot = json.loads(json.dumps([record, snapshot]))
    assert replay_model_run(record, snapshot)


def test_published_sara_clr_kmeans_matches_exhaustive_partition(published_sara):
    project, _ = published_sara
    dl, so, wl, el, zero_asphaltenes, ss = GeochemistryIndex(project).samples
    fields = [column.id for column in project.columns[1:]]
    comparison = compare_samples(project, [dl, so, wl], [el, zero_asphaltenes, ss], fields)
    result = comparison_kmeans(comparison, 2, seed=17, preprocessing='clr')

    # Exhaustive reference over the 15 unordered two-way partitions.
    assert result['inertia'] == pytest.approx(0.6669773625716665, abs=1e-12, rel=0)
    assert result['excluded'] == {zero_asphaltenes: {fields[3]: 'nonpositive_composition'}}
    labels = {point['sample_id']: point['cluster'] for point in result['membership']}
    assert set(labels) == {dl, so, wl, el, ss}
    assert labels[dl] != labels[so] == labels[wl] == labels[el] == labels[ss]
    assert result['seed'] == 17

    record = archive_model_run(project, comparison, result)
    snapshot = project.view[MODEL_RUNS_KEY]['snapshots'][record['snapshot_id']]
    record, snapshot = json.loads(json.dumps([record, snapshot]))
    assert replay_model_run(record, snapshot)


def test_published_sara_clr_dbscan_matches_distance_reference(published_sara):
    project, _ = published_sara
    dl, so, wl, el, zero_asphaltenes, ss = GeochemistryIndex(project).samples
    fields = [column.id for column in project.columns[1:]]
    comparison = compare_samples(project, [dl, so, wl], [el, zero_asphaltenes, ss], fields)
    result = comparison_dbscan(comparison, radius=0.7, min_samples=4, preprocessing='clr')

    # Pairwise CLR distances put only Wl-15 within radius of three other samples.
    # Dl-25 is more than 1.57 away from every other complete sample.
    assert result['cluster_count'] == result['noise_count'] == 1
    assert result['excluded'] == {zero_asphaltenes: {fields[3]: 'nonpositive_composition'}}
    assert {point['sample_id']: (point['cluster'], point['role'])
            for point in result['membership']} == {
        dl: (None, 'noise'), so: (1, 'border'), wl: (1, 'core'),
        el: (1, 'border'), ss: (1, 'border'),
    }
    assert (result['radius'], result['min_samples']) == (0.7, 4)

    record = archive_model_run(project, comparison, result)
    snapshot = project.view[MODEL_RUNS_KEY]['snapshots'][record['snapshot_id']]
    record, snapshot = json.loads(json.dumps([record, snapshot]))
    assert replay_model_run(record, snapshot)


def test_published_kerogen_elemental_analysis_matches_atomic_ratios():
    # Teerman et al. (1995), Table 6, doi:10.1016/0146-6380(94)00083-2.
    # C/H/O are mass percent of isolated kerogen. H/C and O/C are printed to 0.01.
    published = [
        ('1 / Lab A', 60.80, 4.14, 19.02, 0.82, 0.23),
        ('1 / Lab B', 61.58, 4.57, 24.00, 0.89, 0.29),
        ('2 / Lab A', 55.97, 6.10, 7.96, 1.31, 0.11),
        ('2 / Lab B', 58.56, 6.37, 9.42, 1.31, 0.12),
        ('3 / Lab A', 67.09, 6.15, 7.90, 1.10, 0.09),
        ('3 / Lab B', 69.51, 6.62, 8.54, 1.14, 0.09),
        ('4 / Lab A', 53.31, 5.73, 13.62, 1.29, 0.19),
        ('4 / Lab B', 53.51, 5.69, 15.58, 1.28, 0.22),
    ]
    headers = ['Sample / Name', 'Elem / C [%wt]', 'Elem / H [%wt]', 'Elem / O [%wt]']
    project = build_project(headers, [row[:4] for row in published], range(2, 10),
                            [(i, name, 'number' if i else 'text', '%wt' if i else '')
                             for i, name in enumerate(headers)],
                            {'citation': 'Teerman et al. (1995), Table 6, doi:10.1016/0146-6380(94)00083-2'})
    config = {'method': 'elemental', 'version': 1, 'confirmed': True,
              'context': 'Published isolated kerogen, C/H/O mass percent',
              'inputs': suggest_bindings(project, 'elemental', 'Elem')}
    report = evaluate_project(project, [row.id for row in project.rows], config)
    for (sample, _, _, _, hc, oc), result in zip(published, report['results']):
        assert result['values']['HC'] == pytest.approx(hc, abs=0.005), sample
        assert result['values']['OC'] == pytest.approx(oc, abs=0.005), sample
    assert replay_report(json.loads(json.dumps(report)))['results'] == report['results']


def test_usgs_atomic_ratios_round_trip_database_units_and_preserve_missing_oxygen():
    # USGS OFR 89-108, Table 3. The database input representation is ratio × 100.
    # https://pubs.usgs.gov/of/1989/0108/report.pdf
    published = [
        ('1', 1.27, 0.17),
        ('2', 1.21, 0.14),
        ('3', 1.28, None),
        ('4', 1.30, 0.08),
    ]
    unit = '%mol (a/b)'
    headers = ['Sample', f'Elem / H/C mol [{unit}]', f'Elem / O/C mol [{unit}]']
    project = build_project(
        headers,
        [(sample, round(hc * 100), '' if oc is None else round(oc * 100))
         for sample, hc, oc in published],
        range(2, 6),
        [(i, name, 'number' if i else 'text', unit if i else '') for i, name in enumerate(headers)],
        {'citation': 'USGS OFR 89-108, Table 3'},
    )
    config = {
        'method': 'atomic_ratios', 'version': 1, 'confirmed': True,
        'context': 'Union Annex #1, measured kerogen concentrates',
        'inputs': suggest_bindings(project, 'atomic_ratios', 'Elem'),
    }
    report = evaluate_project(project, [row.id for row in project.rows], config)
    for (sample, hc, oc), result in zip(published, report['results'], strict=True):
        if oc is None:
            assert result['status'] == 'missing', sample
            assert result['values'] == {}, sample
        else:
            assert result['status'] == 'profile', sample
            assert result['values'] == pytest.approx({'HC': hc, 'OC': oc}, abs=1e-12, rel=0), sample
    assert replay_report(json.loads(json.dumps(report)))['results'] == report['results']


def test_usgs_reference_gases_show_normal_and_reversed_chung_profiles(qtbot):
    from autopyro.ui.window import MainWindow

    # USGS SIR 2021–5136, Tables 6 and 7, https://pubs.usgs.gov/sir/2021/5136/sir20215136.pdf.
    published = [
        ('HCG-1', -1.51, -10.22, -15.43),
        ('HCG-2', -43.09, -29.80, -19.35),
        ('HCG-3', -61.39, -45.31, -36.80),
        ('NGS-3', -73.06, None, None),
    ]
    headers = ['Sample / Name', *[f'Gas-Iso / d13C C{i} [per mil]' for i in range(1, 4)]]
    project = build_project(headers, published, range(2, 6),
                            [(i, name, 'number' if i else 'text', 'per mil' if i else '')
                             for i, name in enumerate(headers)], {'citation': 'USGS SIR 2021–5136, Tables 6 and 7'})
    project.view['nonpyro_configs'] = {'gas_isotopes': {
        'method': 'gas_isotopes', 'version': 1, 'context': 'USGS gas reference materials',
        'isotope_reference': 'VPDB', 'inputs': suggest_bindings(project, 'gas_isotopes', 'Gas-Iso'),
    }}
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_nonpyro('gas')
    dialog = owner.geochemistry_windows['gas']
    dialog.method.setCurrentIndex(dialog.method.findData('gas_isotopes'))
    dialog.form.fields['confirmed'][0].setChecked(True)
    dialog.calculate_button.click()
    assert dialog.report is not None, dialog.status.text()
    results = dialog.report['results']
    assert 'Полная изотопная инверсия' == results[0]['interpretation']
    assert all('Нормальный порядок' in item['interpretation'] for item in results[1:3])
    assert results[3]['status'] == 'missing'
    for item, (d21, d32) in zip(results, [(-8.71, -5.21), (13.29, 10.45), (16.08, 8.51)]):
        assert item['values'] == pytest.approx({'delta21': d21, 'delta32': d32})
    dialog.results.selectAll()
    dialog.chung_button.click()
    lines = dialog.chung_canvas.figure.axes[0].lines
    assert len(lines) == 3
    for line, (_, methane, ethane, propane) in zip(lines, published[:3]):
        assert list(line.get_xdata()) == pytest.approx([1 / 3, 1 / 2, 1])
        assert list(line.get_ydata()) == [propane, ethane, methane]
    assert replay_report(json.loads(json.dumps(dialog.report)))['results'] == results
    owner.dirty = False


def test_published_oil_measurements_exercise_independent_well_prediction():
    # Kotarba et al. (2021), Tables 3 and 4, doi:10.1007/s00531-021-02035-7.
    # A: Istebna sandstone wells. B: Ciężkowice sandstone wells.
    # This split and the regression are a numerical benchmark, not the authors' model.
    published = [
        ('Dl-25', 54.1, 0.04), ('Dl-51', 69.2, 0.14), ('So-9', 79.5, 0.08),
        ('Wl-3', 65.5, 0.10), ('Wl-15', 66.5, 0.12),
        ('El-58', 59.3, 0.19), ('Jo-44', 41.8, 0.20), ('Pl-8', 53.7, 0.27), ('SB-32', 66.2, 0.09),
    ]
    well = Column('Well / Well Name')
    saturates = Column('Saturates', 'number', unit='%wt')
    sulfur = Column('Organic sulfur', 'number', unit='%wt')
    project = Project(columns=[well, saturates, sulfur], rows=[
        Row({well.id: code, saturates.id: sat, sulfur.id: amount}) for code, sat, amount in published
    ], source={'citation': 'Kotarba et al. (2021), Tables 3 and 4'})
    samples = list(GeochemistryIndex(project).samples)
    report = compare_samples(project, samples[:5], samples[5:], [saturates.id, sulfur.id])
    run = comparison_prediction(report, sulfur.id, well_cv=True)
    assert run['train_ids'] == samples[:5]
    assert [item['sample_id'] for item in run['predictions']] == samples[5:]
    # Independent closed-form OLS on the printed decimals gives a training mean
    # of 0.096% and slope (3006 / 164996) / 10 in the original units.
    expected = [0.082044534413, 0.050161943320, 0.071842105263, 0.094615384615]
    assert [item['predicted'] for item in run['predictions']] == pytest.approx(expected, abs=1e-11)
    assert run['metrics']['RMSE'] == pytest.approx(0.135456338723, abs=1e-11)
    assert run['baseline']['metrics']['RMSE'] == pytest.approx(0.111763142404, abs=1e-11)
    assert run['predictions'][1]['outside_train_range'] == [saturates.id]
    assert len(run['well_cv']['folds']) == 5
    record = archive_model_run(project, report, run)
    assert replay_model_run(record, project.view[MODEL_RUNS_KEY]['snapshots'][record['snapshot_id']])


def test_linked_source_records_to_well_validated_prediction_in_another_project(tmp_path):
    name = Column('Sample / Name')
    well = Column('Well / Well Name')
    predictor = Column('Predictor', 'number', unit='%wt')
    target = Column('Target', 'number', unit='mg/g')
    columns = [name, well, predictor, target]
    endpoint = {'host': 'example.test', 'port': 5432, 'database': 'geochem'}
    rows = []
    for i in range(8):
        common = {name.id: f'Sample {i}', well.id: f'W{i // 3 + 1}'}
        source = {'sample_id': i, 'sample_type': 'rock', 'preparation': 'whole'}
        rows.append(Row({**common, predictor.id: i, target.id: None},
                        source={**source, 'source_record_id': 2 * i}))
        rows.append(Row({**common, predictor.id: None, target.id: 2 * i + 1},
                        source={**source, 'source_record_id': 2 * i + 1}))
    project = Project(columns=columns, rows=rows, source={'endpoint': endpoint})
    source_path = tmp_path / 'source.apyro'
    project.save(source_path)
    project = Project.load(source_path)

    index = GeochemistryIndex(project)
    samples = list(index.samples)
    assert len(project.rows) == 16 and len(samples) == 8
    assert index.catalog([project.rows[0].id])[0]['row_ids'] == [row.id for row in project.rows[:2]]
    report = compare_samples(project, samples[:6], samples[6:], [predictor.id, target.id])
    result = comparison_prediction(report, target.id, well_cv=True)
    assert [row['predicted'] for row in result['predictions']] == pytest.approx([13, 15])
    assert [fold['train_means'] for fold in result['well_cv']['folds']] == [[4], [1]]
    assert result['metrics']['RMSE'] == pytest.approx(0, abs=1e-12)
    assert result['baseline']['metrics']['RMSE'] == pytest.approx(65**.5)
    run = archive_model_run(project, report, result)
    snapshot = project.view[MODEL_RUNS_KEY]['snapshots'][run['snapshot_id']]
    payload = json.loads(json.dumps({'run': run, 'snapshot': snapshot}, allow_nan=False))

    # The receiving project has a matching predictor but has never measured the target.
    destination = Project(columns=deepcopy(columns[:3]), rows=[
        Row({name.id: 'New sample', well.id: 'W4', predictor.id: 10},
            source={'sample_type': 'rock', 'preparation': 'whole'})
    ])
    assert install_model_run(destination, *verify_model_run_package(payload))
    path = tmp_path / 'destination.apyro'
    destination.save(path)
    restored = Project.load(path)
    archive = restored.view[MODEL_RUNS_KEY]
    imported = archive['runs'][0]
    frozen = archive['snapshots'][imported['snapshot_id']]
    assert replay_model_run(imported, frozen)
    application = apply_prediction(restored, imported, frozen, list(GeochemistryIndex(restored).samples))
    assert application['results'][0]['predicted'] == pytest.approx(21)
    assert application['results'][0]['outside_train_range'] == [predictor.id]
    assert target.id not in restored.rows[0].values
