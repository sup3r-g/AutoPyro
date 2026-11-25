"""Actual NPD workbook + Norwegian wells + T3, with independent tabular checks."""
import hashlib
import json
from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from autopyro.io import read_file, build_project, file_source
from autopyro.mapping import MapSpec, read_layer, build_map
from autopyro.plotting import category_key
from autopyro.regional_comparison import compare_regions, freeze_regional_result
from autopyro.regional_report import export_paired_pdf, export_pairs_csv
from autopyro.regional_geochemistry import RegionalOptions, analyze_regions, export_regional_summary, render_distributions


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / 'examples'


@pytest.mark.skipif(not (EXAMPLES / 'npd.xlsx').exists(), reason='Local NPD examples required')
def test_npd_regional_statistics(tmp_path):
    paths = [EXAMPLES / 'npd.xlsx', *sorted((EXAMPLES / 'maps').glob('*'))]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    headers, rows, numbers = read_file(EXAMPLES / 'npd.xlsx')
    wanted = ['Well', 'Age', 'Depth', 'TOC', 'HI', 'Tmax', 'Method TOC', 'Method Pyrolysis']
    # Preserve qualifiers and non-numeric values literally; analysis parses numbers explicitly.
    project = build_project(headers, rows, numbers, [(headers.index(name), name, 'text', '') for name in wanted], file_source(EXAMPLES / 'npd.xlsx'))
    ids = {c.name: c.id for c in project.columns}
    regions = read_layer(EXAMPLES / 'maps' / 'T3.shp')
    wells = read_layer(EXAMPLES / 'maps' / 'Norwegian_Wells.shp')
    report = build_map(project, [r.id for r in project.rows], MapSpec(location_mode='layer', sample_key=ids['Well'], point_layer=wells['id'], layer_key='wlbWellbor', target_crs='EPSG:5940'), [regions, wells])
    options = RegionalOptions(regions['id'], 'Paleogeogr', toc_id=ids['TOC'], hi_id=ids['HI'], tmax_id=ids['Tmax'],
        well_id=ids['Well'], age_id=ids['Age'], depth_id=ids['Depth'], toc_method_id=ids['Method TOC'], pyrolysis_method_id=ids['Method Pyrolysis'], ages=('T3n-T3r',))
    result = analyze_regions(project, report, options)
    assert result['diagnostics']['scope_rows'] == 32651
    assert result['diagnostics']['filtered_rows'] == 1841
    assert result['diagnostics']['spatial']['no_location'] == 22
    unlocated = [issue for issue in result['spatial_issues'] if issue['status'] == 'no_location']
    assert len(unlocated) == 22 and all(issue['key'] for issue in unlocated)
    # Independent polygon containment and pandas groupby, independent from assign_regions.
    import geopandas as gpd
    frame = pd.DataFrame([{**{name: r.values[ids[name]] for name in wanted}, 'row_id': r.id} for r in project.rows])
    points = gpd.read_file(EXAMPLES / 'maps' / 'Norwegian_Wells.shp').to_crs('EPSG:5940')
    polygons = gpd.read_file(EXAMPLES / 'maps' / 'T3.shp')
    locations = {}
    for _, well in points.iterrows():
        if well.geometry is None or well.geometry.is_empty:
            continue
        matches = polygons[polygons.geometry.contains(well.geometry)]
        if len(matches) == 1:
            locations[well['wlbWellbor']] = matches.iloc[0]['Paleogeogr']
    frame['region'] = frame.Well.map(locations)
    frame = frame[(frame.Age == 'T3n-T3r') & frame.region.notna()]
    assert len(frame) == len(result['assigned_row_ids'])
    for row in result['summary']:
        method_col = 'Method TOC' if row['metric'] == 'TOC' else 'Method Pyrolysis'
        group = frame[(frame.region == row['region']) & (frame[method_col].fillna('') == (row['method'] or ''))]
        values = pd.to_numeric(group[row['metric']].str.replace(',', '.', regex=False), errors='coerce').dropna()
        values = values[np.isfinite(values)]
        assert row['n_total'] == len(group)
        assert row['n_valid'] == len(values)
        if len(values):
            assert row['mean'] == pytest.approx(values.mean())
            assert row['median'] == pytest.approx(values.median())
    export_regional_summary(tmp_path / 'summary.csv', result)
    (tmp_path / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for metric in ['TOC', 'HI', 'Tmax']:
        render_distributions(result, metric).savefig(tmp_path / f'{metric}.png', dpi=130)
    method = next(r['method'] for r in result['summary'] if r['metric'] == 'TOC' and r['n_valid'])
    filtered = analyze_regions(project, report, replace(options, depth_min=2000, depth_max=3500, toc_methods=(method,), by_well=True))
    eligible = frame[pd.to_numeric(frame.Depth, errors='coerce').between(2000, 3500)]
    assert len(eligible) == len(filtered['assigned_row_ids'])
    expected = eligible['Method TOC'].isna() if method is None else eligible['Method TOC'] == method
    assert sum(r['n_total'] for r in filtered['summary'] if r['metric'] == 'TOC') == sum(expected)
    for row in filtered['summary']:
        if row['metric'] != 'TOC':
            continue
        group = eligible.loc[expected & eligible.region.eq(row['region'])].copy()
        values = pd.to_numeric(group.TOC.str.replace(',', '.', regex=False), errors='coerce')
        group['value'] = values.where(np.isfinite(values))
        medians = group.groupby('Well')['value'].median().dropna()
        assert row['n_observations'] == len(medians)
        if len(medians):
            assert row['mean'] == pytest.approx(medians.mean())
    export_regional_summary(tmp_path / 'filtered-summary.csv', filtered)
    snapshots = [freeze_regional_result(item, report, project.columns) for item in (result, filtered)]
    comparison = compare_regions(*snapshots, 'TOC', category_key(method), category_key(method), paired=True)
    assert any(row['paired']['n'] > 0 for row in comparison)
    def toc_medians(samples):
        method_matches = samples['Method TOC'].isna() if method is None else samples['Method TOC'].eq(method)
        samples = samples.loc[method_matches].copy()
        values = pd.to_numeric(samples.TOC.str.replace(',', '.', regex=False), errors='coerce')
        samples['value'] = values.where(np.isfinite(values))
        return samples.groupby(['region', 'Well'])['value'].median().dropna()

    differences = (toc_medians(eligible) - toc_medians(frame)).dropna()
    for row in comparison:
        paired = row['paired']
        expected = differences.xs(row['region']) if row['region'] in differences.index.get_level_values('region') else pd.Series(dtype=float)
        assert paired['n'] == len(expected)
        for pair in paired['wells']:
            assert pair['delta'] == pytest.approx(expected[pair['well']])
        if len(expected):
            assert paired['median'] == pytest.approx(expected.median())
            assert paired['q1'] == pytest.approx(expected.quantile(.25))
            assert paired['q3'] == pytest.approx(expected.quantile(.75))
    pdf_path = tmp_path / 'regional-AB.pdf'
    export_paired_pdf(pdf_path, comparison, snapshots, ('T3', 'T3 · 2000–3500'),
                      (method or 'Метод не задан',) * 2, 'TOC', map_report=None)
    assert pdf_path.read_bytes().startswith(b'%PDF-')
    csv_path = tmp_path / 'regional-AB-pairs.csv'
    export_pairs_csv(csv_path, comparison, 'TOC')
    pairs = pd.read_csv(csv_path, sep=';')
    assert len(pairs) == len(differences)
    assert pairs['delta_b_minus_a'].tolist() == pytest.approx(
        [pair['delta'] for row in comparison for pair in row['paired']['wells']])
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in hashes.items())
