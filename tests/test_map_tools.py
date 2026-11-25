from copy import deepcopy

import numpy as np
import pytest
from shapely.geometry import Point, Polygon, mapping

from autopyro.core import Project
from autopyro.io import build_project
from autopyro.mapping import MapSpec, build_map, render_map, layer_categories
from autopyro.map_tools import filtered_features, numeric_layer_style, assign_regions, renderer_for


def layer(geometries, **options):
    result = dict(id='layer', name='Layer', crs='EPSG:32631', visible=True,
                  color='#333333', fill='#aaccee', alpha=.7, line_width=1,
                  features=[dict(id=str(i), type='Feature', geometry=mapping(geometry),
                                 properties={'name': f'Area {i}', 'value': [1, 10, -1, None][i % 4]})
                            for i, geometry in enumerate(geometries)])
    result.update(options)
    return result


def test_filters_preserve_sources_and_category_palette():
    source = layer([Point(i, i) for i in range(4)], color_by='name')
    before = deepcopy(source)
    colors = layer_categories(source)
    source['filter'] = dict(field='value', op='gt', value='2')
    assert [f['id'] for f in filtered_features(source)] == ['1']
    assert layer_categories(source) == colors
    source['filter'] = dict(field='name', op='contains', value='AREA 2')
    assert [f['id'] for f in filtered_features(source)] == ['2']
    source['filter'] = dict(field='value', op='isnull')
    assert [f['id'] for f in filtered_features(source)] == ['3']
    assert source['features'] == before['features']


def test_numeric_layer_log_scale_missing_and_manual_limits():
    source = layer([Point(i, i) for i in range(4)], color_by='value', color_mode='numeric',
                   color_scale='log', color_min=1, color_max=10, missing_color='#ff0000')
    colors, mapper = numeric_layer_style(source, source['features'])
    assert colors['2'] == colors['3'] == (1, 0, 0, 1)
    assert colors['0'] != colors['1']
    assert float(mapper.norm(np.sqrt(10))) == pytest.approx(.5)
    source['color_min'] = 0
    with pytest.raises(ValueError):
        numeric_layer_style(source, source['features'])


def test_spatial_assignment_flags_holes_boundaries_and_overlaps():
    project = build_project(['X', 'Y'], [[1, 1], [0, 5], [9, 9], [3, 3], [20, 20]], list(range(5)),
                            [(0, 'X', 'number', 'm'), (1, 'Y', 'number', 'm')], {})
    spec = MapSpec(x_id=project.columns[0].id, y_id=project.columns[1].id,
                   source_crs='EPSG:32631', target_crs='EPSG:32631')
    source = layer([Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], holes=[[(2, 2), (4, 2), (4, 4), (2, 4)]]),
                    Polygon([(8, 8), (12, 8), (12, 12), (8, 12)])])
    report = build_map(project, [r.id for r in project.rows], spec, [source])
    before = deepcopy(report)
    records, diagnostics = assign_regions(report, 'layer', ['name'])
    assert [records[r.id]['status'] for r in project.rows] == ['matched', 'boundary', 'overlap', 'outside', 'outside']
    assert records[project.rows[0].id]['values']['name'] == 'Area 0'
    assert all(records[row.id]['values']['name'] is None for row in project.rows[1:])
    assert diagnostics == dict(matched=1, boundary=1, overlap=1, outside=2)
    assert report == before


def test_boundary_takes_priority_over_overlap_and_keeps_unlocated_rows():
    from shapely.geometry import box

    source = layer([box(0, 0, 10, 10), box(5, 5, 15, 15)])
    report = {
        'layers': [source], 'spec': {'target_crs': source['crs']},
        'row_ids': ['edge', 'same-point', 'overlap', 'missing'],
        'points': [
            {'x': 5, 'y': 6, 'row_ids': ['edge', 'same-point']},
            {'x': 6, 'y': 6, 'row_ids': ['overlap']},
        ],
    }
    records, counts = assign_regions(report, source['id'], ['name'])
    assert [record['status'] for record in records.values()] == [
        'boundary', 'boundary', 'overlap', 'no_location',
    ]
    assert counts == {'boundary': 2, 'overlap': 1, 'no_location': 1}
    assert all(record['values'] == {'name': None} for record in records.values())


def test_layer_order_filter_labels_and_legend_controls():
    source = layer([Point(0, 0), Point(10, 10)], label_by='name', color_by='name', legend_hidden=False,
                   filter=dict(field='value', op='gt', value=2))
    bottom = layer([Point(5, 5)], id='bottom')
    report = build_map(Project(), [], MapSpec(location_mode='layers', legend_title='Geography'), [source, bottom])
    figure, result = render_map(report)
    assert [fid for lid, fid, _ in result['layer_features'] if lid == 'layer'] == ['1']
    assert result['layer_features'][-1][0] == 'layer'  # First table row is rendered last/on top.
    assert len(result['axes'].get_legend().get_texts()) == 1
    assert result['axes'].get_legend().get_title().get_text() == 'Geography'
    assert len(result['layer_labels']) == 1
    figure.clear()


@pytest.mark.parametrize('target', ['EPSG:4326', 'EPSG:32631', 'EPSG:5940'])
def test_cartographic_decorations_in_multiple_crs(target):
    source = layer([Point(3, 60), Point(4, 61)], crs='EPSG:4326')
    report = build_map(Project(), [], MapSpec(location_mode='layers', target_crs=target,
                                             scale_bar=True, north_arrow=True), [source])
    figure, result = render_map(report)
    assert result['decorations']['scale_m'] > 0
    assert np.linalg.norm(result['decorations']['north_direction']) == pytest.approx(1)
    assert any(text.get_text().endswith((' km', ' m')) for text in result['axes'].texts)
    figure.clear()


def test_colliding_layer_labels_are_placed_without_overlap():
    source = layer([Point(0, 0) for _ in range(20)], label_by='name')
    report = build_map(Project(), [], MapSpec(location_mode='layers'), [source])
    figure, result = render_map(report)
    renderer = renderer_for(figure)
    boxes = [text.get_window_extent(renderer) for text in result['layer_labels']]
    assert boxes
    assert len(boxes) < 20
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:])
    figure.clear()
