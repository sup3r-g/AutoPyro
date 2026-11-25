import json
from copy import deepcopy
from pathlib import Path

import pytest
from shapely.geometry import MultiPolygon, Polygon, mapping

from autopyro.scheme_geometry import (add_geometry, add_ring_or_part, geometry_diagnostics,
                                      geometry_paths, remove_ring_or_part, replace_vertices, vertices)


@pytest.fixture
def scheme():
    return json.loads((Path(__file__).parents[1] / "examples" / "interpretation-demo.json").read_text(encoding="utf-8"))


def test_edit_keeps_ring_closed_and_preserves_original(scheme):
    original = deepcopy(scheme["regions"][0])
    path = geometry_paths(original)[0]
    values = vertices(original, path)
    values[0] = [.25, 25]
    edited = replace_vertices(original, path, values)
    assert scheme["regions"][0] == original
    assert edited["geometry"]["coordinates"][0][0] == [.25, 25]
    assert edited["geometry"]["coordinates"][0][-1] == [.25, 25]


def test_holes_and_multiple_parts_remain_separate(scheme):
    entry = scheme["regions"][0]
    path = geometry_paths(entry)[0]
    entry = add_ring_or_part(entry, path, [[.2, 100], [.8, 100], [.8, 200], [.2, 200]], hole=True)
    entry = add_ring_or_part(entry, path, [[5, 0], [6, 0], [6, 50], [5, 50]])
    assert entry["geometry"]["type"] == "MultiPolygon"
    paths = geometry_paths(entry)
    assert [p.indices for p in paths] == [(0, 0), (0, 1), (1, 0)]
    changed_hole = replace_vertices(entry, paths[1], [[.3, 100], [.8, 100], [.8, 200], [.3, 200]])
    assert changed_hole["geometry"]["coordinates"][1] == entry["geometry"]["coordinates"][1]
    no_hole = remove_ring_or_part(changed_hole, geometry_paths(changed_hole)[1])
    assert len(no_hole["geometry"]["coordinates"][0]) == 1
    no_part = remove_ring_or_part(no_hole, geometry_paths(no_hole)[1])
    assert len(no_part["geometry"]["coordinates"]) == 1


def test_shapely_tuple_coordinates_are_editable():
    entry = {"id": "multi", "label": "multi", "geometry": mapping(MultiPolygon([
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(3, 0), (4, 0), (4, 1), (3, 1)])]))}
    path = geometry_paths(entry)[1]
    result = replace_vertices(entry, path, [[3, 0], [4, 0], [4, 2], [3, 2]])
    assert result["geometry"]["coordinates"][1][0][2] == [4, 2]


def test_line_parts_and_minimum_vertices(scheme):
    result = add_geometry(scheme, "LineString", [[0, 0], [1, 1]], "line", "Line")
    line = result["lines"][0]
    extended = add_ring_or_part(line, geometry_paths(line)[0], [[3, 3], [4, 4]])
    assert extended["geometry"]["type"] == "MultiLineString"
    assert len(geometry_paths(extended)) == 2
    reduced = remove_ring_or_part(extended, geometry_paths(extended)[1])
    assert len(reduced["geometry"]["coordinates"]) == 1
    with pytest.raises(ValueError, match="минимум"):
        replace_vertices(line, geometry_paths(line)[0], [[1, 1]])


def test_overlap_warning_does_not_repair_geometry(scheme):
    scheme["regions"][1]["geometry"] = mapping(Polygon([(1, 0), (3, 0), (3, 500), (1, 500)]))
    before = deepcopy(scheme)
    result = geometry_diagnostics(scheme)
    assert result["errors"] == []
    assert result["overlaps"][0]["area"] == 500
    assert scheme == before


def test_invalid_polygon_is_reported_without_repair(scheme):
    scheme["regions"][0]["geometry"] = mapping(Polygon([(0, 0), (1, 1), (0, 1), (1, 0)]))
    original = deepcopy(scheme)
    assert geometry_diagnostics(scheme)["errors"]
    assert scheme == original


def test_touching_boundaries_are_not_area_overlaps(scheme):
    assert geometry_diagnostics(scheme) == {"errors": [], "warnings": [], "overlaps": []}


def test_coverage_counts_holes_and_outside_area_without_double_counting(scheme):
    scheme['x'].update(minimum=0, maximum=4)
    scheme['y'].update(minimum=0, maximum=500)
    scheme['regions'][0]['geometry'] = mapping(Polygon(
        [(-1, 0), (2, 0), (2, 500), (-1, 500)],
        holes=[[(0.5, 100), (1.5, 100), (1.5, 200), (0.5, 200)]],
    ))
    scheme['regions'][1]['geometry'] = mapping(Polygon(
        [(1.75, 0), (3, 0), (3, 500), (1.75, 500)],
    ))
    result = geometry_diagnostics(scheme)
    assert result['overlaps'][0]['area'] == pytest.approx(125)
    assert result['coverage']['domain'].area == pytest.approx(2000)
    assert result['coverage']['gaps'].area == pytest.approx(600)
    assert len(result['coverage']['gaps'].geoms) == 2
    assert result['coverage']['outside'].area == pytest.approx(500)


def test_coverage_uses_transformed_axis_limits(scheme):
    scheme['x'].update(minimum=1, maximum=100, transform='log10', origin=0, scale=2)
    scheme['y'].update(minimum=0, maximum=500)
    result = geometry_diagnostics(scheme)
    assert result['coverage']['domain'].bounds == (0, 0, 1, 500)
    assert result['coverage']['gaps'].is_empty
    scheme['x']['minimum'] = 0
    assert 'coverage' not in geometry_diagnostics(scheme)


@pytest.mark.parametrize("values", [[[0, 0], [1, 1]], [[0, 0], [1, 1], [float("nan"), 2]],
                                    [[0, 0], [1, 1], [True, 2]]])
def test_invalid_vertices_are_rejected(scheme, values):
    entry = scheme["regions"][0]
    with pytest.raises(ValueError):
        replace_vertices(entry, geometry_paths(entry)[0], values)
