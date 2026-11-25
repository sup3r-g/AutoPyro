import json
from pathlib import Path

import pytest
from shapely.geometry import Polygon, mapping

from autopyro.interpretation import GeometricEngine, InterpretationInput, evaluate_project, replay_report, validate_scheme
from autopyro.io import build_project
from autopyro.plotting import PlotSpec, render_plot


@pytest.fixture
def scheme():
    return json.loads((Path(__file__).parents[1] / "examples" / "interpretation-demo.json").read_text(encoding="utf-8"))


def run(scheme, x, y, **kwargs):
    return GeometricEngine().evaluate([InterpretationInput("sample", x, y, **kwargs)], scheme)[0]


def test_inside_boundary_outside_and_missing(scheme):
    assert run(scheme, 1, 200).status == "classified"
    assert run(scheme, 1, 200).regions == ["A"]
    edge = run(scheme, 2, 200)
    assert edge.status == "boundary"
    assert edge.boundary_regions == ["A", "B"]
    assert edge.regions == []
    assert run(scheme, 5, 200).status == "outside"
    assert run(scheme, None, 200).status == "insufficient_data"


def test_region_passport_is_validated_and_shown_from_snapshot(scheme):
    from autopyro.interpretation_explanation import explain_sample

    project = build_project(["TOC", "HI"], [["1", "200"]], [1],
                            [(0, "TOC", "number", "%wt"), (1, "HI", "number", "mg HC/g TOC")], {})
    x, y = [column.id for column in project.columns]
    scheme['regions'][0]['passport'] = {'source': 'Тестовый источник', 'limitations': 'Тестовое ограничение'}
    report = evaluate_project(project, project.row_map(), x, y, "%wt", "mg HC/g TOC", scheme)
    snapshot = json.loads(json.dumps(report))
    assert 'Тестовый источник' in explain_sample(snapshot, 0)
    assert 'Тестовое ограничение' in explain_sample(snapshot, 0)
    scheme['regions'][0]['passport']['source'] = 1
    with pytest.raises(ValueError, match="паспорт"):
        validate_scheme(scheme)


def test_boundary_policy_and_overlap_are_explicit(scheme):
    scheme["boundary_policy"] = "include"
    result = run(scheme, 2, 200)
    assert result.status == "ambiguous"
    assert result.regions == ["A", "B"]
    scheme["boundary_policy"] = "exclude"
    assert run(scheme, 2, 200).regions == []
    scheme["regions"][1]["geometry"] = mapping(Polygon([(1, 0), (3, 0), (3, 500), (1, 500)]))
    assert run(scheme, 1.5, 200).status == "ambiguous"


def test_revision_comparison_detects_new_overlap_and_matches_samples_by_uuid(scheme):
    from copy import deepcopy

    from autopyro.schemes.interpretation_revisions import revision_changes

    project = build_project(["TOC", "HI"], [["1.5", "200"], ["3", "200"], ["3.5", "200"]], [1, 2, 3],
                            [(0, "TOC", "number", "%wt"), (1, "HI", "number", "mg HC/g TOC")], {})
    x, y = [column.id for column in project.columns]
    shared, removed, added = [row.id for row in project.rows]
    before = evaluate_project(project, [shared, removed], x, y, "%wt", "mg HC/g TOC", scheme)
    changed = deepcopy(scheme)
    changed["revision"] = "2"
    changed["regions"][1]["geometry"] = mapping(Polygon([(1, 0), (3, 0), (3, 500), (1, 500)]))
    after = evaluate_project(project, [added, shared], x, y, "%wt", "mg HC/g TOC", changed)
    changes = {change.row_id: change.flags for change in revision_changes(project, before, after)}
    assert changes == {shared: ['outcome', 'ambiguity_added'], removed: ['removed'], added: ['added']}
    reversed_changes = {change.row_id: change.flags for change in revision_changes(project, after, before)}
    assert reversed_changes[shared] == ['outcome', 'ambiguity_removed']


def test_polygon_holes_are_not_classified_as_inside(scheme):
    scheme["regions"] = [{"id": "hole", "label": "Hole", "geometry": mapping(
        Polygon([(0, 0), (4, 0), (4, 500), (0, 500)], holes=[[(1, 100), (2, 100), (2, 200), (1, 200)]]))}]
    assert run(scheme, 1.5, 150).status == "outside"
    assert run(scheme, 1, 150).status == "boundary"


def test_invalid_geometries_are_rejected_without_repair(scheme):
    scheme["regions"][0]["geometry"] = mapping(Polygon([(0, 0), (2, 2), (0, 2), (2, 0)]))
    with pytest.raises(ValueError, match="геометрия"):
        validate_scheme(scheme)


def test_log_coordinates_metric_and_applicability(scheme):
    scheme["regions"] = scheme["regions"][:1]
    scheme["x"]["transform"] = "log10"
    scheme["x"]["reference"] = 1
    assert run(scheme, 10, 200).status == "classified"
    assert run(scheme, 0, 200).status == "not_applicable"
    assert run(scheme, 10, 200).distances == {}
    scheme["distance_metric"] = "analysis_euclidean"
    with pytest.raises(ValueError, match="нормировку"):
        validate_scheme(scheme)
    scheme["metric_note"] = "Synthetic example only; analysis coordinates explicitly chosen"
    assert run(scheme, 10, 200).distances["A"] == pytest.approx(1)
    scheme["x"]["maximum"] = 5
    assert run(scheme, 10, 200).status == "not_applicable"


def test_uncertainty_crossing_boundary_and_outside_overlap(scheme):
    assert run(scheme, 1.9, 200, x_bounds=(1.8, 2.1), y_bounds=(190, 210)).status == "ambiguous"
    assert run(scheme, 4.1, 200, x_bounds=(3.9, 4.2)).status == "ambiguous"
    assert run(scheme, 1, 200, x_bounds=(.9, 1.1), y_bounds=(190, 210)).status == "classified"
    assert run(scheme, 1, 200, x_bounds=(1.1, 1.2)).status == "not_applicable"


def test_tolerance_and_lines(scheme):
    scheme["tolerance"] = .01
    assert run(scheme, 2.005, 200).status == "boundary"
    scheme["lines"] = [{"id": "line", "label": "Reference", "geometry": {"type": "LineString", "coordinates": [[0, 100], [4, 100]]}}]
    assert run(scheme, 1, 100).line_hits == ["line"]


def test_threshold_uses_interior_vertices_and_marks_domain_limits(scheme):
    scheme["regions"] = []
    scheme["lines"] = [{
        "id": "cutoff", "label": "Порог",
        "geometry": {"type": "LineString", "coordinates": [[4, 100], [2, 300], [0, 100]]},
        "threshold": {"axis": "y", "less": "Низкий", "greater": "Высокий"},
    }]
    scheme["tolerance"] = 5
    assert run(scheme, 1, 206).status == "classified"
    assert run(scheme, 1, 205).line_hits == ["cutoff"]
    result = run(scheme, 1, 225, x_bounds=(0, 4))
    assert result.status == "ambiguous"
    assert result.thresholds["cutoff"] == {
        "relation": "greater", "possible_relations": ["less", "boundary", "greater"],
    }
    result = run(scheme, 0, 200, x_bounds=(-1, 1))
    assert result.status == "ambiguous"
    assert "outside_domain" in result.thresholds["cutoff"]["possible_relations"]
    assert run(scheme, 5, 200).status == "not_applicable"


def test_threshold_x_rejects_ambiguous_functions_and_replays(scheme):
    scheme["regions"] = []
    line = {
        "id": "cutoff", "label": "Порог",
        "geometry": {"type": "LineString", "coordinates": [[1, 100], [2, 100]]},
        "threshold": {"axis": "x", "less": "Левее", "greater": "Правее"},
    }
    scheme["lines"] = [line]
    with pytest.raises(ValueError, match="однозначной функцией"):
        validate_scheme(scheme)
    line["geometry"]["coordinates"] = [[2, 100], [2, 300]]
    project = build_project(["TOC", "HI"], [["3", "200"]], [1],
                            [(0, "TOC", "number", "%wt"), (1, "HI", "number", "mg HC/g TOC")], {})
    x, y = [column.id for column in project.columns]
    report = evaluate_project(project, project.row_map(), x, y, "%wt", "mg HC/g TOC", scheme)
    assert report["results"][0]["thresholds"]["cutoff"]["relation"] == "greater"
    assert replay_report(json.loads(json.dumps(report))) == report["results"]


def test_unit_binding_qualifiers_and_overlay(scheme):
    project = build_project(["TOC", "HI"], [["1", "200"], ["<0.1", "300"]], [1, 2],
                            [(0, "TOC", "text", "%wt"), (1, "HI", "number", "mg HC/g TOC")], {})
    x, y = [c.id for c in project.columns]
    report = evaluate_project(project, project.row_map(), x, y, "%wt", "mg HC/g TOC", scheme, text_numbers=True)
    assert [r["status"] for r in report["results"]] == ["classified", "insufficient_data"]
    assert replay_report(report) == report["results"]
    with pytest.raises(ValueError, match="Единица"):
        evaluate_project(project, project.row_map(), x, y, "g/g rock", "mg HC/g TOC", scheme)
    spec = PlotSpec(x_id=x, y_id=y, text_numbers=True, interpretation={"scheme": scheme, **report["binding"]})
    figure, result = render_plot(project, project.row_map(), spec)
    assert len(figure.axes[0].patches) == 2
    assert result["plotted"][x] == 1
    spec.x_axis.minimum = 0
    spec.x_axis.maximum = 2
    render_plot(project, project.row_map(), spec)
    assert evaluate_project(project, project.row_map(), x, y, "%wt", "mg HC/g TOC", scheme, text_numbers=True)["results"] == report["results"]


@pytest.mark.parametrize("interpolated", [False, True])
def test_batch_axis_inputs_keep_quality_and_exclusion_precedence(scheme, interpolated):
    values = [["1", "200"], ["<0.1", "300"], ["101", "400"], ["0.01", "ND"]]
    rows = values * 75
    project = build_project(["TOC", "HI"], rows, range(len(rows)),
                            [(0, "TOC", "text", "%wt"), (1, "HI", "number", "mg HC/g TOC")], {})
    x, y = [column.id for column in project.columns]
    project.sync_measurements()
    measurement = project.science.current(project.rows[0].id, x)
    project.view["qc_exclusions"] = {"interpretation": {measurement.id: {"revision": measurement.revision}}}
    if interpolated:
        project.columns[0].metadata["method"] = "linear"
    args = (x, y, "%wt", "mg HC/g TOC", scheme)
    scalar = evaluate_project(project, [row.id for row in project.rows[:8]], *args, text_numbers=True)
    batch = evaluate_project(project, project.row_map(), *args, text_numbers=True)
    assert batch["inputs"][:8] == scalar["inputs"]
    assert batch["results"][:8] == scalar["results"]


def test_alternative_engine_can_be_injected(scheme):
    class FakeEngine:
        def evaluate(self, samples, scheme):
            from autopyro.interpretation import InterpretationResult
            return [InterpretationResult(s.row_id, "outside", reason="Injected engine") for s in samples]
    project = build_project(["X", "Y"], [["1", "200"]], [1], [(0, "X", "number", "%wt"), (1, "Y", "number", "mg HC/g TOC")], {})
    for column, role in zip(project.columns, ('TOC', 'HI'), strict=True):
        column.metadata['property_role'] = role
    x, y = [c.id for c in project.columns]
    report = evaluate_project(project, project.row_map(), x, y, "%wt", "mg HC/g TOC", scheme, engine=FakeEngine())
    assert report["results"][0]["reason"] == "Injected engine"
