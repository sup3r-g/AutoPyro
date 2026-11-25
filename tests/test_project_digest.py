"""Compatibility and stale-preview protection for the complete project digest."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import pytest

from autopyro.importing import digest
from autopyro.io import build_project


def legacy_digest(project):
    return hashlib.sha256(json.dumps(asdict(project), ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode("utf-8")).hexdigest()


@pytest.fixture
def project():
    project = build_project(["Sample", "TOC", "S2"],
        [["001", "2,00", "<0.1"], ["001", "3", "ND"]], [2, 3],
        [(0, "Sample", "text", ""), (1, "TOC", "number", "%wt"),
         (2, "S2", "number", "mg HC/g rock")], {"path": "лаборатория.csv", "sha256": "original"})
    project.sync_measurements()
    project.view["unicode"] = {"кириллица": "≈ ≤ ≥ 🌍\n\"\\", "floats": [-0.0, 1e-100, 1e100],
                               "integer": 2**60 + 1, "tuple": (True, None, [1, "001"])}
    sid, pid = project.rows[0].id, project.columns[2].id
    repeat = project.science.add_result(sid, pid, .05,
        analysis_fields={"method": "Pyrol6", "laboratory": "Лаборатория Б"},
        details={"uncertainty": .01, "uncertainty_type": "standard"})
    repeat.history.append({"previous": {"raw_value": "0,04", "value": .04}})
    project._storage_path = Path("private-location.apyro")
    return project


def test_digest_matches_legacy_json_without_mutating_project(project):
    before = asdict(deepcopy(project))
    assert digest(project) == legacy_digest(project)
    project.view = dict(reversed(list(project.view.items())))
    assert digest(project) == legacy_digest(project)
    assert asdict(project) == before


def test_nested_dataclass_and_shared_record_are_encoded_by_declared_fields(project):
    @dataclass
    class Context:
        name: str
        values: tuple
    context = Context("whole rock", (1, {"nested": [2]}))
    context.transient = "not persisted"
    project.view["contexts"] = [context, context]
    assert digest(project) == legacy_digest(project)


@pytest.mark.parametrize("part", ["value", "original", "row_source", "column", "view", "history",
                                  "source", "measurement", "analysis", "working", "sample"])
def test_digest_detects_changes_in_every_scientific_layer(project, part):
    before = digest(project)
    row, column = project.rows[0], project.columns[2]
    measurement = project.science.current(row.id, column.id)
    if part == "value":
        row.values[column.id] = 12
    elif part == "original":
        row.original[column.id] = "different original"
    elif part == "row_source":
        row.source["row"] = 99
    elif part == "column":
        column.metadata["basis"] = "extracted"
    elif part == "view":
        project.view["qc_exclusions"] = {"plots": {measurement.id: {"revision": 1}}}
    elif part == "history":
        project.event("local change")
    elif part == "source":
        project.science.sources[measurement.source_id].provenance["sha256"] = "changed"
    elif part == "measurement":
        measurement.uncertainty = .02
    elif part == "analysis":
        project.science.analyses[measurement.analysis_id].laboratory = "Changed lab"
    elif part == "working":
        repeat = project.science.results(row.id, column.id)[-1]
        project.science.select(row.id, column.id, repeat.id)
    else:
        project.science.samples[row.id].metadata["reviewed"] = True
    assert digest(project) != before
    assert digest(project) == legacy_digest(project)


@pytest.mark.parametrize("value,error", [(float("nan"), ValueError), (float("inf"), ValueError),
                                        (float("-inf"), ValueError), (Path("unsupported"), TypeError)])
def test_digest_rejects_non_json_values_as_before(project, value, error):
    project.view["invalid"] = value
    with pytest.raises(error):
        legacy_digest(project)
    with pytest.raises(error):
        digest(project)
