"""Compare scientific outcomes with the former deepcopy path in both cores."""
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict
import sqlite3

import pytest

from autopyro import importing
from autopyro.database import refresh
from autopyro.database.sources import read_snapshot
from autopyro.importing import digest
from autopyro.io import build_project, read_sqlite


def measurement_state(project):
    """Compare analytical content, allowing fresh result/source UUIDs and timestamps."""
    results = []
    for row in project.rows:
        for column in project.columns:
            current = project.science.current(row.id, column.id)
            for measurement in project.science.results(row.id, column.id):
                analysis = project.science.analyses[measurement.analysis_id]
                source = project.science.sources.get(measurement.source_id)
                results.append((row.id, column.id, measurement.raw_value, measurement.value,
                    measurement.qualifier, measurement.unit, measurement.basis, measurement.uncertainty,
                    measurement.revision, measurement.history, measurement is current,
                    analysis.method, analysis.laboratory, source.original if source else None,
                    source.provenance if source else None))
    return results


@pytest.mark.parametrize("kind", ["import", "refresh"])
@pytest.mark.parametrize("decision", ["keep", "source"])
def test_core_results_and_input_isolation_match_legacy_copy(tmp_path, monkeypatch, kind, decision):
    path = tmp_path / "lab.sqlite"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE lab (Sample TEXT PRIMARY KEY, TOC TEXT, S2 TEXT)")
        connection.executemany("INSERT INTO lab VALUES (?,?,?)", [("001", "2", "6"), ("002", "3", "ND")])
        connection.commit()
    headers, rows, numbers = read_sqlite(path, "lab")
    selections = [(0, "Sample", "text", ""), (1, "TOC", "number", "%wt"), (2, "S2", "number", "mg HC/g rock")]
    project = build_project(headers, rows, numbers, selections,
        {"type": "sqlite", "path": str(path), "table": "lab", "source_columns": headers, "read_only": True})
    project.sync_measurements()
    sid, pid = project.rows[0].id, project.columns[2].id
    repeat = project.science.add_result(sid, pid, 7, analysis_fields={"laboratory": "Local lab"},
                                       details={"uncertainty": .2})
    project.select_measurement(sid, pid, repeat.id)
    repeat.history.append({"previous": {"notes": ["original"]}})
    before = deepcopy(asdict(project))
    if kind == "import":
        module = importing
        incoming = build_project(headers, [["001", "2", "8"], ["002", "3", "<0.2"]], [2, 3], selections,
                                 {"sha256": "second", "path": "repeat.csv"})

        def preview():
            return importing.preview_import(project, incoming, importing.suggest_mapping(project, incoming),
                                            [incoming.columns[0].id])

        def prepare(plan):
            return importing.prepare_import(project, incoming, plan,
                working_policy="keep" if decision == "keep" else "replace")
    else:
        module = refresh
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("UPDATE lab SET S2='8' WHERE Sample='001'")
            connection.execute("UPDATE lab SET S2='<0.2' WHERE Sample='002'")
            connection.commit()
        incoming = read_snapshot(project.source)
        monkeypatch.setattr(refresh, "timestamp", lambda: "2026-11-03T00:00:00+00:00")

        def preview():
            return refresh.preview_refresh(project, incoming, [project.columns[0].id])

        def prepare(plan):
            choices = {entry["id"]: decision for entry in plan.cells if entry["status"] == "conflict"}
            return refresh.prepare_refresh(project, incoming, plan, choices)

    incoming_before, source_bytes = digest(incoming), path.read_bytes()
    fast_plan = preview()
    fast, _ = prepare(fast_plan)
    monkeypatch.setattr(module, "copy_project", deepcopy)
    legacy_plan = preview()
    legacy, _ = prepare(legacy_plan)
    assert asdict(fast_plan) == asdict(legacy_plan)
    assert [row.values for row in fast.rows] == [row.values for row in legacy.rows]
    assert measurement_state(fast) == measurement_state(legacy)
    assert asdict(project) == before and digest(incoming) == incoming_before
    fast.science.measurements[repeat.id].history[0]["previous"]["notes"].append("changed")
    assert repeat.history[0]["previous"]["notes"] == ["original"]
    assert legacy.science.measurements[repeat.id].history[0]["previous"]["notes"] == ["original"]
    assert path.read_bytes() == source_bytes
