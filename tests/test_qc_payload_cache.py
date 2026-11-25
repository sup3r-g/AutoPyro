"""QC decisions must become stale when shared metadata is edited."""
import pytest

from autopyro.analysis.qc import run_checks, snapshot_for_checks
from autopyro.model.core import Column, Project, Row


def test_decision_saved_by_rc2_is_recognized():
    column = Column("TOC", "number", id="toc", unit="%wt")
    project = Project(id="qc-fixture", columns=[column], rows=[Row({"toc": 101}, id="r1")])
    project.sync_measurements()
    project.science.current("r1", "toc").created_at = "2026-01-01T00:00:00+00:00"
    project.view["qc_profiles"] = {"pyrolysis": {"enabled": True}}
    # Recorded using the packaged 0.1.1rc2 QC implementation on this input.
    decision_id = ("profile_absolute:21fad6b1-c8bb-5b0d-a464-ee04749dac48:"
                   "418a83f59869da74f1f3b4f5b2ec4a714a04f3031ec94d8bfab85a023d3c50e5")
    project.view["qc_decisions"] = {decision_id: {"action": "acknowledged", "comment": "Reviewed"}}
    finding, = (item for item in run_checks(project) if item["rule_id"] == "profile_absolute")
    assert (finding["id"], finding["decision"], finding["comment"]) == (decision_id, "acknowledged", "Reviewed")


@pytest.mark.parametrize("edited", ["column", "analysis"])
def test_shared_payload_edits_invalidate_decisions_between_checks(edited):
    column = Column("TOC", "number", unit="%wt", id="toc")
    project = Project(columns=[column], rows=[Row({"toc": 101}, id="r1")])
    project.sync_measurements()
    initial = run_checks(project)
    project.view["qc_decisions"] = {
        issue["id"]: {"action": "acknowledged", "comment": "Reviewed"}
        for issue in initial
    }
    assert all(issue["decision"] == "acknowledged" for issue in run_checks(project))
    if edited == "column":
        column.metadata["notes"] = {"source": ["Updated declaration"]}
    else:
        measurement = project.science.current("r1", "toc")
        project.science.analyses[measurement.analysis_id].preparation = "Updated preparation"
    updated = run_checks(project)
    assert len(updated) == len(initial)
    assert {i["id"] for i in initial}.isdisjoint(i["id"] for i in updated)
    assert all(issue["decision"] == "open" for issue in updated)


def test_qc_snapshot_preserves_findings_and_isolates_inputs():
    column = Column("TOC", "number", unit="%wt", id="toc")
    project = Project(columns=[column], rows=[Row({"toc": 101}, id="r1")])
    project.sync_measurements()
    project.view["qc_profiles"] = {"pyrolysis": {"enabled": True}}
    project.view["unrelated"] = {"large_artifact": ["skip"]}
    project.event("unrelated")
    measurement = project.science.current("r1", "toc")
    measurement.history.append({"previous": {"value": 100}})
    initial = run_checks(project)
    project.view["qc_decisions"] = {initial[0]["id"]: {"action": "acknowledged", "comment": "Reviewed"}}
    frozen = snapshot_for_checks(project, lambda: None)
    expected = run_checks(project, include_peers=True)
    assert run_checks(frozen, include_peers=True) == expected
    assert not frozen.history and "unrelated" not in frozen.view
    measurement.history[0]["previous"]["value"] = 99
    project.view["qc_decisions"][initial[0]["id"]]["comment"] = "Edited"
    assert run_checks(frozen, include_peers=True) == expected
