"""Transactional import and explicit QC decisions across the scientific workflow."""
from copy import deepcopy
from dataclasses import asdict
import json

import pytest
from PySide6.QtCore import Qt

from autopyro.core import Project
from autopyro.io import build_project
from autopyro.importing import preview_import, prepare_import, suggest_mapping
from autopyro.qc import is_excluded, run_checks


def dataset(rows, sha="first", *, extra=False):
    headers = ["Well", "Sample", "TOC", "S2"] + (["S3"] if extra else [])
    units = ["", "", "%wt", "mg HC/g rock"] + (["mg CO2/g rock"] if extra else [])
    selections = [(i, name, "text" if i < 2 else "number", units[i]) for i, name in enumerate(headers)]
    project = build_project(headers, rows, list(range(2, len(rows) + 2)), selections,
                            {"path": f"{sha}.csv", "sha256": sha, "sheet": None})
    project.sync_measurements()
    return project


def planned(project, incoming, **kwargs):
    mapping = suggest_mapping(project, incoming)
    keys = [c.id for c in incoming.columns[:2]]
    return preview_import(project, incoming, mapping, keys, **kwargs)


def test_append_preserves_working_values_and_keeps_new_analysis_source():
    project = dataset([["A", "001", "2", "6"]])
    incoming = dataset([["A", "001", "2", "8"]], "second")
    before = asdict(deepcopy(project))
    plan = planned(project, incoming, defaults={"method": "Pyrol6", "laboratory": "Lab B"})
    assert plan.rows[0]["status"] == "matched"
    assert len(plan.conflicts) == 1
    result, protocol = prepare_import(project, incoming, plan)
    assert asdict(project) == before
    assert result.id == project.id and len(result.rows) == 1
    sid, pid = project.rows[0].id, project.columns[3].id
    assert result.rows[0].values[pid] == 6
    assert len(result.science.results(sid, pid)) == 2
    measurement = result.science.measurements[protocol["rows"][0]["measurement_ids"][-1]]
    assert result.science.analyses[measurement.analysis_id].method == "Pyrol6"
    assert result.science.sources[measurement.source_id].provenance["sha256"] == "second"
    assert result.rows[0].original[pid] == "6"


@pytest.mark.parametrize("policy,old,expected", [("keep", None, None), ("fill_missing", None, 8),
    ("fill_missing", "ND", "ND"), ("fill_missing", "<0.1", "<0.1"), ("replace", "6", 8)])
def test_working_policy_is_explicit_and_never_fills_censoring(policy, old, expected):
    project = dataset([["A", "001", "2", old]])
    incoming = dataset([["A", "001", "2", "8"]], "second")
    result, _ = prepare_import(project, incoming, planned(project, incoming), working_policy=policy)
    assert result.rows[0].values[project.columns[3].id] == expected


def test_reimport_skips_source_even_after_local_value_edit():
    project = dataset([["A", "001", "2", "6"]])
    project.rows[0].values[project.columns[3].id] = 99
    project.refresh()
    incoming = dataset([["A", "001", "2", "6"]])
    plan = planned(project, incoming)
    assert plan.rows[0]["status"] == "duplicate"
    result, protocol = prepare_import(project, incoming, plan)
    assert protocol["skipped"] == 1 and protocol["analyses"] == 0
    assert len(result.science.measurements) == len(project.science.measurements)
    assert result.rows[0].values[project.columns[3].id] == 99


def test_partial_reimport_adds_only_new_fields_without_repeating_existing_results():
    project = dataset([["A", "001", "2", "6"]])
    incoming = dataset([["A", "001", "2", "6", "1.5"]], extra=True)
    plan = planned(project, incoming)
    result, protocol = prepare_import(project, incoming, plan)
    assert protocol["measurements"] == 1
    assert len(result.columns) == 5
    assert result.rows[0].values[result.columns[-1].id] == 1.5
    assert len(result.science.results(result.rows[0].id, project.columns[3].id)) == 1


def test_ambiguous_key_requires_explicit_choice_and_never_mutates_input():
    project = dataset([["A", "001", "2", "6"], ["A", "001", "3", "7"]])
    incoming = dataset([["A", "001", "2", "8"]], "second")
    plan = planned(project, incoming)
    assert plan.rows[0]["status"] == "ambiguous"
    before = asdict(deepcopy(project))
    with pytest.raises(ValueError, match="неоднозначные"):
        prepare_import(project, incoming, plan)
    assert asdict(project) == before
    result, protocol = prepare_import(project, incoming, plan,
        {incoming.rows[0].id: {"action": "append", "target_id": project.rows[1].id}}, working_policy="replace")
    assert len(result.rows) == 2
    assert result.rows[0].values[project.columns[3].id] == 6
    assert result.rows[1].values[project.columns[3].id] == 8


def test_no_key_does_not_merge_equal_names_from_different_files():
    project = dataset([["A", "001", "2", "6"]])
    incoming = dataset([["A", "001", "2", "8"]], "second")
    plan = preview_import(project, incoming, suggest_mapping(project, incoming))
    result, _ = prepare_import(project, incoming, plan)
    assert len(result.rows) == 2


def test_incompatible_units_and_stale_preview_are_rejected():
    project = dataset([["A", "001", "2", "6"]])
    incoming = dataset([["A", "001", "2", "8"]], "second")
    mapping = suggest_mapping(project, incoming)
    plan = preview_import(project, incoming, mapping)
    project.rows[0].values[project.columns[3].id] = 9
    with pytest.raises(ValueError, match="изменились"):
        prepare_import(project, incoming, plan)
    incoming.columns[3].unit = "ug HC/g rock"
    with pytest.raises(ValueError, match="единица"):
        preview_import(project, incoming, mapping)


def test_reimport_recognizes_separate_property_created_for_conflicting_units():
    project = dataset([["A", "001", "2", "6"]])
    incoming = dataset([["A", "001", "2", "8000"]], "second")
    incoming.columns[3].unit = "ug HC/g rock"
    plan = planned(project, incoming)
    result, _ = prepare_import(project, incoming, plan)
    separate = result.columns[-1]
    assert separate.name == "S2 (2)" and separate.unit == "ug HC/g rock"
    repeated = planned(result, incoming)
    assert repeated.mapping[incoming.columns[3].id] == separate.id
    assert repeated.rows[0]["status"] == "duplicate"
    unchanged, protocol = prepare_import(result, incoming, repeated)
    assert len(unchanged.columns) == len(result.columns)
    assert len(unchanged.science.measurements) == len(result.science.measurements)
    assert protocol["skipped"] == 1


def test_import_name_does_not_resolve_ambiguous_properties():
    project = dataset([["A", "001", "2", "6"]])
    incoming = dataset([["A", "001", "2", "8000"]], "second")
    incoming.columns[3].unit = "ug HC/g rock"
    result, _ = prepare_import(project, incoming, planned(project, incoming))
    separate = deepcopy(result.columns[-1])
    from autopyro.core import new_id
    separate.id = new_id()
    separate.name = "S2 alternative"
    result.columns.append(separate)
    for row in result.rows:
        row.values[separate.id] = None
    # Two equally compatible candidates require a user's mapping decision.
    assert suggest_mapping(result, incoming)[incoming.columns[3].id] == "new"


def test_new_columns_do_not_invent_missing_analyses_for_unrelated_samples(tmp_path):
    project = dataset([["A", "001", "2", "6"]])
    incoming = dataset([["B", "002", "3", "7", "1"]], "second", extra=True)
    result, _ = prepare_import(project, incoming, planned(project, incoming))
    new_column = result.columns[-1]
    assert result.rows[0].values[new_column.id] is None
    assert result.science.results(result.rows[0].id, new_column.id) == []
    path = tmp_path / "import.apyro"
    result.save(path)
    assert Project.load(path).science.results(result.rows[0].id, new_column.id) == []


def test_manual_numeric_import_keeps_qualifiers():
    project = dataset([["A", "001", "<0.01", "ND"]])
    assert project.rows[0].values[project.columns[2].id] == "<0.01"
    assert project.science.current(project.rows[0].id, project.columns[2].id).limit == .01


def test_qc_approximate_nominal_values_receive_physical_review():
    project = dataset([["A", "001", "~101", "~-1"], ["A", "002", "~2", "~6"]])
    before = asdict(deepcopy(project))
    issues = run_checks(project)
    findings = [issue for issue in issues if issue["rule_id"] == "approximate_range"]
    assert len(findings) == 2
    assert {issue["severity"] for issue in findings} == {"warning"}
    assert {issue["sample_id"] for issue in findings} == {project.rows[0].id}
    assert not any(issue["rule_id"] == "range" for issue in issues)
    assert asdict(project) == before
    sid, cid = project.rows[1].id, project.columns[3].id
    project.science.add_result(sid, cid, 8)
    assert not any(issue["rule_id"].startswith("repeat") for issue in run_checks(project))


def test_qc_checks_are_read_only_and_not_scientific_classification():
    project = dataset([["A", "001", "101", "-1"]])
    before = asdict(deepcopy(project))
    issues = run_checks(project)
    assert asdict(project) == before
    assert len([i for i in issues if i["rule_id"] == "range"]) == 2
    assert any(i["rule_id"] == "context" for i in issues)
    assert {i["severity"] for i in issues} == {"error", "warning"}


def test_pyrolysis_profile_checks_exact_and_censored_results_and_persists(tmp_path):
    from autopyro.analysis.qc import PROFILE_DEFAULTS, PROFILE_KEY

    project = dataset([["A", "001", "84", "100"],
                       ["A", "002", "85", "1001"],
                       ["A", "003", "<0.1", "<=0.5"],
                       ["A", "004", "<LOD", "ND"]])
    toc, s2 = project.columns[2:4]
    lod = project.science.current(project.rows[3].id, toc.id)
    lod.lod = .2
    assert not any("profile" in issue for issue in run_checks(project))
    project.view[PROFILE_KEY[0]] = {PROFILE_KEY[1]: {"enabled": True, "thresholds": PROFILE_DEFAULTS}}
    profile = {(i["sample_id"], i["property_id"]): i for i in run_checks(project) if "profile" in i}
    first, second, third, fourth = (row.id for row in project.rows)
    assert (first, toc.id) not in profile and (first, s2.id) not in profile
    assert profile[second, toc.id]["rule_id"] == "profile_typical"
    assert profile[second, s2.id]["severity"] == "critical"
    assert profile[third, toc.id]["rule_id"] == "profile_typical"
    assert profile[third, s2.id]["rule_id"] == "profile_ambiguous"
    assert profile[fourth, toc.id]["profile"]["upper"] == .2
    assert (fourth, s2.id) not in profile
    issue = profile[second, toc.id]
    project.view["qc_decisions"] = {issue["id"]: {"action": "acknowledged", "comment": "reviewed"}}
    path = tmp_path / "thresholds.apyro"
    project.save(path)
    project = Project.load(path)
    assert next(i for i in run_checks(project) if i["id"] == issue["id"])["decision"] == "acknowledged"
    project.view[PROFILE_KEY[0]][PROFILE_KEY[1]]["thresholds"]["TOC"][2] = 80
    changed = next(i for i in run_checks(project) if i["rule_id"] == "profile_typical" and i["sample_id"] == second)
    assert changed["id"] != issue["id"] and changed["decision"] == "open"
    assert not project.view.get("qc_exclusions")


def test_qc_profile_checks_nd_using_declared_lod_and_reports_coverage():
    from autopyro.analysis.qc import PROFILE_KEY

    project = dataset([["A", "001", "2", "ND"], ["A", "002", "2", "ND"], ["A", "003", "2", "ND"]])
    cid = project.columns[3].id
    first, boundary, unknown = (project.science.current(row.id, cid) for row in project.rows)
    first.lod = 100
    first.unit = "ug HC/g rock"
    boundary.lod = 0.5
    project.view[PROFILE_KEY[0]] = {PROFILE_KEY[1]: {"enabled": True}}
    before = asdict(deepcopy(project))
    coverage = []
    issues = [issue for issue in run_checks(project, coverage_details=coverage)
              if issue["property_id"] == cid and "profile" in issue]
    findings = {issue["measurement_id"]: issue for issue in issues}
    assert findings[first.id]["rule_id"] == "profile_typical"
    assert findings[first.id]["profile"]["display_value"] == "[0, 0.1]"
    assert findings[boundary.id]["rule_id"] == "profile_ambiguous"
    assert findings[boundary.id]["profile"]["upper_closed"] is True
    assert unknown.id not in findings
    assert (first.sample_id, cid, "checked") in coverage
    assert (unknown.sample_id, cid, "no_bounds") in coverage
    assert asdict(project) == before


def test_pyrolysis_profile_checks_calculated_censored_hi():
    from autopyro.analysis.calculations import make_calculated_column
    from autopyro.analysis.qc import PROFILE_DEFAULTS, PROFILE_KEY, snapshot_for_checks

    project = dataset([["A", "001", "<2", "6"]])
    toc, s2 = project.columns[2:4]
    config = {"formula": "HI", "version": 1, "analysis_context": "test",
              "compatible_confirmed": True,
              "inputs": {"TOC": {"column_id": toc.id, "unit": toc.unit},
                         "S2": {"column_id": s2.id, "unit": s2.unit}}}
    column, values = make_calculated_column(project, "HI (calc)", config)
    project.columns.append(column)
    row = project.rows[0]
    row.values[column.id] = values[row.id]
    project.view[PROFILE_KEY[0]] = {PROFILE_KEY[1]: {"enabled": True, "thresholds": PROFILE_DEFAULTS}}
    coverage = []
    issues = run_checks(project, coverage_details=coverage)
    assert (row.id, column.id, "checked") in coverage
    assert run_checks(snapshot_for_checks(project, lambda: None)) == issues
    issue = next(i for i in issues if i["property_id"] == column.id)
    assert issue["rule_id"] == "profile_ambiguous"
    assert issue["measurement_id"] is None
    assert issue["profile"]["display_value"] == "(300, ∞)"
    assert issue["profile"]["violations"][0] == {
        "rule": "profile_absolute", "direction": "above", "threshold": 1200.0, "whole": False,
    }
    project.view[PROFILE_KEY[0]][PROFILE_KEY[1]]["sample_scope"] = {"name": "Empty selection", "row_ids": []}
    coverage.clear()
    assert not any("profile" in item for item in run_checks(project, coverage_details=coverage))
    assert coverage == []


def test_qc_limit_validation_is_independent_of_profile_and_methodology_is_versioned():
    from autopyro.analysis.qc import PROFILE_KEY

    project = dataset([["A", "001", "2", "<LOD"],
                       ["A", "002", "2", ">1000"],
                       ["A", "003", "2", "<0"]])
    lod, above, impossible = (project.science.current(row.id, project.columns[3].id) for row in project.rows)
    lod.lod = 0.4
    base = run_checks(project)
    assert not any(i["measurement_id"] == lod.id and i["rule_id"] == "value_format" for i in base)
    assert any(i["measurement_id"] == impossible.id and i["rule_id"] == "range" for i in base)
    profile = {"enabled": True, "version": "v1"}
    project.view[PROFILE_KEY[0]] = {PROFILE_KEY[1]: profile}
    issue = next(i for i in run_checks(project) if i["measurement_id"] == above.id and "profile" in i)
    assert issue["severity"] == "critical"
    assert issue["profile"]["display_value"] == "(1000, ∞)"
    project.view["qc_decisions"] = {issue["id"]: {"action": "acknowledged"}}
    profile["version"] = "v2"
    changed = next(i for i in run_checks(project) if i["measurement_id"] == above.id and "profile" in i)
    assert changed["decision"] == "open"


def test_qc_repeat_checks_convert_compatible_units_without_rounding():
    project = dataset([["A", "001", "2", "6"]])
    sid, cid = project.rows[0].id, project.columns[3].id
    repeat = project.science.add_result(sid, cid, 6000)
    repeat.unit = "ug HC/g rock"
    assert not any(i["rule_id"] == "repeat" for i in run_checks(project))
    repeat.value = 6001
    assert len([i for i in run_checks(project) if i["rule_id"] == "repeat"]) == 2


def test_qc_repeat_compares_declared_intervals_in_compatible_units():
    project = dataset([["A", "001", "2", "6"]])
    sid, cid = project.rows[0].id, project.columns[3].id
    original = project.science.current(sid, cid)
    original.uncertainty, original.uncertainty_type = 0.1, "standard"
    repeat = project.science.add_result(sid, cid, 6200,
                                       details={"uncertainty": 100, "uncertainty_type": "std"})
    repeat.unit = "ug HC/g rock"

    def findings():
        return [issue for issue in run_checks(project, include_peers=True) if issue["rule_id"] == "repeat"]

    assert findings() == []  # Both intervals include 6.1 mg/g.
    repeat.uncertainty = 99
    separated = findings()
    assert len(separated) == 2
    assert {issue["repeat_comparison"] for issue in separated} == {"interval"}
    assert {peer["uncertainty"] for peer in separated[0]["peers"]} == {0.1, 99}

    repeat.uncertainty = 100
    repeat.confidence_level = 0.95
    fallback = findings()
    assert {issue["repeat_comparison"] for issue in fallback} == {"exact"}
    assert {issue["id"] for issue in separated}.isdisjoint(issue["id"] for issue in fallback)
    repeat.confidence_level = None
    repeat.uncertainty_type = "expanded"
    assert {issue["repeat_comparison"] for issue in findings()} == {"exact"}
    repeat.uncertainty = None
    assert {issue["repeat_comparison"] for issue in findings()} == {"exact"}


def test_qc_repeat_requires_a_common_interval_for_all_repeats():
    project = dataset([["A", "001", "2", "100"]])
    sid, cid = project.rows[0].id, project.columns[3].id
    original = project.science.current(sid, cid)
    original.uncertainty, original.uncertainty_type = 10, "relative_percent"
    project.science.add_result(sid, cid, 115,
                              details={"uncertainty": 0.1, "uncertainty_type": "relative_fraction"})
    assert not any(issue["rule_id"] == "repeat" for issue in run_checks(project))
    project.science.add_result(sid, cid, 130,
                              details={"uncertainty": 10, "uncertainty_type": "relative_percent"})
    findings = [issue for issue in run_checks(project) if issue["rule_id"] == "repeat"]
    assert len(findings) == 3
    assert {issue["repeat_comparison"] for issue in findings} == {"interval"}


@pytest.mark.parametrize(("left", "right", "lod", "warning"), [
    ("6", "<6", None, True),
    ("6", "<=6", None, False),
    ("6", "<LOD", 6, True),
    ("6", "ND", 6, False),
    ("7", "ND", 6, True),
    ("7", "ND", None, False),
    ("<=6", ">6", None, True),
    ("<=6", ">=6", None, False),
])
def test_qc_compares_repeats_with_censored_limits(left, right, lod, warning):
    project = dataset([["A", "001", "2", left]])
    sid, cid = project.rows[0].id, project.columns[3].id
    original = project.science.current(sid, cid)
    repeat = project.science.add_result(sid, cid, right, details={"lod": lod})
    issues = run_checks(project, include_peers=True)
    findings = [i for i in issues if i["rule_id"] == "repeat_censored"]
    assert len(findings) == (2 if warning else 0)
    if warning:
        assert {i["measurement_id"] for i in findings} == {original.id, repeat.id}
        assert next(peer for peer in findings[0]["peers"] if peer["id"] == repeat.id)["lod"] == lod
    assert original.raw_value == left and repeat.raw_value == right


def test_qc_censored_repeat_respects_units_and_declared_uncertainty():
    project = dataset([["A", "001", "2", "6.2"]])
    sid, cid = project.rows[0].id, project.columns[3].id
    original = project.science.current(sid, cid)
    original.uncertainty, original.uncertainty_type = 0.2, "standard"
    repeat = project.science.add_result(sid, cid, "<=6000")
    repeat.unit = "ug HC/g rock"
    assert not any(i["rule_id"] == "repeat_censored" for i in run_checks(project))
    original.uncertainty = 0.19
    assert len([i for i in run_checks(project) if i["rule_id"] == "repeat_censored"]) == 2


def test_qc_keeps_exact_repeat_warning_when_limit_is_compatible_with_both():
    project = dataset([["A", "001", "2", "6"]])
    sid, cid = project.rows[0].id, project.columns[3].id
    project.science.add_result(sid, cid, "7")
    project.science.add_result(sid, cid, "<8")
    rules = [i["rule_id"] for i in run_checks(project) if i["rule_id"].startswith("repeat")]
    assert rules == ["repeat", "repeat"]


def test_qc_distinguishes_ratios_from_grouped_indicators():
    from autopyro.analysis.qc import role
    from autopyro.model.core import Column

    for label in ("S2/S3", "S2 / S3", "Pyrol6 / S2 / S3", "Pyrol6 / S2/S3", "S0 / S3"):
        assert role(Column(label)) is None
    assert role(Column("Pyrol6 / S3")) == "S3"
    assert role(Column("Laboratory / Pyrol6 / T-Max")) == "TMAX"
    assert role(Column("S2/S3", metadata={"property_role": "S3"})) == "S3"
    assert role(Column("S3", metadata={"property_role": None})) is None

    project = dataset([["A", "001", "2", "-1"], ["A", "002", "2", "<bad"]])
    ratio = project.columns[3]
    ratio.name = "S2/S3"
    ratio.unit = "1"
    for row in project.rows:
        project.science.current(row.id, ratio.id).unit = "1"
    assert [issue["rule_id"] for issue in run_checks(project) if issue["property_id"] == ratio.id] == ["value_format"]


def test_qc_keeps_repeat_order_and_all_peers_when_first_value_is_valid():
    project = dataset([["A", "001", "2", "6"], ["A", "002", "102", "6"]])
    context = {"method": "Pyrolysis", "laboratory": "Lab"}
    for analysis in project.science.analyses.values():
        analysis.method, analysis.laboratory = context["method"], context["laboratory"]
    first, second = project.rows
    toc = project.columns[2].id
    original = project.science.current(first.id, toc)
    invalid = project.science.current(second.id, toc)
    repeat = project.science.add_result(first.id, toc, "101", analysis_fields=context)
    issues = run_checks(project)
    assert [(issue["rule_id"], issue["measurement_id"]) for issue in issues] == [
        ("range", repeat.id), ("repeat", original.id), ("repeat", repeat.id), ("range", invalid.id),
    ]


def test_qc_acceptance_expires_after_edit_and_roundtrips(tmp_path):
    # No fixture sharing: the test owns the entire editable state.
    project = dataset([["A", "001", "101", "6"]])
    issue = next(i for i in run_checks(project) if i["rule_id"] == "range")
    project.view["qc_decisions"] = {issue["id"]: {"action": "acknowledged", "comment": "lab check requested"}}
    assert next(i for i in run_checks(project) if i["rule_id"] == "range")["decision"] == "acknowledged"
    path = tmp_path / "qc.apyro"
    project.save(path)
    project = Project.load(path)
    assert next(i for i in run_checks(project) if i["rule_id"] == "range")["decision"] == "acknowledged"
    project.rows[0].values[project.columns[2].id] = 102
    project.refresh()
    changed = next(i for i in run_checks(project) if i["rule_id"] == "range")
    assert changed["id"] != issue["id"] and changed["decision"] == "open"


def test_qc_exclusion_is_scoped_revision_bound_and_changes_statistics():
    from autopyro.statistics import StatisticsOptions, calculate_statistics, snapshot
    project = dataset([["A", "001", "2", "6"], ["A", "002", "4", "8"]])
    sid, pid = project.rows[0].id, project.columns[2].id
    measurement = project.science.current(sid, pid)
    project.view["qc_exclusions"] = {"statistics": {measurement.id: {"revision": measurement.revision, "comment": "review"}}}
    report = calculate_statistics(project, project.row_map(), StatisticsOptions((pid,)))
    assert report["summary"][0]["mean"] == 4
    assert report["availability"][0]["excluded_qc"] == 1
    assert not is_excluded(project, sid, pid, "calculations")
    snap = snapshot(project, project.row_map(), [pid])
    assert calculate_statistics(snap, snap.row_map(), StatisticsOptions((pid,)))["summary"][0]["mean"] == 4
    project.rows[0].values[pid] = 3
    project.refresh()
    assert not is_excluded(project, sid, pid, "statistics")


def test_import_undo_redo_preserves_project_views_and_table_filters(qtbot):
    from autopyro.window import MainWindow
    from autopyro.table import ImportCommand
    project = dataset([["A", "001", "2", "6"]])
    incoming = dataset([["B", "002", "3", "7", "1"]], "second", extra=True)
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.columns.item(1).setCheckState(Qt.CheckState.Unchecked)
    window.apply_filter([{"column": project.columns[0].id, "operator": "eq", "value": "A"}], "AND")
    plan = planned(project, incoming)
    prepared, protocol = prepare_import(project, incoming, plan)
    window.model.undo.push(ImportCommand(window.model, prepared, protocol, plan.project_digest))
    assert window.model.project is project and len(project.rows) == 2
    assert window.proxy.rowCount() == 1 and window.table.isColumnHidden(1)
    project.view["plots"] = {"user plot": {"title": "keep"}}
    window.model.undo.undo()
    assert len(project.rows) == 1 and len(project.columns) == 4
    assert project.view["plots"]["user plot"]["title"] == "keep"
    assert project.view["import_runs"][protocol["id"]]["status"] == "undone"
    window.model.undo.redo()
    assert len(project.rows) == 2 and len(project.columns) == 5
    window.dirty = False


@pytest.mark.parametrize(("sample", "policy", "reason"), [
    ("001", "keep", None),
    ("001", "replace", "measurement_changed"),
    ("002", "keep", "binding_changed"),
])
def test_import_rechecks_snapshot_for_all_samples(qtbot, sample, policy, reason):
    from autopyro.schemes.interpretation import evaluate_project
    from autopyro.schemes.standard_interpretation import standard_schemes
    from autopyro.table import ImportCommand, TableModel

    project = dataset([["A", "001", "2", "6"]])
    toc, s2 = project.columns[2:4]
    report = evaluate_project(project, project.row_map(), toc.id, s2.id, toc.unit, s2.unit,
                              standard_schemes()["S2–TOC"])
    report["scope"] = "all"
    project.view["interpretation_result"] = report
    incoming = dataset([["A", sample, "2", "8"]], "second")
    plan = planned(project, incoming)
    prepared, protocol = prepare_import(project, incoming, plan, working_policy=policy)
    model = TableModel(project)
    model.undo.push(ImportCommand(model, prepared, protocol, plan.project_digest))

    assert report["stale"] == (reason is not None)
    assert report.get("stale_reason", {}).get("code") == reason
    if policy == "replace":
        assert project.rows[0].values[s2.id] == 8
        assert report["stale_reason"]["column_id"] == toc.id
    if reason is None:
        model.undo.undo()
        assert not report["stale"]
        model.undo.redo()
        assert not report["stale"]


@pytest.mark.parametrize(("scope", "well", "scope_expands"), [
    ("filtered", "A", True),
    ("filtered", "B", False),
    ("view:Review", "A", True),
    ("view:Review", "B", False),
    ("set:Review", "A", False),
])
def test_import_rechecks_snapshot_against_restored_scope(qtbot, scope, well, scope_expands):
    from autopyro.schemes.interpretation import evaluate_project
    from autopyro.schemes.standard_interpretation import standard_schemes
    from autopyro.table import ImportCommand
    from autopyro.window import MainWindow

    project = dataset([["A", "001", "2", "6"]])
    rules = [{"column": project.columns[0].id, "operator": "eq", "value": "A"}]
    project.view["named"] = {"Review": {"rules": rules, "combine": "AND"}}
    project.view["sample_sets"] = {"Review": [project.rows[0].id]}
    window = MainWindow()
    qtbot.addWidget(window, before_close_func=lambda w: setattr(w, "dirty", False))
    window.set_project(project)
    window.apply_filter(rules if scope == "filtered" else
                        [{"column": project.columns[0].id, "operator": "eq", "value": "B"}], "AND")
    toc, s2 = project.columns[2:4]
    report = evaluate_project(project, window.analysis_scope.rows(scope), toc.id, s2.id, toc.unit, s2.unit,
                              standard_schemes()["S2–TOC"])
    report["scope"] = scope
    project.view["interpretation_result"] = report
    incoming = dataset([[well, "002", "2", "8"]], "second")
    plan = planned(project, incoming)
    prepared, protocol = prepare_import(project, incoming, plan)
    window.model.undo.push(ImportCommand(window.model, prepared, protocol, plan.project_digest))

    if scope_expands:
        assert report["stale_reason"] == {"code": "binding_changed", "fields": ["scope"]}
    else:
        assert not report["stale"]
        window.model.undo.undo()
        assert not report["stale"]
        window.model.undo.redo()
        assert not report["stale"]


def test_import_rechecks_snapshot_against_restored_sample_selection(qtbot):
    from autopyro.schemes.interpretation import evaluate_project
    from autopyro.schemes.standard_interpretation import standard_schemes
    from autopyro.table import ImportCommand
    from autopyro.window import MainWindow

    project = dataset([["A", "001", "2", "6"]])
    window = MainWindow()
    qtbot.addWidget(window, before_close_func=lambda w: setattr(w, "dirty", False))
    window.set_project(project)
    toc, s2 = project.columns[2:4]
    scheme = standard_schemes()["S2–TOC"]

    def snapshot(row_id):
        window.sample_selection.select([row_id])
        report = evaluate_project(project, [row_id], toc.id, s2.id, toc.unit, s2.unit, scheme)
        report["scope"] = "selected"
        project.view["interpretation_result"] = report
        return report

    original_id = project.rows[0].id
    report = snapshot(original_id)
    incoming = dataset([["A", "002", "2", "8"]], "second")
    plan = planned(project, incoming)
    prepared, protocol = prepare_import(project, incoming, plan)
    window.model.undo.push(ImportCommand(window.model, prepared, protocol, plan.project_digest))
    assert window.sample_selection.ids() == {original_id}
    assert not report["stale"]
    window.model.undo.undo()
    assert not report["stale"]
    window.model.undo.redo()
    assert not report["stale"]

    imported_id = project.rows[1].id
    report = snapshot(imported_id)
    window.model.undo.undo()
    assert window.sample_selection.ids() == set()
    assert report["stale_reason"] == {"code": "sample_removed", "row_id": imported_id, "diagram": None}


@pytest.mark.parametrize("axis_changed", [False, True])
@pytest.mark.parametrize("scope", ["all", "filtered", "selected", "view:Review", "set:Review"])
def test_database_refresh_checks_interpretation_inputs(qtbot, axis_changed, scope):
    from autopyro.core import Column
    from autopyro.data_io.importing import digest
    from autopyro.schemes.interpretation import evaluate_project
    from autopyro.schemes.standard_interpretation import standard_schemes
    from autopyro.table import RefreshDatabaseCommand
    from autopyro.window import MainWindow

    project = dataset([["A", "001", "2", "6"]])
    notes = Column("Notes", "text")
    project.columns.append(notes)
    project.rows[0].values[notes.id] = "original"
    project.sync_measurements()
    project.view["named"] = {"Review": {"rules": [], "combine": "AND"}}
    project.view["sample_sets"] = {"Review": [project.rows[0].id]}
    window = MainWindow()
    qtbot.addWidget(window, before_close_func=lambda w: setattr(w, "dirty", False))
    window.set_project(project)
    window.sample_selection.select([project.rows[0].id])
    toc, s2 = project.columns[2:4]
    report = evaluate_project(project, project.row_map(), toc.id, s2.id, toc.unit, s2.unit,
                              standard_schemes()["S2–TOC"])
    report["scope"] = scope
    project.view["interpretation_result"] = report
    prepared = deepcopy(project)
    prepared.view["database_snapshot"] = {}
    prepared.rows[0].values[s2.id if axis_changed else notes.id] = 8 if axis_changed else "updated"
    prepared.refresh()
    model = window.model
    model.undo.push(RefreshDatabaseCommand(model, prepared, {"id": "refresh-test", "working_changed": True},
                                          digest(project)))
    assert report["stale"] == axis_changed
    if axis_changed:
        assert report["stale_reason"]["code"] == "input_changed"
        assert report["stale_reason"]["column_id"] == s2.id
    else:
        model.undo.undo()
        assert not report["stale"]
        model.undo.redo()
        assert not report["stale"]


def test_qc_profile_editor_validates_applies_and_undoes(qtbot):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QCheckBox, QDialogButtonBox, QTableWidget
    from autopyro.ui.window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    project = dataset([["A", "001", "85", "6"]])
    window.set_project(project)
    window.show_qc()
    qc = window.qc_window
    observed = []

    def edit():
        dialog = QApplication.activeModalWidget()
        table = dialog.findChild(QTableWidget)
        buttons = dialog.findChild(QDialogButtonBox)
        ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        dialog.findChild(QCheckBox).setChecked(True)
        table.item(0, 2).setText("101")
        ok.click()
        observed.append(dialog.isVisible())
        table.item(0, 2).setText("84")
        ok.click()

    QTimer.singleShot(0, edit)
    qc.profile_button.click()
    assert observed == [True]
    assert any(i["rule_id"] == "profile_typical" for i in qc.issues)
    window.model.undo.undo()
    qc.reload()
    assert not any("profile" in i for i in qc.issues)
    window.dirty = False


def test_qc_profile_editor_freezes_the_selected_sample_set(qtbot):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QDialogButtonBox
    from autopyro.analysis.qc import pyrolysis_profile
    from autopyro.ui.window import MainWindow

    project = dataset([["A", "001", "101", "6"], ["B", "002", "102", "6"]])
    included, outside = (row.id for row in project.rows)
    project.view["sample_sets"] = {"Lab A": [included]}
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.show_qc()
    qc = window.qc_window

    def edit():
        editor = QApplication.activeModalWidget()
        scope = editor.findChild(QComboBox)
        scope.setCurrentIndex(scope.findData("set:Lab A"))
        editor.findChild(QCheckBox).setChecked(True)
        editor.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()

    QTimer.singleShot(0, edit)
    qc.profile_button.click()
    assert pyrolysis_profile(project)["sample_scope"] == {"name": "Lab A", "row_ids": [included]}
    project.view["sample_sets"]["Lab A"] = [outside]
    qc.refresh_checks()
    assert {issue["sample_id"] for issue in qc.issues if "profile" in issue} == {included}
    assert {issue["sample_id"] for issue in qc.issues if issue["rule_id"] == "range"} == {included, outside}
    window.model.undo.undo()
    qc.reload()
    assert not any("profile" in issue for issue in qc.issues)
    window.dirty = False


def test_qc_exclusion_reason_does_not_transfer_between_analysis_scopes(qtbot):
    from autopyro.ui.window import MainWindow

    project = dataset([["A", "001", "101", "6"]])
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.show_qc()
    qc = window.qc_window
    qc.comment.setText("Decision draft")
    qc.exclusion_comment.setText("Reason for statistics")
    qc.scope.setCurrentIndex(qc.scope.findData("plots"))
    assert not qc.exclusion_comment.text()
    assert qc.comment.text() == "Decision draft"
    first = qc.selected_issue()["id"]
    qc.select_issue_row(1)
    assert not qc.comment.text()
    qc.comment.setText("Other draft")
    qc.select_issue_row(0)
    assert qc.selected_issue()["id"] == first
    assert qc.comment.text() == "Decision draft"
    qc.scope.setCurrentIndex(qc.scope.findData("statistics"))
    assert qc.exclusion_comment.text() == "Reason for statistics"
    qc.scope.setCurrentIndex(qc.scope.findData("plots"))
    qc.exclude(True)
    assert not project.view.get("qc_exclusions")
    qc.exclusion_comment.setText("Reason for plots")
    qc.exclude(True)
    records = project.view["qc_exclusions"]
    assert records["plots"][qc.current_issue()["measurement_id"]]["comment"] == "Reason for plots"
    assert not records.get("statistics")
    window.dirty = False


def test_qc_selects_one_initial_issue_with_shift_held(qtbot, qapp):
    from autopyro.ui.window import MainWindow

    project = dataset([["A", "001", "101", "6"]])
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)

    qtbot.keyPress(window, Qt.Key.Key_Shift, Qt.KeyboardModifier.ShiftModifier)
    try:
        assert qapp.keyboardModifiers() == Qt.KeyboardModifier.ShiftModifier
        window.show_qc()
        qc = window.qc_window
        assert qc.results_model.rowCount() > 1
        assert qc.selected_spans() == [(0, 0)]
        assert qc.selected_issue()["id"] == qc.current_issue()["id"]
        assert qc.acknowledge.isEnabled()
    finally:
        qtbot.keyRelease(window, Qt.Key.Key_Shift)


def test_qc_gui_decision_and_exclusion_are_undoable(qtbot):
    from autopyro.window import MainWindow
    from autopyro.ui.i18n import tr
    from autopyro.analysis.qc import exclusion_time
    project = dataset([["A", "001", "101", "6"]])
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.show_qc()
    dialog = window.qc_window
    assert window.desktop.documents.currentWidget() is dialog
    assert window.desktop.properties_stack.currentWidget() is dialog.parameter_tabs
    assert callable(dialog.property)  # QObject.property must not be shadowed by a filter control.
    issue = dialog.current_issue()
    assert issue is not None
    dialog.comment.setText("Проверить лабораторный протокол")
    dialog.decide("acknowledged")
    assert project.view["qc_decisions"][issue["id"]]["action"] == "acknowledged"
    stamp = project.view["qc_decisions"][issue["id"]]["at"]
    time_label = tr('Время решения: {0}', exclusion_time(stamp))
    assert time_label in dialog.details.toPlainText()
    dialog.exclusion_comment.setText("Проверить лабораторный протокол")
    dialog.exclude(True)
    assert is_excluded(project, issue["sample_id"], issue["property_id"], "statistics")
    window.model.undo.undo()
    assert not is_excluded(project, issue["sample_id"], issue["property_id"], "statistics")
    window.model.undo.undo()
    assert issue["id"] not in project.view["qc_decisions"]
    dialog.reload()
    assert time_label not in dialog.details.toPlainText()
    assert "decision_at" not in dialog.current_issue()
    dialog.go_to_sample()
    assert window.desktop.documents.currentIndex() == 0
    window.desktop.documents.setCurrentWidget(dialog)
    assert window.desktop.properties_stack.currentWidget() is dialog.parameter_tabs
    window.dirty = False


def test_qc_exclusion_list_updates_while_open_and_keeps_selection(qtbot, monkeypatch):
    from PySide6.QtWidgets import QDialog, QTableView
    from autopyro.ui.table.commands import QCCommand
    from autopyro.ui.window import MainWindow

    project = dataset([["A", "001", "2", "6"]])
    measurement = project.science.current(project.rows[0].id, project.columns[3].id)
    key = ("qc_exclusions", "statistics")
    record = {"revision": measurement.revision, "comment": "Original"}
    project.view["qc_exclusions"] = {"statistics": {measurement.id: record}}
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.show_qc()

    def inspect(dialog):
        table = dialog.findChild(QTableView)
        table.selectRow(0)
        window.model.undo.push(QCCommand(window.model, key, {**record, "comment": "Updated"},
                                        measurement_id=measurement.id))
        qtbot.waitUntil(lambda: table.model().rows[0][-1] == "Updated")
        assert [index.row() for index in table.selectionModel().selectedRows()] == [0]
        window.model.undo.push(QCCommand(window.model, key, None, measurement_id=measurement.id))
        qtbot.waitUntil(lambda: table.model().rowCount() == 0)
        assert not table.selectionModel().hasSelection()
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", inspect)
    window.qc_window.manage_exclusions()
    window.dirty = False


def test_qc_exclusion_list_refreshes_before_acting_on_updated_records(qtbot, monkeypatch):
    from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTableView
    from autopyro.ui.i18n import tr
    from autopyro.ui.table.commands import QCCommand
    from autopyro.ui.window import MainWindow

    project = dataset([["A", "001", "2", "6"]])
    measurement = project.science.current(project.rows[0].id, project.columns[3].id)
    key = ("qc_exclusions", "statistics")
    original = {"revision": measurement.revision, "comment": "Original"}
    updated = {**original, "comment": "Updated"}
    project.view["qc_exclusions"] = {"statistics": {measurement.id: original}}
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.show_qc()

    def inspect(dialog):
        table = dialog.findChild(QTableView)
        remove = next(button for button in dialog.findChild(QDialogButtonBox).buttons()
                      if button.text() == tr('Снять исключение'))
        table.selectRow(0)
        window.model.undo.push(QCCommand(window.model, key, updated, measurement_id=measurement.id))
        remove.click()
        assert project.view["qc_exclusions"]["statistics"][measurement.id] == updated
        assert window.model.undo.count() == 1
        assert table.model().rows[0][-1] == "Updated"
        assert not table.selectionModel().hasSelection()
        table.selectRow(0)
        remove.click()
        assert project.view["qc_exclusions"]["statistics"] == {}
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", inspect)
    window.qc_window.manage_exclusions()
    window.model.undo.undo()
    assert project.view["qc_exclusions"]["statistics"][measurement.id] == updated
    window.dirty = False


def test_qc_batch_removal_rejects_changes_after_live_list_refresh(qtbot, monkeypatch):
    from PySide6.QtWidgets import QDialog, QDialogButtonBox, QMessageBox, QTableView
    from autopyro.ui.i18n import tr
    from autopyro.ui.table.commands import QCCommand
    from autopyro.ui.window import MainWindow

    project = dataset([["A", "001", "2", "6"]])
    measurement = project.science.current(project.rows[0].id, project.columns[3].id)
    record = {"revision": measurement.revision, "comment": "Original"}
    project.view["qc_exclusions"] = {scope: {measurement.id: deepcopy(record)} for scope in ("plots", "statistics")}
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.show_qc()

    def inspect(dialog):
        table = dialog.findChild(QTableView)
        table.selectAll()

        def confirm(*args):
            window.model.undo.push(QCCommand(window.model, ("qc_exclusions", "statistics"),
                                            {**record, "comment": "Updated"}, measurement_id=measurement.id))
            qtbot.waitUntil(lambda: any(row[-1] == "Updated" for row in table.model().rows))
            return QMessageBox.StandardButton.Yes

        monkeypatch.setattr(QMessageBox, "question", confirm)
        remove = next(button for button in dialog.findChild(QDialogButtonBox).buttons()
                      if button.text() == tr('Снять исключения ({0})', 2))
        remove.click()
        assert window.model.undo.count() == 1
        assert project.view["qc_exclusions"]["statistics"][measurement.id]["comment"] == "Updated"
        assert project.view["qc_exclusions"]["plots"][measurement.id] == record
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", inspect)
    window.qc_window.manage_exclusions()
    window.dirty = False


def test_qc_removes_selected_exclusions_across_scopes_with_one_undo(qtbot, monkeypatch):
    from PySide6.QtCore import QItemSelectionModel
    from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QMessageBox, QTableView
    from autopyro.ui.i18n import tr
    from autopyro.ui.window import MainWindow

    project = dataset([["A", "001", "2", "6"]])
    sid, cid = project.rows[0].id, project.columns[3].id
    first = project.science.current(sid, cid)
    other = project.science.add_result(sid, cid, 7)
    record = {"revision": first.revision, "comment": "Review"}
    before = {"statistics": {first.id: {**record, "at": "2026-10-01T09:00:00Z"}, other.id: record},
              "plots": {first.id: {**record, "at": "2026-10-02T09:00:00Z"}}}
    project.view["qc_exclusions"] = deepcopy(before)
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.show_qc()

    def inspect(dialog):
        table = dialog.findChild(QTableView)
        for row in (0, 2):
            table.selectionModel().select(table.model().index(row, 0),
                QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        table.selectionModel().setCurrentIndex(table.model().index(2, 0), QItemSelectionModel.SelectionFlag.NoUpdate)
        order = next(combo for combo in dialog.findChildren(QComboBox) if combo.findData("newest") >= 0)
        order.setCurrentIndex(order.findData("newest"))
        assert {index.row() for index in table.selectionModel().selectedRows()} == {0, 1}
        assert table.currentIndex().row() == 0
        remove = next(b for b in dialog.findChild(QDialogButtonBox).buttons()
                      if b.text() == tr('Снять исключения ({0})', 2))
        answers = iter((QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes))
        monkeypatch.setattr(QMessageBox, "question", lambda *args: next(answers))
        remove.click()
        assert project.view["qc_exclusions"] == before
        remove.click()
        assert table.model().rowCount() == 1
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", inspect)
    window.qc_window.manage_exclusions()
    assert project.view["qc_exclusions"] == {"statistics": {other.id: record}, "plots": {}}
    assert window.model.undo.count() == 1
    window.model.undo.undo()
    assert project.view["qc_exclusions"] == before
    window.model.undo.redo()
    assert set(project.view["qc_exclusions"]["statistics"]) == {other.id}
    window.dirty = False


def test_qc_exclusion_filters_keep_only_visible_selected_records(qtbot, monkeypatch):
    from PySide6.QtCore import QItemSelectionModel
    from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QLineEdit, QTableView
    from autopyro.ui.i18n import tr
    from autopyro.ui.window import MainWindow

    project = dataset([["A", "001", "2", "6"]])
    sid, cid = project.rows[0].id, project.columns[3].id
    first = project.science.current(sid, cid)
    other = project.science.add_result(sid, cid, 7)
    record = {"revision": first.revision, "comment": "Review"}
    before = {"statistics": {first.id: record, other.id: record}, "plots": {first.id: record}}
    project.view["qc_exclusions"] = deepcopy(before)
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.show_qc()

    def inspect(dialog):
        table = dialog.findChild(QTableView)
        search = dialog.findChild(QLineEdit)
        scope = next(combo for combo in dialog.findChildren(QComboBox) if combo.findData("plots") >= 0)
        for row in (0, 2):
            table.selectionModel().select(table.model().index(row, 0),
                QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        table.selectionModel().setCurrentIndex(table.model().index(2, 0), QItemSelectionModel.SelectionFlag.NoUpdate)
        search.setText("Review")
        assert [i.row() for i in table.selectionModel().selectedRows()] == [0, 2]
        assert table.currentIndex().row() == 2
        scope.setCurrentIndex(scope.findData("plots"))
        assert table.model().rowCount() == 1
        assert [i.row() for i in table.selectionModel().selectedRows()] == [0]
        scope.setCurrentIndex(0)
        assert [i.row() for i in table.selectionModel().selectedRows()] == [2]
        remove = next(button for button in dialog.findChild(QDialogButtonBox).buttons()
                      if button.text() == tr('Снять исключение'))
        remove.click()
        assert project.view["qc_exclusions"] == {"statistics": before["statistics"], "plots": {}}
        table.selectRow(0)
        search.setText("no match")
        assert not table.selectionModel().hasSelection() and not remove.isEnabled()
        search.clear()
        assert not table.selectionModel().hasSelection() and not remove.isEnabled()
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", inspect)
    window.qc_window.manage_exclusions()
    window.model.undo.undo()
    assert project.view["qc_exclusions"] == before
    window.dirty = False


def test_qc_exclusion_list_opens_each_repeat_by_measurement_id(qtbot, monkeypatch):
    from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QTableView
    from autopyro.ui.dialogs.measurement import MeasurementDialog
    from autopyro.ui.i18n import tr
    from autopyro.ui.window import MainWindow

    project = dataset([["A", "001", "2", "6"]])
    sid, cid = project.rows[0].id, project.columns[3].id
    working = project.science.current(sid, cid)
    repeats = [project.science.add_result(sid, cid, value) for value in (7, 8, 9)]
    records = {m.id: {"revision": m.revision, "comment": "Same reason"} for m in repeats}
    records[repeats[0].id]["at"] = "2026-10-01T12:34:56+03:00"
    records[repeats[2].id]["at"] = "2026-10-01T10:00:00+00:00"
    project.view["qc_exclusions"] = {"statistics": deepcopy(records)}
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.show_qc()
    opened = []

    def inspect_card(card):
        opened.append(card.selected_id())
        return QDialog.DialogCode.Rejected

    def inspect_manager(manager):
        table = manager.findChild(QTableView)
        assert [row[2] for row in table.model().rows] == ["7", "8", "9"]
        assert [row[5] for row in table.model().rows] == [
            "2026-10-01 09:34:56 UTC", "—", "2026-10-01 10:00:00 UTC",
        ]
        order = next(combo for combo in manager.findChildren(QComboBox) if combo.findData("newest") >= 0)
        order.setCurrentIndex(order.findData("oldest"))
        assert [row[2] for row in table.model().rows] == ["7", "9", "8"]
        order.setCurrentIndex(order.findData("newest"))
        assert [row[2] for row in table.model().rows] == ["9", "7", "8"]
        button = next(b for b in manager.findChild(QDialogButtonBox).buttons() if b.text() == tr('Измерение…'))
        assert not button.isEnabled()
        for index in range(3):
            table.selectRow(index)
            button.click()
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", inspect_manager)
    monkeypatch.setattr(MeasurementDialog, "exec", inspect_card)
    window.qc_window.manage_exclusions()
    assert opened == [repeats[index].id for index in (2, 0, 1)]
    assert project.science.current(sid, cid).id == working.id
    assert project.view["qc_exclusions"]["statistics"] == records
    assert not window.dirty


def test_qc_drafts_do_not_move_and_exclusions_survive_a_hidden_issue(qtbot):
    from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLineEdit, QTableView
    from autopyro.analysis.qc import PROFILE_KEY
    from autopyro.ui.window import MainWindow

    project = dataset([["A", "001", "85", "6"], ["B", "002", "86", "6"]])
    for analysis in project.science.analyses.values():
        analysis.method, analysis.laboratory = "Method", "Lab"
    profile = {"enabled": True}
    project.view[PROFILE_KEY[0]] = {PROFILE_KEY[1]: profile}
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.show_qc()
    dialog = window.qc_window
    dialog.state.setCurrentIndex(dialog.state.findData("open"))
    issue = dialog.current_issue()
    dialog.exclusion_comment.setText("Reason")
    dialog.exclude(True)
    dialog.comment.setText("First sample only")
    dialog.decide("acknowledged")
    dialog.table.selectRow(0)
    assert dialog.current_issue()["id"] != issue["id"]
    assert not dialog.comment.text() and not dialog.exclusion_comment.text()
    dialog.decide("acknowledged")
    assert len(project.view["qc_decisions"]) == 1

    profile["enabled"] = False
    dialog.reload()
    assert dialog.results_model.rowCount() == 0
    assert is_excluded(project, issue["sample_id"], issue["property_id"], "statistics")

    def remove():
        manager = dialog.findChild(QDialog)
        table = manager.findChild(QTableView)
        search = manager.findChild(QLineEdit)
        assert table.model().rowCount() == 1
        search.setText("does not exist")
        assert table.model().rowCount() == 0
        search.setText("Reason")
        assert table.model().rowCount() == 1
        table.selectRow(0)
        buttons = manager.findChild(QDialogButtonBox)
        next(b for b in buttons.buttons() if b.text() == dialog.include_button.toolTip()).click()
        assert table.model().rowCount() == 0
        manager.close()

    from PySide6.QtCore import QTimer
    QTimer.singleShot(0, remove)
    dialog.manage_exclusions()
    assert not is_excluded(project, issue["sample_id"], issue["property_id"], "statistics")
    assert project.history[-1]["measurement_id"] == issue["measurement_id"]
    assert project.history[-1]["value"] is None
    window.model.undo.undo()
    assert is_excluded(project, issue["sample_id"], issue["property_id"], "statistics")
    window.dirty = False


def test_qc_sorts_and_reveals_the_problem_cell_without_losing_the_filter(qtbot, monkeypatch):
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QMessageBox
    from autopyro.ui.window import MainWindow

    project = dataset([["A", "001", "101", "6"], ["B", "002", "2", "6"]])
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.show()
    window.show_qc()
    dialog = window.qc_window
    header = dialog.table.horizontalHeader()
    assert dialog.results_model.issues[0]["severity"] == "error"
    qtbot.mouseClick(header.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(20, 10))
    assert dialog.results_model.issues[0]["severity"] == "warning"

    dialog.data_scope.setCurrentIndex(dialog.data_scope.findData("all"))
    issue = dialog.current_issue()
    other = next(row.id for row in project.rows if row.id != issue["sample_id"])
    window.proxy.set_rules([], row_ids=[other])
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    dialog.go_to_sample()
    assert window.row_ids() == [issue["sample_id"]]
    assert project.columns[window.table.currentIndex().column()].id == issue["property_id"]
    dialog.restore_table_filter()
    assert window.row_ids() == [other]
    window.dirty = False


def test_qc_refreshes_working_status_when_measurement_card_closes(qtbot, monkeypatch):
    from autopyro.ui.dialogs.measurement import MeasurementDialog
    from autopyro.ui.i18n import tr
    from autopyro.ui.window import MainWindow

    project = dataset([["A", "001", "2", "6"]])
    sid, cid = project.rows[0].id, project.columns[3].id
    repeat = project.science.add_result(sid, cid, 8)
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.show_qc()
    qc = window.qc_window
    index = next(i for i, issue in enumerate(qc.results_model.issues)
                 if issue["rule_id"] == "repeat" and issue["measurement_id"] == repeat.id)
    qc.table.selectRow(index)
    assert tr('Не выбран рабочим') in qc.details.toPlainText()

    def change_working(card):
        card.select_result()
        return card.DialogCode.Accepted

    monkeypatch.setattr(MeasurementDialog, "exec", change_working)
    qc.open_measurement()
    assert project.science.current(sid, cid).id == repeat.id
    assert tr('Не выбран рабочим') not in qc.details.toPlainText()
    assert tr('Рабочий результат') in qc.details.toPlainText()
    window.dirty = False


def test_qc_opens_the_non_working_repeat_without_changing_working_selection(qtbot, monkeypatch):
    from autopyro.ui.dialogs.measurement import MeasurementDialog
    from autopyro.ui.i18n import tr
    from autopyro.ui.window import MainWindow

    project = dataset([["A", "001", "2", "6"]])
    sid, cid = project.rows[0].id, project.columns[3].id
    working = project.science.current(sid, cid)
    project.science.analyses[working.analysis_id].method = "Shared name"
    repeat = project.science.add_result(sid, cid, "8", analysis_fields={"laboratory": "Shared name"})
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.show_qc()
    qc = window.qc_window
    row = next(i for i, issue in enumerate(qc.results_model.issues)
               if issue["rule_id"] == "repeat" and issue["measurement_id"] == repeat.id)
    qc.table.selectRow(row)
    details = qc.details.toPlainText()
    assert f"{tr('Метод')}: Shared name · {tr('Лаборатория')}: —" in details
    assert f"{tr('Метод')}: — · {tr('Лаборатория')}: Shared name" in details
    opened = []

    def inspect(dialog):
        opened.append(dialog.selected_id())
        return dialog.DialogCode.Rejected

    monkeypatch.setattr(MeasurementDialog, "exec", inspect)
    qc.measurement_button.click()
    assert opened == [repeat.id]
    assert project.science.current(sid, cid).id == working.id
    assert not window.dirty


def test_qc_compact_workspace_keeps_actions_and_comments_accessible(qtbot, qapp, request):
    import os
    from pathlib import Path
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QBoxLayout, QCheckBox, QPushButton
    from autopyro.ui.i18n import tr
    from autopyro.ui.window import MainWindow

    # The Windows offscreen plugin does not load the system font directory.
    if system_root := os.environ.get("WINDIR"):
        font_path = Path(system_root) / "Fonts" / "arial.ttf"
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        if font_id >= 0:
            original = qapp.font()
            qapp.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0], 10))

            def restore_font():
                qapp.setFont(original)
                QFontDatabase.removeApplicationFont(font_id)

            request.addfinalizer(restore_font)

    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(dataset([["A", "001", "101", "6"]]))
    window.resize(1024, 600)
    window.show()
    window.show_qc()
    qc = window.qc_window
    qtbot.waitUntil(lambda: window.minimumWidth() <= 1024)
    window.resize(1024, 600)
    qtbot.waitUntil(lambda: (qc.decision_actions.direction() == QBoxLayout.Direction.TopToBottom
                             and qc.table.horizontalScrollBar().maximum() == 0
                             and qc.comment.width() >= 250))
    assert window.width() == 1024
    assert qc.table.horizontalScrollBar().maximum() == 0
    assert qc.table.verticalHeader().isHidden()
    assert qc.comment.width() >= 250 and qc.exclusion_comment.width() >= 250
    assert qc.acknowledge.isVisible() and qc.source_button.isVisible() and qc.exclude_button.isVisible()
    before = qc.table.viewport().height()
    toggles = {box.text(): box for box in qc.findChildren(QCheckBox)}
    toggles[tr('Детали замечания')].click()
    toggles[tr('Действия с замечанием')].click()
    qtbot.waitUntil(lambda: qc.table.viewport().height() > before)
    assert not qc.details.isVisible() and not qc.acknowledge.isVisible()
    toggles[tr('Действия с замечанием')].click()
    assert qc.acknowledge.isVisible()
    columns = next(button for button in qc.parameter_tabs.findChildren(QPushButton)
                   if button.text() == tr('Колонки замечаний'))
    for action, column in zip(columns.menu().actions(), (3, 6, 7)):
        assert qc.table.isColumnHidden(column)
        action.trigger()
        assert not qc.table.isColumnHidden(column)
        action.trigger()
        assert qc.table.isColumnHidden(column)
    qc.restore_filter_button.show()
    assert window.width() == 1024

    window.resize(1366, 768)
    qtbot.waitUntil(lambda: qc.decision_actions.direction() == QBoxLayout.Direction.LeftToRight)
    assert qc.table.verticalHeader().isVisible()


def test_qc_reveals_the_hidden_problem_column_and_keeps_saved_view(qtbot, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QMessageBox
    from autopyro.ui.window import MainWindow

    project = dataset([["A", "001", "101", "6"], ["B", "002", "2", "6"]])
    sid, other = (row.id for row in project.rows)
    project.view["sample_sets"] = {"Other": [other]}
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.columns.item(2).setCheckState(Qt.CheckState.Unchecked)
    window.active_sample_set = "Other"
    window.proxy.set_rules([], row_ids=[other])
    window.show_qc()
    qc = window.qc_window
    qc.data_scope.setCurrentIndex(qc.data_scope.findData("all"))
    qc.table.selectRow(next(i for i, issue in enumerate(qc.results_model.issues) if issue["rule_id"] == "range"))
    issue = qc.current_issue()
    assert issue["sample_id"] == sid and issue["property_id"] == project.columns[2].id
    window.dirty = False

    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.No)
    qc.go_to_sample()
    assert window.table.isColumnHidden(2) and window.row_ids() == [other]

    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    qc.go_to_sample()
    assert not window.table.isColumnHidden(2) and window.row_ids() == [sid]
    assert window.table.currentIndex().column() == 2
    assert not window.dirty
    window.path = tmp_path / "qc-temporary-view.apyro"
    assert window.save()
    saved_view = Project.load(window.path).view["current"]
    assert project.columns[2].id not in saved_view["visible"]
    assert saved_view["sample_set"] == "Other"

    qc.restore_table_filter()
    assert window.table.isColumnHidden(2) and window.row_ids() == [other]
    assert window.active_sample_set == "Other"
    assert window.columns.item(2).checkState() == Qt.CheckState.Unchecked
    assert not window.dirty


def test_qc_coverage_explains_why_measurements_were_skipped(qtbot, monkeypatch):
    from PySide6.QtWidgets import QDialog, QTableView
    from autopyro.model.core import Column, Row
    from autopyro.ui.i18n import tr
    from autopyro.ui.window import MainWindow

    project = dataset([["A", "001", "2", "6"], ["A", "002", "2", "ND"],
                       ["A", "003", "2", "<0"], ["A", "004", "2", "6"]])
    toc, s2 = project.columns[2:4]
    unknown = Column("Unknown indicator", "number", unit="ppm")
    project.columns.append(unknown)
    for row in project.rows:
        row.values[unknown.id] = 1
    project.view["qc_profiles"] = {"pyrolysis": {"enabled": True}}
    window = MainWindow()
    qtbot.addWidget(window)
    # A physical column may have no measurement at all for a selected sample.
    project.sync_measurements()
    project.science.current(project.rows[-1].id, s2.id).unit = "unsupported"
    project.rows.append(Row({column.id: None for column in project.columns}))
    window.set_project(project, prepared=True)
    window.show_qc()
    qc = window.qc_window
    assert tr('Записи профиля: проверено {0}, пропущено {1}.', 5, 3) in qc.profile_status.text()
    assert tr('Пар проба–показатель без записи: {0}.', 2) in qc.profile_status.text()

    def inspect(dialog):
        rows = {row[0]: row[1:] for row in dialog.findChild(QTableView).model().rows}
        assert rows[toc.name][2:5] == [4, "—", 1]
        assert rows[s2.name][2] == 1
        for reason in ('Неподдерживаемая единица', 'Недопустимый предел', 'Нет числа или диапазона'):
            assert f"{tr(reason)}: 1" in rows[s2.name][3]
        assert rows["Unknown indicator"][:2] == [tr('Не распознана'), tr('Не предусмотрен')]
        assert rows["Unknown indicator"][-1] == tr('Формат записи')
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", inspect)
    qc.coverage_button.click()


def test_qc_summary_distinguishes_visible_levels_from_scope_review(qtbot):
    from autopyro.ui.i18n import tr
    from autopyro.ui.window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(dataset([["A", "001", "101", "6"]]))
    window.show_qc()
    dialog = window.qc_window
    total = len(dialog.issues)
    assert total > 1

    dialog.severity.setCurrentIndex(dialog.severity.findData("error"))
    assert dialog.results_model.rowCount() == 1
    assert tr('В выдаче: ошибок {0} · критических {1} · предупреждений {2}', 1, 0, 0) in dialog.status.text()
    assert tr('В области: к рассмотрению {0} · принято {1}', total, 0) in dialog.status.text()

    dialog.comment.setText("Checked")
    dialog.decide("acknowledged")
    assert tr('В области: к рассмотрению {0} · принято {1}', total - 1, 1) in dialog.status.text()
    dialog.state.setCurrentIndex(dialog.state.findData("open"))
    assert dialog.results_model.rowCount() == 0
    assert tr('По текущим фильтрам совпадений нет.') in dialog.status.text()
    assert tr('В области: к рассмотрению {0} · принято {1}', total - 1, 1) in dialog.status.text()

    window.model.undo.undo()
    dialog.reload()
    assert dialog.results_model.rowCount() == 1
    assert tr('В области: к рассмотрению {0} · принято {1}', total, 0) in dialog.status.text()
    dialog.severity.setCurrentIndex(dialog.severity.findData("all"))
    dialog.property_filter.setText("S2")
    qtbot.waitUntil(lambda: dialog.results_model.rowCount() == 1)
    assert tr('В выдаче: ошибок {0} · критических {1} · предупреждений {2}', 0, 0, 1) in dialog.status.text()
    window.dirty = False


def test_qc_source_button_applies_pending_search_before_opening(qtbot, monkeypatch):
    from PySide6.QtWidgets import QDialog
    from autopyro.ui.window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(dataset([["A", "001", "101", "6"]]))
    window.show_qc()
    qc = window.qc_window
    opened = []
    monkeypatch.setattr(QDialog, "exec", lambda dialog: opened.append(dialog.windowTitle()))
    qc.source.setText("no matching source")
    assert qc.filter_timer.isActive()
    qc.source_button.click()
    assert opened == []
    assert qc.results_model.rowCount() == 0


def test_qc_navigation_applies_pending_search_before_switching_document(qtbot):
    from autopyro.ui.window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(dataset([["A", "001", "101", "6"]]))
    window.show_qc()
    qc = window.qc_window
    assert qc.selected_issue() is not None
    assert window.desktop.documents.currentWidget() is qc
    qc.source.setText("no matching source")
    assert qc.filter_timer.isActive()
    qc.go.click()
    assert qc.results_model.rowCount() == 0
    assert window.desktop.documents.currentWidget() is qc


def test_qc_source_card_keeps_file_locator_and_searchable_full_provenance(qtbot, monkeypatch):
    from PySide6.QtWidgets import QDialog, QTextEdit
    from autopyro.ui.i18n import tr
    from autopyro.ui.dialogs.qc import source_location
    from autopyro.ui.window import MainWindow

    path = r"C:\Laboratory\Exports\Pyro.xlsx"
    project = dataset([["A", "001", "101", "6"]])
    for source in project.science.sources.values():
        source.provenance.update(path=path, sheet="Batch A", sha256="technical-hash")
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.show_qc()
    qc = window.qc_window
    card = qc.details.toPlainText()
    assert tr('Файл: {0}', "Pyro.xlsx") in card
    assert tr('Лист: {0}', "Batch A") in card
    assert tr('Строка: {0}', 2) in card
    assert path not in card and "technical-hash" not in card

    def inspect(editor):
        details = editor.findChild(QTextEdit).toPlainText()
        assert path in details and "technical-hash" in details
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", inspect)
    qc.source_button.click()
    qc.source.setText("no matching file")
    qtbot.waitUntil(lambda: qc.results_model.rowCount() == 0)
    qc.source.setText(path.upper())
    qtbot.waitUntil(lambda: qc.results_model.rowCount() > 0)
    qc.source.setText(tr('Строка: {0}', 2))
    qtbot.waitUntil(lambda: not qc.filter_timer.isActive())
    assert qc.results_model.rowCount() > 0
    assert source_location({"endpoint": {"database": "geochem"}, "relation": "raw_cells", "row_number": 0}) == (
        f"{tr('База: {0}', 'geochem')} · {tr('Таблица: {0}', 'raw_cells')} · {tr('Строка: {0}', 0)}"
    )


def test_qc_batch_decision_only_changes_selected_findings_and_undoes_once(qtbot, monkeypatch):
    from PySide6.QtCore import QItemSelectionModel
    from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLineEdit
    from autopyro.ui.i18n import tr
    from autopyro.ui.window import MainWindow

    project = dataset([["A", "001", "101", "6"], ["A", "002", "102", "6"], ["A", "003", "103", "6"]])
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.show_qc()
    qc = window.qc_window
    qc.severity.setCurrentIndex(qc.severity.findData("error"))
    assert qc.results_model.rowCount() == 3
    selected = {qc.results_model.issues[row]["id"] for row in (0, 2)}
    other = qc.results_model.issues[1]["id"]

    def choose_rows():
        selection = qc.table.selectionModel()
        selection.clearSelection()
        for row in (0, 2):
            selection.select(qc.results_model.index(row, 0),
                             QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        assert qc.batch_button.isEnabled()
        assert not qc.acknowledge.isEnabled() and not qc.exclude_button.isEnabled()

    def accept(editor):
        buttons = editor.findChild(QDialogButtonBox)
        accept_button = next(button for button in buttons.buttons() if button.text() == tr('Принять ({0})', 2))
        assert not accept_button.isEnabled()
        editor.findChild(QLineEdit).setText("Reviewed together")
        accept_button.click()
        return editor.result()

    revision = window.analysis_revision
    count = window.model.undo.count()

    def change_filter_while_open(editor):
        qc.severity.setCurrentIndex(qc.severity.findData("warning"))
        return accept(editor)

    choose_rows()
    monkeypatch.setattr(QDialog, "exec", change_filter_while_open)
    qc.batch_button.click()
    assert not project.view.get("qc_decisions")
    assert window.model.undo.count() == count

    qc.severity.setCurrentIndex(qc.severity.findData("error"))
    choose_rows()

    def type_search_before_confirmation(editor):
        qc.sample.setText("no matching sample")
        assert qc.filter_timer.isActive()
        return accept(editor)

    monkeypatch.setattr(QDialog, "exec", type_search_before_confirmation)
    qc.batch_button.click()
    assert not project.view.get("qc_decisions")
    assert window.model.undo.count() == count
    assert qc.results_model.rowCount() == 0
    qc.sample.clear()
    qtbot.waitUntil(lambda: qc.results_model.rowCount() == 3)
    choose_rows()
    selection = qc.table.selectionModel()
    selection.setCurrentIndex(qc.results_model.index(2, 0), QItemSelectionModel.SelectionFlag.NoUpdate)
    focused_id = qc.current_issue()["id"]
    header = qc.table.horizontalHeader()
    header.setSortIndicator(2, Qt.SortOrder.DescendingOrder)
    assert {qc.results_model.issues[index.row()]["id"] for index in selection.selectedRows()} == selected
    assert qc.current_issue()["id"] == focused_id
    qc.sample.setText("A")
    qtbot.waitUntil(lambda: not qc.filter_timer.isActive())
    assert {qc.results_model.issues[index.row()]["id"] for index in selection.selectedRows()} == selected
    monkeypatch.setattr(QDialog, "exec", accept)
    qc.batch_button.click()
    records = project.view["qc_decisions"]
    assert set(records) == selected and other not in records
    assert all(record["comment"] == "Reviewed together" for record in records.values())
    assert window.analysis_revision == revision + 1
    assert window.model.undo.count() == count + 1
    assert set(project.history[-1]["issue_ids"]) == selected
    window.model.undo.undo()
    qc.reload()
    assert not records
    window.model.undo.redo()
    qc.reload()
    assert set(records) == selected

    def reopen(editor):
        buttons = editor.findChild(QDialogButtonBox)
        next(button for button in buttons.buttons() if button.text() == tr('Вернуть ({0})', 2)).click()
        return editor.result()

    choose_rows()
    monkeypatch.setattr(QDialog, "exec", reopen)
    qc.batch_button.click()
    assert set(records) == selected
    assert all(record["action"] == "open" for record in records.values())
    window.model.undo.undo()
    assert all(record["action"] == "acknowledged" for record in records.values())
    window.dirty = False


def test_qc_filtered_out_findings_require_selection_again(qtbot):
    from autopyro.ui.window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(dataset([["A", "001", "101", "6"]]))
    window.show_qc()
    qc = window.qc_window
    qc.table.selectAll()
    assert qc.batch_button.isEnabled()
    qc.sample.setText("no matching sample")
    qtbot.waitUntil(lambda: qc.results_model.rowCount() == 0)
    assert not qc.table.selectionModel().hasSelection()
    qc.sample.clear()
    qtbot.waitUntil(lambda: qc.results_model.rowCount() > 0)
    assert not qc.table.selectionModel().hasSelection()
    assert not qc.acknowledge.isEnabled() and not qc.batch_button.isEnabled()


def test_qc_cancelled_first_check_waits_for_explicit_retry(qtbot, monkeypatch):
    from autopyro.ui.dialogs import qc_checks as qc
    from autopyro.ui.window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(dataset([["A", "001", "101", "6"]]))
    calls = []

    def progress(parent, title, check, *, cancellable):
        calls.append(cancellable)
        if len(calls) == 1:
            raise qc.TaskCancelled()
        return check(lambda *_: None)

    monkeypatch.setattr(qc, "large_snapshot", lambda project: True)
    monkeypatch.setattr(qc, "run_with_progress", progress)
    window.show_qc()
    dialog = window.qc_window
    dialog.on_activated()
    assert calls == [True]
    assert not dialog.retry.isHidden()
    assert not dialog.acknowledge.isEnabled()
    assert dialog.results_model.rowCount() == 0
    dialog.retry.click()
    assert calls == [True, True]
    assert dialog.retry.isHidden()
    assert dialog.results_model.rowCount() > 0


def test_append_dialog_preview_and_commit(qtbot):
    from autopyro.window import MainWindow
    from autopyro.import_dialog import AppendImportDialog
    project = dataset([["A", "001", "2", "6"]])
    incoming = dataset([["A", "001", "2", "8"]], "second")
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    dialog = AppendImportDialog(window, incoming)
    qtbot.addWidget(dialog)
    dialog.mapping.cellWidget(0, 2).setChecked(True)
    dialog.mapping.cellWidget(1, 2).setChecked(True)
    dialog.calculate()
    qtbot.waitUntil(lambda: dialog.worker is not None and not dialog.worker.isRunning() and dialog.plan is not None, timeout=10000)
    assert dialog.plan.rows[0]["status"] == "matched"
    dialog.apply_import()
    qtbot.waitUntil(lambda: dialog.result() == dialog.DialogCode.Accepted, timeout=10000)
    assert len(project.rows) == 1
    assert len(project.science.results(project.rows[0].id, project.columns[3].id)) == 2
    window.dirty = False


def test_new_duplicate_keys_inside_file_need_a_decision():
    project = dataset([["A", "001", "2", "6"]])
    incoming = dataset([["B", "002", "3", "7"], ["B", "002", "4", "8"]], "second")
    plan = planned(project, incoming)
    assert [item["status"] for item in plan.rows] == ["input_repeat", "input_repeat"]
    with pytest.raises(ValueError, match="неоднозначные"):
        prepare_import(project, incoming, plan)
    decisions = {row.id: {"action": "new"} for row in incoming.rows}
    result, _ = prepare_import(project, incoming, plan, decisions)
    assert len(result.rows) == 3  # Explicit separate-sample choice; never silently merge equal keys.


def test_qc_exclusions_flow_through_pyro_plots_maps_and_geometry():
    from autopyro.calculations import calculate_column
    from autopyro.plotting import PlotSpec, numeric
    from autopyro.mapping import MapSpec, number
    from autopyro.interpretation import evaluate_project, replay_report
    from pathlib import Path
    project = dataset([["A", "001", "2", "6"]])
    sid, toc, s2 = project.rows[0].id, project.columns[2], project.columns[3]
    measurement = project.science.current(sid, toc.id)
    record = {measurement.id: {"revision": measurement.revision, "comment": "questioned measurement"}}
    project.view["qc_exclusions"] = {scope: deepcopy(record) for scope in ("calculations", "plots", "interpretation")}
    config = {"formula": "HI", "version": 1, "analysis_context": "Pyrol6", "compatible_confirmed": True,
              "inputs": {"TOC": {"column_id": toc.id, "unit": toc.unit}, "S2": {"column_id": s2.id, "unit": s2.unit}}}
    values, diagnostics = calculate_column(project, config)
    assert values[sid] is None and diagnostics[sid]["status"] == "qc_excluded"
    assert numeric(project, project.rows[0], toc.id, PlotSpec())[0] == "qc_excluded"
    assert number(project, project.rows[0], toc.id, MapSpec())[0] == "qc_excluded"
    scheme = json.loads(Path("examples/interpretation-demo.json").read_text(encoding="utf-8"))
    # The geometry's Y has HI units; a separate known-value column suffices for this input-policy test.
    from autopyro.core import Column
    hi = Column("HI", "number", unit="mg HC/g TOC")
    project.columns.append(hi)
    project.rows[0].values[hi.id] = 300
    project.sync_measurements()
    report = evaluate_project(project, [sid], toc.id, hi.id, toc.unit, hi.unit, scheme)
    assert report["inputs"][0]["statuses"][0] == "qc_excluded"
    assert report["results"][0]["status"] == "insufficient_data"
    assert replay_report(report) == report["results"]


def test_qc_report_keeps_only_profile_scope_members_in_the_report(tmp_path):
    from autopyro.analysis.qc import PROFILE_KEY, pyrolysis_profile, snapshot_for_checks
    from autopyro.reports.reporting import ReportOptions, build_report, render_html

    project = dataset([["A", "001", "101", "6"], ["A", "002", "102", "6"], ["B", "003", "103", "6"]])
    included, hidden, outside = (row.id for row in project.rows)
    scope = {"name": "Lab A", "row_ids": [included, hidden]}
    project.view[PROFILE_KEY[0]] = {PROFILE_KEY[1]: {"enabled": True, "sample_scope": scope}}
    path = tmp_path / "scoped-profile.apyro"
    project.save(path)
    loaded = Project.load(path)
    assert pyrolysis_profile(loaded)["sample_scope"] == scope
    assert run_checks(snapshot_for_checks(loaded, lambda: None)) == run_checks(loaded)
    report = build_report(loaded, [included, outside], ReportOptions((loaded.columns[2].id,)))
    exported_scope = report["qc"]["pyrolysis_profile"]["sample_scope"]
    assert exported_scope == {"name": "Lab A", "row_ids": [included]}
    assert report["project"]["view"][PROFILE_KEY[0]][PROFILE_KEY[1]]["sample_scope"] == exported_scope
    assert {i["sample_id"] for i in report["qc"]["issues"] if "profile" in i} == {included}
    assert hidden not in json.dumps(report)
    assert "Проб отчёта в области профиля: 1" in render_html(report)
    assert pyrolysis_profile(loaded)["sample_scope"] == scope


def test_qc_report_keeps_scoped_policy_and_never_leaks_unselected_sample():
    from autopyro.reporting import ReportOptions, build_report, render_html
    from autopyro.statistics import StatisticsOptions, calculate_statistics
    from autopyro.core import Column, Row
    from autopyro.measurements import MeasurementStore
    project = dataset([["A", "001", "101", "6"], ["B", "002", "4", "8"]])
    sid, pid = project.rows[0].id, project.columns[2].id
    measurement = project.science.current(sid, pid)
    project.view["qc_exclusions"] = {"statistics": {measurement.id: {"revision": measurement.revision, "comment": "review"}}}
    report = build_report(project, [sid], ReportOptions((pid,)))
    assert report["statistics"]["availability"][0]["excluded_qc"] == 1
    assert report["qc"]["issues"]
    assert [(item["measurement_id"], item["comment"]) for item in report["qc"]["active_exclusions"]] == [
        (measurement.id, "review"),
    ]
    assert project.rows[1].id not in json.dumps(report)
    assert "Действующие исключения QC" in render_html(report)
    stored = report["project"]
    replay = Project(id=stored["id"], columns=[Column(**c) for c in stored["columns"]],
                     rows=[Row(**r) for r in stored["rows"]], science=MeasurementStore.from_dict(stored["science"]), view=stored["view"])
    assert calculate_statistics(replay, [sid], StatisticsOptions((pid,)))["availability"][0]["excluded_qc"] == 1
    measurement.revision += 1
    stale = build_report(project, [sid], ReportOptions((pid,)))["qc"]
    assert stale["active_exclusions"] == []
    assert measurement.id in stale["exclusions"]["statistics"]


def test_sparse_measurement_can_be_selected_and_undone(qtbot):
    from autopyro.table import TableModel, SelectMeasurementCommand
    project = dataset([["A", "001", "2", "6"]])
    incoming = dataset([["B", "002", "3", "7", "1"]], "second", extra=True)
    result, _ = prepare_import(project, incoming, planned(project, incoming))
    model = TableModel(result)
    sid, pid = result.rows[0].id, result.columns[-1].id
    assert result.science.current(sid, pid) is None
    measurement = result.science.add_result(sid, pid, "2", analysis_fields={"method": "Pyrol6"})
    model.undo.push(SelectMeasurementCommand(model, sid, pid, measurement.id))
    assert result.rows[0].values[pid] == 2
    model.undo.undo()
    assert result.rows[0].values[pid] is None
    assert result.science.current(sid, pid) is None
    result.validate()


def test_qc_detects_different_pyro_contexts_without_changing_values():
    project = dataset([["A", "001", "2", "6"]], extra=False)
    from autopyro.core import Column
    s1 = Column("S1", "number", unit="mg HC/g rock")
    project.columns.append(s1)
    project.rows[0].values[s1.id] = 1
    project.sync_measurements()
    sid = project.rows[0].id
    source_analysis = project.science.analyses[project.science.current(sid, s1.id).analysis_id]
    source_analysis.method = "Pyrol6"
    second = project.science.add_result(sid, project.columns[3].id, 7, analysis_fields={"method": "Pyrol2"})
    project.select_measurement(sid, project.columns[3].id, second.id)
    issues = run_checks(project)
    assert len([i for i in issues if i["rule_id"] == "compatibility"]) == 2
    assert project.rows[0].values[s1.id] == 1


@pytest.mark.parametrize("field", ["method", "basis"])
def test_qc_reports_context_of_saved_calculations_without_changing_values(qtbot, field):
    from autopyro.analysis.calculations import make_calculated_column
    from autopyro.ui.table.model import TableModel

    project = dataset([["A", "001", "2", "6"], ["A", "002", "2", "8"]])
    row, outside = project.rows
    toc, s2 = project.columns[2:]
    first = project.science.current(row.id, toc.id)
    if field == "method":
        project.science.analyses[first.analysis_id].method = "Method A"
        repeat = project.science.add_result(row.id, s2.id, 6, analysis_fields={"method": "Method B"})
    else:
        first.basis = "dry"
        repeat = project.science.add_result(row.id, s2.id, 6, details={"basis": "as_received"})
    project.select_measurement(row.id, s2.id, repeat.id)
    config = {"formula": "HI", "version": 1, "compatible_confirmed": True,
              "analysis_context": "Legacy", "scope_row_ids": [row.id],
              "inputs": {c.name: {"column_id": c.id, "unit": c.unit} for c in (toc, s2)}}
    for name, spec in (("Legacy HI", config), ("Strict HI", {**config, "check_context": True})):
        column, values = make_calculated_column(project, name, spec)
        project.columns.append(column)
        for sample in project.rows:
            sample.values[column.id] = values[sample.id]
    legacy, strict = project.columns[-2:]
    model = TableModel(project)
    before = asdict(deepcopy(project))
    findings = [i for i in run_checks(project, include_peers=True) if i["rule_id"] == "calculation_context"]
    assert asdict(project) == before
    assert [(i["sample_id"], i["property_id"]) for i in findings] == [(row.id, legacy.id), (row.id, strict.id)]
    assert all(i["context_fields"] == [field] and i["measurement_id"] is None for i in findings)
    assert {peer["id"] for peer in findings[0]["peers"]} == {first.id, repeat.id}
    assert row.values[legacy.id] == 300 and row.values[strict.id] is None
    assert outside.values[legacy.id] is None
    index = model.index(0, len(project.columns) - 2)
    assert "Различается аналитический контекст" in index.data(Qt.ItemDataRole.ToolTipRole)
    if field == "method":
        project.science.analyses[repeat.analysis_id].method = "Method C"
    else:
        repeat.basis = "ash_free"
    changed = next(i for i in run_checks(project) if i["rule_id"] == "calculation_context")
    assert changed["id"] != findings[0]["id"]


def test_bad_numeric_mapping_collects_error_without_partial_changes():
    project = dataset([["A", "001", "2", "6"]])
    incoming = build_project(["Well", "Sample", "TOC", "S2"], [["A", "001", "2", "abc"]], [2],
        [(0, "Well", "text", ""), (1, "Sample", "text", ""), (2, "TOC", "number", "%wt"),
         (3, "S2", "text", "mg HC/g rock")], {"path": "bad.csv", "sha256": "bad"})
    mapping = suggest_mapping(project, incoming)
    mapping[incoming.columns[3].id] = project.columns[3].id
    before = asdict(deepcopy(project))
    plan = preview_import(project, incoming, mapping, [c.id for c in incoming.columns[:2]])
    assert len(plan.errors) == 1
    with pytest.raises(ValueError, match="ошибки"):
        prepare_import(project, incoming, plan)
    assert asdict(project) == before


def test_cancelling_append_review_never_modifies_the_project(qtbot):
    from autopyro.window import MainWindow
    from autopyro.import_dialog import AppendImportDialog
    project = dataset([["A", "001", "2", "6"]])
    incoming = dataset([["A", "001", "2", "8"]], "second")
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    before = asdict(deepcopy(project))
    dialog = AppendImportDialog(window, incoming)
    qtbot.addWidget(dialog)
    dialog.calculated(planned(project, incoming))
    dialog.reject()
    assert asdict(project) == before
    assert not window.model.undo.canUndo()


@pytest.mark.parametrize("single", [True, False])
def test_interpretation_qc_exclusion_invalidates_only_affected_snapshot(qtbot, single):
    from autopyro.schemes.interpretation import evaluate_project
    from autopyro.schemes.standard_interpretation import standard_schemes
    from autopyro.table import QCCommand, TableModel

    project = dataset([["A", "001", "2", "6"], ["A", "002", "3", "7"]])
    first, second = project.rows
    toc, s2 = project.columns[2:4]
    report = evaluate_project(project, [first.id], toc.id, s2.id, toc.unit, s2.unit,
                              standard_schemes()["S2–TOC"])
    project.view["interpretation_result"] = report
    model = TableModel(project)
    key = ("qc_exclusions", "interpretation")

    def exclude(row):
        measurement = project.science.current(row.id, toc.id)
        record = {"revision": measurement.revision, "comment": "review"}
        if single:
            command = QCCommand(model, key, record, measurement_id=measurement.id)
        else:
            records = project.view.get(key[0], {}).get(key[1], {})
            command = QCCommand(model, key, {**records, measurement.id: record})
        model.undo.push(command)

    exclude(second)
    assert not report["stale"]
    model.undo.undo()
    assert not report["stale"]
    exclude(first)
    assert report["stale_reason"]["code"] == "qc_changed"
    assert report["stale_reason"]["row_id"] == first.id
    assert report["stale_reason"]["column_id"] == toc.id


def test_qc_calculation_exclusion_recomputes_and_undo_restores_derived_column(qtbot):
    from autopyro.table import TableModel, AddColumnCommand, QCCommand
    from autopyro.calculations import make_calculated_column
    from autopyro.schemes.interpretation import evaluate_project
    from autopyro.schemes.standard_interpretation import standard_schemes

    project = dataset([["A", "001", "2", "6"], ["A", "002", "3", "6"]])
    model = TableModel(project)
    sid, toc, s2 = project.rows[0].id, project.columns[2], project.columns[3]
    config = {"formula": "HI", "version": 1, "analysis_context": "test", "compatible_confirmed": True,
              "inputs": {"TOC": {"column_id": toc.id, "unit": toc.unit}, "S2": {"column_id": s2.id, "unit": s2.unit}}}
    hi, values = make_calculated_column(project, "HI", config)
    model.undo.push(AddColumnCommand(model, hi, values))
    assert project.rows[0].values[hi.id] == 300
    report = evaluate_project(project, [sid], toc.id, hi.id, toc.unit, hi.unit,
                              standard_schemes()["HI–TOC"])
    project.view["interpretation_result"] = report

    other = project.rows[1]
    other_measurement = project.science.current(other.id, toc.id)
    model.undo.push(QCCommand(model, ("qc_exclusions", "calculations"),
        {"revision": other_measurement.revision, "comment": "review other TOC"},
        measurement_id=other_measurement.id))
    assert other.values[hi.id] is None
    assert not report["stale"]
    model.undo.undo()
    assert other.values[hi.id] == 200
    assert not report["stale"]

    measurement = project.science.current(sid, toc.id)
    model.undo.push(QCCommand(model, ("qc_exclusions", "calculations"),
        {measurement.id: {"revision": measurement.revision, "comment": "review TOC"}}))
    assert project.rows[0].values[hi.id] is None
    assert report["stale_reason"] == {
        "code": "input_changed", "row_id": sid, "column_id": hi.id, "axis": "y", "diagram": None,
    }
    model.undo.undo()
    assert project.rows[0].values[hi.id] == 300


def test_source_file_changed_after_preview_cannot_be_imported(qtbot, tmp_path, monkeypatch):
    from autopyro.dialogs import ImportDialog, QMessageBox
    path = tmp_path / "lab.csv"
    path.write_text("TOC;S2\n2;6\n", encoding="utf-8")
    errors = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, message: errors.append(message))
    dialog = ImportDialog(path=path)
    qtbot.addWidget(dialog)
    dialog.load()
    qtbot.waitUntil(lambda: dialog.raw is not None and dialog.worker is not None and not dialog.worker.isRunning(), timeout=10000)
    path.write_text("TOC;S2\n3;9\n", encoding="utf-8")
    dialog.import_data()
    qtbot.waitUntil(lambda: bool(errors), timeout=10000)
    assert "изменился" in errors[0]
    assert dialog.project is None
