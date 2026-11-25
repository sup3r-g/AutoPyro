import csv
from dataclasses import replace

import pytest

from autopyro.core import Column
from autopyro.io import build_project
from autopyro.statistics import (
    StatisticsOptions, calculate_statistics, classify, compare_statistics,
    export_statistics_csv, interpolation_comparison, interpolation_pairs, repeat_statistics,
    snapshot, snapshot_steps, threshold_diagnostics,
)


def test_calculated_snapshot_exports_full_diagnostics_and_reproduces_statistics(data):
    import json
    from dataclasses import asdict
    from autopyro.calculations import make_calculated_column
    from autopyro.core import Project, Column, Row

    x, y = data.columns[1:]
    x.unit, y.unit = "%wt", "mg HC/g rock"
    config = {"formula": "HI", "version": 1, "compatible_confirmed": True,
              "analysis_context": "Computational QA",
              "inputs": {"TOC": {"column_id": x.id, "unit": x.unit},
                         "S2": {"column_id": y.id, "unit": y.unit}}}
    column, values = make_calculated_column(data, "HI", config)
    data.columns.append(column)
    for row in data.rows:
        row.values[column.id] = values[row.id]
    ids = [row.id for row in data.rows[:4]]
    frozen = snapshot(data, ids, [column.id])
    expected = {rid: item for rid, item in column.metadata["diagnostics"].items() if rid in ids}
    payload = json.loads(json.dumps({"columns": [asdict(c) for c in frozen.columns],
                                     "rows": [asdict(r) for r in frozen.rows]}))
    assert payload["columns"][0]["metadata"]["diagnostics"] == expected
    assert payload["columns"][0]["metadata"]["calculation"] == config
    column.metadata["diagnostics"][ids[0]]["inputs_canonical"]["TOC"] = 999
    assert frozen.columns[0].metadata["diagnostics"][ids[0]]["inputs_canonical"]["TOC"] == 1
    reconstructed = Project(columns=[Column(**c) for c in payload["columns"]],
                            rows=[Row(**r) for r in payload["rows"]])
    options = StatisticsOptions((column.id,))
    assert calculate_statistics(reconstructed, ids, options)["summary"] == calculate_statistics(frozen, ids, options)["summary"]


def test_batched_snapshot_matches_synchronous_copy(data):
    data.sync_measurements()
    ids = [row.id for row in data.rows[:4]]
    columns = [column.id for column in data.columns[1:]]
    steps = list(snapshot_steps(data, ids, columns, batch_size=2))
    assert len(steps) > 1 and all(step is None for step in steps[:-1])
    assert steps[-1] == snapshot(data, ids, columns)


def test_snapshot_scopes_diagnostics_and_freezes_working_result_and_qc(data):
    data.sync_measurements()
    row = data.rows[0]
    column = data.columns[1]
    old = data.science.current(row.id, column.id)
    old.history.append({"previous": {"value": 7}})
    repeat = data.science.add_result(row.id, column.id, 99,
                                    analysis_fields={"laboratory": "Repeat lab"})
    data.view["qc_exclusions"] = {"statistics": {
        old.id: {"revision": old.revision, "comment": {"notes": ["reviewed"]}}}}
    column.metadata["diagnostics"] = {
        row.id: {"trace": ["included"]}, data.rows[1].id: {"trace": ["excluded"]}}
    notes = ["source"]
    row.source.update({"notes": notes, "shared_notes": notes})
    frozen = snapshot(data, [row.id], [column.id])
    data.select_measurement(row.id, column.id, repeat.id)
    data.view["qc_exclusions"]["statistics"][old.id]["comment"]["notes"].append("changed")
    column.metadata["diagnostics"][row.id]["trace"].append("changed")
    notes.append("changed")
    old.history[0]["previous"]["value"] = 10
    repeat.history.append({"previous": {"value": 98}})
    assert frozen.science.current(row.id, column.id).id == old.id
    assert repeat.id in frozen.science.measurements
    assert frozen.science.measurements[old.id].history[0]["previous"]["value"] == 7
    assert frozen.science.measurements[repeat.id].history == []
    assert frozen.view["qc_exclusions"]["statistics"][old.id]["comment"]["notes"] == ["reviewed"]
    assert frozen.columns[0].metadata["diagnostics"] == {row.id: {"trace": ["included"]}}
    assert frozen.rows[0].values[column.id] == 1
    assert frozen.rows[0].source["notes"] == ["source"]
    assert frozen.rows[0].source["notes"] is frozen.rows[0].source["shared_notes"]


@pytest.fixture
def data():
    rows = [["A", "1", "2"], ["A", "2", "4"], ["A", "3", "6"],
            ["A", "4", None], ["B", None, "10"], [None, "10", "20"]]
    return build_project(["Well", "X", "Y"], rows, list(range(2, 8)),
                         [(0, "Well", "text", ""), (1, "X", "number", "m"),
                          (2, "Y", "number", "mg/g")], {"type": "test"})


def test_reference_summary_and_pair_counts(data):
    _, x, y = [c.id for c in data.columns]
    report = calculate_statistics(data, [r.id for r in data.rows[:4]], StatisticsOptions((x, y)))
    summary = report["summary"][0]
    assert summary["n"] == 4
    assert summary["mean"] == 2.5
    assert summary["median"] == 2.5
    assert summary["std"] == pytest.approx((5 / 3) ** 0.5)
    assert summary["q1"] == 1.75
    assert summary["q3"] == 3.25
    assert summary["p10"] == pytest.approx(1.3)
    assert summary["p90"] == pytest.approx(3.7)
    assert summary["iqr"] == 1.5
    pair = report["correlations"][0]
    assert pair["n_pairs"] == 3
    assert pair["pearson"] == pytest.approx(1)
    assert pair["spearman"] == pytest.approx(1)


def test_summary_recovers_large_metrics_and_marks_unrepresentable_results():
    import json

    values = [[1e308, -1.7e308, 1], [1e308, -1.7e308, 2],
              [1e308, 1.7e308, 3], [1e308, 1.7e308, 4]]
    names = ["Constant", "Extreme", "Ordinary"]
    project = build_project(names, values, [2, 3, 4, 5],
                            [(i, name, "number", "") for i, name in enumerate(names)], {})
    ids = [row.id for row in project.rows]
    report = calculate_statistics(project, ids, StatisticsOptions(tuple(c.id for c in project.columns), correlations=False))
    constant, extreme, ordinary = report["summary"]
    assert (constant["mean"], constant["std"], constant["outliers"]) == (1e308, 0, 0)
    assert constant["summary_status"] is None
    assert (extreme["mean"], extreme["median"], extreme["std"], extreme["iqr"], extreme["outliers"]) == (0, 0, None, None, None)
    assert extreme["numeric_issues"] == ["std", "iqr", "outlier_limits"]
    assert extreme["summary_status"] == "numerical_warning"
    assert ordinary["median"] == 2.5
    assert report["numerical_warning_count"] == 1
    json.dumps(report, allow_nan=False)


def test_comparison_excludes_qc_and_checks_measurement_context(data):
    cid = data.columns[1].id
    first, excluded, other = (row.id for row in data.rows[:3])
    data.sync_measurements()
    for analysis in data.science.analyses.values():
        analysis.method = "Reference method"
    measurement = data.science.current(excluded, cid)
    data.view["qc_exclusions"] = {"statistics": {measurement.id: {"revision": measurement.revision}}}
    options = StatisticsOptions((cid,))
    result, = compare_statistics(data, [first, excluded], [other, excluded], options)
    assert result["delta"] == 2
    assert result["samples"]["b"] == [(other, "numeric"), (excluded, "excluded_qc")]
    assert (result["a_used"], result["a_excluded"], result["b_used"], result["b_excluded"]) == (1, 1, 1, 1)

    repeat = data.science.add_result(other, cid, 3, analysis_fields={"method": "Other method"})
    data.select_measurement(other, cid, repeat.id)
    result, = compare_statistics(data, [first], [other], options)
    assert result["b_median"] == 3
    assert result["delta"] is None
    assert result["comparison_status"] == "Разный аналитический контекст"
    result, = compare_statistics(data, [], [other], options)
    assert result["a_used"] == result["a_excluded"] == 0
    assert result["delta"] is None


@pytest.mark.parametrize(("left", "right", "missing", "mismatched", "delta"), [
    ({}, {}, ["method"], [], None),
    ({"method": "M"}, {}, ["method"], [], None),
    ({"method": "M", "laboratory": "A"}, {"method": "M"}, ["laboratory"], [], None),
    ({"method": "M", "laboratory": "A"}, {"method": "M", "laboratory": "B"}, [], ["laboratory"], None),
    ({"method": "M", "instrument": "A"}, {"method": "M", "instrument": "B"}, [], ["instrument"], None),
    ({"method": " M "}, {"method": "M"}, [], [], 1),
])
def test_comparison_requires_known_method_and_consistent_declared_context(data, left, right, missing, mismatched, delta):
    data.sync_measurements()
    cid = data.columns[1].id
    a, b = (row.id for row in data.rows[:2])
    for rid, fields in ((a, left), (b, right)):
        analysis = data.science.analyses[data.science.current(rid, cid).analysis_id]
        for name, value in fields.items():
            setattr(analysis, name, value)
    result, = compare_statistics(data, [a], [b], StatisticsOptions((cid,)))
    assert (result["a_median"], result["b_median"], result["delta"]) == (1, 2, delta)
    assert result["context_missing"] == missing
    assert result["context_mismatched"] == mismatched
    assert result["comparison_status"] == (
        "Разный аналитический контекст" if mismatched else "Аналитический контекст неизвестен" if missing else "OK"
    )


def test_summary_excludes_mixed_units_and_bases_per_group(data):
    group, x, y = data.columns
    x.unit = "%"
    for row in data.rows:
        row.values[group.id] = "First" if row in data.rows[:3] else "Second"
    data.sync_measurements()
    data.science.current(data.rows[0].id, x.id).unit = "ppm"
    for row in data.rows[3:]:
        value = data.science.current(row.id, x.id)
        if value is not None:
            value.unit = "ppm"

    options = StatisticsOptions((x.id, y.id), (group.id,))
    report = calculate_statistics(data, data.row_map(), options)
    first, second = report["summary"][::2]
    assert (first["median"], first["n"], first["excluded_context"]) == (None, 0, 3)
    assert first["summary_status"] == "incompatible_context"
    assert report["correlations"][0]["n_pairs"] == 0
    assert second["n"] and second["unit"] == "ppm"

    ids = [row.id for row in data.rows[:3]]
    result, = compare_statistics(data, ids[:1], ids[1:], StatisticsOptions((x.id,)))
    assert (result["a_unit"], result["b_unit"], result["delta"]) == ("ppm", "%", None)
    assert result["context_details"]["unit"] == {"a": ["ppm"], "b": ["%"]}
    a, b = [data.science.current(rid, x.id) for rid in ids[:2]]
    a.unit = b.unit
    a.basis, b.basis = "dry", "wet"
    summary = calculate_statistics(data, ids, options)["summary"][0]
    assert summary["summary_status"] == "incompatible_context"


def test_groups_include_missing_category_and_empty_scope(data):
    group, x, y = [c.id for c in data.columns]
    report = calculate_statistics(data, data.row_map(), StatisticsOptions((x, y), (group,)))
    entries = {(tuple(e["group"]), e["column_id"]): e for e in report["summary"]}
    assert entries[(("A",), x)]["n"] == 4
    assert (entries[(("A",), y)]["n"], entries[(("A",), y)]["missing"]) == (3, 1)
    assert entries[(("B",), x)]["mean"] is None
    assert entries[(("B",), x)]["missing"] == 1
    assert entries[((None,), x)]["std"] is None
    assert [pair["n_pairs"] for pair in report["correlations"]] == [3, 0, 1]
    empty = calculate_statistics(data, [], StatisticsOptions((x, y)))
    assert empty["summary"] == []
    assert empty["row_ids"] == []


def test_repeat_statistics_pairs_samples_and_respects_context_and_qc(data):
    group, column = data.columns[:2]
    data.sync_measurements()
    first, second, outside = (row.id for row in data.rows[:3])
    def result(rid, value, method="M1", lab="A", basis="dry"):
        return data.science.add_result(rid, column.id, value,
                                       analysis_fields={"method": method, "laboratory": lab},
                                       details={"basis": basis})

    repeated = result(first, 2)
    duplicate = replace(repeated, id="same-analysis", value=4, raw_value=4)
    data.science.measurements[duplicate.id] = duplicate
    result(first, 7)
    result(first, 11, lab="B")
    result(first, 13, method="M2")
    result(first, 300, lab="B", basis="wet")
    result(second, 1)
    result(second, 5, lab="B")
    result(outside, 100, lab="B")
    rejected = result(first, 99)
    result(first, "<0.1")
    data.view["qc_exclusions"] = {"statistics": {rejected.id: {"revision": rejected.revision}}}

    options = StatisticsOptions((column.id,), (group.id,))
    report = calculate_statistics(data, [first, second], options)
    entries = repeat_statistics(data, report)
    assert len(entries) == 4
    spread = next(entry for entry in entries if entry["method_b"] is None)
    assert (spread["group"], spread["n"], spread["median"]) == (["A"], 1, 4)
    assert spread["samples"][0]["a_count"] == 2
    labs = next(entry for entry in entries if entry["lab_b"] == "B" and entry["method_a"] == "M1")
    assert [sample["difference"] for sample in labs["samples"]] == [6, 4]
    assert (labs["n"], labs["median"]) == (2, 5)
    assert labs["p10"] == pytest.approx(4.2)
    assert labs["p90"] == pytest.approx(5.8)


def test_text_censoring_is_explicit_and_never_zero_imputed():
    values = [[v] for v in ["1,5", "2.5", "<0.1", "≥3", "ND", None, "NaN", "unverified"]]
    project = build_project(["TOC"], values, list(range(1, 9)), [(0, "TOC", "text", "%")], {})
    cid = project.columns[0].id
    raw = calculate_statistics(project, project.row_map(), StatisticsOptions((cid,)))
    assert raw["summary"][0]["n"] == 0
    options = StatisticsOptions((cid,), text_numbers=True)
    report = calculate_statistics(project, project.row_map(), options)
    entry = report["availability"][0]
    assert entry["n"] == 2
    assert entry["missing"] == 1
    assert entry["censored"] == 2
    assert entry["non_detect"] == 1
    assert entry["other"] == 2
    assert entry["coverage_pct"] == 25
    assert report["summary"][0]["mean"] == 2
    assert project.rows[0].values[cid] == "1,5"


def test_interpolation_comparison_fills_only_blank_sources_and_explains_exclusions():
    values = ["2", None, "<0.1", "ND", "8", "bad", None, "100"]
    project = build_project(["TOC"], [[value] for value in values], list(range(1, 9)),
                            [(0, "TOC", "text", "%")], {})
    project.sync_measurements()
    source = project.columns[0]
    estimate = Column("Linear", "number", unit="%", metadata={"method": "linear", "y_id": source.id})
    stale = Column("Old", "number", unit="%", metadata={
        **estimate.metadata, "stale": True, "interpolation_protocol": "autopyro:protocol:zlib-json:1:corrupt",
    })
    project.columns.extend([estimate, stale])
    for index, row in enumerate(project.rows):
        row.values[estimate.id] = row.values[stale.id] = None if index == 6 else 4
    working = project.science.current(project.rows[-1].id, source.id)
    project.view["qc_exclusions"] = {"statistics": {working.id: {"revision": working.revision}}}
    options = StatisticsOptions((source.id,), text_numbers=True)
    ids = list(project.row_map())
    pairs = interpolation_pairs(project, options.column_ids)
    data = snapshot(project, ids, (source.id, estimate.id, stale.id))
    report = calculate_statistics(data, ids, options)
    current, old = interpolation_comparison(data, report, pairs, options)
    assert (current["measured"], current["added"], current["no_estimate"]) == (2, 1, 1)
    assert [project.rows[6].id, None, None, "no_estimate"] in current["samples"]
    assert (current["before_pct"], current["after_pct"]) == (25, 37.5)
    assert (current["before_median"], current["after_median"], current["delta"]) == (5, 4, -1)
    assert [current[key] for key in (
        "source_numeric", "source_censored", "source_non_detect", "source_other", "source_excluded_qc",
    )] == [2, 1, 1, 1, 1]
    assert (old["stale"], old["added"], old["after_median"]) == (7, 0, 5)
    assert current["sample_provenance"] == old["sample_provenance"] == {}


def test_threshold_diagnostics_groups_working_records_and_separates_literal_limits():
    values = ["<0,1", "≤0.3", "<LOD", ">=5", "н/о", "0", None, "ND", "<2", ">9"]
    project = build_project(["TOC"], [[value] for value in values], list(range(1, 11)),
                            [(0, "TOC", "text", "%")], {})
    cid = project.columns[0].id
    project.sync_measurements()
    for index, (row, value) in enumerate(zip(project.rows, values)):
        result = project.science.add_result(
            row.id, cid, value, analysis_fields={"method": "M2" if index == 8 else "M1",
                                                 "laboratory": "B" if index == 7 else "A"},
            details={"lod": 0.04, "loq": 0.09} if index == 0 else {},
        )
        project.select_measurement(row.id, cid, result.id)
    result = project.science.current(project.rows[0].id, cid)
    project.view["qc_exclusions"] = {"statistics": {result.id: {"revision": result.revision}}}
    project.science.add_result(project.rows[0].id, cid, "<100", analysis_fields={"method": "M1", "laboratory": "A"})
    ids = [row.id for row in project.rows[:-1]]
    entries = threshold_diagnostics(project, ids, StatisticsOptions((cid,)))
    by_context = {(entry["method"], entry["lab"]): entry for entry in entries}
    group = by_context["M1", "A"]
    assert (group["n"], group["below"], group["above"], group["non_detect"]) == (6, 3, 1, 1)
    assert (group["below_min"], group["below_max"], group["above_min"], group["above_max"]) == (0.1, 0.3, 5, 5)
    assert (group["below_pct"], group["no_numeric_limit"], group["excluded_qc"]) == (50, 1, 1)
    assert group["non_detect_pct"] == pytest.approx(100 / 6)
    assert group["samples"][0] == {"row_id": project.rows[0].id, "value": "<0,1", "status": "below",
                                   "limit": 0.1, "lod": 0.04, "loq": 0.09, "excluded_qc": True}
    assert by_context["M1", "B"]["non_detect_pct"] == 100
    assert by_context["M2", "A"]["below_min"] == 2


def test_constant_and_insufficient_pairs_have_no_coefficient(data):
    x, y = [c.id for c in data.columns[1:]]
    options = StatisticsOptions((x, y))
    small = calculate_statistics(data, [r.id for r in data.rows[:2]], options)
    assert small["correlations"][0]["status"] == "insufficient_pairs"
    for row in data.rows:
        row.values[x] = 1
    constant = calculate_statistics(data, data.row_map(), options)
    assert constant["correlations"][0]["status"] == "constant"
    assert constant["correlations"][0]["pearson"] is None


def test_spearman_handles_ties_and_pair_alignment():
    project = build_project(["X", "Y"], [["1", "30"], ["1", "20"], ["2", "10"], [None, "999"]],
                            [1, 2, 3, 4], [(0, "X", "number", ""), (1, "Y", "number", "")], {})
    result = calculate_statistics(project, project.row_map(), StatisticsOptions(tuple(c.id for c in project.columns)))
    assert result["correlations"][0]["spearman"] == pytest.approx(-(3 ** 0.5) / 2)
    assert result["correlations"][0]["n_pairs"] == 3


def test_spearman_reranks_partial_pairs_after_complete_pairs():
    names = ['X', 'Y', 'Z', 'W']
    project = build_project(names, [[1, 4, 1, 4], [2, 3, 1, None], [3, 2, 2, 2], [4, 1, 3, 1]],
                            [1, 2, 3, 4], [(i, name, 'number', '') for i, name in enumerate(names)], {})
    options = StatisticsOptions(tuple(c.id for c in project.columns))
    result = calculate_statistics(project, project.row_map(), options)
    pairs = {(pair['x'], pair['y']): pair for pair in result['correlations']}
    assert pairs['X', 'Y']['n_pairs'] == 4
    assert pairs['X', 'Y']['spearman'] == pytest.approx(-1)
    assert pairs['X', 'Z']['spearman'] == pytest.approx(3 / (10 ** 0.5))
    assert pairs['X', 'W']['n_pairs'] == 3
    assert pairs['X', 'W']['spearman'] == pytest.approx(-1)


def test_interpolations_excluded_by_default_and_stale_never_used(data):
    c = data.columns[1]
    c.metadata.update(method="linear", stale=False)
    options = StatisticsOptions((c.id,))
    excluded = calculate_statistics(data, data.row_map(), options)
    assert excluded["availability"][0]["excluded_interpolation"] == 5
    included = calculate_statistics(data, data.row_map(), replace(options, include_interpolated=True))
    assert included["availability"][0]["interpolated_used"] == 5
    c.metadata["stale"] = True
    stale = calculate_statistics(data, data.row_map(), replace(options, include_interpolated=True))
    assert stale["summary"][0]["n"] == 0
    assert stale["notices"]


def test_outlier_does_not_modify_or_drop_rows():
    project = build_project(["S2"], [[str(v)] for v in [1, 1, 1, 1, 10]], list(range(1, 6)),
                            [(0, "S2", "number", "")], {})
    before = [r.values.copy() for r in project.rows]
    report = calculate_statistics(project, project.row_map(), StatisticsOptions((project.columns[0].id,)))
    assert report["outliers"][0]["row_id"] == project.rows[-1].id
    assert report["summary"][0]["outliers"] == 1
    assert before == [r.values for r in project.rows]


def test_snapshot_isolates_worker_from_edits_and_export_context(data, tmp_path):
    cid = data.columns[1].id
    snap = snapshot(data, [data.rows[0].id], [cid])
    data.rows[0].values[cid] = 999
    assert snap.rows[0].values[cid] == 1
    report = calculate_statistics(snap, snap.row_map(), StatisticsOptions((cid,)))
    report["scope"] = "Selection"
    path = tmp_path / "summary.csv"
    export_statistics_csv(path, report, "summary")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.reader(stream, delimiter=";"))
    assert records[1][1:3] == ["Selection", "1"]


def test_invalid_options_and_missing_row_are_rejected(data):
    with pytest.raises(ValueError):
        calculate_statistics(data, [], StatisticsOptions(()))
    with pytest.raises(ValueError):
        calculate_statistics(data, ["unknown"], StatisticsOptions((data.columns[1].id,)))
    with pytest.raises(ValueError):
        calculate_statistics(data, [], StatisticsOptions((data.columns[1].id,), min_pairs=2))


@pytest.mark.parametrize("value", ["inf", float("nan"), True, "NaN", "1e999"])
def test_invalid_numbers_are_not_observations(value):
    assert classify(value, numeric=True) == ("other", None)


def test_statistics_csv_preserves_previous_export_on_failure(data, tmp_path, monkeypatch):
    from autopyro.analysis import statistics

    report = calculate_statistics(data, data.row_map(), StatisticsOptions((data.columns[1].id,)))
    path = tmp_path / "summary.csv"
    export_statistics_csv(path, report, "summary", language="en")
    original = path.read_bytes()
    assert "Calculated UTC;Scope" in original.decode("utf-8-sig")
    def fail(target, *args):
        target.write_bytes(b"incomplete")
        raise OSError("write failed")
    monkeypatch.setattr(statistics, "export_table_csv", fail)
    with pytest.raises(OSError):
        export_statistics_csv(path, report, "summary")
    assert path.read_bytes() == original
