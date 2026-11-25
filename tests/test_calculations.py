import pytest
from copy import deepcopy

from autopyro.calculations import FORMULAS, calculate_column, make_calculated_column, recompute
from autopyro.core import Project
from autopyro.io import build_project


def test_diagnostic_snapshot_keeps_complete_data_and_isolates_mutations():
    from autopyro.calculations import snapshot_diagnostics

    original = {
        "ok": {"status": "ok", "inputs_canonical": {"TOC": 2., "S2": 2**70}},
        "bad": {"status": "missing", "reason": "TOC missing"},
        "extended": {"status": "ok", "inputs_canonical": {"S2": [1, 2]},
                     "custom": {"trace": (1, {"history": [3]})}},
        "nonstandard": None,
    }
    expected = deepcopy(original)
    copied = snapshot_diagnostics(original)
    assert copied == expected
    original["ok"]["inputs_canonical"]["TOC"] = 99
    original["bad"]["reason"] = "changed"
    original["extended"]["inputs_canonical"]["S2"].append(4)
    original["extended"]["custom"]["trace"][1]["history"].append(5)
    assert copied == expected
    assert copied["ok"]["inputs_canonical"]["S2"] == 2**70


@pytest.mark.parametrize("new_value", [8., "ND", 0.])
def test_partial_recalculation_matches_full_with_dependency_chain(pyro, new_value):
    from autopyro.core import Row

    pyro.rows.append(Row(deepcopy(pyro.rows[0].values)))
    gp, values = make_calculated_column(pyro, "GP", config(pyro, "GP"))
    pyro.columns.append(gp)
    for row in pyro.rows:
        row.values[gp.id] = values[row.id]
    hi_config = config(pyro, "HI")
    hi_config["inputs"]["S2"] = {"column_id": gp.id, "unit": gp.unit}
    hi, values = make_calculated_column(pyro, "HI from GP", hi_config)
    pyro.columns.insert(0, hi)  # Dependency order differs from table order.
    for row in pyro.rows:
        row.values[hi.id] = values[row.id]
    untouched = hi.metadata["diagnostics"][pyro.rows[1].id]
    s2 = next(column for column in pyro.columns if column.name == "S2")
    pyro.rows[0].values[s2.id] = new_value
    reference = deepcopy(pyro)
    recompute(reference)
    recompute(pyro, rows=[pyro.rows[0]])
    assert [row.values for row in pyro.rows] == [row.values for row in reference.rows]
    for actual, expected in zip(pyro.columns, reference.columns):
        assert actual.metadata.get("diagnostics") == expected.metadata.get("diagnostics")
    assert hi.metadata["diagnostics"][pyro.rows[1].id] is untouched


@pytest.fixture
def pyro():
    return build_project(["TOC", "S1", "S2", "S3"], [["2", ".5", "6", "1.2"]], [2],
                         [(0, "TOC", "number", "%wt"), (1, "S1", "number", "mg HC/g rock"),
                          (2, "S2", "number", "mg HC/g rock"), (3, "S3", "number", "mg CO2/g rock")], {})


def config(project, formula):
    columns = {c.name: c for c in project.columns}
    return {"formula": formula, "version": 1, "analysis_context": "Pyrol6, whole rock, test",
            "compatible_confirmed": True, "inputs": {
                role: {"column_id": columns[role].id, "unit": columns[role].unit}
                for role in FORMULAS[formula]["roles"]}}


@pytest.mark.parametrize("scope_kind", ["all", "subset", "empty", "duplicates", "unknown"])
def test_saved_scope_matches_scalar_interpretation_on_full_and_single_row(pyro, scope_kind):
    from autopyro.calculations import calculate_cell
    from autopyro.core import Row

    pyro.rows.extend([Row(deepcopy(pyro.rows[0].values)) for _ in range(3)])
    scopes = {"all": [r.id for r in pyro.rows], "subset": [pyro.rows[1].id], "empty": [],
              "duplicates": [pyro.rows[1].id] * 2, "unknown": ["absent UUID"]}
    cfg = config(pyro, "HI")
    cfg["scope_row_ids"] = scopes[scope_kind]
    for row_ids in (None, [pyro.rows[1].id]):
        values, diagnostics = calculate_column(pyro, cfg, row_ids=row_ids)
        selected = pyro.rows if row_ids is None else [pyro.rows[1]]
        for row in selected:
            expected_value, expected_diagnostic = calculate_cell(pyro, row, cfg)
            assert values[row.id] == expected_value
            assert diagnostics[row.id] == expected_diagnostic


@pytest.mark.parametrize("formula,expected", [("HI", 300), ("OI", 60), ("PI", 1 / 13), ("GP", 6.5), ("OSI", 25)])
def test_reference_formulas(pyro, formula, expected):
    values, diagnostics = calculate_column(pyro, config(pyro, formula))
    assert values[pyro.rows[0].id] == pytest.approx(expected)
    assert diagnostics[pyro.rows[0].id]["status"] == "ok"


def test_context_conflicts_are_row_specific_and_block_lab_comparison(pyro):
    from autopyro.analysis.calculations import calculate_cell, compare_lab_results

    pyro.sync_measurements()
    row = pyro.rows[0]
    toc, s1, s2, lab = pyro.columns
    for column, method in ((toc, "A"), (s1, "B"), (s2, "A"), (lab, "C")):
        working = pyro.science.add_result(row.id, column.id, row.values[column.id],
                                          analysis_fields={"method": method})
        pyro.select_measurement(row.id, column.id, working.id)
    osi = dict(config(pyro, "OSI"), check_context=True)
    values, diagnostics = calculate_column(pyro, osi)
    assert values[row.id] is None
    assert diagnostics[row.id] == {"status": "context_conflict", "fields": ["method"]}
    assert calculate_cell(pyro, row, osi) == (None, diagnostics[row.id])
    hi, _ = calculate_column(pyro, dict(config(pyro, "HI"), check_context=True))
    lab_values = compare_lab_results(pyro, lab, "HI", hi, [row.id], "mg HC/g TOC", 1000,
                                     text_numbers=False, inputs=(toc.id, s2.id))
    assert lab_values.loc[row.id, "status"] == "context_conflict"
    assert lab_values.loc[row.id, "context_fields"] == ("method",)


@pytest.mark.parametrize("s2_basis, lab_basis, calculation_conflict, lab_conflict", [
    ("as_received", "dry", True, True),
    (" dry ", "DRY", False, False),
    ("", "", False, False),
    ("dry", "as_received", False, True),
])
def test_measurement_basis_controls_formula_and_lab_comparability(
    pyro, s2_basis, lab_basis, calculation_conflict, lab_conflict
):
    from autopyro.analysis.calculations import calculate_cell, compare_lab_results, context_conflicts

    pyro.sync_measurements()
    row = pyro.rows[0]
    toc, lab, s2, _ = pyro.columns
    for column, value, basis in ((toc, 2, "dry"), (s2, 6, s2_basis), (lab, 300, lab_basis)):
        measurement = pyro.science.add_result(
            row.id, column.id, value, analysis_fields={"method": "RE6", "laboratory": "A"},
            details={"basis": basis},
        )
        pyro.select_measurement(row.id, column.id, measurement.id)
    settings = dict(config(pyro, "HI"), check_context=True)
    values, diagnostics = calculate_column(pyro, settings)
    inputs = (toc.id, s2.id)
    assert context_conflicts(pyro, row.id, inputs) == (["basis"] if calculation_conflict else [])
    assert calculate_cell(pyro, row, settings) == (values[row.id], diagnostics[row.id])
    assert values[row.id] == (None if calculation_conflict else 300)

    result = compare_lab_results(pyro, lab, "HI", {row.id: 300}, [row.id], "mg HC/g TOC", 0,
                                 text_numbers=False, inputs=inputs)
    assert result.at[row.id, "status"] == ("context_conflict" if lab_conflict else "match")
    assert result.at[row.id, "context_fields"] == (("basis",) if lab_conflict else ())
    with pytest.raises(ValueError, match="вход формулы"):
        compare_lab_results(pyro, s2, "HI", values, [row.id], "mg HC/g TOC", 300,
                            text_numbers=False, inputs=inputs)


def test_scope_filters_rows_before_input_classification(pyro, monkeypatch):
    from autopyro.analysis import calculations
    from autopyro.model.core import Row

    selected = pyro.rows[0]
    pyro.rows.append(Row(deepcopy(selected.values)))
    classify = calculations.classify_series

    def classify_scoped(values, **kwargs):
        assert list(values.index) == [selected.id]
        return classify(values, **kwargs)

    monkeypatch.setattr(calculations, "classify_series", classify_scoped)
    settings = dict(config(pyro, "HI"), scope_row_ids=[selected.id])
    values, diagnostics = calculate_column(pyro, settings)
    assert values == {selected.id: 300, pyro.rows[1].id: None}
    assert diagnostics[pyro.rows[1].id]["status"] == "outside_scope"


@pytest.mark.parametrize("formula,contributions", [
    ("HI", {"TOC": 15, "S2": 10}),
    ("OI", {"TOC": 3, "S3": 2.5}),
    ("PI", {"S1": 6 * .03 / 6.5**2, "S2": .5 * .2 / 6.5**2}),
    ("GP", {"S1": .03, "S2": .2}),
    ("OSI", {"TOC": 1.25, "S1": 1.5}),
])
def test_standard_uncertainty_reports_each_contribution(pyro, formula, contributions):
    from math import hypot
    from autopyro.calculations import calculate_cell

    pyro.sync_measurements()
    row = pyro.rows[0]
    for column, amount in zip(pyro.columns, (.1, .03, .2, .05)):
        measurement = pyro.science.current(row.id, column.id)
        measurement.uncertainty = amount
        measurement.uncertainty_type = "standard"
    cfg = dict(config(pyro, formula), propagate_uncertainty=True)
    values, diagnostics = calculate_column(pyro, cfg)
    info = diagnostics[row.id]["uncertainty"]
    assert info["status"] == "ok"
    assert info["standard"] == pytest.approx(hypot(*contributions.values()))
    assert info["contributions"] == pytest.approx(contributions)
    scalar, scalar_info = calculate_cell(pyro, row, cfg)
    assert scalar == pytest.approx(values[row.id])
    assert scalar_info["uncertainty"]["standard"] == pytest.approx(info["standard"])


def test_uncertainty_converts_units_and_requires_all_working_inputs():
    pyro = build_project(["TOC", "S2"], [["0.02", "6000"]], [2],
                         [(0, "TOC", "number", "g/g rock"),
                          (1, "S2", "number", "ug HC/g rock")], {})
    pyro.sync_measurements()
    row = pyro.rows[0]
    toc, s2 = [pyro.science.current(row.id, c.id) for c in pyro.columns]
    toc.uncertainty, toc.uncertainty_type = 0, "standard"
    s2.uncertainty, s2.uncertainty_type = 200, "standard"
    cfg = dict(config(pyro, "HI"), propagate_uncertainty=True)
    _, diagnostics = calculate_column(pyro, cfg)
    assert diagnostics[row.id]["uncertainty"]["contributions"] == {"TOC": 0, "S2": 10}

    toc.uncertainty = None
    s2.uncertainty_type = "expanded"
    values, diagnostics = calculate_column(pyro, cfg)
    assert values[row.id] == 300
    assert diagnostics[row.id]["uncertainty"] == {
        "status": "unavailable", "inputs": {"TOC": "missing", "S2": "unsupported_type"},
    }


@pytest.mark.parametrize("formula,raw,bounds", [
    ("PI", {"S1": "<0,10", "S2": .90}, (0, .1, True, False)),
    ("PI", {"S1": .90, "S2": "≤0.10"}, (.9, 1, True, True)),
    ("PI", {"S1": "<0.1", "S2": "<0.2"}, (0, 1, True, True)),
    ("PI", {"S1": "<0.1", "S2": 0}, (1, 1, True, True)),
    ("PI", {"S1": "<=0", "S2": 0}, None),
    ("HI", {"TOC": "<2"}, (300, None, False, False)),
    ("HI", {"TOC": "<200"}, (6, None, True, False)),
    ("HI", {"TOC": "<2", "S2": 0}, (0, 0, True, True)),
    ("OI", {"S3": "<=0.2"}, (0, 10, True, True)),
    ("OSI", {"S1": "<0.1"}, (0, 5, True, False)),
    ("GP", {"S1": "<0.1"}, (6, 6.1, True, False)),
    ("GP", {"S1": "<0"}, None),
    ("GP", {"S1": ">1"}, None),
    ("GP", {"S1": "<0.1", "S2": "ND"}, None),
])
def test_censored_ranges_preserve_boundaries_and_leave_numeric_value_empty(pyro, formula, raw, bounds):
    from autopyro.calculations import calculate_cell

    row = pyro.rows[0]
    row.values.update({c.id: raw[c.name] for c in pyro.columns if c.name in raw})
    cfg = config(pyro, formula)
    assert calculate_column(pyro, cfg)[1][row.id]["status"] != "censored_range"
    cfg["show_censored_ranges"] = True
    values, diagnostics = calculate_column(pyro, cfg)
    assert values[row.id] is None
    info = diagnostics[row.id]
    assert calculate_cell(pyro, row, cfg) == (None, info)
    if bounds is None:
        assert info["status"] != "censored_range"
        return
    lower, upper, lower_closed, upper_closed = bounds
    assert info["status"] == "censored_range"
    assert info["range"] == {
        "lower": pytest.approx(lower), "upper": pytest.approx(upper) if upper is not None else None,
        "lower_closed": lower_closed, "upper_closed": upper_closed,
    }


@pytest.mark.parametrize("raw,interpolated,reason", [
    (None, False, "missing"),
    ("ND", False, "non_detect"),
    (">0.2", False, "lower_limit"),
    (0.9, True, "interpolation_input"),
])
def test_censored_range_explains_blocking_second_input(pyro, raw, interpolated, reason):
    from autopyro.calculations import calculate_cell

    row = pyro.rows[0]
    s1, s2 = pyro.columns[1:3]
    row.values.update({s1.id: "<0.1", s2.id: raw})
    if interpolated:
        s2.metadata["method"] = "linear"
    cfg = dict(config(pyro, "PI"), show_censored_ranges=True)
    values, diagnostics = calculate_column(pyro, cfg)
    expected = {"status": "censored", "range_issue": reason, "role": "S2"}
    assert diagnostics[row.id] == expected
    assert values[row.id] is None
    assert calculate_cell(pyro, row, cfg) == (None, expected)


def test_censored_diagnostic_shows_all_blocking_inputs(pyro):
    from autopyro.calculations import calculate_cell
    from autopyro.ui.table.model import range_issue_text

    row = pyro.rows[0]
    s1, s2 = pyro.columns[1:3]
    row.values.update({s1.id: "<LOD", s2.id: "ND"})
    cfg = dict(config(pyro, "PI"), show_censored_ranges=True)
    values, diagnostics = calculate_column(pyro, cfg)
    info = diagnostics[row.id]
    assert values[row.id] is None
    assert info["range_issues"] == [
        {"range_issue": "missing_lod", "role": "S1"},
        {"range_issue": "non_detect", "role": "S2"},
    ]
    assert calculate_cell(pyro, row, cfg) == (None, info)
    assert range_issue_text(info).splitlines()[-1] == "S2: ND"
    assert range_issue_text(info).startswith("S1: ")


def test_symbolic_lod_uses_working_result_and_respects_qc():
    pyro = build_project(["S1", "S2"], [["<LOD", "0.9"]], [2],
                         [(0, "S1", "number", "ug HC/g rock"),
                          (1, "S2", "number", "mg HC/g rock")], {})
    pyro.sync_measurements()
    row = pyro.rows[0]
    s1, s2 = pyro.columns
    cfg = dict(config(pyro, "PI"), show_censored_ranges=True)
    assert calculate_column(pyro, cfg)[1][row.id]["status"] == "censored"
    pyro.science.current(row.id, s1.id).lod = 100
    assert calculate_column(pyro, cfg)[1][row.id]["range"]["upper"] == pytest.approx(.1)
    excluded = pyro.science.current(row.id, s2.id)
    pyro.view["qc_exclusions"] = {"calculations": {excluded.id: {"revision": excluded.revision}}}
    assert calculate_column(pyro, cfg)[1][row.id]["status"] != "censored_range"


@pytest.mark.parametrize("formula,column_index,raw,expected", [
    ("HI", 2, 6000, 300), ("OSI", 1, 500, 25),
])
def test_explicit_unit_conversion_and_text_policy(pyro, formula, column_index, raw, expected):
    toc, numerator = pyro.columns[0], pyro.columns[column_index]
    toc.unit, numerator.unit = "g/g rock", "ug HC/g rock"
    pyro.rows[0].values[toc.id], pyro.rows[0].values[numerator.id] = .02, raw
    assert calculate_column(pyro, config(pyro, formula))[0][pyro.rows[0].id] == pytest.approx(expected)
    toc.kind = "text"
    pyro.rows[0].values[toc.id] = "0,02"
    cfg = config(pyro, formula)
    assert calculate_column(pyro, cfg)[0][pyro.rows[0].id] is None
    cfg["text_numbers"] = True
    assert calculate_column(pyro, cfg)[0][pyro.rows[0].id] == pytest.approx(expected)


@pytest.mark.parametrize("value,status", [(0, "zero_denominator"), (-1, "invalid_range"), (101, "invalid_range"),
                                          (None, "missing"), ("<0.1", "censored"), ("ND", "non_detect")])
def test_bad_inputs_have_no_invented_result(pyro, value, status):
    pyro.rows[0].values[pyro.columns[0].id] = value
    values, diagnostics = calculate_column(pyro, config(pyro, "HI"))
    assert values[pyro.rows[0].id] is None
    assert diagnostics[pyro.rows[0].id]["status"] == status


def test_units_and_context_cannot_be_implicitly_overridden(pyro):
    cfg = config(pyro, "HI")
    cfg["inputs"]["TOC"]["unit"] = "g/g rock"
    with pytest.raises(ValueError, match="не совпадает"):
        calculate_column(pyro, cfg)
    cfg = config(pyro, "HI")
    cfg["compatible_confirmed"] = False
    with pytest.raises(ValueError, match="совместимость"):
        calculate_column(pyro, cfg)


def test_scope_interpolated_input_and_zero_pi(pyro):
    cfg = config(pyro, "HI")
    cfg["scope_row_ids"] = []
    assert calculate_column(pyro, cfg)[1][pyro.rows[0].id]["status"] == "outside_scope"
    cfg.pop("scope_row_ids")
    pyro.columns[2].metadata["method"] = "linear"
    assert calculate_column(pyro, cfg)[1][pyro.rows[0].id]["status"] == "interpolation_input"
    pyro.columns[2].metadata.clear()
    pyro.rows[0].values[pyro.columns[1].id] = 0
    pyro.rows[0].values[pyro.columns[2].id] = 0
    assert calculate_column(pyro, config(pyro, "PI"))[1][pyro.rows[0].id]["status"] == "zero_denominator"


def test_recompute_and_reload_replace_cached_values(pyro, tmp_path):
    column, values = make_calculated_column(pyro, "HI", config(pyro, "HI"))
    pyro.columns.append(column)
    pyro.rows[0].values[column.id] = -999
    path = tmp_path / "calc.apyro"
    pyro.save(path)
    restored = Project.load(path)
    assert restored.rows[0].values[column.id] == 300
    restored.rows[0].values[restored.columns[2].id] = 8
    recompute(restored)
    assert restored.rows[0].values[column.id] == 400
    restored.columns.pop(2)
    recompute(restored)
    assert restored.rows[0].values[column.id] is None
    assert restored.columns[-1].metadata["diagnostics"][restored.rows[0].id]["status"] == "invalid_definition"


def test_cycle_is_explicit_and_cannot_reuse_old_output(pyro):
    column, values = make_calculated_column(pyro, "GP", config(pyro, "GP"))
    pyro.columns.append(column)
    pyro.rows[0].values[column.id] = values[pyro.rows[0].id]
    column.metadata["calculation"]["inputs"]["S2"]["column_id"] = column.id
    recompute(pyro)
    assert pyro.rows[0].values[column.id] is None
    assert column.metadata["diagnostics"][pyro.rows[0].id]["status"] == "cyclic_dependency"
