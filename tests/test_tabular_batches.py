"""Batch operations preserve scalar diagnostics, identifiers and literal source values."""

from copy import deepcopy
import sqlite3

import pandas as pd
import pytest

from autopyro.calculations import FORMULAS, calculate_cell, calculate_column
from autopyro.core import Column, Project, Row
from autopyro.database.geochem import project_from_records
from autopyro.database.readers import read_sqlite
from autopyro.importing import preview_import, suggest_mapping
from autopyro.io import build_project
from autopyro.plotting import PlotSpec, numeric, plot_columns
from autopyro.tabular import exclusion_mask, project_frame


def test_sqlite_mixed_numeric_column_keeps_large_integer_and_null(tmp_path):
    path = tmp_path / "mixed.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE data (value)")
        connection.executemany("INSERT INTO data VALUES (?)", [(9007199254740993,), (None,), (1.5,)])
    assert read_sqlite(path, "data") == (["value"], [["9007199254740993"], [None], ["1.5"]], [1, 2, 3])


def test_geochem_pivot_keeps_record_order_missing_fields_and_integer_precision():
    fields = [{"source_field_id": 1, "group_name": "G", "source_label": "N",
               "source_dtype": "int64", "unit_label": ""}]
    records = [{"source_record_id": 3, "sample_id": "001"},
               {"source_record_id": 1, "sample_id": "002"},
               {"source_record_id": 2, "sample_id": "003"}]
    results = [{"source_record_id": rid, "source_field_id": 1, "group_name": "G",
                "value_int": value, "value_float": None, "value_text": None,
                "analysis_run_id": rid, "repeat_number": 1}
               for rid, value in [(2, 1), (3, 9007199254740993)]]
    before = deepcopy((records, results))
    project = project_from_records(records, results, fields, source={})
    cid = project.columns[-1].id
    assert [row.values[cid] for row in project.rows] == [9007199254740993, None, 1]
    assert [row.source["source_record_id"] for row in project.rows] == [3, 1, 2]
    assert project.rows[1].source["analysis_runs"] == {}
    assert (records, results) == before
    with pytest.raises(ValueError, match="несколько результатов"):
        project_from_records(records, results + [results[0]], fields, source={})
    with pytest.raises(ValueError, match="несколько результатов"):
        project_from_records(records, results + [results[0], {**results[0], "source_record_id": 999}], fields, source={})
    assert project_from_records([], [], fields, source={}).rows == []


def pyro(kind="number"):
    columns = [Column(name, kind, unit=unit) for name, unit in
               [("TOC", "%wt"), ("S1", "mg HC/g rock"),
                ("S2", "mg HC/g rock"), ("S3", "mg CO2/g rock")]]
    raw = [[2, .5, 6, 1.2], [0, 1, 2, 3], [None, 2, 4, 6], [101, -1, 1, 1],
           [True, "ND", "<0.1", None], ["1,2", "1,5", "3", "bad"],
           [1e-308, 1e308, 1e308, 1e308], [2, 1e308, 1e308, 1e308]]
    project = Project(columns=columns, rows=[Row(dict(zip([c.id for c in columns], values))) for values in raw])
    # Only the valid first row needs a working measurement for exclusion tests.
    project.sync_measurements(rows=project.rows[:1])
    return project


@pytest.mark.parametrize("formula", FORMULAS)
@pytest.mark.parametrize("kind", ["text", "number"])
def test_batch_formulas_match_scalar_values_and_diagnostic_priority(formula, kind):
    project = pyro(kind)
    by_name = {column.name: column for column in project.columns}
    config = {"formula": formula, "version": 1, "analysis_context": "Test",
              "compatible_confirmed": True, "text_numbers": True,
              "inputs": {role: {"column_id": by_name[role].id, "unit": by_name[role].unit}
                         for role in reversed(FORMULAS[formula]["roles"])}}
    for mode in ("all", "scope", "qc", "interpolation", "empty"):
        if mode == "scope":
            config["scope_row_ids"] = [row.id for row in project.rows[::2]]
        elif mode == "qc":
            config.pop("scope_row_ids")
            column = next(iter(config["inputs"].values()))["column_id"]
            measurement = project.science.current(project.rows[0].id, column)
            project.view["qc_exclusions"] = {"calculations": {measurement.id: {"revision": measurement.revision}}}
        elif mode == "interpolation":
            by_name[next(iter(config["inputs"]))].metadata["method"] = "linear"
        elif mode == "empty":
            project.rows = []
        expected = {row.id: calculate_cell(project, row, config) for row in project.rows}
        values, diagnostics = calculate_column(project, config)
        assert values == {rid: result[0] for rid, result in expected.items()}
        assert diagnostics == {rid: result[1] for rid, result in expected.items()}


def test_batch_plotting_matches_scalar_exclusions_and_revisions():
    project = pyro()
    column = project.columns[0]
    first, second = project.rows[:2]
    measurement = project.science.current(first.id, column.id)
    project.view["qc_exclusions"] = {"plots": {measurement.id: {"revision": measurement.revision}}}
    for interpolated in (False, True):
        column.metadata.update(method="linear" if interpolated else "", stale=False)
        frame = project_frame(project, column_ids=[column.id])
        data = plot_columns(project, frame, [column.id], PlotSpec())[column.id]
        assert list(data.itertuples(index=False, name=None)) == [
            numeric(project, row, column.id, PlotSpec()) for row in project.rows]
    measurement.revision += 1
    assert not exclusion_mask(project, pd.Index([first.id, second.id]), column.id, "plots").any()
    assert project_frame(project, [second.id, first.id], [column.id]).index.tolist() == [first.id, second.id]


def test_import_matching_keeps_typed_keys_nulls_and_duplicate_candidates():
    def table(values):
        return build_project(["ID", "Value"], values, list(range(2, len(values) + 2)),
                             [(0, "ID", "text", ""), (1, "Value", "number", "")], {})
    project = table([[1, 10], ["1", 20], [None, 30],
                     ["repeat", 40], ["repeat", 50], ["repeat", 60], ["repeat", 70]])
    incoming = table([["1", 22], [1, 11], [None, 33], ["repeat", 44], ["new", 1], ["new", 2]])
    mapping = suggest_mapping(project, incoming)
    plan = preview_import(project, incoming, mapping, [incoming.columns[0].id])
    assert [item["status"] for item in plan.rows] == ["matched", "matched", "new", "ambiguous", "input_repeat", "input_repeat"]
    assert plan.rows[0]["target_id"] == project.rows[1].id
    assert plan.rows[1]["target_id"] == project.rows[0].id
    assert plan.rows[3]["candidates"] == [row.id for row in project.rows[3:]]
