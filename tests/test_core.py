import sqlite3

import pytest
from openpyxl import Workbook

from autopyro.core import Project, parse_number
from autopyro.interpolation import propose
from autopyro.io import build_project, export_csv, infer_kind, read_file, read_sqlite


@pytest.fixture
def project():
    headers = ["Well", "Depth", "S2"]
    return build_project(headers, [["A", "10", "2"], ["A", "20", None], ["A", "30", "6"],
                                   ["B", "20", None], ["A", "40", None]], list(range(2, 7)),
                         [(0, "Well", "text", ""), (1, "Depth", "number", "m"),
                          (2, "S2", "number", "mg HC/g rock")], {"file": "synthetic"})


def test_atomic_round_trip_and_original(project, tmp_path):
    path = tmp_path / "test.apyro"
    project.save(path)
    restored = Project.load(path)
    assert restored == project
    restored.rows[0].values[restored.columns[2].id] = float("nan")
    with pytest.raises(ValueError):
        restored.save(path)
    assert Project.load(path) == project
    assert not list(tmp_path.glob("*.tmp"))


def test_numeric_validation_and_ids():
    assert parse_number("9007199254740993") == 9007199254740993
    assert parse_number("1,2") == 1.2
    assert parse_number("  ") is None
    assert infer_kind(["001", "002"]) == "text"
    assert infer_kind(["0.1", "<0.01"]) == "text"
    for value in ("NaN", "inf", "<0.01", "ND"):
        with pytest.raises(ValueError):
            parse_number(value)


def test_csv_excel_preserve_qualifiers_and_headers(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("Well;S2\n001;<0.01\n002;ND\n", encoding="utf-8-sig")
    headers, rows, numbers = read_file(csv)
    assert rows == [["001", "<0.01"], ["002", "ND"]]
    assert numbers == [2, 3]
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Pyrol6", None])
    sheet.append(["TOC", "S2"])
    sheet.append([1.5, "<0.1"])
    path = tmp_path / "data.xlsx"
    workbook.save(path)
    headers, rows, numbers = read_file(path, header_rows=2)
    assert headers == ["Pyrol6 / TOC", "Pyrol6 / S2"]
    assert rows == [["1.5", "<0.1"]]
    assert numbers == [3]


@pytest.mark.parametrize("method", ["linear", "nearest", "pchip"])
def test_interpolation_scoped_and_no_extrapolation(project, method):
    well, depth, s2 = [c.id for c in project.columns]
    result = propose(project, depth, s2, [well], set(project.row_map()), method, 30)
    assert len(result) == 1
    assert result[0].row_id == project.rows[1].id
    assert result[0].value == (2 if method == "nearest" else 4)
    assert project.rows[1].values[s2] is None
    assert propose(project, depth, s2, [well], set(project.row_map()), method, 10) == []
    assert propose(project, depth, s2, [well], {project.rows[1].id}, method, 30) == []


def test_interpolation_rejects_duplicate_coordinates(project):
    well, depth, s2 = [c.id for c in project.columns]
    project.rows[2].values[depth] = 10
    with pytest.raises(ValueError, match="Повторные"):
        propose(project, depth, s2, [well], set(project.row_map()))


def test_sqlite_read_only_and_identifiers(tmp_path):
    path = tmp_path / "source.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE "strange table" (name TEXT, value REAL)')
        connection.execute('INSERT INTO "strange table" VALUES (?, ?)', ("<LOD", 1.5))
    before = path.read_bytes()
    headers, rows, _ = read_sqlite(path, "strange table")
    assert headers == ["name", "value"]
    assert rows == [["<LOD", "1.5"]]
    assert path.read_bytes() == before
    with pytest.raises(ValueError):
        read_sqlite(path, 'strange table"; DROP TABLE source;--')
    with pytest.raises(ValueError, match="лимит"):
        read_sqlite(path, "strange table", limit=0)


def test_export_uses_requested_order(project, tmp_path):
    path = tmp_path / "export.csv"
    export_csv(path, project, [project.rows[2].id, project.rows[0].id], [project.columns[2].id])
    assert path.read_text(encoding="utf-8-sig").splitlines() == ["S2 [mg HC/g rock]", "6", "2"]


def test_invalid_project_version(project, tmp_path):
    path = tmp_path / "bad.apyro"
    project.save(path)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=999")
    with pytest.raises(ValueError):
        Project.load(path)
