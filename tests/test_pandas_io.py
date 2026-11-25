"""Literal cell values and CSV compatibility across the pandas boundary."""

import csv
from datetime import datetime

import pandas as pd
import pytest
from openpyxl import Workbook
from openpyxl.styles import Font

from autopyro.io import ImportValidationError, build_project, excel_sheet_names, export_frame_csv, export_table_csv, read_file
from autopyro.database.tools.import_lithostratigraphy import HEADERS, read_rows


def test_excel_preserves_literals_empty_columns_and_source_rows(tmp_path):
    book = Workbook()
    sheet = book.active
    sheet.title = "Data"
    sheet.append(["ID", "ID", "Result", "Date", "Flag", None])
    sheet.append(["001", "NA", "=1+1", datetime(2026, 1, 2, 3, 4), True, None])
    sheet.append([None] * 6)
    sheet.append(["002", "NULL", "#DIV/0!", None, False, None])
    # Styled empty cells still establish the source's column bounds.
    sheet["F1"].font = Font(bold=True)
    book.create_sheet("Other")
    path = tmp_path / "literal.xlsx"
    book.save(path)
    before = path.read_bytes()

    assert excel_sheet_names(path) == ["Data", "Other"]
    headers, rows, numbers = read_file(path)
    assert headers == ["ID", "ID", "Result", "Date", "Flag", "Колонка 6"]
    assert rows == [
        ["001", "NA", "=1+1", "2026-01-02T03:04:00", "True", None],
        ["002", "NULL", "#DIV/0!", None, "False", None],
    ]
    assert numbers == [2, 4]
    assert path.read_bytes() == before


def test_excel_range_can_extend_into_empty_columns(tmp_path):
    book = Workbook()
    book.active.append(["A"])
    book.active.append([1])
    path = tmp_path / "range.xlsx"
    book.save(path)
    assert read_file(path, last_column=3) == (
        ["A", "Колонка 2", "Колонка 3"], [["1", None, None]], [2],
    )
    with pytest.raises(ValueError, match="заголовка"):
        read_file(path, header_row=3)


def test_csv_export_preserves_integer_precision_and_protects_text(tmp_path):
    path = tmp_path / "export.csv"
    export_table_csv(path, ["N", "Text", "Bool"], [
        [9007199254740993, "001", True],
        [None, " =SUM(A1)", False],
        [-2, 'quoted;"\nvalue', None],
    ])
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        assert list(csv.reader(stream, delimiter=";")) == [
            ["N", "Text", "Bool"], ["9007199254740993", "001", "True"],
            ["", "' =SUM(A1)", "False"], ["-2", 'quoted;"\nvalue', ""],
        ]
    export_table_csv(path, ["N", "Text"], [])
    assert path.read_text(encoding="utf-8-sig") == "N;Text\n"


def test_csv_export_protects_formula_like_headers(tmp_path):
    path = tmp_path / "headers.csv"
    export_table_csv(path, ["=1+1", "Value"], [["@literal", -2]])
    with path.open(encoding="utf-8-sig", newline="") as stream:
        assert list(csv.reader(stream, delimiter=";")) == [["'=1+1", "Value"], ["'@literal", "-2"]]


def test_csv_export_with_progress_preserves_bytes_across_batches(tmp_path):
    regular, batched = tmp_path / "regular.csv", tmp_path / "batched.csv"
    headers = ["=Header", "N", "Text"]
    rows = [["001", 9007199254740993, 'a;"\nb'], ["@text", None, "-"]] * 501
    updates = []
    export_table_csv(regular, headers, rows)
    export_table_csv(batched, headers, rows, progress=lambda count, total: updates.append((count, total)))
    assert batched.read_bytes() == regular.read_bytes()
    assert (1000, 1002) in updates
    assert updates[-1] == (1002, 1002)


@pytest.mark.parametrize("project_export", [False, True])
@pytest.mark.parametrize("known_total", [False, True])
def test_csv_writes_batches_before_consuming_remaining_rows(tmp_path, monkeypatch, project_export, known_total):
    from autopyro.data_io import io
    from autopyro.model.core import Column, Project, Row

    column = Column("=Value")
    project = Project(columns=[column])
    values = ["001", "a\r\nb", None, 9007199254740993, "@text"] * 401
    consumed = 0
    batches = []
    write_frame = io.export_frame_csv

    def rows():
        nonlocal consumed
        for value in values:
            consumed += 1
            yield Row(values={column.id: value})

    def write_batch(path, frame, **kwargs):
        batches.append((consumed, len(frame)))
        write_frame(path, frame, **kwargs)

    expected, actual = tmp_path / "expected.csv", tmp_path / "actual.csv"
    write_frame(expected, pd.DataFrame([[value] for value in values], columns=[column.name], dtype=object))
    monkeypatch.setattr(io, "export_frame_csv", write_batch)
    updates = []
    total = len(values) if known_total else None

    def progress(done, total):
        updates.append((done, total, consumed))

    if project_export:
        io.export_project_rows_csv(actual, project, rows(), [column.id], row_count=total, progress=progress)
    else:
        io.export_table_csv(actual, [column.name], ([row.values[column.id]] for row in rows()),
                            row_count=total, progress=progress)
    assert updates == [(0, total, 0), (1000, total, 1000), (2000, total, 2000), (2005, 2005, 2005)]
    assert batches == [(1000, 1000), (2000, 1000), (2005, 5)]
    assert actual.read_bytes() == expected.read_bytes()


def test_csv_export_reuses_frame_with_duplicate_headers_and_literal_values(tmp_path):
    frame = pd.concat([
        pd.Series([9007199254740993, -2, None], dtype=object),
        pd.Series([" =SUM(A1)", "001", None], dtype="string"),
        pd.Series(["a\nb", "@formula", "NA"], dtype=object),
    ], axis=1)
    frame.columns = ["Value", "Value", "Text"]
    original = frame.copy(deep=True)
    prepared, records = tmp_path / "frame.csv", tmp_path / "records.csv"
    export_frame_csv(prepared, frame)
    export_table_csv(records, list(frame.columns), frame.itertuples(index=False, name=None))
    assert prepared.read_bytes() == records.read_bytes()
    pd.testing.assert_frame_equal(frame, original)


def test_large_import_preserves_literal_values_and_error_order():
    rows = [["001", "1,2", "<10"], ["002", "9007199254740993", "ND"]] * 150
    numbers = range(3, 603, 2)
    selections = [(0, "ID", "text", ""), (1, "A", "number", ""), (2, "B", "number", "")]
    source = {"first_column": 3, "path": "sample.csv"}
    project = build_project(["ID", "A", "B"], rows, numbers, selections, source)
    keys = [column.id for column in project.columns]
    assert [project.rows[0].values[key] for key in keys] == ["001", 1.2, "<10"]
    assert [project.rows[1].values[key] for key in keys] == ["002", 9007199254740993, "ND"]
    assert list(project.rows[1].original.values()) == rows[1]
    assert project.rows[1].source == {"row": 5, **source}

    invalid = [["001", "1,2", "3,4"], ["002", "1.2", "3,4"]] * 150
    with pytest.raises(ImportValidationError) as cached:
        build_project(["ID", "A", "B"], invalid, numbers, selections, source, decimal_separator=".")
    with pytest.raises(ImportValidationError) as scalar:
        build_project(["ID", "A", "B"], iter(invalid), numbers, selections, source, decimal_separator=".")
    assert cached.value.errors == scalar.value.errors
    assert [(error["row"], error["column"]) for error in cached.value.errors[:4]] == [
        (3, "A"), (3, "B"), (5, "B"), (7, "A"),
    ]


def test_lithostratigraphy_frames_preserve_sheet_and_row_provenance(tmp_path):
    book = Workbook()
    sheet = book.active
    sheet.title = "First"
    sheet.append(HEADERS)
    sheet.append(["001", 0, 10.5, "NA", "FORMATION"])
    sheet.append([None] * 5)
    sheet.append(["002", 10.5, 20, "Unit", "MEMBER"])
    other = book.create_sheet("Second")
    other.append(HEADERS)
    other.append(["003", 20, 30, "Unit", "GROUP"])
    path = tmp_path / "litho.xlsx"
    book.save(path)
    digest, rows = read_rows(path)
    assert len(digest) == 64
    assert [(r[0], r[1], r[2]) for r in rows] == [
        ("First", 2, "001"), ("First", 4, "002"), ("Second", 2, "003"),
    ]
    assert rows[0][7] == dict(zip(HEADERS, ["001", 0, 10.5, "NA", "FORMATION"]))


@pytest.mark.parametrize("column,value,reason", [
    (1, "=1+1", "Formula/error"), (2, "#DIV/0!", "Formula/error"),
    (2, "1", "Invalid depth"), (2, True, "Invalid depth"),
    (3, -1, "Nonpositive/reversed"), (4, " ", "Invalid name/unit/level"),
    (5, "INVALID", "Unsupported level"),
])
def test_lithostratigraphy_rejects_invalid_cells(tmp_path, column, value, reason):
    book = Workbook()
    book.active.append(HEADERS)
    book.active.append(["Well", 0, 10, "Unit", "FORMATION"])
    book.active.cell(2, column).value = value
    path = tmp_path / "invalid.xlsx"
    book.save(path)
    with pytest.raises(ValueError, match=reason):
        read_rows(path)


def test_exported_selection_can_be_imported(tmp_path):
    from autopyro.data_io.io import export_project_rows_xlsx

    project = build_project(["ID"], [["001"], ["002"]], [2, 3], [(0, "ID", "text", "")], {})
    project.sync_measurements()
    path = tmp_path / "export.xlsx"
    export_project_rows_xlsx(path, project, project.rows, [project.columns[0].id])
    headers, rows, lines = read_file(path, sheet="Samples")
    assert [row[-1] for row in rows] == ["001", "002"]
    assert lines == [2, 3]


def test_excel_mask_preserves_visible_sample_key(tmp_path):
    book = Workbook()
    book.active.append(["Sample"])
    book.active.append([7])
    book.active["A2"].number_format = "00000"
    book.active.append(["7"])
    path = tmp_path / "keys.xlsx"
    book.save(path)
    assert read_file(path)[1] == [["00007"], ["7"]]
