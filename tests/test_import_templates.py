"""Range provenance, explicit decimal policy and portable versioned import templates."""
from copy import deepcopy
import json

import pytest
from openpyxl import Workbook

from autopyro.core import Column, Project
from autopyro.io import read_file, build_project, ImportValidationError
from autopyro.import_templates import create, store, reference, resolve_sources, resolve_append, load_template, save_template


def reader_template(view=None, **changes):
    settings = {"sheet": None, "delimiter": ";", "encoding": "utf-8-sig", "header_rows": 1,
                "header_row": 1, "data_end_row": None, "first_column": 1, "last_column": None, "decimal_separator": ","}
    columns = [{"header": "Lab no", "aliases": ["SampleID"], "selected": True, "name": "Sample", "kind": "text", "unit": ""},
               {"header": "S2raw", "aliases": ["S2 alias"], "selected": True, "name": "S2", "kind": "number", "unit": "mg HC/g rock"}]
    return create("Laboratory", "reader", view or {}, file_type="delimited", settings=settings, columns=columns, **changes)


def project(values, sha="source"):
    return build_project(["Lab no", "S2raw"], values, list(range(2, len(values) + 2)),
                         [(0, "Sample", "text", ""), (1, "S2", "number", "mg HC/g rock")], {"sha256": sha})


def append_template(target, view=None):
    return create("Append laboratory", "append", view or {}, columns=[
        {"header": "Lab no", "aliases": ["SampleID"], "target": {"id": target.columns[0].id, "name": "Sample", "kind": "text", "unit": "", "basis": ""}},
        {"header": "S2raw", "aliases": ["S2 alias"], "target": {"id": target.columns[1].id, "name": "S2", "kind": "number", "unit": "mg HC/g rock", "basis": ""}}],
        keys=[0], bindings={}, defaults={"method": "Pyrol6", "laboratory": "Lab A"}, working_policy="keep")


def test_csv_range_retains_physical_rows_and_column_coordinates(tmp_path):
    path = tmp_path / "lab.csv"
    path.write_text("Laboratory export\nIgnore;Lab no;S2raw;Notes\nx;001;1,2;ok\nx;002;<0,1;ok\nfooter\n", encoding="utf-8")
    settings = {"header_row": 2, "data_end_row": 4, "first_column": 2, "last_column": 3}
    headers, rows, numbers = read_file(path, **settings)
    assert headers == ["Lab no", "S2raw"] and numbers == [3, 4]
    result = build_project(headers, rows, numbers, [(0, "Sample", "text", ""), (1, "S2", "number", "mg HC/g rock")],
                           settings, decimal_separator=",")
    assert [c.metadata["source_index"] for c in result.columns] == [1, 2]
    assert result.rows[0].values[result.columns[1].id] == 1.2
    assert result.rows[0].original[result.columns[1].id] == "1,2"
    assert result.rows[1].values[result.columns[1].id] == "<0,1"
    assert result.rows[0].source["row"] == 3


def test_quoted_multiline_csv_tracks_starting_physical_line(tmp_path):
    path = tmp_path / "multiline.csv"
    path.write_text('Sample;Notes\n001;"first\nsecond"\n002;end\n', encoding="utf-8", newline="")
    _, rows, numbers = read_file(path, data_end_row=3)
    assert numbers == [2, 4] and rows[0][1] == "first\nsecond"


def test_excel_offset_and_group_header_outside_selected_columns(tmp_path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Lab"
    sheet.append(["Export title"])
    sheet.append(["Sample", "Pyrol6", None, None])
    sheet.append(["ID", "TOC", "S2", "Tmax"])
    sheet.append(["001", 2, 6, 440])
    sheet.append(["002", 3, 9, 445])
    path = tmp_path / "lab.xlsx"
    workbook.save(path)
    headers, rows, numbers = read_file(path, sheet="Lab", header_row=2, header_rows=2,
                                      first_column=3, last_column=4, data_end_row=4)
    assert headers == ["Pyrol6 / S2", "Pyrol6 / Tmax"]
    assert rows == [["6", "440"]] and numbers == [4]


@pytest.mark.parametrize("settings", [{"header_row": 0}, {"header_row": 3, "data_end_row": 2},
    {"first_column": 3, "last_column": 2}, {"first_column": 99}, {"header_rows": 3}])
def test_bad_ranges_fail_before_building_project(tmp_path, settings):
    path = tmp_path / "lab.csv"
    path.write_text("A;B\n1;2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_file(path, **settings)


def test_numeric_errors_are_collected_and_original_cells_remain_literal():
    rows = [["001", "1,2"], ["002", "bad"], ["003", "2,3"]]
    original = deepcopy(rows)
    with pytest.raises(ImportValidationError) as error:
        build_project(["Sample", "S2"], rows, [5, 6, 7], [(0, "Sample", "text", ""), (1, "S2", "number", "mg/g")], {}, decimal_separator=".")
    assert [item["row"] for item in error.value.errors] == [5, 6, 7]
    assert rows == original


def test_templates_reorder_by_headers_and_explicit_aliases_only():
    template = reader_template()
    assert resolve_sources(template, ["S2 alias", "Unrequested", "SampleID"]) == {0: 2, 1: 0}
    for headers in (["Lab no", "S2raw", "S2 alias"], ["Lab no", "S2raw", "S2raw"], ["Wrong ID", "S2raw"]):
        with pytest.raises(ValueError):
            resolve_sources(template, headers)


def test_revisions_are_immutable_and_survive_active_revision_rollback(tmp_path):
    view = {}
    first = reader_template()
    store(view, first)
    second = reader_template(view)
    second["columns"][1]["unit"] = "ug HC/g rock"
    store(view, second)
    store(view, first)  # Explicitly select an earlier edition.
    third = reader_template(view)
    assert third["revision"] == 3 and third["id"] == first["id"]
    altered = deepcopy(first)
    altered["columns"][1]["unit"] = "incorrect"
    before = deepcopy(view)
    with pytest.raises(ValueError, match="редакция"):
        store(view, altered)
    assert view == before
    path = tmp_path / "template.json"
    save_template(path, second)
    assert load_template(path) == second
    assert "source" not in json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("change", [{"password": "never accepted"}, {"version": 999}, {"stage": "python"}])
def test_template_json_rejects_unknown_fields_versions_and_executable_stage(tmp_path, change):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({**reader_template(), **change}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_template(path)


@pytest.mark.parametrize("field", ["settings", "columns", "id"])
def test_reader_template_requires_complete_definition(tmp_path, field):
    template = reader_template()
    template.pop(field)
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(template), encoding="utf-8")
    with pytest.raises(ValueError):
        load_template(path)


@pytest.mark.parametrize("field", ["header_row", "first_column"])
def test_template_rejects_null_start_coordinates(tmp_path, field):
    template = reader_template()
    template["settings"][field] = None
    path = tmp_path / "null.json"
    path.write_text(json.dumps(template), encoding="utf-8")
    with pytest.raises(ValueError):
        load_template(path)


def test_append_template_is_portable_but_rejects_changed_semantics():
    target = project([["001", "6"]])
    template = append_template(target)
    other = project([["001", "7"]], "other")
    incoming = project([["001", "8"]], "incoming")
    incoming.columns.reverse()
    resolved = resolve_append(template, other, incoming)
    assert resolved["mapping"][incoming.columns[0].id] == other.columns[1].id
    assert resolved["keys"] == [incoming.columns[1].id]
    target.columns[1].unit = "ug HC/g rock"
    target.columns.append(Column("S2", "number", unit="mg HC/g rock"))
    with pytest.raises(ValueError, match="изменилось"):
        resolve_append(template, target, incoming)
    incoming.columns[0].unit = "unknown"
    with pytest.raises(ValueError, match="единица"):
        resolve_append(template, other, incoming)


def test_reader_gui_applies_template_imports_with_provenance_and_archives(tmp_path, qtbot):
    from autopyro.dialogs import ImportDialog
    path = tmp_path / "new-lab.csv"
    path.write_text("S2 alias;SampleID;Unrequested\n1,2;001;extra\n", encoding="utf-8")
    template = reader_template()
    dialog = ImportDialog(path=path)
    qtbot.addWidget(dialog)
    dialog.apply_template(template)
    qtbot.waitUntil(lambda: dialog.active_template is not None and dialog.worker is not None and not dialog.worker.isRunning(), timeout=10000)
    assert not dialog.mapping.cellWidget(2, 0).isChecked()
    assert dialog.mapping.item(0, 4).text() == "mg HC/g rock"
    dialog.import_data()
    qtbot.waitUntil(lambda: dialog.project is not None, timeout=10000)
    result = dialog.project
    assert [c.name for c in result.columns] == ["S2", "Sample"]
    assert result.rows[0].values[result.columns[0].id] == 1.2
    assert result.rows[0].original[result.columns[0].id] == "1,2"
    assert result.rows[0].values[result.columns[1].id] == "001"
    assert result.source["import_template"] == reference(template)
    assert result.view["import_template_revisions"][template["id"]]["1"] == template
    saved = tmp_path / "project.apyro"
    result.save(saved)
    assert Project.load(saved).view["import_template_revisions"] == result.view["import_template_revisions"]


def test_ambiguous_reader_template_never_partially_assigns_units(tmp_path, qtbot):
    from autopyro.dialogs import ImportDialog
    path = tmp_path / "ambiguous.csv"
    path.write_text("Lab no;S2raw;S2 alias\n001;6;8\n", encoding="utf-8")
    dialog = ImportDialog(path=path)
    qtbot.addWidget(dialog)
    dialog.apply_template(reader_template())
    qtbot.waitUntil(lambda: dialog.worker is not None and not dialog.worker.isRunning() and dialog.raw is not None, timeout=10000)
    assert dialog.active_template is None
    assert "неоднозначные" in dialog.summary.text()
    assert all(dialog.mapping.item(i, 4).text() == "" for i in range(3))
    assert not dialog.buttons.button(dialog.buttons.StandardButton.Ok).isEnabled()


def test_reader_errors_tab_lists_all_failed_cells(tmp_path, qtbot):
    from autopyro.dialogs import ImportDialog
    path = tmp_path / "bad-values.csv"
    path.write_text("Lab no;S2raw\n001;wrong\n002;bad\n", encoding="utf-8")
    dialog = ImportDialog(path=path)
    qtbot.addWidget(dialog)
    dialog.apply_template(reader_template())
    qtbot.waitUntil(lambda: dialog.active_template is not None and not dialog.worker.isRunning(), timeout=10000)
    dialog.import_data()
    qtbot.waitUntil(lambda: len(dialog.import_errors) == 2, timeout=10000)
    assert dialog.project is None and dialog.tabs.currentWidget() is dialog.errors
    assert "Строка 2" in dialog.errors.toPlainText() and "Строка 3" in dialog.errors.toPlainText()


def test_reader_typed_columns_roundtrip_and_edit(tmp_path, qtbot):
    from autopyro.ui.data_io.import_file import ImportDialog
    from autopyro.ui.table.filter_rules import validated_rules
    from autopyro.ui.table.model import TableModel

    path = tmp_path / "typed.csv"
    path.write_text("Date;Time;Timestamp;Accepted\n31.12.2025;09:15;2025-12-31T12:15:00+03:00;0\n"
                    "2026-01-01;09:16:05;2026-01-01T09:16:05Z;\n", encoding="utf-8")
    dialog = ImportDialog(path=path)
    qtbot.addWidget(dialog)
    dialog.load()
    qtbot.waitUntil(lambda: dialog.raw is not None, timeout=10000)
    assert [dialog.mapping.cellWidget(i, 3).currentData() for i in range(4)] == [
        "date", "time", "datetime", "number"
    ]
    choices = dialog.mapping.cellWidget(3, 3)
    choices.setCurrentIndex(choices.findData("boolean"))
    dialog.save_template("Typed data")
    assert dialog.active_template["columns"][3]["kind"] == "boolean"
    dialog.import_data()
    qtbot.waitUntil(lambda: dialog.project is not None, timeout=10000)

    saved = tmp_path / "typed.apyro"
    dialog.project.save(saved)
    project = Project.load(saved)
    ids = [column.id for column in project.columns]
    assert [project.rows[0].values[cid] for cid in ids[:3]] == [
        "2025-12-31", "09:15:00", "2025-12-31T09:15:00+00:00"
    ]
    assert project.rows[0].values[ids[3]] is False
    assert project.rows[1].values[ids[3]] is None
    assert project.rows[0].original[ids[0]] == "31.12.2025"
    assert project.view["import_template_revisions"]
    assert validated_rules(project, [{"column": ids[3], "operator": "eq", "value": "нет"}])[0]["value"] is False

    model = TableModel(project)
    assert not model.setData(model.index(0, 0), "31.02.2026")
    assert model.setData(model.index(0, 3), "да")
    assert project.rows[0].values[ids[3]] is True
    model.undo.undo()
    assert project.rows[0].values[ids[3]] is False


def test_excel_dates_and_times_and_invalid_typed_cells(tmp_path):
    from datetime import date, datetime, time
    from autopyro.data_io.io import infer_kind

    path = tmp_path / "typed.xlsx"
    workbook = Workbook()
    workbook.active.append(["Date", "Time", "Timestamp", "Accepted"])
    workbook.active.append([date(2026, 1, 1), time(14, 30), datetime(2026, 1, 1, 14, 30), True])
    workbook.save(path)
    raw = read_file(path)
    assert infer_kind([raw[1][0][3]]) == "boolean"
    selections = [(i, name, kind, "") for i, (name, kind) in enumerate(
        zip(raw[0], ("date", "time", "datetime", "boolean"), strict=True)
    )]
    result = build_project(*raw, selections, {})
    assert list(result.rows[0].values.values()) == ["2026-01-01", "14:30:00", "2026-01-01T14:30:00", True]
    with pytest.raises(ImportValidationError) as error:
        build_project(raw[0], [["2026-02-29", "24:00", "2026-01-01", "2"]], [42], selections, {})
    assert [(cell["row"], cell["column"]) for cell in error.value.errors] == [(42, name) for name in raw[0]]


def test_additional_types_import_edit_filter_and_reopen(tmp_path, qtbot):
    from PySide6.QtWidgets import QStyleOptionViewItem
    from autopyro.ui.data_io.import_file import ImportDialog
    from autopyro.ui.table.batch_edit import parse_cell_change
    from autopyro.ui.table.filter_rules import validated_rules
    from autopyro.ui.table.model import TableModel
    from autopyro.ui.table.proxy import FilterProxy
    from autopyro.ui.table.view import DataTable

    path = tmp_path / "additional-types.csv"
    path.write_text("Elapsed;Lithology;Sample;Repeats\n27:30:00;ND;00123;2\n"
                    "09:00:00;Sand;00002;9007199254740993\n100:00:00;;;\n", encoding="utf-8")
    dialog = ImportDialog(path=path)
    qtbot.addWidget(dialog)
    dialog.load()
    qtbot.waitUntil(lambda: dialog.raw is not None, timeout=10000)
    kinds = ("duration", "category", "identifier", "integer")
    for i, kind in enumerate(kinds):
        combo = dialog.mapping.cellWidget(i, 3)
        assert combo.findData(kind) >= 0
        combo.setCurrentIndex(combo.findData(kind))
    dialog.save_template("Additional types")
    dialog.import_data()
    qtbot.waitUntil(lambda: dialog.project is not None, timeout=10000)
    model = TableModel(dialog.project)
    project = model.project
    ids = [column.id for column in project.columns]
    assert list(project.rows[0].values.values()) == [99000, "ND", "00123", 2]
    assert project.science.current(project.rows[0].id, ids[1]).qualifier == "exact"
    assert model.index(0, 0).data() == "27:30:00"
    assert project.rows[1].values[ids[3]] == 9007199254740993
    assert not model.setData(model.index(0, 3), "2.5")
    assert not model.setData(model.index(0, 0), "27:60:00")
    assert model.setData(model.index(0, 0), "28:00:00")
    assert parse_cell_change(project, model.index(0, 0), "29:00:00")[2] == 104400
    model.undo.undo()
    assert project.rows[0].values[ids[0]] == 99000
    model.undo.redo()

    proxy = FilterProxy()
    proxy.setSourceModel(model)
    rules = validated_rules(project, [{"column": ids[0], "operator": "gt", "value": "24:00:00"}])
    proxy.set_rules(rules)
    assert proxy.rowCount() == 2
    proxy.set_rules(validated_rules(project, [{"column": ids[2], "operator": "eq", "value": "00123"}]))
    assert proxy.rowCount() == 1
    view = DataTable()
    qtbot.addWidget(view)
    view.setModel(proxy)
    index = proxy.index(0, 1)
    editor = view.itemDelegate().createEditor(view, QStyleOptionViewItem(), index)
    assert [editor.itemText(i) for i in range(editor.count())] == ["", "ND", "Sand"]
    view.itemDelegate().setEditorData(editor, index)
    editor.setEditText("Clay")
    view.itemDelegate().setModelData(editor, proxy, index)
    assert project.rows[0].values[ids[1]] == "Clay"
    proxy.set_rules(validated_rules(project, [{"column": ids[1], "operator": "eq", "value": "Clay"}]))
    assert proxy.rowCount() == 1
    proxy.set_rules(validated_rules(project, [{"column": ids[3], "operator": "ge", "value": "9007199254740993"}]))
    assert proxy.rowCount() == 1
    project.view["named"] = {"Large counts": {"rules": proxy.rules, "combine": "AND"}}
    saved = tmp_path / "additional-types.apyro"
    project.save(saved)
    reopened = Project.load(saved)
    assert [column.kind for column in reopened.columns] == list(kinds)
    assert [row.values for row in reopened.rows] == [row.values for row in project.rows]
    assert reopened.rows[0].original[ids[0]] == "27:30:00"
    assert reopened.view["named"] == project.view["named"]
    assert reopened.view["import_template_revisions"] == project.view["import_template_revisions"]


def test_elapsed_excel_duration_and_integer_precision(tmp_path):
    from datetime import timedelta
    from autopyro.model.field_types import format_duration, parse_typed_value

    path = tmp_path / "elapsed.xlsx"
    workbook = Workbook()
    workbook.active.append(["Elapsed", "Sample", "Repeats"])
    workbook.active.append([timedelta(hours=27, minutes=30), "00123", 2])
    workbook.active["A2"].number_format = "[h]:mm:ss"
    workbook.save(path)
    project = build_project(*read_file(path), [(0, "Elapsed", "duration", ""),
                            (1, "Sample", "identifier", ""), (2, "Repeats", "integer", "")], {})
    assert list(project.rows[0].values.values()) == [99000, "00123", 2]
    assert parse_typed_value("9007199254740993", "integer") == 9007199254740993
    assert parse_typed_value("2,0", "integer") == 2
    assert format_duration(parse_typed_value("-100:30:00.125", "duration")) == "-100:30:00.125"
    for kind in ("integer", "duration"):
        for value in ("NaN", "inf", True):
            with pytest.raises(ValueError):
                parse_typed_value(value, kind)


def test_reader_saves_ranges_and_manual_edit_clears_template_provenance(tmp_path, qtbot):
    from autopyro.dialogs import ImportDialog
    path = tmp_path / "offset.csv"
    path.write_text("Export\nSkip;Lab no;S2raw\nx;001;1,2\nFooter\n", encoding="utf-8")
    dialog = ImportDialog(path=path)
    qtbot.addWidget(dialog)
    dialog.header_row.setValue(2)
    dialog.data_end.setValue(3)
    dialog.first_column.setValue(2)
    dialog.last_column.setValue(3)
    dialog.load()
    qtbot.waitUntil(lambda: dialog.raw is not None and not dialog.worker.isRunning(), timeout=10000)
    dialog.mapping.item(1, 4).setText("mg HC/g rock")
    dialog.save_template("Offset export")
    template = deepcopy(dialog.active_template)
    assert template["settings"]["header_row"] == 2
    assert template["settings"]["last_column"] == 3
    assert not dialog.project  # Saving a draft does not publish a project.
    dialog.mapping.item(1, 4).setText("ug HC/g rock")
    assert dialog.active_template is None
    dialog.import_data()
    qtbot.waitUntil(lambda: dialog.project is not None, timeout=10000)
    assert "import_template" not in dialog.project.source
    assert "reader:Offset export" in dialog.project.view["import_templates"]
    assert dialog.project.rows[0].source["row"] == 3


def test_append_template_and_library_changes_are_undoable(qtbot, tmp_path):
    from autopyro.window import MainWindow
    from autopyro.import_dialog import AppendImportDialog
    from autopyro.template_dialog import TemplateLibraryDialog
    target = project([["001", "6"]])
    incoming = project([["001", "8"]], "new")
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(target)
    dialog = AppendImportDialog(window, incoming)
    qtbot.addWidget(dialog)
    template = append_template(target)
    template["defaults"].update(instrument="RE6", fraction="whole rock", date="2026-10-25")
    dialog.apply_template(template)
    assert dialog.mapping.cellWidget(0, 2).isChecked()
    assert dialog.method.text() == "Pyrol6"
    assert dialog.defaults["instrument"].text() == "RE6"
    dialog.calculate()
    qtbot.waitUntil(lambda: dialog.plan is not None and not dialog.worker.isRunning(), timeout=10000)
    dialog.apply_import()
    qtbot.waitUntil(lambda: dialog.result() == dialog.DialogCode.Accepted, timeout=10000)
    protocol = next(iter(target.view["import_runs"].values()))["protocol"]
    assert protocol["import_template"] == reference(template)
    assert any(a.instrument == "RE6" and a.fraction == "whole rock" and a.date == "2026-10-25"
               for a in target.science.analyses.values())
    assert "append:Append laboratory" in target.view["import_templates"]
    window.model.undo.undo()
    assert target.view["import_templates"] == {}
    assert template["id"] in target.view["import_template_revisions"]
    window.model.undo.redo()
    library = TemplateLibraryDialog(window)
    qtbot.addWidget(library)
    library.table.selectRow(0)
    library.remove()
    library.apply_library()
    assert target.view["import_templates"] == {}
    window.model.undo.undo()
    assert "append:Append laboratory" in target.view["import_templates"]
    window.dirty = False
