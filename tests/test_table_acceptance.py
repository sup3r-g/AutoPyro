"""Table workflows keep scientific identities through Qt views and persistence."""
import csv
import json
from copy import deepcopy

import pytest
from openpyxl import load_workbook
from PySide6.QtCore import QPoint, QItemSelection, QItemSelectionModel, Qt
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox, QInputDialog, QLineEdit, QStyleOptionViewItem, QWidget
from shiboken6 import delete

from autopyro.calculations import make_calculated_column
from autopyro.core import Column, Project, Row
from autopyro.io import build_project
from autopyro.measurements import SourceRecord
from autopyro.ui.dialogs.batch_edit import BatchEditPreview
from autopyro.ui.dialogs.measurement import DerivedCellDialog
from autopyro.ui.dialogs.sample_sets import SampleSetComparisonDialog
from autopyro.ui.table import AddColumnCommand, TableModel
from autopyro.ui.table.commands import EditCommand, QCCommand, SelectMeasurementCommand
from autopyro.ui.window import MainWindow
from autopyro.ui.i18n import tr


@pytest.fixture
def table_window(qtbot):
    headers = ["Well", "Sample", "Depth", "TOC", "S2"]
    records = [["A", "001", "30", "2", "6"],
               ["B", "001", "20", "9", "90"],
               ["A", "001", "10", "2", "2"],
               ["A", "001", "20", "2", ""],
               ["A", "001", "25", "2", "ND"]]
    project = build_project(headers, records, list(range(2, 7)),
                            [(i, name, "text" if i < 2 else "number", unit)
                             for i, (name, unit) in enumerate(zip(
                                 headers, ["", "", "m", "%wt", "mg HC/g rock"], strict=True))],
                            {"type": "synthetic acceptance"})
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.may_replace = lambda: True
    yield window
    window.dirty = False
    window.close()


def test_sample_set_roundtrip_and_analysis_scope(table_window, monkeypatch, tmp_path):
    window = table_window
    project = window.model.project
    table = window.table
    selection = table.selectionModel()
    select = QItemSelectionModel.SelectionFlag.Select
    window.proxy.sort(2, Qt.SortOrder.AscendingOrder)
    expected = [window.row_ids()[i] for i in (0, 3)]
    for i in (0, 3):
        selection.select(window.proxy.index(i, 1), select)
    monkeypatch.setattr(QInputDialog, "getText", lambda *args: ("Review", True))
    monkeypatch.setattr(QInputDialog, "getItem", lambda *args, **kwargs: ("Review", True))
    window.save_sample_set()
    assert project.view["sample_sets"]["Review"] == expected
    window.open_sample_set()
    assert window.row_ids() == expected
    assert window.analysis_scope.rows() == expected
    window.path = tmp_path / "sets.apyro"
    assert window.save()
    window.set_project(Project.load(window.path), window.path)
    assert window.row_ids() == expected
    window.close_sample_set()
    assert len(window.row_ids()) == len(project.rows)
    assert window.analysis_scope.rows("set:Review") == expected

    messages = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args: messages.append(args[2]))
    window.model.project.view["sample_sets"]["Review"].append("deleted-row")
    window.open_sample_set()
    assert window.row_ids() == expected
    assert messages == [tr('Проб в наборе, отсутствующих в проекте: {0}', 1)]


def test_clear_hidden_selection_from_table_and_edit_menu(table_window, qtbot):
    window = table_window
    project = window.model.project
    well = project.columns[0].id
    for clear in (lambda: qtbot.keyClick(window.table, Qt.Key.Key_Escape),
                  window.clear_selection_action.trigger, window.table.clearSelection):
        window.apply_filter([], 'AND')
        window.sample_selection.select([project.rows[1].id])
        window.apply_filter([{'column': well, 'operator': 'eq', 'value': 'A'}], 'AND')
        assert window.sample_selection.counts() == (0, 1)
        assert window.selection_status.text() == tr('Выбрано проб: {0:,} · скрыто: {1:,}', 1, 1)
        clear()
        assert not window.sample_selection.ids()
        assert window.sample_selection.export_snapshot()[0] == 0
        assert not window.clear_selection_action.isEnabled()
        window.apply_filter([], 'AND')
        assert not window.table.selectionModel().hasSelection()


def test_headers_switch_from_multicolumn_sort_to_single_sort(table_window, qtbot):
    window = table_window
    table, proxy = window.table, window.proxy
    columns = window.model.project.columns
    window.show()
    window.pin_column(2, True)
    qtbot.waitUntil(table.frozen.isVisible)
    proxy.set_sort_rules([(columns[2].id, True), (columns[0].id, False)])
    header = table.horizontalHeader()
    assert header.sortIndicatorSection() == 2
    assert header.sortIndicatorOrder() == Qt.SortOrder.DescendingOrder
    assert table.frozen.horizontalHeader().sortIndicatorOrder() == Qt.SortOrder.DescendingOrder

    def click(section, heading):
        qtbot.mouseClick(heading.viewport(), Qt.MouseButton.LeftButton,
                         pos=QPoint(heading.sectionViewportPosition(section) + 5, heading.height() // 2))

    click(3, header)
    assert not proxy.sort_rules
    assert header.sortIndicatorSection() == proxy.sortColumn() == 3
    assert proxy.index(0, 3).data() == '2'
    click(2, table.frozen.horizontalHeader())
    assert proxy.index(0, 2).data() == '10'
    click(2, table.frozen.horizontalHeader())
    assert proxy.index(0, 2).data() == '30'
    proxy.set_sort_rules([(columns[0].id, True), (columns[3].id, False)])
    table.sortByColumn(2, Qt.SortOrder.AscendingOrder)
    assert not proxy.sort_rules
    assert header.sortIndicatorSection() == 2
    assert proxy.index(0, 2).data() == '10'


def test_named_table_view_overwrite_requires_confirmation(table_window, monkeypatch):
    window = table_window
    panel = window.desktop.table_properties
    view = window.model.project.view
    panel.view_name.setText('Review')
    panel.save_view()
    saved = deepcopy(view['named']['Review'])
    window.apply_filter([{'column': window.model.project.columns[0].id, 'operator': 'eq', 'value': 'A'}], 'AND')
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.No)
    panel.save_view()
    assert view['named']['Review'] == saved
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.Yes)
    panel.save_view()
    assert view['named']['Review']['rules'] == window.proxy.rules
    assert len(window.analysis_scope.rows('view:Review')) == 4


def test_filter_draft_survives_column_removal_and_undo(table_window):
    window = table_window
    model = window.model
    panel = window.desktop.table_properties
    column = Column('Pending', id='temporary')
    model.undo.push(AddColumnCommand(model, column, {}))
    panel.refresh()
    panel.combine.setCurrentText('OR')
    panel.add_rule({'rules': [{'column': column.id, 'operator': 'eq', 'value': 'draft'}]})
    model.undo.undo()
    panel.refresh()
    rule, = panel.rules[0].rules
    assert panel.combine.currentText() == 'OR'
    assert rule.column.currentText() == tr('Колонка недоступна: {0}', 'Pending')
    assert rule.value.text() == 'draft'
    panel.apply_filter()
    assert panel.filter_error.text() == tr('Колонка недоступна: {0}', 'Pending')
    assert not window.proxy.rules
    model.undo.redo()
    panel.refresh()
    rule, = panel.rules[0].rules
    assert rule.column.currentData() == column.id
    assert rule.column.currentText() == 'Pending'
    assert rule.value.text() == 'draft'


def test_export_uses_selected_format_and_checks_suffix(table_window, monkeypatch, tmp_path):
    window = table_window
    window.table.selectRow(0)
    target = tmp_path / 'samples.csv'
    target.write_bytes(b'existing file')
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warnings.append(args[2]))
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(target), 'JSON (*.json)'))
    window.export_selected_samples()
    assert warnings == [tr('Расширение файла должно соответствовать выбранному формату ({0}).', '.json')]
    assert target.read_bytes() == b'existing file'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(target.with_suffix('')), 'JSON (*.json)'))
    window.export_selected_samples()
    with target.with_suffix('.json').open(encoding='utf-8') as exported:
        assert json.load(exported)['rows'][0]['id'] == window.model.project.rows[0].id


def test_cancelled_fill_does_not_resolve_source_rows_before_prompt(table_window, monkeypatch):
    window = table_window
    proxy = window.proxy
    window.table.selectionModel().select(
        QItemSelection(proxy.index(0, 0), proxy.index(4, 4)), QItemSelectionModel.SelectionFlag.Select,
    )
    original = proxy.mapToSource
    mapped = []

    def map_index(index):
        if index.isValid():
            mapped.append(index.row())
        return original(index)

    monkeypatch.setattr(QInputDialog, 'getText', lambda *args: ('', False))
    with monkeypatch.context() as patch:
        patch.setattr(proxy, 'mapToSource', map_index)
        window.fill_cells()
    assert mapped == []


def test_sample_set_restores_previous_filter_and_broken_view_fails_closed(table_window, monkeypatch, tmp_path):
    window = table_window
    project = window.model.project
    well = project.columns[0]
    rule = {"column": well.id, "operator": "eq", "value": "A"}
    window.apply_filter([rule], "AND")
    original = window.row_ids()
    project.view["sample_sets"] = {"B": [project.rows[1].id]}
    monkeypatch.setattr(QInputDialog, "getItem", lambda *args, **kwargs: ("B", True))
    window.open_sample_set()
    assert window.row_ids() == [project.rows[1].id]
    window.path = tmp_path / "filter-return.apyro"
    assert window.save()
    window.set_project(Project.load(window.path), window.path)
    window.close_sample_set()
    assert window.row_ids() == original
    window.proxy.sort(2, Qt.SortOrder.DescendingOrder)
    saved = window.capture_view()
    saved["sort_column"] = 0
    window.restore_view(saved)
    assert window.proxy.sortColumn() == 2
    window.restore_view({"rules": [{**rule, "column": "removed"}]})
    assert window.row_ids() == []
    window.apply_filter([], "AND")
    assert len(window.row_ids()) == len(project.rows)


def test_column_precision_only_changes_display_and_survives_project_roundtrip(
    table_window, monkeypatch, tmp_path,
):
    window = table_window
    model = window.model
    column = model.project.columns[4]
    assert model.setData(model.index(0, 4), "1.234567")
    assert model.setData(model.index(1, 4), "9007199254740993")
    window.pin_column(4, True)
    revision = model.data_revision
    rows = window.proxy.ordered_row_ids()
    monkeypatch.setattr(QInputDialog, "getInt", lambda *args: (2, True))
    window.choose_column_precision(4)

    def displayed(view, row):
        index = window.proxy.index(row, 4)
        option = QStyleOptionViewItem()
        view.itemDelegateForColumn(4).initStyleOption(option, index)
        return option.text

    assert displayed(window.table, 0) == displayed(window.table.frozen, 0) == "1.23"
    assert displayed(window.table, 1) == "9007199254740993.00"
    assert displayed(window.table, 4) == "ND"
    assert window.proxy.index(0, 4).data(Qt.ItemDataRole.EditRole) == "1.234567"
    assert model.data_revision == revision
    assert window.proxy.ordered_row_ids() is rows
    window.table.selectionModel().select(window.proxy.index(0, 4), QItemSelectionModel.SelectionFlag.Select)
    window.copy_cells()
    assert QApplication.clipboard().text() == "1.234567\n"

    view = window.capture_view()
    window.restore_view({})
    assert window.table.itemDelegateForColumn(4) is None
    window.restore_view(view)
    assert displayed(window.table, 0) == "1.23"
    window.path = tmp_path / "precision.apyro"
    assert window.save()
    window.set_project(Project.load(window.path), window.path)
    assert window.column_precision == {column.id: 2}
    assert displayed(window.table.frozen, 0) == "1.23"


def test_rename_and_delete_sample_set_retarget_live_and_saved_scopes(table_window, monkeypatch, tmp_path):
    window = table_window
    project = window.model.project
    row_ids = [project.rows[1].id, project.rows[2].id]
    view = project.view
    view["sample_sets"] = {"Old": row_ids, "Other": [project.rows[0].id]}
    view["plots"] = {"Set plot": {"scope": "set:Old"}, "View plot": {"scope": "view:Old"}}
    view["map_current"] = {"scope": "set:Old"}
    view["statistics"] = {"scope": "set:Old", "comparison_a": "Old"}
    window.analysis_scope.refresh()
    window.analysis_scope.set_mode("set:Old")
    monkeypatch.setattr(QInputDialog, "getItem", lambda *args, **kwargs: ("Old", True))
    window.open_sample_set()
    window.path = tmp_path / "set-management.apyro"
    assert window.save()

    membership = window.proxy.row_id_filter
    monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: (" New ", True))
    window.rename_sample_set()
    assert view["sample_sets"]["New"] is row_ids
    assert window.proxy.row_id_filter is membership
    assert window.active_sample_set == view["current"]["sample_set"] == "New"
    assert window.analysis_scope_combo.currentData() == "set:New"
    assert view["plots"]["Set plot"]["scope"] == view["map_current"]["scope"] == "set:New"
    assert window.save()
    window.set_project(Project.load(window.path), window.path)
    assert window.active_sample_set == "New"
    assert window.analysis_scope.rows() == row_ids

    monkeypatch.setattr(QInputDialog, "getItem", lambda *args, **kwargs: ("New", True))
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.No)
    window.delete_sample_set()
    assert window.active_sample_set == "New"
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    window.delete_sample_set()
    view = window.model.project.view
    assert list(view["sample_sets"]) == ["Other"]
    assert window.active_sample_set is None
    assert "sample_set" not in view["current"]
    assert window.analysis_scope_combo.currentData() == "filtered"
    assert view["plots"]["Set plot"]["scope"] == view["map_current"]["scope"] == "filtered"
    assert view["plots"]["View plot"]["scope"] == "view:Old"
    assert view["statistics"] == {"scope": "filtered", "comparison_a": "Old"}


def test_selected_samples_csv_preserves_table_order_and_deduplicates_cells(
    table_window, monkeypatch, tmp_path,
):
    window = table_window
    project = window.model.project
    well, sample, depth, toc, s2 = project.columns
    project.rows[2].values[sample.id] = "=1+1"
    window.apply_filter([{"column": well.id, "operator": "eq", "value": "A"}], "AND")
    window.proxy.sort(2, Qt.SortOrder.AscendingOrder)
    for i in (0, 3):
        window.columns.item(i).setCheckState(Qt.CheckState.Unchecked)
    header = window.table.horizontalHeader()
    header.moveSection(header.visualIndex(4), 0)
    selection = window.table.selectionModel()
    for row, column in ((3, 2), (0, 1), (0, 4)):
        selection.select(window.proxy.index(row, column), QItemSelectionModel.SelectionFlag.Select)
    path = tmp_path / "selection.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), "CSV (*.csv)"))
    monkeypatch.setattr(project, "row_map", lambda: pytest.fail("Selection export scanned the whole project"))

    window.export_selected_samples()

    with path.open(encoding="utf-8-sig", newline="") as stream:
        assert list(csv.reader(stream, delimiter=";")) == [
            ["S2 [mg HC/g rock]", "Sample", "Depth [m]"],
            ["2", "'=1+1", "10"],
            ["6", "001", "30"],
        ]


@pytest.mark.parametrize("selected", [False, True])
def test_failed_table_export_preserves_existing_file(table_window, monkeypatch, tmp_path, selected):
    window = table_window
    window.table.selectionModel().select(window.proxy.index(0, 0), QItemSelectionModel.SelectionFlag.Select)
    path = tmp_path / "selection.csv"
    path.write_bytes(b"original")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), "CSV (*.csv)"))
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))

    def failed_export(temporary, *args):
        temporary.write_bytes(b"partial")
        raise OSError("write failed")

    if selected:
        monkeypatch.setattr("autopyro.ui.data_io.table_export.export_project_rows_csv", failed_export)
        window.export_selected_samples()
    else:
        monkeypatch.setattr("autopyro.ui.data_io.table_export.export_project_rows_csv", failed_export)
        window.export()

    assert warnings == ["write failed"]
    assert path.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("selection,suffix", [(False, "csv"), (True, "csv"), (True, "json"), (True, "xlsx")])
@pytest.mark.parametrize("change", ["edit", "project"])
def test_export_rejects_source_change_before_replacing_file(table_window, monkeypatch, tmp_path, selection, suffix, change):
    from autopyro.ui.data_io import table_export as module

    window = table_window
    window.sample_selection.select([window.model.project.rows[0].id])
    assert window.sample_selection.export_snapshot()[0] > 0
    monkeypatch.setattr(QMessageBox, "information", lambda *args: pytest.fail(str(args[2])))
    path = tmp_path / f"selection.{suffix}"
    path.write_bytes(b"previous export")
    warnings = []
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), f"{suffix.upper()} (*.{suffix})"))
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    exporter_name = f"export_project_rows_{suffix}"
    exporter = getattr(module, exporter_name)

    def changed_export(*args, **kwargs):
        exporter(*args, **kwargs)
        if change == "edit":
            assert window.model.setData(window.model.index(0, 2), "42")
        else:
            window.set_project(Project(), prepared=True)

    monkeypatch.setattr(module, exporter_name, changed_export)
    (window.export_selected_samples if selection else window.export)()

    assert warnings == [tr('Таблица изменилась. Повторите операцию.')]
    assert path.read_bytes() == b"previous export"
    assert list(tmp_path.iterdir()) == [path]


def test_selected_samples_json_exports_working_measurements_and_provenance(
    table_window, monkeypatch, tmp_path,
):
    window = table_window
    project = window.model.project
    row = project.rows[0]
    s2 = project.columns[4]
    previous = project.science.current(row.id, s2.id)
    repeat_source = SourceRecord("repeat-source", {"path": "repeat.csv", "row": 9})
    project.science.sources[repeat_source.id] = repeat_source
    repeat = project.science.add_result(
        row.id, s2.id, "<0.25", analysis_fields={"source_id": repeat_source.id, "method": "Repeat"},
        details={"lod": 0.25},
    )
    project.select_measurement(row.id, s2.id, repeat.id)
    for i in range(4):
        window.columns.item(i).setCheckState(Qt.CheckState.Unchecked)
    window.table.selectionModel().select(window.proxy.index(0, 4), QItemSelectionModel.SelectionFlag.Select)
    path = tmp_path / "selection.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), "JSON (*.json)"))

    window.export_selected_samples()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert (payload["format"], payload["version"], payload["project_id"]) == (
        "autopyro.table_selection", 1, project.id,
    )
    assert [column["id"] for column in payload["columns"]] == [s2.id]
    assert payload["columns"][0]["unit"] == "mg HC/g rock"
    exported, = payload["rows"]
    assert exported["id"] == row.id
    assert exported["source"] == row.source
    assert exported["values"] == {s2.id: "<0.25"}
    measurement = exported["measurements"][s2.id]
    assert measurement["id"] == repeat.id
    assert measurement["id"] != previous.id
    assert measurement["qualifier"] == "below_limit"
    assert measurement["raw_value"] == "<0.25"
    assert measurement["limit"] == measurement["lod"] == 0.25
    assert measurement["method"] == "Repeat"
    assert measurement["source"] == repeat_source.provenance


def test_selected_samples_xlsx_preserves_literals_order_and_provenance(table_window, monkeypatch, tmp_path):
    window = table_window
    project = window.model.project
    _, sample, depth, _, s2 = project.columns
    assert window.model.setData(window.model.index(0, 1), "=1+1")
    assert window.model.setData(window.model.index(0, 2), "9007199254740993")
    window.proxy.sort(4, Qt.SortOrder.DescendingOrder)
    window.columns.item(0).setCheckState(Qt.CheckState.Unchecked)
    window.columns.item(3).setCheckState(Qt.CheckState.Unchecked)
    header = window.table.horizontalHeader()
    header.moveSection(header.visualIndex(4), 0)
    first, second = project.rows[:2]
    window.sample_selection.select([first.id, second.id])
    path = tmp_path / "selected"

    def choose_file(*args):
        assert "XLSX (*.xlsx)" in args[-1]
        window.table.selectionModel().clearSelection()
        return str(path), "XLSX (*.xlsx)"

    monkeypatch.setattr(QFileDialog, "getSaveFileName", choose_file)
    window.export_selected_samples()

    workbook = load_workbook(path.with_suffix(".xlsx"), read_only=True)
    try:
        assert workbook.properties.identifier == project.id
        header, first_exported, second_exported = workbook["Samples"].iter_rows()
        assert [cell.value for cell in header] == ["sample_id", "S2 [mg HC/g rock]", "Sample", "Depth [m]"]
        assert [cell.value for cell in first_exported] == [second.id, 90, "001", 20]
        assert [cell.value for cell in second_exported] == [first.id, 6, "=1+1", "9007199254740993"]
        assert second_exported[2].data_type == "s"
        assert [entry[0] for entry in list(workbook["Columns"].values)[1:]] == [s2.id, sample.id, depth.id]
        sources = dict(list(workbook["Sources"].values)[1:])
        assert json.loads(sources[first.id]) == first.source
        fields, *records = workbook["Measurements"].values
        measurement = next(dict(zip(fields, record)) for record in records if record[:2] == (first.id, s2.id))
        active = project.science.current(first.id, s2.id)
        assert measurement["id"] == active.id
        assert measurement["raw_value"] == active.raw_value
        assert json.loads(measurement["source_json"]) == project.science.sources[measurement["source_id"]].provenance
    finally:
        workbook.close()


def test_selection_export_aborts_if_proxy_order_changes_in_save_dialog(table_window, monkeypatch, tmp_path):
    window = table_window
    window.sample_selection.select([window.model.project.rows[0].id])
    assert window.sample_selection.export_snapshot()[0] > 0
    monkeypatch.setattr(QMessageBox, "information", lambda *args: pytest.fail(str(args[2])))
    path = tmp_path / "selection.csv"
    warnings = []

    def choose_file(*args):
        window.proxy.sort(2, Qt.SortOrder.AscendingOrder)
        return str(path), "CSV (*.csv)"

    monkeypatch.setattr(QFileDialog, "getSaveFileName", choose_file)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    window.export_selected_samples()
    assert warnings == [tr('Таблица изменилась. Повторите операцию.')]
    assert not path.exists()


def test_xlsx_export_preserves_existing_file_when_text_exceeds_cell_limit(table_window, monkeypatch, tmp_path):
    window = table_window
    project = window.model.project
    project.rows[0].values[project.columns[0].id] = "X" * 32_768
    window.sample_selection.select([project.rows[0].id])
    path = tmp_path / "selection.xlsx"
    path.write_bytes(b"original")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), "XLSX (*.xlsx)"))
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))

    window.export_selected_samples()

    assert warnings and "32,767" in warnings[0]
    assert path.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("cancel_at", [1, "packing", 100, None])
def test_large_xlsx_export_packs_and_cancels_safely(table_window, monkeypatch, tmp_path, cancel_at):
    from zipfile import ZipFile
    from autopyro.ui.tasks import TaskCancelled

    window = table_window
    row = window.model.project.rows[0]
    monkeypatch.setattr(window.sample_selection, "export_snapshot", lambda: (2400, iter([row] * 2400)))
    path = tmp_path / "selection.xlsx"
    path.write_bytes(b"previous workbook")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), "XLSX (*.xlsx)"))
    updates = []
    warnings = []
    packing_sheet = False
    sheet_checkpoints = 0
    archive_write = ZipFile.write

    def write_member(archive, filename, arcname=None, *args, **kwargs):
        nonlocal packing_sheet
        packing_sheet = bool(arcname and "worksheets/" in arcname)
        try:
            return archive_write(archive, filename, arcname, *args, **kwargs)
        finally:
            packing_sheet = False

    monkeypatch.setattr(ZipFile, "write", write_member)

    def run(parent, title, write, *, cancellable, prepare):
        assert cancellable

        def update(stage, percent):
            nonlocal sheet_checkpoints
            updates.append(percent)
            if packing_sheet:
                sheet_checkpoints += 1
            packing_cancelled = cancel_at == "packing" and sheet_checkpoints == 5
            progress_cancelled = isinstance(cancel_at, int) and percent >= cancel_at
            if packing_cancelled or progress_cancelled:
                raise TaskCancelled()

        assert next(prepare) is None
        assert next(prepare) is None
        with pytest.raises(StopIteration) as finished:
            next(prepare)
        write(update, finished.value.value)

    monkeypatch.setattr("autopyro.ui.data_io.table_export.run_with_progress", run)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    window.export_selected_samples()

    assert updates[0] == 0
    assert not warnings
    if cancel_at is None:
        assert updates[-1] == 100
        with path.open("rb") as stream:
            workbook = load_workbook(stream, read_only=True)
            try:
                assert sum(1 for _ in workbook["Samples"].values) == 2401
            finally:
                workbook.close()
    else:
        if cancel_at == "packing":
            assert updates[-1] == 99
        else:
            assert updates[-1] >= cancel_at
        assert path.read_bytes() == b"previous workbook"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("selected", [False, True])
def test_cancel_csv_during_write_preserves_previous_file(table_window, monkeypatch, tmp_path, selected):
    from autopyro.ui.tasks import TaskCancelled

    window = table_window
    row = window.model.project.rows[0]
    monkeypatch.setattr(window.sample_selection, "export_snapshot", lambda: (2400, iter([row] * 2400)))
    resolved = []

    def visible_rows():
        for index in range(10_000):
            resolved.append(index)
            yield row

    monkeypatch.setattr(window.proxy, "visible_row_snapshot", lambda: (10_000, visible_rows()))
    path = tmp_path / "selection.csv"
    path.write_bytes(b"previous CSV")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), "CSV (*.csv)"))
    warnings = []
    updates = []

    def run(parent, title, write, *, cancellable, prepare=None):
        assert cancellable

        def update(stage, percent):
            updates.append(percent)
            if 0 < percent < 99:  # The first CSV batch has already been written.
                raise TaskCancelled()

        if selected:
            assert next(prepare) is None
            assert next(prepare) is None
            with pytest.raises(StopIteration) as finished:
                next(prepare)
            write(update, finished.value.value)
        else:
            assert not resolved
            assert next(prepare) is None
            assert len(resolved) == 1024
            with pytest.raises(StopIteration) as finished:
                while True:
                    next(prepare)
            write(update, finished.value.value)

    monkeypatch.setattr("autopyro.ui.data_io.table_export.run_with_progress", run)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    (window.export_selected_samples if selected else window.export)()

    assert updates[-1] == (41 if selected else 9)
    assert not warnings
    assert path.read_bytes() == b"previous CSV"
    assert list(tmp_path.iterdir()) == [path]


def test_sample_set_comparison_shows_groups_without_changing_main_table(table_window, monkeypatch):
    window = table_window
    project = window.model.project
    a, b, c = (row.id for row in project.rows[:3])
    project.view["sample_sets"] = {"A": [a, b, "missing"], "B": [b, c]}
    window.apply_filter([{"column": project.columns[0].id, "operator": "eq", "value": "A"}], "AND")
    original_rows = window.row_ids()
    window.dirty = False

    def inspect(dialog):
        assert dialog.proxy.ordered_row_ids() == (b,)
        assert dialog.group.itemText(1) == tr('Только в первом наборе ({0:,})', 2)
        dialog.group.setCurrentIndex(1)
        assert dialog.proxy.ordered_row_ids() == (a,)
        assert dialog.status.text() == tr('Проб в таблице: {0:,}. Отсутствуют в проекте: {1:,}.', 1, 1)
        dialog.group.setCurrentIndex(2)
        assert dialog.proxy.ordered_row_ids() == (c,)
        dialog.second.setCurrentText("A")
        assert dialog.proxy.ordered_row_ids() == ()
        dialog.group.setCurrentIndex(0)
        assert dialog.proxy.ordered_row_ids() == (a, b)

    monkeypatch.setattr(SampleSetComparisonDialog, "exec", inspect)
    window.compare_sample_sets()
    assert window.row_ids() == original_rows
    assert not window.dirty


def test_nested_filter_editor_and_named_view_roundtrip(table_window, tmp_path):
    window = table_window
    project = window.model.project
    well, _, depth, _, s2 = project.columns
    editor = window.desktop.table_properties
    editor.refresh()
    editor.add_rule({"column": well.id, "operator": "eq", "value": "A"})
    editor.add_group_button.click()
    editor.apply_filter()
    assert editor.filter_error.text() == tr('Добавьте хотя бы одно условие в каждую группу.')
    group = editor.rules[-1]
    group.combine.setCurrentText("OR")
    group.add_rule({"column": depth.id, "operator": "le", "value": "10"})
    group.add_rule({"combine": "AND", "negate": True, "rules": [
        {"column": s2.id, "operator": "not_empty"},
        {"column": depth.id, "operator": "ge", "value": "20"},
    ]})
    editor.apply_filter()
    expected = [project.rows[i].id for i in (2, 3)]
    assert window.row_ids() == expected
    assert not editor.filter_error.text()
    window.analysis_scope.create_view("Grouped")
    window.path = tmp_path / "grouped.apyro"
    assert window.save()
    window.set_project(Project.load(window.path), window.path)
    assert window.row_ids() == expected
    window.restore_view({})
    assert len(window.row_ids()) == 5
    assert window.analysis_scope.rows("view:Grouped") == expected


def test_unit_filter_follows_working_measurement_and_undo(table_window):
    window = table_window
    project = window.model.project
    column = project.columns[-1]
    row = project.rows[0]
    repeat = project.science.add_result(row.id, column.id, "6")
    repeat.unit = "mg/g"
    editor = window.desktop.table_properties
    editor.refresh()
    editor.add_rule({"column": column.id, "operator": "unit_eq", "value": "  mg/g  "})
    editor.apply_filter()
    assert not editor.filter_error.text()
    assert window.row_ids() == []
    window.model.undo.push(SelectMeasurementCommand(window.model, row.id, column.id, repeat.id))
    assert window.row_ids() == [row.id]
    window.model.undo.undo()
    assert window.row_ids() == []
    window.model.undo.redo()
    assert window.row_ids() == [row.id]
    window.apply_filter([{"column": column.id, "operator": "unit_missing"}], "AND")
    assert window.row_ids() == []
    repeat.unit = ""
    window.model.notify_changed(rows=[row])
    assert window.row_ids() == [row.id]


def test_derived_cell_card_follows_sorted_selection_and_scopes_provenance(table_window, monkeypatch):
    window = table_window
    project = window.model.project
    row = project.rows[2]
    row.source = {"file": "samples.csv", "row": 4}
    result = Column("Derived result", "number", unit="mg/g", metadata={
        "method": "linear", "stale": True, "max_gap": 12,
        "proposals": [
            {"row_id": project.rows[0].id, "left_id": "unrelated-left", "right_id": "unrelated-right"},
            {"row_id": row.id, "left_id": "left-support", "right_id": "right-support"},
        ],
        "diagnostics": {row.id: {"status": "ok"}, project.rows[0].id: {"reason": "unrelated"}},
        "interpolation_protocol": {"payload": "full protocol"},
    })
    project.columns.append(result)
    for item in project.rows:
        item.values[result.id] = 1.25 if item is row else None
    window.set_project(project, prepared=True)
    window.proxy.sort(2, Qt.SortOrder.AscendingOrder)
    window.table.setCurrentIndex(window.proxy.index(0, 5))
    cards = []

    def inspect(dialog):
        cards.append(dialog)
        text = dialog.details.toPlainText()
        assert dialog.details.isReadOnly()
        assert row.id in text
        assert "1.25" in text and "mg/g" in text
        assert tr('Результат устарел') in text
        assert "left-support" in text and "right-support" in text
        assert '"status": "ok"' in text and "samples.csv" in text
        assert "unrelated" not in text and "full protocol" not in text

    monkeypatch.setattr(DerivedCellDialog, "exec", inspect)
    window.show_measurements()
    assert len(cards) == 1


def test_interpolation_support_navigation_resolves_sorted_hidden_cell(table_window, monkeypatch):
    window = table_window
    project = window.model.project
    support, target = project.rows[0], project.rows[2]
    source_column = project.columns[4]
    result = Column("Interpolated", "number", metadata={
        "method": "linear", "y_id": source_column.id,
        "proposals": [{"row_id": target.id, "left_id": support.id, "right_id": support.id}],
    })
    project.columns.append(result)
    for row in project.rows:
        row.values[result.id] = 2 if row is target else None
    window.set_project(project, prepared=True)
    window.proxy.sort(2, Qt.SortOrder.AscendingOrder)
    window.columns.item(4).setCheckState(Qt.CheckState.Unchecked)
    project.view["sample_sets"] = {"Target": [target.id]}
    window.active_sample_set = "Target"
    window.proxy.set_rules([], row_ids=[target.id])
    window.table.setCurrentIndex(window.proxy.index(0, 5))
    answers = iter((QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes))
    monkeypatch.setattr(QMessageBox, "question", lambda *args: next(answers))

    def inspect(dialog):
        from PySide6.QtWidgets import QDialogButtonBox

        box = dialog.findChild(QDialogButtonBox)
        button, = [b for b in box.buttons() if box.buttonRole(b) == QDialogButtonBox.ButtonRole.ActionRole]
        button.click()
        assert window.active_sample_set == "Target"
        assert window.table.isColumnHidden(4)
        button.click()
        assert window.active_sample_set is None
        assert not window.table.isColumnHidden(4)

    monkeypatch.setattr(DerivedCellDialog, "exec", inspect)
    window.show_measurements()
    source = window.proxy.mapToSource(window.table.currentIndex())
    assert project.rows[source.row()].id == support.id
    assert project.columns[source.column()].id == source_column.id
    assert window.proxy.sortColumn() == 2


def test_stale_filter_updates_all_values_when_column_status_changes(table_window):
    window = table_window
    project = window.model.project
    output = Column("Fit", "number", metadata={"curve_fit": {"run_id": "test"}, "stale": False})
    project.columns.append(output)
    for i, row in enumerate(project.rows):
        row.values[output.id] = i if i < 3 else None
    window.set_project(project, prepared=True)
    current = {"column": output.id, "operator": "up_to_date"}
    stale = {"column": output.id, "operator": "stale"}
    window.apply_filter([current], "AND")
    expected = [row.id for row in project.rows[:3]]
    assert window.row_ids() == expected
    project.view["named"] = {"Stale": {"rules": [stale]}}
    assert window.analysis_scope.rows("view:Stale") == []
    output.metadata["stale"] = True
    window.model.notify_changed(rows=[project.rows[0]])
    assert window.row_ids() == []
    assert window.analysis_scope.rows("view:Stale") == expected
    window.apply_filter([stale], "AND")
    assert window.row_ids() == expected


def test_measurement_filters_follow_edits_and_qc_undo(table_window):
    window = table_window
    project = window.model.project
    s2 = project.columns[-1]
    row = project.rows[-1]
    window.apply_filter([{"column": s2.id, "operator": "non_detect"}], "AND")
    assert window.row_ids() == [row.id]
    window.apply_filter([{"column": s2.id, "operator": "below_limit"}], "AND")
    assert window.row_ids() == []
    assert window.model.setData(window.model.index(0, 4), "<0.1")
    assert window.row_ids() == [project.rows[0].id]
    window.apply_filter([{"column": s2.id, "operator": "qc_excluded"}], "AND")
    measurement = project.science.current(row.id, s2.id)
    window.model.undo.push(QCCommand(window.model, ("qc_exclusions", "plots"),
                                    {measurement.id: {"revision": measurement.revision}}))
    assert window.row_ids() == [row.id]
    window.model.undo.undo()
    assert window.row_ids() == []


@pytest.mark.parametrize("change", ["edit", "reset", "project"])
def test_batch_preview_invalidates_when_source_changes(table_window, qtbot, change):
    window = table_window
    model = window.model
    row = model.project.rows[0]
    column = model.project.columns[2]
    command = EditCommand(model, [(row.id, column.id, 99)], row_positions={row.id: 0})
    dialog = BatchEditPreview("Preview", "One change", command, window)
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog.model.rowCount() == 1

    if change == "edit":
        assert model.setData(model.index(0, 2), "41")
    elif change == "reset":
        model.beginResetModel()
        assert not dialog.apply_button.isEnabled()
        model.endResetModel()
    else:
        window.set_project(Project(), prepared=True)
        dialog.accept()

    assert not dialog.apply_button.isEnabled()
    assert dialog.summary.text() == tr('Таблица изменилась. Повторите операцию.')
    assert dialog.model.rowCount() == 0
    assert not dialog.model.rows
    assert not dialog.model.positions
    assert len(command.changes) == 1
    dialog.accept()
    assert dialog.isVisible()
    assert row.values[column.id] == (41 if change == "edit" else 30)
    dialog.reject()


@pytest.mark.parametrize("operation", ["edit", "paste", "fill"])
def test_joint_fit_outputs_are_read_only_and_do_not_become_measurements(table_window, monkeypatch, operation):
    window = table_window
    project = window.model.project
    column = Column("Joint prediction", "number", metadata={
        "joint_curve_fit": {"output": "prediction", "run_id": "test-run"}})
    project.columns.append(column)
    for row in project.rows:
        row.values[column.id] = 3.5
    window.set_project(project)
    index = window.proxy.index(0, len(project.columns) - 1)
    window.table.setCurrentIndex(index)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    monkeypatch.setattr(BatchEditPreview, "exec", lambda *_: pytest.fail("Unexpected edit preview"))
    assert not index.flags() & Qt.ItemFlag.ItemIsEditable
    if operation == "edit":
        assert not window.proxy.setData(index, "42")
    elif operation == "paste":
        QApplication.clipboard().setText("42")
        window.paste_cells()
    else:
        monkeypatch.setattr(QInputDialog, "getText", lambda *args: ("42", True))
        window.fill_cells()
    if operation != "edit":
        assert warnings and tr('Производные результаты доступны только для чтения') in warnings[0]
    assert project.rows[0].values[column.id] == 3.5
    assert project.science.current(project.rows[0].id, column.id) is None
    assert window.model.undo.count() == 0


def test_batch_preview_can_cancel_without_writing(table_window, monkeypatch):
    window = table_window
    window.table.setCurrentIndex(window.proxy.index(0, 2))
    QApplication.clipboard().setText("11\t12\n21\t22\n31\t32\n20\t2")
    monkeypatch.setattr(window.model, "resolve_row_positions",
                        lambda *_: pytest.fail("Preview already has source positions"))

    def cancel(dialog):
        assert dialog.summary.text() == tr('Будет изменено ячеек: {0}. Строк: {1}.', 6, 3)
        model = dialog.table.model()
        assert model.rowCount() == 6
        assert [model.index(0, column).data() for column in range(5)] == ["001", "1", "Depth", "30", "11"]
        assert [model.index(5, column).data() for column in range(5)] == ["001", "3", "TOC", "2", "32"]
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(BatchEditPreview, "exec", cancel)
    window.paste_cells()
    assert window.model.undo.count() == 0
    assert window.proxy.index(0, 2).data() == "30"


def test_batch_preview_finds_project_row_and_shows_full_value(table_window, qtbot):
    model = table_window.model
    project = model.project
    column = project.columns[1]
    first, second = (row.id for row in project.rows[:2])
    long_value = 'Новое значение ' * 100
    command = EditCommand(model, [(first, column.id, '001', '002'), (second, column.id, '001', long_value)],
                          row_positions={first: 0, second: 1}, validated=True)
    dialog = BatchEditPreview('Preview', 'Summary', command, table_window)
    qtbot.addWidget(dialog)
    dialog.search.setText('2')
    dialog.search_button.click()
    assert dialog.table.currentIndex().row() == 1
    dialog.table.setCurrentIndex(dialog.model.index(1, 4))
    assert dialog.cell_value.toPlainText() == long_value
    dialog.search.setText('does not exist')
    dialog.search_button.click()
    assert dialog.search_status.text() == tr('Совпадений нет.')
    dialog.invalidate_preview()
    assert not dialog.search_button.isEnabled()
    assert not dialog.cell_value.toPlainText()


def test_batch_preview_handles_destruction_with_its_source(table_window, qtbot):
    parent = QWidget()
    project = table_window.model.project
    model = TableModel(project, parent, prepared=True)
    row, column = project.rows[0], project.columns[2]
    command = EditCommand(model, [(row.id, column.id, 99)], row_positions={row.id: 0})
    dialog = BatchEditPreview('Preview', 'One change', command, parent)
    dialog.show()
    with qtbot.captureExceptions() as errors:
        delete(parent)
    assert not errors


def test_bulk_fill_keeps_selected_source_cells_when_sort_changes(table_window, monkeypatch):
    window = table_window
    project = window.model.project
    depth, toc = project.columns[2:4]
    selection = window.table.selectionModel()
    selection.select(window.proxy.index(0, 2), QItemSelectionModel.SelectionFlag.Select)
    selection.select(window.proxy.index(1, 3), QItemSelectionModel.SelectionFlag.Select)

    def enter_value(*args):
        assert args[2] == tr('Значение для {0} ячеек (пустое — очистить)', 2)
        window.proxy.sort(2, Qt.SortOrder.AscendingOrder)
        return "42", True

    monkeypatch.setattr(QInputDialog, "getText", enter_value)
    monkeypatch.setattr(BatchEditPreview, "exec", lambda self: QDialog.DialogCode.Accepted)
    window.fill_cells()
    assert [row.values[depth.id] for row in project.rows] == [42, 20, 10, 20, 25]
    assert [row.values[toc.id] for row in project.rows] == [2, 42, 2, 2, 2]
    window.model.undo.undo()
    assert project.rows[0].values[depth.id] == 30
    assert project.rows[1].values[toc.id] == 9


def test_pinned_columns_stay_visible_when_scrolling_and_restore(table_window, qtbot):
    window = table_window
    table = window.table
    window.show()
    table.setRowHeight(0, 70)
    for i in range(window.model.columnCount()):
        table.setColumnWidth(i, 450)
    window.pin_column(2, True)
    window.pin_column(0, True)
    qtbot.waitUntil(lambda: table.frozen.isVisible() and table.horizontalScrollBar().maximum() > 0)
    scrollbar = table.horizontalScrollBar()
    frozen_geometry = table.frozen.geometry()
    scrollbar.setValue(scrollbar.maximum())
    assert table.frozen.geometry() == frozen_geometry
    assert table.frozen.columnViewportPosition(2) == 0
    assert table.frozen.columnViewportPosition(0) == 450
    assert table.frozen.selectionModel() is table.selectionModel()
    assert table.frozen.rowHeight(0) == 70
    table.frozen.setColumnWidth(0, 425)
    assert table.columnWidth(0) == 425
    table.setCurrentIndex(window.proxy.index(0, 2))
    qtbot.keyClick(table, Qt.Key.Key_F2)
    editor = table.frozen.findChild(QLineEdit)
    assert editor is not None
    qtbot.keyClick(editor, Qt.Key.Key_Escape)
    view = window.capture_view()
    window.pin_column(2, False)
    assert table.frozen.isColumnHidden(2)
    window.restore_view(view)
    assert table.pinned_columns == (2, 0)


@pytest.mark.parametrize("text,expected", [
    ("", ["001", "001", "001"]),
    ("\r\n", [None, "001", "001"]),
    ("new\r\n\r\nlast\r\n", ["new", None, "last"]),
])
def test_paste_blank_rows_clears_cells_and_undo_restores_them(table_window, monkeypatch, text, expected):
    window = table_window
    project = window.model.project
    sample = project.columns[1]
    before = [row.values[sample.id] for row in project.rows]
    window.table.setCurrentIndex(window.proxy.index(0, 1))
    QApplication.clipboard().setText(text)
    monkeypatch.setattr(BatchEditPreview, "exec", lambda self: QDialog.DialogCode.Accepted)

    window.paste_cells()

    assert [row.values[sample.id] for row in project.rows[:3]] == expected
    assert window.model.undo.count() == int(bool(text))
    if text:
        window.model.undo.undo()
    assert [row.values[sample.id] for row in project.rows] == before


def test_paste_rejects_unclosed_quoted_cell_before_preview(table_window, monkeypatch):
    window = table_window
    window.table.setCurrentIndex(window.proxy.index(0, 1))
    QApplication.clipboard().setText('valid\n"unclosed')
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    monkeypatch.setattr(BatchEditPreview, "exec", lambda *_: pytest.fail("Unexpected preview"))

    window.paste_cells()

    assert warnings == [tr('Не удалось прочитать табличные данные из буфера обмена.')]
    assert window.proxy.index(0, 1).data() == "001"
    assert window.model.undo.count() == 0


@pytest.mark.parametrize("interrupt", [None, "cancel", "sort"])
def test_large_clipboard_is_parsed_with_progress_before_preview(table_window, monkeypatch, interrupt):
    from autopyro.ui.tasks import TaskCancelled

    window = table_window
    column = Column("Sample")
    project = Project(columns=[column], rows=[Row(values={column.id: str(i)}) for i in range(256)])
    window.set_project(project, prepared=True)
    window.table.setCurrentIndex(window.proxy.index(0, 0))
    value = "X" * 256
    QApplication.clipboard().setText((value + "\n") * 256)
    updates, warnings, previews = [], [], []

    def run(parent, title, read, *, cancellable):
        assert cancellable

        def update(stage, percent):
            updates.append(percent)
            if interrupt == "cancel" and percent > 0:
                raise TaskCancelled()

        result = read(update)
        if interrupt == "sort":
            window.proxy.sort(0, Qt.SortOrder.DescendingOrder)
        return result

    def preview(title, targets, *, cell_count):
        previews.append((cell_count, [(index.row(), text) for index, text in targets]))

    monkeypatch.setattr("autopyro.ui.table.cell_actions.run_with_progress", run)
    monkeypatch.setattr(window, "preview_batch_edit", preview)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    window.paste_cells()

    assert updates == ([0, 49] if interrupt == "cancel" else [0, 49, 99, 100])
    assert previews == ([(256, [(i, value) for i in range(256)])] if interrupt is None else [])
    assert warnings == ([tr('Таблица изменилась. Повторите операцию.')] if interrupt == "sort" else [])
    assert project.rows[0].values[column.id] == "0"
    assert window.model.undo.count() == 0


@pytest.mark.parametrize("interrupt", [None, "cancel", "sort"])
def test_large_paste_validates_in_batches_before_preview(table_window, monkeypatch, interrupt):
    from autopyro.ui.tasks import TaskCancelled, run_with_progress

    window = table_window
    column = Column("Sample")
    project = Project(columns=[column], rows=[Row(values={column.id: str(i)}) for i in range(10_000)])
    window.set_project(project, prepared=True)
    window.table.setCurrentIndex(window.proxy.index(0, 0))
    QApplication.clipboard().setText("new\n" * 10_000)
    validated = []
    parsed_change = window.parsed_change
    warnings = []

    def parse(index, value):
        validated.append(index.row())
        return parsed_change(index, value)

    def run(parent, title, work, *, cancellable, prepare):
        assert cancellable and not validated
        assert next(prepare) is None
        assert len(validated) == 1024
        if interrupt is None:
            return run_with_progress(parent, title, work, cancellable=cancellable, prepare=prepare)
        if interrupt == "cancel":
            raise TaskCancelled()
        window.proxy.sort(0, Qt.SortOrder.DescendingOrder)
        next(prepare)

    monkeypatch.setattr(window, "parsed_change", parse)
    monkeypatch.setattr("autopyro.ui.table.cell_actions.run_with_progress", run)
    if interrupt is None:
        monkeypatch.setattr(BatchEditPreview, "exec", lambda *_: QDialog.DialogCode.Accepted)
    else:
        monkeypatch.setattr(BatchEditPreview, "exec", lambda *_: pytest.fail("Unexpected preview"))
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    window.paste_cells()

    if interrupt is None:
        assert not warnings
        assert len(validated) == 10_000
        assert all(row.values[column.id] == "new" for row in project.rows)
        assert window.model.undo.count() == 1
        window.model.undo.undo()
        assert all(row.values[column.id] == str(i) for i, row in enumerate(project.rows))
        return
    assert len(validated) == 1024
    assert warnings == ([tr('Таблица изменилась. Повторите операцию.')] if interrupt == "sort" else [])
    assert project.rows[0].values[column.id] == "0"
    assert window.model.undo.count() == 0


def test_qualified_paste_sorted_filtered_hidden_columns_and_atomic_undo(table_window, monkeypatch):
    window = table_window
    project = window.model.project
    well, sample, depth, toc, s2 = project.columns
    originals = deepcopy([(r.original, r.source) for r in project.rows])
    measurement_ids = deepcopy(project.science.working)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    monkeypatch.setattr(BatchEditPreview, "exec", lambda self: QDialog.DialogCode.Accepted)
    window.apply_filter([{"column": well.id, "operator": "eq", "value": "A"}], "AND")
    window.proxy.sort(2, Qt.SortOrder.AscendingOrder)
    window.columns.item(0).setCheckState(Qt.CheckState.Unchecked)
    window.columns.item(1).setCheckState(Qt.CheckState.Unchecked)
    window.columns.item(3).setCheckState(Qt.CheckState.Unchecked)
    # Put S2 before depth: paste follows visible header order, not source order.
    header = window.table.horizontalHeader()
    header.moveSection(header.visualIndex(4), 0)
    targets = window.row_ids()[:2]
    assert targets == [project.rows[2].id, project.rows[3].id]
    before = [deepcopy(row.values) for row in project.rows]
    window.table.setCurrentIndex(window.proxy.index(0, 4))
    QApplication.clipboard().setText("ND\t11\n<0,25\t21")
    count = window.model.undo.count()
    window.paste_cells()
    assert not warnings
    assert window.model.undo.count() == count + 1
    for rid, value, qualifier, new_depth in zip(
            targets, ["ND", "<0,25"], ["non_detect", "below_limit"], [11, 21], strict=True):
        row = project.row_map()[rid]
        result = project.science.current(rid, s2.id)
        assert row.values[s2.id] == value
        assert row.values[depth.id] == new_depth
        assert result.id == measurement_ids[rid][s2.id]
        assert result.qualifier == qualifier
        assert result.raw_value == value
    assert project.science.current(targets[1], s2.id).limit == .25
    assert project.rows[1].values == before[1]  # Hidden well never receives a paste.
    assert [(r.original, r.source) for r in project.rows] == originals
    window.model.undo.undo()
    assert [r.values for r in project.rows] == before
    assert project.science.current(targets[0], s2.id).value == 2
    window.model.undo.redo()
    assert project.science.current(targets[0], s2.id).qualifier == "non_detect"
    assert project.science.working == measurement_ids

    # Invalid second row must roll back the entire paste, including its first row.
    after = deepcopy([row.values for row in project.rows])
    count = window.model.undo.count()
    window.table.setCurrentIndex(window.proxy.index(0, 4))
    QApplication.clipboard().setText("8\t12\n7\tnot a number")
    window.paste_cells()
    assert len(warnings) == 1
    assert window.model.undo.count() == count
    assert [r.values for r in project.rows] == after


def test_interpolation_preview_marks_rows_with_a_populated_bottom(table_window, monkeypatch, tmp_path):
    window = table_window
    project = window.model.project
    _, sample, depth, _, s2 = project.columns
    project.rows[0].values[sample.id] = "ok"
    project.rows[1].values[sample.id] = None
    bottom = Column("Bottom", "number", unit="m")
    project.columns.append(bottom)
    for i, row in enumerate(project.rows):
        row.values[bottom.id] = 26 if i == 4 else None
    window.set_project(project)
    window.interpolate()
    dialog = window.interpolation_window
    dialog.x.setCurrentIndex(dialog.x.findData(depth.id))
    dialog.y.setCurrentIndex(dialog.y.findData(s2.id))
    dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
    assert dialog.interval_bottom.currentData() is None
    dialog.interval_bottom.setCurrentIndex(dialog.interval_bottom.findData(bottom.id))
    dialog.calculate()
    assert [(proposal.row_id, proposal.value) for proposal in dialog.proposals] == [(project.rows[3].id, 4)]
    excluded = ["001", "A", "25", tr('Исключена интервальная проба'), "", project.rows[4].id]
    assert excluded in dialog.diagnostic_model.rows
    all_rows = dialog.diagnostic_model.rows
    first = next(i for i, row in enumerate(all_rows) if row[-1] == project.rows[0].id)
    assert dialog.diagnostic_model.index(first, 0).data() == "ok"
    assert next(row[0] for row in all_rows if row[-1] == project.rows[1].id) == tr('Строка {0}', 2)
    dialog.only_excluded.setChecked(True)
    assert dialog.diagnostic_model.rows == [excluded]
    path = tmp_path / "diagnostics.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), "CSV (*.csv)"))
    summary = dialog.summary.text()
    announcements = []
    monkeypatch.setattr(dialog, "announce_summary", lambda text=None: announcements.append(text))
    dialog.export_diagnostics_button.click()
    assert dialog.export_status.text() == announcements[-1] == tr('CSV сохранён в файл «{0}». Строк: {1}.', path.name, 1)
    assert dialog.summary.text() == summary
    with path.open(encoding="utf-8-sig", newline="") as stream:
        assert list(csv.reader(stream, delimiter=";")) == [
            [tr('Проба'), tr('Ствол'), "X", tr('Статус'), tr('Причина'), tr('UUID пробы')], *dialog.diagnostic_model.rows,
        ]
    dialog.only_excluded.setChecked(False)
    assert dialog.diagnostic_model.rows is all_rows
    dialog.export_diagnostics_button.click()
    assert announcements[-1] == tr('CSV сохранён в файл «{0}». Строк: {1}.', path.name, len(all_rows))
    with path.open(encoding="utf-8-sig", newline="") as stream:
        assert list(csv.reader(stream, delimiter=";"))[1:] == all_rows
    dialog.only_excluded.setChecked(True)
    dialog.y.setCurrentIndex(dialog.y.findData(depth.id))
    dialog.calculate()  # Invalid X == Y leaves the last successful diagnostics visible.
    assert not dialog.export_diagnostics_button.isEnabled()
    dialog.only_excluded.setChecked(False)
    dialog.only_excluded.setChecked(True)
    assert not dialog.export_diagnostics_button.isEnabled()
    window.apply_filter([{"column": depth.id, "operator": "lt", "value": 25}], "AND")
    answers = iter((QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes))
    prompts = []

    def confirm_reveal(*args):
        prompts.append(args)
        return next(answers)

    monkeypatch.setattr(QMessageBox, "question", confirm_reveal)
    dialog.diagnostics.doubleClicked.emit(dialog.diagnostic_model.index(0, 1))
    assert not prompts
    assert dialog.diagnostic_hint.text() == tr('Предыдущие результаты доступны только для просмотра.')
    dialog.y.setCurrentIndex(dialog.y.findData(s2.id))
    dialog.scope.setCurrentIndex(dialog.scope.findData("all"))
    dialog.calculate()
    assert dialog.diagnostic_hint.text() == tr('Enter или двойной щелчок открывает исходную ячейку пробы.')
    index = dialog.diagnostic_model.index(0, 1)
    dialog.diagnostics.doubleClicked.emit(index)
    assert window.proxy.rules
    dialog.diagnostics.doubleClicked.emit(index)
    source = window.proxy.mapToSource(window.table.currentIndex())
    assert project.rows[source.row()].id == project.rows[4].id
    assert project.columns[source.column()].id == s2.id
    assert not window.proxy.rules


@pytest.mark.parametrize("existing", [False, True])
def test_cancel_interpolation_diagnostics_export_is_atomic(table_window, monkeypatch, tmp_path, existing):
    from threading import Event
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QFileDialog, QPushButton
    from autopyro.ui.dialogs import interpolation_diagnostics as interpolation
    from autopyro.ui.tasks import TaskDialog

    window = table_window
    window.interpolate()
    dialog = window.interpolation_window
    project = window.model.project
    dialog.x.setCurrentIndex(dialog.x.findData(project.columns[2].id))
    dialog.y.setCurrentIndex(dialog.y.findData(project.columns[4].id))
    dialog.calculate()
    rows = dialog.diagnostic_model.rows
    assert rows
    dialog.diagnostic_model.set_rows(rows * 500)
    target = tmp_path / "diagnostics.csv"
    summary = dialog.summary.text()
    announcements = []
    monkeypatch.setattr(dialog, "announce_summary", lambda text=None: announcements.append(text))
    if existing:
        target.write_bytes(b"previous export")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(target), "CSV (*.csv)"))

    reached, released = Event(), Event()
    export = interpolation.export_table_csv

    def pause_after_first_batch(*args, progress):
        def update(written, total):
            if written == 1000:
                reached.set()
                assert released.wait(5)
            progress(written, total)

        return export(*args, progress=update)

    monkeypatch.setattr(interpolation, "export_table_csv", pause_after_first_batch)
    timer = QTimer()

    def cancel_when_ready():
        if reached.is_set():
            dialog.findChild(TaskDialog).findChild(QPushButton).click()
            released.set()
            timer.stop()

    timer.timeout.connect(cancel_when_ready)
    timer.start(5)
    try:
        dialog.export_diagnostics()
    finally:
        timer.stop()
    assert reached.is_set() and released.is_set()
    assert dialog.export_status.text() == announcements[-1] == tr('Экспорт отменён.')
    assert dialog.summary.text() == summary
    assert list(tmp_path.iterdir()) == ([target] if existing else [])
    if existing:
        assert target.read_bytes() == b"previous export"


def test_copy_and_fill_exclude_hidden_cells_and_reject_sparse_copy(table_window, monkeypatch):
    window = table_window
    project = window.model.project
    window.columns.item(1).setCheckState(Qt.CheckState.Unchecked)
    window.table.selectRow(0)
    window.copy_cells()
    assert QApplication.clipboard().text() == "A\t30\t2\t6\n"
    monkeypatch.setattr(QInputDialog, "getText", lambda *args: ("8", True))
    monkeypatch.setattr(BatchEditPreview, "exec", lambda self: QDialog.DialogCode.Accepted)
    window.fill_cells()
    assert project.rows[0].values[project.columns[1].id] == "001"
    selection = window.table.selectionModel()
    selection.clearSelection()
    for column in (2, 4):
        selection.select(window.proxy.index(0, column), QItemSelectionModel.SelectionFlag.Select)
    notifications = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args: notifications.append(args[2]))
    window.copy_cells()
    assert notifications
    assert QApplication.clipboard().text() == "A\t30\t2\t6\n"


@pytest.mark.parametrize("action", ["complete", "cancel", "sort"])
def test_large_copy_prepares_in_batches_and_preserves_clipboard_on_abort(table_window, monkeypatch, action):
    from PySide6.QtCore import QItemSelection
    from autopyro.ui.tasks import TaskCancelled

    window = table_window
    column = Column("Sample")
    project = Project(columns=[column], rows=[Row(values={column.id: str(i)}) for i in range(10000)])
    window.set_project(project, prepared=True)
    selection = QItemSelection(window.proxy.index(0, 0), window.proxy.index(9999, 0))
    window.table.selectionModel().select(selection, QItemSelectionModel.SelectionFlag.ClearAndSelect)
    QApplication.clipboard().setText("previous clipboard")
    warnings = []

    def run(parent, title, work, *, cancellable, prepare):
        assert cancellable
        assert next(prepare) is None
        assert QApplication.clipboard().text() == "previous clipboard"
        if action == "cancel":
            raise TaskCancelled()
        if action == "sort":
            window.proxy.sort(0, Qt.SortOrder.DescendingOrder)
        with pytest.raises(StopIteration) as finished:
            while True:
                next(prepare)
        return work(lambda *_: None, finished.value.value)

    monkeypatch.setattr("autopyro.ui.table.cell_actions.run_with_progress", run)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    window.copy_cells()
    assert warnings == ([tr('Таблица изменилась. Повторите операцию.')] if action == "sort" else [])
    expected = "".join(f"{i}\n" for i in range(10000)) if action == "complete" else "previous clipboard"
    assert QApplication.clipboard().text() == expected
    assert window.model.undo.count() == 0


def test_copy_joins_ranges_in_visible_order_after_filter_and_sort(table_window):
    window = table_window
    well = window.model.project.columns[0]
    window.apply_filter([{"column": well.id, "operator": "eq", "value": "A"}], "AND")
    window.proxy.sort(2, Qt.SortOrder.AscendingOrder)
    window.table.hideColumn(1)
    header = window.table.horizontalHeader()
    header.moveSection(header.visualIndex(4), header.visualIndex(1))
    selection = window.table.selectionModel()
    for row in (0, 1):
        for column in (0, 4):
            selection.select(window.proxy.index(row, column), QItemSelectionModel.SelectionFlag.Select)

    window.copy_cells()

    assert QApplication.clipboard().text() == "A\t2\nA\t\n"


def test_interpolation_preview_explains_excluded_duplicate_coordinates(table_window, monkeypatch):
    window = table_window
    _, _, depth, _, s2 = window.model.project.columns
    assert window.model.setData(window.model.index(4, 2), "10")
    window.interpolate()
    dialog = window.interpolation_window
    dialog.x.setCurrentIndex(dialog.x.findData(depth.id))
    dialog.y.setCurrentIndex(dialog.y.findData(s2.id))
    dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
    dialog.calculate()
    assert tr('Повторные координаты внутри группы: уточните группировку или выборку') in dialog.summary.text()
    conflicts = dialog.diagnostic_model.rows
    assert len(conflicts) == 2
    assert {row[2] for row in conflicts} == {"10"}
    assert dialog.export_diagnostics_button.isEnabled()
    assert dialog.preview_tabs.currentWidget() is dialog.diagnostics.parentWidget()
    opened = []
    monkeypatch.setattr(window, "open_source_cell", lambda row_id, column_id, parent: opened.append((row_id, column_id)))
    dialog.open_diagnostic_sample(dialog.diagnostic_model.index(0, 0))
    assert opened == [(conflicts[0][-1], depth.id)]
    dialog.duplicates.setCurrentIndex(dialog.duplicates.findData("exclude"))
    dialog.calculate()
    assert dialog.proposals == []
    duplicates = [row for row in dialog.diagnostic_model.rows if row[3] == tr('Повторяющаяся координата X')]
    assert {row[4] for row in duplicates} == {tr('Числовая опора'), tr('Y не обнаружен (ND)')}
    assert tr('Из-за совпадений X исключено числовых опор: {0}.', 1) in dialog.summary.text()


def test_extrapolation_preview_marks_results_and_requires_linear_method(table_window, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog

    window = table_window
    project = window.model.project
    _, _, depth, _, s2 = project.columns
    assert window.model.setData(window.model.index(3, 2), "31")
    window.interpolate()
    dialog = window.interpolation_window
    dialog.x.setCurrentIndex(dialog.x.findData(depth.id))
    dialog.y.setCurrentIndex(dialog.y.findData(s2.id))
    dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
    dialog.calculate()
    assert dialog.proposals == []
    assert not dialog.extrapolation_limit.isEnabled()

    dialog.extrapolate.setChecked(True)
    dialog.extrapolation_limit.setValue(1)
    dialog.calculate()
    proposal, = dialog.proposals
    assert proposal.row_id == project.rows[3].id
    assert proposal.extrapolated and proposal.value == pytest.approx(6.2)
    assert tr('Экстраполяция') in dialog.preview_model.index(0, 3).data()
    dialog.extrapolation_limit.setValue(0.5)
    dialog.calculate()
    rejected = next(row for row in dialog.diagnostic_model.rows if row[-1] == proposal.row_id)
    assert rejected[3] == tr('Слишком далеко от крайней опоры')
    assert rejected[4] == tr('Фактически {0} при пределе {1} (в единицах X)', 1.0, 0.5)
    path = tmp_path / "limits.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), "CSV (*.csv)"))
    dialog.export_diagnostics()
    assert rejected[4] in path.read_text(encoding="utf-8-sig")

    dialog.extrapolation_limit.setValue(1)
    dialog.gap.setValue(10)
    dialog.calculate()
    rejected = next(row for row in dialog.diagnostic_model.rows if row[-1] == proposal.row_id)
    assert rejected[3] == tr('Слишком большой разрыв между крайними опорами')
    assert rejected[4] == tr('Фактически {0} при пределе {1} (в единицах X)', 20.0, 10.0)
    dialog.gap.setValue(100)
    dialog.min_anchors.setValue(3)
    dialog.calculate()
    assert dialog.proposals == []
    rejected = next(row for row in dialog.diagnostic_model.rows if row[-1] == proposal.row_id)
    assert rejected[4].startswith(tr('Найдено опор: {0}. Требуется: {1}.', 2, 3))
    assert tr('Метод пробы неизвестен. Опоры с известным методом считаются отдельно.') in rejected[4]
    dialog.method.setCurrentIndex(dialog.method.findData("spline"))
    assert dialog.min_anchors.minimum() == dialog.min_anchors.value() == 4
    assert not dialog.extrapolate.isEnabled()
    assert not dialog.extrapolate.isChecked()
    assert not dialog.extrapolation_limit.isEnabled()


def test_interpolation_preview_view_undo_and_sqlite_roundtrip(table_window, tmp_path, monkeypatch):
    window = table_window
    project = window.model.project
    well, sample, depth, toc, s2 = project.columns
    baseline = deepcopy(project.science)
    original_values = [deepcopy(row.values) for row in project.rows]
    window.apply_filter([{"column": well.id, "operator": "eq", "value": "A"}], "AND")
    window.proxy.sort(2, Qt.SortOrder.DescendingOrder)
    window.columns.item(1).setCheckState(Qt.CheckState.Unchecked)
    properties = window.desktop.table_properties
    properties.view_name.setText("Well A, descending depth")
    properties.save_view()
    saved_scope = window.row_ids()
    window.interpolate()
    dialog = window.interpolation_window
    dialog.x.setCurrentIndex(dialog.x.findData(depth.id))
    dialog.y.setCurrentIndex(dialog.y.findData(s2.id))
    dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
    dialog.calculate()
    assert project.science == baseline
    assert [r.values for r in project.rows] == original_values
    assert len(dialog.proposals) == 1
    target = project.rows[3]
    proposal = dialog.proposals[0]
    assert (proposal.row_id, proposal.value) == (target.id, 4)
    assert (proposal.left_id, proposal.right_id) == (project.rows[2].id, project.rows[0].id)
    assert dialog.preview_model.rows is dialog.proposals
    assert dialog.preview_model.index(0, 2).data() == "4.0"
    dialog.preview_tabs.setCurrentWidget(dialog.chart_view)
    predicted = dialog.chart_view.chart().series()[1]
    assert [(point.x(), point.y()) for point in predicted.points()] == [(20, 4)]
    dialog.apply()
    created = project.columns[-1]
    assert target.values[s2.id] is None
    assert target.values[created.id] == 4
    assert project.rows[4].values[s2.id] == "ND"
    assert project.rows[4].values[created.id] is None
    assert project.rows[1].values[created.id] is None
    assert not window.model.flags(window.model.index(3, 5)) & Qt.ItemFlag.ItemIsEditable
    assert project.science == baseline
    assert created.metadata["scope_row_ids"] == sorted(saved_scope)
    window.model.undo.undo()
    assert [r.values for r in project.rows] == original_values
    window.model.undo.redo()
    assert target.values[created.id] == 4

    path = tmp_path / "table-acceptance.apyro"
    window.path = path
    assert window.save()
    restored = Project.load(path)
    window.set_project(restored, path)
    assert window.row_ids() == saved_scope
    assert window.table.isColumnHidden(1)
    assert restored.science == baseline
    assert restored.row_map()[target.id].values[created.id] == 4
    assert restored.columns[-1].metadata == created.metadata
    window.apply_filter([], "AND")
    assert window.proxy.rowCount() == 5
    properties = window.desktop.table_properties
    properties.saved_views.setCurrentText("Well A, descending depth")
    properties.load_view()
    assert window.row_ids() == saved_scope
    window.interpolate()
    dialog = window.interpolation_window
    assert dialog.saved_runs.currentData() == created.id
    # Verification uses frozen inputs after a live edit, without revalidating the column.
    window.model.setData(window.model.index(0, 4), "60")
    assert restored.columns[-1].metadata["stale"]
    before_replay = deepcopy(restored)
    revision = window.model.data_revision
    summary = dialog.summary.text()
    announcements = []
    monkeypatch.setattr(dialog, "announce_summary", lambda text=None: announcements.append(text or dialog.summary.text()))
    dialog.replay_button.click()
    assert not dialog.replay_status.isHidden()
    assert announcements == [dialog.replay_status.text()] == [
        tr('Расчёт воспроизведён из снимка: {0} значений. Таблица не изменена.', 1)
    ]
    assert dialog.summary.text() == summary
    dialog.load_run_button.click()
    assert announcements[-1] == tr('Параметры загружены. Предпросмотр использует текущую область и схему.')
    assert restored == before_replay
    assert window.model.data_revision == revision


def test_loading_saved_interpolation_without_well_requires_fresh_confirmation(table_window):
    window = table_window
    window.interpolate()
    dialog = window.interpolation_window
    project = window.model.project
    well, _, depth, _, s2 = project.columns
    dialog.x.setCurrentIndex(dialog.x.findData(depth.id))
    dialog.y.setCurrentIndex(dialog.y.findData(s2.id))
    dialog.well.setCurrentIndex(dialog.well.findData(None))
    dialog.without_well.setChecked(True)
    dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
    dialog.calculate()
    assert len(dialog.proposals) == 1
    dialog.apply()
    assert project.columns[-1].metadata["constraints"]["well_id"] is None

    dialog.well.setCurrentIndex(dialog.well.findData(well.id))
    dialog.load_saved_parameters()
    assert dialog.well.currentData() is None
    assert not dialog.without_well.isChecked()
    assert not dialog.preview_button.isEnabled()
    assert tr('Выберите ствол или подтвердите расчёт без разделения по стволам.') in dialog.summary.text()
    dialog.without_well.setChecked(True)
    assert dialog.preview_button.isEnabled()


@pytest.mark.parametrize("qualified", ["ND", "<0,25", ">10", "≈3"])
def test_bulk_fill_recalculates_hi_and_undo_by_uuid(table_window, monkeypatch, qualified):
    monkeypatch.setattr(BatchEditPreview, "exec", lambda self: QDialog.DialogCode.Accepted)
    window = table_window
    project = window.model.project
    well, sample, depth, toc, s2 = project.columns
    column, values = make_calculated_column(project, "HI", {
        "formula": "HI", "version": 1, "analysis_context": "Synthetic whole rock",
        "compatible_confirmed": True,
        "inputs": {c.name: {"column_id": c.id, "unit": c.unit} for c in (toc, s2)}})
    window.model.undo.push(AddColumnCommand(window.model, column, values))
    window.apply_filter([{"column": well.id, "operator": "eq", "value": "A"}], "AND")
    window.proxy.sort(2, Qt.SortOrder.AscendingOrder)
    selected = window.table.selectionModel()
    selected.select(window.proxy.index(0, 4), QItemSelectionModel.SelectionFlag.ClearAndSelect)
    rid = window.row_ids()[0]
    original = project.science.current(rid, s2.id)
    source = deepcopy(project.science.sources[original.source_id])
    assert project.row_map()[rid].values[column.id] == 100
    monkeypatch.setattr(QInputDialog, "getText", lambda *args: (qualified, True))
    window.fill_cells()
    assert project.row_map()[rid].values[s2.id] == qualified
    assert project.row_map()[rid].values[column.id] is None
    assert project.science.current(rid, s2.id).id == original.id
    window.model.undo.undo()
    assert project.row_map()[rid].values[column.id] == 100
    assert project.science.current(rid, s2.id).value == 2
    window.model.undo.redo()
    assert project.row_map()[rid].values[column.id] is None
    assert project.science.sources[original.source_id] == source
    assert [project.rows[window.proxy.mapToSource(index).row()].id
            for index in selected.selectedIndexes()] == [rid]
    # Qualified values in the interpolation preview are excluded, as in its engine.
    window.interpolate()
    dialog = window.interpolation_window
    dialog.x.setCurrentIndex(dialog.x.findData(depth.id))
    dialog.y.setCurrentIndex(dialog.y.findData(s2.id))
    dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
    dialog.calculate()
    dialog.preview_tabs.setCurrentWidget(dialog.chart_view)
    measured = dialog.chart_view.chart().series()[0]
    assert measured.count() == 1
    assert dialog.proposals == []


def test_large_interpolation_and_replay_use_worker(table_window, monkeypatch):
    from PySide6.QtCore import QThread
    from autopyro.ui.dialogs import interpolation_calculation, interpolation_runs

    monkeypatch.setattr(interpolation_calculation, "BACKGROUND_ROWS", 1)
    monkeypatch.setattr(interpolation_runs, "BACKGROUND_ROWS", 1)
    threads = []
    prepare, replay = interpolation_calculation.prepare, interpolation_runs.replay

    def tracked_prepare(*args, **kwargs):
        threads.append(QThread.currentThread())
        return prepare(*args, **kwargs)

    def tracked_replay(protocol, **kwargs):
        threads.append(QThread.currentThread())
        return replay(protocol, **kwargs)

    monkeypatch.setattr(interpolation_calculation, "prepare", tracked_prepare)
    monkeypatch.setattr(interpolation_runs, "replay", tracked_replay)
    window = table_window
    window.interpolate()
    dialog = window.interpolation_window
    dialog.x.setCurrentIndex(dialog.x.findData(window.model.project.columns[2].id))
    dialog.y.setCurrentIndex(dialog.y.findData(window.model.project.columns[4].id))
    dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
    dialog.calculate()
    assert len(dialog.proposals) == 1
    dialog.apply()
    assert window.model.columnCount() == 6
    dialog.replay_button.click()
    assert len(threads) == 2
    assert all(thread != QThread.currentThread() for thread in threads)


@pytest.mark.parametrize("operation", ["prepare", "validation_scores", "replay"])
def test_cancel_interpolation_during_calculation_preserves_previous_preview(table_window, monkeypatch, operation):
    from threading import Event
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QDialogButtonBox, QPushButton
    from autopyro.ui.dialogs import interpolation_calculation, interpolation_diagnostics, interpolation_runs
    from autopyro.ui.tasks import TaskDialog

    window = table_window
    window.interpolate()
    dialog = window.interpolation_window
    project = window.model.project
    if operation == "validation_scores":
        assert window.model.setData(window.model.index(4, 4), "5")
    dialog.x.setCurrentIndex(dialog.x.findData(project.columns[2].id))
    dialog.y.setCurrentIndex(dialog.y.findData(project.columns[4].id))
    dialog.calculate()
    if operation == "replay":
        dialog.apply()
        dialog.calculate()
    previous = dialog.protocol
    assert previous and len(dialog.proposals) == 1
    previous_summary = dialog.summary.text()
    column_count = window.model.columnCount()
    announcements = []
    monkeypatch.setattr(dialog, "announce_summary", lambda text=None: announcements.append(text or dialog.summary.text()))

    reached, released = Event(), Event()
    component = {
        "prepare": interpolation_calculation,
        "validation_scores": interpolation_diagnostics,
        "replay": interpolation_runs,
    }[operation]
    calculate = getattr(component, operation)

    def pause_inside_calculation(*args, checkpoint, **kwargs):
        calls = 0

        def pause():
            nonlocal calls
            calls += 1
            if calls == 2:
                reached.set()
                assert released.wait(5)
            checkpoint()

        return calculate(*args, checkpoint=pause, **kwargs)

    monkeypatch.setattr(component, operation, pause_inside_calculation)
    if operation != "validation_scores":
        monkeypatch.setattr(component, "BACKGROUND_ROWS", 1)
    timer = QTimer()

    def cancel_when_ready():
        if reached.is_set():
            task = dialog.findChild(TaskDialog)
            task.findChild(QPushButton).click()
            released.set()
            timer.stop()

    timer.timeout.connect(cancel_when_ready)
    timer.start(5)
    try:
        {"prepare": dialog.calculate, "validation_scores": dialog.validate_methods,
         "replay": dialog.replay_saved}[operation]()
    finally:
        timer.stop()
    assert reached.is_set() and released.is_set()
    if operation == "prepare":
        assert dialog.summary.text().splitlines()[0] == tr('Расчёт отменён.')
    else:
        assert dialog.summary.text() == previous_summary
    if operation == "replay":
        assert not dialog.replay_status.isHidden()
        assert announcements == [dialog.replay_status.text()] == [tr('Проверка снимка отменена.')]
    elif operation == "validation_scores":
        assert not dialog.validation_status.isHidden()
        assert announcements == [dialog.validation_status.text()] == [tr('Сравнение отменено')]
    assert dialog.protocol is previous and dialog.preview_model.rowCount() == 1
    assert dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled() == (operation != "prepare")
    assert window.model.columnCount() == column_count
    if operation == "replay":
        monkeypatch.setattr(interpolation_runs, "replay", calculate)
        dialog.replay_saved()
        assert announcements[-1] == dialog.replay_status.text() == tr(
            'Расчёт воспроизведён из снимка: {0} значений. Таблица не изменена.', 1
        )
        assert dialog.summary.text() == previous_summary


@pytest.mark.parametrize("change", ["data", "proxy_filter", "proxy_sort"])
def test_interpolation_discards_preview_if_data_change_during_worker(table_window, monkeypatch, change):
    from threading import Event
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QDialogButtonBox
    from autopyro.ui.dialogs import interpolation_calculation as interpolation

    monkeypatch.setattr(interpolation, "BACKGROUND_ROWS", 1)
    changed = Event()
    prepare = interpolation.prepare

    def delayed_prepare(*args, **kwargs):
        assert changed.wait(5)
        return prepare(*args, **kwargs)

    monkeypatch.setattr(interpolation, "prepare", delayed_prepare)
    window = table_window
    window.interpolate()
    dialog = window.interpolation_window
    dialog.x.setCurrentIndex(dialog.x.findData(window.model.project.columns[2].id))
    dialog.y.setCurrentIndex(dialog.y.findData(window.model.project.columns[4].id))
    dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)

    def change_source():
        if change == "data":
            window.model.setData(window.model.index(0, 4), "60")
        elif change == "proxy_filter":
            window.proxy.set_rules([{"column": window.model.project.columns[0].id, "operator": "eq", "value": "A"}])
        else:
            window.proxy.sort(2, Qt.SortOrder.DescendingOrder)
        changed.set()

    QTimer.singleShot(0, change_source)
    dialog.calculate()
    assert not dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    assert dialog._preview_source is None
    assert window.model.columnCount() == 5


@pytest.mark.parametrize("surface", ["diagnostics", "proposals", "chart", "export", "validation"])
def test_interpolation_navigation_checks_current_proxy_order(table_window, monkeypatch, surface):
    from PySide6.QtCore import QPointF

    window = table_window
    window.interpolate()
    dialog = window.interpolation_window
    project = window.model.project
    dialog.x.setCurrentIndex(dialog.x.findData(project.columns[2].id))
    dialog.y.setCurrentIndex(dialog.y.findData(project.columns[4].id))
    dialog.calculate()
    assert dialog.proposals
    opened = []
    monkeypatch.setattr(window, "open_source_cell", lambda *args: opened.append(args))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: pytest.fail("Opened export for stale diagnostics"))
    window.proxy.sort(2, Qt.SortOrder.AscendingOrder)
    if surface == "diagnostics":
        dialog.open_diagnostic_sample(dialog.diagnostic_model.index(0, 0))
    elif surface == "proposals":
        dialog.open_proposal_sample(dialog.preview_model.index(0, 0))
    elif surface == "chart":
        dialog.open_chart_sample(dialog.chart_view.chart(), {(10.0, 2.0): [project.rows[2].id]}, QPointF(10, 2))
    elif surface == "export":
        dialog.export_diagnostics()
    else:
        dialog.validate_methods()
    assert not opened
    assert not dialog._diagnostics_current
    assert not dialog.export_diagnostics_button.isEnabled()


def test_interpolation_export_rechecks_scope_after_save_dialog(table_window, monkeypatch, tmp_path):
    window = table_window
    window.interpolate()
    dialog = window.interpolation_window
    project = window.model.project
    dialog.x.setCurrentIndex(dialog.x.findData(project.columns[2].id))
    dialog.y.setCurrentIndex(dialog.y.findData(project.columns[4].id))
    dialog.calculate()
    target = tmp_path / "diagnostics.csv"
    target.write_bytes(b"previous export")

    def choose_file(*args):
        window.proxy.sort(2, Qt.SortOrder.AscendingOrder)
        return str(target), "CSV (*.csv)"

    monkeypatch.setattr(QFileDialog, "getSaveFileName", choose_file)
    dialog.export_diagnostics()
    assert target.read_bytes() == b"previous export"
    assert not dialog._diagnostics_current


@pytest.mark.parametrize("damaged", [True, "broken snapshot"])
def test_saved_interpolation_reports_malformed_snapshot(table_window, damaged, monkeypatch):
    from autopyro.core import Column

    window = table_window
    column = Column("Damaged interpolation", "number",
                    metadata={"method": "linear", "interpolation_protocol": damaged})
    window.model.undo.push(AddColumnCommand(window.model, column, {}))
    window.interpolate()
    dialog = window.interpolation_window
    before = deepcopy(window.model.project)
    summary = dialog.summary.text()
    announcements = []
    monkeypatch.setattr(dialog, "announce_summary", lambda text=None: announcements.append(text))
    dialog.replay_button.click()
    assert not dialog.replay_status.isHidden()
    assert announcements == [dialog.replay_status.text()] == [
        tr('Ошибка: {0}', tr('Неподдерживаемая версия протокола интерполяции'))
    ]
    assert dialog.summary.text() == summary
    assert window.model.project == before
