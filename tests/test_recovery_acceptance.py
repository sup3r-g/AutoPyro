"""Migration, failed writes and recovery with derived columns and repeat results."""
import json
from dataclasses import asdict

import pytest
from PySide6.QtWidgets import QFileDialog

from autopyro.calculations import FORMULAS, make_calculated_column
from autopyro.core import Project
from autopyro.io import build_project
from autopyro.ui.window import MainWindow
from autopyro.ui.table.commands import SelectMeasurementCommand


def test_wide_table_edit_repeat_undo_background_recovery_and_reopen(qtbot, tmp_path):
    """Use the actual window commands and autosave publication on 80 source columns."""
    from autopyro.core import Column
    project = build_project(["Sample", "TOC", "S2"],
        [["001", "2", "6" if i % 3 == 0 else "ND" if i % 3 == 1 else "<0.1"]
         for i in range(1300)], list(range(2, 1302)),
        [(0, "Sample", "text", ""), (1, "TOC", "number", "%wt"),
         (2, "S2", "number", "mg HC/g rock")], {"type": "synthetic wide recovery QA"})
    for index in range(77):
        column = Column(f"Property {index}", "number")
        project.columns.append(column)
        for position, row in enumerate(project.rows):
            row.values[column.id] = position + index
            row.original[column.id] = str(position + index)
    project.sync_measurements()
    toc, s2 = project.columns[1:3]
    hi, values = make_calculated_column(project, "HI", {
        "formula": "HI", "version": 1, "compatible_confirmed": True,
        "analysis_context": "Synthetic wide recovery QA",
        "inputs": {"TOC": {"column_id": toc.id, "unit": toc.unit},
                   "S2": {"column_id": s2.id, "unit": s2.unit}}})
    project.columns.append(hi)
    for row in project.rows:
        row.values[hi.id] = values[row.id]
    row = project.rows[0]
    original = project.science.current(row.id, s2.id)
    original_data = asdict(original)
    repeat = project.science.add_result(row.id, s2.id, "8",
        analysis_fields={"method": "Pyrol6", "laboratory": "Lab B"},
        details={"uncertainty": .3, "uncertainty_type": "standard", "confidence_level": .95})
    path = tmp_path / "wide.apyro"
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project, path)
    assert owner.save()
    before = path.read_bytes()
    try:
        owner.model.undo.push(SelectMeasurementCommand(owner.model, row.id, s2.id, repeat.id))
        assert owner.model.setData(owner.model.index(0, 2), "9")
        assert row.values[hi.id] == 450
        owner.model.undo.undo()
        assert row.values[hi.id] == 400
        owner.model.undo.redo()
        assert row.values[hi.id] == 450
        owner.model.undo.undo()
        owner.model.undo.undo()
        assert project.science.current(row.id, s2.id) is original
        assert row.values[hi.id] == 300
        owner.model.undo.redo()
        owner.model.undo.redo()
        assert row.values[hi.id] == 450
        assert asdict(original) == original_data
        assert row.original[s2.id] == "6"
        recovery = owner.autosave_path()
        owner.autosave()
        assert owner._background_autosave is not None  # >100k table cells.
        qtbot.waitUntil(lambda: owner._background_autosave.worker is None, timeout=30000)
        assert recovery.exists() and path.read_bytes() == before
        owner.load_autosave(recovery)
        recovered = owner.model.project
        assert owner.dirty and owner.path == path
        assert recovered.science == project.science
        assert recovered.columns == project.columns and recovered.rows == project.rows
        assert recovered.history == project.history and recovered.source == project.source
        assert recovered.science.current(row.id, s2.id).id == repeat.id
        assert recovered.rows[1].values[s2.id] == "ND"
        assert recovered.rows[2].values[s2.id] == "<0.1"
        assert owner.save()
        assert not recovery.exists()
        reopened = Project.load(path)
        assert reopened.rows == recovered.rows and reopened.columns == recovered.columns
        assert reopened.science == recovered.science and reopened.history == recovered.history
        assert reopened.science.measurements[repeat.id].uncertainty == .3
        assert reopened.science.measurements[repeat.id].confidence_level == .95
        assert reopened.science.measurements[original.id] == original
    finally:
        if owner._background_autosave is not None:
            owner._background_autosave.shutdown()
        owner.dirty = False


@pytest.fixture
def legacy_pyro(tmp_path):
    names = ["Sample", "TOC", "S1", "S2", "S3"]
    units = ["", "%wt", "mg HC/g rock", "mg HC/g rock", "mg CO2/g rock"]
    project = build_project(names, [["001", "2", ".5", "6", "1.2"],
                                    ["001", "1", ".2", "ND", ".4"]], [2, 3],
        [(i, name, "text" if i == 0 else "number", units[i]) for i, name in enumerate(names)],
        {"path": "synthetic-legacy.csv", "sha256": "qa"})
    columns = {column.name: column for column in project.columns}
    for formula, definition in FORMULAS.items():
        config = {"formula": formula, "version": 1, "compatible_confirmed": True,
                  "analysis_context": "Synthetic QA, canonical units",
                  "inputs": {role: {"column_id": columns[role].id, "unit": columns[role].unit}
                             for role in definition["roles"]}}
        column, values = make_calculated_column(project, formula, config)
        project.columns.append(column)
        for row in project.rows:
            row.values[column.id] = values[row.id]
    payload = asdict(project)
    payload.pop("science")
    original = json.dumps({"format": "autopyro.table", "version": 1, **payload}, ensure_ascii=False).encode("utf-8")
    path = tmp_path / "legacy.apyro"
    path.write_bytes(original)
    return path, original


def test_failed_migration_can_retry_with_original_data_and_formulas(legacy_pyro, monkeypatch):
    from autopyro.database import storage

    path, original = legacy_pyro
    project = Project.load(path)
    reopened = Project.load(path)
    assert set(project.science.measurements) == set(reopened.science.measurements)
    assert path.read_bytes() == original
    replace_file = storage.os.replace

    def fail_commit(source, destination):
        if str(destination) == str(path.resolve()):
            raise OSError("QA commit failure")
        return replace_file(source, destination)

    with monkeypatch.context() as fault:
        fault.setattr(storage.os, "replace", fail_commit)
        with pytest.raises(OSError, match="QA commit failure"):
            project.save(path)
    assert path.read_bytes() == original
    assert not list(path.parent.glob("*.tmp"))
    project.save(path)
    backup, = path.parent.glob("legacy.apyro.v1-*.bak")
    assert backup.read_bytes() == original
    restored = Project.load(path)
    assert [row.id for row in restored.rows] == [row.id for row in project.rows]
    assert set(restored.science.measurements) == set(project.science.measurements)
    assert all(row.values == other.values for row, other in zip(restored.rows, project.rows))
    assert [column.metadata for column in restored.columns] == [column.metadata for column in project.columns]


def test_migrated_repeat_recovery_conflict_and_save_as(legacy_pyro, qtbot, monkeypatch, tmp_path):
    from autopyro.ui import window as window_module
    from autopyro.database import storage

    path, original_bytes = legacy_pyro
    owner = MainWindow()
    qtbot.addWidget(owner)
    project = Project.load(path)
    owner.set_project(project, path)
    assert owner.save()
    columns = {column.name: column for column in project.columns}
    sid, s2, hi = project.rows[0].id, columns["S2"].id, columns["HI"].id
    original = project.science.current(sid, s2)
    repeat = project.science.add_result(sid, s2, 8,
        analysis_fields={"method": "Pyrol6", "laboratory": "Lab B"},
        details={"uncertainty": .3, "uncertainty_type": "standard"})
    owner.model.undo.push(SelectMeasurementCommand(owner.model, sid, s2, repeat.id))
    project.view["qc_exclusions"] = {"statistics": {
        repeat.id: {"revision": repeat.revision, "comment": "QA review"}}}
    assert project.rows[0].values[hi] == 400
    recovery = owner.autosave_path()
    try:
        owner.autosave()
        qtbot.waitUntil(lambda: owner._background_autosave.worker is None, timeout=10000)
        assert recovery.exists()
        assert Project.load(path).rows[0].values[hi] == 300
        owner.load_autosave(recovery)
        recovered = owner.model.project
        assert owner.dirty
        assert recovered.science.current(sid, s2).id == repeat.id
        assert recovered.rows[0].values[hi] == 400
        assert recovered.science.measurements[repeat.id].uncertainty == .3
        assert recovered.view["qc_exclusions"]["statistics"][repeat.id]["comment"] == "QA review"
        assert recovered.science.measurements[original.id].value == 6
        assert recovered.rows[0].original[s2] == "6"
        before = path.read_bytes()
        replace_file = storage.os.replace

        def fail_commit(source, destination):
            if str(destination) == str(path.resolve()):
                raise OSError("QA recovered save failure")
            return replace_file(source, destination)

        with monkeypatch.context() as fault:
            fault.setattr(storage.os, "replace", fail_commit)
            with pytest.raises(OSError, match="QA recovered save failure"):
                recovered.save(path)
        assert path.read_bytes() == before
        assert recovery.exists()
        assert not list(tmp_path.glob("*.tmp"))

        external = Project.load(path)
        external.rows[0].values[s2] = 9
        external.refresh()
        external.save(path)
        external_bytes = path.read_bytes()
        with pytest.raises(ValueError, match="другим процессом"):
            recovered.save(path)
        assert path.read_bytes() == external_bytes
        alternative = tmp_path / "recovered-copy.apyro"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(alternative), ""))
        assert owner.save(as_new=True)
        assert path.read_bytes() == external_bytes
        assert not recovery.exists()
        restored = Project.load(alternative)
        assert restored.science.current(sid, s2).id == repeat.id
        assert restored.rows[0].values[hi] == 400
        assert restored.rows[1].values[s2] == "ND"
        assert set(restored.science.results(sid, s2)[i].id for i in range(2)) == {original.id, repeat.id}
        backup, = tmp_path.glob("legacy.apyro.v1-*.bak")
        assert backup.read_bytes() == original_bytes
    finally:
        if owner._background_autosave is not None:
            owner._background_autosave.shutdown()
        owner.dirty = False
