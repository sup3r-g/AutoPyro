"""End-to-end acceptance through existing import, Qt, report and storage APIs."""
import json
import os
from pathlib import Path
import time

import pytest
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QDialogButtonBox, QFileDialog, QMessageBox, QPushButton
from pypdf import PdfReader

from autopyro.core import Project
from autopyro.io import read_file, build_project, file_source
from autopyro.plotting import PlotSpec
from autopyro.ui.window import MainWindow
from autopyro.ui.dialogs.calculation import CalculationDialog
from autopyro.ui.dialogs.measurement import MeasurementDialog
from autopyro.ui.dialogs.report import ReportDialog
from autopyro.ui.table.commands import QCCommand
from autopyro.ui.data_io.append_import import AppendImportDialog
from autopyro.ui.data_io.import_file import ImportDialog
from autopyro.ui import i18n


def test_laboratory_route_from_import_commands_to_qc_and_report(qtbot, qapp, tmp_path, monkeypatch, request):
    # The Windows offscreen plugin lacks the installed system fonts.
    if qapp.platformName() == "offscreen" and os.name == "nt":
        font_path = Path(os.environ["WINDIR"]) / "Fonts" / "segoeui.ttf"
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        if font_id >= 0:
            original_font = qapp.font()
            qapp.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0], 9))

            def restore_font():
                qapp.setFont(original_font)
                QFontDatabase.removeApplicationFont(font_id)

            request.addfinalizer(restore_font)
    source = Path(__file__).resolve().parents[1] / "examples" / "pyrolysis.csv"
    repeat = tmp_path / "repeat.csv"
    repeat.write_text("Well;MD;S2\nDemo-A;1000;3.6\n", encoding="utf-8")
    originals = {path: path.read_bytes() for path in (source, repeat)}
    paths = iter([source, repeat])
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(next(paths)), ""))
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    store = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    store.setValue("general/autosave_enabled", False)
    monkeypatch.setattr(i18n, "settings", lambda: store)
    i18n.initialize()
    tr = i18n.tr
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1024, 700)
    window.show()

    def activate(label):
        window.actions[tr(label)].trigger()
        assert not warnings, warnings

    def button(dialog, text):
        return next(button for button in dialog.findChildren(QPushButton) if button.text() == tr(text))

    def drive_modal(dialog_class, interaction):
        execute = dialog_class.exec

        def execute_with_actions(dialog):
            failures = []

            def interact():
                try:
                    dialog.resize(970, 640)
                    assert dialog.width() <= 992 and dialog.height() <= 652
                    interaction(dialog)
                except BaseException as exc:
                    failures.append(exc)
                    dialog.reject()

            QTimer.singleShot(0, interact)
            result = execute(dialog)
            if failures:
                raise failures[0]
            return result

        monkeypatch.setattr(dialog_class, "exec", execute_with_actions)

    def import_table(dialog):
        button(dialog, "Прочитать / обновить предпросмотр").click()
        qtbot.waitUntil(lambda: dialog.raw is not None and dialog.isEnabled(), timeout=10000)
        units = {"MD": "m", "S2": "mg HC/g rock"}
        for index in range(dialog.mapping.rowCount()):
            name = dialog.mapping.item(index, 1).text()
            if name in units:
                kind = dialog.mapping.cellWidget(index, 3)
                kind.setCurrentIndex(kind.findData("number"))
                dialog.mapping.item(index, 4).setText(units[name])
        dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).click()
        qtbot.waitUntil(lambda: dialog.result() == dialog.DialogCode.Accepted, timeout=10000)

    def append_repeat(dialog):
        for index, column in enumerate(dialog.incoming.columns):
            if column.name in ("Well", "MD"):
                dialog.mapping.cellWidget(index, 2).setChecked(True)
        button(dialog, "Сравнить с проектом").click()
        qtbot.waitUntil(lambda: dialog.apply_button.isEnabled(), timeout=10000)
        assert [row["status"] for row in dialog.plan.rows] == ["matched"]
        dialog.apply_button.click()
        qtbot.waitUntil(lambda: dialog.result() == dialog.DialogCode.Accepted, timeout=10000)

    drive_modal(ImportDialog, import_table)
    drive_modal(AppendImportDialog, append_repeat)
    activate("Открыть таблицу как новый проект…")
    project_id = window.model.project.id
    activate("Добавить измерения в текущий проект…")
    project = window.model.project
    assert project.id == project_id and len(project.rows) == 6
    sample = project.rows[0]
    position, s2 = next((index, col) for index, col in enumerate(project.columns) if col.name == "S2")
    measurements = project.science.results(sample.id, s2.id)
    assert {measurement.value for measurement in measurements} == {2.4, 3.6}
    chosen = next(measurement for measurement in measurements if measurement.value == 3.6)

    def choose_repeat(dialog):
        dialog.table.selectRow(next(row for row in range(dialog.table.rowCount())
                                    if dialog.table.item(row, 0).data(Qt.ItemDataRole.UserRole) == chosen.id))
        dialog.select_button.click()
        button(dialog, "Закрыть").click()

    drive_modal(MeasurementDialog, choose_repeat)
    window.table.setCurrentIndex(window.proxy.mapFromSource(window.model.index(0, position)))
    activate("Измерения выбранной ячейки…")
    assert project.science.current(sample.id, s2.id).id == chosen.id
    assert sample.values[s2.id] == 3.6

    activate("Контроль качества…")
    qc = window.qc_window
    qtbot.waitUntil(lambda: any(issue["measurement_id"] == chosen.id for issue in qc.results_model.issues))
    row = next(index for index, issue in enumerate(qc.results_model.issues)
               if issue["measurement_id"] == chosen.id)
    qc.select_issue_row(row)
    issue_id = qc.current_issue()["id"]
    comment = "Repeat verified against the source record"
    qc.comment.setText(comment)
    qc.acknowledge.click()
    assert project.view["qc_decisions"][issue_id]["action"] == "acknowledged"
    assert window.size().toTuple() == (1024, 700)

    report_files = {kind: tmp_path / f"report.{kind}" for kind in ("pdf", "json")}
    export_paths = iter(report_files.values())
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(next(export_paths)), ""))

    def export_report(dialog):
        dialog.qc_open_only.setChecked(False)
        button(dialog, "Собрать отчёт").click()
        assert dialog.report is not None, dialog.status.text()
        for kind in report_files:
            action = next(action for action in dialog.exports if action.text() == tr("Экспорт {0}", kind.upper()))
            assert action.isEnabled(), dialog.status.text()
            action.trigger()
        button(dialog, "Закрыть").click()

    drive_modal(ReportDialog, export_report)
    activate("Отчёт…")
    snapshot = json.loads(report_files["json"].read_text(encoding="utf-8"))
    issue = next(issue for issue in snapshot["qc"]["issues"] if issue["id"] == issue_id)
    assert issue["decision"] == "acknowledged" and issue["comment"] == comment
    pages = PdfReader(report_files["pdf"]).pages
    assert comment in " ".join(" ".join(page.extract_text().split()) for page in pages)
    saved = tmp_path / "lab.apyro"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(saved), ""))
    assert window.save()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(saved), ""))
    activate("Открыть проект…")
    restored = window.model.project
    assert restored.id == project_id and restored.science.current(sample.id, s2.id).id == chosen.id
    assert restored.view["qc_decisions"][issue_id]["comment"] == comment
    assert all(path.read_bytes() == original for path, original in originals.items())


def test_pyro_working_repeat_qc_plot_report_and_reopen(qtbot, tmp_path, monkeypatch):
    timings = {}
    source = tmp_path / "laboratory.csv"
    source.write_text("Well;Sample;Depth;TOC;S1;S2;S3\n"
                      "A;001;1000;2;.5;6;1.2\nA;001;1010;1;.2;ND;.4\n"
                      "B;002;1020;0;.3;3;.6\n", encoding="utf-8")
    started = time.perf_counter()
    headers, rows, numbers = read_file(source)
    units = ["", "", "m", "%wt", "mg HC/g rock", "mg HC/g rock", "mg CO2/g rock"]
    project = build_project(headers, rows, numbers,
                            [(i, name, "text" if i < 2 else "number", units[i])
                             for i, name in enumerate(headers)], file_source(source))
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    timings["import_and_table_seconds"] = time.perf_counter() - started
    owner.may_replace = lambda: True
    original_columns = {column.name: column for column in project.columns}
    sid = project.rows[0].id
    assert project.rows[1].id != sid
    assert project.rows[0].values[original_columns["Sample"].id] == "001"
    try:
        started = time.perf_counter()
        calculation = CalculationDialog(owner)
        qtbot.addWidget(calculation)
        calculation.context.setText("Synthetic QA, canonical units, whole rock")
        calculation.confirmed.setChecked(True)
        for index in range(4):
            calculation.outputs.cellWidget(index, 0).setChecked(True)
        calculation.calculate()
        assert len(calculation.prepared) == 4
        calculation.apply()
        timings["calculate_seconds"] = time.perf_counter() - started
        columns = {column.name: column for column in project.columns}
        hi, s2 = columns["HI"], columns["S2"]
        assert project.rows[0].values[hi.id] == 300
        assert project.rows[1].values[hi.id] is None
        assert project.rows[2].values[hi.id] is None
        original = project.science.current(sid, s2.id)
        repeat = project.science.add_result(sid, s2.id, 8,
                    analysis_fields={"method": "Pyrol6", "laboratory": "Lab B"},
                    details={"uncertainty": .3, "uncertainty_type": "standard"})
        card = MeasurementDialog(owner.model, sid, s2.id, owner)
        qtbot.addWidget(card)
        card.table.selectRow(next(i for i in range(card.table.rowCount())
                                  if card.table.item(i, 0).data(Qt.ItemDataRole.UserRole) == repeat.id))
        started = time.perf_counter()
        card.select_result()
        timings["select_repeat_seconds"] = time.perf_counter() - started
        assert project.rows[0].values[hi.id] == 400
        assert project.science.current(sid, s2.id).id == repeat.id
        assert project.rows[0].values[columns["GP"].id] == 8.5
        assert project.rows[0].values[columns["PI"].id] == pytest.approx(.5 / 8.5)
        assert project.rows[0].values[columns["OI"].id] == 60

        owner.show_plots()
        plot = owner.plot_window
        plot.spec = PlotSpec(x_id=columns["Depth"].id, y_id=hi.id)
        plot.load_controls()
        plot.schedule()
        qtbot.waitUntil(lambda: plot.result is not None and not plot.timer.isActive(), timeout=10000)
        assert plot.result["plotted"][columns["Depth"].id] == 1
        assert plot.result["artists"][0]._autopyro_row_ids == [sid]

        owner.model.undo.push(QCCommand(owner.model, ("qc_exclusions", "calculations"),
                              {repeat.id: {"revision": repeat.revision, "comment": "QA exclusion"}}))
        assert project.rows[0].values[hi.id] is None
        assert hi.metadata["diagnostics"][sid]["status"] == "qc_excluded"
        qtbot.waitUntil(lambda: plot.result is not None and not plot.timer.isActive(), timeout=10000)
        assert plot.result["plotted"][columns["Depth"].id] == 0
        owner.model.undo.undo()
        assert project.rows[0].values[hi.id] == 400
        owner.model.undo.undo()
        assert project.rows[0].values[hi.id] == 300
        owner.model.undo.redo()
        assert project.rows[0].values[hi.id] == 400
        qtbot.waitUntil(lambda: plot.result is not None and not plot.timer.isActive(), timeout=10000)

        report = ReportDialog(owner)
        qtbot.addWidget(report)
        for index in range(report.columns.count()):
            item = report.columns.item(index)
            item.setCheckState(Qt.CheckState.Checked if item.data(Qt.ItemDataRole.UserRole) == hi.id
                               else Qt.CheckState.Unchecked)
        report.plots.item(0).setCheckState(Qt.CheckState.Checked)
        started = time.perf_counter()
        report.build()
        assert report.report is not None, report.status.text()
        for kind in ("json", "html"):
            path = tmp_path / f"report.{kind}"
            monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, path=path: (str(path), ""))
            report.export(kind)
            assert path.stat().st_size > 100
        timings["report_and_exports_seconds"] = time.perf_counter() - started
        exported = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert exported["project"]["science"]["working"][sid][s2.id] == repeat.id
        assert {original.id, repeat.id} <= set(exported["project"]["science"]["measurements"])
        assert exported["project"]["rows"][0]["values"][hi.id] == 400
        assert exported["figures"][0]["plotted"][columns["Depth"].id] == 1
        frozen_report = report.report
        assert owner.model.setData(owner.model.index(0, project.columns.index(s2)), "9")
        assert project.rows[0].values[hi.id] == 450
        assert frozen_report["project"]["rows"][0]["values"][hi.id] == 400
        assert report.report is None and not any(button.isEnabled() for button in report.exports)
        owner.model.undo.undo()
        qtbot.waitUntil(lambda: plot.result is not None and not plot.timer.isActive(), timeout=10000)

        started = time.perf_counter()
        owner.path = tmp_path / "acceptance.apyro"
        assert owner.save()
        loaded = Project.load(owner.path)
        owner.set_project(loaded, owner.path)
        timings["save_and_reopen_seconds"] = time.perf_counter() - started
        assert [row.id for row in loaded.rows] == [row.id for row in project.rows]
        assert loaded.science.current(sid, s2.id).id == repeat.id
        assert loaded.science.measurements[repeat.id].uncertainty == .3
        assert loaded.science.measurements[original.id].value == 6
        assert loaded.rows[0].original[s2.id] == "6"
        assert loaded.rows[1].values[s2.id] == "ND"
        assert loaded.rows[0].values[hi.id] == 400
        assert loaded.view["plot_current"]["y_id"] == hi.id
        assert file_source(source)["sha256"] == project.source["sha256"]
        started = time.perf_counter()
        owner.show_plots()
        reopened_plot = owner.plot_window
        qtbot.waitUntil(lambda: reopened_plot.result is not None and not reopened_plot.timer.isActive(), timeout=10000)
        assert reopened_plot.spec.y_id == hi.id
        assert reopened_plot.result["plotted"][columns["Depth"].id] == 1
        assert reopened_plot.result["artists"][0]._autopyro_row_ids == [sid]
        timings["restore_plot_seconds"] = time.perf_counter() - started
        print(json.dumps({"acceptance_timings": timings}))
        (tmp_path / "timings.json").write_text(json.dumps(timings), encoding="utf-8")
    finally:
        owner.dirty = False
