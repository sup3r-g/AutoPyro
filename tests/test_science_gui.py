import json
from copy import deepcopy
from pathlib import Path

import pytest
from PySide6.QtWidgets import QFileDialog, QMessageBox

from autopyro.calculation_dialog import CalculationDialog
from autopyro.core import Project
from autopyro.io import build_project
from autopyro.window import MainWindow


@pytest.fixture
def window(qtbot):
    project = build_project(["TOC", "S1", "S2", "S3"], [["2", ".5", "6", "1.2"], ["1", ".1", "1", ".3"]], [1, 2],
                            [(0, "TOC", "number", "%wt"), (1, "S1", "number", "mg HC/g rock"),
                             (2, "S2", "number", "mg HC/g rock"), (3, "S3", "number", "mg CO2/g rock")], {})
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.may_replace = lambda: True
    yield window
    for dialog in (window.plot_window, window.interpretation_window):
        if dialog:
            dialog.close()
    window.dirty = False


def add_calculations(window, qtbot):
    dialog = CalculationDialog(window)
    qtbot.addWidget(dialog)
    dialog.context.setText("Pyrol6 whole rock, test series")
    dialog.confirmed.setChecked(True)
    dialog.outputs.cellWidget(1, 0).setChecked(True)
    dialog.calculate()
    assert len(dialog.prepared) == 4
    dialog.apply()
    return dialog


def ready(qtbot, dialog):
    qtbot.waitUntil(lambda: dialog.report is not None and not dialog.timer.isActive(), timeout=10000)


def test_formula_dialog_recalculation_and_macro_undo(window, qtbot):
    add_calculations(window, qtbot)
    model = window.model
    hi = next(c for c in model.project.columns if c.name == "HI")
    assert model.project.rows[0].values[hi.id] == 300
    creation_event = next(e for e in model.project.history if e["action"] == "add_column" and e["column"] == hi.id)
    historical_formula = deepcopy(creation_event["metadata"]["calculation"])
    assert not model.setData(model.index(0, 4), "999")
    assert model.setData(model.index(0, 2), "8")
    assert model.project.rows[0].values[hi.id] == 400
    assert creation_event["metadata"]["calculation"] == historical_formula
    model.undo.undo()
    assert model.project.rows[0].values[hi.id] == 300
    model.undo.undo()
    assert model.columnCount() == 4
    model.undo.redo()
    assert model.columnCount() == 8
    assert model.project.rows[0].values[hi.id] == 300


def test_osi_is_available_in_calculation_dialog_and_recomputes(window, qtbot):
    dialog = CalculationDialog(window)
    qtbot.addWidget(dialog)
    dialog.context.setText("Whole rock")
    dialog.confirmed.setChecked(True)
    for i in range(dialog.outputs.rowCount()):
        check = dialog.outputs.cellWidget(i, 0)
        if check.text() == "OSI":
            assert not check.isChecked()
        check.setChecked(check.text() == "OSI")
    dialog.calculate()
    assert dialog.preview.item(0, 1).text() == "25.0"
    dialog.apply()
    model = window.model
    osi = model.project.columns[-1]
    assert (osi.name, osi.unit) == ("OSI", "mg HC/g TOC")
    row = model.project.rows[0]
    assert model.setData(model.index(0, 0), "0")
    assert row.values[osi.id] is None
    assert osi.metadata["diagnostics"][row.id]["status"] == "zero_denominator"
    model.undo.undo()
    assert row.values[osi.id] == 25


def test_interpretation_live_updates_overlay_and_reload(window, qtbot, tmp_path):
    add_calculations(window, qtbot)
    window.show_interpretation()
    dialog = window.interpretation_window
    scheme = json.loads((Path(__file__).parents[1] / "examples" / "interpretation-demo.json").read_text(encoding="utf-8"))
    dialog.set_scheme(scheme)
    dialog.store_scheme()
    ready(qtbot, dialog)
    assert dialog.report["results"][0]["status"] == "boundary"
    assert dialog.report["results"][1]["status"] == "classified"
    dialog.show_on_plot()
    plot = dialog.plot
    qtbot.waitUntil(lambda: plot.result is not None and not plot.timer.isActive(), timeout=10000)
    from matplotlib.patches import PathPatch
    assert sum(isinstance(patch, PathPatch) for patch in plot.figure.axes[0].patches) == 2
    window.model.setData(window.model.index(0, 0), "1.5")
    assert window.model.project.view["interpretation_result"]["stale"]
    ready(qtbot, dialog)
    assert dialog.report["results"][0]["status"] == "classified"
    assert not window.model.project.view["interpretation_result"]["stale"]
    window.path = tmp_path / "science.apyro"
    assert window.save()
    restored = Project.load(window.path)
    assert not restored.view["interpretation_result"]["stale"]
    hi = next(c for c in restored.columns if c.name == "HI")
    restored.rows[0].values[restored.columns[2].id] = 20
    restored.save(window.path)
    altered = Project.load(window.path)
    assert altered.rows[0].values[hi.id] == pytest.approx(100 * 20 / 1.5)
    assert altered.view["interpretation_result"]["stale"]


def test_geometry_changes_require_new_revision(window, qtbot, monkeypatch):
    from autopyro.ui.dialogs.scheme_revision import SchemeRevisionDialog
    from PySide6.QtWidgets import QDialog

    add_calculations(window, qtbot)
    window.show_interpretation()
    dialog = window.interpretation_window
    scheme = json.loads((Path(__file__).parents[1] / "examples" / "interpretation-demo.json").read_text(encoding="utf-8"))
    dialog.set_scheme(scheme)
    dialog.store_scheme()
    ready(qtbot, dialog)
    changed = deepcopy(scheme)
    changed["regions"][0]["label"] = "Edited label"
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    dialog.editor.setPlainText(json.dumps(changed))
    dialog.apply_editor()
    assert warnings and "revision" in warnings[-1]
    assert window.model.project.view["interpretation_schemes"][scheme["id"]] == scheme
    changed["revision"] = "2"
    monkeypatch.setattr(SchemeRevisionDialog, 'exec', lambda self: QDialog.DialogCode.Accepted)
    dialog.editor.setPlainText(json.dumps(changed))
    dialog.apply_editor()
    ready(qtbot, dialog)
    assert dialog.report["scheme"]["revision"] == "2"


def test_interpretation_snapshot_reproduces_results(window, qtbot, tmp_path, monkeypatch):
    from autopyro.schemes.interpretation import replay_report
    add_calculations(window, qtbot)
    window.show_interpretation()
    dialog = window.interpretation_window
    scheme = json.loads((Path(__file__).parents[1] / "examples" / "interpretation-demo.json").read_text(encoding="utf-8"))
    dialog.set_scheme(scheme)
    dialog.store_scheme()
    ready(qtbot, dialog)
    path = tmp_path / "snapshot.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), "JSON (*.json)"))
    dialog.save_report()
    report = json.loads(path.read_text(encoding="utf-8"))["report"]
    assert replay_report(report) == report["results"]


def test_storing_scheme_updates_the_main_plot_once(window, monkeypatch):
    window.show_interpretation()
    window.show_plots()
    dialog = window.interpretation_window
    scheme = json.loads((Path(__file__).parents[1] / "examples" / "interpretation-demo.json").read_text(encoding="utf-8"))
    dialog.set_scheme(scheme)
    dialog.timer.stop()
    window.plot_window.timer.stop()
    window.dirty = False
    updates = []
    monkeypatch.setattr(window.plot_window, "schedule", lambda: updates.append(window.dirty))

    dialog.store_scheme()
    dialog.store_scheme()

    assert updates == [True]
    assert window.model.project.view["interpretation_schemes"][scheme["id"]] == scheme


def test_graphical_editor_commits_revision_and_preserves_history(window, qtbot, monkeypatch, tmp_path):
    from autopyro.geometry_editor import GeometryEditor
    from autopyro.ui.dialogs.scheme_revision import SchemeRevisionDialog
    from PySide6.QtWidgets import QDialog

    add_calculations(window, qtbot)
    window.show_interpretation()
    dialog = window.interpretation_window
    scheme = json.loads((Path(__file__).parents[1] / "examples" / "interpretation-demo.json").read_text(encoding="utf-8"))
    dialog.set_scheme(scheme)
    dialog.store_scheme()
    ready(qtbot, dialog)
    dialog.show_on_plot()
    plot = dialog.plot
    qtbot.waitUntil(lambda: plot.result is not None and not plot.timer.isActive(), timeout=10000)
    def edit(editor):
        qtbot.addWidget(editor)
        editor.vertex_table.item(0, 0).setText("0.25")
        editor.apply()
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(GeometryEditor, "exec", edit)
    monkeypatch.setattr(SchemeRevisionDialog, 'exec', lambda self: QDialog.DialogCode.Accepted)
    dialog.open_geometry_editor()
    assert window.model.project.view["interpretation_result"]["stale"]
    ready(qtbot, dialog)
    qtbot.waitUntil(lambda: plot.result is not None and not plot.timer.isActive(), timeout=10000)
    history = window.model.project.view["interpretation_scheme_revisions"][scheme["id"]]
    assert history["1"] == scheme
    assert history["2"]["regions"][0]["geometry"]["coordinates"][0][0] == [.25, 0]
    assert dialog.report["scheme"]["revision"] == "2"
    # The embedded plot follows the active editor; the previous revision stays
    # frozen in project history rather than being pinned in a separate window.
    assert plot.spec.interpretation["scheme"] == history["2"]
    window.path = tmp_path / "revisions.apyro"
    assert window.save()
    assert Project.load(window.path).view["interpretation_scheme_revisions"] == window.model.project.view["interpretation_scheme_revisions"]
    reuse = deepcopy(history["2"])
    reuse["revision"] = "1"
    with pytest.raises(ValueError, match="revision"):
        dialog.commit_graphical_scheme(reuse)
    assert dialog.scheme["revision"] == "2"


def test_revision_preview_selection_and_cancellation(window, qtbot, monkeypatch):
    from types import SimpleNamespace
    from PySide6.QtWidgets import QDialog
    from autopyro.ui.dialogs.scheme_revision import SchemeRevisionDialog

    add_calculations(window, qtbot)
    window.show_interpretation()
    dialog = window.interpretation_window
    scheme = json.loads((Path(__file__).parents[1] / "examples" / "interpretation-demo.json").read_text(encoding="utf-8"))
    dialog.commit_graphical_scheme(scheme, auto_bind=True)
    ready(qtbot, dialog)
    changed = deepcopy(scheme)
    changed['revision'] = '2'
    changed['x']['maximum'] = 0.5
    project = window.model.project

    def inspect(preview):
        assert project.view['interpretation_schemes'][scheme['id']] == scheme
        assert preview.table.model().rowCount() > 0
        assert all(axis.patches for axis in preview.canvas.figure.axes)
        points, highlight = preview.canvas.figure.axes[0].collections
        after_points, after_highlight = preview.canvas.figure.axes[1].collections
        preview.canvas.callbacks.process('pick_event', SimpleNamespace(artist=points, ind=[0]))
        assert preview.table.selectionModel().selectedRows()
        preview.canvas.callbacks.process('pick_event', SimpleNamespace(artist=after_points, ind=[0]))
        assert preview.table.selectionModel().selectedRows()
        assert preview.table.model().headers[0] == 'Проба'
        preview.table.clearSelection()
        preview.table.selectRow(0)
        assert highlight.get_visible()
        assert after_highlight.get_visible()
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(SchemeRevisionDialog, 'exec', inspect)
    dialog.commit_graphical_scheme(changed)
    assert dialog.scheme == scheme
    assert '2' not in project.view['interpretation_scheme_revisions'][scheme['id']]


def test_cancel_revision_during_capture_preserves_scheme(window, qtbot, monkeypatch):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QPushButton
    from autopyro.schemes.standard_interpretation import standard_schemes
    from autopyro.ui.dialogs import interpretation_schemes as module
    from autopyro.ui.tasks import TaskDialog

    window.show_interpretation()
    dialog = window.interpretation_window
    old = standard_schemes()['S2–TOC']
    dialog.commit_graphical_scheme(old, auto_bind=True)
    ready(qtbot, dialog)
    report = dialog.report
    new = deepcopy(old)
    new['revision'] += '-test'
    new['tolerance'] = .01

    def paused_capture(*_):
        progress = dialog.findChild(TaskDialog)
        cancel = progress.findChild(QPushButton)
        QTimer.singleShot(0, cancel.click)
        while True:
            yield

    monkeypatch.setattr(module, 'capture_interpretation_steps', paused_capture)
    monkeypatch.setattr(module, 'evaluate_project', lambda *_args, **_kwargs: pytest.fail('Started after cancellation'))
    dialog.commit_graphical_scheme(new)
    assert dialog.report is report
    assert dialog.scheme == old
    assert new['revision'] not in window.model.project.view['interpretation_scheme_revisions'][old['id']]


def test_revision_preview_uses_old_and_new_axis_bindings(window, qtbot, monkeypatch):
    from PySide6.QtWidgets import QDialog, QLabel
    from autopyro.schemes.standard_interpretation import standard_schemes
    from autopyro.ui.dialogs.scheme_revision import SchemeRevisionDialog

    window.show_interpretation()
    dialog = window.interpretation_window
    old = standard_schemes()['S2–TOC']
    dialog.commit_graphical_scheme(old, auto_bind=True)
    ready(qtbot, dialog)
    old_y = dialog.y.currentData()
    changed = deepcopy(old)
    changed['revision'] = '2'
    changed['y']['property'] = 'S1'

    def inspect(preview):
        headings = [label.text() for label in preview.findChildren(QLabel)]
        assert any(text.startswith('До · ') and 'S2' in text for text in headings)
        assert any(text.startswith('После · ') and 'S1' in text for text in headings)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(SchemeRevisionDialog, 'exec', inspect)
    dialog.commit_graphical_scheme(changed)
    ready(qtbot, dialog)
    assert dialog.y.currentData() != old_y
    assert dialog.report['scheme']['y']['property'] == 'S1'


def test_cancel_graphical_editor_keeps_project_scheme_unchanged(window, qtbot, monkeypatch):
    from autopyro.geometry_editor import GeometryEditor
    from PySide6.QtWidgets import QDialog
    add_calculations(window, qtbot)
    window.show_interpretation()
    dialog = window.interpretation_window
    scheme = json.loads((Path(__file__).parents[1] / "examples" / "interpretation-demo.json").read_text(encoding="utf-8"))
    dialog.set_scheme(scheme)
    dialog.store_scheme()
    ready(qtbot, dialog)
    view_before = deepcopy(window.model.project.view)
    def cancel(editor):
        qtbot.addWidget(editor)
        editor.vertex_table.item(0, 0).setText("0.25")
        editor.reject()
        return QDialog.DialogCode.Rejected
    monkeypatch.setattr(GeometryEditor, "exec", cancel)
    dialog.open_geometry_editor()
    assert window.model.project.view == view_before


def test_new_graphical_scheme_wizard_binds_project_columns(window, qtbot, monkeypatch):
    from autopyro.geometry_editor import GeometryEditor, NewSchemeDialog
    from autopyro.scheme_geometry import add_geometry
    from PySide6.QtWidgets import QDialog

    add_calculations(window, qtbot)
    window.show_interpretation()
    dialog = window.interpretation_window
    def properties(wizard):
        qtbot.addWidget(wizard)
        values = {"name": "Synthetic new scheme", "source": "Synthetic GUI test",
                  "x_property": "TOC", "x_unit": "%wt", "y_property": "HI", "y_unit": "mg HC/g TOC"}
        for key, value in values.items():
            wizard.fields[key].setText(value)
        wizard.apply()
        return QDialog.DialogCode.Accepted
    def geometry(editor):
        qtbot.addWidget(editor)
        editor.push(add_geometry(editor.draft, "Polygon", [[0, 0], [4, 0], [4, 500], [0, 500]], "A", "A"), "Create area")
        editor.apply()
        assert editor.edited_scheme is not None
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(NewSchemeDialog, "exec", properties)
    monkeypatch.setattr(GeometryEditor, "exec", geometry)
    dialog.new_graphical_scheme()
    ready(qtbot, dialog)
    assert dialog.scheme["revision"] == "1"
    assert all(result["status"] == "classified" for result in dialog.report["results"])
    assert window.model.project.view["interpretation_scheme_revisions"][dialog.scheme["id"]]["1"] == dialog.scheme


def test_unbound_created_scheme_survives_reopening(window, qtbot, tmp_path):
    from autopyro.geometry_editor import new_scheme
    from autopyro.scheme_geometry import add_geometry
    window.show_interpretation()
    dialog = window.interpretation_window
    scheme = new_scheme("Unmapped X", "1", "Unmapped Y", "1")
    scheme["source"] = "Synthetic unbound scheme"
    scheme = add_geometry(scheme, "Polygon", [[0, 0], [1, 0], [1, 1]], "A", "A")
    dialog.commit_graphical_scheme(scheme, auto_bind=True)
    assert dialog.x.currentData() == ""
    window.path = tmp_path / "unbound.apyro"
    assert window.save()
    window.set_project(Project.load(window.path), window.path)
    window.show_interpretation()
    assert window.interpretation_window.scheme == scheme
