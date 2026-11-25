import json
from copy import deepcopy
from pathlib import Path

import pytest
from matplotlib.backend_bases import MouseEvent
from matplotlib.collections import LineCollection
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLineEdit

from autopyro.geometry_editor import GeometryEditor, new_scheme
from autopyro.interpretation import InterpretationInput
from autopyro.scheme_geometry import geometry_paths, replace_object, replace_vertices, vertices


@pytest.fixture
def scheme():
    return json.loads((Path(__file__).parents[1] / "examples" / "interpretation-demo.json").read_text(encoding="utf-8"))


@pytest.fixture
def editor(qtbot, scheme):
    editor = GeometryEditor(scheme, samples=[InterpretationInput("inside", 1, 100), InterpretationInput("edge", 2, 200)])
    qtbot.addWidget(editor)
    editor.show()
    editor.canvas.draw()
    return editor


def event(editor, name, x, y, button=1):
    editor.canvas.draw()
    px, py = editor.ax.transData.transform((x, y))
    mouse = MouseEvent(name, editor.canvas, px, py, button=button)
    editor.canvas.callbacks.process(name, mouse)


def click(editor, x, y):
    event(editor, "button_press_event", x, y)
    event(editor, "button_release_event", x, y)


def draw(editor, points):
    for x, y in points:
        click(editor, x, y)
    editor.finish_drawing()


def test_drag_vertex_and_exact_table_edit_have_undo(editor, scheme):
    before = deepcopy(scheme)
    event(editor, "button_press_event", 0, 0)
    event(editor, "motion_notify_event", .25, 25)
    event(editor, "button_release_event", .25, 25)
    point = vertices(editor.draft["regions"][0], geometry_paths(editor.draft["regions"][0])[0])[0]
    assert point == pytest.approx([.25, 25])
    assert editor.undo_stack.count() == 1
    assert scheme == before
    editor.vertex_table.item(0, 0).setText("0.5")
    changed = editor.current_entry()["geometry"]["coordinates"][0][0]
    assert changed[0] == .5
    assert changed[1] == pytest.approx(25)
    editor.undo_stack.undo()
    assert editor.current_entry()["geometry"]["coordinates"][0][0] == pytest.approx([.25, 25])
    editor.undo_stack.undo()
    assert editor.current_entry()["geometry"] == before["regions"][0]["geometry"]
    editor.undo_stack.redo()
    assert editor.current_entry()["geometry"]["coordinates"][0][-1] == pytest.approx([.25, 25])


def test_editor_view_modes_preserve_geometry_and_limits(editor, qtbot):
    before = deepcopy(editor.draft)
    editor.ax.set_xlim(.25, 1.75)
    limits = editor.ax.get_xlim(), editor.ax.get_ylim()
    editor.canvas_view.mode_combo.setCurrentIndex(1)
    assert editor.canvas.width() == 8 * 96
    assert editor.canvas.height() == 6 * 96
    editor.canvas_view.mode_combo.setCurrentIndex(0)
    qtbot.waitUntil(lambda: editor.canvas.size() == editor.canvas_view.scroll.viewport().size())
    assert (editor.ax.get_xlim(), editor.ax.get_ylim()) == limits
    assert editor.draft == before


def test_passport_and_coverage_domain_are_undoable(editor, monkeypatch):
    monkeypatch.setattr(editor, 'edit_fields', lambda *_: ['Тестовый источник', 'Описание', 'Ограничение'])
    editor.region_passport.click()
    assert editor.current_entry()['passport']['source'] == 'Тестовый источник'
    editor.undo_stack.undo()
    assert 'passport' not in editor.current_entry()
    editor.undo_stack.redo()

    monkeypatch.setattr(editor, 'edit_fields', lambda *_: ['0', '5', '0', '500'])
    editor.edit_coverage_domain()
    editor.show_coverage.setChecked(True)
    editor.fit_view()
    assert editor.validation['coverage']['gaps'].area == pytest.approx(500)
    assert '80 %' in editor.diagnostics.text()
    assert any(isinstance(collection, LineCollection) for collection in editor.ax.collections)
    assert editor.ax.get_xlim()[1] > 5
    editor.undo_stack.undo()
    assert 'coverage' not in editor.validation


def test_invalid_draft_cannot_be_applied_and_undo_restores(editor):
    entry = editor.current_entry()
    invalid = replace_vertices(entry, editor.selected_path, [[0, 0], [2, 500], [0, 500], [2, 0]])
    editor.push(replace_object(editor.draft, "regions", invalid), "Self intersection")
    assert editor.validation["errors"]
    assert not editor.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    editor.apply()
    assert editor.result() != QDialog.DialogCode.Accepted
    editor.undo_stack.undo()
    assert not editor.validation["errors"]
    editor.apply()
    assert editor.edited_scheme["revision"] == "2"


def test_new_scheme_draw_polygon_and_line(qtbot):
    scheme = new_scheme("TOC", "%wt", "HI", "mg HC/g TOC")
    scheme["source"] = "Synthetic test"
    editor = GeometryEditor(scheme, new=True)
    qtbot.addWidget(editor)
    editor.canvas.draw()
    editor.mode.setCurrentIndex(editor.mode.findData("polygon"))
    draw(editor, [(.1, .1), (.8, .1), (.8, .8), (.1, .8)])
    assert len(editor.draft["regions"]) == 1
    editor.mode.setCurrentIndex(editor.mode.findData("line"))
    draw(editor, [(.1, .2), (.6, .2)])
    assert len(editor.draft["lines"]) == 1
    editor.threshold_axis.setCurrentIndex(editor.threshold_axis.findData('y'))
    editor.threshold_less.setText('Низкий')
    editor.threshold_greater.setText('Высокий')
    editor.apply_threshold.click()
    assert editor.draft['lines'][0]['threshold']['axis'] == 'y'
    editor.undo_stack.undo()
    assert 'threshold' not in editor.draft['lines'][0]
    editor.undo_stack.redo()
    editor.apply()
    assert editor.result() == QDialog.DialogCode.Accepted
    assert editor.edited_scheme["revision"] == "1"
    assert editor.edited_scheme['lines'][0]['threshold']['greater'] == 'Высокий'
    assert scheme["regions"] == []


def test_drawing_hole_changes_preview_and_cancel_does_not_apply(editor, scheme):
    editor.mode.setCurrentIndex(editor.mode.findData("hole"))
    draw(editor, [(.5, 50), (1.5, 50), (1.5, 150), (.5, 150)])
    assert len(editor.draft["regions"][0]["geometry"]["coordinates"]) == 2
    assert next(r for r in editor.preview_results if r.row_id == "inside").status == "outside"
    editor.reject()
    assert editor.edited_scheme is None
    assert len(scheme["regions"][0]["geometry"]["coordinates"]) == 1


def test_vertex_insert_delete_and_nan_rejection(editor):
    editor.vertex_table.selectRow(0)
    editor.insert_vertex()
    assert editor.vertex_table.rowCount() == 5
    editor.delete_vertex()
    assert editor.vertex_table.rowCount() == 4
    before = deepcopy(editor.current_entry())
    editor.vertex_table.item(0, 0).setText("NaN")
    assert editor.current_entry() == before
    assert "NaN" in editor.diagnostics.text()


def test_editor_uses_analysis_coordinates_for_log_axes(qtbot, scheme):
    scheme["x"]["transform"] = "log10"
    editor = GeometryEditor(scheme, samples=[InterpretationInput("log", 10, 100)])
    qtbot.addWidget(editor)
    assert editor.preview_results[0].status == "classified"
    samples = editor.ax.collections[-1].get_offsets()
    assert samples[0].tolist() == pytest.approx([1, 100])
    assert "analysis" in editor.ax.get_xlabel()


def test_qt_mouse_events_move_vertex(editor, qtbot):
    def point(x, y):
        px, py = editor.ax.transData.transform((x, y))
        return QPoint(round(px), round(editor.canvas.height() - py))
    start, end = point(0, 0), point(.3, 30)
    qtbot.mousePress(editor.canvas, Qt.MouseButton.LeftButton, pos=start)
    qtbot.mouseMove(editor.canvas, pos=end)
    qtbot.mouseRelease(editor.canvas, Qt.MouseButton.LeftButton, pos=end)
    value = editor.draft["regions"][0]["geometry"]["coordinates"][0][0]
    assert value[0] == pytest.approx(.3, abs=.02)
    assert value[1] == pytest.approx(30, abs=2)


def test_delegate_keyboard_edit_commits_coordinate_without_closing(editor, qtbot):
    index = editor.vertex_table.model().index(0, 0)
    editor.vertex_table.setCurrentIndex(index)
    editor.vertex_table.edit(index)
    line = editor.vertex_table.findChild(QLineEdit)
    assert line is not None
    qtbot.keyClick(line, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClicks(line, "0.375")
    qtbot.keyClick(line, Qt.Key.Key_Return)
    qtbot.waitUntil(lambda: editor.current_entry()["geometry"]["coordinates"][0][0][0] == .375, timeout=3000)
    assert editor.edited_scheme is None
    assert editor.isVisible()
