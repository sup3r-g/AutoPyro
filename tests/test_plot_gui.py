import json
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from autopyro.core import Project
from autopyro.window import MainWindow
from autopyro.io import build_project


def ready(qtbot, dialog):
    qtbot.waitUntil(lambda: dialog.result is not None and not dialog.timer.isActive(), timeout=10000)


@pytest.fixture
def window(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.demo()
    window.may_replace = lambda: True  # GUI teardown must not open a save confirmation.
    yield window
    if window.plot_window:
        window.plot_window.close()
    window.dirty = False


def test_local_plot_filter_leaves_table_scope_and_exports_same_points(window, qtbot, monkeypatch, tmp_path):
    from autopyro.ui.dialogs.filter import FilterDialog

    window.show_plots()
    dialog = window.plot_window
    ready(qtbot, dialog)
    well, depth, s2 = [window.model.project.columns[i].id for i in (0, 2, 4)]
    dialog.data.load({'x_id': depth, 'y_id': s2})
    dialog.schedule()
    ready(qtbot, dialog)
    table_ids = window.row_ids()

    def accept(editor):
        editor.rules = [{'combine': 'OR', 'negate': True, 'rules': [
            {'column': well, 'operator': 'eq', 'value': 'A'},
            {'column': depth, 'operator': 'lt', 'value': 0},
        ]}]
        return editor.DialogCode.Accepted

    monkeypatch.setattr(FilterDialog, 'exec', accept)
    dialog.local_filter_button.click()
    ready(qtbot, dialog)
    wanted = {row.id for row in window.model.project.rows if row.values[well] == 'B'}
    shown = {rid for artist in dialog.result['artists'] for rid in artist._autopyro_row_ids}
    assert shown == wanted
    assert window.row_ids() == table_ids
    assert window.model.project.view['plot_current']['filter_rules'] == dialog.spec.filter_rules
    path, manifest = tmp_path / 'local.svg', tmp_path / 'manifest.json'
    dialog.start_export([('Local', path, dialog.spec)], manifest)
    qtbot.waitUntil(lambda: not dialog.export_busy(), timeout=30000)
    payload = json.loads(manifest.read_text(encoding='utf-8'))
    assert payload['plots'][0]['plotted'][depth] == len(shown)
    assert payload['plots'][0]['spec']['filter_rules'] == dialog.spec.filter_rules


def test_live_plot_style_limits_and_filter(window, qtbot):
    window.show_plots()
    dialog = window.plot_window
    ready(qtbot, dialog)
    well, depth, s2 = [window.model.project.columns[i].id for i in (0, 2, 4)]
    dialog.data.load({"x_id": depth, "y_id": s2})
    dialog.style.load({"color_by": well, "shape_by": well, "size_by": depth})
    dialog.x_form.load({"minimum": 950, "maximum": 1100, "label": "Глубина", "minor_grid": True})
    dialog.schedule()
    ready(qtbot, dialog)
    assert dialog.figure.axes[0].get_xlabel() == "Глубина"
    assert dialog.figure.axes[0].get_xlim() == (950, 1100)
    assert dialog.result["plotted"][depth] == 4
    originals = dialog.data.fields["include_original"][0]
    originals.setChecked(False)
    ready(qtbot, dialog)
    assert dialog.result["plotted"][depth] == 0
    assert window.model.project.view["plot_current"]["include_original"] is False
    originals.setChecked(True)
    ready(qtbot, dialog)
    assert dialog.result["plotted"][depth] == 4
    dialog.layout_form.fields["show_result_type"][0].setChecked(True)
    ready(qtbot, dialog)
    assert dialog.figure.axes[0].get_xlabel() == "Глубина · исходные данные"
    assert window.model.project.view["plot_current"]["show_result_type"] is True
    assert dialog.category_table.rowCount() == 4  # two colors, two shapes
    category = next(i for i in range(dialog.category_table.rowCount())
                    if dialog.category_table.item(i, 0).text() == "Цвет: B")
    dialog.category_table.item(category, 0).setCheckState(Qt.CheckState.Unchecked)
    ready(qtbot, dialog)
    assert dialog.result["plotted"][depth] == 2
    assert len(window.row_ids()) == 5
    assert dialog.category_table.item(category, 0).checkState() == Qt.CheckState.Unchecked
    dialog.category_table.item(category, 0).setCheckState(Qt.CheckState.Checked)
    ready(qtbot, dialog)
    assert dialog.result["plotted"][depth] == 4
    window.apply_filter([{"column": well, "operator": "eq", "value": "A"}], "AND")
    ready(qtbot, dialog)
    assert dialog.result["plotted"][depth] == 2
    window.model.setData(window.model.index(1, 4), "4")
    ready(qtbot, dialog)
    assert dialog.result["plotted"][depth] == 3


def test_pick_selection_and_empty_selected_scope(window, qtbot):
    window.show_plots()
    dialog = window.plot_window
    ready(qtbot, dialog)
    artist = dialog.result["artists"][0]
    wanted_id = artist._autopyro_row_ids[0]
    dialog.on_pick(SimpleNamespace(artist=artist, ind=[0]))
    assert dialog.selection_ids() == {wanted_id}
    dialog.data.load({"scope": "selected"})
    dialog.schedule()
    ready(qtbot, dialog)
    assert len(dialog.rendered_ids) == 1
    window.table.selectionModel().clearSelection()
    ready(qtbot, dialog)
    assert dialog.rendered_ids == []
    assert sum(dialog.result["plotted"].values()) == 0


def test_export_visible_limits_keeps_saved_axis_settings(window, qtbot, tmp_path, monkeypatch):
    window.show_plots()
    plot = window.plot_window
    ready(qtbot, plot)
    ax = plot.result['axes'][0]
    ax.set_xlim(1007, 1027)
    ax.set_ylim(1, 7)
    saved = plot.spec.to_dict()
    path = tmp_path / 'visible.svg'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), 'SVG (*.svg)'))
    capture = plot.start_export
    captured = []

    def export(descriptions):
        captured.extend(spec for _, _, spec in descriptions)
        capture(descriptions)

    monkeypatch.setattr(plot, 'start_export', export)
    plot.export_visible_limits()
    qtbot.waitUntil(lambda: not plot.export_busy(), timeout=30000)
    spec, = captured
    assert (spec.x_axis.minimum, spec.x_axis.maximum) == (1007, 1027)
    assert (spec.y_axis.minimum, spec.y_axis.maximum) == (1, 7)
    assert path.is_file()
    assert plot.spec.to_dict() == saved


def test_plot_size_units_preserve_physical_export_size(window, qtbot, tmp_path):
    from PIL import Image
    from autopyro.visualization.plotting import save_plot

    window.show_plots()
    dialog = window.plot_window
    ready(qtbot, dialog)
    dialog.layout_form.load({"width_mm": 181.23, "height_mm": 130.09, "dpi": 100})
    for unit in ("cm", "in", "mm"):
        dialog.size_unit.setCurrentIndex(dialog.size_unit.findText(unit))
        spec = dialog.read_controls()
        assert (spec.width_mm, spec.height_mm) == (181.23, 130.09)
    dialog.size_unit.setCurrentIndex(dialog.size_unit.findText("in"))
    dialog.layout_form.fields["width_mm"][0].setValue(5)
    dialog.layout_form.fields["height_mm"][0].setValue(4)
    spec = dialog.read_controls()
    assert (spec.width_mm, spec.height_mm) == (127, 101.6)
    path = tmp_path / "size.png"
    save_plot(path, window.model.project, window.row_ids(), spec)
    with Image.open(path) as image:
        assert image.size == (500, 400)


def test_plot_spec_persists_and_export_uses_spec(window, qtbot, tmp_path, monkeypatch):
    window.show_plots()
    dialog = window.plot_window
    ready(qtbot, dialog)
    dialog.layout_form.load({"title": "TOC and S2", "width_mm": 101.6, "height_mm": 76.2, "dpi": 100})
    dialog.schedule()
    ready(qtbot, dialog)
    path = tmp_path / "plot.svg"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), "SVG (*.svg)"))
    dialog.export_plot()
    qtbot.waitUntil(lambda: not dialog.export_busy(), timeout=30000)
    assert path.exists(), dialog.status.text()
    assert "TOC and S2" in path.read_text(encoding="utf-8")
    path = tmp_path / "plot.json"
    dialog.export_spec()
    assert json.loads(path.read_text(encoding="utf-8"))["spec"]["width_mm"] == 101.6
    window.path = tmp_path / "test.apyro"
    assert window.save()
    window.set_project(Project.load(window.path), window.path)
    window.show_plots()
    ready(qtbot, window.plot_window)
    assert window.plot_window.spec.title == "TOC and S2"


def test_depth_planchette_track_settings(window, qtbot, tmp_path):
    window.show_plots()
    dialog = window.plot_window
    ready(qtbot, dialog)
    s2, md = window.model.project.columns[4].id, window.model.project.columns[2].id
    dialog.data.load({"kind": "multi_depth", "y_id": md})
    dialog.tracks.item(4).setCheckState(Qt.CheckState.Checked)
    dialog.y_form.load({"inverted": True})
    dialog.schedule()
    ready(qtbot, dialog)
    assert len(dialog.result["axes"]) == 1
    dialog.x_target.setCurrentIndex(dialog.x_target.findData(s2))
    dialog.x_form.load({"major_step": 1, "label": "S2 (log)", "minimum": .1, "maximum": 10})
    scale = dialog.x_form.fields['scale'][0]
    scale.setCurrentIndex(scale.findData('log'))
    dialog.schedule()
    ready(qtbot, dialog)
    assert dialog.result["axes"][0].get_xscale() == "log"
    assert dialog.result["axes"][0].get_xlabel() == "S2 (log)"
    assert dialog.result["axes"][0].yaxis.get_inverted()
    assert dialog.spec.track_axes[s2].major_step is None
    assert not dialog.x_form.fields['major_step'][0].isEnabled()
    assert not dialog.x_form.fields['number_format'][0].isEnabled()
    path = tmp_path / 'log-track.svg'
    dialog.start_export([('Log track', path, dialog.spec)])
    qtbot.waitUntil(lambda: not dialog.export_busy(), timeout=30000)
    assert 'S2 (log)' in path.read_text(encoding='utf-8')
    scale.setCurrentIndex(scale.findData('linear'))
    ready(qtbot, dialog)
    assert dialog.result['axes'][0].get_xscale() == 'linear'
    assert dialog.x_form.fields['major_step'][0].isEnabled()
    assert dialog.x_form.fields['number_format'][0].isEnabled()


def test_rectangle_selection_uses_ids_after_sort(window, qtbot):
    window.show_plots()
    dialog = window.plot_window
    ready(qtbot, dialog)
    window.table.sortByColumn(2, Qt.SortOrder.DescendingOrder)
    ready(qtbot, dialog)
    ax = dialog.result["axes"][0]
    # Initial axes: MD and S2. Only the A sample at 1020, 6 belongs to this box.
    dialog.on_rectangle(SimpleNamespace(xdata=1019, ydata=5.5, inaxes=ax),
                        SimpleNamespace(xdata=1021, ydata=6.5, inaxes=ax))
    assert dialog.selection_ids() == {window.model.project.rows[2].id}


def test_plot_configuration_undo_redo(window, qtbot):
    window.show_plots()
    dialog = window.plot_window
    ready(qtbot, dialog)
    previous_title = dialog.spec.title
    dialog.layout_form.load({"title": "New title"})
    dialog.schedule()
    ready(qtbot, dialog)
    assert dialog.undo_stack.canUndo()
    dialog.undo_stack.undo()
    ready(qtbot, dialog)
    assert dialog.spec.title == previous_title
    dialog.undo_stack.redo()
    ready(qtbot, dialog)
    assert dialog.spec.title == "New title"


def test_json_template_remaps_columns_by_name_and_unit(window, qtbot, tmp_path, monkeypatch):
    window.show_plots()
    dialog = window.plot_window
    ready(qtbot, dialog)
    path = tmp_path / "template.json"
    dialog.spec.filter_rules = [{'column': window.model.project.columns[0].id, 'operator': 'eq', 'value': 'B'}]
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), "JSON (*.json)"))
    dialog.export_spec()
    source = window.model.project
    headers = [c.name for c in source.columns]
    rows = [[r.values[c.id] for c in source.columns] for r in source.rows]
    selections = [(i, c.name, c.kind, c.unit) for i, c in enumerate(source.columns)]
    replacement = build_project(headers, rows, list(range(1, len(rows) + 1)), selections, {})
    window.set_project(replacement)
    window.show_plots()
    dialog = window.plot_window
    ready(qtbot, dialog)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(path), "JSON (*.json)"))
    dialog.import_spec()
    ready(qtbot, dialog)
    assert dialog.spec.x_id == replacement.columns[2].id
    assert dialog.spec.y_id == replacement.columns[4].id
    assert dialog.spec.filter_rules[0]['column'] == replacement.columns[0].id
    assert dialog.result["plotted"][dialog.spec.x_id] == 2


def test_batch_export_preserves_existing_files_and_writes_manifest(window, qtbot, tmp_path, monkeypatch):
    window.show_plots()
    dialog = window.plot_window
    ready(qtbot, dialog)
    window.model.project.view["plots"] = {"First": dialog.spec.to_dict(), "Second": dialog.spec.to_dict()}
    existing = tmp_path / "plot-001-First.png"
    existing.write_bytes(b"existing")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path))
    monkeypatch.setattr(QInputDialog, "getItem", lambda *args, **kwargs: ("png", True))
    monkeypatch.setattr(QMessageBox, "information", lambda *args: None)
    dialog.batch_export()
    qtbot.waitUntil(lambda: not dialog.export_busy(), timeout=30000)
    manifest = json.loads((tmp_path / "plots-manifest.json").read_text(encoding="utf-8"))
    assert existing.read_bytes() == b"existing"
    assert [p["name"] for p in manifest["plots"]] == ["First", "Second"]
    assert manifest["plots"][0]["file"] == "plot-001-First-2.png"
    assert all((tmp_path / p["file"]).stat().st_size > 100 for p in manifest["plots"])
    assert manifest["errors"] == []


@pytest.mark.parametrize('stage', ['selection', 'directory', 'format', 'panels'])
def test_named_export_stops_when_project_changes_in_a_dialog(window, qtbot, monkeypatch, tmp_path, stage):
    from copy import deepcopy
    from autopyro.ui.dialogs.plot_export_selection import PlotExportSelectionDialog

    window.show_plots()
    plot = window.plot_window
    ready(qtbot, plot)
    project = window.model.project
    project.view['plots'] = {'First': plot.spec.to_dict(), 'Second': plot.spec.to_dict()}
    replacement = deepcopy(project)

    def replace_project():
        window.set_project(replacement)
        assert window.model.project is replacement

    def choose_directory(*args):
        if stage == 'directory':
            replace_project()
        return str(tmp_path)

    def choose_format(*args, **kwargs):
        assert stage != 'directory', 'Export continued after the project changed'
        replace_project()
        return 'svg', True

    def select_plots(dialog):
        for index in range(dialog.choices.count()):
            dialog.choices.item(index).setCheckState(Qt.CheckState.Checked)
        replace_project()
        return dialog.DialogCode.Accepted

    def choose_path(*args):
        replace_project()
        return str(tmp_path / 'panels.svg'), 'SVG (*.svg)'

    def start_export(*args, **kwargs):
        pytest.fail('Export started using a replaced project')

    monkeypatch.setattr(plot, 'start_export', start_export)
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', choose_directory)
    monkeypatch.setattr(QInputDialog, 'getItem', choose_format)
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', choose_path)
    monkeypatch.setattr(PlotExportSelectionDialog, 'exec', select_plots)
    if stage == 'selection':
        plot.export_selected_plots()
    elif stage == 'panels':
        plot.export_panels(['First', 'Second'], columns=2)
    else:
        plot.batch_export()
    assert window.model.project is replacement
    assert not list(tmp_path.iterdir())


def test_saved_scope_and_matrix_without_x(window, monkeypatch):
    from autopyro.visualization.plotting import PlotSpec

    project = window.model.project
    x, y = [column.id for column in project.columns[2:4]]
    project.view['plot_current'] = PlotSpec(scope='selected', x_id=x, y_id=y).to_dict()
    window.analysis_scope.set_mode('all')
    window.show_plots()
    dialog = window.plot_window
    dialog.redraw()
    assert dialog.rendered_ids == []
    assert dialog.scope.currentData() == 'selected'
    assert window.analysis_scope.mode == 'all'

    project.view['plots'] = {'matrix': PlotSpec(kind='mpl:imshow', scope='all',
                                               track_ids=[x, y], text_numbers=True).to_dict()}
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args, **kwargs: ('matrix', True))
    dialog.load_named()
    dialog.redraw()
    assert dialog.result is not None, dialog.status.text()
    assert dialog.spec.x_id == ''
    assert 'Выбранные показатели' in dialog.status.text()
    assert window.analysis_scope.mode == 'all'

    dialog.schedule()
    assert not dialog.frame_notice.isHidden()
    assert not dialog.export_button.isEnabled()
    dialog.cancel_preparation()
    assert dialog.retry_plot_action.isEnabled()
    dialog.redraw()
    assert dialog.frame_notice.isHidden()


def test_replacing_named_plot_requires_confirmation(window, monkeypatch):
    window.show_plots()
    dialog = window.plot_window
    dialog.redraw()
    plots = window.model.project.view['plots'] = {'Existing': {'title': 'Old'}}
    monkeypatch.setattr(QInputDialog, 'getText', lambda *args: ('Existing', True))
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.No)
    dialog.save_named()
    assert plots['Existing'] == {'title': 'Old'}
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.Yes)
    dialog.save_named()
    assert plots['Existing']['scope'] == window.analysis_scope.mode



@pytest.mark.parametrize('stage', ['load', 'name', 'replace'])
def test_named_plot_dialog_stops_after_project_replacement(window, qtbot, monkeypatch, stage):
    from copy import deepcopy

    window.show_plots()
    plot = window.plot_window
    ready(qtbot, plot)
    project = window.model.project
    saved = plot.spec.to_dict() | {'title': 'Original named plot'}
    project.view['plots'] = {'Existing': deepcopy(saved)}
    replacement = deepcopy(project)

    def change_project():
        window.set_project(replacement)

    def choose_name(*args):
        if stage == 'name':
            change_project()
        return 'Existing', True

    def confirm_replacement(*args):
        assert stage == 'replace'
        change_project()
        return QMessageBox.StandardButton.Yes

    def choose_plot(*args, **kwargs):
        change_project()
        return 'Existing', True

    def apply_stale_plot():
        pytest.fail('A graph from the replaced project was applied')

    monkeypatch.setattr(QInputDialog, 'getText', choose_name)
    monkeypatch.setattr(QInputDialog, 'getItem', choose_plot)
    monkeypatch.setattr(QMessageBox, 'question', confirm_replacement)
    monkeypatch.setattr(plot, 'load_controls', apply_stale_plot)
    if stage == 'load':
        plot.load_named()
    else:
        plot.save_named()
    assert window.model.project is replacement
    assert project.view['plots'] == replacement.view['plots'] == {'Existing': saved}
    assert not window.dirty


def test_archived_fit_layer_uses_saved_curve_in_preview_and_batch_export(window, qtbot, tmp_path, monkeypatch):
    project = window.model.project
    x, y = project.columns[2], project.columns[4]
    project.view['curve_fit_results'] = [{
        'id': 'archive-id', 'name': 'Saved fit', 'settings': {'x': x.id, 'y': y.id, 'method': 'ols'},
        'axis_units': [x.unit, y.unit], 'grid': [990, 1020], 'curve': [1, 7],
        'source_digest': 'archived-input-digest', 'source_digest_version': 2, 'stale': True,
    }]
    window.show_plots()
    dialog = window.plot_window
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args, **kwargs: ('Saved fit · archive-id', True))
    dialog.add_archived_curve()
    qtbot.waitUntil(lambda: dialog.result is not None, timeout=30000)
    line = next(line for line in dialog.figure.axes[0].lines if line.get_gid() == 'archived-curve')
    assert line.get_ydata().tolist() == [1, 7]
    assert 'OLS' in line.get_label() and 'устарел' in line.get_label()
    assert dialog.spec.curve_archive_id == 'archive-id'

    path, manifest = tmp_path / 'fit.svg', tmp_path / 'manifest.json'
    dialog.start_export([('Fit', path, dialog.spec)], manifest)
    qtbot.waitUntil(lambda: not dialog.export_busy(), timeout=30000)
    assert 'id="archived-curve"' in path.read_text(encoding='utf-8')
    entry = json.loads(manifest.read_text(encoding='utf-8'))['plots'][0]['curve_archive']
    assert entry['settings']['method'] == 'ols'
    assert entry['source_digest'] == 'archived-input-digest'
    dialog.remove_archived_curve()
    dialog.redraw()
    assert not any(line.get_gid() == 'archived-curve' for line in dialog.figure.axes[0].lines)



def test_positioned_labels_validate_and_survive_undo_and_export(window, qtbot, monkeypatch, tmp_path):
    from matplotlib.backend_bases import MouseButton
    from PySide6.QtWidgets import QMenu
    from autopyro.ui.dialogs.plot_annotations import PlotAnnotationsDialog

    window.show_plots()
    plot = window.plot_window
    ready(qtbot, plot)

    def edit(dialog):
        row = dialog.table.rowCount() - 1
        dialog.table.item(row, 0).setText('Archive marker')
        dialog.table.item(row, 2).setText('2')
        dialog.accept()
        assert dialog.error.text()
        dialog.table.item(row, 2).setText('0,3')
        dialog.add_row()
        shape = dialog.table.cellWidget(row + 1, 5)
        shape.setCurrentIndex(shape.findData('arrow'))
        dialog.table.item(row + 1, 6).setText('0.7')
        dialog.table.item(row + 1, 7).setText('0.8')
        dialog.accept()
        return dialog.result()

    monkeypatch.setattr(PlotAnnotationsDialog, 'exec', edit)
    ax = plot.figure.axes[0]
    x, y = ax.transAxes.transform((.3, .65))
    plot.annotation_context_menu(SimpleNamespace(button=MouseButton.RIGHT, inaxes=ax, x=x, y=y))
    menu = plot.canvas.findChild(QMenu, 'plotAnnotationMenu')
    menu.actions()[0].trigger()
    menu.hide()
    ready(qtbot, plot)
    assert plot.spec.annotations[0]['x'] == .3
    assert plot.spec.annotations[1]['kind'] == 'arrow'
    assert next(t for t in plot.figure.axes[0].texts if t.get_gid() == 'plot-annotation-1').get_position() == (.3, .65)
    assert window.model.project.view['plot_current']['annotations'] == plot.spec.annotations
    plot.undo_stack.undo()
    ready(qtbot, plot)
    assert plot.spec.annotations == []
    plot.undo_stack.redo()
    ready(qtbot, plot)

    path = tmp_path / 'label.svg'
    plot.start_export([('Label', path, plot.spec)])
    qtbot.waitUntil(lambda: not plot.export_busy(), timeout=30000)
    svg = path.read_text(encoding='utf-8')
    assert 'id="plot-annotation-1"' in svg
    assert 'id="plot-annotation-2"' in svg



@pytest.mark.parametrize('incompatible', [
    {'kind': 'arrow', 'coordinates': 'axes', 'x2': .8, 'y2': .8},
    {'coordinates': 'data'},
])
def test_3d_annotation_editor_rejects_invisible_objects(window, qtbot, monkeypatch, tmp_path, incompatible):
    from autopyro.ui.dialogs.plot_annotations import PlotAnnotationsDialog

    window.show_plots()
    plot = window.plot_window
    ready(qtbot, plot)
    plot.spec.kind = 'mpl3d:scatter'
    plot.spec.z_id = window.model.project.columns[2].id
    plot.spec.annotations = [{'text': 'Visible 3D label', 'x': .3, 'y': .7} | incompatible]
    plot.load_controls()
    plot.redraw()
    assert plot.result is not None, plot.status.text()

    def edit(dialog):
        mode, kind = dialog.table.cellWidget(0, 1), dialog.table.cellWidget(0, 5)
        assert not mode.model().item(mode.findData('data')).isEnabled()
        assert all(not kind.model().item(index).isEnabled() for index in range(1, kind.count()))
        dialog.accept()
        assert dialog.result() == 0 and '3D' in dialog.error.text()
        mode.setCurrentIndex(mode.findData('axes'))
        kind.setCurrentIndex(kind.findData('text'))
        dialog.accept()
        return dialog.result()

    monkeypatch.setattr(PlotAnnotationsDialog, 'exec', edit)
    plot.edit_annotations()
    ready(qtbot, plot)
    assert plot.spec.annotations == [{'text': 'Visible 3D label', 'coordinates': 'axes',
                                       'x': .3, 'y': .7, 'track_id': ''}]
    assert any(text.get_text() == 'Visible 3D label' for text in plot.figure.axes[0].texts)
    path = tmp_path / 'annotation-3d.svg'
    plot.start_export([('3D annotation', path, plot.spec)])
    qtbot.waitUntil(lambda: not plot.export_busy(), timeout=30000)
    assert 'Visible 3D label' in path.read_text(encoding='utf-8')


@pytest.mark.parametrize('coordinate', ['x', 'x2', 'y', 'track'])
def test_annotation_editor_checks_log_coordinates_on_selected_track(window, qtbot, monkeypatch, tmp_path, coordinate):
    from autopyro.visualization.plotting import AxisSpec, PlotSpec
    from autopyro.ui.dialogs.plot_annotations import PlotAnnotationsDialog

    window.show_plots()
    plot = window.plot_window
    ready(qtbot, plot)
    md, toc, s2 = [window.model.project.columns[index].id for index in (2, 3, 4)]
    annotation = {'kind': 'arrow', 'text': 'Checked arrow', 'coordinates': 'data',
                  'x': 1, 'x2': 2, 'y': 1000, 'y2': 1010, 'track_id': s2}
    invalid_key = 'x' if coordinate == 'track' else coordinate
    annotation[invalid_key] = 0
    plot.spec = PlotSpec(kind='multi_depth', track_ids=[toc, s2], x_id=toc, y_id=md,
                         track_axes={toc: AxisSpec(minimum=-1, maximum=4), s2: AxisSpec(scale='log')},
                         y_axis=AxisSpec(scale='log' if coordinate == 'y' else 'linear'),
                         annotations=[annotation])
    plot.load_controls()
    plot.redraw()
    assert plot.result is not None, plot.status.text()

    def edit(dialog):
        dialog.accept()
        assert dialog.result() == 0
        assert 'логарифмической' in dialog.error.text()
        if coordinate == 'track':
            target = dialog.table.cellWidget(0, 4)
            target.setCurrentIndex(target.findData(toc))
        else:
            column = {'x': 2, 'x2': 6, 'y': 3}[coordinate]
            assert dialog.table.currentColumn() == column
            dialog.table.item(0, column).setText('1' if coordinate != 'y' else '1000')
        dialog.accept()
        return dialog.result()

    monkeypatch.setattr(PlotAnnotationsDialog, 'exec', edit)
    plot.edit_annotations()
    ready(qtbot, plot)
    target_axis = plot.figure.axes[0 if coordinate == 'track' else 1]
    assert any(artist.get_gid() == 'plot-annotation-1' for artist in target_axis.get_children())
    path = tmp_path / 'checked-arrow.svg'
    plot.start_export([('Checked arrow', path, plot.spec)])
    qtbot.waitUntil(lambda: not plot.export_busy(), timeout=30000)
    assert 'Checked arrow' in path.read_text(encoding='utf-8')


@pytest.mark.parametrize('kind', ['multi_depth', 'scatter'])
def test_annotation_editor_requires_rebinding_removed_track(window, qtbot, monkeypatch, tmp_path, kind):
    from autopyro.visualization.plotting import PlotSpec
    from autopyro.ui.dialogs.plot_annotations import PlotAnnotationsDialog

    window.show_plots()
    plot = window.plot_window
    ready(qtbot, plot)
    md, toc, s2 = [window.model.project.columns[index].id for index in (2, 3, 4)]
    annotation = {'text': 'Recovered label', 'coordinates': 'axes', 'x': .2, 'y': .8, 'track_id': toc}
    plot.spec = PlotSpec(kind=kind, x_id=s2, y_id=md, track_ids=[s2], annotations=[annotation])
    plot.load_controls()
    plot.redraw()
    assert plot.result is not None, plot.status.text()
    assert not any(text.get_text() == 'Recovered label' for text in plot.figure.axes[0].texts)

    def edit(dialog):
        target = dialog.table.cellWidget(0, 4)
        assert target.currentData() == toc
        assert not target.model().item(target.currentIndex()).isEnabled()
        dialog.accept()
        assert dialog.result() == 0 and 'недоступен' in dialog.error.text()
        assert dialog.table.currentColumn() == 4
        assert plot.spec.annotations == [annotation]
        target.setCurrentIndex(target.findData(s2 if kind == 'multi_depth' else ''))
        dialog.accept()
        return dialog.result()

    monkeypatch.setattr(PlotAnnotationsDialog, 'exec', edit)
    plot.edit_annotations()
    ready(qtbot, plot)
    assert any(text.get_text() == 'Recovered label' for text in plot.figure.axes[0].texts)
    path = tmp_path / 'rebound-label.svg'
    plot.start_export([('Recovered label', path, plot.spec)])
    qtbot.waitUntil(lambda: not plot.export_busy(), timeout=30000)
    assert 'Recovered label' in path.read_text(encoding='utf-8')


def test_snapshot_replays_identical_export_without_source_project(window, qtbot, tmp_path, monkeypatch):
    from PIL import Image

    window.show_plots()
    plot = window.plot_window
    project = window.model.project
    color = project.columns[0].id
    plot.style.load({'color_by': color})
    plot.data.load({'scope': 'filtered'})
    window.apply_filter([{'column': color, 'operator': 'eq', 'value': 'A'}], 'AND')
    plot.layout_form.load({'title': 'Portable graph'})
    annotation = {'kind': 'rectangle', 'coordinates': 'axes', 'x': .2, 'y': .2, 'x2': .4, 'y2': .4}
    plot.spec.annotations = [annotation]
    plot.schedule()
    qtbot.waitUntil(lambda: plot.result is not None, timeout=30000)

    original, snapshot, replay = (tmp_path / name for name in ('original.png', 'fit.aplot.json', 'replay.png'))
    plot.start_export([('Original', original, plot.spec)])
    qtbot.waitUntil(lambda: not plot.export_busy(), timeout=30000)
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(snapshot), 'JSON'))
    plot.captured_input = None
    plot.export_snapshot()
    qtbot.waitUntil(lambda: not plot.export_busy(), timeout=30000)
    payload = json.loads(snapshot.read_text(encoding='utf-8'))
    assert payload['ids'] == plot.rendered_ids
    assert payload['spec']['annotations'] == [annotation]
    assert set(payload['global_values'][color]) == set(project.row_map())
    assert len(payload['ids']) < len(project.rows)

    window.set_project(Project())
    window.show_plots()
    empty_plot = window.plot_window
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(snapshot), 'JSON'))
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(replay), 'PNG (*.png)'))
    empty_plot.replay_snapshot()
    qtbot.waitUntil(lambda: not empty_plot.export_busy(), timeout=30000)
    with Image.open(original) as expected, Image.open(replay) as actual:
        assert actual.size == expected.size
        assert actual.tobytes() == expected.tobytes()



@pytest.mark.parametrize('profile', [False, True])
def test_scientific_template_menu_and_manifest(window, qtbot, tmp_path, profile):
    project = build_project(['TOC', 'S2', 'MD'], [[1, 2, 1000], [2, 3, 1010]], [1, 2],
                            [(0, 'TOC', 'number', '%'), (1, 'S2', 'number', 'mg HC/g rock'),
                             (2, 'MD', 'number', 'm')], {})
    window.set_project(project)
    window.show_plots()
    plot = window.plot_window
    plot.use_template('TOC–MD' if profile else 'S2–TOC')
    qtbot.waitUntil(lambda: plot.result is not None, timeout=30000)
    expected_id = 'pyro.profile-toc-md.v1' if profile else 'pyro.s2-toc.v1'
    assert project.view['plot_current']['passport_id'] == expected_id
    assert plot.figure.axes[0].yaxis.get_inverted() == profile
    path = tmp_path / 'passport.svg'
    manifest = tmp_path / 'manifest.json'
    plot.start_export([('Passport', path, plot.spec)], manifest)
    qtbot.waitUntil(lambda: not plot.export_busy(), timeout=30000)
    entry = json.loads(manifest.read_text(encoding='utf-8'))['plots'][0]['passport']
    assert path.exists()
    assert entry['version'] == 1 and entry['axes']['x']['unit'] == '%wt'
    assert entry['source'] == 'SCIENTIFIC_METHODS.md'
    assert entry.get('depth_reference') == ('MD' if profile else None)



def test_md_tablet_template_sets_order_and_exports_passport(window, qtbot, tmp_path):
    from autopyro.visualization.plot_passports import TABLET_TITLE
    from autopyro.visualization.plotting import PlotSpec, render_plot

    names = ['MD', 'Tmax', 'S2', 'TOC', 'S1']
    units = ['m', '°C', 'mg HC/g rock', '%wt', 'mg HC/g rock']
    project = build_project(names, [[1000, 430, 6, 2, 1], [1020, 440, 7, 3, 2]], [1, 2],
                            [(i, name, 'number', units[i]) for i, name in enumerate(names)], {})
    window.set_project(project)
    window.show_plots()
    plot = window.plot_window
    plot.use_template(TABLET_TITLE)
    qtbot.waitUntil(lambda: plot.result is not None, timeout=30000)
    expected = ['TOC', 'S1', 'S2', 'Tmax']
    columns = {column.id: column.name for column in project.columns}
    assert [columns[cid] for cid in plot.spec.track_ids] == expected
    assert len(plot.figure.axes) == 4
    assert all(ax.yaxis.get_inverted() for ax in plot.figure.axes)

    stored = PlotSpec.from_dict(project.view['plot_current'])
    stored.track_ids.reverse()
    with pytest.raises(ValueError, match='TOC'):
        render_plot(project, list(project.row_map()), stored)

    path, manifest = tmp_path / 'tablet.svg', tmp_path / 'manifest.json'
    plot.start_export([('Tablet', path, plot.spec)], manifest)
    qtbot.waitUntil(lambda: not plot.export_busy(), timeout=30000)
    assert path.exists()
    passport = json.loads(manifest.read_text(encoding='utf-8'))['plots'][0]['passport']
    assert passport['id'] == 'pyro.tablet-basic-md.v1'
    assert passport['depth_reference'] == 'MD'
    assert [track['property'] for track in passport['tracks']] == expected



def test_interval_md_template_exports_both_depth_columns(window, qtbot, tmp_path):
    project = build_project(['MD_BOTTOM', 'TOC', 'MD_TOP'], [[1010, 2, 1000]], [1],
                            [(0, 'MD_BOTTOM', 'number', 'm'), (1, 'TOC', 'number', '%wt'),
                             (2, 'MD_TOP', 'number', 'm')], {})
    window.set_project(project)
    window.show_plots()
    plot = window.plot_window
    plot.style.load({'shape_by': project.columns[1].id})
    plot.data.load({'connect': True})
    plot.use_template('TOC–MD interval')
    qtbot.waitUntil(lambda: plot.result is not None, timeout=30000)
    assert plot.figure.axes[0].yaxis.get_inverted()
    assert plot.spec.depth_bottom_id == project.columns[0].id
    path, manifest = tmp_path / 'interval.svg', tmp_path / 'manifest.json'
    plot.start_export([('Interval', path, plot.spec)], manifest)
    qtbot.waitUntil(lambda: not plot.export_busy(), timeout=30000)
    assert path.exists()
    passport = json.loads(manifest.read_text(encoding='utf-8'))['plots'][0]['passport']
    assert passport['id'] == 'pyro.interval-toc-md.v1'
    assert passport['depth_reference'] == 'MD'
    assert passport['axes']['y']['property'] == 'MD_TOP'
    assert passport['axes']['bottom']['property'] == 'MD_BOTTOM'



def test_numeric_color_classes_controls_persist_in_export(window, qtbot, tmp_path):
    window.show_plots()
    plot = window.plot_window
    color = window.model.project.columns[2].id
    plot.style.load({'color_by': color, 'color_mode': 'numeric', 'color_scale': 'classes',
                     'color_class_edges': '990; 1010; 1030'})
    plot.schedule()
    qtbot.waitUntil(lambda: plot.result is not None, timeout=30000)
    assert not plot.style.fields['color_min'][0].isEnabled()
    assert list(plot.result['artists'][0].norm.boundaries) == [990, 1010, 1030]
    path, manifest = tmp_path / 'classes.svg', tmp_path / 'manifest.json'
    plot.start_export([('Classes', path, plot.spec)], manifest)
    qtbot.waitUntil(lambda: not plot.export_busy(), timeout=30000)
    assert path.exists()
    saved = json.loads(manifest.read_text(encoding='utf-8'))['plots'][0]['spec']
    assert saved['color_class_edges'] == '990; 1010; 1030'



@pytest.mark.parametrize('suffix', ['png', 'svg'])
def test_export_saved_plots_as_one_image(window, qtbot, monkeypatch, tmp_path, suffix):
    from PIL import Image

    window.show_plots()
    plot = window.plot_window
    plot.redraw()
    config = plot.spec.to_dict() | {'text_numbers': True}
    first = config | {'width_mm': 50.8, 'height_mm': 38.1, 'dpi': 100, 'background': '#f00000', 'title': 'First panel'}
    second = config | {'width_mm': 38.1, 'height_mm': 38.1, 'dpi': 200, 'background': '#0000f0', 'title': 'Second panel'}
    window.model.project.view['plots'] = {'Left': first, 'Right': second}
    path = tmp_path / f'panels.{suffix}'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path.with_suffix('')), f'{suffix.upper()} (*.{suffix})'))
    if suffix == 'svg':
        from PySide6.QtWidgets import QDialog, QLabel, QListWidget, QPushButton, QSpinBox, QTableWidget

        window.model.project.view['plots']['Spare'] = first

        def reorder(dialog):
            choices = dialog.findChild(QListWidget)
            columns = dialog.findChild(QSpinBox)
            preview = dialog.findChild(QTableWidget)
            assert not columns.isEnabled()
            for row in range(choices.count()):
                choices.item(row).setCheckState(Qt.CheckState.Checked)
            assert columns.isEnabled() and columns.maximum() == 3
            columns.setValue(3)
            choices.item(2).setCheckState(Qt.CheckState.Unchecked)
            assert columns.maximum() == columns.value() == 2
            assert (preview.rowCount(), preview.columnCount()) == (1, 2)
            assert preview.item(0, 1).text() == 'Right'
            columns.setValue(1)
            up, down = (next(button for button in dialog.findChildren(QPushButton) if button.text() == label)
                        for label in ('Вверх', 'Вниз'))
            choices.setCurrentRow(1)
            assert up.isEnabled() and down.isEnabled()
            qtbot.keyClick(up, Qt.Key.Key_Space)
            assert choices.currentItem().text() == 'Right'
            assert not up.isEnabled() and down.isEnabled()
            assert (preview.rowCount(), preview.columnCount()) == (2, 1)
            assert [preview.item(row, 0).text() for row in range(2)] == ['Right', 'Left']
            assert '38.1 × 38.1 mm' in preview.item(0, 0).toolTip()
            summary = next(label.text() for label in dialog.findChildren(QLabel) if label.text().startswith('SVG ·'))
            assert '50.8mm × 79.2mm' in summary
            assert '200 DPI (Right)' in summary
            return QDialog.DialogCode.Accepted

        monkeypatch.setattr(QDialog, 'exec', reorder)
        plot.export_selected_plots(panels=True)
    else:
        plot.export_panels(['Left', 'Right'], columns=2)
    qtbot.waitUntil(lambda: not plot.export_busy(), timeout=30000)
    if suffix == 'png':
        with Image.open(path) as image:
            assert image.size == (412, 150)
            assert image.info['dpi'] == pytest.approx((100, 100), abs=.01)
            assert image.getpixel((0, 0)) == (240, 0, 0, 255)
            assert image.getpixel((205, 0))[3] == 0
            assert image.getpixel((250, 0)) == (0, 0, 240, 255)
    else:
        import re
        from xml.etree import ElementTree as ET

        svg = '{http://www.w3.org/2000/svg}'
        root = ET.parse(path).getroot()
        assert (root.get('width'), root.get('height')) == ('50.8mm', '79.2mm')
        panels = root.findall(f'{svg}svg')
        assert [(panel.get('x'), panel.get('y')) for panel in panels] == [('6.35', '0'), ('0', '41.1')]
        for panel, title in zip(panels, ('Second panel', 'First panel'), strict=True):
            assert title in [node.text for node in panel.iter(f'{svg}text')]
            assert not list(panel.iter(f'{svg}image'))
        ids = [node.get('id') for node in root.iter() if 'id' in node.attrib]
        assert len(ids) == len(set(ids))
        refs = re.findall(r'url\(#([^)]*)\)|href="#([^"]*)"', path.read_text(encoding='utf-8'))
        assert refs and all((url or href) in ids for url, href in refs)
    assert list(tmp_path.iterdir()) == [path]
    assert window.model.project.view['plots']['Right']['dpi'] == 200



def test_batch_uses_current_visual_style_without_changing_saved_bindings(window, qtbot, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QCheckBox, QDialog, QListWidget

    window.show_plots()
    plot = window.plot_window
    plot.layout_form.load({'background': '#f0eeee', 'title_size': 19})
    plot.x_form.load({'label': 'Current label', 'label_size': 15, 'minimum': 1000})
    plot.redraw()
    spec = plot.spec.to_dict()
    saved = spec | {'title': 'Saved title', 'background': '#ffffff', 'dpi': 100,
                    'color_scale': 'log', 'color_min': 2, 'color_max': 100,
                    'x_axis': spec['x_axis'] | {'label': 'Saved axis', 'minimum': 900, 'label_size': 8}}
    plots = window.model.project.view['plots'] = {'Saved': saved}

    def choose_style(dialog):
        dialog.findChild(QListWidget).item(0).setCheckState(Qt.CheckState.Checked)
        dialog.findChild(QCheckBox).setChecked(True)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, 'exec', choose_style)
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *args: str(tmp_path))
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args, **kwargs: ('svg', True))
    plot.export_selected_plots()
    qtbot.waitUntil(lambda: not plot.export_busy(), timeout=30000)
    entry = json.loads((tmp_path / 'plots-manifest.json').read_text(encoding='utf-8'))['plots'][0]
    effective = entry['spec']
    assert effective['background'] == '#f0eeee' and effective['title_size'] == 19
    assert effective['x_axis']['label_size'] == 15
    assert effective['x_axis']['label'] == 'Saved axis' and effective['x_axis']['minimum'] == 900
    assert effective['title'] == 'Saved title' and effective['dpi'] == 100
    assert effective['color_scale'] == 'log' and effective['color_min'] == 2
    assert plots['Saved']['background'] == '#ffffff'
    assert plots['Saved']['x_axis']['label_size'] == 8
    assert '#f0eeee' in (tmp_path / entry['file']).read_text(encoding='utf-8')



def test_export_one_track_uses_its_axis_and_annotations(window, qtbot, monkeypatch, tmp_path):
    from autopyro.visualization.plotting import AxisSpec, PlotSpec

    window.show_plots()
    plot = window.plot_window
    md, toc, s2 = [window.model.project.columns[i].id for i in (2, 3, 4)]
    annotation = {'coordinates': 'axes', 'x': .2, 'y': .8}
    plot.spec = PlotSpec(kind='multi_depth', x_id=toc, y_id=md, track_ids=[toc, s2],
                         text_numbers=True, y_axis=AxisSpec(inverted=True),
                         track_axes={s2: AxisSpec(label='Second track only')},
                         annotations=[annotation | {'text': 'Other label', 'track_id': toc},
                                      annotation | {'text': 'Selected label', 'track_id': s2}])
    plot.load_controls()
    plot.redraw()
    assert plot.result is not None, plot.status.text()
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args, **kwargs: (args[3][1], True))
    path = tmp_path / 'track.svg'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), 'SVG (*.svg)'))
    menu = plot.plot_actions_menu
    action = next(action for action in menu.actions() if action.text() == 'Экспортировать отдельный трек…')
    menu.aboutToShow.emit()
    assert action.isEnabled()
    action.trigger()
    qtbot.waitUntil(lambda: not plot.export_busy(), timeout=30000)
    svg = path.read_text(encoding='utf-8')
    assert 'Second track only' in svg and 'Selected label' in svg
    assert 'Other label' not in svg
    assert plot.spec.track_ids == [toc, s2]
