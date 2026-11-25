"""View modes survive project storage independently of scientific settings."""

from copy import deepcopy
from pathlib import Path
import json

import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

from autopyro.core import Project
from autopyro.io import build_project
from autopyro.mapping import MapSpec, build_map
from autopyro.ui.window import MainWindow
from autopyro.ui.dialogs.geometry_editor import GeometryEditor
from autopyro.ui.dialogs.map import SpatialInterpolationDialog
from autopyro.ui.widgets.figure_canvas import CanvasView, FittedFigureCanvas


def visualizations(owner, qtbot):
    owner.show_plots()
    owner.plot_window.timer.stop()
    owner.show_maps()
    owner.map_window.timer.stop()
    owner.show_stratigraphy()
    owner.show_interpretation()
    owner.interpretation_window.timer.stop()
    owner.interpretation_window.plot.timer.stop()
    scheme = json.loads((Path(__file__).parents[1] / 'examples/interpretation-demo.json').read_text(encoding='utf-8'))
    editor = GeometryEditor(scheme, owner.interpretation_window)
    qtbot.addWidget(editor)
    project = owner.model.project
    report = build_map(project, [row.id for row in project.rows], MapSpec(
        x_id=project.columns[0].id, y_id=project.columns[1].id,
        source_crs='EPSG:32631', target_crs='EPSG:32631'), [])
    spatial = SpatialInterpolationDialog(report, owner.map_window)
    qtbot.addWidget(spatial)
    return {
        'plot': owner.plot_window.canvas_view,
        'interpretation_plot': owner.interpretation_window.plot.canvas_view,
        'map': owner.map_window.canvas_view,
        'stratigraphy_column': owner.stratigraphy_window.canvas_view,
        'spatial_interpolation': spatial.canvas_view,
        'geometry_editor': editor.canvas_view,
    }, [editor, spatial]


@pytest.mark.parametrize('reverse', [False, True])
def test_all_visualization_modes_survive_project_reopening(qtbot, tmp_path, reverse):
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.may_replace = lambda: True
    project = build_project(['X', 'Y', 'TOC'], [[0, 0, 1], [100, 0, 2], [0, 100, 3]], [1, 2, 3],
                            [(0, 'X', 'number', 'm'), (1, 'Y', 'number', 'm'), (2, 'TOC', 'number', '%wt')], {})
    owner.set_project(project)
    controllers, transient = visualizations(owner, qtbot)
    revision = owner.analysis_revision
    owner._autosaved_revision = revision
    expected = {}
    for index, (key, view) in enumerate(controllers.items()):
        assert view.fit
        actual = bool(index % 2) != reverse
        view.mode_combo.setCurrentIndex(1)
        if not actual:
            view.mode_combo.setCurrentIndex(0)
        expected[key] = 'actual' if actual else 'fit'
    assert owner.dirty
    assert owner.analysis_revision == revision
    assert owner._autosaved_revision is None
    assert project.view['visualization_view_modes'] == expected
    for dialog in transient:
        dialog.close()
    owner.desktop.close_documents()
    owner.path = tmp_path / 'view-modes.apyro'
    assert owner.save()
    restored = Project.load(owner.path)
    assert restored.view['visualization_view_modes'] == expected
    owner.set_project(restored, owner.path)
    reopened, transient = visualizations(owner, qtbot)
    for key, view in reopened.items():
        assert view.fit == (expected[key] == 'fit')
        assert view.mode_combo.currentData() == view.fit
        assert view.scroll.widgetResizable() == view.fit
    assert restored.view['visualization_view_modes'] == expected
    for dialog in transient:
        dialog.close()
    owner.desktop.close_documents()
    owner.set_project(Project())
    owner.show_plots()
    owner.plot_window.timer.stop()
    assert owner.plot_window.canvas_view.fit
    assert 'visualization_view_modes' not in owner.model.project.view
    owner.desktop.close_documents()
    owner.dirty = False


@pytest.mark.parametrize('saved', [None, {'plot': 'unexpected'}, ['invalid']])
def test_missing_or_unknown_mode_defaults_to_fit_without_edit(qtbot, saved):
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.may_replace = lambda: True
    if saved is not None:
        owner.model.project.view['visualization_view_modes'] = saved
    before = deepcopy(owner.model.project.view)
    owner.dirty = False
    view = CanvasView(FittedFigureCanvas(Figure()))
    qtbot.addWidget(view.scroll)
    toolbar = NavigationToolbar2QT(view.canvas, view.scroll)
    view.add_to_toolbar(toolbar)
    view.bind_project(owner, 'plot')
    assert view.fit and view.mode_combo.currentData() is True
    assert owner.model.project.view == before
    assert not owner.dirty
