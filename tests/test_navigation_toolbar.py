"""All Matplotlib views share navigation, while domain commands stay in the app UI."""

import pytest
from matplotlib.figure import Figure
from PySide6.QtWidgets import QFileDialog, QWidgetAction

from autopyro.io import build_project
from autopyro.mapping import MapSpec, build_map
from autopyro.ui.dialogs.geometry_editor import GeometryEditor, new_scheme
from autopyro.ui.dialogs.map import SpatialInterpolationDialog
from autopyro.ui.widgets.figure_canvas import CanvasView, FittedFigureCanvas
from autopyro.ui.widgets.navigation import NavigationToolbar
from autopyro.ui.window import MainWindow


@pytest.fixture
def owner(qtbot):
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.may_replace = lambda: True
    project = build_project(['X', 'Y'], [[i, 2 * i + 1] for i in range(6)], list(range(2, 8)),
                            [(0, 'X', 'number', 'm'), (1, 'Y', 'number', 'm')], {})
    owner.set_project(project)
    yield owner
    owner.desktop.close_documents()
    for document in (owner.plot_window, getattr(owner.interpretation_window, 'plot', None)):
        if document is not None:
            document.preview.shutdown()
    owner.dirty = False


def test_viewers_use_common_navigation_and_no_backend_edit_or_save_tools(owner, qtbot):
    owner.show_plots()
    owner.plot_window.timer.stop()
    owner.show_maps()
    owner.map_window.timer.stop()
    owner.show_stratigraphy()
    owner.show_interpretation()
    owner.interpretation_window.timer.stop()
    owner.interpretation_window.plot.timer.stop()
    owner.show_curve_fitting()
    editor = GeometryEditor(new_scheme())
    qtbot.addWidget(editor)
    project = owner.model.project
    snapshot = build_map(project, [row.id for row in project.rows], MapSpec(
        x_id=project.columns[0].id, y_id=project.columns[1].id,
        source_crs='EPSG:32631', target_crs='EPSG:32631'), [])
    spatial = SpatialInterpolationDialog(snapshot, owner.map_window)
    qtbot.addWidget(spatial)
    viewers = [owner.plot_window, owner.map_window, owner.interpretation_window.plot,
               owner.curve_fitting_window, editor, spatial]
    toolbars = [viewer.toolbar for viewer in viewers] + [owner.stratigraphy_window.column_toolbar]
    for toolbar in toolbars:
        assert type(toolbar) is NavigationToolbar
        assert list(toolbar._actions) == ['home', 'back', 'forward', 'pan', 'zoom']
        commands = [action for action in toolbar.actions() if not isinstance(action, QWidgetAction)]
        if toolbar is owner.interpretation_window.plot.toolbar:
            plot = owner.interpretation_window.plot
            assert commands == [toolbar._actions[key] for key in ('home', 'pan', 'zoom')] + [plot.export_button]
            assert all(toolbar._actions[key] in plot.plot_actions_menu.actions() for key in ('back', 'forward'))
        else:
            assert commands == list(toolbar._actions.values())
            widget_count = sum(isinstance(action, QWidgetAction) for action in toolbar.actions())
            assert widget_count == (1 if toolbar is owner.plot_window.toolbar else 2)
    assert owner.plot_window.plot_actions_button.menu() is owner.plot_window.plot_actions_menu
    assert owner.curve_fitting_window.canvas_view.fit
    plot = owner.plot_window
    plot.redraw()
    original = plot.style.values()['marker_size']
    plot.style.load({'marker_size': original + 3})
    plot.redraw()
    plot.plot_actions_menu.actions()[0].trigger()
    assert plot.style.values()['marker_size'] == original
    plot.plot_actions_menu.actions()[1].trigger()
    assert plot.style.values()['marker_size'] == original + 3
    spatial.close()
    editor.close()


def test_common_navigation_preserves_history_and_exclusive_mouse_tools(qtbot):
    figure = Figure()
    canvas = FittedFigureCanvas(figure)
    view = CanvasView(canvas)
    qtbot.addWidget(view.scroll)
    toolbar = NavigationToolbar(canvas, view.scroll)
    view.add_to_toolbar(toolbar)
    ax = figure.subplots()
    ax.plot([0, 10], [0, 10])
    canvas.draw()
    original = ax.get_xlim()
    toolbar.push_current()
    ax.set_xlim(2, 4)
    toolbar.push_current()
    toolbar.back()
    assert ax.get_xlim() == original
    toolbar.forward()
    assert ax.get_xlim() == (2, 4)
    toolbar.home()
    assert ax.get_xlim() == original
    toolbar.pan()
    assert toolbar._actions['pan'].isChecked()
    toolbar.zoom()
    assert toolbar._actions['zoom'].isChecked() and not toolbar._actions['pan'].isChecked()
    toolbar.cancel_drag()
    assert not canvas.widgetlock.locked()
    assert not toolbar._actions['zoom'].isChecked()


def test_curve_plot_export_is_available_in_application_controls(owner, qtbot, monkeypatch, tmp_path):
    owner.show_curve_fitting()
    dialog = owner.curve_fitting_window
    assert not dialog.export_plot_button.isEnabled()
    dialog.calculate()
    qtbot.waitUntil(lambda: dialog.result is not None and dialog.worker is not None
                   and not dialog.worker.isRunning(), timeout=15000)
    assert dialog.tabs.currentWidget() is dialog.plot_page
    assert dialog.export_plot_button.isEnabled()
    path = tmp_path / 'fit.svg'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **kw: (str(path), 'SVG (*.svg)'))
    dialog.export_plot_button.click()
    assert path.exists() and '<svg' in path.read_text(encoding='utf-8')
    dialog.initial.setText('1, 1')
    assert not dialog.export_plot_button.isEnabled()
