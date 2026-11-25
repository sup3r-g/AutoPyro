import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QGroupBox
from matplotlib.colors import get_named_colors_mapping
from matplotlib.markers import MarkerStyle

from autopyro.io import build_project
from autopyro.plot_catalog import colormap_names, library_kinds
from autopyro.ui import i18n
from autopyro.ui.window import MainWindow


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    store = QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)
    monkeypatch.setattr(i18n, 'settings', lambda: store)
    i18n.initialize()
    window = MainWindow()
    qtbot.addWidget(window)
    rows = [[x, y, x+y] for y in range(1, 5) for x in range(1, 5)]
    project = build_project(['X', 'Y', 'Z'], rows, list(range(2, 18)),
                            [(i, key, 'number', '') for i, key in enumerate(('X', 'Y', 'Z'))], {})
    window.set_project(project)
    window.may_replace = lambda: True
    yield window
    window.dirty = False


@pytest.mark.parametrize('kind', ['mpl:plot', 'mpl:imshow', 'mpl3d:scatter', 'seaborn:scatterplot', 'ultraplot:scatter'])
def test_library_plot_controls_render_and_switch_back(window, kind):
    if kind not in library_kinds():
        pytest.skip('Optional visualization library is not installed')
    window.show_plots()
    dialog = window.plot_window
    dialog.timer.stop()
    dialog.canvas_view.mode_combo.setCurrentIndex(1)
    x, y, z = [column.id for column in window.model.project.columns]
    dialog.data.load({'kind': kind})
    dialog.spec.kind = kind
    dialog.spec.track_ids = [x, y, z]
    dialog.spec.library_bindings = {'x': x, 'y': y, 'xs': x, 'ys': y, 'zs': z}
    dialog.load_controls()
    dialog.redraw()
    assert dialog.result is not None, dialog.status.text()
    assert dialog.export_button.isEnabled()
    if kind.startswith('mpl3d:'):
        dialog.z_form.load({'scale': 'log', 'log_base': 2})
        dialog.redraw()
        assert dialog.figure.axes[0].zaxis.get_transform().base == 2
    dialog.data.load({'kind': 'scatter'})
    dialog.redraw()
    assert dialog.result is not None, dialog.status.text()
    assert dialog.figure.axes[0].name == 'rectilinear'
    assert not dialog.canvas_view.fit
    assert dialog.canvas_view.mode_combo.currentData() is False
    assert dialog.canvas.width() == round(dialog.spec.width_mm / 25.4 * 96)


def test_complete_styles_log_base_and_reset(window):
    window.show_plots()
    dialog = window.plot_window
    dialog.timer.stop()
    palette = dialog.style.fields['color'][0]
    assert 'xkcd:sky blue' in [palette.itemData(i) for i in range(palette.count())]
    marker = dialog.style.fields['marker'][0]
    assert {marker.itemData(i) for i in range(marker.count())} == set(MarkerStyle.markers)
    cmap = dialog.style.fields['cmap'][0]
    assert {cmap.itemData(i) for i in range(cmap.count())} == set(colormap_names())
    dialog.style.load({'color': 'xkcd:sky blue', 'marker': 11})
    dialog.x_form.load({'scale': 'log', 'log_base': 2})
    assert dialog.x_form.fields['log_base'][0].isEnabled()
    dialog.redraw()
    assert dialog.result is not None, dialog.status.text()
    assert dialog.spec.marker == 11
    assert dialog.figure.axes[0].xaxis.get_transform().base == 2
    dialog.restore_parameters()
    assert dialog.x_form.values()['log_base'] == 10
    assert not dialog.x_form.fields['log_base'][0].isEnabled()


@pytest.mark.parametrize('method,attribute,field', [
    ('show_plots', 'plot_window', 'x'), ('show_maps', 'map_window', 'data'),
    ('show_statistics', 'statistics_window', 'scope'), ('show_interpretation', 'interpretation_window', 'x'),
    ('interpolate', 'interpolation_window', 'method'), ('show_calculations', 'calculation_window', 'scope'),
    ('show_stratigraphy', 'stratigraphy_window', 'name'), ('show_nonpyro', 'nonpyro_window', 'method'),
    ('show_qc', 'qc_window', 'severity'),
])
def test_all_module_parameters_have_named_group_boxes(window, method, attribute, field):
    getattr(window, method)()
    dialog = getattr(window, attribute)
    if hasattr(dialog, 'timer'):
        dialog.timer.stop()
    tabs = getattr(dialog, 'parameter_tabs', None) or dialog.tabs
    groups = tabs.findChildren(QGroupBox)
    assert len(groups) >= 2
    assert all(group.title() for group in groups)
    widget = getattr(dialog, field)
    if field == 'data':
        widget = next(iter(widget.fields.values()))[0]
    parent = widget.parentWidget()
    while parent is not None and not isinstance(parent, QGroupBox):
        parent = parent.parentWidget()
    assert parent is not None
