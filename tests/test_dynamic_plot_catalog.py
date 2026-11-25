"""Exercise runtime catalogs and the actual rendering of discovered commands."""

import math

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap, get_named_colors_mapping
from matplotlib.markers import MarkerStyle
import pytest

from autopyro.io import build_project
from autopyro.plot_catalog import COLORS, MARKERS, KINDS, colormap_names, library_kinds, library_parameters
from autopyro.plotting import AxisSpec, PlotSpec, render_plot


@pytest.fixture
def grid_project():
    names = ['x', 'y', 'z', 'u', 'v', 'w', 'dx', 'dy', 'dz', 'filled']
    rows = [[x, y, x + y, .3, .4, .5, .3, .4, .5, 1] for y in range(1, 9) for x in range(1, 9)]
    return build_project(names, rows, list(range(2, 66)),
                         [(i, name, 'number', '') for i, name in enumerate(names)], {})


def test_live_catalogs_include_runtime_registrations(monkeypatch):
    mapping = get_named_colors_mapping()
    monkeypatch.setitem(mapping, 'autopyro:test-color', '#123456')
    monkeypatch.setitem(MarkerStyle.markers, 'test-marker', 'circle')
    monkeypatch.setattr(Axes, 'test_chart', Axes.scatter, raising=False)
    cmap = ListedColormap(['red', 'blue'], name='autopyro-test')
    mpl.colormaps.register(cmap)
    try:
        assert 'autopyro:test-color' in COLORS
        assert 'test-marker' in MARKERS
        assert 'mpl:test_chart' in KINDS
        assert ('mpl:test_chart', KINDS['mpl:test_chart']) in dict(KINDS.groups())['Другие графики']
        assert 'autopyro-test' in colormap_names()
        assert set(get_named_colors_mapping()) <= set(COLORS)
        assert set(MarkerStyle.markers) == set(MARKERS)
        assert set(mpl.colormaps) == set(colormap_names())
    finally:
        mpl.colormaps.unregister('autopyro-test')


@pytest.mark.parametrize('kind', [key for key in library_kinds() if key.startswith(('mpl:', 'mpl3d:'))])
def test_every_discovered_matplotlib_command_renders(grid_project, kind):
    columns = {column.name: column.id for column in grid_project.columns}
    parameters = library_parameters(kind)
    bindings = {}
    for index, key in enumerate(parameters):
        name = key.lower().removesuffix('s')
        if name not in columns:
            name = ('x', 'y', 'z')[min(index, 2)]
        bindings[key] = columns[name]
    spec = PlotSpec(kind=kind, x_id=columns['x'], y_id=columns['y'],
                    track_ids=[columns[key] for key in ('x', 'y', 'z')], library_bindings=bindings)
    if kind in ('mpl:cohere', 'mpl:specgram'):
        spec.library_options = '{"NFFT": 16, "noverlap": 8}'
    figure, result = render_plot(grid_project, grid_project.row_map(), spec)
    FigureCanvasAgg(figure).draw()
    assert result['plotted'][spec.x_id] == len(grid_project.rows)
    assert figure.axes
    figure.clear()


@pytest.mark.parametrize('base', [2, 10, math.e, .5])
def test_logarithm_base_is_serialized_and_used_on_each_axis(grid_project, base):
    x, y = grid_project.columns[:2]
    spec = PlotSpec(x_id=x.id, y_id=y.id, x_axis=AxisSpec(scale='log', log_base=base),
                    y_axis=AxisSpec(scale='log', log_base=base))
    restored = PlotSpec.from_dict(spec.to_dict())
    figure, _ = render_plot(grid_project, grid_project.row_map(), restored)
    ax = figure.axes[0]
    assert ax.xaxis.get_transform().base == pytest.approx(base)
    assert ax.yaxis.get_transform().base == pytest.approx(base)


@pytest.mark.parametrize('base', [0, -1, 1, math.inf, math.nan])
def test_invalid_logarithm_base_is_rejected(grid_project, base):
    with pytest.raises(ValueError, match='Основание'):
        render_plot(grid_project, grid_project.row_map(), PlotSpec(x_axis=AxisSpec(log_base=base)))


def test_old_axis_specs_use_base_ten():
    assert PlotSpec.from_dict({'x_axis': {'scale': 'log'}}).x_axis.log_base == 10


@pytest.mark.parametrize('kind,xscale,yscale', [
    ('mpl:semilogx', 'log', 'linear'), ('mpl:semilogy', 'linear', 'log'),
    ('mpl:loglog', 'log', 'log'),
])
@pytest.mark.parametrize('options', ['{"linestyle": "--"}',
                                    '{"linestyle": ":", "base": 2, "subs": [1, 2], "nonpositive": "mask"}'])
def test_legacy_log_lines_migrate_and_render_identically(grid_project, kind, xscale, yscale, options):
    import numpy as np
    x, y = [column.id for column in grid_project.columns[:2]]
    legacy = PlotSpec(kind=kind, library_bindings={'x': x, 'y': y},
                      x_axis=AxisSpec(log_base=2), library_options=options)
    migrated = PlotSpec.from_dict(legacy.to_dict())
    assert migrated.kind == 'mpl:plot'
    assert (migrated.x_axis.scale, migrated.y_axis.scale) == (xscale, yscale)
    assert migrated.x_axis.log_base == 2
    assert migrated.library_bindings == legacy.library_bindings
    assert migrated.library_options == legacy.library_options
    assert PlotSpec.from_dict(migrated.to_dict()).to_dict() == migrated.to_dict()
    images = []
    for spec in (legacy, migrated):
        figure, _ = render_plot(grid_project, grid_project.row_map(), spec)
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        images.append(np.asarray(canvas.buffer_rgba()).copy())
    np.testing.assert_array_equal(*images)


@pytest.mark.parametrize('kind', ['seaborn:scatterplot', 'seaborn:lineplot', 'seaborn:histplot',
                                 'seaborn:heatmap', 'ultraplot:scatter', 'ultraplot:plot', 'ultraplot:heatmap'])
def test_optional_library_renders_when_installed(grid_project, kind):
    if kind not in library_kinds():
        pytest.skip('Optional visualization library is not installed')
    x, y, z = [column.id for column in grid_project.columns[:3]]
    spec = PlotSpec(kind=kind, x_id=x, y_id=y, track_ids=[x, y, z])
    figure, result = render_plot(grid_project, grid_project.row_map(), spec)
    FigureCanvasAgg(figure).draw()
    assert result['plotted'][x] == 64
