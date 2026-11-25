"""Hover lookup follows rendered coordinates and scientific polygon geometry."""
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.transforms import IdentityTransform

from autopyro.io import build_project
from autopyro.plotting import PlotSpec, render_plot
from autopyro.plot_hover import HoverIndex
from autopyro.standard_interpretation import standard_schemes


def frame():
    project = build_project(['TOC', 'S2'], [['1.2', '3'], ['1.8', '4']], [1,2],
                            [(0,'TOC','number','%wt'), (1,'S2','number','mg HC/g rock')], {})
    x,y = [c.id for c in project.columns]
    binding = dict(x_id=x, y_id=y, x_unit='%wt', y_unit='mg HC/g rock',
                   scheme=standard_schemes()['S2–TOC'])
    spec = PlotSpec(x_id=x, y_id=y, interpretation=binding, interpretation_labels=False)
    figure, result = render_plot(project, project.row_map(), spec)
    FigureCanvasAgg(figure).draw()
    return project, figure, result, binding


def test_point_index_is_reused_and_rebuilt_after_zoom():
    project, figure, result, binding = frame()
    ax = figure.axes[0]
    index = HoverIndex(result, binding)
    pixel = ax.transData.transform((1.2,3))
    hit = index.point(ax, *pixel)
    assert hit[0] == project.rows[0].id
    assert hit[1] == (1.2,3)
    cached = index.points[ax]
    index.point(ax, *pixel)
    assert index.points[ax] is cached
    assert index.point(ax, -1000, -1000) is None
    ax.set_xlim(1, 1.5)
    assert index.point(ax, *ax.transData.transform((1.2,3)))[0] == hit[0]
    assert index.points[ax] is not cached


def test_hidden_labels_and_shared_boundary_are_queryable():
    _, _, result, binding = frame()
    index = HoverIndex(result, binding)
    assert index.region_labels(1.5, 3) == ['TOC: хороший; S2: удовлетворительный']
    assert len(index.region_labels(1, 3)) == 2
    assert index.region_labels(-1, 3) == []


def test_stacked_hover_counts_samples_and_keeps_index():
    project, figure, result, binding = frame()
    ax = figure.axes[0]
    duplicate = ax.scatter([1.2], [3])
    duplicate._autopyro_row_ids = ['another-sample']
    result['artists'].append(duplicate)
    index = HoverIndex(result, binding)
    pixel = ax.transData.transform((1.2, 3))
    hit = index.point(ax, *pixel, with_count=True)
    assert hit[0] in {project.rows[0].id, 'another-sample'}
    assert hit[3] == 2
    cached = index.points[ax]
    assert index.point(ax, *pixel, with_count=True) == hit
    assert index.points[ax] is cached


def test_point_index_filters_by_screen_distance_and_counts_the_radius_boundary():
    _, figure, _, _ = frame()
    ax = figure.axes[0]
    x, y = (int(value) for value in ax.transAxes.transform((.5, .5)))
    artist = ax.scatter([x - 8, x + 2, x + 7, x], [y, y + 1, y + 7, y + 9])
    artist.set_offset_transform(IdentityTransform())
    artist._autopyro_row_ids = ['boundary', 'nearest', 'diagonal outside', 'outside']
    index = HoverIndex({'artists': [artist]}, None)
    assert index.point(ax, x, y, with_count=True) == ('nearest', (x + 2, y + 1), ax, 2)
    assert index.point(ax, x - 16, y) is None

def test_region_holes_and_log_analysis_coordinates():
    _, _, result, binding = frame()
    scheme = binding['scheme']
    scheme['x']['transform'] = 'log10'
    scheme['regions'] = [dict(id='hole', label='Полное название', geometry=dict(type='Polygon', coordinates=[
        [(0,0),(2,0),(2,10),(0,10),(0,0)],
        [(.5,2),(1.5,2),(1.5,8),(.5,8),(.5,2)]]))]
    index = HoverIndex(result, binding)
    assert index.region_labels(10,5) == []
    assert index.region_labels(2,5) == ['Полное название']



def test_3d_hover_uses_current_camera_and_preserves_row_identity():
    from mpl_toolkits.mplot3d.proj3d import proj_transform

    project = build_project(['X', 'Y', 'Z'], [['1', '2', '3'], ['4', '5', '6']], [1, 2],
                            [(i, name, 'number', '') for i, name in enumerate(('X', 'Y', 'Z'))], {})
    x, y, z = [column.id for column in project.columns]
    spec = PlotSpec(kind='mpl3d:scatter', x_id=x, y_id=y, z_id=z, camera_3d=(18, 25, 0))
    figure, result = render_plot(project, project.row_map(), PlotSpec.from_dict(spec.to_dict()))
    FigureCanvasAgg(figure).draw()
    ax = figure.axes[0]
    index = HoverIndex(result, None)
    assert (ax.elev, ax.azim) == (18, 25)
    for angle in (25, 60):
        ax.view_init(elev=18, azim=angle)
        px, py, _ = proj_transform(1, 2, 3, ax.get_proj())
        hit = index.point(ax, *ax.transData.transform((px, py)))
        assert hit[:2] == (project.rows[0].id, (1, 2, 3))
