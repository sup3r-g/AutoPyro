"""Scientific overlays must not control sample zoom or overlap region labels."""
from copy import deepcopy
from io import BytesIO

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from autopyro.io import build_project
from autopyro.plotting import AxisSpec, PlotSpec, render_plot
from autopyro.standard_interpretation import standard_schemes
from autopyro.interpretation_labels import RegionLabels


def chart(rows=((1.2, 3), (1.8, 4)), *, log=False):
    project = build_project(['TOC', 'S2'], [[str(a), str(b)] for a,b in rows],
                            list(range(1, len(rows) + 1)),
                            [(0, 'TOC', 'number', '%wt'), (1, 'S2', 'number', 'mg HC/g rock')], {})
    x,y = [c.id for c in project.columns]
    spec = PlotSpec(x_id=x, y_id=y, x_axis=AxisSpec(scale='log' if log else 'linear'))
    binding = dict(x_id=x, y_id=y, x_unit='%wt', y_unit='mg HC/g rock',
                   scheme=standard_schemes()['S2–TOC'])
    return project, spec, binding


@pytest.mark.parametrize('log', [False, True])
def test_limits_are_identical_with_and_without_large_polygons_and_lines(log):
    project, spec, binding = chart(log=log)
    baseline, _ = render_plot(project, project.row_map(), spec)
    spec.interpretation = binding
    # A huge reference curve must not influence auto limits either.
    binding['scheme']['lines'] = [dict(id='far', label='far', geometry=dict(
        type='LineString', coordinates=[[.1, 0], [1e6, 1e6]]))]
    figure, _ = render_plot(project, project.row_map(), spec)
    ax = figure.axes[0]
    np.testing.assert_allclose(ax.get_xlim(), baseline.axes[0].get_xlim())
    np.testing.assert_allclose(ax.get_ylim(), baseline.axes[0].get_ylim())
    # A subsequent Matplotlib autoscale must still use only the sample bounds.
    ax.autoscale()
    np.testing.assert_allclose(ax.get_xlim(), baseline.axes[0].get_xlim())
    np.testing.assert_allclose(ax.get_ylim(), baseline.axes[0].get_ylim())


def test_manual_limits_and_empty_sample_fallback():
    project, spec, binding = chart()
    spec.interpretation = binding
    spec.x_axis.minimum, spec.x_axis.maximum = .5, 8
    spec.y_axis.minimum, spec.y_axis.maximum = 0, 12
    figure, _ = render_plot(project, project.row_map(), spec)
    assert figure.axes[0].get_xlim() == (.5, 8)
    assert figure.axes[0].get_ylim() == (0, 12)
    spec.x_axis, spec.y_axis = AxisSpec(), AxisSpec()
    empty, _ = render_plot(project, [], spec)
    assert empty.axes[0].get_xlim()[1] >= 100
    assert empty.axes[0].get_ylim()[1] >= 1000


def assert_label_layout(figure):
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    ax = figure.axes[0]
    layer = next(a for a in ax.artists if isinstance(a, RegionLabels))
    renderer = canvas.get_renderer()
    bounds = [text.get_window_extent(renderer) for text in layer.labels]
    for i, rect in enumerate(bounds):
        assert ax.bbox.contains(rect.x0, rect.y0) and ax.bbox.contains(rect.x1, rect.y1)
        assert all(not rect.overlaps(other) for other in bounds[i + 1:])
    return layer


def test_labels_fit_visible_areas_after_zoom_resize_and_export():
    project, spec, binding = chart()
    spec.interpretation = binding
    figure, _ = render_plot(project, project.row_map(), spec)
    layer = assert_label_layout(figure)
    assert layer.labels  # A visible label is relocated from the full polygon centre.
    before = figure.axes[0].transData.inverted().transform(layer.labels[0].get_position())
    figure.axes[0].set_xlim(1.1, 1.9)
    figure.axes[0].set_ylim(2.6, 4.9)
    layer = assert_label_layout(figure)
    assert layer.labels
    after = figure.axes[0].transData.inverted().transform(layer.labels[0].get_position())
    assert not np.allclose(after, before)
    figure.set_size_inches(3, 2)
    figure.set_dpi(200)
    assert_label_layout(figure)
    for fmt in ('svg', 'pdf', 'png'):
        buffer = BytesIO()
        figure.savefig(buffer, format=fmt, dpi=300)
        assert buffer.tell() > 100


def test_overlapping_regions_and_tiny_views_do_not_overlap_labels():
    project, spec, binding = chart()
    s = binding['scheme']
    original = next(e for e in s['regions'] if e['id'] == '2:1')
    s['regions'] = [dict(deepcopy(original), id=str(i), label='Область ' + str(i)) for i in range(20)]
    spec.interpretation = binding
    figure, _ = render_plot(project, project.row_map(), spec)
    layer = assert_label_layout(figure)
    assert 0 < len(layer.labels) < 20
    figure.set_size_inches(.8, .6)
    assert_label_layout(figure)
