"""Remote RGBA must survive highlight draws at non-integral figure sizes."""
import numpy as np
from matplotlib.figure import Figure
from autopyro.ui.plot_preview import PreviewCanvas


def test_present_keeps_renderer_for_fractional_figure_bounds(qtbot):
    figure=Figure(figsize=(3.14159,2.12345),dpi=96)
    canvas=PreviewCanvas(figure)
    qtbot.addWidget(canvas)
    width,height=canvas.get_width_height(physical=True)
    pixels=np.empty((height,width,4),dtype=np.uint8)
    pixels[:]=[32,64,96,255]
    canvas.present(pixels)
    renderer=canvas.renderer
    assert canvas.get_renderer() is renderer
    np.testing.assert_array_equal(np.asarray(canvas.get_renderer().buffer_rgba()),pixels)


def test_present_survives_size_rounding_before_highlight_draw(qtbot):
    figure=Figure(figsize=(3,2),dpi=96)
    canvas=PreviewCanvas(figure)
    qtbot.addWidget(canvas)
    width,height=canvas.get_width_height(physical=True)
    pixels=np.empty((height,width,4),dtype=np.uint8)
    pixels[:]=[32,64,96,255]
    figure.set_size_inches((width+1)/96,(height+1)/96,forward=False)
    canvas.present(pixels)
    renderer=canvas.renderer
    assert canvas.get_renderer() is renderer
    np.testing.assert_array_equal(np.asarray(renderer.buffer_rgba()),pixels)
