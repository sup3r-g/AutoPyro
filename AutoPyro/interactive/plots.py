from io import BytesIO, StringIO

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
from holoviews import opts
from holoviews.annotators import PointAnnotator, PolyAnnotator
from holoviews.plotting.links import DataLink
from holoviews.streams import PointDraw, PolyDraw, PolyEdit
from skimage import (
    color,
    exposure,
    filters,
    io,
    measure,
    morphology,
    segmentation,
    transform,
    util,
)
from sklearn.cluster import KMeans
from PIL import Image

# from core.maps import Map

hv.extension("bokeh")
pn.extension(sizing_mode="scale_both")


class PlotUI(param.Parameterized):
    image_file = pn.widgets.FileInput(accept=".png,.jpg,.tiff")
    downscale = param.Boolean(
        False, doc="A sample Boolean parameter", label="Downscale"
    )
    image = param.Parameter()
    dataframe = param.DataFrame()

    def __init__(self, **params):
        super().__init__(**params)

    @param.depends("image_file.value", "downscale", watch=True)
    def _preprocess_image(self):
        if self.image_file.value:
            bytes_io = BytesIO(self.image_file.value)
            self.image_file.save(bytes_io)
            self.image = np.array(Image.open(bytes_io))
            print(self.downscale)
            if self.downscale:
                self.image = transform.rescale(
                    self.image,
                    0.5,
                    anti_aliasing=True,
                    channel_axis=-1,
                )

    # @param.depends("image")
    def image_view(self):
        image = self.image
        if image is not None:
            rgb = hv.RGB(image, bounds=(-1, -1, 1, 1)).apply.opts(
                active_tools=["box_zoom"],
                tools=["hover"],
            )
            return pn.pane.HoloViews(rgb, sizing_mode="scale_both")

    def polygons(self):
        polys = hv.Polygons(
            [hv.Box(*i, spec=np.random.rand() / 3) for i in np.random.rand(10, 2)]
        ).opts(min_height=500)
        poly_annotator = hv.annotate.instance()
        poly_annotate = poly_annotator(
            polys,
            annotations=["Класс"],
            name="Аннотации полигонов",
        )

        print(poly_annotate.Annotator.I.Table)
        return poly_annotate

    def points(self):
        points = hv.Points([(0.0, 0.0), (1.0, 1.0), (20.0, 20.0)]).opts(
            size=10, min_height=500
        )
        point_annotator = hv.annotate.instance()
        point_layout = point_annotator(
            points,
            annotations=["Класс"],
            name="Аннотации точек",
        )
        print(point_layout.annotator)
        return point_layout  # PointAnnotator(points.opts(), annotations=['Label'], name="Point Annotations")

    def view(self):
        return pn.Column(
            "# Графики",
            pn.Column(
                self.image_file,
                pn.Param(
                    self.param.downscale,
                    name="",
                    widgets={
                        "downscale": {
                            "widget_type": pn.widgets.Toggle,
                            "name": "Масштабировать вниз",
                            "button_type": "primary",
                        }
                    },
                ),
            ),
            self.image_view,
            # pn.layout.VSpacer(),
            sizing_mode="scale_both",
        )


PLOT_UI = PlotUI().view()

if __name__ == "__main__":
    pn.Row(PlotUI().view).show()

"Значения параметров:"

"# Построение графиков",

# pn.template.MaterialTemplate(
#     title="Графики",
#     logo="img/graph-up.svg",
#     favicon="img/graph-up.svg",
#     sidebar=[pn.pane.Markdown("### Меню"), menu],
#     main=[
#         description,
#         sample_data_app_view,
#         voltage_app_view,
#     ],
# ).servable()
