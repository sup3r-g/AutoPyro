from textwrap import fill
from typing import Any, Iterable, Literal, Optional, Union

import numpy as np
from matplotlib.artist import ArtistInspector
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.pyplot import get_cmap, subplots, gca
from matplotlib.colors import to_rgba
from shapely import (
    Geometry,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    get_coordinates,
)

from AutoPyro.core.charts import Chart
from AutoPyro.core.maps import Map

COLORS = get_cmap("hsv")(np.linspace(0, 1, 20))
np.random.default_rng().shuffle(COLORS)


MATPLOTLIB_SHAPES_MAP = {
    LineString: Line2D,
    Point: Line2D,
    Polygon: PathPatch,
}


def matplotlib_args(geometry: Geometry) -> dict[str, Any]:
    return ArtistInspector(MATPLOTLIB_SHAPES_MAP.get(geometry)).get_setters()


def annotation_helper(
    line: LineString, place: Literal["left", "right", "center"] = "left"
) -> tuple[tuple[float, float], tuple[int, int]]:
    if place == "left":
        return tuple(list(line.coords)[0]), (15, 10)
    if place == "right":
        return tuple(list(line.coords)[-1]), (-15, 10)
    if place == "center":
        return tuple(line.centroid.coords[0]), (15, 10)

    # return (x, y), (x_text, y_text)


"""
Modified code from shapely.plotting:
https://github.com/shapely/shapely/blob/main/
"""


def _default_ax():
    ax = gca()
    ax.grid(True)
    ax.set_aspect("equal")
    return ax


def _path_from_polygon(polygon) -> Path:
    if isinstance(polygon, MultiPolygon):
        return Path.make_compound_path(
            *[_path_from_polygon(poly) for poly in polygon.geoms]
        )

    return Path.make_compound_path(
        Path(np.asarray(polygon.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in polygon.interiors],
    )


def plot_polygon(
    polygon,
    ax=None,
    add_points: bool = False,
    color=None,
    facecolor=None,
    edgecolor=None,
    linewidth=None,
    **kwargs,
):
    if ax is None:
        ax = _default_ax()

    if color is None:
        color = "C0"
    color = to_rgba(color)

    if facecolor is None:
        facecolor = list(color)
        facecolor[-1] = 0.3
        facecolor = tuple(facecolor)

    if edgecolor is None:
        edgecolor = color

    patch = PathPatch(
        _path_from_polygon(polygon),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        **kwargs,
    )
    ax.add_patch(patch)
    ax.autoscale_view()

    if add_points:
        line = plot_points(polygon, ax=ax, color=color)
        return patch, line

    return patch


def plot_line(
    line, ax=None, add_points: bool = False, color=None, linewidth=2, **kwargs
):
    if ax is None:
        ax = _default_ax()

    if color is None:
        color = "C0"

    if isinstance(line, MultiLineString):
        path = Path.make_compound_path(
            *[Path(np.asarray(mline.coords)[:, :2]) for mline in line.geoms]
        )
    else:
        path = Path(np.asarray(line.coords)[:, :2])

    patch = PathPatch(
        path, facecolor="none", edgecolor=color, linewidth=linewidth, **kwargs
    )
    ax.add_patch(patch)
    ax.autoscale_view()

    if add_points:
        line = plot_points(line, ax=ax, color=color)
        return patch, line

    return patch


def plot_points(geom, ax=None, color=None, marker="o", **kwargs):
    if ax is None:
        ax = _default_ax()

    coords = get_coordinates(geom)
    (line,) = ax.plot(
        coords[:, 0], coords[:, 1], linestyle="", marker=marker, color=color, **kwargs
    )
    return line


class CanvasPlot:
    # __slots__ = "plot", "figure", "axes"

    def __init__(
        self,
        plot_object: Chart,
        figsize: tuple[int],
        axes: Optional[Axes] = None,
        **figure_kwargs,
    ) -> None:
        self.plot = plot_object
        if axes is None:
            _, self.axes = subplots(
                figsize=figsize, layout="constrained", **figure_kwargs
            )
        else:
            self.axes = axes

        self._set_canvas_props()

    def add_areas(
        self, alpha: float = 0.6, annotations: bool = False, **mpl_kwargs
    ) -> None:
        if self.plot.areas:
            for i, area in enumerate(self.plot.areas):
                plot_polygon(
                    area.geometry,
                    add_points=False,
                    ax=self.axes,
                    alpha=alpha if alpha else area.style.alpha,
                    facecolor=area.style.color if area.style.color else COLORS[i],
                    edgecolor="k",
                    label=str(area.label),
                    **mpl_kwargs,
                )
                if annotations:
                    self.axes.annotate(
                        str(area.label),
                        xy=area.centroid.coords[0],
                        xytext=(0, 0),
                        textcoords="offset points",
                        fontsize=18,
                        ha="center",
                        va="center",
                    )
        else:
            raise KeyError("No areas in the 'Chart'")

    def add_curves(
        self,
        annotations: bool = False,
        place: Literal["left", "right", "center"] = "left",
        **mpl_kwargs,
    ) -> None:
        if self.plot.curves:
            for _, curve in enumerate(self.plot.curves):
                label = str(curve.label) if annotations else ""
                self.axes.plot(
                    *curve.xy,
                    # add_points=False,
                    color=curve.style.color,
                    linewidth=curve.style.width,
                    label=label,
                    **mpl_kwargs,
                )

                if annotations:
                    xy, xytext = annotation_helper(curve, place)
                    self.axes.annotate(
                        label,
                        xy=xy,
                        xytext=xytext,
                        textcoords="offset points",  # "offset fontsize"
                        fontsize=14,
                        ha="center",
                        va="center",
                    )
        else:
            raise KeyError("No curves in the 'Chart'")

    def add_points(self, color: str, annotations: bool = False, **mpl_kwargs) -> None:
        if self.plot.points:
            for _, point in enumerate(self.plot.points):
                coords = get_coordinates(point)
                self.axes.plot(
                    coords[:, 0],
                    coords[:, 1],
                    linestyle="",
                    marker="o",
                    color=color,
                    edgecolors="k",
                    label=str(point.label) if annotations else "",
                    **mpl_kwargs,
                )
        else:
            raise KeyError("No points in the 'Chart'")

    def _set_canvas_props(
        self,
        title: str,
        labels: Iterable[str],
        grid: bool = False,
        log: bool = False,
    ) -> None:
        self.axes.set(xlim=self.limits[0], ylim=self.limits[1])
        self.axes.set_title(title, fontweight="bold", fontsize=18)
        self.axes.tick_params(axis="both", labelsize=14)
        if grid:
            self.axes.grid(True, which="major", axis="both", linestyle="-")
            self.axes.set_axisbelow(True)
        if log:
            self.axes.loglog()
        if labels:
            self.axes.set_xlabel(labels[0], fontweight="bold", fontsize=14)
            self.axes.set_ylabel(labels[1], fontweight="bold", fontsize=14)

    def draw(self, interactive: bool = False):
        self.axes.legend()

        figure = self.axes.get_figure()
        if interactive:
            figure.show()
        else:
            return figure, self.axes


class CanvasMap:
    # __slots__ = "plot", "figure", "axes"

    def __init__(
        self,
        map_object: Map,
        figsize: tuple[int],
        axes: Union[Axes, None] = None,
        **figure_kwargs,
    ) -> None:
        self.map = map_object
        self.legend_label = []
        if axes is None:
            _, self.axes = subplots(
                figsize=figsize, layout="constrained", **figure_kwargs
            )
        else:
            self.axes = axes

        self._set_canvas_props()

    def add_shapes(
        self, label: str, alpha: float, color: str, legend: bool = False, **mpl_kwargs
    ) -> None:
        if self.map.elements:
            for i, shape in enumerate(self.map.elements):
                color = color if color else COLORS[i]
                shape.plot(
                    alpha=alpha,
                    color=color,
                    edgecolor="black",
                    linewidth=width,
                    ax=self.axes,
                    label=label,
                    **mpl_kwargs,
                )
                # if legend:
                #     self.legend_label.append(
                #         self.__legend_patches(shape)(
                #             alpha=alpha,
                #             facecolor=color,
                #             edgecolor="black",
                #             label=fill(label, 20),
                #             [0],
                #             [0],
                #             marker="o",
                #             color="black",
                #             markerfacecolor="black",
                #             markersize=16,
                #             ls="",
                #         )
                #     )
        else:
            raise KeyError("No shapes in the 'Map'")

    def _set_canvas_props(
        self,
        title: str,
        labels: Iterable[str],
        grid: bool = False,
        log: bool = False,
    ) -> None:
        self.axes.set(xlim=self.limits[0], ylim=self.limits[1])
        self.axes.set_title(title, fontweight="bold", fontsize=18)
        self.axes.tick_params(axis="both", labelsize=14)
        if grid:
            self.axes.grid(True, which="major", axis="both", linestyle="-")
            self.axes.set_axisbelow(True)
        if log:
            self.axes.loglog()
        if labels:
            self.axes.set_xlabel(labels[0], fontweight="bold", fontsize=14)
            self.axes.set_ylabel(labels[1], fontweight="bold", fontsize=14)

    def draw(self, title: str, interactive: bool = False):
        self.axes.set_title(title, fontweight="bold", fontsize=18)
        self.axes.legend(
            handles=self.legend_label,
            title="Условные обозначения",
            fontsize=16,
            title_fontsize=20,
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            borderaxespad=0.0,
            frameon=False,
            handleheight=2,
            handlelength=3,
        )

        figure = self.axes.get_figure()
        if interactive:
            figure.show()
        else:
            return figure, self.axes
