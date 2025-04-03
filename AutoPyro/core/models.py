from dataclasses import dataclass
from typing import Any, Optional, Sequence
from matplotlib.artist import ArtistInspector
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.typing import ColorType
from shapely import LineString, Point, Polygon


@dataclass
class LabelModel:
    name: str  # "GENERATION POTENTIAL"
    value: str | Sequence[str]  # ["Fair", "Poor"]


@dataclass
class EquationModel:
    curve_type: Optional[str] = None
    params: Sequence[float] = ()  # [1.2, 4, 7.5]


# @dataclass
# class Style:
#     color: ColorType
#     edgecolor: Any
#     facecolor: Any
#     width: float
#     joinstyle: str = "miter"
#     alpha: Optional[float] = None
#     capstyle: str = "butt"
#     fillstyle: str = "full"
#     linestyle: str = "-"
#     linewidth: float = 1.5
#     # Marker
#     marker: Optional[str] = None
#     markeredgecolor: ColorType = "C0"
#     markeredgewidth: float = 1.0
#     markerfacecolor: ColorType = "C0"
#     markerfacecoloralt: str = "none"
#     markersize: float = 6.0
#     # Solid
#     solid_capstyle: str = "projecting"
#     solid_joinstyle: str = "round"
#     # Dash
#     dash_capstyle: str = "butt"
#     dash_joinstyle: str = "round"


class Style:

    MATPLOTLIB_SHAPES_MAP = {
        LineString: Line2D,
        Point: Line2D,
        Polygon: PathPatch,
    }

    IGNORE = (
        "agg_filter",
        "animated",
        "antialiased",
        "clip_box",
        "clip_on",
        "clip_path",
        "data",
        "figure",
        "gid",
        "in_layout",
        "label",
        "markevery",
        "mouseover",
        "path_effects",
        "picker",
        "pickradius",
        "rasterized",
        "sketch_params",
        "snap",
        "transform",
        "url",
        "visible",
        "xdata",
        "ydata",
        "zorder",
    )

    def __init__(self, geometry) -> None:
        inspector = ArtistInspector(geometry)
        self.arguments = inspector.get_setters()
        self.valid_values = [inspector.get_valid_values(arg) for arg in self.arguments]

    def __getitem__(self, item):
        return getattr(self, item)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def verify(self, *style_args, **style_kwargs) -> list[Any]:
        return [arg for arg in style_args if arg in self.arguments] + [
            arg for arg in style_kwargs.keys() if arg in self.arguments
        ]


class AreasStyle:
    pass


class CurvesStyle:
    pass


class PointsStyle:
    pass


@dataclass
class PointModel:
    x: Sequence[float]
    y: Sequence[float]
    label: LabelModel
    name: str = ""


@dataclass
class CurveModel:
    style: Style
    label: LabelModel
    divider: Optional[bool]
    equation: EquationModel
    points: Sequence[PointModel]
    name: str = ""  # "GENERATION POTENTIAL: Very good, Good"


@dataclass
class AreaModel:
    label: LabelModel
    # equation: EquationModel
    points: Sequence[PointModel]
    name: str = ""  # "GENERATION POTENTIAL: Fair"


@dataclass
class DataModel:
    curves: Sequence[CurveModel]
    areas: Sequence[AreaModel]
    points: Sequence[PointModel]


@dataclass
class PlotSettingsModel:
    xlim: tuple[float, float]  # (0.1, 1000.0)
    ylim: tuple[float, float]  # (0.1, 1000.0)
    log: bool
    grid: bool
    legend: bool
    zlim: Optional[tuple[float, float]] = None  # (0.1, 1000.0)


@dataclass
class PlotModel:
    name: str
    title: str
    settings: PlotSettingsModel
    labels: dict[str, Sequence[str]]
    data: DataModel
