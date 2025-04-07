from dataclasses import dataclass
from typing import Any, Optional, Sequence, TypeAlias

from matplotlib.artist import Artist, ArtistInspector
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.typing import ColorType
from shapely import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry


class Style:
    __slots__ = "valid_properties", "values"

    MATPLOTLIB_SHAPES_MAP = {
        LineString: Line2D,
        Point: Line2D,
        Polygon: PathPatch,
    }

    # color: ColorType
    # edgecolor: Any
    # facecolor: Any
    # width: float
    # joinstyle: str = "miter"
    # alpha: Optional[float] = None
    # capstyle: str = "butt"
    # fillstyle: str = "full"
    # linestyle: str = "-"
    # linewidth: float = 1.5
    # # Marker
    # marker: Optional[str] = None
    # markeredgecolor: ColorType = "C0"
    # markeredgewidth: float = 1.0
    # markerfacecolor: ColorType = "C0"
    # markerfacecoloralt: str = "none"
    # markersize: float = 6.0
    # # Solid
    # solid_capstyle: str = "projecting"
    # solid_joinstyle: str = "round"
    # # Dash
    # dash_capstyle: str = "butt"
    # dash_joinstyle: str = "round"

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
        "markevery",  # ?
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

    def __init__(self, obj: BaseGeometry | Artist, **style_kwargs: Any) -> None:
        if isinstance(obj, BaseGeometry):
            obj = self.MATPLOTLIB_SHAPES_MAP.get(obj, Line2D)

        inspector = ArtistInspector(obj)
        self.valid_properties = {
            arg: inspector.get_valid_values(arg)
            for arg in inspector.get_setters()
            if arg not in self.IGNORE
        }
        self.values = {
            k: v for k, v in style_kwargs.items() if k in self.valid_properties
        }

    def __getitem__(self, item):
        return getattr(self.values, item)

    def update(self, values: dict[str, Any]) -> None:
        self.values.update(values)

    def validate(self, *style_args, **style_kwargs: Any) -> list[Any]:
        return [arg for arg in style_args if arg in self.valid_properties] + [
            arg for arg in style_kwargs.keys() if arg in self.valid_properties
        ]

    def kwargs_passthrough(self, kwargs, mpl_kwargs):
        """
        This will not modify kwargs for you.
        This function is taken from:
        https://github.com/mpl-extensions/mpl-interactions
        """

        kwargs = dict(kwargs)
        passthrough = {}
        for k in mpl_kwargs:
            if k in kwargs:
                passthrough[k] = kwargs.pop(k)

        return kwargs, passthrough


class AreasStyle(Style):
    def __init__(self, **style_kwargs: Any) -> None:
        super().__init__(Polygon, **style_kwargs)


class CurvesStyle(Style):
    def __init__(self, **style_kwargs: Any) -> None:
        super().__init__(LineString, **style_kwargs)


class PointsStyle(Style):
    def __init__(self, **style_kwargs: Any) -> None:
        super().__init__(Point, **style_kwargs)


JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None
LabelModel: TypeAlias = dict  # {"GENERATION POTENTIAL": ["Fair", "Poor"]}


@dataclass
class EquationModel:
    curve_type: Optional[str] = None
    params: Sequence[float] = ()  # [1.2, 4, 7.5]


@dataclass
class PointModel:
    x: Sequence[float]
    y: Sequence[float]
    label: Optional[LabelModel]
    name: str = ""


@dataclass
class CurveModel:
    style: Style
    label: Optional[LabelModel]
    divider: Optional[bool]
    equation: EquationModel
    points: Sequence[PointModel]
    name: str = ""  # "GENERATION POTENTIAL: Very good, Good"


@dataclass
class AreaModel:
    label: Optional[LabelModel]
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
