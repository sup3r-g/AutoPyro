from dataclasses import dataclass
from typing import Any, Optional, Sequence
from matplotlib.typing import ColorType


@dataclass
class LabelModel:
    name: str  # "GENERATION POTENTIAL"
    value: str | Sequence[str]  # ["Fair", "Poor"]


@dataclass
class EquationModel:
    curve_type: Optional[str] = None
    params: Sequence[float] = []  # [1.2, 4, 7.5]


@dataclass
class StyleModel:
    color: ColorType
    edgecolor: Any
    facecolor: Any
    width: float
    joinstyle: str = 'miter'
    alpha: Optional[float] = None
    capstyle: str = 'butt'
    fillstyle: str = "full"
    linestyle: str = "-"
    linewidth: float = 1.5
    # Marker
    marker: Optional[str] = None
    markeredgecolor: ColorType = "C0"
    markeredgewidth: float = 1.0
    markerfacecolor: ColorType = "C0"
    markerfacecoloralt: str = "none"
    markersize: float = 6.0
    # Solid
    solid_capstyle: str = "projecting"
    solid_joinstyle: str = "round"
    # Dash
    dash_capstyle: str = "butt"
    dash_joinstyle: str = "round"


@dataclass
class PointModel:
    x: Sequence[float]
    y: Sequence[float]
    label: LabelModel
    name: str = ""


@dataclass
class CurveModel:
    style: StyleModel
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
