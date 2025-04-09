from dataclasses import dataclass
from typing import Any, Optional, Sequence, TypeAlias


@dataclass
class AreaStyle:
    pass


@dataclass
class CurveStyle:
    pass


@dataclass
class PointStyle:
    pass


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
    style: Optional[PointStyle]
    z: Optional[Sequence[float]] = None


@dataclass
class CurveModel:  # "GENERATION POTENTIAL: Very good, Good"
    points: Sequence[PointModel]
    equation: EquationModel
    divider: Optional[bool]
    label: Optional[LabelModel]
    style: Optional[CurveStyle]


@dataclass
class AreaModel:  # "GENERATION POTENTIAL: Fair"
    points: Sequence[PointModel]
    label: Optional[LabelModel]
    style: Optional[AreaStyle]


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
