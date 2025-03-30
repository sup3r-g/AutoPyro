from collections import defaultdict
from typing import Any, Literal, Optional, Self, Sequence

import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata
from shapely import (
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
    contains,
    distance,
    line_interpolate_point,
)

from AutoPyro.core.base import Direction, Equation, GeometryList, Label, LabelGeometry
from AutoPyro.core.charts import CurveFitter
from AutoPyro.core.functions import MODELS


class LabelPoint(LabelGeometry):

    def __init__(self, x: float, y: float, label: Optional[Label] = None) -> None:
        super().__init__(Point(x, y), label)

    # @classmethod
    # def from_dict(cls, init_dict: dict[str, Any]) -> Self:
    #     return cls(
    #         init_dict["x"],
    #         init_dict["y"],
    #         Label.from_dict(init_dict["label"]),
    #     )


class LabelArea(LabelGeometry):
    # Area = Polygon

    def __init__(self, coordinates: Sequence, label: Optional[Label] = None) -> None:
        super().__init__(Polygon(coordinates), label)

    # @classmethod
    # def from_dict(cls, init_dict: dict) -> Self:
    #     return cls(
    #         coordinates=list(zip(*init_dict["points"].values())),
    #         label=Label.from_dict(init_dict["label"]),
    #     )

    # Remove this method
    def contains_points(
        self, *points: LabelPoint
    ) -> tuple[npt.NDArray, list[LabelPoint]]:
        points_geoms = [point.geometry for point in points]
        mask = np.nonzero(contains(self.geometry, points_geoms))[0]

        return mask, [points[i] for i in mask]


class LabelCurve(LabelGeometry):
    __slots__ = "equation", "style"
    # Curve = LineString

    def __init__(
        self,
        coordinates: Sequence,
        label: Optional[Label] = None,
        equation: Optional[Equation] = None,
        **style: Any,  # color: str, width: str
    ) -> None:
        super().__init__(LineString(coordinates), label)
        self.equation = equation
        self.style = style

    # @classmethod
    # def from_function(cls, function: Callable, x: Sequence, *func_args):
    #     cls(function(x, *func_args))

    # @classmethod
    # def from_dict(cls, init_dict: dict[str, Any]) -> Self:
    #     return cls(
    #         coordinates=list(zip(*init_dict["points"].values())),
    #         label=Label.from_dict(init_dict["label"]),
    #         equation=(
    #             Equation.from_dict(init_dict["equation"])
    #             if init_dict["equation"]["curve_type"]
    #             else None
    #         ),
    #         color=init_dict["color"],
    #         width=init_dict["width"],
    #     )

    def fit(
        self, strategy: Literal["ols", "odr"], model: str = "linear", initial_guess=None
    ):
        fitter = CurveFitter(*np.asarray(self))  # self.points()
        if strategy == "ols":
            return fitter.fit_ols(model, initial_guess)

        if strategy == "odr":
            return fitter.fit_odr(model, initial_guess)

    def resample_equation(self, x_new: Sequence[float]) -> Sequence[float]:
        # Add Shapely interpolate method here
        if not self.equation:
            raise AttributeError("curve_type is not defined for this curve")

        return MODELS[self.curve_type](x_new, *self.params)

    def resample_interpolate(self, x_new: Sequence[float]):
        return line_interpolate_point(self.geometry, x_new, normalized=True)

    def normals(
        self, length: float = 50.0, direction: Direction = "up"
    ) -> tuple[tuple[list[tuple[float, float]], Any], ...]:
        x, y = np.asarray(self)  # self.points()
        x1, y1, x2, y2 = x[:-1], y[:-1], x[1:], y[1:]
        x_vect, y_vect = x2 - x1, y2 - y1
        norm = (np.hypot(x_vect, y_vect) * 1 / length).flatten()

        if direction == "up":
            return (x1, x1 - y_vect / norm), (y1, y1 + x_vect / norm)
        if direction == "down":
            return (x1, x1 + y_vect / norm), (y1, y1 - x_vect / norm)

        raise ValueError("Invalid 'direction' value")


class LabelMultiPoint(LabelGeometry):

    def __init__(self, *points: Point, label: Optional[Label] = None) -> None:
        super().__init__(MultiPoint(points), label)


class LabelMultiCurve(LabelGeometry):
    # MultiCurve = MultiLineString

    def __init__(self, *lines: LineString, label: Optional[Label] = None) -> None:
        super().__init__(MultiLineString(lines), label)


def resample_equal_points(
    *curves: LabelCurve,
    strategy: Literal["longest", "shortest", "both"] = "longest",
    points_number: Optional[int] = None,
) -> tuple[LabelCurve, ...]:
    if strategy == "longest":
        points_number = max((len(curve.coords) for curve in curves))
    elif strategy == "shortest":
        points_number = min((len(curve.coords) for curve in curves))
    elif strategy == "both" and not points_number:
        raise ValueError("Invalid 'strategy' value")

    steps = np.linspace(0, 1, points_number)

    return tuple(
        (
            LabelCurve(
                line_interpolate_point(curve, steps, normalized=True),
                curve.label.copy(),
            )
            for curve in curves
        )
    )


# FIX: What the actual fuck is this code here? God help me please
def average_curves(
    curve_one: LabelCurve, curve_two: LabelCurve, ratios: Sequence[float] = (0.5,)
) -> dict[str, LabelCurve]:
    if not isinstance(curve_one, LabelCurve) or not isinstance(curve_two, LabelCurve):
        raise TypeError(
            "Both of the curves must be of type 'LabelCurve'",
            f"Provided Types: {type(curve_one)}, {type(curve_two)}",
        )

    if len(curve_one.coords) != len(curve_two.coords):
        curve_one, curve_two = resample_equal_points(
            curve_one, curve_two, strategy="shortest"
        )

    curves = {}
    for ratio in ratios:
        curves[str(ratio)] = LabelCurve(
            ratio * np.asarray(curve_one.coords)
            + (1 - ratio) * np.asarray(curve_two.coords),
            label=Label(curve_one.label.name, ratio),
        )

    return curves


def ranked_distances(
    points: Sequence[LabelPoint],
    curves: Sequence[LabelCurve],
    k: int = 2,
    indices_only: bool = False,
) -> dict[str, dict]:
    if k > len(points):
        raise ValueError(
            "Number of distances must me less or equal (<=) to number of 'points'"
        )

    points_geoms = GeometryList(points).geometries
    # [point.geometry for point in points]
    distances = [distance(curve.geometry, points_geoms) for curve in curves]
    # np.array()
    ranks = rankdata(distances, axis=0, method="dense", nan_policy="omit")
    indices = np.argwhere(np.isin(ranks, np.arange(1, k + 1)))

    # i - curve, j - point
    if indices_only:
        result = defaultdict(list)
        for i, j in indices:
            result[j].append(i)
    else:
        result = defaultdict(dict)
        for i, j in indices:
            result[j][i] = (ranks[i, j], distances[i][j])  # [i, j]

        # {point_id: {curve_id: (rank, distance)}}

    return dict(result)


def minimal_distances(points: Sequence[LabelPoint], curves: Sequence[LabelCurve]):
    points_geoms = GeometryList(points).geometries
    # [point.geometry for point in points]
    distances = [distance(curve.geometry, points_geoms) for curve in curves]
    # np.array()
    # i - curve, j - point
    return np.argmin(distances, axis=0)
