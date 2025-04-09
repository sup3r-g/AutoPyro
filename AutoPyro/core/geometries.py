from collections import defaultdict
from typing import Any, Literal, Optional, Sequence

import numpy as np
from base import Direction, Equation, GeometryList, LabelGeometry, Labels, Style
from functions import MODELS, CurveFitter
from scipy.stats import rankdata
from shapely import (
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
    distance,
    line_interpolate_point,
)

__all__ = [
    "LabelPoint",
    "LabelArea",
    "LabelCurve",
    "LabelMultiPoint",
    "LabelMultiCurve",
    "resample_equal_points",
    "average_curves",
    "ranked_distances",
    "minimal_distances",
]


class LabelPoint(LabelGeometry):

    def __init__(
        self,
        x: float,
        y: float,
        label: Optional[Labels | dict[str, Any]] = None,
        style: Optional[Style | dict[str, Any]] = None,
        **properties,
    ) -> None:
        super().__init__(Point(x, y), label, style, **properties)


class LabelArea(LabelGeometry):
    # Area = Polygon

    def __init__(
        self,
        coordinates: Sequence,
        label: Optional[Labels | dict[str, Any]] = None,
        style: Optional[Style | dict[str, Any]] = None,
        **properties,
    ) -> None:
        super().__init__(Polygon(coordinates), label, style, **properties)

    # # Remove this method
    # def contains_points(
    #     self, *points: LabelPoint
    # ) -> tuple[npt.NDArray, list[LabelPoint]]:
    #     points_geoms = [point.geometry for point in points]
    #     mask = np.nonzero(contains(self.geometry, points_geoms))[0]

    #     return mask, [points[i] for i in mask]


class LabelCurve(LabelGeometry):
    __slots__ = "equation"
    # Curve = LineString

    def __init__(
        self,
        coordinates: Sequence,
        equation: Optional[Equation] = None,
        label: Optional[Labels | dict[str, Any]] = None,
        style: Optional[Style | dict[str, Any]] = None,
        **properties: Any,  # color: str, width: str
    ) -> None:
        super().__init__(LineString(coordinates), label, style, **properties)
        self.equation = equation

    def fit(
        self, strategy: Literal["ols", "odr"], model: str = "linear", initial_guess=None
    ):
        fitter = CurveFitter(*np.asarray(self))
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
        x, y = np.asarray(self)
        x1, y1, x2, y2 = x[:-1], y[:-1], x[1:], y[1:]
        x_vect, y_vect = x2 - x1, y2 - y1
        norm = (np.hypot(x_vect, y_vect) * 1 / length).flatten()

        if direction == "up":
            return (x1, x1 - y_vect / norm), (y1, y1 + x_vect / norm)
        if direction == "down":
            return (x1, x1 + y_vect / norm), (y1, y1 - x_vect / norm)

        raise ValueError("Invalid 'direction' value")


class LabelMultiPoint(LabelGeometry):

    def __init__(
        self,
        *points: Point,
        label: Optional[Labels | dict[str, Any]] = None,
        style: Optional[Style | dict[str, Any]] = None,
        **properties,
    ) -> None:
        super().__init__(MultiPoint(points), label, style, **properties)


class LabelMultiCurve(LabelGeometry):
    # MultiCurve = MultiLineString

    def __init__(
        self,
        *lines: LineString,
        label: Optional[Labels] = None,
        style: Optional[Style | dict[str, Any]] = None,
        **properties,
    ) -> None:
        super().__init__(MultiLineString(lines), label, style, **properties)


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
                label=curve.label,
            )
            for curve in curves
        )
    )


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
            label=Labels(curve_one.label, ratio),
        )

    return curves


def ranked_distances(
    points: Sequence[LabelPoint],
    curves: Sequence[LabelCurve],
    k: int = 2,
    indices_only: bool = False,
) -> dict[int, dict[int, Any] | list[int]]:
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


def minimal_distances(
    points: Sequence[LabelPoint], curves: Sequence[LabelCurve]
) -> Sequence[int]:
    points_geoms = GeometryList(points).geometries
    # [point.geometry for point in points]
    distances = [distance(curve.geometry, points_geoms) for curve in curves]
    # np.array()
    # i - curve, j - point
    return np.argmin(distances, axis=0)
