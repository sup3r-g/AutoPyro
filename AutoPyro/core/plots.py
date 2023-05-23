import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, Union
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import odr
from scipy.optimize import curve_fit
from scipy.stats import rankdata
from shapely import contains, distance
from shapely.geometry import (
    LineString,
    Point,
    Polygon,
)
from shapely.plotting import plot_line, plot_points, plot_polygon

from AutoPyro.core.functions import MODELS


class Equation:
    __slots__ = "curve_type", "params"

    def __init__(
        self,
        curve_type: Union[str, None] = None,
        params: Union[list[float], None] = None,
    ) -> None:
        self.curve_type = curve_type
        self.params = params

    @classmethod
    def from_dict(cls, init_dict: dict) -> "Equation":
        return cls(
            init_dict["curve_type"],
            init_dict["params"],
        )

    def get_dict(self) -> dict:
        return {"curve_type": self.curve_type, "params": self.params}

    def get_tuple(self) -> tuple:
        return (self.curve_type, self.params)

    def copy(self) -> "Equation":
        return Equation(self.curve_type, self.params)


class Label:
    __slots__ = "name", "value"

    def __init__(self, name: str, value: Union[str, int]) -> None:
        self.name = name
        self.value = value

    @classmethod
    def from_dict(cls, init_dict: dict) -> "Label":
        return cls(
            init_dict["name"],
            init_dict["value"],
        )

    def __str__(self) -> str:
        if isinstance(self.value, (list, tuple)):
            value = ", ".join(self.value)
        else:
            value = self.value

        return f"Label {self.name}: {value}"

    def string(self, name: bool = False) -> str:
        if isinstance(self.value, (list, tuple)):
            value = ", ".join(self.value)
        else:
            value = self.value

        if name:
            return f"{self.name}: {value}"
        else:
            return str(value)

    def get_dict(self) -> dict:
        return {"name": self.name, "value": self.value}

    def get_tuple(self) -> tuple:
        return (self.name, self.value)

    def copy(self) -> "Label":
        return Label(self.name, self.value)


class LabelledPoint:
    __slots__ = "geometry", "label"

    def __init__(self, x: float, y: float, label: Label | None = None) -> None:
        self.geometry = Point(x, y)
        self.label = label

    def __str__(self) -> str:
        return f"Point {self.get_coords()}\n{str(self.label)}"

    @classmethod
    def from_json(cls, file_path: Union[str, os.PathLike]) -> "LabelledPoint":
        with open(file_path, "r", encoding="utf-8") as fp:
            init_dict = json.load(fp=fp)

        return cls.from_dict(init_dict)

    @classmethod
    def from_dict(cls, init_dict: dict) -> "LabelledPoint":
        return cls(
            init_dict["x"],
            init_dict["y"],
            Label.from_dict(init_dict["label"]),
        )

    def has_label(self) -> bool:
        return bool(self.label)

    def get_coords(self) -> tuple[float, float]:
        return self.geometry.x, self.geometry.y

    # def get_min_distance_curves(self, curves: list[LabelledCurve]) -> tuple:


class LabelledArea:
    __slots__ = "geometry", "label"

    def __init__(self, coordinates: Sequence, label: Label) -> None:
        self.geometry = Polygon(coordinates)
        self.label = label

    @classmethod
    def from_json(cls, file_path: Union[str, os.PathLike]):
        with open(file_path, "r", encoding="utf-8") as fp:
            init_dict = json.load(fp=fp)

        return cls.from_dict(init_dict)

    @classmethod
    def from_dict(cls, init_dict: dict):
        return cls(
            coordinates=list(zip(*init_dict["points"].values())),
            label=Label.from_dict(init_dict["label"]),
        )

    def contains_points(
        self, *points: list[LabelledPoint]
    ) -> tuple[np.array, list[LabelledPoint]]:
        points_geoms = [point.geometry for point in self.points]
        mask = np.nonzero(contains(self.geometry, points_geoms))[0]

        return mask, [points[i] for i in mask]


class LabelledCurve:
    __slots__ = (
        "geometry",
        "label",
        "color",
        "width",
        "equation",
    )

    def __init__(
        self,
        coordinates: Sequence,
        label: Label,
        color: Union[str, None] = None,
        width: Union[str, None] = None,
        equation: Union[Equation, None] = None,
    ) -> None:
        self.geometry = LineString(coordinates)
        self.label = label
        self.color = color
        self.width = width
        self.equation = equation

    def __str__(self) -> str:
        return f"Curve {str(self.label)}"

    # @classmethod
    # def from_function(cls, function: Callable, x: Sequence, *func_args):
    #     cls(function(x, *func_args))

    @classmethod
    def from_json(cls, file_path: Union[str, os.PathLike]):
        with open(file_path, "r", encoding="utf-8") as fp:
            init_dict = json.load(fp=fp)

        return cls.from_dict(init_dict)

    @classmethod
    def from_dict(cls, init_dict: dict):
        return cls(
            coordinates=list(zip(*init_dict["points"].values())),
            label=Label.from_dict(init_dict["label"]),
            color=init_dict["color"],
            width=init_dict["width"],
            equation=Equation.from_dict(init_dict["equation"])
            if init_dict["equation"]["curve_type"]
            else None,
        )

    def get_points(self) -> tuple[float, float]:
        return list(self.geometry.coords)

    def fit_ols(self, model="linear", initial_guess=None):
        X, Y = self.get_points()
        model_func = MODELS[model]
        params, _ = curve_fit(
            model_func,
            X,
            Y,
            p0=initial_guess if initial_guess else self.generate_initial_guess(),
        )
        self.curve_type = model
        self.params = params

        # Y_trend = self.model_func(X, *params)

        return self.model_func, self.params

    def fit_odr(self, model="linear", initial_guess=None):
        X, Y = self.get_points()
        model_func = odr.Model(MODELS[model])
        data = odr.Data(X, Y)
        odr = odr.ODR(
            data,
            model_func,
            beta0=initial_guess if initial_guess else self.generate_initial_guess(),
        )

        output = odr.run()

        self.curve_type = model
        self.params = output.beta

        return self.model_func, self.params

    def resample_equation(self, x_new) -> Sequence[float]:
        # Add Shapely interpolate method here
        if not self.equation:
            raise AttributeError("curve_type is not defined for this curve")

        return MODELS[self.curve_type](x_new, *self.params)

    def resample_interp(self, x_new):
        pass

    def generate_initial_guess():
        pass

    def get_normals(
        self, length: float = 50, direction: Literal["up", "down"] = "up"
    ) -> tuple[tuple, tuple]:
        x, y = self.get_points()
        x1, y1, x2, y2 = x[:-1], y[:-1], x[1:], y[1:]
        x_vect = x2 - x1
        y_vect = y2 - y1
        norm = (np.hypot(x_vect, y_vect) * 1 / length).flatten()

        if direction == "up":
            return (x1, x1 - y_vect / norm), (y1, y1 + x_vect / norm)
        elif direction == "down":
            return (x1, x1 + y_vect / norm), (y1, y1 - x_vect / norm)
        else:
            raise ValueError("Invalid 'direction' value")


class Plot:
    __slots__ = "name", "limits", "curves", "areas", "points", "figure"

    def __init__(
        self,
        name: str,
        limits: tuple,
        curves: list[LabelledCurve],
        areas: Union[list[LabelledArea], None] = None,
    ) -> None:
        self.name: str = name
        self.limits: tuple = limits
        self.curves: list[LabelledCurve] = curves
        self.areas: list[LabelledArea] = areas
        self.points: list[LabelledPoint] = []
        self.figure = None

    @classmethod
    def from_author(cls, author: str) -> "Plot":
        author_path = (
            Path(__file__).parents[1].joinpath("resources/plots", f"{author}.json")
        )
        if not author_path.exists():
            raise ValueError("Invalid 'author' value")

        return cls.from_json(author_path)

    @classmethod
    def from_json(cls, file_path: Union[str, os.PathLike]) -> "Plot":
        with open(file_path, "r", encoding="utf-8") as fp:
            init_dict = json.load(fp=fp)

        return cls.from_dict(init_dict)

    @classmethod
    def from_dict(cls, init_dict: dict) -> "Plot":
        return cls(
            init_dict["name"],
            limits=(init_dict["settings"]["xlim"], init_dict["settings"]["ylim"]),
            curves=[
                LabelledCurve.from_dict(curve)
                for curve in init_dict["data"]["curves"].values()
            ],
            areas=[
                LabelledArea.from_dict(area)
                for area in init_dict["data"]["areas"].values()
            ],
        )

    def add_points(self, *points) -> None:
        for point in points:
            if isinstance(point, LabelledPoint):
                self.points.append(point)
            else:
                raise TypeError(
                    "Only objects of type 'LabelledPoint' can be added",
                    f"Supplied object is of type {type(point)}",
                )

    def add_curves(self, *curves) -> None:
        for curve in curves:
            if isinstance(curve, LabelledCurve):
                self.curves.append(curve)
            else:
                raise TypeError(
                    "Only objects of type 'LabelledCurve' can be added",
                    f"Supplied object is of type {type(curve)}",
                )

    def add_areas(self, *areas) -> None:
        for area in areas:
            if isinstance(area, LabelledArea):
                self.areas.append(area)
            else:
                raise TypeError(
                    "Only objects of type 'LabelledArea' can be added",
                    f"Supplied object is of type {type(area)}",
                )

    def add(self, *components) -> None:
        for component in components:
            if isinstance(component, LabelledPoint):
                self.points.append(component)
            elif isinstance(component, LabelledCurve):
                self.curves.append(component)
            elif isinstance(component, LabelledArea):
                self.areas.append(component)
            else:
                raise TypeError(
                    "Only objects of types 'LabelledPoint', 'LabelledCurve', 'LabelledArea' can be added",
                    f"Supplied object is of type {type(component)}",
                )

    def get_points_dataframe(self) -> pd.DataFrame:
        data = []
        for point in self.points:
            data.append([*point.get_coords(), *point.label.get_tuple()])

        return pd.DataFrame(columns=[""])  # X_name, Y_name

    @staticmethod
    def resample_equal_points(
        curve_one: LabelledCurve,
        curve_two: LabelledCurve,
        strategy: Literal["longest", "shortest", "both"] = "longest",
        points_number: Union[int, None] = None,
    ) -> tuple[LabelledCurve, LabelledCurve]:
        if strategy == "longest":
            points_number = max(
                len(curve_one.geometry.coords), len(curve_two.geometry.coords)
            )
        if strategy == "shortest":
            points_number = min(
                len(curve_one.geometry.coords), len(curve_two.geometry.coords)
            )
        if strategy == "both" and not points_number:
            raise ValueError("Invalid 'strategy' value")

        steps = np.linspace(0, 1, points_number)

        return LabelledCurve(
            [curve_one.geometry.interpolate(step, normalized=True) for step in steps],
            curve_one.label.copy(),
        ), LabelledCurve(
            [curve_two.geometry.interpolate(step, normalized=True) for step in steps],
            curve_two.label.copy(),
        )

    @staticmethod
    def average_curves(
        curve_one: LabelledCurve, curve_two: LabelledCurve, ratios: list[float] = (0.5,)
    ) -> dict[str, LabelledCurve] | tuple[str, LabelledCurve]:
        if not isinstance(curve_one, LabelledCurve) or not isinstance(
            curve_two, LabelledCurve
        ):
            raise TypeError(
                "Both of the curves must be of type 'LabelledCurve'",
                f"Curve 1: {type(curve_one)}, Curve 2: {type(curve_two)}",
            )

        if curve_one.geometry.intersects(curve_two.geometry):
            xy = sorted(
                list(curve_one.geometry.intersection(curve_two.geometry).geoms),
                key=lambda curve: curve.xy[1],  # Value with maximum y coordinate
                reverse=True,
            )
            coords = list(curve_one.coords)
            for i, p in enumerate(coords):
                pd = curve_one.project(Point(p))
                if pd == distance:
                    curve_one = LineString(coords[: i + 1])
            coords = list(curve_two.coords)
            for i, p in enumerate(coords):
                pd = curve_one.project(Point(p))
                if pd == distance:
                    curve_two = LineString(coords[: i + 1])

        if len(curve_one.geometry.coords) != len(curve_two.geometry.coords):
            curve_one, curve_two = Plot.resample_equal_points(
                curve_one, curve_two, strategy="shortest"
            )

        curves = {}
        for ratio in ratios:
            curves[str(ratio)] = LabelledCurve(
                ratio * np.asarray(curve_one.geometry.coords)
                + (1 - ratio) * np.asarray(curve_two.geometry.coords),
                label=Label(curve_one.label.name, ratio),
            )

        return curves if len(curves) > 1 else tuple(curves.items())[0]

    def _list_geometries(
        self, component: Literal["curves", "points", "areas"]
    ) -> list[Any]:
        if hasattr(self, component):
            return [part.geometry for part in getattr(self, component)]
        else:
            raise AttributeError(f"Invalid 'component' {component} supplied")

    def get_ranked_distances(
        self, k: int = 2, return_indices_only: bool = False
    ) -> dict[str, dict]:
        if k > len(self.points):
            raise ValueError(
                "Number of distances must me less or equal (<=) to number of points"
            )

        points_geoms = [point.geometry for point in self.points]
        distances_all = np.array(
            [distance(curve.geometry, points_geoms) for curve in self.curves]
        )
        ranks = rankdata(distances_all, axis=0, method="dense", nan_policy="omit")
        indices = np.argwhere(np.isin(ranks, np.arange(1, k + 1)))

        # i - curve, j - point
        if return_indices_only:
            result = defaultdict(list)
            for i, j in indices:
                result[j].append(i)
        else:
            result = defaultdict(dict)
            for i, j in indices:
                result[j][i] = (ranks[i, j], distances_all[i, j])
            print("Schema: {point_id: {curve_id: (rank, distance)}}")

        return dict(result)

    @staticmethod
    def get_min_distances(
        points: list[LabelledPoint], curves: list[LabelledCurve]
    ) -> dict[str, dict]:
        points_geoms = [point.geometry for point in points]
        distances = np.array(
            [distance(curve.geometry, points_geoms) for curve in curves]
        )
        # i - curve, j - point
        return np.argmin(distances, axis=0)

    @staticmethod
    def get_ranked_distances_static(
        points: list[LabelledPoint],
        curves: list[LabelledCurve],
        k: int = 2,
        return_indices_only: bool = False,
    ) -> dict[str, dict]:
        if k > len(points):
            raise ValueError(
                "Number of distances must me less or equal (<=) to number of points"
            )

        points_geoms = [point.geometry for point in points]
        distances_all = np.array(
            [distance(curve.geometry, points_geoms) for curve in curves]
        )
        ranks = rankdata(distances_all, axis=0, method="dense", nan_policy="omit")
        indices = np.argwhere(np.isin(ranks, np.arange(1, k + 1)))

        # i - curve, j - point
        if return_indices_only:
            result = defaultdict(list)
            for i, j in indices:
                result[j].append(i)
        else:
            result = defaultdict(dict)
            for i, j in indices:
                result[j][i] = (ranks[i, j], distances_all[i, j])
            print("Schema: {point_id: {curve_id: (rank, distance)}}")

        return dict(result)

    def fit_all(
        self,
        model="linear",
        initial_guess=None,
        fit_type: Literal["OLS", "ODR"] = "OLS",
    ) -> None:
        for curve in self.curves:
            if fit_type == "OLS":
                curve.fit_ols(model, initial_guess)
            if fit_type == "ODR":
                curve.fit_odr(model, initial_guess)

    def classify_area(self, return_result: bool = True) -> dict[str, dict]:
        if not self.points or not self.areas:
            raise KeyError("Either points or areas are not present in the 'Plot'")

        points_geoms = [point.geometry for point in self.points]
        inclusions = np.array(
            [contains(area.geometry, points_geoms) for area in self.areas]
        )
        for i, j in np.argwhere(inclusions):
            self.points[j].label = self.areas[i].label.copy()

        if return_result:
            return [
                point.label.get_tuple()
                for point in self.points
                if point.has_label()
            ]

    def classify_distance(self, return_result: bool = True) -> tuple[str, float]:
        if not self.points or not self.curves:
            raise KeyError("Either points or curves are not present in the 'Plot'")

        indices = self.get_ranked_distances(k=1, return_indices_only=True)
        print(indices)

        for j, i in indices.items():
            self.points[j].label = self.curves[i[0]].label.copy()

        if return_result:
            return [
                point.label.get_tuple()
                for point in self.points
                if point.has_label()
            ]

    def plot(
        self,
        title: str,
        labels: tuple[str],
        figsize: tuple[int],
        grid=False,
        log=False,
        plot_curves: bool = True,
        plot_areas: bool = False,
        add_points: bool = False,
        ax: Union[Axes, None] = None,
        place: Literal["left", "right", "center"] = "left",
        **fig_kw,
    ) -> tuple[Figure, Axes]:
        colors = plt.get_cmap("hsv")(np.linspace(0, 1, 20))
        rng = np.random.default_rng()
        rng.shuffle(colors)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, layout="constrained", **fig_kw)

        ax.set(xlim=self.limits[0], ylim=self.limits[1])
        ax.set_title(title, fontweight="bold", fontsize=18)
        ax.tick_params(axis="both", labelsize=14)
        if grid:
            ax.grid(True, which="major", axis="both", linestyle="-")
            ax.set_axisbelow(True)
        if log:
            ax.loglog()
        if labels:
            ax.set_xlabel(labels[0], fontweight="bold", fontsize=14)
            ax.set_ylabel(labels[1], fontweight="bold", fontsize=14)

        if plot_areas and self.areas:
            for i, area in enumerate(self.areas):
                plot_polygon(
                    area.geometry,
                    add_points=False,
                    ax=ax,
                    alpha=0.6,
                    facecolor=colors[i],
                    edgecolor="k",
                    label=area.label.string(),
                )
                x, y = area.geometry.centroid.coords[0]
                # ax.annotate(
                #     area.label.string(),
                #     xy=(x, y),
                #     xytext=(0, 0),
                #     textcoords="offset points",
                #     fontsize=18,
                #     ha="center",
                #     va="center",
                # )

        if plot_curves and self.curves:
            for i, curve in enumerate(self.curves):
                ax.plot(
                    *curve.geometry.xy,
                    # add_points=False,
                    color=curve.color,
                    linewidth=curve.width,
                    # label=curve.label.string(True),
                )

                if place == "left":
                    x, y = list(curve.geometry.coords)[0]
                    x_p, y_p = 15, 10
                elif place == "right":
                    x, y = list(curve.geometry.coords)[-1]
                    x_p, y_p = -15, 10
                elif place == "center":
                    x, y = curve.geometry.centroid.coords[0]
                    x_p, y_p = 15, 10

                ax.annotate(
                    curve.label.string(),
                    xy=(x, y),
                    xytext=(x_p, y_p),
                    textcoords="offset points",
                    fontsize=14,
                    ha="center",
                    va="center",
                )

        if add_points and self.points:
            for i, point in enumerate(self.points):
                plot_points(
                    point.geometry,
                    ax=ax,
                    color="g",
                    edgecolors="k",
                    # label=point.label.string(),
                )

        ax.legend()

        return fig, ax


if __name__ == "__main__":
    author = "author"
    print(Plot.from_author(author))
