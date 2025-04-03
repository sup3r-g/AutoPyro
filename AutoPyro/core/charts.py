from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Self

import numpy as np
import pandas as pd
from shapely import contains

from base import GeometryList, Serializable
from geometries import LabelArea, LabelCurve, LabelPoint, ranked_distances

COMPONENTS = Literal["curves", "areas", "points"]


class Chart(Serializable):
    __slots__ = "name", "limits", "curves", "areas", "points"

    def __init__(
        self,
        name: str,
        # limits: tuple,
        curves: Iterable[LabelCurve],
        areas: Optional[Iterable[LabelArea]] = None,
        points: Optional[Iterable[LabelPoint]] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.curves = GeometryList(curves)
        self.areas = GeometryList(areas)
        self.points = GeometryList(points)

    @classmethod
    def from_author(cls, author: str) -> Self:
        return cls.from_json(
            Path(__file__).parents[1].joinpath("resources/json", f"{author}.json")
        )

    @classmethod
    def from_dict(cls, init_dict: dict[str, Any]) -> Self:
        return cls(
            init_dict["name"],
            # limits=(init_dict["settings"]["xlim"], init_dict["settings"]["ylim"]),
            curves=[
                LabelCurve.from_dict(curve)
                for curve in init_dict["data"]["curves"].values()
            ],
            areas=[
                LabelArea.from_dict(area)
                for area in init_dict["data"]["areas"].values()
            ],
        )

    def add(self, *components: LabelPoint | LabelCurve | LabelArea) -> None:
        for component in components:
            if isinstance(component, LabelPoint):
                self.points.append(component)
            elif isinstance(component, LabelCurve):
                self.curves.append(component)
            elif isinstance(component, LabelArea):
                self.areas.append(component)
            else:
                raise TypeError(
                    "Only objects of types "
                    "'LabelPoint', 'LabelCurve', 'LabelArea' can be added.",
                    f"Supplied object is of type {type(component)}",
                )

    def to_dataframe(
        self, component: COMPONENTS = "points", **geopandas_kwargs
    ) -> pd.DataFrame:
        return getattr(self, component).to_geopandas(**geopandas_kwargs)

    def geometries(self, component: COMPONENTS):
        try:
            return getattr(self, component).geometries
        except AttributeError as exc:
            raise AttributeError(f"Invalid 'component' {component} supplied") from exc

    def fit(
        self,
        model: str = "linear",
        initial_guess: Optional[Iterable[float]] = None,
        fit_type: Literal["OLS", "ODR"] = "OLS",
    ) -> None:
        for curve in self.curves:
            if fit_type == "OLS":
                curve.fit_ols(model, initial_guess)
            if fit_type == "ODR":
                curve.fit_odr(model, initial_guess)

    def classify_area(self, return_result: bool = True) -> list[tuple[Any, ...]] | None:
        if not self.points or not self.areas:
            raise KeyError("Either points or areas are not present in the 'Chart'")

        points_geoms = self.points.geometries
        inclusions = [contains(area.geometry, points_geoms) for area in self.areas]
        # np.array()

        for i, j in np.argwhere(inclusions):
            self.points[j].label = self.areas[i].label.copy()

        if return_result:
            return self.points.labels

    def classify_distance(
        self, return_result: bool = True
    ) -> list[tuple[Any, ...]] | None:
        if not self.points or not self.curves:
            raise KeyError("Either points or curves are not present in the 'Chart'")

        indices = ranked_distances(self.points, self.curves, k=1, indices_only=True)

        for j, i in indices.items():
            self.points[j].label = self.curves[i[0]].label.copy()

        if return_result:
            return self.points.labels


if __name__ == "__main__":
    print(Chart.from_author(author="author"))
