import json
import os
from copy import deepcopy
from enum import Enum
from itertools import chain
from typing import Any, Generator, Iterable, Optional, Self

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
from shapely import Geometry

from AutoPyro.core.charts import Chart
from AutoPyro.core.geometries import LabelPoint

# from shapely.geometry import mapping, shape


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"


class Serializable:

    def __slots(self):
        return chain.from_iterable(
            getattr(cls, "__slots__", tuple()) for cls in reversed(type(self).__mro__)
        )

    def to_dict(self, values_only: bool = True) -> dict[str, Any]:
        return {
            slot: (
                deepcopy(getattr(self, slot))
                if values_only
                else {
                    "value": (value := deepcopy(getattr(self, slot))),
                    "class": value.__class__.__name__,
                }
            )
            for slot in self.__slots()
        }

    def to_tuple(
        self, values_only: bool = True
    ) -> tuple[Any | dict[str, Any | str], ...]:
        # (self.name, self.value)
        return tuple(
            (
                (
                    deepcopy(getattr(self, slot))
                    if values_only
                    else {
                        "value": (value := deepcopy(getattr(self, slot))),
                        "class": value.__class__.__name__,
                        # "name": slot,
                    }
                )
                for slot in self.__slots()
            )
        )

    @classmethod
    def from_dict(cls, init_dict: dict[str, Any]) -> Self:
        return cls(**init_dict)

    @classmethod
    def from_json(cls, file_path: str | os.PathLike) -> Self:
        with open(file_path, "r", encoding="utf-8") as fp:
            return cls.from_dict(json.load(fp=fp))


class Equation(Serializable):
    __slots__ = "curve_type", "params"

    def __init__(
        self,
        curve_type: Optional[str] = None,
        params: Optional[list[float]] = None,
    ) -> None:
        super().__init__()
        self.curve_type = curve_type
        self.params = params

    def copy(self) -> Self:
        return self.__class__(self.curve_type, self.params)


class Label(Serializable):
    __slots__ = "name", "value"

    def __init__(self, name: str, value: str | int) -> None:
        super().__init__()
        self.name = name
        self.value = value

    def __str_value(self):
        return (
            ", ".join(self.value)
            if isinstance(self.value, (list, tuple))
            else self.value
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__} → {self.name}: {self.__str_value()}"

    def copy(self):
        return Label(self.name, self.value)


class LabelGeometry(Serializable):
    __slots__ = "geometry", "_label"

    def __init__(self, geometry: Geometry, label: Optional[Label] = None) -> None:
        super().__init__()
        self.geometry = geometry
        self._label = label if label else Label("", "")
        # self.style = Style()

    def __array__(self) -> npt.NDArray[Any]:
        # (self.geometry, dtype=np.object_)
        return np.asarray(self.geometry.coords)

    def __str__(self) -> str:
        return f"{str(self.geometry)} → {str(self.label)}"

    def __getattr__(self, attr) -> Any:
        return getattr(self.geometry, attr)

    def __iter__(self) -> Generator[Any, Any, None]:
        yield self.geometry.coords

    @property
    def __geo_interface__(self) -> dict[str, Any]:
        # return mapping(self.geometry)
        return gpd.GeoDataFrame(
            self.label.to_dict(), geometry=[self.geometry], index=[0]
        ).__geo_interface__

    @property
    def label(self) -> Label | None:
        return self._label

    @label.setter
    def label(self, value) -> None:
        self._label = value

    @classmethod
    def make(
        cls, geometry: Geometry, *geometry_args, label: Optional[Label] = None
    ) -> None:
        return cls(geometry(*geometry_args), label)

    @classmethod
    def from_dict(cls, init_dict: dict[str, Any]) -> Self:
        # return cls(shape(init_dict))
        series = gpd.GeoDataFrame.from_features(init_dict).iloc[0]

        return cls(series.pop("geometry"), Label.from_dict(series.to_dict()))

    # def to_dict(self) -> dict[str, Any]:
    #     return self.__geo_interface__

    def has_label(self) -> bool:
        return bool(self._label)


class GeometryList(list):
    def __init__(self, iterable: Optional[Iterable[LabelGeometry]] = None) -> None:
        super().__init__(
            (self.__validate(item) for item in iterable) if iterable is not None else ()
        )

    def __setitem__(self, index, item) -> None:
        super().__setitem__(index, self.__validate(item))

    def __array__(self) -> npt.NDArray[Any]:
        return np.asarray(self)  # dtype=np.object_

    @property
    def geometries(self) -> list[Geometry]:
        return [item.geometry for item in self]

    @property
    def labels(self) -> list[tuple]:
        return [item.label.to_tuple() for item in self]  # if item.has_label()

    def insert(self, index, item) -> None:
        super().insert(index, self.__validate(item))

    def append(self, item) -> None:
        super().append(self.__validate(item))

    def to_geopandas(self, pandas: bool = False) -> pd.Series:
        return gpd.GeoSeries(
            data=[item.geometry for item in self],
            index=[item.label.value for item in self],
            name=self[0].label.name,
        )
        # return pd.DataFrame(
        #     [
        #         [*item.coords, *item.label.tuple()]
        #         for item in self
        #     ],
        #     columns=[""],
        # )  # X_name, Y_name

    def extend(self, other) -> None:
        if isinstance(other, type(self)):
            super().extend(other)
        else:
            super().extend(self.__validate(item) for item in other)

    def __validate(self, value: Any) -> LabelGeometry:
        if isinstance(value, LabelGeometry):
            return value

        raise TypeError(
            f"Value of type 'LabelGeometry' expected, got {type(value).__name__} instead"
        )


class BaseCalculator:
    COLUMN_NAME = "NAME"

    # def __init__(self, column_name: str = COLUMN_NAME) -> None:
    #     self.column_name = column_name

    def __call__(self):
        pass


# TODO: Completely rework this shit!!!
class BaseClassifier:
    COLUMN_NAME = "Base"

    @staticmethod
    def classify_single(
        X: float | npt.NDArray[np.float_],
        Y: float | npt.NDArray[np.float_],
        author: str,
    ) -> list[tuple[Any, ...]] | None:
        plot = Chart.from_author(author)
        plot.add(
            *[
                LabelPoint(x, y, Label("", ""))
                for x, y in zip(X, Y)
                if not np.isnan(x) and not np.isnan(y)
            ]
        )
        if plot.areas:
            return plot.classify_area()

        return plot.classify_distance()

    @classmethod
    def classify_multi(
        cls,
        X: float | npt.NDArray[np.float_],
        Y: float | npt.NDArray[np.float_],
        *authors: list[str],
        return_all: bool = False,
    ) -> list[list[tuple[Any, ...]] | None] | tuple[NDArray[intp], ...]:
        statistics = [cls.classify_single(X, Y, author) for author in authors]
        labels_authors = np.array(
            [point["value"] for info in statistics for point in info]
        )

        labels, ratios = np.unique(labels_authors, return_counts=True)
        mode = np.argwhere(ratios == np.max(ratios))
        ratios /= labels_authors.shape[0]

        if return_all:
            return statistics

        return mode, (labels, ratios)
