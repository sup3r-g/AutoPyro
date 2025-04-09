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

from matplotlib.artist import Artist, ArtistInspector
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.typing import ColorType

from shapely import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"


def stringify(value: Any, concatenator: str = ", "):
    return concatenator.join(value) if isinstance(value, (list, tuple)) else value

    # flat_dict = {}
    # for k, v in d.items():
    #     new_key = f"{parent_key}{k}_" if parent_key else k
    #     if isinstance(v, dict):
    #         flat_dict.update(flatten_dict(v, new_key))
    #     else:
    #         flat_dict[new_key[:-1]] = v
    # return flat_dict


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

    def update(self, value: Self | dict[str, Any]) -> None:
        if isinstance(value, Style):
            value = value.values

        self.values.update(value)

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


class Labels(dict):
    pass


class LabelGeometry(Serializable):
    __slots__ = "geometry", "_label", "_style"

    def __init__(
        self,
        geometry: BaseGeometry,
        label: Optional[Labels | dict[str, Any]] = None,
        style: Optional[Style | dict[str, Any]] = None,
        **properties: Any,
    ) -> None:
        super().__init__()
        # Initialized geometry only!
        self.geometry = geometry
        self._label = Labels(label if label is not None else {})
        self._style = style if style else Style(geometry, **properties)

    def __array__(self) -> npt.NDArray[Any]:
        # (self.geometry, dtype=np.object_)
        return np.asarray(self.geometry.coords)

    def __str__(self) -> str:
        return (
            f"Geometry: {str(self.geometry)}\n"
            f"Label: {str(self._label)}\n"
            f"Style: {str(self._style)}\n"
        )

    def __getattr__(self, attr) -> Any:
        return getattr(self.geometry, attr)

    def __iter__(self) -> Generator[Any, Any, None]:
        yield self.geometry.coords

    @property
    def __geo_interface__(self) -> dict[str, Any]:
        return gpd.GeoDataFrame(
            self.label, geometry=[self.geometry], index=[0]
        ).__geo_interface__

    @property
    def label(self) -> Labels:
        return self._label

    @label.setter
    def label(self, value) -> None:
        if not isinstance(value, (Labels, dict)):
            raise TypeError(
                f"Not 'Labels' or 'dict' type. Provided type: {type(value)}"
            )

        self._label.update(value)

    @property
    def style(self) -> Style | dict[str, Any]:
        return self._style

    @style.setter
    def style(self, value) -> None:
        if not isinstance(value, (Style, dict)):
            raise TypeError(f"Not 'Style' or 'dict' type. Provided type: {type(value)}")

        self._style.update(value)

    @classmethod
    def make(
        cls,
        geometry: BaseGeometry,
        *geometry_args: Any,
        label: Optional[Labels] = None,
        style: Optional[Style | dict[str, Any]] = None,
    ) -> Self:
        return cls(geometry(*geometry_args), label, style)

    @classmethod
    def from_iterables(
        cls, *geometries: BaseGeometry, labels: Optional[Labels] = None
    ) -> Self:
        return GeometryList(geometries)

    @classmethod
    def from_dict(cls, init_dict: dict[str, Any]) -> Self:
        series = gpd.GeoDataFrame.from_features(init_dict).iloc[0]

        return cls(series.pop("geometry"), Labels(series.to_dict()))

    # def to_dict(self) -> dict[str, Any]:
    #     return self.__geo_interface__


class GeometryList(list):
    def __init__(self, iterable: Optional[Iterable[LabelGeometry]] = None) -> None:
        super().__init__(
            (self.__validate(item) for item in iterable) if iterable is not None else ()
        )

    def __setitem__(self, index, item) -> None:
        super().__setitem__(index, self.__validate(item))

    def __array__(self) -> npt.NDArray[Any]:
        return np.asarray(self)  # dtype=np.object_

    @classmethod
    def create(cls, geometry: BaseGeometry, labels):
        return cls((LabelGeometry(geometry, label) for label in labels))

    @property
    def geometries(self) -> list[BaseGeometry]:
        return [item.geometry for item in self]

    @property
    def labels(self) -> list[Labels]:
        return [item.label for item in self]

    @property
    def styles(self) -> list[Style]:
        return [item.style for item in self]

    def insert(self, index, item) -> None:
        super().insert(index, self.__validate(item))

    def append(self, item) -> None:
        super().append(self.__validate(item))

    def to_pandas(self, geo: bool = True) -> gpd.GeoDataFrame | pd.DataFrame:
        if geo:
            return gpd.GeoDataFrame(
                [self.labels, self.styles], geometry=self.geometries
            )

        return pd.DataFrame(
            [
                [item.label for item in self],
                [item.style for item in self],
                [item.geometry for item in self],
            ]
        )

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


# Do I need this Base class at all?
class BaseCalculator:
    COLUMN_NAME = "NAME"

    # def __init__(self, column_name: str = COLUMN_NAME) -> None:
    #     self.column_name = column_name

    def __call__(self):
        pass
