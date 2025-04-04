import os
from pathlib import Path
from typing import Any, Literal, Self, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.ops import polygonize, split, unary_union
from shapely.geometry import MultiPolygon, Point, Polygon

from base import Style
from data import DataTable
from digitizers import MapDigitizer


class MapElement:
    __slots__ = "name", "data", "style"

    def __init__(self, name: str, data: gpd.GeoDataFrame) -> None:
        self.name = name
        self.data = data
        self.style = Style()

    @property
    def __geo_interface__(self) -> dict[str, Any]:
        return self.data.__geo_interface__

    @classmethod
    def from_file(cls, file: str | os.PathLike, **geopandas_kwargs) -> Self:
        return cls(name = Path(file).stem, gpd.read_file(file, **geopandas_kwargs))


class Map:
    # __slots__ = "elements"

    def __init__(self, **elements: MapElement) -> None:
        self.elements = elements

    @classmethod
    def from_files(
        cls,
        *files: str | os.PathLike,
        **named_files: str | os.PathLike,
    ) -> Self:
        elements = {}

        if files:
            for file in files:
                elements[Path(file).stem] = gpd.read_file(file)

        if named_files:
            for name, file in named_files.items():
                elements[name] = gpd.read_file(file)

        return cls(*elements)

    # @classmethod
    # def from_digitizer(cls, digitizer: MapDigitizer) -> Self:
    #     cls(*digitizer.create_polygons())

    def get_columns(
        self, select_geometry: Literal["base", "separator", "point"] = "base"
    ) -> list:
        element = self.elements.get(select_geometry)
        if element is not None:
           return element.data.columns.to_list()

        raise AttributeError("Invalid dataframe selection")

    def get_column_values(
        self,
        column: str,
        select_geometry: Literal["base", "separator", "point"] = "base",
    ) -> pd.Series:
        if column not in self.get_columns(select_geometry):
            raise KeyError("Specified column not in the dataframe")

        element = self.elements.get(select_geometry)
        if element is not None:
           return element.data[column]

        raise AttributeError("Invalid dataframe selection")

    def intersect(
        self, base_names: list[str], separator_names: list[str], union: bool = False
    ) -> None:
        base_geometry = self.elements["base"][base_names]
        separator_geometry = self.elements["separator"][separator_names]
        if union:
            base_geometry = unary_union(base_geometry)

        split(base_geometry, separator_geometry)

    def unite(
        self, base_names: list[str], separator_names: list[str], union: bool = False
    ) -> None:
        pass

    def add_points(
        self, X: Union[pd.Series, gpd.GeoSeries], Y: Union[pd.Series, gpd.GeoSeries]
    ) -> None:
        self.elements["points"] = gpd.GeoSeries([Point(x, y) for x, y in zip(X, Y)])

    def to_file(self, file_type="shp", **kwargs) -> None:
        gpd.to_file(f"dataframe.{file_type}", mode="w", **kwargs)

    def to_image(self, format="png", *args, **kwargs) -> None:
        fig, ax = plt.subplots(**kwargs)

        for entry in self.elements:
            entry.plot(ax=ax)

        fig.savefig(*args, **kwargs)
