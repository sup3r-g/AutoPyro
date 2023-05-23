import os
from pathlib import Path
from typing import Literal, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import polygonize, split, unary_union

from AutoPyro.core.data import DataTable
from AutoPyro.digitizers import MapDigitizer


class Map:
    __slots__ = "shapes"

    def __init__(self, shapes: gpd.GeoDataFrame) -> None:
        self.shapes = shapes

    @classmethod
    def from_files(
        cls,
        *files: list[Union[str, os.PathLike]],
        **named_files: dict[str, Union[str, os.PathLike]],
    ) -> "Map":
        shapes = {}
        for file in files:
            shapes[Path(file).stem] = gpd.read_file(file)

        for name, file in named_files.items():
            shapes[name] = gpd.read_file(file)

        return cls(shapes=shapes)

    # @classmethod
    # def from_digitizer(
    #     cls,
    #     tif_file: Union[str, os.PathLike],
    #     *extra_files: Union[str, os.PathLike]
    # ) -> "Map":
    #     digitizer = MapDigitizer.from_tif(tif_file)

    #     digitizer.create_polygons()

    def get_columns(
        self, select_geometry: Literal["base", "separator", "point"] = "base"
    ) -> list:
        if select_geometry == "base":
            return self.base_polygons.columns.to_list()
        if select_geometry == "separator":
            return self.separator_polygons.columns.to_list()
        if select_geometry == "point":
            return self.points.columns.to_list()

        raise AttributeError("Invalid dataframe selection")

    def get_column_values(
        self,
        column: str,
        select_geometry: Literal["base", "separator", "point"] = "base",
    ) -> pd.Series:
        if column not in self.get_columns(select_geometry):
            raise KeyError("Specified column not in the dataframe")

        if select_geometry == "base":
            return self.base_polygons[column]
        if select_geometry == "separator":
            return self.separator_polygons[column]
        if select_geometry == "point":
            return self.points[column]

        raise AttributeError("Invalid dataframe selection")

    def intersect_maps(
        self, base_names: list[str], separator_names: list[str], union: bool = False
    ) -> None:
        base_geometry = self.base_polygons[base_names]
        separator_geometry = self.separator_polygons[separator_names]
        if union:
            base_geometry = unary_union(base_geometry)

        split(base_geometry, separator_geometry)

    def unite_maps(
        self, base_names: list[str], separator_names: list[str], union: bool = False
    ) -> None:
        base_geometry = self.base_polygons[base_names]
        separator_geometry = self.separator_polygons[separator_names]
        if union:
            base_geometry = unary_union(base_geometry)

        split(base_geometry, separator_geometry)

    def add_points(
        self, X: Union[pd.Series, gpd.GeoSeries], Y: Union[pd.Series, gpd.GeoSeries]
    ) -> None:
        self.shapes["Points"] = gpd.GeoSeries([Point(x, y) for x, y in zip(X, Y)])

    def interpolate(self, dependency) -> None:
        pass

    def to_file(self, file_type="shp", **kwargs) -> None:
        gpd.to_file(f"dataframe.{file_type}", mode="w")

    def to_image(self, format="png", *args, **kwargs) -> None:
        fig, ax = plt.subplots(**kwargs)
        
        for entry in self.shapes:
            entry.plot(ax=ax)
        
        fig.savefig(*args, **kwargs)
