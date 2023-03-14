from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import shapely

from AutoPyro.core.data import DataTable


@dataclass(slots=True)
class Map:
    
    # __slots__ = ()
    
    def __init__(self, parameter: str, geodata: str, data: DataTable) -> None:
        self.polygons = []
        self.geodata =  gpd.read_file(geodata)
        self.data = data
    
    def construct_map(self):
        pass
    
    def interpolate(self, dependency):
        pass
    
    def to_file(self, file_type="shp", **kwargs):
        for key, values in grouped:
            values.to_file(f"{key}.{file_type}")

    def to_image(self, format="png"):
        pass