from io import BytesIO

from AutoPyro.core.data import DataTable

class Map:
    
    # __slots__ = ()
    
    def __init__(self, parameter: str, data: DataTable) -> None:
        self.polygons = []
        self.data = data
    
    def interpolate(self, dependency):
        pass
    
    def to_shp(self, **kwargs):
        pass

    def to_image(self, format="png"):
        pass