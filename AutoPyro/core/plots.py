from pathlib import Path
from scipy import odr
from scipy.spatial.distance import cdist
import numpy as np

from dataclasses import dataclass

@dataclass(repr=True, slots=True)
class Point:
    x: float
    y: float
    label: str
    
    def get_coordinates(self) -> tuple[float, float]:
        return self.x, self.y


@dataclass()
class Curve:
    curve_type: str
    equation: str
    params: list[float]
    points: list[Point]
    
    def get_points(self) -> tuple[float, float]:
        return self.points.get_coordinates()
    
    def fit(self, model="linear", initial_guess=None):
        X, Y = self.get_points()
        model = odr.Model(MODELS[model])
        data = odr.Data(X, Y)
        odr_obj = odr.ODR(
            data, model, beta0=initial_guess if initial_guess else self.generate_initial_guess()
        )
    
        output = odr_obj.run()
        return output
    
    def resample(self, x_new):
        return MODELS[self.curve_type](x_new, *self.params)
    
    def generate_initial_guess():
        pass


class Plot:
    
    # __slots__ = ()
    
    def __init__(self) -> None:
        self.name: str = name
        self.curves: list[Curve] = curves
        self.points: list[Point] = points
        self.annotations: list[str] = annotations
        self.properties = properties

    @classmethod
    def from_file(self):
        with open() as file:
            file  
    
    def fit_all(self):
        for curve in self.curves:
            curve.fit(self.properties[curve])

    def get_normals(x, y, length=70):
        x1, y1, x2, y2 = x[:-1], y[:-1], x[1:], y[1:]
        x_vect = x2-x1
        y_vect = y2-y1
        norm = (np.hypot(x_vect, y_vect) * 1/length).flatten()

        return (x1, x1 + y_vect / norm), (y1, y1 - x_vect / norm)

    def get_min_distance(self, x: np.ndarray, y: np.ndarray, points: list[Point]):
        distance = cdist(
            np.asarray([p.get_coordinates() for p in points]),
            np.column_stack([x, y])
        )

        return np.nanargmin(distance, axis=1), np.nanmin(distance, axis=1)

    def classify(self) -> tuple:
        distances = {
            prop: self.get_min_distance(prop.x, prop.y, points) for prop in self.properties
        }
        
        distance_values = list(distances.values())
        ratio_val = distances[prop]/np.sum(distance_values)
        
        for prop in self.properties:
            if prop.cu
        
        return prop, ratio_val