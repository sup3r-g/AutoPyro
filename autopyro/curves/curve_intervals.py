"""Conservative interval search using the 1-Lipschitz curve-distance bound.

For a cell with centre c and half-diagonal r, every curve distance lies in
[d(c)-r, d(c)+r]. A curve with d(c)>min(d(c))+2r+tolerance cannot win anywhere
in that cell. Subdivision tightens the enclosure; a bounded work budget never
turns unexamined space into a claimed unique answer.
"""
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import Point, LineString, box

from autopyro.schemes.interpretation import transform


def analysis_bounds(sample, scheme):
    bounds = []
    for key, value, declared in (('x', sample.x, sample.x_bounds), ('y', sample.y, sample.y_bounds)):
        value = float(value)
        low, high = map(float, declared if declared is not None else (value, value))
        if not np.isfinite([low, high]).all() or not low <= value <= high:
            raise ValueError('Интервал должен быть конечным, упорядоченным и содержать измерение')
        bounds.append((transform(low, scheme[key]), transform(high, scheme[key])))
    return bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1]


def footprint(bounds):
    left,bottom,right,top = bounds
    if left != right and bottom != top:
        return box(left,bottom,right,top)
    if left != right or bottom != top:
        return LineString([(left,bottom),(right,top)])
    return Point(left,bottom)


@dataclass
class _Cell:
    bounds: tuple
    candidates: np.ndarray
    radius: float
    domain_proven: bool


class IntervalCurveSearch:
    def __init__(self, geometries, scheme, initial_hi=None):
        self.geometries = geometries
        self.tolerance = max(1e-12, scheme.get('distance_tolerance', 0))
        maximum = scheme.get('max_distance')
        self.maximum = float('inf') if maximum is None else maximum
        self.budget = scheme.get('interval_max_evaluations', 32)
        self.resolution = max(1e-9, scheme.get('hi_resolution', 0))
        self.initial_hi = initial_hi
        self.cache = {}

    def _evaluate(self, locations, geometries=None):
        points = shapely.points(np.asarray(locations))
        geometries = self.geometries if geometries is None else geometries
        return shapely.distance(points[:, None], geometries[None, :])

    def _cell(self, bounds, distances, indices=None):
        left,bottom,right,top = bounds
        radius = float(np.hypot(right-left, top-bottom) / 2)
        minimum = float(distances.min())
        candidates = np.flatnonzero(distances <= minimum + 2*radius + self.tolerance)
        if indices is not None:
            candidates = indices[candidates]
        return _Cell(bounds, candidates, radius, minimum + radius <= self.maximum + 1e-12)

    def search(self, bounds):
        if bounds in self.cache:
            return self.cache[bounds]
        result = self._search(bounds)
        if len(self.cache) >= 1024:
            self.cache.clear()
        self.cache[bounds] = result
        return result

    def _search(self, bounds):
        left,bottom,right,top = bounds
        locations = [((left+right)/2, (bottom+top)/2), (left,bottom),
                     (left,top), (right,bottom), (right,top)]
        distances = self._evaluate(locations)
        evaluations = len(locations)
        if np.any(distances.min(axis=1) > self.maximum + 1e-12):
            return None, {}, 'Часть интервала дальше допустимого расстояния от семейства кривых; экстраполяция запрещена'
        observed = set(np.where(distances <= distances.min(axis=1)[:,None] + self.tolerance)[1].tolist())
        cells = [self._cell(bounds, distances[0])]
        # Curves excluded on the whole root box can never win in a child box.
        active = cells[0].candidates
        active_geometries = self.geometries[active]
        exhausted = False
        while True:
            candidates = np.unique(np.concatenate([c.candidates for c in cells]))
            domain_proven = all(c.domain_proven for c in cells)
            if self.initial_hi is None:
                resolved = set(candidates).issubset(observed)
                gap = None
            else:
                outer = self.initial_hi[candidates]
                witnessed = self.initial_hi[list(observed)]
                gap = max(float(witnessed.min()-outer.min()), float(outer.max()-witnessed.max()), 0)
                resolved = gap <= 1e-9 or float(np.ptp(outer)) <= self.resolution
            if resolved and domain_proven:
                break
            refinable = []
            for i, cell in enumerate(cells):
                if self.initial_hi is None:
                    unresolved = not set(cell.candidates).issubset(observed)
                else:
                    values = self.initial_hi[cell.candidates]
                    unresolved = values.min() < witnessed.min() - 1e-9 or values.max() > witnessed.max() + 1e-9
                if cell.radius > 1e-12 and (not cell.domain_proven or not resolved and unresolved):
                    refinable.append((i, cell))
            if evaluations + 2 > self.budget or not refinable:
                exhausted = True
                break
            index, cell = max(refinable, key=lambda item: item[1].radius)
            a,b,c,d = cell.bounds
            if c-a >= d-b:
                middle=(a+c)/2
                children=[(a,b,middle,d),(middle,b,c,d)]
            else:
                middle=(b+d)/2
                children=[(a,b,c,middle),(a,middle,c,d)]
            centers=[((a+c)/2,(b+d)/2) for a,b,c,d in children]
            child_distances=self._evaluate(centers, active_geometries)
            evaluations += 2
            if np.any(child_distances.min(axis=1) > self.maximum + 1e-12):
                return None, {}, 'Внутри интервала есть координаты вне допустимого расстояния; экстраполяция запрещена'
            observed.update(active[np.where(child_distances <= child_distances.min(axis=1)[:,None] + self.tolerance)[1]].tolist())
            cells[index:index+1] = [self._cell(rect,row,active) for rect,row in zip(children,child_distances,strict=True)]
        if not all(c.domain_proven for c in cells):
            return None, {}, 'Для всего интервала не подтверждён допустимый предел расстояния; бюджет уточнения исчерпан'
        conservative = (not set(candidates).issubset(observed) if self.initial_hi is None else gap > 1e-9)
        details = dict(method='adaptive_lipschitz', conservative=conservative,
                       evaluations=evaluations, budget_exhausted=exhausted, hi_bound_gap=gap)
        return candidates, details, None
