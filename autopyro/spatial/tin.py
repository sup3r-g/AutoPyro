"""Linear and cubic interpolation on a Delaunay triangulation."""
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
from scipy.spatial import QhullError, cKDTree


def predict_tin(xy, values, queries, _options):
    return _predict_triangulated(xy, values, queries, LinearNDInterpolator, support_size=3)


def predict_clough_tocher(xy, values, queries, _options):
    # Gradients of the cubic surface are estimated across the triangulation.
    return _predict_triangulated(xy, values, queries, CloughTocher2DInterpolator, support_size=len(xy))


def _predict_triangulated(xy, values, queries, interpolator, *, support_size):
    predicted = np.full(len(queries), np.nan)
    if len(xy) >= 3:
        try:
            predicted = interpolator(xy, values)(queries)
        except QhullError:
            pass
    valid = np.isfinite(predicted)
    distance = np.full(len(queries), np.inf)
    if valid.any():
        distance[valid], _ = cKDTree(xy).query(queries[valid])
    return predicted, np.where(valid, support_size, 0), distance
