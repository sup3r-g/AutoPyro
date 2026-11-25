"""Thin-plate spline and nearest-neighbor baselines for spatial comparisons."""
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
import shapely
from shapely.geometry import MultiPoint


def predict_rbf(xy, values, queries, _options, *, smoothing):
    estimates = np.full(len(queries), np.nan)
    distances = np.full(len(queries), np.inf)
    hull = MultiPoint(xy).convex_hull
    if hull.geom_type == 'Polygon':
        inside = shapely.covers(hull, shapely.points(queries))
        if inside.any():
            origin = xy[0]
            try:
                spline = RBFInterpolator((xy - origin) / 1000, values, smoothing=smoothing,
                                         kernel='thin_plate_spline')
                estimates[inside] = spline((queries[inside] - origin) / 1000)
            except np.linalg.LinAlgError:
                pass
            valid = np.isfinite(estimates)
            if valid.any():
                distances[valid], _ = cKDTree(xy).query(queries[valid])
    return estimates, np.where(np.isfinite(estimates), len(xy), 0), distances


def predict_nearest(xy, values, queries, options):
    distances, indices = cKDTree(xy).query(queries, distance_upper_bound=options.max_distance)
    valid = np.isfinite(distances)
    estimates = np.full(len(queries), np.nan)
    estimates[valid] = values[indices[valid]]
    return estimates, valid.astype(int), distances
