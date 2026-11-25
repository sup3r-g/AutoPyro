"""Batched interval glyphs and polygon intersection markers for interpretation."""
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

from autopyro.schemes.interpretation import (
    GeometricEngine, InterpretationInput, interpretation_binding_key, interval_marker_ids, saved_interval_markers,
)
from autopyro.schemes.measurement_uncertainty import measurement_interval, interval_from_metadata


def draw_measurement_error_bars(ax, project, points, column_ids, axes, columns):
    """Draw stored intervals on the requested coordinates, skipping unbound axes."""
    snapshot = project.view.get('measurement_uncertainty')
    metadata = [snapshot.get(cid, {}) for cid in column_ids] if snapshot is not None else None
    if metadata is not None and not any(metadata):
        return []
    units = [columns[cid].unit if cid is not None else None for cid in column_ids]
    segments = []
    warnings = {}
    for row_id, *coordinates in points:
        for index, (cid, value, unit, axis) in enumerate(zip(column_ids, coordinates, units, axes)):
            if cid is None:
                continue
            if metadata is None:
                bounds, _, warning = measurement_interval(project, row_id, cid, value, unit)
            else:
                item = metadata[index].get(row_id)
                if item is None:
                    continue
                bounds, warning = interval_from_metadata(item, value, unit)
            if bounds and axis.scale == 'log' and bounds[0] <= 0:
                bounds = None
                warning = 'Интервал неопределённости имеет неположительную границу на логарифмической оси'
            if warning:
                key = (cid, warning)
                warnings[key] = warnings.get(key, 0) + 1
            if bounds:
                low, high = coordinates.copy(), coordinates.copy()
                low[index], high[index] = bounds
                segments.append((low, high))
    if segments:
        style = dict(colors='#0072b2', alpha=.65, linewidths=1, zorder=1.5)
        if ax.name == '3d':
            from mpl_toolkits.mplot3d.art3d import Line3DCollection

            bars = Line3DCollection(segments, **style)
            ax.add_collection3d(bars)
        else:
            bars = LineCollection(segments, **style)
            ax.add_collection(bars)
            ax.autoscale_view()
        bars.set_gid(f'measurement-uncertainty-{ax.figure.axes.index(ax) + 1}')
    return [f'{columns[cid].name}: {warning} ({count})' for (cid, warning), count in warnings.items()]


def draw_uncertainty(ax, project, spec, artists):
    binding = spec.interpretation
    if not binding:
        return {}, 0
    scheme = binding['scheme']
    snapshot = project.view.get('measurement_uncertainty')
    if snapshot is not None and not any(snapshot.get(binding[f'{key}_id']) for key in ('x', 'y')):
        return {}, 0
    rectangles, segments, ids, centers, samples = [], [], [], [], []
    skipped_log = 0
    for artist in artists:
        for rid, (x, y) in zip(artist._autopyro_row_ids, artist.get_offsets(), strict=True):
            intervals = []
            for key, value in (('x', float(x)), ('y', float(y))):
                cid, unit = binding[f'{key}_id'], binding[f'{key}_unit']
                if snapshot is not None:
                    metadata = snapshot.get(cid, {}).get(rid)
                    interval = interval_from_metadata(metadata, value, unit)[0] if metadata else None
                else:
                    interval = measurement_interval(project, rid, cid, value, unit)[0]
                intervals.append(interval)
            xb, yb = intervals
            if xb is None and yb is None:
                continue
            xb, yb = xb or (float(x), float(x)), yb or (float(y), float(y))
            if ((spec.x_axis.scale == 'log' and xb[0] <= 0)
                    or (spec.y_axis.scale == 'log' and yb[0] <= 0)):
                skipped_log += 1
                continue
            if spec.show_uncertainty:
                if xb[0] != xb[1] and yb[0] != yb[1]:
                    rectangles.append([(xb[0], yb[0]), (xb[1], yb[0]), (xb[1], yb[1]), (xb[0], yb[1])])
                else:
                    segments.append([(xb[0], yb[0]), (xb[1], yb[1])])
            samples.append(InterpretationInput(rid, float(x), float(y), x_bounds=xb, y_bounds=yb))
            ids.append(rid)
            centers.append((float(x), float(y)))
    if rectangles:
        ax.add_collection(PolyCollection(rectangles, facecolors='#0072b2', edgecolors='#0072b2',
                                         alpha=.18, linewidths=.8, zorder=.6))
    if segments:
        ax.add_collection(LineCollection(segments, colors='#0072b2', alpha=.6, linewidths=1, zorder=.6))
    crossings = {}
    if samples:
        if snapshot is None:
            covered, cached = saved_interval_markers(project, binding, set(ids))
        else:
            covered, cached = project.view.get('interval_ambiguity', {}).get(
                interpretation_binding_key(binding), (set(), {}))
        missing = [sample for sample in samples if sample.row_id not in covered]
        crossings = {rid: cached[rid] for rid in ids if rid in cached}
        if missing:
            crossings.update({result.row_id: interval_marker_ids(vars(result))
                              for result in GeometricEngine().evaluate(missing, scheme)
                              if result.status == 'ambiguous'})
    if crossings:
        selected = np.asarray([p for rid, p in zip(ids, centers, strict=True) if rid in crossings])
        ax.scatter(selected[:,0], selected[:,1], s=(spec.marker_size + 4)**2, marker='D',
                   facecolors='none', edgecolors='#d55e00', linewidths=1.4, zorder=3)
    if rectangles or segments:
        ax.autoscale_view()
    return crossings, skipped_log
