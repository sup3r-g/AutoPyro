"""Project sample positions for standalone stratigraphy tracks."""

from collections import defaultdict
from typing import NamedTuple

from autopyro.analysis.statistics import classify
from .stratigraphy import UNITS, validate_binding


class SamplePosition(NamedTuple):
    row_id: str
    top: float
    bottom: float | None


def sample_positions(project, row_ids, scheme, binding, wells, *, checkpoint=None):
    validate_binding(project, scheme, binding)
    columns = {column.id: column for column in project.columns}
    well_id = binding['well_id']
    depth_ids = [binding['depth_id']]
    if binding.get('sample_mode') == 'interval':
        depth_ids.append(binding['bottom_id'])
    if any(columns[key].metadata.get('method') for key in [well_id, *depth_ids]):
        return {}
    numeric = [columns[key].kind == 'number' or binding.get('text_numbers', False) for key in depth_ids]
    factor = UNITS[binding['depth_unit']] / UNITS[scheme['depth_unit']]
    rows = project.row_map()
    groups = defaultdict(list)
    for index, key in enumerate(row_ids):
        if checkpoint and index % 256 == 0:
            checkpoint()
        row = rows[key]
        well = row.values.get(well_id)
        if not isinstance(well, str) or well not in wells:
            continue
        depths = [classify(row.values.get(column), numeric=allow)[1]
                  for column, allow in zip(depth_ids, numeric, strict=True)]
        if any(depth is None for depth in depths):
            continue
        top = depths[0]
        bottom = depths[1] if len(depths) == 2 else None
        if bottom is not None and bottom <= top:
            continue
        groups[well].append(SamplePosition(key, top * factor, None if bottom is None else bottom * factor))
    return dict(groups)


def draw_sample_track(axis, samples, origin):
    from matplotlib.collections import LineCollection

    points = [sample for sample in samples if sample.bottom is None]
    intervals = [sample for sample in samples if sample.bottom is not None]
    artists = {}
    if points:
        dots = axis.scatter([.5] * len(points), [sample.top - origin for sample in points],
                            s=18, color='#276c91', zorder=3)
        dots.set_pickradius(5)
        artists[dots] = points
    if intervals:
        segments = []
        for number, sample in enumerate(intervals):
            lane = .2 + (number % 6) * .12
            segments.append(((lane, sample.top - origin), (lane, sample.bottom - origin)))
        bars = LineCollection(segments, colors='#276c91', linewidths=2.4, pickradius=4)
        axis.add_collection(bars)
        artists[bars] = intervals
    axis.set_xlim(0, 1)
    axis.set_xticks([])
    axis._autopyro_sample_artists = artists
