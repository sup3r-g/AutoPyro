"""Indexed hover targets for rendered scatter points and scientific regions."""

import numpy as np
from mpl_toolkits.mplot3d import proj3d


class HoverIndex:
    """Build screen-space point indices lazily, once per axes/view transform."""

    def __init__(self, result, binding):
        self.artists = result['artists']
        self.points = {}
        self.binding = binding
        self.region_indices = {}
        self.regions, self.region_tree = [], None
        if binding:
            from shapely import STRtree
            from autopyro.schemes.interpretation import validate_scheme

            self.regions, _ = validate_scheme(binding['scheme'])
            if self.regions:
                self.region_tree = STRtree([geometry for _, geometry in self.regions])

    def point(self, ax, x, y, radius=8, *, with_count=False):
        projection = ax.get_proj() if ax.name == '3d' else None
        key = (tuple(ax.get_xlim()), tuple(ax.get_ylim()), tuple(ax.bbox.bounds),
               ax.get_xscale(), ax.get_yscale(),
               tuple(projection.ravel()) if projection is not None else None)
        cached = self.points.get(ax)
        if cached is None or cached[0] != key:
            coordinates, sources = [], []
            for artist in self.artists:
                ids = getattr(artist, '_autopyro_row_ids', ())
                if artist.axes is not ax or not ids or not hasattr(artist, 'get_offsets'):
                    continue
                if projection is not None:
                    xyz = getattr(artist, '_offsets3d', None)
                    if xyz is None or len(xyz[0]) != len(ids):
                        continue
                    offsets = np.ma.filled(np.ma.column_stack(xyz).astype(float), np.nan)
                    px, py, _ = proj3d.proj_transform(*offsets.T, projection)
                    pixels = ax.transData.transform(np.column_stack((px, py)))
                else:
                    offsets = np.ma.filled(artist.get_offsets(), np.nan)
                    if offsets.shape != (len(ids), 2):
                        continue
                    pixels = artist.get_offset_transform().transform(offsets)
                valid = np.isfinite(pixels).all(axis=1)
                valid &= ((pixels[:, 0] >= ax.bbox.x0) & (pixels[:, 0] <= ax.bbox.x1)
                          & (pixels[:, 1] >= ax.bbox.y0) & (pixels[:, 1] <= ax.bbox.y1))
                positions = np.flatnonzero(valid)
                coordinates.append(pixels[valid])
                sources.append((artist, positions, offsets))
            counts = np.cumsum([len(positions) for _, positions, _ in sources])
            pixels = np.concatenate(coordinates) if coordinates else np.empty((0, 2))
            # Limit each query to a vertical strip without importing a spatial
            # engine while Qt is installing an ordinary plot's first frame.
            order = pixels[:, 0].argsort(kind='stable')
            sorted_x = pixels[order, 0]
            cached = key, pixels, order, sorted_x, sources, counts
            self.points[ax] = cached
        _, pixels, order, sorted_x, sources, counts = cached
        left = sorted_x.searchsorted(x - radius)
        right = sorted_x.searchsorted(x + radius, side='right')
        candidates = order[left:right]
        if not len(candidates):
            return None
        delta = pixels[candidates] - [x, y]
        distances = np.square(delta).sum(axis=1)
        nearest = distances.argmin()
        limit = radius * radius
        if distances[nearest] >= limit:
            return None
        index = candidates[nearest]
        part = int(np.searchsorted(counts, index, side='right'))
        artist, positions, offsets = sources[part]
        position = int(positions[index - (counts[part - 1] if part else 0)])
        hit = (artist._autopyro_row_ids[position],
               tuple(float(v) for v in offsets[position]), artist.axes)
        if with_count:
            return (*hit, int(np.count_nonzero(distances <= limit)))
        return hit

    def region_labels(self, x, y, ax=None):
        if ax is not None and hasattr(ax, '_autopyro_interpretation'):
            if ax not in self.region_indices:
                self.region_indices[ax] = HoverIndex({'artists': []}, ax._autopyro_interpretation)
            return self.region_indices[ax].region_labels(x, y)
        if self.region_tree is None:
            return []
        from shapely.geometry import Point
        from autopyro.schemes.interpretation import transform

        scheme = self.binding['scheme']
        try:
            point = Point(transform(x, scheme['x']), transform(y, scheme['y']))
        except (ValueError, OverflowError, TypeError):
            return []
        matches = self.region_tree.query(point, predicate='intersects')
        return [self.regions[i][0]['label'] for i in sorted(matches)]
