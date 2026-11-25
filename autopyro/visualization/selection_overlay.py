"""Transient UUID highlights over an already rendered Matplotlib figure.

Build the UUID index once per render. Selection changes touch only highlight
artists; 2D canvases reuse the background, 3D falls back to canvas drawing.
"""
from dataclasses import dataclass

import numpy as np
from matplotlib.collections import LineCollection


@dataclass
class _Layer:
    artist: object
    coordinates: np.ndarray
    sizes: np.ndarray
    interval: bool = False

    def select(self, positions):
        coordinates = self.coordinates[positions]
        if self.interval:
            segments = np.empty((len(positions), 2, 2))
            segments[:, :, 0] = coordinates[:, :1]
            segments[:, 0, 1] = coordinates[:, 1]
            segments[:, 1, 1] = coordinates[:, 2]
            self.artist.set_segments(segments)
        else:
            self.artist.set_offsets(coordinates[:, :2])
            if coordinates.shape[1] == 3:
                self.artist.set_3d_properties(coordinates[:, 2], 'z')
            if not len(positions):
                sizes = np.array([])
            elif len(self.sizes) == 1:
                sizes = self.sizes
            else:
                sizes = self.sizes[positions]
            self.artist.set_sizes(sizes + 70)


@dataclass
class SelectionIndex:
    """Sorted membership keys and parallel arrays, without per-point containers.

    Local keys are UUID strings. Remote keys are input-row numbers, resolved
    through a shared UUID lookup. Neither form truncates or hashes away IDs.
    """
    keys: np.ndarray
    layers: np.ndarray
    positions: np.ndarray
    row_lookup: dict | None = None

    @classmethod
    def from_parts(cls, parts):
        if not parts:
            return cls(np.empty(0, dtype=object), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.intp))
        keys, layers, positions = (np.concatenate([part[i] for part in parts]) for i in range(3))
        order = keys.argsort(kind='stable')
        return cls(keys[order], layers[order], positions[order])

    def lookup(self, row_ids, layer_count):
        ids = np.asarray(list(set(row_ids)), dtype=object)
        # Remote frames use input-row numbers. The immutable UUID -> row map
        # is shared by all frames of that input; the strings are not rebuilt.
        requested = (ids if self.row_lookup is None else
                     np.fromiter((self.row_lookup.get(rid, -1) for rid in ids), dtype=np.intp))
        left = np.searchsorted(self.keys, requested, side='left')
        right = np.searchsorted(self.keys, requested, side='right')
        counts = right - left
        selected = set(ids[counts > 0])
        # Expand the matching ranges in bulk; a UUID may occur on several tracks.
        starts = np.cumsum(counts) - counts
        matches = np.repeat(left, counts) + np.arange(counts.sum()) - np.repeat(starts, counts)
        layers = self.layers[matches]
        return selected, [np.unique(self.positions[matches[layers == i]]) for i in range(layer_count)]

    def clear(self):
        self.keys = np.empty(0, dtype=object)
        self.layers = np.empty(0, dtype=np.int32)
        self.positions = np.empty(0, dtype=np.intp)
        self.row_lookup = None

    def __bool__(self):
        return bool(len(self.keys))


class SelectionOverlay:
    def __init__(self, canvas, result, *, interval_width=2, compact=False):
        self.canvas = canvas
        self.layers, self.locations = [], {}
        self.background = None
        self.selected_ids = set()
        self._location_parts = [] if compact else None
        self.blit = canvas.supports_blit and all(ax.name == 'rectilinear' for ax in canvas.figure.axes)
        prepared = result.pop('_selection_overlay', None)
        if prepared is not None:
            self.layers, self.locations = prepared
        elif 'items' in result:  # Points or site aggregates on a map.
            base = result['artist']
            self._add(base.axes, np.asarray([[p['x'], p['y']] for p in result['items']]).reshape(-1, 2),
                      [p['row_ids'] for p in result['items']], base.get_sizes())
        else:
            wire_members = result.get('_preview_membership')
            for index, base in enumerate(result.get('artists', [])):
                members = wire_members[index] if wire_members is not None else base._autopyro_row_ids
                if hasattr(base, '_autopyro_intervals'):
                    self._add(base.axes, np.asarray(base._autopyro_intervals).reshape(-1, 3),
                              members, np.array([]), interval_width=interval_width, single=True)
                else:
                    coordinates = (np.column_stack(base._offsets3d) if hasattr(base, '_offsets3d')
                                   else np.asarray(base.get_offsets()))
                    self._add(base.axes, coordinates, members, base.get_sizes(), single=True)
        if compact and prepared is None:
            self.locations = SelectionIndex.from_parts(self._location_parts)
        self._location_parts = None
        self.connection = canvas.mpl_connect('draw_event', self._drawn)
        if not self.layers:
            self.blit = False
        self.resize_connection = canvas.mpl_connect('resize_event', self._invalidate)
        self.axis_connections = []
        for ax in {layer.artist.axes for layer in self.layers}:
            for signal in ('xlim_changed', 'ylim_changed'):
                self.axis_connections.append((ax.callbacks, ax.callbacks.connect(signal, self._invalidate)))

    def _invalidate(self, *_):
        self.background = None

    def _add(self, ax, coordinates, members, sizes, *, interval_width=None, single=False):
        if not len(coordinates):
            return
        limits = [ax.get_xlim(), ax.get_ylim()]
        automatic = [ax.get_autoscalex_on(), ax.get_autoscaley_on()]
        if ax.name == '3d':
            limits.append(ax.get_zlim())
            automatic.append(ax.get_autoscalez_on())
        if interval_width is not None:
            artist = LineCollection([], colors='#d62728', linewidths=interval_width + 2, zorder=10)
            ax.add_collection(artist, autolim=False)
        else:
            # A single transparent face and size preserve Matplotlib's shared-marker fast path.
            artist = ax.scatter(*([] for _ in range(coordinates.shape[1])), facecolors=[(0, 0, 0, 0)],
                                edgecolors='#d62728', linewidths=1.5, zorder=10)
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        ax.set_autoscalex_on(automatic[0])
        ax.set_autoscaley_on(automatic[1])
        if ax.name == '3d':
            ax.set_zlim(limits[2])
            ax.set_autoscalez_on(automatic[2])
        artist.set_animated(self.blit)
        layer_index = len(self.layers)
        self.layers.append(_Layer(artist, coordinates, np.asarray(sizes), interval_width is not None))
        if self._location_parts is not None:
            positions = np.arange(len(members))
            if single:
                keys = members if isinstance(members, np.ndarray) else np.asarray(members, dtype=object)
            else:
                counts = np.fromiter((len(ids) for ids in members), dtype=np.intp)
                keys = np.fromiter((rid for ids in members for rid in ids), dtype=object)
                positions = np.repeat(positions, counts)
            self._location_parts.append((keys, np.full(len(keys), layer_index, dtype=np.int32), positions))
        else:
            for index, row_ids in enumerate(zip(members) if single else members):
                for rid in row_ids:
                    self.locations.setdefault(rid, []).append((layer_index, index))

    def set_selected(self, row_ids, *, redraw=True):
        if isinstance(self.locations, SelectionIndex):
            selected_ids, positions = self.locations.lookup(row_ids, len(self.layers))
        else:
            selected_ids = set(row_ids).intersection(self.locations)
            grouped = [set() for _ in self.layers]
            for rid in selected_ids:
                for layer, index in self.locations[rid]:
                    grouped[layer].add(index)
            positions = [np.fromiter(group, dtype=int) for group in grouped]
        if redraw and selected_ids == self.selected_ids:
            return
        self.selected_ids = selected_ids
        for layer, selected in zip(self.layers, positions):
            layer.select(selected)
        if redraw and self.layers:
            if self.blit and self.background is not None:
                self.canvas.restore_region(self.background)
                self._paint()
            else:
                self.canvas.draw_idle()

    def _drawn(self, event):
        if self.blit:
            self.background = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
            self._paint()

    def _paint(self):
        for layer in self.layers:
            layer.artist.axes.draw_artist(layer.artist)
        self.canvas.blit(self.canvas.figure.bbox)

    def disconnect(self):
        self.canvas.mpl_disconnect(self.connection)
        self.canvas.mpl_disconnect(self.resize_connection)
        for callbacks, connection in self.axis_connections:
            callbacks.disconnect(connection)
        self.axis_connections.clear()

    def dispose(self):
        self.disconnect()
        self.background = None
        for layer in self.layers:
            layer.artist.remove()
        self.layers.clear()
        self.locations.clear()
        self.selected_ids.clear()
