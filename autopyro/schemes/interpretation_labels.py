"""Place region labels in visible polygon interiors using renderer font metrics."""

import numpy as np
import shapely
from matplotlib.artist import Artist
from matplotlib.text import Text
from matplotlib.transforms import IdentityTransform
from shapely.geometry import box


class RegionLabels(Artist):
    """Draw at most one label per region; never change the axes' data limits.

    Layout runs at draw time, so navigation, DPI and vector exports use the same
    collision checks. Only region geometry is examined, independent of row count.
    """

    def __init__(self, regions, *, fontsize, fontfamily):
        super().__init__()
        self.regions = regions
        self.fontsize = fontsize
        self.fontfamily = fontfamily
        self.labels = []
        self.set_zorder(0.4)
        self.set_in_layout(False)

    def draw(self, renderer):
        if not self.get_visible():
            return
        ax = self.axes
        viewport = box(*ax.bbox.extents)
        padding = renderer.points_to_pixels(3)
        occupied = [text.get_window_extent(renderer).padded(padding)
                    for text in ax.texts if text.get_visible()]
        legend = ax.get_legend()
        if legend is not None and legend.get_visible():
            occupied.append(legend.get_window_extent(renderer).padded(padding))
        self.labels = []
        visible = []
        for label, geometry in self.regions:
            projected = shapely.transform(geometry, ax.transData.transform)
            if not np.isfinite(shapely.get_coordinates(projected)).all():
                continue
            clipped = projected.intersection(viewport)
            if not clipped.is_empty and clipped.area > 0:
                visible.append((label, clipped))
        # Give the smaller fields a chance before the large background fields.
        for label, geometry in sorted(visible, key=lambda item: item[1].area):
            parts = list(geometry.geoms) if hasattr(geometry, 'geoms') else [geometry]
            for part in sorted(parts, key=lambda item: item.area, reverse=True):
                if part.area <= 0:
                    continue
                left, bottom, right, top = part.bounds
                width = min(right - left - 2 * padding, renderer.points_to_pixels(150))
                if width <= 0:
                    continue
                text = Text(fontsize=self.fontsize, fontfamily=self.fontfamily,
                            ha='center', va='center', transform=IdentityTransform())
                text.set_figure(ax.figure)
                text.axes = ax
                text.set_clip_on(True)
                text.set_clip_box(ax.bbox)
                # Greedy wrapping measured in output pixels, not character counts.
                lines, current = [], ''
                for word in label.split():
                    trial = (current + ' ' + word).strip()
                    text.set_text(trial)
                    if current and text.get_window_extent(renderer).width > width:
                        lines.append(current)
                        current = word
                    else:
                        current = trial
                text.set_text('\n'.join(lines + [current]))
                anchor = part.representative_point()
                candidates = [(anchor.x, anchor.y)]
                candidates.extend((left + (right - left) * x, bottom + (top - bottom) * y)
                                  for y in (.5, .75, .25) for x in (.5, .25, .75))
                for position in candidates:
                    text.set_position(position)
                    bounds = text.get_window_extent(renderer).padded(padding)
                    if (part.covers(box(*bounds.extents))
                            and not any(bounds.overlaps(other) for other in occupied)):
                        text.draw(renderer)
                        self.labels.append(text)
                        occupied.append(bounds)
                        break
                else:
                    continue
                break
        self.stale = False
