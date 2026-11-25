"""Positioned text and shapes shared by graph preview and export."""
from math import isfinite

from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle


ANNOTATION_KINDS = ('text', 'arrow', 'rectangle', 'ellipse')


def validate_annotations(annotations):
    if not isinstance(annotations, list):
        raise ValueError('Некорректный список аннотаций графика')
    for number, item in enumerate(annotations, 1):
        if not isinstance(item, dict):
            raise ValueError(f'Некорректная аннотация графика №{number}')
        kind = item.get('kind', 'text')
        text = item.get('text', '')
        if (kind not in ANNOTATION_KINDS or not isinstance(text, str)
                or (kind == 'text' and not text.strip()) or item.get('coordinates') not in ('axes', 'data')
                or not isinstance(item.get('track_id', ''), str)):
            raise ValueError(f'Некорректная аннотация графика №{number}')
        keys = ('x', 'y') if kind == 'text' else ('x', 'y', 'x2', 'y2')
        coordinates = [item.get(key) for key in keys]
        try:
            finite = all(type(value) in (int, float) and isfinite(value) for value in coordinates)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f'Координаты аннотации №{number} должны быть конечными числами')
        if item['coordinates'] == 'axes' and not all(0 <= value <= 1 for value in coordinates):
            raise ValueError(f'Координаты аннотации №{number} в области осей должны быть от 0 до 1')
        if kind != 'text':
            x, y, x2, y2 = coordinates
            dx, dy = abs(float(x2) - float(x)), abs(float(y2) - float(y))
            if (not isfinite(dx) or not isfinite(dy)
                    or (not dx and not dy) or (kind != 'arrow' and (not dx or not dy))):
                raise ValueError(f'У аннотации №{number} должен быть конечный ненулевой размер')


def draw_annotations(axes, spec):
    if not spec.annotations:
        return []
    tracks = dict(zip(spec.track_ids, axes)) if spec.kind == 'multi_depth' else {}
    notices = []
    for number, item in enumerate(spec.annotations, 1):
        track = item.get('track_id', '')
        if track and track not in tracks:
            notices.append(f'Аннотация №{number}: выбранный трек отсутствует на графике')
            continue
        ax = tracks[track] if track else axes[0]
        kind = item.get('kind', 'text')
        relative = item['coordinates'] == 'axes'
        x, y = item['x'], item['y']
        end = (item['x2'], item['y2']) if kind != 'text' else (x, y)
        if ((ax.name == '3d' and (not relative or kind != 'text'))
                or (not relative and ax.get_xscale() == 'log' and min(x, end[0]) <= 0)
                or (not relative and ax.get_yscale() == 'log' and min(y, end[1]) <= 0)):
            notices.append(f'Аннотация №{number}: тип или координаты несовместимы с осями графика')
            continue
        transform = ax.transAxes if relative else ax.transData
        gid = f'plot-annotation-{number}'
        if kind != 'text':
            x2, y2 = end
            if kind == 'arrow':
                artist = FancyArrowPatch((x, y), end, arrowstyle='->', mutation_scale=13, shrinkA=0, shrinkB=0)
            elif kind == 'rectangle':
                artist = Rectangle((min(x, x2), min(y, y2)), abs(x2 - x), abs(y2 - y), fill=False)
            else:
                artist = Ellipse((x / 2 + x2 / 2, y / 2 + y2 / 2), abs(x2 - x), abs(y2 - y), fill=False)
            artist.set(transform=transform, edgecolor=spec.edge_color, linewidth=spec.line_width, zorder=5, gid=gid)
            # add_artist keeps annotations out of the scientific data limits.
            ax.add_artist(artist)
            artist.set_clip_path(ax.patch)
        if item.get('text'):
            text = ax.text2D if ax.name == '3d' else ax.text
            artist = text(x, y, item['text'], transform=transform, va='top', clip_on=True,
                          parse_math=False, fontsize=spec.label_size, fontfamily=spec.font_family,
                          color=spec.edge_color if kind != 'text' else None, zorder=6)
            artist.set_gid(gid if kind == 'text' else f'{gid}-text')
    return notices
