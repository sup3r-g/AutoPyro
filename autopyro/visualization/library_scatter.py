"""Common style channels for two- and three-dimensional library scatter views."""
import numpy as np
from matplotlib.lines import Line2D

from .plot_catalog import colormaps
from autopyro.model.tabular import project_frame


def draw_scatter(ax, project, ids, arrays, parameters, spec, *, selected_ids=(), options=None, prepared=None):
    from .plotting import ensure_style_maps, category_key, legend_entries, plot_columns, _range, numeric_color_norm

    # The same category contract is used by native scatter and library scatter.
    ensure_style_maps(project, spec, prepared=prepared)
    style_ids = list(dict.fromkeys(cid for cid in (spec.color_by, spec.shape_by, spec.size_by, spec.label_by) if cid))
    if any(cid not in {column.id for column in project.columns} for cid in style_ids):
        raise ValueError('Колонка оформления удалена; выберите замену')
    read_frame = prepared.frame if prepared is not None else lambda ids=None, cids=None: project_frame(project, ids, cids)
    frame = read_frame(ids, style_ids)
    numeric_ids = list(dict.fromkeys(cid for cid, enabled in (
        (spec.color_by, spec.color_mode == 'numeric'), (spec.size_by, spec.size_mode == 'numeric')) if cid and enabled))
    numeric = plot_columns(project, read_frame(None, numeric_ids), numeric_ids, spec, prepared=prepared) if numeric_ids else {}
    channels = {}
    for channel, cid in (('color', spec.color_by), ('shape', spec.shape_by), ('size', spec.size_by)):
        categorical = channel == 'shape' or (spec.color_mode if channel == 'color' else spec.size_mode) == 'category'
        if cid and categorical:
            styles = {key: entry['style'] for key, entry in spec.style_maps[channel][cid].items()}
            channels[channel] = frame.loc[ids, cid].map(category_key).map(styles).tolist()
    colors = np.asarray(channels.get('color', [spec.color] * len(ids)))
    color_norm = None
    if spec.color_by and spec.color_mode == 'numeric':
        colors = numeric[spec.color_by].loc[ids, 'number'].to_numpy(dtype=float, na_value=np.nan)
        color_norm = numeric_color_norm(project, spec, numeric[spec.color_by])
    sizes = np.asarray(channels.get('size', [spec.marker_size] * len(ids)), dtype=float) ** 2
    if spec.size_by and spec.size_mode == 'numeric':
        values = numeric[spec.size_by].loc[ids, 'number'].to_numpy(dtype=float, na_value=np.nan)
        norm = _range(project, spec.size_by, spec, spec.size_value_min, spec.size_value_max, numeric[spec.size_by])
        sizes = np.where(np.isnan(values), spec.marker_size,
                         spec.size_min + norm(values) * (spec.size_max - spec.size_min)) ** 2
    groups = {}
    for i, marker in enumerate(channels.get('shape', [spec.marker] * len(ids))):
        groups.setdefault(marker, []).append(i)
    coordinates = [arrays[key] for key in parameters]
    artists = []
    for marker, positions in groups.items():
        kwargs = dict(c=colors[positions] if spec.color_by else [spec.color],
                      s=sizes[positions] if spec.size_by else spec.marker_size ** 2,
                      marker=marker, alpha=spec.alpha,
                      edgecolors=spec.edge_color, linewidths=spec.edge_width, picker=5)
        if color_norm is not None:
            kwargs.update(norm=color_norm, cmap=colormaps[spec.cmap].with_extremes(bad=spec.missing_color),
                          plotnonfinite=True)
        kwargs.update(options or {})
        artist = ax.scatter(*(values[positions] for values in coordinates), **kwargs)
        artist._autopyro_row_ids = [ids[i] for i in positions]
        artists.append(artist)
    selected = set(selected_ids)
    positions = [i for i, rid in enumerate(ids) if rid in selected]
    if positions:
        ax.scatter(*(values[positions] for values in coordinates), s=sizes[positions] + 60,
                   facecolors='none', edgecolors='red', linewidths=1.5)
    if spec.label_by:
        for i, rid in enumerate(ids):
            value = frame.loc[rid, spec.label_by]
            if value is not None and str(value):
                ax.text(*(values[i] for values in coordinates), str(value), fontsize=spec.label_size,
                        fontfamily=spec.font_family)
    handles = []
    for channel, cid in (('color', spec.color_by), ('shape', spec.shape_by), ('size', spec.size_by)):
        if channel not in channels:
            continue
        visible = set(frame.loc[ids, cid].map(category_key)) if spec.legend_visible_only else None
        for key, entry in legend_entries(spec.style_maps[channel][cid], spec.legend_order):
            if ((channel == "color" and entry.get("hidden") is True)
                    or visible is not None and key not in visible):
                continue
            handles.append(Line2D([], [], linestyle='none', label=entry['label'],
                marker=entry['style'] if channel == 'shape' else spec.marker,
                color=entry['style'] if channel == 'color' else spec.color,
                markersize=float(entry['style']) if channel == 'size' else spec.marker_size))
    return artists, handles
