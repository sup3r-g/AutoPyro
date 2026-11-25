"""Adapters from worksheet columns to discovered library plotting commands."""

import inspect
import json
from dataclasses import replace

import numpy as np
import pandas as pd

from autopyro.model.tabular import project_frame
from matplotlib import cbook
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable

from .plot_catalog import library_kinds, library_parameters, colormaps, load_extensions
from .plot_controls import controls_for
from .plot_data import LIBRARY_MEASUREMENT_ERROR_KINDS
from .plot_inputs import library_columns, requires_regular_grid


def render_library_plot(project, row_ids, spec, *, figure=None, selected_ids=(), prepared=None, language="ru",
                        include_membership=False):
    from .plotting import _axis, apply_scatter_aspect, axis_label, draw_colorbar, plot_columns

    provider, name = spec.kind.split(":", 1)
    if spec.interpretation or spec.stratigraphy:
        raise ValueError("Удалите интерпретационный или стратиграфический слой перед выбором библиотечного графика")
    cls, _, function = library_kinds()[spec.kind]
    parameters = library_parameters(spec.kind)
    xconfig, yconfig = spec.x_axis, spec.y_axis
    if name in ('loglog', 'semilogx'):
        xconfig = replace(xconfig, scale='log')
    if name in ('loglog', 'semilogy'):
        yconfig = replace(yconfig, scale='log')
    try:
        options = json.loads(spec.library_options)
    except (ValueError, TypeError) as exc:
        raise ValueError("Параметры библиотеки должны быть объектом JSON") from exc
    if not isinstance(options, dict):
        raise ValueError("Параметры библиотеки должны быть объектом JSON")
    if {"ax", "data", "figure"} & options.keys():
        raise ValueError("ax, data и figure задаются приложением")
    columns = {column.id: column for column in project.columns}
    roles = dict(controls_for(spec.kind).axes)
    bindings, active = library_columns(spec, columns, parameters, provider, roles)
    frame = (prepared.frame(row_ids, dict.fromkeys(active.values())) if prepared is not None
             else project_frame(project, row_ids, dict.fromkeys(active.values())))
    numeric_data = plot_columns(project, frame, active.values(), spec, prepared=prepared)
    data = pd.DataFrame(index=frame.index)
    rejected = pd.Series(None, index=frame.index, dtype=object)
    details = {}
    for key, cid in active.items():
        column_data = numeric_data[cid]
        values = (frame[cid] if provider == "seaborn" and columns[cid].kind == "text" and not spec.text_numbers
                  else column_data["number"])
        blocked = column_data["status"].isin(
            ("qc_excluded", "excluded_original", "excluded_calculation", "excluded_interpolation"))
        missing = rejected.isna() & (values.isna() | values.eq("") | blocked)
        rejected.loc[missing] = column_data.loc[missing, "status"]
        axis = (xconfig if key.lower() in ("x", "xs") else yconfig if key.lower() in ("y", "ys")
                else spec.z_axis if key.lower() in ("z", "zs") else None)
        excluded_here = missing
        if axis and axis.scale == "log":
            nonpositive = rejected.isna() & values.map(lambda value: isinstance(value, (int, float)) and value <= 0)
            rejected.loc[nonpositive] = "nonpositive_log"
            excluded_here = missing | nonpositive
        if excluded_here.any():
            counts = details.setdefault(cid, {})
            for reason, count in rejected.loc[excluded_here].value_counts(sort=False).items():
                counts[reason] = counts.get(reason, 0) + count
        data[key] = values
        if key in ('xerr', 'yerr', 'zerr') and (values.dropna() < 0).any():
            raise ValueError('Погрешность должна быть неотрицательной')
    data = data.loc[rejected.isna()]
    ids = data.index.tolist()
    excluded = rejected.dropna().value_counts(sort=False).to_dict()
    if not ids:
        raise ValueError("Нет подходящих данных для выбранного графика")
    arrays = {key: np.asarray(data[key].tolist()) for key in active}
    matrix = np.column_stack(list(arrays.values())) if parameters == ["matrix"] else None
    if provider == 'ultraplot' and (figure is None or not isinstance(figure, load_extensions()['ultraplot'].Figure)):
        figure = load_extensions()['ultraplot'].Figure(figsize=(spec.width_mm / 25.4, spec.height_mm / 25.4), dpi=96, tight=False)
    if provider != 'ultraplot' and figure is not None and type(figure) is not Figure:
        figure = Figure(figsize=(spec.width_mm / 25.4, spec.height_mm / 25.4), dpi=96)
    figure = figure or Figure(figsize=(spec.width_mm / 25.4, spec.height_mm / 25.4), dpi=96)
    figure.clear()
    figure.set_facecolor(spec.background)
    if provider == "mpl3d":
        ax = figure.add_subplot(projection="3d")
    elif provider == "ultraplot":
        ax = figure.add_subplot(111)
    else:
        ax = figure.add_subplot()
    ax.set_facecolor(spec.background)
    if spec.kind == 'mpl:scatter':
        apply_scatter_aspect(ax, spec)
    ax.set_axisbelow(spec.grid_below)
    if provider == 'ultraplot':
        width, height = spec.width_mm / 25.4, spec.height_mm / 25.4
        figure.gridspec.update(left=f'{spec.left * width}in', right=f'{(1-spec.right) * width}in',
                               bottom=f'{spec.bottom * height}in', top=f'{(1-spec.top) * height}in')
    else:
        figure.subplots_adjust(left=spec.left, right=spec.right, bottom=spec.bottom, top=spec.top)
    figure.suptitle(spec.title, fontsize=spec.title_size, fontfamily=spec.font_family, y=0.98)
    if spec.subtitle:
        figure.text(.5, .92, spec.subtitle, ha='center',
                    fontsize=max(7, spec.title_size - 3), fontfamily=spec.font_family)
    command = function if provider == "seaborn" else getattr(ax, name)
    kwargs = {}
    signature = inspect.signature(function).parameters
    # Only pass shared style keywords supported by the actual library method.
    for key, value in (("cmap", spec.cmap), ("alpha", spec.alpha), ("color", spec.color),
                       ("linewidth", spec.line_width), ("bins", spec.bins), ("density", spec.density)):
        if key in signature:
            kwargs[key] = value
    artists, style_handles = [], []
    if provider in ('mpl', 'mpl3d') and name in ('scatter', 'scatter3D'):
        from .library_scatter import draw_scatter
        artists, style_handles = draw_scatter(ax, project, ids, arrays, parameters, spec,
                                              selected_ids=selected_ids, options=options, prepared=prepared)
    elif provider == "seaborn":
        kwargs.update(ax=ax)
        if matrix is not None:
            kwargs["data"] = matrix
        else:
            kwargs.update(arrays)
        kwargs.update(options)
        command(**kwargs)
    else:
        args = [matrix] if matrix is not None else [arrays[key] for key in parameters if key in arrays and key not in ("xerr", "yerr", "zerr")]
        kwargs.update({key: arrays[key] for key in ("xerr", "yerr", "zerr") if key in arrays})
        if name == "bxp":
            args = [cbook.boxplot_stats(args[0])]
        elif name == "violin":
            from scipy.stats import gaussian_kde
            args = [cbook.violin_stats(args[0], lambda data, coords: gaussian_kde(data)(coords))]
        elif name == "broken_barh":
            # First column is interval start, second its width.
            args = [np.column_stack(args), options.pop("yrange", (0, 1))]
        elif name == 'grouped_bar':
            args = [np.atleast_2d(args[0])]
        elif requires_regular_grid(spec.kind):
            # Pivot measured regular-grid coordinates. Never invent missing nodes.
            x, y = arrays[parameters[0]], arrays[parameters[1]]
            xs, ys = np.unique(x), np.unique(y)
            if len(xs) * len(ys) != len(x) or len(set(zip(x, y))) != len(x):
                raise ValueError("Для этого графика нужна полная регулярная сетка X–Y без повторов")
            if len(xs) * len(ys) > 1_000_000:
                raise ValueError("Сетка слишком велика для предпросмотра")
            order = np.lexsort((x, y))
            grids = [arrays[key][order].reshape(len(ys), len(xs)) for key in parameters[2:]]
            args = [*np.meshgrid(xs, ys), *grids] if name != "streamplot" else [xs, ys, *grids]
        elif name == "voxels":
            coordinates = [np.unique(arrays[key], return_inverse=True) for key in ('x', 'y', 'z')]
            shape = tuple(len(unique) for unique, _ in coordinates)
            if np.prod(shape) > 1_000_000:
                raise ValueError("Матрица voxels слишком велика для предпросмотра")
            filled = np.zeros(shape, dtype=bool)
            filled[tuple(indices for _, indices in coordinates)] = arrays['filled'].astype(bool)
            edges = []
            for unique, _ in coordinates:
                if len(unique) == 1:
                    edges.append(np.array([unique[0] - .5, unique[0] + .5]))
                else:
                    edges.append(np.r_[unique[0] - (unique[1] - unique[0]) / 2,
                                       (unique[:-1] + unique[1:]) / 2,
                                       unique[-1] + (unique[-1] - unique[-2]) / 2])
            args = [*np.meshgrid(*edges, indexing='ij'), filled]
        if name in ("acorr", "xcorr"):
            kwargs.setdefault("maxlags", min(10, len(ids) - 1))
        if name in ("plot", "plot3D", "step", "errorbar", "semilogx", "semilogy", "loglog"):
            kwargs.update(color=spec.color, linewidth=spec.line_width, marker=spec.marker,
                          markersize=spec.marker_size, alpha=spec.alpha,
                          markeredgecolor=spec.edge_color, markeredgewidth=spec.edge_width,
                          label=axis_label(project, spec.y_id))
        if name in ('bar', 'barh', 'hist'):
            kwargs.update(color=spec.color, alpha=spec.alpha, edgecolor=spec.edge_color, linewidth=spec.edge_width)
        if name in ('hist', 'hist2d'):
            kwargs.update(bins=spec.bins, density=spec.density)
        if name in ('plot_wireframe', 'quiver', 'streamplot'):
            kwargs.update(color=spec.color, linewidth=spec.line_width)
            if name != 'streamplot':
                kwargs['alpha'] = spec.alpha
        if name in ('contour', 'contourf', 'imshow', 'matshow', 'pcolormesh', 'plot_surface',
                    'plot_trisurf', 'tripcolor', 'tricontour', 'tricontourf', 'hist2d', 'hexbin'):
            kwargs.update(cmap=spec.cmap, alpha=spec.alpha)
        if name in ("scatter", "scatter3D"):
            kwargs.pop('cmap', None)
            kwargs.update(color=spec.color, marker=spec.marker, s=spec.marker_size ** 2, alpha=spec.alpha)
        if matrix is not None and name != "spy":
            kwargs.setdefault("cmap", spec.cmap)
        kwargs.update(options)
        if provider == 'mpl' and name == 'plot':
            # Saved semilog/loglog commands accepted scale keywords alongside
            # line options. Keep them readable after merging into ordinary plot.
            scale_options = {key: kwargs.pop(key) for key in ('base', 'subs', 'nonpositive') if key in kwargs}
            if scale_options:
                for dimension, config in (('x', xconfig), ('y', yconfig)):
                    if config.scale == 'log':
                        getattr(ax, f'set_{dimension}scale')('log', **scale_options)
        # Explicit data options may replace optional arguments, but not duplicate them.
        try:
            if provider == 'mpl3d' and name in ('loglog', 'semilogx', 'semilogy'):
                ax.plot(*args, **kwargs)
            else:
                command(*args, **kwargs)
        except (TypeError, ValueError, ImportError, NotImplementedError) as exc:
            raise ValueError(f"{spec.kind}: {exc}") from exc
    notices = []
    qc_outlines = []
    if spec.qc_outline and spec.kind == 'mpl:scatter' and not spec.depth_bottom_id:
        from .plot_qc import draw_qc_outlines

        qc_outlines = draw_qc_outlines(ax, project, artists, (active[roles['x']], active[roles['y']]))
    if spec.measurement_error_bars and spec.kind in LIBRARY_MEASUREMENT_ERROR_KINDS and not spec.depth_bottom_id:
        from .uncertainty_plot import draw_measurement_error_bars
        dimensions = ('x', 'y', 'z') if provider == 'mpl3d' else ('x', 'y')
        axis_specs = (xconfig, yconfig, spec.z_axis)[:len(dimensions)]
        # Explicit errorbar arguments take precedence independently on each axis.
        column_ids = tuple(
            None if name == 'errorbar' and kwargs.get(dimension + 'err') is not None else active[roles[dimension]]
            for dimension in dimensions
        )
        notices = draw_measurement_error_bars(
            ax, project, zip(ids, *(arrays[roles[dimension]] for dimension in dimensions), strict=True),
            column_ids, axis_specs, columns)
    for dimension, config in (("x", xconfig), ("y", yconfig)):
        cid = bindings.get(roles.get(dimension), '')
        _axis(ax, config, dimension, axis_label(project, cid), spec, column=columns.get(cid), language=language)
    if provider == 'mpl3d':
        zkey = next((key for key in parameters if key.lower() in ('z', 'zs')), '')
        cid = bindings.get(zkey, '')
        _axis(ax, spec.z_axis, 'z', axis_label(project, cid), spec, column=columns.get(cid), language=language)
        if spec.camera_3d is not None:
            ax.view_init(*spec.camera_3d)
    mappables = [artist for artist in [*ax.collections, *ax.images]
                 if isinstance(artist, ScalarMappable) and artist.get_array() is not None]
    for artist in mappables:
        if artist in artists and spec.color_by and spec.color_mode == 'numeric':
            continue  # Scatter already configured the numeric channel and its missing color.
        artist.set_cmap(colormaps[spec.cmap])
        artist.set_clim(spec.color_min, spec.color_max)
    if mappables:
        color_id = spec.color_by if mappables[0] in artists and spec.color_mode == 'numeric' else ''
        draw_colorbar(figure, ax, mappables[0], spec, axis_label(project, color_id))
    handles, labels = ax.get_legend_handles_labels()
    if style_handles:
        handles = style_handles
        labels = [handle.get_label() for handle in handles]
        handles.extend(qc_outlines)
        labels.extend(outline.get_label() for outline in qc_outlines)
    if handles and spec.legend:
        ax.legend(handles, labels, loc=spec.legend_position,
                  prop={'family': spec.font_family, 'size': spec.legend_size},
                  ncols=spec.legend_columns, frameon=spec.legend_frame)
    elif ax.get_legend() is not None and not spec.legend:
        ax.get_legend().remove()
    if spec.annotation:
        text = ax.text2D if provider == 'mpl3d' else ax.text
        text(.02, .98, spec.annotation, transform=ax.transAxes, va='top',
             fontsize=spec.label_size, fontfamily=spec.font_family)
    if spec.annotations:
        from .plot_annotations import draw_annotations
        notices.extend(draw_annotations([ax], spec))
    summary_column = spec.x_id
    label_column = active.get(roles.get('x'), '')
    result = {"axes": [ax], "artists": artists, "excluded": {summary_column: dict(excluded)},
                    "summary_columns": {summary_column: label_column},
                    "excluded_by_column": {summary_column: details},
                    "plotted": {summary_column: len(ids)}, "notices": notices}
    if include_membership:
        result['plotted_row_ids'] = {summary_column: ids}
    return figure, result
