"""Isolated image export. The child publishes complete files and reports progress."""
from dataclasses import replace
import os
from pathlib import Path


_COMMON_STYLE_FIELDS = (
    'font_family', 'title_size', 'label_size', 'legend_size', 'legend_frame',
    'background', 'grid_color', 'grid_alpha', 'grid_width', 'grid_style', 'grid_below',
)
_AXIS_STYLE_FIELDS = ('label_size', 'tick_size', 'color', 'rotation', 'label_pad', 'major_grid', 'minor_grid')


def with_export_style(spec, source):
    """Copy visual settings while preserving data bindings and output geometry."""
    if source is None:
        return spec

    def axis_style(target, style):
        return replace(target, **{field: getattr(style, field) for field in _AXIS_STYLE_FIELDS})

    return replace(spec, **{field: getattr(source, field) for field in _COMMON_STYLE_FIELDS},
                   x_axis=axis_style(spec.x_axis, source.x_axis),
                   y_axis=axis_style(spec.y_axis, source.y_axis),
                   z_axis=axis_style(spec.z_axis, source.z_axis),
                   track_axes={cid: axis_style(axis, source.x_axis) for cid, axis in spec.track_axes.items()})


def freeze_palette(name):
    """Transfer runtime registrations as RGBA data, not library/global objects."""
    import numpy as np
    from .plot_catalog import colormaps
    palette = colormaps[name]
    return {'name': name, 'colors': palette(np.linspace(0, 1, palette.N)).tolist(),
            'bad': palette.get_bad().tolist(), 'under': palette.get_under().tolist(),
            'over': palette.get_over().tolist()}


def install_palette(palette):
    """Restore a frozen runtime palette in a rendering process."""
    if not palette:
        return
    import matplotlib
    from matplotlib.colors import ListedColormap
    import numpy as np
    name = palette['name']
    existing = matplotlib.colormaps[name] if name in matplotlib.colormaps else None
    same = (existing is not None
            and np.array_equal(existing(np.linspace(0, 1, len(palette['colors']))), palette['colors'])
            and all(np.array_equal(getattr(existing, 'get_' + key)(), palette[key])
                    for key in ('bad', 'under', 'over')))
    if not same:
        cmap = ListedColormap(palette['colors'], name=name).with_extremes(
            bad=palette['bad'], under=palette['under'], over=palette['over'])
        matplotlib.colormaps.register(cmap, name=name, force=True)


def export_files(jobs, cancelled, connection):
    # No Qt canvas or parent Matplotlib state is used in this process.
    import matplotlib
    matplotlib.use('Agg', force=True)
    from .plotting import _write_plot
    from autopyro.data_io.file_output import publish_without_overwrite
    from .plot_data import PlotDataCache, PreparationCancelled
    cache = PlotDataCache()
    exported, errors = [], []
    try:
        for index, job in enumerate(jobs):
            if cancelled.is_set():
                break
            try:
                record = {'name': job['name'], 'file': Path(job['path']).name}
                if 'panels' in job:
                    from .plot_panels import write_panels

                    def progress(done, total):
                        connection.send({'progress': done, 'total': total})

                    warnings = write_panels(job['staging'], job['panels'], job['columns'], cache, cancelled, progress)
                    if warnings:
                        record['warnings'] = warnings
                elif job.get('snapshot'):
                    from .plot_snapshot import write_snapshot
                    write_snapshot(job['staging'], job, cancelled)
                elif 'matrix' in job:
                    import numpy as np
                    from .statistics_matrix import correlation_matrix_figure, scatter_matrix_figure

                    matrix = job['matrix']
                    if matrix['kind'] == 'heatmap':
                        figure = correlation_matrix_figure(matrix, job['name'])
                    else:
                        figure = scatter_matrix_figure(matrix, np.asarray(matrix['values'], dtype=float), job['name'])
                    figure.savefig(job['staging'], dpi=150)
                else:
                    if 'replay' in job:
                        from .plot_snapshot import read_snapshot
                        spec, source, ids, palette, revision = read_snapshot(job['replay'])
                    else:
                        spec, source, ids = job['spec'], job['source'], job['ids']
                        palette, revision = job.get('palette'), job['revision']
                    install_palette(palette)
                    prepared = job.get('prepared')
                    if prepared is None:
                        prepared = cache.prepare(source, ids, spec, cancelled)
                    result = _write_plot(job['staging'], prepared.project, ids, spec, prepared=prepared,
                                         include_membership=job.get('no_overwrite', False))
                    record.update(row_ids=ids, spec=spec.to_dict(), project_id=prepared.project.id,
                                  data_revision=revision, plotted=result['plotted'], excluded=result['excluded'],
                                  excluded_by_column=result.get('excluded_by_column', {}))
                    for key in ('plotted_row_ids', 'panel_row_ids'):
                        if key in result:
                            record[key] = result[key]
                    if result.get('layout_warnings'):
                        record['warnings'] = result['layout_warnings']
                    if 'curve_archive' in result:
                        record['curve_archive'] = result['curve_archive']
                    if 'passport' in result:
                        record['passport'] = result['passport']
                with open(job['staging'], 'rb+') as stream:
                    os.fsync(stream.fileno())
                if cancelled.is_set():
                    break
                if job.get('no_overwrite'):
                    record['file'] = publish_without_overwrite(job['staging'], job['path']).name
                else:
                    os.replace(job['staging'], job['path'])
                exported.append(record)
            except PreparationCancelled:
                break
            except Exception as exc:
                errors.append(f"{job['name']}: {exc}")
            if 'panels' not in job:
                connection.send({'progress': index + 1, 'total': len(jobs)})
        connection.send({'result': {'plots': exported, 'errors': errors, 'cancelled': cancelled.is_set()}})
    except BaseException as exc:
        try:
            connection.send({'fatal': str(exc)})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()
