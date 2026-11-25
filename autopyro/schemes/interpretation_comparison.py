"""Resolve and render a fixed set of complementary pyrolysis diagrams."""
from copy import deepcopy
from dataclasses import replace

from .interpretation import validate_binding, evaluate_project
from .interpretation_quality import column_property_name, property_name

TYPE_PANEL_NAMES = ('HI–Tmax', 'HI–OI')
PANEL_NAMES = (*TYPE_PANEL_NAMES, 'S2–TOC')


def type_agreement(reports, sample_count):
    """Compare published type IDs in reports with the same sample order."""
    if any(name not in reports for name in TYPE_PANEL_NAMES):
        return ['unavailable'] * sample_count

    def types(item):
        if item['status'] in ('classified', 'ambiguous'):
            return set(item.get('possible_types') or item.get('line_hits', ()))
        return set()

    first, second = (reports[name]['results'] for name in TYPE_PANEL_NAMES)
    agreement = []
    for a, b in zip(first, second, strict=True):
        left, right = types(a), types(b)
        if not left or not right:
            agreement.append('unavailable')
        elif left == right:
            agreement.append('equal')
        elif left.isdisjoint(right):
            agreement.append('different')
        else:
            agreement.append('partial')
    return agreement


def comparison_panels(project, schemes, primary, text_numbers):
    """Require unambiguous properties and explicit units; never invent a mapping."""
    columns = {c.id: c for c in project.columns}
    axes = primary['x_id'], primary['y_id']
    if axes[0] == axes[1] or any(column_id not in columns for column_id in axes):
        raise ValueError("Выберите две разные существующие колонки")
    explicit = {}
    for key in ('x', 'y'):
        explicit[property_name(primary['scheme'][key]['property'])] = (
            primary[f'{key}_id'], primary[f'{key}_unit'])
    groups = {columns[column_id].name.rpartition(' / ')[0] for column_id in axes}
    panels = []
    for name in PANEL_NAMES:
        scheme = schemes[name]
        panel = {'name': name}
        try:
            binding = {'scheme': deepcopy(scheme), 'text_numbers': text_numbers}
            for key in ('x', 'y'):
                role = property_name(scheme[key]['property'])
                if role in explicit:
                    cid, unit = explicit[role]
                else:
                    matches = [c for c in columns.values()
                               if column_property_name(c) == role
                               and c.name.rpartition(' / ')[0] in groups]
                    if len(matches) != 1:
                        raise ValueError(f'{role}: показатель отсутствует или найдено несколько колонок')
                    cid, unit = matches[0].id, matches[0].unit
                if not unit:
                    raise ValueError(f'{role}: укажите единицу исходной колонки')
                binding[f'{key}_id'], binding[f'{key}_unit'] = cid, unit
            validate_binding(project, binding['x_id'], binding['y_id'],
                             binding['x_unit'], binding['y_unit'], scheme)
            panel['binding'] = binding
        except ValueError as exc:
            panel['error'] = str(exc)
        panels.append(panel)
    return panels


def comparison_reports(project, ids, panels, *, primary_report=None, checkpoint=None):
    ids = tuple(ids)
    reports = {}
    for panel in panels:
        if 'binding' not in panel:
            continue
        binding = panel['binding']
        if (primary_report is not None and primary_report['scheme'] == binding['scheme']
                and all(primary_report['binding'].get(key) == binding.get(key)
                        for key in ('x_id', 'y_id', 'x_unit', 'y_unit', 'text_numbers'))):
            # Reuse immutable row arrays, but not the report container: the parent
            # later stores these reports and must never acquire a JSON cycle.
            reports[panel['name']] = {key: value for key, value in primary_report.items()
                                     if key not in ('comparisons', 'comparison_panels')}
            continue
        reports[panel['name']] = evaluate_project(project, ids, binding['x_id'], binding['y_id'],
            binding['x_unit'], binding['y_unit'], binding['scheme'], text_numbers=binding['text_numbers'],
            checkpoint=checkpoint)
    return reports


def comparison_shape(spec, count):
    columns = min(2, count) if spec.comparison_grid else count
    return (count + columns - 1) // columns, columns


def render_comparison(project, row_ids, spec, *, figure=None, selected_ids=(), prepared=None, language="ru",
                      include_membership=False):
    from matplotlib.figure import Figure
    from autopyro.visualization.plotting import render_plot

    row_ids = tuple(row_ids)
    figure = figure if type(figure) is Figure else Figure(figsize=(spec.width_mm / 25.4, spec.height_mm / 25.4), dpi=96)
    figure.clear()
    figure.set_facecolor(spec.background)
    panels = spec.interpretation_panels
    if spec.comparison_preview_panel:
        panels = [panel for panel in panels if panel['name'] == spec.comparison_preview_panel] or panels
    rows, columns = comparison_shape(spec, len(panels))
    grid = figure.subplots(rows, columns, squeeze=False).ravel()
    axes = grid[:len(panels)]
    for empty in grid[len(panels):]:
        figure.delaxes(empty)
    figure.subplots_adjust(left=spec.left, right=spec.right, bottom=spec.bottom, top=spec.top,
                           wspace=spec.wspace, hspace=.65)
    figure.suptitle(spec.title, fontsize=spec.title_size, fontfamily=spec.font_family, y=.98)
    if spec.subtitle:
        figure.text(.5, .92, spec.subtitle, ha='center', fontfamily=spec.font_family, fontsize=max(7, spec.title_size - 3))
    result = dict(axes=list(axes), artists=[], excluded={}, plotted={}, notices=[])
    if include_membership:
        result['panel_row_ids'] = []
    for ax, panel in zip(axes, panels, strict=True):
        ax.set_title(panel['name'], fontsize=spec.title_size, fontfamily=spec.font_family)
        binding = panel.get('binding')
        ax._autopyro_interpretation = binding
        if binding is None:
            ax.text(.5, .5, panel['error'], ha='center', va='center', wrap=True,
                    transform=ax.transAxes, fontsize=spec.label_size)
            ax.set_axis_off()
            result['notices'].append(panel['name'] + ': ' + panel['error'])
            continue
        # Axis typography is shared; labels, scales and ranges belong to each diagram.
        x = binding['scheme']['x']
        y = binding['scheme']['y']
        child = replace(spec, interpretation_panels=[], kind='scatter', annotations=[],
                        x_id=binding['x_id'], y_id=binding['y_id'], interpretation=binding,
                        x_axis=replace(spec.x_axis, label=f"{x['property']} [{x['unit']}]",
                                       minimum=None, maximum=None,
                                       scale='log' if x.get('transform') == 'log10' else 'linear'),
                        y_axis=replace(spec.y_axis, label=f"{y['property']} [{y['unit']}]",
                                       minimum=None, maximum=None,
                                       scale='log' if y.get('transform') == 'log10' else 'linear'))
        _, plotted = render_plot(project, row_ids, child, figure=figure, selected_ids=selected_ids,
                                 prepared=prepared, axes=[ax], language=language,
                                 include_membership=include_membership)
        if include_membership:
            result['panel_row_ids'].append({'panel': panel['name'], 'tracks': plotted['plotted_row_ids']})
        for key in ('artists', 'notices'):
            result[key].extend(plotted[key])
        result.setdefault('uncertainty_crossings', {}).update(plotted.get('uncertainty_crossings', {}))
        for key in ('excluded', 'plotted'):
            result[key].update(plotted[key])
        spec.style_maps = child.style_maps
    if spec.annotations:
        from autopyro.visualization.plot_annotations import draw_annotations
        first = spec.interpretation_panels[0]
        if first in panels:
            result['notices'].extend(draw_annotations([axes[panels.index(first)]], spec))
    return figure, result
