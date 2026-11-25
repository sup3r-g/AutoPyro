"""Input controls and column requirements for native and library plots."""

from dataclasses import dataclass

from .plot_controls import Controls, controls_for


DEPTH_KINDS = frozenset({'depth', 'multi_depth'})
PAIRED_INPUT_KINDS = DEPTH_KINDS | {'scatter'}
OPTIONAL_LIBRARY_ARGUMENTS = frozenset({'hue', 'size', 'style', 'weights', 'xerr', 'yerr', 'zerr'})


@dataclass(frozen=True)
class InputControls:
    library: Controls | None
    axes: frozenset[str]
    extra_arguments: frozenset[str]
    paired: bool
    depth: bool
    tracks: bool
    matrix: bool
    histogram: bool


def input_controls_for(kind):
    library = controls_for(kind) if ':' in kind else None
    paired = kind in PAIRED_INPUT_KINDS
    axis_bindings = library.axes if library else ()
    parameters = frozenset(library.parameters) if library else frozenset()
    matrix = 'matrix' in parameters
    axes = frozenset(axis for axis, _ in axis_bindings) if library else frozenset(('x', 'y') if paired else ('x',))
    return InputControls(
        library=library,
        axes=axes,
        extra_arguments=parameters - {argument for _, argument in axis_bindings} - {'matrix'},
        paired=paired,
        depth=kind in DEPTH_KINDS,
        tracks=kind == 'multi_depth' or matrix,
        matrix=matrix,
        histogram=kind == 'hist' or kind.partition(':')[2] in ('hist', 'hist2d'),
    )


def requires_regular_grid(kind):
    provider, _, name = kind.partition(':')
    return name in ('plot_surface', 'plot_wireframe', 'streamplot') or (
        provider == 'mpl3d' and name in ('contour', 'contourf', 'contour3D', 'contourf3D')
    )


def native_columns(spec, columns):
    """Validate the chosen native plot columns and return the X tracks and pairing mode."""
    x_ids = spec.track_ids if spec.kind == 'multi_depth' else [spec.x_id]
    if spec.depth_bottom_id:
        if spec.kind not in DEPTH_KINDS or spec.depth_bottom_id not in columns or spec.depth_bottom_id == spec.y_id:
            raise ValueError('Низ отбора требует глубинного графика и отдельной колонки')
        top_column, bottom_column = columns.get(spec.y_id), columns[spec.depth_bottom_id]
        if top_column and top_column.unit and bottom_column.unit and top_column.unit != bottom_column.unit:
            raise ValueError('Верх и низ отбора на графике должны иметь одинаковые единицы')
        if spec.shape_by or spec.connect:
            raise ValueError('Для интервальных проб отключите форму по свойству и соединение точек')
    if not x_ids or any(cid not in columns for cid in x_ids):
        raise ValueError('Выберите показатель X / треки планшета')
    paired = spec.kind in PAIRED_INPUT_KINDS
    if paired and spec.y_id not in columns:
        raise ValueError('Выберите показатель Y / глубину')
    channels = (spec.color_by, spec.shape_by, spec.size_by, spec.label_by, spec.line_group_by) if paired else (spec.color_by,)
    if any(cid and cid not in columns for cid in channels):
        raise ValueError('Колонка оформления удалена; выберите другую')
    if len(x_ids) != len(set(x_ids)):
        raise ValueError('Треки не должны повторяться')
    if len(x_ids) > 10:
        raise ValueError('Планшет ограничен 10 треками')
    return x_ids, paired


def library_columns(spec, columns, parameters, provider, axis_roles):
    """Resolve legacy axis bindings and validate required and assigned optional columns."""
    bindings = dict(spec.library_bindings)
    if parameters == ['matrix']:
        if not spec.track_ids or any(cid not in columns for cid in spec.track_ids):
            raise ValueError('Выберите числовые колонки матрицы в списке треков')
        return bindings, {cid: cid for cid in spec.track_ids}

    optional = OPTIONAL_LIBRARY_ARGUMENTS | {'y'} if provider == 'seaborn' else OPTIONAL_LIBRARY_ARGUMENTS
    required = [key for key in parameters if key not in optional]
    axis_bindings = {argument: getattr(spec, axis + '_id') for axis, argument in axis_roles.items()}
    for key in parameters:
        bindings.setdefault(key, axis_bindings.get(key, ''))
    if any(not bindings.get(key) or bindings[key] not in columns for key in required):
        raise ValueError('Выберите колонки аргументов: ' + ', '.join(required))
    active = {key: cid for key, cid in bindings.items() if key in parameters and cid}
    if any(cid not in columns for cid in active.values()):
        raise ValueError('Колонка аргумента удалена; выберите замену')
    return bindings, active
