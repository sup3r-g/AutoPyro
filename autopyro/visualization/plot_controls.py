"""Explicit user-facing contracts for plotting commands, independent of Qt.

Discovery adds experimental commands. Only the families declared here are
presented as supported views; backend argument names stay inside the adapter.
"""
from dataclasses import dataclass
from functools import lru_cache
import json
import math

from .plot_catalog import library_parameters


@dataclass(frozen=True)
class Option:
    key: str
    label: str
    default: object
    choices: tuple = ()
    minimum: float = 0
    maximum: float = 1000


@dataclass(frozen=True)
class Controls:
    title: str
    parameters: tuple
    axes: tuple  # (user axis, backend argument)
    options: tuple = ()
    capabilities: frozenset = frozenset()
    experimental: bool = False


TITLES = {
    'plot': 'Линия', 'plot3D': 'Линия', 'scatter': 'Точки', 'scatter3D': 'Точки',
    'step': 'Ступенчатая линия', 'errorbar': 'Точки с погрешностью',
    'semilogx': 'Линия · логарифмическая X', 'semilogy': 'Линия · логарифмическая Y',
    'loglog': 'Линия · логарифмические X/Y', 'bar': 'Столбцы', 'barh': 'Горизонтальные столбцы',
    'hist': 'Гистограмма', 'hist2d': 'Двумерная гистограмма', 'hexbin': 'Шестиугольные бины',
    'imshow': 'Матрица значений', 'matshow': 'Матрица с индексами', 'pcolormesh': 'Цветовая матрица',
    'contour': 'Изолинии', 'contourf': 'Заполненные изолинии',
    'plot_surface': 'Поверхность на регулярной сетке', 'plot_wireframe': 'Каркас на регулярной сетке',
    'plot_trisurf': 'Триангулированная поверхность', 'tripcolor': 'Цветовая триангуляция',
    'tricontour': 'Триангулированные изолинии', 'tricontourf': 'Заполненные триангулированные изолинии',
    'quiver': 'Векторы', 'streamplot': 'Линии тока',
}
ROLE_LABELS = {
    'xerr': 'Погрешность X', 'yerr': 'Погрешность Y', 'zerr': 'Погрешность Z',
    'u': 'Компонента вектора X', 'v': 'Компонента вектора Y', 'w': 'Компонента вектора Z',
    'filled': 'Заполнение ячейки', 'hue': 'Цвет по свойству', 'style': 'Форма по свойству',
    'size': 'Размер по свойству', 'weights': 'Вес наблюдения', 'height': 'Высота столбца',
    'width': 'Ширина столбца', 'xmin': 'Начало X', 'xmax': 'Конец X',
    'ymin': 'Начало Y', 'ymax': 'Конец Y', 'dx': 'Размер по X', 'dy': 'Размер по Y',
    'dz': 'Размер по Z', 'x2': 'Вторая координата X', 'y2': 'Вторая координата Y',
    'z2': 'Вторая координата Z',
}
CHOICE_LABELS = {'pre': 'До точки', 'post': 'После точки', 'mid': 'Посередине',
                 'auto': 'Автоматически', 'none': 'Нет', 'nearest': 'Ближайший сосед',
                 'bilinear': 'Билинейная', 'upper': 'Сверху', 'lower': 'Снизу', 'equal': 'Равные размеры'}
CHOICE_LABELS.update({'-': 'Сплошная', '--': 'Штриховая', ':': 'Точечная', '-.': 'Штрихпунктирная'})
LINE = frozenset({'color', 'alpha', 'line', 'marker', 'edge'})
SCATTER = frozenset({'color', 'alpha', 'marker', 'edge', 'channels', 'labels'})
MATRIX = frozenset({'alpha', 'cmap'})
OPTIONS = {
    'step': (Option('where', 'Положение ступени', 'pre', ('pre', 'post', 'mid')),),
    'errorbar': (Option('capsize', 'Размер засечек, pt', 0., minimum=0, maximum=30),),
    'bar': (Option('width', 'Ширина столбца', .8, minimum=.01, maximum=1000),),
    'barh': (Option('height', 'Толщина столбца', .8, minimum=.01, maximum=1000),),
    'imshow': (Option('interpolation', 'Интерполяция изображения', 'auto', ('auto', 'none', 'nearest', 'bilinear')),
               Option('origin', 'Начало матрицы', 'upper', ('upper', 'lower')),
               Option('aspect', 'Соотношение ячеек', 'equal', ('equal', 'auto'))),
    'hexbin': (Option('gridsize', 'Разрешение бинов', 100, minimum=2, maximum=500),),
    'contour': (Option('levels', 'Число уровней', 7, minimum=2, maximum=100),),
    'contourf': (Option('levels', 'Число уровней', 7, minimum=2, maximum=100),),
    'plot_surface': (Option('rcount', 'Разрешение по строкам', 50, minimum=2, maximum=500),
                     Option('ccount', 'Разрешение по колонкам', 50, minimum=2, maximum=500)),
    'plot_wireframe': (Option('rcount', 'Линий по строкам', 50, minimum=2, maximum=500),
                       Option('ccount', 'Линий по колонкам', 50, minimum=2, maximum=500)),
    'streamplot': (Option('density', 'Плотность линий тока', 1., minimum=.1, maximum=10),),
}
for _command in ('plot', 'plot3D', 'step', 'errorbar', 'semilogx', 'semilogy', 'loglog'):
    OPTIONS[_command] = OPTIONS.get(_command, ()) + (
        Option('linestyle', 'Линия', '-', ('-', '--', ':', '-.', 'none')),)


@lru_cache(maxsize=256)
def controls_for(kind):
    provider, name = kind.split(':', 1)
    parameters = tuple(library_parameters(kind))
    axes, used = [], set()
    aliases = {'xs': 'x', 'ys': 'y', 'zs': 'z', 'x1': 'x', 'y1': 'y', 'z1': 'z',
               'left': 'x', 'height': 'y', 'width': 'x'}
    for key in parameters:
        axis = aliases.get(key.lower(), key.lower())
        if axis in ('x', 'y', 'z') and axis not in used:
            axes.append((axis, key))
            used.add(axis)
    if parameters != ('matrix',) and not axes and parameters:
        axes = [('x', parameters[0])]
    supported = name in TITLES and provider in ('mpl', 'mpl3d')
    if name in ('scatter', 'scatter3D'):
        capabilities = SCATTER
    elif name in ('plot', 'plot3D', 'step', 'errorbar', 'semilogx', 'semilogy', 'loglog'):
        capabilities = LINE
    elif name in ('hist', 'bar', 'barh'):
        capabilities = frozenset({'color', 'alpha', 'edge'})
    elif name == 'plot_wireframe':
        capabilities = frozenset({'color', 'alpha', 'line'})
    elif name in ('quiver', 'streamplot'):
        capabilities = frozenset({'color', 'line'} | ({'alpha'} if name != 'streamplot' else set()))
    else:
        capabilities = MATRIX
    if provider not in ('mpl', 'mpl3d'):
        capabilities = frozenset({'color', 'alpha', 'marker'}) if name == 'scatter' else frozenset()
    title = TITLES.get(name, name)
    if provider == 'mpl3d':
        title += ' · 3D'
    return Controls(title, parameters, tuple(axes), OPTIONS.get(name, ()), capabilities, not supported)


def options_object(text):
    try:
        result = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ValueError('Дополнительные параметры должны быть объектом JSON') from exc
    if not isinstance(result, dict):
        raise ValueError('Дополнительные параметры должны быть объектом JSON')
    return result


def merge_line_kind(spec):
    """Turn legacy log-line commands into ordinary lines with explicit scales."""
    if spec.kind not in ('mpl:semilogx', 'mpl:semilogy', 'mpl:loglog'):
        return
    if spec.kind in ('mpl:semilogx', 'mpl:loglog'):
        spec.x_axis.scale = 'log'
    if spec.kind in ('mpl:semilogy', 'mpl:loglog'):
        spec.y_axis.scale = 'log'
    spec.kind = 'mpl:plot'


def migrate_bindings(spec):
    """Move old library axis bindings into the common axis fields once on load."""
    if ':' not in spec.kind:
        return
    merge_line_kind(spec)
    for axis, argument in controls_for(spec.kind).axes:
        if spec.library_bindings.get(argument):
            if axis == 'z':
                spec.z_id = spec.library_bindings[argument]
            else:
                setattr(spec, axis + '_id', spec.library_bindings[argument])
    try:
        options = options_object(spec.library_options)
    except ValueError:
        return  # Invalid legacy drafts remain editable in the expert section.
    shared = {'color': 'color', 'alpha': 'alpha', 'cmap': 'cmap', 'marker': 'marker',
              'markersize': 'marker_size', 'markeredgecolor': 'edge_color', 'markeredgewidth': 'edge_width'}
    shared['linewidth'] = 'edge_width' if spec.kind.split(':')[1] in ('scatter', 'scatter3D') else 'line_width'
    for key, attribute in shared.items():
        if key in options and isinstance(options[key], (str, int, float)):
            setattr(spec, attribute, options.pop(key))
    spec.library_options = json.dumps(options)


def option_is_editable(option, value):
    """Legacy array-valued or custom options stay intact in the expert editor."""
    if option.choices:
        return value in option.choices
    return (type(value) in (int, float) and math.isfinite(value)
            and option.minimum <= value <= option.maximum
            and (type(option.default) is not int or type(value) is int))


def bindings_from_axes(spec, extra):
    result = {key: value for key, value in extra.items() if value}
    for axis, argument in controls_for(spec.kind).axes:
        result[argument] = getattr(spec, axis + '_id')
    return result


def column_compatible(column, source):
    """Older templates declare only units; new ones also declare type and basis."""
    return (column.unit == source['unit']
            and ('kind' not in source or column.kind == source['kind'])
            and ('basis' not in source or column.metadata.get('basis', '') == source['basis']))
