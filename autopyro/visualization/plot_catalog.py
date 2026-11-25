"""Live visualization catalogs supplied by the installed plotting libraries.

Matplotlib has registries for styles, but no registry of plot types. Plotting
commands are discovered from the public Axes API and its documentation; adapters
in ``library_plot`` supply the different data shapes those commands require.
"""

from collections.abc import Mapping, Sequence
from functools import lru_cache
import importlib
import importlib.util
import inspect
import re

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.colors import get_named_colors_mapping
from matplotlib.markers import MarkerStyle


@lru_cache(maxsize=1)
def load_extensions():
    """Import installed extensions once so they can register their palettes."""
    modules = {}
    for name in ("seaborn", "ultraplot"):
        if importlib.util.find_spec(name) is not None:
            with mpl.rc_context():
                modules[name] = importlib.import_module(name)
                if name == 'ultraplot':
                    # UltraPlot initializes its registries lazily on attribute access.
                    modules[name].CartesianAxes
    return modules


def colors():
    load_extensions()
    mapping = get_named_colors_mapping()
    # UltraPlot's ColorDatabase combines MutableMapping and dict. Read the
    # concrete dict keys, including runtime registrations, on those versions.
    return list(dict.keys(mapping) if isinstance(mapping, dict) else mapping) + ["none"]


def markers():
    return list(MarkerStyle.markers)


def colormap_names():
    load_extensions()
    return list(mpl.colormaps)


class LiveColormaps(Mapping):
    def __iter__(self):
        return iter(colormap_names())

    def __len__(self):
        return len(colormap_names())

    def __getitem__(self, name):
        load_extensions()
        return mpl.colormaps[name]


colormaps = LiveColormaps()


def marker_value(value):
    """Keep integer marker codes typed when edited as text in a table."""
    if isinstance(value, str) and value.isdigit() and int(value) in MarkerStyle.markers:
        return int(value)
    if value == 'None':
        return None
    return value


def validate_marker(value):
    MarkerStyle(value)


class LiveSequence(Sequence):
    def __init__(self, source):
        self.source = source

    def __len__(self):
        return len(self.source())

    def __getitem__(self, index):
        return self.source()[index]

    def __iter__(self):
        return iter(self.source())


def _commands(cls):
    for name, function in inspect.getmembers(cls, callable):
        if name.startswith("_"):
            continue
        doc = inspect.getdoc(function) or ""
        lines = doc.splitlines()
        headline = (lines[4] if doc.startswith('===') and len(lines) > 4 else lines[0] if lines else '').lower()
        # Ignore labels, axes decorations and pre-created Artist helpers.
        if 'twin axes' in headline:
            continue
        if re.match(r"(?:plot |make (?:a |an )|draw (?:a |an |contour|stream|an unstructured)|create (?:a |an )|compute and plot |fill the area |display data as |a scatter plot|generate a .*plot|ax.voxels|add 2d bar)", headline):
            yield name, function


def library_kinds():
    from mpl_toolkits.mplot3d import Axes3D
    modules = load_extensions()
    classes = [Axes, Axes3D, *(module.CartesianAxes for name, module in modules.items() if name == 'ultraplot')]
    fingerprint = tuple(tuple((name, id(value)) for name, value in vars(cls).items()) for cls in classes)
    fingerprint += tuple(tuple((name, id(value)) for name, value in vars(module).items()) for module in modules.values())
    return {key: value for key, value in _discover_kinds(fingerprint).items()
            if key not in HIDDEN_KINDS}


@lru_cache(maxsize=4)
def _discover_kinds(fingerprint):
    from mpl_toolkits.mplot3d import Axes3D

    result = {}
    for prefix, cls in (("mpl", Axes), ("mpl3d", Axes3D)):
        for name, function in _commands(cls):
            if prefix == "mpl3d" and name not in Axes3D.__dict__:
                continue
            result[f"{prefix}:{name}"] = (cls, name, function)
    for name, module in load_extensions().items():
        if name == "seaborn":
            for command, function in inspect.getmembers(module, inspect.isfunction):
                if not command.startswith("_") and "ax" in inspect.signature(function).parameters and function.__module__.startswith(('seaborn.distributions', 'seaborn.relational', 'seaborn.categorical', 'seaborn.regression', 'seaborn.matrix')):
                    result[f"{name}:{command}"] = (module, command, function)
        elif name == "ultraplot":
            cls = module.CartesianAxes
            for command, function in _commands(cls):
                result[f"{name}:{command}"] = (cls, command, function)
    return result


def library_parameters(kind):
    """Array inputs, separate from scalar options, for a public plotting command."""
    cls, name, function = library_kinds()[kind]
    if kind.startswith('ultraplot:') and f'mpl:{name}' in library_kinds():
        return library_parameters(f'mpl:{name}')
    if kind.startswith("seaborn:"):
        parameters = inspect.signature(function).parameters
        if "x" in parameters:
            return [key for key in ("x", "y", "hue", "size", "style", "weights") if key in parameters]
        return ["matrix"]
    # Variable-argument APIs need explicit shapes; the catalog itself is discovered.
    if name in ("contour", "contourf") and kind.startswith("mpl3d:"):
        return ["x", "y", "z"]
    if name in ("imshow", "matshow", "spy", "pcolor", "pcolormesh", "pcolorfast", "contour", "contourf", "heatmap", "chord_diagram"):
        return ["matrix"]
    if name in ("plot_surface", "plot_wireframe", "contour3D", "contourf3D"):
        return ["x", "y", "z"]
    if name == "voxels":
        return ["x", "y", "z", "filled"]
    if name in ("stem", "stem3D"):
        return ["x", "y", "z"] if kind.startswith("mpl3d:") else ["x", "y"]
    if name in ("quiver", "barbs", "streamplot"):
        return ["x", "y", "z", "u", "v", "w"] if kind.startswith("mpl3d:") else ["x", "y", "u", "v"]
    if name in ("tricontour", "tricontourf", "tripcolor", "plot_trisurf"):
        return ["x", "y", "z"]
    if name in ("plot", "loglog", "semilogx", "semilogy", "fill", "triplot", "stackplot"):
        return ["x", "y", "z"] if kind.startswith("mpl3d:") else ["x", "y"]
    signature = inspect.signature(function)
    inputs = [key for key, value in signature.parameters.items()
              if key not in ("self", "axes") and value.default is inspect.Parameter.empty
              and value.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    # Error bars must be bound explicitly, never invented from scatter data.
    if name == "errorbar":
        inputs += ["xerr", "yerr"]
        if kind.startswith('mpl3d:'):
            inputs += ['zerr']
    if kind.startswith("mpl3d:") and name in ("scatter", "stem", "scatter3D", "plot3D") and "zs" not in inputs:
        inputs += ["zs"]
    if not inputs and any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in signature.parameters.values()):
        # Extension wrappers document their accepted arrays in *args.
        match = re.search(r'\*args\s*:\s*([^\n]+)', inspect.getdoc(function) or '')
        if match:
            choices = match.group(1).split(' or ')[-1]
            inputs = re.findall(r'\b(?:x|y|z|y1|y2|x1|x2|u|v|values|data)\b', choices)
        if not inputs:
            inputs = ['x', 'y']
    return inputs


# Hide signal-processing, vector-field and redundant commands from the
# application catalog. Keep raw discovery independent of application policy.
HIDDEN_KINDS = frozenset(f"mpl:{name}" for name in (
    "angle_spectrum", "magnitude_spectrum", "phase_spectrum",
    "psd", "csd", "cohere", "specgram", "acorr", "xcorr",
    "quiver", "streamplot", "barbs", "spy",
    "bxp", "violin", "pcolor", "pcolorfast", "matshow",
    "hlines", "vlines", "fill", "triplot", "pie",
)) | frozenset(f"mpl3d:{name}" for name in (
    "scatter3D", "plot3D", "contour3D", "contourf3D", "stem3D", "quiver3D",
    "semilogx", "semilogy", "loglog", "bar", "bar3d", "stem", "quiver",
    "voxels", "fill_between", "tricontour", "tricontourf",
))

LIBRARY_LABELS = {
    "mpl:bar": "Столбчатая диаграмма",
    "mpl:barh": "Горизонтальная столбчатая диаграмма",
    "mpl:boxplot": "Ящик с усами",
    "mpl:broken_barh": "Горизонтальные интервалы",
    "mpl:contour": "Изолинии на регулярной сетке",
    "mpl:contourf": "Залитые контуры на регулярной сетке",
    "mpl:ecdf": "Накопленное распределение",
    "mpl:errorbar": "Точки XY с погрешностями",
    "mpl:eventplot": "Штриховая диаграмма событий",
    "mpl:fill_between": "Заливка между кривыми",
    "mpl:fill_betweenx": "Заливка между кривыми по Y",
    "mpl:grouped_bar": "Сгруппированные столбцы",
    "mpl:hexbin": "Плотность точек в шестиугольных ячейках",
    "mpl:hist": "Гистограмма распределения",
    "mpl:hist2d": "Двумерная гистограмма",
    "mpl:imshow": "Тепловая карта матрицы",
    "mpl:loglog": "Линия с логарифмическими осями X и Y",
    "mpl:pcolormesh": "Цветовая карта сеточных данных",
    "mpl:plot": "Линейный график XY",
    "mpl:scatter": "Диаграмма рассеяния XY",
    "mpl:semilogx": "Линия с логарифмической осью X",
    "mpl:semilogy": "Линия с логарифмической осью Y",
    "mpl:stackplot": "Диаграмма площадей с накоплением",
    "mpl:stairs": "Ступенчатый график по границам интервалов",
    "mpl:stem": "Стержневая диаграмма",
    "mpl:step": "Ступенчатая линия XY",
    "mpl:tricontour": "Изолинии по нерегулярным точкам",
    "mpl:tricontourf": "Залитые контуры по нерегулярным точкам",
    "mpl:tripcolor": "Цветовая карта на треугольной сетке",
    "mpl:violinplot": "Скрипичная диаграмма распределения",
    "mpl3d:scatter": "Точки XYZ",
    "mpl3d:plot": "Линия XYZ",
    "mpl3d:errorbar": "Точки XYZ с погрешностями",
    "mpl3d:plot_surface": "Поверхность на регулярной сетке",
    "mpl3d:plot_trisurf": "Триангулированная поверхность",
    "mpl3d:plot_wireframe": "Каркас поверхности",
    "mpl3d:contour": "Изолинии в 3D",
    "mpl3d:contourf": "Залитые контуры в 3D",
}

# These backend identifiers remain readable in saved plots, but share selectors.
MERGED_KINDS = {
    'mpl:scatter': 'scatter', 'mpl:hist': 'hist',
    'mpl:boxplot': 'box', 'mpl:violinplot': 'violin',
    'mpl:semilogx': 'mpl:plot', 'mpl:semilogy': 'mpl:plot', 'mpl:loglog': 'mpl:plot',
    'mpl:barh': 'mpl:bar', 'mpl:contourf': 'mpl:contour',
    'mpl:tricontourf': 'mpl:tricontour', 'mpl:fill_betweenx': 'mpl:fill_between',
    'mpl3d:contourf': 'mpl3d:contour',
}

# The public selector holds the family; the mode retains the exact saved command.
PAIRED_KINDS = {
    'mpl:bar': (('Вертикальные столбцы', 'mpl:bar'), ('Горизонтальные столбцы', 'mpl:barh')),
    'mpl:contour': (('Линии', 'mpl:contour'), ('С заливкой', 'mpl:contourf')),
    'mpl:tricontour': (('Линии', 'mpl:tricontour'), ('С заливкой', 'mpl:tricontourf')),
    'mpl:fill_between': (('По оси X', 'mpl:fill_between'), ('По оси Y', 'mpl:fill_betweenx')),
    'mpl3d:contour': (('Линии', 'mpl3d:contour'), ('С заливкой', 'mpl3d:contourf')),
}

PLOT_GROUPS = {
    'Зависимости и кривые': (
        'scatter', 'mpl:plot', 'mpl:errorbar', 'mpl:step', 'mpl:stem',
        'mpl:fill_between', 'mpl3d:scatter', 'mpl3d:plot', 'mpl3d:errorbar',
    ),
    'Глубина, интервалы и события': (
        'depth', 'multi_depth', 'mpl:broken_barh', 'mpl:eventplot', 'mpl:stairs',
    ),
    'Распределения и плотность': (
        'hist', 'box', 'violin', 'mpl:ecdf', 'mpl:hist2d', 'mpl:hexbin',
    ),
    'Сравнение значений и состав': ('mpl:bar', 'mpl:grouped_bar', 'mpl:stackplot'),
    'Матрицы, цветовые поля и изолинии': (
        'mpl:imshow', 'mpl:pcolormesh', 'mpl:contour', 'mpl:tricontour',
        'mpl:tripcolor', 'mpl3d:contour',
    ),
    'Поверхности': ('mpl3d:plot_surface', 'mpl3d:plot_trisurf', 'mpl3d:plot_wireframe'),
}


class KindCatalog(Mapping):
    # Application-specific views retain their serialized names.
    application = {
        "scatter": "Диаграмма рассеяния XY", "depth": "Глубинный профиль",
        "multi_depth": "Глубинный планшет", "hist": "Гистограмма распределения",
        "box": "Ящик с усами", "violin": "Скрипичная диаграмма",
    }

    def snapshot(self):
        return {**self.application, **{key: LIBRARY_LABELS.get(key, key.replace(":", " · "))
                                    for key in library_kinds() if key not in MERGED_KINDS}}

    def __iter__(self):
        return iter(self.snapshot())

    def __len__(self):
        return len(self.snapshot())

    def __contains__(self, key):
        return key in self.application or key in self.snapshot()

    def __getitem__(self, key):
        return self.snapshot()[key]

    def items(self):
        return self.snapshot().items()

    def groups(self):
        snapshot = self.snapshot()
        groups = []
        assigned = set()
        for title, keys in PLOT_GROUPS.items():
            entries = [(key, snapshot[key]) for key in keys if key in snapshot]
            if entries:
                groups.append((title, entries))
                assigned.update(key for key, _ in entries)
        extra = [(key, label) for key, label in snapshot.items() if key not in assigned]
        if extra:
            groups.append(('Другие графики', extra))
        return groups


COLORS = LiveSequence(colors)
MARKERS = LiveSequence(markers)
KINDS = KindCatalog()
