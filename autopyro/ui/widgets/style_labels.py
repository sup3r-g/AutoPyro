"""Localized presentation of library values; serialized identifiers stay intact."""

import colorsys
from functools import lru_cache

import numpy as np
from matplotlib.colors import to_hex, to_rgba
from matplotlib.markers import MarkerStyle
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.transforms import Affine2D
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import QApplication

from autopyro.visualization.plot_catalog import colormaps
from ..i18n import language, tr


@lru_cache(maxsize=8)
def _names(locale):
    return {
        'point': tr('Точка'), 'pixel': tr('Пиксель'), 'circle': tr('Круг'),
        'triangle_down': tr('Треугольник вниз'), 'triangle_up': tr('Треугольник вверх'),
        'triangle_left': tr('Треугольник влево'), 'triangle_right': tr('Треугольник вправо'),
        'tri_down': tr('Трёхлучевой знак вниз'), 'tri_up': tr('Трёхлучевой знак вверх'),
        'tri_left': tr('Трёхлучевой знак влево'), 'tri_right': tr('Трёхлучевой знак вправо'),
        'octagon': tr('Восьмиугольник'), 'square': tr('Квадрат'), 'pentagon': tr('Пятиугольник'),
        'star': tr('Звезда'), 'hexagon1': tr('Шестиугольник'), 'hexagon2': tr('Повёрнутый шестиугольник'),
        'plus': tr('Плюс'), 'x': tr('Косой крест'), 'diamond': tr('Ромб'),
        'thin_diamond': tr('Узкий ромб'), 'vline': tr('Вертикальная черта'), 'hline': tr('Горизонтальная черта'),
        'plus_filled': tr('Заполненный плюс'), 'x_filled': tr('Заполненный косой крест'),
        'tickleft': tr('Штрих влево'), 'tickright': tr('Штрих вправо'),
        'tickup': tr('Штрих вверх'), 'tickdown': tr('Штрих вниз'),
        'caretleft': tr('Уголок влево'), 'caretright': tr('Уголок вправо'),
        'caretup': tr('Уголок вверх'), 'caretdown': tr('Уголок вниз'),
        'caretleftbase': tr('Уголок влево с основанием'), 'caretrightbase': tr('Уголок вправо с основанием'),
        'caretupbase': tr('Уголок вверх с основанием'), 'caretdownbase': tr('Уголок вниз с основанием'),
        'nothing': tr('Без маркера'),
        'black': tr('Чёрный'), 'white': tr('Белый'), 'gray': tr('Серый'),
        'red': tr('Красный'), 'orange': tr('Оранжевый'), 'yellow': tr('Жёлтый'),
        'green': tr('Зелёный'), 'cyan': tr('Голубой'), 'blue': tr('Синий'),
        'purple': tr('Фиолетовый'), 'magenta': tr('Пурпурный'), 'pink': tr('Розовый'),
        'brown': tr('Коричневый'), 'beige': tr('Бежевый'), 'gold': tr('Золотой'),
        'silver': tr('Серебристый'), 'navy': tr('Тёмно-синий'), 'teal': tr('Бирюзовый'),
        'olive': tr('Оливковый'), 'lime': tr('Лаймовый'), 'coral': tr('Коралловый'),
        'salmon': tr('Лососевый'), 'violet': tr('Лиловый'), 'indigo': tr('Индиго'),
        'skyblue': tr('Небесно-голубой'), 'steelblue': tr('Стальной синий'),
        'turquoise': tr('Бирюзовый'), 'khaki': tr('Хаки'), 'maroon': tr('Бордовый'),
        'best': tr('Автоматически'), 'upper right': tr('Вверху справа'),
        'upper left': tr('Вверху слева'), 'lower left': tr('Внизу слева'),
        'lower right': tr('Внизу справа'), 'center left': tr('По центру слева'),
        'center right': tr('По центру справа'), 'upper center': tr('Вверху по центру'),
        'lower center': tr('Внизу по центру'), 'center': tr('По центру'),
        'right': tr('Справа'), 'left': tr('Слева'), 'top': tr('Сверху'), 'bottom': tr('Снизу'),
    }


def marker_label(value):
    if value is None:
        return _names(language())['nothing']
    key = MarkerStyle.markers.get(value) if isinstance(value, (str, int)) else None
    if key in _names(language()):
        return _names(language())[key]
    if isinstance(value, str) and value.startswith('$'):
        return tr('Математический символ: {0}', value)
    return tr('Пользовательский маркер: {0}', value)


def marker_icon(value):
    color = QApplication.palette().color(QPalette.ColorRole.Text).name()
    try:
        return _marker_icon(value, color)
    except (TypeError, ValueError):
        return QIcon()


@lru_cache(maxsize=256)
def _marker_icon(value, color):
    """Use Matplotlib's own paths, including ticks and custom MathText markers."""
    marker = MarkerStyle(value)
    renderer = RendererAgg(64, 40, 144)
    context = renderer.new_gc()
    context.set_foreground(color)
    context.set_linewidth(1)
    scale = 2 if value == ',' else 28
    transform = marker.get_transform() + Affine2D().scale(scale).translate(32, 20)
    renderer.draw_path(context, marker.get_path(), transform,
                       to_rgba(color) if marker.is_filled() else None)
    context.restore()
    pixels = np.asarray(renderer.buffer_rgba())
    image = QImage(pixels.data, 64, 40, 64 * 4, QImage.Format.Format_RGBA8888).copy()
    pixmap = QPixmap.fromImage(image)
    pixmap.setDevicePixelRatio(2)
    return QIcon(pixmap)


def legend_label(value):
    if isinstance(value, int):
        value = next((name for name, code in Legend.codes.items() if code == value), value)
    return _names(language()).get(value, str(value))


def line_style_label(value):
    names = {
        '-': tr('Сплошная'), 'solid': tr('Сплошная'),
        '--': tr('Штриховая'), 'dashed': tr('Штриховая'),
        ':': tr('Пунктирная'), 'dotted': tr('Пунктирная'),
        '-.': tr('Штрихпунктирная'), 'dashdot': tr('Штрихпунктирная'),
        'none': tr('Без линии'), '': tr('Без линии'), ' ': tr('Без линии'),
    }
    key = value.lower() if isinstance(value, str) else 'none' if value is None else None
    return names.get(key, tr('Пользовательская линия: {0}', value))


def line_style_icon(value):
    color = QApplication.palette().color(QPalette.ColorRole.Text).name()
    try:
        return _line_style_icon(value, color)
    except (TypeError, ValueError):
        return QIcon()


@lru_cache(maxsize=256)
def _line_style_icon(value, color):
    renderer = RendererAgg(128, 40, 144)
    line = Line2D([6, 122], [20, 20], color=color, linewidth=1.5,
                  linestyle='None' if value is None else value)
    line.set_figure(Figure(dpi=144))
    line.set_transform(Affine2D())
    line.draw(renderer)
    pixels = np.asarray(renderer.buffer_rgba())
    image = QImage(pixels.data, 128, 40, 128 * 4, QImage.Format.Format_RGBA8888).copy()
    pixmap = QPixmap.fromImage(image)
    pixmap.setDevicePixelRatio(2)
    return QIcon(pixmap)


def color_label(value):
    try:
        red, green, blue, alpha = to_rgba(value)
    except (TypeError, ValueError):
        return str(value)
    if alpha == 0:
        return tr('Без цвета')
    names = _names(language())
    key = str(value).lower().removeprefix('xkcd:').removeprefix('tab:').replace(' ', '').replace('grey', 'gray')
    key = {'b': 'blue', 'g': 'green', 'r': 'red', 'c': 'cyan', 'm': 'magenta', 'y': 'yellow', 'k': 'black', 'w': 'white', 'fuchsia': 'magenta', 'aqua': 'cyan'}.get(key, key)
    color_keys = ('black', 'white', 'gray', 'red', 'orange', 'yellow', 'green', 'cyan', 'blue',
                  'purple', 'magenta', 'pink', 'brown', 'beige', 'gold', 'silver', 'navy',
                  'teal', 'olive', 'lime', 'coral', 'salmon', 'violet', 'indigo', 'skyblue',
                  'steelblue', 'turquoise', 'khaki', 'maroon')
    name = names.get(key) if key in color_keys else None
    if name is None:
        hue, saturation, brightness = colorsys.rgb_to_hsv(red, green, blue)
        lightness = (max(red, green, blue) + min(red, green, blue)) / 2
        if saturation < .04:
            name = names['black' if brightness < .12 else 'white' if brightness > .95 else 'gray']
        elif .04 < hue < .15 and lightness < .45:
            name = names['brown']
        elif .04 < hue < .15 and saturation < .4 and lightness > .65:
            name = names['beige']
        elif (hue < .04 or hue > .9) and lightness > .7:
            name = names['pink']
        else:
            keys = ('red', 'orange', 'yellow', 'lime', 'green', 'teal', 'cyan', 'blue', 'blue', 'purple', 'magenta', 'pink')
            name = names[keys[int((hue * 12 + .5) % 12)]]
        if .12 <= brightness and lightness < .25:
            name = tr('Тёмно-{0}', name.lower())
        elif (.65 < brightness <= .95 and saturation < .04) or (lightness > .75 and name not in (names['white'], names['pink'], names['beige'])):
            name = tr('Светло-{0}', name.lower())
    return f'{name} ({to_hex((red, green, blue, alpha), keep_alpha=alpha < 1)})'


@lru_cache(maxsize=8)
def _colormap_names(locale):
    names = {
        'magma': tr('Магма'), 'inferno': tr('Инферно'), 'plasma': tr('Плазма'),
        'viridis': tr('Виридис'), 'cividis': tr('Цивидис'), 'twilight': tr('Сумерки'),
        'turbo': tr('Турбо'), 'berlin': tr('Берлин'), 'managua': tr('Манагуа'), 'vanimo': tr('Ванимо'),
        'Blues': tr('Синие оттенки'), 'Greens': tr('Зелёные оттенки'), 'Greys': tr('Серые оттенки'),
        'Oranges': tr('Оранжевые оттенки'), 'Purples': tr('Фиолетовые оттенки'), 'Reds': tr('Красные оттенки'),
        'BrBG': tr('Коричневый — бирюзовый'), 'BuGn': tr('Синий — зелёный'),
        'BuPu': tr('Синий — фиолетовый'), 'GnBu': tr('Зелёный — синий'),
        'OrRd': tr('Оранжевый — красный'), 'PRGn': tr('Фиолетовый — зелёный'),
        'PiYG': tr('Розовый — жёлто-зелёный'), 'PuBu': tr('Фиолетовый — синий'),
        'PuBuGn': tr('Фиолетовый — синий — зелёный'), 'PuOr': tr('Фиолетовый — оранжевый'),
        'PuRd': tr('Фиолетовый — красный'), 'RdBu': tr('Красный — синий'),
        'RdGy': tr('Красный — серый'), 'RdPu': tr('Красный — фиолетовый'),
        'RdYlBu': tr('Красный — жёлтый — синий'), 'RdYlGn': tr('Красный — жёлтый — зелёный'),
        'YlGn': tr('Жёлтый — зелёный'), 'YlGnBu': tr('Жёлтый — зелёный — синий'),
        'YlOrBr': tr('Жёлтый — оранжевый — коричневый'), 'YlOrRd': tr('Жёлтый — оранжевый — красный'),
        'CMRmap': tr('Шкала CMR'), 'Spectral': tr('Спектральная'), 'Wistia': tr('Глициния'),
        'afmhot': tr('Горячая AFM'), 'autumn': tr('Осень'), 'binary': tr('Двоичная'),
        'bone': tr('Костяная'), 'brg': tr('Синий — красный — зелёный'),
        'bwr': tr('Синий — белый — красный'), 'cool': tr('Холодная'), 'coolwarm': tr('Холодная — тёплая'),
        'copper': tr('Медная'), 'cubehelix': tr('Цветовая спираль'), 'flag': tr('Флаг'),
        'gist_earth': tr('Земная GIST'), 'gist_gray': tr('Серая GIST'), 'gist_heat': tr('Тепловая GIST'),
        'gist_ncar': tr('Спектральная NCAR'), 'gist_rainbow': tr('Радужная GIST'), 'gist_stern': tr('Шкала Штерна'),
        'gist_yarg': tr('Обратная серая GIST'), 'gnuplot': tr('Спектральная Gnuplot'), 'gnuplot2': tr('Спектральная Gnuplot 2'),
        'gray': tr('Серая'), 'hot': tr('Горячая'), 'hsv': tr('Цветовой круг HSV'), 'jet': tr('Радужная Jet'),
        'nipy_spectral': tr('Спектральная NIPY'), 'ocean': tr('Океан'), 'pink': tr('Розовая'),
        'prism': tr('Призма'), 'rainbow': tr('Радуга'), 'seismic': tr('Сейсмическая'),
        'spring': tr('Весна'), 'summer': tr('Лето'), 'terrain': tr('Рельеф'), 'winter': tr('Зима'),
        'Accent': tr('Акцентные цвета'), 'okabe_ito': tr('Окабэ — Ито'), 'Dark2': tr('Тёмная категориальная'),
        'Paired': tr('Парные цвета'), 'Pastel1': tr('Пастельная 1'), 'Pastel2': tr('Пастельная 2'),
        'Set1': tr('Категориальный набор 1'), 'Set2': tr('Категориальный набор 2'), 'Set3': tr('Категориальный набор 3'),
        'tab10': tr('Категориальная палитра 10'), 'tab20': tr('Категориальная палитра 20'),
        'tab20b': tr('Категориальная палитра 20 B'), 'tab20c': tr('Категориальная палитра 20 C'),
        'rocket': tr('Ракета'), 'mako': tr('Мако'), 'flare': tr('Сияние'), 'crest': tr('Гребень'),
        'vlag': tr('Флаг'), 'icefire': tr('Лёд — огонь'),
    }
    return {key.casefold(): label for key, label in names.items()}


def colormap_label(value):
    if value.endswith('_r'):
        return tr('{0} (обратная)', colormap_label(value[:-2]))
    if value.endswith('_shifted'):
        return tr('{0} (со сдвигом)', colormap_label(value[:-8]))
    key = value.casefold()
    key = {'grey': 'gray', 'grays': 'greys', 'gist_grey': 'gist_gray', 'gist_yerg': 'gist_yarg'}.get(key, key)
    return _colormap_names(language()).get(key, tr('Цветовая шкала «{0}»', value.replace('_', ' ')))


@lru_cache(maxsize=2048)
def _swatch(samples):
    pixmap = QPixmap(64 if len(samples) > 1 else 24, 14)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    for x in range(pixmap.width()):
        color = samples[min(len(samples) - 1, x * len(samples) // pixmap.width())]
        painter.setPen(QColor(*color))
        painter.drawLine(x, 0, x, pixmap.height() - 1)
    painter.setPen(QColor('#808080'))
    painter.drawRect(0, 0, pixmap.width() - 1, pixmap.height() - 1)
    painter.end()
    return QIcon(pixmap)


def color_icon(value):
    try:
        return _swatch((tuple(round(channel * 255) for channel in to_rgba(value)),))
    except (TypeError, ValueError):
        return QIcon()


def colormap_icon(value):
    samples = colormaps[value](np.linspace(0, 1, 64), bytes=True)
    return _swatch(tuple(tuple(int(channel) for channel in sample) for sample in samples))
