"""Font families available to Qt and to the visualization renderer."""

from matplotlib import font_manager
from PySide6.QtGui import QFontDatabase


def font_families():
    # Matplotlib also ships fonts (including the default DejaVu Sans) that
    # may not be installed as system fonts and therefore are absent from Qt.
    families = set(QFontDatabase.families())
    families.update(font.name for font in font_manager.fontManager.ttflist)
    return sorted(families, key=str.casefold)
