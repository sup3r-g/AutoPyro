"""KDE Breeze palettes and locally bundled Lucide interface icons."""

from configparser import ConfigParser
from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QIconEngine, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication, QStyle, QStyleFactory

ASSETS = Path(__file__).with_name("assets")
THEMES = ("breeze_light", "breeze_dark")
ICON_ALIASES = {
    "document-new": "file-plus", "document-open": "folder-open",
    "document-save": "save", "document-save-as": "save-all",
    "document-import": "import", "document-export": "log-out",
    "edit-undo": "undo-2", "edit-redo": "redo-2", "edit-copy": "copy",
    "edit-paste": "clipboard-paste", "configure": "settings-2",
}


def breeze_palette(theme):
    if theme not in THEMES:
        raise ValueError(f"Unknown theme: {theme}")
    config = ConfigParser(interpolation=None)
    config.read(ASSETS / ("BreezeDark.colors" if theme == "breeze_dark" else "BreezeLight.colors"), encoding="utf-8")

    def color(section, key):
        return QColor(*(int(component) for component in config["Colors:" + section][key].split(",")))

    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: ("Window", "BackgroundNormal"),
        QPalette.ColorRole.WindowText: ("Window", "ForegroundNormal"),
        QPalette.ColorRole.Base: ("View", "BackgroundNormal"),
        QPalette.ColorRole.AlternateBase: ("View", "BackgroundAlternate"),
        QPalette.ColorRole.Text: ("View", "ForegroundNormal"),
        QPalette.ColorRole.Button: ("Button", "BackgroundNormal"),
        QPalette.ColorRole.ButtonText: ("Button", "ForegroundNormal"),
        QPalette.ColorRole.Highlight: ("Selection", "BackgroundNormal"),
        QPalette.ColorRole.HighlightedText: ("Selection", "ForegroundNormal"),
        QPalette.ColorRole.ToolTipBase: ("Tooltip", "BackgroundNormal"),
        QPalette.ColorRole.ToolTipText: ("Tooltip", "ForegroundNormal"),
        QPalette.ColorRole.Link: ("View", "ForegroundLink"),
        QPalette.ColorRole.LinkVisited: ("View", "ForegroundVisited"),
        QPalette.ColorRole.PlaceholderText: ("View", "ForegroundInactive"),
        QPalette.ColorRole.BrightText: ("View", "ForegroundNegative"),
        QPalette.ColorRole.Accent: ("Selection", "BackgroundNormal"),
    }
    for role, (section, key) in roles.items():
        palette.setColor(role, color(section, key))
    # Keep selected text legible against the Breeze accent in both themes.
    if theme == "breeze_light":
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#126392"))
        palette.setColor(QPalette.ColorRole.Accent, QColor("#126392"))
    else:
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#17222b"))
    window = palette.window().color()
    for role, shade in ((QPalette.ColorRole.Light, window.lighter(115)),
                        (QPalette.ColorRole.Midlight, window.lighter(105)),
                        (QPalette.ColorRole.Mid, window.darker(125)),
                        (QPalette.ColorRole.Dark, window.darker(145)),
                        (QPalette.ColorRole.Shadow, window.darker(180))):
        palette.setColor(role, shade)
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, color("View", "ForegroundInactive"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, window)
    return palette


def apply_theme(app, theme="breeze_light"):
    if app.property("autopyroTheme") == theme:
        return
    native = next((key for key in QStyleFactory.keys() if key.lower() == "breeze"), None)
    app.setStyle(QStyleFactory.create(native or "Fusion"))
    palette = breeze_palette(theme)
    app.setPalette(palette)
    accent = palette.highlight().color().name()
    border = palette.mid().color().name()
    app.setStyleSheet(f"""
        QToolBar {{ border: none; spacing: 3px; padding: 3px; }}
        QToolButton {{ padding: 4px; border-radius: 3px; }}
        QDockWidget {{ font-weight: 500; }}
        QDockWidget::title {{ padding: 7px 8px; background: palette(window); border-bottom: 1px solid {border}; }}
        QTabWidget::pane {{ border: 1px solid {border}; }}
        QTabBar::tab {{ padding: 7px 12px; }}
        QTabBar::tab:selected {{ border-bottom: 2px solid {accent}; }}
        QHeaderView::section {{ padding: 5px 7px; border: none; border-right: 1px solid {border}; border-bottom: 1px solid {border}; }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ min-height: 23px; }}
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {accent}; }}
        QTreeView {{ border: none; outline: none; }}
        QTreeView::item {{ min-height: 25px; }}
        QStatusBar {{ border-top: 1px solid {border}; }}
        QGroupBox {{ margin-top: 12px; padding-top: 8px; }}
        QLabel#workspaceHeading {{ font-size: 20px; font-weight: 600; }}
        QLabel#workspaceHint {{ color: palette(placeholder-text); }}
    """)
    app.setProperty("autopyroTheme", theme)
    app.setProperty("autopyroStyleEngine", native or "Fusion")


@lru_cache(maxsize=128)
def _icon(name, foreground):
    path = ASSETS / "lucide" / (ICON_ALIASES.get(name, name) + ".svg")
    if not path.is_file():
        return QApplication.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)
    data = path.read_text(encoding="utf-8").replace("currentColor", foreground)
    renderer = QSvgRenderer(QByteArray(data.encode("utf-8")))
    result = QIcon()
    for size in (16, 22, 32, 48, 64):
        image = QPixmap(size * 2, size * 2)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        renderer.render(painter, QRectF(0, 0, size * 2, size * 2))
        painter.end()
        image.setDevicePixelRatio(2)
        result.addPixmap(image)
    return result


class _PaletteIconEngine(QIconEngine):
    """Resolve colors when painted so open dialogs follow theme changes too."""

    def __init__(self, name):
        super().__init__()
        self.name = name

    def clone(self):
        return _PaletteIconEngine(self.name)

    def pixmap(self, size, mode, state):
        group = QPalette.ColorGroup.Disabled if mode == QIcon.Mode.Disabled else QPalette.ColorGroup.Active
        role = QPalette.ColorRole.HighlightedText if mode == QIcon.Mode.Selected else QPalette.ColorRole.WindowText
        foreground = QApplication.palette().color(group, role).name()
        return _icon(self.name, foreground).pixmap(size)

    def paint(self, painter, rect, mode, state):
        painter.drawPixmap(rect, self.pixmap(rect.size() * painter.device().devicePixelRatioF(), mode, state))


def icon(name):
    return QIcon(_PaletteIconEngine(name))
