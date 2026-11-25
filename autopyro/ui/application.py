import sys
from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase, QIcon
from PySide6.QtWidgets import QApplication

from .i18n import initialize


def main():
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("AutoPyro.Desktop")
    app = QApplication(sys.argv)
    # The Windows offscreen platform has no system fonts registered by default.
    # Register one for headless rendering; native Windows uses its normal fonts.
    if not QFontDatabase.families():
        font = Path("C:/Windows/Fonts/segoeui.ttf")
        if font.exists():
            QFontDatabase.addApplicationFont(str(font))
            app.setFont(QFont("Segoe UI", 10))
    app.setApplicationName("AutoPyro")
    app.setOrganizationName("AutoPyro")
    app.setWindowIcon(QIcon(str(Path(__file__).parent / "assets" / "autopyro-icon.png")))
    initialize()
    from .window import MainWindow

    window = MainWindow()
    window.show()
    return app.exec()
