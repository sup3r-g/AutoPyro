"""Native Qt regressions that offscreen widget tests cannot detect."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_empty_workspace_defers_scientific_imports_until_interpolation(tmp_path):
    script = '''
import sys
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from autopyro.ui import i18n
from autopyro.ui.window import MainWindow

app = QApplication([])
store = QSettings(sys.argv[1], QSettings.Format.IniFormat)
i18n.settings = lambda: store
i18n.initialize()
window = MainWindow()
assert 'autopyro.ui.dialogs.interpolation' not in sys.modules
assert 'scipy.stats' not in sys.modules
window.interpolate()
from autopyro.ui.dialogs.interpolation import InterpolationDialog
assert isinstance(window.interpolation_window, InterpolationDialog)
assert window.desktop.documents.currentWidget() is window.interpolation_window
window.dirty = False
window.close()
app.aboutToQuit.emit()
app.processEvents()
'''
    result = subprocess.run(
        [sys.executable, '-c', script, str(tmp_path / 'settings.ini')],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(sys.platform != "win32" or os.environ.get("AUTOPYRO_NATIVE_SMOKE") != "1",
                    reason="Opt in to opening native Windows Qt windows with AUTOPYRO_NATIVE_SMOKE=1")
@pytest.mark.parametrize("theme", ["breeze_light", "breeze_dark"])
def test_native_startup_with_current_and_legacy_layouts(tmp_path, theme):
    script = tmp_path / "native_startup.py"
    script.write_text('''
import faulthandler
import sys
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication, QDockWidget, QMainWindow
from autopyro.ui import i18n
from autopyro.ui.window import MainWindow
from autopyro.ui.workspace import LAYOUT_VERSION

faulthandler.enable()
app = QApplication([])
store = QSettings(sys.argv[1], QSettings.Format.IniFormat)
store.setValue("appearance/theme", sys.argv[2])
i18n.settings = lambda: store
i18n.initialize()
first = MainWindow()
first.resize(1024, 700)
first.show()
app.processEvents()
assert first.isVisible()
desktop = first.desktop
welcome = desktop.table_page.currentWidget()
assert welcome.horizontalScrollBar().maximum() == 0
for button in desktop.welcome_buttons:
    assert button.isVisible() and button.width() >= button.sizeHint().width()
    assert welcome.viewport().rect().contains(button.mapTo(welcome.viewport(), button.rect().bottomRight()))
first.demo()
for method in (first.show_calculations, first.show_statistics, first.show_qc, first.show_plots, first.show_maps,
               first.show_stratigraphy, first.show_geochemistry_workspace, first.show_nonpyro, first.interpolate,
               first.show_curve_fitting, first.show_interpretation):
    method()
    app.processEvents()
    assert first.size().toTuple() == (1024, 700), (method.__name__, first.size().toTuple())
    document = first.desktop.documents.currentWidget()
    parameters = getattr(document, "parameter_tabs", None) or document.tabs
    assert first.desktop.properties_stack.currentWidget() is parameters
    assert desktop.project_dock.isVisible() and desktop.properties_dock.isVisible()
first.desktop.close_document(first.desktop.documents.currentIndex())
app.processEvents()
assert first.desktop.documents.indexOf(first.interpretation_window) < 0
assert first.size().toTuple() == (1024, 700)
first.desktop.documents.tabBar().moveTab(0, first.desktop.documents.count() - 1)
first.desktop.open_table()
assert first.desktop.documents.currentWidget() is first.desktop.table_page
assert first.size().toTuple() == (1024, 700)
first.desktop.close_documents(keep=first.desktop.table_page)
assert first.desktop.documents.count() == 1
first.desktop.close_documents()
assert first.desktop.documents.count() == 0
first.show_interpretation()
assert first.desktop.documents.indexOf(first.interpretation_window) == 0
first.desktop.close_document(0)
assert first.desktop.documents.count() == 0
first.desktop.open_table()
assert first.desktop.documents.currentWidget() is first.desktop.table_page
app.processEvents()
store.setValue("workspace/state", first.saveState(LAYOUT_VERSION))
second = MainWindow()
second.show()
app.processEvents()
assert second.isVisible()

legacy = QMainWindow()
for name in ("projectExplorerDock", "columnVisibilityDock", "propertiesExplorerDock"):
    dock = QDockWidget(name, legacy)
    dock.setObjectName(name)
    legacy.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
legacy.tabifyDockWidget(legacy.findChild(QDockWidget, "projectExplorerDock"),
                       legacy.findChild(QDockWidget, "columnVisibilityDock"))
legacy.findChild(QDockWidget, "propertiesExplorerDock").hide()
store.setValue("workspace/state", legacy.saveState(1))
third = MainWindow()
third.show()
app.processEvents()
assert third.isVisible() and not third.desktop.properties_dock.isHidden()
for window in (first, second, third):
    window.may_replace = lambda: True
    window.close()
print("Native startup and layout round trips passed")
''', encoding="utf-8")
    env = {**os.environ, "QT_QPA_PLATFORM": "windows"}
    result = subprocess.run([sys.executable, str(script), str(tmp_path / "settings.ini"), theme],
                            cwd=Path(__file__).resolve().parents[1], env=env,
                            # Includes cold imports and opening eleven native analysis documents.
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Native startup and layout round trips passed" in result.stdout
