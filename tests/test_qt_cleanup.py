"""A failing GUI test must still close its window without a save prompt."""
from pathlib import Path
import os
import shutil
import subprocess
import sys


def test_failed_gui_test_discards_edits_before_closing(tmp_path):
    shutil.copy2(Path(__file__).with_name('conftest.py'), tmp_path / 'conftest.py')
    marker = tmp_path / 'dirty-at-close.txt'
    test = tmp_path / 'test_failing_window.py'
    test.write_text('''
from pathlib import Path
from autopyro.ui.window import MainWindow

class RecordingWindow(MainWindow):
    def closeEvent(self, event):
        Path(__file__).with_name('dirty-at-close.txt').write_text(str(self.dirty))
        event.accept()

def test_deliberate_failure(qtbot):
    window = RecordingWindow()
    qtbot.addWidget(window)
    window.dirty = True
    assert False, 'deliberate GUI failure'
''', encoding='utf-8')
    package_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', test.name, '-q', '--tb=short',
         f'--rootdir={tmp_path}', f'--confcutdir={tmp_path}'],
        cwd=tmp_path,
        env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen',
             'PYTHONPATH': str(package_root) + os.pathsep + os.environ.get('PYTHONPATH', '')},
        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'deliberate GUI failure' in result.stdout
    assert marker.read_text() == 'False'
