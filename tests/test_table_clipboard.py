"""Native Windows clipboard contention must not silently lose table operations."""
import subprocess
import sys
from threading import Thread
from time import monotonic

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox, QWidget

from autopyro.io import build_project
from autopyro.ui.dialogs.batch_edit import BatchEditPreview
from autopyro.ui.table.clipboard import read_text, write_text
from autopyro.ui.window import MainWindow


def hold_clipboard(seconds):
    script = """
import ctypes, sys, time
user = ctypes.WinDLL('user32', use_last_error=True)
deadline = time.monotonic() + 3
while not user.OpenClipboard(None):
    if time.monotonic() > deadline:
        raise RuntimeError('Could not acquire test clipboard')
    time.sleep(.01)
print('locked', flush=True)
time.sleep(float(sys.argv[1]))
user.CloseClipboard()
"""
    process = subprocess.Popen([sys.executable, "-c", script, str(seconds)], stdout=subprocess.PIPE, text=True)
    output = []
    reader = Thread(target=lambda: output.append(process.stdout.readline().strip()), daemon=True)
    reader.start()
    deadline = monotonic() + 4
    while reader.is_alive() and monotonic() < deadline:
        QApplication.processEvents()
        reader.join(.005)
    assert output == ["locked"]
    return process


@pytest.fixture
def owner(qapp, qtbot):
    if sys.platform != "win32" or QApplication.platformName() != "windows":
        pytest.skip("Requires native Windows clipboard")
    widget = QWidget()
    qtbot.addWidget(widget)
    return widget


def test_native_unicode_roundtrip_after_temporary_lock(owner, qapp):
    write_text(owner, "Пробы\t001\n😀\tND\n")
    process = hold_clipboard(.15)
    try:
        assert read_text(owner) == "Пробы\t001\n😀\tND\n"
    finally:
        process.wait(timeout=5)
    process = hold_clipboard(.15)
    try:
        write_text(owner, "Новая\t=literal\n")
    finally:
        process.wait(timeout=5)
    assert qapp.clipboard().text() == "Новая\t=literal\n"


def test_large_windows_copy_does_not_read_the_payload_to_verify_it(owner, qapp, monkeypatch):
    def unexpected_read(*_):
        pytest.fail('Clipboard write materialized the Windows payload')

    monkeypatch.setattr('autopyro.ui.table.clipboard.read_text', unexpected_read)
    text = 'Пробы\t😀\t"многострочное\nзначение"\n' * 100_000
    write_text(owner, text)
    assert qapp.clipboard().text() == text
    assert read_text(owner) == text


def test_table_copy_and_paste_retry_and_report_persistent_lock(owner, qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window, before_close_func=lambda widget: setattr(widget, "dirty", False))
    project = build_project(["Sample"], [["001"]], [2], [(0, "Sample", "text", "")], {})
    window.set_project(project)
    window.table.selectRow(0)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    monkeypatch.setattr(BatchEditPreview, "exec", lambda self: QDialog.DialogCode.Accepted)

    process = hold_clipboard(.15)
    try:
        window.copy_cells()
    finally:
        process.wait(timeout=5)
    assert read_text(owner) == "001\n"
    write_text(owner, "002")
    process = hold_clipboard(.15)
    try:
        window.paste_cells()
    finally:
        process.wait(timeout=5)
    assert project.rows[0].values[project.columns[0].id] == "002"
    assert not warnings

    process = hold_clipboard(2)
    try:
        window.paste_cells()
        window.copy_cells()
        assert len(warnings) == 2
    finally:
        process.wait(timeout=5)
    assert window.model.undo.count() == 1
    assert read_text(owner) == "002"
