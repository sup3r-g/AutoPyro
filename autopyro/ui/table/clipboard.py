"""Checked Unicode clipboard access for the native Windows Qt platform."""
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
from time import monotonic

from PySide6.QtCore import QEventLoop, QMimeData, QTimer
from PySide6.QtWidgets import QApplication

from ..i18n import tr

CF_UNICODETEXT = 13


def _windows_api():
    user = ctypes.WinDLL("user32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = (
        (user.OpenClipboard, [wintypes.HWND], wintypes.BOOL),
        (user.CloseClipboard, [], wintypes.BOOL),
        (user.IsClipboardFormatAvailable, [wintypes.UINT], wintypes.BOOL),
        (user.GetClipboardData, [wintypes.UINT], wintypes.HANDLE),
        (kernel.GlobalLock, [wintypes.HGLOBAL], ctypes.c_void_p),
        (kernel.GlobalUnlock, [wintypes.HGLOBAL], wintypes.BOOL),
        (kernel.GlobalSize, [wintypes.HGLOBAL], ctypes.c_size_t),
    )
    for function, arguments, result in signatures:
        function.argtypes, function.restype = arguments, result
    return user, kernel


@contextmanager
def _open_clipboard(owner):
    user, kernel = _windows_api()
    deadline = monotonic() + 0.6
    window_id = int(owner.winId())
    while not user.OpenClipboard(window_id):
        if monotonic() >= deadline:
            raise ValueError(tr('Буфер обмена занят другим приложением. Повторите операцию.'))
        loop = QEventLoop()
        QTimer.singleShot(25, loop.quit)
        loop.exec(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
    try:
        yield user, kernel
    finally:
        user.CloseClipboard()


def read_text(owner):
    if QApplication.platformName() != "windows":
        return QApplication.clipboard().text()
    with _open_clipboard(owner) as (user, kernel):
        if not user.IsClipboardFormatAvailable(CF_UNICODETEXT):
            return ""
        handle = user.GetClipboardData(CF_UNICODETEXT)
        pointer = kernel.GlobalLock(handle) if handle else None
        if not pointer:
            raise ValueError(tr('Не удалось прочитать табличные данные из буфера обмена.'))
        try:
            # Bound the read by the clipboard allocation, including malformed external data.
            data = ctypes.string_at(pointer, kernel.GlobalSize(handle))
            return data.decode("utf-16-le").split("\0", 1)[0].replace("\r\n", "\n")
        finally:
            kernel.GlobalUnlock(handle)


def write_text(owner, text, *, check_source=None):
    clipboard = QApplication.clipboard()
    if QApplication.platformName() != "windows":
        if check_source is not None:
            check_source()
        clipboard.setText(text)
        return
    error = tr('Не удалось записать данные в буфер обмена. Повторите копирование.')
    if "\0" in text:
        raise ValueError(error)
    deadline = monotonic() + 0.6
    while True:
        with _open_clipboard(owner):
            if check_source is not None:
                check_source()
        mime = QMimeData()
        mime.setText(text)
        clipboard.setMimeData(mime)
        # Check ownership of this write without materializing the full Windows payload.
        if clipboard.ownsClipboard() and clipboard.mimeData() is mime:
            return
        if monotonic() >= deadline:
            raise ValueError(error)
        loop = QEventLoop()
        QTimer.singleShot(25, loop.quit)
        loop.exec(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
