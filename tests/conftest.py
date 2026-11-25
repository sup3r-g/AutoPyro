"""Version-specific storage fixtures and cleanup of Qt-owned test services."""
from contextlib import closing
import json
import sqlite3

import pytest


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_teardown(item):
    """Discard test-only edits before pytest-qt closes registered windows.

    pytest-qt closes widgets before yield-fixture finalizers run, so clearing
    MainWindow.dirty in a fixture's finalizer is too late to avoid a modal save
    prompt. This hook also runs when an assertion prevents end-of-test cleanup.
    """
    widgets = getattr(item, 'qt_widgets', ())
    if widgets:
        from PySide6.QtWidgets import QMainWindow
        for reference, _ in widgets:
            widget = reference()
            if isinstance(widget, QMainWindow) and hasattr(widget, 'dirty'):
                widget.dirty = False
    yield


@pytest.fixture(autouse=True)
def flush_qt_deferred_deletes():
    """Dispose widgets closed by pytest-qt before the next test changes styles.

    pytest-qt closes its widgets before fixture teardown, but processEvents()
    alone does not deliver DeferredDelete without a running QApplication.exec().
    Keeping those closed windows alive makes later palette/style changes visit
    every widget created by all previous tests.
    """
    yield
    from PySide6.QtCore import QCoreApplication, QEvent
    if QCoreApplication.instance() is not None:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture(scope="session", autouse=True)
def shutdown_qt_services():
    """pytest has no QApplication.exec() exit to stop app-owned worker services."""
    yield
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is not None:
        app.aboutToQuit.emit()
        app.processEvents()


@pytest.fixture
def write_v3_project():
    def write(project, path):
        from autopyro.database.storage import encode_record
        project.save(path)
        with closing(sqlite3.connect(path)) as connection:
            for column in project.columns:
                connection.execute("UPDATE table_columns SET payload=? WHERE id=?", (encode_record(column), column.id))
            for position, event in enumerate(project.history):
                protocol_id = connection.execute(
                    "SELECT protocol_id FROM history_events WHERE position=?", (position,)).fetchone()[0]
                payload = {**event, "protocol": None} if protocol_id is not None else event
                connection.execute("UPDATE history_events SET payload=? WHERE position=?",
                                   (json.dumps(payload, ensure_ascii=False, allow_nan=False), position))
            for table in ("column_protocols", "history_nested_protocols", "stored_protocol_parts", "stored_protocols"):
                connection.execute(f"DROP TABLE {table}")
            connection.execute("PRAGMA user_version=3")
            connection.commit()
        return path
    return write


@pytest.fixture
def write_v2_project(write_v3_project):
    def write(project, path):
        # Recreate the former metadata-only history layout in a disposable test file.
        write_v3_project(project, path)
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("INSERT INTO metadata VALUES ('history',?)",
                               (json.dumps(project.history, ensure_ascii=False, allow_nan=False),))
            connection.execute("DELETE FROM metadata WHERE key='history_event_count'")
            connection.execute("DROP TABLE history_events")
            connection.execute("DROP TABLE history_protocols")
            connection.execute("PRAGMA user_version=2")
            connection.commit()
        return path
    return write
