"""Large source signals retain extrema while keeping preview work bounded."""
from threading import Event

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFileDialog

from autopyro.model.core import Project, Row
from autopyro.ui.widgets import pyrograms


def test_large_preview_preserves_spikes_temperature_turns_and_order():
    x = np.arange(300000, dtype=float)
    y = np.zeros(len(x))
    x[12345], x[12346] = -100, 400000
    y[54321], y[54322] = -90, 150
    reduced_x, reduced_y = pyrograms.preview_signal(x, y)
    assert len(reduced_x) <= 6000
    assert (-100, 400000) == (reduced_x.min(), reduced_x.max())
    assert (-90, 150) == (reduced_y.min(), reduced_y.max())
    assert (reduced_x[0], reduced_x[-1]) == (x[0], x[-1])
    assert list(reduced_x).index(-100) < list(reduced_x).index(400000)


def test_import_keeps_gui_timer_running(qtbot, monkeypatch):
    project = Project(rows=[Row(values={})])
    project.sync_measurements()
    rid = project.rows[0].id
    dialog = pyrograms.PyrogramDialog(None, project, rid)
    qtbot.addWidget(dialog)
    started, resumed = Event(), Event()
    ticks = []

    def read(*args):
        started.set()
        assert resumed.wait(3), "GUI timer stalled during import"
        return dict(file="large.json", sha256="test", run=dict(label="Large", traces=[]))

    timer = QTimer()
    timer.setInterval(1)

    def heartbeat():
        if started.is_set():
            ticks.append(True)
            resumed.set()

    timer.timeout.connect(heartbeat)
    monkeypatch.setattr(pyrograms, "read_pyrogram", read)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("large.json", ""))
    timer.start()
    try:
        dialog.add_run()
    finally:
        timer.stop()
    assert ticks
    assert project.science.samples[rid].metadata["pyrograms"][0]["file"] == "large.json"


def test_zoom_restores_source_detail_and_crossing_segments():
    x = np.arange(1000000, dtype=float)
    y = np.sin(x)
    zoom_x, zoom_y = pyrograms.preview_signal(x, y, bounds=(501000, 501100))
    assert np.array_equal(zoom_x, x[500999:501102])
    assert np.array_equal(zoom_y, y[500999:501102])
    reversed_x, _ = pyrograms.preview_signal(x, y, bounds=(501100, 501000))
    assert np.array_equal(zoom_x, reversed_x)
    crossing_x, _ = pyrograms.preview_signal([0, 10, 0], [1, 2, 3], bounds=(4, 6))
    assert crossing_x.tolist() == [0, 10, 0]
