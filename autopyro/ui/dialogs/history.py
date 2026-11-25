"""Select an action and expand only its protocol, keeping the journal compact."""
from copy import deepcopy
import json

from PySide6.QtWidgets import QListWidget, QSplitter, QTextEdit, QVBoxLayout

from autopyro.model.audit import MAGIC, read_protocol
from ..i18n import tr
from ..tasks import TaskDialog


class HistoryDialog(TaskDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("История изменений · исходные значения доступны в подсказках ячеек"))
        self.resize(1000, 650)
        self.events = list(project.history)
        self.actions = QListWidget()
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        splitter = QSplitter()
        splitter.addWidget(self.actions)
        splitter.addWidget(self.details)
        splitter.setStretchFactor(1, 1)
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)
        for index, event in enumerate(self.events):
            label = f"{event.get('at', '')} · {event.get('action', '')}" if isinstance(event, dict) else str(event)
            self.actions.addItem(f"{index + 1}. {label}")
        self.actions.currentRowChanged.connect(self.show_event)
        if self.events:
            self.actions.setCurrentRow(len(self.events) - 1)

    def show_event(self, index):
        if not 0 <= index < len(self.events):
            return
        event = self.events[index]

        def render():
            if isinstance(event, dict):
                data = {key: deepcopy(value) for key, value in event.items() if key != "protocol"}
                if "protocol" in event:
                    data["protocol"] = read_protocol(event["protocol"])
                metadata = data.get("metadata")
                if isinstance(metadata, dict):
                    snapshot = metadata.get("interpolation_protocol")
                    if isinstance(snapshot, str) and snapshot.startswith(MAGIC):
                        metadata["interpolation_protocol"] = read_protocol(snapshot)
            else:
                data = deepcopy(event)
            return json.dumps(data, ensure_ascii=False, indent=2)

        self.run_task(render, self.details.setPlainText)
