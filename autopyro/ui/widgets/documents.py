"""Shared layout and preview context for docked analysis documents."""

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QSplitter, QTabWidget, QVBoxLayout, QWidget

from ..i18n import tr
from ..theme import icon


def analysis_document(dialog, *, parameter_title=None, parameter_icon='settings-2', scroll_parameters=True):
    outer = QVBoxLayout(dialog)
    splitter = QSplitter()
    outer.addWidget(splitter)
    dialog.parameter_tabs = QTabWidget()
    splitter.addWidget(dialog.parameter_tabs)
    settings = parameter_page(dialog.parameter_tabs, parameter_title or tr("Параметры"), parameter_icon,
                              scrollable=scroll_parameters)
    content = QWidget()
    results = QVBoxLayout(content)
    results.setContentsMargins(0, 0, 0, 0)
    splitter.addWidget(content)
    splitter.setSizes([390, 850])
    return settings, results


def parameter_page(tabs, title, icon_name, *, scrollable=True):
    parameters = QWidget()
    layout = QVBoxLayout(parameters)
    page = parameters
    if scrollable:
        page = QScrollArea()
        page.setWidgetResizable(True)
        page.setWidget(parameters)
    tabs.addTab(page, icon(icon_name), title)
    return layout


def source_token(owner):
    scope = owner.analysis_scope
    order = owner.proxy.ordered_row_ids() if scope.mode in ("filtered", "selected") else ()
    return id(owner.model), owner.analysis_revision, order, scope.revision


def refill_combo(combo, choices):
    if [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())] == choices:
        return False
    selected = combo.currentData()
    blocker = QSignalBlocker(combo)
    combo.clear()
    for label, identifier in choices:
        combo.addItem(label, identifier)
    combo.setCurrentIndex(max(0, combo.findData(selected)))
    del blocker
    return True


class _CheckCaption(QLabel):
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            checkbox = self.buddy()
            if checkbox is not None and checkbox.isEnabled():
                checkbox.setFocus(Qt.FocusReason.MouseFocusReason)
                checkbox.click()
                event.accept()
                return
        super().mouseReleaseEvent(event)


def wrapped_check(layout, checkbox, *, clickable=False):
    text = checkbox.text()
    checkbox.setProperty("parameterLabel", text)
    checkbox.setAccessibleName(text)
    checkbox.setToolTip(text)
    checkbox.setText("")
    label = _CheckCaption(text) if clickable else QLabel(text)
    label.setWordWrap(True)
    label.setBuddy(checkbox)
    row = QHBoxLayout()
    row.addWidget(checkbox)
    row.addWidget(label, 1)
    layout.addLayout(row)
    return row
