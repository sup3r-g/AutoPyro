"""Icons for standard Qt controls and explicitly registered dialog commands."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialogButtonBox, QLineEdit, QPushButton, QToolBar

from . import icon


STANDARD_BUTTON_ICONS = {
    QDialogButtonBox.StandardButton.Ok: 'check',
    QDialogButtonBox.StandardButton.Cancel: 'x',
    QDialogButtonBox.StandardButton.Close: 'x',
    QDialogButtonBox.StandardButton.Apply: 'check',
    QDialogButtonBox.StandardButton.Save: 'save',
    QDialogButtonBox.StandardButton.Open: 'folder-open',
    QDialogButtonBox.StandardButton.Yes: 'check',
    QDialogButtonBox.StandardButton.No: 'x',
    QDialogButtonBox.StandardButton.RestoreDefaults: 'rotate-ccw',
    QDialogButtonBox.StandardButton.Help: 'circle-question-mark',
}


def add_command_button(box, icon_name, text, role):
    button = QPushButton(icon(icon_name), text)
    box.addButton(button, role)
    return button


def add_search_icon(field):
    field.addAction(icon('search'), QLineEdit.ActionPosition.LeadingPosition)


def apply_control_icons(widget):
    """Style standard Qt buttons by role and enable icons on text-only toolbars."""
    for box in widget.findChildren(QDialogButtonBox):
        for standard, name in STANDARD_BUTTON_ICONS.items():
            button = box.button(standard)
            if button is not None:
                button.setIcon(icon(name))
    for toolbar in widget.findChildren(QToolBar):
        if toolbar.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonTextOnly:
            toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
