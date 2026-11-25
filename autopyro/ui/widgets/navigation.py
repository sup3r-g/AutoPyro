"""Common viewing tools; editing and export belong to the application UI."""

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QMenu, QToolButton

from ..i18n import tr
from ..theme import icon


def add_tab_sections_menu(tabs, accessible_name):
    button = QToolButton(tabs)
    button.setText(tr('Разделы…'))
    fallback_icon = icon('layers')
    button.setIcon(fallback_icon)
    button.setAccessibleName(accessible_name)
    button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    menu = QMenu(button)
    for index in range(tabs.count()):
        action = menu.addAction(fallback_icon, tabs.tabText(index))
        action.setCheckable(True)
        action.triggered.connect(lambda checked=False, page=index: tabs.setCurrentIndex(page))

    def sync_menu():
        for index, action in enumerate(menu.actions()):
            action.setChecked(index == tabs.currentIndex())
            tab_icon = tabs.tabIcon(index)
            action.setIcon(fallback_icon if tab_icon.isNull() else tab_icon)

    menu.aboutToShow.connect(sync_menu)
    button.setMenu(menu)
    tabs.setCornerWidget(button, Qt.Corner.TopRightCorner)
    return button


class NavigationToolbar(NavigationToolbar2QT):
    toolitems = tuple(item for item in NavigationToolbar2QT.toolitems
                      if item[3] in ('home', 'back', 'forward', 'pan', 'zoom'))

    def __init__(self, canvas, parent=None, coordinates=True):
        super().__init__(canvas, parent, coordinates=coordinates)
        self.setMovable(False)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.setIconSize(QSize(22, 22))
        labels = {
            'home': (tr('Исходный вид'), tr('Вернуться к исходному виду осей'), 'rotate-ccw'),
            'back': (tr('Предыдущий вид'), tr('Назад по истории просмотра'), 'undo-2'),
            'forward': (tr('Следующий вид'), tr('Вперёд по истории просмотра'), 'redo-2'),
            'pan': (tr('Перемещение'), tr('Перемещать область просмотра мышью'), 'move'),
            'zoom': (tr('Масштабирование'), tr('Выделить рамкой область для увеличения'), 'zoom-in'),
        }
        for key, (label, help_text, image) in labels.items():
            action = self._actions[key]
            action.setText(label)
            action.setToolTip(help_text)
            action.setWhatsThis(help_text)
            action.setIcon(icon(image))

    def remap_history(self, old_axes, new_axes):
        # Remote frames replace Axes objects, but keep the navigation history.
        mapping = dict(zip(old_axes, new_axes))
        for entry in self._nav_stack:
            replacement = {mapping[ax]: value for ax, value in entry.items() if ax in mapping}
            entry.clear()
            entry.update(replacement)

    def reconnect_canvas(self):
        self._id_press = self.canvas.mpl_connect('button_press_event', self._zoom_pan_handler)
        self._id_release = self.canvas.mpl_connect('button_release_event', self._zoom_pan_handler)
        self._id_drag = self.canvas.mpl_connect('motion_notify_event', self.mouse_move)

    def cancel_drag(self):
        """A hidden canvas may never receive the mouse-release event."""
        if self._pan_info is not None:
            self.release_pan(None)
        if self._zoom_info is not None:
            self.canvas.mpl_disconnect(self._zoom_info.cid)
            self.remove_rubberband()
            self._zoom_info = None
        if self.mode.name == 'PAN':
            self.pan()
        elif self.mode.name == 'ZOOM':
            self.zoom()
