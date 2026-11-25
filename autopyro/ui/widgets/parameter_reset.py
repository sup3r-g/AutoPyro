"""Individual resets that reuse declared defaults and existing change signals."""

from copy import deepcopy
from html import escape

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFormLayout, QHBoxLayout,
    QLineEdit, QListWidget, QPlainTextEdit, QSpinBox, QDoubleSpinBox,
    QTextEdit, QToolButton, QWidget, QMenu,
)
from shiboken6 import isValid

from ..i18n import tr
from ..theme import icon
from .parameter_defaults import ParameterDefaults


class ResetParameterButton(QToolButton):
    def __init__(self):
        super().__init__()
        self.setIcon(icon('edit-undo'))

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            self.setIcon(icon('edit-undo'))


def parameter_field(widget):
    """The field placed in its layout, including an individual reset button."""
    return getattr(widget, 'parameter_holder', widget)


def notify_parameter(widget):
    """Notify existing preview/validation handlers after a blocked restoration."""
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        widget.valueChanged.emit(widget.value())
    elif isinstance(widget, QComboBox):
        widget.currentIndexChanged.emit(widget.currentIndex())
        widget.currentTextChanged.emit(widget.currentText())
        if hasattr(widget, 'editingFinished'):
            widget.editingFinished.emit()
    elif isinstance(widget, QCheckBox):
        widget.toggled.emit(widget.isChecked())
    elif isinstance(widget, QLineEdit):
        widget.textChanged.emit(widget.text())
        widget.textEdited.emit(widget.text())
        widget.editingFinished.emit()
    elif isinstance(widget, (QTextEdit, QPlainTextEdit)):
        widget.textChanged.emit()
    elif isinstance(widget, QListWidget) and widget.count():
        widget.itemChanged.emit(widget.item(0))


def _wrap(widget, button):
    # Keep layouts and controls alive: their signal connections and public
    # attributes continue to refer to the original editor.
    parent = widget.parentWidget()
    if parent is None:
        return False
    layouts = parent.findChildren(QHBoxLayout) + parent.findChildren(QFormLayout)
    if parent.layout() is not None:
        layouts.insert(0, parent.layout())
    layout = next((item for item in layouts if item.indexOf(widget) >= 0), None)
    if layout is None:
        return False
    holder = QWidget(parent)
    holder.setObjectName('parameterWithReset')
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(3)
    holder.setSizePolicy(widget.sizePolicy())
    replaced = layout.replaceWidget(widget, holder)
    if replaced is None:
        holder.deleteLater()
        return False
    row.addWidget(widget, 1)
    row.addWidget(button, 0, Qt.AlignmentFlag.AlignTop)
    holder.setFocusProxy(widget)
    widget.parameter_holder = holder
    return True


def install_parameter_resets(root, defaults=None, *, changed=None):
    snapshot = ParameterDefaults(root, exclude=getattr(defaults, 'excluded', ()))
    declared = dict(defaults.entries) if defaults is not None else {}
    if defaults is not None:
        install_table_resets(defaults, changed=changed)
    for widget, captured in snapshot.entries:
        if hasattr(widget, 'parameter_reset_button'):
            continue
        # Cell editors belong to parameter tables; wrapping a cell widget
        # would change the QTableWidget.cellWidget() API used by calculations.
        ancestor = widget.parentWidget()
        in_table = False
        form = root if root.property('parameterForm') else None
        while ancestor is not None and ancestor is not root:
            if isinstance(ancestor, QAbstractItemView):
                in_table = True
                break
            if ancestor.property('parameterForm'):
                form = ancestor
                break
            ancestor = ancestor.parentWidget()
        if in_table:
            continue
        key = next((key for key, (field, kind) in form.fields.items() if field is widget), None) if form is not None else None
        state = deepcopy(getattr(widget, 'parameter_default_state', declared.get(widget, captured)))
        default = form.defaults[key] if form is not None and key is not None else state[2] if state[0] == 'combo' else state[1]
        sensitive = isinstance(widget, QLineEdit) and widget.echoMode() != QLineEdit.EchoMode.Normal
        if not sensitive and state[0] != 'list':
            if isinstance(default, bool):
                default = tr('Включено') if default else tr('Выключено')
            elif default is None or default == '':
                default = tr('Пусто / автоматически')
            elif form is not None and key is not None and form.fields[key][1] == 'combo':
                index = widget.findData(default)
                if index >= 0:
                    default = widget.itemText(index)
            if tr('По умолчанию: {0}', '').strip() not in widget.whatsThis():
                widget.setWhatsThis(widget.whatsThis() + '<br>' + escape(tr('По умолчанию: {0}', default)))
                widget.setToolTip(widget.whatsThis())
        button = ResetParameterButton()
        button.setObjectName('resetParameter')
        title = widget.property('parameterLabel') or widget.accessibleName()
        caption = tr('Сбросить параметр «{0}»', title) if title else tr('Сбросить параметр')
        button.setAccessibleName(caption)
        button.setToolTip(escape(caption) + '<br>' + widget.whatsThis())
        button.setToolTipDuration(20000)
        button.setAutoRaise(True)
        button.setFixedSize(24, 24)
        button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        if not _wrap(widget, button):
            button.deleteLater()
            continue

        def restore(checked=False, *, field=widget, owner=form, name=key, initial=state):
            if owner is not None and name is not None:
                owner.reset_parameter(name)
            else:
                single = ParameterDefaults.__new__(ParameterDefaults)
                single.entries = [(field, initial)]
                single.table_items = []
                single.restore()
                notify_parameter(field)
                if changed is not None:
                    changed()

        button.clicked.connect(restore)
        widget.parameter_reset_button = button


def install_table_resets(defaults, *, changed=None):
    """Reset a clicked parameter cell without adding actions to result tables."""
    states = dict(defaults.entries)
    for table, values in defaults.table_items:
        if hasattr(table, 'reset_parameter_cell'):
            continue
        items = {(row, col): (table.item(row, col), text) for row, col, text in values}
        fields = {(row, col): (widget, states[widget])
                  for row in range(table.rowCount()) for col in range(table.columnCount())
                  if (widget := table.cellWidget(row, col)) in states}

        def reset_cell(row, col, *, target=table, originals=items, controls=fields):
            if (row, col) in controls:
                widget, state = controls[row, col]
                if not isValid(widget) or target.cellWidget(row, col) is not widget:
                    return
                single = ParameterDefaults.__new__(ParameterDefaults)
                single.entries, single.table_items = [(widget, state)], []
                single.restore()
                notify_parameter(widget)
            elif (row, col) in originals:
                item, text = originals[row, col]
                if not isValid(item) or target.item(row, col) is not item or not item.flags() & Qt.ItemFlag.ItemIsEditable:
                    return
                item.setText(text)
            if changed is not None:
                changed()

        install_cell_reset_menu(table, reset_cell)
        for item, text in items.values():
            item.setToolTip(table.whatsThis() + '<br>' + tr('Сброс отдельной ячейки доступен через контекстное меню.'))


def install_cell_reset_menu(table, callback):
    def menu(position):
        index = table.indexAt(position)
        if not index.isValid():
            index = table.currentIndex()
        if not index.isValid():
            return
        row, col = index.row(), index.column()
        item = table.item(row, col)
        if table.cellWidget(row, col) is None and (item is None or not item.flags() & Qt.ItemFlag.ItemIsEditable):
            return
        popup = QMenu(table)
        action = popup.addAction(icon('rotate-ccw'), tr('Сбросить параметр'))
        action.triggered.connect(lambda: callback(row, col))
        popup.exec(table.viewport().mapToGlobal(position))

    table.reset_parameter_cell = callback
    table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    table.customContextMenuRequested.connect(menu)
