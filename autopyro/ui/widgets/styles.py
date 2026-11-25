"""Library-backed choices in category style tables."""

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QComboBox, QStyledItemDelegate, QStyleOptionViewItem
from autopyro.visualization.plot_catalog import markers
from autopyro.visualization.plot_catalog import marker_value
from ..i18n import tr
from .forms import ColorCombo
from .style_labels import color_icon, color_label, marker_icon, marker_label


class StyleDelegate(QStyledItemDelegate):
    def __init__(self, table, entries=None, *, automatic_color=None):
        super().__init__(table)
        self.entries = entries or (lambda: [('color',)] * table.rowCount())
        self.automatic_color = automatic_color

    def createEditor(self, parent, option, index):
        entries = self.entries()
        if index.row() >= len(entries) or entries[index.row()][0] not in ('color', 'shape'):
            return super().createEditor(parent, option, index)
        if entries[index.row()][0] == 'color':
            editor = ColorCombo('', parent)
            if self.automatic_color is not None:
                editor.lineEdit().setPlaceholderText(tr('Автоматически'))
            return editor
        editor = QComboBox(parent)
        editor.setEditable(True)
        editor.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        editor.setIconSize(QSize(32, 20))
        for value in markers():
            editor.addItem(marker_icon(value), marker_label(value), value)
        return editor

    def setEditorData(self, editor, index):
        if isinstance(editor, ColorCombo):
            editor.setText(index.data())
        elif isinstance(editor, QComboBox):
            item = editor.findData(marker_value(str(index.data())))
            if item < 0:
                item = editor.findData(index.data())
            if item >= 0:
                editor.setCurrentIndex(item)
            else:
                editor.setEditText(str(index.data()))
        else:
            super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if isinstance(editor, ColorCombo):
            model.setData(index, editor.text() or '')
        elif isinstance(editor, QComboBox):
            value = editor.currentData() if editor.currentText() == editor.itemText(editor.currentIndex()) else editor.currentText()
            model.setData(index, str(value))
        else:
            super().setModelData(editor, model, index)

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        entries = self.entries()
        if index.row() >= len(entries):
            return
        if entries[index.row()][0] == 'color':
            color = index.data()
            if self.automatic_color is not None and not (color or '').strip():
                color = self.automatic_color(index)
                option.text = f"{tr('Автоматически')} · {color}"
            else:
                option.text = color_label(color)
            option.icon = color_icon(color)
            option.features |= QStyleOptionViewItem.ViewItemFeature.HasDecoration
            option.decorationSize = QSize(24, 14)
        elif entries[index.row()][0] == 'shape':
            option.text = marker_label(marker_value(str(index.data())))
            option.icon = marker_icon(marker_value(str(index.data())))
            option.features |= QStyleOptionViewItem.ViewItemFeature.HasDecoration
            option.decorationSize = QSize(32, 20)
