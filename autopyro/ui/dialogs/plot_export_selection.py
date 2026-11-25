"""Choose named plots, their order and the layout of a multi-panel export."""

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from ..i18n import tr
from autopyro.visualization.plotting import PlotSpec
from autopyro.visualization.plot_panels import panel_layout as raster_layout, svg_grid
from ..theme import icon


class PlotExportSelectionDialog(QDialog):
    def __init__(self, plots, parent=None, *, panels=False, can_use_style=False):
        super().__init__(parent)
        self.setWindowTitle(tr('Собрать многопанельный рисунок…') if panels else tr('Экспорт выбранных графиков'))
        layout = QVBoxLayout(self)
        choices = self.choices = QListWidget()
        if panels:
            hint = QLabel(tr('Отметьте графики. Перетащите их, чтобы изменить порядок панелей.'))
            hint.setWordWrap(True)
            layout.addWidget(hint)
            choices.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        for name in plots:
            item = QListWidgetItem(name, choices)
            item.setCheckState(Qt.CheckState.Unchecked)
        layout.addWidget(choices)

        columns = None
        if panels:
            order_buttons = QHBoxLayout()
            move_up = QPushButton(icon('upload'), tr('Вверх'))
            move_down = QPushButton(icon('download'), tr('Вниз'))
            order_buttons.addWidget(move_up)
            order_buttons.addWidget(move_down)
            order_buttons.addStretch()
            layout.addLayout(order_buttons)

            def update_order_buttons():
                row = choices.currentRow()
                move_up.setEnabled(row > 0)
                move_down.setEnabled(0 <= row < choices.count() - 1)

            def move_panel(offset):
                row = choices.currentRow()
                target = row + offset
                if row < 0 or not 0 <= target < choices.count():
                    return
                item = choices.takeItem(row)
                choices.insertItem(target, item)
                choices.setCurrentItem(item)
                update_order_buttons()
                update_preview()

            move_up.clicked.connect(lambda: move_panel(-1))
            move_down.clicked.connect(lambda: move_panel(1))
            choices.currentRowChanged.connect(update_order_buttons)
            choices.model().rowsMoved.connect(update_order_buttons)
            update_order_buttons()
            columns = QSpinBox()
            columns.setRange(1, 2)
            columns.setValue(2)
            columns.setEnabled(False)
            panel_layout = QFormLayout()
            panel_layout.addRow(tr('Число столбцов'), columns)
            layout.addLayout(panel_layout)
            preview = QTableWidget()
            preview.setAccessibleName(tr('Схема панелей'))
            preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            preview.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
            preview.horizontalHeader().hide()
            preview.verticalHeader().hide()
            preview.setMaximumHeight(130)
            layout.addWidget(QLabel(tr('Схема панелей')))
            layout.addWidget(preview)
            dimensions = QLabel()
            dimensions.setWordWrap(True)
            layout.addWidget(dimensions)

            def update_preview():
                names = self.selected_names()
                count = min(columns.value(), len(names))
                preview.clearContents()
                preview.setColumnCount(count)
                preview.setRowCount((len(names) + count - 1) // count if count else 0)
                for index, name in enumerate(names):
                    row, column = divmod(index, count)
                    item = QTableWidgetItem(name)
                    config = plots[name]
                    item.setToolTip(f"{name}\n{config.get('width_mm', 180):g} × {config.get('height_mm', 130):g} mm")
                    preview.setItem(row, column, item)
                dimensions.clear()
                if len(names) >= 2:
                    try:
                        specs = [PlotSpec.from_dict(plots[name]) for name in names]
                        svg, _, _ = svg_grid(specs, count)
                        vector = f"SVG · {svg.get('width')} × {svg.get('height')}"
                        dpi, _, _, _, (width, height) = raster_layout('panels.png', specs, count)
                        dimensions.setText(f'{vector}\nPNG · {width} × {height} px · {dpi} DPI ({names[0]})')
                    except (ValueError, TypeError, KeyError):
                        dimensions.setText(tr('Проверьте настройки экспорта выбранных графиков.'))

            columns.valueChanged.connect(update_preview)
            choices.model().rowsMoved.connect(update_preview)
        common_style = self.common_style = QCheckBox(tr('Единое оформление по текущему графику'))
        common_style.setToolTip(tr('Переносит фон, сетку и типографику. Настройки данных и размеры сохраняются.'))
        common_style.setEnabled(can_use_style)
        layout.addWidget(common_style)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        minimum = 2 if panels else 1

        def update_selection():
            count = len(self.selected_names())
            buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(count >= minimum)
            if columns is not None:
                with QSignalBlocker(columns):
                    columns.setMaximum(max(2, count))
                columns.setEnabled(count >= minimum)
                update_preview()

        choices.itemChanged.connect(update_selection)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.columns = columns

    def selected_names(self):
        return [self.choices.item(i).text() for i in range(self.choices.count())
                if self.choices.item(i).checkState() == Qt.CheckState.Checked]

    def panel_columns(self):
        return self.columns.value() if self.columns is not None else None
