"""Options for saving a set of well columns with one appearance."""

from PySide6.QtWidgets import QCheckBox, QComboBox, QDialogButtonBox, QFileDialog, QFormLayout, QLabel

from ..i18n import tr
from ..theme import icon
from .stratigraphy_labels import CheckedValuesDialog


class ColumnExportDialog(QFileDialog):
    FORMATS = {'PNG (*.png)': 'png', 'SVG (*.svg)': 'svg', 'PDF (*.pdf)': 'pdf'}

    def __init__(self, include_samples, parent):
        super().__init__(parent)
        self.setOption(QFileDialog.Option.DontUseNativeDialog)
        self.setWindowTitle(tr('Экспорт колонки'))
        self.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        self.setNameFilters(list(self.FORMATS))
        self.selectNameFilter('PNG (*.png)')
        self.setDefaultSuffix('png')
        self.selectFile('column')
        self.filterSelected.connect(self.update_suffix)
        self.include_samples = QCheckBox(tr('Включить дорожку проб и замеров'))
        self.include_samples.setChecked(include_samples)
        layout = self.layout()
        layout.addWidget(self.include_samples, layout.rowCount(), 0, 1, layout.columnCount())

    def update_suffix(self, file_filter):
        self.setDefaultSuffix(self.FORMATS[file_filter])


class ColumnBatchDialog(CheckedValuesDialog):
    def __init__(self, wells, selected, templates, parent):
        super().__init__(tr('Экспорт комплекта колонок'), [(well, well) for well in wells], selected, parent)
        self.resize(540, 510)
        buttons = self.findChild(QDialogButtonBox)
        form = QFormLayout()
        self.format = QComboBox()
        self.format.addItems(['PNG', 'SVG', 'PDF'])
        form.addRow(tr('Формат'), self.format)
        self.template = QComboBox()
        self.template.addItem(tr('Текущее оформление'), None)
        for name in sorted(templates, key=str.casefold):
            self.template.addItem(name, name)
        form.addRow(tr('Шаблон оформления'), self.template)
        self.individual = QCheckBox(tr('Отдельная колонка для каждого ствола'))
        self.comparison = QCheckBox(tr('Общий лист сравнения'))
        for option in (self.individual, self.comparison):
            option.setChecked(True)
            option.toggled.connect(self.update_actions)
            form.addRow(option)
        self.include_samples = QCheckBox(tr('Включить дорожку проб и замеров'))
        form.addRow(self.include_samples)
        hint = QLabel(tr('Комплект сохраняется в новой папке. Используются выбранные уровни и шкала глубины.'))
        hint.setWordWrap(True)
        form.addRow(hint)
        self.layout().insertLayout(self.layout().indexOf(buttons), form)
        self.export_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.export_button.setText(tr('Экспортировать'))
        self.export_button.setIcon(icon('log-out'))
        self.list.itemChanged.connect(self.update_actions)
        self.update_actions()

    def update_actions(self):
        count = len(self.values())
        self.comparison.setEnabled(count > 1)
        self.export_button.setEnabled(count > 0 and (self.individual.isChecked() or self.includes_comparison()))

    def includes_comparison(self):
        return self.comparison.isEnabled() and self.comparison.isChecked()
