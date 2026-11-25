"""Edit the zero reference and named ties between wellbores."""

from copy import deepcopy

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QScrollArea, QTabWidget, QVBoxLayout, QWidget,
)

from autopyro.geology.stratigraphy import column_style
from ..i18n import tr
from ..theme import icon
from ..theme.controls import apply_control_icons
from .stratigraphy_labels import geological_label


class CorrelationDialog(QDialog):
    def __init__(self, wells, intervals, settings, unit, parent):
        super().__init__(parent)
        self.setWindowTitle(tr('Корреляционные горизонты'))
        self.resize(720, 420)
        layout = QVBoxLayout(self)
        self.mode = QComboBox()
        for label, edge in [(tr('Исходная глубина'), None), (tr('По кровле'), 'top'),
                            (tr('По подошве'), 'bottom')]:
            self.mode.addItem(label, edge)
        self.saved_settings = deepcopy(settings)
        self.interval_wells = {item['id']: item['wellbore'] for item in intervals}
        self.last_edge = settings.get('edge')
        self.mode.setCurrentIndex(max(0, self.mode.findData(self.last_edge)) if settings.get('enabled', True) else 0)
        layout.addWidget(self.mode)
        hint = QLabel(tr('Выберите один интервал для каждого ствола. Выбранная граница станет нулём. '
                         'Пределы глубин в оформлении относятся к активной шкале.'))
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.panel = QWidget()
        form = QFormLayout(self.panel)
        self.selectors = {}
        by_well = {well: [] for well in wells}
        for item in intervals:
            if item['wellbore'] in by_well:
                by_well[item['wellbore']].append(item)
        for well, items in by_well.items():
            combo = QComboBox()
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(20)
            combo.addItem(tr('Выберите интервал'), None)
            for item in sorted(items, key=lambda item: (item['top'], item['bottom'])):
                label = (f"{item['name']} · {geological_label(item['level'])} · "
                         f"{item['top']:.10g}–{item['bottom']:.10g} {unit}")
                combo.addItem(label, item['id'])
                combo.setItemData(combo.count() - 1, label, Qt.ItemDataRole.ToolTipRole)
            combo.setCurrentIndex(max(0, combo.findData(settings.get('anchors', {}).get(well))))
            self.selectors[well] = combo
            form.addRow(well, combo)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.scroll_panel(self.panel), icon('layers'), tr('Нулевой горизонт'))
        self.tabs.addTab(self.extra_panel(settings), icon('list-plus'), tr('Дополнительные горизонты'))
        layout.addWidget(self.tabs, 1)
        self.validation = QLabel()
        self.validation.setWordWrap(True)
        layout.addWidget(self.validation)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.apply_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.apply_button.setText(tr('Применить'))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        for combo in (self.mode, *self.selectors.values()):
            combo.currentIndexChanged.connect(self.update_actions)
        self.update_actions()
        apply_control_icons(self)

    @staticmethod
    def scroll_panel(panel):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        return scroll

    def extra_panel(self, settings):
        self.horizons = deepcopy(settings.get('horizons', []))
        panel = QWidget()
        layout = QVBoxLayout(panel)
        tools = QHBoxLayout()
        self.horizon_list = QComboBox()
        self.horizon_list.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.horizon_list.setMinimumContentsLength(18)
        tools.addWidget(self.horizon_list, 1)
        self.add_horizon_button = QPushButton(icon('list-plus'), tr('Добавить'))
        self.add_horizon_button.setEnabled(len(self.selectors) >= 2)
        self.remove_horizon_button = QPushButton(icon('trash'), tr('Удалить'))
        tools.addWidget(self.add_horizon_button)
        tools.addWidget(self.remove_horizon_button)
        layout.addLayout(tools)
        hint = QLabel(tr('Для каждого горизонта укажите границу и интервалы минимум в двух стволах. '
                         'Ствол без интервала прерывает линию связи.'))
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.extra_fields = QWidget()
        form = QFormLayout(self.extra_fields)
        self.horizon_name = QLineEdit()
        form.addRow(tr('Название горизонта'), self.horizon_name)
        self.horizon_edge = QComboBox()
        self.horizon_edge.addItem(tr('По кровле'), 'top')
        self.horizon_edge.addItem(tr('По подошве'), 'bottom')
        form.addRow(tr('Граница'), self.horizon_edge)
        self.horizon_selectors = {}
        for well, reference in self.selectors.items():
            combo = QComboBox()
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(20)
            combo.setModel(reference.model())
            self.horizon_selectors[well] = combo
            form.addRow(well, combo)
        layout.addWidget(self.scroll_panel(self.extra_fields), 1)
        self.extra_index = -1
        self.refresh_horizons(0)
        self.horizon_list.currentIndexChanged.connect(self.load_horizon)
        self.add_horizon_button.clicked.connect(self.add_horizon)
        self.remove_horizon_button.clicked.connect(self.remove_horizon)
        self.horizon_name.textChanged.connect(self.store_horizon)
        self.horizon_edge.currentIndexChanged.connect(self.store_horizon)
        for combo in self.horizon_selectors.values():
            combo.currentIndexChanged.connect(self.store_horizon)
        return panel

    def refresh_horizons(self, index):
        with QSignalBlocker(self.horizon_list):
            self.horizon_list.clear()
            for number, horizon in enumerate(self.horizons, 1):
                self.horizon_list.addItem(horizon['name'] or tr('Горизонт {0}', number))
            self.horizon_list.setCurrentIndex(index if self.horizons else -1)
        self.load_horizon(self.horizon_list.currentIndex())

    def load_horizon(self, index):
        self.extra_index = index
        self.extra_fields.setEnabled(index >= 0)
        self.remove_horizon_button.setEnabled(index >= 0)
        horizon = self.horizons[index] if index >= 0 else {'name': '', 'edge': 'top', 'anchors': {}}
        with QSignalBlocker(self.horizon_edge):
            self.horizon_edge.setCurrentIndex(self.horizon_edge.findData(horizon['edge']))
        with QSignalBlocker(self.horizon_name):
            self.horizon_name.setText(horizon['name'])
        for well, combo in self.horizon_selectors.items():
            with QSignalBlocker(combo):
                combo.setCurrentIndex(max(0, combo.findData(horizon['anchors'].get(well))))
        if hasattr(self, 'apply_button'):
            self.update_actions()

    def store_horizon(self):
        if self.extra_index < 0:
            return
        horizon = self.horizons[self.extra_index]
        horizon.update(name=self.horizon_name.text().strip(), edge=self.horizon_edge.currentData())
        for well, combo in self.horizon_selectors.items():
            if combo.currentData():
                horizon['anchors'][well] = combo.currentData()
            else:
                horizon['anchors'].pop(well, None)
        self.horizon_list.setItemText(self.extra_index, horizon['name'] or tr('Горизонт {0}', self.extra_index + 1))
        self.update_actions()

    def add_horizon(self):
        number = len(self.horizons) + 1
        names = {item['name'].casefold() for item in self.horizons}
        while tr('Горизонт {0}', number).casefold() in names:
            number += 1
        self.horizons.append({'name': tr('Горизонт {0}', number), 'edge': 'top', 'anchors': {}})
        self.refresh_horizons(len(self.horizons) - 1)
        self.tabs.setCurrentIndex(1)

    def remove_horizon(self):
        if self.extra_index >= 0:
            del self.horizons[self.extra_index]
            self.refresh_horizons(min(self.extra_index, len(self.horizons) - 1))

    def update_actions(self):
        edge = self.mode.currentData()
        active = edge is not None
        if active:
            self.last_edge = edge
        self.panel.setEnabled(active)
        self.tabs.setTabEnabled(1, active)
        error = self.configuration_error()
        self.apply_button.setEnabled(not active or not error)
        self.validation.setText(error if active else (
            tr('Корреляционные горизонты скрыты. Сохранённые настройки можно включить снова.')
            if self.last_edge else ''))

    def configuration_error(self):
        complete = bool(self.selectors) and all(combo.currentData() for combo in self.selectors.values())
        names = [horizon['name'].casefold() for horizon in self.horizons]
        if not complete:
            return tr('Укажите нулевой горизонт для каждого ствола.')
        if not all(names) or len(names) != len(set(names)):
            return tr('Укажите разные названия дополнительных горизонтов.')
        for horizon in self.horizons:
            valid = sum(self.interval_wells.get(key) == well for well, key in horizon['anchors'].items())
            if valid < 2:
                return tr('Горизонт «{0}»: выберите существующие интервалы хотя бы в двух стволах.', horizon['name'])
        return ''

    def values(self):
        active = self.mode.currentData() is not None
        result = deepcopy(self.saved_settings)
        if self.last_edge:
            candidate = {'edge': self.last_edge, 'anchors': {**result.get('anchors', {}), **{
                well: combo.currentData() for well, combo in self.selectors.items() if combo.currentData()}}}
            if self.horizons:
                candidate['horizons'] = deepcopy(self.horizons)
            try:
                column_style({'correlation': candidate})
            except ValueError:
                pass  # An unfinished new horizon keeps the last saved configuration.
            else:
                result = candidate
        if result and not active:
            result['enabled'] = False
        return result
