"""Structured editors for stratapy options; persisted dictionaries stay outside the UI."""

from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget, QScrollArea, QCheckBox, QGroupBox,
)

from autopyro.model.core import parse_number
from autopyro.geology.stratigraphy import COLUMN_STYLE, LEVELS
from ..i18n import tr
from ..theme import icon
from .forms import Form
from .lithology import LithologyCatalog, LithologyPicker
from .stratigraphy_labels import CheckedValuesDialog, geological_label
from .styles import StyleDelegate
from .style_labels import legend_label
from .parameters import set_parameter_help


class RowsEditor(QWidget):
    """Small add/remove table shared by the three structured option editors."""
    edited = Signal()

    def __init__(self, headers, parent=None, readonly_columns=()):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, len(headers))
        self.readonly_columns = readonly_columns
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setMinimumHeight(180)
        layout.addWidget(self.table)
        self.buttons = QHBoxLayout()
        layout.addLayout(self.buttons)
        self.add_button(tr('Добавить'), self.add_row, 'plus')
        self.add_button(tr('Удалить выделенные'), self.remove_rows, 'trash')
        self.table.itemChanged.connect(lambda *_: self.edited.emit())

    def add_button(self, text, callback, icon_name):
        button = QPushButton(icon(icon_name), text)
        button.clicked.connect(callback)
        self.buttons.addWidget(button)
        return button

    def add_row(self, checked=False, values=None):
        row = self.table.rowCount()
        with QSignalBlocker(self.table):
            self.table.insertRow(row)
            for column in range(self.table.columnCount()):
                value = values[column] if values is not None else ''
                cell = QTableWidgetItem(str(value))
                if column in self.readonly_columns:
                    cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, column, cell)
        self.table.setCurrentCell(row, 0)
        self.edited.emit()
        return row

    def remove_rows(self):
        rows = {index.row() for index in self.table.selectedIndexes()}
        with QSignalBlocker(self.table):
            for row in sorted(rows, reverse=True):
                self.table.removeRow(row)
        if rows:
            self.edited.emit()

    def load_rows(self, rows):
        with QSignalBlocker(self):
            self.table.setRowCount(0)
            for values in rows:
                self.add_row(values=values)
        self.table.resizeColumnsToContents()


class GrainScaleEditor(QWidget):
    edited = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.mode = QComboBox()
        self.mode.addItem(tr('Использовать выбранную шкалу пород'), 'preset')
        self.mode.addItem(tr('Задать свои деления'), 'custom')
        layout.addWidget(self.mode)
        self.ticks = RowsEditor([tr('Подпись деления'), tr('Положение на шкале')])
        layout.addWidget(self.ticks)
        self.group_mode = QComboBox()
        for label, code in [(tr('Группы из выбранной шкалы'), 'preset'),
                            (tr('Не показывать группы'), 'none'), (tr('Свои группы делений'), 'custom')]:
            self.group_mode.addItem(label, code)
        layout.addWidget(self.group_mode)
        self.groups = RowsEditor([tr('Название группы'), tr('Деления группы')], readonly_columns=(1,))
        self.groups.add_button(tr('Выбрать деления…'), self.choose_members, 'columns-3')
        self.groups.table.cellDoubleClicked.connect(lambda row, column: self.choose_members() if column == 1 else None)
        layout.addWidget(self.groups)
        self.error = QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        self.preset = 'clastic'
        self.mode.currentIndexChanged.connect(self.update_enabled)
        self.group_mode.currentIndexChanged.connect(self.update_enabled)
        self.ticks.edited.connect(self.edited)
        self.groups.edited.connect(self.edited)
        self.update_enabled()

    def update_enabled(self, *_):
        self.ticks.setEnabled(self.mode.currentData() == 'custom')
        self.groups.setEnabled(self.group_mode.currentData() == 'custom')
        self.edited.emit()

    def tick_values(self):
        if self.mode.currentData() == 'preset':
            return None
        result = {}
        for row in range(self.ticks.table.rowCount()):
            name = self.ticks.table.item(row, 0).text().strip()
            value = parse_number(self.ticks.table.item(row, 1).text())
            if not name or name in result or value is None or value < 0:
                raise ValueError(tr('У каждого деления нужны уникальная подпись и неотрицательное число.'))
            result[name] = value
        if not result:
            raise ValueError(tr('Добавьте деления или выберите готовую шкалу.'))
        return result

    def choose_members(self):
        try:
            self._choose_members()
            self.error.clear()
        except ValueError as exc:
            self.error.setText(str(exc))

    def _choose_members(self):
        row = self.groups.table.currentRow()
        if row < 0:
            return
        from stratapy.utils import parse_params

        ticks = self.tick_values()
        if ticks is None:
            ticks = parse_params({'grain_preset': self.preset})['x_ticks_dict']
        cell = self.groups.table.item(row, 1)
        selected = cell.data(Qt.ItemDataRole.UserRole) or []
        picker = CheckedValuesDialog(tr('Деления группы'),
                                    [(geological_label(code), code) for code in ticks], selected, self)
        if picker.exec() == picker.DialogCode.Accepted:
            cell.setData(Qt.ItemDataRole.UserRole, picker.values())
            cell.setText(', '.join(geological_label(code) for code in picker.values()))

    def load(self, ticks, groups):
        self.mode.setCurrentIndex(self.mode.findData('preset' if ticks is None else 'custom'))
        self.ticks.load_rows((ticks or {}).items())
        mode = 'preset' if groups is None else 'custom' if groups else 'none'
        self.group_mode.setCurrentIndex(self.group_mode.findData(mode))
        self.groups.load_rows([(name, ', '.join(values)) for name, values in (groups or {}).items()])
        for row, values in enumerate((groups or {}).values()):
            cell = self.groups.table.item(row, 1)
            cell.setData(Qt.ItemDataRole.UserRole, list(values))
            cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.update_enabled()

    def values(self):
        ticks = self.tick_values()
        mode = self.group_mode.currentData()
        if mode != 'custom':
            return {'x_ticks_dict': ticks, 'grain_brackets': None if mode == 'preset' else {}}
        groups = {}
        for row in range(self.groups.table.rowCount()):
            name = self.groups.table.item(row, 0).text().strip()
            values = self.groups.table.item(row, 1).data(Qt.ItemDataRole.UserRole) or []
            if not name or name in groups or not values:
                raise ValueError(tr('У каждой группы нужны уникальное название и выбранные деления.'))
            if ticks is not None and set(values) - ticks.keys():
                raise ValueError(tr('Группа содержит удалённые деления. Выберите её состав заново.'))
            groups[name] = values
        return {'x_ticks_dict': ticks, 'grain_brackets': groups}


class CustomRocksEditor(RowsEditor):
    def __init__(self, parent=None):
        super().__init__([tr('Идентификатор'), tr('Название породы'), tr('Узор'), tr('Цвет')], parent)
        self.table.setColumnHidden(0, True)
        self.table.setItemDelegateForColumn(3, StyleDelegate(self.table))
        self.catalog = LithologyCatalog()
        self.add_button(tr('Выбрать узор…'), self.choose_pattern, 'layers')
        self.add_button(tr('Изображение…'), self.choose_image, 'folder-open')
        self.table.cellDoubleClicked.connect(lambda row, column: self.choose_pattern() if column == 2 else None)

    def add_row(self, checked=False, values=None):
        if values is None:
            existing = {self.table.item(row, 0).text() for row in range(self.table.rowCount())}
            number = 1
            while f'custom_rock_{number}' in existing:
                number += 1
            values = [f'custom_rock_{number}', tr('Новая порода'), '', '#e0b761']
        row = super().add_row(values=values)
        self.table.item(row, 2).setFlags(self.table.item(row, 2).flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.item(row, 2).setData(Qt.ItemDataRole.UserRole, '')
        return row

    def set_pattern(self, row, value):
        cell = self.table.item(row, 2)
        entry = self.catalog.resolve(value) if value else None
        cell.setData(Qt.ItemDataRole.UserRole, value)
        cell.setText(entry.label if entry else Path(value).name if value else tr('Сплошная заливка'))
        if entry:
            cell.setIcon(self.catalog.icon(entry))

    def choose_pattern(self):
        row = self.table.currentRow()
        if row < 0:
            return
        picker = LithologyPicker(self.catalog, self.table.item(row, 2).data(Qt.ItemDataRole.UserRole) or '', self)
        if picker.exec() == picker.DialogCode.Accepted:
            self.set_pattern(row, picker.selected_code)

    def choose_image(self):
        row = self.table.currentRow()
        if row < 0:
            return
        path, _ = QFileDialog.getOpenFileName(self, tr('Узор породы'), '', tr('Изображения (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)'))
        if path:
            self.set_pattern(row, path)

    def load(self, rocks):
        self.load_rows([(code, value[2], '', str(value[1])) for code, value in rocks.items()])
        for row, value in enumerate(rocks.values()):
            self.set_pattern(row, value[0])
            self.table.item(row, 3).setData(Qt.ItemDataRole.UserRole, (str(value[1]), deepcopy(value[1])))

    def values(self):
        result = {}
        for row in range(self.table.rowCount()):
            code, name, pattern, color = [self.table.item(row, column) for column in range(4)]
            if not name.text().strip():
                raise ValueError(tr('Укажите название пользовательской породы.'))
            original = color.data(Qt.ItemDataRole.UserRole)
            color_value = original[1] if original and color.text() == original[0] else color.text()
            result[code.text()] = [pattern.data(Qt.ItemDataRole.UserRole), color_value, name.text()]
        return result


class ColumnStyleForm(Form):
    """Regular fields plus structured editors, exposing the existing native style dictionary."""
    def check(self, key, label, value=False):
        # Long option captions wrap within the parameter panel instead of widening it.
        widget = QCheckBox()
        widget.setChecked(value)
        widget.setAccessibleName(label)
        widget.toggled.connect(self.edited)
        self.fields[key] = (widget, 'check')
        self.defaults[key] = value
        set_parameter_help(widget, key, label, value)
        caption = QLabel(label)
        caption.setWordWrap(True)
        caption.setBuddy(widget)
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(widget)
        row.addWidget(caption, 1)
        self.form.addRow(holder)
        return widget

    def __init__(self, parent=None):
        super().__init__(parent)
        self._default_depth_label = False
        self.comparison_wells = []
        self.correlation = {}
        self.available_wells = []
        self.comparison_group = QGroupBox(tr('Сравнение скважин и стволов'), self)
        comparison_layout = QVBoxLayout(self.comparison_group)
        self.comparison_button = QPushButton(icon('columns-3'), tr('Выбрать стволы для сравнения…'))
        self.comparison_button.clicked.connect(self.choose_comparison_wells)
        comparison_layout.addWidget(self.comparison_button)
        self.comparison_summary = QLabel()
        self.comparison_summary.setWordWrap(True)
        comparison_layout.addWidget(self.comparison_summary)
        self.correlation_button = QPushButton(icon('layers'), tr('Корреляционные горизонты…'))
        comparison_layout.addWidget(self.correlation_button)
        self.correlation_summary = QLabel()
        comparison_layout.addWidget(self.correlation_summary)
        hint = QLabel(tr('Выбранный ствол колонки показывается всегда. Добавьте другие стволы из этой схемы: глубина и масштаб будут общими. Для возврата к одной колонке снимите дополнительные флажки.'))
        hint.setWordWrap(True)
        comparison_layout.addWidget(hint)
        size_hint = QLabel(tr('Для узких дорожек увеличьте ширину рисунка и выберите «Реальный размер» на панели просмотра.'))
        size_hint.setWordWrap(True)
        comparison_layout.addWidget(size_hint)
        self.hierarchy_group = QGroupBox(tr('Дополнительные уровни рядом с колонкой'), self)
        hierarchy_layout = QVBoxLayout(self.hierarchy_group)
        self.hierarchy_checks = {}
        for level in LEVELS:
            check = QCheckBox(geological_label(level))
            self.hierarchy_checks[level] = check
            hierarchy_layout.addWidget(check)
            check.toggled.connect(self.edited)
        hint = QLabel(tr('Основной уровень задаёт литологию и результаты привязки. Отметьте другие уровни, чтобы показать их на общей шкале глубины.'))
        hint.setWordWrap(True)
        hierarchy_layout.addWidget(hint)
        self.details = QTabWidget()
        self.legend_form = Form()
        self.legend_form.section(tr('Оформление легенды'))
        for key, label, value in [('facecolor', tr('Фон легенды'), 'none'),
                                   ('edgecolor', tr('Цвет рамки'), 'black')]:
            self.legend_form.text(key, label, value, color=True)
        for key, label, value, low, high in [
            ('framealpha', tr('Непрозрачность фона'), 1, 0, 1),
            ('labelspacing', tr('Расстояние между строками'), 2.2, 0, 20),
            ('handletextpad', tr('Расстояние от образца до подписи'), 1, 0, 20),
            ('handlelength', tr('Ширина образца'), 2, 0, 20),
            ('handleheight', tr('Высота образца'), .7, 0, 20),
            ('borderpad', tr('Внутренний отступ'), 1.2, 0, 20),
            ('borderaxespad', tr('Отступ от рисунка'), 0, 0, 20),
        ]:
            self.legend_form.number(key, label, value, low, high)
        self.legend_form.check('fancybox', tr('Скруглённая рамка'), False)
        self.legend_form.combo('location', tr('Точное положение легенды'),
                               [(tr('Автоматически'), 'auto'),
                                *((legend_label(code), code) for code in (
                                    'upper left', 'upper right', 'lower left', 'lower right', 'center',
                                    'upper center', 'lower center', 'center left', 'center right'))])
        self.legend_form.text('anchor_x', tr('Горизонтальный якорь (пусто — автоматически)'), optional=True)
        self.legend_form.text('anchor_y', tr('Вертикальный якорь (пусто — автоматически)'), optional=True)
        self.legend_form.section(tr('Заголовки разделов легенды'))
        self.title_labels = [tr('Литологии'), tr('Минералы'), tr('Осадочные структуры'),
                             tr('Палеонтологические признаки'), tr('Тектонические структуры'),
                             tr('Контакты слоёв'), tr('Места отбора проб')]
        for index, label in enumerate(self.title_labels):
            self.legend_form.text(f'title_{index}', label, label)
        legend_scroll = QScrollArea()
        legend_scroll.setWidgetResizable(True)
        legend_scroll.setWidget(self.legend_form)
        self.details.addTab(legend_scroll, icon('tags'), tr('Легенда'))
        self.scale_editor = GrainScaleEditor()
        self.details.addTab(self.scale_editor, icon('columns-3'), tr('Деления и группы'))
        self.rocks_editor = CustomRocksEditor()
        self.details.addTab(self.rocks_editor, icon('layers'), tr('Свои породы'))
        self.legend_form.edited.connect(self.edited)
        self.scale_editor.edited.connect(self.edited)
        self.rocks_editor.edited.connect(self.edited)
        self._legend_options = {}
        self._legend_baseline = self.legend_form.values()
        self._titles = deepcopy(COLUMN_STYLE['legend_titles'])

    def choose_comparison_wells(self):
        picker = CheckedValuesDialog(tr('Стволы для сравнения'),
                                     [(well, well) for well in self.available_wells],
                                     self.comparison_wells, self)
        if picker.exec() == picker.DialogCode.Accepted:
            wells = picker.values()
            if wells != self.comparison_wells:
                self.comparison_wells = wells
                self.update_comparison_summary()
                self.edited.emit()

    def update_comparison_summary(self):
        self.comparison_summary.setText(
            tr('Дополнительные стволы: {0}', ', '.join(self.comparison_wells))
            if self.comparison_wells else tr('Показан только выбранный ствол колонки.'))

    def set_correlation(self, settings):
        self.correlation = deepcopy(settings)
        edge = self.correlation.get('edge') if self.correlation.get('enabled', True) else None
        summary = (
            tr('Выравнивание по кровле') if edge == 'top' else
            tr('Выравнивание по подошве') if edge == 'bottom' else tr('Исходная глубина'))
        if self.correlation.get('horizons'):
            summary += ' · ' + tr('Дополнительных горизонтов: {0}', len(self.correlation['horizons']))
        self.correlation_summary.setText(summary)

    def load(self, values):
        with QSignalBlocker(self):
            display_values = dict(values)
            if 'y_label' in values:
                self._default_depth_label = values['y_label'] == COLUMN_STYLE['y_label']
                if self._default_depth_label:
                    display_values['y_label'] = tr('Глубина')
            super().load(display_values)
            if 'comparison_wells' in values:
                self.comparison_wells = list(values['comparison_wells'])
                self.update_comparison_summary()
            if 'correlation' in values:
                self.set_correlation(values['correlation'])
            if 'hierarchy_levels' in values:
                for level, check in self.hierarchy_checks.items():
                    check.setChecked(level in values['hierarchy_levels'])
            if 'grain_preset' in values:
                self.scale_editor.preset = values['grain_preset']
            if 'legend_kwargs' in values:
                self._legend_options = deepcopy(values['legend_kwargs'])
                controls = self.legend_form.defaults.copy()
                controls.update({key: value for key, value in self._legend_options.items()
                                 if key in controls})
                controls['location'] = self._legend_options.get('loc', 'auto')
                anchor = self._legend_options.get('bbox_to_anchor')
                if isinstance(anchor, (list, tuple)) and len(anchor) == 2:
                    controls['anchor_x'], controls['anchor_y'] = anchor
                self.legend_form.load(controls)
            if 'legend_titles' in values:
                self._titles = deepcopy(values['legend_titles'])
                self.legend_form.load({f'title_{i}': self.title_labels[i] if title == COLUMN_STYLE['legend_titles'][i] else title
                                       for i, title in enumerate(self._titles)})
            self._legend_baseline = self.legend_form.values()
            if 'x_ticks_dict' in values or 'grain_brackets' in values:
                old = ({} if 'x_ticks_dict' in values and 'grain_brackets' in values
                       else self.scale_editor.values())
                self.scale_editor.load(values.get('x_ticks_dict', old.get('x_ticks_dict')),
                                       values.get('grain_brackets', old.get('grain_brackets')))
            if 'lithologies' in values:
                self.rocks_editor.load(values['lithologies'])

    def values(self):
        values = super().values()
        values['comparison_wells'] = list(self.comparison_wells)
        values['correlation'] = deepcopy(self.correlation)
        values['hierarchy_levels'] = [level for level, check in self.hierarchy_checks.items() if check.isChecked()]
        if self._default_depth_label and values.get('y_label') == tr('Глубина'):
            values['y_label'] = COLUMN_STYLE['y_label']
        controls = self.legend_form.values()
        legend = deepcopy(self._legend_options)
        for key, value in controls.items():
            if (not key.startswith('title_') and key not in ('anchor_x', 'anchor_y', 'location')
                    and value != self._legend_baseline[key]):
                legend[key] = value
        if controls['location'] != self._legend_baseline['location']:
            legend.pop('loc', None)
            if controls['location'] != 'auto':
                legend['loc'] = controls['location']
        if any(controls[key] != self._legend_baseline[key] for key in ('anchor_x', 'anchor_y')):
            if (controls['anchor_x'] is None) != (controls['anchor_y'] is None):
                raise ValueError(tr('Для положения легенды задайте обе координаты якоря.'))
            legend.pop('bbox_to_anchor', None)
            if controls['anchor_x'] is not None:
                legend['bbox_to_anchor'] = [controls['anchor_x'], controls['anchor_y']]
        values['legend_kwargs'] = legend
        values['legend_titles'] = [controls[f'title_{i}'] if controls[f'title_{i}'] != self._legend_baseline[f'title_{i}']
                                  else title for i, title in enumerate(self._titles)]
        values.update(self.scale_editor.values())
        values['lithologies'] = self.rocks_editor.values()
        return values

    def reset_defaults(self):
        self.load(deepcopy(COLUMN_STYLE))
        self.edited.emit()
