from __future__ import annotations

from functools import lru_cache

from matplotlib.colors import to_rgba

from ..i18n import language, tr
from ..theme import icon

from PySide6.QtCore import Signal, QSignalBlocker, Qt, QSize
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QGroupBox,
    QVBoxLayout,
    QWidget,
)

from autopyro.model.core import parse_number
from autopyro.geology.chronostratigraphic_colors import color_catalog
from autopyro.visualization.plot_catalog import colors, colormap_names, markers, marker_value
from .parameters import set_parameter_help
from .documents import wrapped_check
from .fonts import font_families
from .style_labels import color_icon, color_label, colormap_icon, colormap_label, legend_label, marker_icon, marker_label, line_style_icon, line_style_label


def _color_catalog():
    # Registry names alone are insufficient: extensions can redefine a color.
    entries = []
    for name in colors():
        try:
            rgba = to_rgba(name)
        except (TypeError, ValueError):
            rgba = None
        entries.append((name, rgba))
    locale = language()
    return _color_items(locale, tuple(entries)) + _stratigraphic_color_items(locale)


@lru_cache(maxsize=4)
def _color_items(locale, entries):
    """Reuse presentation values; each editor still owns its mutable Qt model."""
    return tuple((name, color_icon(name), color_label(name)) for name, _ in entries)


@lru_cache(maxsize=4)
def _stratigraphic_color_items(locale):
    entries = color_catalog()['colors']
    english = locale == 'en'
    scale = 'Russian scale' if english else 'ОСШ'
    return tuple(
        (color, color_icon(color),
         f"{scale} · {' / '.join(unit[english] for unit in units[-2:])} · {code} · {color}")
        for _, units, code, color in entries
    )


class ColorCombo(QComboBox):
    """Library colors, stratigraphic swatches and arbitrary Matplotlib color text."""
    textEdited = Signal(str)
    editingFinished = Signal()

    def __init__(self, value, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.setIconSize(QSize(24, 14))
        for color, preview, label in _color_catalog():
            self.addItem(preview, label, color)
            self.setItemData(self.count() - 1, color, Qt.ItemDataRole.ToolTipRole)
        self.completer().setFilterMode(Qt.MatchFlag.MatchContains)
        self.setText(value)
        self.currentTextChanged.connect(self.textEdited)
        self.lineEdit().editingFinished.connect(self.editingFinished)
        self.activated.connect(lambda: self.editingFinished.emit())
        self.setMinimumContentsLength(10)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)

    def text(self):
        if self.currentText() == self.itemText(self.currentIndex()):
            return self.currentData()
        index = self.findText(self.currentText())
        return self.itemData(index) if index >= 0 else self.currentText()

    def setText(self, text):
        text = str(text)
        index = self.findData(text)
        if index < 0 and text and color_label(text) != text:
            self.addItem(color_icon(text), color_label(text), text)
            index = self.count() - 1
            self.setItemData(index, text, Qt.ItemDataRole.ToolTipRole)
        if index >= 0:
            self.setCurrentIndex(index)
            self.setEditText(self.itemText(index))
        else:
            self.setEditText(text)


class Form(QWidget):
    edited = Signal()

    def __init__(self, parent=None, *, compact=False):
        super().__init__(parent)
        self.compact = compact
        self.setProperty('parameterForm', True)
        self.sections = QVBoxLayout(self)
        self.form = None
        self.section(tr('Параметры'))
        self.fields = {}
        self.defaults = {}
        self.number_scales = {}

    def section(self, title):
        if self.form is not None and self.form.rowCount() == 0:
            self.form.parentWidget().setTitle(title)
            return
        group = QGroupBox(title, self)
        self.sections.addWidget(group)
        self.form = QFormLayout(group)
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

    def add_field(self, label, widget):
        self.form.addRow(label, widget)
        if self.compact:
            self.form.labelForField(widget).setWordWrap(True)

    def text(self, key, label, value="", color=False, optional=False):
        parent = self.form.parentWidget()
        holder = QWidget(parent) if color else None
        widget = ColorCombo(value, holder) if color else QLineEdit(str(value), parent)
        widget.textEdited.connect(self.edited)
        self.fields[key] = (widget, "optional" if optional else "text")
        self.defaults[key] = value
        set_parameter_help(widget, key, label, value)
        if color:
            layout = QHBoxLayout(holder)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(widget)
            button = QPushButton(icon('sliders-horizontal'), '', holder)
            button.setToolTip(tr('Цвет…'))
            button.setAccessibleName(tr('Цвет…'))
            button.setMaximumWidth(30)

            def choose():
                chosen = QColorDialog.getColor(parent=self)
                if chosen.isValid():
                    widget.setText(chosen.name())
                    self.edited.emit()

            button.clicked.connect(choose)
            layout.addWidget(button)
            self.add_field(label, holder)
        else:
            self.add_field(label, widget)
        return widget

    def number(self, key, label, value, minimum, maximum, decimals=2, integer=False, *, storage_scale=1):
        widget = QSpinBox() if integer else QDoubleSpinBox()
        if not integer:
            widget.setDecimals(decimals)
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.valueChanged.connect(self.edited)
        self.fields[key] = (widget, "number")
        self.number_scales[key] = storage_scale
        self.defaults[key] = value * storage_scale
        set_parameter_help(widget, key, label, value)
        self.add_field(label, widget)
        return widget

    def font_selector(self, key, label, value="DejaVu Sans"):
        widget = self.combo(key, label, font_families())
        self.defaults[key] = value
        self.load({key: value})
        set_parameter_help(widget, key, label, value)
        return widget

    def check(self, key, label, value=False):
        widget = QCheckBox(label)
        widget.setChecked(value)
        widget.toggled.connect(self.edited)
        self.fields[key] = (widget, "check")
        self.defaults[key] = value
        if self.compact:
            holder = QWidget()
            layout = QVBoxLayout(holder)
            layout.setContentsMargins(0, 0, 0, 0)
            wrapped_check(layout, widget, clickable=True)
            self.form.addRow(holder)
        else:
            self.form.addRow(widget)
        set_parameter_help(widget, key, label, value)
        return widget

    def combo(self, key, label, options):
        line_style = key in ('grid_style', 'linestyle', 'line_style') or key.endswith(('_linestyle', '_line_style'))
        if key == "cmap":
            options = [(colormap_label(name), name) for name in colormap_names()]
        elif key == "marker":
            options = [(marker_label(marker), marker) for marker in markers()]
        elif line_style:
            options = [(line_style_label(value[1] if isinstance(value, tuple) else value),
                        value[1] if isinstance(value, tuple) else value) for value in options]
        elif key in ('legend_position', 'legend_loc'):
            if key == 'legend_position':
                from matplotlib.legend import Legend
                options = list(Legend.codes)
            options = [(legend_label(value), value) if not isinstance(value, tuple) else value for value in options]
        if key in ('source_crs', 'target_crs'):
            from .spatial import CRSCombo
            widget = CRSCombo()
        else:
            widget = QComboBox()
        widget.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        widget.setMinimumContentsLength(10)
        if key == 'cmap' or line_style:
            widget.setIconSize(QSize(64, 14))
        elif key == 'marker':
            widget.setIconSize(QSize(32, 20))
        for value in options:
            if isinstance(value, tuple):
                widget.addItem(value[0], value[1])
            else:
                widget.addItem(str(value), value)
            if key == 'cmap':
                widget.setItemIcon(widget.count() - 1, colormap_icon(widget.itemData(widget.count() - 1)))
            elif key == 'marker':
                widget.setItemIcon(widget.count() - 1, marker_icon(widget.itemData(widget.count() - 1)))
            elif line_style:
                widget.setItemIcon(widget.count() - 1, line_style_icon(widget.itemData(widget.count() - 1)))
            if line_style or key in ('cmap', 'marker', 'legend_position', 'legend_loc'):
                widget.setItemData(widget.count() - 1, str(widget.itemData(widget.count() - 1)), Qt.ItemDataRole.ToolTipRole)
        if key == 'marker':
            widget.setEditable(True)
            widget.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            widget.editTextChanged.connect(self.edited)
        widget.currentIndexChanged.connect(self.edited)
        self.fields[key] = (widget, "combo")
        self.defaults[key] = widget.currentData()
        set_parameter_help(widget, key, label, widget.currentText())
        self.add_field(label, widget)
        return widget

    def values(self):
        result = {}
        for key, (widget, kind) in self.fields.items():
            if kind == "check":
                value = widget.isChecked()
            elif kind == "number":
                value = widget.value() * self.number_scales.get(key, 1)
            elif kind == "combo":
                value = widget.currentData()
                if widget.isEditable() and widget.currentText() != widget.itemText(widget.currentIndex()):
                    index = widget.findText(widget.currentText())
                    value = widget.itemData(index) if index >= 0 else widget.currentText()
            elif kind == "optional":
                try:
                    value = parse_number(widget.text())
                except ValueError as exc:
                    error = tr('Введите число или оставьте поле пустым.')
                    widget.setProperty('invalidOptionalNumber', True)
                    widget.setStyleSheet('QLineEdit { border: 1px solid #b74634; }')
                    widget.setToolTip(error)
                    widget.setAccessibleDescription(error)
                    raise ValueError(f'{widget.property("parameterLabel")}: {error}') from exc
                if widget.property('invalidOptionalNumber'):
                    widget.setProperty('invalidOptionalNumber', False)
                    widget.setStyleSheet('')
                    widget.setToolTip(widget.whatsThis())
                    widget.setAccessibleDescription('')
            else:
                value = widget.text()
            result[key] = marker_value(value) if key == 'marker' else value
        return result

    def load(self, values):
        for key, value in values.items():
            if key not in self.fields:
                continue
            widget, kind = self.fields[key]
            blocker = QSignalBlocker(widget)
            if kind == "check":
                widget.setChecked(value)
            elif kind == "number":
                scale = self.number_scales.get(key, 1)
                widget.setValue(value if scale == 1 else value / scale)
            elif kind == "combo":
                index = widget.findData(value)
                if index < 0 and key == 'cmap' and isinstance(value, str):
                    try:
                        preview = colormap_icon(value)
                    except (KeyError, ValueError, TypeError):
                        pass
                    else:
                        widget.addItem(preview, colormap_label(value), value)
                        index = widget.count() - 1
                        widget.setItemData(index, value, Qt.ItemDataRole.ToolTipRole)
                if index < 0:
                    line_style = key in ('grid_style', 'linestyle', 'line_style') or key.endswith(('_linestyle', '_line_style'))
                    text = (marker_label(value) if key == 'marker' else
                            line_style_label(value) if line_style else
                            legend_label(value) if key in ('legend_position', 'legend_loc') else
                            str(value) if widget.isEditable() else tr('Недоступно: {0}', value))
                    widget.addItem(text, value)
                    index = widget.count() - 1
                    if key == 'marker':
                        widget.setItemIcon(index, marker_icon(value))
                    elif line_style:
                        widget.setItemIcon(index, line_style_icon(value))
                widget.setCurrentIndex(index)
            else:
                widget.setText("" if value is None else str(value))
            del blocker
            if key == 'scope' and getattr(widget, '_analysis_scope', None) is not None:
                widget._analysis_scope.set_mode(value)
        if 'log_base' in self.fields:
            self.update_axis_scale()

    def update_axis_scale(self):
        logarithmic = self.fields['scale'][0].currentData() == 'log'
        self.fields['log_base'][0].setEnabled(logarithmic)
        step = self.fields['major_step'][0]
        step.setEnabled(not logarithmic)
        self.fields['number_format'][0].setEnabled(not logarithmic)
        if logarithmic:
            with QSignalBlocker(step):
                step.clear()

    def reset_defaults(self):
        values = dict(self.defaults)
        for key, (widget, kind) in self.fields.items():
            if kind == "combo" and values[key] is None:
                if not widget.count():
                    values.pop(key)
                    continue
                values[key] = widget.itemData(0)
        self.load(values)
        self.edited.emit()

    def reset_parameter(self, key):
        from .parameter_reset import notify_parameter
        widget, kind = self.fields[key]
        value = self.defaults[key]
        if kind == 'combo' and value is None:
            if not widget.count():
                return
            value = widget.itemData(0)
        self.load({key: value})
        notify_parameter(widget)

    def add_reset_button(self):
        from .parameter_reset import install_parameter_resets
        install_parameter_resets(self)
        button = QPushButton(icon('rotate-ccw'), tr("Сбросить параметры") if self.compact else tr("Сбросить параметры вкладки"))
        button.setAutoDefault(False)
        button.setWhatsThis(tr("Возвращает параметры этой вкладки к исходным значениям. Для сброса оформления целиком используйте кнопку внизу окна."))
        button.clicked.connect(self.reset_defaults)
        self.form.addRow(button)
        return button


def axis_form(parent=None):
    form = Form(parent)
    form.section(tr('Диапазон и шкала'))
    form.text("label", tr('Подпись (пусто — из данных)'))
    form.text("minimum", tr('Минимум (пусто — авто)'), optional=True)
    form.text("maximum", tr('Максимум (пусто — авто)'), optional=True)
    form.combo("scale", tr('Шкала'), [(tr('Линейная'), "linear"), (tr('Логарифмическая'), "log")])
    base = form.number("log_base", tr('Основание логарифма'), 10, 0.000001, 1e12, decimals=6)
    base.setEnabled(False)
    form.check("inverted", tr('Инвертировать направление'))
    form.section(tr('Подписи и деления'))
    form.number("label_size", tr('Размер подписи, pt'), 11, 5, 60, integer=True)
    form.number("label_pad", tr('Отступ подписи оси, pt'), 4, 0, 60)
    form.number("tick_size", tr('Размер делений, pt'), 9, 5, 40, integer=True)
    form.text("color", tr('Цвет оси'), "#222222", color=True)
    form.number("rotation", tr('Поворот делений, °'), 0, -180, 180)
    form.text("major_step", tr('Шаг делений (пусто — авто)'), optional=True)
    form.number("minor_divisions", tr('Доп. деления (0 — выкл.)'), 2, 0, 10, integer=True)
    form.combo(
        "number_format",
        tr('Формат чисел'),
        [(tr('Авто'), "auto"), (tr('Обычный'), "plain"), (tr('Научный'), "scientific")],
    )
    form.section(tr('Сетка'))
    form.check("major_grid", tr('Основная сетка'), True)
    form.check("minor_grid", tr('Дополнительная сетка'))
    form.fields['scale'][0].currentIndexChanged.connect(form.update_axis_scale)
    return form
