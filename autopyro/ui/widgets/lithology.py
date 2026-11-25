"""Searchable stratapy catalog with lazy, bounded pattern previews."""

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image
from PySide6.QtCore import QAbstractListModel, QModelIndex, QSize, QSortFilterProxyModel, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QLineEdit, QListView,
    QStyledItemDelegate, QStyleOptionViewItem, QVBoxLayout,
)

from ..i18n import tr
from ..theme.controls import add_search_icon


@dataclass(frozen=True)
class Lithology:
    code: str
    name: str
    style: tuple | None

    @property
    def label(self):
        if not self.code:
            return self.name
        variant = tr(' · цветной') if self.code.endswith('*') else ''
        return f"{self.name}{variant} [{self.code}]"


@lru_cache(maxsize=1)
def standard_lithologies():
    """Read metadata once; importing the catalog never decodes pattern images."""
    from stratapy.setup import formatting

    return tuple(Lithology(code, style[-1], style)
                 for code, style in formatting.lithologies.items())


@lru_cache(maxsize=32)
def pattern_thumbnail(path):
    """Keep only small source tiles, shared by black/white and colored variants."""
    with Image.open(path) as source:
        source.thumbnail((256, 256), Image.Resampling.LANCZOS)
        pixels = np.array(source)
    # Match stratapy's special handling of the bony coal pattern.
    if str(path).endswith('659.png'):
        pixels = np.clip(pixels, 0, .01)
    return pixels


class LithologyCatalog:
    ICON_SIZE = QSize(96, 60)
    CACHE_LIMIT = 128

    def __init__(self, custom=None):
        self.entries = [Lithology('', tr('Без сведений о литологии'), None), *standard_lithologies()]
        if custom:
            self.entries.extend(self._custom_entries(custom))
        self.by_code = {entry.code: entry for entry in self.entries}
        self.by_name = {}
        for entry in self.entries:
            self.by_name.setdefault(entry.name.casefold(), entry)
        self._icons = OrderedDict()

    @staticmethod
    def _custom_entries(custom):
        """Resolve custom styling through stratapy, restoring its global catalog afterwards."""
        import stratapy as sp
        from stratapy.setup import formatting

        original = formatting.lithologies.copy()
        try:
            if set(custom) & formatting.default_lithologies.keys():
                raise ValueError(tr('Для пользовательской литологии укажите новый код'))
            sp.update_lithologies({key: tuple(value) for key, value in custom.items()})
            if set(custom) - formatting.lithologies.keys():
                raise ValueError(tr('Не удалось создать пользовательскую литологию stratapy'))
            return [Lithology(code, formatting.lithologies[code][-1], formatting.lithologies[code])
                    for code in custom]
        finally:
            formatting.lithologies.clear()
            formatting.lithologies.update(original)

    def resolve(self, value):
        value = str(value or '').strip()
        return self.by_code.get(value.lower()) or self.by_name.get(value.casefold())

    def icon(self, entry):
        if entry.code in self._icons:
            self._icons.move_to_end(entry.code)
            return self._icons[entry.code]
        try:
            icon = self._make_icon(entry)
        except (OSError, ValueError, TypeError):
            # An unavailable custom image must not prevent choosing another lithology.
            icon = QIcon()
        self._icons[entry.code] = icon
        if len(self._icons) > self.CACHE_LIMIT:
            self._icons.popitem(last=False)
        return icon

    def _make_icon(self, entry):
        pixmap = QPixmap(self.ICON_SIZE)
        pixmap.fill(Qt.GlobalColor.white)
        if entry.style:
            style = entry.style
            if style[0] == 'fill':
                from matplotlib.colors import to_hex

                pixmap.fill(QColor(to_hex(style[1])))
            else:
                pixels = pattern_thumbnail(style[1])
                if pixels.ndim == 2:
                    from matplotlib import colormaps

                    cmap = colormaps[style[2]] if isinstance(style[2], str) else style[2]
                    low, high = pixels.min(), pixels.max()
                    normalized = (pixels - low) / (high - low) if high > low else np.zeros_like(pixels)
                    pixels = cmap(normalized, bytes=True)
                else:
                    pixels = np.array(Image.fromarray(pixels).convert('RGBA'))
                pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
                image = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0],
                               QImage.Format.Format_RGBA8888).copy()
                painter = QPainter(pixmap)
                painter.drawTiledPixmap(pixmap.rect(), QPixmap.fromImage(image))
                painter.end()
        painter = QPainter(pixmap)
        painter.setPen(Qt.GlobalColor.darkGray)
        painter.drawRect(pixmap.rect().adjusted(0, 0, -1, -1))
        if entry.code == 'no':
            painter.drawLine(0, 0, pixmap.width(), pixmap.height())
            painter.drawLine(0, pixmap.height(), pixmap.width(), 0)
        painter.end()
        return QIcon(pixmap)


class LithologyModel(QAbstractListModel):
    def __init__(self, catalog, parent=None):
        super().__init__(parent)
        self.catalog = catalog

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.catalog.entries)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.catalog.entries):
            return None
        entry = self.catalog.entries[index.row()]
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            return entry.label
        if role == Qt.ItemDataRole.DecorationRole:
            return self.catalog.icon(entry)
        if role == Qt.ItemDataRole.UserRole:
            return entry.code


class LithologyPicker(QDialog):
    def __init__(self, catalog, current='', parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr('Выбор литологии stratapy'))
        self.resize(760, 520)
        self.catalog = catalog
        self.selected_code = current
        self.search = QLineEdit()
        self.search.setPlaceholderText(tr('Поиск по названию породы или коду…'))
        add_search_icon(self.search)
        self.model = LithologyModel(catalog, self)
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.search.textChanged.connect(self._filter_entries)
        self.list = QListView()
        self.list.setModel(self.proxy)
        self.list.setIconSize(catalog.ICON_SIZE)
        self.list.setUniformItemSizes(True)
        self.preview = QLabel()
        self.preview.setFixedSize(240, 150)
        self.description = QLabel()
        self.description.setWordWrap(True)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self.list.selectionModel().currentChanged.connect(self._selection_changed)
        self.list.doubleClicked.connect(lambda index: self.accept())
        layout = QVBoxLayout(self)
        layout.addWidget(self.search)
        content = QHBoxLayout()
        content.addWidget(self.list, 1)
        detail = QVBoxLayout()
        detail.addWidget(self.preview)
        detail.addWidget(self.description)
        detail.addStretch()
        content.addLayout(detail)
        layout.addLayout(content)
        layout.addWidget(self.buttons)
        entry = catalog.resolve(current)
        if entry:
            source = self.model.index(catalog.entries.index(entry), 0)
            self.list.setCurrentIndex(self.proxy.mapFromSource(source))
            self.list.scrollTo(self.list.currentIndex())
        self.search.setFocus()
        from ..theme.controls import apply_control_icons
        apply_control_icons(self)

    def _filter_entries(self, text):
        self.proxy.setFilterFixedString(text)
        if self.list.currentIndex().isValid():
            self.list.scrollTo(self.list.currentIndex())
        else:
            self.list.scrollToTop()

    def _selection_changed(self, current, previous):
        valid = current.isValid()
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(valid)
        self.preview.clear()
        self.description.clear()
        if valid:
            entry = self.catalog.entries[self.proxy.mapToSource(current).row()]
            self.selected_code = entry.code
            icon = self.catalog.icon(entry)
            if icon.isNull():
                self.preview.setText(tr('Предпросмотр недоступен'))
            else:
                self.preview.setPixmap(icon.pixmap(self.catalog.ICON_SIZE).scaled(
                    self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
            self.description.setText(entry.label)

    def accept(self):
        if self.list.currentIndex().isValid():
            super().accept()


class LithologyDelegate(QStyledItemDelegate):
    """Display names and swatches while keeping original codes/names in the table model."""
    def __init__(self, catalog, parent=None):
        super().__init__(parent)
        self.catalog = catalog

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        entry = self.catalog.resolve(index.data())
        if entry:
            option.text = entry.label
            option.icon = self.catalog.icon(entry)
            option.decorationSize = QSize(40, 25)
            option.features |= QStyleOptionViewItem.ViewItemFeature.HasDecoration

    def sizeHint(self, option, index):
        # resizeColumnsToContents must not decode all images in a large interval table.
        entry = self.catalog.resolve(index.data())
        text = entry.label if entry else str(index.data() or '')
        return QSize(option.fontMetrics.horizontalAdvance(text) + 52, 29)

    def createEditor(self, parent, option, index):
        return None
