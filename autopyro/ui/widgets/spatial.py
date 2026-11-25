"""Named CRS selectors and a lazy, read-only spatial attribute table."""

import pandas as pd
import re
from functools import lru_cache
from pyproj import database, enums
from pyproj.exceptions import CRSError
from PySide6.QtCore import QAbstractTableModel, Qt, QTimer
from PySide6.QtWidgets import (QComboBox, QCompleter, QDialog, QDialogButtonBox,
                              QLineEdit, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QLabel)

from autopyro.spatial.mapping import available_crs, crs_label, layer_frame
from ..i18n import tr, settings
from ..theme.controls import add_command_button, add_search_icon


FAVORITE_CRS_KEY = 'mapping/crs/favorites'
RECENT_CRS_KEY = 'mapping/crs/recent'


def saved_crs(store, key):
    values = store.value(key, [])
    values = [values] if isinstance(values, str) else values
    result = []
    for value in values if isinstance(values, (list, tuple)) else []:
        if not isinstance(value, str) or value == 'auto' or value in result:
            continue
        try:
            crs_label(value)
        except (ValueError, TypeError, CRSError):
            continue
        result.append(value)
    return result


@lru_cache(maxsize=1)
def crs_groups():
    return {f'{info.auth_name}:{info.code}': info for info in database.query_crs_info(allow_deprecated=False)}


class CRSDialog(QDialog):
    def __init__(self, combo):
        super().__init__(combo)
        self.setWindowTitle(tr('Выбор системы координат'))
        self.resize(800, 620)
        self.store = settings()
        self.favorites = saved_crs(self.store, FAVORITE_CRS_KEY)
        self.recents = saved_crs(self.store, RECENT_CRS_KEY)[:10]
        for value in dict.fromkeys(self.favorites + self.recents):
            if combo.findData(value) < 0:
                combo.addItem(crs_label(value), value)
        layout = QVBoxLayout(self)
        self.search = QLineEdit()
        self.search.setPlaceholderText(tr('Поиск по названию или коду CRS…'))
        add_search_icon(self.search)
        self.search.setClearButtonEnabled(True)
        layout.addWidget(self.search)
        self.matches = QLabel()
        layout.addWidget(self.matches)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels([tr('Название'), tr('Код')])
        self.tree.setColumnWidth(0, 590)
        layout.addWidget(self.tree)
        self.details = QLabel()
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.favorite_button = add_command_button(self.buttons, 'plus', tr('В избранное'), QDialogButtonBox.ButtonRole.ActionRole)
        self.favorite_button.setCheckable(True)
        self.favorite_button.clicked.connect(self.toggle_favorite)
        layout.addWidget(self.buttons)
        self.leaves, groups = [], {}
        selected = None
        metadata = crs_groups()
        for index in range(combo.count()):
            value = combo.itemData(index)
            label = combo.itemText(index)
            info = metadata.get(value)
            if info and info.type == enums.PJType.PROJECTED_CRS:
                family = 'UTM' if 'UTM zone' in info.name else info.projection_method_name or info.auth_name
                path = (tr('Проецированные системы'), family, info.name.split(' / ')[0])
            elif info:
                path = (tr('Географические системы'), info.auth_name)
            else:
                path = (tr('Автоматические и пользовательские CRS'),)
            parent = self.tree.invisibleRootItem()
            for depth in range(len(path)):
                key = path[:depth + 1]
                if key not in groups:
                    groups[key] = QTreeWidgetItem(parent, [path[depth]])
                parent = groups[key]
            leaf = QTreeWidgetItem(parent, [info.name if info else label, value])
            leaf.setData(0, Qt.ItemDataRole.UserRole, value)
            area = info.area_of_use.name if info and info.area_of_use else ''
            leaf.setToolTip(0, area)
            searchable = ' '.join((*path, label, str(value), area))
            self.leaves.append((leaf, self.normalize(searchable)))
            if value == combo.currentData():
                selected = leaf
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(120)
        self.timer.timeout.connect(self.filter_tree)
        self.search.textChanged.connect(lambda: self.timer.start())
        self.tree.currentItemChanged.connect(self.selection_changed)
        self.tree.itemDoubleClicked.connect(lambda item, _: self.accept() if item.data(0, Qt.ItemDataRole.UserRole) else None)
        self.catalog_leaves = list(self.leaves)
        self.shortcut_groups = []
        self.refresh_shortcuts()
        selected = next((leaf for leaf, _ in self.leaves if leaf.data(0, Qt.ItemDataRole.UserRole) == combo.currentData()), selected)
        self.tree.expandToDepth(0)
        self.matches.setText(tr('Найдено CRS: {0}', len(self.catalog_leaves)))
        if selected:
            self.tree.setCurrentItem(selected)
            parent = selected.parent()
            while parent:
                parent.setExpanded(True)
                parent = parent.parent()
            self.tree.scrollToItem(selected)
        if self.shortcut_groups:
            self.tree.scrollToTop()
        self.selection_changed()
        if combo.isEditable() and combo.currentText() != combo.itemText(combo.currentIndex()):
            self.search.setText(combo.currentText())
        self.search.setFocus()
        from ..theme.controls import apply_control_icons
        apply_control_icons(self)

    @staticmethod
    def normalize(text):
        return re.sub(r'[:/—_\-\s]+', '', text.casefold())

    def refresh_shortcuts(self):
        current = self.tree.currentItem()
        value = current.data(0, Qt.ItemDataRole.UserRole) if current else None
        self.tree.blockSignals(True)
        for group in self.shortcut_groups:
            self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(group))
        self.shortcut_groups = []
        leaves = []
        originals = {leaf.data(0, Qt.ItemDataRole.UserRole): (leaf, text) for leaf, text in self.catalog_leaves}
        for title, values in [(tr('Избранные CRS'), self.favorites), (tr('Недавно использованные CRS'), self.recents)]:
            if not values:
                continue
            group = QTreeWidgetItem([title])
            self.tree.insertTopLevelItem(len(self.shortcut_groups), group)
            self.shortcut_groups.append(group)
            for code in values:
                original, text = originals[code]
                leaf = QTreeWidgetItem(group, [original.text(0), code])
                leaf.setData(0, Qt.ItemDataRole.UserRole, code)
                leaf.setToolTip(0, original.toolTip(0))
                leaves.append((leaf, text))
            group.setExpanded(True)
        self.leaves = leaves + self.catalog_leaves
        if value:
            selected = next((leaf for leaf, _ in self.leaves if leaf.data(0, Qt.ItemDataRole.UserRole) == value), None)
            if selected:
                self.tree.setCurrentItem(selected)
        self.tree.blockSignals(False)

    def toggle_favorite(self):
        item = self.tree.currentItem()
        value = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        if not value or value == 'auto':
            return
        if value in self.favorites:
            self.favorites.remove(value)
        else:
            self.favorites.append(value)
        self.store.setValue(FAVORITE_CRS_KEY, self.favorites)
        self.store.sync()
        self.refresh_shortcuts()
        self.filter_tree()

    def filter_tree(self):
        terms = [self.normalize(term) for term in self.search.text().split()]
        for leaf, text in self.leaves:
            leaf.setHidden(not all(term in text for term in terms))
        def visible(item):
            if not item.childCount():
                return not item.isHidden()
            matches = [visible(item.child(i)) for i in range(item.childCount())]
            item.setHidden(not any(matches))
            item.setExpanded(bool(terms) and any(matches))
            return any(matches)
        for index in range(self.tree.topLevelItemCount()):
            visible(self.tree.topLevelItem(index))
        matches = [leaf for leaf, _ in self.leaves if not leaf.isHidden()]
        self.matches.setText(tr('Найдено CRS: {0}', len({leaf.data(0, Qt.ItemDataRole.UserRole) for leaf in matches})))
        current = self.tree.currentItem()
        if matches and (current is None or current.isHidden()):
            self.tree.setCurrentItem(matches[0])
        if not terms:
            self.tree.expandToDepth(0)
        self.selection_changed()

    def selection_changed(self, *_):
        item = self.tree.currentItem()
        valid = item is not None and not item.isHidden() and bool(item.data(0, Qt.ItemDataRole.UserRole))
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(valid)
        self.details.setText(item.toolTip(0) if valid else '')
        value = item.data(0, Qt.ItemDataRole.UserRole) if valid else None
        self.favorite_button.setEnabled(bool(value) and value != 'auto')
        self.favorite_button.setChecked(value in self.favorites)
        self.favorite_button.setText(tr('Убрать из избранного') if value in self.favorites else tr('В избранное'))

    def accept(self):
        if self.timer.isActive():
            self.timer.stop()
            self.filter_tree()
        item = self.tree.currentItem()
        if item is not None and not item.isHidden() and item.data(0, Qt.ItemDataRole.UserRole):
            self.selected_crs = item.data(0, Qt.ItemDataRole.UserRole)
            if self.selected_crs != 'auto':
                recents = saved_crs(self.store, RECENT_CRS_KEY)
                self.store.setValue(RECENT_CRS_KEY, [self.selected_crs] + [value for value in recents if value != self.selected_crs][:9])
                self.store.sync()
            super().accept()


class CRSCombo(QComboBox):
    def showPopup(self):
        if self.isEditable():
            self.completer().popup().hide()
        dialog = CRSDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.setCurrentIndex(self.findData(dialog.selected_crs))
        self.hidePopup()
        dialog.deleteLater()


def configure_crs_combo(combo, current='auto'):
    combo.addItem(tr('Автоматически из данных'), 'auto')
    for label, value in available_crs():
        combo.addItem(label, value)
    if current != 'auto' and combo.findData(current) < 0:
        combo.addItem(crs_label(current), current)
    combo.setEditable(True)
    combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    completer = QCompleter(combo.model(), combo)
    completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
    completer.setFilterMode(Qt.MatchFlag.MatchContains)
    completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
    combo.setCompleter(completer)
    combo.setCurrentIndex(max(0, combo.findData(current)))


class AttributeModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame = pd.DataFrame()
        self.feature_rows = {}

    def set_layer(self, layer):
        self.beginResetModel()
        if layer is None:
            self.frame = pd.DataFrame()
        else:
            records = layer.get('attributes')
            if records is None:
                spatial = layer_frame(layer)
                self.frame = pd.DataFrame(spatial.drop(columns=spatial.geometry.name))
                self.frame.insert(0, tr('Тип геометрии'), spatial.geom_type.to_numpy(), allow_duplicates=True)
                self.frame.insert(0, 'FID', spatial.index.to_numpy(), allow_duplicates=True)
            else:
                self.frame = pd.DataFrame([r['properties'] for r in records], columns=layer.get('fields'))
                self.frame.insert(0, tr('Тип геометрии'), [r['geometry_type'] for r in records], allow_duplicates=True)
                self.frame.insert(0, 'FID', [r['id'] for r in records], allow_duplicates=True)
        self.feature_rows = {}
        if len(self.frame.columns):
            for row, fid in enumerate(self.frame.iloc[:, 0]):
                self.feature_rows.setdefault(str(fid), row)
        self.endResetModel()

    def rowCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else len(self.frame)

    def columnCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else len(self.frame.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            return None
        value = self.frame.iat[index.row(), index.column()]
        return '—' if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)) else str(value)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            return str(self.frame.columns[section]) if orientation == Qt.Orientation.Horizontal else str(section + 1)
