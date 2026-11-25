"""Inline filter and sorting editor for a saved analysis view."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QPushButton

from .properties import TableProperties
from .filter_rules import saved_sort_column
from .filter_rules import validated_rules
from ..i18n import message, tr
from ..theme import icon


class SavedViewProperties(TableProperties):
    def __init__(self, window):
        self.name = None
        super().__init__(window, QLabel())
        self.headings = []
        for key, page in self.pages.items():
            self.setTabVisible(self.indexOf(page), key in ('filter', 'sort'))
            if key in ('filter', 'sort'):
                heading = QLabel()
                heading.setWordWrap(True)
                page.widget().layout().insertWidget(0, heading)
                self.headings.append(heading)
        self.apply_filter_button.setText(tr('Сохранить фильтр представления'))
        self.apply_sort_button.setText(tr('Сохранить сортировку представления'))
        self.apply_filter_button.setIcon(icon('save'))
        self.apply_sort_button.setIcon(icon('save'))
        self.done_button = QPushButton(icon('check'), tr('Готово'))
        self.done_button.clicked.connect(self.finish)
        self.setCornerWidget(self.done_button, Qt.Corner.TopRightCorner)
        window.analysis_scope.changed.connect(self.scope_changed)

    def finish(self):
        self.window.desktop.finish_view_properties()

    def record(self):
        return self.window.model.project.view.get('named', {}).get(self.name)

    def edit(self, name):
        if name not in self.window.model.project.view.get('named', {}):
            raise ValueError(tr('Сохранённое представление отсутствует'))
        self.name = name
        self._filter_state = self._sort_state = None
        self.refresh()
        self.select_page('filter')

    def rename_target(self, old_name, name):
        if self.name == old_name:
            self.name = name

    def clear_target(self):
        self.name = None
        self._schema = self._filter_state = self._sort_state = None

    def refresh(self):
        if not hasattr(self, 'headings'):
            return
        record = self.record()
        exists = record is not None
        title = tr('Представление: {0}', self.name) if exists else tr('Сохранённое представление отсутствует')
        if exists and (sample_set := record.get('sample_set')):
            title += '\n' + tr('Набор проб: {0}', sample_set)
        for heading in self.headings:
            heading.setText(title)
        for key in ('filter', 'sort'):
            self.pages[key].widget().setEnabled(exists)
        if exists:
            super().refresh()

    def filter_source(self):
        record = self.record() or {}
        return record.get('rules', []), record.get('combine', 'AND')

    def sort_source(self):
        record = self.record() or {}
        return (record.get('sort', []), saved_sort_column(record, self.window.model.project.columns),
                Qt.SortOrder.DescendingOrder if record.get('descending') else Qt.SortOrder.AscendingOrder)

    def scope_changed(self):
        desktop = self.window.desktop
        if desktop.properties_stack.currentWidget() is not self:
            return
        mode = self.window.analysis_scope.mode
        if not mode.startswith('view:'):
            self.finish()
        elif mode[5:] != self.name:
            self.edit(mode[5:])
        else:
            self.refresh()

    def save_changes(self, changes):
        try:
            self.window.analysis_scope.update_view(self.name, changes)
            self.filter_error.clear()
        except ValueError as exc:
            self.filter_error.setText(message(exc))
        self.refresh()

    def apply_filter(self):
        self.refresh()
        try:
            rules = validated_rules(self.window.model.project, [rule.draft() for rule in self.rules])
        except ValueError as exc:
            self.filter_error.setText(message(exc))
            return
        self.save_changes({'rules': rules, 'combine': self.combine.currentText()})

    def reset_filter(self):
        self.save_changes({'rules': [], 'combine': 'AND'})

    def apply_sort(self):
        self.refresh()
        rules = [(item.data(Qt.ItemDataRole.UserRole), item.data(Qt.ItemDataRole.UserRole + 1))
                 for index in range(self.sort_columns.count())
                 if (item := self.sort_columns.item(index)).checkState() == Qt.CheckState.Checked]
        self.save_changes({'sort': rules, 'sort_column': 0 if rules else -1,
                           'sort_column_id': rules[0][0] if rules else None, 'descending': False})

    def reset_sort(self):
        self.save_changes({'sort': [], 'sort_column': -1, 'sort_column_id': None, 'descending': False})
