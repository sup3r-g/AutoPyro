"""Apply and restore the project table's filters and sort order."""

from PySide6.QtCore import Qt

from .filter_rules import leaf_rules, saved_sort_column


class TableFilterSort:
    def __init__(self, window):
        self.window = window
        self.missing_filter_column = False

    def apply_filter(self, rules, combine):
        window = self.window
        if self.missing_filter_column:
            self.missing_filter_column = False
            window.proxy.set_rules(rules, combine, row_ids=window.sample_sets.row_ids())
        else:
            window.proxy.set_rules(rules, combine)
        window.model_changed()

    def restore_filter(self, rules, combine, *, row_ids):
        window = self.window
        self.missing_filter_column = False
        if rules:
            columns = {column.id for column in window.model.project.columns}
            self.missing_filter_column = any(rule["column"] not in columns for rule in leaf_rules(rules))
        window.proxy.set_rules(rules, combine, row_ids=() if self.missing_filter_column else row_ids)

    def condition_count(self):
        return sum(1 for _ in leaf_rules(self.window.proxy.rules))

    def show_filter(self):
        self.window.desktop.show_table_properties("filter")

    def show_sort(self):
        self.window.desktop.show_table_properties("sort")

    def single_sort(self, section):
        window = self.window
        proxy = window.proxy
        order = Qt.SortOrder.AscendingOrder
        if not proxy.sort_rules and proxy.sortColumn() == section and proxy.sortOrder() == order:
            order = Qt.SortOrder.DescendingOrder
        window.table.sortByColumn(section, order)
        window.view_changed()

    def apply_sort(self, rules):
        window = self.window
        primary = next((i for i, column in enumerate(window.model.project.columns)
                        if rules and column.id == rules[0][0]), -1)
        window.proxy.set_sort_rules(rules, primary)
        window.view_changed()

    def capture(self):
        window = self.window
        proxy = window.proxy
        column = proxy.sortColumn()
        columns = window.model.project.columns
        return {
            "rules": proxy.rules,
            "combine": proxy.combine,
            "sort": getattr(proxy, "sort_rules", []),
            "sort_column": column,
            "sort_column_id": columns[column].id if 0 <= column < len(columns) else None,
            "descending": proxy.sortOrder() == Qt.SortOrder.DescendingOrder,
        }

    def restore_sort(self, view, positions):
        window = self.window
        rules = [rule for rule in view.get("sort", []) if rule[0] in positions]
        primary = saved_sort_column(view, window.model.project.columns)
        if primary < 0 and rules:
            primary = positions[rules[0][0]]
        order = Qt.SortOrder.DescendingOrder if view.get("descending") else Qt.SortOrder.AscendingOrder
        window.proxy.set_sort_rules(rules, primary, order)
