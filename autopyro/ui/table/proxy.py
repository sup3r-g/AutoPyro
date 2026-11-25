from __future__ import annotations

import operator as comparison
import math
from contextlib import contextmanager, nullcontext

import numpy as np
import pandas as pd
from PySide6.QtCore import QSortFilterProxyModel, Qt, Signal

from autopyro.analysis.qc import is_excluded
from .filter_rules import MEASUREMENT_STATUS_OPERATORS, leaf_rules

_KEEP_ROW_IDS = object()
COLUMN_CONTEXT_OPERATORS = {"unit_eq", "unit_missing", "stale", "up_to_date"}
COMPLEX_FILTER_OPERATORS = MEASUREMENT_STATUS_OPERATORS | COLUMN_CONTEXT_OPERATORS


def _matches_missing(rule):
    """Only 'empty' accepts None before per-rule negation."""
    result = rule["operator"] == "empty"
    return not result if rule.get("negate") else result


def _matches_measurement_status(project, row, column, operator):
    if operator == "qc_excluded":
        return any(is_excluded(project, row.id, column, scope)
                   for scope in project.view.get("qc_exclusions", {}))
    measurement = project.science.current(row.id, column)
    if measurement is None:
        return False
    if measurement.qualifier == operator:
        return True
    raw = measurement.raw_value
    return operator == "below_limit" and isinstance(raw, str) and raw.strip().casefold() == "<lod"


class FilterProxy(QSortFilterProxyModel):
    filtersChanged = Signal()
    filteringChanged = Signal(bool)
    sortingChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rules = []
        self.combine = "AND"
        self.row_id_filter = None
        self._table_model = None
        self._context_columns = set()
        self._column_context = {}
        self.sort_rules = []
        self._sort_ranks = None
        self._filter_mask = None
        self._row_ids_revision = 0
        self._row_ids_cache = None
        for signal in (self.layoutChanged, self.modelReset, self.rowsInserted, self.rowsRemoved,
                       self.rowsMoved, self.columnsInserted, self.columnsRemoved, self.columnsMoved,
                       self.dataChanged):
            signal.connect(self._invalidate_row_ids)
        self.setSortRole(Qt.ItemDataRole.UserRole)

    def _invalidate_row_ids(self, *args):
        self._row_ids_revision += 1
        self._row_ids_cache = None

    @property
    def row_order_revision(self):
        return self._row_ids_revision

    def visible_row_snapshot(self):
        count = self.rowCount()
        rows = self.sourceModel().project.rows
        if self.row_id_filter is None and not self.rules and self.sortColumn() < 0 and self.columnCount():
            return count, iter(rows)
        return count, (rows[self.mapToSource(self.index(row, 0)).row()] for row in range(count))

    def ordered_row_ids(self):
        """Return an immutable snapshot of current proxy order, without repeated Qt walks."""
        model = self.sourceModel()
        if model is None:
            return ()
        # Identity/revision also covers replacement of project data and isolated
        # derived changes that intentionally do not emit dataChanged.
        key = self._row_ids_key(model)
        if self._row_ids_cache is not None and self._row_ids_cache[0] == key:
            return self._row_ids_cache[1]
        rows = model.project.rows
        if (self.row_id_filter is None and not self.rules and self._filter_mask is None
                and self.sortColumn() < 0 and self.columnCount()):
            identifiers = tuple(row.id for row in rows)
        else:
            identifiers = tuple(rows[self.mapToSource(self.index(i, 0)).row()].id
                                for i in range(self.rowCount()))
        current = self._row_ids_key(model)
        if current == key:
            self._row_ids_cache = key, identifiers
        return identifiers

    def _row_ids_key(self, model):
        return id(model), id(model.project), model.data_revision, self._row_ids_revision

    def setSourceModel(self, model):
        old = self.sourceModel()
        if old is model:
            return
        if old is not None:
            old.derivedStructureChanged.disconnect(self._derived_structure_changed)
            old.qcExclusionsChanged.disconnect(self._qc_exclusions_changed)
            old.columnContextChanged.disconnect(self._column_context_changed)
        self._invalidate_row_ids()
        self._table_model = model
        self._column_context = self._read_column_context()
        super().setSourceModel(model)
        if model is not None:
            model.derivedStructureChanged.connect(self._derived_structure_changed)
            model.qcExclusionsChanged.connect(self._qc_exclusions_changed)
            model.columnContextChanged.connect(self._column_context_changed)

    def _read_column_context(self):
        if self._table_model is None or not self._context_columns:
            return {}
        return {
            column.id: ((column.unit or "").strip(), column.metadata.get("stale"))
            for column in self._table_model.project.columns if column.id in self._context_columns
        }

    def _column_context_changed(self):
        if self._read_column_context() != self._column_context:
            self.set_rules(self.rules, self.combine)

    def _qc_exclusions_changed(self):
        if any(rule["operator"] == "qc_excluded" for rule in leaf_rules(self.rules)):
            self.set_rules(self.rules, self.combine)

    def _derived_structure_changed(self, column_ids):
        project = self.sourceModel().project
        filters_affected = any(rule["column"] in column_ids for rule in leaf_rules(self.rules))
        sort_ids = {cid for cid, _ in self.sort_rules}
        if not sort_ids and 0 <= self.sortColumn() < len(project.columns):
            sort_ids.add(project.columns[self.sortColumn()].id)
        if sort_ids & column_ids:
            with self._sorting(self.sortColumn()):
                self.invalidate()
        elif filters_affected:
            ordering = self._sorting(self.sortColumn()) if self._filter_may_expand(column_ids) else nullcontext()
            with ordering:
                if hasattr(self, "endFilterChange"):
                    self.beginFilterChange()
                    self.endFilterChange(QSortFilterProxyModel.Direction.Rows)
                else:
                    self.invalidateRowsFilter()  # Qt 6.8/6.9 compatibility.
        if filters_affected:
            self.filtersChanged.emit()

    def _filter_may_expand(self, column_ids):
        """Detect possible new members without scanning sample values.

        Absent columns behave as None. Inserting a column replaces that constant
        predicate; removing it restores the constant. An AND/OR group that can
        only lose matches needs no temporary sorting preparation.
        """
        if any("rules" in rule for rule in self.rules):
            return True
        missing = [_matches_missing(rule) for rule in self.rules if rule["column"] in column_ids]
        if not missing:
            return False
        accepts_missing = all(missing) if self.combine == "AND" else any(missing)
        present = {column.id for column in self.sourceModel().project.columns}
        if column_ids <= present:
            return not accepts_missing  # Added columns: old values were all None.
        if column_ids.isdisjoint(present):
            return accepts_missing  # Removed columns: new values are all None.
        return True  # Mixed structural changes: conservatively prepare ordering.

    def _set_filter_rules(self, rules, combine):
        self.rules, self.combine = rules, combine
        self._context_columns = {
            rule["column"] for rule in leaf_rules(rules) if rule["operator"] in COLUMN_CONTEXT_OPERATORS
        }
        self._column_context = self._read_column_context()

    def set_rules(self, rules, combine="AND", *, row_ids=_KEEP_ROW_IDS):
        self.filteringChanged.emit(True)
        try:
            self._apply_rules(rules, combine, row_ids)
        finally:
            self.filteringChanged.emit(False)

    def _apply_rules(self, rules, combine, row_ids):
        membership = self.row_id_filter
        if row_ids is not _KEEP_ROW_IDS:
            membership = None if row_ids is None else frozenset(row_ids)
        model = self._table_model
        large_complex_filter = model is not None and len(model.project.rows) >= 4096 and any(
            "rules" in rule or rule["operator"] in COMPLEX_FILTER_OPERATORS
            for rule in (*rules, *self.rules)
        )
        opening_set = row_ids is not _KEEP_ROW_IDS and membership is not None and rules != self.rules
        if membership != self.row_id_filter or opening_set or large_complex_filter:
            self._set_filter_rules(rules, combine)
            self.row_id_filter = membership
            # A sparse result can change thousands of disjoint spans. Rebuild
            # the mapping once instead of sending a Qt signal for each span.
            sparse_set = membership is not None and len(membership) < 4096
            self._filter_mask = None if sparse_set else self._prepare_filter_mask()
            try:
                with nullcontext() if sparse_set else self._sorting(self.sortColumn()):
                    self.invalidate()
                    if self._filter_mask is not None:
                        self.rowCount()
            finally:
                self._filter_mask = None
            self.filtersChanged.emit()
            return
        modern_filter_change = hasattr(self, "endFilterChange")
        if modern_filter_change:
            self.beginFilterChange()
        self._set_filter_rules(rules, combine)
        self._sort_ranks = self.sort_ranks(self.sortColumn())
        try:
            self._filter_mask = mask = self._prepare_filter_mask()
            source_key = self._row_ids_key(model)[:3] if mask is not None else None
            # Filtering changes row membership, not the order of existing rows.
            if modern_filter_change:
                self.endFilterChange(QSortFilterProxyModel.Direction.Rows)
            else:
                self.invalidateRowsFilter()  # Compatibility with Qt 6.8/6.9.
            if (mask is not None and self._filter_mask is mask and self._table_model is model
                    and membership is None and self.sortColumn() < 0 and self.columnCount()):
                key = self._row_ids_key(model)
                if key[:3] == source_key:
                    # Without sorting, the computed mask already gives Qt's
                    # row order. Cache it before the first analysis asks.
                    rows = model.project.rows
                    identifiers = tuple(rows[i].id for i in np.flatnonzero(mask))
                    if len(identifiers) == self.rowCount():
                        self._row_ids_cache = key, identifiers
        finally:
            self._sort_ranks = None
            self._filter_mask = None
        self.filtersChanged.emit()

    def _prepare_filter_mask(self):
        """Evaluate supported column rules during the full filter change."""
        model = self.sourceModel()
        if not self.rules or model is None or len(model.project.rows) < 4096:
            return None
        return self._group_filter_mask(model.project.rows, self.rules, self.combine)

    def _group_filter_mask(self, rows, rules, combine):
        mask = None
        for rule in rules:
            if "rules" in rule:
                accepted = self._group_filter_mask(rows, rule["rules"], rule["combine"])
                if accepted is not None and rule.get("negate"):
                    accepted = ~accepted
            else:
                accepted = self._column_filter_mask(rows, rule)
            if accepted is None:
                return None
            if mask is None:
                mask = accepted
            elif combine == "AND":
                mask = mask & accepted
            else:
                mask = mask | accepted
        return mask

    def _column_filter_mask(self, rows, rule):
        operator, target = rule["operator"], rule.get("value", "")
        if operator in COLUMN_CONTEXT_OPERATORS:
            project = self._table_model.project
            column = rule["column"]
            result = np.fromiter(
                (self._matches_column_context(project, row, column, operator, target) for row in rows),
                dtype=bool, count=len(rows),
            )
            return ~result if rule.get("negate") else result
        if operator in ("gt", "ge", "lt", "le") and type(target) in (int, float):
            accepted_types = (int, float, bool)
        elif operator in ("contains", "eq", "ne") and type(target) is str:
            accepted_types = (str,)
        else:
            return None
        cid = rule["column"]
        values = []
        for row in rows:
            value = row.values.get(cid)
            if value is not None and type(value) not in accepted_types:
                return None
            values.append(value)
        # Object dtype preserves large integers and literal string values.
        series = pd.Series(values, dtype=object)
        if operator == "contains":
            if series.isna().all():
                accepted = series.notna()
            else:
                accepted = series.str.casefold().str.contains(target.casefold(), regex=False, na=False)
        else:
            accepted = getattr(series, operator)(target)
            if operator == "ne":
                accepted &= series.notna()
        result = accepted.to_numpy(dtype=bool)
        return ~result if rule.get("negate") else result

    def sort_ranks(self, column):
        model = self.sourceModel()
        if model is None or not 0 <= column < len(model.project.columns):
            return None
        project = model.project
        if self.sort_rules:
            return self._multiple_numeric_ranks(project)
        identifier = project.columns[column].id
        values = [row.values.get(identifier) for row in project.rows]
        numeric = all(value is None or isinstance(value, (int, float)) for value in values)
        textual = all(value is None or isinstance(value, str) for value in values)
        if not numeric and not textual:
            return None  # Keep the original mixed numeric/qualified-text comparison.
        keys = [(value is None, (0 if numeric else "") if value is None else value if numeric else value.casefold())
                for value in values]
        ranks = {key: rank for rank, key in enumerate(sorted(set(keys)))}
        return [ranks[key] for key in keys]

    def _multiple_numeric_ranks(self, project):
        """Encode numeric keys exactly; keep mixed/qualified comparisons unchanged."""
        rank_columns = []
        for identifier, descending in self.sort_rules:
            values = [row.values.get(identifier) for row in project.rows]
            if any(value is not None and (type(value) not in (int, float)
                   or type(value) is float and not math.isfinite(value)) for value in values):
                return None
            ordered = sorted({value for value in values if value is not None}, reverse=descending)
            rank_by_value = {value: rank for rank, value in enumerate(ordered)}
            # Missing values are last within each rule, independently of its direction.
            rank_columns.append([len(ordered) if value is None else rank_by_value[value] for value in values])
        # The existing comparator breaks complete ties by original source row.
        keys = list(zip(*rank_columns, range(len(project.rows))))
        ranks = {key: rank for rank, key in enumerate(sorted(keys))}
        return [ranks[key] for key in keys]

    @contextmanager
    def _sorting(self, column):
        self._sort_ranks = self.sort_ranks(column)
        try:
            model = self.sourceModel()
            if model is None:
                yield
            else:
                with model.cache_sort_indexes(column):
                    yield
        finally:
            # Never reuse ranks after an edit or an implicit Qt re-sort.
            self._sort_ranks = None

    def sort(self, column, order=Qt.SortOrder.AscendingOrder):
        with self._sorting(column):
            super().sort(column, order)
        self.sortingChanged.emit()

    def set_sort_rules(self, rules, column=0, order=Qt.SortOrder.AscendingOrder):
        """Change comparison rules and order in one prepared Qt operation.

        Qt skips sort() when column and direction are unchanged, even when our
        comparison rules changed. Only that case needs explicit invalidation.
        """
        rules = [tuple(rule) for rule in rules]
        changed = rules != [tuple(rule) for rule in self.sort_rules]
        self.sort_rules = rules
        if column == self.sortColumn() and order == self.sortOrder():
            if changed and column >= 0:
                with self._sorting(column):
                    self.invalidate()
            if changed:
                self.sortingChanged.emit()
            return
        self.sort(column, order)

    def filterAcceptsRow(self, source_row, parent):
        if self._filter_mask is not None and self.row_id_filter is None:
            return bool(self._filter_mask[source_row])
        if self.row_id_filter is None and not self.rules:
            return True
        project = self._table_model.project
        row = project.rows[source_row]
        if self.row_id_filter is not None and row.id not in self.row_id_filter:
            return False
        if not self.rules:
            return True
        if self._filter_mask is not None:
            return bool(self._filter_mask[source_row])
        return self._matches_rules(project, row, self.rules, self.combine)

    def _matches_rules(self, project, row, rules, combine):
        # Table columns can temporarily disappear during Undo. The scalar leaf
        # evaluator treats their values as None, like the vectorized filter.
        results = []
        for rule in rules:
            if "rules" in rule:
                result = self._matches_rules(project, row, rule["rules"], rule.get("combine", "AND"))
            else:
                result = self._matches_leaf(project, row, rule)
            results.append(not result if rule.get("negate") else result)
        return all(results) if combine == "AND" else any(results)

    def _matches_leaf(self, project, row, rule):
        column = rule["column"]
        operator, target = rule["operator"], rule.get("value", "")
        if operator in MEASUREMENT_STATUS_OPERATORS:
            return _matches_measurement_status(project, row, column, operator)
        if operator in COLUMN_CONTEXT_OPERATORS:
            return self._matches_column_context(project, row, column, operator, target)
        value = row.values.get(column)
        if operator == "empty":
            return value is None or value == ""
        if value is None:
            return False
        if operator == "not_empty":
            return value != ""
        if operator == "contains":
            return str(target).casefold() in str(value).casefold()
        if operator == "eq":
            return value == target
        if operator == "ne":
            return value != target
        if operator in ("gt", "ge", "lt", "le"):
            try:
                return getattr(comparison, operator)(value, target)
            except TypeError:
                # Qualified numeric results are not ordinary numbers.
                return False
        return False

    def _matches_column_context(self, project, row, column, operator, target):
        context = self._column_context.get(column)
        if context is None:
            return False
        unit, stale = context
        if operator in ("stale", "up_to_date"):
            expected_stale = operator == "stale"
            return stale is expected_stale and row.values.get(column) not in (None, "")
        measurement = project.science.current(row.id, column)
        if measurement is not None:
            unit = (measurement.unit or "").strip()
        elif row.values.get(column) in (None, ""):
            return False
        return unit == ("" if operator == "unit_missing" else target)

    def lessThan(self, left, right):
        if self._sort_ranks is not None:
            return self._sort_ranks[left.row()] < self._sort_ranks[right.row()]
        project = self.sourceModel().project
        a_row, b_row = project.rows[left.row()], project.rows[right.row()]
        if self.sort_rules:
            for cid, descending in self.sort_rules:
                a, b = a_row.values.get(cid), b_row.values.get(cid)
                if a == b:
                    continue
                if a is None:
                    return False
                if b is None:
                    return True
                if isinstance(a, str) or isinstance(b, str):
                    a, b = str(a).casefold(), str(b).casefold()
                return a > b if descending else a < b
            return left.row() < right.row()
        column_id = project.columns[left.column()].id
        a, b = a_row.values.get(column_id), b_row.values.get(column_id)
        if a is None:
            return False
        if b is None:
            return True
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a < b
        return str(a).casefold() < str(b).casefold()
