"""Qt-independent local project. Import rows remain distinct, even with equal names."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
from uuid import uuid4

from .measurements import MeasurementStore, parse_number


def new_id() -> str:
    return str(uuid4())


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Column:
    name: str
    kind: str = "text"
    id: str = field(default_factory=new_id)
    unit: str = ""
    metadata: dict = field(default_factory=dict)


class _RowExtensions:
    # Keep extension attributes and weak references without putting the four
    # declared fields in an instance dict traversed by every full collection.
    __slots__ = ('__dict__', '__weakref__')


@dataclass(slots=True)
class Row(_RowExtensions):
    values: dict
    id: str = field(default_factory=new_id)
    source: dict = field(default_factory=dict)
    original: dict = field(default_factory=dict)

    def __setstate__(self, state):
        # Older internal pickles stored one dict; slotted records store
        # (extension attributes, slot values). Restore both through setattr
        # so legacy fields populate slots rather than shadowing them in __dict__.
        attributes, slots = (state, {}) if isinstance(state, dict) else state
        for values in (attributes or {}, slots):
            for name, value in values.items():
                setattr(self, name, value)


@dataclass
class Project:
    columns: list[Column] = field(default_factory=list)
    rows: list[Row] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    view: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)
    id: str = field(default_factory=new_id)
    science: MeasurementStore = field(default_factory=MeasurementStore)

    def row_map(self):
        return {row.id: row for row in self.rows}

    def event(self, action: str, **details):
        self.history.append({"at": timestamp(), "action": action, **deepcopy(details)})

    def sync_measurements(self, *, rows=None, edited_cells=None):
        self.science.sync(self, rows=rows, edited_cells=edited_cells)

    def refresh(self, *, invalidate_interpolation=True, rows=None, sync_measurements=True, changed_columns=None,
                stratigraphy_report=None, edited_cells=None):
        """Recalculate; return scoped affected IDs, or None when a full pass is needed.

        Skip synchronization only for isolated changes to derived columns.
        """
        if rows is not None:
            rows = list(rows)
        if sync_measurements:
            changed_columns = None
            if edited_cells is None:
                self.sync_measurements(rows=rows)
            else:
                # Initializing a sample can create working measurements in any column.
                if all(rid in self.science.samples for rid in edited_cells):
                    changed_columns = {cid for columns in edited_cells.values() for cid in columns}
                self.sync_measurements(rows=rows, edited_cells=edited_cells)
        self.mark_interpolation_stale(all_columns=invalidate_interpolation)
        from autopyro.analysis.calculations import recompute
        from autopyro.analysis.nonpyro import check_saved_reports, recompute as recompute_nonpyro
        from autopyro.geology.stratigraphy import recompute as recompute_stratigraphy
        # Other engines can change columns outside the explicit edit set.
        global_engines = any(c.metadata.get("nonpyro") or c.metadata.get("stratigraphy") for c in self.columns)
        if global_engines:
            changed_columns = None
        if edited_cells is not None and changed_columns is not None and rows is not None:
            groups = {}
            for row in rows:
                columns = frozenset(edited_cells.get(row.id, changed_columns))
                groups.setdefault(columns, []).append(row)
            affected_columns = set()
            for columns, group_rows in groups.items():
                affected_columns.update(recompute(self, rows=group_rows, changed_columns=columns))
        else:
            affected_columns = recompute(self, rows=rows, changed_columns=changed_columns)
        recompute_nonpyro(self)
        stratigraphy_rows = None if any(c.metadata.get("nonpyro") for c in self.columns) else rows
        recompute_stratigraphy(self, rows=stratigraphy_rows, prepared_report=stratigraphy_report)
        if self.view.get("curve_fit_runs") or self.view.get("curve_fit_results"):
            from autopyro.curves.curve_outputs import invalidate_curve_outputs
            invalidate_curve_outputs(self, None if global_engines else rows, invalidate_interpolation,
                                     changed_columns=affected_columns)
        if self.view.get("joint_curve_report") or any(c.metadata.get("joint_curve_fit") for c in self.columns):
            from autopyro.curves.joint_sources import update_joint_status
            update_joint_status(self, changed_rows=None if global_engines else rows,
                                changed_columns=None if sync_measurements or affected_columns is None or changed_columns is None
                                else set(changed_columns) | affected_columns)
        if self.view.get("nonpyro_results"):
            check_saved_reports(self)
        if self.view.get("interpretation_result"):
            from autopyro.schemes.interpretation import check_saved_result

            check_saved_result(self, changed_rows=None if global_engines else rows)
        return affected_columns

    def mark_interpolation_stale(self, *, all_columns=False):
        from autopyro.analysis.interpolation_constraints import input_ids

        column_ids = {column.id for column in self.columns}
        current = self.view.get("stratigraphy") or {}
        for column in self.columns:
            metadata = column.metadata
            if not metadata.get("method") or metadata.get("stale"):
                continue
            if all_columns:
                metadata["stale"] = True
                continue
            constraints = metadata.get("constraints")
            if not isinstance(constraints, dict):
                constraints = {}
            saved = constraints.get("stratigraphy")
            changed_stratigraphy = isinstance(saved, dict) and (
                saved.get("binding") != current.get("binding")
                or bool(constraints.get("levels") and saved.get("scheme") != current.get("scheme"))
            )
            try:
                dependencies = [metadata.get("x_id"), metadata.get("y_id"), *input_ids(constraints)]
                groups = metadata.get("group_ids")
                if isinstance(groups, (list, tuple)):
                    dependencies.extend(groups)
                missing_input = any(cid and cid not in column_ids for cid in dependencies)
            except TypeError:
                missing_input = True
            if missing_input or changed_stratigraphy:
                metadata["stale"] = True

    def select_measurement(self, sample_id, property_id, measurement_id):
        """Select by UUID, update the table projection and recalculate dependents."""
        row = self.row_map().get(sample_id)
        if row is None or property_id not in row.values:
            raise ValueError("Проба или показатель отсутствуют в таблице")
        if measurement_id is None:
            self.science.working.get(sample_id, {}).pop(property_id, None)
            row.values[property_id] = None
        else:
            self.science.select(sample_id, property_id, measurement_id)
            row.values[property_id] = self.science.measurements[measurement_id].table_value()
        self.event("select_measurement", sample_id=sample_id, property_id=property_id, measurement_id=measurement_id)
        # Pyrolysis and stratigraphic assignments are row-local.
        global_engines = any(c.metadata.get("nonpyro") for c in self.columns)
        self.refresh(rows=None if global_engines else [row])

    def validate(self):
        from .measurements import parse_result

        ids = [c.id for c in self.columns]
        column_ids = set(ids)
        if len(column_ids) != len(ids) or len(self.row_map()) != len(self.rows):
            raise ValueError("Повторяющиеся идентификаторы в проекте")
        numeric_ids = [column.id for column in self.columns if column.kind == "number"]
        for row in self.rows:
            if set(row.values) != column_ids:
                raise ValueError("Состав строки не соответствует колонкам")
            for cid in numeric_ids:
                value = row.values[cid]
                value_type = type(value)
                # Plain finite numbers need no temporary measurement dictionary.
                if (value is None
                        or (value_type is float and math.isfinite(value))
                        or (value_type is int and -(2 ** 53) <= value <= 2 ** 53)):
                    continue
                parse_result(value, "number")
        self.science.validate(self)

    def save(self, path: str | Path):
        """Atomic SQLite save; previous project formats are backed up before replacement."""
        from autopyro.database.storage import save_project
        save_project(self, path)

    def save_recovery(self, path: str | Path, *, cancelled=None):
        from autopyro.database.storage import save_project
        save_project(self, path, recovery=True, cancelled=cancelled)

    @classmethod
    def load(cls, path: str | Path):
        from autopyro.database.storage import load_payload, file_digest

        before = file_digest(path)
        data, legacy = load_payload(path)
        if file_digest(path) != before:
            raise ValueError("Проект изменился во время чтения; откройте его повторно")
        data["columns"] = [Column(**column) for column in data["columns"]]
        data["rows"] = [Row(**row) for row in data["rows"]]
        project = cls(**data)
        from .audit import share_history_protocols

        share_history_protocols(project.history)
        if legacy:
            project.sync_measurements()
        project.validate()
        project.mark_interpolation_stale()
        project._storage_path = Path(path).resolve()
        project._storage_digest = before
        if any(column.metadata.get("calculation") for column in project.columns):
            from autopyro.analysis.calculations import recompute

            recompute(project)
        if (any(column.metadata.get("nonpyro") for column in project.columns)
                or "nonpyro_results" in project.view):
            from autopyro.analysis.nonpyro import recompute as recompute_nonpyro, check_saved_reports

            recompute_nonpyro(project)
            check_saved_reports(project)
        if any(column.metadata.get("stratigraphy") for column in project.columns):
            from autopyro.geology.stratigraphy import recompute as recompute_stratigraphy

            recompute_stratigraphy(project)
        if project.view.get("interpretation_result"):
            from autopyro.schemes.interpretation import check_saved_result

            check_saved_result(project)
        return project
