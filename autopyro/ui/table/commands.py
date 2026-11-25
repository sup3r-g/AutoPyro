from __future__ import annotations

from ..i18n import tr

from copy import deepcopy
from autopyro.model.core import timestamp
from autopyro.model.audit import archive_protocol
from autopyro.model.snapshots import copy_table_state
from autopyro.model.measurements import derived
from .snapshot_policy import large_snapshot
from .undo_state import PackedTableState, pack_table_state, restore_table_state

from PySide6.QtCore import QModelIndex
from PySide6.QtGui import QUndoCommand


def _prepare_snapshot(model, state, function):
    """Keep large snapshot work off the GUI thread, before resetting the model."""
    from ..tasks import run_with_progress
    if isinstance(state, PackedTableState):
        large_state = state.cell_count >= 100_000 or state.measurement_count >= 100_000
    else:
        large_state = len(state[0]) * len(state[1]) >= 100_000 or len(state[2].measurements) >= 100_000
    if not large_snapshot(model.project) and not large_state:
        return function()
    project, revision = model.project, model.data_revision
    result = run_with_progress(model.parent(), tr("Подготовка отмены изменений…"), function)
    if model.project is not project or model.data_revision != revision:
        raise ValueError(tr("Данные изменились во время подготовки снимка. Повторите операцию"))
    return result


def _application_snapshot(state, protocol, history_protocol):
    """Isolate the active state, registry and history from the undo command."""
    return {"state": restore_table_state(state), "run_protocol": deepcopy(protocol),
            "history_protocol": deepcopy(history_protocol)}


def _check_interpretation_snapshot(model, code, action):
    project = model.project
    if report := project.view.get("interpretation_result"):
        from autopyro.schemes.interpretation import (
            check_saved_binding, check_saved_result, mark_saved_result_stale,
        )

        scope = report.get("scope")
        scope_controller = getattr(model.parent(), "analysis_scope", None)
        if scope == "all":
            check_saved_result(project)
        elif scope_controller is not None and any(mode == scope for _, mode in scope_controller.choices()):
            check_saved_result(project)
            if not report.get("stale"):
                try:
                    scope_rows = scope_controller.rows(scope)
                except ValueError:
                    mark_saved_result_stale(project, "binding_changed", fields=["scope"])
                else:
                    check_saved_binding(project, {}, scope_rows=scope_rows)
        else:
            mark_saved_result_stale(project, code, action=action)


class ImportCommand(QUndoCommand):
    def __init__(self, model, prepared, protocol, expected_digest):
        from autopyro.data_io.importing import digest
        project = model.project
        after = (prepared.columns, prepared.rows, prepared.science)

        def snapshots():
            if digest(project) != expected_digest:
                raise ValueError(tr('Проект изменился после предпросмотра импорта'))
            owned_protocol = deepcopy(protocol)
            return {"protocol": owned_protocol, "history_protocol": archive_protocol(owned_protocol),
                    "before": pack_table_state(copy_table_state((project.columns, project.rows, project.science))),
                    "after": pack_table_state(copy_table_state(after)),
                    "before_templates": deepcopy(project.view.get("import_templates", {})),
                    "after_templates": deepcopy(prepared.view.get("import_templates", {})),
                    "template_definitions": {key: deepcopy(prepared.view.get(key, {})) for key in
                                             ("import_templates", "import_template_revisions")}}

        data = _prepare_snapshot(model, after, snapshots)
        super().__init__(tr('Добавить результаты из файла'))
        self.model = model
        self.protocol, self.before, self.after = data["protocol"], data["before"], data["after"]
        self.history_protocol = data["history_protocol"]
        self.before_templates, self.after_templates = data["before_templates"], data["after_templates"]
        self.template_definitions = data["template_definitions"]

    def apply(self, state, action):
        project = self.model.project
        from autopyro.data_io.import_templates import merge_views

        def snapshot():
            data = _application_snapshot(state, self.protocol, self.history_protocol)
            definitions = {key: deepcopy(project.view.get(key, {})) for key in
                           ("import_templates", "import_template_revisions")}
            merge_views(definitions, self.template_definitions)
            data["definitions"] = definitions
            data["templates"] = deepcopy(self.after_templates if action == "import" else self.before_templates)
            return data

        data = _prepare_snapshot(self.model, state, snapshot)
        columns, rows, science = data["state"]
        definitions = data["definitions"]
        self.model.beginResetModel()
        project.columns, project.rows, project.science = columns, rows, science
        if definitions.get("import_template_revisions"):
            project.view["import_template_revisions"] = definitions["import_template_revisions"]
            project.view["import_templates"] = data["templates"]
        project.view.setdefault("import_runs", {})[self.protocol["id"]] = {
            "status": "applied" if action == "import" else "undone", "protocol": data["run_protocol"]}
        for report in project.view.get("nonpyro_results", {}).values():
            report["stale"] = True
        # Protocol isolation was prepared in the worker; timestamp the actual commit.
        project.history.append({"at": timestamp(), "action": action, "protocol": data["history_protocol"]})
        self.model.endResetModel()
        self.model.structureChanged.emit()
        _check_interpretation_snapshot(self.model, "import_changed", action)
        self.model.notify_changed()

    def redo(self):
        self.apply(self.after, "import")

    def undo(self):
        self.apply(self.before, "undo_import")


class QCBatchDecisionCommand(QUndoCommand):
    def __init__(self, model, issue_ids, action, comment):
        super().__init__(tr('Групповое решение QC'))
        self.model = model
        self.issue_ids = list(issue_ids)
        records = model.project.view.get("qc_decisions", {})
        self.before = {key: deepcopy(records[key]) for key in self.issue_ids if key in records}
        self.after = {"action": action, "comment": comment, "at": timestamp()}

    def redo(self):
        records = self.model.project.view.setdefault("qc_decisions", {})
        records.update((key, self.after.copy()) for key in self.issue_ids)
        self.model.project.event("qc_decision_batch", operation="redo", issue_ids=self.issue_ids, value=self.after)
        self.model.notify_changed(data_changed=False)

    def undo(self):
        records = self.model.project.view["qc_decisions"]
        for key in self.issue_ids:
            if key in self.before:
                records[key] = deepcopy(self.before[key])
            else:
                records.pop(key, None)
        self.model.project.event("qc_decision_batch", operation="undo", issue_ids=self.issue_ids)
        self.model.notify_changed(data_changed=False)


class QCCommand(QUndoCommand):
    def __init__(self, model, key, value, *, measurement_id=None, measurement_ids=()):
        super().__init__(tr('Изменить решение QC'))
        self.model, self.key, self.value = model, key, deepcopy(value)
        self.measurement_id = measurement_id
        self.measurement_ids = tuple(measurement_ids)
        old = model.project.view.get(key[0], {}).get(key[1])
        if self.measurement_ids:
            self.old = {mid: deepcopy(old[mid]) for mid in self.measurement_ids}
        else:
            self.old = deepcopy(old if measurement_id is None else (old or {}).get(measurement_id))

    def apply(self, value):
        if self.key == ("qc_exclusions", "interpolation"):
            from autopyro.model.tabular import excluded_sample_ids

            project = self.model.project
            candidates = [column for column in project.columns
                          if column.metadata.get("method") and not column.metadata.get("stale")]
            properties = {column.id: {column.metadata.get("x_id"), column.metadata.get("y_id")}
                          for column in candidates}
            before = {column.id: excluded_sample_ids(project, properties[column.id], "interpolation")
                      for column in candidates}
        decisions = self.model.project.view.setdefault(self.key[0], {})
        if self.measurement_ids:
            records = decisions.setdefault(self.key[1], {})
            for mid in self.measurement_ids:
                records.pop(mid, None)
            if value is not None:
                records.update(deepcopy(value))
        elif self.measurement_id is not None:
            records = decisions.setdefault(self.key[1], {})
            if value is None:
                records.pop(self.measurement_id, None)
            else:
                records[self.measurement_id] = deepcopy(value)
        elif value is None:
            decisions.pop(self.key[1], None)
        else:
            decisions[self.key[1]] = deepcopy(value)
        details = {"measurement_id": self.measurement_id} if self.measurement_id is not None else {}
        if self.measurement_ids:
            details["measurement_ids"] = list(self.measurement_ids)
        self.model.project.event("qc_decision", section=self.key[0], key=self.key[1], value=value, **details)
        if self.key == ("qc_exclusions", "interpolation"):
            header_status = self.model._fit_header_status()
            for column in candidates:
                metadata = column.metadata
                after = excluded_sample_ids(project, properties[column.id], "interpolation")
                if (before[column.id] ^ after).intersection(metadata.get("scope_row_ids", ())):
                    metadata["stale"] = True
            self.model._notify_fit_headers(header_status)
            self.model.qcExclusionsChanged.emit()
            self.model.notify_changed()
            return
        if self.key == ("qc_exclusions", "interpretation") and self.model.project.view.get("interpretation_result"):
            from autopyro.schemes.interpretation import check_saved_result

            project = self.model.project
            if self.measurement_id is None:
                check_saved_result(project)
            elif (measurement := project.science.measurements.get(self.measurement_id)) is not None:
                positions = self.model.resolve_row_positions([measurement.sample_id])
                check_saved_result(project, changed_rows=[project.rows[pos] for pos in positions.values()])
        if self.key == ("qc_exclusions", "calculations"):
            self.model.refresh()
        else:
            if self.key[0] == "qc_exclusions":
                if self.key[1] == "statistics":
                    self.model.refresh_joint_qc_status()
                    from autopyro.analysis.nonpyro import check_saved_reports, recompute

                    project = self.model.project
                    check_saved_reports(project)
                    if any(column.metadata.get("nonpyro") for column in project.columns):
                        recompute(project)
                        self.model.qcExclusionsChanged.emit()
                        self.model.notify_changed()
                        return
                self.model.qcExclusionsChanged.emit()
            self.model.notify_changed(data_changed=False)

    def redo(self):
        self.apply(self.value)

    def undo(self):
        self.apply(self.old)


class TemplateLibraryCommand(QUndoCommand):
    def __init__(self, model, state):
        from autopyro.data_io.import_templates import merge_views
        super().__init__(tr('Изменить библиотеку шаблонов импорта'))
        self.model = model
        self.before = deepcopy(model.project.view.get("import_templates", {}))
        self.after = deepcopy(state.get("import_templates", {}))
        self.definitions = {key: deepcopy(model.project.view.get(key, {})) for key in ("import_templates", "import_template_revisions")}
        merge_views(self.definitions, state)

    def apply(self, active):
        from autopyro.data_io.import_templates import merge_views
        merge_views(self.model.project.view, self.definitions)
        self.model.project.view["import_templates"] = deepcopy(active)
        self.model.project.event("import_template_library", templates=list(active))
        self.model.notify_changed()

    def redo(self):
        self.apply(self.after)

    def undo(self):
        self.apply(self.before)


class RefreshDatabaseCommand(QUndoCommand):
    def __init__(self, model, prepared, protocol, expected_digest):
        from autopyro.data_io.importing import digest
        project = model.project
        after = (prepared.columns, prepared.rows, prepared.science, prepared.view["database_snapshot"])

        def snapshots():
            if digest(project) != expected_digest:
                raise ValueError(tr('Проект изменился после сравнения снимков'))
            owned_protocol = deepcopy(protocol)
            return {"protocol": owned_protocol, "history_protocol": archive_protocol(owned_protocol),
                    "before": pack_table_state(copy_table_state((project.columns, project.rows, project.science,
                                         project.view.get("database_snapshot")))),
                    "after": pack_table_state(copy_table_state(after))}

        data = _prepare_snapshot(model, after, snapshots)
        super().__init__(tr('Обновить снимок БД'))
        self.model = model
        self.protocol, self.before, self.after = data["protocol"], data["before"], data["after"]
        self.history_protocol = data["history_protocol"]

    def apply(self, state, action):
        project = self.model.project
        data = _prepare_snapshot(self.model, state,
                                 lambda: _application_snapshot(state, self.protocol, self.history_protocol))
        columns, rows, science, snapshot = data["state"]
        self.model.beginResetModel()
        project.columns, project.rows, project.science = columns, rows, science
        if snapshot is None:
            project.view.pop("database_snapshot", None)
        else:
            project.view["database_snapshot"] = snapshot
        project.view.setdefault("database_refresh_runs", {})[self.protocol["id"]] = {
            "status": "applied" if action == "refresh_database" else "undone", "protocol": data["run_protocol"]}
        if self.protocol["working_changed"]:
            for report in project.view.get("nonpyro_results", {}).values():
                report["stale"] = True
        # Keep the event schema while avoiding another large copy in the GUI.
        project.history.append({"at": timestamp(), "action": action, "protocol": data["history_protocol"]})
        self.model.endResetModel()
        self.model.structureChanged.emit()
        _check_interpretation_snapshot(self.model, "database_changed", action)
        self.model.notify_changed()

    def redo(self):
        self.apply(self.after, "refresh_database")

    def undo(self):
        self.apply(self.before, "undo_refresh_database")


class SelectMeasurementCommand(QUndoCommand):
    def __init__(self, model, sample_id, property_id, measurement_id):
        super().__init__(tr('Выбрать рабочее измерение'))
        self.model, self.sample_id, self.property_id = model, sample_id, property_id
        store = model.project.science
        target = store.measurements.get(measurement_id)
        if target is None or (target.sample_id, target.property_id) != (sample_id, property_id):
            raise ValueError(tr('Измерение относится к другой пробе или показателю'))
        current = store.current(sample_id, property_id)
        self.old_id = current.id if current else None
        self.new_id = measurement_id

    def apply(self, measurement_id):
        project = self.model.project
        rows = [project.row_map()[self.sample_id]]
        with self.model.preserve_recovery_edit(rows, None):
            project.select_measurement(self.sample_id, self.property_id, measurement_id)
            self.model.notify_changed(rows=rows)

    def redo(self):
        self.apply(self.new_id)

    def undo(self):
        self.apply(self.old_id)


class AddMeasurementCommand(QUndoCommand):
    def __init__(self, model, measurement, analysis):
        super().__init__(tr('Добавить результат анализа'))
        self.model, self.measurement, self.analysis = model, measurement, analysis

    def redo(self):
        store = self.model.project.science
        store.analyses[self.analysis.id] = self.analysis
        store.measurements[self.measurement.id] = self.measurement
        self.model.project.event("add_measurement", measurement_id=self.measurement.id, analysis_id=self.analysis.id)
        self.model.refresh(invalidate_interpolation=False)

    def undo(self):
        store = self.model.project.science
        store.measurements.pop(self.measurement.id)
        store.analyses.pop(self.analysis.id)
        self.model.project.event("undo_add_measurement", measurement_id=self.measurement.id)
        self.model.refresh(invalidate_interpolation=False)


class EditCommand(QUndoCommand):
    def __init__(self, model, changes, *, row_positions=None, validated=False):
        super().__init__(tr('Изменить ячейки: {0}', len(changes)))
        self.model = model
        if validated:
            # The batch preview already captured old values and source positions.
            self.row_positions = row_positions
            self.changes = changes
            return
        positions = model.resolve_row_positions((rid for rid, cid, new in changes), row_positions)
        self.row_positions = {}
        self.changes = []
        for rid, cid, new in changes:
            position = positions[rid]
            old = model.project.rows[position].values[cid]
            if old != new:
                self.changes.append((rid, cid, old, new))
                self.row_positions[rid] = position

    def apply(self, undo=False):
        project = self.model.project
        positions = self.model.resolve_row_positions(self.row_positions, self.row_positions)
        rows = [project.rows[positions[rid]] for rid in self.row_positions]
        with self.model.preserve_recovery_edit(rows, ((rid, cid) for rid, cid, _, _ in self.changes)):
            self.apply_cells(project, positions, rows, undo)

    def apply_cells(self, project, positions, rows, undo):
        history = []
        edited_cells = {}
        for rid, cid, old, new in self.changes:
            before, after = (new, old) if undo else (old, new)
            # Cell values are normally immutable scalars. Own any other values
            # in the audit record without copying the entire batch of dicts.
            history.append({"row": rid, "column": cid,
                            "old": before if type(before) in (str, int, float, bool, type(None)) else deepcopy(before),
                            "new": after if type(after) in (str, int, float, bool, type(None)) else deepcopy(after)})
            project.rows[positions[rid]].values[cid] = after
            edited_cells.setdefault(rid, []).append(cid)
        self.row_positions = positions
        project.history.append({"at": timestamp(), "action": "undo_edit" if undo else "edit", "changes": history})
        self.model.refresh(rows=rows, row_positions=positions, edited_cells=edited_cells)

    def redo(self):
        self.apply()

    def undo(self):
        self.apply(undo=True)


class AddColumnCommand(QUndoCommand):
    def __init__(self, model, column, values):
        super().__init__(tr('Добавить: {0}', column.name))
        self.model, self.column, self.values = model, column, values
        # Oversized protocols remain mutable inline JSON. Freeze their history
        # copy before push(), where preparation can fail without changing undo.
        self._prepared_history = self._inline_history_snapshot()

    def _inline_history_snapshot(self):
        if not isinstance(self.column.metadata.get("interpolation_protocol"), dict):
            return None  # Immutable archives and ordinary metadata stay cheap.
        project = self.model.project
        state = (project.columns, project.rows, project.science)
        return _prepare_snapshot(self.model, state, lambda: deepcopy({
            k: v for k, v in self.column.metadata.items() if k != "diagnostics"}))

    def refresh(self):
        if derived(self.column):
            self.model.refresh_derived_structure(self.column.id)
        else:
            self.model.refresh(invalidate_interpolation=False)

    def redo(self):
        self.insert_column()
        self.refresh()

    def insert_column(self):
        model = self.model
        metadata = self._prepared_history
        if metadata is None:
            metadata = self._inline_history_snapshot()
        self._prepared_history = None
        i = len(model.project.columns)
        model.updatingColumns.emit(True)
        model.beginInsertColumns(QModelIndex(), i, i)
        model.project.columns.append(self.column)
        for row in model.project.rows:
            row.values[self.column.id] = self.values.get(row.id)
        model.endInsertColumns()
        model.updatingColumns.emit(False)
        if metadata is None:
            model.project.event(
                "add_column", column=self.column.id,
                metadata={k: v for k, v in self.column.metadata.items() if k != "diagnostics"})
        else:
            # This copy is already independent; Project.event would copy again.
            model.project.history.append({"at": timestamp(), "action": "add_column",
                                          "column": self.column.id, "metadata": metadata})

    def undo(self):
        self.remove_column()
        self.refresh()

    def remove_column(self):
        model = self.model
        i = next(
            i for i, c in enumerate(model.project.columns) if c.id == self.column.id
        )
        model.updatingColumns.emit(True)
        model.beginRemoveColumns(QModelIndex(), i, i)
        model.project.columns.pop(i)
        for row in model.project.rows:
            row.values.pop(self.column.id)
        model.endRemoveColumns()
        model.updatingColumns.emit(False)
        model.project.event("undo_add_column", column=self.column.id)
