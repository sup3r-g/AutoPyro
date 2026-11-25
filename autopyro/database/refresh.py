"""Three-way snapshot refresh. Original records are immutable; source deletions retain data."""
from copy import deepcopy
from dataclasses import dataclass
import json

import pandas as pd

from autopyro.model.core import Row, new_id, parse_number, timestamp
from .sources import descriptor, field_key, identity, row_provenance
from autopyro.data_io.importing import digest
from autopyro.model.measurements import Analysis, Measurement, SourceRecord, derived, parse_result
from autopyro.model.snapshots import copy_project


def parsed(raw, kind):
    result = parse_result(raw, kind)
    if kind == "number" and result["qualifier"] == "unparsed":
        raise ValueError("Не удалось разобрать числовой результат источника")
    return Measurement("", "", "", **result).table_value()


def comparable(raw, kind):
    try:
        return parsed(raw, kind)
    except (ValueError, TypeError, OverflowError):
        return raw


def encoded_key(values):
    if any(value is None or isinstance(value, str) and not value.strip() for value in values):
        raise ValueError("Ключ источника содержит пропуск")
    return json.dumps([(type(value).__name__, value) for value in values], ensure_ascii=False)


def source_row(project, row):
    kind = project.source.get("type")
    if (kind == "sqlite" and row.source.get("type") == kind
            and row.source.get("path") == project.source.get("path")
            and row.source.get("table") == project.source.get("table")):
        return True
    if kind == "postgresql_nonpyro":
        if row.source.get("type"):
            try:
                if identity(row.source) != identity(project.source):
                    return False
            except (ValueError, KeyError):
                return False
        return row.source.get("source_record_id") is not None and not row.source.get("sha256")
    try:
        return identity(row.source) == identity(project.source)
    except (ValueError, KeyError):
        return False


def source_key(project, row, keys, values):
    if project.source["type"] == "postgresql_nonpyro":
        return encoded_key([str(row.source["source_record_id"])])
    if project.source["type"] == "postgresql":
        return encoded_key([str(row.source["row"])])
    columns = {c.id: c for c in project.columns}
    return encoded_key([parse_number(values[cid]) if columns[cid].kind == "number" else values[cid] for cid in keys])


@dataclass
class RefreshPlan:
    project_digest: str
    incoming_digest: str
    baseline: dict
    mapping: dict
    cells: list[dict]
    new_rows: list[dict]
    missing_rows: list[dict]
    schema_changes: dict
    errors: list[str]
    checked_at: str


def initial_baseline(project, keys):
    original_rows = [row for row in project.rows if source_row(project, row)]
    original_ids = {cid for row in original_rows for cid in row.original}
    fields = {c.id: field_key(c) for c in project.columns if not derived(c) and
              c.id in original_ids and
              ("source_header" in c.metadata or project.source["type"] == "postgresql_nonpyro")}
    if not fields:
        raise ValueError("Нет исходных колонок БД для сравнения")
    if project.source["type"] == "sqlite" and (not keys or set(keys) - set(fields) or len(set(keys)) != len(keys)):
        raise ValueError("Выберите уникальный составной ключ SQLite из исходных колонок")
    primary = {a.sample_id: a for a in project.science.analyses.values() if a.metadata.get("origin") == "table_row"}
    origins = {(m.sample_id, m.property_id): m.id for m in project.science.measurements.values()
               if m.source_id == getattr(primary.get(m.sample_id), "source_id", None)}
    rows = {}
    for row in original_rows:
        values = {cid: row.original[cid] for cid in fields if cid in row.original}
        key = source_key(project, row, keys, values)
        if key in rows:
            raise ValueError("Ключ не уникален в исходном снимке. Выберите другие поля")
        rows[key] = {"sample_id": row.id, "values": values, "references": {
            cid: origins.get((row.id, cid)) for cid in values}, "present": True}
    if not rows:
        raise ValueError("Исходные записи БД не найдены; требуется новый импорт с сохранением источника")
    known = project.source.get("source_columns")
    schema = ["name:" + name for name in known] if known and project.source["type"] != "postgresql_nonpyro" else list(fields.values())
    return {"source": descriptor(project.source), "keys": list(keys), "fields": fields, "rows": rows,
            "schema": schema, "checked_at": project.source.get("imported_at", project.source.get("created_at"))}


def preview_refresh(project, incoming, keys=()):
    if identity(project.source) != identity(incoming.source):
        raise ValueError("Обновление относится к другому источнику или другой области выборки")
    before = digest(project)
    project = copy_project(project)
    project.sync_measurements()
    saved = project.view.get("database_snapshot")
    if saved and keys and list(keys) != saved["keys"]:
        raise ValueError("Ключ сохранённого снимка закреплён; для другого ключа нужен новый импорт")
    baseline = deepcopy(saved) or initial_baseline(project, keys)
    if identity(baseline["source"]) != identity(project.source):
        raise ValueError("Источник сохранённого сравнения не соответствует проекту")
    keys = baseline["keys"]
    incoming_fields = {}
    for column in incoming.columns:
        key = field_key(column)
        if key in incoming_fields:
            raise ValueError("Неоднозначные имена полей источника; автоматическое обновление невозможно")
        incoming_fields[key] = column
    columns = {c.id: c for c in project.columns}
    mapping, errors = {}, []
    for cid, external in baseline["fields"].items():
        if cid not in columns:
            errors.append("Исходный показатель удалён из проекта; восстановите его для обновления")
            continue
        candidate = incoming_fields.get(external)
        if candidate:
            if project.source["type"] == "postgresql_nonpyro" and (candidate.unit != columns[cid].unit or candidate.kind != columns[cid].kind):
                errors.append(f"{columns[cid].name}: изменилась единица или тип в БД; нужен явный повторный импорт")
            old_field, new_field = columns[cid].metadata.get("geochem_field"), candidate.metadata.get("geochem_field")
            if old_field and new_field and any(old_field.get(key) != new_field.get(key) for key in ("group", "label", "dtype")):
                errors.append(f"{columns[cid].name}: изменилось определение показателя БД; нужен явный повторный импорт")
            mapping[cid] = candidate.id
    if any(cid not in mapping for cid in keys):
        errors.append("Ключевые поля исчезли из источника")
    schema = list(incoming_fields)
    changes = {"added": sorted(set(schema) - set(baseline["schema"])),
               "removed": sorted(set(baseline["schema"]) - set(schema))}
    remote_rows = {}
    incoming_descriptor = descriptor(incoming.source)
    for row in incoming.rows:
        values = {cid: row.original.get(sid, row.values[sid]) for cid, sid in mapping.items()}
        try:
            if incoming.source["type"] == "sqlite":
                key = encoded_key([parse_number(values[cid]) if columns[cid].kind == "number" else values[cid] for cid in keys])
            else:
                key = source_key(incoming, row, (), {})
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"Ключ строки источника недоступен: {exc}")
            continue
        if key in remote_rows:
            errors.append("Ключ не уникален в новом снимке. Обновление без потери записей невозможно")
        remote_rows[key] = {"values": values, "source": row_provenance(
            incoming.source, row.source, source_descriptor=incoming_descriptor)}
    cells, new_rows, missing = [], [], []
    local_rows = project.row_map()
    base_frame = pd.DataFrame(list(baseline["rows"].items()), columns=["key", "base_record"], dtype=object)
    remote_frame = pd.DataFrame(list(remote_rows.items()), columns=["key", "remote_record"], dtype=object)
    joined = base_frame.merge(remote_frame, on="key", how="left", sort=False, validate="one_to_one", indicator=True)
    comparisons = []
    for key, base, remote, presence in joined.itertuples(index=False, name=None):
        local = local_rows.get(base["sample_id"])
        if local is None:
            errors.append("Проба исходного снимка удалена из локального проекта")
            continue
        if presence == "left_only":
            if base.get("present", True):
                missing.append({"id": "removed:" + local.id, "sample_id": local.id, "key": key, "action": "unresolved"})
            continue
        for cid, raw in remote["values"].items():
            column = columns[cid]
            valid = True
            try:
                new = parsed(raw, column.kind)
            except (ValueError, TypeError, OverflowError):
                valid, new = False, raw
            measurement = project.science.current(local.id, cid)
            comparisons.append((local.id + ":" + cid, local.id, cid, key, base["values"].get(cid),
                local.values[cid], raw, comparable(base["values"].get(cid), column.kind), new, valid,
                measurement.id if measurement else None, base["references"].get(cid), cid in base["values"]))
    frame = pd.DataFrame(comparisons, columns=["id", "sample_id", "property_id", "key", "base", "local", "source",
        "old", "new", "valid", "local_reference", "base_reference", "base_present"], dtype=object)
    local_equal = frame["local"].eq(frame["old"]) | (frame["local"].isna() & frame["old"].isna())
    remote_equal = frame["new"].eq(frame["old"]) | (frame["new"].isna() & frame["old"].isna())
    converged = frame["new"].eq(frame["local"]) | (frame["new"].isna() & frame["local"].isna())
    local_changed = ~local_equal | frame["local_reference"].fillna("").ne(frame["base_reference"].fillna(""))
    remote_changed = ~remote_equal | ~frame["base_present"].astype(bool)
    frame["status"], frame["action"] = "local_only", "keep"
    frame.loc[remote_changed, ["status", "action"]] = ["source_changed", "source"]
    frame.loc[remote_changed & local_changed, ["status", "action"]] = ["conflict", "unresolved"]
    frame.loc[remote_changed & local_changed & converged, ["status", "action"]] = ["converged", "keep"]
    invalid = ~frame["valid"].astype(bool)
    frame.loc[invalid, "status"] = "invalid_source"
    frame.loc[invalid & remote_changed, "action"] = "unresolved"
    frame.loc[invalid & ~remote_changed, "action"] = "keep"
    frame["remote_changed"] = remote_changed
    cells = frame.loc[local_changed | remote_changed, ["id", "sample_id", "property_id", "key", "base", "local",
        "source", "status", "action", "remote_changed"]].to_dict("records")
    for key, remote in remote_rows.items():
        if key not in baseline["rows"]:
            invalid = []
            for cid, raw in remote["values"].items():
                try:
                    parsed(raw, columns[cid].kind)
                except (ValueError, TypeError, OverflowError):
                    invalid.append(columns[cid].name)
            new_rows.append({"id": "new:" + key, "key": key, "action": "unresolved" if invalid else "add",
                             "invalid": invalid, **remote})
    baseline["incoming_rows"] = remote_rows
    return RefreshPlan(before, digest(incoming), baseline, mapping, cells, new_rows, missing, changes, errors, timestamp())


def _validate_decisions(plan, decisions):
    """Validate review choices before copying or changing any project data."""
    entries = plan.cells + plan.new_rows + plan.missing_rows
    if set(decisions) - {entry["id"] for entry in entries}:
        raise ValueError("Решение относится к неизвестному изменению")
    groups = (("cell", plan.cells), ("new", plan.new_rows), ("missing", plan.missing_rows))
    for kind, group in groups:
        for entry in group:
            action = decisions.get(entry["id"], entry["action"])
            if kind == "missing" or entry.get("status") == "invalid_source":
                allowed = ("keep",)
            elif kind == "new":
                allowed = ("skip",) if entry.get("invalid") else ("add", "skip")
            else:
                allowed = ("keep", "source")
            if action not in allowed:
                raise ValueError("Разрешите конфликты и подтвердите сохранение исчезнувших записей")


def prepare_refresh(project, incoming, plan, decisions=None, *, accept_schema_changes=False):
    if digest(project) != plan.project_digest or digest(incoming) != plan.incoming_digest:
        raise ValueError("Данные изменились после сравнения. Обновите предпросмотр")
    if plan.errors:
        raise ValueError("Исправьте ошибки сравнения до обновления")
    if any(plan.schema_changes.values()) and not accept_schema_changes:
        raise ValueError("Подтвердите изменение набора полей источника")
    decisions = dict(decisions or {})
    _validate_decisions(plan, decisions)
    result = copy_project(project)
    result.sync_measurements()
    baseline = deepcopy(plan.baseline)
    remote_rows = baseline.pop("incoming_rows")
    run_id = new_id()
    source_descriptor = descriptor(project.source)
    protocol = {"id": run_id, "at": timestamp(), "checked_at": plan.checked_at, "source": source_descriptor,
                "changes": [], "schema_changes": deepcopy(plan.schema_changes)}
    local_rows = result.row_map()
    columns = {c.id: c for c in result.columns}
    events = {}
    working_selection_changed = False
    for entry in plan.cells:
        action = decisions.get(entry["id"], entry["action"])
        protocol["changes"].append({**entry, "action": action})
        if not entry["remote_changed"] and action == "keep":
            continue
        sid, cid, key = entry["sample_id"], entry["property_id"], entry["key"]
        if sid not in events:
            source = SourceRecord(new_id(), {**deepcopy(source_descriptor), **remote_rows[key]["source"],
                                            "snapshot_at": plan.checked_at}, deepcopy(remote_rows[key]["values"]))
            result.science.sources[source.id] = source
            analysis = Analysis(sid, source_id=source.id, metadata={"database_refresh": run_id})
            result.science.analyses[analysis.id] = analysis
            events[sid] = analysis
        analysis = events[sid]
        try:
            value = parse_result(entry["source"], columns[cid].kind)
        except (ValueError, TypeError, OverflowError):
            value = {"raw_value": entry["source"], "value": None, "qualifier": "unparsed"}
        measurement = Measurement(sid, cid, analysis.id, source_id=analysis.source_id,
                                  unit=columns[cid].unit, basis=columns[cid].metadata.get("basis", ""), **value)
        result.science.measurements[measurement.id] = measurement
        baseline["rows"][key]["references"][cid] = measurement.id
        if action == "source":
            result.science.select(sid, cid, measurement.id)
            local_rows[sid].values[cid] = measurement.table_value()
            # Equal numbers can still have different QC and analytical context.
            working_selection_changed = True
    for entry in plan.new_rows:
        action = decisions.get(entry["id"], entry["action"])
        protocol["changes"].append({**entry, "action": action})
        if action == "skip":
            continue
        values = {c.id: None for c in result.columns}
        try:
            values.update({cid: parsed(raw, columns[cid].kind) for cid, raw in entry["values"].items()})
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("Новая строка содержит некорректное числовое значение; пропустите её или исправьте источник") from exc
        row = Row(values, source=deepcopy(entry["source"]), original=deepcopy(entry["values"]))
        result.rows.append(row)
    result.sync_measurements()
    # Establish references for new rows once; preserve UUIDs for all existing samples.
    new_ids = {row.id for row in result.rows} - set(local_rows)
    for row in result.rows:
        if row.id not in new_ids:
            continue
        key = source_key(result, row, baseline["keys"], row.original)
        baseline["rows"][key] = {"sample_id": row.id, "values": deepcopy(row.original), "references": {
            cid: result.science.current(row.id, cid).id for cid in row.original if result.science.current(row.id, cid)}, "present": True}
    for key, base in baseline["rows"].items():
        if key in remote_rows:
            base["values"].update(deepcopy(remote_rows[key]["values"]))
            base["present"] = True
        else:
            base["present"] = False
    for entry in plan.missing_rows:
        protocol["changes"].append({**entry, "action": decisions.get(entry["id"], entry["action"])})
    baseline.update(schema=[field_key(c) for c in incoming.columns], checked_at=plan.checked_at,
                    applied_at=timestamp(), source=deepcopy(source_descriptor))
    result.view["database_snapshot"] = baseline
    original_rows = project.row_map()
    working_changed = working_selection_changed or len(result.rows) != len(project.rows) or any(
        row.values != original_rows[row.id].values for row in result.rows if row.id in original_rows)
    protocol["working_changed"] = working_changed
    if working_changed:
        result.refresh()
    result.validate()
    return result, protocol


def snapshot_status(project, sample_ids):
    snapshot = project.view.get("database_snapshot")
    if not snapshot:
        return None
    sample_ids = set(sample_ids)
    return {"checked_at": snapshot["checked_at"], "applied_at": snapshot["applied_at"],
            "source": descriptor(snapshot["source"]), "records": [
                {"sample_id": row["sample_id"], "present": row["present"]}
                for row in snapshot["rows"].values() if row["sample_id"] in sample_ids]}
