"""Preview and transactional append of laboratory files; no Qt or source writes."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import json

import pandas as pd

from autopyro.model.core import Column, Row, new_id, timestamp
from autopyro.model.measurements import Analysis, Measurement, SourceRecord, derived, parse_result
from autopyro.model.snapshots import copy_project


ANALYSIS_FIELDS = ("method", "laboratory", "instrument", "heating_program", "preparation", "fraction", "date")


class _ProjectEncoder(json.JSONEncoder):
    def default(self, value):
        if is_dataclass(value) and not isinstance(value, type):
            # Encode declared fields directly, without a second project-sized copy.
            return {field.name: getattr(value, field.name) for field in fields(value)}
        return super().default(value)


def digest(project):
    """Hash the same sorted JSON as asdict(project), without copying every record."""
    return hashlib.sha256(json.dumps(project, cls=_ProjectEncoder, ensure_ascii=False,
                                     sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def raw_value(row, column):
    raw = row.original.get(column.id, row.values[column.id])
    try:
        if Measurement("", "", "", **parse_result(raw, column.kind)).table_value() == row.values[column.id]:
            return raw
    except (ValueError, TypeError):
        pass
    return row.values[column.id]


def fingerprint(source):
    # Without a verified content hash, do not claim that two imports are identical.
    if not source.get("sha256") or source.get("row") is None:
        return None
    return (source["sha256"], source.get("sheet"), source["row"])


def suggest_mapping(project, incoming):
    def descriptors(columns, *, include_import_names=False):
        records = []
        for column in columns:
            names = {column.name}
            if include_import_names and column.metadata.get("append_source_name"):
                names.add(column.metadata["append_source_name"])
            records.extend((column.id, name, column.unit, column.kind, column.metadata.get("basis", ""))
                           for name in sorted(names))
        return pd.DataFrame(
            records,
            columns=["id", "name", "unit", "kind", "basis"], dtype=object,
        )
    targets = descriptors([c for c in project.columns if not derived(c)],
                          include_import_names=True).rename(columns={"id": "target"})
    matches = descriptors(incoming.columns).merge(targets, on=["name", "unit", "kind", "basis"], how="left", sort=False)
    candidates = matches.groupby("id", sort=False)["target"].agg(list)
    return {cid: values[0] if len(values) == 1 and pd.notna(values[0]) else "new"
            for cid, values in candidates.items()}



@dataclass
class ImportPlan:
    project_digest: str
    incoming_digest: str
    mapping: dict
    keys: list[str]
    columns: list[Column]
    rows: list[dict]
    conflicts: list[dict]
    errors: list[str]
    bindings: dict = field(default_factory=dict)
    defaults: dict = field(default_factory=dict)


def preview_import(project, incoming, mapping, keys=(), *, bindings=None, defaults=None):
    original_digest = digest(project)
    project = copy_project(project)
    project.sync_measurements()
    bindings, defaults = dict(bindings or {}), dict(defaults or {})
    source_columns = {c.id: c for c in incoming.columns}
    targets = {c.id: c for c in project.columns}
    if set(mapping) != set(source_columns) or not any(mapping.values()):
        raise ValueError("Сопоставьте хотя бы одну колонку; каждой исходной колонке нужно назначение")
    if set(bindings) - set(ANALYSIS_FIELDS) or set(defaults) - set(ANALYSIS_FIELDS):
        raise ValueError("Неизвестное поле аналитического контекста")
    if set(bindings.values()) - set(source_columns):
        raise ValueError("Колонка аналитического контекста отсутствует")
    resolved, new_columns, used, names = {}, [], set(), {c.name for c in project.columns}
    for sid, tid in mapping.items():
        if tid is None:
            continue
        column = source_columns[sid]
        if derived(column):
            raise ValueError("Этот импорт принимает исходные результаты, а не вычисленные колонки")
        if tid == "new":
            name, suffix = column.name, 2
            while name in names:
                name = f"{column.name} ({suffix})"
                suffix += 1
            names.add(name)
            target = Column(name, column.kind, unit=column.unit, metadata=deepcopy(column.metadata))
            # Keep the input name when a collision adds a suffix to the display name.
            target.metadata["append_source_name"] = column.name
            new_columns.append(target)
        else:
            target = targets.get(tid)
            if target is None or derived(target):
                raise ValueError("Нельзя импортировать в отсутствующую или вычисленную колонку")
            if target.unit != column.unit or target.metadata.get("basis", "") != column.metadata.get("basis", ""):
                raise ValueError(f"{column.name}: единица или основание отличаются; нужен отдельный показатель или явный перевод")
        if target.id in used:
            raise ValueError("Две исходные колонки нельзя назначить одному показателю")
        used.add(target.id)
        resolved[sid] = target.id
    if len(set(keys)) != len(keys) or set(keys) - set(resolved):
        raise ValueError("Ключи должны быть различными импортируемыми колонками")
    if any(resolved[key] not in targets for key in keys):
        raise ValueError("Ключ сопоставления должен ссылаться на существующий показатель проекта")
    all_targets = {**targets, **{c.id: c for c in new_columns}}

    def key_for(values):
        parts = [values.get(resolved[cid]) for cid in keys]
        if not keys or any(v is None or v == "" for v in parts):
            return None
        return tuple((type(value).__name__, value) for value in parts)

    source_index = defaultdict(set)
    project_rows = project.row_map()
    target_keys = pd.DataFrame(
        [(key_for(row.values), row.id) for row in project.rows], columns=["key", "candidate"], dtype=object,
    ).dropna(subset=["key"])
    target_keys["target_order"] = range(len(target_keys))
    # One physical source row can have several result records, but one source sample candidate.
    for measurement in project.science.measurements.values():
        source = project.science.sources.get(measurement.source_id)
        token = fingerprint(source.provenance) if source else None
        if token and measurement.sample_id in project_rows:
            source_index[token].add(measurement.sample_id)
    source_results = defaultdict(list)
    for measurement in project.science.measurements.values():
        source = project.science.sources.get(measurement.source_id)
        token = fingerprint(source.provenance) if source else None
        if token:
            source_results[(measurement.sample_id, measurement.property_id, token)].append((measurement, source))

    rows, conflicts, errors = [], [], []
    prepared = []
    for row in incoming.rows:
        originals, values = {}, {}
        for sid, tid in resolved.items():
            raw = raw_value(row, source_columns[sid])
            originals[tid] = raw
            try:
                values[tid] = Measurement("", "", "", **parse_result(raw, all_targets[tid].kind)).table_value()
            except (ValueError, TypeError, OverflowError) as exc:
                errors.append(f"Строка {row.source.get('row', row.id)}, {source_columns[sid].name}: {exc}")
        prepared.append((row, originals, values, key_for(values)))
    incoming_keys = pd.DataFrame(
        [(row.id, key) for row, _, _, key in prepared], columns=["incoming_id", "key"], dtype=object,
    )
    # Null keys must never match each other; typed keys preserve 1 versus "1".
    matches = incoming_keys.dropna(subset=["key"]).merge(target_keys, on="key", how="inner", sort=False)
    matches = matches.sort_values("target_order", kind="stable")
    candidates_by_row = matches.groupby("incoming_id", sort=False)["candidate"].agg(list).to_dict()
    for row, originals, values, key in prepared:
        token = fingerprint(row.source)
        candidates = candidates_by_row.get(row.id, []) if key is not None else sorted(source_index.get(token, ()))
        duplicate_candidates, duplicate_properties = [], {}
        for sample_id in candidates:
            duplicate_properties[sample_id] = [tid for tid, raw in originals.items() if token and any(
                source.original.get(tid) == raw and tid in source.original
                for _, source in source_results.get((sample_id, tid, token), ()))]
            if len(duplicate_properties[sample_id]) == len(originals):
                duplicate_candidates.append(sample_id)
        if len(duplicate_candidates) == 1:
            status, action, target = "duplicate", "skip", duplicate_candidates[0]
        elif len(candidates) == 1:
            status, action, target = "matched", "append", candidates[0]
        elif len(candidates) > 1:
            status, action, target = "ambiguous", "unresolved", None
        else:
            status, action, target = "new", "new", None
        context = {name: str(defaults.get(name, "")).strip() for name in ANALYSIS_FIELDS}
        for name, sid in bindings.items():
            value = row.values[sid]
            context[name] = "" if value is None else str(value).strip()
        rows.append({"incoming_id": row.id, "source_row": row.source.get("row", row.id),
                     "key": key, "candidates": candidates, "status": status, "action": action,
                     "target_id": target, "values": values, "originals": originals, "context": context})
        rows[-1]["duplicate_properties"] = duplicate_properties
        for candidate in candidates:
            current = project_rows[candidate]
            for tid, value in values.items():
                if current.values.get(tid) != value and tid not in duplicate_properties.get(candidate, []):
                    conflicts.append({"incoming_id": row.id, "sample_id": candidate, "property_id": tid,
                                      "property": all_targets[tid].name, "current": current.values.get(tid), "incoming": value})
    repeated = incoming_keys["key"].notna() & incoming_keys["key"].duplicated(keep=False)
    repeated_ids = set(incoming_keys.loc[repeated, "incoming_id"])
    earlier_new = defaultdict(list)
    for item in rows:
        if item["status"] == "new" and item["incoming_id"] in repeated_ids:
            item["status"], item["action"] = "input_repeat", "unresolved"
            item["new_candidates"] = list(earlier_new[item["key"]])
            earlier_new[item["key"]].append(item["incoming_id"])
    return ImportPlan(original_digest, digest(incoming), resolved, list(keys), new_columns, rows, conflicts, errors,
                      bindings, defaults)


def prepare_import(project, incoming, plan, decisions=None, *, working_policy="keep", template_view=None, template_ref=None):
    """Return a validated private project and protocol. Neither input is modified."""
    if digest(project) != plan.project_digest or digest(incoming) != plan.incoming_digest:
        raise ValueError("Данные изменились после предпросмотра. Рассчитайте его заново")
    if plan.errors:
        raise ValueError("Исправьте ошибки предпросмотра до импорта")
    if working_policy not in ("keep", "fill_missing", "replace"):
        raise ValueError("Неизвестная политика рабочих результатов")
    decisions = decisions or {}
    if set(decisions) - {r["incoming_id"] for r in plan.rows}:
        raise ValueError("Решение относится к отсутствующей исходной строке")
    choices = []
    actions = {}
    for item in plan.rows:
        choice = {"action": item["action"], "target_id": item["target_id"], **decisions.get(item["incoming_id"], {})}
        if choice["action"] not in ("skip", "new", "append", "append_new"):
            raise ValueError(f"Строка {item['source_row']}: разрешите неоднозначные совпадения — выберите пробу, новую запись или пропуск")
        if choice["action"] == "append" and choice["target_id"] not in item["candidates"]:
            raise ValueError("Выберите пробу из проверенных кандидатов")
        if item["status"] == "duplicate" and choice["action"] != "skip":
            raise ValueError("Та же исходная запись уже загружена. Изменённые измерения не восстанавливаются повторным импортом")
        if choice["action"] == "append_new":
            target = choice["target_id"]
            if target not in item.get("new_candidates", ()) or actions.get(target) != "new":
                raise ValueError(f"Строка {item['source_row']}: сначала выберите создание пробы в целевой строке файла")
        choices.append(choice)
        actions[item["incoming_id"]] = choice["action"]
    result = copy_project(project)
    from .import_templates import merge_views, reference
    merge_views(result.view, incoming.view)
    merge_views(result.view, template_view or {}, activate=True)
    if template_ref:
        definition = result.view.get("import_template_revisions", {}).get(template_ref["id"], {}).get(str(template_ref["revision"]))
        if definition is None or reference(definition) != template_ref:
            raise ValueError("Определение использованного шаблона не соответствует ссылке")
    result.sync_measurements()
    run_id = new_id()
    active = [i for i, choice in enumerate(choices) if choice["action"] != "skip"]
    used_targets = {tid for i in active for tid in plan.rows[i]["values"]}
    columns = [deepcopy(c) for c in plan.columns if c.id in used_targets]
    result.columns.extend(columns)
    for row in result.rows:
        row.values.update({c.id: None for c in columns})
    new_rows = {}
    incoming_rows = incoming.row_map()
    for item, choice in zip(plan.rows, choices, strict=True):
        if choice["action"] == "new":
            values = {c.id: None for c in result.columns}
            values.update(item["values"])
            new_rows[item["incoming_id"]] = Row(values, source=deepcopy(incoming_rows[item["incoming_id"]].source),
                                                 original=deepcopy(item["originals"]))
    result.rows.extend(new_rows.values())
    # Backfill only the view; absent properties must not fabricate missing analyses.
    result.sync_measurements()
    target_columns = {c.id: c for c in result.columns}
    target_rows = result.row_map()
    primary = {a.sample_id: a for a in result.science.analyses.values() if a.metadata.get("origin") == "table_row"}
    sample_measurements = defaultdict(list)
    for measurement in result.science.measurements.values():
        sample_measurements[measurement.sample_id].append(measurement.id)
    protocol = {"id": run_id, "at": timestamp(), "source": deepcopy(incoming.source),
                "mapping": dict(plan.mapping), "keys": list(plan.keys), "bindings": plan.bindings,
                "defaults": plan.defaults, "working_policy": working_policy, "rows": [],
                "conflicts": deepcopy(plan.conflicts),
                "new_samples": 0, "analyses": 0, "measurements": 0, "skipped": 0}
    if template_ref:
        protocol["import_template"] = deepcopy(template_ref)
    for item, choice in zip(plan.rows, choices, strict=True):
        entry = {"incoming_id": item["incoming_id"], "source_row": item["source_row"],
                 "status": item["status"], **choice}
        protocol["rows"].append(entry)
        if choice["action"] == "skip":
            protocol["skipped"] += 1
            continue
        incoming_row = incoming_rows[item["incoming_id"]]
        if choice["action"] == "new":
            target = new_rows[item["incoming_id"]]
            analysis = primary[target.id]
            for name, value in item["context"].items():
                setattr(analysis, name, value)
            analysis.metadata["import_run"] = run_id
            measurement_ids = sample_measurements[target.id]
            protocol["new_samples"] += 1
        else:
            targets = new_rows if choice["action"] == "append_new" else target_rows
            target = targets[choice["target_id"]]
            new_properties = {tid: raw for tid, raw in item["originals"].items()
                              if tid not in item["duplicate_properties"].get(target.id, [])}
            if not new_properties:
                protocol["skipped"] += 1
                entry["action"] = "skip"
                continue
            source = SourceRecord(new_id(), deepcopy(incoming_row.source), deepcopy(item["originals"]))
            result.science.sources[source.id] = source
            analysis = Analysis(target.id, source_id=source.id, **item["context"], metadata={"import_run": run_id})
            result.science.analyses[analysis.id] = analysis
            measurement_ids = []
            for tid, raw in new_properties.items():
                column = target_columns[tid]
                measurement = Measurement(target.id, tid, analysis.id, source_id=source.id, unit=column.unit,
                                          basis=column.metadata.get("basis", ""), **parse_result(raw, column.kind))
                result.science.measurements[measurement.id] = measurement
                measurement_ids.append(measurement.id)
                current = result.science.current(target.id, tid)
                if current is None or working_policy == "replace" or working_policy == "fill_missing" and current.qualifier == "missing":
                    result.science.select(target.id, tid, measurement.id)
                    target.values[tid] = measurement.table_value()
        entry.update(target_id=target.id, analysis_id=analysis.id, measurement_ids=measurement_ids)
        protocol["analyses"] += 1
        protocol["measurements"] += len(measurement_ids)
    if active:
        result.refresh()
    result.validate()
    return result, protocol
