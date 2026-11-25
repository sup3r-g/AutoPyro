"""SELECT-only catalog and focused snapshots of the existing geochem schema."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pandas as pd

from autopyro.model.core import Column, Project, Row, timestamp
from autopyro.model.measurements import derived
from autopyro.model.snapshots import copy_project
from autopyro.analysis.nonpyro import (
    METHODS, field_matches, preview_profile_coverage, profile_inputs_complete,
    source_identity, suggest_bindings, variant_tag,
)
from .connection import connect, local_parameters


def snapshot_profile_coverage(project, method_ids, *, group='', variant='', combined=False):
    """Assess the working values of the snapshot on each selected measurement basis."""
    row_ids = [row.id for row in project.rows]
    tags = {column.id: variant_tag(source_identity(column)[1]) for column in project.columns}
    variants = ('h', 'a', 'ch', 'ca', 'vol', 'mol', 'un', '') if combined else (variant,)
    coverage = []
    for method in dict.fromkeys(method_ids):
        required = {name for name, role in METHODS[method]['roles'].items() if not role['optional']}
        matched = False
        for basis in variants:
            bindings = suggest_bindings(project, method, group, basis)
            if combined and basis not in {tags[item['column_id']] for item in bindings.values()}:
                continue
            matched = True
            complete, effective, error = 0, None, ''
            if profile_inputs_complete(method, bindings):
                bound = {name: item['column_id'] for name, item in bindings.items()}
                complete = sum(profile_inputs_complete(method, {
                    name for name, cid in bound.items() if row.values.get(cid) not in (None, '')
                }) for row in project.rows)
                # Isotope readiness is conditional on the analyst confirming these prerequisites.
                config = {'version': 1, 'method': method, 'variant': basis, 'inputs': bindings,
                          'isotope_reference': 'VPDB', 'material': 'oil_C15plus'}
                try:
                    effective = preview_profile_coverage(project, row_ids, config)
                except ValueError as exc:
                    error = str(exc)
            coverage.append((method, basis, (len(required & bindings.keys()), len(required)),
                             complete, effective, error))
        if not matched:
            coverage.append((method, '', None, 0, None, ''))
    return coverage


def append_geochemistry_snapshot(project, incoming):
    """Append database rows while preserving each source analysis and repeat."""
    result, snapshot = copy_project(project), copy_project(incoming)
    snapshot.sync_measurements()
    targets = {}

    def column_key(column):
        field = column.metadata.get("geochem_field", {})
        return column.name, column.kind, column.unit, field.get("group"), field.get("label")

    for column in result.columns:
        if not derived(column):
            targets.setdefault(column_key(column), []).append(column)
    mapping = {}
    existing_columns = {column.id for column in result.columns}
    for column in snapshot.columns:
        candidates = targets.get(column_key(column), ())
        if len(candidates) > 1:
            raise ValueError(f"Неоднозначная колонка проекта: {column.name}")
        if candidates:
            mapping[column.id] = candidates[0].id
        else:
            if column.id in existing_columns:
                raise ValueError(f"Совпали идентификаторы колонок: {column.name}")
            result.columns.append(column)
            mapping[column.id] = column.id
    if set(result.row_map()) & snapshot.row_map().keys():
        raise ValueError("Этот снимок уже содержит строки с теми же идентификаторами")
    for row in snapshot.rows:
        row.values = {mapping[key]: value for key, value in row.values.items()}
        row.original = {mapping[key]: value for key, value in row.original.items()}
        if snapshot.source.get("endpoint"):
            row.source["geochemistry_endpoint"] = deepcopy(snapshot.source["endpoint"])
    result.rows.extend(snapshot.rows)
    for collection in ("samples", "sources", "analyses", "measurements", "working"):
        target = getattr(result.science, collection)
        records = getattr(snapshot.science, collection)
        if target.keys() & records.keys():
            raise ValueError("Снимок содержит совпадающие идентификаторы научных записей")
        if collection == "measurements":
            for record in records.values():
                record.property_id = mapping[record.property_id]
        elif collection == "working":
            records = {row_id: {mapping[key]: value for key, value in choices.items()}
                       for row_id, choices in records.items()}
        target.update(records)
    for key, definition in snapshot.science.properties.items():
        result.science.properties.setdefault(mapping[key], replace(definition, id=mapping[key]))
    result.refresh(sync_measurements=False)
    return result


def inventory(parameters):
    with connect(parameters) as connection:
        connection.execute("SET LOCAL statement_timeout='60s'")
        fields = connection.execute("""SELECT f.source_field_id,f.group_name,f.source_label,f.unit_label,f.source_dtype,
            count(r.analysis_run_id)::integer AS populated
            FROM geochem.source_field f LEFT JOIN geochem.analysis_result r USING(source_field_id)
            WHERE f.target_table='analysis_result' AND f.group_name NOT LIKE 'Py%%'
            AND NOT (f.group_name='Isotopes' AND f.source_label LIKE 'd13C Pyrol%%')
            GROUP BY f.source_field_id ORDER BY f.group_name,f.source_label""").fetchall()
        batches = connection.execute("""SELECT import_batch_id,source_name,source_sha256,imported_at::text
            FROM geochem.import_batch ORDER BY import_batch_id""").fetchall()
        groups = connection.execute("""SELECT group_name,count(*)::integer AS runs,count(DISTINCT source_record_id)::integer AS records,
            max(repeat_number) AS max_repeat FROM geochem.analysis_run WHERE group_name NOT LIKE 'Py%%'
            GROUP BY group_name ORDER BY group_name""").fetchall()
        read_only = connection.execute("SHOW transaction_read_only").fetchone()[
            "transaction_read_only"
        ]
    return {
        "created_at": timestamp(),
        "fields": fields,
        "groups": groups,
        "batches": batches,
        "read_only": read_only,
    }


def profile_fields(catalog, method_id, group="", variant=""):
    definition = METHODS[method_id]
    required = []
    allowed_groups = {group} if group else set(definition["groups"])
    if not definition["same_group"]:
        allowed_groups = set(definition["groups"])
    metadata = {
        "Lab",
        "Lab#",
        "Date",
        "Method",
        "Eqp",
        "Quality",
        "Prep",
        "Sam type",
        "Lab comment",
    }
    for field in catalog["fields"]:
        if field["group_name"] not in allowed_groups:
            continue
        is_metadata = field["source_label"] in metadata
        matches = any(
            field_matches(field["group_name"], field["source_label"], role)
            for role in definition["roles"].values()
        )
        if is_metadata or (
            matches
            and (
                not variant
                or not variant_tag(field["source_label"])
                or variant_tag(field["source_label"]) == variant
            )
        ):
            required.append(field)
    return required


def read_profile(
    parameters,
    method_id,
    *,
    group="",
    variant="",
    batch_id=None,
    well_name="",
    repeat_number=1,
    limit=10000,
    method_ids=None,
    include_repeats=False,
):
    """One row per source_record, exact run number; never join on sample names."""
    if (
        method_id not in METHODS
        or not 1 <= int(limit) <= 100000
        or int(repeat_number) < 1
    ):
        raise ValueError("Некорректная методика, лимит или номер повтора")
    methods = list(dict.fromkeys(method_ids or [method_id]))
    if not methods or any(key not in METHODS for key in methods):
        raise ValueError('Неизвестный набор геохимических показателей')
    with connect(parameters) as connection:
        connection.execute("SET LOCAL statement_timeout='60s'")
        fields = connection.execute("""SELECT source_field_id,group_name,source_label,unit_label,source_dtype
            FROM geochem.source_field WHERE target_table='analysis_result' ORDER BY source_field_id""").fetchall()
        selected_ids = {field['source_field_id'] for key in methods
                        for field in profile_fields({'fields': fields}, key, group, variant)}
        selected = [field for field in fields if field['source_field_id'] in selected_ids]
        selected = [f for f in selected if not f["group_name"].startswith("Py")]
        if not selected:
            raise ValueError("Нет полей выбранного профиля в каталоге БД")
        groups = sorted({field["group_name"] for field in selected})
        records = connection.execute(
            """SELECT sr.source_record_id,sr.sample_id,sr.import_batch_id,sr.row_number,
            w.well_name,s.name AS sample_name,s.type AS sample_type,s.category,s.top_md,s.base_md,s.md_datum,
            s.preparation,s.rock_type,s.liquid_type,s.gas_type
            FROM geochem.source_record sr JOIN geochem.sample s USING(sample_id)
            LEFT JOIN geochem.well w USING(well_id)
            WHERE (%s::bigint IS NULL OR sr.import_batch_id=%s) AND (%s='' OR w.well_name=%s)
            AND EXISTS (SELECT 1 FROM geochem.analysis_run a WHERE a.source_record_id=sr.source_record_id
                AND a.group_name=ANY(%s) AND (%s OR a.repeat_number=%s))
            ORDER BY sr.source_record_id LIMIT %s""",
            (
                batch_id,
                batch_id,
                well_name,
                well_name,
                groups,
                bool(include_repeats),
                repeat_number,
                limit + 1,
            ),
        ).fetchall()
        if len(records) > limit:
            raise ValueError(
                f"Выборка превышает {limit} записей; уточните скважину или загрузку. Усечение не выполнено."
            )
        record_ids = [r["source_record_id"] for r in records]
        results = connection.execute(
            """SELECT a.source_record_id,a.analysis_run_id,a.group_name,a.repeat_number,
            r.source_field_id,r.value_float,r.value_int,r.value_text
            FROM geochem.analysis_run a JOIN geochem.analysis_result r USING(analysis_run_id,group_name)
            WHERE a.source_record_id=ANY(%s::bigint[]) AND (%s OR a.repeat_number=%s) AND r.source_field_id=ANY(%s::bigint[])
            ORDER BY a.source_record_id,a.analysis_run_id,r.source_field_id""",
            (record_ids, bool(include_repeats), repeat_number, [f["source_field_id"] for f in selected]),
        ).fetchall()
        batches = connection.execute(
            """SELECT import_batch_id,source_name,source_sha256 FROM geochem.import_batch
            WHERE import_batch_id=ANY(%s::bigint[]) ORDER BY import_batch_id""",
            (sorted({r["import_batch_id"] for r in records}),),
        ).fetchall()
        endpoint = {
            "host": connection.info.host,
            "port": connection.info.port,
            "database": connection.info.dbname,
        }
        read_only = connection.execute("SHOW transaction_read_only").fetchone()[
            "transaction_read_only"
        ]
    builder = project_with_repeats if include_repeats or method_ids is not None else project_from_records
    return builder(
        records,
        results,
        selected,
        source={
            "type": "postgresql_nonpyro",
            "read_only": read_only == "on",
            "endpoint": endpoint,
            "created_at": timestamp(),
            "groups": groups,
            "variant": variant,
            "repeat_number": repeat_number,
            "batch_id": batch_id,
            "well_name": well_name,
            "batches": batches,
            "method_profile": method_id,
            "group": group,
            **({'method_profiles': methods, 'include_repeats': bool(include_repeats)}
               if method_ids is not None or include_repeats else {}),
        },
    )


def read_profiles(parameters, method_ids, **options):
    methods = list(dict.fromkeys(method_ids))
    if not methods:
        raise ValueError('Выберите хотя бы один набор показателей')
    return read_profile(parameters, methods[0], method_ids=methods, **options)


def project_with_repeats(records, results, fields, *, source):
    """Keep all raw runs; only the requested repeat becomes the table projection."""
    from autopyro.model.measurements import Analysis, Measurement, SourceRecord, migrated_id, parse_result

    repeat = int(source.get('repeat_number') or 1)
    working = [item for item in results if item['repeat_number'] == repeat]
    project = project_from_records(records, working, fields, source=source)
    project.sync_measurements()
    field_columns = {column.metadata['geochem_field']['id']: column for column in project.columns
                     if column.metadata.get('geochem_field')}
    field_ids = {column.id for column in field_columns.values()}
    for mid in [mid for mid, measurement in project.science.measurements.items() if measurement.property_id in field_ids]:
        del project.science.measurements[mid]
    rows = {row.source['source_record_id']: row for row in project.rows}
    for row in rows.values():
        row.source['geochemistry_endpoint'] = deepcopy(source.get('endpoint', {}))
        for cid in field_ids:
            project.science.working[row.id].pop(cid, None)
    seen = set()
    for item in results:
        key = (item['source_record_id'], item['analysis_run_id'], item['group_name'], item['source_field_id'])
        if key in seen or item['source_record_id'] not in rows or item['source_field_id'] not in field_columns:
            raise ValueError('Повторяющийся или неизвестный результат геохимического анализа')
        seen.add(key)
        row = rows[item['source_record_id']]
        column = field_columns[item['source_field_id']]
        if column.metadata['geochem_field']['group'] != item['group_name']:
            raise ValueError('Группа результата не соответствует показателю')
        aid = migrated_id(project.id, 'geochemistry_analysis', str(item['source_record_id']),
                          str(item['analysis_run_id']), item['group_name'])
        sid = migrated_id(project.id, 'geochemistry_source', aid)
        if aid not in project.science.analyses:
            provenance = {**deepcopy(row.source), 'analysis_run_id': item['analysis_run_id'],
                          'group': item['group_name'], 'repeat_number': item['repeat_number'],
                          'endpoint': deepcopy(source.get('endpoint', {}))}
            project.science.sources[sid] = SourceRecord(sid, provenance, {})
            project.science.analyses[aid] = Analysis(row.id, source_id=sid, id=aid, method=item['group_name'],
                metadata={'analysis_run_id': item['analysis_run_id'], 'repeat_number': item['repeat_number'],
                          'source_record_id': item['source_record_id']})
        elif project.science.analyses[aid].metadata['repeat_number'] != item['repeat_number']:
            raise ValueError('Для одного анализа указаны разные номера повтора')
        raw = next((item[key] for key in ('value_float', 'value_int', 'value_text') if item.get(key) is not None), None)
        measurement = Measurement(row.id, column.id, aid, unit=column.unit, source_id=sid,
            id=migrated_id(project.id, 'geochemistry_measurement', aid, column.id), **parse_result(raw, column.kind))
        project.science.measurements[measurement.id] = measurement
        project.science.sources[sid].original[column.id] = deepcopy(raw)
        if item['repeat_number'] == repeat:
            project.science.select(row.id, column.id, measurement.id)
    project.validate()
    return project


def project_from_records(records, results, fields, *, source):
    basics = [
        ("source_record_id", "DB / source_record_id", "text", ""),
        ("sample_id", "DB / sample_id", "text", ""),
        ("well_name", "Well / Well Name", "text", ""),
        ("sample_name", "Sample / Name", "text", ""),
        ("sample_type", "Sample / Type", "text", ""),
        ("category", "Sample / Category", "text", ""),
        ("top_md", "Sample / Top MD [m]", "number", "m"),
        ("base_md", "Sample / Base MD [m]", "number", "m"),
        ("md_datum", "Sample / MD datum", "text", ""),
        ("preparation", "Sample / Preparation", "text", ""),
    ]
    project = Project(source=deepcopy(source))
    core_columns = {
        key: Column(name, kind, unit=unit) for key, name, kind, unit in basics
    }
    project.columns = list(core_columns.values())
    field_columns = {}
    for field in fields:
        column = Column(
            f"{field['group_name']} / {field['source_label']}",
            "text" if field["source_dtype"] == "string" else "number",
            unit=field["unit_label"] or "",
            metadata={
                "geochem_field": {
                    "id": field["source_field_id"],
                    "group": field["group_name"],
                    "label": field["source_label"],
                    "dtype": field["source_dtype"],
                }
            },
        )
        field_columns[field["source_field_id"]] = column
        project.columns.append(column)
    record_ids = [record["source_record_id"] for record in records]
    result_frame = pd.DataFrame(results, columns=[
        "source_record_id", "source_field_id", "value_float", "value_int", "value_text",
        "analysis_run_id", "repeat_number", "group_name",
    ], dtype=object)
    run_frame = result_frame[["source_record_id", "group_name", "analysis_run_id", "repeat_number"]].drop_duplicates()
    errors = pd.Series(None, index=result_frame.index, dtype=object)
    errors.loc[run_frame.index[run_frame.duplicated(["source_record_id", "group_name"])]] = "Несколько повторов одной группы в исходной записи; выберите один повтор"
    errors.loc[result_frame.duplicated(["source_record_id", "source_field_id"])] = "В одной строке найдено несколько результатов одного поля; повторы не объединены"
    errors.loc[~result_frame["source_field_id"].isin(field_columns)] = "Неизвестный показатель снимка"
    errors.loc[~result_frame["source_record_id"].isin(record_ids)] = "Результат относится к записи вне снимка"
    if errors.notna().any():
        # Preserve the first source row's diagnostic and its validation priority.
        raise ValueError(errors.dropna().iloc[0])
    if pd.Index(record_ids).has_duplicates:
        raise ValueError("Повторяющиеся исходные записи БД")
    runs = {rid: {} for rid in record_ids}
    for rid, group, run_id, repeat in run_frame.itertuples(index=False, name=None):
        runs[rid][group] = {"analysis_run_id": run_id, "repeat_number": repeat}
    values = result_frame["value_float"]
    values = values.where(values.notna(), result_frame["value_int"])
    result_frame["value"] = values.where(values.notna(), result_frame["value_text"])
    result_frame["column_id"] = result_frame["source_field_id"].map({key: c.id for key, c in field_columns.items()})
    values = result_frame.pivot(index="source_record_id", columns="column_id", values="value")
    values = values.reindex(index=record_ids, columns=[c.id for c in field_columns.values()])
    basics_frame = pd.DataFrame(records, columns=list(core_columns), dtype=object).rename(
        columns={key: column.id for key, column in core_columns.items()},
    )
    basics_frame.index = record_ids
    cells_frame = pd.concat([basics_frame, values], axis=1).astype(object)
    cells_frame = cells_frame.where(cells_frame.notna(), None)
    for record, cells in zip(records, cells_frame.to_dict("records"), strict=True):
        provenance = {
            key: record.get(key)
            for key in (
                "source_record_id",
                "sample_id",
                "import_batch_id",
                "row_number",
                "sample_type",
                "rock_type",
                "liquid_type",
                "gas_type",
            )
        }
        provenance["analysis_runs"] = runs[record["source_record_id"]]
        project.rows.append(
            Row(values=cells, original=deepcopy(cells), source=provenance)
        )
    project.event(
        "postgresql_nonpyro_snapshot",
        rows=len(project.rows),
        groups=source.get("groups", []),
        read_only=source.get("read_only"),
    )
    project.source["source_columns"] = [c.name for c in project.columns]
    return project
