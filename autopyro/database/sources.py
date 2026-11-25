"""Read-only source adapters and credential-free snapshot descriptors."""
from copy import deepcopy
from pathlib import Path

from autopyro.model.core import timestamp
from autopyro.data_io.io import build_project
from .readers import read_sqlite, read_postgres_snapshot


def descriptor(source):
    kind = source.get("type")
    if kind == "sqlite":
        result = {"type": kind, "path": str(Path(source["path"]).resolve()), "table": source["table"]}
    elif kind == "postgresql":
        if source.get("relation") != "geochem.source_record.raw_cells":
            raise ValueError("Этот адаптер поддерживает только geochem.source_record.raw_cells")
        result = {"type": kind, **{name: source[name] for name in ("host", "port", "dbname", "relation")}}
    elif kind == "postgresql_nonpyro":
        result = {"type": kind, "endpoint": {name: source["endpoint"][name] for name in ("host", "port", "database")},
                  **{name: deepcopy(source.get(name)) for name in
                     ("method_profile", "groups", "group", "variant", "repeat_number", "batch_id", "well_name")}}
        result["groups"] = sorted(result["groups"] or [])
        result["group"] = result["group"] or (result["groups"][0] if len(result["groups"]) == 1 else "")
        result["variant"] = result["variant"] or ""
        result["repeat_number"] = result["repeat_number"] or 1
        result["well_name"] = result["well_name"] or ""
        for key in ('method_profiles', 'include_repeats'):
            if key in source:
                result[key] = deepcopy(source[key])
    else:
        raise ValueError("Для этого проекта нет поддерживаемого источника БД")
    result["read_only"] = True
    return result


def identity(source):
    source = descriptor(source)
    source.pop("read_only")
    if "port" in source:
        source["port"] = str(source["port"])
    if "endpoint" in source:
        source["endpoint"]["port"] = str(source["endpoint"]["port"])
    return source


def field_key(column):
    field = column.metadata.get("geochem_field")
    return f"field:{field['id']}" if field else "name:" + column.metadata.get("source_header", column.name)


def row_provenance(source, row, *, source_descriptor=None):
    """Persist source identity and record coordinates, never connection options."""
    # Snapshot readers can resolve their shared source once, outside the row loop.
    result = descriptor(source) if source_descriptor is None else deepcopy(source_descriptor)
    for name in ("source_columns", "imported_at", "created_at"):
        if name in source:
            result[name] = deepcopy(source[name])
    if source.get("batches"):
        result["batches"] = [{key: value for key, value in batch.items()
                              if key in ("import_batch_id", "source_name", "source_sha256", "imported_at")}
                             for batch in source["batches"]]
    for name in ("row", "row_number", "source_record_id", "sample_id", "import_batch_id", "sample_type",
                 "rock_type", "liquid_type", "gas_type"):
        if name in row:
            result[name] = deepcopy(row[name])
    if row.get("analysis_runs"):
        result["analysis_runs"] = {group: {key: value for key, value in run.items()
                                            if key in ("analysis_run_id", "repeat_number")}
                                    for group, run in row["analysis_runs"].items()}
    return result


def read_snapshot(source, parameters=None):
    """Parameters exist only in memory. Descriptor and project contain no login data."""
    source = descriptor(source)
    kind = source["type"]
    parameters = dict(parameters or {})
    if kind == "sqlite":
        headers, rows, numbers = read_sqlite(source["path"], source["table"])
    else:
        endpoint = source if kind == "postgresql" else source["endpoint"]
        parameters.update(host=endpoint["host"], port=endpoint["port"],
                          dbname=endpoint.get("dbname", endpoint.get("database")))
        if kind == "postgresql_nonpyro":
            from .geochem import read_profile
            if source.get('method_profiles'):
                return read_profile(parameters, source['method_profile'], method_ids=source['method_profiles'],
                    include_repeats=source.get('include_repeats', False), group=source.get('group', ''),
                    variant=source.get('variant', ''), repeat_number=source.get('repeat_number', 1),
                    batch_id=source.get('batch_id'), well_name=source.get('well_name', ''), limit=100000)
            groups = source.get("groups") or []
            return read_profile(parameters, source["method_profile"],
                group=source.get("group") or (groups[0] if len(groups) == 1 else ""),
                variant=source.get("variant") or "", repeat_number=source.get("repeat_number") or 1,
                batch_id=source.get("batch_id"), well_name=source.get("well_name") or "", limit=100000)
        from psycopg.conninfo import make_conninfo
        headers, rows, numbers = read_postgres_snapshot(make_conninfo(**parameters))
    source.update(imported_at=timestamp(), source_columns=list(headers))
    # Literal input permits the refresh preview to show malformed new numeric results.
    return build_project(headers, rows, numbers,
                         [(i, name, "text", "") for i, name in enumerate(headers)], source)
