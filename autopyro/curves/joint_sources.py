"""Provenance shared by a joint fit's report snapshot and output columns."""

from hashlib import sha256
from itertools import islice
import json

from autopyro.model.tabular import excluded_sample_ids


def source_schema(project, ids):
    columns = {column.id: column for column in project.columns}
    return [(columns[cid].kind, columns[cid].unit) if cid in columns else None for cid in ids]


def joint_source_digest(schema, cells, excluded):
    digest = sha256()
    encode = json.JSONEncoder(ensure_ascii=False, default=str, separators=(",", ":")).encode
    digest.update(encode(schema).encode("utf-8"))
    cells = iter(cells)
    while batch := list(islice(cells, 256)):
        digest.update(encode(batch).encode("utf-8"))
    digest.update(encode(sorted(excluded)).encode("utf-8"))
    return digest.hexdigest()


def joint_report_snapshots(project):
    snapshots = dict(project.view.get("joint_curve_reports", {}))
    if latest := project.view.get("joint_curve_report"):
        snapshots[latest.get("id", "legacy")] = latest
    return snapshots


def joint_report_snapshot(project, snapshot_id=None):
    if snapshot_id is None:
        return project.view.get("joint_curve_report")
    latest = project.view.get("joint_curve_report")
    if latest and latest.get("id", "legacy") == snapshot_id:
        return latest
    return project.view.get("joint_curve_reports", {}).get(snapshot_id)


def update_joint_status(project, *, changed_rows=None, changed_columns=None, artifacts_project=None, rows_by_id=None):
    """Check each referenced run once and update its visible artifacts."""
    target = project if artifacts_project is None else artifacts_project
    artifacts = [column.metadata for column in target.columns if column.metadata.get("joint_curve_fit")]
    snapshot = target.view.get("joint_curve_report")
    if snapshot:
        artifacts.append(snapshot)
    if not artifacts:
        return
    runs = project.view.get("joint_curve_runs", {})
    changed = {row.id for row in changed_rows} if changed_rows is not None else None
    statuses = {}
    for artifact in artifacts:
        spec = artifact.get("joint_curve_fit", artifact)
        run_id = spec.get("run_id")
        if run_id not in statuses:
            source = runs.get(run_id)
            if not source or source.get("version") != 1:
                statuses[run_id] = "unverified"
            else:
                ids, row_ids = source["source_ids"], source["row_ids"]
                if (changed_columns is not None and not set(ids).intersection(changed_columns)
                        or changed is not None and changed.isdisjoint(row_ids)):
                    continue
                schema = source_schema(project, ids)
                if rows_by_id is None:
                    rows_by_id = project.row_map()
                if None in schema or any(rid not in rows_by_id for rid in row_ids):
                    statuses[run_id] = "stale"
                else:
                    excluded = excluded_sample_ids(project, source["numeric_ids"], "statistics")
                    cells = ([rid, *(rows_by_id[rid].values.get(cid) for cid in ids)] for rid in row_ids)
                    digest = joint_source_digest(schema, cells, (rid for rid in row_ids if rid in excluded))
                    statuses[run_id] = "current" if digest == source["digest"] else "stale"
        artifact["source_status"] = statuses[run_id]
