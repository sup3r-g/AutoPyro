"""Apply a fitted curve without refitting; prepare immutable output columns."""
from copy import deepcopy
import hashlib
import json

import numpy as np

from .curve_bootstrap import BOOTSTRAP_BAND_METHODS
from autopyro.model.core import Column, new_id, timestamp
from autopyro.model.tabular import classify


def source_digest(project, column_ids, row_ids, *, version=2):
    """Canonical signatures ignore table ordering; v1 remains readable for old runs."""
    wanted = set(row_ids)
    digest = hashlib.sha256()
    if version == 2:
        columns = {c.id: c for c in project.columns}
        sources = sorted(set(column_ids))
        rows = {row.id: row for row in project.rows if row.id in wanted}
        for cid in sources:
            column = columns.get(cid)
            digest.update(json.dumps([cid, column.kind, column.unit] if column else [cid, None], ensure_ascii=False).encode())
        for rid in sorted(wanted):
            row = rows.get(rid)
            digest.update(json.dumps([rid, *[row.values.get(cid) for cid in sources]] if row else [rid, None],
                                     ensure_ascii=False, default=str).encode())
        return digest.hexdigest()
    # Original order-sensitive representation, retained for existing project files.
    for c in project.columns:
        if c.id in column_ids:
            digest.update(json.dumps([c.id, c.kind, c.unit], ensure_ascii=False).encode())
    for row in project.rows:
        if row.id in wanted:
            digest.update(json.dumps([row.id, *[row.values.get(cid) for cid in column_ids]],
                                     ensure_ascii=False, default=str).encode())
    return digest.hexdigest()


def output_columns(result, rows, settings, *, bootstrap=None, confidence=.95):
    """Prepare values in bounded chunks; bootstrap memory does not grow as N*B."""
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("Уровень доверия должен быть между 0 и 1")
    x_id, y_id = settings["x"], settings["y"]
    xs = np.array([classify(row.values.get(x_id), numeric=True)[1] for row in rows], float)
    ys = np.array([classify(row.values.get(y_id), numeric=True)[1] for row in rows], float)
    valid = np.isfinite(xs)
    if result.model.positive_x:
        valid &= xs > 0
    use_bootstrap = bootstrap is not None and settings.get("band_method") != "analytic"
    with_bands = use_bootstrap or (result.covariance is not None and settings.get("band_method") not in BOOTSTRAP_BAND_METHODS)
    calibration_range = None
    if use_bootstrap and bootstrap.max_deviations is not None:
        calibration_range = np.min(bootstrap.grid), np.max(bootstrap.grid)
    keys = ["prediction", "residual"] + (["confidence_lower", "confidence_upper", "prediction_lower", "prediction_upper"] if with_bands else [])
    values = {key: {} for key in keys}
    indices = np.flatnonzero(valid)
    for offset in range(0, len(indices), 512):
        idx = indices[offset:offset+512]
        grid = xs[idx]
        mean = result.predict(grid)
        if use_bootstrap:
            bands = bootstrap.bands_at(result, grid, confidence=confidence)[1:]
        elif with_bands:
            bands = result.bands(grid, confidence=confidence)[1:]
        for j, position in enumerate(idx):
            rid = rows[position].id
            values["prediction"][rid] = float(mean[j])
            values["residual"][rid] = float(ys[position]-mean[j]) if np.isfinite(ys[position]) else None
            if with_bands:
                for key, band in zip(keys[2:], bands):
                    if (calibration_range is not None and key.startswith("confidence_")
                            and not calibration_range[0] <= grid[j] <= calibration_range[1]):
                        values[key][rid] = None
                    else:
                        values[key][rid] = float(band[j])
    return values


def invalidate_curve_outputs(project, rows, invalidate, *, changed_columns=None):
    runs = project.view.get("curve_fit_runs", {})
    records = list(runs.values()) + list(project.view.get("curve_fit_results", []))
    if not records:
        return
    existing = {c.id for c in project.columns}
    changed_rows = {row.id for row in rows} if rows is not None else None
    for record in records:
        sources, row_ids = record.get("source_ids", []), record.get("source_rows", [])
        if not set(sources) <= existing:
            record["stale"] = True
        elif (invalidate and not record.get("stale")
              and (changed_columns is None or not changed_columns.isdisjoint(sources))
              and (changed_rows is None or not changed_rows.isdisjoint(row_ids))):
            if changed_rows is not None and changed_columns is None:
                # A working measurement can change its QC eligibility without changing its value.
                record["stale"] = True
            else:
                version = record.get("source_digest_version", 1)
                record["stale"] = (version not in (1, 2)
                                   or record.get("source_digest") != source_digest(project, sources, row_ids, version=version))
    for column in project.columns:
        spec = column.metadata.get("curve_fit")
        if spec:
            column.metadata["stale"] = bool(runs.get(spec["run_id"], {}).get("stale", True))


def fit_source_ids(settings, bootstrap):
    keys = ["x", "y", "sx", "sy"]
    if bootstrap is not None and bootstrap.group_count is not None:
        keys.append("bootstrap_group")
    sources = [settings[key] for key in keys if settings.get(key)]
    selection = settings.get("selection", {})
    sources.extend(selection[key] for key in ("holdout_column", "split_column") if selection.get(key))
    return list(dict.fromkeys(sources))


def prepare_output_columns(project, rows, result, settings, row_ids, *, bootstrap=None, confidence=.95,
                           labels=None, prefix="Fit"):
    values = output_columns(result, rows, settings, bootstrap=bootstrap, confidence=confidence)
    used_bootstrap = bootstrap if settings.get("band_method") != "analytic" else None
    source_ids = fit_source_ids(settings, used_bootstrap)
    source_rows = list(dict.fromkeys([*row_ids, *[r.id for r in rows]]))
    run_id = new_id()
    record = {"id": run_id, "at": timestamp(), "settings": deepcopy(settings),
              "parameters": result.parameters.tolist(), "formula": result.model.formula,
              "confidence": confidence,
              "band_method_actual": "bootstrap" if used_bootstrap is not None else
                  ("analytic" if result.covariance is not None and settings.get("band_method") not in BOOTSTRAP_BAND_METHODS else None),
              "training_rows": list(row_ids), "source_ids": source_ids, "source_rows": source_rows,
              "training_x": result.x.tolist(), "training_y": result.y.tolist(),
              "training_sx": result.sx.tolist() if result.sx is not None else None,
              "training_sy": result.sy.tolist() if result.sy is not None else None,
              "bootstrap": used_bootstrap.diagnostics(confidence=confidence) if used_bootstrap is not None else None,
              "confidence_calibration_grid": (used_bootstrap.grid.tolist()
                  if used_bootstrap is not None and used_bootstrap.max_deviations is not None else None),
              "bootstrap_parameters": used_bootstrap.parameters.tolist() if used_bootstrap is not None else None,
              "bootstrap_prediction_draws": used_bootstrap.prediction_draws.tolist() if used_bootstrap is not None and used_bootstrap.prediction_draws is not None else None,
              "source_digest": source_digest(project, source_ids, source_rows), "source_digest_version": 2, "stale": False}
    unit = next(c.unit for c in project.columns if c.id == settings["y"])
    labels = labels or {}
    columns = [(Column(name=f"{prefix}: {labels.get(key, key)}", kind="number", unit=unit,
                       metadata={"curve_fit": {"run_id": run_id, "output": key}, "stale": False}), data)
               for key, data in values.items()]
    return record, columns
