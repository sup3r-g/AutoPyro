"""Interpolation produces a proposal; applying it never overwrites measurements."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np
import scipy
from scipy.interpolate import CubicSpline, PchipInterpolator

from autopyro.model.core import Column, Project, Row, timestamp
from autopyro.model.tabular import project_frame, classify_series
from autopyro.model.protocol_json import protocol_json_chunks


SPLINE_MIN_ANCHORS = 4
SPLINE_BOUNDARY = "natural"


def minimum_anchor_count(method):
    return SPLINE_MIN_ANCHORS if method == "spline" else 2


@dataclass
class Proposal:
    row_id: str
    x: float
    value: int | float
    left_id: str
    right_id: str
    extrapolated: bool = False

    def record(self):
        data = {"row_id": self.row_id, "x": self.x, "value": self.value,
                "left_id": self.left_id, "right_id": self.right_id}
        if self.extrapolated:
            data["extrapolated"] = True
        return data


class DuplicateCoordinatesError(ValueError):
    def __init__(self, coordinates):
        super().__init__("Повторные координаты внутри группы: уточните группировку или выборку")
        self.coordinates = coordinates


def propose(
    project: Project,
    x_id: str,
    y_id: str,
    group_ids: list[str],
    row_ids: set[str],
    method="linear",
    max_gap=100.0,
    *, constraints=None, max_extrapolation=None, min_anchors=None, duplicate_policy="reject",
):
    return _calculate(project, x_id, y_id, group_ids, row_ids, method, max_gap,
                      constraints=constraints, max_extrapolation=max_extrapolation,
                      min_anchors=min_anchors, duplicate_policy=duplicate_policy, exact_values=True)


def _calculate(project, x_id, y_id, group_ids, row_ids, method, max_gap, *, supports=None,
               constraints=None, eligibility=None, max_extrapolation=None, min_anchors=None,
               duplicate_policy="reject", diagnostic_details=False, extrapolation_diagnostics=False,
               qualified_y_diagnostics=False, limit_diagnostics=False, anchor_diagnostics=False,
               legacy=False, checkpoint=None, exact_values=False):
    columns = {c.id: c for c in project.columns}
    if x_id not in columns or y_id not in columns:
        raise ValueError("Для интерполяции нужны две числовые колонки")
    if columns[x_id].kind != "number" or columns[y_id].kind != "number" or x_id == y_id:
        raise ValueError("Выберите разные числовые колонки X и Y")
    if method not in ("linear", "nearest", "pchip", "spline") or not math.isfinite(max_gap) or max_gap <= 0:
        raise ValueError("Некорректные параметры интерполяции")
    required = minimum_anchor_count(method)
    if min_anchors is not None:
        if type(min_anchors) is not int or min_anchors < required:
            raise ValueError("Минимум опор должен быть целым числом не ниже порога метода")
        required = min_anchors
    # The spline checks its threshold separately within each continuous span.
    group_minimum = 2 if method == "spline" else required
    if max_extrapolation is not None:
        if method != "linear":
            raise ValueError("Экстраполяция доступна только для линейного метода")
        if isinstance(max_extrapolation, bool) or not math.isfinite(max_extrapolation) or max_extrapolation <= 0:
            raise ValueError("Предел экстраполяции должен быть положительным конечным числом")
    if not group_ids:
        raise ValueError("Выберите группировку (например, скважину и метод)")
    if duplicate_policy not in ("reject", "exclude"):
        raise ValueError("Неизвестная политика повторяющихся координат")
    if len(set(group_ids)) != len(group_ids) or any(cid not in columns for cid in group_ids):
        raise ValueError("Некорректные колонки группировки")
    if constraints is not None and constraints.get("exclude_intervals_by") == x_id:
        raise ValueError("Выберите отдельную числовую колонку подошвы пробы")
    partitions = None
    if constraints is not None:
        from .interpolation_constraints import partition
        partitions, diagnostics = partition(project, set(row_ids), y_id, constraints, checkpoint=checkpoint)
        if eligibility is not None:
            eligibility.update(diagnostics)
    frame = project_frame(project, partitions if partitions is not None else row_ids,
                          [*dict.fromkeys([*group_ids, x_id, y_id])])
    x_data = classify_series(frame[x_id], numeric=True)
    if checkpoint:
        checkpoint()
    y_data = classify_series(frame[y_id], numeric=True)
    if checkpoint:
        checkpoint()
    has_group = frame[group_ids].notna().all(axis=1)
    if not legacy:
        for cid in group_ids:
            has_group &= frame[cid].map(lambda value: not isinstance(value, str) or bool(value.strip()))
        frame[x_id] = x_data["number"]
    has_x = x_data["number"].notna()
    valid = has_group & has_x

    def mark(indices, status, actual=None, limit=None):
        if not diagnostic_details or eligibility is None:
            return
        if limit_diagnostics and limit is not None:
            for rid, value in zip(indices, np.broadcast_to(actual, len(indices)), strict=True):
                eligibility.setdefault(rid, {}).update(
                    status=status, actual=float(value) if math.isfinite(value) else "∞", limit=float(limit),
                )
        else:
            for rid in indices:
                eligibility.setdefault(rid, {})["status"] = status

    def mark_anchor_shortage(indices, status, found):
        if eligibility is None:
            return
        for rid, count in zip(indices, np.broadcast_to(found, len(indices)), strict=True):
            record = eligibility.setdefault(rid, {})
            record["status"] = status
            if anchor_diagnostics:
                record.update(anchors_found=int(count), anchors_required=required)

    if diagnostic_details:
        missing_y = frame[y_id].isna() if legacy else y_data["status"].eq("missing")
        mark(frame.index[(missing_y & ~has_group) if legacy else ~has_group], "missing_group")
        invalid_x = has_group & ~has_x
        mark(frame.index[(missing_y & invalid_x) if legacy else invalid_x], "invalid_x")
        if not legacy:
            for status in ("non_detect", "censored", "other"):
                reason = f"y_{status}" if qualified_y_diagnostics else "qualified_y"
                mark(frame.index[valid & y_data["status"].eq(status)], reason)
    partition_column = None
    grouping = list(group_ids)
    if partitions is not None:
        # A private column object cannot collide with a user column UUID.
        partition_column = object()
        frame[partition_column] = [partitions[rid] for rid in frame.index]
        grouping.append(partition_column)
    frame = frame.loc[valid]
    if duplicate_policy == "reject":
        keys = list(dict.fromkeys([*grouping, x_id]))
        repeated = frame.loc[frame.duplicated(subset=keys, keep=False), x_id]
        if not repeated.empty:
            raise DuplicateCoordinatesError(repeated.to_dict())
    proposals = []
    for _, rows in frame.groupby(grouping, sort=False, dropna=False):
        if checkpoint:
            checkpoint()
        missing = (rows[y_id].isna() if legacy else y_data.loc[rows.index, "status"].eq("missing"))
        if duplicate_policy == "exclude":
            duplicates = rows[x_id].duplicated(keep=False)
            if duplicates.any():
                if eligibility is not None:
                    for rid in rows.index[duplicates]:
                        eligibility.setdefault(rid, {}).update(status="duplicate_coordinate")
                rows = rows.loc[~duplicates]
                if rows.empty:
                    continue
        rows = rows.sort_values(x_id, kind="stable")
        missing = missing.loc[rows.index]
        numeric_y = y_data.loc[rows.index, "number"]
        anchors = rows.loc[numeric_y.notna()]
        if supports is not None:
            supports.append({
                "group": [rows.iloc[0][cid] for cid in group_ids],
                "anchors": [{"row_id": rid, "x": x, "y": y}
                            for rid, x, y in zip(anchors.index, anchors[x_id],
                                                anchors[y_id] if legacy else numeric_y.loc[anchors.index], strict=True)],
            })
            if partition_column is not None:
                supports[-1]["partition"] = list(rows.iloc[0][partition_column])
        if len(anchors) < group_minimum:
            if min_anchors is not None or diagnostic_details:
                mark_anchor_shortage(rows.index[missing], "insufficient_anchors", len(anchors))
            continue
        xp = anchors[x_id].to_numpy(dtype=float)
        anchor_y = anchors[y_id] if legacy else numeric_y.loc[anchors.index]
        yp = anchor_y.to_numpy(dtype=object if exact_values and method == "nearest" else float)
        if exact_values and method != "nearest" and any(
            type(v) is int and int(float(v)) != v for v in anchor_y
        ):
            raise ValueError("Значения Y теряют точность при интерполяции.")
        x_limit = max(abs(xp[0]), abs(xp[-1])) + (max_extrapolation or 0)
        if x_limit >= 2**53 and any(
            type(value) is int and int(float(value)) != value for value in rows[x_id]
        ):
            raise ValueError("Координаты X теряют точность при интерполяции. Используйте смещения от начала шкалы.")
        inside = rows[x_id].gt(xp[0]) & rows[x_id].lt(xp[-1])
        if diagnostic_details:
            mark(rows.index[missing & ~inside], "outside_range")
        if max_extrapolation is not None:
            before, after = _linear_exterior(rows.loc[missing, x_id], anchors.index, xp, yp,
                                             max_gap, max_extrapolation,
                                             mark=mark if extrapolation_diagnostics else None)
            proposals.extend(before)
            if diagnostic_details:
                mark((item.row_id for item in (*before, *after)), "eligible")
        targets = rows.loc[missing & inside]
        x = targets[x_id].to_numpy(dtype=float)
        right = np.searchsorted(xp, x)
        left = right - 1
        gaps = xp[right] - xp[left]
        within_gap = gaps <= max_gap
        if diagnostic_details:
            mark(targets.index[~within_gap], "gap_too_large", gaps[~within_gap], max_gap)
        targets, x = targets.loc[within_gap], x[within_gap]
        left, right = left[within_gap], right[within_gap]
        if method == "linear":
            values = np.interp(x, xp, yp)
        elif method == "nearest":
            values = yp[np.where(x - xp[left] <= xp[right] - x, left, right)]
        elif method == "pchip" and legacy:
            values = PchipInterpolator(xp, yp, extrapolate=False)(x)
        else:
            counts = np.zeros(len(x), dtype=int) if anchor_diagnostics else None
            values, applicable = _segment_values(
                xp, yp, x, left, max_gap, required, method, checkpoint=checkpoint, anchor_counts=counts,
            )
            if eligibility is not None:
                status = "insufficient_spline_anchors" if method == "spline" else "insufficient_anchors"
                if method == "pchip" and anchor_diagnostics:
                    status = "insufficient_segment_anchors"
                mark_anchor_shortage(targets.index[~applicable], status, counts[~applicable] if counts is not None else 0)
            targets, values = targets.loc[applicable], values[applicable]
            left, right = left[applicable], right[applicable]
        exact_nearest = exact_values and method == "nearest"
        if not legacy and not exact_nearest and not np.isfinite(values).all():
            raise ValueError("Интерполяция вернула неконечное значение. Проверьте диапазон Y.")
        proposals.extend(
            Proposal(rid, original_x, value if exact_nearest else float(value), anchors.index[lo], anchors.index[hi])
            for rid, original_x, value, lo, hi in zip(targets.index, targets[x_id], values, left, right, strict=True)
        )
        if max_extrapolation is not None:
            proposals.extend(after)
    return proposals


def _linear_exterior(targets, anchor_ids, xp, yp, max_gap, max_distance, *, mark=None):
    x = targets.to_numpy(dtype=float)

    def side(lo, hi, edge, distances):
        selected = (distances > 0) & (distances <= max_distance)
        if mark is not None:
            too_far = distances > max_distance
            mark(targets.index[too_far], "extrapolation_distance", distances[too_far], max_distance)
        gap = xp[hi] - xp[lo]
        if not 0 < gap <= max_gap:
            if mark is not None:
                mark(targets.index[selected], "extrapolation_gap", gap, max_gap)
            return []
        if not selected.any():
            return []
        with np.errstate(over="ignore", invalid="ignore"):
            values = yp[edge] + (x[selected] - xp[edge]) * ((yp[hi] - yp[lo]) / gap)
        if not np.isfinite(values).all():
            raise ValueError("Линейная экстраполяция вернула неконечное значение")
        return [
            Proposal(rid, original_x, float(value), anchor_ids[lo], anchor_ids[hi], extrapolated=True)
            for (rid, original_x), value in zip(targets[selected].items(), values, strict=True)
        ]

    return side(0, 1, 0, xp[0] - x), side(-2, -1, -1, x - xp[-1])


def _segment_values(xp, yp, x, left, max_gap, min_anchors, method, *, checkpoint=None, anchor_counts=None):
    """Fit each contiguous span independently and leave unsupported targets out."""
    values = np.empty(len(x), dtype=float)
    applicable = np.zeros(len(x), dtype=bool)
    if not len(x):
        return values, applicable
    gaps = np.diff(xp)
    if (gaps <= 0).any():
        raise ValueError("Координаты опор сплайна совпадают после числового преобразования")
    boundaries = [0, *(np.flatnonzero(gaps > max_gap) + 1), len(xp)]
    for start, stop in zip(boundaries, boundaries[1:]):
        if checkpoint:
            checkpoint()
        if anchor_counts is not None:
            first, last = np.searchsorted(left, (start, stop - 1))
            anchor_counts[first:last] = stop - start
        if stop - start < min_anchors:
            continue
        if anchor_counts is None:
            first, last = np.searchsorted(left, (start, stop - 1))
        if first == last:
            continue
        if method == "pchip":
            spline = PchipInterpolator(xp[start:stop], yp[start:stop], extrapolate=False)
        else:
            spline = CubicSpline(xp[start:stop], yp[start:stop], bc_type=SPLINE_BOUNDARY, extrapolate=False)
        values[first:last] = spline(x[first:last])
        applicable[first:last] = True
    if not np.isfinite(values[applicable]).all():
        raise ValueError("Интерполяция вернула неконечное значение. Проверьте диапазон Y.")
    return values, applicable


def validation_scores(protocol, *, limit=120, checkpoint=None, progress=None):
    """Compare methods on a bounded, deterministic sample of interior anchors."""
    from autopyro.model.audit import read_protocol

    snapshot = read_protocol(protocol, checkpoint=checkpoint) if isinstance(protocol, str) else protocol
    if checkpoint:
        checkpoint()
    max_gap = snapshot["parameters"]["max_gap"]
    configured_minimum = snapshot["parameters"].get("min_anchors", 0)
    groups = [group["anchors"] for group in snapshot["supports"] if len(group["anchors"]) > 2]
    candidates = sum(len(anchors) - 2 for anchors in groups)
    sampled = min(limit, candidates)
    positions = iter(i * candidates // sampled for i in range(sampled))
    position = next(positions, None)
    methods = {method: {} for method in ("linear", "nearest", "pchip", "spline")}
    offset = 0
    completed = 0
    for anchors in groups:
        if checkpoint:
            checkpoint()
        if position is None:
            break
        end = offset + len(anchors) - 2
        if position >= end:
            offset = end
            continue
        xp = np.array([anchor["x"] for anchor in anchors], dtype=float)
        yp = np.array([anchor["y"] for anchor in anchors], dtype=float)
        exact_y = [anchor["y"] for anchor in anchors]
        has_lossy_y = any(type(value) is int and int(float(value)) != value for value in exact_y)
        while position is not None and position < end:
            if checkpoint:
                checkpoint()
            i = position - offset + 1
            position = next(positions, None)
            completed += 1
            if progress:
                progress(10 + completed * 89 // sampled)
            if not 0 < xp[i + 1] - xp[i - 1] <= max_gap:
                continue
            x = xp[i:i + 1]
            xx, yy = np.delete(xp, i), np.delete(yp, i)
            for method, errors in methods.items():
                minimum = max(configured_minimum, minimum_anchor_count(method))
                if len(xx) < minimum or (has_lossy_y and method != "nearest"):
                    continue
                try:
                    if method == "linear":
                        prediction = np.interp(x, xx, yy)[0]
                    elif method == "nearest":
                        neighbor = i - 1 if xp[i] - xp[i - 1] <= xp[i + 1] - xp[i] else i + 1
                        prediction = exact_y[neighbor]
                    else:
                        values, applicable = _segment_values(
                            xx, yy, x, np.array([i - 1]), max_gap, minimum, method, checkpoint=checkpoint)
                        if not applicable[0]:
                            continue
                        prediction = values[0]
                    error = abs(prediction - exact_y[i])
                    if math.isfinite(error):
                        errors[anchors[i]["row_id"]] = error
                except (ValueError, FloatingPointError):
                    continue
        offset = end

    common = sorted(set.intersection(*(set(errors) for errors in methods.values())))

    def metrics(errors):
        if not errors:
            return None, None
        scale = max(errors)
        if not scale:
            return 0.0, 0.0
        normalized = np.array(errors) / scale
        return float(normalized.mean() * scale), float(np.sqrt(np.mean(normalized**2)) * scale)

    scores = {
        method: (len(errors), *metrics([errors[rid] for rid in common]))
        for method, errors in methods.items()
    }
    if (distance := snapshot["parameters"].get("max_extrapolation")) is not None:
        eligible = [anchors for anchors in groups if len(anchors) > max(2, configured_minimum)]
        count = min(limit, 2 * len(eligible))
        errors = []
        for index in range(count):
            if checkpoint:
                checkpoint()
            group, side = divmod(index * 2 * len(eligible) // count, 2)
            endpoint, near, far = eligible[group][:3] if side == 0 else eligible[group][-1:-4:-1]
            span = abs(far["x"] - near["x"])
            delta = abs(endpoint["x"] - near["x"])
            if not (0 < span <= max_gap and 0 < delta <= distance):
                continue
            values = (endpoint["y"], near["y"], far["y"])
            if any(type(value) is int and int(float(value)) != value for value in values):
                continue
            observed, y_near, y_far = map(float, values)
            error = abs(y_near + delta * ((y_near - y_far) / span) - observed)
            if math.isfinite(error):
                errors.append(error)
        scores["extrapolation"] = (len(errors), *metrics(errors))
    return sampled, len(common), scores


PROTOCOL_VERSION = 1
LEGACY_ALGORITHM = "autopyro.interpolation/1"
ALGORITHM = "autopyro.interpolation/3"
POLICIES = {"extrapolation": False, "duplicates": "reject",
            "targets": "missing_only", "anchors": "finite_numeric_only"}
COMPACT_MEASUREMENT_THRESHOLD = 100_000
# Encoding v3 freezes these omitted nulls explicitly; changing this map requires
# a new encoding version. IDs, timestamps, units,
# qualifiers, revisions and every non-null value remain in each record.
NULL_MEASUREMENT_DEFAULTS = dict.fromkeys(("raw_value", "value", "limit", "lod", "loq",
                                          "uncertainty", "confidence_level", "source_id"))
REQUIRED_MEASUREMENT_FIELDS = frozenset(("id", "sample_id", "property_id", "analysis_id", "created_at",
                                        "revision", "qualifier", "unit", "basis", "comparator", "uncertainty_type", "qc_status"))


def _policies_for_method(method, max_extrapolation=None, min_anchors=None, duplicate_policy="reject"):
    policies = dict(POLICIES)
    policies["extrapolation"] = max_extrapolation is not None
    policies["duplicates"] = duplicate_policy
    if method == "spline":
        policies["spline"] = {"boundary": SPLINE_BOUNDARY,
                              "min_anchors": SPLINE_MIN_ANCHORS if min_anchors is None else min_anchors,
                              "split_on_max_gap": True}
    return policies


def _compact_measurements(measurements, *, checkpoint=None):
    # The default declaration must cost less than the omitted fields. Each
    # removed entry saves len(key)+8 compact JSON bytes (including its comma);
    # required fields ensure the record itself never becomes empty.
    overhead = len(json.dumps({"measurement_defaults": NULL_MEASUREMENT_DEFAULTS}, separators=(",", ":"))) - 1
    for index, record in enumerate(measurements.values()):
        if checkpoint and index % 256 == 0:
            checkpoint()
        for key in NULL_MEASUREMENT_DEFAULTS:
            if key in record and record[key] is None:
                overhead -= len(key) + 8
        if overhead < 0:
            break
    else:
        return False
    for index, record in enumerate(measurements.values()):
        if checkpoint and index % 256 == 0:
            checkpoint()
        for key in NULL_MEASUREMENT_DEFAULTS:
            if key in record and record[key] is None:
                del record[key]
    return True


def _checksum(protocol, *, checkpoint=None):
    payload = {key: value for key, value in protocol.items() if key != "checksum"}
    digest = hashlib.sha256()
    for chunk in protocol_json_chunks(payload, sort_keys=True):
        if checkpoint:
            checkpoint()
        digest.update(chunk)
    return digest.hexdigest()


def prepare(project, x_id, y_id, group_ids, row_ids, method="linear", max_gap=100.0, *,
            constraints=None, max_extrapolation=None, min_anchors=None, duplicate_policy="reject",
            checkpoint=None):
    """Calculate once and freeze only selected inputs and their working provenance.

    Supports are stored once per group, including every spline anchor. A snapshot
    includes missing/qualified rows too, so replay preserves target eligibility.
    It never synchronizes or otherwise mutates the caller's project.
    """
    selected = set(row_ids)
    constraints = deepcopy(constraints)
    extrapolation_diagnostics = constraints is not None and max_extrapolation is not None
    supports, eligibility = [], {}
    proposals = _calculate(project, x_id, y_id, group_ids, selected, method, max_gap,
                           supports=supports, constraints=constraints, eligibility=eligibility,
                           max_extrapolation=max_extrapolation, min_anchors=min_anchors,
                           duplicate_policy=duplicate_policy, diagnostic_details=constraints is not None,
                           extrapolation_diagnostics=extrapolation_diagnostics,
                           qualified_y_diagnostics=constraints is not None,
                           limit_diagnostics=constraints is not None, anchor_diagnostics=constraints is not None,
                           checkpoint=checkpoint, exact_values=True)
    from .interpolation_constraints import input_ids
    ids = list(dict.fromkeys([*group_ids, x_id, y_id, *input_ids(constraints or {})]))
    columns = {column.id: column for column in project.columns}
    rows, measurements, analyses, sources = [], {}, {}, {}
    for index, row in enumerate(project.rows):
        if checkpoint and index % 256 == 0:
            checkpoint()
        if row.id not in selected:
            continue
        working = {}
        for cid in ids:
            measurement = project.science.current(row.id, cid)
            if measurement is None:
                continue  # Derived inputs retain their raw projection and definition.
            working[cid] = measurement.id
            # Revision and current result suffice; do not duplicate edit history.
            measurements[measurement.id] = {
                key: deepcopy(value) for key, value in vars(measurement).items() if key != "history"
            }
            analysis = project.science.analyses[measurement.analysis_id]
            if analysis.id not in analyses:
                analyses[analysis.id] = asdict(analysis)
            for sid in (measurement.source_id, analysis.source_id):
                if sid and sid not in sources:
                    source = project.science.sources[sid]
                    sources[sid] = {"id": sid, "provenance": deepcopy(source.provenance),
                                    "original": {cid: deepcopy(value) for cid, value in source.original.items()
                                                 if cid in ids}}
        rows.append({"id": row.id, "values": {cid: deepcopy(row.values[cid]) for cid in ids},
                     "working": working})
    protocol = {
        "version": PROTOCOL_VERSION, "algorithm": ALGORITHM, "created_at": timestamp(),
        "environment": {"numpy": np.__version__, "scipy": scipy.__version__},
        "project_id": project.id,
        "parameters": {"x_id": x_id, "y_id": y_id, "group_ids": list(group_ids),
                       "method": method, "max_gap": max_gap},
        "policies": _policies_for_method(method, max_extrapolation, min_anchors, duplicate_policy),
        "columns": [{"id": cid, "name": columns[cid].name, "kind": columns[cid].kind,
                     "unit": columns[cid].unit,
                     "metadata": {key: deepcopy(value) for key, value in columns[cid].metadata.items()
                                  if key not in ("diagnostics", "proposals", "interpolation_protocol")}}
                    for cid in ids],
        "rows": rows, "measurements": measurements, "analyses": analyses, "sources": sources,
        "supports": supports, "proposals": [proposal.record() for proposal in proposals],
    }
    if max_extrapolation is not None:
        protocol["parameters"]["max_extrapolation"] = max_extrapolation
        if extrapolation_diagnostics:
            protocol["parameters"]["extrapolation_diagnostics"] = True
    if min_anchors is not None:
        protocol["parameters"]["min_anchors"] = min_anchors
    protocol["parameters"]["exact_values"] = True
    if duplicate_policy != "reject":
        protocol["parameters"]["duplicate_policy"] = duplicate_policy
        protocol["eligibility"] = eligibility
    if constraints is not None:
        protocol.update(version=2, eligibility=eligibility)
        protocol["parameters"]["constraints"] = constraints
        protocol["parameters"]["diagnostic_details"] = True
        protocol["parameters"]["qualified_y_diagnostics"] = True
        protocol["parameters"]["limit_diagnostics"] = True
        protocol["parameters"]["anchor_diagnostics"] = True
    if len(measurements) >= COMPACT_MEASUREMENT_THRESHOLD and _compact_measurements(
        measurements, checkpoint=checkpoint
    ):
        protocol.update(version=3, measurement_defaults=dict(NULL_MEASUREMENT_DEFAULTS))
    protocol["checksum"] = _checksum(protocol, checkpoint=checkpoint)
    return proposals, protocol


def replay(protocol, *, checkpoint=None, progress=None):
    """Recompute from frozen inputs; reject damaged or incompatible protocols.

    No live project is required. Dependency versions are recorded, while a small
    numeric tolerance allows replay across compatible NumPy/SciPy releases.
    """
    if isinstance(protocol, str):
        from autopyro.model.audit import read_protocol
        try:
            protocol = read_protocol(protocol, checkpoint=checkpoint)
        except ValueError as exc:
            raise ValueError("Нарушена целостность протокола интерполяции") from exc
    if checkpoint:
        checkpoint()
    if progress:
        progress(15)
    if (not isinstance(protocol, dict)
            or (protocol.get("version"), protocol.get("algorithm")) not in
            ((1, LEGACY_ALGORITHM), (2, "autopyro.interpolation/2"),
             (3, LEGACY_ALGORITHM), (3, "autopyro.interpolation/2"),
             (1, ALGORITHM), (2, ALGORITHM), (3, ALGORITHM))):
        raise ValueError("Неподдерживаемая версия протокола интерполяции")
    try:
        if protocol.get("checksum") != _checksum(protocol, checkpoint=checkpoint):
            raise ValueError("Нарушена целостность протокола интерполяции")
        if progress:
            progress(30)
        parameters = protocol["parameters"]
        if protocol["policies"] != _policies_for_method(
            parameters["method"], parameters.get("max_extrapolation"), parameters.get("min_anchors"),
            parameters.get("duplicate_policy", "reject"),
        ):
            raise ValueError("Неподдерживаемая версия протокола интерполяции")
        def checked(items):
            for index, item in enumerate(items):
                if checkpoint and index % 256 == 0:
                    checkpoint()
                yield item

        defaults = {}
        if protocol["version"] == 3:
            defaults = protocol["measurement_defaults"]
            if type(defaults) is not dict or defaults != NULL_MEASUREMENT_DEFAULTS:
                raise ValueError("Неподдерживаемая версия протокола интерполяции")
            if any(not REQUIRED_MEASUREMENT_FIELDS.issubset(record) or record["id"] != mid
                   for mid, record in checked(protocol["measurements"].items())):
                raise ValueError("Нарушена целостность протокола интерполяции")
        snapshot = Project(
            id=protocol["project_id"],
            columns=[Column(**deepcopy(column)) for column in protocol["columns"]],
            rows=[Row(id=row["id"], values=deepcopy(row["values"])) for row in checked(protocol["rows"])],
        )
        constrained = "constraints" in parameters
        if constrained:
            from autopyro.model.measurements import Analysis, Measurement
            # These temporary records are read-only; nested provenance is never mutated.
            snapshot.science.measurements = {mid: Measurement(**({**defaults, **record} if defaults else record))
                                            for mid, record in checked(protocol["measurements"].items())}
            snapshot.science.analyses = {aid: Analysis(**record)
                                        for aid, record in checked(protocol["analyses"].items())}
            snapshot.science.working = {row["id"]: deepcopy(row["working"]) for row in checked(protocol["rows"])}
        supports, eligibility = [], {}
        if progress:
            progress(60)
        proposals = _calculate(snapshot, row_ids=set(snapshot.row_map()),
                               supports=supports, eligibility=eligibility,
                               legacy=protocol["algorithm"] != ALGORITHM, checkpoint=checkpoint, **parameters)
        expected = protocol["proposals"]
        if progress:
            progress(90)
        checks_eligibility = (constrained
                              or parameters.get("duplicate_policy") == "exclude")
        if (supports != protocol["supports"] or len(proposals) != len(expected)
                or (checks_eligibility and eligibility != protocol["eligibility"])):
            raise ValueError("Результат интерполяции не совпадает с протоколом")
        for actual, saved in checked(zip(proposals, expected, strict=True)):
            if (any(getattr(actual, key) != saved[key] for key in ("row_id", "x", "left_id", "right_id"))
                    or actual.extrapolated is not saved.get("extrapolated", False)
                    or (actual.value != saved["value"] if parameters.get("exact_values") and parameters["method"] == "nearest"
                        else not math.isclose(actual.value, saved["value"], rel_tol=1e-12, abs_tol=1e-12))):
                raise ValueError("Результат интерполяции не совпадает с протоколом")
        return proposals
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("Нарушена целостность протокола интерполяции") from exc
