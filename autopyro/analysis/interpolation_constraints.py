"""Explicit sample partitions for interpolation; no inferred well or method names."""
CONTEXT_FIELDS = ("method", "laboratory", "instrument", "heating_program", "preparation", "fraction")


def well_column_ids(constraints):
    binding = constraints.get("stratigraphy", {}).get("binding", {})
    return list(dict.fromkeys(cid for cid in (constraints.get("well_id"), binding.get("well_id")) if cid))


def input_ids(constraints):
    ids = [constraints.get("well_id"), constraints.get("exclude_intervals_by")]
    stratigraphy = constraints.get("stratigraphy")
    binding = stratigraphy.get("binding") if isinstance(stratigraphy, dict) else None
    if not isinstance(binding, dict):
        binding = {}
    ids += [binding.get(key) for key in ("well_id", "depth_id")]
    if binding.get("sample_mode", "point") == "interval":
        ids.append(binding.get("bottom_id"))
    return list(dict.fromkeys(cid for cid in ids if cid))


def partition(project, row_ids, y_id, constraints, *, checkpoint=None):
    """Return immutable grouping keys and explicit per-row eligibility diagnostics.

    Unknown analytical metadata form their own context, never a wildcard matching
    a known method. A declared bottom column can exclude interval samples.
    Stratigraphic groups use interval UUIDs, not potentially repeated unit names.
    """
    columns = {column.id: column for column in project.columns}
    if any(cid not in columns for cid in input_ids(constraints)):
        raise ValueError("Отсутствуют колонки ограничений интерполяции")
    stratigraphy = constraints.get("stratigraphy", {})
    binding = stratigraphy.get("binding", {})
    mode = binding.get("sample_mode", "point")
    if mode not in ("point", "interval"):
        raise ValueError("Режим проб: point или interval")
    bottom_id = constraints.get("exclude_intervals_by")
    if mode == "interval" and bottom_id is None:
        raise ValueError("Интерполяция интервальных проб не поддерживается; середины интервалов не используются")
    diagnostics = {rid: {"status": "excluded_qc"}
                   for rid in constraints.get("excluded_row_ids", ()) if rid in row_ids}
    row_ids = row_ids - diagnostics.keys()
    if bottom_id is not None:
        bottom_column = columns.get(bottom_id)
        if bottom_column is None or bottom_column.kind != "number" or bottom_id in (binding.get("depth_id"), y_id):
            raise ValueError("Выберите отдельную числовую колонку подошвы пробы")
        if mode == "interval" and bottom_id != binding.get("bottom_id"):
            raise ValueError("Выберите колонку подошвы из привязки стратиграфии")
        for index, row in enumerate(project.rows):
            if checkpoint and index % 256 == 0:
                checkpoint()
            if row.id not in row_ids:
                continue
            bottom = row.values.get(bottom_id)
            if bottom is not None and not (isinstance(bottom, str) and not bottom.strip()):
                diagnostics[row.id] = {"status": "interval_sample_excluded"}
        row_ids = row_ids - diagnostics.keys()
        if mode == "interval":
            # Only explicitly accepted point rows reach the stratigraphic engine.
            binding = {**binding, "sample_mode": "point"}
    levels = constraints.get("levels", [])
    stratigraphy_results = {}
    if levels:
        from autopyro.geology.stratigraphy import LEVELS, evaluate_project
        if len(set(levels)) != len(levels) or any(level not in LEVELS for level in levels):
            raise ValueError("Некорректные уровни ограничений интерполяции")
        if not stratigraphy.get("scheme") or not binding:
            raise ValueError("Для ограничения по границам нужна схема и привязка стратиграфии")
        result = evaluate_project(project, row_ids, stratigraphy["scheme"], binding, checkpoint=checkpoint)
        stratigraphy_results = {item["row_id"]: item for item in result["results"]}
    well_ids = well_column_ids(constraints)
    keys = {}
    for index, row in enumerate(project.rows):
        if checkpoint and index % 256 == 0:
            checkpoint()
        if row.id not in row_ids:
            continue
        key = []
        diagnostic = {"status": "eligible"}
        wells = [row.values.get(cid) for cid in well_ids]
        if any(well is None or not str(well).strip() for well in wells):
            diagnostics[row.id] = {"status": "missing_wellbore"}
            continue
        key.extend(wells)
        if constraints.get("analytical_context", False):
            measurement = project.science.current(row.id, y_id)
            analysis = project.science.analyses.get(measurement.analysis_id) if measurement else None
            context = [getattr(analysis, field, "") for field in CONTEXT_FIELDS]
            context += [measurement.unit if measurement else columns[y_id].unit,
                        measurement.basis if measurement else columns[y_id].metadata.get("basis", "")]
            key.extend(context)
            diagnostic["analytical_context"] = dict(zip((*CONTEXT_FIELDS, "unit", "basis"), context, strict=True))
            diagnostic["method_known"] = bool(context[0])
        for level in levels:
            membership = stratigraphy_results[row.id]["levels"][level]
            if membership["status"] != "matched":
                diagnostic.update(status="stratigraphy_excluded", level=level,
                                  reason=membership["status"])
                break
            key.append(membership["interval_ids"][0])
        diagnostics[row.id] = diagnostic
        if diagnostic["status"] == "eligible":
            keys[row.id] = tuple(key)
    return keys, diagnostics
