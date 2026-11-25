"""Selection summary and charts based on report statistics."""

import base64
import io
from collections import defaultdict
from dataclasses import replace
from textwrap import shorten

from matplotlib.figure import Figure

from autopyro.analysis.qc import role
from autopyro.model.tabular import classify

from .well_profile import depth_columns, depth_datum_column, field_name


def overview(project, columns, rows, statistics, options):
    binding = (project.view.get("stratigraphy") or {}).get("binding") or {}
    depth_id, bottom_id = depth_columns(columns, binding, options.profile_depth_id)
    depth = columns.get(depth_id)
    depth_range = None
    datums = set()
    if depth and not depth.metadata.get("method"):
        datum_id = depth_datum_column(columns, depth_id)
        strict = depth_id == binding.get("depth_id") and binding.get("sample_mode") == "interval"
        bounds = []
        for row in rows:
            top = classify(row.values.get(depth_id), numeric=True)[1]
            bottom = classify(row.values.get(bottom_id), numeric=True)[1] if bottom_id else top
            if top is not None and (bottom is not None or not strict):
                bottom = top if bottom is None else bottom
                if bottom >= top:
                    bounds.append((top, bottom))
                    if datum_id:
                        datums.add(row.values.get(datum_id))
        if bounds and len(datums) <= 1:
            depth_range = {
                "column": depth.name, "minimum": min(top for top, _ in bounds),
                "maximum": max(bottom for _, bottom in bounds), "samples": len(bounds),
                "unit": depth.unit or (binding.get("depth_unit") if depth_id == binding.get("depth_id") else ""),
            }
    by_column = defaultdict(list)
    for row in statistics["summary"]:
        if row["n"]:
            by_column[row["column_id"]].append(row)
    indicators = []
    for cid in options.column_ids:
        column = columns[cid]
        entries = by_column[cid]
        if entries and (role(column) or role(replace(column, name=field_name(column)))):
            count = sum(row["n"] for row in entries)
            indicators.append({
                "column_id": cid, "name": column.name, "unit": column.unit, "n": count,
                "minimum": min(row["min"] for row in entries),
                "maximum": max(row["max"] for row in entries),
                "mean": sum(row["mean"] * (row["n"] / count) for row in entries),
            })
    return {"depth": depth_range, "mixed_depth_datums": len(datums) > 1, "indicators": indicators}


def comparison_figures(statistics, display, checkpoint, *, language="ru"):
    if not statistics["groups"]:
        return []
    groups = defaultdict(list)
    for row in statistics["summary"]:
        if row["n"]:
            groups[row["column_id"]].append(row)
    images = []
    for cid, rows in groups.items():
        if len(rows) < 2:
            continue
        for offset in range(0, len(rows), 12):
            checkpoint()
            page = rows[offset:offset + 12]
            name = f"Сопоставление медиан: {page[0]['property']}"
            if len(rows) > 12:
                name += f" ({offset + 1}–{offset + len(page)})"
            figure = Figure(figsize=(8, max(2.8, 1.5 + len(page) * .33)), layout="constrained")
            axis = figure.subplots()
            labels = [shorten(" / ".join(display(value) for value in row["group"]), width=55,
                              placeholder="…") + f" (n={row['n']})"
                      for row in page]
            axis.set_yticks(range(len(page)), labels, fontsize=8)
            axis.invert_yaxis()
            axis.scatter([row["median"] for row in page], range(len(page)), color="#235789")
            axis.set_xlabel(page[0]["unit"] or page[0]["property"])
            axis.grid(axis="x", alpha=.25)
            stream = io.BytesIO()
            figure.savefig(stream, format="png", dpi=140)
            figure.clear()
            images.append({
                "name": name, "column_id": cid, "groups": [row["group"] for row in page],
                "caption": ("Points show medians. n is the number of values used. Only groups with numeric data are shown."
                            if language == "en" else "Точки обозначают медианы. n — число значений в расчёте. "
                            "Показаны группы с числовыми данными."),
                "png": base64.b64encode(stream.getvalue()).decode("ascii"),
            })
    return images
