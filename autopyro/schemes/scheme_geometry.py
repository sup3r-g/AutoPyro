"""Draft geometry operations. Coordinates stay in the scheme's analysis space."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

from shapely import STRtree, union_all
from shapely.errors import ShapelyError
from shapely.geometry import box

from .interpretation import transform, validate_scheme


@dataclass(frozen=True)
class GeometryPath:
    indices: tuple[int, ...]
    closed: bool
    label: str


def editable_entry(entry):
    result = deepcopy(entry)

    def lists(value):
        return (
            [lists(item) for item in value]
            if isinstance(value, (list, tuple))
            else value
        )

    result["geometry"]["coordinates"] = lists(result["geometry"]["coordinates"])
    return result


def geometry_paths(entry):
    geometry = entry["geometry"]
    kind, coordinates = geometry["type"], geometry["coordinates"]
    if kind == "Polygon":
        return [
            GeometryPath(
                (ring,), True, "Внешний контур" if ring == 0 else f"Отверстие {ring}"
            )
            for ring in range(len(coordinates))
        ]
    if kind == "MultiPolygon":
        return [
            GeometryPath(
                (part, ring),
                True,
                f"Часть {part + 1}: "
                + ("внешний контур" if ring == 0 else f"отверстие {ring}"),
            )
            for part, polygon in enumerate(coordinates)
            for ring in range(len(polygon))
        ]
    if kind == "LineString":
        return [GeometryPath((), False, "Линия")]
    if kind == "MultiLineString":
        return [
            GeometryPath((part,), False, f"Линия {part + 1}")
            for part in range(len(coordinates))
        ]
    raise ValueError("Неподдерживаемый тип геометрии")


def vertices(entry, path):
    coordinates = entry["geometry"]["coordinates"]
    for index in path.indices:
        coordinates = coordinates[index]
    values = [list(point) for point in coordinates]
    if path.closed and values and values[0] == values[-1]:
        values.pop()
    return values


def checked_vertices(values, closed):
    result = []
    for point in values:
        if len(point) != 2 or any(isinstance(v, bool) for v in point):
            raise ValueError("Вершина должна содержать две числовые координаты")
        try:
            x, y = map(float, point)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("Нечисловая координата") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Координаты должны быть конечными")
        result.append([x, y])
    if closed and len(result) > 1 and result[0] == result[-1]:
        result.pop()
    if len(result) < (3 if closed else 2):
        raise ValueError("Нужно минимум три вершины контура или две вершины линии")
    return result


def replace_vertices(entry, path, values):
    """Return a new entry; validity is checked separately, never repaired silently."""
    result = editable_entry(entry)
    values = checked_vertices(values, path.closed)
    if path.closed:
        values.append(values[0].copy())
    if not path.indices:
        result["geometry"]["coordinates"] = values
    else:
        container = result["geometry"]["coordinates"]
        for index in path.indices[:-1]:
            container = container[index]
        container[path.indices[-1]] = values
    return result


def add_geometry(scheme, kind, points, object_id, label, color="#0072B2"):
    if kind not in ("Polygon", "LineString"):
        raise ValueError("Создавайте Polygon или LineString")
    if not object_id or any(
        entry["id"] == object_id
        for group in ("regions", "lines")
        for entry in scheme.get(group, [])
    ):
        raise ValueError("Идентификатор объекта должен быть уникальным")
    values = checked_vertices(points, kind == "Polygon")
    if kind == "Polygon":
        values = [values + [values[0].copy()]]
    result = deepcopy(scheme)
    result.setdefault("regions" if kind == "Polygon" else "lines", []).append(
        {
            "id": object_id,
            "label": label,
            "color": color,
            "geometry": {"type": kind, "coordinates": values},
        }
    )
    return result


def add_ring_or_part(entry, path, values, *, hole=False):
    result = editable_entry(entry)
    geometry = result["geometry"]
    kind = geometry["type"]
    polygon = kind in ("Polygon", "MultiPolygon")
    if hole and not polygon:
        raise ValueError("Отверстие можно добавить только к полигону")
    values = checked_vertices(values, polygon)
    if polygon:
        values.append(values[0].copy())
    if hole:
        target = (
            geometry["coordinates"]
            if kind == "Polygon"
            else geometry["coordinates"][path.indices[0]]
        )
        target.append(values)
    elif polygon:
        if kind == "Polygon":
            geometry["type"] = "MultiPolygon"
            geometry["coordinates"] = [geometry["coordinates"], [values]]
        else:
            geometry["coordinates"].append([values])
    elif kind == "LineString":
        geometry["type"] = "MultiLineString"
        geometry["coordinates"] = [geometry["coordinates"], values]
    else:
        geometry["coordinates"].append(values)
    return result


def remove_ring_or_part(entry, path):
    result = editable_entry(entry)
    geometry = result["geometry"]
    kind = geometry["type"]
    if kind == "Polygon" and path.indices[0] > 0:
        geometry["coordinates"].pop(path.indices[0])
    elif kind == "MultiPolygon":
        part, ring = path.indices
        if ring > 0:
            geometry["coordinates"][part].pop(ring)
        elif len(geometry["coordinates"]) > 1:
            geometry["coordinates"].pop(part)
        else:
            raise ValueError("Последнюю внешнюю часть удаляйте вместе с объектом")
    elif kind == "MultiLineString" and len(geometry["coordinates"]) > 1:
        geometry["coordinates"].pop(path.indices[0])
    else:
        raise ValueError("Для удаления внешнего контура используйте «Удалить объект»")
    return result


def coverage_domain(scheme):
    """Return the declared finite rectangular domain in analysis coordinates."""
    limits = []
    for bound in ("minimum", "maximum"):
        for key in ("x", "y"):
            axis = scheme[key]
            value = axis.get(bound)
            if value is None:
                return None
            try:
                limits.append(transform(value, axis))
            except (ValueError, OverflowError):
                return None
    if not all(map(math.isfinite, limits)):
        return None
    domain = box(*limits)
    return domain if 0 < domain.area < math.inf else None


def geometry_diagnostics(scheme):
    try:
        regions, lines = validate_scheme(scheme)
    except (ValueError, TypeError, KeyError, ShapelyError) as exc:
        return {"errors": [str(exc)], "warnings": [], "overlaps": []}
    overlaps, warnings = [], []
    geometries = [geometry for _, geometry in regions]
    try:
        tree = STRtree(geometries)
        for i, (first, a) in enumerate(regions):
            for j in sorted(tree.query(a)):
                if j <= i:
                    continue
                second, b = regions[j]
                area = a.intersection(b).area
                if area > 0:
                    overlaps.append({"first": first["id"], "second": second["id"], "area": area})
                    warnings.append(
                        f"Перекрытие {first['label']} / {second['label']}: площадь {area:.6g} в analysis²"
                    )
        domain = coverage_domain(scheme) if regions and scheme.get("strategy", "regions") == "regions" else None
        coverage = None
        if domain is not None:
            union = union_all(geometries)
            gaps = domain.difference(union)
            outside = union.difference(domain)
            coverage = {"domain": domain, "gaps": gaps, "outside": outside}
    except ShapelyError as exc:
        return {"errors": [str(exc)], "warnings": warnings, "overlaps": overlaps}
    for entry, line in lines:
        if not line.is_simple:
            warnings.append(f"Линия {entry['label']} пересекает сама себя")
    result = {"errors": [], "warnings": warnings, "overlaps": overlaps}
    if coverage is not None:
        result["coverage"] = coverage
    return result


def next_revision(revision):
    return str(int(revision) + 1) if str(revision).isdigit() else f"{revision}.1"


def object_at(scheme, collection, object_id):
    return next(
        entry for entry in scheme.get(collection, []) if entry["id"] == object_id
    )


def replace_object(scheme, collection, entry):
    result = deepcopy(scheme)
    index = next(
        i for i, item in enumerate(result[collection]) if item["id"] == entry["id"]
    )
    result[collection][index] = deepcopy(entry)
    return result
