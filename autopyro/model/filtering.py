"""Scalar filter evaluation shared by saved plot filters and table fallback."""
import operator as comparison


def leaf_rules(rules):
    for rule in rules:
        if "rules" in rule:
            yield from leaf_rules(rule["rules"])
        else:
            yield rule


def matches_rules(project, row, rules, combine="AND"):
    def leaf(rule):
        cid, op, target = rule["column"], rule["operator"], rule.get("value", "")
        column = next((c for c in project.columns if c.id == cid), None)
        if column is None:
            raise ValueError("Колонка локального фильтра удалена")
        value = row.values.get(cid)
        metadata = project.view.get("plot_filter_measurements")
        if metadata is None:
            m = project.science.current(row.id, cid)
            info = dict(unit=m.unit, qualifier=m.qualifier, raw=m.raw_value) if m else {}
            from autopyro.analysis.qc import is_excluded
            excluded = any(is_excluded(project, row.id, cid, scope)
                           for scope in project.view.get("qc_exclusions", {}))
        else:
            info = metadata.get(cid, {}).get(row.id, {})
            excluded = info.get("excluded", False)
        if op == "qc_excluded":
            return excluded
        if op in ("non_detect", "below_limit"):
            return (info.get("qualifier") == op
                    or op == "below_limit" and str(info.get("raw", "")).strip().casefold() == "<lod")
        if op in ("stale", "up_to_date"):
            stale = project.view.get("plot_filter_stale", {}).get(cid, column.metadata.get("stale"))
            return stale is (op == "stale") and value not in (None, "")
        if op in ("unit_eq", "unit_missing"):
            if "unit" not in info and value in (None, ""):
                return False
            return (info.get("unit", column.unit) or "").strip() == ("" if op == "unit_missing" else target)
        if op == "empty":
            return value is None or value == ""
        if value is None:
            return False
        if op == "not_empty":
            return value != ""
        if op == "contains":
            return str(target).casefold() in str(value).casefold()
        if op in ("eq", "ne", "gt", "ge", "lt", "le"):
            try:
                return getattr(comparison, op)(value, target)
            except TypeError:
                return False
        raise ValueError("Неизвестное условие локального фильтра")

    values = []
    for rule in rules:
        result = (matches_rules(project, row, rule["rules"], rule.get("combine", "AND"))
                  if "rules" in rule else leaf(rule))
        values.append(not result if rule.get("negate") else result)
    return all(values) if combine == "AND" else any(values)
