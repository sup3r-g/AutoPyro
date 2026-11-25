"""Operators, validation and traversal for table and named-view filters."""

from autopyro.model.core import parse_number
from autopyro.model.field_types import TYPED_KINDS, parse_typed_value
from ..i18n import tr


MEASUREMENT_STATUS_OPERATORS = {"non_detect", "below_limit", "qc_excluded"}
OPERATORS = [
    ("Содержит", "contains"),
    ("Равно", "eq"),
    ("Не равно", "ne"),
    (">", "gt"),
    ("≥", "ge"),
    ("<", "lt"),
    ("≤", "le"),
    ("Пусто", "empty"),
    ("Не пусто", "not_empty"),
    ("Не обнаружено (ND)", "non_detect"),
    ("Ниже предела обнаружения (<LOD)", "below_limit"),
    ("Исключено по QC", "qc_excluded"),
    ("Единица измерения равна", "unit_eq"),
    ("Единица измерения не указана", "unit_missing"),
    ("Результат устарел", "stale"),
    ("Результат актуален", "up_to_date"),
]

VALUELESS_OPERATORS = MEASUREMENT_STATUS_OPERATORS | {"empty", "not_empty", "unit_missing", "stale", "up_to_date"}


def validated_rules(project, rules):
    """Validate a UI draft for either the dock editor or the legacy dialog."""
    columns = {column.id: column for column in project.columns}
    return _validated_rules(columns, rules)


def _validated_rules(columns, rules):
    result = []
    for draft in rules:
        rule = dict(draft)
        if "rules" in rule:
            if not rule["rules"]:
                raise ValueError(tr('Добавьте хотя бы одно условие в каждую группу.'))
            rule["rules"] = _validated_rules(columns, rule["rules"])
            result.append(rule)
            continue
        column_name = rule.pop("column_name", None)
        column = columns.get(rule["column"])
        if column is None:
            if rule["column"]:
                raise ValueError(tr('Колонка недоступна: {0}', column_name or rule["column"]))
            raise ValueError(tr('Выберите колонку для каждого условия.'))
        operator = rule["operator"]
        if operator in VALUELESS_OPERATORS:
            rule["value"] = ""
        elif operator == "unit_eq":
            rule["value"] = rule["value"].strip()
            if not rule["value"]:
                raise ValueError(tr('Введите единицу измерения.'))
        elif operator != "contains" and column.kind == "number":
            rule["value"] = parse_number(rule["value"])
            if rule["value"] is None:
                raise ValueError(tr('Для числового сравнения введите значение'))
        elif operator != "contains" and column.kind in TYPED_KINDS:
            rule["value"] = parse_typed_value(rule["value"], column.kind)
            if rule["value"] is None:
                raise ValueError(tr('Введите значение для сравнения'))
        if column.kind in ("text", "category", "identifier") and operator in ("gt", "ge", "lt", "le"):
            raise ValueError(tr('Операторы > и < недоступны для текстовых колонок'))
        result.append(rule)
    return result


def leaf_rules(rules):
    for rule in rules:
        if "rules" in rule:
            yield from leaf_rules(rule["rules"])
        else:
            yield rule


def saved_sort_column(view, columns):
    if "sort_column_id" in view:
        cid = view["sort_column_id"]
        return next((i for i, column in enumerate(columns) if column.id == cid), -1)
    index = view.get("sort_column", -1)
    return index if type(index) is int and 0 <= index < len(columns) else -1


