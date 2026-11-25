"""Preparation optimizations retain source spelling, precision and legacy IDs."""
from uuid import NAMESPACE_URL, uuid5

import pytest

from autopyro.core import parse_number
from autopyro.measurements import migrated_id, parse_result


@pytest.mark.parametrize("raw,value,qualifier,limit,comparator", [
    (2**100, 2**100, "exact", None, ""),
    (2.5, 2.5, "exact", None, ""),
    (-0.0, -0.0, "exact", None, ""),
    ("0002,00", 2, "exact", None, ""),
    ("−2,5", -2.5, "exact", None, ""),
    ("  ", None, "missing", None, ""),
    (None, None, "missing", None, ""),
    ("н/о", None, "non_detect", None, ""),
    ("≤0,5", None, "below_limit", .5, "≤"),
    (">=2", None, "above_limit", 2, ">="),
    ("~2,5", 2.5, "approximate", None, ""),
    ("<bad", None, "unparsed", None, ""),
])
def test_numeric_parser_keeps_complete_result(raw, value, qualifier, limit, comparator):
    assert parse_result(raw, "number") == {
        "raw_value": raw, "value": value, "qualifier": qualifier,
        "limit": limit, "comparator": comparator}


@pytest.mark.parametrize("raw", [True, False, float("nan"), float("inf"), -float("inf")])
def test_numeric_parser_rejects_boolean_and_nonfinite_inputs(raw):
    with pytest.raises(ValueError):
        parse_result(raw, "number")
    with pytest.raises(ValueError):
        parse_number(raw)


def test_numeric_subclass_keeps_its_original_qualifier_spelling():
    class ReportedFloat(float):
        def __str__(self):
            return "<0.5"

    raw = ReportedFloat(.5)
    parsed = parse_result(raw, "number")
    assert parsed["raw_value"] is raw
    assert parsed["qualifier"] == "below_limit"
    assert parsed["limit"] == .5


@pytest.mark.parametrize("raw", ["001", 2**100, 2.5, True])
def test_exact_text_properties_are_not_coerced(raw):
    result = parse_result(raw, "text")
    assert result["raw_value"] is raw
    assert result["value"] is raw
    assert result["qualifier"] == "exact"


@pytest.mark.parametrize("project_id,kind,parts", [
    ("project", "source", ("row",)),
    ("project", "analysis", ("row",)),
    ("project", "measurement", ("row", "property")),
    ("проект:нефть", "measurement", ("проба 01", "δ¹³C")),
    ("", "measurement", ("", "")),
])
def test_migration_identifiers_match_the_existing_uuid5_format(project_id, kind, parts):
    name = ":".join(("autopyro", project_id, kind, *parts))
    assert migrated_id(project_id, kind, *parts) == str(uuid5(NAMESPACE_URL, name))
