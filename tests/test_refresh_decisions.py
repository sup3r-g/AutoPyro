"""Review policies are independent of how a change's group is looked up."""
from copy import deepcopy

import pytest

from autopyro.database.refresh import RefreshPlan, _validate_decisions


def plan_for(group, entry):
    groups = {"cells": [], "new_rows": [], "missing_rows": []}
    groups[group] = [{"id": "change", **deepcopy(entry)}]
    return RefreshPlan("", "", {}, {}, groups["cells"], groups["new_rows"], groups["missing_rows"], {}, [], "")


@pytest.mark.parametrize("group,entry,allowed", [
    ("cells", {"status": "source_changed", "action": "source"}, {"keep", "source"}),
    ("cells", {"status": "invalid_source", "action": "unresolved"}, {"keep"}),
    ("new_rows", {"invalid": [], "action": "add"}, {"add", "skip"}),
    ("new_rows", {"invalid": ["S2"], "action": "unresolved"}, {"skip"}),
    ("missing_rows", {"action": "unresolved"}, {"keep"}),
])
@pytest.mark.parametrize("action", ["keep", "source", "add", "skip", "unresolved", "unknown"])
def test_allowed_choices_match_review_policy(group, entry, allowed, action):
    plan = plan_for(group, entry)
    before = deepcopy(plan)
    decisions = {"change": action}
    if action in allowed:
        _validate_decisions(plan, decisions)
    else:
        with pytest.raises(ValueError, match="конфликты"):
            _validate_decisions(plan, decisions)
    assert plan == before and decisions == {"change": action}


@pytest.mark.parametrize("group,entry,ready", [
    ("cells", {"status": "source_changed", "action": "source"}, True),
    ("cells", {"status": "conflict", "action": "unresolved"}, False),
    ("cells", {"status": "invalid_source", "action": "unresolved"}, False),
    ("new_rows", {"invalid": [], "action": "add"}, True),
    ("new_rows", {"invalid": ["S2"], "action": "unresolved"}, False),
    ("missing_rows", {"action": "unresolved"}, False),
])
def test_defaults_never_resolve_pending_choices_implicitly(group, entry, ready):
    plan = plan_for(group, entry)
    if ready:
        _validate_decisions(plan, {})
    else:
        with pytest.raises(ValueError, match="конфликты"):
            _validate_decisions(plan, {})


def test_unknown_change_is_rejected_before_applying_choices():
    plan = plan_for("cells", {"status": "source_changed", "action": "source"})
    with pytest.raises(ValueError, match="неизвестному"):
        _validate_decisions(plan, {"other": "keep"})
