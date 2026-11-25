"""Safety partitions separate wellbores, methods and actual stratigraphic intervals."""
from copy import deepcopy
import json

import pytest

from autopyro.core import Column, Project, Row
from autopyro.interpolation import prepare, propose, replay
from autopyro.io import build_project
from autopyro import interpolation


@pytest.fixture
def data():
    headers = ["Well", "Bore", "Depth", "Y"]
    project = build_project(headers,
                            [["A", "A-1", "0", "2"], ["A", "A-1", "5", ""],
                             ["A", "A-1", "10", "6"], ["A", "A-1", "15", ""],
                             ["A", "A-1", "20", "10"]], list(range(2, 7)),
                            [(i, name, "text" if i < 2 else "number", "m" if i == 2 else "")
                             for i, name in enumerate(headers)], {})
    project.sync_measurements()
    return project


def options(project):
    well, bore, depth, y = project.columns[:4]
    return dict(x_id=depth.id, y_id=y.id, group_ids=[well.id], row_ids=set(project.row_map()),
                constraints={"well_id": bore.id, "analytical_context": True})


def stratigraphy(project):
    _, bore, depth, _ = project.columns[:4]
    return {"binding": {"well_id": bore.id, "depth_id": depth.id, "sample_mode": "point",
                        "depth_unit": "m", "depth_reference": "MD RKB"},
            "scheme": {"format": "autopyro.stratigraphy", "version": 1, "id": "demo",
                       "revision": "1", "name": "Test", "source": "Synthetic", "depth_unit": "m",
                       "depth_reference": "MD RKB", "intervals": [
                           {"id": "upper", "wellbore": "A-1", "name": "Repeated name", "level": "FORMATION", "top": -1, "bottom": 8},
                           {"id": "lower", "wellbore": "A-1", "name": "Repeated name", "level": "FORMATION", "top": 8, "bottom": 30}]}}


@pytest.mark.parametrize("field", ["method", "laboratory", "instrument", "heating_program", "preparation", "fraction"])
@pytest.mark.parametrize("compact", [False, True])
def test_distinct_analytical_contexts_never_supply_anchors(data, field, compact, monkeypatch):
    if compact:
        monkeypatch.setattr(interpolation, "COMPACT_MEASUREMENT_THRESHOLD", 0)
    cfg = options(data)
    assert len(propose(data, **cfg)) == 2
    y = data.columns[3]
    result = data.science.current(data.rows[0].id, y.id)
    setattr(data.science.analyses[result.analysis_id], field, "Other context")
    proposals, protocol = prepare(data, **cfg)
    assert [p.row_id for p in proposals] == [data.rows[3].id]
    assert replay(json.loads(json.dumps(protocol))) == proposals


@pytest.mark.parametrize("field", ["unit", "basis"])
def test_distinct_measurement_units_or_basis_never_supply_anchors(data, field):
    result = data.science.current(data.rows[0].id, data.columns[3].id)
    setattr(result, field, "Other")
    assert [p.row_id for p in propose(data, **options(data))] == [data.rows[3].id]


def test_unknown_context_is_not_a_wildcard(data):
    y = data.columns[3]
    for row in data.rows:
        measurement = data.science.current(row.id, y.id)
        data.science.analyses[measurement.analysis_id].method = "Known"
    target = data.science.current(data.rows[1].id, y.id)
    data.science.analyses[target.analysis_id].method = ""
    proposals, protocol = prepare(data, **options(data))
    assert [p.row_id for p in proposals] == [data.rows[3].id]
    assert protocol["eligibility"][data.rows[1].id]["method_known"] is False


def test_nearest_preserves_integer_y_in_snapshot_and_replay(data):
    y = data.columns[3]
    value = 2**53 + 1
    data.rows[0].values[y.id] = value
    data.sync_measurements(edited_cells={data.rows[0].id: {y.id}})
    proposals, protocol = prepare(data, method="nearest", **options(data))
    assert proposals[0].value == value
    assert type(proposals[0].value) is int
    assert replay(json.loads(json.dumps(protocol))) == proposals
    with pytest.raises(ValueError, match="Y"):
        prepare(data, method="linear", **options(data))


def test_validation_scores_observe_anchor_minimum_and_extrapolation_limits(data):
    _, protocol = prepare(data, min_anchors=3, max_extrapolation=10, max_gap=20, **options(data))
    checked, common, scores = interpolation.validation_scores(protocol)
    assert (checked, common) == (1, 0)
    assert scores["nearest"] == scores["extrapolation"] == (0, None, None)

    _, protocol = prepare(data, min_anchors=2, max_extrapolation=10, max_gap=20, **options(data))
    _, _, scores = interpolation.validation_scores(protocol)
    assert scores["nearest"][0] == 1
    assert scores["extrapolation"] == (2, 0, 0)
    _, protocol = prepare(data, max_extrapolation=5, max_gap=20, **options(data))
    assert interpolation.validation_scores(protocol)[2]["extrapolation"] == (0, None, None)


def test_explicit_and_bound_wellbores_are_both_respected(data):
    cfg = options(data)
    data.rows[0].values[data.columns[1].id] = "A-2"
    assert [p.row_id for p in propose(data, **cfg)] == [data.rows[3].id]
    cfg["constraints"]["well_id"] = data.columns[0].id  # General well must not override bound bore.
    cfg["constraints"]["stratigraphy"] = stratigraphy(data)
    assert [p.row_id for p in propose(data, **cfg)] == [data.rows[3].id]


def test_linear_extrapolation_is_opt_in_bounded_and_replayed():
    records = [(-2, ""), (-1, ""), (-0.5, "ND"), (0, "10"), (4, "18"),
               (5, "20"), (6, ""), (7, "")]
    project = build_project(
        ["Well", "Bore", "Depth", "Y"], [["A", "A-1", str(x), y] for x, y in records],
        list(range(2, len(records) + 2)),
        [(0, "Well", "text", ""), (1, "Bore", "text", ""),
         (2, "Depth", "number", "m"), (3, "Y", "number", "")], {},
    )
    config = options(project)
    assert propose(project, **config) == []
    proposals, protocol = prepare(project, max_gap=4, max_extrapolation=1, **config)
    assert [(item.x, item.value, item.extrapolated) for item in proposals] == [(-1, 8, True), (6, 22, True)]
    assert (proposals[0].left_id, proposals[0].right_id) == (project.rows[3].id, project.rows[4].id)
    assert (proposals[1].left_id, proposals[1].right_id) == (project.rows[4].id, project.rows[5].id)
    assert protocol["parameters"]["max_extrapolation"] == 1
    assert protocol["policies"]["extrapolation"] is True
    assert replay(json.loads(json.dumps(protocol))) == proposals
    constrained, detailed = prepare(project, max_gap=2, max_extrapolation=1, **config)
    assert [item.x for item in constrained] == [6]
    for row, reason, actual, limit in ((0, "extrapolation_distance", 2, 1), (1, "extrapolation_gap", 4, 2)):
        record = detailed["eligibility"][project.rows[row].id]
        assert (record["status"], record["actual"], record["limit"]) == (reason, actual, limit)
    assert detailed["eligibility"][project.rows[6].id]["status"] == "eligible"
    assert replay(detailed) == constrained
    historical = deepcopy(detailed)
    historical["parameters"].pop("limit_diagnostics")
    for record in historical["eligibility"].values():
        record.pop("actual", None)
        record.pop("limit", None)
    historical["checksum"] = interpolation._checksum(historical)
    assert replay(historical) == constrained
    historical["parameters"].pop("extrapolation_diagnostics")
    for record in historical["eligibility"].values():
        if record["status"] in {"extrapolation_distance", "extrapolation_gap"}:
            record["status"] = "outside_range"
    historical["checksum"] = interpolation._checksum(historical)
    assert replay(historical) == constrained
    assert propose(project, max_gap=4, max_extrapolation=1, min_anchors=4, **config) == []
    with pytest.raises(ValueError, match="только для линейного"):
        propose(project, method="spline", max_extrapolation=1, **config)


@pytest.mark.parametrize("constrained", [False, True])
def test_duplicate_policy_excludes_all_matching_rows_only_within_each_group(constrained):
    records = [
        ("A", 0, "0"), ("A", 5, "100"), ("A", 5, "200"), ("A", 6, ""), ("A", 6, ""),
        ("A", 7, ""), ("A", 10, "10"), ("B", 0, "0"), ("B", 5, ""), ("B", 10, "20"),
        ("C", 5, "0"), ("C", 5, ""),
    ]
    project = build_project(
        ["Well", "Bore", "Depth", "Y"], [[well, well, str(x), y] for well, x, y in records],
        list(range(2, len(records) + 2)),
        [(0, "Well", "text", ""), (1, "Bore", "text", ""),
         (2, "Depth", "number", "m"), (3, "Y", "number", "")], {},
    )
    config = options(project)
    if not constrained:
        config.pop("constraints")
    with pytest.raises(ValueError, match="Повторные координаты"):
        propose(project, **config)

    proposals, protocol = prepare(project, duplicate_policy="exclude", **config)
    assert [(item.row_id, item.value) for item in proposals] == [
        (project.rows[5].id, 7), (project.rows[8].id, 10),
    ]
    assert {rid for rid, diagnostic in protocol["eligibility"].items()
            if diagnostic["status"] == "duplicate_coordinate"} == {
        project.rows[i].id for i in (1, 2, 3, 4, 10, 11)
    }
    assert protocol["policies"]["duplicates"] == "exclude"
    assert replay(json.loads(json.dumps(protocol))) == proposals


def test_cubic_spline_requires_four_anchors_per_contiguous_span_and_replays():
    records = [
        (-1, None), (0, 0), (0.5, None), (1, 1), (2, 0), (3, 1),
        (8, None), (10, 21), (10.5, None), (11, 23), (12, 25), (13, None),
    ]
    columns = ["Well", "Bore", "Depth", "Y"]
    project = build_project(columns, [["A", "A-1", str(x), "" if y is None else str(y)] for x, y in records],
                            list(range(2, len(records) + 2)),
                            [(0, "Well", "text", ""), (1, "Bore", "text", ""),
                             (2, "Depth", "number", "m"), (3, "Y", "number", "")], {})
    config = options(project)
    proposals, protocol = prepare(project, method="spline", max_gap=2, **config)
    only, = proposals
    assert (only.row_id, only.left_id, only.right_id) == (
        project.rows[2].id, project.rows[1].id, project.rows[3].id,
    )
    assert only.value == pytest.approx(0.75)
    record = protocol["eligibility"][project.rows[8].id]
    assert (record["status"], record["anchors_found"], record["anchors_required"]) == (
        "insufficient_spline_anchors", 3, 4,
    )
    assert protocol["policies"]["spline"] == {"boundary": "natural", "min_anchors": 4, "split_on_max_gap": True}
    assert len(protocol["supports"][0]["anchors"]) == 7
    assert replay(json.loads(json.dumps(protocol))) == proposals
    stricter, frozen = prepare(project, method="spline", max_gap=2, min_anchors=5, **config)
    assert stricter == replay(frozen) == []
    assert frozen["policies"]["spline"]["min_anchors"] == 5
    assert frozen["eligibility"][project.rows[2].id]["status"] == "insufficient_spline_anchors"
    pchip, pchip_protocol = prepare(project, method="pchip", max_gap=2, min_anchors=4, **config)
    record = pchip_protocol["eligibility"][project.rows[8].id]
    assert (record["status"], record["anchors_found"], record["anchors_required"]) == (
        "insufficient_segment_anchors", 3, 4,
    )
    assert replay(pchip_protocol) == pchip
    pchip_protocol["parameters"].pop("anchor_diagnostics")
    for diagnostic in pchip_protocol["eligibility"].values():
        if diagnostic["status"] == "insufficient_segment_anchors":
            diagnostic["status"] = "insufficient_anchors"
            diagnostic.pop("anchors_found")
            diagnostic.pop("anchors_required")
    pchip_protocol["checksum"] = interpolation._checksum(pchip_protocol)
    assert replay(pchip_protocol) == pchip
    with pytest.raises(ValueError, match="не ниже порога"):
        propose(project, method="spline", min_anchors=3, **config)


def test_minimum_anchors_for_linear_interpolation_is_frozen_and_diagnosed(data):
    config = options(data)
    assert len(propose(data, min_anchors=3, **config)) == 2
    proposals, protocol = prepare(data, min_anchors=4, **config)
    assert proposals == replay(protocol) == []
    assert protocol["parameters"]["min_anchors"] == 4
    for position in (1, 3):
        record = protocol["eligibility"][data.rows[position].id]
        assert (record["status"], record["anchors_found"], record["anchors_required"]) == (
            "insufficient_anchors", 3, 4,
        )


def test_interpolation_rejects_lossy_integer_coordinates_and_reports_large_gaps(data):
    config = options(data)
    proposals, protocol = prepare(data, max_gap=2, **config)
    assert proposals == replay(protocol) == []
    record = protocol["eligibility"][data.rows[1].id]
    assert (record["status"], record["actual"], record["limit"]) == ("gap_too_large", 10, 2)
    depth_id = config["x_id"]
    for i, row in enumerate(data.rows):
        row.values[depth_id] = 2**53 + i
    with pytest.raises(ValueError, match="теряют точность"):
        propose(data, **config)


@pytest.mark.parametrize("method", ["linear", "nearest", "pchip"])
@pytest.mark.parametrize("compact", [False, True])
def test_stratigraphic_interval_ids_not_names_partition_anchors(data, method, tmp_path, compact, monkeypatch):
    if compact:
        monkeypatch.setattr(interpolation, "COMPACT_MEASUREMENT_THRESHOLD", 0)
    cfg = options(data)
    cfg.update(method=method)
    cfg["constraints"].update(stratigraphy=stratigraphy(data), levels=["FORMATION"])
    proposals, protocol = prepare(data, **cfg)
    assert [p.row_id for p in proposals] == [data.rows[3].id]
    assert protocol["version"] == (3 if compact else 2)
    before_replay = deepcopy(protocol)
    assert replay(protocol) == proposals
    assert protocol == before_replay
    frozen = deepcopy(protocol)
    cfg["constraints"]["stratigraphy"]["scheme"]["intervals"].clear()
    assert protocol == frozen
    # Definitions, analytical contexts and all required depth inputs survive storage.
    data.columns[3].metadata["interpolation_protocol"] = protocol
    path = tmp_path / "restricted.apyro"
    data.save(path)
    assert replay(Project.load(path).columns[3].metadata["interpolation_protocol"]) == proposals


@pytest.mark.parametrize("status", ["outside", "ambiguous"])
def test_unassigned_or_ambiguous_stratigraphy_is_excluded_with_reason(data, status):
    cfg = options(data)
    scheme = stratigraphy(data)
    if status == "outside":
        scheme["scheme"]["intervals"] = []
        scheme["scheme"]["intervals"].append({"id": "elsewhere", "wellbore": "A-1", "name": "Elsewhere",
                                               "level": "FORMATION", "top": 100, "bottom": 200})
    else:
        scheme["scheme"]["intervals"][0]["bottom"] = 30
    cfg["constraints"].update(stratigraphy=scheme, levels=["FORMATION"])
    proposals, protocol = prepare(data, **cfg)
    target = protocol["eligibility"][data.rows[3].id]
    assert target["status"] == "stratigraphy_excluded"
    assert target["reason"] == status
    assert proposals == []
    assert replay(protocol) == []


def test_interval_samples_are_rejected_even_without_selected_boundaries(data):
    cfg = options(data)
    scheme = stratigraphy(data)
    scheme["binding"].update(sample_mode="interval", bottom_id=data.columns[3].id)
    cfg["constraints"]["stratigraphy"] = scheme
    with pytest.raises(ValueError, match="Интерполяция интервальных"):
        prepare(data, **cfg)


@pytest.mark.parametrize("with_binding", [False, True])
def test_mixed_scope_excludes_explicit_bottom_values_and_replays(data, with_binding):
    well, bore, depth, y = data.columns
    bottom = Column("Bottom", "number", unit="m")
    data.columns.append(bottom)
    for row in data.rows:
        row.values[bottom.id] = None
    # A qualified bottom is still populated and is excluded conservatively.
    data.rows[1].values[bottom.id] = "ND"
    interval_anchor = Row({well.id: "A", bore.id: "A-1", depth.id: 12, y.id: 100, bottom.id: 14})
    data.rows.append(interval_anchor)
    data.sync_measurements()
    config = options(data)
    config["constraints"]["exclude_intervals_by"] = bottom.id
    if with_binding:
        state = stratigraphy(data)
        state["binding"].update(sample_mode="interval", bottom_id=bottom.id)
        config["constraints"].update(stratigraphy=state, levels=["FORMATION"])

    proposals, protocol = prepare(data, **config)
    assert [(item.row_id, item.value) for item in proposals] == [(data.rows[3].id, 8)]
    assert protocol["eligibility"][interval_anchor.id]["status"] == "interval_sample_excluded"
    assert protocol["eligibility"][data.rows[1].id]["status"] == "interval_sample_excluded"
    assert bottom.id in {column["id"] for column in protocol["columns"]}
    assert protocol["parameters"]["constraints"]["exclude_intervals_by"] == bottom.id
    assert replay(json.loads(json.dumps(protocol))) == proposals


def test_bound_depth_is_frozen_even_when_x_is_another_property(data):
    from autopyro.core import Column

    cfg = options(data)
    depth = data.columns[2]
    x = Column("Independent X", "number")
    data.columns.append(x)
    for row in data.rows:
        row.values[x.id] = row.values[depth.id] * 2
    cfg["x_id"] = x.id
    cfg["constraints"].update(stratigraphy=stratigraphy(data), levels=["FORMATION"])
    proposals, protocol = prepare(data, **cfg)
    assert [p.row_id for p in proposals] == [data.rows[3].id]
    assert depth.id in {c["id"] for c in protocol["columns"]}
    data.rows[3].values[depth.id] = 0
    assert replay(protocol) == proposals


def test_boundaries_delegate_depth_units_and_datum_validation(data):
    cfg = options(data)
    cfg["constraints"].update(stratigraphy=stratigraphy(data), levels=["FORMATION"])
    scheme = cfg["constraints"]["stratigraphy"]["scheme"]
    scheme["depth_unit"] = "ft"
    for item in scheme["intervals"]:
        item["top"] /= .3048
        item["bottom"] /= .3048
    proposals, protocol = prepare(data, **cfg)
    assert [p.row_id for p in proposals] == [data.rows[3].id]
    assert replay(protocol) == proposals
    cfg["constraints"]["stratigraphy"]["binding"]["depth_reference"] = "TVD SS"
    with pytest.raises(ValueError):
        prepare(data, **cfg)


def test_missing_well_is_excluded_not_grouped_with_other_missing_wells(data):
    cfg = options(data)
    bore = data.columns[1].id
    for row in data.rows:
        row.values[bore] = None
    proposals, protocol = prepare(data, **cfg)
    assert proposals == []
    assert {d["status"] for d in protocol["eligibility"].values()} == {"missing_wellbore"}
    assert replay(protocol) == []


def test_gui_defaults_boundaries_diagnostics_creation_and_replay(data, qtbot):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDialogButtonBox
    from autopyro.ui.window import MainWindow

    data.view["stratigraphy"] = stratigraphy(data)
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(data)
    window.may_replace = lambda: True
    try:
        window.interpolate()
        dialog = window.interpolation_window
        dialog.x.setCurrentIndex(dialog.x.findData(data.columns[2].id))
        dialog.y.setCurrentIndex(dialog.y.findData(data.columns[3].id))
        dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
        assert dialog.well.currentData() == data.columns[1].id
        assert dialog.boundaries.item(0).checkState() == Qt.CheckState.Checked
        dialog.calculate()
        assert [p.row_id for p in dialog.proposals] == [data.rows[3].id]
        assert dialog.diagnostic_model.rowCount() == 5
        dialog.apply()
        created = data.columns[-1]
        assert data.rows[1].values[created.id] is None
        assert data.rows[3].values[created.id] == 8
        assert replay(created.metadata["interpolation_protocol"])[0].value == 8
        window.model.undo.undo()
        assert len(data.columns) == 4
        window.model.undo.redo()
        assert data.rows[3].values[created.id] == 8
        # Declaring interval samples blocks the next preview, rather than using midpoints.
        data.view["stratigraphy"]["binding"].update(sample_mode="interval", bottom_id=data.columns[3].id)
        dialog.calculate()
        assert not dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
        assert len(data.columns) == 5
    finally:
        window.dirty = False
        window.close()


def test_pchip_respects_segment_boundaries_and_replays_legacy(data, monkeypatch):
    cfg = options(data)
    cfg.update(method="pchip", max_gap=10, constraints=None)
    x_id, y_id = cfg["x_id"], cfg["y_id"]
    data.rows[-1].values.update({x_id: 100, y_id: 2})
    proposals, protocol = prepare(data, **cfg)
    assert [p.value for p in proposals] == [4]
    assert replay(protocol) == proposals

    calculate = interpolation._calculate
    with monkeypatch.context() as patch:
        patch.setattr(interpolation, "ALGORITHM", interpolation.LEGACY_ALGORITHM)
        patch.setattr(interpolation, "_calculate", lambda *args, **kwargs: calculate(*args, legacy=True, **kwargs))
        historical, old_protocol = prepare(data, **cfg)
    assert historical[0].value != proposals[0].value
    assert replay(old_protocol) == historical


@pytest.mark.parametrize(("value", "reason"), [
    ("ND", "y_non_detect"), ("<0.1", "y_censored"), ("unparsed", "y_other"),
])
def test_numeric_strings_and_blank_targets_share_numeric_interpolation(data, value, reason):
    cfg = options(data)
    x_id, y_id = cfg["x_id"], cfg["y_id"]
    data.rows[0].values[x_id] = "0"
    data.rows[1].values.update({x_id: "5,0", y_id: " "})
    data.rows[2].values[y_id] = "6,0"
    data.rows[3].values[y_id] = value
    proposals, protocol = prepare(data, **cfg)
    assert [(p.x, p.value) for p in proposals] == [(5, 4)]
    record = protocol["eligibility"][data.rows[3].id]
    assert record["status"] == reason
    assert replay(protocol) == proposals
    protocol["parameters"].pop("qualified_y_diagnostics")
    record["status"] = "qualified_y"
    protocol["checksum"] = interpolation._checksum(protocol)
    assert replay(protocol) == proposals


def test_preview_filters_each_qualified_y_reason(data, qtbot):
    from autopyro.ui.dialogs.interpolation import InterpolationDialog, diagnostic_labels
    from autopyro.ui.i18n import tr

    x_id, y_id = (column.id for column in data.columns[2:])
    cases = [("ND", "y_non_detect"), (">1", "y_censored"), ("unparsed", "y_other")]
    for row, (value, _) in zip(data.rows[:3], cases, strict=True):
        row.values[y_id] = value
    dialog = InterpolationDialog(data, set(data.row_map()))
    qtbot.addWidget(dialog)
    dialog.x.setCurrentIndex(dialog.x.findData(x_id))
    dialog.y.setCurrentIndex(dialog.y.findData(y_id))
    dialog.calculate()
    labels = diagnostic_labels()
    for row, (_, reason) in zip(data.rows[:3], cases, strict=True):
        dialog.diagnostic_status.setCurrentIndex(dialog.diagnostic_status.findData(labels[reason]))
        assert [item[-1] for item in dialog.diagnostic_model.rows] == [row.id]
        assert dialog.diagnostic_count.text() == tr('Показано проб: {0} из {1}', 1, 5)
    dialog.diagnostic_status.setCurrentIndex(dialog.diagnostic_status.findData(labels["anchor"]))
    assert dialog.diagnostic_model.rowCount() == 1
    dialog.only_excluded.setChecked(True)
    assert dialog.diagnostic_model.rowCount() == 0
    assert dialog.diagnostic_count.text() == (
        tr('Показано проб: {0} из {1}', 0, 5) + "\n" + tr('Фильтры скрыли все строки диагностики.')
    )
    assert not dialog.export_diagnostics_button.isEnabled()
    dialog.diagnostic_status.setCurrentIndex(0)
    dialog.only_excluded.setChecked(False)
    assert dialog.diagnostic_count.text() == tr('Показано проб: {0} из {1}', 5, 5)


def test_duplicate_rejection_reports_every_conflicting_coordinate(data):
    cfg = options(data)
    x_id = cfg["x_id"]
    data.rows[1].values[x_id] = "0"
    data.rows[3].values[x_id] = 10
    with pytest.raises(interpolation.DuplicateCoordinatesError) as failure:
        prepare(data, **cfg)
    assert failure.value.coordinates == {
        row.id: coordinate for row, coordinate in zip(data.rows[:4], (0, 0, 10, 10), strict=True)
    }


def test_validation_scores_report_coverage_and_errors(data):
    _, protocol = prepare(data, **options(data))
    checked, common, scores = interpolation.validation_scores(protocol)
    assert (checked, common) == (1, 0)
    assert scores == {"linear": (1, None, None), "nearest": (1, None, None),
                      "pchip": (1, None, None), "spline": (0, None, None)}


def test_validation_scores_compare_only_common_predictions():
    protocol = {
        "parameters": {"max_gap": 2},
        "supports": [
            {"anchors": [{"row_id": f"long-{x}", "x": x, "y": x} for x in range(5)]},
            {"anchors": [{"row_id": f"short-{x}", "x": x, "y": y}
                         for x, y in enumerate((0, 100, 0))]},
        ],
    }
    checked, common, scores = interpolation.validation_scores(protocol)
    assert (checked, common) == (4, 3)
    assert scores == {"linear": (4, 0, 0), "nearest": (4, 1, 1),
                      "pchip": (4, 0, 0), "spline": (3, 0, 0)}


def test_preview_labels_show_explicit_and_bound_well_columns(data, qtbot):
    from PySide6.QtCore import Qt
    from autopyro.ui.dialogs.interpolation import InterpolationDialog

    well, bore, depth, y = data.columns
    formation = Column("Formation", "text")
    data.columns.append(formation)
    for row in data.rows:
        row.values[formation.id] = "F1"
    for group, start in (("F1", 3), ("F2", 13)):
        for x, value in ((0, start), (5, None), (10, start + 4)):
            data.rows.append(Row(values={well.id: "A", bore.id: "A-2", depth.id: x, y.id: value,
                                         formation.id: group}))
    data.view["stratigraphy"] = stratigraphy(data)
    dialog = InterpolationDialog(data, set(data.row_map()))
    qtbot.addWidget(dialog)
    dialog.x.setCurrentIndex(dialog.x.findData(depth.id))
    dialog.y.setCurrentIndex(dialog.y.findData(y.id))
    dialog.well.setCurrentIndex(dialog.well.findData(well.id))
    dialog.groups.item(4).setCheckState(Qt.CheckState.Checked)
    dialog.boundaries.item(0).setCheckState(Qt.CheckState.Unchecked)
    dialog.calculate()

    assert dialog.preview_model.rowCount() == 4
    for row, wellbore, group_number in ((0, "A-1", 1), (2, "A-2", 2)):
        label = dialog.preview_model.index(row, 5).data()
        assert f"Well: A · Bore: {wellbore}" in label
        assert dialog.chart_well.itemText(dialog.chart_well.findData(f"group:{group_number}")) == label
    dialog.preview_tabs.setCurrentWidget(dialog.chart_view)
    primary = dialog.chart_well.findData("well:A")
    bound = dialog.chart_well.findData("bound-well:A-2")
    assert dialog.chart_well.itemText(primary) == "Well: A"
    assert dialog.chart_well.itemText(bound) == "Bore: A-2"
    dialog.chart_well.setCurrentIndex(bound)
    measured, predicted = dialog.chart_view.chart().series()
    assert measured.count() == 4
    assert {point.y() for point in predicted.points()} == {5, 15}
    dialog.chart_well.setCurrentIndex(primary)
    measured, predicted = dialog.chart_view.chart().series()
    assert measured.count() == 7 and predicted.count() == 4


def test_preview_labels_distinguish_same_named_stratigraphic_intervals(data, qtbot):
    from autopyro.ui.dialogs.interpolation import InterpolationDialog

    well, bore, depth, y = data.columns
    data.rows.append(Row(values={well.id: "A", bore.id: "A-1", depth.id: 7, y.id: 5}))
    data.view["stratigraphy"] = stratigraphy(data)
    dialog = InterpolationDialog(data, set(data.row_map()))
    qtbot.addWidget(dialog)
    dialog.x.setCurrentIndex(dialog.x.findData(depth.id))
    dialog.y.setCurrentIndex(dialog.y.findData(y.id))
    dialog.calculate()

    assert dialog.preview_model.rowCount() == 2
    for i, interval in enumerate(("-1–8 m", "8–30 m")):
        label = dialog.preview_model.index(i, 5).data()
        assert f"Repeated name [{interval}]" in label
        assert dialog.chart_well.itemText(dialog.chart_well.findData(f"group:{i + 1}")) == label


@pytest.mark.parametrize("dependency", ["group", "well", "bottom", "bound_depth", None])
def test_reopening_marks_interpolation_with_missing_dependencies_stale(data, tmp_path, dependency):
    well, bore, depth, y = data.columns
    metadata = {"method": "linear", "x_id": depth.id, "y_id": y.id, "group_ids": [well.id],
                "constraints": {"well_id": bore.id, "exclude_intervals_by": None}, "stale": False}
    if dependency == "group":
        metadata["group_ids"].append("removed-column")
    elif dependency == "well":
        metadata["constraints"]["well_id"] = "removed-column"
    elif dependency == "bottom":
        metadata["constraints"]["exclude_intervals_by"] = "removed-column"
    elif dependency == "bound_depth":
        data.view["stratigraphy"] = {"binding": {"well_id": bore.id, "depth_id": "removed-column"}}
        metadata["constraints"]["stratigraphy"] = deepcopy(data.view["stratigraphy"])
    column = Column("Estimate", "number", metadata=metadata)
    data.columns.append(column)
    for row in data.rows:
        row.values[column.id] = None
    path = tmp_path / "dependencies.apyro"
    data.save(path)

    restored = Project.load(path)
    assert restored.columns[-1].metadata["stale"] is (dependency is not None)


def test_changed_stratigraphy_marks_interpolation_stale(data, qtbot, tmp_path):
    from PySide6.QtCore import Qt
    from autopyro.ui.dialogs.stratigraphy import SetStratigraphyCommand
    from autopyro.ui.window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(data)
    try:
        state = stratigraphy(data)
        window.model.undo.push(SetStratigraphyCommand(window.model, state))
        window.interpolate()
        dialog = window.interpolation_window
        dialog.x.setCurrentIndex(dialog.x.findData(data.columns[2].id))
        dialog.y.setCurrentIndex(dialog.y.findData(data.columns[3].id))
        dialog.calculate()
        assert [(p.x, p.value) for p in dialog.proposals] == [(15, 8)]
        dialog.apply()
        column = data.columns[-1]
        revised = deepcopy(state)
        revised["scheme"]["revision"] = "2"
        revised["scheme"]["intervals"][0]["bottom"] = 12
        revised["scheme"]["intervals"][1]["top"] = 12
        window.model.undo.push(SetStratigraphyCommand(window.model, revised))
        assert column.metadata["stale"]
        assert window.model.headerData(4, Qt.Orientation.Horizontal) != column.name
        column.metadata["stale"] = False  # Simulate a project saved before invalidation was fixed.
        path = tmp_path / "old-staleness.apyro"
        data.save(path)
        assert Project.load(path).columns[-1].metadata["stale"]
        dialog.calculate()
        assert [(p.x, p.value) for p in dialog.proposals] == [(5, 4)]
    finally:
        window.dirty = False
        window.close()


def test_preview_separates_wellbores_even_with_additional_group(data, qtbot):
    from PySide6.QtCore import Qt
    from autopyro.ui.dialogs.interpolation import InterpolationDialog

    bore = data.columns[1]
    for row in data.rows[2:]:
        row.values[bore.id] = "A-2"
    dialog = InterpolationDialog(data, set(data.row_map()))
    qtbot.addWidget(dialog)
    dialog.x.setCurrentIndex(dialog.x.findData(data.columns[2].id))
    dialog.y.setCurrentIndex(dialog.y.findData(data.columns[3].id))
    dialog.groups.item(0).setCheckState(Qt.CheckState.Checked)
    assert dialog.well.currentData() == bore.id
    dialog.calculate()
    assert [(p.x, p.value) for p in dialog.proposals] == [(15, 8)]
    dialog.preview_tabs.setCurrentWidget(dialog.chart_view)
    dialog.chart_well.setCurrentIndex(dialog.chart_well.findData("group:1"))
    assert dialog.chart_view.chart().series()[0].count() == 1
    assert dialog.chart_view.chart().series()[1].count() == 0
    dialog.well.setCurrentIndex(0)
    assert not dialog.preview_button.isEnabled()
    dialog.without_well.setChecked(True)
    dialog.calculate()
    assert [(p.x, p.value) for p in dialog.proposals] == [(5, 4), (15, 8)]


@pytest.mark.parametrize("property_index", [2, 3])
def test_qc_scope_excludes_anchor_from_preview_and_snapshot(data, qtbot, property_index):
    from autopyro.ui.table.commands import QCCommand
    from autopyro.ui.window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(data)
    try:
        window.interpolate()
        dialog = window.interpolation_window
        dialog.x.setCurrentIndex(dialog.x.findData(data.columns[2].id))
        dialog.y.setCurrentIndex(dialog.y.findData(data.columns[3].id))
        dialog.calculate()
        dialog.apply()
        column = data.columns[-1]
        property_id = data.columns[property_index].id
        anchor = data.rows[2]
        measurement = data.science.current(anchor.id, property_id)
        window.model.undo.push(QCCommand(window.model, ("qc_exclusions", "interpolation"), {
            measurement.id: {"revision": measurement.revision, "comment": "QC regression"},
        }))
        assert column.metadata["stale"]
        dialog.calculate()
        assert anchor.id in dialog.parameters["constraints"]["excluded_row_ids"]
        assert any(row[-1] == anchor.id for row in dialog._excluded_rows)
        assert replay(dialog.protocol) == dialog.proposals
        dialog.preview_tabs.setCurrentWidget(dialog.chart_view)
        assert dialog.chart_view.chart().series()[0].count() == 2
        dialog.chart_excluded.setChecked(True)
        assert dialog.chart_view.chart().series()[-1].count() == 1
        window.model.undo.undo()
        dialog.calculate()
        assert not dialog.parameters["constraints"]["excluded_row_ids"]
    finally:
        window.dirty = False
        window.close()
