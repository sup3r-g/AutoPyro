import json
from copy import deepcopy

import pytest
from matplotlib.figure import Figure

from autopyro.core import Column, Project, Row
from autopyro.io import build_project
from autopyro.plotting import PlotSpec, render_plot
from autopyro.reporting import ReportOptions, build_report, render_html, render_pdf
from autopyro.stratigraphy import (evaluate_project, from_records, overlap_pairs,
    recompute, result_column, validate_scheme)


@pytest.fixture
def data():
    project = build_project(["Wellbore", "MD", "TOC"],
        [["A", "10", "1"], ["A", "20", "2"], ["A", "30", "3"], ["A R", "15", "4"],
         [" A", "15", "5"], ["A", "ND", "6"], ["A", "<15", "7"]], list(range(1, 8)),
        [(0, "Wellbore", "text", ""), (1, "MD", "text", "m"), (2, "TOC", "number", "%wt")], {})
    scheme = {"format": "autopyro.stratigraphy", "version": 1, "id": "demo", "revision": "1",
        "name": "Synthetic strata", "depth_reference": "MD RKB", "depth_unit": "m", "source": "test",
        "intervals": [{"id": "upper", "wellbore": "A", "level": "FORMATION", "name": "Upper", "top": 0, "bottom": 20, "color": "#aaccff"},
                      {"id": "lower", "wellbore": "A", "level": "FORMATION", "name": "Lower", "top": 20, "bottom": 30, "color": "#ffccaa"},
                      {"id": "group", "wellbore": "A", "level": "GROUP", "name": "Group", "top": 0, "bottom": 30}]}
    well, depth, _ = [c.id for c in project.columns]
    binding = {"well_id": well, "depth_id": depth, "depth_reference": "MD RKB", "depth_unit": "m", "text_numbers": True}
    project.view["stratigraphy"] = {"scheme": scheme, "binding": binding}
    return project, scheme, binding


def test_half_open_bounds_exact_wellbores_and_qualifiers(data):
    project, scheme, binding = data
    report = evaluate_project(project, [r.id for r in project.rows], scheme, binding)
    results = [r["levels"]["FORMATION"] for r in report["results"]]
    assert results[0]["names"] == ["Upper"]
    assert results[1]["names"] == ["Lower"]
    assert [r["status"] for r in results[2:]] == ["outside", "unknown_wellbore", "unknown_wellbore", "insufficient_data", "insufficient_data"]
    assert overlap_pairs(scheme) == []
    assert evaluate_project(project, [], scheme, binding)["results"] == []


@pytest.mark.parametrize("language", ["ru", "en"])
def test_report_depth_profile_marks_qc_and_estimates_with_lithology(data, monkeypatch, language):
    project, scheme, _ = data
    scheme["intervals"][0]["lithology"] = "shale"
    toc = project.columns[2]
    estimate = Column("TOC estimate", "number", metadata={"method": "linear", "y_id": toc.id})
    project.columns.append(estimate)
    project.rows[1].values[estimate.id] = 2.5
    project.sync_measurements()
    measured = project.science.current(project.rows[0].id, toc.id)
    project.view["qc_exclusions"] = {"statistics": {measured.id: {"revision": measured.revision}}}
    ids = [row.id for row in project.rows[:3]]
    savefig = Figure.savefig

    def verify_axes(figure, *args, **kwargs):
        if any(axis.get_label() == "lithology" for axis in figure.axes):
            figure.canvas.draw()
            assert [axis.get_label() for axis in figure.axes[-2:]] == ["stratigraphy", "lithology"]
            if language == "en":
                assert figure.axes[-1].get_title() == "Lithology"
                assert "Excluded by QC" in [text.get_text() for text in figure.legends[0].get_texts()]
            for boundary in (10, 20, 30):
                positions = [axis.transData.transform((0, boundary))[1] for axis in figure.axes]
                assert positions == pytest.approx([positions[0]] * len(positions))
            for track in figure.axes[-2:]:
                assert any(line.get_visible() for line in track.get_ygridlines())
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", verify_axes)
    report = build_report(project, ids, ReportOptions((toc.id,), sections=("profile",), language=language))
    profile, = report["well_profiles"]
    assert profile["row_ids"] == ids
    assert profile["status_counts"] == {
        toc.id: {"qc": 1, "measured": 2}, estimate.id: {"interpolated": 1},
    }
    html = render_html(report)
    if language == "en":
        assert "Depth profile: A" in html
        assert "Lithology is shown for the scheme intervals." in html
    else:
        assert "Литология показана по интервалам схемы." in html
    assert render_pdf(report).startswith(b"%PDF")


@pytest.mark.parametrize(("language", "notice"), [
    ("ru", "Стратиграфический слой недоступен: Неизвестная литология stratapy: custom-rock-123"),
    ("en", "Stratigraphic layer unavailable: Unknown stratapy lithology: custom-rock-123"),
])
def test_depth_profile_localizes_layer_validation_errors(data, language, notice):
    project, scheme, _ = data
    scheme["intervals"][0]["lithology"] = "custom-rock-123"
    report = build_report(project, [project.rows[0].id], ReportOptions(
        (project.columns[2].id,), sections=("profile",), language=language,
    ))
    profile, = report["well_profiles"]
    assert profile["png"]
    assert notice in profile["notices"]
    assert notice in render_html(report)


def test_report_limits_stratigraphy_and_rejects_mismatched_profile_datum(data):
    project, scheme, _ = data
    scheme["intervals"][0]["lithology"] = "shale"
    scheme["intervals"].append({**scheme["intervals"][0], "id": "other", "wellbore": "A R"})
    datum = Column("MD datum")
    project.columns.append(datum)
    for row in project.rows:
        row.values[datum.id] = "RKB"
    project.rows[1].values[datum.id] = "MSL"
    ids = [row.id for row in project.rows[:2]]
    report = build_report(project, ids, ReportOptions(
        (project.columns[2].id,), sections=("profile", "stratigraphy"),
    ))
    strat = report["stratigraphy"]
    assert strat["row_ids"] == ids[:1]
    assert {interval["wellbore"] for interval in strat["scheme"]["intervals"]} == {"A"}
    assert strat["source_scheme_hash"] != strat["scheme_hash"]
    assert len(scheme["intervals"]) == 4
    rkb, msl = report["well_profiles"]
    assert any("Литология показана" in note for note in rkb["notices"])
    assert not any("Литология показана" in note for note in msl["notices"])
    assert any("MSL" in note and "не наложена" in note for note in report["well_profile_notices"])


@pytest.mark.parametrize("interval", [False, True])
def test_english_stratigraphy_report_keeps_names_and_protocol_statuses(data, interval):
    project, scheme, binding = data
    scheme["intervals"][0]["name"] = "Верхний пласт"
    if interval:
        bottom = Column("Base", "number", unit="m")
        project.columns.append(bottom)
        for row in project.rows:
            row.values[bottom.id] = 25
        binding.update(sample_mode="interval", bottom_id=bottom.id)
    report = build_report(project, [project.rows[0].id], ReportOptions(
        (project.columns[2].id,), sections=("stratigraphy",), language="en",
    ))
    strat = report["stratigraphy"]
    frozen = deepcopy(strat)
    html = render_html(report)
    assert "Stratigraphic assignment" in html
    assert "Stratigraphy · Formation" in html
    assert "<th>Wellbore</th>" in html and "Верхний пласт" in html
    assert strat == frozen
    status = strat["results"][0]["levels"]["FORMATION"]["status"]
    if interval:
        assert status == "cross_boundary"
        assert "Sample intersections · Formation" in html
        assert "Crosses a unit boundary" in html and "Coverage fraction (0–1)" in html
    else:
        assert status == "matched" and ">Matched<" in html


def test_bulk_depth_parsing_preserves_selection_order(data):
    project, scheme, binding = data
    originals = list(project.rows)
    for _ in range(40):
        project.rows.extend(Row(row.values.copy()) for row in originals)
    ids = [row.id for row in reversed(project.rows)]
    scalar = evaluate_project(project, ids[:7], scheme, binding)
    batch = evaluate_project(project, ids, scheme, binding)
    assert batch["row_ids"] == ids
    assert batch["results"][:7] == scalar["results"]


def test_overlap_preserved_and_ambiguous_even_same_name(data):
    project, scheme, binding = data
    item = deepcopy(scheme["intervals"][0])
    item.update(id="duplicate", top=5, bottom=15)
    scheme["intervals"].append(item)
    assert len(overlap_pairs(scheme)) == 1
    result = evaluate_project(project, [project.rows[0].id], scheme, binding)["results"][0]["levels"]
    assert result["FORMATION"]["status"] == "ambiguous"
    assert result["GROUP"]["status"] == "matched"


def test_unit_conversion_and_reference_validation(data):
    project, scheme, binding = data
    project.columns[1].unit = "ft"
    project.rows[0].values[binding["depth_id"]] = "50"
    binding["depth_unit"] = "ft"
    result = evaluate_project(project, [project.rows[0].id], scheme, binding)
    assert result["results"][0]["levels"]["FORMATION"]["names"] == ["Upper"]
    binding["depth_reference"] = "TVD MSL"
    with pytest.raises(ValueError, match="начало отсчёта"):
        evaluate_project(project, [], scheme, binding)


@pytest.mark.parametrize("field,value", [("top", float("nan")), ("bottom", float("inf")), ("bottom", 0), ("top", True), ("color", "notacolor")])
def test_invalid_interval_rejected(data, field, value):
    _, scheme, _ = data
    scheme["intervals"][0][field] = value
    with pytest.raises(ValueError):
        validate_scheme(scheme)


def test_derived_columns_recalculate_and_roundtrip(data, tmp_path):
    project, _, binding = data
    name, status = result_column("FORMATION"), result_column("FORMATION", status=True)
    project.columns.extend([name, status])
    recompute(project)
    assert project.rows[0].values[name.id] == "Upper"
    project.rows[0].values[binding["depth_id"]] = "25"
    recompute(project)
    assert project.rows[0].values[name.id] == "Lower"
    project.save(tmp_path/"strata.apyro")
    restored = Project.load(tmp_path/"strata.apyro")
    assert restored.rows[0].values[name.id] == "Lower"
    del project.view["stratigraphy"]
    recompute(project)
    assert project.rows[0].values[name.id] is None
    assert name.metadata["diagnostics"][project.rows[0].id]["status"] == "invalid_definition"


def test_scoped_refresh_rebuilds_all_rows_when_binding_changes(data):
    project, _, binding = data
    name = result_column('FORMATION')
    project.columns.append(name)
    recompute(project)
    assert project.rows[1].values[name.id] == 'Lower'

    binding['text_numbers'] = False
    recompute(project, rows=project.rows[:1])
    assert project.rows[1].values[name.id] is None
    assert name.metadata['diagnostics'][project.rows[1].id]['status'] == 'insufficient_data'


def test_csv_view_schema_and_provenance():
    headers = ["wellbore_name", "unit_name", "unit_level", "top_depth_m", "bottom_depth_m"]
    scheme = from_records(headers, [["A R", "Formation", "FORMATION", "1,5", "20"]], [42],
                          name="Export", depth_reference="MD RKB", source={"sha256": "abc"})
    interval = scheme["intervals"][0]
    assert interval["top"] == 1.5 and interval["source_row"] == 42
    assert interval["raw_cells"]["wellbore_name"] == "A R"
    with pytest.raises(ValueError, match="система глубин"):
        from_records(headers + ["depth_reference"], [["A", "F", "FORMATION", "0", "2", "TVD"]], [2],
                     name="x", depth_reference="MD RKB", source={})


def test_depth_layer_keeps_limits_and_rejects_mixed_wells(data):
    project, scheme, binding = data
    spec = PlotSpec(kind="multi_depth", track_ids=[project.columns[-1].id], y_id=binding["depth_id"],
        text_numbers=True, stratigraphy={"scheme": scheme, "binding": binding, "level": "FORMATION"})
    ids = [r.id for r in project.rows[:3]]
    plain = deepcopy(spec)
    plain.stratigraphy = None
    _, reference = render_plot(project, ids, plain)
    figure, result = render_plot(project, ids, spec)
    assert result["axes"][0].get_ylim() == reference["axes"][0].get_ylim()
    column = next(ax for ax in figure.axes if ax.get_label() == "stratigraphy")
    assert len(column.patches) == 2
    assert column.get_ylim() == reference["axes"][0].get_ylim()
    assert column.get_shared_y_axes().joined(column, result["axes"][0])
    # A remote interval must not create labels outside the visible plot.
    scheme["intervals"].append({"id": "remote", "wellbore": "A", "level": "FORMATION", "name": "Remote", "top": 1000, "bottom": 2000})
    remote_figure, remote = render_plot(project, ids, spec)
    assert "Remote" not in [text.get_text() for ax in remote_figure.axes
                            for text in ax.texts if text.get_visible()]
    with pytest.raises(ValueError, match="одного ствола"):
        render_plot(project, [r.id for r in project.rows], spec)
    spec.kind = "scatter"
    spec.x_id = project.columns[-1].id
    with pytest.raises(ValueError, match="глубинного графика"):
        render_plot(project, ids, spec)


def test_interpolated_depth_excluded_and_report_frozen(data):
    project, scheme, binding = data
    project.columns[1].metadata["method"] = "linear"
    assert evaluate_project(project, [project.rows[0].id], scheme, binding)["results"][0]["levels"]["FORMATION"]["status"] == "interpolation_excluded"
    report = build_report(project, [project.rows[0].id], ReportOptions((project.columns[-1].id,)))
    scheme["intervals"][0]["name"] = "Changed"
    assert report["stratigraphy"]["scheme"]["intervals"][0]["name"] == "Upper"
    assert "Стратиграфическая привязка" in render_html(report)
    assert len(report["stratigraphy"]["results"]) == 1


@pytest.mark.parametrize("bounds", [[(10, 20)], [(0, 15), (22, 35)], [(0, 25), (20, 35)]])
def test_stratapy_columns_preserve_geometry_and_zoom(data, bounds, tmp_path, monkeypatch):
    from stratapy.core import LogObject
    from matplotlib.colors import to_rgba

    project, scheme, binding = data
    scheme["depth_unit"] = "ft"
    scheme["intervals"] = [
        {"id": str(i), "wellbore": "A", "level": "FORMATION", "name": f"Layer {i}",
         "top": top / 0.3048, "bottom": bottom / 0.3048, "color": "#aaccff",
         "lithology": "607" if i == 0 else ""}
        for i, (top, bottom) in enumerate(bounds)
    ]
    original = deepcopy(scheme)
    calls = []
    plot = LogObject.plot

    def record(self, *args, **kwargs):
        calls.append(self.original_df.copy())
        return plot(self, *args, **kwargs)

    monkeypatch.setattr(LogObject, "plot", record)
    spec = PlotSpec(kind="multi_depth", track_ids=[project.columns[-1].id, binding["depth_id"]],
                    y_id=binding["depth_id"], text_numbers=True, stratigraphy_alpha=0.8,
                    stratigraphy={"scheme": scheme, "binding": binding, "level": "FORMATION"})
    fig, result = render_plot(project, [r.id for r in project.rows[:3]], spec)
    assert len(calls) == 2  # Both columns go through the real stratapy renderer.
    strat, lith = fig.axes[-2:]
    assert strat.get_label() == "stratigraphy" and lith.get_label() == "lithology"
    for frame in calls:
        assert list(frame["height/age"]) == pytest.approx([top for top, bottom in bounds])
        assert list(frame["thickness"]) == pytest.approx([bottom - top for top, bottom in bounds])
    for patch, (top, bottom) in zip(strat.patches, bounds):
        assert min(patch.get_xy()[:, 1]) == pytest.approx(top)
        assert max(patch.get_xy()[:, 1]) == pytest.approx(bottom)
        assert patch.get_facecolor() == pytest.approx(to_rgba("#aaccff", 0.8))
    assert lith.images  # Native FGDC raster pattern, not an application hatch substitute.
    assert scheme == original
    result["axes"][0].set_ylim(27, 12)
    assert all(ax.get_ylim() == (27, 12) for ax in fig.axes)
    fig.savefig(tmp_path / "columns.png")
    fig.savefig(tmp_path / "columns.svg")
    fig.savefig(tmp_path / "columns.pdf")
    if len(bounds) > 1 and bounds[0][1] > bounds[1][0]:
        assert any("перекрытия" in notice for notice in result["notices"])


def test_unknown_lithology_is_not_silently_replaced(data):
    project, scheme, binding = data
    scheme["intervals"][0]["lithology"] = "unknown rock"
    spec = PlotSpec(kind="depth", x_id=project.columns[-1].id, y_id=binding["depth_id"],
                    text_numbers=True,
                    stratigraphy={"scheme": scheme, "binding": binding, "level": "FORMATION"})
    with pytest.raises(ValueError, match="Неизвестная литология"):
        render_plot(project, [project.rows[0].id], spec)


@pytest.mark.parametrize("well, level, count, limits", [
    ("A", "FORMATION", 2, (30, 0)), ("A", "GROUP", 1, (30, 0)),
])
def test_standalone_column_needs_only_scheme(data, well, level, count, limits, tmp_path):
    from autopyro.stratigraphy import render_columns

    _, scheme, _ = data
    original = deepcopy(scheme)
    fig, notices = render_columns(scheme, well, level)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].patches) == count
    assert fig.axes[0].get_ylim() == limits
    assert fig.axes[0].get_ylabel() == "Глубина [m]"
    assert scheme == original
    assert not notices
    fig.savefig(tmp_path / "column.pdf")


def test_column_warns_only_about_overlaps_on_visible_tracks(data, monkeypatch):
    from autopyro.geology import stratigraphy

    _, scheme, _ = data
    group = scheme['intervals'][-1]
    scheme['intervals'].append({**group, 'id': 'nested-group', 'top': 5, 'bottom': 10})

    def unexpected_pair_listing(*args):
        pytest.fail('Column rendering should not enumerate overlap pairs')

    monkeypatch.setattr(stratigraphy, 'overlap_pairs', unexpected_pair_listing)
    _, formation_notices = stratigraphy.render_columns(scheme, 'A', 'FORMATION')
    _, group_notices = stratigraphy.render_columns(scheme, 'A', 'GROUP')
    assert not formation_notices
    assert group_notices == ['Стратиграфическая колонка содержит перекрытия на одном уровне']


def test_overlap_count_matches_pairs_and_handles_dense_tracks(data):
    from autopyro.geology.stratigraphy import overlap_count

    _, scheme, _ = data
    first = scheme['intervals'][0]
    scheme['intervals'].extend([
        {**first, 'id': 'nested', 'top': 5, 'bottom': 10},
        {**first, 'id': 'straddling', 'top': 5, 'bottom': 25},
        {**first, 'id': 'other-well', 'wellbore': 'B'},
    ])
    assert overlap_count(scheme) == len(overlap_pairs(scheme)) == 4

    size = 10_000
    scheme['intervals'] = [{**first, 'id': str(index)} for index in range(size)]
    assert overlap_count(scheme) == size * (size - 1) // 2


def test_standalone_lithology_and_missing_level(data):
    from autopyro.stratigraphy import render_columns

    _, scheme, _ = data
    scheme["intervals"][0]["lithology"] = "607"
    fig, _ = render_columns(scheme, "A", "FORMATION")
    assert [ax.get_label() for ax in fig.axes] == ["stratigraphy", "lithology"]
    assert fig.axes[1].images
    assert fig.axes[0].get_ylim() == fig.axes[1].get_ylim() == (30, 0)
    with pytest.raises(ValueError, match="нет интервалов"):
        render_columns(scheme, "missing well", "FORMATION")


def test_hierarchy_tracks_share_depth_and_load_each_level_once(data, monkeypatch, tmp_path):
    import stratapy as sp
    from autopyro.stratigraphy import render_columns

    _, scheme, _ = data
    scheme['intervals'][0]['lithology'] = '607'
    scheme['intervals'][2].update(top=-5, bottom=35, lithology='608')
    original = deepcopy(scheme)
    loads = []
    load = sp.load
    def record(*args, **kwargs):
        loads.append(args)
        return load(*args, **kwargs)
    monkeypatch.setattr(sp, 'load', record)
    figure, notices = render_columns(scheme, 'A', 'FORMATION', options={
        'hierarchy_levels': ['MEMBER', 'GROUP'], 'width_mm': 220})
    assert len(loads) == 2  # Stratigraphy and lithology reuse the primary log.
    assert [(axis._autopyro_level, axis._autopyro_kind) for axis in figure.axes] == [
        ('GROUP', 'stratigraphy'), ('FORMATION', 'stratigraphy'), ('FORMATION', 'lithology')]
    assert all(axis.get_ylim() == (35, -5) for axis in figure.axes)
    assert all(figure.axes[0].get_shared_y_axes().joined(figure.axes[0], axis)
               for axis in figure.axes[1:])
    assert notices == ['Нет интервалов уровня: MEMBER']  # Parent/child is not a conflict.
    assert scheme == original
    assert figure.axes[-1].images
    positions = [axis.get_position() for axis in figure.axes]
    assert all(left.x1 < right.x0 for left, right in zip(positions, positions[1:]))
    figure.axes[-1].set_ylim(25, 10)
    assert all(axis.get_ylim() == (25, 10) for axis in figure.axes)
    figure.savefig(tmp_path / 'hierarchy.svg')


def test_missing_primary_does_not_infer_lithology_from_other_levels(data):
    from autopyro.stratigraphy import render_columns

    _, scheme, _ = data
    scheme['intervals'][2]['lithology'] = '607'
    figure, notices = render_columns(scheme, 'A', 'MEMBER', options={
        'hierarchy_levels': ['GROUP'], 'display_mode': 'default'})
    assert len(figure.axes) == 1
    assert figure.axes[0]._autopyro_level == 'GROUP'
    assert figure.axes[0]._autopyro_kind == 'stratigraphy'
    assert notices == ['Нет интервалов уровня: MEMBER']


def test_well_comparison_preserves_tracks_and_shares_absolute_depth(data, monkeypatch, tmp_path):
    import stratapy as sp
    from autopyro.stratigraphy import render_columns

    _, scheme, _ = data
    scheme['intervals'][0]['lithology'] = '607'
    for well, offset in [('B', 50), ('excluded', 1000)]:
        for source in list(scheme['intervals'][:3]):
            item = deepcopy(source)
            item.update(id=f"{well}-{source['id']}", wellbore=well,
                        top=source['top'] + offset, bottom=source['bottom'] + offset)
            scheme['intervals'].append(item)
    original = deepcopy(scheme)
    loads = []
    load = sp.load
    def record(*args, **kwargs):
        loads.append(args)
        return load(*args, **kwargs)
    monkeypatch.setattr(sp, 'load', record)
    figure, notices = render_columns(scheme, 'A', 'FORMATION', options={
        'comparison_wells': ['B', 'A'], 'hierarchy_levels': ['GROUP'], 'width_mm': 300})
    assert len(loads) == 4  # One load for each (well, level), reused for lithology.
    assert [axis._autopyro_well for axis in figure.axes] == ['A'] * 3 + ['B'] * 3
    assert [axis._autopyro_level for axis in figure.axes] == ['GROUP', 'FORMATION', 'FORMATION'] * 2
    assert all(axis.get_ylim() == (80, 0) for axis in figure.axes)
    assert all(figure.axes[0].get_shared_y_axes().joined(figure.axes[0], axis)
               for axis in figure.axes[1:])
    assert {text.get_text() for text in figure.texts} >= {'A', 'B'}
    assert not notices
    assert scheme == original
    figure.axes[-1].set_ylim(70, 10)
    assert all(axis.get_ylim() == (70, 10) for axis in figure.axes)
    figure.savefig(tmp_path / 'comparison.svg')
    figure.savefig(tmp_path / 'comparison.png')


def test_comparison_missing_levels_are_reported_per_well(data):
    from autopyro.stratigraphy import render_columns

    _, scheme, _ = data
    scheme['intervals'].append({'id': 'B-group', 'wellbore': 'B', 'level': 'GROUP',
                                'name': 'B group', 'top': 50, 'bottom': 100})
    figure, notices = render_columns(scheme, 'A', 'FORMATION', options={
        'comparison_wells': ['B', 'missing'], 'hierarchy_levels': ['GROUP']})
    assert [axis._autopyro_well for axis in figure.axes] == ['A', 'A', 'B']
    assert all(axis.get_ylim() == (100, 0) for axis in figure.axes)
    assert 'Нет интервалов уровня: FORMATION · B' in notices
    assert 'Нет интервалов уровня: GROUP · missing' in notices
    assert 'Нет интервалов уровня: FORMATION · missing' in notices


@pytest.mark.parametrize('enabled', [True, False])
def test_correlation_connection_stops_at_a_missing_well_anchor(data, enabled):
    from matplotlib.patches import ConnectionPatch
    from autopyro.stratigraphy import render_columns

    _, scheme, _ = data
    original = next(item for item in scheme['intervals'] if item['wellbore'] == 'A' and item['level'] == 'FORMATION')
    for well in ('B', 'C'):
        scheme['intervals'].append({**original, 'id': well, 'wellbore': well})
    figure, notices = render_columns(scheme, 'A', 'FORMATION', options={
        'comparison_wells': ['B', 'C'],
        'correlation': {'enabled': enabled, 'edge': 'top', 'anchors': {'A': original['id'], 'B': 'B', 'C': 'C'},
                        'horizons': [{'name': 'Marker', 'edge': 'bottom',
                                      'anchors': {'A': original['id'], 'B': 'deleted', 'C': 'C'}}]},
    })
    assert not any(isinstance(artist, ConnectionPatch) for artist in figure.artists)
    assert any('Marker' in notice and 'B' in notice for notice in notices) == enabled
    assert any(text.get_text() == 'Marker' for axis in figure.axes for text in axis.texts) == enabled


@pytest.mark.parametrize('wells', ['A', ['A', 'A'], [''], [None]])
def test_comparison_rejects_invalid_well_selection(wells):
    from autopyro.stratigraphy import column_style

    with pytest.raises(ValueError, match='стволов для сравнения'):
        column_style({'comparison_wells': wells})


@pytest.mark.parametrize('legend_kwargs, x', [({}, .25), ({'bbox_to_anchor': None}, .25),
                                              ({'bbox_to_anchor': [.2, .08, .5, .1]}, .1)])
def test_comparison_legends_stay_in_their_well_regions(data, legend_kwargs, x):
    from autopyro.stratigraphy import render_columns

    _, scheme, _ = data
    scheme['intervals'].append({'id': 'B-formation', 'wellbore': 'B', 'level': 'FORMATION',
                                'name': 'B formation', 'top': 0, 'bottom': 40, 'lithology': '608'})
    scheme['intervals'][0]['lithology'] = '607'
    figure, _ = render_columns(scheme, 'A', 'FORMATION', options={
        'comparison_wells': ['B'], 'legend': True, 'legend_loc': 'bottom', 'width_mm': 300,
        'legend_kwargs': legend_kwargs})
    assert len(figure.legends) == 2
    left, right = [legend.get_bbox_to_anchor().transformed(figure.transFigure.inverted())
                   for legend in figure.legends]
    assert left.x0 == pytest.approx(x)
    assert right.x0 == pytest.approx(x + .5)


@pytest.mark.parametrize('levels', ['GROUP', ['UNKNOWN'], ['GROUP', 'GROUP'], [None]])
def test_invalid_hierarchy_levels_rejected(levels):
    from autopyro.stratigraphy import column_style

    with pytest.raises(ValueError, match='стратиграфические уровни'):
        column_style({'hierarchy_levels': levels})


@pytest.mark.parametrize("mode, location", [("log", "right"), ("default", "left"),
                                           ("grainsize", "bottom"), ("default", "top")])
def test_column_style_reaches_native_renderer(data, mode, location, tmp_path, monkeypatch):
    import matplotlib as mpl
    from stratapy.core import LogObject
    from stratapy.setup import formatting
    from autopyro.stratigraphy import render_columns

    _, scheme, _ = data
    scheme["intervals"][0].update(lithology="607", top_grain="0.5", bottom_grain="1")
    original_sizes = formatting.fontsizes.copy()
    original_family = list(mpl.rcParams["font.family"])
    calls = []
    plot = LogObject.plot

    def record(self, *args, **kwargs):
        calls.append((kwargs.copy(), self.original_df.copy()))
        return plot(self, *args, **kwargs)

    monkeypatch.setattr(LogObject, "plot", record)
    options = dict(display_mode=mode, feature_mode="merge", legend=True, legend_loc=location,
                   unit_borders=False, spines=True, ppi=250, dpi=120, width_mm=180,
                   height_mm=240, labels=False, x_axis=True, x_label="Grain size", y_label="MD",
                   depth_min=5, depth_max=28, alpha=.65, label_size=9, transparent=True,
                   axis_size=11, tick_size=8, legend_size=8, title="Custom section")
    fig, _ = render_columns(scheme, "A", "FORMATION", options=options)
    assert len(calls) == 2
    for kwargs, frame in calls:
        assert kwargs["display_mode"] == mode and kwargs["feature_mode"] == "merge"
        assert kwargs["ppi"] == 250 and kwargs["unit_borders"] is False
        assert frame.iloc[0]["top_grain"] == .5
        assert frame.iloc[0]["bottom_grain"] == 1
    assert fig.dpi == 120
    assert list(fig.get_size_inches()) == pytest.approx([180 / 25.4, 240 / 25.4])
    assert fig.axes[0].get_ylabel() == "MD [m]"
    assert all(ax.get_ylim() == (28, 5) for ax in fig.axes)
    assert not any(text.get_text() == "Upper" for ax in fig.axes for text in ax.texts)
    assert fig.legends
    assert fig.patch.get_alpha() == 0
    assert formatting.fontsizes == original_sizes
    assert list(mpl.rcParams["font.family"]) == original_family
    fig.savefig(tmp_path / f"{mode}-{location}.png")
    fig.savefig(tmp_path / f"{mode}-{location}.svg")


@pytest.mark.parametrize("options", [{"dpi": 0}, {"alpha": float("nan")},
                                    {"depth_min": 10, "depth_max": 5}, {"legend_columns": 1.5},
                                    {"legend_kwargs": []}, {"x_ticks_dict": {"f": -1}},
                                    {"display_mode": "bad"}, {"legend_titles": []}])
def test_invalid_column_style_rejected(options):
    from autopyro.stratigraphy import column_style

    with pytest.raises(ValueError):
        column_style(options)


def test_custom_lithology_and_grain_axis_are_local(data):
    from stratapy.setup import formatting
    from autopyro.stratigraphy import render_columns

    _, scheme, _ = data
    scheme["intervals"][0].update(lithology="custom_sand", top_grain="fine", bottom_grain="coarse")
    original = dict(formatting.lithologies)
    fig, _ = render_columns(scheme, "A", "FORMATION", options={
        "display_mode": "default", "x_axis": True,
        "x_ticks_dict": {"fine": .5, "coarse": 2}, "grain_brackets": {},
        "lithologies": {"custom_sand": ["607", "#e0b761", "Custom sand"]}})
    assert fig.axes[1].images
    assert formatting.lithologies == original
    assert "custom_sand" not in formatting.lithologies


def test_failed_render_restores_stratapy_styles(data):
    from stratapy.setup import formatting
    from autopyro.stratigraphy import render_columns

    _, scheme, _ = data
    scheme["intervals"][0]["lithology"] = "missing"
    original = dict(formatting.fontsizes)
    with pytest.raises(ValueError, match="Неизвестная литология"):
        render_columns(scheme, "A", "FORMATION", options={"axis_size": 25})
    assert formatting.fontsizes == original



def test_native_features_survive_loading(data, monkeypatch):
    from stratapy.core import LogObject
    from autopyro.stratigraphy import render_columns

    _, scheme, _ = data
    scheme["intervals"][0].update(lithology="607", top_grain="f", bottom_grain="m",
                                   minerals="biotite", features="ammonite", contact="sharp")
    original = LogObject.plot
    frames = []
    def record(self, *args, **kwargs):
        frames.append(self.original_df.copy())
        return original(self, *args, **kwargs)
    monkeypatch.setattr(LogObject, "plot", record)
    fig, _ = render_columns(scheme, "A", "FORMATION",
                            options={"display_mode": "default", "feature_mode": "merge"})
    assert frames[1].iloc[0]["minerals"] == "biotite"
    assert frames[1].iloc[0]["features"] == "ammonite"
    assert frames[1].iloc[0]["top_grain"] != frames[1].iloc[0]["bottom_grain"]
    assert fig.axes[1].collections or fig.axes[1].images or fig.axes[1].artists
    fig, _ = render_columns(scheme, "A", "FORMATION",
                            options={"display_mode": "default", "x_axis": True})
    texts = [text.get_text() for text in fig.axes[0].texts]
    assert texts.count("Upper") == texts.count("Lower") == 1
    assert "sand" in texts  # Native bracket labels are not overwritten by unit labels.
def test_interval_color_uses_scale_only_without_explicit_color():
    from autopyro.geology.stratigraphy import interval_color

    assert interval_color({"name": "Юрская система"}) == "#4C78B6"
    assert interval_color({"name": "Верхняя юра", "color": ""}) == "#60BACF"
    assert interval_color({"name": "Late Jurassic"}) == "#60BACF"
    assert interval_color({"name": "Jurassic", "color": "#112233"}) == "#112233"
    assert interval_color({"name": "Jurassic", "color": "#c7ddeb"}) == "#4C78B6"
    assert interval_color({"name": "Jurassic", "color": "#c7ddeb", "color_manual": True}) == "#c7ddeb"
    assert interval_color({"name": "Верхний"}) == "#c7ddeb"


def test_saved_legacy_column_and_plot_colors_survive_project_roundtrip(tmp_path):
    from matplotlib import rcParams
    from matplotlib.colors import to_hex
    from autopyro.geology.stratigraphy import render_columns
    from autopyro.visualization.plotting import category_key

    project = Project(columns=[Column("TOC", "number"), Column("S2", "number"),
                               Column("Возраст", "text")])
    x, y, age = (column.id for column in project.columns)
    project.rows = [Row({x: 1, y: 2, age: "Jurassic"}), Row({x: 3, y: 4, age: "Triassic"})]
    project.view["stratigraphy_scheme"] = {
        "format": "autopyro.stratigraphy", "version": 1, "id": "legacy", "revision": "1",
        "name": "Legacy", "depth_reference": "MD RKB", "depth_unit": "m",
        "intervals": [
            {"id": "j", "wellbore": "A", "level": "FORMATION", "name": "Jurassic",
             "top": 0, "bottom": 10, "color": "#c7ddeb"},
            {"id": "t", "wellbore": "A", "level": "FORMATION", "name": "Triassic",
             "top": 10, "bottom": 20, "color": "#923456"},
        ],
    }
    cycle = rcParams["axes.prop_cycle"].by_key()["color"]
    project.view["plots"] = {"Legacy": PlotSpec(
        x_id=x, y_id=y, color_by=age,
        style_maps={"color": {age: {
            category_key("Jurassic"): {"label": "Jurassic", "style": cycle[0]},
            category_key("Triassic"): {"label": "Triassic", "style": "#345678"},
        }}},
    ).to_dict()}
    old_path = tmp_path / "legacy.apyro"
    project.save(old_path)

    current_path = tmp_path / "resaved.apyro"
    for path in (old_path, current_path):
        loaded = Project.load(path)
        scheme = loaded.view["stratigraphy_scheme"]
        column, _ = render_columns(scheme, "A", "FORMATION")
        assert [to_hex(patch.get_facecolor()) for patch in column.axes[0].patches] == [
            "#4c78b6", "#923456",
        ]
        spec = PlotSpec.from_dict(loaded.view["plots"]["Legacy"])
        figure, _ = render_plot(loaded, loaded.row_map(), spec)
        assert [to_hex(color) for color in figure.axes[0].collections[0].get_facecolors()] == [
            "#4c78b6", "#345678",
        ]
        if path == old_path:
            loaded.view["plots"]["Legacy"] = spec.to_dict()
            loaded.save(current_path)

