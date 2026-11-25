"""Cached plotting preserves scientific masks, global styles and input isolation."""
from copy import deepcopy
from threading import Event
import subprocess
import sys

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from autopyro.io import build_project
from autopyro.plotting import PlotSpec, render_plot
from autopyro.plot_data import PlotDataCache, capture_input, input_token, PreparationCancelled


def test_plot_imports_do_not_load_scipy():
    subprocess.run([sys.executable, '-c',
        'import sys; import autopyro.plotting, autopyro.plot_data; assert "scipy" not in sys.modules'],
        check=True, timeout=30)


@pytest.fixture
def project():
    result = build_project(['X', 'Y', 'Group', 'Bottom'],
        [[1, 2, 'A', 3], [2, '<0.1', 'B', 4], [3, 'ND', 'B', 5], [4, 5, 'C', 6]], [2, 3, 4, 5],
        [(0, 'X', 'number', ''), (1, 'Y', 'number', 'm'), (2, 'Group', 'text', ''), (3, 'Bottom', 'number', 'm')], {})
    result.sync_measurements()
    return result


def prepare(project, spec, ids=None, cache=None, revision=0):
    ids = [r.id for r in project.rows] if ids is None else ids
    cache = cache or PlotDataCache()
    source = capture_input(project, ids, [spec], input_token(project, revision, [spec]), cache.snapshot)
    return cache.prepare(source, ids, spec), cache


@pytest.mark.parametrize('kind', ['scatter', 'depth', 'multi_depth', 'hist', 'box', 'violin',
                                 'mpl:plot', 'mpl:scatter', 'mpl:imshow', 'mpl3d:scatter'])
def test_cached_renderer_matches_existing_counts_and_global_categories(project, kind):
    x, y, group, bottom = [c.id for c in project.columns]
    spec = PlotSpec(kind=kind, x_id=x, y_id=y, z_id=bottom, track_ids=[x, bottom])
    if kind in ('scatter', 'depth', 'multi_depth', 'hist', 'box', 'violin', 'mpl:scatter', 'mpl3d:scatter'):
        spec.color_by = group
    ids = [r.id for r in project.rows[:2]]
    reference = deepcopy(spec)
    fig1, expected = render_plot(project, ids, reference)
    data, _ = prepare(project, spec, ids)
    fig2, actual = render_plot(data.project, ids, spec, prepared=data)
    FigureCanvasAgg(fig2).draw()
    assert actual['plotted'] == expected['plotted']
    assert actual['excluded'] == expected['excluded']
    assert spec.style_maps == reference.style_maps
    fig1.clear()
    fig2.clear()


def test_warm_style_update_does_not_build_frames_or_reparse(project, monkeypatch):
    import autopyro.plot_data as module
    x, y, group, _ = [c.id for c in project.columns]
    spec = PlotSpec(x_id=x, y_id=y, color_by=group)
    data, cache = prepare(project, spec)
    misses = cache.misses
    def fail(*args, **kwargs):
        raise AssertionError('Warm style changes must reuse prepared data')
    monkeypatch.setattr(module, 'classify_series', fail)
    monkeypatch.setattr(module, 'project_frame', fail)
    spec.title = 'New style'
    warm = cache.prepare(data.source, [r.id for r in project.rows], spec)
    assert cache.misses == misses and cache.hits > 0
    assert warm.local is data.local
    render_plot(warm.project, list(warm.scope), spec, prepared=warm)


def test_snapshot_is_independent_and_contains_no_scientific_history(project):
    spec = PlotSpec(x_id=project.columns[0].id, y_id=project.columns[1].id)
    data, _ = prepare(project, spec, [project.rows[0].id])
    project.rows[0].values[spec.x_id] = 999
    assert data.project.rows[0].values[spec.x_id] == 1
    assert len(data.project.rows) == 1
    assert data.project.science.measurements == {} and data.project.history == []


def test_qc_decision_invalidates_without_a_table_revision(project):
    x, y = [c.id for c in project.columns[:2]]
    spec = PlotSpec(x_id=x, y_id=y)
    data, cache = prepare(project, spec)
    before = data.source.token
    measurement = project.science.current(project.rows[0].id, y)
    project.view['qc_exclusions'] = {'plots': {measurement.id: {'revision': measurement.revision}}}
    updated, _ = prepare(project, spec, cache=cache)
    assert updated.source.token != before
    _, result = render_plot(updated.project, list(updated.scope), spec, prepared=updated)
    assert result['excluded'][x]['qc_excluded'] == 1


def test_policy_change_and_small_budget_do_not_reuse_wrong_classification(project):
    x, y = [c.id for c in project.columns[:2]]
    project.columns[1].metadata.update(method='linear', stale=False)
    spec = PlotSpec(x_id=x, y_id=y)
    cache = PlotDataCache(max_bytes=1)
    data, _ = prepare(project, spec, cache=cache)
    assert cache.bytes <= cache.max_bytes
    assert set(data.columns(data.local, [y])[y]['status']) == {'excluded_interpolation'}
    spec.include_interpolated = True
    updated, _ = prepare(project, spec, cache=cache)
    assert updated.columns(updated.local, [y])[y]['number'].notna().sum() == 2


def test_calculated_policy_is_independent_and_invalidates_parsed_cache(project, tmp_path):
    from autopyro.plotting import save_plot

    x, y = [column.id for column in project.columns[:2]]
    project.columns[0].metadata['calculation'] = {'formula': 'test'}
    spec = PlotSpec(x_id=x, y_id=y, include_interpolated=True)
    ids = list(project.row_map())
    enabled, cache = prepare(project, spec, ids)
    _, included = render_plot(project, ids, spec, prepared=enabled)
    assert included['plotted'][x] == 2
    spec.include_calculated = False
    spec = PlotSpec.from_dict(spec.to_dict())
    disabled, cache = prepare(project, spec, ids, cache)
    _, excluded = render_plot(project, ids, spec, prepared=disabled)
    assert excluded['plotted'][x] == 0
    assert excluded['excluded'][x]['excluded_calculation'] == len(ids)
    exported = save_plot(tmp_path / 'calculated-policy.svg', project, ids, spec, prepared=disabled)
    assert exported['plotted'] == excluded['plotted']
    _, direct = render_plot(project, ids, spec)
    assert direct['excluded'] == excluded['excluded']
    spec.include_calculated = True
    restored, _ = prepare(project, spec, ids, cache)
    _, included_again = render_plot(project, ids, spec, prepared=restored)
    assert included_again['plotted'] == included['plotted']


def test_original_policy_is_independent_of_derived_types_and_cached(project, tmp_path):
    from autopyro.plotting import save_plot

    x, y = project.columns[:2]
    ids = list(project.row_map())
    spec = PlotSpec(x_id=x.id, y_id=y.id)
    prepared, cache = prepare(project, spec, ids)
    _, before = render_plot(project, ids, spec, prepared=prepared)
    assert before['plotted'][x.id] == 2
    spec.include_original = False
    spec = PlotSpec.from_dict(spec.to_dict())
    prepared, cache = prepare(project, spec, ids, cache)
    _, excluded = render_plot(project, ids, spec, prepared=prepared)
    assert excluded['excluded'][x.id]['excluded_original'] == len(ids)
    assert save_plot(tmp_path / 'original-policy.svg', project, ids, spec, prepared=prepared)['plotted'][x.id] == 0
    spec.include_original = True
    prepared, _ = prepare(project, spec, ids, cache)
    _, restored = render_plot(project, ids, spec, prepared=prepared)
    assert restored['plotted'] == before['plotted']
    x.metadata['calculation'] = {'formula': 'test'}
    y.metadata['method'] = 'linear'
    spec.include_original = False
    spec.include_interpolated = True
    prepared, _ = prepare(project, spec, ids)
    _, derived = render_plot(project, ids, spec, prepared=prepared)
    assert derived['plotted'] == before['plotted']
    _, direct = render_plot(project, ids, spec)
    assert direct['plotted'] == derived['plotted']


@pytest.mark.parametrize('kind', ['scatter', 'mpl:scatter'])
def test_local_filter_uses_frozen_values_and_measurement_metadata(project, kind, tmp_path):
    from autopyro.plotting import save_plot

    x, y, group = [column.id for column in project.columns[:3]]
    spec = PlotSpec(kind=kind, x_id=x, y_id=y, filter_combine='AND', filter_rules=[
        {'column': group, 'operator': 'contains', 'value': 'a'},
        {'column': x, 'operator': 'gt', 'value': 0},
    ])
    ids = list(project.row_map())
    prepared, _ = prepare(project, spec, ids)
    project.rows[0].values[group] = 'changed'
    figure, result = render_plot(project, ids, spec, prepared=prepared)
    assert result['artists'][0]._autopyro_row_ids == [ids[0]]
    exported = save_plot(tmp_path / 'filtered.svg', project, ids, spec, prepared=prepared)
    assert exported['plotted'] == result['plotted']
    spec.filter_rules = [{'column': y, 'operator': 'unit_eq', 'value': 'custom'}]
    measurement = project.science.current(ids[0], y)
    measurement.unit = 'custom'
    prepared, _ = prepare(project, spec, ids)
    measurement.unit = 'changed'
    _, unit_filtered = render_plot(project, ids, spec, prepared=prepared)
    assert unit_filtered['artists'][0]._autopyro_row_ids == [ids[0]]
    spec.filter_rules = [{'column': y, 'operator': 'non_detect', 'negate': True}]
    spec.filter_combine = 'OR'
    _, result = render_plot(project, ids, spec, prepared=prepared)
    assert result['plotted'][x] == 2
    spec.filter_rules = [{'column': 'removed', 'operator': 'not_empty'}]
    with pytest.raises(ValueError, match='Колонка локального фильтра удалена'):
        render_plot(project, ids, spec, prepared=prepared)


@pytest.mark.parametrize("kind", ["scatter", "mpl:scatter"])
@pytest.mark.parametrize("metadata_key", ["nonpyro", "stratigraphy"])
def test_derived_result_policy_matches_axis_type_and_frozen_export(project, kind, metadata_key, tmp_path):
    from autopyro.plotting import save_plot

    x, y = project.columns[:2]
    x.metadata[metadata_key] = {"output": "value"}
    y.metadata[metadata_key] = {"output": "value"}
    ids = list(project.row_map())
    spec = PlotSpec(kind=kind, x_id=x.id, y_id=y.id, show_result_type=True,
                    include_original=False, include_calculated=True)
    prepared, cache = prepare(project, spec, ids)
    figure, result = render_plot(project, ids, spec, prepared=prepared)
    assert result["plotted"][x.id] == 2
    assert figure.axes[0].get_xlabel().endswith(" · расчёт")
    spec.include_calculated = False
    excluded, cache = prepare(project, spec, ids, cache)
    assert set(excluded.columns(excluded.local, [x.id])[x.id]["status"]) == {"excluded_calculation"}
    if kind == "mpl:scatter":
        with pytest.raises(ValueError, match="Нет подходящих данных"):
            render_plot(project, ids, spec, prepared=excluded)
        with pytest.raises(ValueError, match="Нет подходящих данных"):
            render_plot(project, ids, spec)
    else:
        _, hidden = render_plot(project, ids, spec, prepared=excluded)
        assert hidden["plotted"][x.id] == 0
        _, direct = render_plot(project, ids, spec)
        assert direct["excluded"] == hidden["excluded"]
    spec.include_calculated = True
    prepared, _ = prepare(project, spec, ids, cache)
    path = tmp_path / "derived.svg"
    exported = save_plot(path, project, ids, spec, prepared=prepared)
    assert exported["plotted"] == result["plotted"]
    assert "расчёт" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("prepared_input", [False, True])
def test_categorical_library_arguments_respect_qc_and_result_type_policy(project, monkeypatch, prepared_input, tmp_path):
    from types import SimpleNamespace
    from autopyro.visualization import library_plot, plotting

    x, y, group = project.columns[:3]
    ids = [project.rows[0].id, project.rows[-1].id]
    group.metadata["nonpyro"] = {"output": "category"}
    excluded = project.science.current(ids[0], group.id)
    project.view["qc_exclusions"] = {"plots": {excluded.id: {"revision": excluded.revision}}}
    received = []

    def scatterplot(*, x, y, hue, ax):
        received.append(list(hue))
        ax.scatter(x, y)

    kind = "seaborn:scatterplot"
    monkeypatch.setattr(library_plot, "library_kinds", lambda: {kind: (None, "", scatterplot)})
    monkeypatch.setattr(library_plot, "library_parameters", lambda _: ["x", "y", "hue"])
    monkeypatch.setattr(library_plot, "controls_for",
                        lambda _: SimpleNamespace(axes=(("x", "x"), ("y", "y"))))
    monkeypatch.setattr(plotting, "KINDS", {kind: kind})
    spec = PlotSpec(kind=kind, x_id=x.id, y_id=y.id, library_bindings={"hue": group.id})
    prepared = prepare(project, spec, ids)[0] if prepared_input else None
    figure, result = library_plot.render_library_plot(project, ids, spec, prepared=prepared)
    assert received[-1] == ["C"]
    assert result["excluded"][x.id] == {"qc_excluded": 1}
    exported = plotting.save_plot(tmp_path / "categorical.svg", project, ids, spec, prepared=prepared)
    assert exported["plotted"] == result["plotted"]
    assert received[-1] == ["C"]
    spec.include_calculated = False
    prepared = prepare(project, spec, ids)[0] if prepared_input else None
    with pytest.raises(ValueError, match="Нет подходящих данных"):
        library_plot.render_library_plot(project, ids, spec, prepared=prepared)
    assert len(received) == 2


def test_cancelled_preparation_is_not_returned(project):
    spec = PlotSpec(x_id=project.columns[0].id, y_id=project.columns[1].id)
    source = capture_input(project, [r.id for r in project.rows], [spec], (0,))
    stop = Event()
    stop.set()
    with pytest.raises(PreparationCancelled):
        PlotDataCache().prepare(source, [r.id for r in project.rows], spec, stop)


@pytest.mark.parametrize('kind', ['scatter', 'mpl:scatter', 'multi_depth'])
def test_result_type_axis_labels_use_column_metadata_in_preview_and_export(project, kind, tmp_path):
    from autopyro.plotting import save_plot

    x, y, _, other_x = project.columns
    y.metadata['method'] = 'linear'
    x.metadata['calculation'] = {'formula': 'demo'}
    spec = PlotSpec(kind=kind, x_id=x.id, y_id=y.id, track_ids=[x.id, other_x.id],
                    include_interpolated=True, show_result_type=True)
    spec.x_axis.label = 'Custom X'
    ids = list(project.row_map())
    spec = PlotSpec.from_dict(spec.to_dict())
    prepared, _ = prepare(project, spec, ids)
    figure, result = render_plot(project, ids, spec, prepared=prepared)
    assert figure.axes[0].get_xlabel() == 'Custom X · расчёт'
    assert figure.axes[0].get_ylabel() == 'Y [m] · интерполяция'
    if kind == 'multi_depth':
        assert figure.axes[1].get_xlabel() == 'Custom X · исходные данные'
    path = tmp_path / 'result-types.svg'
    exported = save_plot(path, project, ids, spec, prepared=prepared)
    assert exported['plotted'] == result['plotted']
    assert 'Custom X · расчёт' in path.read_text(encoding='utf-8')


@pytest.mark.parametrize('kind', ['scatter', 'depth', 'multi_depth', 'mpl:scatter'])
def test_qc_outline_uses_frozen_working_measurement_and_respects_exclusions(project, kind):
    x, y, *_ = [column.id for column in project.columns]
    first, second = project.rows[0].id, project.rows[-1].id
    marked = project.science.current(first, x)
    marked.qc_status = 'expert pending'
    excluded = project.science.current(second, y)
    excluded.qc_status = 'review'
    project.view['qc_exclusions'] = {'plots': {excluded.id: {'revision': excluded.revision}}}
    spec = PlotSpec(kind=kind, x_id=x, y_id=y, track_ids=[x], qc_outline=True)
    prepared, _ = prepare(project, spec, [first, second])
    assert prepared.project.view['plot_measurement_qc'][x][first] == (marked.id, marked.revision, 'expert pending')
    marked.qc_status = 'unchecked'
    figure, result = render_plot(project, [first, second], spec, prepared=prepared)
    axes = figure.axes[0]
    outline, = [artist for artist in axes.collections if artist.get_gid() == 'measurement-qc-outline']
    assert outline.get_offsets().tolist() == [[1, 2]]
    assert outline not in result['artists']
    assert sum(result['plotted'].values()) == 1
    assert [label.get_text() for label in axes.get_legend().get_texts()] == ['QC · expert pending']


@pytest.mark.parametrize('kind', ['scatter', 'mpl:scatter'])
def test_qc_legend_distinguishes_working_status_combinations(project, kind, tmp_path):
    from autopyro.plotting import save_plot

    x, y, group, *_ = [column.id for column in project.columns]
    first, second = project.rows[0].id, project.rows[-1].id
    project.science.current(first, x).qc_status = 'review'
    project.science.current(second, x).qc_status = 'review'
    project.science.current(second, y).qc_status = 'accepted'
    spec = PlotSpec(kind=kind, x_id=x, y_id=y, color_by=group, qc_outline=True)
    prepared, _ = prepare(project, spec, [first, second])
    figure, result = render_plot(project, [first, second], spec, prepared=prepared)
    labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert 'QC · review' in labels
    assert 'QC · accepted / review' in labels
    outlines = [artist for artist in figure.axes[0].collections
                if (artist.get_gid() or '').startswith('measurement-qc-outline')]
    assert len(outlines) == 2
    assert sorted(len(artist.get_offsets()) for artist in outlines) == [1, 1]
    marker_before = next(artist.get_paths()[0].vertices.copy() for artist in outlines
                         if artist.get_label() == 'QC · review')
    filtered_figure, _ = render_plot(project, [first], spec, prepared=prepared)
    marker_after, = [artist for artist in filtered_figure.axes[0].collections
                     if artist.get_label() == 'QC · review']
    assert (marker_before == marker_after.get_paths()[0].vertices).all()
    assert sum(result['plotted'].values()) == 2
    path = tmp_path / 'qc-statuses.svg'
    save_plot(path, project, [first, second], spec, prepared=prepared)
    assert 'QC · accepted / review' in path.read_text(encoding='utf-8')



@pytest.mark.parametrize('kind', ['scatter', 'depth', 'multi_depth', 'mpl:scatter', 'mpl:plot', 'mpl:step',
                                  'mpl:errorbar', 'mpl:semilogx', 'mpl:semilogy', 'mpl:loglog'])
def test_measurement_bars_capture_current_results_and_keep_valid_log_axis(project, kind):
    from matplotlib.collections import LineCollection

    x, y, _, _ = [c.id for c in project.columns]
    rid = project.rows[0].id
    for column, uncertainty in [(x, 1.5), (y, .25)]:
        measurement = project.science.current(rid, column)
        measurement.uncertainty = uncertainty
        measurement.uncertainty_type = 'absolute'
    spec = PlotSpec(kind=kind, x_id=x, y_id=y, track_ids=[x])
    ids = [row.id for row in project.rows]
    disabled, cache = prepare(project, spec, ids)
    assert not disabled.project.view['measurement_uncertainty']
    spec.measurement_error_bars = True
    spec.x_axis.scale = 'log'
    assert not disabled.source.covers(input_token(project, 0, [spec]), ids, [spec])
    prepared, _ = prepare(project, spec, ids, cache)
    project.science.current(rid, y).uncertainty = .75
    fig, result = render_plot(project, ids, spec, prepared=prepared)
    bars = [c for c in fig.axes[0].collections if isinstance(c, LineCollection)]
    assert len(bars) == 1
    assert [segment.tolist() for segment in bars[0].get_segments()] == [[[1, 1.75], [1, 2.25]]]
    assert result['plotted'][x] == 2
    assert 'логарифмической оси' in result['notices'][0]


def test_library_errorbar_prefers_explicit_errors_on_each_bound_axis(project, tmp_path):
    from autopyro.plotting import save_plot

    x, y, _, error = [column.id for column in project.columns]
    rid = project.rows[0].id
    for cid in (x, y):
        measurement = project.science.current(rid, cid)
        measurement.uncertainty = .25
        measurement.uncertainty_type = 'absolute'
    spec = PlotSpec(kind='mpl:errorbar', x_id=x, y_id=y, measurement_error_bars=True,
                    library_bindings={'xerr': error})
    prepared, _ = prepare(project, spec, [rid])
    figure, _ = render_plot(project, [rid], spec, prepared=prepared)
    explicit, recorded = figure.axes[0].collections
    assert explicit.get_segments()[0].tolist() == [[-2, 2], [4, 2]]
    assert recorded.get_gid() == 'measurement-uncertainty-1'
    assert recorded.get_segments()[0].tolist() == [[1, 1.75], [1, 2.25]]
    path = tmp_path / 'errorbar.svg'
    save_plot(path, project, [rid], spec, prepared=prepared)
    assert 'id="measurement-uncertainty-1"' in path.read_text(encoding='utf-8')


@pytest.mark.parametrize('kind, log_z', [('mpl3d:scatter', False), ('mpl3d:scatter', True),
                                         ('mpl3d:errorbar', False)])
def test_3d_uncertainty_preserves_z_binding_and_explicit_error_precedence(project, kind, log_z, tmp_path):
    from autopyro.plotting import save_plot
    from autopyro.visualization.plot_controls import controls_for

    x, y, _, z = [column.id for column in project.columns]
    rid = project.rows[0].id
    for cid, uncertainty in ((x, .25), (y, .5), (z, 4 if log_z else 1.25)):
        measurement = project.science.current(rid, cid)
        measurement.uncertainty = uncertainty
        measurement.uncertainty_type = 'absolute'
    bindings = {dict(controls_for(kind).axes)['z']: z}
    if kind == 'mpl3d:errorbar':
        bindings['zerr'] = x
    spec = PlotSpec(kind=kind, x_id=x, y_id=y, library_bindings=bindings, measurement_error_bars=True)
    if log_z:
        spec.z_axis.scale = 'log'
    prepared, _ = prepare(project, spec, [rid])
    assert set(prepared.project.view['measurement_uncertainty']) == {x, y, z}
    figure, result = render_plot(project, [rid], spec, prepared=prepared)
    ax = figure.axes[0]
    bars, = [artist for artist in ax.collections if artist.get_gid() == 'measurement-uncertainty-1']
    expected = [[[.75, 2, 3], [1.25, 2, 3]], [[1, 1.5, 3], [1, 2.5, 3]]]
    if log_z:
        assert ax.get_zscale() == 'log'
        assert any('логарифмической оси' in notice for notice in result['notices'])
    elif kind == 'mpl3d:scatter':
        expected.append([[1, 2, 1.75], [1, 2, 4.25]])
        assert ax.get_zlim()[0] < 1.75 and ax.get_zlim()[1] > 4.25
    assert [list(segment) for segment in bars._segments3d] == expected
    assert sum(result['plotted'].values()) == 1
    path = tmp_path / 'uncertainty-3d.svg'
    save_plot(path, project, [rid], spec, prepared=prepared)
    assert 'id="measurement-uncertainty-1"' in path.read_text(encoding='utf-8')


@pytest.mark.parametrize('kind', ['depth', 'multi_depth'])
def test_interval_samples_draw_measurement_uncertainty_at_sampling_midpoint(project, kind, tmp_path):
    from autopyro.plotting import save_plot

    x, top, _, bottom = [column.id for column in project.columns]
    rid = project.rows[0].id
    for cid in (x, top, bottom):
        measurement = project.science.current(rid, cid)
        measurement.uncertainty = .25
        measurement.uncertainty_type = 'absolute'
    spec = PlotSpec(kind=kind, x_id=x, y_id=top, depth_bottom_id=bottom, track_ids=[x],
                    measurement_error_bars=True)
    prepared, _ = prepare(project, spec, [rid])
    assert set(prepared.project.view['measurement_uncertainty']) == {x}
    figure, result = render_plot(project, [rid], spec, prepared=prepared)
    assert result['artists'][0].get_segments()[0].tolist() == [[1, 2], [1, 3]]
    bars, = [artist for artist in figure.axes[0].collections
             if artist.get_gid() == 'measurement-uncertainty-1']
    assert bars.get_segments()[0].tolist() == [[.75, 2.5], [1.25, 2.5]]
    path = tmp_path / 'interval.svg'
    save_plot(path, project, [rid], spec, prepared=prepared)
    assert 'id="measurement-uncertainty-1"' in path.read_text(encoding='utf-8')



def test_snapshot_keeps_archived_curve_uncertainty_and_qc(project, tmp_path):
    from matplotlib.collections import LineCollection
    from autopyro.visualization.plot_snapshot import read_snapshot, write_snapshot

    x, y, _, _ = [column.id for column in project.columns]
    ids = [row.id for row in project.rows]
    record = {'id': 'fit', 'name': 'Fit', 'settings': {'x': x, 'y': y, 'method': 'ols'},
              'grid': [1, 4], 'curve': [2, 5], 'axis_units': ['', 'm']}
    project.view['curve_fit_results'] = [record]
    uncertain = project.science.current(ids[0], y)
    uncertain.uncertainty = .25
    uncertain.uncertainty_type = 'absolute'
    excluded = project.science.current(ids[3], y)
    project.view['qc_exclusions'] = {'plots': {excluded.id: {'revision': excluded.revision}}}
    spec = PlotSpec(x_id=x, y_id=y, curve_archive_id='fit', measurement_error_bars=True)
    prepared, _ = prepare(project, spec)
    path = tmp_path / 'plot.aplot.json'
    write_snapshot(path, {'spec': spec, 'source': prepared.source, 'ids': ids, 'revision': 0})
    spec, source, captured, _, _ = read_snapshot(path)
    replay = PlotDataCache().prepare(source, captured, spec)
    figure, result = render_plot(source.project, captured, spec, prepared=replay)
    assert result['excluded'][x]['qc_excluded'] == 1
    assert figure.axes[0].lines[0].get_ydata().tolist() == [2, 5]
    intervals = next(c for c in figure.axes[0].collections if isinstance(c, LineCollection))
    assert intervals.get_segments()[0].tolist() == [[1, 1.75], [1, 2.25]]
