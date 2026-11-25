import pytest

from autopyro.core import Project
from autopyro.io import build_project
from autopyro.plotting import AxisSpec, PlotSpec, render_plot, save_plot
from autopyro.reporting import ReportOptions, build_report, render_html
from autopyro.stratigraphy import evaluate_project, recompute, result_column, validate_binding


@pytest.fixture
def data():
    project = build_project(["Well", "Top", "Bottom", "TOC"], [["A", "2", "8", "1"]], [2],
        [(0,"Well","text",""),(1,"Top","number","m"),(2,"Bottom","number","m"),(3,"TOC","number","%wt")], {})
    well, top, bottom, _ = [c.id for c in project.columns]
    scheme = {"format":"autopyro.stratigraphy","version":1,"id":"demo","revision":"1","name":"Demo",
        "source":"Synthetic","depth_unit":"m","depth_reference":"MD RKB", "intervals":[
            {"id":"a","wellbore":"A","name":"Upper","level":"FORMATION","top":0,"bottom":10},
            {"id":"b","wellbore":"A","name":"Lower","level":"FORMATION","top":10,"bottom":20},
            {"id":"g","wellbore":"A","name":"Group","level":"GROUP","top":0,"bottom":20}]}
    binding = {"well_id":well,"depth_id":top,"bottom_id":bottom,"sample_mode":"interval",
        "depth_unit":"m","depth_reference":"MD RKB"}
    project.view["stratigraphy"] = {"scheme":scheme,"binding":binding}
    return project, scheme, binding


def assign(data, top, bottom):
    project, scheme, binding = data
    project.rows[0].values[binding["depth_id"]] = top
    project.rows[0].values[binding["bottom_id"]] = bottom
    return evaluate_project(project, [project.rows[0].id], scheme, binding)["results"][0]["levels"]["FORMATION"]


@pytest.mark.parametrize("top,bottom,status,names,fraction", [
    (2,8,"matched",["Upper"],1), (0,10,"matched",["Upper"],1),
    (10,20,"matched",["Lower"],1), (8,12,"cross_boundary",["Upper","Lower"],1),
    (-5,5,"partial",["Upper"],.5), (18,22,"partial",["Lower"],.5),
    (20,25,"outside",[],0), (3,3,"invalid_interval",[],None),
    (8,2,"invalid_interval",[],None), (None,8,"insufficient_data",[],None),
    ("<2",8,"insufficient_data",[],None), (2,"ND","insufficient_data",[],None)])
def test_interval_statuses(data,top,bottom,status,names,fraction):
    result = assign(data,top,bottom)
    assert result["status"] == status
    assert result["names"] == names
    assert result["coverage_fraction"] == fraction


@pytest.mark.parametrize(('decimal', 'separator'), [(',', ';'), ('.', ',')])
def test_revision_comparison_detects_redistributed_interval_sample(data, tmp_path, decimal, separator):
    import csv
    from codecs import BOM_UTF8
    from copy import deepcopy
    from types import SimpleNamespace
    from autopyro.data_io.file_output import write_excel_csv
    from autopyro.stratigraphy_revisions import interval_changes, sample_changes
    from autopyro.ui.widgets.stratigraphy_revisions import RevisionComparisonDialog

    project, before, binding = data
    sample = project.rows[0]
    sample.values[binding['depth_id']] = 8
    sample.values[binding['bottom_id']] = 14
    after = deepcopy(before)
    after['intervals'][0]['bottom'] = 12
    after['intervals'][1]['top'] = 12
    _, identities = interval_changes(before, after)
    changes = sample_changes(project, [sample.id], before, after, binding, identities)
    assert len(changes) == 1
    _, level, old, new = changes[0]
    assert level == 'FORMATION'
    assert old['status'] == new['status'] == 'cross_boundary'
    assert old['names'] == new['names'] == ['Upper', 'Lower']
    assert [part['length'] for part in old['overlaps']] == [2, 4]
    assert [part['length'] for part in new['overlaps']] == [4, 2]
    report = SimpleNamespace(compared=(before, after), binding=binding, csv_results=[None, changes])
    headers, rows = RevisionComparisonDialog.csv_table(report, 1)
    path = tmp_path / 'samples.csv'
    write_excel_csv(path, headers, rows, decimal)
    assert path.read_bytes().startswith(BOM_UTF8)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream, delimiter=separator))
    totals = [row for row in records if row['Тип строки'] == 'Итог по пробе']
    assert [(row['Сторона сравнения'], row['Покрытие пробы, %']) for row in totals] == [
        ('До', '100'), ('После', '100')]
    details = [row for row in records if row['Тип строки'] == 'Подразделение']
    assert [row['Длина отбора в подразделении'] for row in details] == ['2', '4', '4', '2']
    assert all(row['Покрытие пробы, %'] == '' for row in details)


def test_sample_track_preserves_interval_and_shifts_to_reference_horizon(data):
    from autopyro.stratigraphy import render_columns
    from autopyro.stratigraphy_samples import sample_positions

    project, scheme, binding = data
    row = project.rows[0]
    positions = sample_positions(project, [row.id], scheme, binding, {'A'})
    figure, _ = render_columns(scheme, 'A', 'FORMATION', samples=positions, options={
        'correlation': {'edge': 'bottom', 'anchors': {'A': 'a'}},
    })
    axis = figure.axes[-1]
    collection, samples = next(iter(axis._autopyro_sample_artists.items()))
    assert tuple(samples[0]) == (row.id, 2, 8)
    assert collection.get_segments()[0][:, 1].tolist() == [-8, -2]
    assert axis.get_ylim() == figure.axes[0].get_ylim()
    row.values[binding['bottom_id']] = 1
    assert sample_positions(project, [row.id], scheme, binding, {'A'}) == {}


def test_gaps_overlap_lengths_and_union_are_independent(data):
    project, scheme, binding = data
    scheme["intervals"][1]["top"] = 12
    result = assign(data,8,16)
    assert result["status"] == "partial"
    assert result["coverage_fraction"] == .75
    assert result["uncovered_length"] == 2
    scheme["intervals"][1]["top"] = 8
    result = assign(data,5,15)
    assert result["status"] == "ambiguous"
    assert result["ambiguous_length"] == 2
    assert result["coverage_fraction"] == 1
    assert sum(item["fraction"] for item in result["overlaps"]) == pytest.approx(1.2)
    assert result["covered_length"] == 10
    assert evaluate_project(project,[project.rows[0].id],scheme,binding)["results"][0]["levels"]["GROUP"]["status"] == "matched"


def test_unit_validation_and_lengths_in_sample_units(data):
    project, scheme, binding = data
    project.columns[1].unit = project.columns[2].unit = "ft"
    binding["depth_unit"] = "ft"
    result = assign(data,10,20)
    assert result["status"] == "matched"
    assert result["covered_length"] == pytest.approx(10)
    project.columns[2].unit = "m"
    with pytest.raises(ValueError,match="единиц"):
        validate_binding(project,scheme,binding)
    project.columns[2].unit = "ft"
    binding["bottom_id"] = binding["depth_id"]
    with pytest.raises(ValueError,match="разные"):
        validate_binding(project,scheme,binding)


def test_interpolation_excluded_even_only_lower_depth(data):
    project, _, _ = data
    project.columns[2].metadata["method"] = "linear"
    assert assign(data,2,8)["status"] == "interpolation_excluded"


def test_columns_report_and_roundtrip_preserve_ranges(data,tmp_path):
    project, scheme, binding = data
    name, status = result_column("FORMATION"), result_column("FORMATION",status=True)
    project.columns.extend([name,status])
    assign(data,8,12)
    recompute(project)
    assert project.rows[0].values[name.id] is None
    assert "границ" in project.rows[0].values[status.id]
    report = build_report(project,[project.rows[0].id],ReportOptions((project.columns[3].id,)))
    assert report["stratigraphy"]["sample_mode"] == "interval"
    assert report["stratigraphy"]["results"][0]["bottom"] == 12
    assert "Доля покрытия" in render_html(report)
    project.save(tmp_path/"interval.apyro")
    restored = Project.load(tmp_path/"interval.apyro")
    assert restored.view["stratigraphy"]["binding"]["bottom_id"] == binding["bottom_id"]
    assert restored.rows[0].values[name.id] is None


def spec_for(data):
    project,scheme,binding = data
    return PlotSpec(kind="depth", x_id=project.columns[3].id,y_id=binding["depth_id"],
        depth_bottom_id=binding["bottom_id"],y_axis=AxisSpec(inverted=True),
        stratigraphy={"scheme":scheme,"binding":binding,"level":"FORMATION"})


def test_interval_plot_draws_full_ranges_and_numeric_styles(data,tmp_path):
    project, _, _ = data
    spec = spec_for(data)
    spec.color_by = spec.size_by = project.columns[3].id
    spec.color_mode = "numeric"
    figure,result = render_plot(project,[project.rows[0].id],spec,selected_ids=[project.rows[0].id])
    assert result["artists"][0].get_segments()[0].tolist() == [[1,2],[1,8]]
    assert result["artists"][0]._autopyro_row_ids == [project.rows[0].id]
    assert result["axes"][0].yaxis.get_inverted()
    assert min(result["axes"][0].get_ylim()) < 2 and max(result["axes"][0].get_ylim()) > 8
    assert not any(hasattr(artist,"_offsets") and artist.__class__.__name__ == "PathCollection" for artist in result["axes"][0].collections)
    save_plot(tmp_path/"interval.svg",project,[project.rows[0].id],spec)
    assert (tmp_path/"interval.svg").stat().st_size > 1000
    spec.depth_bottom_id = ""
    with pytest.raises(ValueError,match="низ"):
        render_plot(project,[project.rows[0].id],spec)


def test_invalid_plot_range_excluded_and_wrong_kind_rejected(data):
    project,_,binding = data
    spec = spec_for(data)
    project.rows[0].values[binding["bottom_id"]] = 1
    _,result = render_plot(project,[project.rows[0].id],spec)
    assert result["plotted"][spec.x_id] == 0
    assert result["excluded"][spec.x_id]["invalid_depth_interval"] == 1
    spec.kind = "scatter"
    with pytest.raises(ValueError,match="глубинного"):
        render_plot(project,[],spec)


def test_long_interval_legend_fits_track(data):
    from autopyro.core import Column
    project,_,_ = data
    label = Column("Стратиграфическая принадлежность")
    project.columns.append(label)
    project.rows[0].values[label.id] = "Пересекает границу подразделений"
    spec = spec_for(data)
    spec.color_by = label.id
    spec.width_mm = 90
    figure,result = render_plot(project,[project.rows[0].id],spec)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(figure).draw()
    ax = result["axes"][0]
    legend = ax.get_legend()
    assert "\n" in legend.get_texts()[0].get_text()
    assert legend.get_window_extent().width < ax.get_window_extent().width
    for key in ("legend_size", "legend_columns"):
        original = getattr(spec,key)
        setattr(spec,key,0)
        with pytest.raises(ValueError,match="легенды"):
            render_plot(project,[project.rows[0].id],spec)
        setattr(spec,key,original)


@pytest.mark.parametrize("legend_columns", [1, 2])
@pytest.mark.parametrize("legend_size", [8, 10])
def test_depth_legend_measured_after_stratigraphic_layout(data, tmp_path, legend_columns, legend_size):
    from copy import deepcopy
    from autopyro.core import Column, Row
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    project, _, _ = data
    label = Column("Стратиграфическая принадлежность")
    project.columns.append(label)
    first = "Пересекает границу подразделений"
    second = "Принадлежит выбранному подразделению"
    project.rows[0].values[label.id] = first
    other = Row(deepcopy(project.rows[0].values))
    other.values[label.id] = second
    project.rows.append(other)
    spec = spec_for(data)
    spec.color_by = label.id
    spec.width_mm = 90
    spec.legend_size, spec.legend_columns = legend_size, legend_columns
    figure, result = render_plot(project, project.row_map(), spec)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    ax = result["axes"][0]
    legend = ax.get_legend()
    assert legend.get_window_extent().width < ax.get_window_extent().width
    assert legend.get_window_extent().height < ax.get_window_extent().height
    assert [text.get_fontsize() for text in legend.get_texts()] == [legend_size, legend_size]
    for text, value in zip(legend.get_texts(), (first, second)):
        assert "".join(text.get_text().split()) == "".join(f"{label.name}: {value}".split())
    figure.savefig(tmp_path / "depth-legend.png", dpi=144)
