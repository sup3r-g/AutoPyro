from copy import deepcopy

import numpy as np
import pytest
from matplotlib.colors import to_rgb
from PIL import Image

from autopyro.io import build_project
from autopyro.plotting import AxisSpec, PlotSpec, category_key, render_plot, save_plot


@pytest.fixture
def data():
    return build_project(["Well", "TOC", "S2", "MD", "Facies"],
                         [["A", "1", "2", "1000", "shale"], ["A", "2", "4", "1010", "sand"],
                          ["B", "3", "6", "1020", "shale"], ["B", "4", "8", "1030", None]], [2, 3, 4, 5],
                         [(0, "Well", "text", ""), (1, "TOC", "number", "%"),
                          (2, "S2", "number", "mg/g"), (3, "MD", "number", "m"), (4, "Facies", "text", "")], {})


def scatter(data, **kwargs):
    return PlotSpec(x_id=data.columns[1].id, y_id=data.columns[2].id, **kwargs)


def test_template_remapping_keeps_payload_reusable(data, tmp_path):
    from dataclasses import replace
    from uuid import uuid4
    from autopyro.ui.plot_templates import bind_plot_template, read_plot_template, write_plot_template

    x, y, depth = [column.id for column in data.columns[1:4]]
    spec = PlotSpec(kind='multi_depth', track_ids=[x, y], x_id=x, y_id=depth,
                    filter_rules=[{'column': x, 'operator': 'gt', 'value': 0}],
                    annotations=[{'text': 'Track label', 'coordinates': 'axes', 'x': .2, 'y': .8, 'track_id': y}])
    path = tmp_path / 'template.json'
    write_plot_template(path, spec, data.columns)
    payload = read_plot_template(path)
    original = deepcopy(payload)
    columns = [replace(column, id=str(uuid4())) for column in data.columns]
    first = bind_plot_template(payload, columns)
    assert first.filter_rules[0]['column'] == columns[1].id
    assert first.annotations[0]['track_id'] == columns[2].id
    assert payload == original
    other_columns = [replace(column, id=str(uuid4())) for column in data.columns]
    second = bind_plot_template(payload, other_columns)
    assert second.filter_rules[0]['column'] == other_columns[1].id
    assert second.annotations[0]['track_id'] == other_columns[2].id
    assert first.filter_rules[0]['column'] == columns[1].id
    assert first.annotations[0]['track_id'] == columns[2].id
    assert payload == original


def test_category_color_defaults_to_scale_and_keeps_saved_override(data):
    column_id = data.columns[-1].id
    data.rows[0].values[column_id] = "Jurassic"
    spec = scatter(data, color_by=column_id)
    figure, _ = render_plot(data, data.row_map(), spec)
    entry = spec.style_maps["color"][column_id][category_key("Jurassic")]
    assert entry["style"] == "#4C78B6"
    assert np.allclose(figure.axes[0].collections[0].get_facecolors()[0, :3], to_rgb("#4C78B6"))
    entry["style"] = "#112233"
    entry["automatic"] = False
    restored = PlotSpec.from_dict(spec.to_dict())
    figure, _ = render_plot(data, data.row_map(), restored)
    assert np.allclose(figure.axes[0].collections[0].get_facecolors()[0, :3], to_rgb("#112233"))


def test_legacy_category_default_updates_without_replacing_manual_styles(data):
    from matplotlib import rcParams
    from autopyro.visualization.plotting import ensure_style_maps

    column_id = data.columns[-1].id
    cycle = rcParams["axes.prop_cycle"].by_key()["color"]
    data.rows[0].values[column_id] = "Jurassic"
    data.rows[1].values[column_id] = "Triassic"
    styles = {
        category_key("Jurassic"): {"label": "Jurassic", "style": cycle[0], "hidden": True},
        category_key("Triassic"): {"label": "Triassic", "style": "#112233"},
    }
    spec = scatter(data, color_by=column_id, style_maps={"color": {column_id: styles}})
    ensure_style_maps(data, spec)
    assert styles[category_key("Jurassic")] == {
        "label": "Jurassic", "style": "#4C78B6", "automatic": True, "hidden": True,
    }
    assert styles[category_key("Triassic")]["style"] == "#112233"
    styles[category_key("Triassic")].update(style=cycle[1], automatic=False)
    ensure_style_maps(data, spec)
    assert styles[category_key("Triassic")]["style"] == cycle[1]


def test_spec_roundtrip_and_limits_are_not_scientific_classification(data):
    spec = scatter(data, x_axis=AxisSpec(minimum=.5, maximum=5, label="Organic carbon", minor_grid=True),
                   y_axis=AxisSpec(inverted=True), title="Pyrolysis")
    restored = PlotSpec.from_dict(spec.to_dict())
    assert restored == spec
    figure, result = render_plot(data, data.row_map(), restored)
    assert result["plotted"][spec.x_id] == 4
    assert figure.axes[0].get_xlim() == (.5, 5)
    assert figure.axes[0].yaxis.get_inverted()
    assert figure.axes[0].get_xlabel() == "Organic carbon"


@pytest.mark.parametrize("kind", ["scatter", "mpl:scatter", "multi_depth"])
def test_axis_label_padding_is_saved_and_applied_per_axis(data, kind):
    x, y = data.columns[1:3]
    spec = scatter(data, kind=kind, x_axis=AxisSpec(label_pad=7), y_axis=AxisSpec(label_pad=18),
                   track_ids=[x.id, data.columns[3].id], track_axes={x.id: AxisSpec(label_pad=25)})
    restored = PlotSpec.from_dict(spec.to_dict())
    figure, _ = render_plot(data, data.row_map(), restored)
    assert figure.axes[0].xaxis.labelpad == (25 if kind == "multi_depth" else 7)
    assert figure.axes[0].yaxis.labelpad == 18
    if kind == "multi_depth":
        assert figure.axes[1].xaxis.labelpad == 7
    restored.y_axis.label_pad = float("nan")
    with pytest.raises(ValueError, match="Отступ подписи оси Y"):
        render_plot(data, data.row_map(), restored)


@pytest.mark.parametrize("kind", ["scatter", "mpl:scatter"])
def test_numeric_aspect_takes_precedence_and_preserves_axis_limits(data, kind, tmp_path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    spec = scatter(data, kind=kind, square_plot_area=True, data_aspect=2,
                   width_mm=152.4, height_mm=101.6, dpi=100,
                   x_axis=AxisSpec(minimum=0, maximum=5), y_axis=AxisSpec(minimum=0, maximum=10))
    spec = PlotSpec.from_dict(spec.to_dict())
    figure, _ = render_plot(data, data.row_map(), spec)
    FigureCanvasAgg(figure).draw()
    ax = figure.axes[0]
    assert ax.bbox.height / ax.bbox.width == pytest.approx(4)
    assert ax.get_xlim() == (0, 5)
    assert ax.get_ylim() == (0, 10)
    path = tmp_path / "aspect.png"
    save_plot(path, data, data.row_map(), spec)
    with Image.open(path) as image:
        assert image.size == (600, 400)
    spec.data_aspect = 0
    with pytest.raises(ValueError, match="Соотношение масштабов"):
        render_plot(data, data.row_map(), spec)


def test_independent_style_channels_and_stable_palette(data):
    spec = scatter(data, color_by=data.columns[0].id, shape_by=data.columns[4].id, size_by=data.columns[3].id)
    _, all_points = render_plot(data, data.row_map(), spec)
    palette = deepcopy(spec.style_maps)
    assert len(all_points["artists"]) == 3  # shale, sand and missing
    _, filtered = render_plot(data, [data.rows[2].id, data.rows[3].id], spec)
    assert spec.style_maps == palette
    for artist in filtered["artists"]:
        assert set(artist._autopyro_row_ids) <= {data.rows[2].id, data.rows[3].id}
    key = category_key("B")
    spec.style_maps["color"][spec.color_by][key]["style"] = "#ff0000"
    _, red = render_plot(data, [data.rows[2].id], spec)
    assert red["artists"][0].get_facecolors()[0][:3] == pytest.approx([1, 0, 0])


@pytest.mark.parametrize("kind", ["scatter", "mpl:scatter", "hist"])
def test_legend_sorts_custom_category_labels_without_reordering_styles(data, kind, tmp_path):
    spec = scatter(data, kind=kind, color_by=data.columns[0].id, legend_order="ascending")
    ids = list(data.row_map())
    render_plot(data, ids, spec)
    styles = spec.style_maps["color"][spec.color_by]
    styles[category_key("A")]["label"] = "Zulu category"
    styles[category_key("B")]["label"] = "alpha category"
    spec = PlotSpec.from_dict(spec.to_dict())
    original = deepcopy(spec.style_maps)
    figure, result = render_plot(data, ids, spec)
    labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert labels[0].endswith("alpha category") and labels[1].endswith("Zulu category")
    path = tmp_path / "ordered.svg"
    save_plot(path, data, ids, spec)
    svg = path.read_text(encoding="utf-8")
    assert svg.index("alpha category") < svg.index("Zulu category")
    spec.legend_order = "descending"
    figure, descending = render_plot(data, ids, spec)
    assert [text.get_text() for text in figure.axes[0].get_legend().get_texts()] == labels[::-1]
    assert descending["plotted"] == result["plotted"]
    assert spec.style_maps == original


@pytest.mark.parametrize('kind', ['scatter', 'mpl:scatter', 'mpl3d:scatter'])
def test_shared_typography_reaches_legend_and_subtitle_in_export(data, kind, tmp_path):
    from xml.etree import ElementTree as ET
    from matplotlib import rc_context

    spec = scatter(data, kind=kind, color_by=data.columns[0].id,
                   title='Main title', subtitle='Context subtitle', title_size=19,
                   font_family='DejaVu Serif', legend_size=14)
    if kind == 'mpl3d:scatter':
        spec.z_id = data.columns[3].id
    spec = PlotSpec.from_dict(spec.to_dict())
    figure, _ = render_plot(data, data.row_map(), spec)
    subtitle = next(text for text in figure.texts if text.get_text() == spec.subtitle)
    assert subtitle.get_position() == (.5, .92)
    assert subtitle.get_fontsize() == 16
    for text in [figure._suptitle, subtitle, *figure.axes[0].get_legend().get_texts()]:
        assert text.get_fontfamily() == [spec.font_family]
    assert all(text.get_fontsize() == 14 for text in figure.axes[0].get_legend().get_texts())

    path = tmp_path / 'typography.svg'
    with rc_context({'svg.fonttype': 'none'}):
        save_plot(path, data, data.row_map(), spec)
    texts = list(ET.parse(path).getroot().iter('{http://www.w3.org/2000/svg}text'))
    exported_subtitle = next(text for text in texts if text.text == spec.subtitle)
    assert '16px' in exported_subtitle.get('style')
    labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    for label in ['Main title', 'Context subtitle', *labels]:
        exported = next(text for text in texts if text.text == label)
        assert spec.font_family in exported.get('style')


@pytest.mark.parametrize("kind", ["scatter", "mpl:scatter"])
def test_hidden_color_category_roundtrips_and_filters_render_and_export(data, tmp_path, kind):
    spec = scatter(data, kind=kind, color_by=data.columns[4].id)
    ids = list(data.row_map())
    render_plot(data, ids, spec)
    spec.style_maps["color"][spec.color_by][category_key("shale")]["hidden"] = True
    restored = PlotSpec.from_dict(spec.to_dict())
    figure, result = render_plot(data, ids, restored)
    assert [rid for artist in result["artists"] for rid in artist._autopyro_row_ids] == [ids[1], ids[3]]
    assert all("shale" not in label.get_text() for label in figure.axes[0].get_legend().get_texts())
    result = save_plot(tmp_path / "filtered.svg", data, ids, restored)
    assert sum(result["plotted"].values()) == 2


@pytest.mark.parametrize("color", ["#123456", (.1, .2, .3)])
def test_uniform_markers_keep_points_and_selection_ids_with_shared_style(data, color):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import to_rgba

    ids = [row.id for row in data.rows[:3]]
    spec = scatter(data, color=color, marker_size=9, alpha=.6)
    figure, result = render_plot(data, ids, spec)
    FigureCanvasAgg(figure).draw()
    artist = result["artists"][0]
    assert artist._autopyro_row_ids == ids
    np.testing.assert_allclose(artist.get_offsets(), [[1, 2], [2, 4], [3, 6]])
    np.testing.assert_allclose(artist.get_sizes(), [81])
    np.testing.assert_allclose(artist.get_facecolors(), [to_rgba(color, alpha=.6)])


def test_log_exclusions_and_no_silent_coercion(data):
    x = data.columns[1]
    data.rows[0].values[x.id] = 0
    data.rows[1].values[x.id] = -1
    spec = scatter(data, x_axis=AxisSpec(scale="log"))
    _, result = render_plot(data, data.row_map(), spec)
    assert result["plotted"][x.id] == 2
    assert result["excluded"][x.id] == {"nonpositive_log": 2}
    x.kind = "text"
    data.rows[2].values[x.id] = "<0.1"
    data.rows[3].values[x.id] = "1,2"
    spec.text_numbers = True
    _, result = render_plot(data, data.row_map(), spec)
    assert result["plotted"][x.id] == 1
    assert result["excluded"][x.id]["censored"] == 1


@pytest.mark.parametrize("kind", ["scatter", "mpl:scatter"])
def test_log_color_scale_keeps_nonpositive_points_in_missing_color(data, kind, tmp_path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LogNorm, to_rgba

    color = data.columns[3].id
    for row, value in zip(data.rows, (0, -1, 10, 1000), strict=True):
        row.values[color] = value
    spec = scatter(data, kind=kind, color_by=color, color_mode="numeric", color_scale="log",
                   missing_color="#ff00ff", alpha=1)
    spec = PlotSpec.from_dict(spec.to_dict())
    figure, result = render_plot(data, [row.id for row in data.rows[:3]], spec)
    FigureCanvasAgg(figure).draw()
    artist = result["artists"][0]
    assert isinstance(artist.norm, LogNorm)
    assert (artist.norm.vmin, artist.norm.vmax) == (10, 1000)
    assert figure.axes[1].get_yscale() == "log"
    assert len(artist._autopyro_row_ids) == 3
    assert artist.get_facecolors()[0] == pytest.approx(to_rgba("#ff00ff"))
    assert artist.get_facecolors()[1] == pytest.approx(to_rgba("#ff00ff"))
    path = tmp_path / "log-color.svg"
    save_plot(path, data, data.row_map(), spec)
    assert "#ff00ff" in path.read_text(encoding="utf-8")
    spec.color_min = 0
    with pytest.raises(ValueError, match="должны быть положительными"):
        render_plot(data, data.row_map(), spec)


def test_numeric_color_and_size_use_whole_project_ranges(data):
    spec = scatter(data, color_by=data.columns[3].id, color_mode="numeric", size_by=data.columns[3].id,
                   size_min=2, size_max=10)
    figure, result = render_plot(data, [data.rows[1].id], spec)
    artist = result["artists"][0]
    assert artist.norm.vmin == 1000
    assert artist.norm.vmax == 1030
    assert artist.get_sizes()[0] == pytest.approx((2 + (10 - 2) / 3) ** 2)
    assert len(figure.axes) == 2  # main axis plus colorbar


@pytest.mark.parametrize("kind", ["scatter", "mpl:scatter", "mpl3d:scatter"])
@pytest.mark.parametrize('location', ['top', 'right'])
def test_colorbar_position_and_visibility_roundtrip(data, kind, location, tmp_path):
    spec = scatter(data, kind=kind, color_by=data.columns[3].id, color_mode="numeric",
                   colorbar_location=location, font_family='DejaVu Serif', legend_size=14)
    if kind == 'mpl3d:scatter':
        spec.z_id = data.columns[3].id
    spec = PlotSpec.from_dict(spec.to_dict())
    automatic, _ = render_plot(data, data.row_map(), spec)
    colorbar = automatic.axes[-1]
    axis = colorbar.xaxis if location == 'top' else colorbar.yaxis
    assert axis.label.get_text() == 'MD [m]'
    assert axis.label.get_fontfamily() == [spec.font_family]
    ticks = axis.get_ticklabels()
    assert ticks and all(tick.get_fontfamily() == [spec.font_family] for tick in ticks)
    assert all(tick.get_fontsize() == 14 for tick in ticks)
    automatic_path = tmp_path / 'automatic.svg'
    save_plot(automatic_path, data, data.row_map(), spec)
    assert 'MD [m]' in automatic_path.read_text(encoding='utf-8')

    spec.colorbar_label = 'Depth scale'
    figure, result = render_plot(data, data.row_map(), spec)
    axes, bar = figure.axes
    if location == 'top':
        assert bar.get_xlabel() == "Depth scale"
        assert bar.get_position().y0 >= axes.get_position().y1
    else:
        assert bar.get_ylabel() == "Depth scale"
        assert bar.get_position().x0 >= axes.get_position().x1
    spec.colorbar_location = "none"
    hidden, hidden_result = render_plot(data, data.row_map(), spec)
    assert len(hidden.axes) == 1
    assert hidden_result["plotted"] == result["plotted"]
    path = tmp_path / "hidden.svg"
    save_plot(path, data, data.row_map(), spec)
    assert "Depth scale" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("kind", ["scatter", "mpl:scatter"])
def test_square_plot_area_keeps_axis_limits_and_export_dimensions(data, kind, tmp_path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    spec = scatter(data, kind=kind, square_plot_area=True, width_mm=152.4, height_mm=101.6, dpi=100,
                   x_axis=AxisSpec(minimum=0, maximum=5), y_axis=AxisSpec(minimum=0, maximum=10))
    spec = PlotSpec.from_dict(spec.to_dict())
    figure, _ = render_plot(data, data.row_map(), spec)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    ax = figure.axes[0]
    assert ax.bbox.width == pytest.approx(ax.bbox.height)
    assert ax.get_xlim() == (0, 5)
    assert ax.get_ylim() == (0, 10)
    path = tmp_path / "square-area.png"
    save_plot(path, data, data.row_map(), spec)
    with Image.open(path) as image:
        assert image.size == (600, 400)


@pytest.mark.parametrize("style_values", [
    [None, "<1100", 1000, 1030], [None, "ND", None, None], [990, 1000, 1010, 1020],
])
def test_numeric_style_arrays_preserve_missing_values_clipping_and_marker_groups(data, style_values):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize
    from autopyro.statistics import classify

    style = data.columns[3]
    for row, value in zip(data.rows, style_values):
        row.values[style.id] = value
    spec = scatter(data, color_by=style.id, color_mode="numeric", size_by=style.id,
                   shape_by=data.columns[0].id, marker_size=7, size_min=2, size_max=10,
                   size_value_min=1000, size_value_max=1010, color_min=1000, color_max=1010)
    figure, result = render_plot(data, data.row_map(), spec)
    FigureCanvasAgg(figure).draw()
    norm = Normalize(1000, 1010, clip=True)
    expected = {}
    for row, raw in zip(data.rows, style_values):
        _, value = classify(raw, numeric=True)
        expected[row.id] = (7 if value is None else 2 + float(norm(value)) * 8)**2
    seen = set()
    for artist in result["artists"]:
        ids = artist._autopyro_row_ids
        seen.update(ids)
        np.testing.assert_allclose(artist.get_sizes(), [expected[rid] for rid in ids])
        colors = np.ma.asarray(artist.get_array())
        for index, rid in enumerate(ids):
            row = data.row_map()[rid]
            _, number = classify(row.values[style.id], numeric=True)
            if number is None:
                assert np.ma.getmaskarray(colors)[index]
            else:
                assert colors[index] == number
    assert seen == set(data.row_map())
    assert result["plotted"][spec.x_id] == 4


@pytest.mark.parametrize("kind", ["scatter", "depth", "multi_depth", "hist", "box", "violin"])
def test_plot_kinds_render_and_do_not_mutate_cells(data, kind):
    values = [r.values.copy() for r in data.rows]
    spec = scatter(data, kind=kind, color_by=data.columns[0].id)
    if kind in ("depth", "multi_depth"):
        spec.y_id = data.columns[3].id
        spec.y_axis.inverted = True
    if kind == "multi_depth":
        spec.track_ids = [data.columns[1].id, data.columns[2].id]
        spec.track_axes[spec.track_ids[1]] = AxisSpec(scale="log")
    figure, result = render_plot(data, data.row_map(), spec)
    figure.savefig(__import__("io").BytesIO(), format="png")
    assert [r.values for r in data.rows] == values
    assert len(result["axes"]) == (2 if kind == "multi_depth" else 1)
    if kind == "multi_depth":
        assert all(ax.yaxis.get_inverted() for ax in result["axes"])
        assert result["axes"][1].get_xscale() == "log"


def test_histogram_groups_share_bin_edges(data):
    spec = scatter(data, kind="hist", bins=4, color_by=data.columns[0].id)
    figure, _ = render_plot(data, data.row_map(), spec)
    patches = figure.axes[0].patches
    assert [p.get_x() for p in patches[:4]] == [p.get_x() for p in patches[4:]]


@pytest.mark.parametrize("extension", ["png", "tif", "svg", "pdf"])
def test_exports_have_physical_size_and_valid_content(data, tmp_path, extension):
    spec = scatter(data, width_mm=101.6, height_mm=76.2, dpi=100, title="Pyrolysis")
    path = tmp_path / f"plot.{extension}"
    save_plot(path, data, data.row_map(), spec)
    assert path.stat().st_size > 100
    if extension in ("png", "tif"):
        with Image.open(path) as image:
            assert image.size == (400, 300)
    if extension == "svg":
        assert "<svg" in path.read_text(encoding="utf-8")
    if extension == "pdf":
        assert path.read_bytes().startswith(b"%PDF")


def test_empty_scope_stays_empty_and_invalid_configuration_is_rejected(data):
    spec = scatter(data)
    _, result = render_plot(data, [], spec)
    assert result["plotted"][spec.x_id] == 0
    assert result["artists"] == []
    spec.x_axis.minimum, spec.x_axis.maximum = 10, 1
    with pytest.raises(ValueError, match="Минимум"):
        render_plot(data, data.row_map(), spec)


def test_stale_interpolation_is_excluded(data):
    spec = scatter(data, include_interpolated=True)
    data.columns[1].metadata = {"method": "linear", "stale": True}
    _, result = render_plot(data, data.row_map(), spec)
    assert result["excluded"][spec.x_id] == {"excluded_interpolation": 4}


def test_numeric_scale_is_not_limited_to_100_categories():
    data = build_project(["X", "Y"], [[str(i), str(i * 2)] for i in range(150)], list(range(150)),
                         [(0, "X", "number", ""), (1, "Y", "number", "")], {})
    x, y = [c.id for c in data.columns]
    spec = PlotSpec(x_id=x, y_id=y, color_by=x, color_mode="numeric")
    _, result = render_plot(data, data.row_map(), spec)
    assert result["plotted"][x] == 150
    assert not spec.style_maps.get("color")



def test_line_groups_can_break_at_excluded_probes_independently_of_color(data):
    well, x, y, depth, _ = [column.id for column in data.columns]
    for row in data.rows:
        row.values[well] = 'A'
    data.rows[1].values[y] = None
    spec = PlotSpec(kind='depth', x_id=y, y_id=depth, color_by=x, color_mode='numeric',
                    line_group_by=well, connect=True, connect_across_gaps=False)
    figure, result = render_plot(data, data.row_map(), spec)
    assert [list(line.get_ydata()) for line in figure.axes[0].lines] == [[1000], [1020, 1030]]
    assert result['excluded_by_column'][y] == {y: {'missing': 1}}



def test_positioned_label_follows_track_uuid_and_checks_log_coordinates(data):
    x, y, depth = [column.id for column in data.columns[1:4]]
    spec = PlotSpec(kind='multi_depth', track_ids=[y, x], y_id=depth, track_axes={x: AxisSpec(scale='log')},
                    annotations=[{'text': 'Note', 'coordinates': 'data', 'x': 2, 'y': 1010, 'track_id': x}])
    spec = PlotSpec.from_dict(spec.to_dict())
    figure, result = render_plot(data, data.row_map(), spec)
    assert not figure.axes[0].texts
    label = figure.axes[1].texts[0]
    assert label.get_position() == (2, 1010)
    spec.annotations[0]['x'] = 0
    figure, result = render_plot(data, data.row_map(), spec)
    assert not figure.axes[1].texts
    assert 'несовместимы' in result['notices'][0]



def test_graphical_annotations_keep_axes_limits_and_validate_shape_endpoints(data):
    from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle

    spec = scatter(data)
    original, _ = render_plot(data, data.row_map(), spec)
    spec.annotations = [
        {'kind': kind, 'coordinates': 'data', 'x': 100, 'y': 100, 'x2': 200, 'y2': 200}
        for kind in ('arrow', 'rectangle', 'ellipse')
    ]
    restored = PlotSpec.from_dict(spec.to_dict())
    figure, _ = render_plot(data, data.row_map(), restored)
    ax = figure.axes[0]
    assert ax.get_xlim() == original.axes[0].get_xlim()
    assert ax.get_ylim() == original.axes[0].get_ylim()
    shapes = [artist for artist in ax.patches if artist.get_gid()]
    assert [type(artist) for artist in shapes] == [FancyArrowPatch, Rectangle, Ellipse]
    assert all(artist.get_clip_path() is not None or artist.get_clip_box() is not None for artist in shapes)

    restored.annotations[1]['x2'] = 100
    with pytest.raises(ValueError, match='ненулевой размер'):
        PlotSpec.from_dict(restored.to_dict())



@pytest.mark.parametrize('title', [
    'HI–OI', 'HI–Tmax', 'S2–TOC', 'S1–TOC', 'S1–S2', 'PI–Tmax', 'HI–TOC', 'S1+S2–TOC',
    'TOC–MD', 'S1–MD', 'S2–MD', 'S3–MD', 'Tmax–MD', 'HI–MD', 'OI–MD', 'PI–MD',
])
def test_scientific_passport_survives_preparation_and_validates_axis_units(title):
    from autopyro.visualization.plot_data import PlotDataCache, capture_input, input_token
    from autopyro.visualization.plot_passports import PASSPORTS, axis_requirements, template_kind

    requirements = axis_requirements(title)
    project = build_project(['Horizontal', 'Vertical'], [[1, 2]], [1],
                            [(i, name, 'number', unit)
                             for i, (name, (_, unit)) in enumerate(zip(('Horizontal', 'Vertical'), requirements))], {})
    for column, (role, _) in zip(project.columns, requirements):
        column.metadata['property_role'] = role
    x, y = project.columns
    spec = PlotSpec(kind=template_kind(title), x_id=x.id, y_id=y.id)
    ids = list(project.row_map())
    original = capture_input(project, ids, [spec], input_token(project, 0, [spec]))
    spec.passport_id = PASSPORTS[title][0]
    spec = PlotSpec.from_dict(spec.to_dict())
    token = input_token(project, 0, [spec])
    source = capture_input(project, ids, [spec], token, original)
    prepared = PlotDataCache().prepare(source, ids, spec)
    _, result = render_plot(project, ids, spec, prepared=prepared)
    assert result['passport']['id'] == spec.passport_id
    assert result['passport']['classification_layer'] == 'none'
    assert result['passport']['axes']['y']['unit'] == requirements[1][1]
    if spec.kind == 'depth':
        assert result['passport']['depth_reference'] == 'MD'
        assert result['passport']['axes']['y']['unit'] == 'm'
        y.metadata['property_role'] = 'TVD'
        with pytest.raises(ValueError, match='MD'):
            render_plot(project, ids, spec)
        y.metadata['property_role'] = 'MD'
    y.unit = 'incompatible'
    with pytest.raises(ValueError, match='требуется'):
        render_plot(project, ids, spec)



@pytest.mark.parametrize('role', ['TOC', 'S1', 'S2', 'S3', 'Tmax', 'HI', 'OI', 'PI'])
def test_interval_md_passport_checks_both_depth_bindings(role):
    from autopyro.visualization.plot_passports import INTERVAL_PASSPORTS, axis_requirements

    title = f'{role}–MD interval'
    _, x_role, _, _ = INTERVAL_PASSPORTS[title]
    x_unit = axis_requirements(title)[0][1]
    project = build_project([x_role, 'MD top', 'MD bottom'], [[2, 1000, 1010], [3, 1020, 1015]],
                            [1, 2], [(0, x_role, 'number', x_unit), (1, 'MD top', 'number', 'm'),
                                     (2, 'MD bottom', 'number', 'm')], {})
    x, top, bottom = project.columns
    spec = PlotSpec(kind='depth', x_id=x.id, y_id=top.id, depth_bottom_id=bottom.id,
                    passport_id=INTERVAL_PASSPORTS[title][0])
    spec = PlotSpec.from_dict(spec.to_dict())
    ids = list(project.row_map())
    _, result = render_plot(project, ids, spec)
    assert result['passport']['axes']['y']['column_id'] == top.id
    assert result['passport']['axes']['bottom']['column_id'] == bottom.id
    assert result['excluded'][x.id] == {'invalid_depth_interval': 1}
    assert result['artists'][0].get_segments()[0].tolist() == [[2, 1000], [2, 1010]]
    bottom.unit = 'ft'
    with pytest.raises(ValueError, match='MD_BOTTOM'):
        render_plot(project, ids, spec)
    bottom.unit = 'm'
    spec.y_id, spec.depth_bottom_id = bottom.id, top.id
    with pytest.raises(ValueError, match='MD_TOP'):
        render_plot(project, ids, spec)



@pytest.mark.parametrize('kind', ['scatter', 'mpl:scatter'])
def test_numeric_color_classes_are_discrete_and_keep_missing_points(data, kind, tmp_path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import BoundaryNorm, to_rgba

    color = data.columns[3].id
    for row, value in zip(data.rows, (-1, 1.5, 100, None), strict=True):
        row.values[color] = value
    spec = scatter(data, kind=kind, color_by=color, color_mode='numeric', color_scale='classes',
                   color_class_edges='0; 1,5; 10', color_min=5, color_max=1,
                   missing_color='#ff00ff', alpha=1)
    spec = PlotSpec.from_dict(spec.to_dict())
    figure, result = render_plot(data, list(data.row_map()), spec)
    FigureCanvasAgg(figure).draw()
    artist = result['artists'][0]
    assert isinstance(artist.norm, BoundaryNorm)
    assert list(artist.norm.boundaries) == [0, 1.5, 10]
    assert list(figure.axes[1].get_yticks()) == [0, 1.5, 10]
    colors = artist.get_facecolors()
    assert colors[0] == pytest.approx(artist.cmap(0))
    assert colors[0] != pytest.approx(colors[1])
    assert colors[1] == pytest.approx(colors[2])
    assert colors[3] == pytest.approx(to_rgba('#ff00ff'))
    path = tmp_path / 'classes.svg'
    save_plot(path, data, data.row_map(), spec)
    assert '#ff00ff' in path.read_text(encoding='utf-8')
    spec.color_class_edges = '0; 1,5; 1,5'
    with pytest.raises(ValueError, match='возрастающих границ'):
        render_plot(data, data.row_map(), spec)
