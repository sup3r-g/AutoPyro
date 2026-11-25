import json
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
import shapely
from shapely.geometry import Point, Polygon, mapping
from pyogrio import raw

from autopyro.io import build_project, file_source
from autopyro.mapping import (MapSpec,build_map,export_map,list_layers,read_layer,render_map,transform_geometry)


@pytest.fixture
def data():
    rows=[["A",0,0,"1","F1","Upper","2020-01-01",100],
          ["B",10,0,"3","F2","Lower","2021-01-01",200],
          ["C",0,10,"ND","F1","Upper","2022-01-01",300],
          ["D",10,10,"7","F2","Lower","bad date",400],
          ["E",5,5,"<2","F1","Upper","2020-06-01",150]]
    names=["Well","X","Y","TOC","Family","Formation","Date","MD"]
    project=build_project(names,rows,list(range(2,7)),[(i,name,"number" if name in ("X","Y","MD") else "text","m" if name in ("X","Y","MD") else "") for i,name in enumerate(names)],{})
    ids={c.name:c.id for c in project.columns}
    spec=MapSpec(x_id=ids["X"],y_id=ids["Y"],source_crs="EPSG:32631",target_crs="EPSG:32631",text_numbers=True)
    return project,ids,spec


def layer(geometries,name="sites"):
    return {"id":name,"name":name,"crs":"EPSG:32631","visible":True,"color":"#333333","fill":"#aaccee","alpha":.2,"line_width":1,
        "features":[{"type":"Feature","id":str(i),"geometry":mapping(g),"properties":{"well":chr(65+i)}} for i,g in enumerate(geometries)]}


def test_projection_axis_order_and_snapshot(data):
    point=transform_geometry(Point(3,0),"EPSG:4326","EPSG:32631")
    assert point.x==pytest.approx(500000) and point.y==pytest.approx(0)
    project,ids,spec=data
    before=asdict(deepcopy(project))
    report=build_map(project,[r.id for r in project.rows],spec,[])
    assert asdict(project)==before
    project.rows[0].values[ids["TOC"]]="100"
    assert report["points"][0]["values"][ids["TOC"]]=="1"
    assert report["crs_operation"]["network"] is False
    assert build_map(project,[],spec,[])["points"]==[]


@pytest.mark.parametrize("suffix,driver",[(".gpkg","GPKG"),(".geojson","GeoJSON"),(".fgb","FlatGeobuf"),(".shp","ESRI Shapefile")])
def test_real_vector_import_readonly(tmp_path,suffix,driver):
    path=tmp_path/f"wells{suffix}"
    raw.write(path,np.array([shapely.to_wkb(Point(3,60))],dtype=object),[np.array(["Well A"],dtype=object)],["well"],
              crs="EPSG:4326",driver=driver,layer="wells",geometry_type="Point")
    before=file_source(path)["sha256"]
    assert list_layers(path)
    result=read_layer(path,"wells")
    assert result["features"][0]["properties"]["well"]=="Well A"
    assert file_source(path)["sha256"]==before
    assert result["source"]["sha256"]==before


def test_exact_layer_join_duplicate_keys_excluded(data):
    project,ids,spec=data
    wells=layer([Point(1,2),Point(3,4)],"wells")
    spec.location_mode="layer"
    spec.point_layer="wells"
    spec.sample_key=ids["Well"]
    spec.layer_key="well"
    report=build_map(project,[r.id for r in project.rows],spec,[wells])
    assert len(report["points"])==2
    wells["features"][1]["properties"]["well"]="A"
    report=build_map(project,[r.id for r in project.rows],spec,[wells])
    assert report["points"]==[]
    assert report["diagnostics"]["coordinates"]["unmatched_or_ambiguous_location"]==5
    assert report['diagnostics']['excluded_rows'] == {
        'location_key_duplicate': [project.rows[0].id],
        'location_key_unmatched': [row.id for row in project.rows[1:]],
    }


def test_filters_and_coordinate_diagnostics(data, tmp_path):
    project,ids,spec=data
    spec.depth_id=ids["MD"]
    spec.depth_min=100
    spec.depth_max=200
    spec.strat_id=ids["Formation"]
    spec.strat_value="Upper"
    spec.date_id=ids["Date"]
    spec.date_from="2020-01-01"
    spec.date_to="2020-12-31"
    report=build_map(project,[r.id for r in project.rows],spec,[])
    assert report["row_ids"]==[project.rows[0].id,project.rows[4].id]
    assert sum(report["diagnostics"]["filtered"].values())==3
    spec.strat_id=""
    spec.depth_min=spec.depth_max=None
    spec.date_to="2025-01-01"
    report=build_map(project,[r.id for r in project.rows],spec,[])
    assert report["diagnostics"]["filtered"]["invalid_date"]==1
    assert report['diagnostics']['excluded_rows']['invalid_date'] == [project.rows[3].id]
    for suffix in ('json', 'geojson'):
        path = tmp_path / f'map.{suffix}'
        export_map(path, report)
        exported = path.read_text(encoding='utf-8')
        assert project.rows[3].id not in exported
        assert 'invalid_date' in exported
    spec.date_id=spec.date_from=spec.date_to=""
    spec.source_crs="EPSG:4326"
    project.columns[1].unit=project.columns[2].unit=""
    project.rows[0].values[ids["Y"]]=100
    report=build_map(project,[r.id for r in project.rows],spec,[])
    assert report["diagnostics"]["coordinates"]["invalid_longitude_latitude"]==1
    assert report['diagnostics']['excluded_rows']['invalid_longitude_latitude'] == [project.rows[0].id]


def test_site_aggregation_counts_qualifiers_holes_and_boundaries(data):
    project,ids,spec=data
    sites=layer([Polygon([(-1,-1),(11,-1),(11,11),(-1,11)],holes=[[(4,4),(6,4),(6,6),(4,6)]] )])
    spec.site_layer="sites"
    spec.site_value=ids["TOC"]
    spec.site_family=ids["Family"]
    report=build_map(project,[r.id for r in project.rows],spec,[sites])
    site=report["sites"][0]
    assert site["n_samples"]==4 and site["n_valid"]==3
    assert site["value"]==pytest.approx(11/3)
    assert site["excluded"]=={"non_detect":1}
    assert site["family"] is None and site["family_status"]=="tie"
    assert report["diagnostics"]["site_assignment"]["outside_sites"]==1
    sites=layer([Polygon([(-1,-1),(5,-1),(5,11),(-1,11)]),Polygon([(5,-1),(11,-1),(11,11),(5,11)])])
    report=build_map(project,[r.id for r in project.rows],spec,[sites])
    assert report["diagnostics"]["site_assignment"]["ambiguous_site"]==1
    assert sum(s["n_samples"] for s in report["sites"])==4
    assert sum(s["n_boundary"] for s in report["sites"])==0


def test_stable_family_palette_and_exports(data,tmp_path):
    project,ids,spec=data
    spec.color_by=ids["Family"]
    spec.size_by=ids["TOC"]
    report=build_map(project,[r.id for r in project.rows],spec,[])
    styles=deepcopy(spec.style_maps)
    subset=build_map(project,[project.rows[1].id],spec,[])
    assert styles==spec.style_maps
    subset_figure, subset_view = render_map(subset)
    assert [label.get_text() for label in subset_view['axes'].get_legend().get_texts()] == ['F2']
    subset_figure.clear()
    figure,result=render_map(report)
    assert len(result["artist"]._autopyro_members)==5
    for suffix in ("png","svg","json","geojson"):
        export_map(tmp_path/f"map.{suffix}",report)
        assert (tmp_path/f"map.{suffix}").stat().st_size>100
    assert json.loads((tmp_path/"map.geojson").read_text(encoding="utf-8"))["type"]=="FeatureCollection"
    assert "EPSG" in figure.texts[0].get_text()


def test_category_limit_uses_only_mapped_samples():
    names = ['X', 'Y', 'Family']
    project = build_project(names, [[i, i, f'F{i}'] for i in range(101)], list(range(101)),
                            [(0, 'X', 'number', 'm'), (1, 'Y', 'number', 'm'), (2, 'Family', 'text', '')], {})
    x, y, family = [column.id for column in project.columns]
    spec = MapSpec(x_id=x, y_id=y, source_crs='EPSG:32631', color_by=family, size_by=family, size_mode='category')
    ids = [row.id for row in project.rows]
    with pytest.raises(ValueError, match='100 значениями'):
        build_map(project, ids, spec, [])
    report = build_map(project, ids[:2], spec, [])
    assert {channel: len(keys) for channel, keys in report['point_categories'].items()} == {'color': 2, 'size': 2}
    figure, view = render_map(report)
    assert [label.get_text() for label in view['axes'].get_legend().get_texts()] == ['F0', 'F1']
    figure.clear()


def test_coordinate_unit_contradiction_is_not_silently_reprojected(data):
    project,ids,spec=data
    spec.source_crs="EPSG:4326" # source columns explicitly declare metres
    with pytest.raises(ValueError,match="единица|Единица"):
        build_map(project,[r.id for r in project.rows],spec,[])


def test_interval_depth_filter_uses_overlap_not_midpoint(data):
    from autopyro.core import Column
    project,ids,spec=data
    bottom=Column("Bottom","number",unit="m")
    project.columns.append(bottom)
    for row,value in zip(project.rows,[110,210,310,410,160]):
        row.values[bottom.id]=value
    spec.depth_id=ids["MD"]
    spec.depth_bottom_id=bottom.id
    spec.depth_min,spec.depth_max=105,155
    report=build_map(project,[r.id for r in project.rows],spec,[])
    assert report["row_ids"]==[project.rows[0].id,project.rows[4].id]
    spec.depth_min=spec.depth_max=110
    assert build_map(project,[r.id for r in project.rows],spec,[])["points"]==[]
    spec.depth_min=spec.depth_max=100
    assert build_map(project,[r.id for r in project.rows],spec,[])["row_ids"]==[project.rows[0].id]
    project.rows[0].values[bottom.id]="ND"
    assert build_map(project,[project.rows[0].id],spec,[])["diagnostics"]["filtered"]["invalid_depth_interval"]==1


def test_failed_map_export_keeps_previous_file(data,tmp_path,monkeypatch):
    import autopyro.mapping as module
    project,ids,spec=data
    report=build_map(project,[],spec,[])
    path=tmp_path/"map.json"
    path.write_text("old",encoding="utf-8")
    monkeypatch.setattr(module.os,"replace",lambda *args: (_ for _ in ()).throw(OSError("locked")))
    with pytest.raises(OSError):
        export_map(path,report)
    assert path.read_text(encoding="utf-8")=="old"
    assert list(tmp_path.iterdir())==[path]


def test_auto_crs_uses_metadata_without_changing_saved_mode(data):
    from autopyro.mapping import crs_label
    project, ids, spec = data
    spec.source_crs = spec.target_crs = 'auto'
    project.columns[1].metadata['crs'] = 'EPSG:32631'
    report = build_map(project, [r.id for r in project.rows], spec, [])
    assert report['spec']['source_crs'] == report['spec']['target_crs'] == 'EPSG:32631'
    assert spec.source_crs == spec.target_crs == 'auto'
    assert 'WGS 84 / UTM zone 31N' in crs_label(report['spec']['target_crs'])
    project.columns[2].metadata['crs'] = 'EPSG:3857'
    with pytest.raises(ValueError, match='противоречат'):
        build_map(project, [], spec, [])


def test_auto_crs_does_not_guess_from_coordinate_magnitude(data):
    project, ids, spec = data
    spec.source_crs = 'auto'
    with pytest.raises(ValueError, match='CRS координат неизвестна'):
        build_map(project, [], spec, [])
    for column, name in zip(project.columns[1:3], ('Longitude', 'Latitude')):
        column.name, column.unit = name, 'deg'
    spec.target_crs = 'auto'
    report = build_map(project, [r.id for r in project.rows], spec, [])
    assert report['spec']['source_crs'] == report['spec']['target_crs'] == 'EPSG:4326'


def test_auto_crs_tracks_point_layer(data):
    from pyproj import CRS
    project, ids, spec = data
    spec.location_mode, spec.point_layer = 'layer', 'wells'
    spec.sample_key, spec.layer_key = ids['Well'], 'well'
    spec.source_crs = spec.target_crs = 'auto'
    wells = layer([Point(1, 2)], 'wells')
    report = build_map(project, [r.id for r in project.rows], spec, [wells])
    assert CRS(report['crs_operation']['source_crs']) == CRS(wells['crs'])
    assert report['spec']['target_crs'] == wells['crs']


def test_import_keeps_null_geometry_attributes_and_fids(tmp_path):
    import geopandas as gpd
    path = tmp_path / 'attributes.gpkg'
    gpd.GeoDataFrame({'name': ['Скважина', None], 'value': [2, None]},
                     geometry=[Point(3, 60), None], crs='EPSG:4326').to_file(path, layer='wells')
    imported = read_layer(path, 'wells')
    assert len(imported['features']) == 1
    assert len(imported['attributes']) == 2
    assert imported['diagnostics']['null_geometry'] == 1
    assert imported['attributes'][0]['id'] == imported['features'][0]['id']
    assert imported['attributes'][1]['properties'] == {'name': None, 'value': None}
    json.dumps(imported, allow_nan=False)


def test_missing_crs_requires_explicit_declaration(tmp_path):
    import geopandas as gpd
    path = tmp_path / 'unknown.shp'
    gpd.GeoDataFrame({'well': ['A']}, geometry=[Point(3, 60)]).to_file(path)
    with pytest.raises(ValueError, match='CRS слоя неизвестна'):
        read_layer(path)
    imported = read_layer(path, declared_crs='EPSG:4326')
    assert 'WGS 84' in imported['crs']
    with pytest.raises(ValueError, match='противоречит'):
        known = tmp_path / 'known.gpkg'
        gpd.GeoDataFrame(geometry=[Point(3, 60)], crs='EPSG:4326').to_file(known)
        read_layer(known, declared_crs='EPSG:3857')


def test_empty_polygon_layer_and_empty_sample_scope(data):
    project, ids, spec = data
    spec.site_layer = 'sites'
    assert build_map(project, [], spec, [layer([])])['sites'] == []
    report = build_map(project, [r.id for r in project.rows], spec, [layer([])])
    assert report['diagnostics']['site_assignment'] == {'outside_sites': 5}


def test_missing_xy_placeholder_keeps_attributes_without_distorting_map(tmp_path, data):
    import geopandas as gpd
    from autopyro.mapping import layer_frame
    path = tmp_path / 'wells.shp'
    missing = -np.finfo(np.float64).max
    gpd.GeoDataFrame({'well': ['A', 'B']}, geometry=[Point(1, 2), Point(missing, missing)],
                     crs='EPSG:32631').to_file(path)
    imported = read_layer(path)
    assert len(imported['attributes']) == 2
    assert len(imported['features']) == 1
    assert imported['diagnostics'] == {'missing_coordinates': 1}
    assert layer_frame(imported).total_bounds.tolist() == [1, 2, 1, 2]
    project, ids, spec = data
    spec.location_mode, spec.point_layer = 'layer', imported['id']
    spec.sample_key, spec.layer_key = ids['Well'], 'well'
    report = build_map(project, [r.id for r in project.rows], spec, [imported])
    assert len(report['points']) == 1
    assert report['diagnostics']['coordinates']['unmatched_or_ambiguous_location'] == 4
    assert report['diagnostics']['layers'][imported['id']] == {'missing_coordinates': 1}
    figure, artists = render_map(report)
    assert max(abs(value) for value in artists['axes'].get_xlim()) < 100
    figure.clear()


@pytest.mark.parametrize('mode', ['columns', 'layers'])
def test_spatial_layers_without_any_sample_table(mode, tmp_path):
    from autopyro.core import Project
    from autopyro.mapping import crs
    regions = layer([Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])])
    spec = MapSpec(location_mode=mode, color_by='deleted-column', strat_id='deleted-column')
    report = build_map(Project(), [], spec, [regions])
    assert report['spec']['location_mode'] == 'layers'
    assert crs(report['spec']['target_crs']) == crs(regions['crs'])
    assert report['points'] == report['sites'] == report['row_ids'] == []
    assert report['crs_operation']['source_crs'] is None
    figure, result = render_map(report)
    assert len(result['layer_features']) == 1
    export_map(tmp_path / f'{mode}.png', report)
    figure.clear()


def test_short_map_export_reserves_space_for_axis_text_and_footer(data, tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    from autopyro.spatial.map_tools import renderer_for

    project, _, spec = data
    spec.height_mm = 45
    spec.width_mm = 120
    report = build_map(project, [row.id for row in project.rows], spec, [])
    savefig = Figure.savefig

    def check_layout(figure, *args, **kwargs):
        renderer = renderer_for(figure)
        axes = figure.axes[0]
        bounds = axes.get_tightbbox(renderer, bbox_extra_artists=[])
        footer = figure.texts[0].get_window_extent(renderer)
        assert figure.bbox.ymin <= footer.ymin < footer.ymax < bounds.ymin
        assert bounds.ymax <= figure.bbox.ymax
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', check_layout)
    path = tmp_path / 'short-map.svg'
    export_map(path, report)
    assert path.is_file()


def test_layer_categories_render_distinct_colors_and_legend(tmp_path):
    from autopyro.core import Project
    from autopyro.mapping import layer_categories
    regions = layer([Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                     Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]), Point(5, 5)])
    for feature, value in zip(regions['features'], ['Шельф', 'Равнина', None]):
        feature['properties']['Paleogeogr'] = value
    regions['color_by'] = 'Paleogeogr'
    categories = layer_categories(regions)
    report = build_map(Project(), [], MapSpec(location_mode='layers'), [regions])
    figure, result = render_map(report)
    texts = [item.get_text().replace('\n', ' ') for item in result['axes'].get_legend().get_texts()]
    assert len(texts) == 3
    assert any('Шельф' in text for text in texts)
    assert any('Нет значения' in text for text in texts)
    patches = result['axes'].patches
    assert patches[0].get_facecolor() != patches[1].get_facecolor()
    regions['features'].reverse()
    assert layer_categories(regions) == categories
    export_map(tmp_path / 'categories.svg', report)
    figure.clear()
    report['spec']['legend'] = False
    figure, result = render_map(report)
    assert result['axes'].get_legend() is None
    figure.clear()


def test_polygon_layer_batches_paths_without_losing_holes_or_feature_styles(tmp_path):
    from matplotlib.collections import PathCollection
    from matplotlib.path import Path
    from shapely.geometry import MultiPolygon, box
    from autopyro.core import Project

    with_hole = Polygon(box(0, 0, 10, 10).exterior.coords, [box(2, 2, 4, 4).exterior.coords])
    regions = layer([with_hole, MultiPolygon([box(20, 0, 30, 10), box(40, 0, 50, 10)])])
    regions['color_by'] = 'well'
    report = build_map(Project(), [], MapSpec(location_mode='layers'), [regions])
    figure, result = render_map(report)
    ax = result['axes']
    batch, = [artist for artist in ax.collections if isinstance(artist, PathCollection) and len(artist.get_paths()) == 3]
    first, second, third = batch.get_paths()
    assert np.count_nonzero(first.codes == Path.MOVETO) == 2
    assert np.count_nonzero(first.codes == Path.CLOSEPOLY) == 2
    assert second.vertices[:, 0].min() == 20
    assert third.vertices[:, 0].max() == 50
    colors = batch.get_facecolors()
    assert not np.allclose(colors[0], colors[1])
    assert np.allclose(colors[1], colors[2])
    assert np.allclose(colors[:, 3], regions['alpha'])
    assert ax.get_xlim()[0] < 0 < 50 < ax.get_xlim()[1]
    assert [fid for _, fid, _ in result['layer_features']] == ['0', '1']
    export_map(tmp_path / 'polygons.svg', report)
    figure.clear()
