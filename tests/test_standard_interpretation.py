"""Strategy tests use synthetic curves with analytically known answers."""
from copy import deepcopy

import pytest

from autopyro.interpretation import GeometricEngine, InterpretationInput, validate_scheme
from autopyro.standard_interpretation import standard_schemes


def curves(strategy="nearest_curve"):
    s = standard_schemes()["HI–Tmax"]
    s["strategy"] = strategy
    s.pop("max_distance")
    s["x"] = dict(property="Tmax", unit="°C", scale=1)
    s["y"] = dict(property="HI", unit="mg HC/g TOC", scale=1)
    s["lines"] = [dict(id=str(i), label=str(i), geometry=dict(type="LineString", coordinates=c))
                  for i, c in enumerate([[(0, 100), (10, 0)], [(0, 200), (10, 0)]])]
    s["intermediate_curves"] = 1
    return s


def evaluate(s, x, y, **kwargs):
    return GeometricEngine().evaluate([InterpretationInput("row", x, y, **kwargs)], s)[0]


def test_catalog_is_valid_and_independent():
    for s in standard_schemes().values():
        validate_scheme(s)
    a = standard_schemes()
    a["HI–Tmax"]["lines"].clear()
    assert standard_schemes()["HI–Tmax"]["lines"]


def test_svg_catalog_groups_and_source_calibration():
    from collections import Counter

    from shapely.geometry import shape
    from autopyro.schemes.interpretation import inverse

    imported = [scheme for scheme in standard_schemes().values() if 'group' in scheme]
    assert Counter(scheme['group'] for scheme in imported) == {
        'HI–OI': 6, 'HI–Tmax': 13, 'HI–TOC': 1, 'PI–Tmax': 3,
        'S1+S2–TOC': 3, 'S2–TOC': 4, 'TOCo': 1,
    }
    assert len({scheme['id'] for scheme in imported}) == len(imported)
    assert all(scheme['scientific_status'] == 'user_supplied' for scheme in imported)

    s2 = next(s for s in imported if s['name'].startswith('S2–TOC · Edress'))
    assert s2['x']['transform'] == s2['y']['transform'] == 'log10'
    result = evaluate(s2, 1.5, 7)
    region = next(entry for entry in s2['regions'] if entry['id'] == result.regions[0])
    assert region['label'] == 'TOC: Good · S2: Good'

    hi = next(s for s in imported if s['name'].startswith('HI–Tmax · Akram'))
    guide = next(line for line in hi['lines'] if line.get('reference_only'))
    p = shape(guide['geometry']).interpolate(.5, normalized=True)
    result = evaluate(hi, inverse(p.x, hi['x']), inverse(p.y, hi['y']))
    assert result.line_hits
    assert guide['id'] not in result.line_hits

    jarvie = next(s for s in imported if s['group'] == 'TOCo')
    assert jarvie['y']['property'] == 'Kerogen Yield'
    assert len(jarvie['lines']) == 23


def test_nearest_curve_and_equal_distance():
    s = curves()
    assert evaluate(s, 0, 100).line_hits == ["0"]
    assert evaluate(s, 10, 0).status == "ambiguous"
    assert evaluate(s, None, 100).status == "insufficient_data"
    assert evaluate(s, float("nan"), 100).status == "not_applicable"
    assert evaluate(s, 1, 100, x_bounds=(0, 2)).status == "classified"


def test_intermediate_curve_initial_hi_and_shared_endpoint():
    s = curves("initial_hi")
    r = evaluate(s, 5, 75)
    assert r.value == pytest.approx(150)
    assert r.status == "classified"
    r = evaluate(s, 10, 0)
    assert r.value is None
    assert r.status == "ambiguous"
    # The leftmost point is independent of the digitization direction.
    for entry in s["lines"]:
        entry["geometry"]["coordinates"].reverse()
    assert evaluate(s, 5, 75).value == pytest.approx(150)


def test_curve_validation():
    for key, value in [("strategy", "typo"), ("distance_metric", None),
                       ("intermediate_curves", -1)]:
        s = curves("initial_hi")
        s[key] = value
        with pytest.raises(ValueError):
            validate_scheme(s)


def test_polygon_catalog_returns_both_independent_properties():
    r = evaluate(standard_schemes()["S2–TOC"], 1.5, 12)
    assert r.status == "classified"
    assert r.regions == ["2:3"]
    assert evaluate(standard_schemes()["S2–TOC"], .5, 12).status == "boundary"
    assert evaluate(standard_schemes()["PI–Tmax"], 450, 1.1).status == "not_applicable"


def test_normalized_alignment_ignores_direction_and_vertex_density():
    s = curves("initial_hi")
    s["curve_alignment"] = "normalized_arclength"
    first = evaluate(s, 5, 75)
    s["lines"][1]["geometry"]["coordinates"] = [[10, 0], [7.5, 50], [5, 100], [0, 200]]
    second = evaluate(s, 5, 75)
    assert first.value == second.value == pytest.approx(150)


def test_no_hio_extrapolation():
    s = curves("initial_hi")
    s["no_extrapolation"] = True
    assert evaluate(s, 5, 150).status == "not_applicable"
    assert evaluate(s, 5, 75).value == pytest.approx(150)
    s["max_distance"] = .01
    assert evaluate(s, 5, 76).value is None


def test_published_reference_coordinates_and_gp_thresholds():
    from autopyro.interpretation import inverse
    s = standard_schemes()["HI–Tmax"]
    # Independently calibrated first point of the Type I vector path in Fig.13.3b.
    entry = next(e for e in s["lines"] if e["id"] == "I")
    x, y = entry["geometry"]["coordinates"][0]
    assert inverse(x, s["x"]) == pytest.approx(424.60509, abs=1e-4)
    assert inverse(y, s["y"]) == pytest.approx(866.07791, abs=.01)
    assert s["revision"] == "4"
    assert len(standard_schemes()["HI–OI"]["lines"]) == 4
    s = standard_schemes()["S1+S2–TOC"]
    assert evaluate(s, 1.5, 1.9).regions == ["2:0"]
    assert evaluate(s, 1.5, 2).status == "boundary"
    assert evaluate(s, 1.5, 5).regions == ["2:1"]
    assert evaluate(s, 1.5, 6).status == "boundary"
    assert evaluate(s, 1.5, 20).status == "classified"


def test_bundled_nearest_curves_reject_distant_points_and_intervals():
    from autopyro.schemes.interpretation import inverse

    schemes = standard_schemes()
    akram = next(s for name, s in schemes.items() if name.startswith('HI–Tmax · Akram'))
    assert akram['revision'] == '3'
    assert evaluate(akram, 520, 1000).status == 'not_applicable'
    assert evaluate(schemes['HI–Tmax'], 520, 1100).status == 'not_applicable'
    curve = next(line for line in akram['lines'] if not line.get('reference_only'))
    x, y = curve['geometry']['coordinates'][0]
    t, hi = inverse(x, akram['x']), inverse(y, akram['y'])
    assert evaluate(akram, t, hi).status in ('classified', 'ambiguous')
    assert evaluate(akram, t, hi, x_bounds=(t, 520), y_bounds=(hi, 1000)).status == 'not_applicable'


def test_published_hio_anchor_and_crossing_are_not_confused():
    from shapely.geometry import shape
    from autopyro.interpretation import inverse
    s = standard_schemes()["HIo · HI–Tmax"]
    anchor = evaluate(s, 424.60509, 866.07824)
    assert anchor.value == pytest.approx(866.07824, abs=.01)
    assert anchor.value_bounds[0] <= anchor.value <= anchor.value_bounds[1]
    lines = {e['id']: shape(e['geometry']) for e in s['lines']}
    crossing = lines['I'].intersection(lines['II'])
    assert crossing.geom_type == 'Point'
    result = evaluate(s, inverse(crossing.x, s['x']), inverse(crossing.y, s['y']))
    assert result.status == 'ambiguous'
    assert result.value is None
    assert result.value_bounds[1] - result.value_bounds[0] > 100
