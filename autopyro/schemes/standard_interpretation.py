"""Published screening criteria and traceable vector-digitized standard curves."""
from shapely.geometry import box, mapping

QUALITY_SOURCE = "https://emrlibrary.gov.yk.ca/ygs/open_files/2008/2008-1/2008_1.pdf"
MATURITY_SOURCE = "https://geoinfo.nmt.edu/publications/openfile/downloads/500-599/502/ofr-502.pdf"

_SVG_POTENTIAL_REGIONS = {
    # Region IDs in order from poor to excellent, at TOC = 10 wt%.
    "svg:S1+S2-TOC:1e97d702ca61": ("path12", "path8", "path9", "path10", "path11"),
    "svg:S1+S2-TOC:e56fa7fa0899": ("path9", "path5", "path6", "path7", "path8"),
    "svg:S1+S2-TOC:98e85d91ef97": ("path5", "path6", "path7", "path8", "path9"),
}

# Fixed coordinates in the original units. Each entry includes interior
# points, an interval spanning two regions and a TOC beyond the domain.
_SVG_S2_PROBES = {
    "svg:S2-TOC:b9255d809b81": (
        [("path7", 4.4, 2), ("path8", 5.1, 6.5), ("path9", 4.5, 18),
         ("path5", 2, 14), ("path6", .8, 14)],
        (5, 4, [2, 6], ["path7", "path8"]), 7,
    ),
    "svg:S2-TOC:df50f2543c30": (
        [(f"{i}:{j}", toc, s2)
         for i, toc in enumerate((.25, .7, 1.4, 3, 20))
         for j, s2 in enumerate((1, 3.5, 7, 17, 50))],
        (.7, 2, [1, 4], ["1:0", "1:1"]), 101,
    ),
    "svg:S2-TOC:5c345b9ca9e3": (
        [("path19", 3, .5), ("path15", .3, 50), ("path16", 1.4, 70),
         ("path17", 2.8, 100), ("path18", 20, 100)],
        (3, 2, [.5, 4], ["path19", "path15"]), 101,
    ),
    "svg:S2-TOC:667b0a10f2e2": (
        [("path16", 2.2, .9), ("path15", 2.5, 4), ("path14", 2.1, 5.4),
         ("path13", 1.2, 5.5), ("path12", .4, 5.7)],
        (2.4, 2, [1, 4], ["path16", "path15"]), 4,
    ),
}


_SVG_HI_OI_PROBES = {
    # Curve anchors in OI and HI units, with the type II and I anchors last.
    "svg:HI-OI:04b8097cd514": (
        [("path1543", 48.402, 70.385), ("path1591", 9.290, 363.146), ("path419", 4.654, 410.329)],
        (7, 385, [4, 10], [355, 415]), (125, 800),
    ),
    "svg:HI-OI:8d3c3c0d68be": (
        [("path7", 75.044, 122.032), ("path6", 33.638, 265.869),
         ("path5", 11.864, 455.031), ("path4", 8.222, 530.728)],
        (10, 490, [8, 12], [450, 535]), (200, 1000),
    ),
    "svg:HI-OI:b15ed7d16439": (
        [("path6", 80.657, 78.174), ("path5", 13.682, 432.659), ("path4", 11.364, 507.462)],
        (12, 465, [11, 14], [430, 510]), (160, 1000),
    ),
    "svg:HI-OI:5d1cf6e0d0b3": (
        [("path996", 93.900, 80.282), ("path952", 82.741, 196.861),
         ("path1447", 9.401, 430.539), ("path552", 7.789, 499.841)],
        (8.5, 465, [7.5, 10], [425, 505]), (200, 1000),
    ),
    "svg:HI-OI:bee7e9eb73be": (
        [("path11", 85.963, 78.809), ("path10", 12.913, 388.945), ("path9", 9.638, 365.292)],
        (11, 377, [9, 14], [360, 395]), (155, 700),
    ),
    "svg:HI-OI:7093b510ed99": (
        [("path4", 74.317, 60.823), ("path3", 66.376, 160.782),
         ("path2", 15.334, 348.595), ("path1", 14.564, 413.506)],
        (15, 380, [14, 16], [345, 420]), (150, 800),
    ),
}


def _svg_hi_oi_reference_cases(anchors, interval, far_point):
    x, y, x_bounds, y_bounds = interval
    far_x, far_y = far_point
    return [
        dict(id=f"near-{curve_id}", x=oi, y=hi,
             expected=dict(status="classified", possible_types=[curve_id]))
        for curve_id, oi, hi in anchors
    ] + [
        dict(id="interval-spans-types-i-ii", x=x, y=y,
             x_bounds=list(x_bounds), y_bounds=list(y_bounds),
             expected=dict(status="ambiguous", possible_types=[curve_id for curve_id, _, _ in anchors[-2:]])),
        dict(id="far-from-curves", x=far_x, y=far_y,
             expected=dict(status="not_applicable", possible_types=[])),
        dict(id="outside-oi-domain", x=far_x + 1, y=y,
             expected=dict(status="not_applicable", possible_types=[])),
    ]


_SVG_HI_TMAX_PROBES = {
    # Frozen anchors on the digitized curves and a distant point in each domain.
    "svg:HI-Tmax:838bc85e9290": (
        [("path2974", 463.344, 39.130), ("path9362", 470.128, 56.006), ("path2842", 450.438, 176.715),
         ("path9390", 461.860, 270.828), ("path2452", 459.903, 406.023)], (520, 1000)),
    "svg:HI-Tmax:0258ef283fb9": (
        [("path1372", 455.559, 146.113), ("path1056", 448.407, 320.285),
         ("path920", 448.943, 431.452)], (520, 1000)),
    "svg:HI-Tmax:3780d6ee1fe0": (
        [("path1939", 440.721, 123.204), ("path1587", 443.112, 496.887),
         ("path1443", 452.599, 303.133)], (520, 900)),
    "svg:HI-Tmax:f1671e93f078": (
        [("path1", 441.241, 128.490), ("path2", 449.914, 328.087),
         ("path3", 453.803, 312.537)], (525, 900)),
    "svg:HI-Tmax:ebdd91983b82": (
        [("path6", 443.717, 122.826), ("path5", 442.127, 564.727),
         ("path4", 454.968, 318.153)], (525, 900)),
    "svg:HI-Tmax:9d41858f217d": (
        [("path4", 451.740, 342.920), ("path5", 448.486, 299.760),
         ("path6", 446.218, 149.512), ("path7", 446.381, 80.330)], (510, 900)),
    "svg:HI-Tmax:e143c8a66a14": (
        [("path4", 453.301, 467.660), ("path5", 445.608, 337.558),
         ("path6", 449.045, 161.662), ("path7", 449.495, 112.362)], (510, 900)),
    "svg:HI-Tmax:74624a1ffb7e": (
        [("path4", 443.146, 118.987), ("path5", 447.007, 441.113),
         ("path6", 454.459, 346.611)], (520, 900)),
    "svg:HI-Tmax:c174608a53bb": (
        [("III", 445.134, 62.882), ("II", 441.993, 475.600), ("I", 453.714, 387.877)], (500, 900)),
    "svg:HI-Tmax:73cbf5c2f7f3": (
        [("path4", 445.570, 44.825), ("path5", 445.946, 128.154), ("path6", 445.849, 194.232),
         ("path7", 445.033, 313.286), ("path8", 451.906, 447.331)], (500, 900)),
    "svg:HI-Tmax:4c20f476085b": (
        [("path4", 459.322, 113.100), ("path5", 432.410, 295.307), ("path6", 467.378, 149.728),
         ("path8", 446.504, 504.894), ("path7", 447.336, 547.150)], (550, 1000)),
    "svg:HI-Tmax:a1f1c826f301": (
        [("path4", 456.541, 133.847), ("path5", 461.137, 205.557),
         ("path6", 449.765, 518.517), ("path7", 463.151, 649.254)], (510, 1000)),
}


def _svg_hi_tmax_reference_cases(anchors, far_point):
    far_x, far_y = far_point
    return [
        dict(id=f"near-{curve_id}", x=tmax, y=hi,
             x_bounds=[round(tmax - .02, 3), round(tmax + .02, 3)],
             y_bounds=[round(hi - .2, 3), round(hi + .2, 3)],
             expected=dict(status="classified", possible_types=[curve_id]))
        for curve_id, tmax, hi in anchors
    ] + [
        dict(id="far-from-curves", x=far_x, y=far_y,
             expected=dict(status="not_applicable", possible_types=[])),
        dict(id="outside-tmax-domain", x=far_x + 1, y=far_y / 2,
             expected=dict(status="not_applicable", possible_types=[])),
    ]


_SVG_PI_PROBES = {
    "svg:PI-Tmax:6aa20f026e5f": (
        [("path15", 497, .05), ("path16", 437, .05), ("path14", 497, .55),
         ("path13", 465, .55), ("path12", 445, .55), ("path11", 418, .55)],
        (445, .1, [.08, .15], ["path16", "path12"]), 521,
    ),
    "svg:PI-Tmax:779eeb130103": (
        [("path14", 480, .05), ("path18", 487, .55), ("path15", 420, .05),
         ("path13", 468, .55), ("path12", 445, .55), ("path11", 405, .55)],
        (445, .1020745, [.08, .15], ["path15", "path12"]), 501,
    ),
    "svg:PI-Tmax:f1e42f41698c": (
        [("rect9", 418, .05), ("rect10", 450, .25), ("rect11", 473, .575),
         ("rect12", 418, .25), ("rect13", 505, .375)],
        (418, .1, [.05, .25], ["rect9", "rect12"]), 531,
    ),
}


def _svg_polygon_reference_cases(points, interval, outside_x, *, outside_id, outside_y):
    x, y, bounds, region_ids = interval
    return [
        dict(id=f"inside-{region_id}", x=sample_x, y=sample_y,
             expected=dict(status="classified", regions=[region_id]))
        for region_id, sample_x, sample_y in points
    ] + [
        dict(id="interval-crosses-boundary", x=x, y=y, y_bounds=list(bounds),
             expected=dict(status="ambiguous", uncertainty_regions=list(region_ids))),
        dict(id=outside_id, x=outside_x, y=outside_y,
             expected=dict(status="not_applicable", regions=[])),
    ]


def _svg_potential_reference_cases(region_ids):
    # Fixed physical coordinates also exercise the log transforms on both axes.
    classes = ("poor", "fair", "good", "very-good", "excellent")
    cases = [
        dict(id=grade, x=10, y=value, expected=dict(status="classified", regions=[region_id]))
        for grade, value, region_id in zip(classes, (.5, 4, 8, 15, 100), region_ids, strict=True)
    ]
    cases.extend([
        dict(id="interval-crosses-poor-fair", x=10, y=2, y_bounds=[1, 4],
             expected=dict(status="ambiguous", uncertainty_regions=list(region_ids[:2]))),
        dict(id="nonpositive-toc", x=0, y=5,
             expected=dict(status="not_applicable", regions=[])),
        dict(id="toc-above-domain", x=101, y=5,
             expected=dict(status="not_applicable", regions=[])),
    ])
    return cases


def _axis(prop, unit, maximum, minimum=0, scale=1):
    return dict(property=prop, unit=unit, minimum=minimum, maximum=maximum, scale=scale)


def _grid_reference_cases(*, inside, corner, interval_end, outside_x):
    """Probe the lowest cell and its shared edges with fixed coordinates."""
    x, y = inside
    edge_x, edge_y = corner
    return [
        dict(id="inside-first-cell", x=x, y=y,
             expected=dict(status="classified", regions=["0:0"])),
        dict(id="first-x-boundary", x=edge_x, y=y,
             expected=dict(status="boundary", boundary_regions=["0:0", "1:0"])),
        dict(id="first-y-boundary", x=x, y=edge_y,
             expected=dict(status="boundary", boundary_regions=["0:0", "0:1"])),
        dict(id="shared-corner", x=edge_x, y=edge_y,
             expected=dict(status="boundary", boundary_regions=["0:0", "0:1", "1:0", "1:1"])),
        dict(id="interval-crosses-first-y-boundary", x=x, y=y, y_bounds=[y, interval_end],
             expected=dict(status="ambiguous", uncertainty_regions=["0:0", "0:1"])),
        dict(id="outside-x-domain", x=outside_x, y=y,
             expected=dict(status="not_applicable", regions=[])),
    ]


def _grid_cutoff_cases(*, anchor, x_cutoffs=(), y_cutoffs=(), first_index=1):
    """Check shared edges against coordinates fixed independently of the grid."""
    x, y = anchor
    cases = []
    for index, cutoff in enumerate(x_cutoffs, first_index):
        cases.append(dict(id=f"x-boundary-{index}", x=cutoff, y=y,
                          expected=dict(status="boundary",
                                        boundary_regions=[f"{index - 1}:0", f"{index}:0"])))
    for index, cutoff in enumerate(y_cutoffs, first_index):
        cases.append(dict(id=f"y-boundary-{index}", x=x, y=cutoff,
                          expected=dict(status="boundary",
                                        boundary_regions=[f"0:{index - 1}", f"0:{index}"])))
    return cases


def _svg_remaining_reference_cases():
    zhang = [
        dict(id=f"inside-{generation}:{maturity}", x=tmax, y=hi,
             expected=dict(status="classified", regions=[f"{generation}:{maturity}"]))
        for generation, hi in (("path9", 98), ("path8", 246), ("path7", 648))
        for maturity, tmax in (("path13", 490), ("path12", 440), ("path11", 410))
    ]
    zhang.extend([
        dict(id="maturity-interval", x=440, y=98, x_bounds=[415, 445],
             expected=dict(status="ambiguous", uncertainty_regions=["path9:path12", "path9:path11"])),
        dict(id="outside-tmax-domain", x=531, y=300, expected=dict(status="not_applicable")),
    ])
    hi_toc = [
        dict(id=f"inside-{region}", x=toc, y=hi, expected=dict(status="classified", regions=[region]))
        for region, toc, hi in (
            ("path45", .1755, 32), ("path44", 30, 306), ("path43", 10, 555),
            ("path42", 10, 1540), ("path41", 6.6, 2358), ("path40", 2.4, 2358),
        )
    ]
    hi_toc.extend([
        dict(id="source-quality-interval", x=4, y=2358, x_bounds=[2.4, 6.6],
             expected=dict(status="ambiguous", uncertainty_regions=["path40", "path41"])),
        dict(id="outside-toc-domain", x=101, y=200, expected=dict(status="not_applicable")),
    ])
    # Exact endpoints in physical units are needed because this scheme has zero tolerance.
    jarvie = [
        dict(id=f"on-{line}", x=toc, y=yield_value,
             expected=dict(status="boundary", line_hits=[line], regions=[]))
        for line, toc, yield_value in (
            ("path533", 8.05529454395494, 0), ("path527", 12.7099044, 50.847948),
            ("path543", 22.3905192, 89.6420304), ("path519", 22.470303, 89.96175720000001),
            ("path541", 16.9750269, 24.2749872), ("path537", 30, 29.99362110044652),
            ("path523", 30, 60.01702319285026), ("path511", 29.999981100000003, 90.1005144),
            ("path479", 29.9453154, 119.9171112), ("path485", 23.9343765, 119.9085468),
            ("path477", 20.8158549, 119.90410440000001), ("path475", 17.726130299999998, 119.8997016),
            ("path483", 15.439739999999999, 119.8964436), ("path481", 13.6868154, 119.8939464),
            ("path473", 29.981549700000002, 29.975051999999998), ("path547", 29.9791374, 59.9993832),
            ("path515", 22.208352124323973, 0), ("path491", 29.972013, 119.9171484),
            ("path487", 13.803987224403084, 0), ("path495", 9.932671094779856, 0),
            ("path507", 6.86750614600058, 0), ("path505", 15.4478628, 119.89645560000001),
            ("path501", 13.7011695, 119.90527920000001),
        )
    ]
    jarvie.extend([
        dict(id="between-lines", x=15, y=15, expected=dict(status="outside", line_hits=[], regions=[])),
        dict(id="outside-toc-domain", x=31, y=40, expected=dict(status="not_applicable", line_hits=[])),
    ])
    return {
        "svg:HI-Tmax:7db83cea51ca": zhang,
        "svg:HI-TOC:0361ff38ba45": hi_toc,
        "svg:TOCo:4c82336c6d05": jarvie,
    }


def standard_schemes():
    """Return independent JSON-compatible schemes, ready for editing and archiving."""
    schemes = {}
    toc = _axis("TOC", "%wt", 100)
    hc = "mg HC/g rock"
    hi = _axis("HI", "mg HC/g TOC", 1200, scale=1200)

    def create(name, x, y, source=QUALITY_SOURCE, demo=False):
        scheme = dict(format="autopyro.interpretation", version=1,
                      id="standard:" + name, revision="1", name=name, source=source,
                      scientific_status="demonstration" if demo else "validated",
                      coordinate_space="analysis", boundary_policy="flag", tolerance=0,
                      x=x.copy(), y=y.copy(), regions=[], lines=[])
        schemes[name] = scheme
        return scheme

    def grid(scheme, xs, ys, xlabels, ylabels):
        for i, (left, right) in enumerate(zip(xs, xs[1:])):
            for j, (bottom, top) in enumerate(zip(ys, ys[1:])):
                scheme["regions"].append(dict(id=f"{i}:{j}",
                    label=f"{xlabels[i]}; {ylabels[j]}",
                    geometry=mapping(box(left, bottom, right, top))))

    quality = ["бедный", "удовлетворительный", "хороший", "очень хороший", "отличный"]
    for name, prop, limits in [("S2–TOC", "S2", [0, 2.5, 5, 10, 20, 1000]),
                                ("S1–TOC", "S1", [0, .5, 1, 2, 4, 1000])]:
        s = create(name, toc, _axis(prop, hc, 1000))
        grid(s, [0, .5, 1, 2, 4, 100], limits,
             ["TOC: " + q for q in quality], [prop + ": " + q for q in quality])
    s = create("HI–TOC", toc, _axis("HI", "mg HC/g TOC", 1200))
    grid(s, [0, .5, 1, 2, 4, 100], [0, 50, 200, 300, 600, 1200],
         ["TOC: " + q for q in quality],
         ["IV · инертное", "III · газ", "II/III · нефть/газ", "II · нефть", "I · нефть"])
    s = create("PI–Tmax", _axis("Tmax", "°C", 600, 300), _axis("PI", "1", 1), MATURITY_SOURCE)
    grid(s, [300, 435, 470, 600], [0, .1, .4, 1],
         ["Tmax: незрелое", "Tmax: зрелое", "Tmax: перезрелое"],
         ["PI: <0.1", "PI: 0.1–0.4", "PI: >0.4"])
    # Keep reference coordinates independent from the geometry so a cutoff
    # change does not silently change the corresponding test.
    for name, anchor, corner, interval_end, outside_x, x_cutoffs, y_cutoffs in (
        ("S1–TOC", (.25, .25), (.5, .5), .6, 101, (1, 2, 4), (1, 2, 4)),
        ("S2–TOC", (.25, 2), (.5, 2.5), 3, 101, (1, 2, 4), (5, 10, 20)),
        ("HI–TOC", (.25, 40), (.5, 50), 60, 101, (1, 2, 4), (200, 300, 600)),
        ("PI–Tmax", (420, .08), (435, .1), .12, 601, (470,), (.4,)),
    ):
        cases = _grid_reference_cases(inside=anchor, corner=corner, interval_end=interval_end, outside_x=outside_x)
        cases.extend(_grid_cutoff_cases(
            anchor=anchor, x_cutoffs=x_cutoffs, y_cutoffs=y_cutoffs, first_index=2))
        schemes[name].update(revision="3", reference_cases=cases)
    s = create("S1+S2–TOC", toc, _axis("GP", hc, 1000))
    s.update(revision="4", source="Tissot & Welte (1984), p. 513; thresholds 2 and 6 mg HC/g rock, reproduced in Rosenthal, p. 208: https://mspace.lib.umanitoba.ca/bitstream/1993/16928/2/Rosenthal_The_stratigraphy_v2.pdf")
    grid(s, [0, .5, 1, 2, 4, 100], [0, 2, 6, 1000],
         ["TOC: " + q for q in quality], ["GP: бедный", "GP: умеренный", "GP: хороший"])
    # Synthetic probes at a fixed TOC isolate the two published GP cutoffs.
    s["reference_cases"] = [
        dict(id="gp-poor", x=1.5, y=1.9,
             expected=dict(status="classified", regions=["2:0"])),
        dict(id="gp-boundary-2", x=1.5, y=2,
             expected=dict(status="boundary", boundary_regions=["2:0", "2:1"])),
        dict(id="gp-moderate", x=1.5, y=5,
             expected=dict(status="classified", regions=["2:1"])),
        dict(id="gp-boundary-6", x=1.5, y=6,
             expected=dict(status="boundary", boundary_regions=["2:1", "2:2"])),
        dict(id="gp-good", x=1.5, y=20,
             expected=dict(status="classified", regions=["2:2"])),
        dict(id="gp-interval-crosses-2", x=1.5, y=1.9, y_bounds=[1.8, 2.2],
             expected=dict(status="ambiguous", uncertainty_regions=["2:0", "2:1"])),
        dict(id="gp-interval-crosses-6", x=1.5, y=5.9, y_bounds=[5.8, 6.2],
             expected=dict(status="ambiguous", uncertainty_regions=["2:1", "2:2"])),
        dict(id="gp-interval-within-moderate", x=1.5, y=5, y_bounds=[4, 5.8],
             expected=dict(status="classified", regions=["2:1"], uncertainty_regions=["2:1"])),
        dict(id="toc-outside-domain", x=101, y=5,
             expected=dict(status="not_applicable", regions=[])),
    ]
    s["reference_cases"].extend(_grid_cutoff_cases(anchor=(.25, 1), x_cutoffs=(.5, 1, 2, 4)))

    import json
    from pathlib import Path
    from .interpretation import transform
    published = json.loads(Path(__file__).with_name("published_curves.json").read_text(encoding="utf-8"))
    source = "Petroleum Geological Atlas of the Southern Permian Basin Area (2010), chapter 13, p. 227, Fig. 13.3; https://www.nlog.nl/sites/default/files/2018-12/spba-chapter13.pdf"
    for name, x in [("HI–Tmax", _axis("Tmax", "°C", 520, 400, 120)),
                    ("HI–OI", _axis("OI", "mg CO2/g TOC", 300, scale=300))]:
        s = create(name, x, hi, source)
        s.update(revision="2", strategy="nearest_curve", distance_metric="analysis_euclidean",
                 metric_note="Евклидова метрика: X / диапазон оси; HI / 1200. Геометрическая оценка типа по ближайшей опубликованной траектории, не вероятность.",
                 provenance=dict(figure="13.3b" if name == "HI–Tmax" else "13.3c",
                     method="PDF vector Bezier paths, 32 samples per segment; arrowheads excluded",
                     pdf_page=4, printed_page=227, coordinate_precision="5 decimal places; source diagram is schematic"))
        for entry in published[name]:
            s["lines"].append(dict(id=entry["id"], label=entry["label"], geometry=dict(
                type="LineString", coordinates=[[transform(a, x), transform(b, hi)] for a,b in entry["coordinates"]])))
        s["limitations"] = "Оценка по стандартным траекториям; при зрелости и смешении типов возможна неоднозначность. Не заменяет петрографию и контроль качества Rock-Eval."
    from copy import deepcopy
    s = deepcopy(schemes["HI–Tmax"])
    s.update(id="standard:HIo", name="HIo · HI–Tmax", revision="3", strategy="initial_hi",
             intermediate_curves=199, curve_alignment="normalized_arclength",
             max_distance=0.01, no_extrapolation=True, distance_tolerance=0.0001, hi_resolution=10, scientific_status="user_supplied")
    s["limitations"] = "HIo — графическая модельная оценка по интерполяции стандартных траекторий, не опубликованная универсальная калибровка. Начальное HI берётся на самом левом конце опубликованной линии. Вне семейства траекторий экстраполяция запрещена."
    # The Type I anchor comes from the digitized path. The crossing envelope
    # is a regression baseline for this interpolation model and its settings.
    s["reference_cases"] = [
        dict(id="hio-type-i-anchor", x=424.60509, y=866.07824,
             expected=dict(status="classified", possible_types=["I"],
                           value_bounds=[866.07824, 866.07824])),
        dict(id="hio-crossing-interval", x=437.159121, y=259.752705,
             x_bounds=[437.1591, 437.1592], y_bounds=[259.75265, 259.75275],
             expected=dict(status="ambiguous", possible_types=["I", "II"],
                           value_bounds=[331.170162, 866.07824])),
        dict(id="hio-away-from-curves", x=510, y=1100,
             expected=dict(status="not_applicable", possible_types=[], value_bounds=None)),
        dict(id="hio-outside-tmax-domain", x=525, y=500,
             expected=dict(status="not_applicable", possible_types=[], value_bounds=None)),
    ]
    schemes[s["name"]] = s

    # Fixed points on the digitized paths and synthetic applicability checks.
    schemes["HI–Tmax"]["reference_cases"] = [
        dict(id="type-i", x=424.60509, y=866.07824,
             expected=dict(status="classified", possible_types=["I"])),
        dict(id="type-ii", x=416.26327, y=542.53115,
             expected=dict(status="classified", possible_types=["II"])),
        dict(id="type-iii", x=408.03161, y=209.6792,
             expected=dict(status="classified", possible_types=["III"])),
        dict(id="crossing-interval", x=437.159121, y=259.752705,
             x_bounds=[437.14, 437.18], y_bounds=[259.3, 260.1],
             expected=dict(status="ambiguous", possible_types=["I", "II"])),
        dict(id="far-from-curves", x=520, y=1100,
             expected=dict(status="not_applicable", possible_types=[])),
        dict(id="outside-axis-domain", x=521, y=200,
             expected=dict(status="not_applicable", possible_types=[])),
    ]
    schemes["HI–OI"]["reference_cases"] = [
        dict(id="type-i", x=41.82431, y=815.82509,
             expected=dict(status="classified", possible_types=["I"])),
        dict(id="type-ii", x=69.4043, y=577.07855,
             expected=dict(status="classified", possible_types=["II"])),
        dict(id="type-iii", x=151.92002, y=228.65046,
             expected=dict(status="classified", possible_types=["III"])),
        dict(id="type-iii-iv", x=201.02617, y=54.43828,
             expected=dict(status="classified", possible_types=["III/IV"])),
        dict(id="interval-within-type-i", x=42, y=815, x_bounds=[41, 43], y_bounds=[810, 820],
             expected=dict(status="classified", possible_types=["I"])),
        dict(id="interval-spans-types-i-ii", x=55, y=700, x_bounds=[41.8, 69.5], y_bounds=[575, 816],
             expected=dict(status="ambiguous", possible_types=["I", "II"])),
        dict(id="far-from-curves", x=300, y=1100,
             expected=dict(status="not_applicable", possible_types=[])),
        dict(id="outside-axis-domain", x=301, y=200,
             expected=dict(status="not_applicable", possible_types=[])),
    ]
    supplied = json.loads(Path(__file__).with_name("svg_interpretation.json").read_text(encoding="utf-8"))
    remaining_cases = _svg_remaining_reference_cases()
    for scheme in supplied:
        if scheme["id"] in remaining_cases:
            scheme.update(revision="2", reference_cases=remaining_cases[scheme["id"]])
        region_ids = _SVG_POTENTIAL_REGIONS.get(scheme["id"])
        if region_ids is not None:
            scheme.update(revision="2", reference_cases=_svg_potential_reference_cases(region_ids))
        s2_probes = _SVG_S2_PROBES.get(scheme["id"])
        if s2_probes is not None:
            scheme.update(revision="2", reference_cases=_svg_polygon_reference_cases(
                *s2_probes, outside_id="toc-above-domain", outside_y=2))
        pi_probes = _SVG_PI_PROBES.get(scheme["id"])
        if pi_probes is not None:
            cases = _svg_polygon_reference_cases(
                *pi_probes, outside_id="tmax-above-domain", outside_y=.3)
            if scheme["id"] == "svg:PI-Tmax:779eeb130103":
                cases.append(dict(id="conversion-line", x=445, y=.1020745,
                                  expected=dict(status="boundary", regions=["path12"])))
            scheme.update(revision="2", reference_cases=cases)
    schemes.update((scheme["name"], scheme) for scheme in supplied)
    for scheme in schemes.values():
        if scheme.get('strategy') == 'nearest_curve':
            scheme.update(revision='2' if scheme.get('group') else '4', max_distance=.15)
            scheme['limitations'] += (' Геометрический предел нормированного расстояния — 0,15.'
                                      ' Этот предел не является научной калибровкой.')
        hi_oi_probes = _SVG_HI_OI_PROBES.get(scheme["id"])
        if hi_oi_probes is not None:
            scheme.update(revision="3", reference_cases=_svg_hi_oi_reference_cases(*hi_oi_probes))
        hi_tmax_probes = _SVG_HI_TMAX_PROBES.get(scheme["id"])
        if hi_tmax_probes is not None:
            scheme.update(revision="3", reference_cases=_svg_hi_tmax_reference_cases(*hi_tmax_probes))
    return schemes
