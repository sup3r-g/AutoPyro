"""Build the bundled GeoJSON catalog from the supplied Inkscape charts.

Usage: python tools/convert_interpretation_svg.py PATH_TO_SVG_ROOT
Requires fontTools for reading vector paths. Embedded images are not bundled.
"""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from xml.etree import ElementTree

from fontTools.pens.recordingPen import RecordingPen
from fontTools.svgLib.path import parse_path
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box, mapping, shape
from shapely import make_valid


INKSCAPE = '{http://www.inkscape.org/namespaces/inkscape}'
FRAME = box(0, 0, 1, 1)
AXES = {
    'HI-OI': ('OI', 'mg CO2/g TOC', 'HI', 'mg HC/g TOC'),
    'HI-Tmax': ('Tmax', '°C', 'HI', 'mg HC/g TOC'),
    'HI-TOC': ('TOC', '%wt', 'HI', 'mg HC/g TOC'),
    'PI-Tmax': ('Tmax', '°C', 'PI', '1'),
    'S1+S2-TOC': ('TOC', '%wt', 'GP', 'mg HC/g rock'),
    'S2-TOC': ('TOC', '%wt', 'S2', 'mg HC/g rock'),
    'TOCo': ('TOC', '%wt', 'Kerogen Yield', 'mg HC/g rock'),
}
# The remaining frames carry numerical bounds in their Inkscape label.
# These ranges were read from the embedded figure's ticks and the traced frame.
CALIBRATION = {
    ('HI-TOC', 'HI-TOC'): ((.1, 100), (10, 10000), True, True),
    ('PI-Tmax', 'Akram'): ((400, 520), (0, 1), False, False),
    ('PI-Tmax', 'Edress'): ((380, 500), (0, 1), False, False),
    ('PI-Tmax', 'Khatibi'): ((400, 530), (0, .75), False, False),
    ('S1+S2-TOC', 'Edress'): ((.1, 100), (.1, 1000), True, True),
    ('S1+S2-TOC', 'Maky'): ((.2, 100), (.1, 1000), True, True),
    ('S1+S2-TOC', 'Spasennykh'): ((.1, 100), (.1, 1000), True, True),
    ('S2-TOC', 'Chen'): ((0, 6), (0, 27), False, False),
    ('S2-TOC', 'Edress'): ((.1, 100), (.1, 100), True, True),
    ('S2-TOC', 'T.A. Ryazanova'): ((.1, 100), (.1, 500), True, True),
    ('S2-TOC', 'Zeb'): ((0, 3), (0, 11), False, False),
}


def label(element):
    return element.get(INKSCAPE + 'label', '')


def tag(element):
    return element.tag.rsplit('}', 1)[-1]


def translation(element):
    value = element.get('transform')
    if not value:
        return 0, 0
    match = re.fullmatch(r'translate\(([^, ]+)[, ]+([^,) ]+)\)', value)
    if not match:
        raise ValueError(f'Unsupported transform: {value}')
    return tuple(map(float, match.groups()))


def rect(element):
    return tuple(float(element.get(key)) for key in ('x', 'y', 'width', 'height'))


def svg_paths(element):
    if tag(element) == 'rect':
        x, y, w, h = rect(element)
        yield [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)], True
        return
    pen = RecordingPen()
    parse_path(element.get('d'), pen)
    points = []
    for command, coords in pen.value:
        if command in ('moveTo', 'lineTo'):
            points.append(coords[0])
        elif command == 'curveTo':
            start = points[-1]
            for step in range(1, 25):
                t = step / 24
                weights = ((1 - t)**3, 3 * t * (1 - t)**2, 3 * t*t * (1 - t), t**3)
                points.append(tuple(sum(w * v for w, v in zip(weights, axis, strict=True))
                                    for axis in zip(start, *coords, strict=True)))
        elif command in ('closePath', 'endPath'):
            yield points, command == 'closePath'
            points = []
        else:
            raise ValueError(f'Unsupported path command: {command}')


def parts(geometry, kind):
    if geometry.is_empty:
        return []
    if geometry.geom_type == kind:
        return [geometry]
    return [part for child in getattr(geometry, 'geoms', ()) for part in parts(child, kind)]


def axis(prop, unit, limits, logarithmic):
    lo, hi = limits
    start, end = (math.log10(lo), math.log10(hi)) if logarithmic else limits
    return dict(property=prop, unit=unit, minimum=lo, maximum=hi,
                origin=start, scale=end - start, transform='log10' if logarithmic else 'identity')


def convert(path):
    group, title = path.parent.name, path.stem
    data = path.read_bytes()
    root = ElementTree.fromstring(data)
    layers = {label(e): e for e in root if tag(e) == 'g'}
    border_layer = layers['Border']
    border = next(e for e in border_layer if tag(e) == 'rect')
    x, y, width, height = rect(border)
    dx, dy = translation(border)
    lx, ly = translation(border_layer)
    x, y = x + dx + lx, y + dy + ly
    numbers = re.findall(r'-?\d+(?:\.\d+)?', label(border))
    if numbers:
        a, b, c, d = map(float, numbers)
        x_range, y_range, log_x, log_y = (a, c), (b, d), False, False
    else:
        x_range, y_range, log_x, log_y = next(
            value for (category, prefix), value in CALIBRATION.items()
            if category == group and title.startswith(prefix))
    x_prop, x_unit, y_prop, y_unit = AXES[group]
    if title.startswith('Spasennykh'):
        y_prop = 'S0+S1+S2'
    source = f'SVG/{group}/{path.name}'
    source_id = sha256(f'{group}/{title}'.encode()).hexdigest()[:12]
    scheme = dict(format='autopyro.interpretation', version=1, revision='1',
                  id=f'svg:{group}:{source_id}', name=f'{group.replace("-", "–")} · {title}',
                  group=group.replace('-', '–'), source=source, scientific_status='user_supplied',
                  coordinate_space='analysis', boundary_policy='flag', tolerance=0,
                  x=axis(x_prop, x_unit, x_range, log_x), y=axis(y_prop, y_unit, y_range, log_y),
                  regions=[], lines=[], provenance={'source_sha256': sha256(data).hexdigest(),
                  'method': 'Inkscape paths; SVG frame calibrated in linear or log10 coordinates'})
    scheme['limitations'] = 'Оцифровка предоставленного рисунка. Границы применимы в пределах шкал источника.'

    for layer, node in layers.items():
        if layer in ('Image', 'Border'):
            continue
        lx, ly = translation(node)
        for element in node:
            if tag(element) not in ('path', 'rect'):
                continue
            ident = element.get('id')
            title_label = label(element)
            if not title_label and path.stem.startswith('Khatibi'):
                title_label = {'rect11': 'MATURITY: Condensate / Wet Gas Window',
                               'rect12': 'S1 enrichment by HC migration or contamination'}[ident]
            if not title_label and layer == 'Porosity':
                title_label = 'Porosity · ' + ident
            if not title_label:
                raise ValueError(f'{source}: unlabeled geometry {ident}')
            dx, dy = translation(element)
            style = dict(part.split(':', 1) for part in element.get('style', '').split(';') if ':' in part)
            polygons, lines = [], []
            for points, closed in svg_paths(element):
                coords = [(round((px + dx + lx - x) / width, 8),
                           round(1 - (py + dy + ly - y) / height, 8)) for px, py in points]
                if closed:
                    polygons.extend(parts(make_valid(Polygon(coords)).intersection(FRAME), 'Polygon'))
                else:
                    lines.extend(parts(LineString(coords).intersection(FRAME), 'LineString'))
            if polygons:
                geometry = polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
                scheme['regions'].append(dict(id=ident, label=title_label, geometry=mapping(geometry)))
            elif lines:
                geometry = lines[0] if len(lines) == 1 else MultiLineString(lines)
                entry = dict(id=ident, label=title_label, geometry=mapping(geometry))
                stroke = style.get('stroke', element.get('stroke', ''))
                if re.fullmatch(r'#[0-9a-fA-F]{6}', stroke):
                    entry['color'] = stroke
                if style.get('stroke-dasharray', 'none') != 'none':
                    entry['dashed'] = True
                if layer != 'MATTER TYPE':
                    entry['reference_only'] = True
                scheme['lines'].append(entry)
            else:
                raise ValueError(f'{source}: geometry outside frame: {ident}')
    if not scheme['regions'] and group in ('HI-Tmax', 'HI-OI'):
        scheme.update(strategy='nearest_curve', distance_metric='analysis_euclidean',
                      metric_note='Расстояние по осям, нормированным на диапазоны рисунка.')
        scheme['limitations'] += ' Ближайшая траектория даёт геометрическую оценку типа.'
    else:
        for entry in scheme['lines']:
            entry.pop('reference_only', None)
    if title.startswith('Zhang, Xiaohuan'):
        # The two complete thematic layers describe independent attributes.
        # Their intersections form the mutually exclusive output fields.
        original = scheme['regions']
        scheme['regions'] = []
        for generation in (e for e in original if e['label'].startswith('GENERATION:')):
            for maturity in (e for e in original if e['label'].startswith('MATURITY:')):
                polygons = parts(shape(generation['geometry']).intersection(shape(maturity['geometry'])), 'Polygon')
                if polygons:
                    geometry = polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
                    scheme['regions'].append(dict(id=f"{generation['id']}:{maturity['id']}",
                        label=f"{generation['label']} · {maturity['label']}", geometry=mapping(geometry)))
        scheme['limitations'] += ' Кривые типа ОВ и Ro показаны как справочные.'
    if group == 'TOCo':
        scheme['limitations'] += ' Справочная номограмма: численная оценка TOCo не задана.'
    if group == 'S2-TOC' and title.startswith('Edress'):
        divisions = {'X': [], 'Y': []}
        for entry in scheme['lines']:
            name, sides = entry['label'].split(':')
            less, greater = map(str.strip, sides.split(','))
            dimension = name[-1]
            low_x, low_y, _, _ = shape(entry['geometry']).bounds
            divisions[dimension].append((low_x if dimension == 'X' else low_y, less, greater))

        def fields(dimension):
            cuts = sorted(divisions[dimension])
            bounds = [0, *(cut[0] for cut in cuts), 1]
            names = [cuts[0][1], *(cut[2] for cut in cuts)]
            return [(start, end, name) for start, end, name in zip(bounds[:-1], bounds[1:], names, strict=True)]

        for i, (left, right, toc) in enumerate(fields('X')):
            for j, (bottom, top, s2) in enumerate(fields('Y')):
                scheme['regions'].append(dict(id=f'{i}:{j}', label=f'TOC: {toc} · S2: {s2}',
                    geometry=mapping(box(left, bottom, right, top))))
        scheme['lines'].clear()
    return scheme


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    schemes = [convert(path) for path in sorted(args.root.glob('*/*.svg'))]
    if not schemes:
        parser.error('No SVG charts found')
    output = Path(__file__).resolve().parents[1] / 'autopyro/schemes/svg_interpretation.json'
    output.write_text('[\n' + ',\n'.join(json.dumps(s, ensure_ascii=False, separators=(',', ':'))
                                        for s in schemes) + '\n]\n', encoding='utf-8')


if __name__ == '__main__':
    main()
