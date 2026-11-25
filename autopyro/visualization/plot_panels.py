"""PNG and SVG grids for independently rendered saved plots."""
from contextlib import nullcontext
from dataclasses import replace
from io import BytesIO
from math import ceil
from pathlib import Path
import re
from xml.etree import ElementTree as ET

from matplotlib import rc_context
from PIL import Image

from .plot_data import check_cancelled
from .plotting import _save_rendered_plot, render_plot

SVG = 'http://www.w3.org/2000/svg'
XLINK = 'http://www.w3.org/1999/xlink'


def panel_layout(path, specs, columns):
    if len(specs) < 2 or type(columns) is not int or not 1 <= columns <= len(specs):
        raise ValueError('Выберите не менее двух графиков и допустимое число столбцов')
    suffix = Path(path).suffix.lower()
    if suffix not in {'.png', '.svg'}:
        raise ValueError('Многопанельный рисунок сохраняется в PNG или SVG')
    dpi = specs[0].dpi
    if not 36 <= dpi <= 1200:
        raise ValueError('DPI рисунка должен быть от 36 до 1200')
    widths = [spec.width_mm / 25.4 * dpi for spec in specs]
    heights = [spec.height_mm / 25.4 * dpi for spec in specs]
    if not all(0 < value <= 64_000_000 for value in (*widths, *heights)):
        raise ValueError('Некорректные размеры панелей')
    width, height = round(max(widths)), round(max(heights))
    gap = round(3 / 25.4 * dpi)
    rows = ceil(len(specs) / columns)
    size = (columns * width + (columns - 1) * gap, rows * height + (rows - 1) * gap)
    if suffix == '.png' and size[0] * size[1] > 64_000_000:
        raise ValueError('Многопанельный рисунок ограничен 64 млн пикселей; уменьшите размеры или DPI')
    return dpi, width, height, gap, size


def svg_grid(specs, columns):
    width = max(spec.width_mm for spec in specs)
    height = max(spec.height_mm for spec in specs)
    rows = ceil(len(specs) / columns)
    total_width = columns * width + (columns - 1) * 3
    total_height = rows * height + (rows - 1) * 3
    root = ET.Element(f'{{{SVG}}}svg', width=f'{total_width:g}mm', height=f'{total_height:g}mm',
                      viewBox=f'0 0 {total_width:g} {total_height:g}', version='1.1')
    return root, width, height


def append_svg_panel(root, buffer, index, x, y, spec):
    panel = ET.fromstring(buffer.getvalue())
    # Matplotlib restarts artist IDs in each figure. References resolve document-wide.
    prefix = f'panel-{index + 1}-'
    for node in panel.iter():
        for key, value in node.attrib.items():
            if key == 'id':
                node.set(key, prefix + value)
            elif key in {'href', f'{{{XLINK}}}href'} and value.startswith('#'):
                node.set(key, '#' + prefix + value[1:])
            elif 'url(#' in value:
                node.set(key, re.sub(r'url\(#([^)]*)\)', rf'url(#{prefix}\1)', value))
    panel.attrib.update(x=f'{x:g}', y=f'{y:g}', width=f'{spec.width_mm:g}', height=f'{spec.height_mm:g}')
    root.append(panel)


def write_panels(path, jobs, columns, cache, cancelled, progress):
    from .plot_export import install_palette

    specs = [job['spec'] for job in jobs]
    dpi, width, height, gap, size = panel_layout(path, specs, columns)
    check_cancelled(cancelled)
    vector = Path(path).suffix.lower() == '.svg'
    if vector:
        root, cell_width, cell_height = svg_grid(specs, columns)
    warnings = []
    with (nullcontext() if vector else Image.new('RGBA', size)) as result:
        for index, job in enumerate(jobs):
            check_cancelled(cancelled)
            install_palette(job.get('palette'))
            spec = replace(job['spec'], dpi=dpi)
            prepared = job.get('prepared')
            if prepared is None:
                prepared = cache.prepare(job['source'], job['ids'], spec, cancelled)
            figure, _ = render_plot(prepared.project, job['ids'], spec, prepared=prepared)
            panel_warnings = []
            with BytesIO() as buffer:
                if vector:
                    with rc_context({'svg.fonttype': 'none'}):
                        _save_rendered_plot(figure, buffer, spec, format='svg', layout_warnings=panel_warnings)
                else:
                    _save_rendered_plot(figure, buffer, spec, format='png', layout_warnings=panel_warnings)
                check_cancelled(cancelled)
                row, column = divmod(index, columns)
                if vector:
                    x = column * (cell_width + 3) + (cell_width - spec.width_mm) / 2
                    y = row * (cell_height + 3) + (cell_height - spec.height_mm) / 2
                    append_svg_panel(root, buffer, index, x, y, spec)
                else:
                    buffer.seek(0)
                    with Image.open(buffer) as panel:
                        x = column * (width + gap) + (width - panel.width) // 2
                        y = row * (height + gap) + (height - panel.height) // 2
                        result.paste(panel, (x, y))
            warnings.extend(f"{job.get('name', index + 1)}: {warning}" for warning in panel_warnings)
            progress(index + 1, len(jobs))
        check_cancelled(cancelled)
        if vector:
            ET.register_namespace('', SVG)
            ET.register_namespace('xlink', XLINK)
            ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
        else:
            result.save(path, dpi=(dpi, dpi))
    return warnings
