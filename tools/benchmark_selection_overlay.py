"""Measure UUID highlight updates on a cached Agg canvas, separately from rendering."""
import argparse
import json
from pathlib import Path
import platform
from time import perf_counter

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

from autopyro.core import Column, Row, Project
from autopyro.plotting import PlotSpec, render_plot
from autopyro.selection_overlay import SelectionOverlay


def measure(size, repeats, *, compact=False):
    x, y = Column('X', 'number'), Column('Y', 'number')
    project = Project(columns=[x, y], rows=[Row({x.id: i, y.id: 1 + (i % 997) / 100}) for i in range(size)])
    ids = [row.id for row in project.rows]
    spec = PlotSpec(x_id=x.id, y_id=y.id, color_by=x.id, color_mode='numeric')
    preparation_started = perf_counter()
    figure, result = render_plot(project, ids, spec)
    prepared = perf_counter()
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    drawn = perf_counter()
    overlay = SelectionOverlay(canvas, result, compact=compact)
    if compact:
        from pandas import Index
        from autopyro.plot_preview import pack_preview_membership, restore_preview_membership
        result['_selection_overlay'] = overlay.layers, overlay.locations
        pack_preview_membership(result, Index(ids, dtype=object))
        restore_preview_membership(result, np.asarray(ids, dtype=object), {rid: i for i, rid in enumerate(ids)})
        result.pop('_selection_overlay')
    indexed = perf_counter()
    canvas.draw()  # Capture the background before timing the updates.
    updates = {}
    for count in sorted({1, min(100, size), min(10000, size)}):
        samples = []
        for i in range(repeats):
            selected = ids[i:i + count] if i + count <= size else ids[:count]
            overlay.set_selected([])
            started = perf_counter()
            overlay.set_selected(selected)
            samples.append(perf_counter() - started)
        assert len(overlay.selected_ids) == count
        updates[count] = {'runs_s': samples, 'median_s': float(np.median(samples)),
                          'p95_s': float(np.percentile(samples, 95))}
    overlay.dispose()
    figure.clear()
    return {'rows': size, 'prepare_s': prepared - preparation_started,
            'full_draw_s': drawn - prepared, 'index_s': indexed - drawn, 'updates': updates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[1000, 10000, 100000])
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--compact', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 1 for size in args.sizes):
        parser.error('Sizes and repeats must be positive')
    result = {'platform': platform.platform(), 'python': platform.python_version(),
              'matplotlib': matplotlib.__version__,
              'compact_index': args.compact,
              'ordinal_lookup': args.compact,
              'note': 'Synthetic table-only numeric-color scatter. Agg, fixed figure size, no Qt selection signals, '
                      'no RSS. Background capture and clearing selection excluded from update times; '
                      'each update starts empty. Full draw is a single measurement.',
              'measurements': []}
    for size in args.sizes:
        measured = measure(size, args.repeats, compact=args.compact)
        result['measurements'].append(measured)
        print(json.dumps({'rows': size, 'index_s': measured['index_s'],
                          'full_draw_s': measured['full_draw_s'], 'p95_s':
                          {key: value['p95_s'] for key, value in measured['updates'].items()}}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
