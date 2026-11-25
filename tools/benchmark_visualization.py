"""Measure plot preparation, Agg drawing and PNG writing separately on synthetic data.

Run with: python -m tools.benchmark_visualization --output tmp/visualization.json
Input construction is outside timings; this does not measure Qt responsiveness or RSS.
"""
import argparse
from io import BytesIO
import json
import platform
from statistics import median
from time import perf_counter

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from autopyro.core import Column, Project, Row
from autopyro.plotting import PlotSpec, render_plot


def measure(size, repeats, kind='scatter'):
    columns = [Column("Depth", "number", unit="m"), Column("S2", "number", unit="mg HC/g rock"),
               Column("Well", "text")]
    x, y, group = [column.id for column in columns]
    project = Project(columns=columns, rows=[Row({x: i, y: 1 + (i % 997) / 100,
        group: f"Well-{i % 12}"}) for i in range(size)])
    ids = [row.id for row in project.rows]
    measurements = []
    for name, style in (("uniform", {}), ("numeric", {"color_by": x, "color_mode": "numeric"}),
                        ("category", {"color_by": group, "shape_by": group})):
        figure = Figure(dpi=96)
        canvas = FigureCanvasAgg(figure)
        spec = PlotSpec(kind=kind, x_id=x, y_id=y, **style)
        samples = []
        for _ in range(repeats):
            started = perf_counter()
            _, result = render_plot(project, ids, spec, figure=figure)
            prepared = perf_counter()
            canvas.draw()
            drawn = perf_counter()
            with BytesIO() as stream:
                # Includes another render at export DPI, as a real PNG export does.
                figure.savefig(stream, format="png", dpi=150)
                png_bytes = stream.tell()
            finished = perf_counter()
            assert result["plotted"][x] == size
            samples.append({"prepare_s": prepared - started, "draw_s": drawn - prepared,
                            "png_s": finished - drawn, "png_bytes": png_bytes})
        measurements.append({"rows": size, "style": name, "runs": samples,
            "median_s": {key: median(sample[key] for sample in samples)
                         for key in ("prepare_s", "draw_s", "png_s")}})
        figure.clear()
    return measurements


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--kind", choices=['scatter', 'mpl:scatter'], default='scatter')
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 1 for size in args.sizes):
        parser.error("Sizes and repeats must be positive")
    result = {"platform": platform.platform(), "processor": platform.processor(),
              "python": platform.python_version(), "matplotlib": matplotlib.__version__,
              "preview_dpi": 96, "export_dpi": 150, "categories": 12, "kind": args.kind,
              "note": "Synthetic table-only project, no MeasurementStore population; input creation excluded. "
                      "Agg, no Qt loop, no RSS, no filesystem PNG write. Three runs are not a p95 estimate.",
              "measurements": []}
    for size in args.sizes:
        for measured in measure(size, args.repeats, args.kind):
            result["measurements"].append(measured)
            print(json.dumps({key: measured[key] for key in ("rows", "style", "median_s")}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
