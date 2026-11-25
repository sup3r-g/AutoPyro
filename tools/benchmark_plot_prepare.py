"""Profile plot preparation without canvas drawing or Qt event scheduling."""
import argparse
import cProfile
import io
import json
import platform
import pstats
import time
from pathlib import Path

from matplotlib.figure import Figure

from autopyro.plotting import PlotSpec, render_plot
from tools.benchmark_dependent_edit import input_project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = input_project(args.size)
    columns = {column.name: column.id for column in project.columns}
    ids = [row.id for row in project.rows]
    profile = cProfile.Profile()
    runs = {}
    for name, style in (("uniform", {}), ("numeric", {
        "color_by": columns["Depth"], "color_mode": "numeric",
        "size_by": columns["TOC"], "size_mode": "numeric",
    })):
        spec = PlotSpec(x_id=columns["Depth"], y_id=columns["HI"], **style)
        figure = Figure()
        durations = []
        for _ in range(3):
            started = time.perf_counter()
            _, result = render_plot(project, ids, spec, figure=figure)
            durations.append(time.perf_counter() - started)
            assert result["plotted"][columns["Depth"]] == args.size
        runs[name] = durations
        profile.runcall(render_plot, project, ids, spec, figure=figure)
        figure.clear()
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(30)
    result = {"platform": platform.platform(), "rows": args.size, "seconds": runs,
              "profile": stream.getvalue(),
              "note": "Three unprofiled runs per style, reused Figure, no canvas draw. Separate cumulative profile includes both styles."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"seconds": runs, "profile": stream.getvalue()}), flush=True)


if __name__ == "__main__":
    main()
