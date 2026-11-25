"""Compare interpolation with and without a reproducible snapshot on synthetic data.

Run with the project interpreter: python tools/benchmark_interpolation_protocol.py --rows 30000
Timings include warm scientific imports and exclude fixture construction.
"""
import argparse
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autopyro.interpolation import prepare, propose, replay
from autopyro.audit import archive_protocol
from autopyro.io import build_project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--method", choices=("linear", "nearest", "pchip"), default="pchip")
    parser.add_argument("--restricted", action="store_true", help="Partition by explicit wellbore and analytical context")
    parser.add_argument("--archive", action="store_true", help="Replay through the archived storage representation")
    args = parser.parse_args()
    if args.rows < 10:
        parser.error("--rows must be at least 10")
    records = [[f"Well {i // 1000}", str(i % 1000),
                "" if i % 5 == 2 else str((i % 1000) ** 1.2)] for i in range(args.rows)]
    project = build_project(["Well", "Depth", "Y"], records, list(range(2, args.rows + 2)),
                            [(0, "Well", "text", ""), (1, "Depth", "number", "m"),
                             (2, "Y", "number", "mg/g")], {"type": "synthetic benchmark"})
    project.sync_measurements()
    group, x, y = project.columns
    options = dict(x_id=x.id, y_id=y.id, group_ids=[group.id],
                   row_ids=set(project.row_map()), method=args.method, max_gap=10)
    if args.restricted:
        options["constraints"] = {"well_id": group.id, "analytical_context": True}
    propose(project, **options)  # Warm lazy scientific imports before the timed passes.
    started = perf_counter()
    expected = propose(project, **options)
    propose_seconds = perf_counter() - started
    started = perf_counter()
    proposals, protocol = prepare(project, **options)
    prepare_seconds = perf_counter() - started
    started = perf_counter()
    stored = archive_protocol(protocol) if args.archive else protocol
    archive_seconds = perf_counter() - started
    started = perf_counter()
    reproduced = replay(stored)
    replay_seconds = perf_counter() - started
    assert proposals == reproduced == expected
    print(json.dumps({"python": platform.python_version(), "platform": platform.platform(),
                      "environment": protocol["environment"], "rows": args.rows,
                      "measurements": len(project.science.measurements), "method": args.method,
                      "restricted": args.restricted,
                      "proposals": len(proposals), "propose_seconds": propose_seconds,
                      "prepare_seconds": prepare_seconds, "replay_seconds": replay_seconds,
                      "archive_seconds": archive_seconds, "archived": isinstance(stored, str),
                      "stored_protocol_bytes": len(stored.encode("utf-8")) if isinstance(stored, str) else None,
                      "protocol_bytes": len(json.dumps(protocol, ensure_ascii=False,
                                                        separators=(",", ":")).encode("utf-8"))}, indent=2))


if __name__ == "__main__":
    main()
