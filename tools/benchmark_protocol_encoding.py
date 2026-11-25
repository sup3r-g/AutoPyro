"""Isolate temporary encoding allocations; the prepared input is outside tracing.

python tools/benchmark_protocol_encoding.py --rows 30000 --memory [--legacy]
Legacy controls reproduce the former whole-string checksum/archive implementations.
"""
import argparse
import base64
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import tracemalloc
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autopyro import audit
from autopyro.interpolation import _checksum, prepare, replay
from tools.benchmark_interpolation_storage import make_project


def legacy_checksum(protocol):
    payload = {key: value for key, value in protocol.items() if key != "checksum"}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def legacy_archive(protocol):
    if type(protocol) is not dict or not audit._plain_json(protocol):
        return protocol
    try:
        raw = json.dumps(protocol, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, RecursionError):
        return protocol
    if not audit.MIN_PROTOCOL_BYTES <= len(raw) <= audit.MAX_PROTOCOL_BYTES:
        return protocol
    payload = base64.b64encode(zlib.compress(raw, level=1)).decode("ascii")
    stored = f"{audit.PREFIX}{len(raw)}:{hashlib.sha256(raw).hexdigest()}:{payload}"
    return stored if len(stored) < len(raw) else protocol


def measure(function, protocol, memory):
    gc.collect()
    if memory:
        tracemalloc.start()
    started = perf_counter()
    result = function(protocol)
    metrics = {"seconds": perf_counter() - started}
    if memory:
        metrics["retained_python_bytes"], metrics["peak_python_bytes"] = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return result, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    if args.rows < 10:
        parser.error("--rows must be at least 10")
    project = make_project(args.rows)
    project.sync_measurements()
    well, x, y = project.columns
    proposals, protocol = prepare(project, x.id, y.id, [well.id], set(project.row_map()), method="pchip",
                                   constraints={"well_id": well.id, "analytical_context": True})
    checksum = legacy_checksum if args.legacy else _checksum
    archive = legacy_archive if args.legacy else audit.archive_protocol
    digest, checksum_metrics = measure(checksum, protocol, args.memory)
    stored, archive_metrics = measure(archive, protocol, args.memory)
    assert digest == protocol["checksum"]
    assert audit.read_protocol(stored) == protocol
    assert replay(stored) == proposals
    print(json.dumps({"rows": args.rows, "legacy": args.legacy, "tracemalloc": args.memory,
                      "python": platform.python_version(), "environment": protocol["environment"],
                      "checksum": checksum_metrics, "archive": archive_metrics,
                      "stored_bytes": len(stored) if isinstance(stored, str) else None}, indent=2))


if __name__ == "__main__":
    main()
