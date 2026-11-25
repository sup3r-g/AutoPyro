"""Separate snapshot copying, JSON encoding and hashing with a Qt heartbeat."""
import argparse
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys

from tools.benchmark_import_refresh import measure, QApplication
from autopyro.io import build_project
from autopyro.importing import digest
from autopyro.snapshots import copy_project


class RecordEncoder(json.JSONEncoder):
    def default(self, value):
        if is_dataclass(value) and not isinstance(value, type):
            return {field.name: getattr(value, field.name) for field in fields(value)}
        return super().default(value)


def legacy_digest(project):
    return hashlib.sha256(json.dumps(asdict(project), ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def stream_digest(project):
    encoder = RecordEncoder(ensure_ascii=False, sort_keys=True, allow_nan=False)
    checksum = hashlib.sha256()
    chunks, length = [], 0
    for chunk in encoder.iterencode(project):
        chunks.append(chunk)
        length += len(chunk)
        if length >= 65536:
            checksum.update("".join(chunks).encode("utf-8"))
            chunks, length = [], 0
    checksum.update("".join(chunks).encode("utf-8"))
    return checksum.hexdigest()


def benchmark(size, *, memory=False, copy_only=False):
    project = build_project(["Sample", "Well", "TOC", "S2"],
        [[f"{i:07d}", f"W{i % 20}", "2", "ND" if i % 20 == 0 else "6"] for i in range(size)],
        list(range(2, size + 2)), [(0, "Sample", "text", ""), (1, "Well", "text", ""),
        (2, "TOC", "number", "%wt"), (3, "S2", "number", "mg HC/g rock")], {"sha256": "synthetic"})
    project.sync_measurements()
    results = []

    def stage(name, function):
        value, result = measure(name, function, memory=memory)
        results.append(result)
        print(json.dumps({"rows": size, **result}), flush=True)
        return value

    copied = stage("deepcopy", lambda: deepcopy(project))
    assert len(copied.science.measurements) == size * 4
    del copied
    copied = stage("copy_project", lambda: copy_project(project))
    assert digest(copied) == digest(project)
    del copied
    if copy_only:
        return {"rows": size, "stages": results}
    payload = stage("asdict", lambda: asdict(project))
    serialized = stage("json_dumps", lambda: json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    expected = stage("utf8_sha256", lambda: hashlib.sha256(serialized.encode("utf-8")).hexdigest())
    del payload, serialized
    assert stage("legacy_digest", lambda: legacy_digest(project)) == expected
    assert stage("current_digest", lambda: digest(project)) == expected
    assert stage("stream_digest", lambda: stream_digest(project)) == expected
    return {"rows": size, "stages": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000])
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--copy-only", action="store_true", help="Measure only the two copying strategies")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    result = {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
              "memory_instrumentation": args.memory, "copy_only": args.copy_only, "heartbeat_interval_ms": 10,
              "note": "Synthetic four-column project. Separate worker per stage, one pass, input construction excluded. Python allocations exclude pre-existing inputs and native memory.",
              "measurements": []}
    for size in args.sizes:
        result["measurements"].append(benchmark(size, memory=args.memory, copy_only=args.copy_only))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    app.processEvents()


if __name__ == "__main__":
    main()
