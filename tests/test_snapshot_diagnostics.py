"""Diagnostic counters report shallow sizes and GC spans without changing policy."""
import sys

from autopyro.measurements import Measurement
from tools import benchmark_snapshot_parts as benchmark


def test_gc_observations_count_only_matched_spans(monkeypatch):
    ticks = iter([10., 10.25, 11., 11.5])
    monkeypatch.setattr(benchmark, "perf_counter", lambda: next(ticks))
    observed = benchmark.GCObservations()
    observed.callback("stop", {"generation": 2, "collected": 99})
    for count in (3, 5):
        observed.callback("start", {"generation": 1})
        observed.callback("stop", {"generation": 1, "collected": count})
    assert observed.generations[1] == {"count": 2, "seconds": .75,
                                        "max_seconds": .5, "collected": 8}
    assert observed.generations[2]["count"] == 0 and not observed.started


def test_memo_sizes_account_for_containers_without_traversing_references():
    record = Measurement("sample", "property", "analysis")
    memo = {1: record, 2: record.history}
    result = benchmark.memo_sizes(memo)
    assert result["entries"] == 2
    assert result["dict_bytes"] == sys.getsizeof(memo)
    assert result["key_bytes"] == sys.getsizeof(1) + sys.getsizeof(2)
    assert result["measurement_attribute_dict_bytes"] == sys.getsizeof(vars(record))
    assert result["values"]["Measurement"]["shallow_bytes"] == sys.getsizeof(record)
    record.history.append({"large": list(range(1000))})
    assert benchmark.memo_sizes(memo)["measurement_attribute_dict_bytes"] == result["measurement_attribute_dict_bytes"]
