"""Byte compatibility protects old protocol checksums and archive framing."""
import base64
from copy import deepcopy
import hashlib
import json
import zlib

import pytest

from autopyro import audit
from autopyro.interpolation import _checksum
from autopyro.protocol_json import protocol_json_chunks


@pytest.fixture
def protocol():
    sample = {"qualifiers": ["ND", "<0.1", "≈3", None], "large_int": 2**70 + 1,
              "unicode": 'Лаборатория 🌍\n\t"\\\x00', "numeric": [-0.0, 1e-100, 1e100, True, False]}
    return {"z-last": None, "rows": [deepcopy(sample) for _ in range(2001)],
            "records": {f"m-{i:04}": deepcopy(sample) for i in reversed(range(2001))},
            "empty": {}, "tuple_extension": (1, [2, 3]), "a-first": "001"}


@pytest.mark.parametrize("sort_keys", [False, True])
@pytest.mark.parametrize("batch_size", [1, 17, 1000, 5000])
def test_exact_bytes_match_standard_encoder_without_mutation(protocol, sort_keys, batch_size):
    before = deepcopy(protocol)
    expected = json.dumps(protocol, ensure_ascii=False, allow_nan=False, sort_keys=sort_keys,
                          separators=(",", ":")).encode("utf-8")
    chunks = list(protocol_json_chunks(protocol, sort_keys=sort_keys, batch_size=batch_size))
    assert b"".join(chunks) == expected
    assert protocol == before
    if batch_size <= 1000:
        assert max(map(len, chunks)) < len(expected) // 2


def test_checksum_matches_existing_sorted_json_including_ignored_checksum(protocol):
    protocol["checksum"] = "old value"
    payload = {k: v for k, v in protocol.items() if k != "checksum"}
    expected = hashlib.sha256(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()
    assert _checksum(protocol) == expected


def test_archive_has_identical_raw_bytes_digest_order_and_old_reader_compatibility(protocol, monkeypatch):
    protocol.pop("tuple_extension")  # Only strict plain JSON is eligible for archiving.
    raw = json.dumps(protocol, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    archived = audit.archive_protocol(protocol)
    assert isinstance(archived, str)
    length, digest, encoded = archived[len(audit.PREFIX):].split(":", 2)
    assert int(length) == len(raw)
    assert digest == hashlib.sha256(raw).hexdigest()
    assert zlib.decompress(base64.b64decode(encoded)) == raw
    # Archives produced by the former whole-string encoder remain readable.
    legacy = f"{audit.PREFIX}{len(raw)}:{digest}:{base64.b64encode(zlib.compress(raw, 1)).decode('ascii')}"
    assert audit.read_protocol(legacy) == audit.read_protocol(archived) == protocol
    assert list(audit.read_protocol(archived)) == list(protocol)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), object()])
def test_invalid_values_remain_rejected_by_checksum_and_unarchived(bad):
    value = {"rows": [{"value": bad}] * 1001}
    with pytest.raises((ValueError, TypeError)):
        _checksum(value)
    assert audit.archive_protocol(value) is value


def test_oversized_archive_stops_encoding_before_later_records(monkeypatch):
    import autopyro.audit as module

    seen = []
    def chunks(_):
        seen.append("first")
        yield b"x" * 100
        raise AssertionError("encoding must stop at the size limit")

    monkeypatch.setattr(module, "MAX_PROTOCOL_BYTES", 50)
    monkeypatch.setattr(module, "protocol_json_chunks", chunks)
    protocol = {"rows": []}
    assert module.archive_protocol(protocol) is protocol
    assert seen == ["first"]


def test_circular_or_extended_json_keeps_legacy_fallback():
    cyclic = {"rows": []}
    cyclic["rows"].append(cyclic)
    assert audit.archive_protocol(cyclic) is cyclic
    extended = {"tuple": (1, 2)}
    assert audit.archive_protocol(extended) is extended


def test_empty_protocol_and_container_extensions_keep_json_behavior():
    class ExtendedList(list):
        pass

    protocol = {"empty_list": [], "empty_dict": {}, "extension": ExtendedList(range(1002))}
    assert b"".join(protocol_json_chunks({})) == b"{}"
    assert b"".join(protocol_json_chunks(protocol)) == json.dumps(
        protocol, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def test_archive_limits_count_utf8_bytes_at_exact_boundaries(monkeypatch):
    protocol = {"label": "Пробы 🌍" * 1000}
    size = len(json.dumps(protocol, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8"))
    monkeypatch.setattr(audit, "MIN_PROTOCOL_BYTES", size)
    monkeypatch.setattr(audit, "MAX_PROTOCOL_BYTES", size)
    archived = audit.archive_protocol(protocol)
    assert isinstance(archived, str)
    assert audit.read_protocol(archived) == protocol
    monkeypatch.setattr(audit, "MIN_PROTOCOL_BYTES", size + 1)
    assert audit.archive_protocol(protocol) is protocol
    monkeypatch.setattr(audit, "MIN_PROTOCOL_BYTES", 0)
    monkeypatch.setattr(audit, "MAX_PROTOCOL_BYTES", size - 1)
    assert audit.archive_protocol(protocol) is protocol
