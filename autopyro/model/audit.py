"""Versioned JSON protocol snapshots for history and saved calculations."""
import base64
import binascii
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import zlib

from .protocol_json import protocol_json_chunks

MIN_PROTOCOL_BYTES = 64 * 1024
MAX_PROTOCOL_BYTES = 256 * 1024 * 1024
MAGIC = "autopyro:protocol:"
PREFIX = MAGIC + "zlib-json:1:"
_SCALARS = (str, int, float, bool, type(None))


def _plain_json(value, *, checkpoint=None):
    pending, seen = [value], set()
    visited = 0
    while pending:
        if checkpoint:
            if visited % 4096 == 0:
                checkpoint()
            visited += 1
        item = pending.pop()
        kind = type(item)
        if kind in _SCALARS:
            continue
        if kind not in (dict, list):
            return False
        if id(item) in seen:
            continue
        seen.add(id(item))
        if kind is dict:
            if any(type(key) is not str for key in item):
                return False
            pending.extend(item.values())
        else:
            pending.extend(item)
    return True


def archive_protocol(protocol, *, checkpoint=None):
    """Consume a private protocol; use immutable text for large plain JSON data."""
    if type(protocol) is not dict or not _plain_json(protocol, checkpoint=checkpoint):
        return protocol
    digest, size = hashlib.sha256(), 0
    compressor = zlib.compressobj(level=1)
    try:
        with BytesIO() as compressed:
            for chunk in protocol_json_chunks(protocol):
                if checkpoint:
                    checkpoint()
                size += len(chunk)
                if size > MAX_PROTOCOL_BYTES:
                    return protocol
                digest.update(chunk)
                compressed.write(compressor.compress(chunk))
            if size < MIN_PROTOCOL_BYTES:
                return protocol
            compressed.write(compressor.flush())
            payload = base64.b64encode(compressed.getvalue()).decode("ascii")
    except (ValueError, TypeError, RecursionError):
        return protocol
    snapshot = f"{PREFIX}{size}:{digest.hexdigest()}:{payload}"
    return snapshot if len(snapshot) < size else protocol


def read_protocol(value, *, checkpoint=None):
    """Return independent details from either an old inline or archived protocol."""
    if checkpoint:
        checkpoint()
    if type(value) is not str or not value.startswith(MAGIC):
        return deepcopy(value)
    if not value.startswith(PREFIX):
        raise ValueError("Неизвестная версия протокола журнала")
    try:
        length, expected, encoded = value[len(PREFIX):].split(":", 2)
        size = int(length)
        if not 0 < size <= MAX_PROTOCOL_BYTES or len(encoded) > (MAX_PROTOCOL_BYTES + 2) // 3 * 4:
            raise ValueError("Недопустимый размер протокола")
        decoder = zlib.decompressobj()
        compressed = base64.b64decode(encoded, validate=True)
        if checkpoint:
            checkpoint()
        raw = decoder.decompress(compressed, size + 1)
        del compressed
        if checkpoint:
            checkpoint()
        if (len(raw) != size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail
                or hashlib.sha256(raw).hexdigest() != expected):
            raise ValueError("Контрольная сумма или размер не совпадают")
        if checkpoint:
            checkpoint()
        result = json.loads(raw)
        if checkpoint:
            checkpoint()
        if type(result) is not dict:
            raise ValueError("Протокол должен быть объектом")
        return result
    except (ValueError, binascii.Error, zlib.error, UnicodeError) as exc:
        raise ValueError("Повреждён протокол журнала") from exc


def share_history_protocols(history):
    """Validate each unique archived payload once and share its immutable text."""
    snapshots = {}
    for event in history:
        if not isinstance(event, dict):
            continue
        value = event.get("protocol")
        if type(value) is str and value.startswith(MAGIC):
            if value not in snapshots:
                read_protocol(value)
                snapshots[value] = value
            event["protocol"] = snapshots[value]
