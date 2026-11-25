"""Encode protocol collections in batches without building one full JSON string."""
from itertools import batched
import json


def _collection_chunks(value, encoder, batch_size):
    # Only exact built-in containers are batched; extensions retain JSON's own
    # encoding behavior. A batch still uses the fast standard C encoder.
    if type(value) is dict and len(value) > batch_size:
        yield b"{"
        items = sorted(value.items()) if encoder.sort_keys else value.items()
        for index, batch in enumerate(batched(items, batch_size)):
            if index:
                yield b","
            yield encoder.encode(dict(batch))[1:-1].encode("utf-8")
        yield b"}"
    elif type(value) is list and len(value) > batch_size:
        yield b"["
        for index, batch in enumerate(batched(value, batch_size)):
            if index:
                yield b","
            yield encoder.encode(batch)[1:-1].encode("utf-8")
        yield b"]"
    else:
        yield encoder.encode(value).encode("utf-8")


def protocol_json_chunks(protocol, *, sort_keys=False, batch_size=1000):
    """Yield the same UTF-8 bytes as compact json.dumps for a protocol object.

    Protocols have string root keys and large row/record collections directly
    beneath the root. Batching these collections avoids whole-document copies.
    This bounds record count per batch, not the size of an individual record.
    """
    if type(protocol) is not dict or any(type(key) is not str for key in protocol):
        raise TypeError("Protocol root must be a dict with string keys")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, sort_keys=sort_keys,
                               separators=(",", ":"))
    items = sorted(protocol.items()) if sort_keys else protocol.items()
    yield b"{"
    for index, (key, value) in enumerate(items):
        if index:
            yield b","
        yield encoder.encode(key).encode("utf-8")
        yield b":"
        yield from _collection_chunks(value, encoder, batch_size)
    yield b"}"
