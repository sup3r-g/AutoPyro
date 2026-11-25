"""Compressed in-memory undo states; payloads are created here, never imported."""
from dataclasses import dataclass, field
from io import BytesIO
import pickle
import zlib

from autopyro.model.core import Column, Project, Row
from autopyro.model.measurements import Analysis, Measurement, MeasurementStore, PropertyDefinition, Sample, SourceRecord
from autopyro.model.snapshots import copy_table_state


MIN_PACK_BYTES = 1024 * 1024
_RECORD_TYPES = frozenset((Column, Project, Row, Analysis, Measurement, MeasurementStore,
                         PropertyDefinition, Sample, SourceRecord))


class _StatePickler(pickle.Pickler):
    def reducer_override(self, value):
        # Preserve custom extension copy semantics via the ordinary snapshot path.
        if isinstance(value, type) and value in _RECORD_TYPES:
            return NotImplemented
        if type(value) not in _RECORD_TYPES or hasattr(value, "__deepcopy__"):
            raise pickle.PicklingError("Undo packing supports plain application records")
        return NotImplemented


@dataclass(frozen=True)
class PackedTableState:
    payload: bytes = field(repr=False)
    cell_count: int
    measurement_count: int

    def restore(self):
        # Only private bytes produced by pack_table_state enter this decoder.
        return pickle.loads(zlib.decompress(self.payload))


def pack_table_state(state):
    """Pack an already independent snapshot; retain plain data when unsuitable."""
    try:
        stream = BytesIO()
        _StatePickler(stream, protocol=pickle.HIGHEST_PROTOCOL).dump(state)
        payload = stream.getvalue()
    except (pickle.PicklingError, AttributeError, TypeError, ValueError, RecursionError):
        # Extensions can support deepcopy without supporting pickle (e.g. local classes).
        return state
    if len(payload) < MIN_PACK_BYTES:
        return state
    compressed = zlib.compress(payload, level=1)
    if len(compressed) >= len(payload):
        return state
    return PackedTableState(compressed, len(state[0]) * len(state[1]),
                            len(state[2].measurements))


def restore_table_state(state):
    if isinstance(state, PackedTableState):
        return state.restore()
    return copy_table_state(state)
