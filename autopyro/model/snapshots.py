"""Independent project/undo copies with fewer temporary measurement objects."""
from concurrent.futures import CancelledError
from copy import deepcopy
from itertools import batched

from .measurements import Measurement
from .work_batches import cooperative_items

_SCALAR_TYPES = (str, int, float, bool, type(None))
_SCALAR_TYPE_SET = frozenset(_SCALAR_TYPES)


def _check_cancelled(cancelled):
    if cancelled is not None and cancelled.is_set():
        raise CancelledError()


def _copy_items(items, cancelled):
    items = cooperative_items(items)
    if cancelled is None:
        return items

    def checked():
        _check_cancelled(cancelled)
        for batch in batched(items, 512):
            _check_cancelled(cancelled)
            yield from batch

    return checked()


def _memoize_scalar_mapping(mapping, memo, originals):
    """Own the container; only exact immutable built-ins may retain references."""
    if type(mapping) is not dict or id(mapping) in memo:
        return
    if (_SCALAR_TYPE_SET.issuperset(map(type, mapping))
            and _SCALAR_TYPE_SET.issuperset(map(type, mapping.values()))):
        memo[id(mapping)] = mapping.copy()
        # Match deepcopy's keep-alive convention: custom hooks can remove an
        # original container later, but its identity must not be recycled.
        originals.append(mapping)


def _table_memo(rows, store, memo, cancelled=None):
    """Pre-copy the common flat maps; all nested or extended values fall back."""
    originals = memo.setdefault(id(memo), [])
    for row in _copy_items(rows, cancelled):
        _memoize_scalar_mapping(row.values, memo, originals)
        _memoize_scalar_mapping(row.original, memo, originals)
    for source in _copy_items(store.sources.values(), cancelled):
        _memoize_scalar_mapping(source.original, memo, originals)
    for selected in _copy_items(store.working.values(), cancelled):
        _memoize_scalar_mapping(selected, memo, originals)


def _plain_measurements(store, cancelled=None):
    return [record for record in _copy_items(store.measurements.values(), cancelled)
            if type(record) is Measurement and not hasattr(record, "__deepcopy__")]


def _allocate_measurement_copies(records, cancelled=None):
    memo = {}
    # Register every destination first: histories may refer to this or another result.
    for record in _copy_items(records, cancelled):
        memo[id(record)] = object.__new__(Measurement)
    return memo


def _copy_measurement_history(history, memo, originals):
    """Copy revision entries in order, with a shortcut for flat previous states."""
    if type(history) is not list or id(history) in memo:
        return deepcopy(history, memo)
    copied = []
    memo[id(history)] = copied
    for event in history:
        # An ordinary revision has only a timestamp and its previous state.
        # Other entries can contain hooks or references to later revisions.
        if (type(event) is dict and id(event) not in memo and len(event) == 2
                and all(type(key) is str for key in event)
                and type(event.get("at")) is str):
            _memoize_scalar_mapping(event.get("previous"), memo, originals)
        copied.append(deepcopy(event, memo))
    originals.append(history)
    return copied


def _measurement_memo(store, cancelled=None):
    """Pre-copy plain measurements; let deepcopy handle all other data normally."""
    records = _plain_measurements(store, cancelled)
    memo = _allocate_measurement_copies(records, cancelled)
    originals = memo.setdefault(id(memo), [])
    for record in _copy_items(records, cancelled):
        copied = memo[id(record)]
        for name, value in vars(record).items():
            if name == "history":
                value = _copy_measurement_history(value, memo, originals)
            elif type(value) not in _SCALAR_TYPE_SET:
                value = deepcopy(value, memo)
            # Writing declared fields directly lets CPython keep the destination
            # attributes inline until a caller actually requests its dictionary.
            if name in Measurement.__dataclass_fields__:
                setattr(copied, name, value)
            else:
                vars(copied)[name] = value
    return memo


def copy_project(project, *, cancelled=None):
    """Preserve the full object graph, including private project attributes."""
    # Existing synchronous callers and custom copy hooks retain their semantics.
    memo = (_measurement_memo(project.science) if cancelled is None
            else _measurement_memo(project.science, cancelled))
    _table_memo(project.rows, project.science, memo, cancelled)
    _check_cancelled(cancelled)
    result = deepcopy(project, memo)
    _check_cancelled(cancelled)
    return result


def copy_table_state(state):
    """Copy (columns, rows, science[, database_snapshot]) for an undo command."""
    memo = _measurement_memo(state[2])
    _table_memo(state[1], state[2], memo)
    return deepcopy(state, memo)
