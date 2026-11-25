"""Numeric preview buffers roundtrip writable without entering pickle metadata."""
import multiprocessing
from threading import Thread
from types import SimpleNamespace

import numpy as np
import pytest

from autopyro.plot_preview import send_preview_response, receive_preview_response
from autopyro.plot_preview import pack_preview_membership, restore_preview_membership
from autopyro.plot_preview import (
    send_preview_message, receive_preview_message, preview_input, restore_preview_input,
)
from autopyro.selection_overlay import SelectionIndex


def roundtrip(value, sender=send_preview_response, receiver=receive_preview_response):
    reader, writer = multiprocessing.Pipe(duplex=False)
    errors = []
    def send():
        try:
            sender(writer, value)
        except Exception as exc:
            errors.append(exc)
        finally:
            writer.close()
    worker = Thread(target=send)
    worker.start()
    try:
        result = receiver(reader)
    finally:
        reader.close()
        worker.join(10)
    assert not worker.is_alive() and not errors
    return result


def test_numeric_storage_and_aliases_roundtrip_without_metadata_copy():
    data = np.arange(60000, dtype=float).reshape(-1, 3)
    value, stats = roundtrip((data, data, np.arange(100, dtype=np.int32)))
    assert value[0] is value[1]
    np.testing.assert_array_equal(value[0], data)
    value[0][0, 0] = -1
    assert value[1][0, 0] == -1 and data[0, 0] == 0
    assert stats['response_buffers'] == 1  # The small integer array stays inline.
    assert stats['response_metadata_bytes'] < data.nbytes / 10


def test_readonly_strided_and_object_arrays_preserve_values():
    readonly = np.arange(5000, dtype=np.float32)
    readonly.flags.writeable = False
    masked = np.ma.array([1., 2., 3.], mask=[False, True, False])
    values = (readonly, readonly[::2], np.asarray(['a\0', 'текст 🌋', None], dtype=object), masked)
    actual, _ = roundtrip(values)
    for expected, restored in zip(values, actual):
        np.testing.assert_array_equal(restored, expected)
    assert not actual[0].flags.writeable
    np.testing.assert_array_equal(np.ma.getmaskarray(actual[3]), np.ma.getmaskarray(masked))


@pytest.mark.parametrize('value', [(2, None, None), (3, None, 'Ошибка'), (4, np.empty((0, 2)), None)])
def test_empty_error_and_cancelled_responses(value):
    actual, stats = roundtrip(value)
    assert actual[0] == value[0] and actual[2] == value[2]
    if value[1] is not None:
        np.testing.assert_array_equal(actual[1], value[1])
    assert stats['decode_s'] >= 0


def test_membership_reuses_input_ids_and_keeps_multi_track_duplicates():
    import pandas as pd
    ids = pd.Index(['a\0', 'a', 'текст 🌋', 'x' * 200], dtype=object)
    artists = [SimpleNamespace(_autopyro_row_ids=[ids[1], ids[0]]),
               SimpleNamespace(_autopyro_row_ids=[ids[2], ids[1]])]
    index = SelectionIndex.from_parts([(np.asarray([ids[1], ids[0], ids[2], ids[1]], dtype=object),
        np.asarray([0, 0, 1, 1]), np.asarray([0, 1, 0, 1]))])
    result = {'artists': artists, '_selection_overlay': ([], index)}
    expected = [artist._autopyro_row_ids.copy() for artist in artists]
    pack_preview_membership(result, ids)
    assert all(not artist._autopyro_row_ids for artist in artists)
    restored, _ = roundtrip(result)
    restore_preview_membership(restored, ids.to_numpy(dtype=object), {rid: i for i, rid in enumerate(ids)})
    assert [artist._autopyro_row_ids for artist in restored['artists']] == expected
    selected, positions = restored['_selection_overlay'][1].lookup(['a'], 2)
    assert selected == {'a'}
    assert [p.tolist() for p in positions] == [[0], [1]]


def test_membership_rejects_an_id_outside_the_input_scope():
    import pandas as pd
    result = {'artists': [SimpleNamespace(_autopyro_row_ids=['missing'])],
              '_selection_overlay': ([], SelectionIndex.from_parts([]))}
    with pytest.raises(ValueError, match='UUID'):
        pack_preview_membership(result, pd.Index(['present'], dtype=object))
    assert result['artists'][0]._autopyro_row_ids == ['missing']


def test_preview_builds_selection_on_packed_row_positions():
    import pandas as pd
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from autopyro.visualization.selection_overlay import SelectionOverlay

    ids = pd.Index(['first', 'second', 'third'])
    figure = Figure()
    canvas = FigureCanvasAgg(figure)
    artists = [ax.scatter([1, 2], [3, 4]) for ax in figure.subplots(1, 2)]
    expected = [['second', 'first'], ['third', 'second']]
    for artist, members in zip(artists, expected, strict=True):
        artist._autopyro_row_ids = members
    result = {'artists': artists}
    pack_preview_membership(result, ids)
    overlay = SelectionOverlay(canvas, result, compact=True)
    result['_selection_overlay'] = overlay.layers, overlay.locations
    assert overlay.locations.keys.dtype.kind == 'i'
    restore_preview_membership(result, ids.to_numpy(), {rid: i for i, rid in enumerate(ids)})
    assert [artist._autopyro_row_ids for artist in artists] == expected
    selected, positions = overlay.locations.lookup(['second'], 2)
    assert selected == {'second'}
    assert [part.tolist() for part in positions] == [[0], [1]]


def prepared_input(scope=None):
    from autopyro.core import Column, Project, Row
    from autopyro.plot_data import PlotDataCache, capture_input, input_token
    from autopyro.plotting import PlotSpec
    x, y, group = Column('X', 'number'), Column('Y', 'number'), Column('Group')
    ids = ['a\0', 'текст 🌋', 'x' * 200, 'last']
    project = Project(columns=[x, y, group], rows=[
        Row({x.id: value, y.id: value, group.id: str(n % 2)}, id=rid)
        for n, (rid, value) in enumerate(zip(ids, [2 ** 100, '<0.1', None, 1.25], strict=True))])
    spec = PlotSpec(x_id=x.id, y_id=y.id, color_by=x.id, color_mode='numeric', shape_by=group.id)
    source = capture_input(project, ids, [spec], input_token(project, 0, [spec]))
    return PlotDataCache().prepare(source, ids if scope is None else scope, spec)


@pytest.mark.parametrize('index_dtype', ['object', 'str'])
def test_input_shares_equal_indexes_without_mutating_cached_frames(index_dtype):
    import pandas as pd
    prepared = prepared_input()
    frames = [prepared.local, prepared.global_frame, *prepared.parsed.values()]
    for frame in frames:
        frame.index = pd.Index(frame.index.tolist(), dtype=index_dtype)
    original_indexes = [frame.index for frame in frames]
    small = preview_input(prepared)
    transferred = [small.local, small.global_frame, *small.parsed.values()]
    assert all(frame.index is small.local.index for frame in transferred)
    assert all(frame.index is index for frame, index in zip(frames, original_indexes, strict=True))
    assert len(prepared.project.rows) == 4 and small.project.rows == []
    for before, after in zip(frames, transferred, strict=True):
        pd.testing.assert_frame_equal(before, after)


@pytest.mark.parametrize('scope', [None, ['a\0', 'last'], [], ['last', 'absent']])
def test_input_pipe_preserves_scopes_classifications_precision_and_uuid_order(scope):
    import pandas as pd
    prepared = prepared_input(scope)
    small = preview_input(prepared)
    actual, stats = roundtrip(small, send_preview_message, receive_preview_message)
    assert actual.scope == prepared.scope.difference(prepared.local.index)
    restore_preview_input(actual)
    assert actual.scope == prepared.scope and actual.category_values == prepared.category_values
    assert [row.id for row in actual.project.rows] == prepared.local.index.tolist()
    pd.testing.assert_frame_equal(actual.local, prepared.local)
    pd.testing.assert_frame_equal(actual.global_frame, prepared.global_frame)
    for key in prepared.parsed:
        pd.testing.assert_frame_equal(actual.parsed[key], prepared.parsed[key])
    if scope is None:
        assert actual.local.iloc[0, actual.local.columns.get_loc(prepared.project.columns[0].id)] == 2 ** 100
        assert all(frame.index is actual.local.index for frame in actual.parsed.values())
    else:
        assert actual.local.index is not actual.global_frame.index
    assert stats['message_bytes'] >= stats['metadata_bytes']


@pytest.mark.parametrize('difference', ['name', 'order', 'dtype'])
def test_input_does_not_merge_distinct_indexes(difference):
    import pandas as pd
    prepared = prepared_input()
    if difference == 'name':
        prepared.global_frame.index = prepared.global_frame.index.rename('global rows')
    elif difference == 'order':
        prepared.global_frame = prepared.global_frame.iloc[::-1]
    else:
        prepared.global_frame.index = prepared.global_frame.index.astype(object)
    small = preview_input(prepared)
    assert small.local.index is not small.global_frame.index
    pd.testing.assert_frame_equal(small.global_frame, prepared.global_frame)


def test_request_buffers_preserve_aliases_and_control_messages():
    data = np.arange(60000, dtype=np.int64)
    actual, stats = roundtrip((data, data), send_preview_message, receive_preview_message)
    assert actual[0] is actual[1]
    np.testing.assert_array_equal(actual[0], data)
    assert stats['buffers'] == 1 and stats['metadata_bytes'] < 1024
    actual[0][0] = -1
    assert data[0] == 0  # Receiving storage belongs to the other process.
    actual, stats = roundtrip(None, send_preview_message, receive_preview_message)
    assert actual is None and stats['buffers'] == 0


@pytest.mark.parametrize('layout', ['C', 'F', 'strided'])
def test_cooperative_request_preserves_object_cells_aliases_and_layout(layout):
    from functools import partial
    from time import sleep
    data = np.empty((400, 30), dtype=object, order='F' if layout == 'F' else 'C')
    data[:] = 2 ** 100
    data[0, 0], data[0, 1], data[0, 2] = ['nested', None], {'x': '<0.1'}, np.arange(3)
    if layout == 'strided':
        data = data[::2]
    data.flags.writeable = False
    yields = []
    def cooperate():
        yields.append(1)
        sleep(0)
    sender = partial(send_preview_message, cooperate=cooperate)
    actual, _ = roundtrip((data, data), sender, receive_preview_message)
    assert actual[0] is actual[1] and not actual[0].flags.writeable
    assert actual[0].shape == data.shape and actual[0].dtype == object
    assert actual[0][0, 0] == ['nested', None] and actual[0][0, 1] == {'x': '<0.1'}
    np.testing.assert_array_equal(actual[0][0, 2], np.arange(3))
    assert actual[0][-1, -1] == 2 ** 100 and len(yields) > data.size // 4096
    if layout == 'F':
        assert actual[0].flags.f_contiguous


def test_cooperative_request_keeps_cycles_in_object_arrays():
    from functools import partial
    data = np.empty(5000, dtype=object)
    data[0] = data
    data[1] = {'again': data}
    actual, _ = roundtrip(data, partial(send_preview_message, cooperate=lambda: None), receive_preview_message)
    assert actual[0] is actual and actual[1]['again'] is actual


def test_cooperative_request_restores_large_dataframe_blocks_exactly():
    import pandas as pd
    from functools import partial
    frame = pd.DataFrame({'number': [2 ** 100, None, '<0.1', 1.25] * 1500,
                          'status': ['numeric', 'missing', 'censored', 'numeric'] * 1500}, dtype=object)
    frame.index = pd.Index(['sample-' + str(n) for n in range(len(frame))], name='UUID')
    actual, _ = roundtrip(frame, partial(send_preview_message, cooperate=lambda: None), receive_preview_message)
    pd.testing.assert_frame_equal(actual, frame)
