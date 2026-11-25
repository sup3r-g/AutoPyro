"""Agg preview rendering in an isolated, reusable process."""
from copy import copy
from dataclasses import replace
from io import BytesIO
import pickle
from time import perf_counter

from .plot_export import install_palette


def _empty_object_array(shape, fortran):
    import numpy as np
    return np.empty(shape, dtype=object, order='F' if fortran else 'C')


def _fill_object_array(array, state):
    chunks, writeable = state
    # Assign cells individually: nested lists/arrays are cells, not new axes.
    position = 0
    flat = array.flat
    for chunk in chunks:
        for value in chunk:
            flat[position] = value
            position += 1
    array.flags.writeable = writeable


class _ObjectArrayChunks:
    """Pickle list-items lazily so each bounded batch can release the GIL."""
    def __init__(self, array, cooperate):
        self.array, self.cooperate = array, cooperate

    def __reduce_ex__(self, protocol):
        return list, (), None, self.chunks()

    def chunks(self):
        for start in range(0, self.array.size, 4096):
            self.cooperate()
            yield self.array.flat[start:start + 4096].tolist()


class _CooperativePickler(pickle.Pickler):
    """Keep object-valued DataFrame blocks from becoming one long C call."""
    def __init__(self, stream, *, cooperate, **options):
        super().__init__(stream, **options)
        self.cooperate = cooperate

    def reducer_override(self, value):
        import numpy as np
        self.cooperate()
        if type(value) is np.ndarray and value.dtype == object and value.size >= 4096:
            # A state setter memoizes the empty array before reading cells,
            # preserving aliases and cycles just like NumPy's native pickle.
            state = _ObjectArrayChunks(value, self.cooperate), value.flags.writeable
            return (_empty_object_array, (value.shape, value.flags.f_contiguous),
                    state, None, None, _fill_object_array)
        return NotImplemented


def send_preview_message(connection, value, buffer_type=BytesIO, *, cooperate=None):
    """Send metadata and large buffers separately in either pipe direction."""
    buffers = []
    def choose_buffer(buffer):
        # Keep tiny arrays in pickle; separate messages are useful for large
        # storage, not the many short arrays used by ticks and transforms.
        if buffer.raw().nbytes < 64 * 1024:
            return True
        buffers.append(buffer)
        return False
    started = perf_counter()
    with buffer_type() as metadata:
        options = {'protocol': 5, 'buffer_callback': choose_buffer}
        pickler = (pickle.Pickler(metadata, **options) if cooperate is None else
                   _CooperativePickler(metadata, cooperate=cooperate, **options))
        pickler.dump(value)
        del pickler  # Drop memoized temporary chunks before waiting on the pipe.
        raw = [buffer.raw() for buffer in buffers]
        metadata_bytes = metadata.tell()
        encode_s = perf_counter() - started
        started = perf_counter()
        connection.send([buffer.nbytes for buffer in raw])
        connection.send_bytes(metadata.getbuffer())
        for buffer in raw:
            connection.send_bytes(buffer)
    return {'encode_s': encode_s, 'send_s': perf_counter() - started,
            'message_bytes': metadata_bytes + sum(buffer.nbytes for buffer in raw),
            'metadata_bytes': metadata_bytes, 'buffers': len(raw)}


def receive_preview_message(connection, buffer_type=BytesIO):
    sizes = connection.recv()
    started = perf_counter()
    metadata = connection.recv_bytes()
    buffers = []
    for size in sizes:
        # Receive directly into writable storage retained by NumPy. No second
        # full-frame bytes object or conversion to a writable array is needed.
        data = bytearray(size)
        if connection.recv_bytes_into(data) != size:
            raise RuntimeError('Неполный буфер предпросмотра')
        buffers.append(data)
    receive_s = perf_counter() - started
    started = perf_counter()
    with buffer_type(metadata) as stream:
        value = pickle.Unpickler(stream, buffers=buffers).load()
    return value, {'decode_s': perf_counter() - started, 'receive_s': receive_s,
                   'message_bytes': len(metadata) + sum(sizes),
                   'metadata_bytes': len(metadata), 'buffers': len(sizes)}


def send_preview_response(connection, value):
    return send_preview_message(connection, value)


def receive_preview_response(connection, buffer_type=BytesIO):
    value, stats = receive_preview_message(connection, buffer_type)
    return value, {'decode_s': stats['decode_s'], 'receive_s': stats['receive_s'],
                   'response_bytes': stats['message_bytes'],
                   'response_metadata_bytes': stats['metadata_bytes'], 'response_buffers': stats['buffers']}


def pack_preview_membership(result, row_index):
    """Use positions in the immutable input index instead of resending UUIDs."""
    import numpy as np
    def positions(ids):
        found = row_index.get_indexer(ids)
        if np.any(found < 0):
            raise ValueError('UUID графика отсутствует во входном снимке')
        return found
    artists = result.get('artists', [])
    members = [positions(artist._autopyro_row_ids) for artist in artists]
    if '_selection_overlay' in result:
        index = result['_selection_overlay'][1]
        keys = positions(index.keys)
        order = keys.argsort(kind='stable')
        index.keys = keys[order]
        index.layers = index.layers[order]
        index.positions = index.positions[order]
    for artist in artists:
        artist._autopyro_row_ids = []
    result['_preview_membership'] = members


def restore_preview_membership(result, row_ids, row_lookup):
    """Restore UUID references before any frame or picking event reaches Qt."""
    members = result.pop('_preview_membership')
    for artist, positions in zip(result.get('artists', []), members, strict=True):
        artist._autopyro_row_ids = row_ids[positions].tolist()
    result['_selection_overlay'][1].row_lookup = row_lookup


def preview_input(prepared):
    """Pack frame data with the local index as the source of row UUIDs.

    Pandas may wrap the same row index in several objects. Pickle then sends
    their UUID/Arrow buffers repeatedly. Shallow copies leave cached frames
    untouched while making equal indexes a single object in the wire graph.
    On the wire, scope contains only UUIDs absent from the local index.
    """
    indexes = []
    def share_index(frame):
        for index in indexes:
            if frame.index.identical(index):
                if frame.index is index:
                    return frame
                shared = frame.copy(deep=False)
                shared.index = index
                return shared
        indexes.append(frame.index)
        return frame
    local = share_index(prepared.local)
    global_frame = share_index(prepared.global_frame)
    parsed = {key: share_index(frame) for key, frame in prepared.parsed.items()}
    project = copy(prepared.project)
    project.rows = []
    source = replace(prepared.source, project=project, excluded={}, row_ids=frozenset(),
                     global_values=dict.fromkeys(prepared.source.global_values))
    # The local frame contains only rows in scope, with unique UUIDs. Send
    # separately only requested UUIDs that were absent from the snapshot.
    missing = (frozenset() if len(local) == len(prepared.scope)
               else prepared.scope.difference(local.index))
    return replace(prepared, source=source, scope=missing, local=local,
                   global_frame=global_frame, parsed=parsed)


def restore_preview_input(prepared):
    """Restore the scope and renderer rows from the same index."""
    from .plot_data import PlotRows

    rows = PlotRows.from_frame(prepared.local)
    prepared.project.rows = rows
    prepared.scope = prepared.scope.union(rows.ids)
    return prepared


def preview_process(connection, wanted):
    import gc
    import matplotlib
    matplotlib.use('Agg', force=True)
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from .plotting import layout_notice, render_plot
    from .selection_overlay import SelectionOverlay
    current_prepared = None
    try:
        while True:
            job, _ = receive_preview_message(connection)
            if job is None:
                return
            serial, prepared, spec, views, ratio, palette = job
            if prepared is not None:
                current_prepared = restore_preview_input(prepared)
            prepared = current_prepared
            figure = None
            try:
                if wanted.value != serial:
                    send_preview_response(connection, (serial, None, None))
                    continue
                install_palette(palette)
                figure, result = render_plot(prepared.project, prepared.scope, spec, prepared=prepared)
                if wanted.value != serial:
                    send_preview_response(connection, (serial, None, None))
                    continue
                for ax, view in zip(figure.axes, views or ()):
                    ax._set_view(view)
                # Match the physical pixels used by Qt on high-DPI screens.
                figure.set_dpi(96 * ratio)
                canvas = FigureCanvasAgg(figure)
                # Build the selection index against the same integer row
                # positions sent to Qt, avoiding a second sort of UUIDs.
                pack_preview_membership(result, prepared.local.index)
                overlay = SelectionOverlay(canvas, result, interval_width=spec.interval_width, compact=True)
                canvas.draw()
                warning = layout_notice(figure, canvas.get_renderer(), spec.render_language)
                if warning:
                    result['layout_warning'] = warning
                overlay.disconnect()
                result['_selection_overlay'] = overlay.layers, overlay.locations
                figure._original_dpi = figure.dpi
                if wanted.value != serial:
                    send_preview_response(connection, (serial, None, None))
                    continue
                pixels = np.asarray(canvas.buffer_rgba()).copy()
                if spec.kind == 'multi_depth':
                    # The raster already has every tick. Qt needs the axis
                    # transforms and a template tick for each scale, so keep
                    # those without transferring the expanded tick artists.
                    for ax in figure.axes:
                        for axis in ax._axis_map.values():
                            del axis.majorTicks[1:]
                            del axis.minorTicks[1:]
                # Figure's pickle deliberately excludes its canvas/renderer. The
                # receiving thread reconstructs Python artists, never Qt objects.
                send_preview_response(connection, (serial, (figure, result, spec, pixels), None))
            except Exception as exc:
                send_preview_response(connection, (serial, None, str(exc)))
            finally:
                if figure is not None:
                    figure.clear()
                job = prepared = result = pixels = canvas = figure = overlay = None
                # Matplotlib's cyclic Axes/artists must not retain old frames
                # until an arbitrary later generation-2 collection.
                gc.collect()
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        connection.close()
