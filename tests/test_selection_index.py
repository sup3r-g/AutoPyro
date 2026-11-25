"""The compact transfer index preserves UUID membership across tracks/items."""
import pickle

import numpy as np
import pytest

from autopyro.selection_overlay import SelectionIndex


def part(keys, layers, positions):
    return np.asarray(keys, dtype=object), np.asarray(layers, dtype=np.int32), np.asarray(positions, dtype=np.intp)


@pytest.mark.parametrize('selected', [[], ['missing'], ['a'], ['a', 'b', 'a'], ['a', 'b', 'c']])
def test_lookup_keeps_all_tracks_and_deduplicates_site_members(selected):
    index = SelectionIndex.from_parts([part(['a', 'b', 'a'], [0, 0, 1], [5, 5, 9]),
                                       part(['c'], [1], [10])])
    actual, positions = index.lookup(selected, 3)
    reference = {'a': [(0, 5), (1, 9)], 'b': [(0, 5)], 'c': [(1, 10)]}
    assert actual == set(selected) & reference.keys()
    for layer, matches in enumerate(positions):
        expected = {pos for rid in actual for group, pos in reference[rid] if group == layer}
        assert set(matches) == expected


def test_arbitrary_string_ids_roundtrip_without_truncation():
    keys = ['a', 'a\0', 'длинное имя 🌋', 'x' * 200, 'x' * 201]
    index = SelectionIndex.from_parts([part(keys, [0] * len(keys), list(range(len(keys))))])
    restored = pickle.loads(pickle.dumps(index, protocol=pickle.HIGHEST_PROTOCOL))
    for i, key in enumerate(keys):
        selected, positions = restored.lookup([key], 1)
        assert selected == {key}
        np.testing.assert_array_equal(positions[0], [i])
    restored.clear()
    assert not restored
    assert restored.lookup(keys, 2)[0] == set()


def test_empty_index_accepts_empty_or_hidden_selection():
    index = SelectionIndex.from_parts([])
    assert not index
    selected, positions = index.lookup(['hidden'], 2)
    assert not selected and all(len(items) == 0 for items in positions)


def test_retiring_one_frame_does_not_clear_the_shared_uuid_lookup():
    lookup = {'a': 0, 'b': 1}
    first = SelectionIndex(np.asarray([1]), np.asarray([0]), np.asarray([5]), lookup)
    second = SelectionIndex(np.asarray([1]), np.asarray([0]), np.asarray([9]), lookup)
    first.clear()
    selected, positions = second.lookup(['a', 'b', 'unknown'], 1)
    assert selected == {'b'} and positions[0].tolist() == [9]
    assert lookup == {'a': 0, 'b': 1}


@pytest.mark.parametrize('kind', ['scatter', 'errorbar', 'image', 'bar', 'contour'])
def test_retirement_removes_data_artists_without_resetting_axes(kind):
    from matplotlib.figure import Figure
    from autopyro.ui.plot_preview import release_figure
    figure = Figure()
    ax = figure.add_subplot()
    if kind == 'scatter':
        artist = ax.scatter([1, 2], [3, 4], c=[0, 1])
        figure.colorbar(artist, ax=ax)
    elif kind == 'errorbar':
        ax.errorbar([1, 2], [3, 4], yerr=[.1, .2])
    elif kind == 'image':
        figure.colorbar(ax.imshow([[1, 2], [3, 4]]), ax=ax)
    elif kind == 'bar':
        ax.bar([1, 2], [3, 4])
    else:
        ax.clabel(ax.contour([[1, 2], [3, 4]]))
    release_figure(figure)
    assert not figure.axes
    assert not ax.collections and not ax.images and not ax.lines and not ax.patches
