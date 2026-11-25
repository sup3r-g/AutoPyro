"""Column snapshots stay isolated and preserve row/render compatibility."""
from copy import deepcopy
import pickle

import pandas as pd
import pytest

from autopyro.core import Column, Project, Row
from autopyro.plot_data import PlotRows, PlotDataCache, capture_input, input_token
from autopyro.plotting import PlotSpec
from autopyro.tabular import classify, classify_series, project_frame


@pytest.mark.parametrize('numeric', [False, True])
@pytest.mark.parametrize('copies', [1, 64])
@pytest.mark.parametrize('series', [
    pd.Series([None, '', 'ND', '<0.1', '1.25', 2 ** 100, 10 ** 400, True,
               float('nan'), float('inf'), pd.NA], dtype=object),
    pd.Series([1, 2, None], dtype='Int64'),
    pd.Series(['1', 'ND', None], dtype='string'),
    pd.Series(pd.to_datetime(['2020-01-01', None])),
])
def test_column_classification_matches_previous_series_map(series, copies, numeric):
    series = pd.concat([series] * copies, ignore_index=True)
    expected = pd.DataFrame(series.map(lambda value: classify(value, numeric=numeric)).tolist(),
        index=series.index, columns=['status', 'number'], dtype=object)
    pd.testing.assert_frame_equal(classify_series(series, numeric=numeric), expected)


def snapshot():
    x, y, label = Column('X', 'number'), Column('Y', 'number'), Column('Label')
    project = Project(columns=[x, y, label], rows=[
        Row({x.id: 1, y.id: None, label.id: {'values': ['a']}}),
        Row({x.id: 2 ** 100, y.id: '<3', label.id: 'B'}),
        Row({x.id: 3, y.id: 'ND', label.id: 'C'}),
    ])
    spec = PlotSpec(x_id=x.id, y_id=y.id, label_by=label.id)
    source = capture_input(project, [r.id for r in project.rows], [spec], input_token(project, 0, [spec]))
    return project, spec, source


def test_snapshot_columns_preserve_values_order_slices_and_pickle():
    project, spec, source = snapshot()
    rows = source.project.rows
    assert isinstance(rows, PlotRows)
    assert len(rows) == 3 and rows[-1].id == project.rows[-1].id
    assert [r.id for r in rows[1:]] == [r.id for r in project.rows[1:]]
    with pytest.raises(IndexError):
        _ = rows[3]
    restored = pickle.loads(pickle.dumps(source))
    expected = project_frame(project, None, list(source.local_ids))
    pd.testing.assert_frame_equal(restored.project.rows.to_frame(list(source.local_ids)), expected)
    project.rows[0].values[spec.label_by]['values'].append('changed')
    assert rows[0].values[spec.label_by] == {'values': ['a']}


def test_frame_preparation_does_not_materialize_snapshot_rows(monkeypatch):
    project, spec, source = snapshot()
    ids = [row.id for row in project.rows]
    expected = project_frame(project, ids[1:], list(source.local_ids))
    def fail(*args, **kwargs):
        raise AssertionError('Preparation must read columns directly')
    monkeypatch.setattr(PlotRows, '__getitem__', fail)
    prepared = PlotDataCache().prepare(source, ids[1:], spec)
    pd.testing.assert_frame_equal(prepared.local, expected)


def test_empty_column_snapshot_keeps_row_count_and_is_independent():
    rows = PlotRows(['a', 'b'], {})
    assert rows.to_frame([]).shape == (2, 0)
    assert rows.to_frame([], ['b', 'missing']).index.tolist() == ['b']
    assert PlotRows.from_frame(rows.to_frame([])).ids == ['a', 'b']
    copied = deepcopy(rows)
    rows.ids.append('c')
    assert len(copied) == 2
