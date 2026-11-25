from dataclasses import replace

from autopyro.model.core import Column, Project, Row
from autopyro.ui.table.model import TableModel


def test_recent_measurements_follow_replacement_revision_and_removal(qapp):
    project = Project(columns=[Column('Value', 'number', id='v')],
                      rows=[Row({'v': i}, id=str(i)) for i in range(12)])
    project.sync_measurements()
    model = TableModel(project, prepared=True)
    first = model.measurement_results('0', 'v')[0]
    for i in range(1, 12):
        assert model.measurement_results(str(i), 'v')[0].sample_id == str(i)
    assert model.measurement_results('0', 'v') == [first]
    replacement = replace(first, value=99)
    project.science.measurements[first.id] = replacement
    model.data_revision += 1
    assert model.measurement_results('0', 'v') == [replacement]
    del project.science.measurements[first.id]
    assert model.measurement_results('0', 'v') == ()
    assert model.measurement_results('missing', 'v') == ()
