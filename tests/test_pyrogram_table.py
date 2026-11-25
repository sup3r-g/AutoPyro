import pytest

from autopyro.curves.pyrograms import mapped_pyrogram, read_pyrogram, read_pyrogram_table
from autopyro.model.core import Project


S00_SOURCE = (
    '\n[Curves pyro]\n0\t198\t251\t0\t12\n2\t199\t240\t1\t14\n'
    '\n[Curves oxi]\n0\t298\t19\t465\n1\t299\t20\t521\n'
)


def test_rock_eval_import_displays_and_persists_both_stages(qtbot, tmp_path, monkeypatch):
    from copy import deepcopy
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    from autopyro.data_io.io import build_project
    from autopyro.ui.widgets.pyrograms import PyrogramDialog

    project = build_project(['TOC'], [['1']], [1], [(0, 'TOC', 'number', '%wt')], {})
    project.sync_measurements()
    sample_id = project.rows[0].id
    path = tmp_path / 'run.S00'
    path.write_text(S00_SOURCE, encoding='ascii')
    dialog = PyrogramDialog(None, project, sample_id)
    qtbot.addWidget(dialog)
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(path), ''))
    with qtbot.waitSignal(dialog.changed):
        dialog.add_run()
    assert dialog.trace.count() == 5
    reopened = PyrogramDialog(None, project, sample_id)
    qtbot.addWidget(reopened)
    assert reopened.runs.currentRow() == 0
    assert reopened.verify_source_button.isEnabled()
    oxidation = next(i for i in range(dialog.trace.count())
                     if dialog.trace.itemData(i) == ('CO₂', 'Окисление'))
    dialog.trace.setCurrentIndex(oxidation)
    dialog.axis.setCurrentIndex(dialog.axis.findData('temperature'))
    dialog.refresh_traces()
    assert dialog.trace.currentData() == ('CO₂', 'Окисление')
    line = dialog.figure.axes[0].lines[0]
    assert line.get_xdata().tolist() == [298, 299]
    assert line.get_ydata().tolist() == [465, 521]

    project_path = tmp_path / 'pyrogram.apyro'
    project.save(project_path)
    saved, = Project.load(project_path).science.samples[sample_id].metadata['pyrograms']
    assert saved == read_pyrogram(path, project, sample_id)
    assert [trace['signal'] for trace in saved['run']['traces']] == [
        [251, 240], [0, 1], [12, 14], [19, 20], [465, 521],
    ]
    assert saved['run']['traces'][0]['time'] == [0, 2]
    assert saved['run']['traces'][-1]['time'] == [0, 1]
    before = deepcopy(project.science.samples[sample_id].metadata)
    events = list(project.history)
    messages = []
    monkeypatch.setattr(QMessageBox, 'information', lambda *args: messages.append(args[-1]))
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: messages.append(args[-1]))
    dialog.runs.setCurrentRow(0)
    assert dialog.verify_source_button.isEnabled()
    dialog.verify_source_button.click()
    assert 'совпадает' in messages[-1]
    path.write_text(S00_SOURCE + '\n', encoding='ascii')
    dialog.verify_source_button.click()
    assert 'отличается' in messages[-1]
    path.rename(tmp_path / 'moved.S00')
    dialog.verify_source_button.click()
    assert len(messages) == 3
    assert project.science.samples[sample_id].metadata == before
    assert project.history == events


def test_rock_eval_import_rejects_wrong_column_count(tmp_path):
    path = tmp_path / 'run.s00'
    path.write_text(S00_SOURCE.replace('\t465', ''), encoding='ascii')
    with pytest.raises(ValueError, match='Строка 7.*4 столбцов S00'):
        read_pyrogram(path, Project(), 'sample')


def test_csv_maps_named_columns_and_converts_minutes(tmp_path):
    path = tmp_path / 'export.csv'
    path.write_text('Температура, °C;Время, мин;FID, пА\n300;0;1,5\n310;0,5;2,75\n', encoding='cp1251')
    table = read_pyrogram_table(path)
    assert table.delimiter == ';'
    attachment = mapped_pyrogram(table, Project(), 'sample', time=1, signal=2, temperature=0,
                                 time_unit='min', label='Run', channel='FID', stage='Pyrolysis', unit='pA')
    trace, = attachment['run']['traces']
    assert trace['time'] == [0, 30]
    assert trace['signal'] == [1.5, 2.75]
    assert trace['temperature'] == [300, 310]
    assert attachment['table_mapping']['columns']['time'] == 'Время, мин'
    assert attachment['table_mapping']['encoding'] == 'cp1251'


@pytest.mark.parametrize(('rows', 'error'), [
    ('0\t1\n1\t\n', 'Строка 3.*Сигнал'),
    ('0\t1\n0\t2\n', 'строго возрастать'),
    ('0\t1\n1\t2\t3\n', 'Строка 3'),
])
def test_bad_trace_is_rejected(tmp_path, rows, error):
    path = tmp_path / 'export.tsv'
    path.write_text('Время\tСигнал\n' + rows, encoding='utf-8-sig')
    with pytest.raises(ValueError, match=error):
        mapped_pyrogram(read_pyrogram_table(path), Project(), 'sample', time=0, signal=1,
                        label='Run', channel='FID', stage='Pyrolysis', unit='pA')


def test_line_numbers_refer_to_source_after_metadata_header(tmp_path):
    path = tmp_path / 'export.csv'
    path.write_text('Instrument data\nRun 42\nTime;Signal\n0;5\n1;bad\n', encoding='utf-8')
    source = read_pyrogram_table(path, allow_unmapped=True)
    assert not source.headers
    table = source.with_layout(3, ';')
    with pytest.raises(ValueError, match='Строка 5.*Signal'):
        mapped_pyrogram(table, Project(), 'sample', time=0, signal=1,
                        label='Run', channel='FID', stage='Pyrolysis', unit='pA')


def test_select_run_with_clock_time_and_midnight(tmp_path):
    path = tmp_path / 'clock.csv'
    path.write_text(
        'Time,Temp,CO2\n15:42:25,70,1\n15:42:26,71,2\n'
        '23:59:58,70,3\n00:00:00,71,4\n00:00:00,72,5\n00:00:02,73,6\n00:00:01,70,7\n',
        encoding='utf-8',
    )
    table = read_pyrogram_table(path)
    args = dict(time=0, temperature=1, signal=2, time_unit='clock',
                label='Run', channel='CO2', stage='RPO', unit='scaled')
    attachment = mapped_pyrogram(table, Project(), 'sample', first_line=4, last_line=7, **args)
    trace, = attachment['run']['traces']
    assert trace['time'] == [0, 2, 2, 4]
    assert trace['signal'] == [3, 4, 5, 6]
    assert trace['time_order'] == 'nondecreasing'
    assert attachment['table_mapping']['first_line'] == 4
    assert attachment['table_mapping']['last_line'] == 7
    with pytest.raises(ValueError, match='Строка 8.*время убывает'):
        mapped_pyrogram(table, Project(), 'sample', first_line=4, **args)


def test_table_import_attaches_mapped_trace_to_selected_sample(qtbot, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QDialog, QFileDialog

    from autopyro.data_io.io import build_project
    from autopyro.ui.widgets.pyrograms import PyrogramColumnsDialog, PyrogramDialog

    project = build_project(['TOC'], [['1'], ['2']], [1, 2], [(0, 'TOC', 'number', '%wt')], {})
    project.sync_measurements()
    first, second = project.rows
    path = tmp_path / 'trace.csv'
    path.write_text('Instrument data\nRun 42\nt;s\n0;2\n1;3\n', encoding='utf-8')
    dialog = PyrogramDialog(None, project, second.id)
    qtbot.addWidget(dialog)
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(path), ''))

    def map_columns(mapping):
        assert not mapping.import_button.isEnabled()
        assert mapping.table is None
        mapping.header_line.setValue(3)
        mapping.delimiter.setCurrentIndex(mapping.delimiter.findData(';'))
        mapping.columns['time'].setCurrentIndex(1)
        mapping.columns['signal'].setCurrentIndex(2)
        for key, text in {'channel': 'FID', 'stage': 'Pyrolysis', 'unit': 'pA'}.items():
            mapping.fields[key].setText(text)
        assert mapping.import_button.isEnabled()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(PyrogramColumnsDialog, 'exec', map_columns)
    with qtbot.waitSignal(dialog.changed):
        dialog.add_run()
    assert 'pyrograms' not in project.science.samples[first.id].metadata
    assert dialog.figure.axes[0].lines[0].get_ydata().tolist() == [2, 3]
    path = tmp_path / 'pyrogram.apyro'
    project.save(path)
    saved = Project.load(path).science.samples[second.id].metadata['pyrograms'][0]
    assert saved['table_mapping']['columns'] == {'time': 't', 'signal': 's'}
    assert saved['table_mapping']['header_line'] == 3
    assert saved['table_mapping']['delimiter'] == ';'
