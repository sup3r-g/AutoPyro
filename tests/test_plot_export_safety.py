"""Failed export must not corrupt an existing figure or JSON document."""
import json

import pytest
from matplotlib.figure import Figure
from PySide6.QtWidgets import QFileDialog, QMessageBox

from autopyro.file_output import write_json
from autopyro.io import build_project
from autopyro.plotting import PlotSpec, save_plot
from autopyro.ui.window import MainWindow


@pytest.fixture
def project():
    return build_project(['X', 'Y'], [[1, 2], [2, 3]], [2, 3],
        [(0, 'X', 'number', ''), (1, 'Y', 'number', '')], {})


@pytest.mark.parametrize('extension', ['png', 'tif', 'svg', 'pdf'])
def test_partial_figure_write_keeps_previous_file(project, tmp_path, monkeypatch, extension):
    path = tmp_path / ('figure.' + extension)
    path.write_bytes(b'previous complete figure')
    def broken(figure, temporary, **kwargs):
        temporary.write_bytes(b'partial figure')
        raise OSError('Export failed')
    monkeypatch.setattr(Figure, 'savefig', broken)
    x, y = [c.id for c in project.columns]
    with pytest.raises(OSError, match='Export failed'):
        save_plot(path, project, [r.id for r in project.rows], PlotSpec(x_id=x, y_id=y))
    assert path.read_bytes() == b'previous complete figure'
    assert list(tmp_path.iterdir()) == [path]


def test_failure_does_not_publish_a_new_partial_file(project, tmp_path, monkeypatch):
    path = tmp_path / 'new.svg'
    def broken(figure, temporary, **kwargs):
        temporary.write_bytes(b'partial')
        raise OSError('Failed')
    monkeypatch.setattr(Figure, 'savefig', broken)
    with pytest.raises(OSError):
        save_plot(path, project, [r.id for r in project.rows],
                  PlotSpec(x_id=project.columns[0].id, y_id=project.columns[1].id))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('extension', ['png', 'svg', 'pdf', 'tif'])
def test_export_reports_clipped_labels_in_selected_language(project, tmp_path, extension):
    from autopyro.plotting import AxisSpec

    x = project.columns[0].id
    spec = PlotSpec(kind='hist', x_id=x, width_mm=100, height_mm=75, render_language='en',
                    x_axis=AxisSpec(label='A long axis label with physical units [mg/g]', label_pad=12))
    path = tmp_path / f'figure.{extension}'
    result = save_plot(path, project, project.row_map(), spec)
    assert result['layout_warnings'] and 'outside the figure' in result['layout_warnings'][0]
    if extension == 'svg':
        assert 'Sample count' in path.read_text(encoding='utf-8')


def test_json_replacement_failure_cleans_up_and_keeps_previous_file(tmp_path, monkeypatch):
    import autopyro.file_output as module
    path = tmp_path / 'style.json'
    path.write_text('previous', encoding='utf-8')
    def denied(*args):
        raise PermissionError('Target is locked')
    monkeypatch.setattr(module.os, 'replace', denied)
    with pytest.raises(PermissionError):
        write_json(path, {'title': 'New'})
    assert path.read_text(encoding='utf-8') == 'previous'
    assert list(tmp_path.iterdir()) == [path]


def test_json_validation_precedes_temporary_file_creation(tmp_path):
    path = tmp_path / 'style.json'
    with pytest.raises(ValueError):
        write_json(path, {'value': float('nan')})
    assert not list(tmp_path.iterdir())
    write_json(path, {'title': 'Пиролиз'})
    assert json.loads(path.read_text(encoding='utf-8')) == {'title': 'Пиролиз'}


def test_gui_json_export_uses_atomic_writer(project, qtbot, tmp_path, monkeypatch):
    import autopyro.file_output as module
    owner = MainWindow()
    owner.may_replace = lambda: True
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_plots()
    dialog = owner.plot_window
    dialog.timer.stop()
    dialog.redraw()
    path = tmp_path / 'style.json'
    path.write_text('previous', encoding='utf-8')
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), 'JSON'))
    messages = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: messages.append(args[-1]))
    def denied(*args):
        raise PermissionError('locked')
    monkeypatch.setattr(module.os, 'replace', denied)
    dialog.export_spec()
    assert messages and path.read_text(encoding='utf-8') == 'previous'
    assert list(tmp_path.iterdir()) == [path]
    dialog.close()
    owner.dirty = False


def test_partial_json_write_keeps_previous_file(tmp_path, monkeypatch):
    from pathlib import Path
    path = tmp_path / 'style.json'
    path.write_text('previous', encoding='utf-8')
    def broken(temporary, text, **kwargs):
        temporary.write_bytes(b'partial')
        raise OSError('Write failed')
    monkeypatch.setattr(Path, 'write_text', broken)
    with pytest.raises(OSError):
        write_json(path, {'title': 'new'})
    assert path.read_text(encoding='utf-8') == 'previous'
    assert list(tmp_path.iterdir()) == [path]


def test_batch_manifest_lists_only_completed_figures(project, qtbot, tmp_path, monkeypatch):
    import json
    from PySide6.QtWidgets import QInputDialog
    owner = MainWindow()
    owner.may_replace = lambda: True
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_plots()
    dialog = owner.plot_window
    dialog.timer.stop()
    dialog.redraw()
    bad = dialog.spec.to_dict()
    bad['kind'] = 'invalid-kind'
    good = dialog.spec.to_dict() | {'kind': 'hist', 'filter_rules': [
        {'column': project.columns[0].id, 'operator': 'gt', 'value': 1},
    ]}
    project.view['plots'] = {'First': bad, 'Second': good}
    start = dialog.export_service.start

    def collide(jobs, manifest):
        (tmp_path / 'plot-002-Second.png').write_bytes(b'late figure')
        (tmp_path / 'plots-manifest.json').write_bytes(b'late manifest')
        return start(jobs, manifest)

    monkeypatch.setattr(dialog.export_service, 'start', collide)
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *args: str(tmp_path))
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args, **kwargs: ('png', True))
    monkeypatch.setattr(QMessageBox, 'information', lambda *args: None)
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: None)
    dialog.batch_export()
    qtbot.waitUntil(lambda: not dialog.export_busy(), timeout=30000)
    assert (tmp_path / 'plots-manifest.json').read_bytes() == b'late manifest'
    assert (tmp_path / 'plot-002-Second.png').read_bytes() == b'late figure'
    manifest = json.loads((tmp_path / 'plots-manifest-2.json').read_text(encoding='utf-8'))
    assert [p['name'] for p in manifest['plots']] == ['Second']
    assert len(manifest['errors']) == 1
    record, = manifest['plots']
    assert record['file'] == 'plot-002-Second-2.png'
    assert (tmp_path / record['file']).is_file()
    assert record['row_ids'] == [row.id for row in project.rows]
    x = project.columns[0].id
    assert record['plotted_row_ids'][x] == [row.id for row in project.rows if row.values[x] > 1]
    assert len(list(tmp_path.iterdir())) == 4
    dialog.close()
    owner.dirty = False


def test_transient_highlight_is_not_included_in_export(project, qtbot, tmp_path):
    import numpy as np
    from PIL import Image
    from autopyro.ui.table.selection import select_row_ids
    owner = MainWindow()
    owner.may_replace = lambda: True
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_plots()
    dialog = owner.plot_window
    dialog.timer.stop()
    dialog.redraw()
    before, after = tmp_path / 'before.png', tmp_path / 'after.png'
    save_plot(before, project, dialog.scope_ids(), dialog.spec)
    select_row_ids(owner, [project.rows[0].id])
    assert dialog.selection_overlay.selected_ids
    save_plot(after, project, dialog.scope_ids(), dialog.spec)
    with Image.open(before) as first, Image.open(after) as second:
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    dialog.close()
    owner.dirty = False


@pytest.mark.parametrize('field', ['unit', 'kind', 'basis', 'unused_unit'])
def test_template_checks_semantics_even_when_uuid_still_exists(project, qtbot, tmp_path, monkeypatch, field):
    if field == 'unused_unit':
        from autopyro.core import Column
        column = Column('Unused', 'number', unit='m')
        project.columns.append(column)
        for row in project.rows:
            row.values[column.id] = 0
    owner = MainWindow()
    owner.may_replace = lambda: True
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_plots()
    dialog = owner.plot_window
    dialog.timer.stop()
    dialog.redraw()
    path = tmp_path / 'style.json'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), 'JSON'))
    dialog.export_spec()
    if field == 'unused_unit':
        project.columns[-1].unit = 'km'
    elif field == 'basis':
        project.columns[0].metadata['basis'] = 'dry'
    else:
        setattr(project.columns[0], field, 'km' if field == 'unit' else 'text')
    previous = dialog.spec
    messages = []
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(path), 'JSON'))
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: messages.append(args[-1]))
    dialog.import_spec()
    if field == 'unused_unit':
        assert not messages and dialog.spec is not previous
    else:
        assert messages and dialog.spec is previous
    dialog.close()
    owner.dirty = False



@pytest.mark.parametrize('suffix', ['png', 'svg'])
def test_panel_export_stops_between_panels_and_checks_combined_size(project, tmp_path, suffix):
    from dataclasses import replace
    from threading import Event

    from autopyro.visualization.plot_data import (
        PlotDataCache, PreparationCancelled, capture_input, input_token,
    )
    from autopyro.visualization.plot_panels import panel_layout, write_panels

    x, y = [column.id for column in project.columns]
    spec = PlotSpec(x_id=x, y_id=y, width_mm=50.8, height_mm=38.1, dpi=100)
    ids = list(project.row_map())
    source = capture_input(project, ids, [spec], input_token(project, 0, [spec]))
    jobs = [{'spec': spec, 'ids': ids, 'source': source}] * 2
    path = tmp_path / f'panels.{suffix}'
    path.write_bytes(b'previous complete figure')
    cancelled = Event()
    progress = []

    def stop(done, total):
        progress.append((done, total))
        cancelled.set()

    with pytest.raises(PreparationCancelled):
        write_panels(path, jobs, 2, PlotDataCache(), cancelled, stop)
    assert progress == [(1, 2)]
    assert path.read_bytes() == b'previous complete figure'

    large = replace(spec, width_mm=500, height_mm=500, dpi=300)
    if suffix == 'png':
        with pytest.raises(ValueError, match='64 млн'):
            panel_layout(path, [large, large], 2)
    else:
        panel_layout(path, [large, large], 2)



def test_common_export_style_preserves_per_track_axis_semantics():
    from autopyro.visualization.plot_export import with_export_style
    from autopyro.visualization.plotting import AxisSpec

    source = PlotSpec(background='#eeeeee', x_axis=AxisSpec(label_size=17, color='#444444', rotation=30))
    axis = AxisSpec(label='TOC', minimum=0.1, maximum=10, scale='log', inverted=True)
    target = PlotSpec(kind='multi_depth', track_ids=['toc'], track_axes={'toc': axis},
                      y_id='md', width_mm=120, color_by='toc', color_class_edges='0; 1; 10')
    styled = with_export_style(target, source)
    assert styled.track_axes['toc'].label_size == 17
    assert styled.track_axes['toc'].rotation == 30
    assert styled.track_axes['toc'].scale == 'log' and styled.track_axes['toc'].inverted
    assert styled.track_axes['toc'].minimum == 0.1 and styled.track_axes['toc'].maximum == 10
    assert styled.track_axes['toc'].label == 'TOC'
    assert styled.y_id == 'md' and styled.track_ids == ['toc']
    assert styled.color_class_edges == '0; 1; 10' and styled.width_mm == 120
    assert target.background == '#ffffff' and target.track_axes['toc'].label_size == 11
