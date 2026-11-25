"""Uncertainty metadata survives the remote preview and selection synchronization."""
import pytest

from autopyro.ui.window import MainWindow
from test_measurement_uncertainty import data
from test_interpretation_comparison import data as comparison_data
from autopyro.standard_interpretation import standard_schemes


def ready(qtbot, dialog):
    qtbot.waitUntil(lambda: dialog.report is not None and dialog.plot.result is not None
                   and not dialog.plot.timer.isActive(), timeout=30000)


@pytest.mark.parametrize('mode', ['regions', 'comparison', 'threshold', 'single_region'])
def test_uncertainty_crossing_and_selection_in_remote_preview(qtbot, mode):
    comparison = mode == 'comparison'
    if comparison:
        project,binding,_=comparison_data()
        project.sync_measurements()
        m=project.science.current(project.rows[0].id,binding['x_id'])
        m.uncertainty=.2;m.uncertainty_type='absolute'
    else:
        project,binding,_=data()
    if mode == 'single_region':
        binding['scheme']['regions'] = binding['scheme']['regions'][:1]
    if mode == 'threshold':
        binding['scheme']['regions'] = []
        binding['scheme']['lines'] = [{
            'id': 'cutoff', 'label': 'Порог TOC',
            'geometry': {'type': 'LineString', 'coordinates': [[2, 0], [2, 500]]},
            'threshold': {'axis': 'x', 'less': 'Ниже порога', 'greater': 'Выше порога'},
        }]
    window=MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.may_replace=lambda:True
    window.show_interpretation()
    dialog=window.interpretation_window
    dialog.commit_graphical_scheme(binding['scheme'],auto_bind=True)
    dialog.comparison.setChecked(comparison)
    ready(qtbot,dialog)
    rid=project.rows[0].id
    assert rid in dialog.plot.result['uncertainty_crossings']
    dialog.plot.select_rows([rid])
    assert dialog.plot.selection_overlay.selected_ids=={rid}
    assert {i.data() for i in dialog.table.selectionModel().selectedRows()}=={rid}
    assert 'Интервал TOC' in dialog.explanation.toPlainText()
    assert dialog.report['results'][0]['status']=='ambiguous'
    if mode == 'single_region':
        assert dialog.plot.result['uncertainty_crossings'][rid] == ['A']
    if mode == 'threshold':
        assert dialog.plot.result['uncertainty_crossings'][rid] == ['cutoff']
        assert 'Выше порога' in dialog.explanation.toPlainText()
        assert 'Выше порога' in dialog.table.model().rows[0][4]
        assert 'Выше порога' in dialog.plot.point_tooltip(rid, (1.9, 200))
    # A metadata-only change must invalidate the snapshot and remove the old marker.
    m=window.model.project.science.current(rid,binding['x_id'])
    m.uncertainty=.01;m.revision+=1
    if not comparison:
        window.model.notify_changed()
        ready(qtbot,dialog)
        assert rid not in dialog.plot.result['uncertainty_crossings']
        assert dialog.plot.selection_overlay.selected_ids=={rid}
    dialog.close();window.dirty=False



def test_ordinary_chart_updates_measurement_intervals_in_remote_preview(qtbot, tmp_path):
    from matplotlib.collections import LineCollection

    project, binding, _ = data()
    row_id = project.rows[0].id
    x_id, y_id = binding['x_id'], binding['y_id']
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.may_replace = lambda: True
    window.show_plots()
    dialog = window.plot_window
    dialog.data.load({'x_id': x_id, 'y_id': y_id})
    dialog.schedule()
    qtbot.waitUntil(lambda: dialog.result is not None, timeout=30000)
    assert not any(isinstance(c, LineCollection) for c in dialog.figure.axes[0].collections)

    def segment():
        return next(c.get_segments()[0] for c in dialog.figure.axes[0].collections
                    if isinstance(c, LineCollection))

    dialog.layout_form.fields['measurement_error_bars'][0].setChecked(True)
    qtbot.waitUntil(lambda: dialog.result is not None, timeout=30000)
    before = segment()
    measurement = project.science.current(row_id, x_id)
    measurement.uncertainty *= 2
    measurement.revision += 1
    measurement.qc_status = 'expert pending'
    window.model.notify_changed()
    qtbot.waitUntil(lambda: dialog.result is not None, timeout=30000)
    after = segment()
    assert after[1, 0] - after[0, 0] == pytest.approx(2 * (before[1, 0] - before[0, 0]))
    dialog.layout_form.fields['qc_outline'][0].setChecked(True)
    qtbot.waitUntil(lambda: dialog.result is not None, timeout=30000)
    assert any(artist.get_gid() == 'measurement-qc-outline' for artist in dialog.figure.axes[0].collections)
    path = tmp_path / 'uncertainty.svg'
    dialog.start_export([('Uncertainty', path, dialog.spec)])
    qtbot.waitUntil(lambda: not dialog.export_busy(), timeout=30000)
    assert 'id="measurement-uncertainty-1"' in path.read_text(encoding='utf-8')
    assert 'id="measurement-qc-outline"' in path.read_text(encoding='utf-8')
    dialog.spec.kind = 'mpl:plot'
    dialog.load_controls()
    dialog.schedule()
    qtbot.waitUntil(lambda: dialog.result is not None, timeout=30000)
    assert dialog.spec.kind == 'mpl:plot'
    assert not dialog.layout_form.fields['measurement_error_bars'][0].isHidden()
    assert any(artist.get_gid() == 'measurement-uncertainty-1' for artist in dialog.figure.axes[0].collections)
    dialog.spec.kind = 'mpl3d:scatter'
    dialog.spec.z_id = y_id
    dialog.spec.library_bindings = {}
    dialog.load_controls()
    dialog.schedule()
    qtbot.waitUntil(lambda: dialog.result is not None, timeout=30000)
    assert dialog.figure.axes[0].name == '3d'
    assert not dialog.layout_form.fields['measurement_error_bars'][0].isHidden()
    assert any(artist.get_gid() == 'measurement-uncertainty-1' for artist in dialog.figure.axes[0].collections)
    dialog.close()
    window.dirty = False


def test_interval_depth_chart_exposes_uncertainty_control_and_renders_midpoint(qtbot):
    from autopyro.io import build_project
    from autopyro.plotting import PlotSpec

    project = build_project(['X', 'Top', 'Bottom'], [[1, 2, 6]], [2],
                            [(0, 'X', 'number', '%'), (1, 'Top', 'number', 'm'),
                             (2, 'Bottom', 'number', 'm')], {})
    project.sync_measurements()
    x, top, bottom = [column.id for column in project.columns]
    measurement = project.science.current(project.rows[0].id, x)
    measurement.uncertainty = .25
    measurement.uncertainty_type = 'absolute'
    window = MainWindow()
    qtbot.addWidget(window)
    window.may_replace = lambda: True
    window.set_project(project)
    window.show_plots()
    dialog = window.plot_window
    dialog.spec = PlotSpec(kind='depth', x_id=x, y_id=top, depth_bottom_id=bottom)
    dialog.load_controls()
    dialog.schedule()
    qtbot.waitUntil(lambda: dialog.result is not None, timeout=30000)
    assert dialog.spec.depth_bottom_id == bottom
    control = dialog.layout_form.fields['measurement_error_bars'][0]
    assert not control.isHidden()
    control.setChecked(True)
    qtbot.waitUntil(lambda: dialog.result is not None, timeout=30000)
    bars, = [artist for artist in dialog.figure.axes[0].collections
             if artist.get_gid() == 'measurement-uncertainty-1']
    assert bars.get_segments()[0].tolist() == [[.75, 4], [1.25, 4]]
    dialog.close()
    window.dirty = False
