"""Curve interval results remain linked to points, report rows and explanations."""
from unittest.mock import patch

import numpy as np
import pytest

from autopyro.ui.window import MainWindow
from autopyro.standard_interpretation import standard_schemes
from test_interpretation_comparison import data


@pytest.mark.parametrize('name',['HI–Tmax','HI–OI','HIo · HI–Tmax'])
def test_curve_intervals_in_remote_preview(qtbot,name):
    project,_,_=data()
    project.sync_measurements()
    for role,uncertainty in [('Tmax',1),('HI',10),('OI',5)]:
        cid=next(c.id for c in project.columns if c.name==role)
        m=project.science.current(project.rows[0].id,cid)
        m.uncertainty=uncertainty;m.uncertainty_type='absolute'
    window=MainWindow();qtbot.addWidget(window);window.set_project(project)
    window.resize(1550,1050);window.show()
    window.may_replace=lambda:True;window.show_interpretation()
    d=window.interpretation_window
    d.commit_graphical_scheme(standard_schemes()[name],auto_bind=True)
    qtbot.waitUntil(lambda:d.report is not None and d.plot.result is not None and not d.plot.timer.isActive(),timeout=30000)
    qtbot.waitUntil(lambda:not d.plot.preview.busy and not d.plot.viewport_timer.isActive(),timeout=15000)
    item=d.report['results'][0];rid=project.rows[0].id
    assert item['status'] in ('classified','ambiguous')
    assert item['possible_types']
    canvas = d.plot.canvas
    before = np.asarray(canvas.renderer.buffer_rgba()).copy()
    assert np.std(before[:, :, :3]) > 1
    # Selection must paint the overlay immediately. A delayed viewport resize
    # can independently request a fresh frame, so the global serial is not a
    # reliable way to tell whether selection itself scheduled a remote draw.
    with patch.object(canvas, 'draw_idle', wraps=canvas.draw_idle) as draw_idle:
        d.plot.select_rows([rid])
    draw_idle.assert_not_called()
    after = np.asarray(canvas.renderer.buffer_rgba())
    assert np.std(after[:, :, :3]) > 1
    assert not np.array_equal(before, after)
    assert d.plot.selection_overlay.selected_ids=={rid}
    assert {i.data() for i in d.table.selectionModel().selectedRows()}=={rid}
    text=d.explanation.toPlainText()
    assert 'Возможные типы вещества' in text and 'Интервальный метод' in text
    if name.startswith('HIo'):
        assert item['status']=='ambiguous' and item['value'] is None
        assert item['value_bounds'][0]<item['value_bounds'][1]
        assert rid in d.plot.result['uncertainty_crossings']
        assert 'Диапазон HIo' in text and 'Причина неоднозначности' in text
        column = d.table.model().headers.index('Диапазон HIo')
        assert d.table.horizontalHeader().visualIndex(column) == 2
        assert d.table.model().rows[0][column] == ' – '.join(f'{bound:.6g}' for bound in item['value_bounds'])
        report = d.report
        other_id = project.rows[1].id
        for mode, expected_ids in (
            ('ambiguous', [rid]),
            ('not_applicable', [other_id]),
            ('quality', [rid, other_id]),
            ('all', [rid, other_id]),
        ):
            d.result_filter.setCurrentIndex(d.result_filter.findData(mode))
            qtbot.waitUntil(lambda: d.plot.result is not None and not d.plot.timer.isActive(), timeout=15000)
            assert d.report is report
            assert list(d.result_rows) == d.plot.rendered_ids == expected_ids
            assert [row[0] for row in d.table.model().rows] == expected_ids
            assert {row_id for artist in d.plot.result['artists']
                    for row_id in artist._autopyro_row_ids} == set(expected_ids)
            if mode == 'ambiguous':
                assert d.plot.selection_overlay.selected_ids == {rid}
            if mode == 'not_applicable':
                assert not d.result_selection
                d.table.selectRow(0)
                assert d.plot.selection_overlay.selected_ids == {other_id}
                assert other_id in d.explanation.toPlainText()
                d.plot.select_rows([other_id])
                assert {i.data() for i in d.table.selectionModel().selectedRows()} == {other_id}
    d.close();window.dirty=False
