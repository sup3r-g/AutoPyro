"""Structured stratigraphy settings preserve native data and expose localized controls."""
from copy import deepcopy

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLineEdit

from autopyro.stratigraphy import COLUMN_STYLE, column_style
from autopyro.ui.widgets.stratigraphy_style import ColumnStyleForm
from autopyro.ui.widgets.stratigraphy_labels import CheckedValuesDialog, geological_label


def make_form(qtbot):
    form = ColumnStyleForm()
    qtbot.addWidget(form)
    # In production the dialog adds the scalar fields; this test isolates structured options.
    form.load(deepcopy(COLUMN_STYLE))
    return form


def test_structured_options_roundtrip_without_editing(qtbot):
    form = make_form(qtbot)
    options = dict(
        legend_kwargs={'facecolor': '#ffaa00', 'labelspacing': 3.5, 'bbox_to_anchor': [.8, .3],
                       'loc': 'upper right', 'shadow': True, 'title': 'Legacy title'},
        legend_titles=['Rocks', 'Minerals', 'Sediment', 'Fossils', 'Tectonic', 'Contacts', 'Samples'],
        x_ticks_dict={'fine': .5, 'coarse': 2}, grain_brackets={'size': ['fine', 'coarse']},
        lithologies={'sand': ['607', '#e0b761', 'Sandstone'], 'solid': [None, 'red', 'Solid']})
    form.load(options)
    values = form.values()
    for key, value in options.items():
        assert values[key] == value
    form.legend_form.load({'facecolor': 'blue', 'anchor_x': .2})
    values = form.values()
    assert values['legend_kwargs']['facecolor'] == 'blue'
    assert values['legend_kwargs']['bbox_to_anchor'] == [.2, .3]
    assert values['legend_kwargs']['shadow'] is True
    assert values['legend_kwargs']['title'] == 'Legacy title'


def test_invalid_scale_reset_does_not_read_invalid_rows(qtbot):
    form = make_form(qtbot)
    scale = form.scale_editor
    scale.mode.setCurrentIndex(scale.mode.findData('custom'))
    scale.ticks.add_row(values=['Fine', 'broken'])
    with pytest.raises(ValueError):
        form.values()
    form.reset_defaults()
    result = form.values()
    for key in ('legend_kwargs', 'legend_titles', 'x_ticks_dict', 'grain_brackets', 'lithologies'):
        assert result[key] == COLUMN_STYLE[key]


def test_hierarchy_checkboxes_roundtrip_and_reset(qtbot):
    form = make_form(qtbot)
    form.load({'hierarchy_levels': ['GROUP', 'MEMBER']})
    assert form.values()['hierarchy_levels'] == ['GROUP', 'MEMBER']
    with qtbot.waitSignal(form.edited):
        form.hierarchy_checks['SUBGROUP'].setChecked(True)
    assert form.values()['hierarchy_levels'] == ['GROUP', 'SUBGROUP', 'MEMBER']
    form.reset_defaults()
    assert form.values()['hierarchy_levels'] == []


def test_custom_rocks_use_picker_values_and_generated_identifiers(qtbot):
    form = make_form(qtbot)
    row = form.rocks_editor.add_row()
    table = form.rocks_editor.table
    assert table.isColumnHidden(0)
    table.item(row, 1).setText('My sand')
    form.rocks_editor.set_pattern(row, '607')
    rocks = form.values()['lithologies']
    assert rocks == {'custom_rock_1': ['607', '#e0b761', 'My sand']}
    assert table.item(row, 2).text() != '607'
    assert not table.item(row, 2).flags() & Qt.ItemFlag.ItemIsEditable


def test_geological_selection_preserves_checked_hidden_values(qtbot):
    picker = CheckedValuesDialog('Minerals', [('Quartz', 'quartz'), ('Pyrite', 'pyrite')],
                                 ['quartz', '^legacy'], None)
    qtbot.addWidget(picker)
    picker.filter_items('Pyrite')
    assert picker.values() == ['quartz', '^legacy']
    assert geological_label('quartz') != 'quartz'
