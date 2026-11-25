"""User workflows for library views through the shared plot properties."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog

from autopyro.io import build_project
from autopyro.plotting import AxisSpec, PlotSpec, render_plot
from autopyro.plot_controls import controls_for
from autopyro.plot_catalog import library_kinds, MERGED_KINDS, KINDS, PAIRED_KINDS
from autopyro.ui.window import MainWindow


@pytest.fixture
def window(qtbot):
    owner = MainWindow()
    owner.may_replace = lambda: True
    qtbot.addWidget(owner)
    rows = [[x, y, x + y, .1, f'G{x % 2}'] for y in range(1, 4) for x in range(1, 4)]
    owner.set_project(build_project(['X', 'Y', 'Z', 'Error', 'Group'], rows, list(range(2, 11)),
        [(i, name, 'text' if name == 'Group' else 'number', '')
         for i, name in enumerate(['X', 'Y', 'Z', 'Error', 'Group'])], {}))
    owner.show_plots()
    owner.plot_window.timer.stop()
    yield owner
    if owner.plot_window:
        owner.plot_window.close()
    owner.dirty = False


def choose(dialog, kind):
    index = dialog.kind.findData(MERGED_KINDS.get(kind, kind))
    assert index >= 0, kind
    dialog.kind.setCurrentIndex(index)
    dialog.sync_variant(kind)
    dialog.kind_changed()
    dialog.plot_mode.setCurrentIndex(int(kind in ('mpl:scatter', 'mpl:hist', 'mpl:boxplot', 'mpl:violinplot')))
    if kind in ('mpl:semilogx', 'mpl:semilogy', 'mpl:loglog'):
        dialog.x_form.load({'scale': 'log' if kind in ('mpl:semilogx', 'mpl:loglog') else 'linear'})
        dialog.y_form.load({'scale': 'log' if kind in ('mpl:semilogy', 'mpl:loglog') else 'linear'})
    dialog.timer.stop()


def test_plot_catalog_groups_keep_all_views_and_skip_headings(window, qtbot):
    dialog = window.plot_window
    groups = KINDS.groups()
    entries = [key for _, items in groups for key, _ in items]
    assert len(entries) == len(set(entries)) == len(KINDS) == 32
    assert set(entries) == set(KINDS)
    assert dialog.kind.currentData() == 'scatter'
    offset = 0
    for title, items in groups:
        assert dialog.kind.itemText(offset) == title
        assert dialog.kind.itemData(offset) is None
        flags = dialog.kind.model().item(offset).flags()
        assert not flags & (Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        for index, (key, label) in enumerate(items, offset + 1):
            assert dialog.kind.itemData(index) == key
            assert dialog.kind.itemText(index) == label
        offset += len(items) + 1
    dialog.kind.setCurrentIndex(dialog.kind.findData(groups[0][1][-1][0]))
    qtbot.keyClick(dialog.kind, Qt.Key.Key_Down)
    assert dialog.kind.currentData() == groups[1][1][0][0]
    qtbot.keyClick(dialog.kind, Qt.Key.Key_Up)
    assert dialog.kind.currentData() == groups[0][1][-1][0]
    dialog.timer.stop()


@pytest.mark.parametrize('kind', ['mpl:scatter', 'mpl:hist', 'mpl:boxplot', 'mpl:violinplot'])
def test_merged_view_restores_legacy_options_and_native_mode(window, kind):
    dialog = window.plot_window
    x, y = [c.id for c in window.model.project.columns[:2]]
    options = {'orientation': 'horizontal'} if kind in ('mpl:boxplot', 'mpl:violinplot') else {}
    legacy = PlotSpec(kind=kind, library_bindings={controls_for(kind).parameters[0]: x},
                      x_id=x, y_id=y, library_options=json.dumps(options))
    if kind == 'mpl:scatter':
        legacy.library_bindings['y'] = y
    dialog.spec = PlotSpec.from_dict(legacy.to_dict())
    dialog.load_controls()
    assert sum(dialog.kind.itemData(i) in KINDS for i in range(dialog.kind.count())) == len(KINDS) == 32
    assert dialog.kind.findData(kind) == -1
    assert dialog.kind.currentData() == MERGED_KINDS[kind]
    assert dialog.plot_mode.currentData() is True
    render(dialog)
    saved = dialog.read_controls().to_dict()
    assert saved['kind'] == kind
    assert all(json.loads(saved['library_options'])[key] == value for key, value in options.items())
    dialog.spec = PlotSpec.from_dict(saved)
    dialog.load_controls()
    render(dialog)
    dialog.plot_mode.setCurrentIndex(0)
    render(dialog)
    assert dialog.spec.kind == MERGED_KINDS[kind]
    dialog.plot_mode.setCurrentIndex(1)
    render(dialog)
    assert all(json.loads(dialog.spec.library_options)[key] == value for key, value in options.items())


def render(dialog):
    dialog.timer.stop()
    dialog.redraw()
    if dialog.result is None:
        render_plot(dialog.owner.model.project, dialog.scope_ids(), dialog.spec)
    assert dialog.result is not None, dialog.status.text()


def test_legacy_log_line_opens_as_one_editable_line_view(window):
    dialog = window.plot_window
    x, y = [c.id for c in window.model.project.columns[:2]]
    for kind in ('mpl:semilogx', 'mpl:semilogy', 'mpl:loglog'):
        dialog.spec = PlotSpec(kind=kind, library_bindings={'x': x, 'y': y})
        dialog.load_controls()
        assert dialog.kind.currentData() == 'mpl:plot'
        assert dialog.kind.findData(kind) == -1
        assert dialog.x_form.fields['scale'][0].isEnabled()
        assert dialog.y_form.fields['scale'][0].isEnabled()
        render(dialog)
        assert dialog.figure.axes[0].get_xscale() == ('log' if kind != 'mpl:semilogy' else 'linear')
        assert dialog.figure.axes[0].get_yscale() == ('log' if kind != 'mpl:semilogx' else 'linear')
        dialog.x_form.load({'scale': 'linear'})
        dialog.y_form.load({'scale': 'linear'})
        render(dialog)
        assert PlotSpec.from_dict(dialog.read_controls().to_dict()).kind == 'mpl:plot'
        assert dialog.figure.axes[0].get_xscale() == dialog.figure.axes[0].get_yscale() == 'linear'


@pytest.mark.parametrize('kind', ['box', 'violin'])
def test_category_axis_disables_numeric_fields_and_restores_them_for_scatter(window, kind):
    dialog = window.plot_window
    group = window.model.project.columns[-1].id
    dialog.style.load({'color_by': group})
    choose(dialog, kind)
    render(dialog)
    for key in ('minimum', 'maximum', 'scale', 'log_base', 'major_step',
                'minor_divisions', 'number_format'):
        assert not dialog.y_form.fields[key][0].isEnabled(), key
    assert dialog.y_form.fields['label'][0].isEnabled()
    labels = [text.get_text() for text in dialog.figure.axes[0].get_yticklabels()]
    assert set(labels) == {'G0', 'G1'}
    choose(dialog, 'scatter')
    render(dialog)
    for key in ('minimum', 'maximum', 'scale', 'major_step', 'minor_divisions', 'number_format'):
        assert dialog.y_form.fields[key][0].isEnabled(), key
    dialog.y_form.fields['maximum'][0].setText('10')
    render(dialog)
    assert dialog.figure.axes[0].get_ylim()[1] == 10


@pytest.mark.parametrize('family', PAIRED_KINDS)
def test_paired_views_restore_switch_and_save_both_legacy_variants(window, family):
    dialog = window.plot_window
    x, y, z = [c.id for c in window.model.project.columns[:3]]
    commands = [kind for _, kind in PAIRED_KINDS[family]]
    for index, kind in enumerate(commands):
        if family == 'mpl:bar':
            options = {'width' if index == 0 else 'height': .4}
        elif family == 'mpl:fill_between':
            options = {'y2' if index == 0 else 'x2': .25, 'interpolate': True}
        else:
            options = {'levels': [2, 3, 4]}
        original = PlotSpec(kind=kind, x_id=x, y_id=y, z_id=z, track_ids=[x, y, z],
                            library_options=json.dumps(options),
                            x_axis=AxisSpec(label='X label'))
        dialog.spec = PlotSpec.from_dict(original.to_dict())
        dialog.load_controls()
        assert sum(dialog.kind.itemData(i) in KINDS for i in range(dialog.kind.count())) == 32
        assert dialog.kind.currentData() == family
        assert dialog.variant.currentData() == kind
        assert dialog.variant.count() == 2
        assert dialog.kind.findData(commands[1]) == -1
        render(dialog)
        saved = dialog.read_controls().to_dict()
        assert saved['kind'] == kind
        assert all(json.loads(saved['library_options'])[key] == value for key, value in options.items())
        dialog.spec = PlotSpec.from_dict(saved)
        dialog.load_controls()
        render(dialog)
        dialog.variant.setCurrentIndex(1 - index)
        render(dialog)
        assert dialog.spec.kind == commands[1 - index]
        if family in ('mpl:bar', 'mpl:fill_between'):
            assert (dialog.spec.x_id, dialog.spec.y_id) == (y, x)
            assert dialog.spec.y_axis.label == 'X label'
        else:
            assert json.loads(dialog.spec.library_options)['levels'] == [2, 3, 4]
        dialog.variant.setCurrentIndex(index)
        render(dialog)
        assert dialog.spec.kind == kind
        assert (dialog.spec.x_id, dialog.spec.y_id) == (x, y)
        assert all(json.loads(dialog.spec.library_options)[key] == value for key, value in options.items())


def test_common_axes_replace_old_bindings_without_a_library_tab(window):
    dialog = window.plot_window
    x, y, z = [c.id for c in window.model.project.columns[:3]]
    dialog.spec = PlotSpec(kind='mpl:plot', library_bindings={'x': x, 'y': y})
    dialog.load_controls()
    assert dialog.x.currentData() == x and dialog.y.currentData() == y
    assert dialog.y.isEnabled()
    assert not any(dialog.tabs.tabText(i) == 'Библиотека' for i in range(dialog.tabs.count()))
    assert not dialog.expert_options.isChecked()
    dialog.data.load({'x_id': z})
    render(dialog)
    assert dialog.spec.library_bindings == {'x': z, 'y': y}
    np.testing.assert_array_equal(dialog.figure.axes[0].lines[0].get_xdata(), [r.values[z] for r in window.model.project.rows])


@pytest.mark.parametrize('kind,option,value', [('mpl:step', 'where', 'post'),
    ('mpl:errorbar', 'capsize', 4.), ('mpl:imshow', 'origin', 'lower'),
    ('mpl3d:plot_surface', 'rcount', 20), ('mpl:contourf', 'levels', 12)])
def test_kind_parameters_are_typed_and_export_without_json(window, tmp_path, kind, option, value):
    dialog = window.plot_window
    ids = [c.id for c in window.model.project.columns[:4]]
    choose(dialog, kind)
    dialog.data.load({'x_id': ids[0], 'y_id': ids[1], 'z_id': ids[2]})
    for i in range(3):
        dialog.tracks.item(i).setCheckState(Qt.CheckState.Checked)
    if kind == 'mpl:errorbar':
        dialog.library_form.load({'yerr': ids[3]})
    dialog.kind_options.load({option: value})
    render(dialog)
    assert json.loads(dialog.spec.library_options)[option] == value
    assert not dialog.expert_options.isChecked()
    from autopyro.plotting import save_plot
    path = tmp_path / 'view.svg'
    save_plot(path, window.model.project, dialog.scope_ids(), PlotSpec.from_dict(dialog.spec.to_dict()))
    assert '<svg' in path.read_text(encoding='utf-8')
    if kind == 'mpl:imshow':
        assert 'матрицы' in dialog.tracks_label.text()
        assert not dialog.x.isEnabled() and not dialog.y.isEnabled()
        assert dialog.figure.axes[0].images[0].origin == 'lower'
    if kind == 'mpl:step':
        assert dialog.figure.axes[0].lines[0].get_drawstyle() == 'steps-post'
    if kind == 'mpl:errorbar':
        container = dialog.figure.axes[0].containers[0]
        assert container.has_yerr and not container.has_xerr


def test_switching_types_keeps_compatible_axes_and_separate_options(window):
    dialog = window.plot_window
    choose(dialog, 'mpl:step')
    dialog.kind_options.load({'where': 'post'})
    before = (dialog.x.currentData(), dialog.y.currentData())
    choose(dialog, 'mpl:errorbar')
    dialog.kind_options.load({'capsize': 6.})
    render(dialog)
    assert 'where' not in json.loads(dialog.spec.library_options)
    choose(dialog, 'mpl:step')
    assert dialog.kind_options.values()['where'] == 'post'
    assert (dialog.x.currentData(), dialog.y.currentData()) == before


def test_3d_scatter_uses_common_channels_and_axis_roles(window):
    dialog = window.plot_window
    x, y, z, err, group = [c.id for c in window.model.project.columns]
    choose(dialog, 'mpl3d:scatter')
    dialog.data.load({'x_id': x, 'y_id': y, 'z_id': z, 'label_by': group})
    dialog.style.load({'color_by': group, 'shape_by': group, 'size_by': z, 'edge_color': '#ff0000'})
    render(dialog)
    assert dialog.color_by.isEnabled() and dialog.shape_by.isEnabled()
    assert len(dialog.result['artists']) == 2
    assert {channel for channel, _, _ in dialog.category_entries} == {'color', 'shape'}
    assert sum(len(a._autopyro_row_ids) for a in dialog.result['artists']) == 9
    assert len(dialog.figure.axes[0].texts) == 9
    dialog.data.load({'z_id': err})
    render(dialog)
    assert dialog.spec.library_bindings['zs'] == err
    for artist in dialog.result['artists']:
        np.testing.assert_allclose(artist._offsets3d[2], .1)


def test_library_scatter_keeps_numeric_ranges_and_uuid_picking(window):
    dialog = window.plot_window
    x, y, z = [c.id for c in window.model.project.columns[:3]]
    choose(dialog, 'mpl:scatter')
    render(dialog)
    assert len(dialog.result['artists'][0].get_facecolors()) == 1
    assert len(dialog.result['artists'][0].get_sizes()) == 1
    dialog.style.load({'color_by': z, 'color_mode': 'numeric', 'shape_by': y, 'size_by': z})
    render(dialog)
    artist = dialog.result['artists'][0]
    assert artist.norm.vmin == 2 and artist.norm.vmax == 6
    wanted = artist._autopyro_row_ids[0]
    dialog.on_pick(SimpleNamespace(artist=artist, ind=[0]))
    assert dialog.selection_ids() == {wanted}
    dialog.style.load({'color_by': y, 'color_mode': 'category', 'shape_by': '', 'size_by': y})
    render(dialog)
    assert dialog.result['artists'][0].get_array() is None
    assert len(set(dialog.result['artists'][0].get_sizes())) == 3


def test_library_template_remaps_all_roles(window, tmp_path, monkeypatch):
    dialog = window.plot_window
    choose(dialog, 'mpl3d:scatter')
    x, y, z = [c.id for c in window.model.project.columns[:3]]
    dialog.data.load({'x_id': x, 'y_id': y, 'z_id': z})
    render(dialog)
    path = tmp_path / 'style.json'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), 'JSON'))
    dialog.export_spec()
    # An independent project with the same semantics and different UUIDs.
    old = window.model.project
    replacement = build_project([c.name for c in old.columns],
        [[r.values[c.id] for c in old.columns] for r in old.rows], list(range(2, 11)),
        [(i, c.name, c.kind, c.unit) for i, c in enumerate(old.columns)], {})
    window.set_project(replacement)
    window.show_plots()
    dialog = window.plot_window
    dialog.timer.stop()
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(path), 'JSON'))
    dialog.import_spec()
    render(dialog)
    assert dialog.spec.z_id == replacement.columns[2].id
    assert set(dialog.spec.library_bindings.values()) <= {c.id for c in replacement.columns}


@pytest.mark.parametrize('kind', ['mpl:semilogx', 'mpl:semilogy', 'mpl:loglog'])
def test_effective_log_axes_filter_before_counting(window, kind):
    project = build_project(['X', 'Y'], [[-1, 1], [1, -1], [2, 2]], [2, 3, 4],
        [(0, 'X', 'number', ''), (1, 'Y', 'number', '')], {})
    x, y = [c.id for c in project.columns]
    _, result = render_plot(project, [r.id for r in project.rows], PlotSpec(kind=kind, x_id=x, y_id=y))
    expected = 2 if kind == 'mpl:loglog' else 1
    assert result['excluded'][x]['nonpositive_log'] == expected
    assert result['plotted'][x] == 3 - expected


@pytest.mark.parametrize('kind', [kind for kind in library_kinds() if not controls_for(kind).experimental])
def test_every_supported_view_is_configured_through_shared_controls(window, kind):
    dialog = window.plot_window
    choose(dialog, kind)
    x, y, z, err = [c.id for c in window.model.project.columns[:4]]
    dialog.data.load({'x_id': x, 'y_id': y, 'z_id': z})
    contract = controls_for(kind)
    primary = {argument for _, argument in contract.axes}
    for key in contract.parameters:
        if key not in primary and key != 'matrix':
            dialog.library_form.load({key: err})
    if 'matrix' in contract.parameters:
        for i in range(3):
            dialog.tracks.item(i).setCheckState(Qt.CheckState.Checked)
    render(dialog)
    assert not dialog.expert_options.isChecked()
    assert dialog.export_button.isEnabled()
    assert dialog.spec.kind == ('mpl:plot' if kind in ('mpl:semilogx', 'mpl:semilogy', 'mpl:loglog') else kind)


def test_regular_grid_hint_matches_input_validation(window):
    dialog = window.plot_window
    project = window.model.project
    x, y, z = [column.id for column in project.columns[:3]]
    choose(dialog, 'mpl3d:plot_surface')
    dialog.data.load({'x_id': x, 'y_id': y, 'z_id': z})
    assert 'регулярная сетка X–Y' in dialog.kind_note.text()

    ids = [row.id for row in project.rows][:-1]
    with pytest.raises(ValueError, match='полная регулярная сетка X–Y без повторов'):
        render_plot(project, ids, dialog.read_controls())

    choose(dialog, 'mpl3d:plot_trisurf')
    assert 'регулярная сетка' not in dialog.kind_note.text()
    figure, result = render_plot(project, ids, dialog.read_controls())
    assert result['plotted'][x] == len(ids)
    figure.clear()


def test_legacy_custom_option_is_preserved_and_explained(window):
    dialog = window.plot_window
    x, y = [c.id for c in window.model.project.columns[:2]]
    dialog.spec = PlotSpec(kind='mpl:contour', x_id=x, y_id=y,
                           track_ids=[x, y], library_options='{"levels": [1, 2, 3]}')
    dialog.load_controls()
    assert dialog.expert_options.isChecked()
    assert not dialog.kind_options.fields['levels'][0].isEnabled()
    render(dialog)
    assert json.loads(dialog.spec.library_options)['levels'] == [1, 2, 3]


def test_common_legacy_style_options_migrate_to_visible_controls(window):
    dialog = window.plot_window
    x, y = [c.id for c in window.model.project.columns[:2]]
    dialog.spec = PlotSpec(kind='mpl:plot', library_bindings={'x': x, 'y': y},
                           library_options='{"color": "red", "linewidth": 3}')
    dialog.load_controls()
    assert dialog.style.values()['color'] == 'red'
    assert dialog.data.values()['line_width'] == 3
    dialog.style.load({'color': 'green'})
    render(dialog)
    assert dialog.figure.axes[0].lines[0].get_color() == 'green'
    assert 'color' not in json.loads(dialog.spec.library_options)


def test_switching_panel_does_not_rediscover_library_api(window, monkeypatch):
    import autopyro.plot_controls as contracts
    def unexpected(kind):
        raise AssertionError('Library API must not be inspected while switching properties')
    monkeypatch.setattr(contracts, 'library_parameters', unexpected)
    for kind in ('mpl:plot', 'mpl:errorbar', 'mpl:imshow', 'mpl3d:scatter'):
        choose(window.plot_window, kind)
        window.plot_window.read_controls()


def test_project_roundtrip_preserves_shared_3d_data_and_parameters(window, tmp_path):
    from autopyro.core import Project
    dialog = window.plot_window
    choose(dialog, 'mpl3d:plot_surface')
    x, y, z = [c.id for c in window.model.project.columns[:3]]
    dialog.data.load({'x_id': x, 'y_id': y, 'z_id': z})
    dialog.kind_options.load({'rcount': 20})
    render(dialog)
    path = tmp_path / 'surface.apyro'
    window.model.project.save(path)
    window.set_project(Project.load(path))
    window.show_plots()
    dialog = window.plot_window
    render(dialog)
    assert dialog.z.currentData() == z
    assert dialog.kind_options.values()['rcount'] == 20
    assert dialog.spec.library_bindings == {'x': x, 'y': y, 'z': z}


def test_reset_formatting_keeps_additional_data_roles(window):
    dialog = window.plot_window
    choose(dialog, 'mpl:errorbar')
    err = window.model.project.columns[3].id
    dialog.library_form.load({'yerr': err})
    dialog.kind_options.load({'capsize': 6.})
    render(dialog)
    dialog.restore_parameters()
    render(dialog)
    assert dialog.spec.library_bindings['yerr'] == err
    assert json.loads(dialog.spec.library_options)['capsize'] == 0


def test_3d_errorbar_has_explicit_optional_z_uncertainty(window):
    dialog = window.plot_window
    choose(dialog, 'mpl3d:errorbar')
    x, y, z, err = [c.id for c in window.model.project.columns[:4]]
    dialog.data.load({'x_id': x, 'y_id': y, 'z_id': z})
    dialog.library_form.load({'zerr': err})
    render(dialog)
    assert dialog.spec.library_bindings['zerr'] == err
    assert 'xerr' not in dialog.spec.library_bindings and 'yerr' not in dialog.spec.library_bindings
    assert len(dialog.figure.axes[0].collections) == 1
