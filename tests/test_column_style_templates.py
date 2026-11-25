"""Appearance presets must be portable with the project and cheap to apply."""

from copy import deepcopy

import pytest
from PySide6.QtWidgets import QInputDialog, QMessageBox

from autopyro.core import Project
from autopyro.stratigraphy import COLUMN_STYLE
from autopyro.ui.dialogs.stratigraphy import StratigraphyDialog
from autopyro.ui.widgets.stratigraphy_templates import CONTEXT_FIELDS, TEMPLATES_KEY, template_style
from autopyro.ui.window import MainWindow


@pytest.fixture
def editor(qtbot):
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(Project())
    owner.may_replace = lambda: True
    dialog = StratigraphyDialog(owner)
    qtbot.addWidget(dialog)
    return owner, dialog


def save_as(dialog, monkeypatch, name):
    monkeypatch.setattr(QInputDialog, 'getText', lambda *args, **kwargs: (name, True))
    dialog.style_templates.save_button.click()


def test_save_without_intervals_and_project_roundtrip(editor, qtbot, monkeypatch, tmp_path):
    owner, dialog = editor
    form = dialog.column_style_form
    form.fields['label_size'][0].setValue(14)
    form.load({'legend_kwargs': {'labelspacing': 3.5}, 'x_ticks_dict': {'fine': .5, 'coarse': 2},
               'grain_brackets': {'sizes': ['fine', 'coarse']}})
    def unexpected(*args, **kwargs):
        pytest.fail('Saving a preset must not refresh project data or render stratapy')
    monkeypatch.setattr(owner.model, 'refresh', unexpected)
    save_as(dialog, monkeypatch, '  Для печати  ')
    stored = owner.model.project.view[TEMPLATES_KEY]['Для печати']
    assert stored['label_size'] == 14
    assert stored['x_ticks_dict'] == {'fine': .5, 'coarse': 2}
    assert not CONTEXT_FIELDS & stored.keys()
    assert owner.dirty
    assert not dialog._has_column_preview
    for button in (dialog.style_templates.save_button, dialog.style_templates.apply_button,
                   dialog.style_templates.delete_button):
        assert not button.icon().isNull()
    path = tmp_path / 'templates.apyro'
    owner.model.project.save(path)
    restored = Project.load(path)
    assert restored.view[TEMPLATES_KEY] == owner.model.project.view[TEMPLATES_KEY]
    owner.set_project(restored)
    reopened = StratigraphyDialog(owner)
    qtbot.addWidget(reopened)
    reopened.style_templates.combo.setCurrentIndex(1)
    reopened.style_templates.apply_button.click()
    assert reopened.style_values()['label_size'] == 14


def test_apply_preserves_section_data_and_emits_once(editor, monkeypatch):
    owner, dialog = editor
    form = dialog.column_style_form
    save_as(dialog, monkeypatch, 'Default')
    form.load({'title': 'My section', 'depth_min': 10, 'depth_max': 80,
               'hierarchy_levels': ['GROUP'], 'comparison_wells': ['B'],
               'lithologies': {'custom_rock_1': ['607', '#e0b761', 'My sandstone']}})
    form.fields['label_size'][0].setValue(18)
    form.fields['display_mode'][0].setCurrentIndex(form.fields['display_mode'][0].findData('default'))
    before = dialog.style_values()
    draft = deepcopy(dialog.draft)
    view = deepcopy(owner.model.project.view)
    calls = []
    form.edited.connect(lambda: calls.append(True))
    dialog.style_templates.apply_button.click()
    after = dialog.style_values()
    assert after['label_size'] == COLUMN_STYLE['label_size']
    assert after['display_mode'] == 'log'
    assert not form.fields['grain_preset'][0].isEnabled()
    assert all(after[key] == before[key] for key in CONTEXT_FIELDS)
    assert dialog.draft == draft
    assert owner.model.project.view == view  # Applying is a local appearance draft.
    assert calls == [True]


def test_apply_repairs_invalid_appearance_without_reading_it(editor, monkeypatch):
    _, dialog = editor
    save_as(dialog, monkeypatch, 'Valid')
    scale = dialog.column_style_form.scale_editor
    scale.mode.setCurrentIndex(scale.mode.findData('custom'))
    scale.ticks.add_row(values=['Fine', 'broken'])
    with pytest.raises(ValueError):
        dialog.style_values()
    dialog.style_templates.apply_button.click()
    assert dialog.style_values()['x_ticks_dict'] is None


def test_overwrite_cancel_and_delete_preserve_current_style(editor, monkeypatch):
    owner, dialog = editor
    save_as(dialog, monkeypatch, 'Publication')
    original = deepcopy(owner.model.project.view[TEMPLATES_KEY])
    dialog.column_style_form.fields['label_size'][0].setValue(16)
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.No)
    save_as(dialog, monkeypatch, 'Publication')
    assert owner.model.project.view[TEMPLATES_KEY] == original
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.Yes)
    save_as(dialog, monkeypatch, 'Publication')
    assert owner.model.project.view[TEMPLATES_KEY]['Publication']['label_size'] == 16
    current = dialog.style_values()
    dialog.style_templates.delete_button.click()
    assert owner.model.project.view[TEMPLATES_KEY] == {}
    assert dialog.style_values() == current
    assert not dialog.style_templates.apply_button.isEnabled()


def test_cancel_empty_name_and_invalid_saved_template(editor, monkeypatch):
    owner, dialog = editor
    monkeypatch.setattr(QInputDialog, 'getText', lambda *args, **kwargs: ('Cancelled', False))
    dialog.style_templates.save_button.click()
    save_as(dialog, monkeypatch, '   ')
    assert TEMPLATES_KEY not in owner.model.project.view
    owner.model.project.view[TEMPLATES_KEY] = {'Invalid': {'alpha': 5}}
    dialog.style_templates.refresh('Invalid')
    before = dialog.style_values()
    dialog.style_templates.apply_button.click()
    assert dialog.style_values() == before
    assert dialog.style_templates.feedback.text()


def test_old_template_defaults_and_copy_isolation():
    options = {'legend_kwargs': {'labelspacing': 3}}
    style = template_style(options)
    style['legend_kwargs']['labelspacing'] = 4
    assert options['legend_kwargs']['labelspacing'] == 3
    assert style['width_mm'] == COLUMN_STYLE['width_mm']


def test_apply_batches_preview_and_keeps_interval_history(editor, qtbot, monkeypatch):
    from autopyro.ui.dialogs import stratigraphy_column as module

    owner, dialog = editor
    dialog.reference.setText('MD RKB')
    dialog.add_interval(interval={'id': 'one', 'wellbore': 'A', 'level': 'FORMATION',
                                 'name': 'Layer', 'top': 0, 'bottom': 30, 'color': '#aabbcc'})
    dialog.show()
    dialog.show_column()
    dialog.column_preview_timer.setInterval(100)
    save_as(dialog, monkeypatch, 'Default')
    dialog.column_figure.axes[0].set_ylim(25, 10)
    dialog.interval_selection.select_ids({'one'})
    count = dialog.draft_undo.count()
    calls = []
    render = module.render_columns
    def record(*args, **kwargs):
        calls.append(True)
        return render(*args, **kwargs)
    monkeypatch.setattr(module, 'render_columns', record)
    dialog.column_style_form.fields['label_size'][0].setValue(16)
    dialog.style_templates.apply_button.click()
    dialog.style_templates.apply_button.click()
    assert not calls
    qtbot.waitUntil(lambda: dialog.export_column_button.isEnabled(), timeout=3000)
    assert calls == [True]
    assert dialog.column_figure.axes[0].get_ylim() == (25, 10)
    assert dialog.interval_selection.selected_ids() == {'one'}
    assert dialog.draft_undo.count() == count
