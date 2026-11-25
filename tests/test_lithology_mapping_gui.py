"""Import mapping, review, remembered rules and one-step draft undo."""

from copy import deepcopy
import json

import pytest
from PySide6.QtWidgets import QFileDialog, QInputDialog

from autopyro.core import Project
from autopyro.lithology_matching import RULES_KEY, normalize_rock
from autopyro.ui.dialogs.stratigraphy import StratigraphyDialog
from autopyro.ui.widgets.lithology import LithologyCatalog, LithologyPicker
from autopyro.ui.widgets.lithology_mapping import LithologyMappingDialog
from autopyro.ui.widgets.lithology_rules import LithologyRulesDialog
from autopyro.ui.window import MainWindow

from test_lithology_matching import imported


@pytest.mark.parametrize('accept', [False, True])
def test_edit_and_delete_filtered_project_rules(dialog, monkeypatch, tmp_path, accept):
    project = dialog.owner.model.project
    project.view[RULES_KEY] = {'local clay': '620', 'local sand': '607', 'obsolete': 'missing-code'}
    original = dict(project.view[RULES_KEY])
    def review(editor):
        assert editor.model.rowCount() == 3
        editor.search.setText('missing-code')
        assert editor.proxy.rowCount() == 1
        editor.table.selectRow(0)
        editor.delete_button.click()
        assert editor.proxy.rowCount() == 0
        editor.search.setText('local sand')
        editor.table.selectRow(0)

        def choose(picker):
            picker.selected_code = '607*'
            return picker.DialogCode.Accepted

        monkeypatch.setattr(LithologyPicker, 'exec', choose)
        editor.edit_button.click()
        assert project.view[RULES_KEY] == original
        return editor.DialogCode.Accepted if accept else editor.DialogCode.Rejected

    monkeypatch.setattr(LithologyRulesDialog, 'exec', review)
    dialog.manage_lithology_rules()
    expected = {'local clay': '620', 'local sand': '607*'} if accept else original
    assert project.view[RULES_KEY] == expected
    assert dialog.lithology_matcher()[1].match('Local sand') == (expected['local sand'], 'saved')
    path = tmp_path / 'rules.apyro'
    project.save(path)
    assert Project.load(path).view[RULES_KEY] == expected


def test_rule_dictionary_exchange_resolves_conflicts_before_project_apply(dialog, monkeypatch, tmp_path, qtbot):
    from PySide6.QtCore import Qt
    from autopyro.ui.widgets.interval_review import IntervalReviewDialog

    project = dialog.owner.model.project
    project.view[RULES_KEY] = {'sand': '607', 'clay': '620', 'coal': '658'}
    original = dict(project.view[RULES_KEY])
    path = tmp_path / 'rules.json'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), ''))
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(path), ''))

    def review(preview):
        model = preview.table.model()
        assert model.changes() == {'limestone': '627', 'skip': ''}
        preview.table.setCurrentIndex(model.index(0, 4))
        qtbot.keyClick(preview.table, Qt.Key.Key_Space)
        assert model.changes()['sand'] == '607*'
        return preview.DialogCode.Accepted

    def exchange(editor):
        editor.export_rules()
        payload = json.loads(path.read_text(encoding='utf-8'))
        assert payload['rules'] == original
        payload['rules'] = {' SAND ': '607*', 'clay': '607', 'coal': '658',
                            'limestone': '627', 'skip': '', 'absent': 'missing-code'}
        path.write_text(json.dumps(payload), encoding='utf-8')
        monkeypatch.setattr(IntervalReviewDialog, 'exec', lambda self: self.DialogCode.Rejected)
        editor.import_rules()
        assert editor.rules() == original
        monkeypatch.setattr(IntervalReviewDialog, 'exec', review)
        editor.import_rules()
        assert project.view[RULES_KEY] == original
        return editor.DialogCode.Accepted

    monkeypatch.setattr(LithologyRulesDialog, 'exec', exchange)
    dialog.manage_lithology_rules()
    assert project.view[RULES_KEY] == {**original, 'sand': '607*', 'limestone': '627', 'skip': ''}


def test_rule_dictionary_rejects_colliding_normalized_names():
    from autopyro.lithology_matching import RULES_FORMAT, decode_rules

    with pytest.raises(ValueError):
        decode_rules({'format': RULES_FORMAT, 'version': 1, 'rules': {' Sand ': '607', 'sand': '608'}})


def test_import_custom_lithology_requires_review(dialog):
    from PySide6.QtCore import Qt
    from autopyro.ui.widgets.lithology_rules import RuleImportModel

    catalog = LithologyCatalog({'my_sand': ['607', '#e0b761', 'My sandstone']})
    model = RuleImportModel({}, {'local': 'my_sand'}, catalog, dialog)
    assert model.changes() == {}
    assert model.setData(model.index(0, 4), Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole)
    assert model.changes() == {'local': 'my_sand'}


@pytest.fixture
def dialog(qtbot):
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(Project())
    owner.may_replace = lambda: True
    dialog = StratigraphyDialog(owner)
    qtbot.addWidget(dialog)
    dialog.reference.setText('MD RKB')
    return dialog


def test_review_picker_maps_all_occurrences_with_native_preview(dialog, qtbot, monkeypatch):
    scheme = imported(['Local', ' LOCAL ', 'Глина'])
    catalog, matcher = dialog.lithology_matcher()
    picker = LithologyMappingDialog(scheme, catalog, matcher, dialog)
    qtbot.addWidget(picker)
    assert picker.model.rowCount() == 2
    picker.table.setCurrentIndex(picker.model.index(0, 2))
    def choose(rock_picker):
        rock_picker.selected_code = '607*'
        return rock_picker.DialogCode.Accepted
    monkeypatch.setattr(LithologyPicker, 'exec', choose)
    picker.choose_button.click()
    assert [item['lithology'] for item in picker.mapping.apply()['intervals']] == ['607*', '607*', '620']
    assert picker.mapping.learned_rules() == {'local': '607*'}
    from PySide6.QtCore import Qt
    assert not picker.model.data(picker.model.index(0, 2), Qt.ItemDataRole.DecorationRole).isNull()
    assert not picker.choose_button.icon().isNull()


def test_import_remembers_rules_and_reuses_them_after_project_reload(dialog, monkeypatch, tmp_path):
    path = tmp_path / 'rocks.csv'
    path.write_text('wellbore_name;unit_name;unit_level;top_depth_m;bottom_depth_m;Порода\n'
                    'A;Upper;FORMATION;0;10;ПСЧ\nA;Lower;FORMATION;10;20;ПСЧ\n'
                    'A;Clay;FORMATION;20;30;Глина\n', encoding='utf-8')
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(path), ''))
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args, **kwargs: (';', True))
    reviews = []
    def review(picker):
        reviews.append(True)
        group = picker.mapping.groups['псч']
        group.code, group.reason = '607', 'manual'
        return picker.DialogCode.Accepted
    monkeypatch.setattr(LithologyMappingDialog, 'exec', review)
    dialog.import_scheme()
    scheme = dialog.read_scheme()
    assert [item['lithology'] for item in scheme['intervals']] == ['607', '607', '620']
    assert scheme['intervals'][0]['raw_cells']['Порода'] == 'ПСЧ'
    assert scheme['intervals'][0]['source_row'] == 2
    assert dialog.owner.model.project.view[RULES_KEY] == {'псч': '607'}
    assert reviews == [True]
    dialog.import_scheme()
    assert reviews == [True]  # The second import needs no manual assignments.
    dialog.show_column()
    assert dialog.export_column_button.isEnabled(), dialog.status.text()
    project_path = tmp_path / 'rules.apyro'
    dialog.owner.model.project.save(project_path)
    restored = Project.load(project_path)
    assert restored.view[RULES_KEY] == {'псч': '607'}


def test_cancel_review_keeps_existing_draft_and_project(dialog, monkeypatch):
    dialog.draft = imported(['607'])
    dialog.load_draft()
    before = dialog.read_scheme()
    view = deepcopy(dialog.owner.model.project.view)
    monkeypatch.setattr(LithologyMappingDialog, 'exec', lambda self: self.DialogCode.Rejected)
    scheme, rules = dialog.import_lithologies(imported(['mystery']))
    assert scheme is None and rules == {}
    assert dialog.read_scheme() == before
    assert dialog.owner.model.project.view == view


def test_manual_mapping_is_one_undo_step_including_provenance(dialog, monkeypatch):
    dialog.draft = imported(['Local', ' LOCAL ', 'Глина'])
    dialog.load_draft()
    before = dialog.read_scheme()
    def review(picker):
        group = picker.mapping.groups[normalize_rock('Local')]
        group.code, group.reason = '627', 'manual'
        return picker.DialogCode.Accepted
    monkeypatch.setattr(LithologyMappingDialog, 'exec', review)
    dialog.map_lithologies()
    mapped = dialog.read_scheme()
    assert [item['lithology'] for item in mapped['intervals']] == ['627', '627', '620']
    assert dialog.draft_undo.count() == 1
    assert mapped['intervals'][0]['lithology_source'] == 'Local'
    dialog.draft_undo.undo()
    assert dialog.read_scheme() == before
    dialog.draft_undo.redo()
    assert dialog.read_scheme() == mapped


def test_unknown_can_be_left_unassigned_and_column_still_builds(dialog, monkeypatch):
    monkeypatch.setattr(LithologyMappingDialog, 'exec', lambda self: self.DialogCode.Accepted)
    mapped, learned = dialog.import_lithologies(imported(['mystery']))
    assert not learned
    dialog.draft = mapped
    dialog.load_draft()
    dialog.show_column()
    assert dialog.export_column_button.isEnabled(), dialog.status.text()
    assert dialog.read_scheme()['intervals'][0]['lithology_source'] == 'mystery'


def test_source_can_be_selected_from_any_imported_column(dialog, qtbot):
    scheme = imported(['Известняк'], field='Описание')
    catalog, matcher = dialog.lithology_matcher()
    picker = LithologyMappingDialog(scheme, catalog, matcher, dialog)
    qtbot.addWidget(picker)
    assert picker.model.rowCount() == 0
    picker.source.setCurrentIndex(picker.source.findData('Описание'))
    assert picker.model.rowCount() == 1
    assert picker.mapping.apply()['intervals'][0]['lithology'] == '627'
