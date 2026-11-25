import pytest
from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import QDialog, QDialogButtonBox

from autopyro.ui.widgets.spatial import CRSCombo, CRSDialog, configure_crs_combo


@pytest.fixture(autouse=True)
def crs_settings(tmp_path, monkeypatch):
    from autopyro.ui.widgets import spatial
    store = QSettings(str(tmp_path / 'crs.ini'), QSettings.Format.IniFormat)
    monkeypatch.setattr(spatial, 'settings', lambda: store)
    return store


def picker(qtbot):
    combo = CRSCombo()
    qtbot.addWidget(combo)
    configure_crs_combo(combo, 'EPSG:4326')
    dialog = CRSDialog(combo)
    qtbot.addWidget(dialog)
    return combo, dialog


def search(dialog, text):
    dialog.search.setText(text)
    dialog.timer.stop()
    dialog.filter_tree()
    return [leaf for leaf, _ in dialog.leaves if not leaf.isHidden()]


def test_crs_search_by_code_name_and_groups(qtbot):
    combo, dialog = picker(qtbot)
    matches = search(dialog, 'EPSG:32631')
    assert len(matches) == 1
    leaf = matches[0]
    assert leaf.data(0, Qt.ItemDataRole.UserRole) == 'EPSG:32631'
    assert leaf.parent().text(0) == 'WGS 84'
    assert leaf.parent().parent().text(0) == 'UTM'
    assert any(leaf.data(0, Qt.ItemDataRole.UserRole) == 'EPSG:4326' for leaf in search(dialog, 'wgs84 4326'))
    assert any(leaf.data(0, Qt.ItemDataRole.UserRole) == 'EPSG:5940' for leaf in search(dialog, 'Russia Polar'))
    assert search(dialog, 'no such reference system 123456789') == []
    assert not dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    search(dialog, '')
    assert not any(dialog.tree.topLevelItem(i).isHidden() for i in range(dialog.tree.topLevelItemCount()))
    assert combo.currentData() == 'EPSG:4326'


def test_crs_groups_cannot_be_accepted_and_custom_crs_is_preserved(qtbot):
    combo, dialog = picker(qtbot)
    dialog.tree.setCurrentItem(dialog.tree.topLevelItem(0))
    dialog.accept()
    assert dialog.result() != QDialog.DialogCode.Accepted
    dialog.reject()
    assert combo.currentData() == 'EPSG:4326'
    from pyproj import CRS
    custom = CRS.from_proj4('+proj=aeqd +lat_0=60 +lon_0=20 +datum=WGS84 +units=m').to_wkt()
    combo.addItem('Local azimuthal CRS', custom)
    combo.setCurrentIndex(combo.findData(custom))
    combo.removeItem(combo.findData('auto'))
    custom_dialog = CRSDialog(combo)
    qtbot.addWidget(custom_dialog)
    assert all(leaf.data(0, Qt.ItemDataRole.UserRole) != 'auto' for leaf, _ in custom_dialog.leaves)
    custom_dialog.accept()
    assert custom_dialog.selected_crs == custom


def test_combo_dialog_applies_selection_and_cancel_keeps_it(qtbot, monkeypatch):
    combo = CRSCombo()
    qtbot.addWidget(combo)
    configure_crs_combo(combo, 'EPSG:4326')
    def choose(dialog):
        leaf = search(dialog, 'EPSG:5940')[0]
        dialog.tree.setCurrentItem(leaf)
        dialog.accept()
        return dialog.result()
    monkeypatch.setattr(CRSDialog, 'exec', choose)
    combo.showPopup()
    assert combo.currentData() == 'EPSG:5940'
    monkeypatch.setattr(CRSDialog, 'exec', lambda self: QDialog.DialogCode.Rejected)
    combo.showPopup()
    assert combo.currentData() == 'EPSG:5940'


def test_favorites_persist_remove_and_search_without_affecting_recent(qtbot, crs_settings):
    from autopyro.ui.widgets.spatial import FAVORITE_CRS_KEY, RECENT_CRS_KEY
    combo, dialog = picker(qtbot)
    leaf = search(dialog, 'EPSG:5940')[0]
    dialog.tree.setCurrentItem(leaf)
    dialog.favorite_button.click()
    assert crs_settings.value(FAVORITE_CRS_KEY) == ['EPSG:5940']
    assert dialog.shortcut_groups[0].child(0).data(0, Qt.ItemDataRole.UserRole) == 'EPSG:5940'
    assert '1' in dialog.matches.text()  # Duplicate shortcuts do not inflate the match count.
    dialog.reject()
    assert crs_settings.value(RECENT_CRS_KEY, []) == []
    reopened = CRSDialog(combo)
    qtbot.addWidget(reopened)
    assert reopened.favorites == ['EPSG:5940']
    search(reopened, 'EPSG:5940')
    reopened.tree.setCurrentItem(reopened.shortcut_groups[0].child(0))
    reopened.favorite_button.click()
    assert crs_settings.value(FAVORITE_CRS_KEY) == []


def test_recent_crs_are_confirmed_unique_newest_first_and_bounded(qtbot, crs_settings):
    from autopyro.ui.widgets.spatial import RECENT_CRS_KEY
    combo = CRSCombo()
    qtbot.addWidget(combo)
    codes = [f'EPSG:{32600 + i}' for i in range(1, 13)]
    combo.addItem('Automatic', 'auto')
    for code in codes:
        combo.addItem(code, code)
    for code in codes + [codes[-2]]:
        combo.setCurrentIndex(combo.findData(code))
        dialog = CRSDialog(combo)
        qtbot.addWidget(dialog)
        dialog.accept()
    recent = crs_settings.value(RECENT_CRS_KEY)
    assert len(recent) == len(set(recent)) == 10
    assert recent[:2] == [codes[-2], codes[-1]]
    assert codes[0] not in recent
    combo.setCurrentIndex(0)
    dialog = CRSDialog(combo)
    qtbot.addWidget(dialog)
    assert not dialog.favorite_button.isEnabled()
    dialog.accept()
    assert crs_settings.value(RECENT_CRS_KEY) == recent
    persisted = QSettings(crs_settings.fileName(), QSettings.Format.IniFormat)
    assert persisted.value(RECENT_CRS_KEY) == recent


def test_saved_custom_crs_is_available_in_new_selector(qtbot, crs_settings):
    from pyproj import CRS
    from autopyro.ui.widgets.spatial import FAVORITE_CRS_KEY, RECENT_CRS_KEY
    custom = CRS.from_proj4('+proj=aeqd +lat_0=60 +lon_0=20 +datum=WGS84 +units=m').to_wkt()
    crs_settings.setValue(FAVORITE_CRS_KEY, [custom, 'invalid CRS', custom])
    crs_settings.setValue(RECENT_CRS_KEY, [custom])
    combo, dialog = picker(qtbot)
    assert dialog.favorites == [custom]
    assert combo.findData(custom) >= 0
    dialog.tree.setCurrentItem(dialog.shortcut_groups[0].child(0))
    dialog.accept()
    assert dialog.selected_crs == custom
