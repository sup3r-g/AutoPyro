"""Catalog correctness, modal selection, and lazy thumbnail behavior."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialogButtonBox, QStyleOptionViewItem
import pytest

from autopyro.ui.widgets.lithology import (
    LithologyCatalog, LithologyDelegate, LithologyModel, LithologyPicker, pattern_thumbnail,
)


def test_catalog_contains_native_variants_without_loading_images(qapp, monkeypatch):
    from stratapy.setup import formatting
    import autopyro.ui.widgets.lithology as module

    def unexpected_load(path):
        pytest.fail("Catalog metadata must not decode pattern images")
    monkeypatch.setattr(module, 'pattern_thumbnail', unexpected_load)
    catalog = LithologyCatalog()
    model = LithologyModel(catalog)
    assert model.rowCount() == len(formatting.lithologies) + 1
    assert catalog.resolve('607').name == 'Massive sandstone'
    assert catalog.resolve('Massive sandstone').code == '607'
    assert catalog.resolve('607*').code == '607*'
    assert catalog.resolve('').code == ''
    assert catalog.resolve('unrecognized') is None
    delegate = LithologyDelegate(catalog)
    option = QStyleOptionViewItem()
    for row in range(model.rowCount()):
        index = model.index(row, 0)
        assert model.data(index, Qt.ItemDataRole.DisplayRole)
        assert delegate.sizeHint(option, index).width() > 0
    assert not catalog._icons


def test_pattern_previews_are_cached_and_colored(qapp, monkeypatch):
    from stratapy.core import LogObject

    def unexpected_plot(*args, **kwargs):
        pytest.fail("Swatches must not invoke matplotlib/stratapy plotting")
    monkeypatch.setattr(LogObject, 'plot', unexpected_plot)
    pattern_thumbnail.cache_clear()
    catalog = LithologyCatalog()
    plain = catalog.icon(catalog.resolve('607'))
    assert not plain.isNull()
    assert pattern_thumbnail.cache_info().misses == 1
    assert catalog.icon(catalog.resolve('607')).cacheKey() == plain.cacheKey()
    colored = catalog.icon(catalog.resolve('607*'))
    assert pattern_thumbnail.cache_info().misses == 1  # Same source image, different colormap.
    assert pattern_thumbnail.cache_info().hits == 1
    image = colored.pixmap(catalog.ICON_SIZE).toImage()
    assert any(image.pixelColor(x, y).red() != image.pixelColor(x, y).blue()
               for x in range(1, image.width() - 1, 4)
               for y in range(1, image.height() - 1, 4))
    catalog.CACHE_LIMIT = 2
    catalog.icon(catalog.resolve('658'))
    assert len(catalog._icons) == 2
    assert '607' not in catalog._icons


def test_picker_search_selection_empty_results_and_clear(qtbot):
    picker = LithologyPicker(LithologyCatalog(), current='Massive sandstone')
    qtbot.addWidget(picker)
    picker.show()
    assert picker.selected_code == '607'
    assert not picker.preview.pixmap().isNull()
    qtbot.keyClicks(picker.search, 'Massive sandstone')
    assert picker.proxy.rowCount() == 2  # Plain and colored variants.
    picker.list.setCurrentIndex(picker.proxy.index(1, 0))
    assert picker.selected_code == '607*'
    picker.search.setText('no such rock')
    assert picker.proxy.rowCount() == 0
    assert not picker.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    picker.accept()
    assert picker.result() == 0
    picker.search.clear()
    picker.list.setCurrentIndex(picker.proxy.index(0, 0))
    assert picker.selected_code == ''
    picker.accept()
    assert picker.result() == picker.DialogCode.Accepted


def test_custom_lithology_has_preview_without_global_catalog_changes(qtbot):
    from stratapy.setup import formatting

    original = formatting.lithologies.copy()
    catalog = LithologyCatalog({'my_sand': ['607', '#e0b761', 'My sandstone']})
    assert formatting.lithologies == original
    picker = LithologyPicker(catalog, current='my_sand')
    qtbot.addWidget(picker)
    assert picker.selected_code == 'my_sand'
    assert 'My sandstone' in picker.description.text()
    assert not picker.preview.pixmap().isNull()
    assert formatting.lithologies == original


def test_unknown_imported_value_is_not_silently_replaced(qtbot):
    picker = LithologyPicker(LithologyCatalog(), current='legacy_unknown')
    qtbot.addWidget(picker)
    assert picker.selected_code == 'legacy_unknown'
    assert not picker.list.currentIndex().isValid()
    assert not picker.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    picker.reject()
    assert picker.selected_code == 'legacy_unknown'
