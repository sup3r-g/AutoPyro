"""Cached color presentations follow registry changes and keep editors independent."""
from matplotlib.colors import get_named_colors_mapping

from autopyro.ui.widgets.forms import ColorCombo
from autopyro.ui.widgets.style_labels import color_label


def test_redefined_registered_color_refreshes_label_and_swatch(qtbot, monkeypatch):
    registry = get_named_colors_mapping()
    name = 'autopyro-cache-color'
    monkeypatch.setitem(registry, name, '#123456')
    first = ColorCombo(name)
    qtbot.addWidget(first)
    first_image = first.itemIcon(first.currentIndex()).pixmap(24, 14).toImage()
    monkeypatch.setitem(registry, name, '#fedcba')
    second = ColorCombo(name)
    qtbot.addWidget(second)
    assert second.currentText() == color_label(name)
    assert first.currentText() != second.currentText()
    assert first_image != second.itemIcon(second.currentIndex()).pixmap(24, 14).toImage()
    monkeypatch.delitem(registry, name)
    third = ColorCombo('red')
    qtbot.addWidget(third)
    assert third.findData(name) == -1


def test_cached_catalog_does_not_share_custom_entries_or_selection(qtbot):
    first, second = ColorCombo('red'), ColorCombo('blue')
    qtbot.addWidget(first)
    qtbot.addWidget(second)
    first.setText('#123abc')
    assert first.text() == '#123abc'
    assert second.findData('#123abc') == -1
    assert second.text() == 'blue'
    first.removeItem(first.findData('blue'))
    assert second.findData('blue') >= 0


def test_stratigraphic_swatch_saves_source_hex(qtbot):
    editor = ColorCombo('red')
    qtbot.addWidget(editor)
    index = editor.findData('#FFF2D8')
    assert index >= 0
    assert '018/11' in editor.itemText(index)
    assert not editor.itemIcon(index).isNull()
    editor.setCurrentIndex(index)
    assert editor.text() == '#FFF2D8'
