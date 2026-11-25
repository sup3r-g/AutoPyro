"""Discover and load new compiled languages without a Python translation map."""
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QSettings, Qt, QTranslator

from autopyro.ui import i18n
from autopyro.ui.i18n import catalogs
from autopyro.ui.dialogs.settings import SettingsDialog


@pytest.fixture
def isolated_catalogs(tmp_path, monkeypatch, qapp):
    directory = tmp_path / "translations"
    directory.mkdir()
    store = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(i18n, "settings", lambda: store)
    monkeypatch.setattr(i18n, "catalog_directories", lambda: (directory,))
    yield directory
    i18n.save_language("ru", store)
    i18n.initialize(store)


def compile_catalog(directory, code, label, save_translation):
    path = directory / f"autopyro_{code}.ts"
    root = ET.Element("TS", version="2.1", language=code, sourcelanguage="ru_RU")
    for context_name, messages in {
        "AutoPyroLanguage": {"native_name": label},
        "AutoPyro": {"Сохранить": save_translation, "Вход {0}: неподдерживаемая единица": "Input {0}: custom unit"},
    }.items():
        context = ET.SubElement(root, "context")
        ET.SubElement(context, "name").text = context_name
        for source, target in messages.items():
            entry = ET.SubElement(context, "message")
            ET.SubElement(entry, "source").text = source
            ET.SubElement(entry, "translation").text = target
    if code == "de":
        entry = ET.SubElement(context, "message", numerus="yes")
        ET.SubElement(entry, "source").text = "%n проба"
        translation = ET.SubElement(entry, "translation")
        for value in ("%n Probe", "%n Proben"):
            ET.SubElement(translation, "numerusform").text = value
    catalogs.write_ts(path, root)
    catalogs.validate(path)
    subprocess.run([catalogs.lrelease(), str(path), "-qm", str(path.with_suffix(".qm"))], check=True, capture_output=True)
    path.unlink()  # Runtime discovery and translation must require only the .qm.
    return path.with_suffix(".qm")


@pytest.mark.parametrize("code,label,translation,direction", [
    ("de", "Deutsch", "Speichern", Qt.LayoutDirection.LeftToRight),
    ("ar", "العربية", "حفظ", Qt.LayoutDirection.RightToLeft),
])
def test_new_language_qm_is_detected_and_used(isolated_catalogs, qtbot, qapp, code, label, translation, direction):
    compile_catalog(isolated_catalogs, code, label, translation)
    assert i18n.available_languages()[code] == label
    i18n.save_language(code)
    i18n.initialize()
    assert i18n.language() == code
    assert QCoreApplication.translate("AutoPyro", "Сохранить") == translation
    assert i18n.tr("Сохранить") == translation
    assert i18n.message("Вход S2: неподдерживаемая единица") == "Input S2: custom unit"
    assert i18n.tr("Нет такого перевода") == "Нет такого перевода"
    assert qapp.layoutDirection() == direction
    dialog = SettingsDialog()
    qtbot.addWidget(dialog)
    assert dialog.language.currentData() == code
    assert dialog.language.currentText() == label
    if code == "de":
        assert i18n.tr("%n проба", n=1) == "1 Probe"
        assert i18n.tr("%n проба", n=3) == "3 Proben"


def test_invalid_or_removed_catalog_falls_back(isolated_catalogs):
    path = compile_catalog(isolated_catalogs, "de", "Deutsch", "Speichern")
    i18n.save_language("de")
    path.write_bytes(b"not a Qt catalog")
    assert "de" not in i18n.available_languages()
    i18n.initialize()
    assert i18n.language() == "ru"
    assert i18n.tr("Сохранить") == "Сохранить"


def test_catalog_tool_creates_updates_and_validates_placeholders(tmp_path):
    path = tmp_path / "autopyro_fr.ts"
    catalogs.create(path, "fr", "Français", {"Источник: {0}": False, "%n проба": True})
    catalogs.update(path, {"Источник: {0}": False, "%n проба": True, "Новый текст": False})
    catalogs.validate(path)
    root = ET.parse(path).getroot()
    context = next(c for c in root.findall("context") if c.findtext("name") == "AutoPyro")
    assert len(context.findall("message")) == 3
    entry = context.find("message")
    entry.find("translation").attrib.clear()
    entry.find("translation").text = "Source sans paramètre"
    catalogs.write_ts(path, root)
    with pytest.raises(ValueError, match="placeholders"):
        catalogs.validate(path)


def test_bundled_catalog_is_valid_and_packaged():
    directory = Path(i18n.__file__).with_name("translations")
    for code in ("ru", "en"):
        catalogs.validate(directory / f"autopyro_{code}.ts")
        assert (directory / f"autopyro_{code}.qm").is_file()
    assert not Path(i18n.__file__).with_name("en.py").exists()


def test_russian_catalog_is_complete_and_installed(tmp_path, monkeypatch, qapp):
    directory = Path(i18n.__file__).with_name("translations")
    monkeypatch.setattr(i18n, "catalog_directories", lambda: (directory,))
    store = QSettings(str(tmp_path / "russian.ini"), QSettings.Format.IniFormat)
    i18n.save_language("ru", store)
    i18n.initialize(store)
    translator = next(t for t in i18n._translators if Path(t.filePath()).name == "autopyro_ru.qm")
    assert isinstance(translator, QTranslator)
    assert translator.translate("AutoPyroLanguage", "native_name") == "Русский"
    source_messages = catalogs.sources()
    root = ET.parse(directory / "autopyro_ru.ts").getroot()
    context = next(c for c in root.findall("context") if c.findtext("name") == i18n.CONTEXT)
    entries = {entry.findtext("source"): entry for entry in context.findall("message")}
    assert entries.keys() == source_messages.keys()
    for source, entry in entries.items():
        assert entry.findtext("translation") == source
        assert entry.find("translation").get("type") != "unfinished"
        assert translator.translate(i18n.CONTEXT, source) == source
    assert i18n.tr("Источник: {0}", "данные.csv") == "Источник: данные.csv"


def test_russian_catalog_updates_keep_source_text_and_existing_edits(tmp_path):
    path = tmp_path / "autopyro_ru.ts"
    catalogs.create(path, "ru", "Русский", {"Сохранить": False})
    root = ET.parse(path).getroot()
    context = next(c for c in root.findall("context") if c.findtext("name") == i18n.CONTEXT)
    context.find("message/translation").text = "Сохранить изменения"
    catalogs.write_ts(path, root)
    catalogs.update(path, {"Сохранить": False, "Новый текст": False, "%n проба": True})
    context = next(c for c in ET.parse(path).findall("context") if c.findtext("name") == i18n.CONTEXT)
    entries = {entry.findtext("source"): entry for entry in context.findall("message")}
    assert entries["Сохранить"].findtext("translation") == "Сохранить изменения"
    assert entries["Новый текст"].findtext("translation") == "Новый текст"
    # Numerus forms need linguistic editing instead of copying a singular form.
    assert entries["%n проба"].find("translation").get("type") == "unfinished"
    catalogs.validate(path)
