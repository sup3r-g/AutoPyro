"""Qt translation catalogs; project data, identifiers and units stay unchanged."""

import re
from functools import lru_cache
from pathlib import Path
from string import Formatter

from PySide6.QtCore import (
    QCoreApplication, QLibraryInfo, QLocale, QSettings, QStandardPaths, QTranslator,
)
from PySide6.QtWidgets import QApplication

from .source_patterns import SOURCE_PATTERNS

CONTEXT = "AutoPyro"
SOURCE_LANGUAGE = "ru"
SETTINGS_KEY = "interface/language"
_language = "ru"
_translators = []


def settings():
    return QSettings("AutoPyro", "AutoPyro")


def catalog_directories():
    return (
        Path(__file__).with_name("translations"),
        Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.GenericDataLocation))
        / "AutoPyro" / "translations",
    )


def available_catalogs():
    """Discover valid Qt catalogs; user catalogs override bundled ones."""
    result = {}
    for directory in catalog_directories():
        for path in sorted(directory.glob("autopyro_*.qm")):
            code = path.stem.removeprefix("autopyro_")
            if not re.fullmatch(r"[a-zA-Z]{2,3}(?:_[a-zA-Z0-9]{2,8})*", code):
                continue
            translator = QTranslator()
            if translator.load(str(path)):
                result[code] = path
    return result


def available_languages():
    result = {SOURCE_LANGUAGE: "Русский"}
    for code, path in available_catalogs().items():
        translator = QTranslator()
        translator.load(str(path))
        locale = QLocale(code)
        name = translator.translate("AutoPyroLanguage", "native_name")
        if not name:
            name = locale.nativeLanguageName().capitalize() or code
            if "_" in code and locale.nativeTerritoryName():
                name += f" ({locale.nativeTerritoryName()})"
        result[code] = name
    return result


def preferred_language(store=None):
    value = (store if store is not None else settings()).value(SETTINGS_KEY, "ru")
    return value if isinstance(value, str) and value in available_languages() else SOURCE_LANGUAGE


def save_language(language, store=None):
    if language not in available_languages():
        raise ValueError(f"Unsupported interface language: {language}")
    store = store if store is not None else settings()
    store.setValue(SETTINGS_KEY, language)
    store.sync()


def language():
    return _language


def initialize(store=None):
    """Apply the saved language before constructing the first application window."""
    global _language
    _language = preferred_language(store)
    app = QApplication.instance()
    if app is not None:
        for translator in _translators:
            app.removeTranslator(translator)
            translator.deleteLater()
        _translators.clear()
        qt = QTranslator(app)
        directory = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
        if qt.load(QLocale(_language), "qtbase", "_", directory):
            app.installTranslator(qt)
            _translators.append(qt)
        else:
            qt.deleteLater()
        path = available_catalogs().get(_language)
        if path is not None:
            application = QTranslator(app)
            if application.load(str(path)):
                app.installTranslator(application)
                _translators.append(application)
            else:
                application.deleteLater()
        app.setLayoutDirection(QLocale(_language).textDirection())


def tr(source, *arguments, n=-1):
    """Translate a UI template, then insert values without translating user data."""
    text = QCoreApplication.translate(CONTEXT, source, None, n)
    return text.format(*arguments) if arguments else text


@lru_cache(maxsize=1)
def _message_patterns():
    patterns = []
    formatter = Formatter()
    for source in SOURCE_PATTERNS:
        parts, fields, literal_length = [], set(), 0
        for literal, field, spec, conversion in formatter.parse(source):
            parts.append(re.escape(literal))
            literal_length += len(literal)
            if field is not None:
                group = "arg" + field
                parts.append(f"(?P={group})" if field in fields else f"(?P<{group}>.*?)")
                fields.add(field)
        patterns.append((literal_length, re.compile("".join(parts), re.DOTALL), source))
    return sorted(patterns, key=lambda item: item[0], reverse=True)


def message(value):
    """Translate a domain error at the presentation boundary, retaining its values."""
    source = str(value)
    translated = tr(source)
    if translated != source:
        return translated
    for _, pattern, template in _message_patterns():
        match = pattern.fullmatch(source)
        if not match:
            continue
        target = tr(template)
        if target != template:
            # Captures are already formatted (including !r, units and precision).
            return "".join(
                literal + (match.group("arg" + field) if field is not None else "")
                for literal, field, spec, conversion in Formatter().parse(target)
            )
    return source
