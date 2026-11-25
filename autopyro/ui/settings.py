"""User-wide UI preferences and defaults for new operations."""

from dataclasses import dataclass, fields, replace

from . import i18n
from ..database.preferences import connection_parameters, remember_password
from .theme import THEMES

CSV_DELIMITERS = (";", ",", "\t", "|")
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1251", "utf-16")
DECIMAL_SEPARATORS = ("auto", ".", ",")


@dataclass(frozen=True)
class Preferences:
    language: str = "ru"
    theme: str = "breeze_light"
    autosave_enabled: bool = True
    autosave_minutes: int = 1
    csv_delimiter: str = ";"
    csv_encoding: str = "utf-8-sig"
    decimal_separator: str = "auto"
    database_host: str = "127.0.0.1"
    database_port: int = 15432
    database_name: str = "geochemistry"
    database_user: str = "readonly"
    sqlite_path: str = ""
    sqlite_table: str = ""


KEYS = {
    "language": i18n.SETTINGS_KEY,
    "theme": "appearance/theme",
    "autosave_enabled": "general/autosave_enabled",
    "autosave_minutes": "general/autosave_minutes",
    "csv_delimiter": "import/delimiter",
    "csv_encoding": "import/encoding",
    "decimal_separator": "import/decimal_separator",
    "database_host": "database/host",
    "database_port": "database/port",
    "database_name": "database/name",
    "database_user": "database/user",
    "sqlite_path": "database/sqlite_path",
    "sqlite_table": "database/sqlite_table",
}


def _local_parameters():
    from ..database.connection import local_parameters
    return local_parameters()


def load_preferences(store=None, *, use_saved=True, base=None):
    store = store if store is not None else i18n.settings()
    base = base if base is not None else _local_parameters()
    try:
        port = int(base["port"])
        if not 1 <= port <= 65535:
            port = 15432
    except (TypeError, ValueError, OverflowError):
        port = 15432
    defaults = Preferences(
        database_host=base["host"], database_port=port,
        database_name=base["dbname"], database_user=base["user"],
    )
    if not use_saved:
        return defaults
    values = {}
    for field in fields(defaults):
        default = getattr(defaults, field.name)
        raw = store.value(KEYS[field.name], default)
        if isinstance(default, bool):
            raw = raw if isinstance(raw, bool) else {"true": True, "false": False, "1": True, "0": False}.get(str(raw).lower(), default)
        elif isinstance(default, int):
            try:
                raw = int(raw)
            except (TypeError, ValueError, OverflowError):
                raw = default
            maximum = 120 if field.name == "autosave_minutes" else 65535
            raw = raw if 1 <= raw <= maximum else default
        elif not isinstance(raw, str):
            raw = default
        values[field.name] = raw
    values["language"] = i18n.preferred_language(store)
    if values["theme"] not in THEMES:
        values["theme"] = defaults.theme
    for name, choices in (("csv_delimiter", CSV_DELIMITERS), ("csv_encoding", CSV_ENCODINGS), ("decimal_separator", DECIMAL_SEPARATORS)):
        if values[name] not in choices:
            values[name] = getattr(defaults, name)
    for name in ("database_host", "database_name", "database_user"):
        values[name] = values[name].strip() or getattr(defaults, name)
    return Preferences(**values)


def database_parameters(base=None, store=None):
    """Merge saved defaults, without reusing a password for another connection."""
    base = dict(base if base is not None else _local_parameters())
    preferences = load_preferences(store, base=base)
    return connection_parameters(base, preferences)


def save_preferences(preferences, *, password=None, store=None):
    """Validate the complete draft before writing any setting."""
    if preferences.language not in i18n.available_languages():
        raise ValueError(f"Unsupported interface language: {preferences.language}")
    if preferences.theme not in THEMES:
        raise ValueError(f"Unsupported interface theme: {preferences.theme}")
    if not 1 <= preferences.autosave_minutes <= 120:
        raise ValueError(i18n.tr("Интервал автосохранения должен быть от 1 до 120 минут."))
    if not 1 <= preferences.database_port <= 65535:
        raise ValueError(i18n.tr("Порт PostgreSQL должен быть от 1 до 65535."))
    preferences = replace(preferences, database_host=preferences.database_host.strip(),
                          database_name=preferences.database_name.strip(), database_user=preferences.database_user.strip())
    if not all((preferences.database_host, preferences.database_name, preferences.database_user)):
        raise ValueError(i18n.tr("Укажите сервер, базу данных и пользователя PostgreSQL."))
    if (preferences.csv_delimiter not in CSV_DELIMITERS or preferences.csv_encoding not in CSV_ENCODINGS
            or preferences.decimal_separator not in DECIMAL_SEPARATORS):
        raise ValueError(i18n.tr("Некорректные параметры импорта CSV."))
    store = store if store is not None else i18n.settings()
    for field in fields(preferences):
        store.setValue(KEYS[field.name], getattr(preferences, field.name))
    store.sync()
    if store.status() != store.Status.NoError:
        raise OSError(i18n.tr("Не удалось сохранить настройки приложения."))
    if password is not None:
        remember_password(preferences, password)
