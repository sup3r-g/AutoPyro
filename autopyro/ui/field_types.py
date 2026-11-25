"""Localized labels for the persisted table field types."""

from .i18n import tr


def field_kind_label(kind):
    return {
        "text": tr("Текст"),
        "number": tr("Число"),
        "date": tr("Дата"),
        "time": tr("Время"),
        "datetime": tr("Дата и время"),
        "boolean": tr("Логическое значение"),
        "duration": tr("Длительность"),
        "category": tr("Категория"),
        "identifier": tr("Идентификатор"),
        "integer": tr("Целое число"),
    }.get(kind, kind)
