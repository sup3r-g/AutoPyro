"""Scalar types for table columns. Temporal values use JSON-safe ISO strings."""

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
import math
import re


TYPED_KINDS = ("date", "time", "datetime", "boolean", "duration", "category", "identifier", "integer")
FIELD_KINDS = ("text", "number", *TYPED_KINDS)

_ISO_DATE = r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
_LOCAL_DATE = r"[0-9]{1,2}\.[0-9]{1,2}\.[0-9]{4}"
_TIME = r"[0-9]{2}:[0-9]{2}(?::[0-9]{2}(?:[.,][0-9]{1,6})?)?"
_DATETIME = re.compile(rf"({_ISO_DATE}|{_LOCAL_DATE})[T ]({_TIME}(?:Z|[+-][0-9]{{2}}:[0-5][0-9])?)")
_BOOLEAN = {"true": True, "yes": True, "да": True, "истина": True, "1": True,
            "false": False, "no": False, "нет": False, "ложь": False, "0": False}
_DURATION = re.compile(r"([+-]?)([0-9]+):([0-5][0-9])(?::([0-5][0-9](?:[.,][0-9]{1,6})?))?")


def _integer(raw):
    if isinstance(raw, bool):
        raise ValueError("Ожидается целое число без дробной части")
    try:
        value = Decimal(str(raw).strip().replace(",", "."))
        if value.is_finite() and value == value.to_integral_value():
            return int(value)
    except InvalidOperation:
        pass
    raise ValueError("Ожидается целое число без дробной части")


def _duration(raw):
    if isinstance(raw, timedelta):
        raw = raw.total_seconds()
    if isinstance(raw, bool):
        raise ValueError("Ожидается длительность ЧЧ:ММ:СС или число секунд")
    text = str(raw).strip()
    match = _DURATION.fullmatch(text)
    try:
        if match:
            sign, hours, minutes, seconds = match.groups()
            value = Decimal(hours) * 3600 + Decimal(minutes) * 60 + Decimal((seconds or "0").replace(",", "."))
            if sign == "-":
                value = -value
        else:
            value = Decimal(text.replace(",", "."))
        if value.is_finite() and math.isfinite(float(value)):
            return int(value) if value == value.to_integral_value() else float(value)
    except (InvalidOperation, OverflowError):
        pass
    raise ValueError("Ожидается длительность ЧЧ:ММ:СС или число секунд")


def format_duration(value):
    """Format seconds without wrapping at midnight or losing whole hours."""
    seconds = Decimal(str(abs(value)))
    hours, remainder = divmod(int(seconds), 3600)
    minutes, remainder = divmod(remainder, 60)
    fraction = format(seconds, "f").partition(".")[2].rstrip("0")
    tail = f"{remainder:02d}" + (f".{fraction}" if fraction else "")
    return f"{'-' if value < 0 else ''}{hours:02d}:{minutes:02d}:{tail}"


def _date(value):
    if re.fullmatch(_ISO_DATE, value):
        return date.fromisoformat(value)
    if re.fullmatch(_LOCAL_DATE, value):
        day, month, year = map(int, value.split("."))
        return date(year, month, day)
    raise ValueError


def _datetime(value):
    match = _DATETIME.fullmatch(value)
    if match is None:
        raise ValueError
    return datetime.fromisoformat(f"{_date(match[1]).isoformat()}T{match[2]}")


def parse_typed_value(raw, kind):
    if raw is None or isinstance(raw, str) and not raw.strip():
        return None
    text = str(raw).strip()
    if kind in ("category", "identifier"):
        return str(raw)
    if kind == "integer":
        return _integer(raw)
    if kind == "duration":
        return _duration(raw)
    if kind == "boolean":
        try:
            return _BOOLEAN[text.casefold()]
        except KeyError:
            raise ValueError("Ожидается логическое значение: true/false, да/нет, истина/ложь или 1/0") from None
    if kind == "date":
        try:
            # Excel often supplies a date as midnight without a time zone.
            if _DATETIME.fullmatch(text):
                stamp = _datetime(text)
                if stamp.tzinfo is not None or stamp.time() != time():
                    raise ValueError
                return stamp.date().isoformat()
            return _date(text).isoformat()
        except ValueError:
            raise ValueError("Ожидается дата в формате ГГГГ-ММ-ДД или ДД.ММ.ГГГГ") from None
    if kind == "time":
        try:
            if re.fullmatch(_TIME, text):
                return time.fromisoformat(text).isoformat()
        except ValueError:
            pass
        raise ValueError("Ожидается время в формате ЧЧ:ММ или ЧЧ:ММ:СС")
    if kind == "datetime":
        try:
            stamp = _datetime(text)
            if stamp.tzinfo is not None:
                stamp = stamp.astimezone(timezone.utc)
            return stamp.isoformat()
        except (ValueError, OverflowError):
            raise ValueError("Ожидаются дата и время, например 2026-11-11 14:30:00") from None
    raise ValueError(kind)


def typed_kind_hint(value):
    """Suggest a type only for recognizable, unambiguous input."""
    text = str(value).strip()
    if text.casefold() in _BOOLEAN and text not in ("0", "1"):
        return "boolean"
    if _DATETIME.fullmatch(text):
        return "datetime"
    if re.fullmatch(_ISO_DATE, text) or re.fullmatch(_LOCAL_DATE, text):
        return "date"
    if re.fullmatch(_TIME, text):
        return "time"
    return None
