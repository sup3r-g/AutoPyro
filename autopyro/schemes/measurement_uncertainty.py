"""Interpret declared measurement uncertainty without inventing confidence factors."""
import math

from .interpretation import unit_key

ABSOLUTE_TYPES = {'absolute', 'absolute_interval', 'interval', 'standard', 'standard_uncertainty',
                  'std', 'sigma', 'expanded', 'expanded_uncertainty', 'абсолютная', 'стандартная', 'расширенная'}
PERCENT_TYPES = {'relative_percent', 'relative %', '%', 'относительная %'}


def measurement_interval(project, row_id, column_id, value, unit):
    measurement = project.science.current(row_id, column_id)
    if measurement is None or measurement.uncertainty is None:
        return None, None, None
    metadata = dict(measurement_id=measurement.id, revision=measurement.revision,
                    uncertainty=measurement.uncertainty, type=measurement.uncertainty_type,
                    confidence_level=measurement.confidence_level, unit=measurement.unit or unit)
    bounds, warning = interval_from_metadata(metadata, value, unit)
    return bounds, metadata, warning


def interval_from_metadata(metadata, value, unit):
    declared_type = metadata.get('type', metadata.get('uncertainty_type', ''))
    kind = declared_type.strip().casefold() if isinstance(declared_type, str) else ''
    warning = None
    if metadata.get('unit') and unit_key(metadata['unit']) != unit_key(unit):
        warning = 'Единица неопределённости не совпадает с единицей оси'
    elif kind not in ABSOLUTE_TYPES | PERCENT_TYPES | {'relative_fraction'}:
        warning = 'Тип неопределённости не указан или не поддерживается; интервал не использован'
    elif value is None:
        warning = 'Интервал не использован: нет допустимого числового значения'
    if warning:
        return None, warning
    amount = metadata.get('uncertainty')
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        return None, 'Неопределённость должна быть конечной и неотрицательной'
    try:
        finite = math.isfinite(amount)
    except OverflowError:
        finite = False
    if not finite or amount < 0:
        return None, 'Неопределённость должна быть конечной и неотрицательной'
    if kind in PERCENT_TYPES:
        amount = abs(value) * amount / 100
    elif kind == 'relative_fraction':
        amount = abs(value) * amount
    if amount == 0:
        return None, None
    bounds = (value - amount, value + amount)
    if not all(math.isfinite(v) for v in bounds):
        return None, 'Неконечные границы неопределённости; интервал не использован'
    return bounds, None
