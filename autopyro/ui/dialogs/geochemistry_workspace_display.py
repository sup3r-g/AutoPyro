"""Shared display labels and number formatting for geochemistry workspace views."""

from ..geochemistry import status_label
from ..i18n import tr


def _number(value):
    return '—' if value is None else f'{value:.6g}' if isinstance(value, (int, float)) else str(value)


def _biodegradation_labels():
    return {'not_assessed': tr('Не оценивались'), 'insufficient_evidence': tr('Данных недостаточно'),
            'evidence_noted': tr('Признаки отмечены экспертом'), 'no_evidence_noted': tr('Признаки не отмечены экспертом')}


def _biodegradation_indicators():
    return {
        'alkane_depletion': tr('Обеднение ряда н-алканов'),
        'isoprenoid_depletion': tr('Обеднение изопреноидов'),
        'unresolved_background': tr('Неразрешённый фон хроматограммы'),
        'sterane_changes': tr('Изменения стеранов'),
        'hopane_changes': tr('Изменения гопанов'),
    }


def _observation_labels():
    return {'insufficient_data': tr('Недостаточно данных'), 'observed': tr('Наблюдается'),
            'not_observed': tr('Не наблюдается')}


def _status(value):
    labels = {
        'not_selected': tr('Не выбрано'),
        'numeric': tr('Числовое значение'), 'ambiguous': tr('Выберите измерение'),
        'stale_selection': tr('Выбор измерения устарел'), 'qc_excluded': tr('Исключено QC'),
        'unit_mismatch': tr('Несовместимые единицы'), 'basis_mismatch': tr('Разные основания измерения'),
        'below_limit': tr('Ниже предела'), 'above_limit': tr('Выше предела'),
        'approximate': tr('Приближённое значение'), 'unparsed': tr('Не разобрано'),
        'other': tr('Не числовое значение'), 'derived_excluded': tr('Расчётный показатель исключён'),
        'incompatible_inputs': tr('Разные аналитические группы или варианты измерения'),
        'unsupported_correlation_indicator': tr('Вне списка показателей корреляции'),
        'unconfirmed_isotope_basis': tr('Подтвердите фракции C15+ и шкалу VPDB'),
    }
    return labels.get(value, status_label(value))
