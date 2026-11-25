"""Independent analytical workspaces sharing the versioned calculation engine."""

from .i18n import tr


def group_label(group):
    names = {
        'Isotopes': tr('Изотопы нефти'), 'VR': tr('Отражательная способность витринита'),
        'Gas': tr('Молекулярный состав газа'), 'Gas-Iso': tr('Изотопный состав газа'),
        'SARA-Un': tr('SARA: исходные фракции'), 'SARA-FID': tr('SARA: пламенно-ионизационный детектор'),
        'SARA-Calc': tr('SARA: расчётный состав'), 'Elem': tr('Элементный состав'),
        'Sat-GC': tr('Газовая хроматография насыщенной фракции'),
        'WO-GC': tr('Газовая хроматография цельной нефти'),
        'Sat-GCMS': tr('ГХ–МС насыщенной фракции'), 'Arom-GCMS': tr('ГХ–МС ароматической фракции'),
        'VisK-TL': tr('Микроскопия керогена'), 'Extract': tr('Экстракты'),
        'Leco': tr('Органический углерод'),
    }
    return names.get(group, group) + (f' ({group})' if group in names else '')


def status_label(status):
    return {
        'interpreted': tr('Интерпретация получена'), 'profile': tr('Показатели рассчитаны'),
        'boundary': tr('На разделяющей границе'), 'not_applicable': tr('Методика неприменима'),
        'outside_scope': tr('Вне выборки'), 'interpolation_excluded': tr('Интерполяция исключена'),
        'invalid_range': tr('Значение вне диапазона'), 'invalid_definition': tr('Ошибка методики'),
        'numerical_error': tr('Ошибка вычисления'), 'zero_denominator': tr('Деление на ноль'),
        'qc_excluded': tr('Исключено QC'),
        'missing': tr('Нет значения'), 'non_detect': tr('Не обнаружено'),
        'censored': tr('Значение с пределом'), 'invalid': tr('Некорректное значение'),
        'text': tr('Текст вместо числа'), 'non_numeric': tr('Не числовое значение'),
        'non_finite': tr('Не конечное число'),
    }.get(status, status)


def binding_title(method_id, role_key, definition):
    titles = {
        ('sofer', 'sat'): tr('Насыщенная фракция'),
        ('sofer', 'arom'): tr('Ароматическая фракция'),
        ('sofer', 'cv_reported'): tr('Контрольное значение CV'),
        ('vitrinite', 'ro'): tr('Отражательная способность'),
        ('vitrinite', 'n'): tr('Число измерений'),
        ('extract_toc', 'extract'): tr('Выход экстракта'),
        ('extract_toc', 'toc'): tr('Органический углерод породы'),
    }
    return titles.get((method_id, role_key), tr(definition['label']))


def modules():
    return {
        "oil_isotopes": (tr("Изотопы нефти"), ("sofer",)),
        "gas": (tr("Геохимия газа"), ("gas_ratios", "gas_isotopes")),
        "petrography": (tr("Органическая петрография"), ("vitrinite", "microscopy")),
        "sara": (tr("Фракционный состав SARA"), ("sara",)),
        "elemental": (tr("Элементный состав"), ("elemental", "atomic_ratios")),
        "chromatography": (tr("Хроматография и биомаркеры"),
                           ("gc_profile", "alkane_indices", "stereo_profile", "aromatic_profile",
                            "aromatic_indices", "aromatic_indices_from_peaks", "aromatic_steroid_distribution")),
        "steranes": (tr("Регулярные стераны C27–C29"),
                     ("sterane_distribution", "c27_diasterane_ratio", "sterane_ratios",
                      "sterane_hopane_ratios", "ster_hop_from_peaks")),
        "terpanes": (tr("Терпаны и гопаны"),
                     ("terpane_hopane_profile", "homohopane_distribution", "homohopane_epimers",
                      "hopane_indicators", "tricyclic_terpane_ratios")),
        "rallakis": (tr("Методики Rallakis"),
                     ("rallakis_aromatics", "rallakis_steranes", "rallakis_saturated",
                      "rallakis_steroids", "rallakis_gc")),
        "extraction": (tr("Экстракция и TOC"), ("extract_toc",)),
    }
