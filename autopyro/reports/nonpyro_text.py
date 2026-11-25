"""English display text for built-in NonPyro methods and findings."""

import re

ENGLISH_METHODS = {
    'sofer': (
        'Oil isotopes · Sofer CV',
        'CV=-2.53*δ13C_sat+2.22*δ13C_arom-11.65; boundary 0.47. Oil and comparable C15+ fractions only, VPDB scale. Statistical isotope type, not a definitive source assignment; maturity and mixing may affect results.',
    ),
    'vitrinite': (
        'Vitrinite · approximate maturity',
        'BGS reference ranges: oil generation onset 0.5–0.6 %Ro, completion 0.85–1.1; gas range onset 1.0–1.3, upper limit around 3.0. Boundaries overlap and depend on kerogen. The original population is required; modal/user need separate confirmation.',
    ),
    'gas_ratios': (
        'Gas C1–C3 · molecular ratios',
        'The same molar or volume basis and normalization are required. Wetness here covers C1–C3 only, not C1–C5. Molecular dryness alone does not establish origin.',
    ),
    'gas_isotopes': (
        'Gas isotopes · C1–C3 order',
        'VPDB, comparable measurements. Normal, reversed or partly reversed order is identified; this is not an unambiguous diagnosis of mixing, biogenic origin or maturity.',
    ),
    'sara': (
        'SARA · fraction composition and balance',
        'One group/method and mass basis. The sum is not normalized to 100. The user sets balance tolerance; the dominant fraction describes composition, not biodegradation. Missing %Asph in SARA-Calc is not replaced with zero.',
    ),
    'elemental': (
        'Elemental composition · atomic H/C and O/C',
        'Atomic ratios use nominal masses C=12, H=1, O=16; H/C=12H/C, O/C=0.75O/C. Confirm purified kerogen and a shared mass basis for a kerogen diagram. Kerogen type is not assigned automatically.',
    ),
    'atomic_ratios': (
        'Database elemental ratios · normalization',
        'The %mol (a/b) field stores ratio ×100: divide by 100 to obtain the atomic ratio. Mass H/C wt is not substituted for molar H/C mol.',
    ),
    'gc_profile': (
        'GC · Pr/Ph and isoprenoid/n-alkane ratios',
        'One fraction and one ratio measurement basis (area/height/concentration). Profile and QC; Pr/Ph is not an unambiguous redox indicator. Use a validated Shapely scheme for interpretation fields.',
    ),
    'alkane_indices': (
        'GC · n-alkane indices CPI and OEP',
        'Bray–Evans CPI (C24–C34) and Scalan–Smith OEP (centered on C27). Confirm that all inputs come from one analysis and one fraction. Use comparable peak areas or mass concentrations. The indices require geological context.',
    ),
    'stereo_profile': (
        'GC–MS · steranes and hopanes',
        'Stereochemical ratio profile: % (a/(a+b)) is divided by 100. S/R and S/(S+R) are not equated without unit definitions. Equilibrium plateaus are not automatically converted to Ro; source facies and secondary processes need separate assessment.',
    ),
    'sterane_distribution': (
        'Regular C27-C29 steranes (alpha-alpha-alpha 20R)',
        'Use three alpha-alpha-alpha 20R peaks from the saturated fraction on a single analytical basis. Fractions are calculated from the sum of those three peaks. Source type is not assigned automatically.',
    ),
    'c27_diasterane_ratio': (
        'C27 diasterane / regular sterane from peaks',
        'Ratio of two saturated-fraction peaks on a single analytical basis.',
    ),
    'ster_hop_from_peaks': (
        'Ster/Hop from 17 sterane and 16 hopane peaks',
        'The peak set was reconstructed from heights and areas for one Stratum Reservoir sample in the Barents Sea snapshot. Use a single basis for all peaks. The combined St27bbR+ peak is used in full.',
    ),
    'sterane_ratios': (
        'Diasteranes and steranes from reported ratios',
        'Use reported indicators on a single analytical basis. Percentages are converted to fractions and missing values are preserved.',
    ),
    'sterane_hopane_ratios': (
        'Steranes / hopanes',
        'Use reported indicators on a single analytical basis. Percentages are converted to fractions and missing values are preserved. For Ster/Hop, confirm which peaks were included in the source totals.',
    ),
    'hopane_indicators': (
        'Oleanane, diahopanes and norhopanes',
        'Use reported indicators on a single analytical basis. Percentages are converted to fractions and missing values are preserved. An expert assesses biodegradation.',
    ),
    'tricyclic_terpane_ratios': (
        'Reported tri- and tetracyclic terpane ratios',
        "Use reported indicators on a single analytical basis. Percentages are converted to fractions and missing values are preserved. The summed ratio uses the laboratory's original peak set.",
    ),
    'aromatic_indices': (
        'Aromatic GC-MS indices MPI-2, MPR and MDBTR',
        'Use reported indicators on a single analytical basis. Percentages are converted to fractions and missing values are preserved.',
    ),
    'aromatic_steroid_distribution': (
        'Monoaromatic and triaromatic steroid distributions',
        'Use reported indicators on a single analytical basis. Percentages are converted to fractions and missing values are preserved. MAS and TAS fractions are taken from the reported distributions for each group.',
    ),
    'terpane_hopane_profile': (
        'Terpanes and hopanes in the saturated fraction',
        'Assign at least one complete peak pair on a single analytical basis. When a peak is missing, ratios are calculated for the other available pairs.',
    ),
    'homohopane_distribution': (
        'C31-C35 homohopane distribution',
        'For each C31-C35 homologue, assign the reported 22S+22R sum or both epimers on a single analytical basis. The reported sum takes priority when both are available. All five homologues are required.',
    ),
    'homohopane_epimers': (
        'C31-C35 homohopane epimer ratios',
        'Assign separate 22S and 22R peaks for at least one homologue. Ratios are calculated independently for complete pairs and expressed as fractions from 0 to 1.',
    ),
    'aromatic_profile': (
        'Aromatic GC–MS · MPI-1 and DBT/Phen',
        'Original indices from one measurement method. Rc is not calculated from MPI without independently selecting a branch and calibration; Rc is not equated with measured Ro. Profiles, QC and transfer to geometric interpretation are available.',
    ),
    'aromatic_indices_from_peaks': (
        'MPI-2, MPR and MDBTR from peaks',
        'MPI-2 = 3 x 2MP/(Phen+1MP+9MP), MPR = 2MP/1MP, MDBTR = 4MDBT/1MDBT. Assign the peaks for at least one index on a single analytical basis.',
    ),
    'microscopy': (
        'Microscopy · kerogen maceral groups',
        'Area percentages from one classification. Dominant maceral group and balance checks; AOM/Algae and nested categories are not counted twice. Generation potential is not inferred automatically.',
    ),
    'extract_toc': (
        'Extract / organic carbon',
        '100*EOM_mg/g_rock/TOC_%wt. The same original sample and pre-extraction state are required. This is extract mass relative to organic carbon mass, not a transformation percentage or an independent migration diagnosis.',
    ),
    'rallakis_gc': (
        'Rallakis · GC-FID',
        'The shortened CPI is intended for extracts. The n-alkane ratios follow Figs. 7.19–7.21.',
    ),
    'rallakis_saturated': (
        'Rallakis · hopanes, terpanes and steranes',
        'No. 7 uses peaks P, Q, R, T, U and V. Assign TT_T manually because peak T is absent from Table 5.2.1. '
        'Nos. 9 and 12 use four C29 peaks at m/z 217.',
    ),
    'rallakis_steroids': (
        'Rallakis · aromatic steroids',
        'Assign peaks g1 (C28 TA, m/z 231) and H1 (C29 MA, m/z 253) as identified in Figs. 5.2.3–5.2.4.',
    ),
    'rallakis_aromatics': (
        'Rallakis · aromatic indices and Rc',
        'Rc is calculated using the calibrations in Table 5.3. MPI-1 uses the ascending branch. '
        'No. 27 uses four MP isomers from Table 5.2.6.1 and three MDBT signals, with 2+3 as a combined peak.',
    ),
    'rallakis_steranes': (
        'Rallakis · regular C27–C29 ββ 20S steranes',
        'Nos. 13–15 use peaks i, o and s at m/z 218. Assign comparable signals for the three isomers.',
    ),
}

ENGLISH_FINDINGS = {
    'Доли регулярных стеранов ββ 20S в сумме C27–C29': 'Fractions of regular ββ 20S steranes in the C27–C29 total',
    'Показатели Rallakis по измеренным пикам': 'Rallakis indicators calculated from measured peaks',
    'Нет полного набора пиков для показателей Rallakis': 'No complete peak set for the Rallakis indicators',
    'Rc по MPR: для log10 требуется MPR > 0': 'Rc from MPR: log10 requires MPR > 0',
    'Доли регулярных стеранов ααα 20R в сумме C27–C29': 'Fractions of regular ααα 20R steranes in the C27–C29 total',
    'Отношение суммы 17 стерановых и 16 гопановых пиков': 'Ratio of the sums of 17 sterane and 16 hopane peaks',
    'Отношение двух пиков C27': 'Ratio of two C27 peaks',
    'C27 диастеран / стеран': 'C27 diasterane / sterane',
    'Исходные показатели в долях единицы': 'Reported indicators expressed as fractions',
    'Нет допустимых исходных показателей': 'No valid reported indicators',
    'Индексы по измеренным пикам': 'Indices calculated from measured peaks',
    'Нет полного набора пиков для индексов': 'No complete peak set for the indices',
    'Отношения по доступным парам пиков': 'Ratios calculated from available peak pairs',
    'Нет полных пар пиков': 'No complete peak pairs',
    'Доли гомогопанов 17α,21β (22S+22R) в сумме C31–C35': 'Fractions of 17α,21β homohopanes (22S+22R) in the C31–C35 total',
    'Неполный ряд гомогопанов C31–C35': 'Incomplete C31–C35 homohopane series',
    'Доля 22S в сумме эпимеров каждого гомолога': '22S fraction of the epimer total for each homologue',
    'Нет полных пар эпимеров': 'No complete epimer pairs',
    'Основа измерения [un] не определена; сопоставление с другими основами требует проверки': 'Measurement basis [un] is undefined; comparison with other bases needs checking',
    'Численное переполнение': 'Numerical overflow',
    'Рассчитанный CV отличается от исходного CV больше объявленного допуска': 'Calculated CV differs from original CV beyond the declared tolerance',
    'На разделяющей границе CV=0.47': 'On the CV=0.47 separating boundary',
    '; статистическая гипотеза': '; statistical hypothesis',
    'Молекулярный профиль C1–C3; генезис требует совместного анализа изотопов и геологии': 'C1–C3 molecular profile; origin requires joint isotope and geological analysis',
    'Изотопный тип по Sofer: преимущественно ': 'Sofer isotope type: predominantly ',
    'Происхождение популяции не подтверждено; ориентировочная зрелость не назначена': 'Population origin is unconfirmed; approximate maturity was not assigned',
    'Ниже ориентировочного начала нефтеобразования': 'Below approximate oil generation onset',
    'Границы ориентировочные; зависят от типа керогена и истории нагрева': 'Boundaries are approximate and depend on kerogen type and thermal history',
    'Число измерений не указано либо меньше ориентировочных 40–50; проверьте распределение и популяцию': 'Measurement count is missing or below the approximate 40–50; check distribution and population',
    'терригенный': 'terrigenous',
    'морской': 'marine',
    'Переход начала нефтеобразования (0.5–0.6 %Ro)': 'Oil generation onset transition (0.5–0.6 %Ro)',
    'Сумма C1–C3 превышает 100% на объявленной основе; проверьте нормировку': 'C1–C3 sum exceeds 100% on the declared basis; check normalization',
    'Нормальный порядок δ13C1 < δ13C2 < δ13C3': 'Normal order δ13C1 < δ13C2 < δ13C3',
    'Порядок изотопов сам по себе не доказывает источник или механизм образования газа': 'Isotope order alone does not prove gas source or formation mechanism',
    'Ориентировочный нефтяной диапазон': 'Approximate oil range',
    'Полная изотопная инверсия': 'Complete isotope reversal',
    'Частичная инверсия / совпадение значений': 'Partial reversal / equal values',
    'насыщенные': 'saturates',
    'ароматические': 'aromatics',
    'смолы': 'resins',
    'асфальтены': 'asphaltenes',
    'Атомный профиль H/C–O/C по номинальным массам; тип керогена не назначен': 'Atomic H/C–O/C profile using nominal masses; kerogen type is unassigned',
    'Перекрывающийся переход нефтяного/газового диапазонов (0.85–1.3 %Ro)': 'Overlapping oil/gas range transition (0.85–1.3 %Ro)',
    'Преобладающая фракция: ': 'Dominant fraction: ',
    'Максимумы совпадают: ': 'Tied maxima: ',
    'Все фракции нулевые; состав не классифицирован': 'All fractions are zero; composition is unclassified',
    'Остаточные минералы/карбонаты и неодинаковая подготовка влияют на отношения': 'Residual minerals/carbonates and different preparation affect ratios',
    'Атомные отношения приведены к 1, без множителя 100; тип керогена не назначен': 'Atomic ratios are normalized to 1, without a factor of 100; kerogen type is unassigned',
    'Ориентировочный газовый диапазон': 'Approximate gas range',
    'Высокая зрелость; около/выше ориентировочного конца газового диапазона': 'High maturity; near/above the approximate gas range limit',
    'GC-профиль; однозначная фациальная, генетическая или биодеградационная классификация по этим трём отношениям не выполняется': 'GC profile; these three ratios do not provide an unambiguous facies, genetic or biodegradation classification',
    'Индексы чётности н-алканов для подтверждённой основы измерений': 'n-Alkane odd–even indices for the confirmed measurement basis',
    'Стереоотношения в долях 0–1; количественная зрелость без калибровки не назначается': 'Stereochemical ratios as 0–1 fractions; quantitative maturity is not assigned without calibration',
    'Ароматический профиль; ветвь MPI→Rc и фациальная схема не выбраны': 'Aromatic profile; MPI→Rc branch and facies scheme are unselected',
    'Максимумы макрогрупп совпадают': 'Maceral group maxima are tied',
    'Выход экстракта на органический углерод; самостоятельный диагноз миграции не выполняется': 'Extract yield per organic carbon; no independent migration diagnosis is made',
    'Преобладающая макрогруппа: ': 'Dominant maceral group: ',
    'Все макрогруппы нулевые; состав не классифицирован': 'All maceral groups are zero; composition is unclassified',
}

_FINDING_PATTERNS = tuple((re.compile(pattern), translation) for pattern, translation in (
    (r"Изотопный тип по Sofer: преимущественно (терригенный|морской); статистическая гипотеза", None),
    (r"Преобладающая макрогруппа: (.+)", "Dominant maceral group: {}"),
    (r"Исходная Quality \((.+?)\): (.*); код не интерпретирован автоматически",
     "Reported Quality ({}): {}. The code was not interpreted automatically."),
    (r"(.+): нулевой знаменатель", "{}: zero denominator"),
    (r"Сумма SARA=(.+?)%: баланс вне допуска; автоматическая нормировка не выполнялась",
     "SARA total = {}%. The balance is outside tolerance and was not automatically normalized."),
    (r"(.+): сумма исходных долей (.+?)%", "{}: reported fractions total {}%"),
    (r"Сумма трёх макрогрупп (.+?)%: проверьте полноту и исключительность классификации",
     "The three maceral groups total {}%. Check that the classification is complete and mutually exclusive."),
    (r"(.+): неотрицательное δ13C; проверьте исходные данные, ноль не заменён пропуском",
     "{}: non-negative δ13C. Check the source data. Zero was retained as a measured value."),
))


def english_method_text(protocol):
    from autopyro.analysis.nonpyro import METHODS

    definition = protocol["definition"]
    method_id = protocol["config"].get("method")
    builtin = METHODS.get(method_id)
    if builtin is None or method_id not in ENGLISH_METHODS:
        return definition["name"], definition["note"]
    name, note = ENGLISH_METHODS[method_id]
    return (name if definition["name"] == builtin["name"] else definition["name"],
            note if definition["note"] == builtin["note"] else definition["note"])


def english_finding(text):
    if text in ENGLISH_FINDINGS:
        return ENGLISH_FINDINGS[text]
    for prefix, translation in (("Преобладающая фракция: ", "Dominant fraction: "),
                                ("Максимумы совпадают: ", "Fractions tied for the maximum: ")):
        if text.startswith(prefix):
            fractions = text[len(prefix):].split(", ")
            return translation + ", ".join(ENGLISH_FINDINGS.get(item, item) for item in fractions)
    for pattern, translation in _FINDING_PATTERNS:
        if match := pattern.fullmatch(text):
            if translation is None:
                source = ENGLISH_FINDINGS[match[1]]
                return f"Sofer isotope type: predominantly {source}. This is a statistical hypothesis."
            return translation.format(*match.groups())
    return text
