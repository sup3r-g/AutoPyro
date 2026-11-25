"""Versioned non-pyrolysis interpretation with explicit analytical applicability."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from copy import deepcopy

import numpy as np
import pandas as pd

from autopyro.model.core import Column, timestamp
from autopyro.model.measurements import derived, snapshot_mapping
from autopyro.model.tabular import classify, classify_series
from autopyro import __version__

SOFER = "https://doi.org/10.1306/AD460963-16F7-11D7-8645000102C1865D"
RO_SOURCE = "https://earthwise.bgs.ac.uk/index.php/OR/14/055_Technical_information"
RO_QC = "https://wiki.aapg.org/Vitrinite_reflectance"
GAS_SOURCE = "https://pubs.usgs.gov/sir/2011/5195/sir2011-5195_text.pdf"
ISO_SOURCE = "https://www.usgs.gov/publications/carbon-and-hydrogen-isotopic-reversals-deep-basin-gas-evidence-limits-stability"
GC_SOURCE = "https://pubs.usgs.gov/pp/1314a-l/report.pdf"
ALKANE_INDEX_SOURCE = "https://pubs.usgs.gov/of/1998/ofr-98-0034/OA.pdf"
ELEMENT_SOURCE = "https://doi.org/10.1306/3B05BB14-172A-11D7-8645000102C1865D"

UNITS = {
    "delta": {"per mil": 1.0, "‰": 1.0},
    "mass": {"%wt": 1.0, "%wt (a/b)": 1.0, "%": 1.0, "g/g": 100.0},
    "ro": {"%": 1.0, "%Ro": 1.0},
    "count": {"Euc": 1.0, "count": 1.0, "1": 1.0},
    "ratio": {"1": 1.0, "Euc": 1.0, "Euc (a/b)": 1.0, "ratio": 1.0},
    "percent_ratio": {"% (a/b)": 0.01, "1": 1.0, "Euc (a/b)": 1.0},
    "signed": {"1": 1.0, "Euc": 1.0},
    "fraction": {"1": 1.0, "Euc (a/(a+b))": 1.0, "% (a/(a+b))": 0.01, "%": 0.01},
    "atomic": {"1": 1.0, "atomic ratio": 1.0, "%mol (a/b)": 0.01},
    "area": {"%area": 1.0, "%": 1.0},
    "gas": {
        "%vol": 10000.0,
        "%mol": 10000.0,
        "ppm (cm³/m³)": 1.0,
        "ppm (mol)": 1.0,
        "ppm vol": 1.0,
        "ppm mol": 1.0,
    },
    "extract": {"ppt (mg/g) (a/b)": 1.0, "mg/g rock": 1.0, "ppm (µg/g)": 0.001},
    "alkane_signal": {"count": 1.0, "ppm (µg/g)": 1.0},
    "sterane_signal": {"count": 1.0, "ppm (µg/g)": 1.0},
    "terpane_signal": {"count": 1.0, "ppm (µg/g)": 1.0},
}

TERPANE_HOPANE_PAIRS = {
    "C19_C23": ("t19", "t23"),
    "Tet24_C23": ("tet24", "t23"),
    "Gamm_C30": ("gamm", "h30"),
    "Ts_fraction": ("ts", "tm"),
}

HOMOHOPANE_INPUTS = {
    carbon: (f"h{carbon}_total", f"h{carbon}_s", f"h{carbon}_r")
    for carbon in range(31, 36)
}

AROMATIC_INDEX_FORMULAS = {
    "mpi2": ("2mp", ("phen", "1mp", "9mp"), 3),
    "mpr": ("2mp", ("1mp",), 1),
    "mdbtr": ("4mdbt", ("1mdbt",), 1),
}

STER_HOP_STERANES = (
    "St27dbS", "St27dbR", "St28dbS", "St28dbR", "St29dbR",
    "St27aaS", "St27bbR+", "St27bbS", "St27aaR",
    "St28aaS", "St28bbR", "St28bbS", "St28aaR",
    "St29aaS", "St29bbR", "St29bbS", "St29aaR",
)
STER_HOP_HOPANES = (
    "H27Ts", "H27Tm", "H29ab", "H29Ts", "H30D", "H30ab",
    *(f"H{carbon}ab{epimer}" for carbon in range(31, 36) for epimer in ("S", "R")),
)


def ster_hop_ratio(values):
    scale = max(values.values()) or 1
    denominator = math.fsum(values[key] / scale for key in STER_HOP_HOPANES)
    if denominator == 0:
        return None
    return math.fsum(values[key] / scale for key in STER_HOP_STERANES) / denominator


def complete_homohopanes(inputs):
    return all(
        total in inputs or (s in inputs and r in inputs)
        for total, s, r in HOMOHOPANE_INPUTS.values()
    )


def profile_inputs_complete(method_id, available):
    if method_id in RALLAKIS_RATIOS:
        return any(set(numerator + denominator) <= set(available)
                   for _, numerator, denominator, _ in RALLAKIS_RATIOS[method_id])
    if method_id == "homohopane_distribution":
        return complete_homohopanes(available)
    if method_id == "homohopane_epimers":
        return any(s in available and r in available for _, s, r in HOMOHOPANE_INPUTS.values())
    if method_id == "terpane_hopane_profile":
        return any(a in available and b in available for a, b in TERPANE_HOPANE_PAIRS.values())
    if method_id == "aromatic_indices_from_peaks":
        return any(peak in available and all(key in available for key in divisor)
                   for peak, divisor, _ in AROMATIC_INDEX_FORMULAS.values())
    if method_id in REPORTED_BIOMARKER_METHODS:
        return bool(available)
    return all(key in available for key, role in METHODS[method_id]["roles"].items() if not role["optional"])


def role(label, unit, groups, prefixes, optional=False):
    return {
        "label": label,
        "unit_family": unit,
        "groups": tuple(groups),
        "prefixes": tuple(prefixes),
        "optional": optional,
    }


def method(name, roles, outputs, source, note, groups, plot=None, same_group=True):
    return {
        "name": name,
        "version": 1,
        "roles": roles,
        "outputs": outputs,
        "sources": source,
        "note": note,
        "groups": groups,
        "plot": plot,
        "same_group": same_group,
    }


def reported_biomarker_method(name, group, fields, note="", sources=()):
    return method(
        name,
        {
            key: role(label, family, [group], [f"{field}[{tag}]" for tag in ("h", "a", "ch", "ca")],
                      optional=True)
            for key, label, field, family in fields
        },
        {key: (label, "1") for key, label, _, _ in fields},
        list(sources),
        "Выберите готовые показатели одной аналитической основы. "
        "Проценты переводятся в доли единицы, пропуски сохраняются. " + note,
        [group],
    )


REPORTED_BIOMARKER_METHODS = {
    "sterane_ratios": reported_biomarker_method(
        "Диастераны и стераны · исходные отношения", "Sat-GCMS",
        (
            ("dia27_sr", "C27 диастеран 20S / C27 ααα 20R", "St27dbS/St27aaR", "ratio"),
            ("dia27", "Диастераны / ααα-стераны C27", "Dia27/St27aa", "percent_ratio"),
            ("dia29", "Диастераны / ααα-стераны C29", "Dia29/St29aa", "ratio"),
            ("st27_epimers", "C27 ααα 20S/(20S+20R)", "St27 20S/R", "fraction"),
        ),
    ),
    "sterane_hopane_ratios": reported_biomarker_method(
        "Стераны / гопаны", "Sat-GCMS",
        (
            ("ster_hop", "Стераны / гопаны · сумма по данным лаборатории", "Ster/Hop", "ratio"),
            ("st27_h30", "Стераны C27 / C30 αβ-гопан", "St27/H30ab", "ratio"),
            ("st29_h30", "Стераны C29 / C30 αβ-гопан", "St29/H30ab", "ratio"),
        ),
        "Для Ster/Hop требуется подтверждение состава сумм в исходном анализе.",
    ),
    "hopane_indicators": reported_biomarker_method(
        "Олеанан, диагопаны и норгопаны", "Sat-GCMS",
        (
            ("olean", "Олеанан / (олеанан + C30 αβ-гопан)", "OleanIndex", "fraction"),
            ("diahopane", "Диагопан C30 / (диагопан + C30 αβ-гопан)", "DiaHIndex", "fraction"),
            ("h29ts", "C29 Ts / C29 αβ-гопан", "H29Ts/H29ab", "ratio"),
            ("nor25", "25-норгопаны (m/z 177) / C30 αβ-гопан", "25nor(177)/H30ab", "ratio"),
        ),
        "Оценку биодеградации формулирует эксперт.",
    ),
    "tricyclic_terpane_ratios": reported_biomarker_method(
        "Три- и тетрациклические терпаны · исходные отношения", "Sat-GCMS",
        (
            ("c26_c25", "C26TT/C25TT", "t26/t25", "ratio"),
            ("c23_h30", "C23TT/C30 αβ-гопан", "t23/H30ab", "ratio"),
            ("tet24_h30", "C24TeT/C30 αβ-гопан", "tet24/H30ab", "ratio"),
            ("c20_26_h30", "Σ(C20–C26 TT)/C30 αβ-гопан", "t20-26/H30ab", "ratio"),
        ),
        "Суммарное отношение использует исходный набор пиков лаборатории.",
    ),
    "aromatic_indices": reported_biomarker_method(
        "Ароматические GC–MS · MPI-2, MPR и MDBTR", "Arom-GCMS",
        (
            ("mpi2", "MPI-2", "MPI-2", "ratio"),
            ("mpr", "MPR", "MPR", "ratio"),
            ("mdbtr", "MDBTR", "MDBTR", "ratio"),
        ),
    ),
    "aromatic_steroid_distribution": reported_biomarker_method(
        "Моно- и триароматические стероиды · распределения", "Arom-GCMS",
        tuple((f"{family}{carbon}", f"{family} C{carbon}", f"{carbon}{family}", "fraction")
              for family, carbons in (("MAS", (27, 28, 29)), ("TAS", (26, 27, 28)))
              for carbon in carbons),
        "Доли MAS и TAS берутся из исходных распределений каждой группы.",
    ),
}


METHODS = {
    "sofer": method(
        "Изотопы нефти · Sofer CV",
        {
            "sat": role(
                "δ13C C15+ насыщенной фракции", "delta", ["Isotopes"], ["d13C Sat "]
            ),
            "arom": role(
                "δ13C C15+ ароматической фракции", "delta", ["Isotopes"], ["d13C Arom "]
            ),
            "cv_reported": role(
                "Исходный CV для проверки (необязательно)",
                "signed",
                ["Isotopes"],
                ["CV Sofer ["],
                True,
            ),
        },
        {
            "CV": ("CV Sofer", "1"),
            "d13C_sat": ("δ13C Sat canonical", "per mil"),
            "d13C_arom": ("δ13C Arom canonical", "per mil"),
            "CV_difference": ("CV расчёт − исходный CV", "1"),
        },
        [SOFER],
        "CV=-2.53*δ13C_sat+2.22*δ13C_arom-11.65; граница 0.47. Только нефти и сопоставимые C15+ фракции, шкала VPDB. "
        "Статистический изотопный тип, не окончательное установление источника; зрелость и смешение могут влиять.",
        ["Isotopes"],
        ("d13C_sat", "d13C_arom"),
    ),
    "vitrinite": method(
        "Витринит · ориентировочная зрелость",
        {
            "ro": role(
                "Отражательная способность выбранной популяции",
                "ro",
                ["VR"],
                ["Indig pop Mean ", "Modal pop Mean ", "User pop Mean "],
            ),
            "n": role(
                "Число измерений той же популяции",
                "count",
                ["VR"],
                ["Indig pop Num ", "Modal pop Num ", "User pop Num "],
                True,
            ),
        },
        {"Ro": ("Ro выбранной популяции", "%Ro")},
        [RO_SOURCE, RO_QC],
        "Ориентиры BGS: начало нефтеобразования 0.5–0.6 %Ro, завершение 0.85–1.1; начало газового диапазона 1.0–1.3, "
        "верх около 3.0. Границы перекрываются и зависят от керогена. Нужна исходная популяция; modal/user требуют отдельного подтверждения.",
        ["VR"],
    ),
    "gas_ratios": method(
        "Газ C1–C3 · молекулярные отношения",
        {
            "c1": role("Метан C1", "gas", ["Gas"], ["C1 %HC[", "C1["]),
            "c2": role("Этан C2", "gas", ["Gas"], ["C2 %HC[", "C2["]),
            "c3": role("Пропан C3", "gas", ["Gas"], ["C3 %HC[", "C3["]),
        },
        {
            "Bernard": ("C1/(C2+C3)", "1"),
            "dryness3": ("C1/(C1+C2+C3)", "%"),
            "wetness3": ("(C2+C3)/(C1+C2+C3)", "%"),
        },
        [GAS_SOURCE],
        "Одинаковая молярная либо объёмная основа и одинаковая нормировка. Влажность здесь только C1–C3; "
        "это не C1–C5. Генезис по одной молекулярной сухости не устанавливается.",
        ["Gas"],
        ("c1", "Bernard"),
    ),
    "gas_isotopes": method(
        "Изотопы газа · порядок C1–C3",
        {
            "d1": role("δ13C метана", "delta", ["Gas-Iso"], ["d13C C1 ["]),
            "d2": role("δ13C этана", "delta", ["Gas-Iso"], ["d13C C2 ["]),
            "d3": role("δ13C пропана", "delta", ["Gas-Iso"], ["d13C C3 ["]),
        },
        {
            "delta21": ("δ13C2 − δ13C1", "per mil"),
            "delta32": ("δ13C3 − δ13C2", "per mil"),
        },
        [ISO_SOURCE],
        "VPDB, сопоставимые измерения. Определяется нормальный, обратный или частично обратный порядок; "
        "это не однозначный диагноз смешения, биогенности или зрелости.",
        ["Gas-Iso"],
        ("delta21", "delta32"),
    ),
    "sara": method(
        "SARA · состав и баланс фракций",
        {
            "sat": role(
                "Насыщенные", "mass", ["SARA-Un", "SARA-FID", "SARA-Calc"], ["%Sat ["]
            ),
            "arom": role(
                "Ароматические",
                "mass",
                ["SARA-Un", "SARA-FID", "SARA-Calc"],
                ["%Arom ["],
            ),
            "res": role(
                "Смолы", "mass", ["SARA-Un", "SARA-FID", "SARA-Calc"], ["%Res ["]
            ),
            "asph": role(
                "Асфальтены",
                "mass",
                ["SARA-Un", "SARA-FID"],
                ["%Asph [", "%Asph FID ["],
            ),
        },
        {
            "sum": ("Сумма SARA", "%wt"),
            "HC": ("Sat+Arom", "%wt"),
            "nonHC": ("Res+Asph", "%wt"),
            "sat_arom": ("Sat/Arom", "1"),
            "asph_res": ("Asph/Res", "1"),
        },
        [],
        "Одна группа/метод и массовая основа. Сумма не нормируется к 100. Допуск баланса задаёт пользователь; "
        "преобладающая фракция описывает состав, а не степень биодеградации. SARA-Calc без %Asph не дополняется нулём.",
        ["SARA-Un", "SARA-FID", "SARA-Calc"],
        ("HC", "nonHC"),
    ),
    "elemental": method(
        "Элементный состав · атомные H/C и O/C",
        {
            "c": role("C, массовая доля", "mass", ["Elem"], ["C ["]),
            "h": role("H, массовая доля", "mass", ["Elem"], ["H ["]),
            "o": role("O, массовая доля", "mass", ["Elem"], ["O ["]),
        },
        {"HC": ("H/C atomic (12:1)", "1"), "OC": ("O/C atomic (12:16)", "1")},
        [ELEMENT_SOURCE],
        "Атомные отношения по номинальным массам C=12, H=1, O=16; H/C=12H/C, O/C=0.75O/C. "
        "Для диаграммы керогена подтвердите очищенный кероген и общую массовую основу. Тип керогена автоматически не назначается.",
        ["Elem"],
        ("OC", "HC"),
    ),
    "atomic_ratios": method(
        "Элементные отношения из БД · нормировка",
        {
            "hc": role("Молярное H/C", "atomic", ["Elem"], ["H/C mol ["]),
            "oc": role("Молярное O/C", "atomic", ["Elem"], ["O/C mol ["]),
        },
        {"HC": ("H/C atomic", "1"), "OC": ("O/C atomic", "1")},
        [ELEMENT_SOURCE],
        "Поле %mol (a/b) содержит отношение ×100: перевод в атомное отношение выполняется делением на 100. "
        "Массовое H/C wt не подставляется вместо молярного H/C mol.",
        ["Elem"],
        ("OC", "HC"),
    ),
    "gc_profile": method(
        "GC · Pr/Ph и изопреноиды/n-алканы",
        {
            "prph": role("Pr/Ph", "ratio", ["Sat-GC", "WO-GC"], ["Pr/Ph["]),
            "pr17": role("Pr/nC17", "ratio", ["Sat-GC", "WO-GC"], ["Pr/nC17["]),
            "ph18": role("Ph/nC18", "ratio", ["Sat-GC", "WO-GC"], ["Ph/nC18["]),
        },
        {"PrPh": ("Pr/Ph", "1"), "Pr17": ("Pr/nC17", "1"), "Ph18": ("Ph/nC18", "1")},
        [GC_SOURCE],
        "Одна фракция и один способ получения отношений (площадь/высота/концентрация). Профиль и QC; "
        "Pr/Ph не является однозначным индикатором редокса. Для полей интерпретации используйте проверенную схему Shapely.",
        ["Sat-GC", "WO-GC"],
        ("Pr17", "Ph18"),
    ),
    "alkane_indices": method(
        "GC · Индексы н-алканов CPI и OEP",
        {
            f"c{carbon}": role(f"nC{carbon}", "alkane_signal", ["Sat-GC", "WO-GC"],
                              [f"nC{carbon}[a]", f"nC{carbon}[ca]"])
            for carbon in range(24, 35)
        },
        {"CPI": ("CPI C24–C34", "1"), "OEP27": ("OEP C27", "1")},
        [ALKANE_INDEX_SOURCE, "https://doi.org/10.1016/0016-7037(70)90019-0"],
        "CPI Bray–Evans (C24–C34) и OEP Scalan–Smith (центр C27). Подтвердите один анализ и одну фракцию. "
        "Используйте сопоставимые площади пиков или массовые концентрации. Индексы требуют геологического контекста.",
        ["Sat-GC", "WO-GC"],
        ("CPI", "OEP27"),
    ),
    "stereo_profile": method(
        "GC–MS · стераны и гопаны",
        {
            "st20": role(
                "C29 20S/(20S+20R)", "fraction", ["Sat-GCMS"], ["St29 20S/R["]
            ),
            "stbb": role("C29 bb/(aa+bb)", "fraction", ["Sat-GCMS"], ["St29 bb/aa["]),
            "h31": role("C31 22S/(22S+22R)", "fraction", ["Sat-GCMS"], ["H31 22S/R["]),
        },
        {
            "St20S": ("C29 20S/(S+R)", "1"),
            "StBB": ("C29 bb/(aa+bb)", "1"),
            "H31S": ("C31 22S/(S+R)", "1"),
        },
        [],
        "Профиль стереоотношений: % (a/(a+b)) делится на 100. Формы S/R и S/(S+R) не отождествляются без определения единиц. "
        "Плато равновесия не переводится автоматически в Ro; исходная фация и вторичные процессы требуют отдельной оценки.",
        ["Sat-GCMS"],
        ("St20S", "StBB"),
    ),
    "sterane_distribution": method(
        "Регулярные стераны C27–C29 · ααα 20R",
        {f"st{carbon}": role(f"C{carbon} ααα 20R", "sterane_signal", ["Sat-GCMS"], [f"St{carbon}aaR["])
         for carbon in (27, 28, 29)},
        {f"C{carbon}": (f"C{carbon} ααα 20R / Σ(C27–C29 ααα 20R)", "%")
         for carbon in (27, 28, 29)},
        ["https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/MSDMethod.pdf"],
        "Три пика ααα 20R насыщенной фракции на одной аналитической основе. "
        "Доли рассчитываются от суммы этих трёх пиков. Тип источника автоматически не назначается.",
        ["Sat-GCMS"],
    ),
    "c27_diasterane_ratio": method(
        "C27 диастеран / регулярный стеран · расчёт по пикам",
        {
            key: role(label, "sterane_signal", ["Sat-GCMS"],
                      [f"{peak}[{tag}]" for tag in ("h", "a", "ch", "ca")])
            for key, label, peak in (
                ("dia", "C27 13β,17α-диастеран 20S", "St27dbS"),
                ("regular", "C27 ααα-стеран 20R", "St27aaR"),
            )
        },
        {"dia27_sr": ("C27 диастеран 20S / C27 ααα 20R", "1")},
        ["https://archives.datapages.com/data/bulletns/2021/03mar/BLTN19077/images/bltn19077.pdf"],
        "Отношение двух пиков насыщенной фракции на одной аналитической основе.",
        ["Sat-GCMS"],
    ),
    "ster_hop_from_peaks": method(
        "Ster/Hop · 17 стерановых и 16 гопановых пиков",
        {
            peak: role(peak, "terpane_signal", ["Sat-GCMS"], [f"{peak}[h]", f"{peak}[a]"])
            for peak in (*STER_HOP_STERANES, *STER_HOP_HOPANES)
        },
        {"ster_hop": ("Ster/Hop · 17/16 пиков", "1")},
        ["https://factpages.sodir.no/en/wellbore/pageview/other/scientific"],
        "Состав восстановлен по высотам и площадям одной пробы Stratum Reservoir в снимке Barents Sea. "
        "Привяжите все пики на одной основе. Совмещённый пик St27bbR+ используется целиком.",
        ["Sat-GCMS"],
    ),
    **REPORTED_BIOMARKER_METHODS,
    "terpane_hopane_profile": method(
        "Терпаны и гопаны · насыщенная фракция",
        {
            key: role(
                label, "terpane_signal", ["Sat-GCMS"],
                [f"{peak}[{tag}]" for tag in ("h", "a", "ch", "ca")],
                optional=True,
            )
            for key, label, peak in (
                ("t19", "C19 трициклический терпан", "t19"),
                ("t23", "C23 трициклический терпан", "t23"),
                ("tet24", "C24 тетрациклический терпан", "tet24"),
                ("gamm", "Гаммацеран", "Gamm"),
                ("h30", "C30 17α,21β-гопан", "H30ab"),
                ("ts", "C27 Ts", "H27Ts"),
                ("tm", "C27 Tm", "H27Tm"),
            )
        },
        {
            "C19_C23": ("C19TT/C23TT", "1"),
            "Tet24_C23": ("C24TeT/C23TT", "1"),
            "Gamm_C30": ("Гаммацеран/C30 αβ-гопан", "1"),
            "Ts_fraction": ("Ts/(Ts+Tm)", "1"),
        },
        ["https://factpages.sodir.no/pbl/geochemical_pdfs/3314_6.pdf"],
        "Привяжите хотя бы одну полную пару пиков на одной аналитической основе. "
        "При пропуске пика рассчитываются остальные доступные отношения.",
        ["Sat-GCMS"],
    ),
    "homohopane_distribution": method(
        "Распределение гомогопанов C31–C35",
        {
            key: role(
                f"C{carbon} 17α,21β · {label}", "terpane_signal", ["Sat-GCMS"],
                [f"H{carbon}ab{peak}[{tag}]" for tag in ("h", "a", "ch", "ca")],
                optional=True,
            )
            for carbon, keys in HOMOHOPANE_INPUTS.items()
            for key, peak, label in zip(keys, ("S+R", "S", "R"), ("сумма 22S+22R", "22S", "22R"))
        },
        {
            f"C{carbon}": (f"C{carbon} (22S+22R) / Σ(C31–C35)", "%")
            for carbon in HOMOHOPANE_INPUTS
        },
        ["https://factpages.sodir.no/pbl/geochemical_pdfs/5248_2.pdf",
         "https://archives.datapages.com/data/bulletns/2021/03mar/BLTN19077/images/bltn19077.pdf"],
        "Для каждого C31–C35 привяжите готовую сумму 22S+22R или оба эпимера на одной аналитической основе. "
        "Если доступны оба варианта, используется готовая сумма. Для расчёта нужны все пять гомологов.",
        ["Sat-GCMS"],
    ),
    "homohopane_epimers": method(
        "Эпимерные отношения гомогопанов C31–C35",
        {
            key: role(
                f"C{carbon} 17α,21β · 22{epimer}", "terpane_signal", ["Sat-GCMS"],
                [f"H{carbon}ab{epimer}[{tag}]" for tag in ("h", "a", "ch", "ca")],
                optional=True,
            )
            for carbon, (_, s, r) in HOMOHOPANE_INPUTS.items()
            for key, epimer in ((s, "S"), (r, "R"))
        },
        {f"C{carbon}_22S": (f"C{carbon} 22S/(22S+22R)", "1") for carbon in HOMOHOPANE_INPUTS},
        ["https://factpages.sodir.no/pbl/geochemical_pdfs/5248_2.pdf"],
        "Привяжите отдельные пики 22S и 22R хотя бы для одного гомолога. "
        "Отношения рассчитываются независимо по полным парам и выводятся в долях 0–1.",
        ["Sat-GCMS"],
    ),
    "aromatic_profile": method(
        "Ароматические GC–MS · MPI-1 и DBT/Phen",
        {
            "mpi": role("MPI-1", "ratio", ["Arom-GCMS"], ["MPI-1["]),
            "dbt": role("DBT/Phen", "ratio", ["Arom-GCMS"], ["DBT/PHEN["]),
        },
        {"MPI1": ("MPI-1", "1"), "DBTPhen": ("DBT/Phen", "1")},
        [],
        "Исходные индексы одного способа измерения. Расчёт Rc по MPI не выполняется без независимого выбора ветви и калибровки; "
        "Rc не отождествляется с измеренным Ro. Доступны профиль, QC и передача в геометрическую интерпретацию.",
        ["Arom-GCMS"],
        ("MPI1", "DBTPhen"),
    ),
    "aromatic_indices_from_peaks": method(
        "MPI-2, MPR и MDBTR · расчёт по пикам",
        {
            key: role(label, "terpane_signal", ["Arom-GCMS"],
                      [f"{peak}[{tag}]" for tag in ("h", "a", "ch", "ca")], optional=True)
            for key, label, peak in (
                ("phen", "Фенантрен", "Phen"), ("1mp", "1-метилфенантрен", "1MP"),
                ("2mp", "2-метилфенантрен", "2MP"), ("9mp", "9-метилфенантрен", "9MP"),
                ("4mdbt", "4-метилдибензотиофен", "4MDBT"), ("1mdbt", "1-метилдибензотиофен", "1MDBT"),
            )
        },
        {"mpi2": ("MPI-2", "1"), "mpr": ("MPR", "1"), "mdbtr": ("MDBTR", "1")},
        ["https://factpages.sodir.no/pbl/geochemical_pdfs/5248_2.pdf"],
        "MPI-2 = 3 × 2MP/(Phen+1MP+9MP), MPR = 2MP/1MP, MDBTR = 4MDBT/1MDBT. "
        "Привяжите пики для хотя бы одного индекса на одной аналитической основе.",
        ["Arom-GCMS"],
    ),
    "microscopy": method(
        "Микроскопия · макрогруппы керогена",
        {
            "vit": role("Витринит", "area", ["VisK-TL"], ["Vit ["]),
            "lipt": role("Липтинит", "area", ["VisK-TL"], ["Lipt ["]),
            "inert": role("Инертинит", "area", ["VisK-TL"], ["Inert ["]),
        },
        {
            "total": ("Vit+Lipt+Inert", "%area"),
            "Vit": ("Витринит", "%area"),
            "Lipt": ("Липтинит", "%area"),
            "Inert": ("Инертинит", "%area"),
        },
        [],
        "Проценты площади одной классификации. Преобладающая макрогруппа и проверка баланса; "
        "AOM/Algae и вложенные категории не добавляются к итогам повторно. Генерационный потенциал автоматически не выводится.",
        ["VisK-TL"],
        ("Vit", "Lipt"),
    ),
    "extract_toc": method(
        "Экстракт / органический углерод",
        {
            "extract": role(
                "Выход экстракта на исходную породу",
                "extract",
                ["Extract"],
                ["Extr yld ["],
            ),
            "toc": role(
                "TOC исходной породы (не пиролиз)", "mass", ["Leco", "Elem"], ["TOC ["]
            ),
        },
        {"EOM_TOC": ("Экстракт, mg/g TOC", "mg/g TOC")},
        [],
        "100*EOM_mg/g_rock/TOC_%wt. Требуются одна исходная проба и состояние до экстракции. "
        "Это отношение массы экстракта к массе органического углерода, не процент превращения и не самостоятельный диагноз миграции.",
        ["Extract", "Leco", "Elem"],
        same_group=False,
    ),
}


RALLAKIS_RATIOS = {
    "rallakis_gc": (
        ("Pr/Ph", ("Pr",), ("Ph",), 1),
        ("Pr/n-C17", ("Pr",), ("nC17",), 1),
        ("Ph/n-C18", ("Ph",), ("nC18",), 1),
        ("CPI1 · CPI нефти C22–C30", ("nC23", "nC25", "nC27", "nC29"),
         ("nC22", "nC24", "nC24", "nC26", "nC26", "nC28", "nC28", "nC30"), 2),
        ("OEP1 · OEP C23", ("nC21",) + ("nC23",) * 6 + ("nC25",), ("nC22", "nC24") * 4, 1),
        ("OEP2 · OEP C27", ("nC25",) + ("nC27",) * 6 + ("nC29",), ("nC26", "nC28") * 4, 1),
        ("CPI экстракта C25/(C24+C26)", ("nC25",), ("nC24", "nC26"), 1),
        ("n-C25/n-C30", ("nC25",), ("nC30",), 1),
        ("n-C15/n-C20", ("nC15",), ("nC20",), 1),
        ("n-C17/Pr", ("nC17",), ("Pr",), 1),
        ("n-C18/Ph", ("nC18",), ("Ph",), 1),
    ),
    "rallakis_saturated": (
        ("2 · Диагопан/(диагопан+норморетан)", ("H30D",), ("H30D", "H29ba"), 1),
        ("5 · 29Ts/(29Ts+норгопан)", ("H29Ts",), ("H29Ts", "H29ab"), 1),
        ("6 · Бисноргопан/(бисноргопан+норгопан)", ("H28",), ("H28", "H29ab"), 1),
        ("7 · Σ трициклических терпанов/C30 αβ-гопан",
         ("t23", "t24", "t25", "TT_T", "t28", "t29"), ("H30ab",), 1),
        ("9 · C30 αβ-гопан/Σ регулярных стеранов C29", ("H30ab",),
         ("St29aaS", "St29bbR", "St29bbS", "St29aaR"), 1),
        ("12 · Диастераны/(диастераны+регулярные стераны)", ("St27dbS", "St27dbR"),
         ("St27dbS", "St27dbR", "St29aaS", "St29bbR", "St29bbS", "St29aaR"), 1),
    ),
    "rallakis_steroids": (
        ("16 · C20 TA/(C20 TA+C28 TA)", ("TAS20",), ("TAS20", "TA28_g1"), 1),
        ("17 · C28 TA/(C28 TA+C29 MA)", ("TA28_g1",), ("TA28_g1", "MA29_H1"), 1),
    ),
    "rallakis_aromatics": (
        ("18 · MPR", ("2MP",), ("1MP",), 1),
        ("19 · MPI-1", ("2MP", "3MP"), ("Phen", "9MP", "1MP"), 1.5),
        ("20 · MPDF (F1)", ("2MP", "3MP"), ("1MP", "2MP", "3MP", "9MP"), 1),
        ("21 · MDR", ("4MDBT",), ("1MDBT",), 1),
        ("26 · 3-MP/4-MDBT", ("3MP",), ("4MDBT",), 1),
        ("27 · ΣMDBT/ΣMP", ("1MDBT", "3+2MDBT", "4MDBT"), ("1MP", "2MP", "3MP", "9MP"), 1),
    ),
}
RALLAKIS_RC = {
    "18": ("22 · Rc по MPR (Radke)", 1.1, 0.95),
    "19": ("23 · Rc по MPI-1 (возрастающая ветвь)", 0.60, 0.40),
    "20": ("24 · Rc по MPDF (Kvalheim)", 2.242, -0.166),
    "21": ("25 · Rc по MDR (Radke)", 0.073, 0.51),
}
RALLAKIS_SOURCE = "Rallakis (2015), Geochemical Comparison of Oils and Source Rocks from Barents Sea"


def rallakis_rc(index, value):
    """Calculate Rc from a measured or reported aromatic index."""
    if value is None or (index == "18" and value <= 0):
        return None
    _, slope, offset = RALLAKIS_RC[index]
    return slope * (math.log10(value) if index == "18" else value) + offset


def rallakis_peak(peak, groups, optional=True):
    label = {
        "TT_T": "Трициклический терпан T (m/z 191)",
        "TA28_g1": "C28 TA, пик g1 (m/z 231)",
        "MA29_H1": "C29 MA, пик H1 (m/z 253)",
    }.get(peak, f"Пик {peak}")
    return role(label, "terpane_signal", groups, [f"{peak}[{tag}]" for tag in ("h", "a", "ch", "ca")],
                optional=optional)


for method_id, name, groups, note in (
    ("rallakis_gc", "Rallakis · GC-FID", ["Sat-GC", "WO-GC"],
     "Сокращённый CPI предназначен для экстрактов. Отношения н-алканов соответствуют рис. 7.19–7.21."),
    ("rallakis_saturated", "Rallakis · гопаны, терпаны и стераны", ["Sat-GCMS"],
     "№ 7: P, Q, R, T, U, V; колонку TT_T привяжите вручную, так как T отсутствует в таблице 5.2.1. "
     "№ 9 и 12: четыре пика C29, m/z 217."),
    ("rallakis_steroids", "Rallakis · ароматические стероиды", ["Arom-GCMS"],
     "Привяжите именно пики g1 (C28 TA, m/z 231) и H1 (C29 MA, m/z 253) по рис. 5.2.3–5.2.4."),
    ("rallakis_aromatics", "Rallakis · ароматические индексы и Rc", ["Arom-GCMS"],
     "Расчётные Rc по калибровкам таблицы 5.3; для MPI-1 используется возрастающая ветвь. "
     "№ 27: четыре MP-изомера из табл. 5.2.6.1 и три сигнала MDBT (2+3 — совмещённый пик)."),
):
    formulas = RALLAKIS_RATIOS[method_id]
    peaks = dict.fromkeys(peak for _, numerator, denominator, _ in formulas for peak in numerator + denominator)
    outputs = {label.split()[0]: (label, "1") for label, *_ in formulas}
    if method_id == "rallakis_aromatics":
        outputs.update({label.split()[0]: (label, "%") for label, _, _ in RALLAKIS_RC.values()})
    METHODS[method_id] = method(
        name, {peak: rallakis_peak(peak, groups) for peak in peaks}, outputs,
        [RALLAKIS_SOURCE], note, groups,
    )
    METHODS[method_id]["ratio_formulas"] = formulas
    if method_id == "rallakis_aromatics":
        METHODS[method_id]["rc_calibrations"] = RALLAKIS_RC

METHODS["rallakis_steranes"] = method(
    "Rallakis · регулярные ββ 20S-стераны C27–C29",
    {f"st{c}": rallakis_peak(f"St{c}bbS", ["Sat-GCMS"], optional=False) for c in (27, 28, 29)},
    {f"C{c}": (f"C{c} ββ 20S", "%") for c in (27, 28, 29)},
    [RALLAKIS_SOURCE], "№ 13–15, пики i, o, s, m/z 218. Привяжите сопоставимые сигналы трёх изомеров.",
    ["Sat-GCMS"],
)

KNOWN_GROUPS = {g for definition in METHODS.values() for g in definition["groups"]}


def source_identity(column):
    metadata = column.metadata.get("geochem_field", {})
    if metadata:
        return metadata["group"], metadata["label"]
    if " / " in column.name:
        group, label = column.name.split(" / ", 1)
        if group in KNOWN_GROUPS or group.startswith("Py"):
            return group, label
    return "", column.name


def field_matches(group, label, definition):
    return group in definition["groups"] and any(
        label.startswith(prefix) for prefix in definition["prefixes"]
    )


def variant_tag(label):
    match = re.search(r"\[(a|ca|h|ch|un|vol|mol)\]", label)
    return match.group(1) if match else ""


def suggest_bindings(project, method_id, group="", variant=""):
    definitions = METHODS[method_id]["roles"]
    result = {}
    for key, definition in definitions.items():
        candidates = []
        for prefix in definition["prefixes"]:
            matches = [
                c
                for c in project.columns
                if (identity := source_identity(c))[0] in definition["groups"]
                and identity[1].startswith(prefix)
                and (
                    not group
                    or identity[0] == group
                    or not METHODS[method_id]["same_group"]
                )
                and (
                    not variant
                    or not variant_tag(identity[1])
                    or variant_tag(identity[1]) == variant
                )
            ]
            if matches:
                candidates = matches
                break
        if len(candidates) == 1:
            column = candidates[0]
            units = UNITS[definition["unit_family"]]
            if column.unit in units:
                result[key] = {"column_id": column.id, "unit": column.unit}
    return result


def validate_config(project, config, *, require_confirmation=True):
    if config.get("method") not in METHODS or config.get("version") != 1:
        raise ValueError("Неизвестная методика непиролитической интерпретации")
    definition = METHODS[config["method"]]
    if require_confirmation and (
        config.get("confirmed") is not True
        or not isinstance(config.get("context"), str)
        or not config["context"].strip()
    ):
        raise ValueError(
            "Укажите аналитический контекст и подтвердите применимость методики"
        )
    if (
        config["method"] in ("sofer", "gas_isotopes")
        and config.get("isotope_reference") != "VPDB"
    ):
        raise ValueError("Для углеродных изотопов подтвердите шкалу VPDB")
    if config["method"] == "sofer" and config.get("material") != "oil_C15plus":
        raise ValueError(
            "Sofer применяется к C15+ фракциям нефти, а не к пиролизату или керогену"
        )
    if not (0 <= config.get("closure_tolerance", 2) <= 100):
        raise ValueError("Допуск баланса должен быть 0–100 процентных пунктов")
    if not (0 <= config.get("comparison_tolerance", 0.05) <= 100):
        raise ValueError("Допуск сравнения CV должен быть 0–100")
    columns = {c.id: c for c in project.columns}
    inputs = config.get("inputs", {})
    terpane_profile = config["method"] == "terpane_hopane_profile"
    homohopanes = config["method"] == "homohopane_distribution"
    epimers = config["method"] == "homohopane_epimers"
    aromatic_peaks = config["method"] == "aromatic_indices_from_peaks"
    rallakis_ratios = config["method"] in RALLAKIS_RATIOS
    comparable_peaks = terpane_profile or homohopanes or epimers or aromatic_peaks or config["method"] in (
        "sterane_distribution", "c27_diasterane_ratio", "ster_hop_from_peaks", "rallakis_steranes",
    ) or rallakis_ratios
    if set(inputs) - set(definition["roles"]):
        raise ValueError("Неизвестные входы методики")
    if config["method"] in REPORTED_BIOMARKER_METHODS and not inputs:
        raise ValueError("Привяжите хотя бы один исходный показатель")
    required = {
        key for key, role_def in definition["roles"].items() if not role_def["optional"]
    }
    if not required <= set(inputs):
        raise ValueError(
            "Не привязаны обязательные показатели: "
            + ", ".join(sorted(required - set(inputs)))
        )
    if terpane_profile and not profile_inputs_complete(config["method"], inputs):
        raise ValueError("Привяжите хотя бы одну полную пару пиков для расчёта отношения")
    if homohopanes and not profile_inputs_complete(config["method"], inputs):
        raise ValueError("Для каждого C31–C35 привяжите сумму 22S+22R или оба эпимера")
    if epimers and not profile_inputs_complete(config["method"], inputs):
        raise ValueError("Привяжите пики 22S и 22R хотя бы для одного гомолога")
    if aromatic_peaks and not profile_inputs_complete(config["method"], inputs):
        raise ValueError("Привяжите полный набор пиков хотя бы для одного ароматического индекса")
    if rallakis_ratios and not profile_inputs_complete(config["method"], inputs):
        raise ValueError("Привяжите полный набор пиков хотя бы для одного показателя Rallakis")
    ids = [item.get("column_id") for item in inputs.values()]
    if len(set(ids)) != len(ids) or any(cid not in columns for cid in ids):
        raise ValueError("Выберите разные существующие колонки для входов")
    identities = []
    for key, item in inputs.items():
        role_def = definition["roles"][key]
        units = UNITS[role_def["unit_family"]]
        if item.get("unit") not in units:
            raise ValueError(f"{key}: неподдерживаемая единица")
        if config["method"] == "ster_hop_from_peaks" and item["unit"] != "count":
            raise ValueError("Для Ster/Hop выберите высоты или площади пиков в count")
        column = columns[item["column_id"]]
        if column.metadata.get("nonpyro") or (
            (comparable_peaks or config["method"] == "alkane_indices") and derived(column)
        ):
            raise ValueError(
                "Используйте исходный показатель, а не производный результат этого модуля"
            )
        group, label = source_identity(column)
        if group.startswith("Py") or (
            group == "Isotopes" and label.startswith("d13C Pyrol")
        ):
            raise ValueError("Пиролитические показатели исключены из этого модуля")
        if column.unit and (
            column.unit not in units or units[column.unit] != units[item["unit"]]
        ):
            raise ValueError(f"{column.name}: единица колонки противоречит декларации")
        if role_def["unit_family"] == "gas":
            source_basis = (
                "mol" if "mol" in column.unit else "vol" if column.unit else None
            )
            declared_basis = "mol" if "mol" in item["unit"] else "vol"
            if source_basis and source_basis != declared_basis:
                raise ValueError("Нельзя переобозначать объёмную основу как молярную")
        if group and not field_matches(group, label, role_def):
            raise ValueError(
                f"{key}: исходный показатель не соответствует роли методики"
            )
        if config["method"] == "alkane_indices" and group:
            expected_unit = {"a": "count", "ca": "ppm (µg/g)"}.get(variant_tag(label))
            if item["unit"] != expected_unit:
                raise ValueError("Единица н-алкана не соответствует основе измерения")
        if comparable_peaks and group:
            expected_unit = {"a": "count", "h": "count", "ca": "ppm (µg/g)", "ch": "ppm (µg/g)"}.get(variant_tag(label))
            if item["unit"] != expected_unit:
                raise ValueError("Единица пика не соответствует известной основе измерения")
        identities.append((group, label, item["unit"]))
    groups = {group for group, label, unit in identities if group}
    if definition["same_group"] and len(groups) > 1:
        raise ValueError(
            "Нельзя смешивать аналитические группы/фракции в одной методике"
        )
    variants = {
        variant_tag(label) for group, label, unit in identities if variant_tag(label)
    }
    if len(variants) > 1:
        raise ValueError(
            "Нельзя смешивать площадь, высоту, концентрацию или vol/mol/un"
        )
    if config.get("variant") and variants and variants != {config["variant"]}:
        raise ValueError("Способ измерения не совпадает с основой привязанных колонок")
    if config["method"] == "alkane_indices" and len({unit for _, _, unit in identities}) > 1:
        raise ValueError("Для CPI и OEP выберите одну единицу для всех н-алканов")
    if comparable_peaks and len({
        (group, variant_tag(label), unit) for group, label, unit in identities
    }) != 1:
        raise ValueError("Выберите пики с одной основой и единицей измерения")
    if config["method"] == "gas_ratios":
        bases = {"mol" if "mol" in unit else "vol" for group, label, unit in identities}
        normalization = {
            "HC" if "%HC[" in label else "THCG" if "%THCG[" in label else "bulk"
            for group, label, unit in identities
            if group
        }
        if len(bases) > 1 or len(normalization) > 1:
            raise ValueError(
                "Компоненты газа должны иметь одну основу и одну нормировку"
            )
    if config["method"] == "vitrinite":
        ro_label = source_identity(columns[inputs["ro"]["column_id"]])[1]
        if any(text in ro_label for text in ("Rework", "Cave", "Inert")):
            raise ValueError(
                "Переотложенная, обвальная или инертинитовая популяция не используется как Ro исходного витринита"
            )
        if "n" in inputs:
            n_label = source_identity(columns[inputs["n"]["column_id"]])[1]
            if (
                " pop " in ro_label
                and " pop " in n_label
                and ro_label.split(" pop ")[0] != n_label.split(" pop ")[0]
            ):
                raise ValueError(
                    "Число измерений должно относиться к той же популяции Ro"
                )
    return definition


def profile_measurements(project, row_id, column_ids):
    working = project.science.working.get(row_id, {})
    result = {}
    for column_id in column_ids:
        record = project.science.measurements.get(working.get(column_id))
        if record:
            result[column_id] = {"id": record.id, "revision": record.revision,
                                 "analysis_id": record.analysis_id, "basis": record.basis}
    return result


def profile_qc_exclusions(project, measurements):
    exclusions = project.view.get("qc_exclusions", {}).get("statistics", {})
    return {column_id: {"id": record["id"], "revision": record["revision"],
                        **({"comment": exclusion["comment"]} if exclusion.get("comment") else {})}
            for column_id, record in measurements.items()
            if (exclusion := exclusions.get(record["id"], {})).get("revision") == record["revision"]}


def _profile_source_snapshot(source):
    # The database importer stores ordinary scalar fields and a map of analysis runs.
    # Preserve arbitrary source metadata through the normal deep-copy path.
    if type(source) is dict and type(runs := source.get("analysis_runs")) is dict:
        scalar_types = (str, int, float, bool, type(None))
        if (all(type(key) is str and (key == "analysis_runs" or type(value) in scalar_types)
                for key, value in source.items())
                and all(type(group) is str and type(run) is dict
                        and all(type(key) is str and type(value) in scalar_types for key, value in run.items())
                        for group, run in runs.items())):
            result = source.copy()
            result["analysis_runs"] = {group: run.copy() for group, run in runs.items()}
            return result
    return snapshot_mapping(source)


def interpret_cell(project, row, config, columns=None, *, qc=None):
    definition = METHODS[config["method"]]
    columns = columns or {c.id: c for c in project.columns}
    values, warnings = {}, []
    if (
        config.get("scope_row_ids") is not None
        and row.id not in config["scope_row_ids"]
    ):
        return {
            "status": "outside_scope",
            "values": {},
            "interpretation": "Вне зафиксированной выборки",
            "warnings": [],
        }
    for key, item in config["inputs"].items():
        column = columns[item["column_id"]]
        if column.metadata.get("method"):
            return {
                "status": "interpolation_excluded",
                "values": {},
                "interpretation": "Интерполированные входы исключены",
                "warnings": [],
            }
        if qc and column.id in qc:
            status, value = "qc_excluded", None
        else:
            status, value = classify(
                row.values.get(column.id),
                numeric=column.kind == "number" or config.get("text_numbers", False),
            )
        if value is None:
            if definition["roles"][key]["optional"]:
                warnings.append(f"{key}: {status}")
                continue
            return {
                "status": status,
                "values": {},
                "interpretation": f"{key}: {status}",
                "warnings": warnings,
            }
        family = definition["roles"][key]["unit_family"]
        try:
            value = float(value) * UNITS[family][item["unit"]]
        except (OverflowError, ValueError):
            value = float("inf")
        if (
            not math.isfinite(value)
            or (family == "delta" and value <= -1000)
            or (family not in ("delta", "signed") and value < 0)
            or (family in ("mass", "ro", "area") and value > 100)
            or (family == "fraction" and value > 1)
            or (family == "gas" and value > 1e6)
        ):
            if config["method"] in REPORTED_BIOMARKER_METHODS:
                warnings.append(f"{key}: вне допустимого диапазона")
                continue
            return {
                "status": "invalid_range",
                "values": {},
                "interpretation": f"{key}: вне допустимого диапазона",
                "warnings": warnings,
            }
        if family == "count" and (value < 1 or not value.is_integer()):
            return {
                "status": "invalid_range",
                "values": {},
                "interpretation": "Количество измерений должно быть целым и положительным",
                "warnings": warnings,
            }
        if family == "delta" and value >= 0:
            warnings.append(
                f"{key}: неотрицательное δ13C; проверьте исходные данные, ноль не заменён пропуском"
            )
        values[key] = value
    return _interpret_values(row, config, columns, values, warnings)


def _source_notes(config, columns):
    groups = {source_identity(columns[item["column_id"]])[0] for item in config["inputs"].values()}
    quality_columns = [column for column in columns.values()
                       if (identity := source_identity(column))[0] in groups and identity[0] and identity[1] == "Quality"]
    undefined_basis = any(variant_tag(source_identity(columns[item["column_id"]])[1]) == "un"
                          for item in config["inputs"].values())
    return quality_columns, undefined_basis


def _interpret_values(row, config, columns, values, warnings, *, source_notes=None):
    """Format one validated result; scalar and batch paths share all scientific rules."""
    quality_columns, undefined_basis = source_notes if source_notes is not None else _source_notes(config, columns)
    for column in quality_columns:
        if row.values.get(column.id) not in (None, ""):
            group = source_identity(column)[0]
            warnings.append(f"Исходная Quality ({group}): {row.values[column.id]}; код не интерпретирован автоматически")
    if undefined_basis:
        warnings.append("Основа измерения [un] не определена; сопоставление с другими основами требует проверки")
    method_id = config["method"]
    status = "interpreted"
    output = {}
    text = ""

    def ratio(a, b, key):
        if b == 0:
            warnings.append(f"{key}: нулевой знаменатель")
            return None
        return a / b

    def share(a, b, key):
        scale = max(a, b) or 1
        scaled_a = a / scale
        return ratio(scaled_a, scaled_a + b / scale, key)

    if method_id == "sofer":
        cv = -2.53 * values["sat"] + 2.22 * values["arom"] - 11.65
        difference = cv - values["cv_reported"] if "cv_reported" in values else None
        output = {
            "CV": cv,
            "d13C_sat": values["sat"],
            "d13C_arom": values["arom"],
            "CV_difference": difference,
        }
        if difference is not None and abs(difference) > config.get(
            "comparison_tolerance", 0.05
        ):
            warnings.append(
                "Рассчитанный CV отличается от исходного CV больше объявленного допуска"
            )
        if math.isclose(cv, 0.47, abs_tol=1e-10, rel_tol=0):
            status, text = "boundary", "На разделяющей границе CV=0.47"
        else:
            text = (
                "Изотопный тип по Sofer: преимущественно "
                + ("терригенный" if cv > 0.47 else "морской")
                + "; статистическая гипотеза"
            )
    elif method_id == "vitrinite":
        ro = values["ro"]
        output = {"Ro": ro}
        label = source_identity(columns[config["inputs"]["ro"]["column_id"]])[1]
        if "Indig pop Mean" not in label and not config.get(
            "population_confirmed", False
        ):
            status, text = (
                "not_applicable",
                "Происхождение популяции не подтверждено; ориентировочная зрелость не назначена",
            )
        else:
            text = (
                "Ниже ориентировочного начала нефтеобразования"
                if ro < 0.5
                else "Переход начала нефтеобразования (0.5–0.6 %Ro)"
                if ro <= 0.6
                else "Ориентировочный нефтяной диапазон"
                if ro < 0.85
                else "Перекрывающийся переход нефтяного/газового диапазонов (0.85–1.3 %Ro)"
                if ro <= 1.3
                else "Ориентировочный газовый диапазон"
                if ro < 3
                else "Высокая зрелость; около/выше ориентировочного конца газового диапазона"
            )
            warnings.append(
                "Границы ориентировочные; зависят от типа керогена и истории нагрева"
            )
        if values.get("n", 0) < 40:
            warnings.append(
                "Число измерений не указано либо меньше ориентировочных 40–50; проверьте распределение и популяцию"
            )
    elif method_id == "gas_ratios":
        c1, c2, c3 = (values[k] for k in ("c1", "c2", "c3"))
        total = c1 + c2 + c3
        output = {
            "Bernard": ratio(c1, c2 + c3, "Bernard"),
            "dryness3": ratio(100 * c1, total, "dryness3"),
            "wetness3": ratio(100 * (c2 + c3), total, "wetness3"),
        }
        if total > 1e6:
            warnings.append(
                "Сумма C1–C3 превышает 100% на объявленной основе; проверьте нормировку"
            )
        text = "Молекулярный профиль C1–C3; генезис требует совместного анализа изотопов и геологии"
        status = "profile"
    elif method_id == "gas_isotopes":
        d21, d32 = values["d2"] - values["d1"], values["d3"] - values["d2"]
        output = {"delta21": d21, "delta32": d32}
        text = (
            "Нормальный порядок δ13C1 < δ13C2 < δ13C3"
            if d21 > 0 and d32 > 0
            else "Полная изотопная инверсия"
            if d21 < 0 and d32 < 0
            else "Частичная инверсия / совпадение значений"
        )
        warnings.append(
            "Порядок изотопов сам по себе не доказывает источник или механизм образования газа"
        )
    elif method_id == "sara":
        total = sum(values[k] for k in ("sat", "arom", "res", "asph"))
        output = {
            "sum": total,
            "HC": values["sat"] + values["arom"],
            "nonHC": values["res"] + values["asph"],
            "sat_arom": ratio(values["sat"], values["arom"], "Sat/Arom"),
            "asph_res": ratio(values["asph"], values["res"], "Asph/Res"),
        }
        maximum = max(values.values())
        names = {
            "sat": "насыщенные",
            "arom": "ароматические",
            "res": "смолы",
            "asph": "асфальтены",
        }
        dominant = [names[k] for k, value in values.items() if value == maximum]
        text = (
            "Преобладающая фракция: " + ", ".join(dominant)
            if len(dominant) == 1
            else "Максимумы совпадают: " + ", ".join(dominant)
        )
        if total == 0:
            status, text = (
                "not_applicable",
                "Все фракции нулевые; состав не классифицирован",
            )
        if abs(total - 100) > config.get("closure_tolerance", 2):
            warnings.append(
                f"Сумма SARA={total:.6g}%: баланс вне допуска; автоматическая нормировка не выполнялась"
            )
    elif method_id == "elemental":
        output = {
            "HC": ratio(12 * values["h"], values["c"], "H/C"),
            "OC": ratio(0.75 * values["o"], values["c"], "O/C"),
        }
        text = "Атомный профиль H/C–O/C по номинальным массам; тип керогена не назначен"
        status = "profile"
        warnings.append(
            "Остаточные минералы/карбонаты и неодинаковая подготовка влияют на отношения"
        )
    elif method_id == "atomic_ratios":
        output = {"HC": values["hc"], "OC": values["oc"]}
        text = "Атомные отношения приведены к 1, без множителя 100; тип керогена не назначен"
        status = "profile"
    elif method_id == "gc_profile":
        output = dict(
            zip(("PrPh", "Pr17", "Ph18"), (values[k] for k in ("prph", "pr17", "ph18")))
        )
        text = "GC-профиль; однозначная фациальная, генетическая или биодеградационная классификация по этим трём отношениям не выполняется"
        status = "profile"
    elif method_id == "alkane_indices":
        # Scaling by the largest signal preserves ratios and bounds the sums.
        scale = max(values.values()) or 1.0
        signals = {carbon: values[f"c{carbon}"] / scale for carbon in range(24, 35)}
        odd = sum(signals[carbon] for carbon in range(25, 34, 2))
        lower = ratio(odd, sum(signals[carbon] for carbon in range(24, 33, 2)), "CPI C24–C32")
        upper = ratio(odd, sum(signals[carbon] for carbon in range(26, 35, 2)), "CPI C26–C34")
        output = {
            "CPI": 0.5 * lower + 0.5 * upper if lower is not None and upper is not None else None,
            "OEP27": ratio(signals[25] + 6 * signals[27] + signals[29],
                           4 * (signals[26] + signals[28]), "OEP C27"),
        }
        text = "Индексы чётности н-алканов для подтверждённой основы измерений"
        status = "profile"
    elif method_id == "stereo_profile":
        output = {
            "St20S": values["st20"],
            "StBB": values["stbb"],
            "H31S": values["h31"],
        }
        text = "Стереоотношения в долях 0–1; количественная зрелость без калибровки не назначается"
        status = "profile"
    elif method_id in ("sterane_distribution", "rallakis_steranes"):
        scale = max(values.values()) or 1
        scaled = {f"C{carbon}": values[f"st{carbon}"] / scale for carbon in (27, 28, 29)}
        total = math.fsum(scaled.values())
        output = {key: ratio(100 * value, total, "Σ C27–C29") for key, value in scaled.items()}
        isomer = "ββ 20S" if method_id == "rallakis_steranes" else "ααα 20R"
        text = f"Доли регулярных стеранов {isomer} в сумме C27–C29"
        status = "profile"
    elif method_id in RALLAKIS_RATIOS:
        for label, numerator, denominator, multiplier in RALLAKIS_RATIOS[method_id]:
            peaks = set(numerator + denominator)
            if not peaks <= values.keys():
                continue
            scale = max(values[peak] for peak in peaks) or 1
            a = math.fsum(values[peak] / scale for peak in numerator)
            b = math.fsum(values[peak] / scale for peak in denominator)
            output[label.split()[0]] = ratio(multiplier * a, b, label)
        if method_id == "rallakis_aromatics":
            for index, (label, _, _) in RALLAKIS_RC.items():
                value = output.get(index)
                if value is None:
                    continue
                rc = rallakis_rc(index, value)
                if rc is None:
                    warnings.append("Rc по MPR: для log10 требуется MPR > 0")
                    continue
                output[label.split()[0]] = rc
        status, text = ("profile", "Показатели Rallakis по измеренным пикам") if output else (
            "not_applicable", "Нет полного набора пиков для показателей Rallakis"
        )
    elif method_id == "ster_hop_from_peaks":
        output = {"ster_hop": ster_hop_ratio(values)}
        if output["ster_hop"] is None:
            warnings.append("Ster/Hop: нулевой знаменатель")
        status, text = "profile", "Отношение суммы 17 стерановых и 16 гопановых пиков"
    elif method_id == "c27_diasterane_ratio":
        output = {"dia27_sr": ratio(values["dia"], values["regular"], "C27 диастеран / стеран")}
        status, text = "profile", "Отношение двух пиков C27"
    elif method_id in REPORTED_BIOMARKER_METHODS:
        output = values
        status, text = ("profile", "Исходные показатели в долях единицы") if output else (
            "not_applicable", "Нет допустимых исходных показателей"
        )
        if method_id == "aromatic_steroid_distribution":
            for family, keys in (("MAS", ("MAS27", "MAS28", "MAS29")), ("TAS", ("TAS26", "TAS27", "TAS28"))):
                if all(key in values for key in keys):
                    total = 100 * math.fsum(values[key] for key in keys)
                    if abs(total - 100) > config.get("closure_tolerance", 2):
                        warnings.append(f"{family}: сумма исходных долей {total:.6g}%")
    elif method_id == "aromatic_indices_from_peaks":
        for key, (peak, divisor, multiplier) in AROMATIC_INDEX_FORMULAS.items():
            if peak in values and all(item in values for item in divisor):
                scale = max(values[item] for item in (*divisor, peak)) or 1
                denominator = math.fsum(values[item] / scale for item in divisor)
                output[key] = ratio(multiplier * (values[peak] / scale), denominator,
                                    METHODS[method_id]["outputs"][key][0])
        status, text = ("profile", "Индексы по измеренным пикам") if output else (
            "not_applicable", "Нет полного набора пиков для индексов"
        )
    elif method_id == "terpane_hopane_profile":
        for key, (first, second) in TERPANE_HOPANE_PAIRS.items():
            if first not in values or second not in values:
                continue
            calculate = share if key == "Ts_fraction" else ratio
            output[key] = calculate(values[first], values[second], METHODS[method_id]["outputs"][key][0])
        status = "profile" if output else "not_applicable"
        text = "Отношения по доступным парам пиков" if output else "Нет полных пар пиков"
    elif method_id == "homohopane_distribution":
        if complete_homohopanes(values):
            signals = {
                f"C{carbon}": (values[total],) if total in values else (values[s], values[r])
                for carbon, (total, s, r) in HOMOHOPANE_INPUTS.items()
            }
            scale = max(map(max, signals.values())) or 1
            peaks = {key: sum(signal / scale for signal in group) for key, group in signals.items()}
            series_total = math.fsum(peaks.values())
            output = {key: ratio(100 * peak, series_total, "Σ C31–C35") for key, peak in peaks.items()}
            status = "profile"
            text = "Доли гомогопанов 17α,21β (22S+22R) в сумме C31–C35"
        else:
            status, text = "not_applicable", "Неполный ряд гомогопанов C31–C35"
    elif method_id == "homohopane_epimers":
        output = {
            f"C{carbon}_22S": share(values[s], values[r], f"C{carbon} 22S/(22S+22R)")
            for carbon, (_, s, r) in HOMOHOPANE_INPUTS.items()
            if s in values and r in values
        }
        status = "profile" if output else "not_applicable"
        text = "Доля 22S в сумме эпимеров каждого гомолога" if output else "Нет полных пар эпимеров"
    elif method_id == "aromatic_profile":
        output = {"MPI1": values["mpi"], "DBTPhen": values["dbt"]}
        text = "Ароматический профиль; ветвь MPI→Rc и фациальная схема не выбраны"
        status = "profile"
    elif method_id == "microscopy":
        total = sum(values.values())
        output = {
            "total": total,
            "Vit": values["vit"],
            "Lipt": values["lipt"],
            "Inert": values["inert"],
        }
        if abs(total - 100) > config.get("closure_tolerance", 2):
            warnings.append(
                f"Сумма трёх макрогрупп {total:.6g}%: проверьте полноту и исключительность классификации"
            )
        keys = [key for key, value in values.items() if value == max(values.values())]
        text = (
            "Преобладающая макрогруппа: " + ", ".join(keys)
            if len(keys) == 1
            else "Максимумы макрогрупп совпадают"
        )
        if total == 0:
            status, text = (
                "not_applicable",
                "Все макрогруппы нулевые; состав не классифицирован",
            )
    elif method_id == "extract_toc":
        output = {"EOM_TOC": ratio(100 * values["extract"], values["toc"], "EOM/TOC")}
        text = "Выход экстракта на органический углерод; самостоятельный диагноз миграции не выполняется"
        status = "profile"
    if any(value is not None and not math.isfinite(value) for value in output.values()):
        return {
            "status": "numerical_error",
            "values": {},
            "interpretation": "Численное переполнение",
            "warnings": warnings,
        }
    if output and all(value is None for value in output.values()):
        status = "zero_denominator"
    return {
        "status": status,
        "values": output,
        "canonical_inputs": values,
        "interpretation": text,
        "warnings": warnings,
    }



def sara_ternary_points(report):
    """Project SARA results onto Sat, Arom and Res+Asph vertices after the balance check."""
    tolerance = report["config"].get("closure_tolerance", 2)
    points = []
    for index, result in enumerate(report["results"]):
        values = result["values"]
        total = values.get("sum")
        if not total or abs(total - 100) > tolerance:
            continue
        arom = result["canonical_inputs"]["arom"] / total
        res_asph = values["nonHC"] / total
        # Sat is the lower-left vertex, Arom the lower-right and Res+Asph the top.
        points.append((index, arom + res_asph / 2, res_asph * math.sqrt(3) / 2))
    return points


def sterane_ternary_points(report):
    """Project C27, C28 and C29 onto the left, right and top vertices."""
    return [
        (index, (values["C28"] + values["C29"] / 2) / 100,
         values["C29"] * math.sqrt(3) / 200)
        for index, result in enumerate(report["results"])
        if result["status"] == "profile" and (values := result["values"])
    ]


def alkane_distribution(result):
    """Return (carbon number, percent of the C24–C34 signal) from a calculation result."""
    values = result.get("canonical_inputs", {})
    carbons = range(24, 35)
    if not all(f"c{carbon}" in values for carbon in carbons):
        return []
    signals = [values[f"c{carbon}"] for carbon in carbons]
    scale = max(signals)
    if scale == 0:
        return []
    scaled = [signal / scale for signal in signals]
    total = math.fsum(scaled)
    return [(carbon, 100 * signal / total) for carbon, signal in zip(carbons, scaled)]


_BATCH_METHODS = {"gas_ratios", "sara", "elemental", "atomic_ratios", "gc_profile",
                  "stereo_profile", "aromatic_profile", "microscopy", "extract_toc"}


def _interpret_rows(project, rows, config, columns, qc=None, *, check_rows=None):
    """Prepare required profile inputs by column, preserving the first failure."""
    qc = qc or {}
    definition = METHODS[config["method"]]
    if qc or config["method"] not in _BATCH_METHODS or len(rows) < 256:
        return [{"row_id": row.id, **interpret_cell(project, row, config, columns, qc=qc.get(row.id))}
                for row in (rows if check_rows is None else check_rows(rows))]
    frame = pd.DataFrame([row.values for row in rows], index=pd.Index([row.id for row in rows], dtype=object),
                         columns=[item["column_id"] for item in config["inputs"].values()], dtype=object)
    status = pd.Series("ready", index=frame.index, dtype=object)
    reasons = pd.Series(None, index=frame.index, dtype=object)
    canonical = pd.DataFrame(index=frame.index)
    if config.get("scope_row_ids") is not None:
        outside = ~frame.index.isin(config["scope_row_ids"])
        status.loc[outside], reasons.loc[outside] = "outside_scope", "Вне зафиксированной выборки"
    for key, item in config["inputs"].items():
        column = columns[item["column_id"]]
        active = status.eq("ready")
        if column.metadata.get("method"):
            status.loc[active], reasons.loc[active] = "interpolation_excluded", "Интерполированные входы исключены"
            canonical[key] = np.nan
            continue
        data = classify_series(frame[column.id], numeric=column.kind == "number" or config.get("text_numbers", False))
        missing = active & data["number"].isna()
        status.loc[missing] = data.loc[missing, "status"]
        reasons.loc[missing] = key + ": " + data.loc[missing, "status"]
        family = definition["roles"][key]["unit_family"]
        with np.errstate(over="ignore", invalid="ignore"):
            values = data["number"].astype(float) * UNITS[family][item["unit"]]
        invalid = ~np.isfinite(values)
        if family not in ("delta", "signed"):
            invalid |= values.lt(0)
        if family in ("mass", "ro", "area"):
            invalid |= values.gt(100)
        if family == "fraction":
            invalid |= values.gt(1)
        if family == "gas":
            invalid |= values.gt(1e6)
        invalid &= status.eq("ready")
        status.loc[invalid], reasons.loc[invalid] = "invalid_range", f"{key}: вне допустимого диапазона"
        canonical[key] = values
    source_notes = _source_notes(config, columns)
    results = []
    items = zip(rows, status, reasons, canonical.to_dict("records"), strict=True)
    for row, state, reason, inputs in (items if check_rows is None else check_rows(items)):
        item = (_interpret_values(row, config, columns, inputs, [], source_notes=source_notes)
                if state == "ready" else dict(status=state, values={}, interpretation=reason, warnings=[]))
        results.append({"row_id": row.id, **item})
    return results


def _profile_selection(project, row_ids, config):
    ids = list(row_ids)
    rows = project.row_map()
    if len(set(ids)) != len(ids) or set(ids) - set(rows):
        raise ValueError("Некорректная выборка проб")
    columns = {c.id: c for c in project.columns}
    cell_config = config
    if config.get("scope_row_ids") is not None:
        # Prepare membership once; keep the ordered, serializable protocol intact.
        cell_config = dict(config)
        cell_config["scope_row_ids"] = set(config["scope_row_ids"])
    return ids, rows, columns, cell_config


def _profile_qc_snapshot(project, measurements):
    if not project.view.get("qc_exclusions", {}).get("statistics"):
        return {}
    return {rid: excluded for rid, records in measurements.items()
            if (excluded := profile_qc_exclusions(project, records))}


def preview_profile_coverage(project, row_ids, config):
    """Check effective row coverage without assembling or saving a report."""
    definition = validate_config(project, config, require_confirmation=False)
    ids, rows, columns, cell_config = _profile_selection(project, row_ids, config)
    qc = {}
    if project.view.get("qc_exclusions", {}).get("statistics"):
        column_ids = {item["column_id"] for item in config["inputs"].values()}
        qc = _profile_qc_snapshot(project, {rid: profile_measurements(project, rid, column_ids) for rid in ids})
    counts = dict.fromkeys(definition["outputs"], 0)
    failures = {}
    with_values = 0
    for item in _interpret_rows(project, [rows[rid] for rid in ids], cell_config, columns, qc):
        available = [key for key, value in item["values"].items() if value is not None]
        if available:
            with_values += 1
            for key in available:
                counts[key] += 1
        else:
            key = (item["status"], item["interpretation"])
            bucket = failures.setdefault(key, {"status": key[0], "reason": key[1], "count": 0, "examples": []})
            bucket["count"] += 1
            if len(bucket["examples"]) < 3:
                bucket["examples"].append(item["row_id"])
    return {"total": len(ids), "with_values": with_values, "outputs": counts,
            "failures": sorted(failures.values(), key=lambda item: -item["count"])}


def evaluate_project(project, row_ids, config, *, _snapshot_qc=None, _snapshot_measurements=None,
                     checkpoint=None):
    definition = validate_config(project, config)
    ids, rows, columns, cell_config = _profile_selection(project, row_ids, config)

    def checked(items, start, end):
        for index, item in enumerate(items):
            if index % 2048 == 0:
                checkpoint(start + (end - start) * index // len(ids))
            yield item

    cid_set = {item["column_id"] for item in config["inputs"].values()}
    measurement_ids = ids if checkpoint is None else checked(ids, 0, 30)
    measurements = ({rid: records for rid in measurement_ids if (records := profile_measurements(project, rid, cid_set))}
                    if _snapshot_measurements is None else deepcopy(_snapshot_measurements))
    qc = _profile_qc_snapshot(project, measurements) if _snapshot_qc is None else deepcopy(_snapshot_qc)
    results = _interpret_rows(project, [rows[rid] for rid in ids], cell_config, columns, qc,
                              check_rows=None if checkpoint is None else lambda items: checked(items, 30, 60))
    groups = {source_identity(columns[cid])[0] for cid in cid_set}
    cid_set.update(
        c.id
        for c in project.columns
        if (identity := source_identity(c))[0] in groups
        and identity[0]
        and identity[1]
        in {
            "Lab",
            "Lab#",
            "Date",
            "Method",
            "Eqp",
            "Prep",
            "Quality",
            "Lab comment",
            "Sam type",
        }
    )
    inputs = [
        {
            "row_id": rid,
            "values": snapshot_mapping({cid: rows[rid].values.get(cid) for cid in cid_set}),
            "source": _profile_source_snapshot(rows[rid].source),
            "measurements": measurements.get(rid, {}),
            **({"qc_exclusions": qc[rid]} if rid in qc else {}),
        }
        for rid in (ids if checkpoint is None else checked(ids, 60, 90))
    ]
    return {
        "format": "autopyro.nonpyro",
        "version": 1,
        "created_at": timestamp(),
        "autopyro_version": __version__,
        "project_id": project.id,
        "project_source": deepcopy(project.source),
        "stale": False,
        "row_ids": ids,
        "config": deepcopy(config),
        "definition": deepcopy(definition),
        "definition_hash": hashlib.sha256(
            json.dumps(definition, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "columns": [
            {
                "id": cid,
                "name": columns[cid].name,
                "kind": columns[cid].kind,
                "unit": columns[cid].unit,
                "metadata": {
                    k: deepcopy(v)
                    for k, v in columns[cid].metadata.items()
                    if k not in ("diagnostics", "proposals")
                },
            }
            for cid in sorted(cid_set)
        ],
        "inputs": inputs,
        "results": results,
        "summary": dict(Counter(r["status"] for r in results)),
    }


def replay_report(report):
    from autopyro.model.core import Project, Row

    if report.get("format") != "autopyro.nonpyro" or report.get("version") != 1:
        raise ValueError("Неизвестный формат протокола")
    definition = METHODS[report["config"]["method"]]
    current_hash = hashlib.sha256(
        json.dumps(definition, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    embedded_hash = hashlib.sha256(
        json.dumps(report["definition"], sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    if (
        current_hash != report["definition_hash"]
        or embedded_hash != report["definition_hash"]
    ):
        raise ValueError(
            "Определение методики изменилось; требуется соответствующая версия ядра"
        )
    project = Project(
        id=report["project_id"],
        source=deepcopy(report.get("project_source", {})),
        columns=[Column(**deepcopy(item)) for item in report["columns"]],
        rows=[
            Row(
                id=item["row_id"],
                values=deepcopy(item["values"]),
                source=deepcopy(item["source"]),
            )
            for item in report["inputs"]
        ],
    )
    qc = {item["row_id"]: item["qc_exclusions"] for item in report["inputs"] if item.get("qc_exclusions")}
    measurements = {item["row_id"]: item["measurements"] for item in report["inputs"] if "measurements" in item}
    return evaluate_project(project, report["row_ids"], report["config"], _snapshot_qc=qc,
                            _snapshot_measurements=measurements)


def check_saved_reports(project, reports=None):
    columns = {c.id: c for c in project.columns}
    rows = project.row_map()
    if reports is None:
        reports = project.view.get("nonpyro_results", {}).values()
    for report in reports:
        try:
            definition = validate_config(project, report["config"])
            current_hash = hashlib.sha256(
                json.dumps(definition, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            changed = current_hash != report[
                "definition_hash"
            ] or project.source != report.get("project_source", {})
            for item in report["columns"]:
                column = columns[item["id"]]
                changed |= any(
                    getattr(column, key) != item[key]
                    for key in ("name", "kind", "unit")
                )
                changed |= {
                    k: v
                    for k, v in column.metadata.items()
                    if k not in ("diagnostics", "proposals")
                } != item["metadata"]
            input_ids = [binding["column_id"] for binding in report["config"]["inputs"].values()]
            for item in report["inputs"]:
                row = rows[item["row_id"]]
                changed |= row.source != item["source"] or any(
                    row.values.get(cid) != value for cid, value in item["values"].items()
                )
                measurements = profile_measurements(project, row.id, input_ids)
                if "measurements" in item:
                    changed |= measurements != item["measurements"]
                active_qc = profile_qc_exclusions(project, measurements)
                saved_qc = item.get("qc_exclusions", {})
                # Comments document the run. Exclusion identity determines its numerical validity.
                changed |= {cid: (entry["id"], entry["revision"]) for cid, entry in active_qc.items()} != {
                    cid: (entry["id"], entry["revision"]) for cid, entry in saved_qc.items()
                }
            report["stale"] = bool(changed)
        except (ValueError, KeyError, TypeError):
            report["stale"] = True


def result_columns(config):
    definition = METHODS[config["method"]]
    result = []
    for key, (label, unit) in definition["outputs"].items():
        result.append(
            Column(
                f"{definition['name']} · {label}",
                "number",
                unit=unit,
                metadata={"nonpyro": {"config": deepcopy(config), "output": key}},
            )
        )
    for key, label in (
        ("interpretation", "Интерпретация"),
        ("status", "Статус"),
        ("warnings", "Оговорки / QC"),
    ):
        result.append(
            Column(
                f"{definition['name']} · {label}",
                metadata={"nonpyro": {"config": deepcopy(config), "output": key}},
            )
        )
    return result


def sofer_scheme(project, x_id, y_id, row_ids):
    """Published linear discriminator as clipped Shapely regions; no redrawn domains."""
    from shapely.geometry import box, LineString, mapping
    from shapely.ops import split

    columns = {c.id: c for c in project.columns}
    selected = set(row_ids)
    points = [
        (r.values[x_id], r.values[y_id])
        for r in project.rows
        if r.id in selected
        and r.values.get(x_id) is not None
        and r.values.get(y_id) is not None
    ]
    if not points:
        raise ValueError("Нет сопоставимых пар δ13C для диаграммы Sofer")
    xs, ys = zip(*points)
    padx, pady = max(1, (max(xs) - min(xs)) * 0.2), max(1, (max(ys) - min(ys)) * 0.2)
    xmin, xmax, ymin, ymax = (
        min(xs) - padx,
        max(xs) + padx,
        min(ys) - pady,
        max(ys) + pady,
    )
    domain = box(xmin, ymin, xmax, ymax)
    line = LineString(
        [
            (xmin, (0.47 + 11.65 + 2.53 * xmin) / 2.22),
            (xmax, (0.47 + 11.65 + 2.53 * xmax) / 2.22),
        ]
    )
    regions = []
    for i, geometry in enumerate(split(domain, line).geoms):
        point = geometry.representative_point()
        terrigenous = -2.53 * point.x + 2.22 * point.y - 11.65 > 0.47
        regions.append(
            {
                "id": f"sofer-{i}",
                "label": "Терригенный изотопный тип (Sofer)"
                if terrigenous
                else "Морской изотопный тип (Sofer)",
                "color": "#e69f00" if terrigenous else "#56b4e9",
                "geometry": mapping(geometry),
            }
        )
    clipped = line.intersection(domain)
    scheme = {
        "format": "autopyro.interpretation",
        "version": 1,
        "id": "nonpyro-sofer",
        "revision": "1",
        "name": "Sofer CV=0.47 · C15+ нефти",
        "source": SOFER,
        "scientific_status": "validated",
        "coordinate_space": "analysis",
        "boundary_policy": "flag",
        "tolerance": 1e-10 / math.hypot(2.53, 2.22),
        "x": {
            "property": columns[x_id].name,
            "unit": "per mil",
            "transform": "identity",
            "minimum": xmin,
            "maximum": xmax,
        },
        "y": {
            "property": columns[y_id].name,
            "unit": "per mil",
            "transform": "identity",
            "minimum": ymin,
            "maximum": ymax,
        },
        "regions": regions,
        "lines": []
        if clipped.geom_type != "LineString" or clipped.length == 0
        else [{"id": "CV047", "label": "CV=0.47", "geometry": mapping(clipped)}],
    }
    # The clipping window is part of the immutable geometry definition.
    scheme["id"] += (
        "-"
        + hashlib.sha256(json.dumps(scheme, sort_keys=True).encode()).hexdigest()[:16]
    )
    return scheme


def recompute(project):
    groups = {}
    for column in project.columns:
        if column.metadata.get("nonpyro"):
            config = column.metadata["nonpyro"]["config"]
            key = json.dumps(config, sort_keys=True, ensure_ascii=False)
            groups.setdefault(key, []).append(column)
    for serialized, columns in groups.items():
        config = json.loads(serialized)
        try:
            report = evaluate_project(project, [r.id for r in project.rows], config)
            result = {item["row_id"]: item for item in report["results"]}
        except (ValueError, KeyError, TypeError) as exc:
            result = {
                row.id: {
                    "status": "invalid_definition",
                    "interpretation": str(exc),
                    "warnings": [],
                    "values": {},
                }
                for row in project.rows
            }
        for column in columns:
            key = column.metadata["nonpyro"]["output"]
            for row in project.rows:
                item = result[row.id]
                row.values[column.id] = (
                    "; ".join(item["warnings"])
                    if key == "warnings"
                    else item[key]
                    if key in ("status", "interpretation")
                    else item["values"].get(key)
                )
            column.metadata["diagnostics"] = {
                rid: {
                    key: deepcopy(item[key])
                    for key in ("status", "interpretation", "warnings")
                }
                for rid, item in result.items()
            }
