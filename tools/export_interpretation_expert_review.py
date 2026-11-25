"""Freeze the bundled interpretation catalog and its technical review evidence."""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
import re
from types import SimpleNamespace

import shapely

from autopyro.data_io.file_output import atomic_output
from autopyro.schemes.interpretation import (
    GeometricEngine, InterpretationInput, scheme_hash, validate_axis_binding, validate_scheme,
)
from autopyro.schemes.reference_cases import reference_failures
from autopyro.schemes.standard_interpretation import standard_schemes


_AKRAM_PAPER = 'https://www.diva-portal.org/smash/get/diva2%3A1532119/FULLTEXT01.pdf'
_EDRESS_PAPER = 'https://www.degruyterbrill.com/document/doi/10.1515/geo-2020-0306/html'
_RYAZANOVA_POSTER = 'https://www.crru.ru/conference/stend2023/gener_potencial_ryazanova.pdf'
_IDENTIFIED_FIGURES = {
    'svg:S1+S2-TOC:e56fa7fa0899': (
        'https://www.researchgate.net/publication/316159963_HYDROCARBON_POTENTIALITY_OF_SOME_MIOCENE_ROCK_UNITS_IN_THE_CENTRAL_PART_OF_THE_GULF_OF_SUEZ_EGYPT',
        'рис. 14',
    ),
    'svg:S2-TOC:b9255d809b81': (
        'https://publications.gc.ca/collections/collection_2017/rncan-nrcan/M183-2/M183-2-8018-eng.pdf#page=20',
        'рис. 4',
    ),
    'svg:HI-Tmax:3780d6ee1fe0': (
        'https://www.researchgate.net/publication/268461249_Oil-Oil_Correlation_and_Potential_Source_Rocks_in_the_Chotts_Basin_Tunisia',
        'рис. 5B, 9 и 13A',
    ),
    'svg:HI-Tmax:4c20f476085b': (
        'https://repository.geologyscience.ru/server/api/core/bitstreams/d253d059-a9b8-459d-bcb7-c94d1fe9c2ad/content',
        'рис. 7',
    ),
    'svg:HI-Tmax:a1f1c826f301': ('https://old.geors.ru/media/pdf/07_Protcko_s.pdf#page=3', 'рис. 4, с. 53'),
    'svg:HI-OI:7093b510ed99': ('https://www.ngtp.ru/upload/iblock/96c/40_2022.pdf#page=16', 'рис. 8а'),
    'svg:HI-OI:bee7e9eb73be': (
        'https://repository.geologyscience.ru/server/api/core/bitstreams/d77d604a-2054-47db-9b5e-5aefc4577005/content',
        'рис. 6',
    ),
    'svg:HI-Tmax:74624a1ffb7e': (
        'https://sciencejournals.ru/view-article/?a=GeoKhim2103003Borisova&j=geokhim&n=3&v=66&y=2021',
        'рис. 1',
    ),
    'svg:HI-Tmax:9d41858f217d': (
        'https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2024.1371358/full#F5',
        'рис. 5',
    ),
    'svg:TOCo:4c82336c6d05': ('https://doi.org/10.1306/13321446M973489', 'рис. 6A'),
    'svg:HI-OI:04b8097cd514': (
        'https://www.researchgate.net/publication/279592953_Characterization_of_the_Black_Shales_of_the_Temburong_Formation_in_West_Sabah_East_Malaysia',
        'рис. 4',
    ),
    'svg:HI-Tmax:0258ef283fb9': ('https://www.mdpi.com/2075-163X/11/8/811', 'рис. 6'),
    'svg:HI-Tmax:838bc85e9290': (_AKRAM_PAPER, 'рис. 3'),
    'svg:PI-Tmax:6aa20f026e5f': (_AKRAM_PAPER, 'рис. 4'),
    'svg:S2-TOC:df50f2543c30': (_EDRESS_PAPER, 'рис. 5'),
    'svg:S1+S2-TOC:1e97d702ca61': (_EDRESS_PAPER, 'рис. 6'),
    'svg:HI-OI:8d3c3c0d68be': (_EDRESS_PAPER, 'рис. 7'),
    'svg:PI-Tmax:779eeb130103': (_EDRESS_PAPER, 'рис. 10'),
    'svg:HI-Tmax:e143c8a66a14': (_RYAZANOVA_POSTER, 'рис. 2а'),
    'svg:S2-TOC:5c345b9ca9e3': (_RYAZANOVA_POSTER, 'рис. 2б'),
    'svg:S1+S2-TOC:98e85d91ef97': ('https://www.mdpi.com/2076-3263/11/8/307#fig_body_display_geosciences-11-00307-f002', 'рис. 2'),
    'svg:S2-TOC:667b0a10f2e2': ('https://link.springer.com/article/10.1007/s13202-020-00939-9#Fig5', 'рис. 5'),
    'svg:HI-Tmax:7db83cea51ca': ('https://link.springer.com/article/10.1007/s12303-018-0013-4#Fig2', 'рис. 2'),
    'svg:PI-Tmax:f1e42f41698c': ('https://www.mdpi.com/1996-1073/11/6/1406', 'рис. 9'),
}
_IDENTIFIED_PUBLICATIONS = {
    'svg:HI-OI:b15ed7d16439': 'https://doi.org/10.2516/ogst:1977002',
    'svg:HI-OI:5d1cf6e0d0b3': 'https://pubs.usgs.gov/publication/70219079',
    'svg:HI-Tmax:ebdd91983b82': 'https://doi.org/10.1016/j.jaap.2004.11.012',
}
_SOURCE_REVIEW_NOTES = {
    'svg:HI-OI:b15ed7d16439': (
        'Рис. 5 в [отчёте Rock-Eval по скважине 30/7-7]'
        '(https://factpages.sodir.no/pbl/geochemical_pdfs/390_2_Rock_eval_pyrolysis_of_samples_from_well_30_7_7.pdf) '
        'показывает HI–OI. Номер рисунка в оригинале Espitalié остаётся открытым. '
        'Проверка 17.11.2026 по DOI 10.2516/ogst:1977002 и адресам PDF издателя '
        'завершилась HTTP 403. Запись ResearchGate '
        '[239385451](https://www.researchgate.net/publication/239385451) содержит '
        'описание статьи и запрос полного текста. Для подтверждения нужна страница '
        'оригинала с подписью диаграммы HI–OI.'
    ),
    'svg:HI-OI:5d1cf6e0d0b3': 'Запись USGS подтверждает авторов и год. Панель HI–OI в статье пока не подтверждена.',
    'svg:HI-Tmax:ebdd91983b82': 'Совпали авторы и год. Связь конкретной диаграммы HI–Tmax со статьёй требует сверки.',
    'svg:HI-Tmax:f1671e93f078': (
        '[Работа о влиянии минеральной матрицы](https://doi.org/10.1016/0146-6380(84)90059-7) '
        'исследует кероген типа II. Происхождение набора кривых HI–Tmax остаётся открытым.'
    ),
    'svg:HI-Tmax:c174608a53bb': 'Для атрибуции нужны название кафедры или исходный учебный материал.',
    'svg:HI-Tmax:73cbf5c2f7f3': 'Найдены работы автора по баженовской свите. Для атрибуции нужны название работы или год.',
    'svg:HI-TOC:0361ff38ba45': (
        'Близкий аналог по осям и названиям полей есть на [рис. 7c]'
        '(https://cup.edu.cn/petroleumscience/docs/2019-08/20150301.pdf). '
        'Происхождение SVG требует сверки границ.'
    ),
}


def source_register(evidence, repository):
    rows = [
        '# Реестр источников схем', '',
        'Локальные SVG сверяются с хешем каталога при наличии файла. '
        'Найденный рисунок требует отдельной сверки с оцифрованной геометрией.', '',
        '| Схема и редакция | Источник каталога | Рисунок или страница | Что осталось проверить |',
        '| --- | --- | --- | --- |',
    ]
    for item in evidence:
        scheme = item['scheme']
        source = scheme['source']
        provenance = scheme.get('provenance', {})
        expected_hash = provenance.get('source_sha256')
        figure = '—'
        if expected_hash:
            path = repository / source
            if path.is_file():
                matches = sha256(path.read_bytes()).hexdigest() == expected_hash
                file_status = 'SVG совпал по SHA-256' if matches else 'SHA-256 SVG отличается от каталога'
            else:
                file_status = 'SVG отсутствует в рабочей копии'
            source = f'`{source}`<br>SHA-256 SVG: `{expected_hash}`'
            citation = _IDENTIFIED_FIGURES.get(scheme['id'])
            if citation:
                url, label = citation
                figure = f'[{label}]({url})'
                remaining = f'{file_status}. Сверить геометрию с рисунком.'
            elif scheme['id'] in _IDENTIFIED_PUBLICATIONS:
                url = _IDENTIFIED_PUBLICATIONS[scheme['id']]
                figure = f'[Публикация]({url})'
                remaining = f'{file_status}. Установить номер рисунка.'
            else:
                remaining = f'{file_status}. Установить публикацию и рисунок.'
        elif provenance.get('figure') and provenance.get('printed_page'):
            figure = f"рис. {provenance['figure']}, с. {provenance['printed_page']}"
            remaining = 'Сверить геометрию с указанным рисунком.'
        else:
            remaining = 'Уточнить страницу или таблицу для каждого правила.'
        if scheme['id'] == 'standard:HIo':
            remaining += ' Отдельно проверить модель интерполяции HIo.'
        if scheme['id'] in _SOURCE_REVIEW_NOTES:
            remaining += ' ' + _SOURCE_REVIEW_NOTES[scheme['id']]
        rows.append(f"| {scheme['name']} · {scheme['revision']} | {source} | {figure} | {remaining} |")
    return '\n'.join(rows) + '\n'


def unit_checks(axis):
    unit = axis['unit']
    incompatible = '°C' if unit == '%wt' else '%wt'
    checks = []
    for name, column_unit, declared, expected in (
        ('matching', unit, unit, 'accepted'),
        ('missing-declaration', '', '', 'rejected'),
        ('explicit-declaration', '', unit, 'accepted'),
        ('incompatible-declaration', unit, incompatible, 'rejected'),
        ('incompatible-column', incompatible, unit, 'rejected'),
    ):
        column = SimpleNamespace(name=axis['property'], unit=column_unit, metadata={})
        try:
            validate_axis_binding(column, declared, axis)
        except ValueError as error:
            observed, message = 'rejected', str(error)
        else:
            observed, message = 'accepted', None
        checks.append(dict(id=name, column_unit=column_unit, declared_unit=declared,
                           expected=expected, actual=observed, message=message))
    return checks


def review_packets():
    engine = GeometricEngine()
    blind, evidence = [], []
    for scheme in standard_schemes().values():
        validate_scheme(scheme)
        digest = scheme_hash(scheme)
        cases = scheme['reference_cases']
        inputs = [{key: case[key] for key in ('id', 'x', 'y', 'x_bounds', 'y_bounds') if key in case}
                  for case in cases]
        samples = [InterpretationInput(case['id'], case['x'], case['y'],
                                       case.get('x_bounds'), case.get('y_bounds')) for case in inputs]
        blind.append({**{key: scheme[key] for key in ('id', 'name', 'revision', 'source', 'x', 'y')},
                      'sha256': digest, 'cases': inputs})
        evidence.append(dict(sha256=digest, scheme=scheme,
                             actual=[asdict(result) for result in engine.evaluate(samples, scheme)],
                             failures=reference_failures(scheme),
                             unit_checks={key: unit_checks(scheme[key]) for key in ('x', 'y')}))
    return blind, evidence


def update_decision_sheet(path, schemes):
    existing = path.read_text(encoding='utf-8') if path.exists() else ''
    # Match only the checksum column, so a reference in a comment cannot suppress a new row.
    recorded = set(re.findall(r'^\|[^|]*\|[^|]*\|\s*`?([0-9a-f]{64})`?\s*\|',
                              existing, flags=re.MULTILINE))
    pending = [item for item in schemes if item['sha256'] not in recorded]
    if not pending:
        return
    title = '## Дополнительные редакции' if existing else '# Лист экспертных решений'
    rows = [title, '',
            '| Схема | Редакция | SHA-256 | Эталонов | Решение | Основание |',
            '| --- | --- | --- | ---: | --- | --- |']
    for item in pending:
        rows.append(f"| {item['name']} | {item['revision']} | `{item['sha256']}` "
                    f"| {len(item['cases'])} | Ожидает | |")
    # Preserve the original bytes, including the expert's line endings.
    previous = path.read_bytes() if path.exists() else b''
    addition = ('\n\n' if previous else '') + '\n'.join(rows) + '\n'
    with atomic_output(path) as temporary:
        temporary.write_bytes(previous + addition.encode('utf-8'))


def retain_previous_evidence(output, new_files):
    previous = {name: (output / name).read_bytes()
                for name in new_files if (output / name).exists()}
    if not previous or all(content == new_files[name] for name, content in previous.items()):
        return
    hashes = {name: sha256(content).hexdigest() for name, content in previous.items()}
    digest = sha256(json.dumps(hashes, sort_keys=True).encode('utf-8')).hexdigest()
    archive = output / 'history' / digest
    archive.mkdir(parents=True, exist_ok=True)
    # Preserve the complete earlier evidence before replacing any current files.
    for name, content in previous.items():
        target = archive / name
        if not target.exists():
            with atomic_output(target) as temporary:
                temporary.write_bytes(content)


def verify_evidence(path):
    """Replay an exported snapshot without substituting the installed catalog."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('format_version') not in (1, 2) or not isinstance(payload.get('schemes'), list):
        raise ValueError('Неизвестный формат технической выгрузки')
    failures = []
    records = payload['schemes']
    if not records:
        return ['В технической выгрузке отсутствуют схемы.']
    identities = [(item['scheme']['id'], item['scheme']['revision']) for item in records]
    if len(set(identities)) != len(identities):
        failures.append('В технической выгрузке повторяются редакции схем.')
    if payload['format_version'] == 2:
        observed = [dict(id=item['scheme']['id'], revision=item['scheme']['revision'],
                         sha256=item['sha256'], cases=len(item['scheme'].get('reference_cases', [])))
                    for item in records]
        if payload.get('manifest') != observed:
            failures.append('Состав схем, редакций или эталонов отличается от описи выгрузки.')
    engine = GeometricEngine()
    for item in records:
        scheme = item['scheme']
        validate_scheme(scheme)
        label = f"{scheme['id']} · {scheme['revision']}"
        if scheme_hash(scheme) != item['sha256']:
            failures.append(f'{label} — контрольная сумма схемы отличается')
            continue
        cases = scheme.get('reference_cases', [])
        if not cases:
            failures.append(f'{label} — отсутствуют эталоны')
        samples = [InterpretationInput(case['id'], case['x'], case['y'],
                                       case.get('x_bounds'), case.get('y_bounds')) for case in cases]
        actual = [asdict(result) for result in engine.evaluate(samples, scheme)]
        saved = item['actual']
        if [row['row_id'] for row in saved] != [row['row_id'] for row in actual]:
            failures.append(f'{label} — состав или порядок проб отличается')
        else:
            for previous, current in zip(saved, actual, strict=True):
                missing = sorted(current.keys() - previous.keys())
                extra = sorted(previous.keys() - current.keys())
                if missing:
                    failures.append(f"{label} — {current['row_id']} · отсутствуют поля {', '.join(missing)}")
                if extra:
                    failures.append(f"{label} — {current['row_id']} · неизвестные поля {', '.join(extra)}")
                for key in current:
                    if key in previous and previous[key] != current[key]:
                        failures.append(f"{label} — {current['row_id']} · {key}")
        current_failures = reference_failures(scheme)
        if current_failures or item.get('failures'):
            failures.append(f'{label} — есть расхождения с эталонами')
        for axis in ('x', 'y'):
            checks = unit_checks(scheme[axis])
            if checks != item['unit_checks'][axis] or any(c['actual'] != c['expected'] for c in checks):
                failures.append(f'{label} — проверки единиц оси {axis} отличаются')
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output', type=Path)
    mode.add_argument('--verify', type=Path, help='Replay a saved 02-technical-evidence.json')
    args = parser.parse_args()
    if args.verify is not None:
        failures = verify_evidence(args.verify)
        print('\n'.join(failures) if failures else 'Сохранённая выгрузка воспроизведена без расхождений.')
        return int(bool(failures))
    output = args.output
    blind, evidence = review_packets()
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for filename, records in (('01-blind-inputs.json', blind), ('02-technical-evidence.json', evidence)):
        payload = dict(format_version=2, python=platform.python_version(),
                       shapely=shapely.__version__, geos=shapely.geos_version_string, schemes=records)
        payload['manifest'] = [
            dict(id=item['id'], revision=item['revision'], sha256=item['sha256'], cases=len(item['cases']))
            for item in blind
        ]
        files[filename] = (json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)
                           + '\n').encode('utf-8')
    files['04-sources.md'] = source_register(
        evidence, Path(__file__).resolve().parents[1]).encode('utf-8')
    retain_previous_evidence(output, files)
    for filename, content in files.items():
        with atomic_output(output / filename) as temporary:
            temporary.write_bytes(content)
    update_decision_sheet(output / '03-decisions.md', blind)

    failed_cases = sum(len(item['failures']) for item in evidence)
    failed_units = sum(check['actual'] != check['expected'] for item in evidence
                       for checks in item['unit_checks'].values() for check in checks)
    print(f'Schemes: {len(blind)}, cases: {sum(len(item["cases"]) for item in blind)}, '
          f'failed assertions: {failed_cases}, failed unit checks: {failed_units}')
    return int(bool(failed_cases or failed_units))


if __name__ == '__main__':
    raise SystemExit(main())
