"""Plain-language explanation from the immutable interpretation report."""
from .interpretation import STATUSES
from .interpretation_thresholds import threshold_labels


def explain_sample(report, position, *, sample_label=None, highlights=()):
    scheme = report['scheme']
    item, sample = report['results'][position], report['inputs'][position]
    lines = [f"Проба: {sample_label}"] if sample_label else []
    lines.append(f"Результат: {STATUSES[item['status']]}")
    labels = {e['id']: e['label'] for e in scheme.get('regions', []) + scheme.get('lines', [])}
    for key, title in [('regions', 'Области'), ('boundary_regions', 'Границы'),
                       ('possible_types', 'Возможные типы вещества')]:
        if item.get(key):
            lines.append(title + ': ' + '; '.join(labels.get(i, i) for i in item[key]))
    lines.extend(highlights)
    if item.get('value') is not None:
        lines.append(f"HIo: {item['value']:g} {scheme['y']['unit']}")
    if item.get('value_bounds'):
        bounds = '–'.join(f'{v:g}' for v in item['value_bounds'])
        lines.append(f"Диапазон HIo: {bounds} {scheme['y']['unit']}")
    if item['status'] == 'ambiguous':
        lines.append('Причина неоднозначности: ' + item.get('reason', 'Перекрытие областей или близкие траектории'))
    elif item['status'] in ('not_applicable', 'insufficient_data') and item.get('reason'):
        lines.append('Обоснование: ' + item['reason'])
    lines.extend(threshold_labels(scheme, item.get('thresholds', {})))
    issues = sample.get('quality', [])
    lines.extend(['', 'Контроль качества до интерпретации:'])
    unchecked = {'qc_excluded', 'interpolation_excluded', 'unavailable', 'unit_unchecked'}
    skipped = {issue.get('column_id') for issue in issues if issue['code'] in unchecked}
    checked = [column['name'] for column in report.get('quality_policy', {}).get('columns', ())
               if column['id'] not in skipped]
    if checked:
        lines.append('Диапазоны проверены: ' + ', '.join(checked))
    lines.extend(f"{'ОШИБКА' if q['severity'] == 'error' else 'Предупреждение'}: {q['message']}" for q in issues)
    if not issues:
        lines.append('Предупреждений по проверенным показателям нет.')
    lines.extend(['', f"Схема: {scheme['name']} · редакция {scheme['revision']}"])
    for key in ('x', 'y'):
        axis = scheme[key]
        lines.append(f"{axis['property']} [{axis['unit']}]: {sample[key]}")
        bounds = sample.get(f'{key}_bounds')
        metadata = sample.get('uncertainty', {}).get(key)
        if bounds is not None:
            lines.append(f"Интервал {axis['property']}: {bounds[0]:g}–{bounds[1]:g} {axis['unit']}")
        if metadata:
            lines.append(f"Заданная неопределённость: {metadata['uncertainty']:g}; тип: {metadata['type'] or 'не указан'}")
            if metadata.get('confidence_level') is not None:
                lines.append(f"Указанный уровень доверия: {metadata['confidence_level']:g}")
    lines.append('')
    for key, title in [
        ('line_hits', 'Родительские кривые HIo' if scheme.get('strategy') == 'initial_hi' else 'Кривые'),
        ('uncertainty_regions', 'Области неопределённости'),
    ]:
        if item.get(key):
            lines.append(title + ': ' + '; '.join(labels.get(i, i) for i in item[key]))
    region_ids = set(item['regions'] + item['boundary_regions'] + item.get('uncertainty_regions', []))
    if region_ids:
        for region in scheme.get('regions', []):
            passport = region.get('passport')
            if region['id'] in region_ids and passport:
                lines.append(f"Паспорт области «{region['label']}»")
                for key, title in (('source', 'Источник'), ('description', 'Описание'), ('limitations', 'Ограничения')):
                    if passport.get(key):
                        lines.append(f"{title}: {passport[key]}")
    details = item.get('interval_details', {})
    if details:
        lines.append('Интервальный метод: адаптивные оценки расстояний для всей области неопределённости')
        lines.append(f"Вычислений: {details.get('evaluations', 0)}; консервативная оболочка: {bool(details.get('conservative') or details.get('types_conservative'))}")
        if details.get('hi_bound_gap') is not None:
            lines.append(f"Зазор внешних границ HIo относительно подтверждённых значений: {details['hi_bound_gap']:g}")
    if item.get('distances'):
        lines.append('Расстояния в нормированном пространстве: ' + '; '.join(
            f"{labels.get(key, key)}: {value:g}" for key, value in item['distances'].items()))
        lines.append(scheme.get('metric_note', ''))
    if item['status'] not in ('ambiguous', 'not_applicable', 'insufficient_data') and item.get('reason'):
        lines.append('Обоснование: ' + item['reason'])
    lines.extend(['', f"UUID пробы: {item['row_id']}", f"ID схемы: {scheme['id']}",
                  f"Статус схемы: {scheme['scientific_status']}", f"Источник: {scheme['source']}",
                  f"Стратегия: {scheme.get('strategy', 'regions')}",
                  f"Контрольная сумма схемы: {report['scheme_hash']}"])
    if scheme.get('limitations'):
        lines.extend(['', 'Ограничения: ' + scheme['limitations']])
    lines.append('Типичные диапазоны — пользовательские ориентиры, не самостоятельный геологический диагноз.')
    return '\n'.join(lines)
