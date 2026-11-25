"""Conservative, linear-time matching of imported rock names to catalog codes."""

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import unicodedata


RULES_KEY = 'stratigraphy_lithology_mapping'
RULES_FORMAT = 'autopyro.lithology_rules'
ORIGINAL_SOURCE = '@original_lithology'
SOURCE_FIELDS = {'lithology', 'lithology_name', 'rock', 'rock_name', 'порода',
                 'литология', 'название породы'}
# Generic sandstone uses the catalog's plain sandstone pattern. Structured or
# mixed descriptions are deliberately not matched by substrings or fuzzy search.
ALIASES = {
    'песчаник': '607', 'sandstone': '607', 'массивный песчаник': '607',
    'слоистый песчаник': '608', 'глинистый песчаник': '612',
    'известковистый песчаник': '613', 'доломитистый песчаник': '614',
    'алевролит': '616', 'глина': '620', 'аргиллит': '620b',
    'глинистый сланец': '624', 'горючий сланец': '625', 'мел': '626',
    'известняк': '627', 'доломит': '642', 'доломитовая порода': '642',
    'торф': '657', 'уголь': '658', 'бентонит': '662',
    'гипс': '667', 'каменная соль': '668', 'кварцит': '702', 'гнейс': '708',
    'гранит': '719', 'кварц': '732',
}


def normalize_rock(value):
    text = '' if value is None else str(value)
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().replace('ё', 'е').split())


def decode_rules(payload):
    """Validate a portable rule dictionary before it reaches the project."""
    if (not isinstance(payload, dict) or payload.get('format') != RULES_FORMAT
            or type(payload.get('version')) is not int or payload['version'] != 1
            or not isinstance(payload.get('rules'), dict)):
        raise ValueError('Неподдерживаемый формат словаря пород.')
    rules = {}
    for name, code in payload['rules'].items():
        key = normalize_rock(name)
        if not key or not isinstance(code, str) or key in rules:
            raise ValueError('В словаре есть пустое или повторяющееся название либо неверный код.')
        rules[key] = code.strip()
    return rules


class LithologyMatcher:
    def __init__(self, entries, saved=None):
        self.codes = {}
        self.names = defaultdict(set)
        self.saved = saved or {}
        for code, name in entries:
            if not code:
                continue
            self.codes[normalize_rock(code)] = code
            if not code.endswith('*'):
                self.names[normalize_rock(name)].add(code)

    def match(self, value):
        key = normalize_rock(value)
        if key in self.codes:
            return self.codes[key], 'code'
        if key in self.saved:
            code = self.saved[key]
            if code == '' or normalize_rock(code) in self.codes:
                return self.codes.get(normalize_rock(code), ''), 'saved'
        names = self.names.get(key, set())
        if len(names) == 1:
            return next(iter(names)), 'name'
        if len(names) > 1:
            return None, 'ambiguous'
        code = ALIASES.get(key)
        if code in self.codes:
            return self.codes[code], 'alias'
        return None, 'unknown'


def source_value(interval, field):
    if field == ORIGINAL_SOURCE:
        return interval.get('lithology_source', interval.get('lithology', ''))
    return interval.get('lithology', '') if field is None else interval.get('raw_cells', {}).get(field, '')


def source_columns(scheme):
    return sorted({key for item in scheme['intervals'] for key in item.get('raw_cells', {})})


def default_source(scheme, columns):
    if any(normalize_rock(item.get('lithology')) for item in scheme['intervals']):
        return None
    candidates = [column for column in columns if normalize_rock(column) in SOURCE_FIELDS]
    return candidates[0] if len(candidates) == 1 else None


@dataclass
class RockMatch:
    value: str
    count: int
    code: str | None
    reason: str


class LithologyMapping:
    def __init__(self, scheme, matcher, field=None):
        self.scheme, self.field = scheme, field
        self.groups = {}
        for item in scheme['intervals']:
            value = source_value(item, field)
            key = normalize_rock(value)
            if not key:
                continue
            if key not in self.groups:
                code, reason = matcher.match(value)
                self.groups[key] = RockMatch(str(value), 0, code, reason)
            self.groups[key].count += 1

    def apply(self):
        result = deepcopy(self.scheme)
        for item in result['intervals']:
            value = source_value(item, self.field)
            group = self.groups.get(normalize_rock(value))
            if group is None:
                continue
            code = group.code or ''
            if item.get('lithology', '') != code:
                item.setdefault('lithology_source', str(value))
                item.setdefault('lithology_source_field',
                                'lithology_source' if self.field == ORIGINAL_SOURCE else self.field or 'lithology')
                item['lithology'] = code
        return result

    def learned_rules(self):
        return {key: group.code for key, group in self.groups.items() if group.reason == 'manual'}
