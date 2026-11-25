"""Validated source traces attached to a project sample."""

from hashlib import sha256
import csv
from dataclasses import dataclass
from io import StringIO
import json
import math
from pathlib import Path
from autopyro.model.measurements import parse_number


MAX_FILE_BYTES = 16 * 1024 * 1024


def _text(record, key, context):
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{context}: требуется непустое поле {key}')
    return value.strip()


def _series(trace, key, context):
    values = trace.get(key)
    if not isinstance(values, list) or len(values) < 2 or any(
        type(value) not in (int, float) or not math.isfinite(value) for value in values
    ):
        raise ValueError(f'{context}: {key} должен содержать не менее двух конечных чисел')
    return values


def _marks(trace, key, fields, size, context):
    marks = trace.get(key, [])
    if not isinstance(marks, list):
        raise ValueError(f'{context}: {key} должен быть массивом')
    for mark in marks:
        if not isinstance(mark, dict):
            raise ValueError(f'{context}: неверная запись в {key}')
        _text(mark, 'label', context)
        indices = [mark.get(field) for field in fields]
        if any(type(i) is not int or not 0 <= i < size for i in indices) or indices != sorted(indices):
            raise ValueError(f'{context}: неверные индексы в {key}')


def _read_source(path):
    path = Path(path)
    with path.open('rb') as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError('Файл пирограммы превышает 16 МиБ')
    return path.name, data


def read_pyrogram(path, project, sample_id):
    name, data = _read_source(path)
    source = data.decode('utf-8-sig')
    if Path(name).suffix.lower() == '.s00':
        run = _rock_eval_run(source, Path(name).stem)
    else:
        run = json.loads(source)
    return _attachment(name, sha256(data).hexdigest(), run, project, sample_id)


def verify_pyrogram_source(path, attachment):
    """Compare source bytes with the imported source hash, including table mappings."""
    _, data = _read_source(path)
    return sha256(data).hexdigest() == attachment['sha256']


def _rock_eval_run(source, label):
    # The two sections have independent time axes and different detector columns.
    layouts = {
        '[curves pyro]': ('Пиролиз', ('FID', 'CO', 'CO₂')),
        '[curves oxi]': ('Окисление', ('CO', 'CO₂')),
    }
    sections = {}
    columns = None
    for line_number, line in enumerate(source.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if line.startswith('['):
            section = line.casefold()
            if section not in layouts or section in sections:
                raise ValueError(f'Строка {line_number}: неизвестная или повторная секция S00 {line}')
            columns = [[] for _ in range(2 + len(layouts[section][1]))]
            sections[section] = columns
            continue
        if columns is None:
            raise ValueError(f'Строка {line_number}: ожидается заголовок секции S00')
        fields = line.split()
        if len(fields) != len(columns):
            raise ValueError(f'Строка {line_number}: ожидается {len(columns)} столбцов S00')
        try:
            values = tuple(float(field) for field in fields)
        except ValueError as exc:
            raise ValueError(f'Строка {line_number}: ожидаются числовые значения S00') from exc
        for column, value in zip(columns, values):
            column.append(value)

    traces = []
    for section, (stage, channels) in layouts.items():
        if section not in sections:
            raise ValueError(f'Отсутствует секция S00: {stage}')
        times, temperatures, *signals = sections[section]
        for channel, signal in zip(channels, signals):
            traces.append({
                'stage': stage, 'channel': channel, 'unit': 'Исходные единицы S00',
                'time': times, 'temperature': temperatures, 'signal': signal,
            })
    return {
        'format': 'autopyro.pyrogram', 'version': 1, 'label': label,
        'time_unit': 's', 'temperature_unit': '°C', 'traces': traces,
    }


def _attachment(name, digest, run, project, sample_id):
    if (not isinstance(run, dict) or run.get('format') != 'autopyro.pyrogram'
            or type(run.get('version')) is not int or run['version'] != 1):
        raise ValueError('Ожидается формат autopyro.pyrogram версии 1')
    run['label'] = _text(run, 'label', 'Пирограмма')
    if run.get('time_unit') != 's' or run.get('temperature_unit', '°C') != '°C':
        raise ValueError('Поддерживаемые единицы времени и температуры: s и °C')
    if not isinstance(run.get('heating_program', ''), str):
        raise ValueError('Поле heating_program должно быть строкой')
    if 'analysis_id' in run:
        analysis_id = _text(run, 'analysis_id', 'Пирограмма')
        analysis = project.science.analyses.get(analysis_id)
        if analysis is None or analysis.sample_id != sample_id:
            raise ValueError('Указанный анализ отсутствует у выбранной пробы')
        run['analysis_id'] = analysis_id
    traces = run.get('traces')
    if not isinstance(traces, list) or not traces:
        raise ValueError('Добавьте хотя бы один ряд в traces')
    keys = set()
    for index, trace in enumerate(traces, 1):
        context = f'traces[{index}]'
        if not isinstance(trace, dict):
            raise ValueError(f'{context}: ожидается объект')
        key = _text(trace, 'channel', context), _text(trace, 'stage', context)
        if key in keys:
            raise ValueError(f'{context}: канал и стадия уже присутствуют в этом запуске')
        keys.add(key)
        trace['channel'], trace['stage'] = key
        trace['unit'] = _text(trace, 'unit', context)
        times = _series(trace, 'time', context)
        signal = _series(trace, 'signal', context)
        series = [signal]
        if 'temperature' in trace:
            if 'temperature_unit' not in run:
                raise ValueError('Для ряда temperature укажите temperature_unit')
            series.append(_series(trace, 'temperature', context))
        if any(len(values) != len(times) for values in series):
            raise ValueError(f'{context}: time, signal и temperature должны быть одинаковой длины')
        time_order = trace.get('time_order', 'increasing')
        if time_order not in ('increasing', 'nondecreasing'):
            raise ValueError(f'{context}: неизвестный порядок времени')
        if time_order == 'nondecreasing':
            if times[-1] <= times[0] or any(right < left for left, right in zip(times, times[1:])):
                raise ValueError(f'{context}: время не должно убывать, ряд должен иметь ненулевую длительность')
        elif any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError(f'{context}: время должно строго возрастать')
        _marks(trace, 'peaks', ('index',), len(times), context)
        _marks(trace, 'intervals', ('start', 'end'), len(times), context)
    # Reject non-finite values even in optional source metadata before project save.
    json.dumps(run, allow_nan=False)
    return {'file': name, 'sha256': digest, 'run': run}


@dataclass(frozen=True)
class PyrogramTable:
    file: str
    sha256: str
    text: str
    delimiter: str
    encoding: str
    headers: tuple[str, ...]
    header_line: int = 1

    def rows(self):
        return self._reader(self.header_line, self.delimiter)

    def _reader(self, header_line, delimiter):
        stream = StringIO(self.text, newline='')
        for _ in range(header_line - 1):
            if not stream.readline():
                break
        return csv.reader(stream, delimiter=delimiter, strict=True)

    def with_layout(self, header_line, delimiter=None):
        if type(header_line) is not int or header_line < 1:
            raise ValueError('Укажите номер строки заголовка')
        if delimiter not in (None, '\t', ';', ','):
            raise ValueError('Неизвестный разделитель пирограммы')
        delimiters = (delimiter,) if delimiter is not None else ('\t', ';', ',')
        candidates = []
        for separator in delimiters:
            reader = self._reader(header_line, separator)
            try:
                fields = next(reader, [])
            except csv.Error:
                continue
            headers = tuple(field.strip() for field in fields)
            if len(headers) < 2 or not all(headers) or len(set(headers)) != len(headers):
                continue
            try:
                first = next((row for row in reader if any(value.strip() for value in row)), [])
            except csv.Error:
                first = []
            candidates.append((len(first) == len(headers), separator, headers))
        if not candidates:
            raise ValueError(f'Строка {header_line}: нужны непустые уникальные заголовки колонок')
        _, separator, headers = max(candidates, key=lambda item: item[0])
        return PyrogramTable(self.file, self.sha256, self.text, separator, self.encoding, headers, header_line)


def read_pyrogram_table(path, *, allow_unmapped=False):
    """Read a headed delimited export without guessing physical quantities."""
    name, data = _read_source(path)
    try:
        text, encoding = data.decode('utf-8-sig'), 'utf-8-sig'
    except UnicodeDecodeError:
        text, encoding = data.decode('cp1251'), 'cp1251'
    header_line = next((index for index, line in enumerate(StringIO(text, newline=''), 1)
                        if line.strip()), 1)
    delimiter = '\t' if Path(name).suffix.lower() == '.tsv' else None
    table = PyrogramTable(name, sha256(data).hexdigest(), text, delimiter or ';', encoding, (), header_line)
    try:
        return table.with_layout(header_line, delimiter)
    except ValueError:
        if allow_unmapped:
            return table
        raise


def _clock_seconds(text):
    parts = text.strip().split(':')
    if len(parts) != 3 or not all(part.isascii() and part.isdecimal() for part in parts[:2]):
        raise ValueError('ожидается время ЧЧ:ММ:СС')
    hours, minutes = map(int, parts[:2])
    seconds = parse_number(parts[2])
    if hours >= 24 or minutes >= 60 or seconds is None or not 0 <= seconds < 60:
        raise ValueError('ожидается время ЧЧ:ММ:СС в пределах суток')
    return hours * 3600 + minutes * 60 + seconds


def mapped_pyrogram(table, project, sample_id, *, time, signal, temperature=None,
                    time_unit='s', label, channel, stage, unit, first_line=None, last_line=None):
    columns = {'time': time, 'signal': signal}
    if temperature is not None:
        columns['temperature'] = temperature
    if (any(type(index) is not int or not 0 <= index < len(table.headers) for index in columns.values())
            or len(set(columns.values())) != len(columns)):
        raise ValueError('Выберите разные колонки времени, сигнала и температуры')
    if time_unit not in ('s', 'min', 'clock'):
        raise ValueError('Выберите секунды, минуты или время ЧЧ:ММ:СС')
    trace = {'channel': channel, 'stage': stage, 'unit': unit, **{key: [] for key in columns}}
    reader = table.rows()
    next(reader)
    header_end = table.header_line + reader.line_num - 1
    if (first_line is not None and (type(first_line) is not int or first_line <= header_end)
            or last_line is not None and (type(last_line) is not int or last_line < (first_line or header_end + 1))):
        raise ValueError('Укажите диапазон строк данных после заголовка')
    line_number = header_end
    clock_start = previous_clock = None
    day_offset = 0
    try:
        while last_line is None or line_number < last_line:
            start = line_number + 1
            row = next(reader, None)
            if row is None:
                break
            line_number = table.header_line + reader.line_num - 1
            if first_line is not None and line_number < first_line:
                continue
            if (first_line is not None and start < first_line
                    or last_line is not None and line_number > last_line):
                raise ValueError(f'Строка {start}: диапазон разрезает многострочную запись')
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != len(table.headers):
                raise ValueError(f'Строка {line_number}: число полей не совпадает с заголовком')
            for key, index in columns.items():
                try:
                    if key == 'time' and time_unit == 'clock':
                        clock = _clock_seconds(row[index])
                        # Accept a midnight crossing, but reject other clock resets
                        # so consecutive runs cannot silently become one trace.
                        if previous_clock is not None and previous_clock >= 82800 and clock < 3600:
                            day_offset += 86400
                        if clock_start is None:
                            clock_start = clock
                        value = clock + day_offset - clock_start
                        previous_clock = clock
                        if trace['time']:
                            if value < trace['time'][-1]:
                                raise ValueError('время убывает в пределах выбранного запуска')
                            if value == trace['time'][-1]:
                                trace['time_order'] = 'nondecreasing'
                    else:
                        value = parse_number(row[index])
                    if value is None:
                        raise ValueError('пустое значение')
                    value = float(value)
                    if key == 'time' and time_unit == 'min':
                        value *= 60
                    if not math.isfinite(value):
                        raise ValueError('неконечное число')
                except (ValueError, OverflowError) as exc:
                    raise ValueError(f'Строка {line_number}, колонка «{table.headers[index]}»: {exc}') from exc
                trace[key].append(value)
    except csv.Error as exc:
        line_number = table.header_line + reader.line_num - 1
        raise ValueError(f'Строка {line_number}: неверный формат таблицы') from exc
    run = {'format': 'autopyro.pyrogram', 'version': 1, 'label': label,
           'time_unit': 's', 'traces': [trace]}
    if temperature is not None:
        run['temperature_unit'] = '°C'
    attachment = _attachment(table.file, table.sha256, run, project, sample_id)
    attachment['table_mapping'] = {
        'columns': {key: table.headers[index] for key, index in columns.items()},
        'time_unit': time_unit, 'delimiter': table.delimiter, 'encoding': table.encoding,
        'header_line': table.header_line,
    }
    if first_line is not None or last_line is not None:
        attachment['table_mapping']['first_line'] = first_line
        attachment['table_mapping']['last_line'] = last_line
    return attachment
