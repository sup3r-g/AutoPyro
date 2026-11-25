"""Atomic replacement for exported figures and small JSON documents."""
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import tempfile


@contextmanager
def atomic_output(path, *, before_replace=None, overwrite=True):
    path = Path(path)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.' + path.stem + '-', suffix=path.suffix)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        with temporary.open('rb+') as stream:
            os.fsync(stream.fileno())
        if before_replace:
            before_replace()
        if overwrite:
            os.replace(temporary, path)
        else:
            publish_without_overwrite(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_without_overwrite(staging, path):
    """Publish a complete file atomically, picking a new name on collision."""
    path = Path(path)
    candidate = path
    suffix = 2
    while True:
        try:
            if os.name == 'nt':
                os.rename(staging, candidate)
            else:
                os.link(staging, candidate)
            return candidate
        except FileExistsError:
            candidate = path.with_name(f'{path.stem}-{suffix}{path.suffix}')
            suffix += 1


def write_json(path, payload, *, overwrite=True):
    # Serialization errors must leave both the target and its directory untouched.
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    with atomic_output(path, overwrite=overwrite) as temporary:
        temporary.write_text(text, encoding='utf-8')


def write_excel_csv(path, headers, rows, decimal):
    def cell(value):
        if isinstance(value, (int, float)):
            return format(value, '.15g').replace('.', decimal)
        if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')):
            return "'" + value
        return value

    with atomic_output(path) as temporary, temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream, delimiter=';' if decimal == ',' else ',', lineterminator='\r\n')
        writer.writerow(headers)
        writer.writerows([cell(value) for value in row] for row in rows)
