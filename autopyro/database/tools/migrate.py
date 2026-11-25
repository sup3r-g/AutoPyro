"""Plan/check/apply schema migrations. Base schema and role setup are separate."""
import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re

import psycopg

from autopyro.database.connection import local_parameters


MIGRATIONS = Path(__file__).resolve().parents[3] / 'database/migrations'
# Security changes require configure_users.py; never replay them as schema changes.
EXCLUDED = {'004_readonly_access.sql'}


@dataclass(frozen=True)
class Migration:
    name: str
    checksum: str
    statement: str


def load_migrations(directory=MIGRATIONS):
    result = []
    versions = set()
    for path in sorted(directory.glob('*.sql')):
        if path.name in EXCLUDED:
            continue
        match = re.fullmatch(r'(\d{3})_[a-z0-9_]+\.sql', path.name)
        if not match or match[1] in versions:
            raise ValueError(f'Invalid or duplicate migration version: {path.name}')
        versions.add(match[1])
        text = path.read_text(encoding='utf-8-sig').replace('\r\n', '\n').replace('\r', '\n')
        # Existing scripts optionally own one transaction. The runner owns it instead.
        starts = len(re.findall(r'^BEGIN;\s*$', text, flags=re.M))
        ends = len(re.findall(r'^COMMIT;\s*$', text, flags=re.M))
        if (starts, ends) not in ((0, 0), (1, 1)):
            raise ValueError(f'Unexpected transaction wrapper: {path.name}')
        statement = re.sub(r'^(BEGIN|COMMIT);[ \t]*\n?', '', text, flags=re.M)
        result.append(Migration(path.name, hashlib.sha256(text.encode()).hexdigest(), statement))
    if not result:
        raise ValueError('No migrations found')
    return result


def migration_plan(conn, migrations):
    if conn.execute("SELECT to_regclass('geochem.source_field')").fetchone()[0] is None:
        raise ValueError('Base schema missing: apply database/schema.sql first')
    exists = conn.execute("SELECT to_regclass('geochem.schema_migration')").fetchone()[0]
    applied = dict(conn.execute('SELECT filename,sha256 FROM geochem.schema_migration')) if exists else {}
    known = {m.name for m in migrations}
    unknown = applied.keys()-known
    if unknown:
        raise ValueError('Applied migrations are missing locally: '+', '.join(sorted(unknown)))
    pending_seen = False
    for m in migrations:
        if m.name in applied:
            if applied[m.name] != m.checksum:
                raise ValueError('Checksum mismatch for applied migration: '+m.name)
            if pending_seen:
                raise ValueError('Migration history has a gap before '+m.name)
        else:
            pending_seen = True
    return [(m, m.name in applied) for m in migrations]


def apply_migrations(conn, migrations):
    """Caller owns the transaction; all pending migrations and journal rows are atomic."""
    conn.execute("SELECT pg_advisory_xact_lock(hashtextextended('geochem:schema-migrations',0))")
    plan = migration_plan(conn, migrations)
    conn.execute('''CREATE TABLE IF NOT EXISTS geochem.schema_migration (
        filename text PRIMARY KEY,
        sha256 text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
        applied_at timestamptz NOT NULL DEFAULT current_timestamp,
        applied_by text NOT NULL DEFAULT current_user
    )''')
    applied = []
    for migration, done in plan:
        if done:
            continue
        conn.execute(migration.statement)
        conn.execute('INSERT INTO geochem.schema_migration(filename,sha256) VALUES (%s,%s)',
                     (migration.name, migration.checksum))
        applied.append(migration.name)
    if conn.execute("SELECT 1 FROM pg_roles WHERE rolname='readonly'").fetchone():
        conn.execute('GRANT SELECT ON geochem.schema_migration TO readonly')
    return applied


def main():
    parser = argparse.ArgumentParser(__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true', help='Apply and record pending migrations atomically')
    mode.add_argument('--check', action='store_true', help='Exit with code 1 if any migrations are pending')
    args = parser.parse_args()
    os.environ.setdefault('PGUSER', 'admin')
    try:
        migrations = load_migrations()
        with psycopg.connect(**local_parameters()) as conn:
            conn.execute("SET LOCAL lock_timeout='15s'")
            if args.apply:
                applied = apply_migrations(conn, migrations)
            else:
                conn.execute('SET TRANSACTION READ ONLY')
                plan = migration_plan(conn, migrations)
                for migration, done in plan:
                    print(('applied ' if done else 'pending ')+migration.name)
                if args.check and any(not done for _, done in plan):
                    raise SystemExit(1)
        if args.apply:
            print('Applied: '+(', '.join(applied) if applied else 'none; schema is up to date'))
    except (ValueError, psycopg.Error) as exc:
        parser.exit(2, f'Migration failed: {exc}\n')


if __name__ == '__main__':
    main()
