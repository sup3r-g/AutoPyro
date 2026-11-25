"""Rename the existing owner to admin and configure a SELECT-only readonly login.

Uses the existing Compose credentials; stores generated reader credentials in .env.
Never prints passwords. Existing .env values are retained when updating credentials.
"""
import os
import secrets
from pathlib import Path

import psycopg
from psycopg import sql

ROOT = Path(__file__).resolve().parents[3]


def main():
    env_path = ROOT / '.env'
    lines = env_path.read_text(encoding='utf-8-sig').splitlines() if env_path.exists() else []
    env = {}
    for line in lines:
        if '=' in line and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            env[key.strip()] = value.strip().strip('\"\'')
    password = os.getenv('PGPASSWORD', os.getenv('POSTGRES_PASSWORD', env.get('POSTGRES_PASSWORD', 'autopyro_local')))
    reader_password = os.getenv('READONLY_PASSWORD', env.get('READONLY_PASSWORD')) or secrets.token_urlsafe(24)
    parameters = dict(host=os.getenv('PGHOST', '127.0.0.1'),
                      port=os.getenv('PGPORT', env.get('POSTGRES_PORT', '15432')),
                      dbname='geochemistry', password=password, connect_timeout=10)
    old_user = os.getenv('PGUSER', env.get('POSTGRES_USER', 'admin'))
    initial = psycopg.connect(**parameters, user=old_user)
    bootstrap = None
    renamed = False
    try:
        with initial:
            name, superuser = initial.execute('SELECT current_user, rolsuper FROM pg_roles WHERE rolname=current_user').fetchone()
            if not superuser:
                raise RuntimeError('Existing administrator must be a superuser')
            if name not in ('admin', 'geochem'):
                raise RuntimeError('Expected admin or original geochem owner')
            if name == 'geochem':
                if initial.execute("SELECT 1 FROM pg_roles WHERE rolname='admin'").fetchone():
                    raise RuntimeError('admin already exists; review role ownership before renaming')
                # PostgreSQL cannot rename its current session user. A short-lived
                # separate login performs the rename, preserving the original role OID.
                bootstrap = '_autopyro_setup_' + secrets.token_hex(6)
                bootstrap_password = secrets.token_urlsafe(32)
                initial.execute(sql.SQL('CREATE ROLE {} LOGIN SUPERUSER PASSWORD {}').format(
                    sql.Identifier(bootstrap), sql.Literal(bootstrap_password)))
        if bootstrap:
            with psycopg.connect(**{**parameters, 'password': bootstrap_password}, user=bootstrap) as connection:
                connection.execute('ALTER ROLE geochem RENAME TO admin')
                # Explicit reset also supports a legacy MD5 password, cleared by rename.
                connection.execute(sql.SQL('ALTER ROLE admin PASSWORD {}').format(sql.Literal(password)))
            renamed = True
        with psycopg.connect(**parameters, user='admin') as connection:
            if not connection.execute("SELECT 1 FROM pg_roles WHERE rolname='readonly'").fetchone():
                connection.execute('CREATE ROLE readonly LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS NOINHERIT')
            else:
                memberships = connection.execute('''SELECT r.rolname FROM pg_auth_members m
                    JOIN pg_roles r ON r.oid=m.roleid WHERE m.member='readonly'::regrole''').fetchall()
                if memberships:
                    raise RuntimeError('Existing readonly role has memberships; review before changing access')
                connection.execute('ALTER ROLE readonly LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS NOINHERIT')
            connection.execute(sql.SQL('ALTER ROLE readonly PASSWORD {}').format(sql.Literal(reader_password)))
            connection.execute((ROOT / 'database/migrations/004_readonly_access.sql').read_text(encoding='utf-8'))
        updates = {'POSTGRES_USER': 'admin', 'POSTGRES_PASSWORD': password,
                   'READONLY_USER': 'readonly', 'READONLY_PASSWORD': reader_password}
        output = []
        for line in lines:
            key = line.split('=', 1)[0].strip() if '=' in line and not line.lstrip().startswith('#') else None
            if key in updates:
                output.append(key + '=' + updates.pop(key))
            else:
                output.append(line)
        output.extend(key + '=' + value for key, value in updates.items())
        env_path.write_text('\n'.join(output) + '\n', encoding='utf-8')
        print('Configured admin and readonly; credentials saved to .env (passwords not printed).')
    finally:
        if bootstrap:
            # The setup role owns no objects. Connect as the renamed owner to remove it.
            with psycopg.connect(**parameters, user='admin' if renamed else old_user, autocommit=True) as connection:
                connection.execute(sql.SQL('DROP ROLE {}').format(sql.Identifier(bootstrap)))


if __name__ == '__main__':
    main()
