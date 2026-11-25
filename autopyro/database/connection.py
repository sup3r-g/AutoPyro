"""PostgreSQL connection defaults and read-only connections."""

import os
from pathlib import Path

import psycopg
from psycopg.rows import dict_row


def local_parameters():
    """Match the repository's Compose dev connection; never persist credentials."""
    env = {}
    path = Path(".env")
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip().strip("\"'")
    user = os.getenv("PGUSER", env.get("READONLY_USER", "readonly"))
    if user == env.get("READONLY_USER", "readonly"):
        password = os.getenv("PGPASSWORD", os.getenv("READONLY_PASSWORD", env.get("READONLY_PASSWORD", "")))
    else:
        password = os.getenv(
            "PGPASSWORD",
            os.getenv("POSTGRES_PASSWORD", env.get("POSTGRES_PASSWORD", "autopyro_local")),
        )
    return {
        "host": os.getenv("PGHOST", "127.0.0.1"),
        "port": os.getenv("PGPORT", env.get("POSTGRES_PORT", "15432")),
        "dbname": os.getenv("PGDATABASE", "geochemistry"),
        "user": user,
        "password": password,
    }



def connect(parameters):
    connection = psycopg.connect(**parameters, connect_timeout=8, row_factory=dict_row)
    try:
        connection.read_only = True
        connection.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
        connection.execute("SET LOCAL statement_timeout='60s'")
        from .access import verify_read_only
        verify_read_only(connection)
    except BaseException:
        connection.close()
        raise
    return connection

