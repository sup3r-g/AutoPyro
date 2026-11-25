"""Database code stays centralized and independent of its Qt interface."""
import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_database_drivers_are_only_imported_inside_database_package():
    package = ROOT / "autopyro"
    for path in package.rglob("*.py"):
        if "database" in path.relative_to(package).parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(name.split(".")[0] in {"psycopg", "sqlite3"} for name in names), path


@pytest.mark.parametrize("first", ["autopyro.database.readers", "autopyro.io"])
def test_database_backend_imports_without_qt_or_import_cycles(first):
    code = f"""
import importlib
import sys
import autopyro.database
assert 'psycopg' not in sys.modules
importlib.import_module({first!r})
from autopyro.database import access, connection, sources, refresh, geochem, storage, preferences
from autopyro.io import read_sqlite
from autopyro.database.readers import read_sqlite as canonical
assert read_sqlite is canonical
assert not any(name.startswith('PySide6') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_development_tools_keep_existing_asset_paths():
    for name in ("generate_schema", "import_parquet", "import_lithostratigraphy", "configure_users", "validate_users"):
        if name == "import_parquet" and importlib.util.find_spec("pyarrow") is None:
            continue  # Optional dependency of the administrative Parquet tool.
        module = importlib.import_module(f"autopyro.database.tools.{name}")
        expected = ROOT if name in ("configure_users", "validate_users") else ROOT / "database"
        assert module.ROOT == expected
        assert not (ROOT / "database" / f"{name}.py").exists()


def test_database_ui_legacy_submodules_keep_identity_in_fresh_process():
    code = """
import importlib
import autopyro.ui.database
for name in ('snapshot', 'refresh', 'nonpyro'):
    old = importlib.import_module('autopyro.ui.database.' + name)
    current = importlib.import_module('autopyro.database.ui.' + name)
    assert old is current
    assert old.__spec__.name == 'autopyro.database.ui.' + name
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
