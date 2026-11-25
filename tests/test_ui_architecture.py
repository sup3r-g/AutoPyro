"""Import compatibility and dependency boundaries of the presentation package."""
import ast
import importlib
import subprocess
import sys
from importlib.util import resolve_name
from pathlib import Path

import pytest

from autopyro._compat import _DIALOG_EXPORTS, _MODULE_ALIASES


AUTOPYRO = Path(__file__).resolve().parents[1] / "autopyro"
UI = AUTOPYRO / "ui"
LEGACY_IMPORTS = {f"autopyro.{old}": f"autopyro.{new}" for old, new in _MODULE_ALIASES.items()}
LEGACY_IMPORTS["autopyro.dialogs"] = "autopyro.ui.dialogs"
LEGACY_IMPORTS.update({
    f"autopyro.dialogs.{name}": f"{module}.{name}"
    for module, names in _DIALOG_EXPORTS.items()
    for name in names
})


@pytest.mark.parametrize("legacy,current", [
    ("window", "ui.window"),
    ("table", "ui.table"),
    ("calculation_dialog", "ui.dialogs.calculation"),
    ("geometry_editor", "ui.dialogs.geometry_editor"),
    ("interpretation_dialog", "ui.dialogs.interpretation"),
    ("map_dialog", "ui.dialogs.map"),
    ("measurement_dialog", "ui.dialogs.measurement"),
    ("nonpyro_dialog", "ui.dialogs.nonpyro"),
    ("plot_dialog", "ui.dialogs.plot"),
    ("qc_dialog", "ui.dialogs.qc"),
    ("report_dialog", "ui.dialogs.report"),
    ("statistics_dialog", "ui.dialogs.statistics"),
    ("stratigraphy_dialog", "ui.dialogs.stratigraphy"),
    ("import_dialog", "ui.data_io.append_import"),
    ("template_dialog", "ui.data_io.templates"),
    ("database_refresh_dialog", "database.ui.refresh"),
    ("database_access", "database.access"),
    ("database_sources", "database.sources"),
    ("database_refresh", "database.refresh"),
    ("geochem_database", "database.geochem"),
    ("storage", "database.storage"),
    ("ui.database", "database.ui"),
    ("ui.database.snapshot", "database.ui.snapshot"),
    ("ui.database.refresh", "database.ui.refresh"),
    ("ui.database.nonpyro", "database.ui.nonpyro"),
])
def test_legacy_imports_share_module_globals(legacy, current, monkeypatch):
    old = importlib.import_module(f"autopyro.{legacy}")
    new = importlib.import_module(f"autopyro.{current}")
    assert old is new
    assert old.__spec__.name == f"autopyro.{current}"
    sentinel = object()
    monkeypatch.setattr(old, "_compatibility_probe", sentinel, raising=False)
    assert new._compatibility_probe is sentinel


@pytest.mark.parametrize("first", ["autopyro.window", "autopyro.ui.window"])
def test_compatibility_in_a_fresh_process(first):
    code = f"""
import importlib
import sys
from autopyro.model import core
from autopyro import core as legacy_core
assert legacy_core is core
assert not any(name.startswith('PySide6') for name in sys.modules)
assert 'autopyro.ui.window' not in sys.modules
importlib.import_module({first!r})
from autopyro import window, statistics_dialog, dialogs
from autopyro.window import MainWindow
from autopyro.dialogs import ImportDialog, QMessageBox
from autopyro.ui.window import MainWindow as CanonicalWindow
from autopyro.ui.dialogs import statistics
assert MainWindow is CanonicalWindow
assert window is sys.modules['autopyro.ui.window']
assert statistics_dialog is statistics
assert window.__spec__.name == 'autopyro.ui.window'
assert ImportDialog is dialogs.ImportDialog
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_shared_components_and_entry_point_keep_identity():
    from autopyro import __main__, dialogs, map_dialog, nonpyro_dialog, plot_dialog, statistics_dialog, window
    from autopyro.ui.application import main
    from autopyro.ui.data_io.import_file import ImportDialog
    from autopyro.ui.database.nonpyro import NonPyroDatabaseDialog
    from autopyro.ui.database.snapshot import DatabaseDialog
    from autopyro.ui.dialogs.filter import FilterDialog
    from autopyro.ui.dialogs.interpolation import InterpolationDialog
    from autopyro.ui.dialogs.spatial_interpolation import SpatialInterpolationDialog
    from autopyro.ui.tasks import TaskDialog, Worker
    from autopyro.ui.widgets.forms import Form
    from autopyro.ui.widgets.results import ResultsModel

    assert __main__.main is main
    assert dialogs.ImportDialog is ImportDialog
    assert dialogs.FilterDialog is FilterDialog
    assert dialogs.InterpolationDialog is InterpolationDialog
    assert map_dialog.SpatialInterpolationDialog is SpatialInterpolationDialog
    assert dialogs.TaskDialog is TaskDialog
    assert dialogs.Worker is Worker
    assert window.DatabaseDialog is DatabaseDialog
    assert nonpyro_dialog.NonPyroDatabaseDialog is NonPyroDatabaseDialog
    assert plot_dialog.Form is Form
    assert statistics_dialog.ResultsModel is ResultsModel


# Rules for one module also apply to its descendants if it becomes a package.
SHARED_BOUNDARIES = {
    "i18n": ("dialogs",),
    "theme": ("dialogs",),
    "widgets": ("dialogs",),
    "table.properties": ("dialogs",),
    "table.view_properties": ("dialogs",),
    "table.filter_rules": ("dialogs", "table.proxy"),
    "table.commands": ("table.model",),
    "table.snapshot_policy": ("table.model", "table.commands"),
    "widgets.parameter_defaults": ("widgets.forms", "widgets.parameters", "widgets.parameter_reset"),
    "widgets.parameter_reset": ("widgets.forms", "widgets.parameters"),
    "dialogs.spatial_interpolation": ("dialogs.map",),
    "dialogs.spatial_settings_comparison": ("dialogs.spatial_partition_comparison",),
    "dialogs.spatial_partition_comparison": ("dialogs.spatial_settings_comparison",),
}

DIALOG_FAMILIES = {
    "calculation": "calculation",
    "curve": "curve_fitting",
    "geochemistry_workspace": "geochemistry_workspace",
    "geometry_editor": "geometry_editor",
    "interpolation": "interpolation",
    "interpretation": "interpretation",
    "map": "map",
    "nonpyro": "nonpyro",
    "plot": "plot",
    "qc": "qc",
    "regional_geochemistry": "regional_geochemistry",
    "report": "report",
    "spatial": "spatial_interpolation",
    "statistics": "statistics",
    "stratigraphy": "stratigraphy",
}


def is_module_or_child(name, module):
    return name == module or name.startswith(module + ".")


def canonical_import(name):
    head, suffix = name, ""
    while head:
        if head in LEGACY_IMPORTS:
            return LEGACY_IMPORTS[head] + suffix
        head, dot, leaf = head.rpartition(".")
        suffix = dot + leaf + suffix
    return name


def module_imports(path, package_root):
    package = ".".join((package_root.name, *path.parent.relative_to(package_root).parts))
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = resolve_name("." * node.level + base, package)
            imports = [base, *(f"{base}.{alias.name}" for alias in node.names)]
        else:
            continue
        for name in imports:
            yield node.lineno, canonical_import(name)


def forbidden_ui_imports(module):
    # Only the entry point may import the main window.
    if module != "application":
        yield "window"
    for source, forbidden in SHARED_BOUNDARIES.items():
        if is_module_or_child(module, source):
            yield from forbidden
    if module.startswith("dialogs."):
        components = module.split(".")[1:]
        for family, consumer in DIALOG_FAMILIES.items():
            if any(part.startswith(family + "_") and part != consumer for part in components):
                yield "dialogs." + consumer


def ui_dependency_violations(path, package_root=AUTOPYRO):
    parts = path.relative_to(package_root / "ui").with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    forbidden = tuple("autopyro.ui." + name for name in forbidden_ui_imports(".".join(parts)))
    return dependency_violations(path, package_root, forbidden)


def dependency_violations(path, package_root, forbidden):
    return sorted({
        f"{path.relative_to(package_root.parent).as_posix()}:{line}: {name}"
        for line, name in module_imports(path, package_root)
        if any(is_module_or_child(name, boundary) for boundary in forbidden)
    })


@pytest.mark.parametrize("path", sorted(UI.rglob("*.py")), ids=lambda p: p.relative_to(UI).as_posix())
def test_ui_modules_respect_dependency_boundaries(path):
    violations = ui_dependency_violations(path)
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("subpackage", ["widgets", "theme", "i18n"])
def test_nested_package_reports_relative_and_legacy_ui_imports(tmp_path, subpackage):
    package = tmp_path / "autopyro"
    source = package / "ui" / subpackage / "nested" / "__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "from ... import window\n"
        "from ...table import model\n"
        "def open_plot():\n"
        "    from autopyro import plot_dialog\n",
        encoding="utf-8",
    )

    assert ui_dependency_violations(source, package) == [
        f"autopyro/ui/{subpackage}/nested/__init__.py:1: autopyro.ui.window",
        f"autopyro/ui/{subpackage}/nested/__init__.py:4: autopyro.ui.dialogs.plot",
    ]


def test_nested_dialog_helper_reports_its_consumer(tmp_path):
    package = tmp_path / "autopyro"
    source = package / "ui" / "dialogs" / "helpers" / "interpolation_section.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "from .. import interpolation, interpolation_inputs\n"
        "from autopyro.dialogs import InterpolationDialog\n",
        encoding="utf-8",
    )

    assert ui_dependency_violations(source, package) == [
        "autopyro/ui/dialogs/helpers/interpolation_section.py:1: autopyro.ui.dialogs.interpolation",
        "autopyro/ui/dialogs/helpers/interpolation_section.py:2: autopyro.ui.dialogs.interpolation.InterpolationDialog",
    ]


def test_domain_and_persistence_have_no_presentation_imports():
    groups = (
        "model", "data_io", "analysis", "curves", "schemes", "visualization",
        "spatial", "geology", "reports", "database",
    )
    presentation = ("PySide6", "autopyro.ui", "ui", "autopyro.database.ui")
    violations = []
    for group in groups:
        modules = sorted((AUTOPYRO / group).rglob("*.py"))
        assert modules, group
        for module in modules:
            if not module.is_relative_to(AUTOPYRO / "database" / "ui"):
                violations.extend(dependency_violations(module, AUTOPYRO, presentation))
    assert not violations, "\n".join(violations)
