"""Lazy compatibility imports for relocated AutoPyro modules."""

import sys
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader


_MODULE_ALIASES = {
    "audit": "model.audit",
    "calculations": "analysis.calculations",
    "core": "model.core",
    "curve_archive": "curves.curve_archive",
    "curve_bootstrap": "curves.curve_bootstrap",
    "curve_constraints": "curves.curve_constraints",
    "curve_fitting": "curves.curve_fitting",
    "curve_interpretation": "curves.curve_interpretation",
    "curve_intervals": "curves.curve_intervals",
    "curve_numerics": "curves.curve_numerics",
    "curve_outputs": "curves.curve_outputs",
    "curve_search": "curves.curve_search",
    "curve_validation": "curves.curve_validation",
    "file_output": "data_io.file_output",
    "geochemistry_workspace": "analysis.geochemistry_workspace",
    "import_templates": "data_io.import_templates",
    "importing": "data_io.importing",
    "interpolation": "analysis.interpolation",
    "interpolation_constraints": "analysis.interpolation_constraints",
    "interpretation": "schemes.interpretation",
    "interpretation_comparison": "schemes.interpretation_comparison",
    "interpretation_explanation": "schemes.interpretation_explanation",
    "interpretation_labels": "schemes.interpretation_labels",
    "interpretation_quality": "schemes.interpretation_quality",
    "interpretation_thresholds": "schemes.interpretation_thresholds",
    "io": "data_io.io",
    "library_plot": "visualization.library_plot",
    "library_scatter": "visualization.library_scatter",
    "lithology_matching": "geology.lithology_matching",
    "map_tools": "spatial.map_tools",
    "mapping": "spatial.mapping",
    "measurement_uncertainty": "schemes.measurement_uncertainty",
    "measurements": "model.measurements",
    "model_comparison": "curves.model_comparison",
    "nonpyro": "analysis.nonpyro",
    "plot_catalog": "visualization.plot_catalog",
    "plot_controls": "visualization.plot_controls",
    "plot_data": "visualization.plot_data",
    "plot_export": "visualization.plot_export",
    "plot_hover": "visualization.plot_hover",
    "plot_preview": "visualization.plot_preview",
    "plotting": "visualization.plotting",
    "protocol_json": "model.protocol_json",
    "qc": "analysis.qc",
    "regional_comparison": "spatial.regional_comparison",
    "regional_geochemistry": "spatial.regional_geochemistry",
    "regional_report": "reports.regional_report",
    "reporting": "reports.reporting",
    "scheme_geometry": "schemes.scheme_geometry",
    "selection_overlay": "visualization.selection_overlay",
    "snapshots": "model.snapshots",
    "spatial_interpolation": "spatial.spatial_interpolation",
    "standard_interpretation": "schemes.standard_interpretation",
    "statistics": "analysis.statistics",
    "stratigraphy": "geology.stratigraphy",
    "stratigraphy_review": "geology.stratigraphy_review",
    "stratigraphy_revisions": "geology.stratigraphy_revisions",
    "stratigraphy_samples": "geology.stratigraphy_samples",
    "stratigraphy_summary": "geology.stratigraphy_summary",
    "tabular": "model.tabular",
    "uncertainty_plot": "visualization.uncertainty_plot",
    "window": "ui.window",
    "table": "ui.table",
    "calculation_dialog": "ui.dialogs.calculation",
    "geometry_editor": "ui.dialogs.geometry_editor",
    "interpretation_dialog": "ui.dialogs.interpretation",
    "map_dialog": "ui.dialogs.map",
    "measurement_dialog": "ui.dialogs.measurement",
    "nonpyro_dialog": "ui.dialogs.nonpyro",
    "plot_dialog": "ui.dialogs.plot",
    "qc_dialog": "ui.dialogs.qc",
    "report_dialog": "ui.dialogs.report",
    "statistics_dialog": "ui.dialogs.statistics",
    "stratigraphy_dialog": "ui.dialogs.stratigraphy",
    "import_dialog": "ui.data_io.append_import",
    "template_dialog": "ui.data_io.templates",
    "database_refresh_dialog": "database.ui.refresh",
    "database_access": "database.access",
    "database_sources": "database.sources",
    "database_refresh": "database.refresh",
    "geochem_database": "database.geochem",
    "storage": "database.storage",
    "ui.database": "database.ui",
    "ui.database.snapshot": "database.ui.snapshot",
    "ui.database.refresh": "database.ui.refresh",
    "ui.database.nonpyro": "database.ui.nonpyro",
}

_DIALOG_EXPORTS = {
    "autopyro.ui.tasks": ("TaskDialog", "Worker"),
    "autopyro.ui.data_io.import_file": ("ImportDialog",),
    "autopyro.ui.dialogs.filter": ("FilterDialog", "OPERATORS"),
    "autopyro.ui.dialogs.interpolation": ("InterpolationDialog",),
    "PySide6.QtWidgets": ("QMessageBox",),
}


class _CompatibilityImports(MetaPathFinder, Loader):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("autopyro."):
            return None
        name = fullname.removeprefix("autopyro.")
        if name in _MODULE_ALIASES or name == "dialogs":
            return spec_from_loader(fullname, self, is_package=name == "ui.database")
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        name = module.__name__.removeprefix("autopyro.")
        if name == "dialogs":
            module.__doc__ = "Compatibility exports for dialogs in autopyro.ui."
            module.__all__ = []
            for source, names in _DIALOG_EXPORTS.items():
                imported = import_module(source)
                for export in names:
                    setattr(module, export, getattr(imported, export))
                    module.__all__.append(export)
        else:
            # Replace the temporary module after loading the canonical one, so
            # its name, spec and globals remain shared by both import paths.
            sys.modules[module.__name__] = import_module(
                f"autopyro.{_MODULE_ALIASES[name]}"
            )


def install():
    """Register legacy names without eagerly importing Qt or database drivers."""
    if not any(isinstance(finder, _CompatibilityImports) for finder in sys.meta_path):
        sys.meta_path.insert(0, _CompatibilityImports())
