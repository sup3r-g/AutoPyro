"""Language selection, translated UI flows and preservation of project data."""
import ast
import re
from copy import deepcopy
from pathlib import Path
from string import Formatter
import xml.etree.ElementTree as ET

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QAbstractButton, QComboBox, QDialog, QMessageBox

from autopyro.io import build_project
from autopyro.ui import i18n
from autopyro.ui.window import MainWindow

ENGLISH = {
    entry.findtext("source"): entry.findtext("translation")
    for context in ET.parse(Path(i18n.__file__).with_name("translations") / "autopyro_en.ts").findall("context")
    if context.findtext("name") == i18n.CONTEXT
    for entry in context.findall("message")
}


@pytest.fixture
def language_store(tmp_path, monkeypatch, qapp):
    store = QSettings(str(tmp_path / "language.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(i18n, "settings", lambda: store)
    i18n.initialize()
    yield store
    i18n.save_language("ru", store)
    i18n.initialize(store)


@pytest.fixture
def english_window(language_store, qtbot):
    i18n.save_language("en")
    i18n.initialize()
    project = build_project(
        ["Сохранить", "Depth", "TOC", "S1", "S2", "S3"],
        [["Новый проект", "10", "2", ".5", "6", "1.2"],
         ["Проба Б", "20", "1", ".1", "1", ".3"]],
        [2, 3],
        [(0, "Сохранить", "text", ""), (1, "Depth", "number", "m"),
         (2, "TOC", "number", "%wt"), (3, "S1", "number", "mg HC/g rock"),
         (4, "S2", "number", "mg HC/g rock"), (5, "S3", "number", "mg CO2/g rock")],
        {"type": "test", "path": "Русский источник.csv"},
    )
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.may_replace = lambda: True
    yield window
    window.dirty = False


def test_language_selection_is_saved_without_modifying_open_project(english_window, language_store, monkeypatch):
    window = english_window
    project = deepcopy(window.model.project)
    notices = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args: notices.append(args[2]))
    assert "File" in [action.text() for action in window.menuBar().actions()]
    assert "Save" in window.actions
    assert "Rows: 2" in window.statusBar().currentMessage()
    from autopyro.ui.dialogs.settings import SettingsDialog
    dialog = SettingsDialog(window)
    assert dialog.language.currentData() == "en"
    dialog.language.setCurrentIndex(dialog.language.findData("ru"))
    dialog.accept()
    monkeypatch.setattr(SettingsDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    window.show_settings()
    assert i18n.preferred_language() == "ru"
    assert i18n.language() == "en"
    assert "Restart AutoPyro" in notices[0]
    assert window.model.project == project
    assert not window.dirty
    assert window.model.headerData(0, Qt.Orientation.Horizontal) == "Сохранить"
    assert window.model.index(0, 0).data() == "Новый проект"
    # A new settings instance reads the preference for the next application run.
    fresh = QSettings(language_store.fileName(), QSettings.Format.IniFormat)
    i18n.initialize(fresh)
    assert i18n.tr("Сохранить") == "Сохранить"


def test_invalid_preference_defaults_to_russian(language_store):
    language_store.setValue(i18n.SETTINGS_KEY, "unsupported")
    i18n.initialize()
    assert i18n.language() == "ru"
    assert i18n.tr("Сохранить") == "Сохранить"
    with pytest.raises(ValueError):
        i18n.save_language("fr")


def test_formatted_translations_keep_user_values(english_window):
    assert i18n.tr("Совместная CV: блок {0} из {1}", 2, 10) == "Joint CV: fold 2 of 10"
    assert i18n.message("Модель вернула NaN/Inf; проверьте область определения и начальные параметры") == (
        "The model returned NaN/Inf. Check its domain and initial parameters."
    )
    assert i18n.tr("Источник: {0}", "Сохранить.csv") == "Source: Сохранить.csv"
    assert i18n.tr("Строка {0}, {1}: {2!r}\n{3}\n", 5, "Колонка", "1,2", "details") == (
        "Row 5, Колонка: '1,2'\ndetails\n"
    )
    assert i18n.message(ValueError("Вход S2: неподдерживаемая единица")) == "Input S2: unsupported unit"
    assert i18n.message(ValueError("Таблица отсутствует в источнике")) == "Table is missing from source"
    assert i18n.message("native error: Русское имя.csv") == "native error: Русское имя.csv"


@pytest.mark.parametrize("feature", [
    "filter", "interpolation", "calculation", "geometry_editor", "interpretation",
    "map", "nonpyro", "report", "statistics", "stratigraphy", "qc", "measurement",
    "database", "database_profile", "database_refresh", "append_import", "templates", "plot",
])
def test_english_dialogs_use_translated_controls(english_window, qtbot, monkeypatch, feature):
    from autopyro.ui.database.nonpyro import NonPyroDatabaseDialog
    from autopyro.ui.database.refresh import DatabaseRefreshDialog
    from autopyro.ui.database.snapshot import DatabaseDialog
    from autopyro.ui.data_io.append_import import AppendImportDialog
    from autopyro.ui.data_io.templates import TemplateLibraryDialog
    from autopyro.ui.dialogs.calculation import CalculationDialog
    from autopyro.ui.dialogs.filter import FilterDialog
    from autopyro.ui.dialogs.geometry_editor import NewSchemeDialog
    from autopyro.ui.dialogs.interpolation import InterpolationDialog
    from autopyro.ui.dialogs.interpretation import InterpretationDialog
    from autopyro.ui.dialogs.map import MapDialog
    from autopyro.ui.dialogs.measurement import MeasurementDialog
    from autopyro.ui.dialogs.nonpyro import NonPyroDialog
    from autopyro.ui.dialogs.plot import PlotDialog
    from autopyro.ui.dialogs.qc import QCDialog
    from autopyro.ui.dialogs.report import ReportDialog
    from autopyro.ui.dialogs.statistics import StatisticsDialog
    from autopyro.ui.dialogs.stratigraphy import StratigraphyDialog

    window = english_window
    project = window.model.project
    factories = {
        "filter": lambda: FilterDialog(project, [], "AND", window),
        "interpolation": lambda: InterpolationDialog(project, set(project.row_map()), window),
        "calculation": lambda: CalculationDialog(window),
        "geometry_editor": lambda: NewSchemeDialog(window),
        "interpretation": lambda: InterpretationDialog(window),
        "map": lambda: MapDialog(window),
        "nonpyro": lambda: NonPyroDialog(window),
        "report": lambda: ReportDialog(window),
        "statistics": lambda: StatisticsDialog(window),
        "stratigraphy": lambda: StratigraphyDialog(window),
        "qc": lambda: QCDialog(window),
        "measurement": lambda: MeasurementDialog(window.model, project.rows[0].id, project.columns[2].id, window),
        "database": lambda: DatabaseDialog(window),
        "database_profile": lambda: NonPyroDatabaseDialog("atomic_ratios", window),
        "database_refresh": lambda: DatabaseRefreshDialog(window),
        "append_import": lambda: AppendImportDialog(window, deepcopy(project)),
        "templates": lambda: TemplateLibraryDialog(window),
        "plot": lambda: PlotDialog(window),
    }
    if feature == "database_refresh":
        project.source = {"type": "sqlite", "path": "test.sqlite", "table": "data", "source_columns": [c.name for c in project.columns]}
    dialog = factories[feature]()
    qtbot.addWidget(dialog)
    assert dialog.help_button.text() == "What's This?"
    assert dialog.whats_this_action.shortcut().toString() == "Shift+F1"
    if feature not in ("measurement", "templates"):
        assert dialog.reset_defaults_button is not None
        dialog.reset_defaults_button.click()
    assert not re.search("[А-Яа-яЁё]", dialog.windowTitle()), dialog.windowTitle()
    for button in dialog.findChildren(QAbstractButton):
        assert not re.search("[А-Яа-яЁё]", button.text()), button.text()
    if feature == "report":
        assert dialog.preview_sections.accessibleName() == "Jump to a report section"
        assert dialog.preview.accessibleName() == "Report preview"
        assert dialog.preview_stale.text() == "Out of date. Rebuild the report before exporting."
        dialog.apply_builtin_template("Паспорт скважины")
        assert dialog.title.text() == "Well report"

        def inspect_repair(popup):
            assert popup.windowTitle() == "Repair template references"
            combo = popup.findChild(QComboBox)
            assert combo.itemText(0) == "Choose a replacement…"
            assert combo.itemText(1) == "Remove from the template"
            return QDialog.DialogCode.Rejected

        monkeypatch.setattr(QDialog, "exec", inspect_repair)
        options, _ = dialog.composition()
        assert dialog.repair_template(options, [], [("column", "missing", "Missing column", {})]) is None
    if hasattr(dialog, "timer"):
        dialog.timer.stop()
    if hasattr(dialog, "closed"):
        dialog.closed = True


def test_english_import_preserves_types_and_original_cells(english_window, tmp_path, qtbot):
    from autopyro.ui.data_io.import_file import ImportDialog

    source = tmp_path / "данные.csv"
    source.write_text("Сохранить;S2\n001;<0.1\nПроба;3.2\n", encoding="utf-8-sig")
    dialog = ImportDialog(path=source)
    qtbot.addWidget(dialog)
    dialog.load()
    qtbot.waitUntil(lambda: dialog.raw is not None and not dialog.worker.isRunning(), timeout=10000)
    kind = dialog.mapping.cellWidget(1, 3)
    assert kind.itemText(0) == "Text"
    assert kind.itemData(0) == "text"
    assert kind.itemText(1) == "Number"
    assert kind.itemData(1) == "number"
    dialog.import_data()
    qtbot.waitUntil(lambda: dialog.result() == QDialog.DialogCode.Accepted, timeout=10000)
    assert dialog.project.columns[0].name == "Сохранить"
    assert dialog.project.rows[0].values[dialog.project.columns[0].id] == "001"
    assert dialog.project.rows[0].values[dialog.project.columns[1].id] == "<0.1"


def test_calculation_and_undo_work_in_english(english_window, qtbot):
    from autopyro.ui.dialogs.calculation import CalculationDialog

    window = english_window
    dialog = CalculationDialog(window)
    qtbot.addWidget(dialog)
    dialog.context.setText("Pyrol6 whole rock, test series")
    dialog.confirmed.setChecked(True)
    dialog.calculate()
    assert dialog.prepared
    original_count = window.model.columnCount()
    dialog.apply()
    hi = next(column for column in window.model.project.columns if column.name == "HI")
    assert window.model.project.rows[0].values[hi.id] == 300
    window.model.undo.undo()
    assert window.model.columnCount() == original_count
    window.model.undo.redo()
    assert window.model.project.rows[0].values[hi.id] == 300


def test_translated_filter_keeps_operator_ids_and_user_names(english_window, qtbot):
    from autopyro.ui.dialogs.filter import FilterDialog
    from autopyro.ui.widgets.results import ResultsModel

    window = english_window
    dialog = FilterDialog(window.model.project, [], "AND", window)
    qtbot.addWidget(dialog)
    dialog.add_rule()
    operator = dialog.table.cellWidget(0, 1)
    assert operator.itemText(0) == "Contains"
    assert operator.itemData(0) == "contains"
    column = dialog.table.cellWidget(0, 0)
    assert column.itemText(0) == "Сохранить"
    assert column.itemData(0) == window.model.project.columns[0].id
    results = ResultsModel(["Группа: Сохранить", "Статус"], [["Проба", "constant"]])
    assert results.headerData(0, Qt.Orientation.Horizontal) == "Group: Сохранить"
    assert results.headerData(10, Qt.Orientation.Vertical) == "11"
    assert results.index(0, 1).data() == "Constant property"


def test_geochemistry_passports_are_translated(english_window):
    window = english_window
    window.show_geochemistry_workspace()
    document = window.geochemistry_workspace_window
    assert document.windowTitle() == 'Geochemical passports and comparison'
    assert document.passport_tabs.tabText(0) == 'Sample passport'
    assert document.compare_button.text() == 'Compare'
    assert document.export_passport_button.text() == 'Export passport JSON…'


def test_batch_edit_preview_is_translated_and_keeps_user_values(english_window, monkeypatch):
    from autopyro.ui.dialogs.batch_edit import BatchEditPreview

    def inspect(dialog):
        model = dialog.model
        assert dialog.windowTitle() == "Paste"
        assert [model.headerData(i, Qt.Orientation.Horizontal)
                for i in range(model.columnCount())] == ["Sample", "Project row", "Column", "Before", "After"]
        assert model.index(0, 1).data() == "1"
        assert model.index(0, 2).data() == "Сохранить"
        assert model.index(0, 3).data() == "Новый проект"
        assert model.index(0, 4).data() == "Новая проба"
        assert {button.text() for button in dialog.findChildren(QAbstractButton)} >= {"Apply", "Cancel"}
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(BatchEditPreview, "exec", inspect)
    window = english_window
    index = window.model.index(0, 0)
    window.preview_batch_edit(i18n.tr("Вставка"), [(index, "Новая проба")])
    assert index.data() == "Новый проект"


def test_catalog_covers_literal_ui_templates_and_keeps_placeholders():
    ui = Path(__file__).resolve().parents[1] / "autopyro" / "ui"
    paths = [*ui.rglob("*.py"), *(ui.parent / "database" / "ui").rglob("*.py")]
    for path in paths:
        if "i18n" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "tr"
                    and node.args and isinstance(node.args[0], ast.Constant)):
                source = node.args[0].value
                if isinstance(source, str) and re.search("[А-Яа-яЁё]", source):
                    assert source in ENGLISH, (path, source)
    fields = lambda text: sorted((key, spec, conversion) for _, key, spec, conversion in Formatter().parse(text) if key is not None)
    for source, target in ENGLISH.items():
        assert fields(source) == fields(target), source



def test_stratigraphy_structured_controls_are_english(english_window, qtbot):
    from autopyro.ui.dialogs.stratigraphy import StratigraphyDialog
    from autopyro.ui.widgets.stratigraphy_labels import geological_label

    dialog = StratigraphyDialog(english_window)
    qtbot.addWidget(dialog)
    assert [dialog.parameter_tabs.tabText(i) for i in range(3)] == [
        '1. Intervals', '2. Column', '3. Measurements']
    form = dialog.column_style_form
    assert form.fields['display_mode'][0].itemText(0) == 'Lithology column'
    assert form.fields['feature_mode'][0].itemText(0) == 'Hide'
    assert form.fields['grain_preset'][0].itemText(0) == 'Clastic rocks'
    assert dialog.level.currentText() == 'Formation'
    assert geological_label('quartz') == 'Quartz'
    assert geological_label('cross bedding') == 'Cross Bedding'
    assert form.title_labels[1] == 'Minerals'
    assert form.details.tabText(1) == 'Ticks and groups'
    assert form.hierarchy_group.title() == 'Additional levels beside the column'
    assert [check.text() for check in form.hierarchy_checks.values()] == [
        'Group', 'Subgroup', 'Formation', 'Member']


def test_hierarchy_preview_and_export_have_english_titles(english_window, qtbot, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from autopyro.ui.dialogs.stratigraphy import StratigraphyDialog

    dialog = StratigraphyDialog(english_window)
    qtbot.addWidget(dialog)
    dialog.reference.setText('MD RKB')
    for rank in ('GROUP', 'FORMATION'):
        dialog.add_interval(interval={'id': rank, 'name': f'Unit {rank}', 'wellbore': 'A',
                                     'level': rank, 'top': 0, 'bottom': 30, 'color': '#aabbcc'})
    dialog.column_style_form.hierarchy_checks['GROUP'].setChecked(True)
    dialog.column_style_form.hierarchy_checks['MEMBER'].setChecked(True)
    dialog.show_column()
    assert [axis.get_title() for axis in dialog.column_figure.axes] == ['Group', 'Formation']
    assert 'No intervals for level: Member' in dialog.status.text()
    target = tmp_path / 'hierarchy.svg'
    monkeypatch.setattr(QFileDialog, 'exec', lambda self: self.DialogCode.Accepted)
    monkeypatch.setattr(QFileDialog, 'selectedFiles', lambda self: [str(target)])
    dialog.export_column()
    assert 'Column exported' in dialog.status.text()
    assert '<!-- Formation -->' in target.read_text(encoding='utf-8')


def test_comparison_controls_warnings_and_export_are_english(english_window, qtbot, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    from autopyro.ui.dialogs.stratigraphy import StratigraphyDialog

    dialog = StratigraphyDialog(english_window)
    qtbot.addWidget(dialog)
    dialog.reference.setText('MD RKB')
    for well, depth in [('Скважина А', 0), ('Скважина Б', 50)]:
        dialog.add_interval(interval={'id': well, 'name': 'Formation name', 'wellbore': well,
                                     'level': 'FORMATION', 'top': depth, 'bottom': depth + 30,
                                     'color': '#aabbcc'})
    form = dialog.column_style_form
    form.load({'comparison_wells': ['Скважина Б'], 'hierarchy_levels': ['GROUP']})
    assert form.comparison_group.title() == 'Compare wells and wellbores'
    assert form.comparison_button.text() == 'Choose wellbores to compare…'
    assert form.comparison_summary.text() == 'Additional wellbores: Скважина Б'
    dialog.show_column()
    assert [axis._autopyro_well for axis in dialog.column_figure.axes] == ['Скважина А', 'Скважина Б']
    assert 'Wellbore Скважина Б: no intervals for level Group' in dialog.status.text()
    target = tmp_path / 'comparison.svg'
    monkeypatch.setattr(QFileDialog, 'exec', lambda self: self.DialogCode.Accepted)
    monkeypatch.setattr(QFileDialog, 'selectedFiles', lambda self: [str(target)])
    dialog.export_column()
    content = target.read_text(encoding='utf-8')
    assert '<!-- Скважина А -->' in content and '<!-- Скважина Б -->' in content
    assert '<!-- Formation -->' in content


def test_column_appearance_templates_are_localized_and_keep_user_names(english_window, qtbot, monkeypatch):
    from PySide6.QtWidgets import QInputDialog
    from autopyro.stratigraphy import COLUMN_STYLE
    from autopyro.ui.dialogs.stratigraphy import StratigraphyDialog
    from autopyro.ui.widgets.stratigraphy_templates import TEMPLATES_KEY

    dialog = StratigraphyDialog(english_window)
    qtbot.addWidget(dialog)
    presets = dialog.style_templates
    assert presets.title() == 'Appearance templates'
    assert presets.apply_button.text() == 'Apply'
    assert presets.save_button.text() == 'Save as…'
    assert presets.delete_button.text() == 'Delete'
    monkeypatch.setattr(QInputDialog, 'getText', lambda *args, **kwargs: ('Для печати', True))
    presets.save_button.click()
    assert presets.combo.currentText() == 'Для печати'
    stored = english_window.model.project.view[TEMPLATES_KEY]['Для печати']
    assert stored['y_label'] == COLUMN_STYLE['y_label']
    assert stored['legend_titles'] == COLUMN_STYLE['legend_titles']
    presets.apply_button.click()
    assert 'applied to the current appearance' in presets.feedback.text()
    assert dialog.column_style_form.fields['y_label'][0].text() == 'Depth'


def test_lithology_mapping_controls_are_english_and_keep_imported_names(english_window, qtbot):
    from autopyro.ui.dialogs.stratigraphy import StratigraphyDialog
    from autopyro.ui.widgets.lithology_mapping import LithologyMappingDialog

    owner = StratigraphyDialog(english_window)
    qtbot.addWidget(owner)
    catalog, matcher = owner.lithology_matcher()
    scheme = {'intervals': [{'lithology': 'Местная порода'}, {'lithology': 'Глина'}]}
    dialog = LithologyMappingDialog(scheme, catalog, matcher, owner)
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == 'Match rock names'
    assert dialog.source.currentText() == 'Current interval lithology'
    assert dialog.choose_button.text() == 'Choose lithology for this name…'
    assert dialog.model.index(0, 0).data() == 'Местная порода'
    assert dialog.model.index(0, 3).data() == 'Unrecognized'
    assert dialog.model.index(1, 3).data() == 'Known rock name'
