"""Inventory real UI controls in both languages and Breeze palettes."""
import ast
from pathlib import Path
import pytest
from PySide6.QtCore import QSettings, QTimer, QSize
from PySide6.QtGui import QAction, QIcon, QPalette
from PySide6.QtWidgets import QPushButton, QToolButton, QTabWidget, QMenu
from autopyro.ui import i18n
from autopyro.io import build_project
from autopyro.ui.window import MainWindow
from autopyro.ui.theme import ASSETS, ICON_ALIASES, apply_theme, icon
from autopyro.ui.theme.controls import STANDARD_BUTTON_ICONS, apply_control_icons
from autopyro.ui.workspace import DOCUMENT_ICONS

def build_interfaces():
    owner = MainWindow()
    project = build_project(['Well', 'Depth', 'X', 'Y', 'TOC', 'S1', 'S2', 'S3'],
        [['A', 10, 1, 3, 2, .5, 6, 1.2], ['A', 20, 2, 4, 1, .1, 1, .3]], [2, 3],
        [(i, s, 'text' if i == 0 else 'number', '') for i, s in enumerate(['Well', 'Depth', 'X', 'Y', 'TOC', 'S1', 'S2', 'S3'])],
        {'type': 'sqlite', 'path': 'icons.db', 'table': 'samples'})
    owner.set_project(project)
    owner.may_replace = lambda: True
    for method in ('show_plots', 'show_maps', 'show_statistics', 'show_curve_fitting', 'show_interpretation',
                   'interpolate', 'show_calculations', 'show_stratigraphy', 'show_nonpyro', 'show_qc'):
        getattr(owner, method)()
    from autopyro.ui.geochemistry import modules
    for module in modules(): owner.show_nonpyro(module)
    from autopyro.ui.dialogs.settings import SettingsDialog
    from autopyro.ui.dialogs.report import ReportDialog
    from autopyro.ui.dialogs.filter import FilterDialog
    from autopyro.ui.dialogs.geometry_editor import GeometryEditor, NewSchemeDialog, new_scheme
    from autopyro.ui.dialogs.measurement import MeasurementDialog, NewMeasurementDialog
    from autopyro.ui.data_io.import_file import ImportDialog
    from autopyro.ui.data_io.append_import import AppendImportDialog
    from autopyro.ui.data_io.templates import TemplateLibraryDialog
    from autopyro.database.ui.nonpyro import NonPyroDatabaseDialog
    from autopyro.database.ui.refresh import DatabaseRefreshDialog
    from autopyro.database.ui.snapshot import DatabaseDialog
    from autopyro.ui.widgets.lithology import LithologyPicker, LithologyCatalog
    from autopyro.ui.widgets.spatial import CRSDialog, CRSCombo
    from autopyro.ui.widgets.stratigraphy_labels import CheckedValuesDialog
    from autopyro.ui.dialogs.map import SpatialInterpolationDialog
    from autopyro.mapping import MapSpec, build_map
    combo = CRSCombo(owner)
    combo.addItem('WGS 84', 'EPSG:4326')
    combo.hide()
    report = build_map(project, [r.id for r in project.rows], MapSpec(
        x_id=project.columns[2].id, y_id=project.columns[3].id,
        source_crs='EPSG:32631', target_crs='EPSG:32631'), [])
    extra = [ReportDialog(owner), SettingsDialog(owner), FilterDialog(project, [], 'AND', owner), GeometryEditor(new_scheme(), owner),
        NewSchemeDialog(owner), MeasurementDialog(owner.model, project.rows[0].id, project.columns[4].id, owner),
        NewMeasurementDialog(owner.model, project.rows[0].id, project.columns[4].id, owner),
        ImportDialog(path='icons.csv', parent=owner), AppendImportDialog(owner, project), TemplateLibraryDialog(owner),
        NonPyroDatabaseDialog(parent=owner), DatabaseRefreshDialog(owner), DatabaseDialog(owner),
        LithologyPicker(LithologyCatalog(), parent=owner), CRSDialog(combo),
        CheckedValuesDialog('Types', [('a', 'A')], [], owner), SpatialInterpolationDialog(report, owner)]
    for timer in owner.findChildren(QTimer): timer.stop()
    return owner, extra

def inventory(owner):
    missing = {'buttons': [], 'tabs': [], 'menus': []}
    for b in [*owner.findChildren(QPushButton), *owner.findChildren(QToolButton)]:
        if b.icon().isNull() and (b.text() or b.toolTip()):
            missing['buttons'].append([type(b.parent()).__name__, b.text(), b.toolTip()])
    for tabs in owner.findChildren(QTabWidget):
        for i in range(tabs.count()):
            if tabs.tabIcon(i).isNull(): missing['tabs'].append([type(tabs.parent()).__name__, tabs.tabText(i)])
    for menu in owner.findChildren(QMenu):
        for a in menu.actions():
            if not a.isSeparator() and a.icon().isNull(): missing['menus'].append([type(menu.parent()).__name__, a.text()])
    return {k: sorted(set(map(tuple,v))) for k,v in missing.items()}


@pytest.mark.parametrize('language', ['ru', 'en'])
@pytest.mark.parametrize('theme', ['breeze_light', 'breeze_dark'])
def test_interpretation_commands_have_icons(qtbot, qapp, monkeypatch, tmp_path, language, theme):
    store = QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)
    monkeypatch.setattr(i18n, 'settings', lambda: store)
    original_language = i18n.language()
    original_theme = qapp.property('autopyroTheme')
    i18n.save_language(language)
    i18n.initialize()
    apply_theme(qapp, theme)
    window = MainWindow()
    qtbot.addWidget(window)
    try:
        window.show_interpretation()
        dialog = window.interpretation_window
        assert inventory(dialog) == {'buttons': [], 'tabs': [], 'menus': []}
    finally:
        dialog.close()
        window.close()
        i18n.save_language(original_language)
        i18n.initialize()
        apply_theme(qapp, original_theme or 'breeze_light')


@pytest.mark.parametrize('language', ['ru', 'en'])
@pytest.mark.parametrize('theme', ['breeze_light', 'breeze_dark'])
def test_all_module_commands_tabs_and_context_menus_have_icons(qtbot, qapp, monkeypatch, tmp_path, language, theme):
    store = QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)
    store.setValue('appearance/theme', theme)
    monkeypatch.setattr(i18n, 'settings', lambda: store)
    i18n.save_language(language)
    i18n.initialize()
    apply_theme(qapp, theme)
    owner, extra = build_interfaces()
    qtbot.addWidget(owner)
    try:
        assert qapp.property('autopyroTheme') == theme
        assert inventory(owner) == {'buttons': [], 'tabs': [], 'menus': []}
        action = owner.actions[i18n.tr('Новый проект')]
        original_text, icon_key = action.text(), action.icon().cacheKey()
        action.setText(i18n.tr('Удалить набор проб…'))
        try:
            apply_control_icons(owner)
            assert action.icon().cacheKey() == icon_key
        finally:
            action.setText(original_text)
        button = owner.desktop.table_properties.add_rule_button
        original_text, icon_key = button.text(), button.icon().cacheKey()
        button.setText(i18n.tr('Удалить набор проб…'))
        try:
            apply_control_icons(owner)
            assert button.icon().cacheKey() == icon_key
        finally:
            button.setText(original_text)
        # Drawing modes are independent of their captions, even after a rename.
        from autopyro.ui.dialogs.geometry_editor import GeometryEditor
        modes = owner.findChild(GeometryEditor).mode
        move = modes.findData('move')
        mode_icon = modes.itemIcon(move).cacheKey()
        assert not modes.itemIcon(move).isNull()
        modes.setItemText(move, i18n.tr('Новая область'))
        apply_control_icons(owner)
        assert modes.itemIcon(move).cacheKey() == mode_icon
        payloads = [('project', ''), ('table', ''), ('column', owner.model.project.columns[0].id),
            ('column', owner.model.project.columns[1].id),
            ('folder', 'views'), ('view', 'Audit view'), ('saved_plot', 'Audit plot')]
        payloads += [(kind, '') for kind in ('plot', 'map', 'statistics', 'curve_fitting', 'interpretation',
            'interpolation', 'calculation', 'stratigraphy', 'nonpyro', 'qc', 'report')]
        owner.model.project.view.setdefault('named', {})['Audit view'] = owner.capture_view()
        owner.model.project.view.setdefault('plots', {})['Audit plot'] = owner.plot_window.spec.to_dict()
        for payload in payloads:
            menu = owner.desktop.create_project_menu(payload)
            for action in menu.findChildren(QAction):
                if action.text() and not action.isSeparator():
                    assert not action.icon().isNull(), (payload, action.text())
            menu.deleteLater()
        # Render real menus and the borrowed property tabs, beyond isNull().
        owner.resize(1500, 900)
        owner.show()
        owner.desktop.documents.setCurrentWidget(owner.plot_window)
        qtbot.wait(50)
        assert owner.grab().save(str(tmp_path / f'interface-{theme}-{language}.png'))
        menu = owner.desktop.create_project_menu(('table', ''))
        menu.ensurePolished()
        menu.resize(menu.sizeHint())
        assert menu.grab().save(str(tmp_path / f'menu-{theme}-{language}.png'))
    finally:
        for dialog in extra:
            dialog.close()
        owner.desktop.close_documents()
        for plot in (owner.plot_window, owner.interpretation_window.plot):
            plot.preview.shutdown()
        owner.dirty = False
        i18n.save_language('ru')
        i18n.initialize()
        apply_theme(qapp, 'breeze_light')


def test_every_interface_icon_has_a_bundled_svg_and_paints_in_both_palettes(qapp):
    names = set(STANDARD_BUTTON_ICONS.values()) | set(DOCUMENT_ICONS.values())
    for path in Path('autopyro/ui').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'icon'
                    and node.args and isinstance(node.args[0], ast.Constant)):
                names.add(node.args[0].value)
    icons = {name: icon(name) for name in names}
    try:
        painted = {}
        for theme in ('breeze_light', 'breeze_dark'):
            apply_theme(qapp, theme)
            for name, item in icons.items():
                assert (ASSETS / 'lucide' / (ICON_ALIASES.get(name, name) + '.svg')).is_file(), name
                for mode in (QIcon.Mode.Normal, QIcon.Mode.Disabled, QIcon.Mode.Selected):
                    pixels = item.pixmap(QSize(22, 22), mode).toImage()
                    opaque = [pixels.pixelColor(x, y) for x in range(pixels.width()) for y in range(pixels.height())
                              if pixels.pixelColor(x, y).alpha() > 200]
                    assert opaque, (theme, name, mode)
                    group = QPalette.ColorGroup.Disabled if mode == QIcon.Mode.Disabled else QPalette.ColorGroup.Active
                    role = QPalette.ColorRole.HighlightedText if mode == QIcon.Mode.Selected else QPalette.ColorRole.WindowText
                    color = max(opaque, key=lambda color: color.alpha())
                    expected = qapp.palette().color(group, role)
                    assert all(abs(a - b) <= 2 for a, b in zip(color.getRgb()[:3], expected.getRgb()[:3]))
                    if mode == QIcon.Mode.Normal:
                        painted[theme, name] = color.name()
        assert all(painted['breeze_light', name] != painted['breeze_dark', name] for name in names)
    finally:
        apply_theme(qapp, 'breeze_light')

