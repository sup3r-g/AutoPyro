import json
from copy import deepcopy

import pytest
from PySide6.QtWidgets import QDialog, QFileDialog, QInputDialog, QTableView

from autopyro.core import Column,Project,Row
from autopyro.analysis.nonpyro import REPORTED_BIOMARKER_METHODS
from autopyro.nonpyro import (METHODS,evaluate_project,replay_report,suggest_bindings,
    validate_config,result_columns,recompute,check_saved_reports,sofer_scheme)
from autopyro.geochem_database import project_from_records,profile_fields
from autopyro.reporting import ReportOptions,build_report,render_html


def example(method_id,values=None):
    """Manual columns deliberately require explicit analyst bindings."""
    defaults={
        "sofer": {"sat":(-28,"per mil"),"arom":(-27,"per mil"),"cv_reported":(-.75,"Euc")},
        "vitrinite":{"ro":(.7,"%Ro"),"n":(50,"count")},
        "gas_ratios":{"c1":(90,"%vol"),"c2":(6,"%vol"),"c3":(4,"%vol")},
        "gas_isotopes":{"d1":(-50,"per mil"),"d2":(-35,"per mil"),"d3":(-30,"per mil")},
        "sara":{"sat":(60,"%wt"),"arom":(20,"%wt"),"res":(15,"%wt"),"asph":(5,"%wt")},
        "elemental":{"c":(80,"%wt"),"h":(8,"%wt"),"o":(8,"%wt")},
        "atomic_ratios":{"hc":(120,"%mol (a/b)"),"oc":(7.5,"%mol (a/b)")},
        "gc_profile":{"prph":(2,"1"),"pr17":(.8,"1"),"ph18":(.5,"1")},
        "alkane_indices":{f"c{carbon}":(carbon - 23,"count") for carbon in range(24, 35)},
        "stereo_profile":{"st20":(52,"% (a/(a+b))"),"stbb":(60,"% (a/(a+b))"),"h31":(58,"% (a/(a+b))")},
        "sterane_distribution":{"st27":(25,"count"),"st28":(35,"count"),"st29":(40,"count")},
        "c27_diasterane_ratio":{"dia":(4,"count"),"regular":(2,"count")},
        "ster_hop_from_peaks":{peak:(2,"count") for peak in METHODS["ster_hop_from_peaks"]["roles"]},
        "terpane_hopane_profile":{"gamm":(2,"count"),"h30":(10,"count"),"ts":(3,"count"),"tm":(2,"count")},
        "homohopane_distribution":{f"h{carbon}_total":(carbon,"count") for carbon in range(31, 36)},
        "homohopane_epimers":{f"h{carbon}_{epimer}":(value,"count")
                             for carbon in range(31, 36) for epimer,value in (("s",3),("r",2))},
        "aromatic_profile":{"mpi":(.8,"1"),"dbt":(1.5,"1")},
        "aromatic_indices_from_peaks":{"2mp":(4,"count"),"1mp":(2,"count")},
        "microscopy":{"vit":(50,"%area"),"lipt":(30,"%area"),"inert":(20,"%area")},
        "extract_toc":{"extract":(2,"mg/g rock"),"toc":(4,"%wt")},
    }
    pairs = ({key: (0.5, "1") for key in METHODS[method_id]["roles"]}
             if method_id in REPORTED_BIOMARKER_METHODS else
             {key: (2, "count") for key in METHODS[method_id]["roles"]}
             if method_id.startswith("rallakis_") else defaults[method_id])
    cols={key:Column(key,"number",unit=unit) for key,(value,unit) in pairs.items()}
    cells={cols[key].id:(values or {}).get(key,value) for key,(value,unit) in pairs.items()}
    project=Project(columns=list(cols.values()),rows=[Row(cells,source={"source_record_id":1})])
    config={"version":1,"method":method_id,"confirmed":True,"context":"Synthetic QA",
        "isotope_reference":"VPDB","material":"oil_C15plus","population_confirmed":True,
        "inputs":{key:{"column_id":col.id,"unit":col.unit} for key,col in cols.items()}}
    return project,config,cols


def result(project,config):
    return evaluate_project(project,[r.id for r in project.rows],config)["results"][0]


def test_rallakis_gc_uses_the_published_carbon_windows():
    signals = {f"nC{carbon}": 1 for carbon in range(21, 31)}
    signals.update(nC23=3, nC27=5)
    project, config, _ = example("rallakis_gc", signals)
    outputs = result(project, config)["values"]
    assert outputs["CPI1"] == pytest.approx(2.5)
    assert outputs["CPI"] == pytest.approx(0.5)
    assert outputs["OEP1"] == pytest.approx(2.5)
    assert outputs["OEP2"] == pytest.approx(4)


def test_rallakis_aromatic_calibrations():
    project, config, _ = example("rallakis_aromatics", {
        "1MP": 2, "2MP": 4, "3MP": 6, "9MP": 8, "Phen": 10,
        "1MDBT": 2, "3+2MDBT": 4, "4MDBT": 8,
    })
    outputs = result(project, config)["values"]
    assert outputs == pytest.approx({
        "18": 2, "19": 0.75, "20": 0.5, "21": 4, "26": 0.75, "27": 0.7,
        "22": 1.281132995, "23": 0.85, "24": 0.955, "25": 0.802,
    })


@pytest.mark.parametrize('custom_metadata', [False, True])
def test_profile_snapshot_owns_analysis_runs_and_measurements(custom_metadata):
    project, config, columns = example('c27_diasterane_ratio')
    row, = project.rows
    row.source = {'source_record_id': 1, 'analysis_runs': {
        'Sat-GCMS': {'analysis_run_id': 19, 'repeat_number': 2},
    }}
    if custom_metadata:
        row.source['custom'] = {'flags': ['checked']}
    project.sync_measurements()
    column_id = columns['dia'].id
    measurement_id = project.science.working[row.id][column_id]
    project.view['qc_exclusions'] = {'statistics': {measurement_id: {'revision': 1}}}
    report = evaluate_project(project, [row.id], config)
    original = deepcopy(report['inputs'])
    assert report['results'][0]['status'] == 'qc_excluded'
    row.source['analysis_runs']['Sat-GCMS']['repeat_number'] = 3
    if custom_metadata:
        row.source['custom']['flags'].append('edited')
    project.science.measurements[measurement_id].revision += 1
    assert report['inputs'] == original
    replayed = replay_report(report)
    assert replayed['inputs'] == original
    replayed['inputs'][0]['measurements'][column_id]['revision'] += 1
    replayed['inputs'][0]['qc_exclusions'][column_id]['revision'] += 1
    assert report['inputs'] == original


def test_profile_coverage_before_confirmation_uses_qc_and_matches_results(qtbot, monkeypatch):
    from PySide6.QtWidgets import QLabel

    from autopyro.ui.window import MainWindow

    method = 'c27_diasterane_ratio'
    project, config, columns = example(method)
    sample = Column('Sample / Name')
    project.columns.append(sample)
    project.rows = [
        Row({sample.id: name, columns['dia'].id: dia, columns['regular'].id: regular})
        for name, dia, regular in (('Valid', 8, 4), ('Missing', None, 4),
                                   ('Zero denominator', 8, 0), ('QC removed', 8, 4))
    ]
    project.sync_measurements()
    measurement = project.science.working[project.rows[-1].id][columns['dia'].id]
    project.view['qc_exclusions'] = {'statistics': {
        measurement: {'revision': project.science.measurements[measurement].revision},
    }}
    project.view['nonpyro_configs'] = {method: {**config, 'context': '', 'confirmed': False}}
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.show_nonpyro('steranes')
    dialog = window.geochemistry_windows['steranes']
    dialog.method.setCurrentIndex(dialog.method.findData(method))
    assert dialog.preview_button.isEnabled()
    assert not dialog.calculate_button.isEnabled()
    shown = []

    def inspect_coverage(preview):
        summary = preview.findChild(QLabel).text()
        model = preview.findChild(QTableView).model()
        shown.append((summary, model.rows))

    monkeypatch.setattr(QDialog, 'exec', inspect_coverage)
    dialog.preview_button.click()
    summary, rows = shown[0]
    assert '1 из 4' in summary
    assert {row[3] for row in rows[1:]} == {'Missing', 'Zero denominator', 'QC removed'}
    assert all(row[2] == 1 for row in rows)
    assert dialog.report is None and 'nonpyro_results' not in project.view
    dialog.form.fields['context'][0].setText('fixture')
    dialog.form.fields['confirmed'][0].setChecked(True)
    dialog.calculate_button.click()
    assert [row['status'] for row in dialog.report['results']] == [
        'profile', 'missing', 'zero_denominator', 'qc_excluded',
    ]
    window.dirty = False


@pytest.mark.parametrize('method_id', REPORTED_BIOMARKER_METHODS)
def test_reported_biomarker_fields_bind_and_normalize(method_id):
    definition = METHODS[method_id]
    group, = definition['groups']
    units = {'ratio': ('Euc (a/b)', 2, 2), 'fraction': ('% (a/(a+b))', 40, 0.4),
             'percent_ratio': ('% (a/b)', 125, 1.25)}
    fields = []
    expected = {}
    observations = []
    for i, (key, role) in enumerate(definition['roles'].items(), 1):
        unit, raw, expected[key] = units[role['unit_family']]
        fields.append({'source_field_id': i, 'group_name': group,
                       'source_label': f"{role['prefixes'][0]} [{unit}]",
                       'unit_label': unit, 'source_dtype': 'Float64'})
        observations.append({'source_record_id': 1, 'source_field_id': i, 'analysis_run_id': 1,
                             'repeat_number': 1, 'group_name': group, 'value_float': raw})
    assert profile_fields({'fields': fields}, method_id, group, 'h') == fields
    project = project_from_records([{'source_record_id': 1, 'sample_name': 'QA'}],
                                   observations, fields, source={})
    config = {'method': method_id, 'version': 1, 'confirmed': True, 'context': 'QA',
              'inputs': suggest_bindings(project, method_id, group, 'h')}
    assert result(project, config)['values'] == pytest.approx(expected)
    if method_id == 'aromatic_steroid_distribution':
        assert result(project, config)['warnings'] == ['MAS: сумма исходных долей 120%',
                                                       'TAS: сумма исходных долей 120%']
    removed = next(iter(expected))
    project.rows[0].values[config['inputs'][removed]['column_id']] = None
    del expected[removed]
    assert result(project, config)['values'] == pytest.approx(expected)
    project.rows[0].values[config['inputs'][removed]['column_id']] = -1
    row = result(project, config)
    assert row['values'] == pytest.approx(expected)
    assert f'{removed}: вне допустимого диапазона' in row['warnings']


def test_homohopane_epimers_calculate_only_complete_pairs():
    project, config, columns = example('homohopane_epimers',
                                       {'h31_s': 0, 'h32_r': None, 'h33_s': 0, 'h33_r': 0, 'h34_r': 0})
    row = result(project, config)
    assert row['status'] == 'profile'
    assert row['values'] == {'C31_22S': 0, 'C33_22S': None, 'C34_22S': 1,
                             'C35_22S': pytest.approx(0.6)}
    assert any(warning.startswith('C33 22S/') for warning in row['warnings'])
    for carbon in range(31, 36):
        del config['inputs'][f'h{carbon}_r']
    with pytest.raises(ValueError, match='хотя бы для одного гомолога'):
        validate_config(project, config)


def test_homohopane_distribution_requires_full_series_and_one_basis():
    project, config, columns = example('homohopane_distribution',
                                       {f'h{carbon}_total': 0 for carbon in range(31, 36)})
    assert result(project, config)['status'] == 'zero_denominator'
    columns['h35_total'].unit = config['inputs']['h35_total']['unit'] = 'ppm (µg/g)'
    with pytest.raises(ValueError, match='одной основой'):
        validate_config(project, config)
    del config['inputs']['h35_total']
    with pytest.raises(ValueError, match='оба эпимера'):
        validate_config(project, config)


def test_terpane_hopane_profile_independent_ratios_and_binding_validation():
    project, config, columns = example('terpane_hopane_profile', {'ts': 0, 'tm': 0})
    row = result(project, config)
    assert row['status'] == 'profile'
    assert row['values'] == {'Gamm_C30': 0.2, 'Ts_fraction': None}
    assert any('нулевой знаменатель' in warning for warning in row['warnings'])

    project.rows[0].values[columns['h30'].id] = None
    assert result(project, config)['status'] == 'zero_denominator'
    project.rows[0].values[columns['tm'].id] = None
    assert result(project, config)['status'] == 'not_applicable'

    columns['h30'].unit = config['inputs']['h30']['unit'] = 'ppm (µg/g)'
    with pytest.raises(ValueError, match='одной основой'):
        validate_config(project, config)
    del config['inputs']['h30'], config['inputs']['tm']
    with pytest.raises(ValueError, match='полную пару'):
        validate_config(project, config)


def test_geochemistry_workflow_readiness_results_and_stale_actions(qtbot, monkeypatch, tmp_path):
    from autopyro.ui.window import MainWindow
    project, config, cols = example('sofer')
    project.view['nonpyro_configs'] = {'sofer': config}
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_nonpyro('oil_isotopes')
    dialog = owner.geochemistry_windows['oil_isotopes']
    assert not dialog.calculate_button.isEnabled()
    assert not any(button.isEnabled() for button in dialog.result_actions.values())
    dialog.form.fields['confirmed'][0].setChecked(True)
    assert dialog.calculate_button.isEnabled(), dialog.workflow_hint.text()
    dialog.calculate_button.click()
    assert dialog.report is not None, dialog.status.text()
    from PySide6.QtCore import Qt

    model = dialog.results.model()
    assert model.columnCount() == 5 + len(METHODS['sofer']['outputs'])
    assert model.headerData(3, Qt.Orientation.Horizontal) == 'CV Sofer [1]'
    assert float(model.index(0, 3).data()) == pytest.approx(-.75)
    assert project.rows[0].id in model.index(0, 3).data(Qt.ItemDataRole.ToolTipRole)
    assert dialog.result_actions['export'].isEnabled()
    assert not dialog.result_actions['select'].isEnabled()
    dialog.results.selectRow(0)
    assert dialog.result_actions['select'].isEnabled()
    path = tmp_path / 'profile.json'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), ''))
    dialog.result_actions['export'].click()
    assert json.loads(path.read_text('utf-8'))['results'][0]['row_id'] == project.rows[0].id
    dialog.result_actions['select'].click()
    assert [index.row() for index in owner.table.selectionModel().selectedRows()] == [0]
    dialog.form.fields['context'][0].setText('Changed material')
    assert dialog.report is None
    assert not any(button.isEnabled() for button in dialog.result_actions.values())
    owner.dirty = False


def test_large_profile_prepares_and_exports_off_gui_path(qtbot, monkeypatch, tmp_path):
    import csv

    from autopyro.reports.reporting import encode_report, export_report
    from autopyro.ui.dialogs import nonpyro_calculation, nonpyro_reports
    from autopyro.ui.tasks import TaskCancelled
    from autopyro.ui.window import MainWindow

    method = 'c27_diasterane_ratio'
    project, config, _ = example(method)
    project.rows = [Row(dict(project.rows[0].values)) for _ in range(256)]
    project.view['nonpyro_configs'] = {method: config}
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_nonpyro('steranes')
    dialog = owner.geochemistry_windows['steranes']
    dialog.method.setCurrentIndex(dialog.method.findData(method))
    dialog.form.fields['confirmed'][0].setChecked(True)
    operations = []
    checkpoints = []

    def dispatch(parent, title, operation, *, cancellable=False):
        operations.append(title)
        return operation(lambda _, percent: checkpoints.append(percent)) if cancellable else operation()

    monkeypatch.setattr(nonpyro_calculation, 'run_with_progress', dispatch)
    monkeypatch.setattr(nonpyro_reports, 'run_with_progress', dispatch)
    revision = owner.analysis_revision
    dialog.calculate_button.click()
    assert dialog.results.model().rowCount() == 256
    assert owner.analysis_revision == revision
    for suffix in ('csv', 'json'):
        output = tmp_path / f'profile.{suffix}'
        monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(output), ''))
        dialog.export()
        if suffix == 'csv':
            with output.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.reader(stream, delimiter=';'))
            assert len(rows) == 257 and float(rows[1][2]) == 2
        else:
            assert json.loads(output.read_text('utf-8'))['results'] == dialog.report['results']
            before = output.read_bytes()
            assert before == encode_report(dialog.report, 'json')
            invalid = deepcopy(dialog.report)
            invalid['results'][-1]['values']['dia27_sr'] = float('nan')
            with pytest.raises(ValueError):
                export_report(output, invalid, 'json')
            assert output.read_bytes() == before
    assert len(operations) == 3
    assert checkpoints == [0, 30, 60, 90]

    saved = project.view['nonpyro_results'][method]
    history = list(project.history)

    def cancel(_, title, operation, *, cancellable=False):
        assert cancellable

        def update(_, percent):
            if percent == 60:
                raise TaskCancelled()

        return operation(update)

    monkeypatch.setattr(nonpyro_calculation, 'run_with_progress', cancel)
    dialog.calculate_button.click()
    assert dialog.status.text() == 'Расчёт отменён.'
    assert project.view['nonpyro_results'][method] is saved
    assert project.history == history
    owner.dirty = False


def test_chromatography_calculates_alkane_indices(qtbot):
    from autopyro.ui.window import MainWindow
    project, config, _ = example('alkane_indices')
    project.view['nonpyro_configs'] = {'alkane_indices': config}
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_nonpyro('chromatography')
    dialog = owner.geochemistry_windows['chromatography']
    dialog.method.setCurrentIndex(dialog.method.findData('alkane_indices'))
    assert len(dialog.roles) == 11
    assert not dialog.calculate_button.isEnabled()
    dialog.form.fields['confirmed'][0].setChecked(True)
    assert dialog.calculate_button.isEnabled(), dialog.workflow_hint.text()
    dialog.calculate_button.click()
    assert dialog.report['results'][0]['values'] == pytest.approx({'CPI': 36 / 35, 'OEP27': 1})
    assert dialog.result_actions['export'].isEnabled()
    assert not dialog.alkane_profile_button.isEnabled()
    dialog.results.selectRow(0)
    assert dialog.alkane_profile_button.isEnabled()
    dialog.alkane_profile_button.click()
    assert dialog.result_tabs.currentWidget() is dialog.alkane_canvas
    bars = dialog.alkane_canvas.figure.axes[0].patches
    assert [bar.get_height() for bar in bars] == pytest.approx([100 * i / 66 for i in range(1, 12)])
    assert len(project.columns) == 11
    dialog.form.fields['context'][0].setText('Updated context')
    assert not dialog.result_tabs.isTabVisible(dialog.result_tabs.indexOf(dialog.alkane_canvas))
    owner.dirty = False


def test_sara_ternary_filters_balance_and_opens_clicked_result(qtbot):
    from types import SimpleNamespace
    from autopyro.ui.window import MainWindow

    project, config, columns = example('sara', {'sat': 59})
    invalid = Row({**project.rows[0].values, columns['sat'].id: 40})
    project.rows.insert(0, invalid)
    project.view['nonpyro_configs'] = {'sara': config}
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_nonpyro('sara')
    dialog = owner.geochemistry_windows['sara']
    dialog.form.fields['confirmed'][0].setChecked(True)
    dialog.calculate_button.click()
    assert dialog.report['results'][1]['values']['sum'] == 99
    dialog.sara_plot_button.click()
    assert dialog.result_tabs.currentWidget() is dialog.sara_canvas
    scatter = dialog.sara_canvas.figure.axes[0].collections[0]
    assert tuple(scatter.get_offsets()[0]) == pytest.approx((30 / 99, 20 / 99 * 3 ** .5 / 2))
    assert len(scatter.get_offsets()) == 1
    assert '1 из 2' in dialog.status.text()
    assert len(project.columns) == 4
    dialog.select_sara_point(SimpleNamespace(ind=[0], artist=scatter))
    assert dialog.result_tabs.currentWidget() is dialog.results
    assert dialog.results.selectionModel().selectedRows()[0].row() == 1
    dialog.form.fields['context'][0].setText('Updated context')
    assert not dialog.result_tabs.isTabVisible(dialog.result_tabs.indexOf(dialog.sara_canvas))
    owner.dirty = False


def test_sara_profiles_compare_selected_raw_fractions(qtbot):
    from autopyro.ui.window import MainWindow

    project, config, columns = example('sara')
    values = project.rows[0].values
    project.rows.extend([
        Row({**values, columns['sat'].id: 40, columns['res'].id: 5, columns['asph'].id: 15}),
        Row({**values, columns['asph'].id: 'ND'}),
    ])
    project.view['nonpyro_configs'] = {'sara': config}
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_nonpyro('sara')
    dialog = owner.geochemistry_windows['sara']
    dialog.form.fields['confirmed'][0].setChecked(True)
    dialog.calculate_button.click()
    assert not dialog.sara_profiles_button.isEnabled()
    dialog.results.selectAll()
    dialog.sara_profiles_button.click()
    canvas = dialog.sara_profiles_canvas
    assert dialog.result_tabs.currentWidget() is canvas
    assert [list(line.get_ydata()) for line in canvas.figure.axes[0].lines] == [
        [60, 20, 15, 5], [40, 20, 5, 15],
    ]
    assert '2 из 3' in dialog.status.text()
    assert len(project.columns) == 4
    dialog.form.fields['context'][0].setText('Updated context')
    assert not dialog.result_tabs.isTabVisible(dialog.result_tabs.indexOf(canvas))
    owner.dirty = False


def test_chung_plot_uses_reciprocal_carbon_number_and_complete_rows(qtbot):
    from autopyro.ui.window import MainWindow

    project, config, columns = example('gas_isotopes')
    project.rows.append(Row({**project.rows[0].values, columns['d2'].id: 'ND'}))
    project.view['nonpyro_configs'] = {'gas_isotopes': config}
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_nonpyro('gas')
    dialog = owner.geochemistry_windows['gas']
    dialog.method.setCurrentIndex(dialog.method.findData('gas_isotopes'))
    dialog.form.fields['confirmed'][0].setChecked(True)
    dialog.calculate_button.click()
    assert not dialog.chung_button.isEnabled()
    dialog.results.selectAll()
    dialog.chung_button.click()
    assert dialog.result_tabs.currentWidget() is dialog.chung_canvas
    axes = dialog.chung_canvas.figure.axes[0]
    assert len(axes.lines) == 1
    line = axes.lines[0]
    assert list(line.get_xdata()) == pytest.approx([1 / 3, 1 / 2, 1])
    assert list(line.get_ydata()) == [-30, -35, -50]
    assert 'VPDB' in axes.get_ylabel()
    assert '1 из 2' in dialog.status.text()
    assert len(project.columns) == 3
    dialog.form.fields['context'][0].setText('Updated context')
    assert not dialog.result_tabs.isTabVisible(dialog.result_tabs.indexOf(dialog.chung_canvas))
    owner.dirty = False


def test_geochemistry_autobinding_fills_only_missing_roles(qtbot, monkeypatch):
    from autopyro.ui.window import MainWindow
    from autopyro.ui.dialogs import nonpyro_inputs
    project, config, cols = example('gas_ratios')
    project.view['nonpyro_configs'] = {'gas_ratios': config}
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_nonpyro('gas')
    dialog = owner.geochemistry_windows['gas']
    dialog.variant.setCurrentIndex(dialog.variant.findData('mol'))
    assert dialog.roles['c1'][0].currentData() == cols['c1'].id
    dialog.roles['c2'][0].setCurrentIndex(0)
    monkeypatch.setattr(nonpyro_inputs, 'suggest_bindings', lambda *args: {
        'c1': {'column_id': cols['c3'].id, 'unit': '%vol'},
        'c2': {'column_id': cols['c2'].id, 'unit': '%vol'},
    })
    dialog.autobind_button.click()
    assert dialog.roles['c1'][0].currentData() == cols['c1'].id
    assert dialog.roles['c2'][0].currentData() == cols['c2'].id
    assert dialog.roles['c3'][0].currentData() == cols['c3'].id
    owner.dirty = False


def test_geochemistry_bindings_follow_table_column_changes(qtbot):
    from autopyro.ui.window import MainWindow
    from autopyro.ui.table import AddColumnCommand
    project, config, cols = example('sofer')
    project.view['nonpyro_configs'] = {'sofer': config}
    owner = MainWindow()
    qtbot.addWidget(owner)
    owner.set_project(project)
    owner.show_nonpyro('oil_isotopes')
    dialog = owner.geochemistry_windows['oil_isotopes']
    column = Column('Extra isotope measurement', 'number', unit='per mil')
    owner.model.undo.push(AddColumnCommand(owner.model, column, {}))
    assert dialog.roles['sat'][0].findData(column.id) >= 0
    assert dialog.roles['sat'][0].currentData() == cols['sat'].id
    owner.model.undo.undo()
    assert dialog.roles['sat'][0].findData(column.id) == -1
    owner.dirty = False


def test_geochemistry_import_filters_invalidate_loaded_preview(qtbot):
    from autopyro.database.ui.nonpyro import NonPyroDatabaseDialog
    well, sample = Column('Well / Well Name'), Column('Sample / Name')
    project = Project(columns=[well, sample], rows=[Row(
        {well.id: 'A', sample.id: 'Sample 1'},
        source={'source_record_id': 1, 'analysis_runs': []})])
    dialog = NonPyroDatabaseDialog()
    qtbot.addWidget(dialog)
    assert dialog.method.count() == len(METHODS)
    dialog.loaded(project)
    assert dialog.use.isEnabled()
    assert dialog.preview_table.rowCount() == 1
    assert dialog.tabs.currentWidget() is dialog.preview_table
    qtbot.keyClicks(dialog.form.fields['well_name'][0], 'Another well')
    assert dialog.project is None
    assert not dialog.use.isEnabled()
    assert dialog.preview_table.rowCount() == 0


def test_geochemistry_import_keeps_actions_visible_at_compact_height(qtbot):
    from autopyro.database.ui.nonpyro import NonPyroDatabaseDialog

    dialog = NonPyroDatabaseDialog()
    qtbot.addWidget(dialog)
    dialog.combine.setChecked(True)
    dialog.resize(1000, 640)
    dialog.show()
    scroll = dialog.settings_scroll
    qtbot.waitUntil(lambda: scroll.verticalScrollBar().maximum() > 0)

    assert dialog.height() == 640
    assert dialog.rect().contains(dialog.use.geometry())
    assert dialog.rect().contains(dialog.tabs.geometry())
    assert dialog.rect().contains(dialog.status.geometry())
    header = dialog.fields_table.horizontalHeader()
    last = header.count() - 1
    assert header.sectionSize(last) >= header.sectionSizeHint(last)
    scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
    assert scroll.viewport().rect().contains(dialog.profile_list.mapTo(scroll.viewport(), dialog.profile_list.rect().bottomLeft()))


@pytest.mark.parametrize('append', [False, True])
def test_geochemistry_import_cannot_accept_empty_snapshot(qtbot, monkeypatch, append):
    from autopyro.database.ui import nonpyro

    project = Project(columns=[Column('Well / Well Name'), Column('Sample / Name')])
    monkeypatch.setattr(nonpyro, 'read_profile', lambda *args, **kwargs: project)
    dialog = nonpyro.NonPyroDatabaseDialog(append=append)
    qtbot.addWidget(dialog)
    dialog.read()
    qtbot.waitUntil(lambda: not dialog.task_running)

    assert dialog.project is None
    assert not dialog.use.isEnabled()
    assert dialog.preview_table.rowCount() == dialog.coverage_table.rowCount() == 0
    assert 'записи не найдены' in dialog.status.text()


def test_geochemistry_import_checks_working_values_and_qc_before_acceptance(qtbot, monkeypatch):
    from autopyro.database.ui import nonpyro
    from autopyro.ui.geochemistry import status_label

    well, sample = Column('Well / Well Name'), Column('Sample / Name')
    dia, regular = [Column(f'Sat-GCMS / {peak}[h]', 'number', unit='count')
                    for peak in ('St27dbS', 'St27aaR')]
    project = Project(columns=[well, sample, dia, regular], rows=[
        Row({well.id: 'A', sample.id: str(i), dia.id: numerator, regular.id: denominator},
            source={'source_record_id': i, 'analysis_runs': []})
        for i, (numerator, denominator) in enumerate(((8, 4), ('<2', 4), (8, 0), (8, 4), (None, 4)))
    ])
    project.sync_measurements()
    excluded = project.science.working[project.rows[3].id][dia.id]
    project.view['qc_exclusions'] = {'statistics': {
        excluded: {'revision': project.science.measurements[excluded].revision},
    }}
    view_before = deepcopy(project.view)
    monkeypatch.setattr(nonpyro, 'read_profile', lambda *args, **kwargs: project)
    dialog = nonpyro.NonPyroDatabaseDialog('c27_diasterane_ratio')
    qtbot.addWidget(dialog)

    dialog.read()
    qtbot.waitUntil(dialog.use.isEnabled)

    table = dialog.coverage_table
    assert table.item(0, 3).text() == '4'
    assert table.item(0, 4).text() == '1 / 5'
    assert 'Рабочий повтор: 1' in dialog.status.text()
    assert 'пригодных строк: 1 из 5' in dialog.status.text()
    for status in ('censored', 'zero_denominator', 'qc_excluded', 'missing'):
        assert f'{status_label(status)}: 1' in table.item(0, 5).text()
    assert project.view == view_before
    assert dialog.result() == dialog.DialogCode.Rejected


def test_scope_membership_prepared_once_without_changing_protocol_order():
    class ScopeList(list):
        searches = 0

        def __contains__(self, value):
            self.searches += 1
            return super().__contains__(value)

    project, config, _ = example("atomic_ratios")
    second = deepcopy(project.rows[0])
    second.id = "outside-scope"
    project.rows.append(second)
    scope = ScopeList([project.rows[0].id])
    config["scope_row_ids"] = scope
    ids = [second.id, project.rows[0].id]
    report = evaluate_project(project, ids, config)
    assert scope.searches == 0
    assert config["scope_row_ids"] is scope
    assert report["config"]["scope_row_ids"] == list(scope)
    assert [item["row_id"] for item in report["results"]] == ids
    assert [item["status"] for item in report["results"]] == ["outside_scope", "profile"]
    assert replay_report(json.loads(json.dumps(report)))["results"] == report["results"]


@pytest.mark.parametrize("method_id",list(METHODS))
def test_profiles_replay_exactly(method_id):
    project,config,_=example(method_id)
    protocol=evaluate_project(project,[project.rows[0].id],config)
    assert protocol["results"][0]["status"] in ("interpreted","profile")
    replay=replay_report(json.loads(json.dumps(protocol)))
    assert replay["results"]==protocol["results"]
    assert protocol["definition_hash"]==replay["definition_hash"]


def test_alkane_indices_use_both_even_windows_and_weight_c27():
    project, config, columns = example('alkane_indices')
    assert result(project, config)['values'] == pytest.approx({'CPI': 36 / 35, 'OEP27': 1})
    project.rows[0].values[columns['c27'].id] = 10
    assert result(project, config)['values'] == pytest.approx({'CPI': 216 / 175, 'OEP27': 68 / 32})


def test_alkane_distribution_handles_zero_and_unusable_signals():
    from autopyro.nonpyro import alkane_distribution
    project, config, columns = example('alkane_indices')
    row = project.rows[0]
    for column in columns.values():
        row.values[column.id] = 0
    assert alkane_distribution(result(project, config)) == []
    row.values[columns['c24'].id] = 4
    profile = alkane_distribution(result(project, config))
    assert profile == [(24, 100)] + [(carbon, 0) for carbon in range(25, 35)]
    row.values[columns['c34'].id] = 'ND'
    assert alkane_distribution(result(project, config)) == []


@pytest.mark.parametrize('raw,status', [('ND', 'non_detect'), ('<1', 'censored'), (-1, 'invalid_range')])
def test_alkane_indices_reject_nonquantitative_signals(raw, status):
    project, config, _ = example('alkane_indices', {'c25': raw})
    assert result(project, config)['status'] == status


def test_alkane_indices_zero_denominator_and_mixed_units():
    project, config, columns = example('alkane_indices', {'c26': 0, 'c28': 0})
    values = result(project, config)['values']
    assert values['CPI'] > 0 and values['OEP27'] is None
    columns['c25'].unit = config['inputs']['c25']['unit'] = 'ppm (µg/g)'
    with pytest.raises(ValueError, match='одну единицу'):
        result(project, config)


def test_alkane_database_fields_and_bindings_keep_only_requested_basis():
    fields = [{'group_name': 'Sat-GC', 'source_label': f'nC{carbon}[{variant}]'}
              for carbon in range(24, 35) for variant in ('a', 'ca', 'un')]
    selected = profile_fields({'fields': fields}, 'alkane_indices', 'Sat-GC', 'a')
    assert len(selected) == 11 and all(f['source_label'].endswith('[a]') for f in selected)
    assert len(profile_fields({'fields': fields}, 'alkane_indices')) == 22
    columns = [Column(f['source_label'], 'number', unit='count', metadata={'geochem_field': {
        'group': f['group_name'], 'label': f['source_label']}}) for f in selected]
    assert set(suggest_bindings(Project(columns=columns), 'alkane_indices', 'Sat-GC', 'a')) == {
        f'c{carbon}' for carbon in range(24, 35)}


def test_sofer_reference_boundary_and_signed_reported_cv():
    project,config,cols=example("sofer")
    assert result(project,config)["values"]["CV"]==pytest.approx(-.75)
    assert result(project,config)["values"]["CV_difference"]==pytest.approx(0)
    project.rows[0].values[cols["arom"].id]=(.47+11.65+2.53*(-28))/2.22
    assert result(project,config)["status"]=="boundary"
    config["confirmed"]=False
    with pytest.raises(ValueError,match="контекст"):
        result(project,config)


@pytest.mark.parametrize("method,key,expected",[("gas_ratios","Bernard",9),
    ("gas_ratios","dryness3",90),("elemental","HC",1.2),("elemental","OC",.075),
    ("atomic_ratios","HC",1.2),("stereo_profile","St20S",.52),("extract_toc","EOM_TOC",50)])
def test_reference_numbers(method,key,expected):
    project,config,_=example(method)
    assert result(project,config)["values"][key]==pytest.approx(expected)


@pytest.mark.parametrize("value,status",[(None,"missing"),("ND","non_detect"),("<0.2","censored"),(-1,"invalid_range")])
def test_qualifiers_never_imputed(value,status):
    project,config,cols=example("sara",{"asph":value})
    item=result(project,config)
    assert item["status"]==status
    assert item["values"]=={}
    assert project.rows[0].values[cols["asph"].id]==value


def test_zero_denominator_closure_and_invalid_fraction():
    project,config,_=example("extract_toc",{"toc":0})
    assert result(project,config)["status"]=="zero_denominator"
    project,config,_=example("sara",{"sat":40})
    item=result(project,config)
    assert item["values"]["sum"]==80
    assert "баланс" in " ".join(item["warnings"])
    project,config,_=example("stereo_profile",{"st20":101})
    assert result(project,config)["status"]=="invalid_range"


def test_gas_basis_and_source_unit_cannot_be_relabelled():
    project,config,cols=example("gas_ratios")
    config["inputs"]["c2"]["unit"]="%mol"
    with pytest.raises(ValueError):
        validate_config(project,config)
    config["inputs"]["c2"]["unit"]="ppm vol"
    with pytest.raises(ValueError,match="единица"):
        validate_config(project,config)


def test_forbid_pyro_sources_mixed_gc_variants_and_wrong_vitrinite_population():
    project,config,cols=example("sofer")
    cols["sat"].name="Isotopes / d13C Pyrol [per mil]"
    with pytest.raises(ValueError,match="Пиролитические"):
        result(project,config)
    project,config,cols=example("gc_profile")
    for key,prefix,variant in (("prph","Pr/Ph","a"),("pr17","Pr/nC17","h"),("ph18","Ph/nC18","a")):
        cols[key].name=f"Sat-GC / {prefix}[{variant}] [Euc]"
    with pytest.raises(ValueError,match="смешивать"):
        result(project,config)
    project,config,cols=example("vitrinite")
    cols["ro"].name="VR / Modal pop Mean [%]"
    cols["n"].name="VR / Indig pop Num [Euc]"
    with pytest.raises(ValueError,match="популяции"):
        result(project,config)


def test_vitrinite_requires_population_and_warns_low_n():
    project,config,_=example("vitrinite",{"n":2})
    config["population_confirmed"]=False
    assert result(project,config)["status"]=="not_applicable"
    config["population_confirmed"]=True
    assert "40–50" in " ".join(result(project,config)["warnings"])


def test_frozen_scope_interpolations_and_stale_protocol(tmp_path):
    project,config,cols=example("sofer")
    report=evaluate_project(project,[project.rows[0].id],config)
    project.view["nonpyro_results"]={"sofer":report}
    check_saved_reports(project)
    assert not report["stale"]
    project.rows[0].values[cols["sat"].id]=-30
    check_saved_reports(project)
    assert report["stale"]
    assert replay_report(report)["results"]==report["results"]
    config["scope_row_ids"]=[]
    assert result(project,config)["status"]=="outside_scope"
    config.pop("scope_row_ids")
    cols["sat"].metadata["method"]="linear"
    assert result(project,config)["status"]=="interpolation_excluded"


def test_report_recalculates_and_respects_preview_limit():
    project,config,cols=example("sofer")
    project.view["nonpyro_configs"]={"sofer":config}
    project.view["nonpyro_results"]={"sofer":{"stale":True}}
    project.columns.extend(result_columns(config))
    report=build_report(project,[project.rows[0].id],ReportOptions((cols["sat"].id,),data_limit=0))
    assert report["nonpyro"][0]["results"][0]["values"]["CV"]==pytest.approx(-.75)
    assert "Геохимия вне пиролиза" in render_html(report)
    assert project.view["nonpyro_results"]["sofer"]["stale"]
    disabled=build_report(project,[],ReportOptions((cols["sat"].id,),include_nonpyro=False))
    assert disabled["nonpyro"]==[]


@pytest.mark.parametrize(("method", "name", "note", "finding"), [
    ("rallakis_gc", "Rallakis · GC-FID", "The shortened CPI is intended for extracts.",
     "Rallakis indicators calculated from measured peaks"),
    ("rallakis_steranes", "Rallakis · regular C27–C29 ββ 20S steranes",
     "Assign comparable signals for the three isomers.",
     "Fractions of regular ββ 20S steranes in the C27–C29 total"),
])
def test_english_rallakis_report_localizes_method_and_finding(method, name, note, finding):
    project, config, columns = example(method)
    project.view["nonpyro_configs"] = {method: config}
    options = ReportOptions((next(iter(columns.values())).id,), language="en", sections=("nonpyro",))
    report = build_report(project, [project.rows[0].id], options)
    html = render_html(report)
    assert name in html and note in html and finding in html


def test_changed_method_snapshot_rejected():
    project,config,_=example("sofer")
    protocol=evaluate_project(project,[project.rows[0].id],config)
    protocol["definition"]["note"]="Changed definition"
    with pytest.raises(ValueError,match="Определение"):
        replay_report(protocol)


def test_sofer_shapely_agrees_and_windows_have_distinct_ids():
    from autopyro.interpretation import evaluate_project as geometry_evaluate
    project,config,_=example("sofer")
    project.columns.extend(result_columns(config))
    recompute(project)
    outputs={c.metadata["nonpyro"]["output"]:c.id for c in project.columns if "nonpyro" in c.metadata}
    x,y=outputs["d13C_sat"],outputs["d13C_arom"]
    ids=[r.id for r in project.rows]
    scheme=sofer_scheme(project,x,y,ids)
    interpreted=geometry_evaluate(project,ids,x,y,"per mil","per mil",scheme)
    region_id=interpreted["results"][0]["regions"][0]
    assert "Морской" in next(r["label"] for r in scheme["regions"] if r["id"]==region_id)
    project.rows[0].values[x]=-40
    assert sofer_scheme(project,x,y,ids)["id"]!=scheme["id"]


@pytest.mark.parametrize('variant,unit', [
    ('h', 'count'), ('a', 'count'), ('ch', 'ppm (µg/g)'), ('ca', 'ppm (µg/g)'),
])
def test_steranes_accept_complete_database_signal_variants(variant, unit):
    project, config, columns = example('sterane_distribution')
    for carbon in (27, 28, 29):
        column = columns[f'st{carbon}']
        column.name = f'Sat-GCMS / St{carbon}aaR[{variant}] [{unit}]'
        column.unit = unit
    config['inputs'] = suggest_bindings(project, 'sterane_distribution', 'Sat-GCMS', variant)
    assert result(project, config)['values'] == pytest.approx({'C27': 25, 'C28': 35, 'C29': 40})


def test_steranes_reject_mixed_signal_variants_and_diasteranes():
    project, config, columns = example('sterane_distribution')
    for carbon in (27, 28, 29):
        columns[f'st{carbon}'].name = f'Sat-GCMS / St{carbon}aaR[h] [count]'
    columns['st28'].name = 'Sat-GCMS / St28aaR[a] [count]'
    with pytest.raises(ValueError, match='Нельзя смешивать'):
        result(project, config)
    columns['st28'].name = 'Sat-GCMS / St28dbR[h] [count]'
    with pytest.raises(ValueError, match='не соответствует роли'):
        result(project, config)
    columns['st28'].name = 'Sat-GCMS / St28aaR[un] [unknown]'
    with pytest.raises(ValueError, match='известной основе'):
        result(project, config)


def test_database_builder_keeps_duplicate_names_and_provenance():
    records=[{"source_record_id":i,"sample_id":i+10,"sample_name":"Same","import_batch_id":7} for i in (1,2)]
    fields=[{"source_field_id":3,"group_name":"Elem","source_label":"H/C mol [%mol (a/b)]","unit_label":"%mol (a/b)","source_dtype":"Float64"}]
    results=[{"source_record_id":i,"source_field_id":3,"analysis_run_id":i+20,"repeat_number":2,"group_name":"Elem",
              "value_float":120.,"value_int":None,"value_text":None} for i in (1,2)]
    project=project_from_records(records,results,fields,source={"read_only":True})
    assert len(project.rows)==2 and project.rows[0].id!=project.rows[1].id
    assert project.rows[1].source["analysis_runs"]["Elem"]=={"analysis_run_id":22,"repeat_number":2}
    assert project.rows[0].original==project.rows[0].values
    with pytest.raises(ValueError,match="несколько результатов"):
        project_from_records(records,results+[results[0]],fields,source={})
    assert profile_fields({"fields":fields},"atomic_ratios")==fields
    assert profile_fields({"fields":fields},"sofer")==[]


def test_qt_calculation_protection_edit_undo_reload_and_plot(qtbot,tmp_path):
    from PySide6.QtCore import Qt
    from autopyro.window import MainWindow
    from autopyro.nonpyro_dialog import NonPyroDialog
    from autopyro.database.ui.nonpyro import NonPyroDatabaseDialog
    project,config,cols=example("sofer")
    # Saved manual bindings are restored, applicability must be reconfirmed.
    project.view["nonpyro_configs"]={"sofer":config}
    window=MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    dialog=NonPyroDialog(window)
    qtbot.addWidget(dialog)
    assert not dialog.form.values()["confirmed"]
    dialog.form.load({"confirmed":True})
    dialog.calculate()
    assert dialog.report is not None,dialog.status.text()
    dialog.promote()
    output=next(c for c in project.columns if c.metadata.get("nonpyro",{}).get("output")=="CV")
    index=project.columns.index(output)
    assert not window.model.flags(window.model.index(0,index)) & Qt.ItemFlag.ItemIsEditable
    assert not window.model.setData(window.model.index(0,index),"100")
    assert window.model.setData(window.model.index(0,0),"-30")
    assert project.rows[0].values[output.id]==pytest.approx(4.31)
    assert dialog.report is None
    window.model.undo.undo()
    assert project.rows[0].values[output.id]==pytest.approx(-.75)
    path=tmp_path/"nonpyro.apyro"
    project.save(path)
    loaded=Project.load(path)
    assert loaded.rows[0].values[output.id]==pytest.approx(-.75)
    dialog.calculate()
    dialog.plot()
    assert window.plot_window.result is not None,dialog.status.text()
    assert window.plot_window.spec.interpretation["scheme"]["source"]
    db=NonPyroDatabaseDialog("atomic_ratios",window)
    qtbot.addWidget(db)
    assert db.method.currentData()=="atomic_ratios"
    assert not db.use.isEnabled()
    window.dirty=False


def test_profile_rebinds_variant_and_retains_previous_runs(qtbot, monkeypatch):
    from autopyro.ui.window import MainWindow
    method = 'c27_diasterane_ratio'
    columns = [Column(f'Sat-GCMS / {peak}[{tag}]', 'number', unit='count')
               for tag in ('h', 'a') for peak in ('St27dbS', 'St27aaR')]
    project = Project(columns=columns, rows=[Row({column.id: value
                                                 for column, value in zip(columns, (12, 4, 200, 40), strict=True)})])
    window = MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.show_nonpyro('steranes')
    dialog = window.geochemistry_windows['steranes']
    dialog.method.setCurrentIndex(dialog.method.findData(method))
    dialog.form.fields['context'][0].setText('fixture')
    dialog.form.fields['confirmed'][0].setChecked(True)
    dialog.calculate()
    assert dialog.report['results'][0]['values']['dia27_sr'] == 3
    dialog.variant.setCurrentIndex(dialog.variant.findData('a'))
    assert all('[a]' in binding[0].currentText() for binding in dialog.roles.values())
    dialog.calculate()
    assert [run['results'][0]['values']['dia27_sr'] for run in dialog.history_reports()] == [3, 5]
    dialog.method.setCurrentIndex(0)
    dialog.method.setCurrentIndex(dialog.method.findData(method))
    assert dialog.variant.currentData() == 'a'
    assert all('[a]' in binding[0].currentText() for binding in dialog.roles.values())
    monkeypatch.setattr(QInputDialog, 'getItem', lambda parent, title, label, choices, *args: (choices[1], True))
    archived_values = []

    def inspect_archive(archive):
        model = archive.findChild(QTableView).model()
        archived_values.append(model.index(0, 2).data())

    monkeypatch.setattr(QDialog, 'exec', inspect_archive)
    dialog.history_button.click()
    assert archived_values == ['3']
    window.dirty = False


def test_embedded_profile_preserves_bindings_and_blocks_stale_promotion(qtbot):
    from autopyro.window import MainWindow
    project, config, cols = example("sofer")
    project.view["nonpyro_configs"] = {"sofer": config}
    window = MainWindow()
    qtbot.addWidget(window)
    window.may_replace = lambda: True
    window.set_project(project)
    window.show_nonpyro()
    dialog = window.nonpyro_window
    assert window.desktop.properties_stack.currentWidget() is dialog.parameter_tabs
    for key, column in cols.items():
        assert dialog.roles[key][0].currentData() == column.id
        assert dialog.parameter_tabs.isAncestorOf(dialog.roles[key][0])
    dialog.form.load({"confirmed": True})
    dialog.calculate()
    assert dialog.report is not None, dialog.status.text()
    count = window.model.columnCount()
    window.apply_filter([], "AND")
    assert dialog.report is not None
    window.model.setData(window.model.index(0, 0), -29)
    dialog.promote()
    assert dialog.report is None and window.model.columnCount() == count
    dialog.calculate()
    dialog.promote()
    assert window.model.columnCount() > count
    window.model.undo.undo()
    assert window.model.columnCount() == count
    window.model.undo.redo()
    dialog.calculate()
    dialog.plot()
    assert window.desktop.documents.indexOf(dialog) > 0
    assert window.desktop.documents.currentWidget() is window.plot_window
    assert window.plot_window.result is not None, dialog.status.text()
    window.plot_window.timer.stop()
    window.dirty = False


def test_embedded_database_snapshot_replaces_document_safely(qtbot, monkeypatch):
    from types import SimpleNamespace
    from PySide6.QtWidgets import QDialog
    from autopyro.window import MainWindow
    from autopyro.database.ui import nonpyro
    project, config, cols = example("atomic_ratios")
    window = MainWindow()
    qtbot.addWidget(window)
    window.may_replace = lambda: True
    window.show_nonpyro()
    previous = window.nonpyro_window
    previous.method.setCurrentIndex(previous.method.findData("atomic_ratios"))
    monkeypatch.setattr(nonpyro, "NonPyroDatabaseDialog", lambda **kwargs: SimpleNamespace(
        exec=lambda: QDialog.DialogCode.Accepted, project=project, deleteLater=lambda: None))
    project.source['method_profile'] = 'atomic_ratios'
    window.import_geochemistry()
    assert window.model.project is project
    assert window.nonpyro_window is None
    assert window.desktop.documents.currentWidget() is window.desktop.table_page
    window.show_nonpyro()
    assert window.nonpyro_window is not previous
    assert window.nonpyro_window.method.currentData() == "atomic_ratios"
    assert window.desktop.documents.currentWidget() is window.nonpyro_window
    assert window.desktop.properties_stack.currentWidget() is window.nonpyro_window.parameter_tabs
    window.dirty = False


def test_qt_all_scope_plot_ignores_current_filter_and_undo_macro(qtbot):
    from autopyro.window import MainWindow
    from autopyro.nonpyro_dialog import NonPyroDialog
    project,config,cols=example("sofer")
    row=deepcopy(project.rows[0])
    row.id="second-record"
    row.values[cols["sat"].id]=-30
    project.rows.append(row)
    project.view["nonpyro_configs"]={"sofer":config}
    window=MainWindow()
    qtbot.addWidget(window)
    window.set_project(project)
    window.apply_filter([{"column":cols["sat"].id,"operator":"eq","value":-28}],"AND")
    dialog=NonPyroDialog(window)
    qtbot.addWidget(dialog)
    dialog.form.load({"scope":"all","confirmed":True})
    dialog.calculate()
    assert len(dialog.report["results"])==2
    count=len(project.columns)
    dialog.plot()
    assert window.plot_window.spec.scope=="all"
    assert sum(window.plot_window.result["plotted"].values())==2
    window.model.undo.undo()
    assert len(project.columns)==count
    window.model.undo.redo()
    assert len(project.columns)>count
    window.dirty=False
