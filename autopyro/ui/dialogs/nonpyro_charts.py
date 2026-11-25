"""Display profiles, ternary diagrams and project plots."""

from __future__ import annotations

from math import sqrt

from autopyro.analysis.nonpyro import (
    METHODS,
    alkane_distribution,
    sara_ternary_points,
    sterane_ternary_points,
    sofer_scheme,
)
from autopyro.visualization.plotting import PlotSpec, AxisSpec

from ..i18n import tr, message


def show_component_profiles(editor):
    if not editor.current_report():
        return
    method = editor.report['config']['method']
    if method not in ('sara', 'gas_isotopes', 'homohopane_distribution'):
        return
    sara = method == 'sara'
    homohopanes = method == 'homohopane_distribution'
    if sara:
        button, canvas_attr, title = editor.sara_profiles_button, 'sara_profiles_canvas', tr('Профили SARA')
        empty = tr('В выбранных строках нет полного ненулевого состава SARA.')
    elif homohopanes:
        button, canvas_attr, title = editor.homohopane_button, 'homohopane_canvas', tr('Профили C31–C35')
        empty = tr('В выбранных строках нет полного ряда C31–C35 с положительной суммой.')
    else:
        button, canvas_attr, title = editor.chung_button, 'chung_canvas', tr('Диаграмма Chung')
        empty = tr('В выбранных строках нет полного набора значений δ¹³C для C1–C3.')
    rows = sorted(index.row() for index in editor.results.selectionModel().selectedRows())
    canvas = getattr(editor, canvas_attr)
    if not 1 <= len(rows) <= 8:
        editor.status.setText(button.toolTip())
        return
    results = editor.report['results']
    status = 'profile' if homohopanes else 'interpreted'
    included = [row for row in rows if results[row]['status'] == status]
    if not included:
        if homohopanes and canvas is not None:
            editor.result_tabs.setTabVisible(editor.result_tabs.indexOf(canvas), False)
        editor.status.setText(empty)
        return
    if canvas is None:
        from matplotlib.figure import Figure
        from ..widgets.figure_canvas import FittedFigureCanvas
        canvas = FittedFigureCanvas(Figure(layout='constrained'))
        setattr(editor, canvas_attr, canvas)
        editor.result_tabs.addTab(canvas, title)
    figure = canvas.figure
    figure.clear()
    axes = figure.add_subplot()
    if sara:
        roles = METHODS['sara']['roles']
        positions = range(len(roles))
        ticks = [tr(role['label']) for role in roles.values()]
        axes.set_ylim(0, 100)
        axes.set_ylabel(tr('Массовая доля, %'))
    elif homohopanes:
        roles = tuple(editor.report['definition']['outputs'])
        positions = range(31, 36)
        ticks = roles
        axes.set_ylim(bottom=0)
        axes.set_ylabel(tr('Доля в сумме C31–C35, %'))
    else:
        roles = ('d3', 'd2', 'd1')
        positions = (1 / 3, 1 / 2, 1)
        ticks = ('1/3 (C3)', '1/2 (C2)', '1 (C1)')
        axes.set_xlabel(tr('1/n, где n — число атомов углерода'))
        axes.set_ylabel(tr('δ¹³C, ‰ VPDB'))
    for row in included:
        values = results[row]['values' if homohopanes else 'canonical_inputs']
        name = str(editor.results_model.rows[row][1])
        axes.plot(positions, [values[key] for key in roles], marker='o',
                  label=f'{row + 1}. {name}')
    axes.set_xticks(positions, ticks)
    axes.grid(axis='y', color='0.85')
    legend = axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncols=2)
    for label in legend.get_texts():
        label.set_parse_math(False)
    editor.result_tabs.setTabVisible(editor.result_tabs.indexOf(canvas), True)
    editor.result_tabs.setCurrentWidget(canvas)
    canvas.draw_idle()
    counts = len(included), len(rows)
    editor.status.setText(
        tr('Показано {0} из {1} выбранных строк. Массовые доли показаны до нормировки, проверка баланса доступна в результатах.', *counts)
        if sara else tr('Показано {0} из {1} выбранных строк. Линии соединяют измерения одной пробы.', *counts))


def show_ternary(editor):
    if not editor.current_report():
        return
    method = editor.report['config']['method']
    if method not in ('sara', 'sterane_distribution', 'rallakis_steranes'):
        return
    steranes = method in ('sterane_distribution', 'rallakis_steranes')
    title = tr('Треугольная диаграмма стеранов') if steranes else tr('Треугольная диаграмма SARA')
    points = (sterane_ternary_points if steranes else sara_ternary_points)(editor.report)
    if not points:
        editor.status.setText(tr('Нет проб с тремя допустимыми пиками и положительной суммой.') if steranes
                            else tr('Нет проб с полным составом SARA и балансом в пределах допуска.'))
        return
    if editor.sara_canvas is None:
        from matplotlib.figure import Figure
        from ..widgets.figure_canvas import FittedFigureCanvas
        editor.sara_canvas = FittedFigureCanvas(Figure(layout='constrained'))
        editor.result_tabs.addTab(editor.sara_canvas, title)
        editor.sara_canvas.mpl_connect('pick_event', editor.select_sara_point)
    figure = editor.sara_canvas.figure
    figure.clear()
    axes = figure.add_subplot()
    height = sqrt(3) / 2
    for fraction in (0.2, 0.4, 0.6, 0.8):
        inset = fraction / 2
        axes.plot([inset, 1 - inset], [fraction * height] * 2, color='0.8')
        axes.plot([fraction, 0.5 + inset], [0, (1 - fraction) * height], color='0.8')
        axes.plot([1 - fraction, 0.5 - inset], [0, (1 - fraction) * height], color='0.8')
    axes.plot([0, 1, 0.5, 0], [0, 0, height, 0], color='0.3')
    rows, xs, ys = zip(*points)
    scatter = axes.scatter(xs, ys, zorder=3, picker=True)
    scatter._autopyro_result_rows = rows
    if steranes:
        isomer = 'ββ 20S' if method == 'rallakis_steranes' else 'ααα 20R'
        left, right, top = (f'C{carbon} {isomer}' for carbon in (27, 28, 29))
    else:
        left, right, top = tr('Насыщенные'), tr('Ароматические'), tr('Смолы + асфальтены')
    axes.text(0, -0.035, left, ha='center', va='top')
    axes.text(1, -0.035, right, ha='center', va='top')
    axes.text(0.5, height + 0.035, top, ha='center', va='bottom')
    axes.set(xlim=(-0.16, 1.16), ylim=(-0.11, height + 0.11), aspect='equal')
    axes.set_axis_off()
    tab = editor.result_tabs.indexOf(editor.sara_canvas)
    editor.result_tabs.setTabText(tab, title)
    editor.result_tabs.setTabVisible(tab, True)
    editor.result_tabs.setCurrentWidget(editor.sara_canvas)
    editor.sara_canvas.draw_idle()
    editor.status.setText(tr('На диаграмме {0} из {1} проб. Доли приведены к 100%, шаг сетки 20%. Нажмите точку для перехода к результату.',
                           len(points), len(editor.report['results'])))


def select_sara_point(editor, event):
    if len(event.ind) and editor.current_report():
        editor.results.selectRow(event.artist._autopyro_result_rows[event.ind[0]])
        editor.result_tabs.setCurrentWidget(editor.results)


def selected_alkane_distribution(editor):
    rows = editor.results.selectionModel().selectedRows()
    if editor.report is None or len(rows) != 1 or editor.report['config']['method'] != 'alkane_indices':
        return []
    return alkane_distribution(editor.report['results'][rows[0].row()])


def show_alkane_distribution(editor):
    if not editor.current_report():
        return
    distribution = editor.selected_alkane_distribution()
    if not distribution:
        return
    if editor.alkane_canvas is None:
        from matplotlib.figure import Figure
        from ..widgets.figure_canvas import FittedFigureCanvas
        editor.alkane_canvas = FittedFigureCanvas(Figure(layout='constrained'))
        editor.result_tabs.addTab(editor.alkane_canvas, tr('Профиль н-алканов'))
    figure = editor.alkane_canvas.figure
    figure.clear()
    axes = figure.add_subplot()
    carbons, percentages = zip(*distribution)
    axes.bar(carbons, percentages)
    axes.set_xticks(carbons, [f'nC{carbon}' for carbon in carbons])
    axes.set_ylabel(tr('Доля сигнала C24–C34, %'))
    row = editor.results.selectionModel().selectedRows()[0].row()
    axes.set_title(str(editor.results_model.rows[row][1]), parse_math=False)
    editor.result_tabs.setTabVisible(editor.result_tabs.indexOf(editor.alkane_canvas), True)
    editor.result_tabs.setCurrentWidget(editor.alkane_canvas)
    editor.alkane_canvas.draw_idle()


def plot(editor):
    if not editor.current_report():
        editor.status.setText(tr('Сначала выполните актуальный расчёт.'))
        return
    try:
        report = editor.report
        editor.promote()
        config = report["config"]
        definition = METHODS[config["method"]]
        outputs = {
            c.metadata["nonpyro"]["output"]: c.id
            for c in editor.owner.model.project.columns
            if c.metadata.get("nonpyro", {}).get("config") == config
        }
        axes = definition["plot"]
        if axes:
            ids = [
                outputs[key]
                if key in outputs
                else config["inputs"][key]["column_id"]
                for key in axes
            ]
            spec = PlotSpec(
                x_id=ids[0],
                y_id=ids[1],
                title=definition["name"],
                scope=editor.owner.analysis_scope.mode,
                text_numbers=config.get("text_numbers", False),
            )
            if config["method"] == "sofer":
                scheme = sofer_scheme(
                    editor.owner.model.project, ids[0], ids[1], report["row_ids"]
                )
                spec.interpretation = {
                    "scheme": scheme,
                    "x_id": ids[0],
                    "y_id": ids[1],
                    "x_unit": "per mil",
                    "y_unit": "per mil",
                    "text_numbers": False,
                }
                spec.x_axis.label = tr('δ¹³C насыщенной фракции, ‰ VPDB')
                spec.y_axis.label = tr('δ¹³C ароматической фракции, ‰ VPDB')
                spec.interpretation_labels = False
                spec.annotation = tr('C15+ фракции нефти; VPDB.\nCV>0.47: терригенный; CV<0.47: морской тип.\nSofer (1984): статистическая гипотеза.')
        else:
            cid = next(outputs[key] for key in definition["outputs"])
            depths = [
                c.id
                for c in editor.owner.model.project.columns
                if c.name == "Sample / Top MD [m]"
            ]
            spec = PlotSpec(
                kind="depth" if depths else "hist",
                x_id=cid,
                y_id=depths[0] if depths else "",
                title=definition["name"],
                scope=editor.owner.analysis_scope.mode,
                y_axis=AxisSpec(inverted=bool(depths)),
            )
        editor.owner.show_plots()
        dialog = editor.owner.plot_window
        dialog.spec = spec
        dialog.load_controls()
        dialog.redraw()
        if dialog.result is None:
            raise ValueError(
                tr('Проверьте параметры в окне графиков: отрисовка не завершена')
            )
        if not editor.property("workspaceTracked"):
            editor.accept()
    except Exception as exc:
        editor.status.setText(tr('График не построен: {0}', message(exc)))
