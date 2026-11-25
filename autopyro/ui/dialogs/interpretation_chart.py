"""Embed the interpretation plot and manage its presentation in the worksheet."""

from __future__ import annotations

from PySide6.QtCore import QByteArray, Qt

from ..i18n import tr, settings


SPLITTER_SETTING = 'workspace/interpretationResultSplitter'


def create_plot(sheet):
    from .plot import PlotDialog

    sheet.plot = PlotDialog(sheet.owner, interpretation_owner=sheet)
    sheet.comparison_view.currentIndexChanged.connect(sheet.plot.request_navigation)
    sheet.plot.setWindowFlags(Qt.WindowType.Widget)
    sheet.plot.setParent(sheet)
    # Reuse the graph's parameter pages in the analysis properties panel.
    # The chart and report stay visible together, with a draggable divider.
    while sheet.plot.tabs.count():
        page = sheet.plot.tabs.widget(0)
        title = sheet.plot.tabs.tabText(0)
        image = sheet.plot.tabs.tabIcon(0)
        sheet.plot.tabs.removeTab(0)
        if hasattr(page, 'widget') and page.widget() is sheet.plot.z_form:
            page.hide()
            continue
        sheet.parameter_tabs.addTab(page, image, title)
    sheet.plot.tabs.hide()
    sheet.plot.canvas_scroll.setMinimumHeight(180)
    sheet.result_splitter.insertWidget(0, sheet.plot)
    sheet.result_splitter.setStretchFactor(0, 3)
    sheet.result_splitter.setStretchFactor(1, 1)
    sheet.result_splitter.setStretchFactor(2, 1)
    sheet.result_splitter.setSizes([620, 220, 160])
    sheet.result_splitter.restoreState(settings().value(SPLITTER_SETTING, QByteArray(), type=QByteArray))
    sheet.result_splitter.splitterMoved.connect(sheet.save_splitter)
    saved_plot = sheet.owner.model.project.view.get('interpretation_plot_current', {})
    if sheet.scheme:
        if not saved_plot:
            sheet.configure_plot_axes()
        elif ('group' in sheet.scheme and 'interpretation_source_frame' not in saved_plot
              and all(saved_plot.get(f'{key}_axis', {}).get(limit) == sheet.scheme[key].get(limit)
                      for key in ('x', 'y') for limit in ('minimum', 'maximum'))):
            sheet.configure_plot_axes()


def configure_plot_axes(sheet):
    for form, key in ((sheet.plot.x_form, 'x'), (sheet.plot.y_form, 'y')):
        axis = sheet.scheme[key]
        form.load({'label': f"{axis['property']} [{axis['unit']}]",
                   'scale': 'log' if axis.get('transform') == 'log10' else 'linear',
                   'minimum': None, 'maximum': None})


def save_splitter(sheet, *_):
    settings().setValue(SPLITTER_SETTING, sheet.result_splitter.saveState())


def update_preview_context(sheet, axes):
    if not sheet.comparison_panels or sheet.report is None or not axes:
        sheet.comparison_context.hide()
        return
    names = ', '.join(axis.get_title() for axis in axes)
    sheet.comparison_context.setText(tr('Таблица · {0}    График · {1}', sheet.scheme['name'], names))
    sheet.comparison_context.show()


def show_on_plot(sheet):
    """Compatibility entry point; visualization stays in this document."""
    sheet.update_plot()


def on_activated(sheet):
    if sheet.plot.closed:
        sheet.plot.closed = False
        sheet.plot.preview.warmup()
        sheet.update_plot()
    else:
        sheet.plot.on_activated()


def update_plot(sheet):
    if not hasattr(sheet, 'plot'):
        return
    sheet.plot.apply_interpretation_binding()
    sheet.plot.schedule()
