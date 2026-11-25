"""Registration of analysis worksheets shared by menus and the project explorer."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from importlib import import_module

from .geochemistry import modules
from .i18n import tr


@dataclass(frozen=True)
class AnalysisDocument:
    kind: str
    dialog: str
    title: Callable[[], str]
    action_title: Callable[[], str]
    icon: str
    window_method: str
    view_keys: tuple[str, ...] = ()
    shortcut: str = ''
    profile: str = ''
    methods: tuple[str, ...] = ()

    @property
    def slot(self):
        return f'{self.kind}_window'

    @property
    def payload(self):
        return self.kind, self.profile

    def command(self, window):
        method = getattr(window, self.window_method)
        return partial(method, self.profile) if self.profile else method

    def dialog_class(self):
        module, name = self.dialog.rsplit('.', 1)
        return getattr(import_module(f'.dialogs.{module}', __package__), name)


DOCUMENTS = {document.kind: document for document in (
    AnalysisDocument('plot', 'plot.PlotDialog', lambda: tr('График'), lambda: tr('Графики…'),
                     'chart-no-axes-combined', 'show_plots', ('plot_current',), 'Ctrl+Shift+G'),
    AnalysisDocument('map', 'map.MapDialog', lambda: tr('Карта'), lambda: tr('Карты…'),
                     'map', 'show_maps', ('map_current', 'map_layers')),
    AnalysisDocument('statistics', 'statistics.StatisticsDialog', lambda: tr('Статистика'), lambda: tr('Статистика…'),
                     'chart-column', 'show_statistics', ('statistics',), 'Ctrl+Shift+T'),
    AnalysisDocument('curve_fitting', 'curve_fitting.CurveFittingDialog',
                     lambda: tr('Подгонка кривых'), lambda: tr('Подгонка кривых…'),
                     'chart-no-axes-combined', 'show_curve_fitting', ('curve_fitting',)),
    AnalysisDocument('interpretation', 'interpretation.InterpretationDialog',
                     lambda: tr('Интерпретация'), lambda: tr('Интерпретация…'),
                     'tags', 'show_interpretation', ('interpretation_current',)),
    AnalysisDocument('interpolation', 'interpolation.InterpolationDialog',
                     lambda: tr('Интерполяция'), lambda: tr('Интерполяция…'),
                     'spline', 'interpolate', ('interpolation_current',)),
    AnalysisDocument('calculation', 'calculation.CalculationDialog',
                     lambda: tr('Расчёты пиролиза'), lambda: tr('Расчёты пиролиза…'),
                     'calculator', 'show_calculations', ('calculation_current',)),
    AnalysisDocument('stratigraphy', 'stratigraphy.StratigraphyDialog',
                     lambda: tr('Стратиграфия'), lambda: tr('Стратиграфия…'),
                     'layers', 'show_stratigraphy', ('stratigraphy',)),
    AnalysisDocument('nonpyro', 'nonpyro.NonPyroDialog', lambda: tr('Геохимия'), lambda: tr('Все методы'),
                     'flask-conical', 'show_nonpyro', ('legacy_nonpyro_current',)),
    AnalysisDocument('qc', 'qc.QCDialog', lambda: tr('Контроль качества'), lambda: tr('Контроль качества…'),
                     'shield-check', 'show_qc', ('qc_decisions',)),
    AnalysisDocument('geochemistry_workspace', 'geochemistry_workspace.GeochemistryWorkspaceDialog',
                     lambda: tr('Паспорта и сравнение проб'), lambda: tr('Паспорта и сравнение проб…'),
                     'files', 'show_geochemistry_workspace', ('geochemistry_workspace', 'geochemistry_comparison')),
)}


def _geochemistry_document(profile, title, methods):
    return replace(DOCUMENTS['nonpyro'], kind='geochemistry', profile=profile, methods=methods,
                   title=lambda: title, action_title=lambda: title + '…', view_keys=())


def geochemistry_documents():
    return tuple(_geochemistry_document(profile, title, methods) for profile, (title, methods) in modules().items())


def document_descriptor(kind, profile=''):
    if kind == 'geochemistry':
        module = modules().get(profile)
        return _geochemistry_document(profile, *module) if module else None
    return DOCUMENTS.get(kind)
