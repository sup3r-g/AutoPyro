"""Synchronize PlotSpec, plot controls, appearance and editing history."""

from __future__ import annotations

from ..i18n import tr
import json
from copy import deepcopy
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtGui import QUndoCommand, QUndoStack
from PySide6.QtWidgets import QColorDialog, QListWidgetItem, QTableWidgetItem
from autopyro.visualization.plot_data import LIBRARY_MEASUREMENT_ERROR_KINDS
from autopyro.visualization.plot_catalog import marker_value, MERGED_KINDS, PAIRED_KINDS
from autopyro.visualization.plot_controls import (
    bindings_from_axes, controls_for, migrate_bindings, option_is_editable, options_object,
)
from autopyro.visualization.plot_inputs import input_controls_for, requires_regular_grid
from autopyro.visualization.plotting import AxisSpec, PlotSpec, category_label
from ..widgets.parameter_reset import parameter_field


SHARED_APPEARANCE = 'plot_shared_appearance'
LOCAL_LAYOUT_FIELDS = {'title', 'subtitle', 'annotation', 'data_aspect', 'comparison_grid',
                       'interpretation_source_frame', 'stratigraphy_alpha', 'stratigraphy_labels',
                       'stratigraphy_info', 'stratigraphy_label_size'}
LOCAL_AXIS_FIELDS = {'label', 'minimum', 'maximum', 'scale', 'log_base', 'inverted', 'major_step'}


class PlotConfigCommand(QUndoCommand):
    def __init__(self, dialog, before, after):
        super().__init__(tr('Оформление графика'))
        self.dialog, self.before, self.after = dialog, before, after
        self.first = True

    def apply(self, config):
        self.dialog.replaying = True
        self.dialog.spec = PlotSpec.from_dict(config)
        self.dialog.load_controls()
        self.dialog.schedule()

    def redo(self):
        if self.first:
            self.first = False
        else:
            self.apply(self.after)

    def undo(self):
        self.apply(self.before)


def install(dialog):
    dialog.undo_stack = QUndoStack(dialog)
    dialog.last_config = None
    dialog.shared_values = {}
    dialog.shared_palette = None
    dialog.shared_refresh_pending = False
    dialog.replaying = False
    dialog.controls_kind = None
    dialog.kind_drafts = {}


def refresh_scopes(dialog):
    scope = dialog.scope.currentData()
    choices = [(tr('Общая область анализа'), 'global'), *dialog.owner.analysis_scope.choices()]
    with QSignalBlocker(dialog.scope):
        dialog.scope.clear()
        for label, value in choices:
            dialog.scope.addItem(label, value)
        index = dialog.scope.findData(scope)
        if index < 0:
            index = dialog.scope.findData(dialog.spec.scope)
        if index < 0 and dialog.spec.scope:
            dialog.scope.addItem(tr('Недоступная область: {0}', dialog.spec.scope), dialog.spec.scope)
            index = dialog.scope.count() - 1
        dialog.scope.setCurrentIndex(max(0, index))
    if dialog.scope.currentData() != scope or str(scope).startswith(('view:', 'set:')):
        dialog.schedule()


def rename_scope(dialog, old, new):
    index = dialog.scope.findData(old)
    if index >= 0:
        dialog.scope.setItemData(index, new)


def analysis_scope_changed(dialog):
    if dialog.scope.currentData() == 'global':
        dialog.schedule()


def table_scope_changed(dialog):
    scope = dialog.scope.currentData()
    if scope == 'global':
        scope = dialog.owner.analysis_scope.mode
    if scope in ('filtered', 'selected'):
        dialog.schedule()


def restore_parameters(dialog):
    dialog.timer.stop()
    previous = dialog.spec
    dialog.spec = PlotSpec(
        kind=dialog.selected_kind(), scope=dialog.scope.currentData(),
        x_id=dialog.x.currentData() or "", y_id=dialog.y.currentData() or "",
        z_id=dialog.z.currentData() or "",
        depth_bottom_id=dialog.depth_bottom.currentData() or "",
        track_ids=dialog.selected_tracks(),
        text_numbers=dialog.data.fields['text_numbers'][0].isChecked(),
        include_interpolated=dialog.data.fields['include_interpolated'][0].isChecked(),
        bins=dialog.data.fields['bins'][0].value(),
        density=dialog.data.fields['density'][0].isChecked(),
        curve_archive_id=previous.curve_archive_id,
        passport_id=previous.passport_id,
        interpretation=deepcopy(previous.interpretation),
        stratigraphy=deepcopy(previous.stratigraphy),
    )
    if ':' in dialog.spec.kind:
        primary = {argument for _, argument in controls_for(dialog.spec.kind).axes}
        extra = {key: dialog.library_inputs[key].currentData() for key in controls_for(dialog.spec.kind).parameters
                 if key in dialog.library_inputs and key not in primary}
        dialog.spec.library_bindings = bindings_from_axes(dialog.spec, extra)
        dialog.kind_drafts.pop(dialog.spec.kind, None)
    dialog.brush.setChecked(False)
    dialog.template.setCurrentIndex(0)
    dialog.load_controls()
    dialog.category_table.setRowCount(0)
    dialog.update_enabled()
    dialog.schedule()


def sync_columns(dialog, *args):
    columns = dialog.owner.model.project.columns
    for combo, optional in [
        (dialog.x, True),
        (dialog.y, True),
        (dialog.z, True),
        (dialog.depth_bottom, True),
        (dialog.color_by, True),
        (dialog.shape_by, True),
        (dialog.size_by, True),
        (dialog.label_by, True),
        (dialog.line_group_by, True),
        *((combo, True) for combo in dialog.library_inputs.values()),
    ]:
        selected = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        if optional:
            combo.addItem(tr('Не задано'), "")
        for column in columns:
            combo.addItem(column.name, column.id)
        if selected and combo.findData(selected) < 0:
            combo.addItem(tr('Удалённая колонка — выберите замену'), selected)
        combo.setCurrentIndex(max(0, combo.findData(selected)))
        combo.blockSignals(False)
    checked = set(dialog.spec.track_ids)
    dialog.tracks.blockSignals(True)
    dialog.tracks.clear()
    for column in columns:
        item = QListWidgetItem(column.name)
        item.setData(Qt.ItemDataRole.UserRole, column.id)
        item.setCheckState(
            Qt.CheckState.Checked
            if column.id in checked
            else Qt.CheckState.Unchecked
        )
        dialog.tracks.addItem(item)
    for cid in checked - {c.id for c in columns}:
        item = QListWidgetItem(
            tr('Удалённый трек — снимите отметку или выберите замену')
        )
        item.setData(Qt.ItemDataRole.UserRole, cid)
        item.setCheckState(Qt.CheckState.Checked)
        dialog.tracks.addItem(item)
    dialog.tracks.blockSignals(False)
    if not dialog.loading:
        dialog.schedule()


def selected_tracks(dialog):
    return [
        dialog.tracks.item(i).data(Qt.ItemDataRole.UserRole)
        for i in range(dialog.tracks.count())
        if dialog.tracks.item(i).checkState() == Qt.CheckState.Checked
    ]


def tracks_changed(dialog, *args):
    dialog.schedule()


def show_field(widget, visible):
    field = parameter_field(widget)
    field.parentWidget().layout().setRowVisible(field, visible)


def selected_kind(dialog):
    kind = dialog.kind.currentData()
    if kind in PAIRED_KINDS and dialog.variant_family == kind:
        return dialog.variant.currentData() or kind
    if dialog.plot_mode.currentData() and dialog.interpretation_owner is None:
        return {'scatter': 'mpl:scatter', 'hist': 'mpl:hist',
                'box': 'mpl:boxplot', 'violin': 'mpl:violinplot'}.get(kind, kind)
    return kind


def sync_variant(dialog, saved_kind=None):
    family = dialog.kind.currentData()
    if family != dialog.variant_family:
        dialog.variant.blockSignals(True)
        dialog.variant.clear()
        for label, kind in PAIRED_KINDS.get(family, ()):
            dialog.variant.addItem(tr(label), kind)
        dialog.variant_family = family
        dialog.variant.blockSignals(False)
    if saved_kind is not None:
        dialog.variant.blockSignals(True)
        dialog.variant.setCurrentIndex(max(0, dialog.variant.findData(saved_kind)))
        dialog.variant.blockSignals(False)
    dialog.show_field(dialog.variant, family in PAIRED_KINDS and dialog.interpretation_owner is None)


def load_kind_options(dialog, *_):
    kind = dialog.selected_kind()
    if ':' not in (kind or ''):
        return
    try:
        options = options_object(dialog.library_options.text())
    except ValueError:
        return  # Keep the draft visible; redraw reports the invalid JSON.
    values = {}
    for option in controls_for(kind).options:
        value = options.get(option.key, option.default)
        editable = option_is_editable(option, value)
        widget = dialog.kind_options.fields[option.key][0]
        widget.setEnabled(editable)
        widget.setToolTip(tr('Сохранён нестандартный параметр. Он доступен в экспертных настройках.') if not editable else '')
        values[option.key] = value if editable else option.default
        if not editable:
            dialog.expert_options.setChecked(True)
    dialog.kind_options.load(values)


def read_kind_options(dialog, kind):
    options = options_object(dialog.library_options.text())
    values = dialog.kind_options.values()
    for option in controls_for(kind).options:
        if dialog.kind_options.fields[option.key][0].isEnabled():
            options[option.key] = values[option.key]
    return options


def update_enabled(dialog):
    if dialog.interpretation_owner is not None and dialog.kind.currentData() != "scatter":
        dialog.kind.blockSignals(True)
        dialog.kind.setCurrentIndex(dialog.kind.findData("scatter"))
        dialog.kind.blockSignals(False)
    dialog.sync_variant()
    dialog.show_field(dialog.plot_mode, dialog.kind.currentData() in ('scatter', 'hist', 'box', 'violin')
                    and dialog.interpretation_owner is None)
    kind = dialog.selected_kind() or ''
    inputs = input_controls_for(kind)
    paired = inputs.paired
    contract = inputs.library
    library = contract is not None
    capabilities = contract.capabilities if library else frozenset()
    for key, combo in dialog.library_inputs.items():
        dialog.show_field(combo, key in inputs.extra_arguments)
    dialog.library_form.setVisible(bool(inputs.extra_arguments))
    option_keys = {option.key for option in contract.options} if library else set()
    for key, (widget, _) in dialog.kind_options.fields.items():
        dialog.show_field(widget, key in option_keys)
    dialog.kind_options.setVisible(bool(option_keys))
    dialog.expert_options.setVisible(library)
    dialog.expert_form.setVisible(library and dialog.expert_options.isChecked())
    dialog.kind_note.setText(tr('Экспериментальный вид: параметры и возможности могут быть ограничены.')
                           if library and contract.experimental else
                           tr('Данные и оформление задаются общими полями. Неприменимые параметры отключены.')
                           if library else '')
    dialog.kind_note.setVisible(library)
    if requires_regular_grid(kind):
        dialog.kind_note.setText(tr('Нужна полная регулярная сетка X–Y без повторов.'))
    dialog.z_form.setEnabled(kind.startswith('mpl3d:'))
    dialog.x.setEnabled('x' in inputs.axes)
    dialog.y.setEnabled('y' in inputs.axes)
    dialog.show_field(dialog.x, 'x' in inputs.axes)
    dialog.show_field(dialog.y, 'y' in inputs.axes)
    dialog.show_field(dialog.z, 'z' in inputs.axes)
    dialog.depth_bottom.setEnabled(inputs.depth)
    dialog.show_field(dialog.depth_bottom, inputs.depth)
    dialog.interval_note.setVisible(inputs.depth)
    dialog.tracks.setEnabled(inputs.tracks)
    dialog.show_field(dialog.tracks, inputs.tracks)
    parameter_field(dialog.tracks).parentWidget().setVisible(inputs.tracks)
    dialog.tracks_label.setText(tr('Показатели матрицы\n(порядок: перетаскивание)') if inputs.matrix
                              else tr('Треки планшета\n(порядок: перетаскивание)'))
    if inputs.matrix:
        dialog.kind_note.setText(tr('Выбранные показатели образуют колонки матрицы, пробы — строки. Это матрица исходных значений, не корреляций.'))
    y_field = parameter_field(dialog.y)
    y_field.parentWidget().layout().labelForField(y_field).setText(
        tr('Глубина (Y)') if inputs.depth else tr('Y / значение'))
    dialog.x_target.setEnabled(kind == 'multi_depth')
    dialog.x_form.setEnabled(True)
    dialog.y_form.setEnabled(True)
    command = kind.split(':', 1)[1] if library else ''
    for axis, form in (('x', dialog.x_form), ('y', dialog.y_form)):
        categorical = axis == 'y' and kind in ('box', 'violin')
        forced_log = command == 'loglog' or command == 'semilog' + axis
        form.fields['scale'][0].setEnabled(not forced_log and not categorical)
        if forced_log:
            form.load({'scale': 'log'})
        form.update_axis_scale()
        for key in ('minimum', 'maximum', 'minor_divisions'):
            form.fields[key][0].setEnabled(not categorical)
        if categorical:
            for key in ('log_base', 'major_step', 'number_format'):
                form.fields[key][0].setEnabled(False)
    for key in ('bins', 'density'):
        dialog.data.fields[key][0].setEnabled(inputs.histogram)
        dialog.show_field(dialog.data.fields[key][0], inputs.histogram)
    dialog.data.fields['connect'][0].setEnabled(paired)
    for key in ('line_group_by', 'connect_across_gaps'):
        widget = dialog.data.fields[key][0]
        enabled = paired and dialog.data.fields['connect'][0].isChecked()
        widget.setEnabled(enabled)
        dialog.show_field(widget, enabled)
    dialog.data.fields['line_width'][0].setEnabled(paired or 'line' in capabilities)
    dialog.label_by.setEnabled(paired or 'labels' in capabilities)
    for key in ('connect', 'line_width', 'label_by'):
        dialog.show_field(dialog.data.fields[key][0], dialog.data.fields[key][0].isEnabled())
    channel_fields = {'color_by', 'color_mode', 'shape_by', 'size_by', 'size_mode',
                      'size_min', 'size_max', 'size_value_min', 'size_value_max', 'missing_color'}
    style_capabilities = {'color': 'color', 'alpha': 'alpha', 'marker': 'marker',
                          'marker_size': 'marker', 'edge_color': 'edge', 'edge_width': 'edge',
                          'cmap': 'cmap', 'color_min': 'cmap', 'color_max': 'cmap',
                          'colorbar_label': 'cmap', 'colorbar_location': 'cmap'}
    for key, (widget, _) in dialog.style.fields.items():
        enabled = True
        if library:
            enabled = ('channels' in capabilities if key in channel_fields else
                       'channels' in capabilities or style_capabilities.get(key) in capabilities)
            if key == 'interval_width':
                enabled = False
        elif key in ('shape_by', 'size_by', 'size_mode', 'size_min', 'size_max',
                     'size_value_min', 'size_value_max'):
            enabled = paired
        widget.setEnabled(enabled)
    color_scale = dialog.style.fields['color_scale'][0]
    numeric_color = ((paired or 'channels' in capabilities)
                     and bool(dialog.style.fields['color_by'][0].currentData())
                     and dialog.style.fields['color_mode'][0].currentData() == 'numeric')
    dialog.show_field(color_scale, numeric_color)
    classes = numeric_color and color_scale.currentData() == 'classes'
    dialog.show_field(dialog.style.fields['color_class_edges'][0], classes)
    for key in ('color_min', 'color_max'):
        dialog.style.fields[key][0].setEnabled(dialog.style.fields[key][0].isEnabled() and not classes)
    dialog.brush.setEnabled(paired or kind == 'mpl:scatter')
    dialog.category_table.setEnabled(not library or 'channels' in capabilities)
    measurement_layers = (dialog.interpretation_owner is None and not dialog.spec.interpretation
                          and not dialog.spec.interpretation_panels)
    interval_samples = bool(dialog.depth_bottom.currentData())
    dialog.show_field(dialog.layout_form.fields['measurement_error_bars'][0],
                    measurement_layers and (paired or kind in LIBRARY_MEASUREMENT_ERROR_KINDS)
                    and (not interval_samples or inputs.depth))
    dialog.show_field(dialog.layout_form.fields['qc_outline'][0],
                    measurement_layers and (paired or kind == 'mpl:scatter') and not interval_samples)
    dialog.layout_form.fields['legend_visible_only'][0].setEnabled(paired or 'channels' in capabilities)
    for key in ('interpretation_alpha', 'interpretation_labels', 'interpretation_info',
                'stratigraphy_alpha', 'stratigraphy_labels', 'stratigraphy_info', 'stratigraphy_label_size'):
        dialog.layout_form.fields[key][0].setEnabled(not library)
    dialog.show_field(dialog.layout_form.fields['square_plot_area'][0], kind in ('scatter', 'mpl:scatter'))
    dialog.show_field(dialog.layout_form.fields['data_aspect'][0], kind in ('scatter', 'mpl:scatter'))
    dialog.show_field(dialog.layout_form.fields['legend_order'][0],
                    paired or kind == 'hist' or 'channels' in capabilities)
    for key in ('legend', 'legend_position', 'legend_columns', 'legend_size', 'legend_frame'):
        dialog.layout_form.fields[key][0].setEnabled(not library or 'line' in capabilities or 'channels' in capabilities)

    if dialog.interpretation_owner is not None:
        for key in ("kind", "x_id", "y_id", "scope", "text_numbers", "include_interpolated", "include_calculated", "include_original"):
            dialog.show_field(dialog.data.fields[key][0], False)
        dialog.show_field(dialog.x_target, False)
        dialog.show_field(dialog.style.fields["interval_width"][0], False)
        for key in ("stratigraphy_alpha", "stratigraphy_labels", "stratigraphy_info", "stratigraphy_label_size", "wspace"):
            dialog.show_field(dialog.layout_form.fields[key][0], False)

    comparison = dialog.interpretation_owner is not None and dialog.interpretation_owner.comparison.isChecked()
    dialog.show_field(dialog.layout_form.fields['comparison_grid'][0],
                    comparison or bool(dialog.spec.interpretation_panels))
    if dialog.interpretation_owner is not None:
        for form in (dialog.x_form, dialog.y_form):
            for key in ('label', 'minimum', 'maximum', 'scale', 'log_base'):
                form.fields[key][0].setEnabled(not comparison)


def kind_changed(dialog, *args):
    if dialog.loading:
        return
    dialog.sync_variant()
    kind = dialog.selected_kind()
    if dialog.controls_kind and ':' in dialog.controls_kind:
        try:
            dialog.kind_drafts[dialog.controls_kind] = dialog.read_kind_options(dialog.controls_kind)
        except ValueError:
            pass  # An invalid expert draft must not prevent choosing another view.
    previous = dialog.controls_kind
    family = MERGED_KINDS.get(kind, kind)
    if (previous != kind and family in PAIRED_KINDS
            and MERGED_KINDS.get(previous, previous) == family):
        options = dict(dialog.kind_drafts.get(previous, {}))
        if family in ('mpl:bar', 'mpl:fill_between'):
            # Rotate the same data and axis settings when changing orientation.
            dialog.data.load({'x_id': dialog.y.currentData(), 'y_id': dialog.x.currentData()})
            xaxis, yaxis = dialog.x_form.values(), dialog.y_form.values()
            dialog.x_form.load(yaxis)
            dialog.y_form.load(xaxis)
            names = ({'width': 'height', 'height': 'width', 'bottom': 'left', 'left': 'bottom',
                      'xerr': 'yerr', 'yerr': 'xerr'} if family == 'mpl:bar'
                     else {'y2': 'x2', 'x2': 'y2'})
            options = {names.get(key, key): value for key, value in options.items()}
        dialog.kind_drafts.setdefault(kind, options)
    dialog.controls_kind = kind
    dialog.library_options.setText(json.dumps(dialog.kind_drafts.get(kind, {})))
    dialog.expert_options.setChecked(False)
    dialog.load_kind_options()
    if kind in ('depth', 'multi_depth'):
        dialog.y_form.load({'inverted': True})
    else:
        dialog.depth_bottom.setCurrentIndex(0)
    dialog.update_enabled()
    dialog.schedule()


def update_x_targets(dialog):
    current = dialog.x_target.currentData()
    dialog.x_target.blockSignals(True)
    dialog.x_target.clear()
    dialog.x_target.addItem(tr('Общие настройки X'), "")
    columns = {c.id: c.name for c in dialog.owner.model.project.columns}
    for cid in dialog.spec.track_ids:
        dialog.x_target.addItem(columns.get(cid, tr('Удалённая колонка')), cid)
    dialog.x_target.setCurrentIndex(max(0, dialog.x_target.findData(current)))
    dialog.x_target.blockSignals(False)
    dialog.previous_x_target = dialog.x_target.currentData()


def switch_x_target(dialog):
    if dialog.loading:
        return
    try:
        axis = AxisSpec(**dialog.x_form.values())
    except ValueError:
        return
    if dialog.previous_x_target:
        dialog.spec.track_axes[dialog.previous_x_target] = axis
    else:
        dialog.spec.x_axis = axis
    current = dialog.x_target.currentData()
    dialog.x_form.load(vars(dialog.spec.track_axes.get(current, dialog.spec.x_axis)))
    dialog.previous_x_target = current
    dialog.schedule()


def apply_interpretation_binding(dialog):
    """The embedded chart must show exactly the inputs of the current report."""
    document = dialog.interpretation_owner
    if document is None:
        return
    dialog.spec.kind = "scatter"
    dialog.spec.interpretation_panels = deepcopy(document.comparison_panels) if document.report is not None else []
    dialog.spec.stratigraphy = None
    dialog.spec.depth_bottom_id = ""
    dialog.spec.include_interpolated = False
    dialog.spec.include_calculated = True
    dialog.spec.include_original = True
    dialog.spec.filter_rules = []
    dialog.spec.x_id = document.x.currentData() or ""
    dialog.spec.y_id = document.y.currentData() or ""
    dialog.spec.scope = document.scope.currentData()
    dialog.spec.text_numbers = document.text_numbers.isChecked()
    dialog.spec.interpretation = (
        {"scheme": deepcopy(document.scheme), **document.report["binding"]}
        if document.report is not None and not document.editor_dirty else None
    )
    if dialog.spec.interpretation_panels and dialog.spec.interpretation and 'group' in document.scheme:
        dialog.spec.interpretation_panels.insert(0, {
            'name': tr('Выбранная схема'), 'binding': dialog.spec.interpretation,
        })


def change_size_unit(dialog):
    scale = dialog.size_unit.currentData()
    for key in ("width_mm", "height_mm"):
        widget = dialog.layout_form.fields[key][0]
        mm = round(widget.value() * dialog.layout_form.number_scales[key], 2)
        with QSignalBlocker(widget):
            widget.setDecimals({1: 2, 10: 3, 25.4: 6}[scale])
            widget.setRange(30 / scale, 1000 / scale)
            widget.setSuffix(" " + dialog.size_unit.currentText())
            widget.setValue(mm / scale)
        dialog.layout_form.number_scales[key] = scale


def read_controls(dialog):
    from autopyro.model.filtering import leaf_rules

    dialog.sync_shared_appearance(publish=True)
    count = sum(1 for _ in leaf_rules(dialog.spec.filter_rules))
    dialog.local_filter_button.setText(tr('Локальный фильтр графика…') + (f" ({count})" if count else ""))
    for form in (dialog.data, dialog.style, dialog.layout_form):
        for key, value in form.values().items():
            setattr(dialog.spec, key, round(value, 2) if key in ("width_mm", "height_mm") else value)
    dialog.spec.kind = dialog.selected_kind()
    if ':' in dialog.spec.kind:
        contract = controls_for(dialog.spec.kind)
        axis_arguments = {argument for _, argument in contract.axes}
        extra = {key: dialog.library_inputs[key].currentData() for key in contract.parameters
                 if key in dialog.library_inputs and key not in axis_arguments}
        dialog.spec.library_bindings = bindings_from_axes(dialog.spec, extra)
        dialog.spec.library_options = json.dumps(dialog.read_kind_options(dialog.spec.kind))
    dialog.spec.annotation = dialog.spec.annotation.replace("\\n", "\n")
    dialog.spec.track_ids = dialog.selected_tracks()
    x = AxisSpec(**dialog.x_form.values())
    if dialog.previous_x_target:
        dialog.spec.track_axes[dialog.previous_x_target] = x
    else:
        dialog.spec.x_axis = x
    dialog.spec.y_axis = AxisSpec(**dialog.y_form.values())
    dialog.spec.z_axis = AxisSpec(**dialog.z_form.values())
    dialog.apply_interpretation_binding()
    dialog.update_x_targets()
    return dialog.spec


def sync_shared_appearance(dialog, *, publish=False):
    """Merge edits to shared controls, leaving scheme axes and graph content local."""
    view = dialog.owner.model.project.view
    stored = view.get(SHARED_APPEARANCE)
    legacy = view.get('plot_current') or view.get('interpretation_plot_current') or {}
    stored = stored if isinstance(stored, dict) else None
    legacy = legacy if isinstance(legacy, dict) else {}
    forms = (
        ('data', dialog.data, {'label_by'}), ('style', dialog.style, dialog.style.fields.keys() - {'interval_width'}),
        ('layout', dialog.layout_form, dialog.layout_form.fields.keys() - LOCAL_LAYOUT_FIELDS),
        ('x_axis', dialog.x_form, dialog.x_form.fields.keys() - LOCAL_AXIS_FIELDS),
        ('y_axis', dialog.y_form, dialog.y_form.fields.keys() - LOCAL_AXIS_FIELDS),
    )
    palette = view.get('plot_palettes')
    changed = isinstance(palette, dict) and palette is not dialog.shared_palette
    if changed:
        dialog.shared_palette = palette
        for channel, columns in deepcopy(palette).items():
            local = dialog.spec.style_maps.setdefault(channel, {})
            for column, categories in columns.items():
                local.setdefault(column, {}).update(categories)
    resolved = {}
    for section, form, keys in forms:
        current = {key: value for key, value in form.values().items() if key in keys}
        saved = (stored.get(section, {}) if stored is not None
                 else legacy.get(section, {}) if section.endswith('_axis') else legacy)
        saved = saved if isinstance(saved, dict) else {}
        previous = dialog.shared_values.get(section, current)
        # A local edit can be waiting for the next render when a peer publishes.
        updates = {key: saved[key] for key, value in current.items()
                   if key in saved and value == previous[key] and value != saved[key]}
        if updates:
            form.load(updates)
            if dialog.last_config is not None:
                config = dialog.last_config[section] if section.endswith('_axis') else dialog.last_config
                config.update(updates)
            changed = True
        resolved[section] = (current if publish else previous) | updates
    dialog.shared_values = resolved
    if publish and stored != resolved:
        view[SHARED_APPEARANCE] = resolved
        dialog.owner.view_changed()
        dialog.sync_peer_appearance()
    return changed


def sync_peer_appearance(dialog):
    other = (dialog.owner.plot_window if dialog.interpretation_owner is not None
             else getattr(dialog.owner.interpretation_window, 'plot', None))
    if other is not None and other.sync_shared_appearance():
        if other.isVisible():
            other.schedule()
        else:
            other.shared_refresh_pending = True


def on_activated(dialog):
    if dialog.sync_shared_appearance() or dialog.shared_refresh_pending:
        dialog.shared_refresh_pending = False
        dialog.schedule()


def remove_stratigraphy(dialog):
    dialog.spec.stratigraphy = None
    dialog.schedule()


def load_controls(dialog):
    dialog.apply_interpretation_binding()
    dialog.loading = True
    migrate_bindings(dialog.spec)
    data = dialog.spec.to_dict()
    data['kind'] = MERGED_KINDS.get(dialog.spec.kind, dialog.spec.kind)
    dialog.plot_mode.setCurrentIndex(int(dialog.spec.kind in ('mpl:scatter', 'mpl:hist', 'mpl:boxplot', 'mpl:violinplot')))
    for form in (dialog.data, dialog.style, dialog.layout_form):
        form.load(data)
    dialog.sync_variant(dialog.spec.kind)
    dialog.controls_kind = dialog.spec.kind
    dialog.library_form.load({**{key: '' for key in dialog.library_inputs}, **dialog.spec.library_bindings})
    dialog.library_options.setText(dialog.spec.library_options)
    dialog.load_kind_options()
    if ':' in dialog.spec.kind:
        known_options = {option.key for option in controls_for(dialog.spec.kind).options}
        try:
            unknown_options = set(options_object(dialog.spec.library_options)) - known_options
        except ValueError:
            unknown_options = True
        if unknown_options:
            dialog.expert_options.setChecked(True)
    dialog.x_target.setCurrentIndex(0)
    dialog.previous_x_target = ""
    dialog.x_form.load(vars(dialog.spec.x_axis))
    dialog.y_form.load(vars(dialog.spec.y_axis))
    dialog.z_form.load(vars(dialog.spec.z_axis))
    dialog.load_track_order()
    dialog.update_x_targets()
    dialog.loading = False
    dialog.update_enabled()


def load_track_order(dialog):
    # Restore both check state and ordering.
    ordered = dialog.spec.track_ids + [
        dialog.tracks.item(i).data(Qt.ItemDataRole.UserRole)
        for i in range(dialog.tracks.count())
        if dialog.tracks.item(i).data(Qt.ItemDataRole.UserRole)
        not in dialog.spec.track_ids
    ]
    labels = {c.id: c.name for c in dialog.owner.model.project.columns}
    dialog.tracks.blockSignals(True)
    dialog.tracks.clear()
    for cid in ordered:
        if cid not in labels:
            continue
        item = QListWidgetItem(labels[cid])
        item.setData(Qt.ItemDataRole.UserRole, cid)
        item.setCheckState(
            Qt.CheckState.Checked
            if cid in dialog.spec.track_ids
            else Qt.CheckState.Unchecked
        )
        dialog.tracks.addItem(item)
    dialog.tracks.blockSignals(False)


def edit_local_filter(dialog):
    from .filter import FilterDialog

    editor = FilterDialog(dialog.owner.model.project, deepcopy(dialog.spec.filter_rules),
                          dialog.spec.filter_combine, dialog)
    if editor.exec():
        dialog.spec.filter_rules = editor.rules
        dialog.spec.filter_combine = editor.mode.currentText()
        dialog.schedule()
    editor.deleteLater()


def scope_ids(dialog, spec=None, *, selected_ids=None):
    if dialog.interpretation_owner is not None and spec is None:
        return list(dialog.interpretation_owner.result_rows)
    scope = dialog.scope.currentData() if spec is None else spec.scope
    if scope != 'global' and scope not in {mode for _, mode in dialog.owner.analysis_scope.choices()}:
        raise ValueError(tr('Сохранённая область графика недоступна. Выберите область в свойствах.'))
    return dialog.owner.analysis_scope.rows(None if scope == 'global' else scope, selected_ids=selected_ids)


def refresh_categories(dialog):
    dialog.category_table.blockSignals(True)
    dialog.category_table.setRowCount(0)
    dialog.category_entries = []
    library = ':' in dialog.spec.kind
    if library and 'channels' not in controls_for(dialog.spec.kind).capabilities:
        dialog.category_table.blockSignals(False)
        return
    for channel, cid in (
        ("color", dialog.spec.color_by),
        ("shape", dialog.spec.shape_by),
        ("size", dialog.spec.size_by),
    ):
        if channel in ("shape", "size") and not library and dialog.spec.kind not in (
            "scatter",
            "depth",
            "multi_depth",
        ):
            continue
        if (
            not cid
            or (channel == "color" and dialog.spec.color_mode == "numeric")
            or (channel == "size" and dialog.spec.size_mode == "numeric")
        ):
            continue
        for key, entry in (
            dialog.spec.style_maps.get(channel, {}).get(cid, {}).items()
        ):
            r = dialog.category_table.rowCount()
            dialog.category_table.insertRow(r)
            item = QTableWidgetItem(
                f"{ {'color': tr('Цвет'), 'shape': tr('Форма'), 'size': tr('Размер')}[channel] }: {category_label(json.loads(key)[1])}"
            )
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if channel == "color" and dialog.interpretation_owner is None:
                item.setCheckState(Qt.CheckState.Unchecked if entry.get("hidden") is True else Qt.CheckState.Checked)
                item.setToolTip(tr('Показывать цветовую группу на графике и при экспорте'))
            dialog.category_table.setItem(r, 0, item)
            dialog.category_table.setItem(r, 1, QTableWidgetItem(str(entry["style"])))
            dialog.category_table.setItem(r, 2, QTableWidgetItem(entry["label"]))
            dialog.category_entries.append((channel, cid, key))
    dialog.category_table.blockSignals(False)
    dialog.category_table.resizeColumnsToContents()


def category_changed(dialog, item):
    if dialog.loading or item.row() >= len(dialog.category_entries):
        return
    channel, cid, key = dialog.category_entries[item.row()]
    entry = dialog.spec.style_maps[channel][cid][key]
    if item.column() == 0 and channel == "color":
        if item.checkState() == Qt.CheckState.Unchecked:
            entry["hidden"] = True
        else:
            entry.pop("hidden", None)
    elif item.column() == 1:
        entry["style"] = marker_value(item.text()) if channel == 'shape' else item.text()
        if channel == "color":
            entry["automatic"] = False
    elif item.column() == 2:
        entry["label"] = item.text()
    dialog.schedule()


def reset_category_parameter(dialog, row, column):
    if column not in (1, 2) or row >= len(dialog.category_entries):
        return
    from autopyro.visualization.plotting import ensure_style_maps
    defaults = deepcopy(dialog.spec)
    defaults.style_maps = {}
    if ':' in defaults.kind and 'channels' in controls_for(defaults.kind).capabilities:
        defaults.kind = 'scatter'
    ensure_style_maps(dialog.owner.model.project, defaults)
    channel, cid, key = dialog.category_entries[row]
    entry = defaults.style_maps.get(channel, {}).get(cid, {}).get(key)
    if entry is not None:
        dialog.category_table.item(row, column).setText(str(entry['style' if column == 1 else 'label']))
        if column == 1 and channel == "color":
            dialog.spec.style_maps[channel][cid][key]["automatic"] = True


def choose_category_color(dialog, row, column):
    if (
        column != 1
        or row >= len(dialog.category_entries)
        or dialog.category_entries[row][0] != "color"
    ):
        return
    color = QColorDialog.getColor(parent=dialog)
    if color.isValid():
        dialog.category_table.item(row, column).setText(color.name())
