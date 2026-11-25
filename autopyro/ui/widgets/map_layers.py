"""Controls for map layer display, inspection, saved views and attribution."""
from copy import deepcopy
from html import escape

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QUndoCommand
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QDialogButtonBox,
                              QHBoxLayout, QInputDialog, QLabel, QListWidget,
                              QMessageBox, QPushButton, QScrollArea, QSplitter,
                              QTableWidget, QTextBrowser, QVBoxLayout, QWidget)

from autopyro.model.core import Column, timestamp
from autopyro.spatial.mapping import saved_map_layers
from autopyro.spatial.map_tools import assign_regions, numeric
from autopyro.visualization.plotting import category_key
from ..i18n import tr, message
from ..table.commands import AddColumnCommand
from ..tasks import run_with_progress
from .forms import Form
from ..theme import icon
from ..theme.controls import add_command_button


SAMPLE_EXCLUSION_LABELS = {
    'stratigraphy_filter': 'Другое стратиграфическое подразделение',
    'depth_filter': 'Вне диапазона глубин или нет числовой глубины',
    'invalid_depth_interval': 'Некорректный интервал глубин',
    'date_filter': 'Вне диапазона дат',
    'invalid_date': 'Отсутствующая или некорректная дата',
    'coordinate:missing': 'Отсутствуют X или Y',
    'coordinate:non_detect': 'В X или Y указано ND',
    'coordinate:censored': 'В X или Y указано пороговое значение',
    'coordinate:other': 'X или Y не распознаны как число',
    'coordinate:qc_excluded': 'Координаты исключены контролем качества',
    'coordinate:interpolation_excluded': 'Интерполированные координаты исключены',
    'invalid_longitude_latitude': 'Долгота или широта вне допустимого диапазона',
    'coordinate_transform_error': 'Не удалось преобразовать координаты',
    'location_key_missing': 'Отсутствует ключ пробы',
    'location_key_unmatched': 'Нет точного совпадения ключа в слое',
    'location_key_duplicate': 'Ключ неоднозначен: несколько объектов в слое',
    'location_geometry_missing': 'Объект с таким ключом без пригодной точечной геометрии',
}

SPATIAL_JOIN_STATUS_LABELS = dict(matched='Сопоставлено', boundary='На границе', overlap='Перекрытие',
                                  outside='Вне областей', no_location='Нет координат', not_in_scope='Вне выборки')


def layer_source_identity(layer):
    source = layer.get('source') or {}
    if not source.get('sha256'):
        return None
    return dict(
        path=source.get('path'), sha256=source['sha256'], layer=source.get('layer'),
        sidecars=[{key: item[key] for key in ('path', 'sha256')} for item in source.get('sidecars', [])],
        crs=layer['crs'],
    )


class MapViewCommand(QUndoCommand):
    def __init__(self, owner, name, snapshot, title):
        super().__init__(title)
        self.owner = owner
        self.project = owner.model.project
        self.name = name
        self.previous = deepcopy(self.project.view.get('map_views', {}).get(name))
        self.snapshot = snapshot

    def apply(self, snapshot):
        views = self.project.view.setdefault('map_views', {})
        if snapshot is None:
            views.pop(self.name, None)
        else:
            views[self.name] = deepcopy(snapshot)
        dialog = self.owner.map_window
        if dialog is not None and not dialog.closed:
            dialog.sync_named_views()
        self.owner.view_changed()

    def redo(self):
        self.apply(self.snapshot)

    def undo(self):
        self.apply(self.previous)


class LayerOrderTable(QTableWidget):
    reordered = Signal(int, int)

    def __init__(self, *args):
        super().__init__(*args)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setDragDropOverwriteMode(False)
        self.setToolTip(tr('Перетащите слой: первая строка отображается поверх остальных.'))

    def dropEvent(self, event):
        source = self.currentRow()
        destination = self.indexAt(event.position().toPoint()).row()
        if destination < 0:
            destination = self.rowCount() - 1
        if source >= 0 and event.source() is self:
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
            self.reordered.emit(source, destination)
        else:
            event.ignore()


class MapLayerTools:
    def map_status(self, report, result):
        from autopyro.spatial.map_tools import filtered_features
        visible = [layer for layer in report['layers'] if layer.get('visible', True)]
        parts = [tr('Объектов на карте: {0}', len(result['layer_features']))]
        if report['spec']['location_mode'] != 'layers':
            located = len(report['points'])
            after_filters = len(report['row_ids'])
            input_count = report['diagnostics']['input_rows']
            parts.insert(0, tr('Проб на карте: {0} из {1}', located, input_count))
            if input_count != after_filters:
                parts.append(tr('После фильтров карты: {0}', after_filters))
            missing = after_filters - located
            if missing:
                parts.append(tr('Без привязки: {0}', missing))
            filtered = sum(report['diagnostics']['filtered'].values())
            if filtered:
                parts.append(tr('Исключено фильтрами карты: {0}', filtered))
        hidden = sum(len(layer['features']) - len(filtered_features(layer)) for layer in visible)
        if hidden:
            parts.append(tr('Скрыто фильтрами: {0}', hidden))
        unavailable = sum(layer.get('diagnostics', {}).get(key, 0) for layer in visible
                          for key in ('missing_coordinates', 'null_geometry', 'empty_geometry'))
        if unavailable:
            parts.append(tr('Без отображаемой геометрии: {0}', unavailable))
        text = ' · '.join(parts)
        summary = self.spatial_join_summary()
        if summary:
            text += '\n' + summary
        return text

    def spatial_join_summary(self):
        for column in reversed(self.owner.model.project.columns):
            join = column.metadata.get('spatial_join')
            if isinstance(join, dict) and join.get('status_column'):
                counts = '; '.join(tr('{0}: {1}', tr(SPATIAL_JOIN_STATUS_LABELS[key]), count)
                                   for key, count in join['diagnostics'].items())
                return tr('Последнее присоединение — {0}: {1}', join['layer_name'], counts)
        return ''

    def show_sample_diagnostics(self):
        report = self.report
        groups = report['diagnostics']['excluded_rows'] if report else {}
        if not groups:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(tr('Диагностика проб'))
        dialog.resize(540, 280)
        layout = QVBoxLayout(dialog)
        note = QLabel(tr('Для каждой пробы показана первая причина исключения. Выберите причину, чтобы выделить пробы в таблице.'))
        note.setWordWrap(True)
        layout.addWidget(note)
        listing = QListWidget()
        for reason, rows in groups.items():
            listing.addItem(tr('{0} · {1}', tr(SAMPLE_EXCLUSION_LABELS.get(reason, 'Другая причина исключения')), len(rows)))
            listing.item(listing.count() - 1).setData(Qt.ItemDataRole.UserRole, reason)
        listing.setCurrentRow(0)
        layout.addWidget(listing)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        select = add_command_button(buttons, 'table-2', tr('Выделить в таблице'), QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        listing.itemDoubleClicked.connect(dialog.accept)
        select.setEnabled(listing.currentItem() is not None)
        listing.currentItemChanged.connect(lambda item, _: select.setEnabled(item is not None))
        layout.addWidget(buttons)
        try:
            if dialog.exec() == QDialog.DialogCode.Accepted and self.report is report and listing.currentItem():
                self.select_rows(groups[listing.currentItem().data(Qt.ItemDataRole.UserRole)])
                visible = self.owner.table.selectionModel().selection()
                if visible:
                    self.owner.desktop.open_table()
                    self.owner.table.scrollTo(visible[0].topLeft())
        finally:
            dialog.deleteLater()

    def reorder_layers(self, source, destination):
        if source == destination or not (0 <= source < len(self.layers) and 0 <= destination < len(self.layers)):
            return
        self.layers.insert(destination, self.layers.pop(source))
        self.sync_layers()
        self.layer_table.selectRow(destination)
        self.schedule()

    def build_layer_tools(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.layer_settings_layer = QComboBox()
        layout.addWidget(self.layer_settings_layer)
        self.layer_filter_info = QLabel()
        layout.addWidget(self.layer_filter_info)
        self.layer_binding_hint = QLabel(tr('Привязка проб по ключу использует и скрытые скважины. Фильтр и видимость этого слоя меняют только показ скважин на карте.'))
        self.layer_binding_hint.setWordWrap(True)
        self.layer_binding_hint.hide()
        layout.addWidget(self.layer_binding_hint)
        self.layer_settings_layer.currentIndexChanged.connect(self.update_layer_binding_hint)
        self.point_layer.currentIndexChanged.connect(self.update_layer_binding_hint)
        self.location.currentIndexChanged.connect(self.update_layer_binding_hint)
        columns = QSplitter()
        self.layer_display = Form()
        self.layer_display.section(tr('Раскраска слоя'))
        self.layer_display.combo('color_by', tr('Цвет по атрибуту'), [])
        self.layer_display.combo('color_mode', tr('Режим цвета'), [(tr('Категории'), 'category'), (tr('Числовая шкала'), 'numeric')])
        self.layer_display.combo('color_scale', tr('Шкала'), [(tr('Линейная'), 'linear'), (tr('Логарифмическая'), 'log')])
        self.layer_display.combo('cmap', tr('Цветовая шкала'), [])
        self.layer_display.text('color_min', tr('Минимум (авто)'), optional=True)
        self.layer_display.text('color_max', tr('Максимум (авто)'), optional=True)
        self.layer_display.text('missing_color', tr('Цвет пропусков'), '#999999', color=True)
        self.layer_rules = Form()
        self.layer_rules.section(tr('Фильтр объектов'))
        self.layer_rules.combo('filter_field', tr('Поле'), [])
        self.layer_rules.combo('filter_op', tr('Условие'), [(tr('Равно'), 'eq'), (tr('Не равно'), 'ne'),
                              (tr('Содержит'), 'contains'), (tr('Больше'), 'gt'), (tr('Не меньше'), 'ge'),
                              (tr('Меньше'), 'lt'), (tr('Не больше'), 'le'), (tr('Пустое'), 'isnull'), (tr('Не пустое'), 'notnull')])
        self.layer_rules.text('filter_value', tr('Значение'))
        self.layer_rules.check('legend_hidden', tr('Сохранять скрытые категории в легенде'), True)
        self.layer_rules.section(tr('Подписи объектов'))
        self.layer_rules.combo('label_by', tr('Поле подписи'), [])
        self.layer_rules.number('label_size', tr('Размер подписи'), 8, 5, 30, integer=True)
        for form in (self.layer_display, self.layer_rules):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(form)
            columns.addWidget(scroll)
        layout.addWidget(columns)
        self.layer_tabs.addTab(panel, icon('sliders-horizontal'), tr('Настройки слоя'))
        self.layer_settings_layer.currentIndexChanged.connect(self.sync_layer_settings)
        self.layer_display.edited.connect(self.layer_settings_changed)
        self.layer_rules.edited.connect(self.layer_settings_changed)
        self.feature_card = QTextBrowser()
        self.feature_card.setOpenExternalLinks(False)
        self.feature_card.setHtml(tr('Выберите объект на карте или строку атрибутов.'))

    def sync_layer_settings(self, *_):
        layer = next((item for item in self.layers if item['id'] == self.layer_settings_layer.currentData()), None)
        self.layer_display.setEnabled(layer is not None)
        self.layer_rules.setEnabled(layer is not None)
        if layer is None:
            self.layer_filter_info.clear()
            return
        fields = layer.get('fields') or sorted({key for f in layer['features'] for key in f['properties']})
        for form, key in [(self.layer_display, 'color_by'), (self.layer_rules, 'filter_field'), (self.layer_rules, 'label_by')]:
            combo = form.fields[key][0]
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(tr('Не задано'), '')
            for field in fields:
                combo.addItem(field, field)
            combo.blockSignals(False)
        self.layer_display.load({key: layer.get(key, default) for key, default in
                                 dict(color_by='', color_mode='category', color_scale='linear', cmap='viridis',
                                      color_min=None, color_max=None, missing_color='#999999').items()})
        rule = layer.get('filter') or {}
        self.layer_rules.load(dict(filter_field=rule.get('field', ''), filter_op=rule.get('op', 'eq'),
                                   filter_value=rule.get('value', ''), legend_hidden=layer.get('legend_hidden', True),
                                   label_by=layer.get('label_by', ''), label_size=layer.get('label_size', 8)))
        self.update_filter_info(layer)

    def update_filter_info(self, layer):
        from autopyro.spatial.map_tools import filtered_features
        self.layer_filter_info.setText(tr('Отображается объектов: {0} из {1}', len(filtered_features(layer)), len(layer['features'])))

    def update_layer_binding_hint(self, *_):
        layer_id = self.layer_settings_layer.currentData()
        self.layer_binding_hint.setVisible(bool(layer_id) and self.location.currentData() == 'layer'
                                           and self.point_layer.currentData() == layer_id)

    def layer_settings_changed(self):
        if self.loading:
            return
        layer = next((item for item in self.layers if item['id'] == self.layer_settings_layer.currentData()), None)
        if layer is None:
            return
        try:
            display, rule = self.layer_display.values(), self.layer_rules.values()
        except ValueError as exc:
            self.status.setText(message(exc))
            return
        layer.update(display)
        layer.update({key: rule[key] for key in ('label_by', 'label_size', 'legend_hidden')})
        layer['filter'] = dict(field=rule['filter_field'], op=rule['filter_op'], value=rule['filter_value'])
        row = next(i for i, item in enumerate(self.layers) if item['id'] == layer['id'])
        combo = self.layer_table.cellWidget(row, 6)
        combo.blockSignals(True)
        combo.setCurrentIndex(max(0, combo.findData(layer['color_by'])))
        combo.blockSignals(False)
        self.sync_layer_categories()
        self.update_filter_info(layer)
        self.schedule()

    def show_feature_card(self, layer_id, fid):
        layer = next((item for item in self.layers if item['id'] == layer_id), None)
        if layer is None:
            return
        feature = next((f for f in layer.get('attributes', layer['features']) if str(f['id']) == str(fid)), None)
        if feature is None:
            return
        properties = feature['properties']
        rows = ''.join(f'<tr><td>{escape(str(key))}</td><td>{escape(str(value)) if value is not None else "—"}</td></tr>' for key, value in properties.items())
        html = f'<b>{escape(layer["name"])} · {escape(str(fid))}</b>'
        if self.report and self.spec.location_mode == 'layer' and layer_id == self.spec.point_layer:
            key = properties.get(self.spec.layer_key)
            occurrences = sum(category_key(f['properties'].get(self.spec.layer_key)) == category_key(key) for f in layer['features']) if key is not None else 0
            if occurrences == 1:
                row_map = self.owner.model.project.row_map()
                samples = [row_map[rid] for rid in self.report['row_ids'] if category_key(row_map[rid].values.get(self.spec.sample_key)) == category_key(key)]
                html += f'<p>{tr("Связанных замеров: {0}", len(samples))}</p>'
                depth = next((c for c in self.owner.model.project.columns if c.id == self.spec.depth_id or
                              (not self.spec.depth_id and c.name.casefold() in ('depth', 'глубина'))), None)
                values = [v for sample in samples if depth and (v := numeric(sample.values.get(depth.id))) is not None]
                if values:
                    html += f'<p>{escape(depth.name)}: {min(values):g}–{max(values):g} {escape(depth.unit)}</p>'
                html += f'<p>{tr("Показатель / непустых значений")}</p><table>' + ''.join(f'<tr><td>{escape(c.name)}</td><td>{sum(row.values.get(c.id) is not None for row in samples)}</td></tr>'
                                           for c in self.owner.model.project.columns if c.kind == 'number' and any(row.values.get(c.id) is not None for row in samples)) + '</table>'
            elif occurrences > 1:
                html += f'<p>{tr("Неоднозначный ключ скважины")}</p>'
        self.feature_card.setHtml(html + f'<hr><table>{rows}</table>')

    def build_view_tools(self, layout):
        row = QHBoxLayout()
        self.saved_view = QComboBox()
        row.addWidget(self.saved_view, 1)
        save = QPushButton(icon('save'), tr('Сохранить вид…'))
        save.clicked.connect(self.save_named_view)
        row.addWidget(save)
        self.delete_view_button = QPushButton(icon('trash'), tr('Удалить вид'))
        self.delete_view_button.clicked.connect(self.delete_named_view)
        row.addWidget(self.delete_view_button)
        layout.addLayout(row)
        self.saved_view.activated.connect(self.load_named_view)
        self.saved_view.currentIndexChanged.connect(
            lambda: self.delete_view_button.setEnabled(bool(self.saved_view.currentData())))
        self.delete_view_button.setEnabled(False)
        self.sync_named_views()

    def sync_named_views(self):
        value = self.saved_view.currentData()
        self.saved_view.clear()
        self.saved_view.addItem(tr('Сохранённые виды карты'), '')
        for name in self.owner.model.project.view.get('map_views', {}):
            self.saved_view.addItem(name, name)
        self.saved_view.setCurrentIndex(max(0, self.saved_view.findData(value)))

    def save_named_view(self, name=None):
        if not isinstance(name, str):
            name, ok = QInputDialog.getText(self, tr('Сохранить вид'), tr('Название'), text=self.saved_view.currentData() or '')
            if not ok:
                return
        name = name.strip()
        if not name:
            return
        project = self.owner.model.project
        if name in project.view.get('map_views', {}) and QMessageBox.question(
            self, tr('Сохранить вид'), tr('Заменить сохранённый вид «{0}»?', name),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        if self.timer.isActive() or not self.report:
            self.redraw()
        if not self.report or self.owner.model.project is not project:
            return
        limits = self.result['axes'].get_xlim(), self.result['axes'].get_ylim()
        spec = self.spec.to_dict()
        spec.update({key: self.report['spec'][key] for key in ('target_crs', 'location_mode')})
        spec.update(zip(('xmin', 'xmax', 'ymin', 'ymax'), [*limits[0], *limits[1]]))
        excluded = {'features', 'attributes', 'fields', 'source', 'diagnostics', 'crs'}
        snapshot = dict(spec=spec, layers=[{key: deepcopy(value) for key, value in layer.items() if key not in excluded} for layer in self.layers])
        snapshot['layer_sources'] = {
            layer['id']: identity for layer in self.layers if (identity := layer_source_identity(layer))
        }
        self.owner.model.undo.push(MapViewCommand(self.owner, name, snapshot, tr('Сохранить вид')))
        self.saved_view.setCurrentIndex(self.saved_view.findData(name))

    def remember_removed_layer(self, layer):
        identity = layer_source_identity(layer)
        if identity is None:
            return
        for snapshot in self.owner.model.project.view.get('map_views', {}).values():
            if any(item['id'] == layer['id'] for item in snapshot['layers']):
                snapshot.setdefault('layer_sources', {})[layer['id']] = identity

    def restore_imported_layer_id(self, layer):
        identity = layer_source_identity(layer)
        if identity is None:
            return
        active = {item['id'] for item in self.layers}
        candidates = {
            layer_id
            for snapshot in self.owner.model.project.view.get('map_views', {}).values()
            for layer_id, source in snapshot.get('layer_sources', {}).items()
            if source == identity and layer_id not in active
        }
        if len(candidates) == 1:
            layer['id'] = candidates.pop()

    def load_named_view(self, *_):
        name = self.saved_view.currentData()
        snapshot = self.owner.model.project.view.get('map_views', {}).get(name)
        if snapshot is None:
            return
        spec, layers, missing = saved_map_layers(snapshot, self.layers, keep_extra=True)
        if missing:
            if all(layer_id in snapshot.get('layer_sources', {}) for layer_id in missing):
                detail = tr('Слой сохранённого вида отсутствует. Повторно импортируйте исходный файл с той же CRS.')
            else:
                detail = tr('Слой сохранённого вида отсутствует. В виде нет сведений для восстановления связи со слоем.')
            QMessageBox.warning(self, tr('Карта'), detail)
            return
        self.loading = True
        by_id = {layer['id']: layer for layer in self.layers}
        for restored in layers:
            layer = by_id[restored['id']]
            layer.clear()
            layer.update(restored)
        self.layers = [by_id[layer['id']] for layer in layers]
        self.spec = spec
        self.sync_layers()
        for form in self.forms():
            form.load(self.spec.to_dict())
        self.owner.analysis_scope.set_mode(self.spec.scope)
        self.sync_layer_fields()
        self.data.load({'layer_key': self.spec.layer_key})
        self.loading = False
        self.map_viewport = None
        self.schedule()

    def delete_named_view(self):
        name = self.saved_view.currentData()
        if name and QMessageBox.question(
            self, tr('Удалить вид'), tr('Удалить сохранённый вид «{0}»?', name),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes:
            self.owner.model.undo.push(MapViewCommand(self.owner, name, None, tr('Удалить вид')))

    def spatial_join_dialog(self):
        if self.timer.isActive():
            self.redraw()
        if not self.report or not self.report['row_ids']:
            QMessageBox.information(self, tr('Карта'), tr('Сначала задайте привязку проб и постройте карту.'))
            return
        layers = [layer for layer in self.layers if layer['features'] and all(f['geometry']['type'] in ('Polygon', 'MultiPolygon') for f in layer['features'])]
        if not layers:
            QMessageBox.information(self, tr('Карта'), tr('Для этой операции импортируйте полигональный слой областей.'))
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(tr('Присоединить области к пробам'))
        dialog.resize(500, 420)
        layout = QVBoxLayout(dialog)
        selector = QComboBox()
        for layer in layers:
            selector.addItem(layer['name'], layer['id'])
        layout.addWidget(selector)
        layout.addWidget(QLabel(tr('Выберите атрибуты для добавления в таблицу проб.')))
        fields = QListWidget()
        fields.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        layout.addWidget(fields)
        def sync_fields(*_):
            fields.clear()
            layer = next(item for item in layers if item['id'] == selector.currentData())
            names = layer.get('fields') or sorted({key for f in layer['features'] for key in f['properties']})
            fields.addItems(names)
            for i in range(fields.count()):
                fields.item(i).setSelected(fields.item(i).text() == (self.spec.site_label or 'Paleogeogr'))
        selector.currentIndexChanged.connect(sync_fields)
        sync_fields()
        note = QLabel(tr('Граничным пробам атрибуты не назначаются, даже если их учитывает агрегат площадки. Границы и перекрытия отмечаются отдельными статусами. Учитываются текущие фильтры карты и слоя.'))
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        fields.itemSelectionChanged.connect(lambda: buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(fields.selectedItems())))
        buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(fields.selectedItems()))
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted and fields.selectedItems():
            try:
                self.apply_spatial_join(selector.currentData(), [item.text() for item in fields.selectedItems()])
            except Exception as exc:
                QMessageBox.warning(self, tr('Карта'), message(exc))

    def apply_spatial_join(self, layer_id, fields):
        if not self.report or not self.report['row_ids']:
            raise ValueError(tr('Сначала задайте привязку проб и постройте карту.'))
        report = self.report
        records, diagnostics = run_with_progress(
            self, tr('Присоединить области к пробам'), lambda: assign_regions(report, layer_id, fields))
        if self.report is not report or self.closed:
            raise ValueError(tr('Карта изменилась. Повторите присоединение.'))
        layer = next(item for item in report['layers'] if item['id'] == layer_id)
        metadata = dict(spatial_join=dict(layer_id=layer_id, source=deepcopy(layer.get('source', {})), fields=fields,
                                        target_crs=report['spec']['target_crs'], filter=deepcopy(layer.get('filter', {})),
                                        map_spec=deepcopy(report['spec']), created_at=timestamp(), diagnostics=diagnostics))
        names = {c.name for c in self.owner.model.project.columns}
        commands = []
        status_names = {key: tr(label) for key, label in SPATIAL_JOIN_STATUS_LABELS.items()}
        for field in [*fields, None]:
            base = f"{layer['name']} · {field if field is not None else tr('Статус привязки')}"
            name, index = base, 2
            while name in names:
                name, index = f'{base} ({index})', index + 1
            names.add(name)
            values = {rid: record['values'][field] if field is not None else status_names[record['status']] for rid, record in records.items()}
            if field is None:
                values.update({row.id: status_names['not_in_scope'] for row in self.owner.model.project.rows if row.id not in records})
            nonempty = [v for v in values.values() if v is not None]
            kind = 'number' if field is not None and nonempty and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in nonempty) else 'text'
            column = Column(name, kind, metadata=deepcopy(metadata))
            if field is None:
                column.metadata['spatial_join'].update(status_column=True, layer_name=layer['name'])
            commands.append(AddColumnCommand(self.owner.model, column, values))
        undo = self.owner.model.undo
        undo.beginMacro(tr('Присоединить области к пробам'))
        try:
            for command in commands:
                undo.push(command)
        finally:
            undo.endMacro()
        self.status.setText(self.spatial_join_summary())
        return diagnostics
