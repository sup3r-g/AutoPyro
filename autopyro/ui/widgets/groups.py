"""Named parameter sections without replacing controls or their signal wiring."""

from PySide6.QtWidgets import QFormLayout, QGroupBox, QVBoxLayout

from ..i18n import tr


def _contains(item, widget):
    if item is None or widget is None:
        return False
    if item.widget() is not None:
        return item.widget() is widget or item.widget().isAncestorOf(widget)
    layout = item.layout()
    return layout is not None and any(_contains(layout.itemAt(i), widget) for i in range(layout.count()))


def group_form(form, boundaries):
    """Move existing form rows into titled boxes. Boundaries refer to controls."""
    starts = {}
    for widget, title in boundaries:
        for row in range(form.rowCount()):
            if any(_contains(form.itemAt(row, role), widget) for role in
                   (QFormLayout.ItemRole.LabelRole, QFormLayout.ItemRole.FieldRole, QFormLayout.ItemRole.SpanningRole)):
                starts[row] = title
                break
    starts.setdefault(0, tr('Параметры'))
    rows = [form.takeRow(0) for _ in range(form.rowCount())]
    target = None
    for index, row in enumerate(rows):
        if index in starts:
            box = QGroupBox(starts[index])
            target = QFormLayout(box)
            target.setRowWrapPolicy(form.rowWrapPolicy())
            target.setFieldGrowthPolicy(form.fieldGrowthPolicy())
            form.addRow(box)
        label = row.labelItem.widget() if row.labelItem is not None else None
        field = row.fieldItem
        content = field.widget() if field is not None and field.widget() is not None else field.layout() if field is not None else None
        if label is not None:
            target.addRow(label, content)
        elif content is not None:
            target.addRow(content)


def group_layout(layout, boundaries):
    """Group contiguous parameter widgets/layouts; a None title ends a section."""
    starts = {layout.indexOf(anchor): title for anchor, title in boundaries if layout.indexOf(anchor) >= 0}
    stretches = [layout.stretch(index) for index in range(layout.count())]
    items = [layout.takeAt(0) for _ in range(layout.count())]
    target = layout
    for index, item in enumerate(items):
        if index in starts:
            title = starts[index]
            target = layout
            if title is not None:
                box = QGroupBox(title)
                layout.addWidget(box)
                target = QVBoxLayout(box)
        if item.widget() is not None:
            target.addWidget(item.widget(), stretches[index])
        elif item.layout() is not None:
            target.addLayout(item.layout())
        else:
            target.addItem(item)


def group_parameter_forms(dialog):
    """Section remaining plain forms in every module using its named controls."""
    boundaries = [
        ('language', tr('Интерфейс')), ('autosave_enabled', tr('Автосохранение')),
        ('delimiter', tr('Импорт')), ('host', tr('PostgreSQL')), ('sqlite_path', tr('SQLite')),
    ]
    by_module = {
        'interpolation': [('x', tr('Исходные данные')), ('method', tr('Метод интерполяции')), ('name', tr('Результат'))],
        'calculation': [('scope', tr('Область и контекст'))],
        'stratigraphy': [('name', tr('Схема')), ('mode', tr('Привязка проб')), ('level', tr('Результат'))],
        'statistics': [('scope', tr('Область данных')), ('min_pairs', tr('Расчёт корреляций'))],
        'interpretation': [('x', tr('Привязка осей')), ('scope', tr('Область данных'))],
        'qc': [('severity', tr('Фильтры результатов')), ('property_filter', tr('Поиск проб'))],
    }
    boundaries += by_module.get(type(dialog).__module__.rsplit('.', 1)[-1], [])
    for form in dialog.findChildren(QFormLayout):
        if isinstance(form.parentWidget(), QGroupBox) or not form.rowCount():
            continue
        if any(isinstance(form.itemAt(row, QFormLayout.ItemRole.SpanningRole).widget(), QGroupBox)
               for row in range(form.rowCount()) if form.itemAt(row, QFormLayout.ItemRole.SpanningRole) is not None):
            continue
        group_form(form, [(getattr(dialog, key, None), title) for key, title in boundaries])
