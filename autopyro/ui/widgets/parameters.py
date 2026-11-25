"""Resettable parameter drafts and native Qt What's This help."""

from html import escape

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QAbstractSpinBox, QCheckBox, QComboBox, QDialogButtonBox,
    QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QPushButton, QSpinBox, QTableWidget, QTextEdit, QPlainTextEdit, QWhatsThis, QWidget,
)

from ..i18n import tr
from .parameter_defaults import ParameterDefaults
from ..theme import icon


def parameter_description(key):
    key = {"xmin": "minimum", "ymin": "minimum", "xmax": "maximum", "ymax": "maximum",
           "grid_alpha": "alpha", "interpretation_alpha": "alpha", "stratigraphy_alpha": "alpha",
           "edge_color": "color", "missing_color": "color", "background": "color", "grid_color": "color",
           "sqlite_path": "path", "sqlite_table": "table", "max_distance": "radius"}.get(key, key)
    descriptions = {
        "color": tr("Цвет в формате #RRGGBB или допустимое имя цвета Matplotlib. Кнопка рядом открывает палитру. Цвет относится к рисунку, а не к исходным данным."),
        "path": tr("Путь к существующему файлу SQLite. Источник открывается только для чтения; изменения сохраняются в локальном проекте AutoPyro."),
        "table": tr("Имя существующей таблицы или представления SQLite. Приложение читает снимок и не выполняет SQL, введённый пользователем."),
        "raw": tr("Исходная запись результата: число, ND или значение с квалификатором <, >, ≈. Предыдущие результаты сохраняются; новое измерение добавляется отдельной записью."),
        "scope": tr("Область определяет, какие строки участвуют в расчёте: весь проект, текущий фильтр или выделение. Пустое выделение не заменяется всем проектом."),
        "kind": tr("Тип графика определяет способ отображения данных. Для глубинного планшета выберите треки; для гистограммы — числовой показатель X."),
        "x_id": tr("Исходная колонка для координаты X. Названия и значения колонки не изменяются; единицы не пересчитываются автоматически."),
        "y_id": tr("Исходная колонка для координаты Y или глубины. Для интервальных проб это верх отбора; низ задаётся отдельно."),
        "depth_bottom_id": tr("Колонка низа отбора для интервальных проб. Верх и низ должны иметь одинаковые единицы. Пустой выбор означает точечные пробы."),
        "minimum": tr("Нижняя граница оси. Пустое поле включает автоматический выбор по данным. Для логарифмической оси значение должно быть положительным."),
        "maximum": tr("Верхняя граница оси. Пустое поле включает автоматический выбор по данным. Значение должно быть больше нижней границы."),
        "scale": tr("Линейная шкала сохраняет разности; логарифмическая показывает отношения и допускает только положительные значения."),
        "log_base": tr("Основание логарифмической шкалы: положительное число, не равное 1. Например: 2, 10 или 2.718282. Применяется к выбранной оси, включая экспорт."),
        "inverted": tr("Меняет направление оси без изменения исходных значений. Полезно для отображения глубины сверху вниз."),
        "major_step": tr("Шаг основных делений в единицах оси. Пустое поле означает автоматический шаг. Фиксированный шаг применяется к линейной оси."),
        "minor_divisions": tr("Количество дополнительных делений между основными. Ноль отключает дополнительные деления."),
        "rotation": tr("Поворот подписей делений в градусах. Помогает разместить длинные подписи без перекрытия."),
        "number_format": tr("Формат подписей числовой оси: автоматический, обычный или научный. Хранимые числа не меняются."),
        "label": tr("Подпись оси или объекта. Пустая подпись оси формируется из названия показателя и его единицы."),
        "color_by": tr("Колонка для цвета. Категориальный режим присваивает цвет каждой категории; числовой использует непрерывную шкалу."),
        "shape_by": tr("Категориальная колонка для формы маркера. Отображение не меняет исходную классификацию проб."),
        "size_by": tr("Колонка для размера маркера. Числовые значения масштабируются между заданными минимальным и максимальным размерами."),
        "color_mode": tr("Категориальный режим подходит для скважин и групп; числовой — для непрерывных значений показателя."),
        "size_mode": tr("Числовой режим масштабирует размер по значению; категориальный задаёт отдельный размер каждой категории."),
        "marker": tr("Геометрический символ точки. Размер указывается отдельно в пунктах, а не в единицах координат."),
        "marker_size": tr("Диаметр маркера в типографских пунктах. Изменение влияет на рисунок, но не на географическую или числовую координату."),
        "interval_width": tr("Толщина линии интервальной пробы в пунктах. Длина отрезка определяется верхом и низом отбора."),
        "alpha": tr("Непрозрачность: 0 — полностью прозрачно, 1 — полностью непрозрачно. Прозрачность помогает видеть перекрывающиеся объекты."),
        "cmap": tr("Непрерывная цветовая шкала для числовых значений. Категориальные цвета настраиваются отдельно."),
        "color_min": tr("Минимальное значение непрерывной цветовой шкалы. Пустое поле включает автоматический предел."),
        "color_max": tr("Максимальное значение непрерывной цветовой шкалы. Пустое поле включает автоматический предел."),
        "size_min": tr("Минимальный диаметр маркера в пунктах при масштабировании размера по свойству."),
        "size_max": tr("Максимальный диаметр маркера в пунктах при масштабировании размера по свойству."),
        "bins": tr("Количество интервалов гистограммы. Меньшее число сглаживает распределение; большее показывает более мелкие детали."),
        "density": tr("Нормирует гистограмму так, чтобы её площадь была равна единице. Высота столбца становится плотностью, а не числом проб."),
        "connect": tr("Соединяет точки внутри категориальных групп цвета. Перед включением проверьте группировку по скважине, методу и другим условиям сопоставимости."),
        "text_numbers": tr("Разрешает использовать числовые записи из текстовых колонок в расчёте. ND и квалификаторы < / > не подменяются числами; исходная таблица сохраняется."),
        "include_interpolated": tr("Включает актуальные интерполированные значения отдельно от исходных. Устаревшие интерполяции не используются."),
        "source_crs": tr("Система координат исходных проб, например EPSG:4326. Объявляйте фактическую систему; ошибочная CRS переместит точки в неверное место."),
        "target_crs": tr("Система координат отображаемой карты. Для пространственной интерполяции нужна подходящая проецированная CRS в метрах."),
        "location_mode": tr("Координаты берутся из колонок проб или находятся по точному ключу в точечном слое. Похожие названия автоматически не объединяются."),
        "sample_key": tr("Колонка ключа пробы или ствола для точного сопоставления с полем точечного слоя."),
        "layer_key": tr("Поле точечного слоя, соответствующее ключу пробы. Совпадение должно быть точным и однозначным."),
        "point_layer": tr("Локальный точечный слой с координатами скважин или проб. Файл и исходные объекты не редактируются."),
        "site_layer": tr("Полигональный слой площадок для агрегации. Общие границы и перекрытия дают неоднозначную принадлежность."),
        "aggregate": tr("Способ объединения числовых значений внутри площадки: среднее, медиана, минимум или максимум. Исходные измерения не заменяются."),
        "radius": tr("Радиус поиска соседей в метрах проецированной CRS. Слишком малый радиус оставляет узлы без достаточного числа соседей."),
        "power": tr("Степень IDW управляет влиянием расстояния: большая степень усиливает влияние ближайших точек. Параметр автоматически не подбирается."),
        "neighbors": tr("Максимальное число соседних мест для IDW. Повторные пробы в одном месте обрабатываются выбранной политикой повторов."),
        "min_neighbors": tr("Минимальное число соседей, необходимое для оценки. При недостатке соседей значение не рассчитывается."),
        "grid_size": tr("Число узлов сетки по каждой оси. Увеличение детализации повышает время расчёта, но не добавляет измерений."),
        "width_mm": tr("Ширина рисунка при экспорте в миллиметрах. Размер растрового изображения также зависит от DPI."),
        "height_mm": tr("Высота рисунка при экспорте в миллиметрах. Размер растрового изображения также зависит от DPI."),
        "dpi": tr("Число пикселей на дюйм при растровом экспорте. Большое DPI увеличивает файл; векторные линии SVG/PDF сохраняют геометрию."),
        "legend": tr("Показывает соответствие цветов, символов и категорий. Подписи легенды не меняют значения в таблице."),
        "legend_position": tr("Положение легенды относительно рисунка. Автоматическое положение старается уменьшить перекрытие с данными."),
        "font_family": tr("Шрифт подписей рисунка. Для корректного отображения выбранный шрифт должен содержать используемые символы."),
        "label_size": tr("Размер подписей в типографских пунктах. Влияет только на оформление рисунка."),
        "tick_size": tr("Размер подписей делений оси в типографских пунктах."),
        "title_size": tr("Размер заголовка рисунка в типографских пунктах."),
        "context": tr("Описание сопоставимости измерений: метод, подготовка, фракция и серия анализа. Не заменяет проверку исходных единиц и методики."),
        "confirmed": tr("Отмечайте подтверждение только после проверки совместимости и условий применимости. Сброс снимает подтверждение."),
        "min_pairs": tr("Минимальное число полных пар для корреляции. Недостаточные пары не дополняются искусственными значениями."),
        "password": tr("Пароль подключения. Не записывается в проект или пользовательские настройки; после изменения адреса или пользователя вводится заново."),
        "host": tr("Адрес сервера PostgreSQL. Изменение поля не меняет источник уже открытого проекта."),
        "port": tr("TCP-порт PostgreSQL, от 1 до 65535. Должен совпадать с портом сервера или опубликованным портом контейнера."),
        "dbname": tr("Имя базы данных PostgreSQL. Для чтения используется учётная запись с правами SELECT."),
        "database": tr("Имя базы данных PostgreSQL. Для чтения используется учётная запись с правами SELECT."),
        "user": tr("Имя пользователя PostgreSQL. Для источника необходимы права только на чтение; административная роль не используется."),
        "encoding": tr("Кодировка исходного текстового файла. Неверный выбор искажает буквы; UTF-8 с BOM подходит для обмена с Excel."),
        "separator": tr("Символ между полями CSV/TSV. Выберите фактический разделитель файла, затем обновите предпросмотр."),
        "delimiter": tr("Символ между полями CSV/TSV. Выберите фактический разделитель файла, затем обновите предпросмотр."),
        "decimal": tr("Десятичный знак при чтении CSV/TSV. Разделители тысяч не предполагаются; для Excel используется тип исходной ячейки."),
        "language": tr("Язык интерфейса из установленных каталогов Qt. Применяется после перезапуска и не переводит пользовательские данные."),
        "autosave_minutes": tr("Интервал создания копии восстановления в минутах. Автосохранение не заменяет обычное сохранение проекта."),
        "revision": tr("Редакция схемы обеспечивает воспроизводимость. После изменения геометрии задайте новую редакцию перед применением."),
    }
    return descriptions.get(key, "")


def set_parameter_help(widget, key, label="", default=None):
    widget.setProperty("parameterKey", key)
    widget.setProperty("parameterLabel", label)
    if widget.whatsThis():
        widget.setToolTip(widget.whatsThis())
        widget.setToolTipDuration(20000)
        return
    description = parameter_description(key)
    if not description:
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            description = tr("Числовой параметр. Допустимый диапазон: {0}–{1}.", widget.minimum(), widget.maximum())
        elif isinstance(widget, (QTableWidget, QListWidget)):
            description = tr("Выбор колонок и параметров текущей операции. Отметки меняют выборку или настройки; исходные строки проекта не удаляются.")
        elif isinstance(widget, QComboBox):
            description = tr("Выберите один из доступных вариантов. Подпись варианта не меняет исходные значения проекта.")
        elif isinstance(widget, QCheckBox):
            description = tr("Флажок включает или отключает указанную возможность для текущей операции.")
        else:
            description = tr("Введите значение параметра. Изменение применяется к текущим настройкам; исходные измерения не переписываются.")
    title = label.replace("\n", " ") or widget.accessibleName()
    text = ("<b>" + escape(title) + "</b><br>" if title else "") + escape(description)
    if default is not None and not ("password" in key or isinstance(widget, QLineEdit) and widget.echoMode() != QLineEdit.EchoMode.Normal):
        if isinstance(default, bool):
            default = tr("Включено") if default else tr("Выключено")
        elif default == "":
            default = tr("Пусто / автоматически")
        text += "<br>" + escape(tr("По умолчанию: {0}", default))
    widget.setWhatsThis(text)
    if widget.toolTip() and widget.toolTip() not in (description, label):
        text += '<br>' + escape(widget.toolTip())
        widget.setWhatsThis(text)
    widget.setToolTip(text)
    widget.setToolTipDuration(20000)


def annotate_parameters(dialog):
    from ..theme.controls import apply_control_icons
    apply_control_icons(dialog)
    # Reuse one inventory while annotating an unchanged widget tree.
    forms = dialog.findChildren(QFormLayout)
    labels = dialog.findChildren(QLabel)
    names = {id(value): key for key, value in vars(dialog).items() if hasattr(value, "setWhatsThis")}
    for value in vars(dialog).values():
        if isinstance(value, dict):
            names.update({id(widget): key for key, widget in value.items() if hasattr(widget, "setWhatsThis")})
            for key, widgets in value.items():
                if isinstance(widgets, tuple):
                    for widget in widgets:
                        if hasattr(widget, "setWhatsThis"):
                            names.setdefault(id(widget), key)
    for widget in dialog.findChildren(QWidget):
        if not isinstance(widget, (QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QListWidget, QTextEdit, QPlainTextEdit, QTableWidget)):
            continue
        if isinstance(widget.parentWidget(), (QAbstractSpinBox, QComboBox)):
            continue
        key = widget.property("parameterKey") or names.get(id(widget), "")
        label = widget.property("parameterLabel") or (widget.text() if isinstance(widget, QCheckBox) else "")
        if not label and isinstance(widget, QLineEdit):
            label = widget.placeholderText()
        for form in forms:
            item = form.labelForField(widget)
            if isinstance(item, QLabel):
                label = item.text()
                break
        if label and not widget.accessibleName():
            widget.setAccessibleName(label)
        set_parameter_help(widget, key, label)
        from .parameter_reset import parameter_field
        field = parameter_field(widget)
        field.setToolTip(widget.whatsThis())
        field.setToolTipDuration(20000)
        for form in forms:
            item = form.labelForField(field)
            if isinstance(item, QLabel):
                item.setToolTip(widget.whatsThis())
                item.setToolTipDuration(20000)
        for item in labels:
            if item.buddy() is widget:
                item.setToolTip(widget.whatsThis())
                item.setToolTipDuration(20000)


def add_parameter_tools(dialog, *, reset=None, defaults=None, changed=None, tables=(), exclude=(), reset_text=None, help_only=False, layout=None):
    """Install shared reset/help controls after a dialog has built its form."""
    from .groups import group_parameter_forms
    group_parameter_forms(dialog)
    annotate_parameters(dialog)
    defaults = defaults or ParameterDefaults(dialog, tables=tables, exclude=exclude)
    dialog.parameter_defaults = defaults
    from .parameter_reset import install_parameter_resets
    install_parameter_resets(dialog, defaults, changed=changed)
    annotate_parameters(dialog)
    box = next(iter(dialog.findChildren(QDialogButtonBox)), None)
    if box is None:
        row = QHBoxLayout()
        (layout if layout is not None else dialog.layout()).addLayout(row)
    else:
        row = None
    action = QWhatsThis.createAction(dialog)
    action.setText(tr("Что это?"))
    action.setShortcut(QKeySequence("Shift+F1"))
    dialog.addAction(action)
    dialog.whats_this_action = action
    help_button = QPushButton(icon('circle-question-mark'), tr("Что это?"))
    if box is not None:
        box.addButton(help_button, QDialogButtonBox.ButtonRole.HelpRole)
    help_button.setAutoDefault(False)
    action.setIcon(icon("circle-question-mark"))
    help_button.setToolTip(tr("Нажмите, затем выберите параметр, чтобы открыть его описание. Также доступно Shift+F1."))
    help_button.clicked.connect(action.trigger)
    if row is not None:
        row.addStretch()
        row.addWidget(help_button)
    dialog.help_button = help_button
    if help_only:
        return
    button = box.button(QDialogButtonBox.StandardButton.RestoreDefaults) if box else None
    if button is None:
        text = reset_text or tr("По умолчанию")
        button = box.addButton(text, QDialogButtonBox.ButtonRole.ResetRole) if box else QPushButton(text)
        button.setAutoDefault(False)
        if row is not None:
            row.insertWidget(0, button)

        def restore():
            if reset is not None:
                reset()
            else:
                defaults.restore()
                if changed is not None:
                    changed()
            annotate_parameters(dialog)

        button.clicked.connect(restore)
    button.setWhatsThis(tr("Возвращает параметры к исходным настройкам. Исходные измерения и сохранённые схемы не удаляются."))
    button.setIcon(icon("rotate-ccw"))
    dialog.reset_defaults_button = button
