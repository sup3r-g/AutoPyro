"""Localized labels and value selectors for stratigraphic input codes."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QLineEdit,
                              QListWidget, QListWidgetItem, QStyledItemDelegate, QVBoxLayout)
from ..i18n import tr
from ..theme.controls import add_search_icon

# Keys remain stratapy/project identifiers; only the visible captions are translated.
GEOLOGICAL_LABELS = {
    'clay': 'Глина',
    'silt': 'Алеврит',
    'sand': 'Песок',
    'gravel': 'Гравий',
    'vf': 'Очень мелкие зёрна',
    'f': 'Мелкие зёрна',
    'm': 'Средние зёрна',
    'c': 'Крупные зёрна',
    'vc': 'Очень крупные зёрна',
    'p': 'Галька',
    'cb': 'Булыжники',
    'b': 'Валуны',
    'f^': 'Мелкие лапилли',
    'm^': 'Средние лапилли',
    'c^': 'Крупные лапилли',
    'block/bomb': 'Блоки и бомбы',
    'Mst': 'Мадстоун',
    'Wst': 'Вакстоун',
    'Pst': 'Пакстоун',
    'Gst': 'Грейнстоун',
    'Fst': 'Флоатстоун',
    'Rst': 'Рудстоун',
    'Bfst': 'Баффлстоун',
    'Bst': 'Баундстоун',
    'Frst': 'Фреймстоун',
    'crypto': 'Криптокристаллический материал',
    'GROUP': 'Группа',
    'SUBGROUP': 'Подгруппа',
    'FORMATION': 'Формация',
    'MEMBER': 'Пачка',
    'log': 'Литологическая колонка',
    'default': 'Профиль зернистости с заливкой',
    'grainsize': 'Колонка и профиль зернистости',
    'off': 'Не показывать',
    'features_default': 'Все элементы рядом с колонкой',
    'merge': 'Все элементы внутри слоя',
    'semi-merge': 'Минералы и линзы внутри, признаки рядом',
    'clastic': 'Обломочные породы',
    'volcanic': 'Вулканические породы',
    'sedimentary': 'Осадочные породы',
    'dunham': 'Карбонатные породы по Данэму',
    'carbonate': 'Карбонатные породы',
    '': 'Не задано',
    'concave': 'Вогнутый профиль',
    'convex': 'Выпуклый профиль',
    'sawtooth': 'Зубчатый профиль',
    'sawtooth_r': 'Обратный зубчатый профиль',
    'sharp': 'Резкий контакт',
    'gradational': 'Постепенный переход',
    'erosional': 'Эрозионный контакт',
    'amphibole': 'Амфибол',
    'anhydrite': 'Ангидрит',
    'anorthite': 'Анортит',
    'aragonite': 'Арагонит',
    'augite': 'Авгит',
    'biotite': 'Биотит',
    'calcite': 'Кальцит',
    'celestite': 'Целестин',
    'chert': 'Кремнистая порода',
    'chlorite': 'Хлорит',
    'chrysotile': 'Хризотил',
    'clinopyroxene': 'Клинопироксен',
    'diopside': 'Диопсид',
    'dolomite': 'Доломит',
    'epidote': 'Эпидот',
    'flint': 'Кремень',
    'fluorite': 'Флюорит',
    'garnet': 'Гранат',
    'glauconite': 'Глауконит',
    'gypsum': 'Гипс',
    'halite': 'Галит',
    'hematite': 'Гематит',
    'hornblende': 'Роговая обманка',
    'illite': 'Иллит',
    'jasper': 'Яшма',
    'kaolinite': 'Каолинит',
    'k-feldspar': 'Калиевый полевой шпат',
    'lizardite': 'Лизардит',
    'magnetite': 'Магнетит',
    'microcline': 'Микроклин',
    'muscovite': 'Мусковит',
    'olivine': 'Оливин',
    'orthoclase': 'Ортоклаз',
    'orthopyroxene': 'Ортопироксен',
    'plagioclase': 'Плагиоклаз',
    'pyrite': 'Пирит',
    'quartz': 'Кварц',
    'siderite': 'Сидерит',
    'smectite': 'Смектит',
    'spinel': 'Шпинель',
    'titanite': 'Титанит',
    'tourmaline': 'Турмалин',
    'volcanic glass': 'Вулканическое стекло',
    'zeolites': 'Цеолиты',
    'zircon': 'Циркон',
    'ammonite': 'Аммонит',
    'arthropod': 'Членистоногое',
    'archaeocyath sponge': 'Археоциатовая губка',
    'benthic foraminifera': 'Бентосные фораминиферы',
    'bivalve': 'Двустворчатый моллюск',
    'brachiopod': 'Брахиопода',
    'bryozoan': 'Мшанка',
    'charcoal': 'Древесный уголь',
    'colonial coral': 'Колониальный коралл',
    'conodont': 'Конодонт',
    'coralline algae': 'Кораллиновые водоросли',
    'crinoid': 'Морская лилия',
    'diatom': 'Диатомея',
    'echinoderm': 'Иглокожее',
    'fragmented': 'Обломки организмов',
    'fungal spore': 'Спора гриба',
    'gastropod': 'Брюхоногий моллюск',
    'graptolite': 'Граптолит',
    'green algae': 'Зелёные водоросли',
    'leaf': 'Лист',
    'microbial mat': 'Микробный мат',
    'mollusc': 'Моллюск',
    'ooid': 'Оолит',
    'ostracod': 'Остракода',
    'pecten shell': 'Раковина гребешка',
    'pelagic foraminifera': 'Пелагические фораминиферы',
    'pollen': 'Пыльца',
    'radiolarian': 'Радиолярия',
    'rhizolith': 'Ризолит',
    'serpulid': 'Серпулида',
    'shell hash': 'Раковинный детрит',
    'solitary coral': 'Одиночный коралл',
    'stromatolite': 'Строматолит',
    'tooth': 'Зуб',
    'trace fossil': 'След жизнедеятельности',
    'trilobite': 'Трилобит',
    'vertebrate bone': 'Кость позвоночного',
    'wood': 'Древесина',
    'bioturbation high': 'Сильная биотурбация',
    'bioturbation medium': 'Умеренная биотурбация',
    'bioturbation low': 'Слабая биотурбация',
    'bored clast': 'Обломок со следами сверления',
    'clast imbrication': 'Черепитчатое расположение обломков',
    'clasts': 'Обломки',
    'climbing ripples': 'Восходящая рябь',
    'concretions': 'Конкреции',
    'convolute bedding': 'Конволютная слоистость',
    'cross bedding': 'Косая слоистость',
    'current ripple': 'Рябь течения',
    'dewatering structures': 'Структуры обезвоживания',
    'flute marks': 'Флейт-отпечатки',
    'groove marks': 'Бороздчатые отпечатки',
    'herringbone cross stratification': 'Перекрёстная косая слоистость',
    'hummocky cross stratification': 'Холмистая косая слоистость',
    'lens': 'Линза',
    'lenticular bedding': 'Линзовидная слоистость',
    'load clasts': 'Структуры нагрузки',
    'dessication cracks': 'Трещины усыхания',
    'normal grading': 'Нормальная градационная слоистость',
    'planar crossbedding': 'Плоская косая слоистость',
    'planar laminations': 'Параллельная слоистость',
    'reverse grading': 'Обратная градационная слоистость',
    'rip up clasts': 'Обломки размыва',
    'scours': 'Размывы',
    'striations': 'Штриховка',
    'stylolites': 'Стилолиты',
    'swaley cross stratification': 'Ложбинная волнистая слоистость',
    'trough crossbedding': 'Мульдообразная косая слоистость',
    'anticline': 'Антиклиналь',
    'syncline': 'Синклиналь',
    'plunging anticline': 'Погружающаяся антиклиналь',
    'plunging syncline': 'Погружающаяся синклиналь',
    'normal fault': 'Сброс',
    'thrust fault': 'Надвиг',
    'horizontal strata': 'Горизонтальные слои',
    'vertical strata': 'Вертикальные слои',
    'strike slip fault': 'Сдвиг',
}



def geological_label(code):
    return tr(GEOLOGICAL_LABELS.get(code, code))


def geological_options(codes):
    return [(geological_label(code), code) for code in codes]


class CheckedValuesDialog(QDialog):
    """Search never drops checked entries that are temporarily hidden."""
    def __init__(self, title, choices, selected, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(460, 480)
        layout = QVBoxLayout(self)
        search = QLineEdit()
        search.setPlaceholderText(tr('Поиск…'))
        add_search_icon(search)
        layout.addWidget(search)
        self.list = QListWidget()
        known = {code for label, code in choices}
        choices = [*choices, *((code, code) for code in selected if code not in known)]
        for label, code in choices:
            item = QListWidgetItem(label, self.list)
            item.setData(Qt.ItemDataRole.UserRole, code)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if code in selected else Qt.CheckState.Unchecked)
        layout.addWidget(self.list)
        search.textChanged.connect(self.filter_items)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        from ..theme.controls import apply_control_icons
        apply_control_icons(self)

    def filter_items(self, text):
        for row in range(self.list.count()):
            item = self.list.item(row)
            item.setHidden(text.casefold() not in item.text().casefold())

    def values(self):
        return [self.list.item(row).data(Qt.ItemDataRole.UserRole) for row in range(self.list.count())
                if self.list.item(row).checkState() == Qt.CheckState.Checked]


class GeologicalDelegate(QStyledItemDelegate):
    def __init__(self, parent, codes=None, multiple=False):
        super().__init__(parent)
        self.codes = codes
        self.multiple = multiple

    def displayText(self, value, locale):
        return ', '.join(geological_label(code) for code in str(value).split(';') if code) if self.multiple else geological_label(str(value))

    def createEditor(self, parent, option, index):
        if self.multiple:
            return None
        editor = QComboBox(parent)
        for label, code in geological_options(self.codes):
            editor.addItem(label, code)
        return editor

    def setEditorData(self, editor, index):
        value = str(index.data() or '')
        row = editor.findData(value)
        if row < 0:
            editor.addItem(value, value)
            row = editor.count() - 1
        editor.setCurrentIndex(row)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentData())
