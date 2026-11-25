"""Select versioned wellbore interpretations before staging an interval import."""

from PySide6.QtCore import QSortFilterProxyModel, Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QPushButton,
    QTableView, QVBoxLayout,
)

from ...ui.i18n import tr
from ...ui.settings import database_parameters
from ...ui.tasks import TaskDialog
from ...ui.theme import icon
from ...ui.theme.controls import apply_control_icons
from ...ui.widgets.results import ResultsModel
from ..stratigraphy import interpretation_catalog, read_interpretations
from ...ui.theme.controls import add_search_icon


class StratigraphyDatabaseDialog(TaskDialog):
    def __init__(self, depth_reference, parent):
        super().__init__(parent)
        self.setWindowTitle(tr('Импорт стратиграфии из БД'))
        self.resize(850, 600)
        self.scheme = None
        self.catalog = []
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.fields = {}
        parameters = database_parameters()
        for key, label in [('host', tr('Сервер')), ('port', tr('Порт')), ('dbname', tr('База')),
                           ('user', tr('Пользователь')), ('password', tr('Пароль'))]:
            field = QLineEdit(str(parameters.get(key, '')))
            self.fields[key] = field
            form.addRow(label, field)
        self.fields['password'].setEchoMode(QLineEdit.EchoMode.Password)
        self.reference = QLineEdit(depth_reference)
        self.reference.setPlaceholderText('MD RKB')
        form.addRow(tr('Тип / начало отсчёта интервалов'), self.reference)
        layout.addLayout(form)
        hint = QLabel(tr('Укажите систему глубин явно. Выберите по одной редакции на ствол. '
                         'Для выбора нескольких строк используйте Ctrl или Shift.'))
        hint.setWordWrap(True)
        layout.addWidget(hint)
        load = QPushButton(icon('refresh-cw'), tr('Загрузить редакции'))
        load.clicked.connect(self.load_catalog)
        layout.addWidget(load)
        search = QLineEdit()
        search.setPlaceholderText(tr('Поиск…'))
        add_search_icon(search)
        layout.addWidget(search)
        self.table = QTableView()
        self.model = ResultsModel([tr('Ствол'), tr('Редакция'), tr('Статус'), tr('Создано')], [], self.table)
        self.proxy = QSortFilterProxyModel(self.table)
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterKeyColumn(-1)
        self.proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        search.textChanged.connect(self.proxy.setFilterFixedString)
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        for column, width in enumerate((210, 190, 140)):
            self.table.setColumnWidth(column, width)
        layout.addWidget(self.table, 1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.read_button = buttons.addButton(tr('Загрузить интервалы'), QDialogButtonBox.ButtonRole.ActionRole)
        self.read_button.setIcon(icon('import'))
        self.read_button.clicked.connect(self.read_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        for key, field in self.fields.items():
            field.textChanged.connect(self.clear_catalog)
            if key != 'password':
                field.textChanged.connect(self.fields['password'].clear)
        self.reference.textChanged.connect(self.update_actions)
        self.table.selectionModel().selectionChanged.connect(self.update_actions)
        self.update_actions()
        apply_control_icons(self)

    def parameters(self):
        return {key: field.text() if key == 'password' else field.text().strip()
                for key, field in self.fields.items()}

    def clear_catalog(self):
        self.catalog = []
        self.model.set_rows([])
        self.status.clear()
        self.update_actions()

    def load_catalog(self):
        self.clear_catalog()
        parameters = self.parameters()
        self.run_task(lambda: interpretation_catalog(parameters), self.show_catalog)

    def show_catalog(self, rows):
        self.catalog = rows
        statuses = {'draft': tr('Черновик'), 'published': tr('Опубликована')}
        self.model.set_rows([(row['wellbore_name'], row['version_tag'], statuses.get(row['status'], row['status']),
                              row['created_at']) for row in rows])
        self.status.setText(tr('Доступно редакций: {0}', len(rows)))
        self.update_actions()

    def update_actions(self):
        self.read_button.setEnabled(bool(self.table.selectionModel().selectedRows() and self.reference.text().strip()))

    def read_selection(self):
        rows = [self.catalog[self.proxy.mapToSource(index).row()]
                for index in self.table.selectionModel().selectedRows()]
        if len({row['wellbore_name'] for row in rows}) != len(rows):
            self.status.setText(tr('Выберите только одну редакцию для каждого имени ствола.'))
            return
        ids = [row['interpretation_id'] for row in rows]
        parameters, reference = self.parameters(), self.reference.text()
        self.run_task(lambda: read_interpretations(parameters, ids, reference), self.finish_read)

    def finish_read(self, scheme):
        self.scheme = scheme
        self.accept()
