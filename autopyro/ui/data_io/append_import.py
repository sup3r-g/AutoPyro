"""Column mapping and paged conflict review before appending laboratory results."""


from ..i18n import tr, message
from collections import Counter
from copy import deepcopy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..tasks import TaskDialog
from autopyro.data_io.importing import (
    ANALYSIS_FIELDS,
    preview_import,
    prepare_import,
    suggest_mapping,
)
from autopyro.model.measurements import derived
from ..table import ImportCommand
from ..table.model import snapshot_project
from ..widgets.parameters import add_parameter_tools
from ..theme import icon


STATUSES = {"new": "Новая проба", "matched": "Найдена проба", "ambiguous": "Неоднозначно", "duplicate": "Уже загружено",
            "input_repeat": "Повтор ключа среди новых проб"}


ACTIONS = {"unresolved": "Выберите решение", "append": "Добавить анализ", "append_new": "Добавить повтор к новой пробе", "new": "Создать отдельную пробу", "skip": "Пропустить"}


PAGE_SIZE = 100


class AppendImportDialog(TaskDialog):
    def __init__(self, owner, incoming):
        super().__init__(owner)
        self.owner, self.incoming = owner, incoming
        self.plan, self.decisions, self.page = None, {}, 0
        self.active_template = None
        self.setWindowTitle(tr('Добавить данные · сопоставление и предварительное сравнение'))
        self.resize(970, 640)
        layout = QVBoxLayout(self)
        note = QLabel(tr('Выберите ключи идентификации пробы. Совпадения точные, без объединения по похожим именам. Новые анализы сохраняются отдельно; рабочие значения по умолчанию остаются прежними.'))
        note.setWordWrap(True)
        layout.addWidget(note)
        from .templates import TemplateBar, template_state
        from autopyro.data_io.import_templates import merge_views
        state = template_state(owner.model.project.view)
        merge_views(state, incoming.view, activate=True)
        self.templates = TemplateBar("append", state, self)
        self.templates.applyRequested.connect(self.apply_template)
        self.templates.saveRequested.connect(self.save_template)
        layout.addWidget(self.templates)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        settings = QWidget()
        form_layout = QVBoxLayout(settings)
        self.mapping = QTableWidget(len(incoming.columns), 3)
        self.mapping.setHorizontalHeaderLabels([tr('Исходная колонка / единица'), tr('Показатель проекта'), tr('Ключ пробы')])
        self.mapping.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        suggested = suggest_mapping(owner.model.project, incoming)
        for i, column in enumerate(incoming.columns):
            self.mapping.setItem(i, 0, QTableWidgetItem(f"{column.name} [{column.unit or tr('без единицы')}]"))
            target = QComboBox()
            target.addItem(tr('Создать новый показатель'), "new")
            target.addItem(tr('Не импортировать'), None)
            for candidate in owner.model.project.columns:
                if not derived(candidate):
                    target.addItem(f"{candidate.name} [{candidate.unit or tr('без единицы')}]", candidate.id)
            target.setCurrentIndex(max(0, target.findData(suggested[column.id])))
            target.currentIndexChanged.connect(self.invalidate)
            self.mapping.setCellWidget(i, 1, target)
            key = QCheckBox()
            key.toggled.connect(self.invalidate)
            self.mapping.setCellWidget(i, 2, key)
        self.mapping.resizeColumnsToContents()
        form_layout.addWidget(self.mapping)
        context = QFormLayout()
        self.context, self.defaults = {}, {}
        labels = {"method": tr('Метод'), "laboratory": tr('Лаборатория'), "instrument": tr('Прибор'),
                  "heating_program": tr('Программа нагрева'), "preparation": tr('Подготовка'), "fraction": tr('Фракция'), "date": tr('Дата анализа')}
        for name in ANALYSIS_FIELDS:
            combo = QComboBox()
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(16)
            combo.addItem(tr('Не задано / общее значение справа'), None)
            for column in incoming.columns:
                combo.addItem(column.name, column.id)
            combo.currentIndexChanged.connect(self.invalidate)
            self.context[name] = combo
            default = QLineEdit()
            default.setPlaceholderText(tr('Общее значение'))
            default.setToolTip(tr('Используется, если колонка не назначена. Пустые значения назначенной колонки сохраняются.'))
            default.textChanged.connect(self.invalidate)
            self.defaults[name] = default
            row = QHBoxLayout()
            row.addWidget(combo, 1)
            row.addWidget(default, 1)
            context.addRow(labels[name], row)
        self.method, self.laboratory = self.defaults["method"], self.defaults["laboratory"]
        self.policy = QComboBox()
        self.policy.addItem(tr('Сохранить текущие; выбирать только первое измерение нового показателя'), "keep")
        self.policy.addItem(tr('Заполнить только отсутствующие результаты (не ND/LOD)'), "fill_missing")
        self.policy.addItem(tr('Выбрать новые результаты рабочими; прежние измерения сохранить'), "replace")
        self.policy.currentIndexChanged.connect(self.clear_template)
        context.addRow(tr('Рабочие результаты'), self.policy)
        form_layout.addLayout(context)
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setWidget(settings)
        tabs.addTab(settings_scroll, icon('columns-3'), tr('Сопоставление и контекст'))
        review = QWidget()
        review_layout = QVBoxLayout(review)
        self.rows = QTableWidget(0, 6)
        self.rows.setHorizontalHeaderLabels([tr('Строка источника'), tr('Ключ'), tr('Состояние'), tr('Кандидаты / проба'), tr('Решение'), tr('Расхождения')])
        self.rows.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        review_layout.addWidget(self.rows)
        paging = QHBoxLayout()
        self.previous, self.next = QPushButton(icon('undo-2'), tr('Предыдущие 100')), QPushButton(icon('redo-2'), tr('Следующие 100'))
        self.previous.clicked.connect(lambda: self.change_page(-1))
        self.next.clicked.connect(lambda: self.change_page(1))
        self.page_label = QLabel()
        for widget in (self.previous, self.page_label, self.next):
            paging.addWidget(widget)
        review_layout.addLayout(paging)
        tabs.addTab(review, icon('shield-check'), tr('Пробы и решения'))
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        tabs.addTab(self.details, icon('circle-question-mark'), tr('Расхождения и ошибки'))
        self.status = QLabel(tr('Предпросмотр ещё не рассчитан.'))
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        preview = QPushButton(icon('files'), tr('Сравнить с проектом'))
        preview.clicked.connect(self.calculate)
        self.apply_button = QPushButton(icon('check'), tr('Применить импорт'))
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.apply_import)
        close = QPushButton(icon('x'), tr('Отмена'))
        close.clicked.connect(self.reject)
        for button in (preview, self.apply_button, close):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        add_parameter_tools(self, changed=self.invalidate, tables=(self.mapping,))

    def invalidate(self, *_):
        self.plan = None
        self.active_template = None
        if hasattr(self, "apply_button"):
            self.apply_button.setEnabled(False)
            self.status.setText(tr('Настройки изменены. Выполните сравнение заново.'))

    def clear_template(self, *_):
        self.active_template = None

    def save_template(self, name):
        from autopyro.data_io.import_templates import create, resolve_append
        try:
            targets = {c.id: c for c in self.owner.model.project.columns}
            indices = {c.id: i for i, c in enumerate(self.incoming.columns)}
            columns = []
            for i, column in enumerate(self.incoming.columns):
                tid = self.mapping.cellWidget(i, 1).currentData()
                target = tid
                if tid not in (None, "new"):
                    c = targets[tid]
                    target = {"id": c.id, "name": c.name, "kind": c.kind, "unit": c.unit, "basis": c.metadata.get("basis", "")}
                columns.append({"header": column.metadata.get("source_header", column.name),
                                "aliases": list(column.metadata.get("source_aliases", [])), "target": target})
            template = create(name, "append", self.templates.state, columns=columns,
                keys=[i for i in range(len(columns)) if self.mapping.cellWidget(i, 2).isChecked()],
                bindings={key: indices[combo.currentData()] for key, combo in self.context.items() if combo.currentData()},
                defaults={key: field.text() for key, field in self.defaults.items() if field.text()}, working_policy=self.policy.currentData())
            resolve_append(template, self.owner.model.project, self.incoming)
            self.templates.put(template)
            self.active_template = template
            self.status.setText(tr('Шаблон сохранится при применении импорта. JSON можно экспортировать отдельно.'))
        except (ValueError, TypeError, KeyError) as exc:
            self.status.setText(message(exc))

    def apply_template(self, template):
        from autopyro.data_io.import_templates import resolve_append
        try:
            resolved = resolve_append(template, self.owner.model.project, self.incoming)
            self.templates.put(template)
            for i, column in enumerate(self.incoming.columns):
                combo = self.mapping.cellWidget(i, 1)
                combo.setCurrentIndex(combo.findData(resolved["mapping"][column.id]))
                self.mapping.cellWidget(i, 2).setChecked(column.id in resolved["keys"])
            for name, combo in self.context.items():
                combo.setCurrentIndex(max(0, combo.findData(resolved["bindings"].get(name))))
            for name, field in self.defaults.items():
                field.setText(resolved["defaults"].get(name, ""))
            self.policy.setCurrentIndex(self.policy.findData(resolved["working_policy"]))
            self.active_template = deepcopy(template)
            self.status.setText(tr('Шаблон применён. Выполните сравнение и проверьте решения перед импортом.'))
        except (ValueError, KeyError, TypeError) as exc:
            self.status.setText(message(exc))

    def calculate(self):
        active = self.active_template
        self.invalidate()
        self.active_template = active
        mapping = {c.id: self.mapping.cellWidget(i, 1).currentData() for i, c in enumerate(self.incoming.columns)}
        keys = [c.id for i, c in enumerate(self.incoming.columns) if self.mapping.cellWidget(i, 2).isChecked()]
        bindings = {name: combo.currentData() for name, combo in self.context.items() if combo.currentData()}
        defaults = {key: field.text() for key, field in self.defaults.items() if field.text()}
        snapshot = snapshot_project(self.owner.model.project, self)
        self.run_task(lambda: preview_import(snapshot, self.incoming, mapping, keys, bindings=bindings, defaults=defaults), self.calculated)

    def calculated(self, plan):
        self.plan, self.decisions, self.page = plan, {}, 0
        counts = Counter(item["status"] for item in plan.rows)
        self.status.setText("; ".join(f"{tr(STATUSES[name])}: {count}" for name, count in counts.items()) + tr('. Ошибок: {0}.', len(plan.errors)))
        source_rows = {r["incoming_id"]: r["source_row"] for r in plan.rows}
        self.details.setPlainText("\n".join([message(error) for error in plan.errors] + [
            tr('Строка {0}, проба {1}, {2}: текущее {3!r} → входящее {4!r}', source_rows[conflict['incoming_id']], conflict['sample_id'], conflict['property'], conflict['current'], conflict['incoming'])
            for conflict in plan.conflicts]))
        self.apply_button.setEnabled(not plan.errors)
        self.render_page()

    def render_page(self):
        if not self.plan:
            return
        items = self.plan.rows[self.page * PAGE_SIZE:(self.page + 1) * PAGE_SIZE]
        self.rows.setRowCount(len(items))
        conflicts = Counter(c["incoming_id"] for c in self.plan.conflicts)
        project_rows = self.owner.model.project.row_map()
        source_rows = {item["incoming_id"]: item["source_row"] for item in self.plan.rows}
        for i, item in enumerate(items):
            rid = item["incoming_id"]
            choice = self.decisions.get(rid, item)
            for column, value in ((0, item["source_row"]), (1, "; ".join(str(value) for _, value in item["key"]) if item["key"] else tr('не задан')),
                                  (2, tr(STATUSES[item["status"]])), (5, conflicts[rid])):
                self.rows.setItem(i, column, QTableWidgetItem(str(value)))
            target = QComboBox()
            target.addItem(tr('Не выбрана'), None)
            for candidate in item["candidates"]:
                row = project_rows[candidate]
                label = " / ".join(str(row.values.get(self.plan.mapping[cid], "")) for cid in self.plan.keys)
                target.addItem(f"{label} · {candidate}", candidate)
            for candidate in item.get("new_candidates", ()):
                target.addItem(tr('Новая проба из строки {0}', source_rows[candidate]), candidate)
            target.setCurrentIndex(max(0, target.findData(choice.get("target_id"))))
            action = QComboBox()
            for name, label in ACTIONS.items():
                if name != "append_new" or item.get("new_candidates"):
                    action.addItem(tr(label), name)
            action.setCurrentIndex(action.findData(choice["action"]))
            if item["status"] == "duplicate":
                action.setEnabled(False)
                target.setEnabled(False)
            def changed(*_, rid=rid, action=action, target=target):
                self.decisions[rid] = {"action": action.currentData(), "target_id": target.currentData()}
            action.currentIndexChanged.connect(changed)
            target.currentIndexChanged.connect(changed)
            self.rows.setCellWidget(i, 3, target)
            self.rows.setCellWidget(i, 4, action)
        self.rows.resizeColumnsToContents()
        self.page_label.setText(tr('Страница {0}; строк всего: {1}. Перед применением разрешите все неоднозначности.', self.page + 1, len(self.plan.rows)))
        self.previous.setEnabled(self.page > 0)
        self.next.setEnabled((self.page + 1) * PAGE_SIZE < len(self.plan.rows))

    def change_page(self, delta):
        self.page += delta
        self.render_page()

    def apply_import(self):
        if not self.plan:
            return
        project = snapshot_project(self.owner.model.project, self)
        plan, decisions, policy = self.plan, deepcopy(self.decisions), self.policy.currentData()
        from autopyro.data_io.import_templates import reference
        state = deepcopy(self.templates.state)
        ref = reference(self.active_template) if self.active_template else None
        self.run_task(lambda: prepare_import(project, self.incoming, plan, decisions, working_policy=policy,
                                            template_view=state, template_ref=ref), self.prepared)

    def prepared(self, outcome):
        prepared, protocol = outcome
        try:
            command = ImportCommand(self.owner.model, prepared, protocol, self.plan.project_digest)
            self.owner.model.undo.push(command)
            self.accept()
        except ValueError as exc:
            self.status.setText(message(exc))
            self.apply_button.setEnabled(False)
