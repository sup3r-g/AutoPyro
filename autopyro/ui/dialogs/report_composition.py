"""Manage report scope, selected content, section order, comments and logo."""

from __future__ import annotations

import base64

from PySide6.QtCore import QBuffer, QIODevice, Qt
from PySide6.QtGui import QImageReader
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QInputDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from autopyro.reports.reporting import REPORT_SECTIONS, ReportOptions, report_blocks

from ..i18n import language, message, tr
from ..theme import icon


def scope_changed(sheet):
    sheet.update_scope_summary()
    sheet.invalidate()


def update_scope_summary(sheet):
    try:
        mode = sheet.scope.currentData()
        if mode == "all":
            count = len(sheet.owner.model.project.rows)
        elif mode == "filtered":
            count = sheet.owner.proxy.rowCount()
        else:
            count = len(sheet.owner.analysis_scope.rows())
    except ValueError as exc:
        sheet.scope_summary.setText(message(exc))
    else:
        sheet.scope_summary.setText(tr('Проб в выборке: {0}', count))


def composition(sheet):
    column_ids = tuple(
        sheet.columns.item(i).data(Qt.ItemDataRole.UserRole)
        for i in range(sheet.columns.count())
        if sheet.columns.item(i).checkState() == Qt.CheckState.Checked
    )
    group_ids = tuple(dict.fromkeys(
        column_id for column_id in (sheet.group.currentData(), sheet.subgroup.currentData())
        if column_id is not None
    ))
    options = ReportOptions(
        column_ids,
        group_ids,
        sheet.title.text(),
        sheet.author.text(),
        sheet.conclusions.toPlainText(),
        sheet.scope.currentText(),
        sheet.text_numbers.isChecked(),
        sheet.interpolated.isChecked(),
        sheet.interpretation.isChecked(),
        sheet.limit.value(),
        include_stratigraphy=sheet.stratigraphy.isChecked(),
        include_nonpyro=sheet.nonpyro.isChecked(),
        sections=sheet.section_order,
        profile_depth_id=sheet.profile_depth.currentData(),
        logo_image=sheet.logo_image,
        section_titles=sheet.section_titles,
        block_comments=sheet.block_comments,
        language=language(),
        joint_snapshot_id=sheet.joint_snapshot.currentData(),
        qc_open_only=sheet.qc_open_only.isChecked(),
        qc_limit=sheet.qc_limit.value(),
        protocol_limit=sheet.protocol_limit.value(),
        map_view_name=sheet.map_view.currentData(),
        detailed_correlations=sheet.detailed_correlations.isChecked(),
    )
    plots = [
        (
            sheet.plots.item(i).text(),
            sheet.plots.item(i).data(Qt.ItemDataRole.UserRole),
        )
        for i in range(sheet.plots.count())
        if sheet.plots.item(i).checkState() == Qt.CheckState.Checked
    ]
    return options, plots


def selection_context(sheet, scope, resolved_ids=None):
    project = sheet.owner.model.project
    view = sheet.owner.capture_view(include_sample_set=True)
    if scope.startswith('view:'):
        view = project.view['named'][scope[5:]]
    elif scope.startswith('set:'):
        view = {"sample_set": scope[4:]}
    context = {key: view.get(key, 'AND' if key == 'combine' else []) for key in ("rules", "combine", "sort")}
    if "sample_set" in view and scope != "all":
        context["sample_set"] = view["sample_set"]
    if scope.startswith('set:') and resolved_ids is not None:
        saved = project.view.get("sample_sets", {}).get(scope[4:], ())
        context["missing_sample_count"] = len(saved) - len(resolved_ids)
    context["applied_to_report"] = scope != "all"
    return context


def edit_sections(sheet):
    dialog = QDialog(sheet)
    dialog.setWindowTitle(tr('Разделы отчёта'))
    layout = QVBoxLayout(dialog)
    instruction = QLabel(tr('Отметьте разделы. Используйте кнопки «Выше» и «Ниже», Alt+↑ и Alt+↓ или перетаскивание.'))
    instruction.setWordWrap(True)
    layout.addWidget(instruction)
    items = QListWidget()
    items.setAccessibleName(tr('Разделы и порядок вывода'))
    items.setDragDropMode(QListWidget.DragDropMode.InternalMove)
    enabled = sheet.section_order
    if enabled is None:
        enabled = tuple(key for key in REPORT_SECTIONS if key not in {"profile", "comparison_figures"})
    titles = dict(sheet.section_titles)
    for key in (*enabled, *(key for key in REPORT_SECTIONS if key not in enabled)):
        item = QListWidgetItem(titles.get(key) or tr(REPORT_SECTIONS[key]), items)
        if key == "joint_curve_fit":
            item.setIcon(icon("chart-no-axes-combined"))
        item.setData(Qt.ItemDataRole.UserRole, key)
        item.setCheckState(Qt.CheckState.Checked if key in enabled else Qt.CheckState.Unchecked)
    layout.addWidget(items)
    actions = QHBoxLayout()
    up = QPushButton(tr('Выше'))
    down = QPushButton(tr('Ниже'))
    rename = QPushButton(tr('Изменить заголовок…'))
    for button in (up, down, rename):
        actions.addWidget(button)
    layout.addLayout(actions)

    def update_move_buttons(row):
        up.setEnabled(row > 0)
        down.setEnabled(0 <= row < items.count() - 1)

    def move_section(offset):
        row = items.currentRow()
        target = row + offset
        if row >= 0 and 0 <= target < items.count():
            items.insertItem(target, items.takeItem(row))
            items.setCurrentRow(target)

    up.clicked.connect(lambda: move_section(-1))
    down.clicked.connect(lambda: move_section(1))
    sheet.report_shortcut(items, ("Alt+Up",), lambda: move_section(-1))
    sheet.report_shortcut(items, ("Alt+Down",), lambda: move_section(1))
    items.currentRowChanged.connect(update_move_buttons)
    items.setCurrentRow(0)
    items.setFocus()

    def rename_section():
        item = items.currentItem()
        if item is None:
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        title, accepted = QInputDialog.getText(
            dialog, tr('Заголовок раздела'), tr('Пустое поле восстанавливает исходный заголовок'),
            text=titles.get(key, ""),
        )
        if accepted:
            if title.strip():
                titles[key] = title.strip()
            else:
                titles.pop(key, None)
            item.setText(titles.get(key) or tr(REPORT_SECTIONS[key]))

    rename.clicked.connect(rename_section)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    if dialog.exec():
        order = tuple(
            items.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(items.count())
            if items.item(i).checkState() == Qt.CheckState.Checked
        )
        if order != tuple(enabled) or titles != sheet.section_titles:
            sheet.section_titles = titles
            sheet.section_order = order
            sheet.invalidate()


def edit_comments(sheet):
    if sheet.preview_report is None:
        sheet.announce_status(tr('Сначала соберите отчёт, чтобы выбрать таблицу или рисунок.'))
        return
    blocks = [(key, title) for _, key, _, title, _ in report_blocks(sheet.preview_report, localized=True) if key is not None]
    if not blocks:
        sheet.announce_status(tr('В отчёте нет таблиц или рисунков для комментариев.'))
        return
    dialog = QDialog(sheet)
    dialog.setWindowTitle(tr('Комментарии к таблицам и рисункам'))
    layout = QVBoxLayout(dialog)
    choice = QComboBox()
    for number, (key, title) in enumerate(blocks, 1):
        choice.addItem(f"{number}. {title}", key)
    layout.addWidget(choice)
    editor = QTextEdit()
    editor.setAcceptRichText(False)
    layout.addWidget(editor)
    comments = dict(sheet.block_comments)
    active = choice.currentData()
    editor.setPlainText(comments.get(active, ""))

    def store():
        comment = editor.toPlainText().strip()
        if comment:
            comments[active] = comment
        else:
            comments.pop(active, None)

    def select_block():
        nonlocal active
        store()
        active = choice.currentData()
        editor.setPlainText(comments.get(active, ""))

    choice.currentIndexChanged.connect(select_block)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    if dialog.exec():
        store()
        if sheet.block_comments != comments:
            sheet.block_comments = comments
            sheet.invalidate()


def choose_logo(sheet):
    path, _ = QFileDialog.getOpenFileName(sheet, tr('Логотип для DOCX'), "", "PNG / JPEG (*.png *.jpg *.jpeg)")
    if not path:
        return
    reader = QImageReader(path)
    size = reader.size()
    if not size.isValid() or size.width() * size.height() > 16_000_000:
        sheet.announce_status(tr('Выберите изображение PNG или JPEG размером до 16 мегапикселей.'))
        return
    image = reader.read()
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if image.isNull() or not image.scaled(800, 350, Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation).save(buffer, "PNG"):
        sheet.announce_status(tr('Не удалось прочитать логотип.'))
        return
    sheet.set_logo(base64.b64encode(bytes(buffer.data())).decode("ascii"))


def set_logo(sheet, image):
    sheet.logo_image = image
    sheet.logo_button.setText(tr('Заменить логотип…') if image else tr('Выбрать логотип…'))
    sheet.invalidate()
