"""Geometry worksheet and its public dialog interface."""

from __future__ import annotations

from copy import deepcopy

from PySide6.QtCore import Qt
from PySide6.QtGui import QUndoStack
from PySide6.QtWidgets import QDialog

from autopyro.schemes.scheme_geometry import next_revision

from ..i18n import tr
from ..widgets.navigation import NavigationToolbar as GeometryToolbar
from . import (
    geometry_editor_draft,
    geometry_editor_forms,
    geometry_editor_interaction,
    geometry_editor_objects,
    geometry_editor_preview,
)
from .geometry_editor_draft import DraftCommand as DraftCommand
from .geometry_editor_new_scheme import NewSchemeDialog as NewSchemeDialog, new_scheme as new_scheme


class GeometryEditor(QDialog):
    """Compose the private scheme draft, geometry controls and preview."""

    def __init__(self, scheme, parent=None, *, samples=(), new=False):
        super().__init__(parent)
        self.original = deepcopy(scheme)
        self.is_new = new
        self.draft = deepcopy(scheme)
        if not new:
            self.draft["revision"] = next_revision(scheme["revision"])
        self.samples = list(samples)
        self._source_samples = list(samples)
        self.preview_results = []
        self.undo_stack = QUndoStack(self)
        self.selected = None
        self.selected_path = None
        self.drawing_points = []
        self.drawing_target = None
        self.drag = None
        self.loading = False
        self.edited_scheme = None
        self.setWindowTitle(tr('Графический редактор схемы · координаты analysis'))
        self.resize(1320, 880)
        geometry_editor_forms.build(self, parent)

    def scope_samples_changed(self):
        return geometry_editor_preview.scope_samples_changed(self)

    def restore_parameters(self):
        return geometry_editor_preview.restore_parameters(self)

    def push(self, after, label):
        return geometry_editor_draft.push(self, after, label)

    def set_draft(self, scheme):
        return geometry_editor_draft.set_draft(self, scheme)

    def metadata_changed(self, *args):
        return geometry_editor_draft.metadata_changed(self, *args)

    def current_entry(self):
        return geometry_editor_objects.current_entry(self)

    def path_index(self, path):
        return geometry_editor_objects.path_index(self, path)

    def select_object(self, *args):
        return geometry_editor_objects.select_object(self, *args)

    def select_path(self, *args):
        return geometry_editor_objects.select_path(self, *args)

    def fill_table(self, values=None):
        return geometry_editor_objects.fill_table(self, values)

    def table_changed(self, item):
        return geometry_editor_objects.table_changed(self, item)

    def vertex_selection_changed(self):
        return geometry_editor_objects.vertex_selection_changed(self)

    def choose_color(self):
        return geometry_editor_objects.choose_color(self)

    def object_properties_changed(self):
        return geometry_editor_objects.object_properties_changed(self)

    def edit_fields(self, title, fields):
        return geometry_editor_objects.edit_fields(self, title, fields)

    def edit_region_passport(self):
        return geometry_editor_objects.edit_region_passport(self)

    def edit_coverage_domain(self):
        return geometry_editor_objects.edit_coverage_domain(self)

    def threshold_changed(self):
        return geometry_editor_objects.threshold_changed(self)

    def insert_vertex(self):
        return geometry_editor_objects.insert_vertex(self)

    def delete_vertex(self):
        return geometry_editor_objects.delete_vertex(self)

    def delete_path(self):
        return geometry_editor_objects.delete_path(self)

    def delete_object(self):
        return geometry_editor_objects.delete_object(self)

    def mode_changed(self, *args):
        return geometry_editor_interaction.mode_changed(self, *args)

    def cancel_drawing(self):
        return geometry_editor_interaction.cancel_drawing(self)

    def finish_drawing(self):
        return geometry_editor_interaction.finish_drawing(self)

    def on_press(self, event):
        return geometry_editor_interaction.on_press(self, event)

    def on_motion(self, event):
        return geometry_editor_interaction.on_motion(self, event)

    def on_release(self, event):
        return geometry_editor_interaction.on_release(self, event)

    def on_key(self, event):
        return geometry_editor_interaction.on_key(self, event)

    def redraw(self, *args, preview_only=False):
        return geometry_editor_preview.redraw(self, *args, preview_only=preview_only)

    def draw_coverage(self, coverage):
        return geometry_editor_preview.draw_coverage(self, coverage)

    def coverage_note(self, diagnostics):
        return geometry_editor_preview.coverage_note(self, diagnostics)

    def fit_view(self):
        return geometry_editor_preview.fit_view(self)

    def update_limit_fields(self):
        return geometry_editor_preview.update_limit_fields(self)

    def apply_limits(self):
        return geometry_editor_preview.apply_limits(self)

    def apply(self):
        return geometry_editor_draft.apply(self)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and (self.drawing_points or self.drag):
            self.cancel_drawing()
            event.accept()
        elif (
            event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.drawing_points
        ):
            self.finish_drawing()
            event.accept()
        else:
            super().keyPressEvent(event)
