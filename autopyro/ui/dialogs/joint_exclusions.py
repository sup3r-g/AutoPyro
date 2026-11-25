"""Compact row diagnostics for the joint fit's prepared selection."""

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

from autopyro.model.tabular import classify_series
from ..i18n import tr


def prepare_joint_rows(frame, x_id, y_id, sy_id, series_id, qc_excluded, positive_x):
    fields = (("X", x_id), ("Y", y_id), ("σY", sy_id))
    numbers = {}
    codes = np.zeros((len(frame), 5), dtype=np.uint8)
    status_codes = {"missing": 1, "non_detect": 2, "censored": 3, "other": 4}
    for col, (_, cid) in enumerate(fields):
        if cid is None:
            continue
        parsed = classify_series(frame[cid], numeric=True)
        values = numbers[cid] = parsed["number"].to_numpy(dtype=float)
        invalid = ~np.isfinite(values)
        if invalid.any():
            codes[invalid, col] = parsed.loc[invalid, "status"].map(status_codes).to_numpy(dtype=np.uint8)
        if (col == 0 and positive_x) or col == 2:
            codes[values <= 0, col] = 5
    series = frame[series_id]
    empty = np.fromiter((isinstance(value, str) and not value.strip() for value in series), bool, count=len(series))
    codes[:, 3] = np.where(series.isna().to_numpy() | empty, 1, 0)
    codes[:, 4] = qc_excluded
    rejected = codes.any(axis=1)
    return numbers, ~rejected, frame.index[rejected].tolist(), codes[rejected]


class JointExclusionModel(QAbstractTableModel):
    def __init__(self, row_ids, codes, parent=None):
        super().__init__(parent)
        self.row_ids = row_ids
        self.codes = codes
        self.fields = ("X", "Y", "σY", tr("Серия"))
        self.reasons = ("", tr("Пропуск"), tr("Ниже предела обнаружения (ND)"),
                        tr("Цензурированное значение"), tr("Не числовое значение"),
                        tr("Требуется положительное значение"))
        self.qc = tr("Исключено QC")

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.row_ids)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 2

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Vertical:
                return section+1
            return ("UUID", tr("Причины исключения"))[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            return None
        row = index.row()
        if index.column() == 0:
            return self.row_ids[row]
        codes = self.codes[row]
        reasons = [f"{field} · {self.reasons[code]}" for field, code in zip(self.fields, codes) if code]
        if codes[-1]:
            reasons.append(self.qc)
        return " / ".join(reasons)
