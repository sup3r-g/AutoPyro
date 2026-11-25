"""Human-readable sample labels shared by the interface and frozen reports."""

_ALIASES = {
    "well": {"well", "wellname", "wellbore", "wellborename", "скважина"},
    "sample": {"sample", "samplename", "sampleid", "samplenumber", "проба", "номерпробы", "образец"},
    "depth": {"depth", "depthm", "md", "глубина", "глубинам"},
}


def _sample_column(columns):
    names = {name.strip().casefold(): cid for cid, name in columns}
    return names.get("sample / name") or names.get("sample") or names.get("проба")


def sample_label_column(columns):
    return _sample_column((column.id, column.name) for column in columns)


def sample_label_fields(columns):
    columns = tuple(columns)
    sample = _sample_column(columns)
    fields = {"sample": sample} if sample else {}
    for cid, name in columns:
        normalized = "".join(char for char in name.casefold() if char.isalnum())
        for key, aliases in _ALIASES.items():
            if normalized in aliases:
                fields.setdefault(key, cid)
    return tuple(fields[key] for key in _ALIASES if key in fields)


def sample_label(row_id, values, fields):
    return " · ".join(str(values[cid]) for cid in fields if values.get(cid) not in (None, "")) or row_id
