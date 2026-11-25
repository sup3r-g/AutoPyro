"""Reusing column ID strings must not change row data, order or validation."""
from contextlib import closing
from dataclasses import asdict
import json
import sqlite3

import pytest

from autopyro.core import Project
from autopyro.io import build_project


@pytest.fixture
def project():
    headers = ["Sample", "TOC", "S2"] + [f"Property {i}" for i in range(40)]
    raw = [["001", "2.00", "ND"] + [str(i) for i in range(40)],
           ["001", "3.00", "<0.1"] + [str(i + 1) for i in range(40)]]
    project = build_project(headers, raw, [2, 3],
                            [(i, name, "text" if i == 0 else "number", "%wt" if i == 1 else "")
                             for i, name in enumerate(headers)], {})
    project.rows[0].original["retired-column-id"] = {"notes": ["preserved"]}
    project.sync_measurements()
    return project


@pytest.mark.parametrize("version", [2, 3, 4])
def test_loaded_row_maps_share_known_column_keys_but_not_mutable_data(project, tmp_path, write_v2_project, write_v3_project, version):
    path = tmp_path / "wide.apyro"
    if version == 2:
        write_v2_project(project, path)
    elif version == 3:
        write_v3_project(project, path)
    else:
        project.save(path)
    before = path.read_bytes()
    loaded = Project.load(path)
    assert asdict(loaded) == asdict(project) and path.read_bytes() == before
    ids = {column.id: column.id for column in loaded.columns}
    for row, original in zip(loaded.rows, project.rows, strict=True):
        for name in ("values", "original"):
            values = getattr(row, name)
            assert list(values) == list(getattr(original, name))
            assert all(key is ids[key] for key in values if key in ids)
        assert row.values is not row.original
    assert loaded.rows[0].values is not loaded.rows[1].values
    loaded.rows[0].original["retired-column-id"]["notes"].append("changed")
    assert project.rows[0].original["retired-column-id"] == {"notes": ["preserved"]}
    toc = loaded.columns[1].id
    loaded.rows[0].values[toc] = 99
    assert loaded.rows[0].original[toc] == "2.00"
    assert loaded.rows[1].values[toc] == 3
    assert loaded.science.current(loaded.rows[0].id, toc).value == 2
    assert path.read_bytes() == before


@pytest.mark.parametrize("damage", ["missing", "unknown"])
def test_column_key_reuse_does_not_repair_invalid_row_composition(project, tmp_path, damage):
    path = tmp_path / "invalid.apyro"
    project.save(path)
    with closing(sqlite3.connect(path)) as connection:
        rid, payload = connection.execute("SELECT id,payload FROM table_rows ORDER BY position LIMIT 1").fetchone()
        record = json.loads(payload)
        if damage == "missing":
            record["values"].pop(project.columns[1].id)
        else:
            record["values"]["unknown-column-id"] = 7
        connection.execute("UPDATE table_rows SET payload=? WHERE id=?", (json.dumps(record), rid))
        connection.commit()
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Состав строки"):
        Project.load(path)
    assert path.read_bytes() == before
