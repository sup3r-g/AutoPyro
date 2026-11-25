"""Versioned JSON templates. Header aliases are explicit; positions never resolve ambiguity."""
from copy import deepcopy
import hashlib
import json
import os
import tempfile
from pathlib import Path
import re

from autopyro.model.core import new_id
from autopyro.model.field_types import FIELD_KINDS
from .importing import ANALYSIS_FIELDS
from autopyro.model.measurements import derived

SETTINGS = {"sheet", "delimiter", "encoding", "header_rows", "header_row", "data_end_row",
            "first_column", "last_column", "decimal_separator"}
TEMPLATE_KEYS = {"format", "version", "id", "revision", "name", "stage", "file_type", "settings",
                 "columns", "keys", "bindings", "defaults", "working_policy"}


def validate(template):
    if not isinstance(template, dict) or set(template) - TEMPLATE_KEYS:
        raise ValueError("Неизвестные поля шаблона")
    if template.get("format") != "autopyro.import-template" or type(template.get("version")) is not int or template.get("version") != 1:
        raise ValueError("Неизвестный формат или версия шаблона")
    for key in ("id", "name"):
        if not isinstance(template.get(key), str) or not template[key].strip():
            raise ValueError(f"Не задано поле шаблона: {key}")
    if type(template.get("revision")) is not int or template["revision"] < 1:
        raise ValueError("Некорректная редакция шаблона")
    stage = template.get("stage")
    if stage not in ("reader", "append") or not isinstance(template.get("columns"), list):
        raise ValueError("Неизвестный этап или состав колонок")
    common = {"format", "version", "id", "revision", "name", "stage", "columns"}
    if set(template) - (common | ({"file_type", "settings"} if stage == "reader" else {"keys", "bindings", "defaults", "working_policy"})):
        raise ValueError("Параметры относятся к другому этапу импорта")
    required = common | ({"file_type", "settings"} if stage == "reader" else {"keys", "bindings", "defaults", "working_policy"})
    if required - set(template):
        raise ValueError("Отсутствуют обязательные поля шаблона")
    if stage == "reader":
        if template.get("file_type") not in ("delimited", "xlsx"):
            raise ValueError("Неизвестный тип источника шаблона")
        settings = template.get("settings", {})
        if not isinstance(settings, dict) or set(settings) - SETTINGS:
            raise ValueError("Недопустимые параметры чтения")
        if settings.get("encoding", "utf-8-sig") not in ("utf-8-sig", "utf-8", "cp1251", "utf-16"):
            raise ValueError("Неподдерживаемая кодировка")
        if settings.get("delimiter", ";") not in (";", ",", "\t", "|") or settings.get("decimal_separator", "auto") not in ("auto", ",", "."):
            raise ValueError("Недопустимый разделитель или десятичный знак")
        for key in ("header_row", "first_column", "last_column", "data_end_row"):
            value = settings.get(key)
            if key in settings and key in ("header_row", "first_column") and value is None:
                raise ValueError("Начало диапазона не может быть пустым")
            if value is not None and (type(value) is not int or not 1 <= value <= 2_000_000_000):
                raise ValueError("Диапазоны шаблона должны быть положительными целыми числами")
        if type(settings.get("header_rows", 1)) is not int or settings.get("header_rows", 1) not in (1, 2):
            raise ValueError("Поддерживаются один или два уровня заголовков")
        if settings.get("last_column") is not None and settings["last_column"] < settings.get("first_column", 1):
            raise ValueError("Некорректный диапазон колонок")
        if settings.get("data_end_row") is not None and settings["data_end_row"] < settings.get("header_row", 1) + settings.get("header_rows", 1) - 1:
            raise ValueError("Конец диапазона выше заголовка")
        if settings.get("sheet") is not None and not isinstance(settings["sheet"], str):
            raise ValueError("Название листа должно быть строкой")
        if template["file_type"] == "xlsx" and settings.get("decimal_separator", "auto") != "auto":
            raise ValueError("Десятичный знак задаётся только для CSV/TSV")
        if template["file_type"] == "delimited" and settings.get("sheet") is not None:
            raise ValueError("У CSV/TSV нет листов Excel")
    elif template.get("working_policy") not in ("keep", "fill_missing", "replace"):
        raise ValueError("Неизвестная политика рабочего результата")
    for spec in template["columns"]:
        allowed = {"header", "aliases"} | ({"selected", "name", "kind", "unit"} if stage == "reader" else {"target"})
        if not isinstance(spec, dict) or set(spec) - allowed or not isinstance(spec.get("header"), str) or not spec["header"]:
            raise ValueError("Некорректное описание исходной колонки")
        if re.fullmatch(r"(?:.* / )?Колонка \d+", spec["header"]):
            raise ValueError("Для надёжного шаблона нужны явные заголовки; автоматически названные колонки сопоставьте вручную")
        if not isinstance(spec.get("aliases", []), list) or not all(isinstance(a, str) and a for a in spec.get("aliases", [])):
            raise ValueError("Синонимы должны быть списком непустых заголовков")
        if stage == "reader":
            if type(spec.get("selected")) is not bool or spec.get("kind") not in FIELD_KINDS or not isinstance(spec.get("unit"), str) or not isinstance(spec.get("name"), str) or not spec["name"].strip():
                raise ValueError("Для колонки задайте выбор, имя, тип и единицу")
        else:
            target = spec.get("target")
            if target is not None and target != "new":
                if not isinstance(target, dict) or set(target) != {"id", "name", "kind", "unit", "basis"}:
                    raise ValueError("Недопустимый целевой показатель")
                if not all(isinstance(v, str) for v in target.values()) or target["kind"] not in FIELD_KINDS or not target["name"].strip():
                    raise ValueError("Некорректная семантика целевого показателя")
    if stage == "reader":
        names = [c["name"].strip() for c in template["columns"] if c["selected"]]
        if not names or len(names) != len(set(names)):
            raise ValueError("Выберите колонки с непустыми уникальными именами")
    if stage == "append":
        size = len(template["columns"])
        keys = template.get("keys", [])
        bindings = template.get("bindings", {})
        if not isinstance(keys, list) or any(type(i) is not int or not 0 <= i < size for i in keys) or len(set(keys)) != len(keys):
            raise ValueError("Некорректные ключи шаблона")
        if not isinstance(bindings, dict) or set(bindings) - set(ANALYSIS_FIELDS) or any(type(i) is not int or not 0 <= i < size for i in bindings.values()):
            raise ValueError("Некорректные привязки аналитического контекста")
        defaults = template.get("defaults", {})
        if not isinstance(defaults, dict) or set(defaults) - set(ANALYSIS_FIELDS) or not all(isinstance(v, str) for v in defaults.values()):
            raise ValueError("Некорректные общие параметры анализа")
        if any(not isinstance(template["columns"][i].get("target"), dict) for i in keys):
            raise ValueError("Ключ шаблона должен ссылаться на существующий показатель")
    return template


def template_hash(template):
    validate(template)
    return hashlib.sha256(json.dumps(template, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def reference(template):
    return {key: template[key] for key in ("id", "revision", "name", "stage")} | {"sha256": template_hash(template)}


def library_key(template):
    return template["stage"] + ":" + template["name"]


def store(view, template, *, activate=True):
    template = deepcopy(validate(template))
    revisions = view.get("import_template_revisions", {}).get(template["id"], {})
    old = revisions.get(str(template["revision"]))
    if old is not None and template_hash(old) != template_hash(template):
        raise ValueError("Эта редакция шаблона уже существует с другим содержимым")
    view.setdefault("import_template_revisions", {}).setdefault(template["id"], {})[str(template["revision"])] = template
    if activate:
        view.setdefault("import_templates", {})[library_key(template)] = deepcopy(template)


def merge_views(target, source, *, activate=False):
    if not any(source.get(key) for key in ("import_templates", "import_template_revisions")):
        return
    staged = {key: deepcopy(target.get(key, {})) for key in ("import_templates", "import_template_revisions")}
    for revisions in source.get("import_template_revisions", {}).values():
        for template in revisions.values():
            store(staged, template, activate=False)
    for template in source.get("import_templates", {}).values():
        store(staged, template, activate=activate or library_key(template) not in staged.get("import_templates", {}))
    for key in ("import_templates", "import_template_revisions"):
        if key in staged:
            target[key] = staged[key]


def create(name, stage, view, **fields):
    old = view.get("import_templates", {}).get(stage + ":" + name.strip())
    revisions = view.get("import_template_revisions", {}).get(old["id"], {}) if old else {}
    revision = max([old["revision"] if old else 0] + [t["revision"] for t in revisions.values()]) + 1
    template = {"format": "autopyro.import-template", "version": 1, "id": old["id"] if old else new_id(),
                "revision": revision, "name": name.strip(), "stage": stage, **deepcopy(fields)}
    validate(template)
    return template


def resolve_sources(template, headers):
    """Return template-column index → source-column index, or fail before applying anything."""
    validate(template)
    index = {}
    for i, header in enumerate(headers):
        index.setdefault(header, []).append(i)
    resolved, used = {}, set()
    for number, spec in enumerate(template["columns"]):
        matches = {i for label in [spec["header"], *spec.get("aliases", [])] for i in index.get(label, [])}
        required = spec.get("selected", True) if template["stage"] == "reader" else (
            spec.get("target") is not None or number in template.get("keys", []) or number in template.get("bindings", {}).values())
        if not required:
            continue
        if len(matches) != 1:
            raise ValueError(f"{spec['header']}: {'нет соответствующей колонки' if not matches else 'неоднозначные заголовки/синонимы'}. Настройте файл вручную")
        match = matches.pop()
        if match in used:
            raise ValueError("Несколько полей шаблона соответствуют одной колонке источника")
        used.add(match)
        resolved[number] = match
    return resolved


def resolve_append(template, project, incoming):
    indices = resolve_sources(template, [c.metadata.get("source_header", c.name) for c in incoming.columns])
    mapping = {c.id: None for c in incoming.columns}
    for i, j in indices.items():
        target = template["columns"][i]["target"]
        source = incoming.columns[j]
        if isinstance(target, dict):
            matches = [c for c in project.columns if not derived(c) and c.name == target["name"] and
                       c.kind == target["kind"] and c.unit == target["unit"] and c.metadata.get("basis", "") == target["basis"]]
            same_id = [c for c in matches if c.id == target["id"]]
            if any(c.id == target["id"] for c in project.columns) and not same_id:
                raise ValueError(f"{target['name']}: определение сохранённого показателя изменилось")
            if len(same_id or matches) != 1:
                raise ValueError(f"{target['name']}: целевой показатель отсутствует или неоднозначен. Настройте сопоставление вручную")
            target = (same_id or matches)[0]
            if source.unit != target.unit or source.metadata.get("basis", "") != target.metadata.get("basis", ""):
                raise ValueError(f"{source.name}: единица или основание не соответствуют шаблону")
            target = target.id
        mapping[source.id] = target
    return {"mapping": mapping, "keys": [incoming.columns[indices[i]].id for i in template["keys"]],
            "bindings": {name: incoming.columns[indices[i]].id for name, i in template["bindings"].items()},
            "defaults": deepcopy(template["defaults"]), "working_policy": template["working_policy"]}


def load_template(path):
    return deepcopy(validate(json.loads(Path(path).read_text(encoding="utf-8-sig"))))


def save_template(path, template):
    payload = json.dumps(validate(template), ensure_ascii=False, indent=2, allow_nan=False)
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
