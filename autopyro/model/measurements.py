"""Scientific records behind the editable table; no Qt or database dependencies.

An old row is one sample, never a claim that equal sample names are identical.
Properties retain column UUIDs. Derived columns are views, not measurements.
"""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from functools import cache
from hashlib import sha1
from itertools import batched, groupby
from operator import itemgetter
from uuid import NAMESPACE_URL, uuid4

from .field_types import TYPED_KINDS, parse_typed_value
from .work_batches import cooperative_items


_URL_NAMESPACE_BYTES = NAMESPACE_URL.bytes
_SCALAR_TYPES = frozenset((str, int, float, bool, type(None)))


def identifier():
    return str(uuid4())


def migrated_id(project_id, kind, *parts):
    """Stable even when an unmodified v1 file is opened in two sessions."""
    name = ":".join(("autopyro", project_id, kind, *parts)).encode("utf-8")
    digest = sha1(_URL_NAMESPACE_BYTES + name).hexdigest()
    # UUIDv5 uses the first 128 bits. Set the version and RFC variant nibbles.
    variant = "89ab"[int(digest[16], 16) & 3]
    return f"{digest[:8]}-{digest[8:12]}-5{digest[13:16]}-{variant}{digest[17:20]}-{digest[20:32]}"


def snapshot_mapping(values):
    """Copy a mapping, sharing only exact immutable scalars in flat dictionaries."""
    if (type(values) is dict
            and set(map(type, values)).issubset(_SCALAR_TYPES)
            and set(map(type, values.values())).issubset(_SCALAR_TYPES)):
        return values.copy()
    return deepcopy(values)


def now():
    return datetime.now(timezone.utc).isoformat()


def derived(column):
    return any(column.metadata.get(key) for key in
               ("method", "calculation", "stratigraphy", "nonpyro", "curve_fit", "joint_curve_fit"))


@dataclass
class Sample:
    id: str
    metadata: dict = field(default_factory=dict)

    def __reduce_ex__(self, protocol):
        # Most imported samples have no metadata. Reconstruct them directly
        # without retaining both the copied state dict and the instance dict.
        # The ordinary traversal still copies and memoizes the metadata.
        if (type(self) is Sample and tuple(vars(self)) == ("id", "metadata")
                and type(self.id) is str and type(self.metadata) is dict and not self.metadata):
            return Sample, (self.id, self.metadata)
        return super().__reduce_ex__(protocol)


@dataclass
class PropertyDefinition:
    id: str
    name: str
    kind: str
    unit: str = ""
    basis: str = ""


@dataclass
class SourceRecord:
    id: str
    provenance: dict = field(default_factory=dict)
    original: dict = field(default_factory=dict)


@dataclass
class Analysis:
    sample_id: str
    source_id: str | None = None
    id: str = field(default_factory=identifier)
    method: str = ""
    laboratory: str = ""
    instrument: str = ""
    heating_program: str = ""
    preparation: str = ""
    fraction: str = ""
    date: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class Measurement:
    sample_id: str
    property_id: str
    analysis_id: str
    raw_value: object = None
    value: object = None
    qualifier: str = "missing"
    unit: str = ""
    basis: str = ""
    limit: float | int | None = None
    comparator: str = ""
    lod: float | None = None
    loq: float | None = None
    uncertainty: float | None = None
    uncertainty_type: str = ""
    confidence_level: float | None = None
    qc_status: str = "unchecked"
    source_id: str | None = None
    id: str = field(default_factory=identifier)
    revision: int = 1
    created_at: str = field(default_factory=now)
    history: list[dict] = field(default_factory=list)

    def table_value(self):
        # Keep qualifier spelling, leading zeros and explicit text columns literal.
        qualifier = self.qualifier
        if qualifier == "exact":
            return self.value
        if qualifier in ("below_limit", "above_limit", "non_detect", "approximate", "unparsed"):
            return self.raw_value
        return self.value


def parsed_table_value(result):
    """Project parsed input without creating a temporary measurement record."""
    if result["qualifier"] in ("below_limit", "above_limit", "non_detect", "approximate", "unparsed"):
        return result["raw_value"]
    return result["value"]


def parse_number(value):
    """Parse a finite number, retaining arbitrary-precision source integers."""
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, bool):
        raise ValueError("Логическое значение не является измерением")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        number = value
    else:
        text = str(value).strip().replace("\u2212", "-")
        try:
            return int(text)
        except ValueError:
            number = float(text.replace(",", "."))
    if not math.isfinite(number):
        raise ValueError("NaN и бесконечность недопустимы")
    return number


def parse_result(raw, kind):
    if raw is None or isinstance(raw, str) and not raw.strip():
        return {"raw_value": raw, "value": None, "qualifier": "missing", "limit": None, "comparator": ""}
    if kind in TYPED_KINDS:
        value = parse_typed_value(raw, kind)
        return {"raw_value": raw, "value": value, "qualifier": "exact", "limit": None, "comparator": ""}
    if kind == "number" and type(raw) in (int, float):
        return {"raw_value": raw, "value": parse_number(raw), "qualifier": "exact", "limit": None, "comparator": ""}
    text = str(raw).strip()
    result = {"raw_value": raw, "value": None, "limit": None, "comparator": ""}
    if text.casefold() in {"nd", "n.d.", "n/d", "н/о", "не обнаружено"}:
        return {**result, "qualifier": "non_detect"}
    for operator in ("<=", ">=", "≤", "≥", "<", ">"):
        if text.startswith(operator):
            try:
                threshold = parse_number(text[len(operator):])
                if threshold is None or threshold < 0:
                    raise ValueError("Недопустимый предел")
            except (TypeError, ValueError):
                return {**result, "qualifier": "unparsed"}
            return {**result, "qualifier": "below_limit" if operator[0] in "<≤" else "above_limit",
                    "limit": threshold, "comparator": operator}
    if text.startswith(("~", "≈")):
        try:
            value = parse_number(text[1:])
            if value is None:
                raise ValueError("Нет значения")
            return {**result, "value": value, "qualifier": "approximate"}
        except (TypeError, ValueError):
            return {**result, "qualifier": "unparsed"}
    value = parse_number(raw) if kind == "number" else raw
    return {**result, "value": value, "qualifier": "exact"}


def symbolic_lod_operator(raw):
    if not isinstance(raw, str):
        return None
    return {"<lod": "<", "<=lod": "<=", "≤lod": "≤"}.get("".join(raw.split()).casefold())


def _loaded_reference(value, records):
    if type(value) is not str:
        return value
    record = records.get(value)
    return record.id if record is not None and record.id == value else value


@cache
def _snapshot_field_names(record_type):
    return tuple(definition.name for definition in fields(record_type))


_MEASUREMENT_SNAPSHOT_FIELDS = frozenset(_snapshot_field_names(Measurement))


def snapshot_record(record):
    """Copy declared scientific fields, sharing only immutable built-in scalars.

    A new field is automatically included. One memo preserves relationships
    between nested fields while isolating them from the original record.
    """
    record_type = type(record)
    if record_type is Measurement and vars(record).keys() == _MEASUREMENT_SNAPSHOT_FIELDS:
        values = vars(record).copy()
        history = values.pop("history")
        if type(history) is list and not history and set(map(type, values.values())).issubset(_SCALAR_TYPES):
            return Measurement(**values, history=[])

    values = {}
    memo = {}
    for name in _snapshot_field_names(record_type):
        value = getattr(record, name)
        if type(value) not in _SCALAR_TYPES:
            value = deepcopy(value, memo)
        values[name] = value
    return record_type(**values)


@dataclass
class MeasurementStore:
    samples: dict[str, Sample] = field(default_factory=dict)
    properties: dict[str, PropertyDefinition] = field(default_factory=dict)
    sources: dict[str, SourceRecord] = field(default_factory=dict)
    analyses: dict[str, Analysis] = field(default_factory=dict)
    measurements: dict[str, Measurement] = field(default_factory=dict)
    working: dict[str, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload):
        names = ("samples", "properties", "sources", "analyses", "measurements")
        return cls.from_payloads({name: payload[name].items() for name in names}, payload["working"])

    @classmethod
    def from_payloads(cls, collections, working):
        """Build records from (id, field-dict) iterables and own the working selections."""
        def restore(name, record_type, share=None):
            records = {}
            # SQLite stores measurements in declaration order. Older or
            # externally edited records can still use the keyword constructor.
            names = _snapshot_field_names(Measurement) if record_type is Measurement else None
            for key, value in collections[name]:
                if names is not None and type(value) is dict and tuple(value) == names:
                    record = record_type(*value.values())
                else:
                    record = record_type(**value)
                if type(key) is str and record.id == key:
                    record.id = key
                if share is not None:
                    share(record)
                records[key] = record
            return records

        # Restore dependencies first so duplicate strings can be released per record.
        # Unknown references and mismatched keys remain intact for validation.
        reference = _loaded_reference
        samples = restore("samples", Sample)
        properties = restore("properties", PropertyDefinition)

        def share_source(source):
            source.original = {reference(pid, properties): value for pid, value in source.original.items()}

        sources = restore("sources", SourceRecord, share_source)

        def share_analysis(analysis):
            analysis.sample_id = reference(analysis.sample_id, samples)
            analysis.source_id = reference(analysis.source_id, sources)

        analyses = restore("analyses", Analysis, share_analysis)

        texts = {value: value for value in ("exact", "missing", "below_limit", "above_limit",
                 "non_detect", "approximate", "unparsed", "unchecked")}
        for prop in properties.values():
            for value in (prop.unit, prop.basis):
                if type(value) is str:
                    texts.setdefault(value, value)

        def shared_text(value):
            return texts.get(value, value) if type(value) is str else value

        def share_measurement(measurement):
            measurement.sample_id = reference(measurement.sample_id, samples)
            measurement.property_id = reference(measurement.property_id, properties)
            measurement.analysis_id = reference(measurement.analysis_id, analyses)
            measurement.source_id = reference(measurement.source_id, sources)
            measurement.qualifier = shared_text(measurement.qualifier)
            measurement.qc_status = shared_text(measurement.qc_status)
            measurement.unit = shared_text(measurement.unit)
            measurement.basis = shared_text(measurement.basis)
        measurements = restore("measurements", Measurement, share_measurement)
        working = {
            reference(sid, samples): {reference(pid, properties): reference(mid, measurements)
                                      for pid, mid in selected.items()}
            for sid, selected in deepcopy(working).items()
        }
        return cls(samples=samples, properties=properties, sources=sources, analyses=analyses,
                   measurements=measurements, working=working)

    @classmethod
    def from_sqlite_payloads(cls, collections, working_rows):
        """Restore references while streaming the working selections into their final map."""
        store = cls.from_payloads(collections, {})
        reference = _loaded_reference
        for sid, rows in groupby(working_rows, key=itemgetter(0)):
            selected = store.working.setdefault(reference(sid, store.samples), {})
            for _, pid, mid in rows:
                selected[reference(pid, store.properties)] = reference(mid, store.measurements)
        return store

    def current(self, sample_id, property_id):
        mid = self.working.get(sample_id, {}).get(property_id)
        return self.measurements.get(mid)

    def results(self, sample_id, property_id):
        return [m for m in self.measurements.values() if m.sample_id == sample_id and m.property_id == property_id]

    def subset(self, sample_ids, property_ids=None, *, measurement_ids=None):
        """Freeze selected records and their provenance without leaking other samples."""
        return next(self.subset_steps(sample_ids, property_ids, measurement_ids=measurement_ids))

    def subset_steps(self, sample_ids, property_ids=None, *, batch_size=None, measurement_ids=None):
        """Yield checkpoints and then the isolated measurement store."""
        sample_ids = set(sample_ids)
        property_ids = set(self.properties) if property_ids is None else set(property_ids)
        def batches(items):
            return batched(items, batch_size) if batch_size else (items,)

        measurements = {}
        analysis_ids, source_ids = set(), set()
        candidates = (self.measurements.items() if measurement_ids is None else
                      ((mid, self.measurements[mid]) for mid in measurement_ids))
        for batch in batches(candidates):
            for mid, measurement in batch:
                if measurement.sample_id in sample_ids and measurement.property_id in property_ids:
                    measurements[mid] = snapshot_record(measurement)
                    analysis_ids.add(measurement.analysis_id)
                    source_ids.add(measurement.source_id)
            if batch_size and len(batch) == batch_size:
                yield None

        analyses = {}
        for batch in batches(analysis_ids):
            analyses.update((aid, snapshot_record(self.analyses[aid])) for aid in batch)
            if batch_size and len(batch) == batch_size:
                yield None
        source_ids.update(a.source_id for a in analyses.values())
        sources = {}
        source_records = (self.sources.items() if measurement_ids is None else
                          ((sid, self.sources[sid]) for sid in source_ids if sid in self.sources))
        for batch in batches(source_records):
            sources.update((sid, snapshot_record(replace(source, original={
                pid: value for pid, value in source.original.items() if pid in property_ids
            }))) for sid, source in batch if sid in source_ids)
            if batch_size and len(batch) == batch_size:
                yield None
        samples = {}
        sample_records = (self.samples.items() if measurement_ids is None else
                          ((sid, self.samples[sid]) for sid in sample_ids if sid in self.samples))
        for batch in batches(sample_records):
            samples.update((sid, snapshot_record(sample)) for sid, sample in batch if sid in sample_ids)
            if batch_size and len(batch) == batch_size:
                yield None
        working = {}
        working_records = (self.working.items() if measurement_ids is None else
                           ((sid, self.working[sid]) for sid in sample_ids if sid in self.working))
        for batch in batches(working_records):
            working.update((sid, {pid: mid for pid, mid in selected.items() if pid in property_ids})
                           for sid, selected in batch if sid in sample_ids)
            if batch_size and len(batch) == batch_size:
                yield None
        yield MeasurementStore(
            samples=samples,
            properties={pid: snapshot_record(prop) for pid, prop in self.properties.items() if pid in property_ids},
            sources=sources, analyses=analyses, measurements=measurements, working=working)

    def sync(self, project, *, rows=None, edited_cells=None):
        """Bridge legacy API writes. Never deduplicate rows or fabricate methods.

        An explicit edited-cell map limits synchronization for registered samples.
        New samples still import all original columns and provenance.
        Removed columns' records remain archived for undo and traceability.
        """
        originals = [
            PropertyDefinition(c.id, c.name, c.kind, c.unit, c.metadata.get("basis", ""))
            for c in project.columns if not derived(c)
        ]
        for column in originals:
            self.properties[column.id] = column
        originals_by_id = {column.id: column for column in originals} if edited_cells is not None else {}
        primary = None
        known_results = None
        for row in project.rows if rows is None else rows:
            columns = originals
            if row.id not in self.samples:
                self.samples[row.id] = Sample(row.id)
                source = SourceRecord(migrated_id(project.id, "source", row.id),
                                      snapshot_mapping(row.source), snapshot_mapping(row.original))
                self.sources[source.id] = source
                # A migrated row is an unclassified source event, not a known laboratory method.
                analysis = Analysis(row.id, source_id=source.id, id=migrated_id(project.id, "analysis", row.id),
                                    metadata={"origin": "table_row", "analysis_runs": deepcopy(row.source.get("analysis_runs", {}))})
                self.analyses[analysis.id] = analysis
                if primary is not None:
                    primary[row.id] = analysis
            elif edited_cells is not None:
                columns = (originals_by_id[cid] for cid in edited_cells.get(row.id, ()) if cid in originals_by_id)
            selected = self.working.setdefault(row.id, {})
            measurements = self.measurements
            values = row.values
            for column in columns:
                column_id = column.id
                current = measurements.get(selected.get(column_id))
                raw = values.get(column_id)
                if current is None:
                    # A property absent from a source is not an analysis with a missing result.
                    if raw is None:
                        if column_id not in row.original:
                            continue
                        if known_results is None:
                            known_results = {(m.sample_id, m.property_id) for m in self.measurements.values()}
                        if (row.id, column_id) in known_results:
                            continue
                    if primary is None:
                        primary = {a.sample_id: a for a in self.analyses.values() if a.metadata.get("origin") == "table_row"}
                    analysis = primary[row.id]
                    imported = row.original.get(column_id, raw)
                    try:
                        parsed = parse_result(imported, column.kind)
                        if parsed_table_value(parsed) != raw:
                            parsed = parse_result(raw, column.kind)
                    except (TypeError, ValueError):
                        parsed = parse_result(raw, column.kind)
                    measurement = Measurement(row.id, column_id, analysis.id, unit=column.unit,
                                              basis=column.basis, source_id=analysis.source_id,
                                              id=migrated_id(project.id, "measurement", row.id, column_id),
                                              **parsed)
                    # The immutable source retains exact imported text separately from the editable result.
                    self.measurements[measurement.id] = measurement
                    selected[column_id] = measurement.id
                elif current.table_value() != raw:
                    self.set_value(row.id, column_id, raw, column.kind)

    def set_value(self, sample_id, property_id, value, kind):
        measurement = self.current(sample_id, property_id)
        if measurement is None:
            raise ValueError("Рабочее измерение отсутствует")
        parsed = parse_result(value, kind)
        if measurement.table_value() == value:
            return
        previous = {key: deepcopy(value) for key, value in vars(measurement).items() if key != "history"}
        measurement.history.append({"at": now(), "previous": previous})
        for key, item in parsed.items():
            setattr(measurement, key, item)
        measurement.revision += 1

    def prepare_result(self, sample_id, property_id, raw, *, analysis_fields=None, details=None):
        if sample_id not in self.samples or property_id not in self.properties:
            raise ValueError("Неизвестная проба или показатель")
        definition = self.properties[property_id]
        analysis = Analysis(sample_id, **(analysis_fields or {}))
        allowed = {"basis", "lod", "loq", "uncertainty", "uncertainty_type", "confidence_level", "qc_status"}
        details = details or {}
        if set(details) - allowed:
            raise ValueError("Неизвестные метаданные измерения")
        details = {"basis": definition.basis, **details}
        measurement = Measurement(sample_id, property_id, analysis.id, unit=definition.unit,
                                  **parse_result(raw, definition.kind), **details)
        self._validate_measurement(measurement)
        return measurement, analysis

    def add_result(self, sample_id, property_id, raw, *, analysis_fields=None, details=None):
        measurement, analysis = self.prepare_result(sample_id, property_id, raw,
                                                    analysis_fields=analysis_fields, details=details)
        self.analyses[analysis.id] = analysis
        self.measurements[measurement.id] = measurement
        return measurement

    def select(self, sample_id, property_id, measurement_id):
        measurement = self.measurements.get(measurement_id)
        if measurement is None or (measurement.sample_id, measurement.property_id) != (sample_id, property_id):
            raise ValueError("Измерение относится к другой пробе или показателю")
        self.working.setdefault(sample_id, {})[property_id] = measurement_id

    @staticmethod
    def _validate_measurement(measurement):
        if measurement.qualifier not in {"exact", "missing", "below_limit", "above_limit", "non_detect", "approximate", "unparsed"}:
            raise ValueError("Неизвестный квалификатор")
        if (measurement.lod is not None or measurement.loq is not None
                or measurement.uncertainty is not None or measurement.limit is not None):
            for name in ("lod", "loq", "uncertainty", "limit"):
                value = getattr(measurement, name)
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                          or not math.isfinite(value) or value < 0):
                    raise ValueError(f"{name}: требуется конечное неотрицательное число")
        confidence = measurement.confidence_level
        if confidence is not None and (isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 < confidence <= 1):
            raise ValueError("Уровень доверия должен быть в интервале (0, 1]")
        if measurement.revision < 1:
            raise ValueError("Недопустимая версия измерения")
        if isinstance(measurement.value, float) and not math.isfinite(measurement.value):
            raise ValueError("Неконечное значение измерения")

    def validate(self, project):
        for collection in (self.samples, self.properties, self.sources, self.analyses):
            if any(key != value.id for key, value in collection.items()):
                raise ValueError("Идентификатор записи не соответствует ключу")
        for analysis in self.analyses.values():
            if analysis.sample_id not in self.samples or analysis.source_id is not None and analysis.source_id not in self.sources:
                raise ValueError("Нарушена связь анализа с пробой или источником")
        for key, measurement in cooperative_items(self.measurements.items()):
            if key != measurement.id:
                raise ValueError("Идентификатор записи не соответствует ключу")
            self._validate_measurement(measurement)
            analysis = self.analyses.get(measurement.analysis_id)
            if analysis is None or analysis.sample_id != measurement.sample_id or measurement.property_id not in self.properties:
                raise ValueError("Нарушена связь измерения с анализом или показателем")
            if measurement.source_id is not None and measurement.source_id not in self.sources:
                raise ValueError("Источник измерения отсутствует")
        for sample_id, selections in self.working.items():
            if sample_id not in self.samples:
                raise ValueError("Неизвестная проба рабочего представления")
            for property_id, measurement_id in selections.items():
                result = self.measurements.get(measurement_id)
                if result is None or result.sample_id != sample_id or result.property_id != property_id:
                    raise ValueError("Некорректная ссылка на рабочее измерение")
        original_columns = [column for column in project.columns if not derived(column)]
        for row in project.rows:
            if row.id not in self.samples:
                raise ValueError("Проба строки отсутствует")
            selected = self.working.get(row.id, {})
            for column in original_columns:
                current = self.measurements.get(selected.get(column.id))
                value = row.values[column.id]
                if current is None and value is None:
                    continue
                if current is None or current.table_value() != value:
                    raise ValueError("Таблица не соответствует рабочим измерениям")
                if current.unit != column.unit:
                    raise ValueError("Единица рабочего измерения отличается от колонки; необходим явный перевод единиц")
