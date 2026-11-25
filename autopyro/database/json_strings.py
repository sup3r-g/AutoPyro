"""Share immutable JSON strings during one project load, never mutable data."""
import json

from autopyro.model.audit import MAGIC

MAX_SHARED_STRINGS = 1_000_000
MAX_SHORT_STRING_LENGTH = 128
MAX_SHARED_CELL_STRINGS = 32_768


class JsonStringPool:
    def __init__(self):
        self._strings = {}
        self._cells = {}

    def cell(self, value):
        """Share short literal cells without growing the protocol string pool."""
        if type(value) is not str or len(value) > MAX_SHORT_STRING_LENGTH:
            return value
        if len(self._cells) < MAX_SHARED_CELL_STRINGS:
            return self._cells.setdefault(value, value)
        return self._cells.get(value, value)

    def existing_cell(self, value):
        """Reuse a cell already seen in the table without extending the pool."""
        return self._cells.get(value, value)

    def string(self, value):
        existing = self._strings.get(value)
        if existing is not None:
            return existing
        # Long free text is unlikely to repeat. Archived protocols are an
        # exception: their immutable text can appear in several metadata blocks.
        if len(value) > MAX_SHORT_STRING_LENGTH and not value.startswith(MAGIC):
            return value
        if len(self._strings) < MAX_SHARED_STRINGS:
            self._strings[value] = value
        return value

    def existing(self, value):
        """Reuse a known protocol ID without growing the pool for source records."""
        return self._strings.get(value, value)

    def _value(self, value):
        if type(value) is str:
            return self.string(value)
        if type(value) is list:
            for i, item in enumerate(value):
                value[i] = self._value(item)
        return value

    def _object(self, values):
        # Child objects have already passed this hook. Only lists need a walk;
        # do not repeatedly traverse dictionaries containing complete protocols.
        share, known = self.string, self._strings.get
        changed_keys = False
        for key, value in values.items():
            if type(value) is str:
                values[key] = share(value)
            elif type(value) is list:
                self._value(value)
            if known(key, key) is not key:
                changed_keys = True
        # Standard JSON already shares field names within a document. Rebuild
        # only maps whose UUID keys were also encountered as string values.
        if changed_keys:
            return {known(key, key): value for key, value in values.items()}
        return values

    def loads(self, payload):
        return self._value(json.loads(payload, object_hook=self._object))

    def text_loader(self):
        """Reuse a decoder across records without keeping a reference cycle."""
        decode = json.JSONDecoder(object_hook=self._object).decode

        def load(payload):
            if type(payload) is str and not payload.startswith("\ufeff"):
                return self._value(decode(payload))
            return self.loads(payload)

        return load
