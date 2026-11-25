"""Resolve names against the bundled Russian stratigraphic scale."""

import json
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache
from importlib.resources import files


_SHORT_NAMES = {
    "четвертичная": "антропоген",
    "меловая": "мел",
    "юрская": "юра",
    "триасовая": "триас",
    "пермская": "пермь",
    "каменноугольная": "карбон",
    "девонская": "девон",
    "силурийская": "силур",
    "ордовикская": "ордовик",
    "кембрийская": "кембрий",
}
_POSITION = {
    "верхняя": "верхний", "верхнее": "верхний", "поздний": "верхний",
    "поздняя": "верхний", "позднее": "верхний", "late": "upper",
    "средняя": "средний", "среднее": "средний",
    "нижняя": "нижний", "нижнее": "нижний", "ранний": "нижний",
    "ранняя": "нижний", "раннее": "нижний", "early": "lower",
}
_RANK = {"система", "период", "отдел", "эпоха", "эра", "эратема",
         "system", "period", "series", "epoch", "era", "erathem"}


def _normalize(name):
    words = re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", name).casefold().replace("ё", "е"))
    return " ".join(_POSITION.get(word, word) for word in words if word not in _RANK)


@lru_cache(maxsize=1)
def color_catalog():
    resource = files("autopyro.geology").joinpath("chronostratigraphic_colors.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _names(unit):
    names = set()
    for name in unit:
        names.add(_normalize(name))
        base = _normalize(name.split("(", 1)[0])
        names.add(base)
        if short := _SHORT_NAMES.get(base):
            names.add(short)
    return names - {""}


@lru_cache(maxsize=1)
def _colors_by_name():
    candidates = defaultdict(set)
    positions = {"верхний", "средний", "нижний", "upper", "middle", "lower"}
    for path, color in color_catalog()["unit_defaults"]:
        for name in _names(path[-1]):
            if name not in positions:
                candidates[name].add(color)
            if len(path) > 1:
                for parent in _names(path[-2]):
                    candidates[f"{parent} {name}"].add(color)
                    candidates[f"{name} {parent}"].add(color)
    return {name: next(iter(colors)) for name, colors in candidates.items() if len(colors) == 1}


def stratigraphic_color(name):
    """Return the scale swatch for an unambiguous name, or None."""
    return _colors_by_name().get(_normalize(name)) if isinstance(name, str) else None
