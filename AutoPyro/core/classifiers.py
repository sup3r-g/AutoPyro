"""Classifiers - depend on initial conditions, require Plots."""

from typing import Any
import pandas as pd
from AutoPyro.core.base import BaseClassifier


class MatterType(BaseClassifier):
    COLUMN_NAME = "Organic Matter Type"
    SELECTOR = "MATTER TYPE"


class Maturity(BaseClassifier):
    COLUMN_NAME = "Maturity Level"
    SELECTOR = "MATURITY | Ro"


class GenerativePotential(BaseClassifier):
    COLUMN_NAME = "Generative Potential"


# IMPORTANT: New stuff
class ValueClassifier(BaseClassifier):
    # TABLE = pd.read_json()

    @classmethod
    def classify_single(cls, value: Any) -> tuple[str, float]:
        table = cls.TABLE
        return pd.cut(
            value,
            bins=table[type(value).__name__].to_list(),
            labels=table.index,
            include_lowest=True,
            right=False,
        )

    @classmethod
    def classify_multi(cls, values) -> tuple[str, float]:
        return cls.classify_single(values)


CLASSIFIERS_MAP = {
    "matter": MatterType,
    "maturity": Maturity,
    "potential": GenerativePotential,
}
