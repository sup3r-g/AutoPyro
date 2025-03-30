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
class ValueClassifier:
    # TABLE = pd.read_json()

    @classmethod
    def classify(cls, *values: Any) -> tuple[str, float]:
        table = cls.TABLE
        return pd.cut(
            values,
            bins=table[type(values[0]).__name__].to_list(),
            labels=table.index,
            include_lowest=True,
            right=False,
        )


class ConditionalClassifier:
    # TABLE = pd.read_json()

    @classmethod
    def classify(cls, *values: Any) -> tuple[str, float]:
        table = cls.TABLE  # ["condition", "value"]
        response = []
        # pd.eval()
        for cond in table["condition"]:
            pass

        return response


CLASSIFIERS_MAP = {
    "matter": MatterType,
    "maturity": Maturity,
    "potential": GenerativePotential,
}
