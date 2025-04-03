"""Classifiers - depend on initial conditions, require Plots."""

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from base import Label
from charts import Chart
from geometries import LabelPoint


# TODO: Completely rework this shit!!!
class ChartClassifier:
    COLUMN_NAME = "Base"

    @staticmethod
    def __classify(
        X: float | npt.NDArray[np.float_],
        Y: float | npt.NDArray[np.float_],
        author: str,
    ) -> list[tuple[Any, ...]] | None:
        plot = Chart.from_author(author)
        plot.add(
            *[
                LabelPoint(x, y, Label("", ""))
                for x, y in zip(X, Y)
                if not np.isnan(x) and not np.isnan(y)
            ]
        )
        if plot.areas:
            return plot.classify_area()

        return plot.classify_distance()

    @classmethod
    def classify(
        cls,
        X: float | npt.NDArray[np.float_],
        Y: float | npt.NDArray[np.float_],
        *authors: str,
        return_all: bool = False,
    ) -> list[tuple[Any, ...]] | None | list[list[tuple[Any, ...]]]:
        if len(authors) == 1:
            return cls.__classify(X, Y, authors[0])

        statistics = [cls.__classify(X, Y, author) for author in authors]
        labels_authors = np.array(
            [point["value"] for info in statistics for point in info]
        )

        labels, ratios = np.unique(labels_authors, return_counts=True)
        mode = np.argwhere(ratios == np.max(ratios))
        ratios /= labels_authors.shape[0]

        if return_all:
            return statistics

        return mode, (labels, ratios)


class MatterType(ChartClassifier):
    COLUMN_NAME = "Organic Matter Type"
    SELECTOR = "MATTER TYPE"


class Maturity(ChartClassifier):
    COLUMN_NAME = "Maturity Level"
    SELECTOR = "MATURITY | Ro"


class GenerativePotential(ChartClassifier):
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
    def classify(cls, *values: Any) -> list[str, float]:
        table = cls.TABLE  # ["condition", "value"]
        response = [val for cond, val in table.itertuples(index=False) if pd.eval(cond)]

        return response


CLASSIFIERS_MAP = {
    "matter": MatterType,
    "maturity": Maturity,
    "potential": GenerativePotential,
}
