from typing import Union

import numpy as np
import numpy.typing as npt

from AutoPyro.core.plots import LabelledPoint, Plot

# Classifiers - depend on initial conditions, require Plots


class BaseClassifier:
    column_name = "Base"

    @staticmethod
    def classify_single(
        X: Union[float, npt.NDArray[np.float_]],
        Y: Union[float, npt.NDArray[np.float_]],
        author: str,
    ) -> dict[str, dict] | tuple[str, float]:
        plot_author = Plot.from_author(author)
        plot_author.add_points(
            *[
                LabelledPoint(x, y)
                for x, y in zip(X, Y)
                if not np.isnan(x) and not np.isnan(y)
            ]
        )
        if plot_author.areas:
            return plot_author.classify_area()
        else:
            return plot_author.classify_distance()

    @classmethod
    def classify_multi(
        cls,
        X: Union[float, npt.NDArray[np.float_]],
        Y: Union[float, npt.NDArray[np.float_]],
        *authors: list[str],
        return_all: bool = False
    ) -> Union[float, dict[str, float]]:
        statistics = [cls.classify_single(X, Y, author) for author in authors]
        labels_authors = [
            point["value"] for info in statistics.values() for point in info
        ]

        labels, ratios = np.unique(labels_authors, return_counts=True)
        mode = np.argwhere(ratios == np.max(ratios))
        ratios /= labels_authors.shape[0]

        if return_all:
            return statistics

        return mode, (labels, ratios)


class MatterType(BaseClassifier):
    column_name = "Organic Matter Type"
    selector = "MATTER TYPE"

    pass


class Maturity(BaseClassifier):
    column_name = "Maturity Level"
    selector = "MATURITY | Ro"

    pass


class GenerativePotential(BaseClassifier):
    column_name = "Generative Potential"

    pass


CLASSIFIERS_MAP = {
    "matter": MatterType,
    "maturity": Maturity,
    "potential": GenerativePotential,
}
