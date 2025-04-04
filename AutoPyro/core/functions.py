from enum import StrEnum, auto
from typing import Iterable, Optional

from helpers import get_all_of_object_type
from numpy import exp, log, sum
from scipy import odr
from scipy.optimize import curve_fit

__all__ = []


def linear(x, a, b) -> float:
    """Linear trend model function"""
    return a * x + b


def power(x, a, b, c, d) -> float:
    """Power trend model function"""
    return a * (x - d) ** b + c


def exponential(x, a, b, c, d, e) -> float:
    """Exponential trend model function"""
    # a*exp(b*x)
    return a * b ** (e * (x - d)) + c


def logarithmic(x, a, b, c, d, e) -> float:
    """
    Logarithm trend model function
    log(b, x) = ln(x) / ln(b)
    """
    return a * log(e * (x - d)) / log(b) + c


def polynomial(x, *coefficients) -> float:
    """
    N-th order polynomial trend model function
    sum(a_i*x^i), i=0..n
    """
    # return a*x**2+b*x+c
    return sum((c * x**i for i, c in enumerate(reversed(coefficients))))


def gaussian(x, a, b, c, d):
    return a * exp ** (-((x - b) ** 2) / (2 * c**2)) + d


def generalized_sigmoid(x, A, K, B, Q, C, M) -> float:
    return (K - A) / (Q * exp(B * (x - M)) + C) + A


def generalized_sigmoid_odr(B, x) -> float:
    return (B[1] - B[0]) / (B[3] * exp(B[2] * (x - B[5])) + B[4]) + B[0]


MODELS = get_all_of_object_type(__name__, "function")


IMPLEMENTED_MODELS = StrEnum(
    "ImplementedModels", [(name.upper(), auto()) for name in MODELS]
)


class CurveFitter:
    __slots__ = "x", "y"

    def __init__(self, x: Iterable[float], y: Iterable[float]) -> None:
        self.x = x
        self.y = y

    def __initial_guess(self):
        pass

    def fit_ols(
        self,
        model: IMPLEMENTED_MODELS = "linear",
        initial_guess: Optional[Iterable[float]] = None,
    ):
        model_func = MODELS[model]
        params, _ = curve_fit(
            model_func,
            self.x,
            self.y,
            p0=initial_guess if initial_guess else self.__initial_guess(),
        )

        # Y_trend = self.model_func(X, *params)

        return model_func, params

    def fit_odr(
        self,
        model: IMPLEMENTED_MODELS = "linear",
        initial_guess: Optional[Iterable[float]] = None,
    ):
        model_func = odr.Model(MODELS[model])
        data = odr.Data(self.x, self.y)
        odr_obj = odr.ODR(
            data,
            model_func,
            beta0=initial_guess if initial_guess else self.__initial_guess(),
        )

        output = odr_obj.run()

        return model_func, output.beta


if __name__ == "__main__":
    print(MODELS, IMPLEMENTED_MODELS.__members__)
