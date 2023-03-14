from numpy import exp, log, log10, sum

MODELS = {
    "linear": linear,
    "power": power,
    "exponential": exponential,
    "logarithmic": logarithmic,
    "polynomial": polynomial,
    "generalized_sigmoid": generalized_sigmoid,
    "generalized_sigmoid_odr": generalized_sigmoid_odr
}


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
    return sum([c * x**i for i, c in enumerate(reversed(coefficients))])


def generalized_sigmoid(x, A, K, B, Q, C, M) -> float:
    return (K - A) / (Q * exp(B * (x - M)) + C) + A


def generalized_sigmoid_odr(B, x) -> float:
    return (B[1] - B[0]) / (B[3] * exp(B[2] * (x - B[5])) + B[4]) + B[0]
