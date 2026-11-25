"""Small numerical helpers for scaling and local finite differences."""
import numpy as np


def residual_autocorrelation(x, residuals):
    """Estimate ACF at lags 1–40 in increasing X order, normalized by lag zero."""
    from scipy.fft import irfft, next_fast_len, rfft

    if len(x) < 3:
        raise ValueError("Для автокорреляции нужны хотя бы три точки")
    order = np.argsort(x)
    sorted_x = x[order]
    if np.any(sorted_x[1:] == sorted_x[:-1]):
        raise ValueError("Для автокорреляции значения X должны быть различны")
    ordered = residuals[order]
    if np.all(ordered == ordered[0]):
        raise ValueError("Автокорреляция не определена для постоянных остатков")
    # Scale before centering to keep the sum of squares finite.
    scaled = ordered / np.max(np.abs(ordered))
    centered = scaled - np.mean(scaled)
    max_lag = min(40, len(x)-1)
    fft_size = next_fast_len(len(x)+max_lag, real=True)
    spectrum = rfft(centered, fft_size)
    correlations = irfft(np.abs(spectrum)**2, fft_size)
    return correlations[1:max_lag+1] / correlations[0]


def _center_transform(design):
    """Reparameterize columns around a constant column, without changing the model.

    If D_centered = D @ T and gamma is fitted in centered coordinates, beta=T@gamma.
    This also allows a stable covariance factor in the original parameter units.
    """
    transform = np.eye(design.shape[1])
    constant = np.flatnonzero((np.ptp(design, axis=0) == 0) & (design[0] != 0))
    if len(constant):
        intercept = int(constant[0])
        for column in range(design.shape[1]):
            if column != intercept:
                transform[intercept, column] = -np.mean(design[:, column])/design[0, intercept]
    return transform


def linear_coefficients(design, values, sigma):
    """Return coefficients and an unscaled covariance factor from one SVD."""
    transform = _center_transform(design)
    constant = np.flatnonzero((np.ptp(design, axis=0) == 0) & (design[0] != 0))
    response_origin = values[0] if len(constant) else 0.
    weighted = (design @ transform)/sigma[:, None]
    scale = np.linalg.norm(weighted, axis=0)
    if np.any(scale == 0) or not np.all(np.isfinite(scale)):
        raise ValueError("Недостаточно различных X для определения параметров")
    left, singular, vt = np.linalg.svd(weighted/scale, full_matrices=False)
    if singular[-1] <= np.finfo(float).eps*max(weighted.shape)*singular[0]:
        raise ValueError("Недостаточно различных X для определения параметров")
    factor = transform @ ((vt.T/singular)/scale[:, None])
    coefficients = factor @ (left.T @ ((values-response_origin)/sigma))
    if len(constant):
        coefficients[constant[0]] += response_origin/design[0, constant[0]]
    return coefficients, factor


def linear_leverage(design, sigma):
    """Diagonal of the weighted projection from the thin QR basis."""
    weighted = (design @ _center_transform(design))/sigma[:, None]
    weighted /= np.linalg.norm(weighted, axis=0)
    basis, _ = np.linalg.qr(weighted, mode="reduced")
    return np.clip(np.einsum("ij,ij->i", basis, basis), 0, 1)


def parameter_covariance(jacobian, variance, *, unweighted=None):
    """Return Cov and a factor F with Cov=F@F.T, retaining stable CI evaluation."""
    transform = _center_transform(unweighted if unweighted is not None else jacobian)
    centered = jacobian @ transform
    scale = np.linalg.norm(centered, axis=0)
    if np.any(scale == 0) or not np.all(np.isfinite(scale)):
        raise ValueError("Параметры не определяются однозначно: вырожденная модель или данные")
    _, singular, vt = np.linalg.svd(centered/scale, full_matrices=False)
    if singular[-1] <= np.finfo(float).eps*max(jacobian.shape)*singular[0]:
        raise ValueError("Параметры не определяются однозначно: вырожденная модель или данные")
    factor = transform @ ((vt.T/singular)/scale[:, None])*np.sqrt(variance)
    return factor @ factor.T, factor


def finite_difference(evaluate, value, step, baseline, *, lower=-np.inf, upper=np.inf):
    """Shrink a bounded step when units make its effect nonlocal or nonfinite.

    The usual relative step is kept for ordinary scales. Small rates multiplied
    by large X need a smaller step even when their magnitude is close to zero.
    """
    for _ in range(16):
        plus, minus = np.minimum(value+step, upper), np.maximum(value-step, lower)
        if np.any(plus == minus):
            break
        try:
            above, below = evaluate(plus), evaluate(minus)
        except (ValueError, ArithmeticError):
            step = step*.1
            continue
        change = max(float(np.max(np.abs(above-baseline)/(1+np.abs(baseline)))),
                     float(np.max(np.abs(below-baseline)/(1+np.abs(baseline)))))
        if change <= 1e-3:
            return (above-below)/(plus-minus)
        step = step*.1
    raise ValueError("Не удалось вычислить производную; измените масштаб данных или границы параметров")
def curve_axis_scale(*values):
    """Keep plot spans and ticks finite for extreme but valid coordinates."""
    limit = np.finfo(float).max/16
    return 1e308 if any(np.max(np.abs(value)) > limit for value in values) else 1.

