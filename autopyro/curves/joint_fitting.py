"""Joint least squares with shared and series-specific coefficients."""
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

from .curve_fitting import CurveModel, _derivatives, _initial, _linear_components
from .curve_numerics import _center_transform
from .curve_validation import factor_groups, prediction_metrics


@dataclass
class JointFitResult:
    model: CurveModel
    shared: tuple[str, ...]
    labels: tuple
    coefficients: np.ndarray
    x: np.ndarray
    y: np.ndarray
    fitted: np.ndarray
    sy: np.ndarray | None
    codes: np.ndarray
    order: np.ndarray
    boundaries: np.ndarray
    metrics: dict
    series_rmse: np.ndarray
    series_mae: np.ndarray
    bounds: tuple[np.ndarray, np.ndarray] | None = None
    active_bounds: np.ndarray | None = None
    evaluations: int | None = None
    loss: str = "linear"
    f_scale: float = 1.
    max_evaluations: int = 1000

    def indices(self, series):
        return self.order[self.boundaries[series]:self.boundaries[series+1]]

    def predict(self, series, x):
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)) or self.model.positive_x and np.any(x <= 0):
            raise ValueError("Некорректные координаты прогноза")
        return self.model.evaluate(x, self.coefficients[series])


def _eliminate_local(local, targets, sigma):
    """Project shared features and response off the local design.

    The returned map gives the local coefficients for each target, so the local
    solution after fitting shared beta is map[:, -1] - map[:, :-1] @ beta.
    """
    transform = _center_transform(local)
    constant = np.flatnonzero((np.ptp(local, axis=0) == 0) & (local[0] != 0))
    if len(constant):
        intercept = int(constant[0])
        origin = targets[0]/local[0, intercept]
        targets = targets-local[:, intercept, None]*origin
    weighted = (local @ transform)/sigma[:, None]
    scale = np.linalg.norm(weighted, axis=0)
    if np.any(scale == 0) or not np.all(np.isfinite(scale)):
        raise ValueError("Локальные коэффициенты одной из серий не определяются однозначно")
    left, singular, right = np.linalg.svd(weighted/scale, full_matrices=False)
    if singular[-1] <= np.finfo(float).eps*max(weighted.shape)*singular[0]:
        raise ValueError("Локальные коэффициенты одной из серий не определяются однозначно")
    targets = targets/sigma[:, None]
    coordinates = left.T @ targets
    reduced = targets-left @ coordinates
    if len(local) == local.shape[1]:
        reduced.fill(0.)  # A saturated local design contributes no shared information.
    local_map = (transform @ ((right.T/singular)/scale[:, None])) @ coordinates
    if len(constant):
        local_map[intercept] += origin
    return reduced, local_map, np.linalg.norm(targets[:, :-1], axis=0)


def _fit_linear(design, y, sigma, order, boundaries, common, cancelled):
    count = len(boundaries)-1
    n_local, n_shared = int((~common).sum()), int(common.sum())
    targets = np.column_stack((design[:, common], y))[order]
    local = design[:, ~common][order]
    ordered_sigma = sigma[order]
    local_maps = np.empty((count, n_local, n_shared+1))
    original_scale = np.zeros(n_shared)
    for series_index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if cancelled is not None and cancelled():
            return None
        block = slice(start, end)
        targets[block], local_maps[series_index], scale = _eliminate_local(local[block], targets[block], ordered_sigma[block])
        original_scale = np.hypot(original_scale, scale)
    reduced = targets[:, :-1]
    scale = np.linalg.norm(reduced, axis=0)
    tolerance = 16*np.finfo(float).eps*design.shape[1]
    if np.any(scale <= tolerance*original_scale) or not np.all(np.isfinite(scale)):
        raise ValueError("Общие коэффициенты не определяются однозначно")
    shared_beta, _, rank, _ = np.linalg.lstsq(reduced/scale, targets[:, -1], rcond=None)
    if rank != n_shared:
        raise ValueError("Общие коэффициенты не определяются однозначно")
    shared_beta /= scale
    coefficients = np.empty((count, design.shape[1]))
    coefficients[:, common] = shared_beta
    coefficients[:, ~common] = local_maps[:, :, -1]-local_maps[:, :, :-1] @ shared_beta
    return coefficients


def _joint_bounds(model, labels, common, bounds, overrides):
    shape = (len(labels), len(model.parameters))
    try:
        lower, upper = (np.broadcast_to(np.asarray(side, float), shape).copy() for side in bounds)
    except (TypeError, ValueError) as exc:
        raise ValueError("Некорректные границы параметров") from exc
    if np.any(lower[:, common] != lower[0, common]) or np.any(upper[:, common] != upper[0, common]):
        raise ValueError("Общие коэффициенты должны иметь одинаковые границы во всех сериях")
    if overrides is not None:
        by_label = {label: i for i, label in enumerate(labels)}
        by_parameter = {name: i for i, name in enumerate(model.parameters) if not common[i]}
        if not isinstance(overrides, Mapping):
            raise ValueError("Границы серий должны быть объектом JSON")
        for label, parameters in overrides.items():
            if label not in by_label or not isinstance(parameters, Mapping):
                raise ValueError("В границах серий указана неизвестная серия или неверный формат")
            for name, interval in parameters.items():
                if name not in by_parameter or not isinstance(interval, (tuple, list)) or len(interval) != 2:
                    raise ValueError("Для серии задайте две границы отдельного коэффициента")
                i, j = by_label[label], by_parameter[name]
                lo, hi = interval
                try:
                    lower[i, j] = -np.inf if lo is None else float(lo)
                    upper[i, j] = np.inf if hi is None else float(hi)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Некорректные границы параметров") from exc
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(lower >= upper):
        raise ValueError("Некорректные границы параметров")
    return lower, upper


class _JointCancelled(Exception):
    pass


class JointIdentifiabilityError(ValueError):
    """The free coefficients cannot be distinguished by the data at the solution."""


def _check_joint_rank(derivatives, blocks, parameter_ids, common, active, check_cancelled):
    # The Jacobian uses finite differences. Test normalized columns so the
    # threshold applies to near-dependence, independent of parameter units.
    tolerance = 1e-8
    shared = common & (active[parameter_ids[0]] == 0)
    reduced = derivatives[:, shared]
    shared_scale = np.linalg.norm(reduced, axis=0)
    for block, ids in zip(blocks, parameter_ids):
        check_cancelled()
        local = derivatives[block][:, ~common & (active[ids] == 0)]
        if local.shape[1] == 0:
            continue
        scale = np.linalg.norm(local, axis=0)
        if np.any(scale == 0) or not np.all(np.isfinite(scale)):
            raise JointIdentifiabilityError("Отдельные коэффициенты совместной подгонки не определяются однозначно")
        basis, singular, _ = np.linalg.svd(local/scale, full_matrices=False)
        if singular[-1] <= tolerance*singular[0]:
            raise JointIdentifiabilityError("Отдельные коэффициенты совместной подгонки не определяются однозначно")
        reduced[block] -= basis @ (basis.T @ reduced[block])
    if not np.any(shared):
        return
    error = "Общие коэффициенты совместной подгонки не определяются однозначно"
    if np.any(shared_scale == 0) or not np.all(np.isfinite(shared_scale)):
        raise JointIdentifiabilityError(error)
    singular = np.linalg.svd(reduced/shared_scale, compute_uv=False)
    if singular[-1] <= tolerance:
        raise JointIdentifiabilityError(error)


def _fit_iterative(model, x, y, sigma, codes, order, boundaries, common, bounds, initial,
                   linear, max_evaluations, loss, f_scale, cancelled):
    count, p = bounds[0].shape
    n_shared = int(common.sum())
    parameter_ids = np.empty((count, p), dtype=int)
    parameter_ids[:, common] = np.arange(n_shared)
    size = n_shared+count*(p-n_shared)
    parameter_ids[:, ~common] = np.arange(n_shared, size).reshape(count, -1)

    def pack(values):
        return np.r_[values[0, common], values[:, ~common].ravel()]

    def check_cancelled():
        if cancelled is not None and cancelled():
            raise _JointCancelled

    x_ordered, y_ordered, sigma_ordered = x[order], y[order], sigma[order]
    blocks = [slice(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    linear_start = None
    if linear is not None:
        baseline, design = linear
        linear_start = _fit_linear(design, y-baseline, sigma, order, boundaries, common, cancelled)
        if linear_start is None:
            raise _JointCancelled
    if initial is not None:
        try:
            start = np.broadcast_to(np.asarray(initial, float), (count, p)).copy()
        except (TypeError, ValueError) as exc:
            raise ValueError("Некорректное начальное приближение") from exc
    elif linear_start is not None:
        start = linear_start
    else:
        fallback = _initial(model, x, y) if np.ptp(x) > 0 else np.ones(p)
        start = np.array([_initial(model, x_ordered[b], y_ordered[b]) if np.ptp(x_ordered[b]) > 0
                          else fallback for b in blocks])
    if not np.all(np.isfinite(start)):
        raise ValueError("Некорректное начальное приближение")
    start[:, common] = np.median(start[:, common], axis=0)
    lower, upper = map(pack, bounds)
    initial_vector = np.clip(pack(start), lower, upper)
    # Starting exactly at a bound can leave the trust region too small to move inward.
    with np.errstate(over="ignore"):
        margin = np.minimum(1e-4*np.maximum(1., np.abs(initial_vector)), (upper-lower)/4)
    initial_vector = np.clip(initial_vector, lower+margin, upper-margin)
    columns = parameter_ids[codes[order]].ravel()
    pointers = np.arange(0, p*(len(x)+1), p)

    def sparse(values):
        return csr_matrix((values.ravel(), columns, pointers), shape=(len(x), size))

    if linear is not None:
        baseline, design = linear
        jacobian = sparse(design[order]/sigma_ordered[:, None])
        offset = (baseline[order]-y_ordered)/sigma_ordered

        def residual(vector):
            check_cancelled()
            return jacobian @ vector + offset

        def jac(vector):
            # SciPy's robust loss rescales the Jacobian in place.
            return jacobian if loss == "linear" else jacobian.copy()
    else:
        derivatives = np.empty((len(x), p))

        def residual(vector):
            check_cancelled()
            coefficients = vector[parameter_ids]
            fitted = np.empty_like(y_ordered)
            for i, block in enumerate(blocks):
                fitted[block] = model.evaluate(x_ordered[block], coefficients[i], allow_nonfinite=True)
            with np.errstate(over="ignore", invalid="ignore"):
                return (fitted-y_ordered)/sigma_ordered

        def jac(vector):
            check_cancelled()
            coefficients = vector[parameter_ids]
            for i, block in enumerate(blocks):
                derivatives[block] = _derivatives(model, x_ordered[block], coefficients[i], (bounds[0][i], bounds[1][i]))
            return sparse(derivatives/sigma_ordered[:, None])

    if not np.all(np.isfinite(residual(initial_vector))):
        raise ValueError("Модель не определена при начальных коэффициентах совместной подгонки")
    solution = least_squares(residual, initial_vector, jac=jac, bounds=(lower, upper), x_scale="jac",
                             tr_solver="lsmr", tr_options={"atol": 1e-9, "btol": 1e-9},
                             ftol=1e-10, xtol=1e-10, gtol=1e-9, max_nfev=max_evaluations,
                             loss=loss, f_scale=f_scale)
    if not solution.success:
        raise ValueError("Совместная подгонка не сошлась. Уточните начальные значения или границы")
    if linear is None:
        _check_joint_rank(derivatives, blocks, parameter_ids, common, solution.active_mask, check_cancelled)
    return solution.x[parameter_ids], solution.active_mask[parameter_ids], solution.nfev


def _solve_joint(model, x, y, sigma, codes, order, boundaries, common, limits, initial, linear,
                 max_evaluations, loss, f_scale, cancelled):
    if linear is not None and loss == "linear" and not any(np.any(np.isfinite(side)) for side in limits):
        baseline, design = linear
        return _fit_linear(design, y-baseline, sigma, order, boundaries, common, cancelled), None, None
    try:
        return _fit_iterative(model, x, y, sigma, codes, order, boundaries, common, limits, initial,
                              linear, max_evaluations, loss, f_scale, cancelled)
    except _JointCancelled:
        return None, None, None


def fit_joint_curve(x, y, series, model, *, shared, sy=None, initial=None, bounds=(-np.inf, np.inf),
                    series_bounds=None, max_evaluations=1000, loss="linear", f_scale=1., cancelled=None):
    """Fit OLS/WLS. Series overrides map labels to local parameter bounds."""
    if type(max_evaluations) is not int or max_evaluations < 1:
        raise ValueError("Число итераций должно быть положительным целым")
    if loss not in ("linear", "huber", "soft_l1"):
        raise ValueError("Неизвестная функция потерь")
    if not np.isfinite(f_scale) or f_scale <= 0:
        raise ValueError("Масштаб робастной функции должен быть положительным и конечным")
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or not len(x) or not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("Для сравнения нужны конечные пары X/Y")
    if model.positive_x and np.any(x <= 0):
        raise ValueError("Некорректные координаты прогноза")
    shared = set(shared)
    if not shared < set(model.parameters) or not shared:
        raise ValueError("Выберите хотя бы один общий и один отдельный коэффициент")
    common = np.array([name in shared for name in model.parameters])
    codes, count = factor_groups(series, len(x), error="Для совместной подгонки укажите серию для каждой точки")
    if count < 2:
        raise ValueError("Для совместной подгонки нужны хотя бы две серии")
    sizes = np.bincount(codes)
    n_local, n_shared = int((~common).sum()), int(common.sum())
    if np.any(sizes < n_local) or len(x) <= n_shared+count*n_local:
        raise ValueError("Недостаточно точек для общих и отдельных коэффициентов")
    sigma = np.ones_like(x) if sy is None else np.broadcast_to(np.asarray(sy, float), x.shape)
    if np.any(sigma <= 0) or not np.all(np.isfinite(sigma)):
        raise ValueError("Погрешности должны быть положительными и конечными")
    order = np.argsort(codes, kind="stable")
    boundaries = np.r_[0, np.cumsum(sizes)]
    labels = tuple(np.asarray(series, dtype=object)[order[boundaries[:-1]]])
    limits = _joint_bounds(model, labels, common, bounds, series_bounds)
    bounded = any(np.any(np.isfinite(side)) for side in limits)
    linear = _linear_components(model, x) if model.linear else None
    coefficients, active, evaluations = _solve_joint(model, x, y, sigma, codes, order, boundaries, common, limits,
                                                     initial, linear, max_evaluations, loss, f_scale, cancelled)
    if coefficients is None:
        return None
    if model.linear:
        baseline, design = linear
        fitted = baseline + np.einsum("ij,ij->i", design, coefficients[codes])
    else:
        fitted = np.empty_like(y)
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            indices = order[start:end]
            fitted[indices] = model.evaluate(x[indices], coefficients[i])
    metrics = prediction_metrics(y, fitted)
    errors = y-fitted
    rmse = np.sqrt(np.bincount(codes, errors**2)/sizes)
    mae = np.bincount(codes, np.abs(errors))/sizes
    return JointFitResult(model, tuple(name for name in model.parameters if name in shared), labels, coefficients,
                          x.copy(), y.copy(), fitted, sigma.copy() if sy is not None else None,
                          codes, order, boundaries, metrics, rmse, mae, limits if bounded else None,
                          active, evaluations, loss, f_scale, max_evaluations)
