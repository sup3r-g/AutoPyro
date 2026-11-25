"""One-dimensional curve fitting, independent of Qt and project storage.

OLS minimizes vertical residuals. ODR minimizes corrections in both coordinates;
unit uncertainties give TLS in the supplied coordinate units. Bands are pointwise
Student-t intervals from local linearization, conditional on an exact future X.
"""

from __future__ import annotations

import ast
import keyword
import operator
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, diags, hstack, vstack
from scipy.stats import t
from .curve_numerics import finite_difference, linear_coefficients, linear_leverage, parameter_covariance

_DELETION_MARGIN = 32*np.finfo(float).eps


@dataclass(frozen=True)
class CurveModel:
    name: str
    parameters: tuple[str, ...]
    function: Callable
    formula: str
    positive_x: bool = False
    linear: bool = False
    parameter_expressions: tuple[tuple[str, str], ...] = ()

    def evaluate(self, x, parameters, *, allow_nonfinite=False):
        x = np.asarray(x, dtype=float)
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                values = np.asarray(self.function(x, *parameters))
        except ArithmeticError as exc:
            raise ValueError("Модель вернула NaN/Inf; проверьте область определения и начальные параметры") from exc
        if np.iscomplexobj(values):
            raise ValueError("Модель должна возвращать вещественные значения Y")
        values = np.asarray(values, dtype=float)
        try:
            values = np.broadcast_to(values, x.shape)
        except ValueError as exc:
            raise ValueError("Модель должна возвращать одно значение Y для каждого X") from exc
        if not allow_nonfinite and not np.all(np.isfinite(values)):
            raise ValueError("Модель вернула NaN/Inf; проверьте область определения и начальные параметры")
        return values


_FUNCTIONS = {name: getattr(np, name) for name in
              ("exp", "log", "log10", "sqrt", "sin", "cos", "tan", "abs", "tanh")}
_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.Pow: operator.pow}


def custom_model(expression: str, parameters: tuple[str, ...], *, name="Custom") -> CurveModel:
    """Compile an arithmetic expression into vectorized calls, without eval/exec.

    Allowed: x, named parameters, pi, e, arithmetic, and functions in _FUNCTIONS.
    Python attributes, indexing, imports, comprehensions and arbitrary calls fail.
    """
    parameters = tuple(parameters)
    reserved = set(_FUNCTIONS) | {"x", "pi", "e"}
    if (not parameters or len(set(parameters)) != len(parameters)
            or any(not p.isidentifier() or keyword.iskeyword(p) or p in reserved for p in parameters)):
        raise ValueError("Укажите уникальные имена параметров, отличные от X и имён функций")
    if len(expression) > 1000 or len(parameters) > 20:
        raise ValueError("Формула ограничена 1000 символами и 20 параметрами")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("Некорректная формула модели") from exc
    if sum(1 for _ in ast.walk(tree)) > 150:
        raise ValueError("Слишком сложная формула")
    used = set()

    def compile_node(node):
        if isinstance(node, ast.Constant) and type(node.value) in (float, int):
            value = float(node.value)
            if not np.isfinite(value):
                raise ValueError("Константы должны быть конечными")
            return lambda values: value
        if isinstance(node, ast.Name) and node.id in {"x", *parameters, "pi", "e"}:
            used.add(node.id)
            return lambda values: values[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = compile_node(node.left), compile_node(node.right)
            operation = _OPERATORS[type(node.op)]
            return lambda values: operation(left(values), right(values))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = compile_node(node.operand)
            sign = -1 if isinstance(node.op, ast.USub) else 1
            return lambda values: sign * operand(values)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _FUNCTIONS and len(node.args) == 1 and not node.keywords):
            function, argument = _FUNCTIONS[node.func.id], compile_node(node.args[0])
            return lambda values: function(argument(values))
        raise ValueError("Допустимы только арифметика, X, параметры и разрешённые функции")

    evaluate = compile_node(tree.body)
    if set(parameters) - used:
        raise ValueError("Все объявленные параметры должны использоваться в формуле")

    def function(x, *beta):
        return evaluate(dict(zip(parameters, beta), x=x, pi=np.pi, e=np.e))

    return CurveModel(name, parameters, function, expression)


def standard_model(name="linear", *, degree=2) -> CurveModel:
    if name == "polynomial":
        if not isinstance(degree, int) or not 1 <= degree <= 10:
            raise ValueError("Степень полинома должна быть от 1 до 10")
        return CurveModel(name, tuple(f"c{i}" for i in range(degree + 1)),
                          lambda x, *p: np.polynomial.polynomial.polyval(x, p),
                          " + ".join(f"c{i}*x**{i}" for i in range(degree + 1)), linear=True)
    models = {
        "linear": CurveModel(name, ("a", "b"), lambda x, a, b: a*x+b, "a*x + b", linear=True),
        "exponential": CurveModel(name, ("a", "b", "c"), lambda x, a, b, c: a*np.exp(b*x)+c,
                                  "a*exp(b*x) + c"),
        "power": CurveModel(name, ("a", "b"), lambda x, a, b: a*x**b, "a*x**b", positive_x=True),
        "logarithmic": CurveModel(name, ("a", "b"), lambda x, a, b: a*np.log(x)+b,
                                  "a*log(x) + b", positive_x=True, linear=True),
        "constant": CurveModel(name, ("c",), lambda x, c: np.full_like(x, c), "c", linear=True),
    }
    if name not in models:
        raise ValueError(f"Неизвестная модель: {name}")
    return models[name]


def _linear_components(model, x):
    count = len(model.parameters)
    baseline = model.evaluate(x, np.zeros(count))
    design = np.column_stack([model.evaluate(x, np.eye(count)[j])-baseline for j in range(count)])
    return baseline, design


def _derivatives(model, x, beta, bounds=None):
    """Batched finite differences, staying inside parameter bounds when supplied."""
    if model.linear:
        return _linear_components(model, x)[1]
    step = np.cbrt(np.finfo(float).eps) * np.maximum(1., np.abs(beta))
    baseline = model.evaluate(x, beta)
    jac = np.empty((len(x), len(beta)))
    for j, delta in enumerate(step):
        def evaluate(value):
            changed = beta.copy()
            changed[j] = value
            return model.evaluate(x, changed)
        jac[:, j] = finite_difference(evaluate, beta[j], delta, baseline,
                                      lower=bounds[0][j] if bounds is not None else -np.inf,
                                      upper=bounds[1][j] if bounds is not None else np.inf)
    return jac


def _x_derivative(model, x, beta):
    step = np.cbrt(np.finfo(float).eps) * np.maximum(1., np.abs(x))
    if model.positive_x:
        step = np.minimum(step, x / 2)
    return finite_difference(lambda value: model.evaluate(value, beta), x, step, model.evaluate(x, beta),
                             lower=np.finfo(float).tiny if model.positive_x else -np.inf)


def _initial(model, x, y):
    if model.name == "power" and np.all(y > 0):
        b, log_a = np.polyfit(np.log(x), np.log(y), 1)
        return np.array([np.exp(log_a), b])
    if model.name == "exponential":
        c = float(np.min(y) - max(np.ptp(y)*.1, 1e-6))
        b, log_a = np.polyfit(x, np.log(y-c), 1)
        return np.array([np.exp(log_a), b, c])
    return np.ones(len(model.parameters))


@dataclass
class FitResult:
    model: CurveModel
    method: str
    parameters: np.ndarray
    covariance: np.ndarray | None
    x: np.ndarray
    y: np.ndarray
    fitted: np.ndarray
    adjusted_x: np.ndarray
    metrics: dict[str, float]
    residual_variance: float
    objective: float
    warnings: tuple[str, ...]
    loss: str = "linear"
    f_scale: float = 1.
    bounds: tuple[np.ndarray, np.ndarray] | None = None
    active_bounds: tuple[str, ...] = ()
    sx: np.ndarray | None = None
    sy: np.ndarray | None = None
    absolute_sigma: bool = False
    max_evaluations: int = 1000
    start_count: int = 1
    start_seed: int = 0
    start_diagnostics: tuple[dict, ...] = ()
    covariance_factor: np.ndarray | None = None

    @property
    def supports_influence(self):
        return (self.model.linear and self.method == "ols" and self.loss == "linear"
                and (self.bounds is None or not np.any(np.isfinite(self.bounds))))

    def _influence_leverage(self):
        if not self.supports_influence:
            raise ValueError("Диагностика влияния доступна для линейного OLS/WLS без ограничений")
        _, design = _linear_components(self.model, self.x)
        sigma = self.sy if self.sy is not None else np.ones_like(self.x)
        return linear_leverage(design, sigma)

    def influence(self):
        """Return weighted leverage and Cook's distance, in fitted-row order."""
        leverage = self._influence_leverage()
        cook = np.full(len(self.x), np.nan)
        residuals = self.y-self.fitted
        eps = np.finfo(float).eps
        roundoff = 4*eps*np.maximum(np.abs(self.y), np.abs(self.fitted))
        if np.all(np.abs(residuals) <= roundoff):
            return leverage, cook
        weighted = residuals if self.sy is None else residuals/self.sy
        scaled = weighted/np.max(np.abs(weighted))
        p = len(self.parameters)
        mse = (scaled @ scaled)/(len(self.x)-p)
        remaining = 1-leverage
        valid = remaining > _DELETION_MARGIN
        cook[valid] = ((scaled[valid]/remaining[valid])**2)*leverage[valid]/(p*mse)
        return leverage, cook

    def leave_one_out(self, *, leverage=None):
        """Return deletion predictions in original Y units, using NaN for singular deletions.

        Pass the leverage from this result's influence() to reuse the QR calculation.
        """
        if leverage is None or not self.supports_influence:
            leverage = self._influence_leverage()
        else:
            leverage = np.asarray(leverage, dtype=float)
            if (leverage.shape != self.x.shape or not np.all(np.isfinite(leverage))
                    or np.any((leverage < 0) | (leverage > 1))):
                raise ValueError("Некорректные значения рычага")
        remaining = 1-leverage
        valid = remaining > _DELETION_MARGIN
        predicted = np.full_like(self.y, np.nan)
        with np.errstate(over="ignore", invalid="ignore"):
            correction = leverage[valid]*(self.y[valid]-self.fitted[valid])/remaining[valid]
            predicted[valid] = self.fitted[valid]-correction
        predicted[~np.isfinite(predicted)] = np.nan
        return predicted

    @property
    def standard_errors(self):
        if self.covariance is None:
            return None
        if self.covariance_factor is not None:
            return np.hypot.reduce(self.covariance_factor, axis=1, initial=0.)
        return np.sqrt(np.maximum(0, np.diag(self.covariance)))

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)) or self.model.positive_x and np.any(x <= 0):
            raise ValueError("Некорректные координаты прогноза")
        return self.model.evaluate(x, self.parameters)

    def bands(self, x, *, confidence=.95, prediction_sigma=None):
        """Return mean, confidence lower/upper, prediction lower/upper.

        Prediction assumes independent homoscedastic Y noise unless its future
        standard deviation is supplied. X uncertainty is not propagated here.
        """
        if self.covariance is None:
            raise ValueError("Полосы недоступны для робастной регрессии или активных границ параметров")
        if not np.isfinite(confidence) or not 0 < confidence < 1:
            raise ValueError("Уровень доверия должен быть между 0 и 1")
        x = np.atleast_1d(np.asarray(x, dtype=float))
        if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)) or (self.model.positive_x and np.any(x <= 0)):
            raise ValueError("Некорректные координаты прогноза")
        mean = self.predict(x)
        jac = _derivatives(self.model, x, self.parameters, self.bounds)
        variance = (np.sum((jac @ self.covariance_factor)**2, axis=1) if self.covariance_factor is not None else
                    np.maximum(0, np.einsum("ij,jk,ik->i", jac, self.covariance, jac)))
        noise = self.residual_variance
        if prediction_sigma is not None:
            sigma = np.asarray(prediction_sigma, dtype=float)
            if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
                raise ValueError("Погрешность прогноза должна быть положительной и конечной")
            noise = np.broadcast_to(sigma, x.shape)**2
        critical = t.ppf((1+confidence)/2, len(self.x)-len(self.parameters))
        ci, pi = critical*np.sqrt(variance), critical*np.sqrt(variance+noise)
        return mean, mean-ci, mean+ci, mean-pi, mean+pi


def fit_curve(x, y, model=None, *, method="ols", initial=None, sx=None, sy=None,
              absolute_sigma=False, max_evaluations=1000, loss="linear", f_scale=1.,
              bounds=(-np.inf, np.inf), starts=1, start_seed=0) -> FitResult:
    """Fit finite paired arrays. sx/sy are positive standard deviations.

    Missing values must be excluded by the caller, preserving paired rows.
    OLS with sy becomes weighted least squares. For ODR, unspecified coordinate
    uncertainties equal one. absolute_sigma uses known supplied uncertainties;
    otherwise covariance is scaled by the objective divided by n-p.
    Huber/soft_l1 act on standardized coordinate residuals with cutoff f_scale.
    Bounds contain lower/upper scalars or arrays in model parameter order.
    """
    model = model or standard_model()
    if type(starts) is not int or not 1 <= starts <= 100 or type(start_seed) is not int or start_seed < 0:
        raise ValueError("Число стартов должно быть от 1 до 100, Seed — неотрицательным целым")
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    p = len(model.parameters)
    if (x.ndim != 1 or y.shape != x.shape or len(x) <= p
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))):
        raise ValueError("Нужны конечные пары X/Y; число точек должно превышать число параметров")
    if model.positive_x and np.any(x <= 0):
        raise ValueError("Для этой модели все X должны быть положительными")
    if method not in {"ols", "odr"}:
        raise ValueError("Метод должен быть ols или odr")
    if loss not in {"linear", "huber", "soft_l1"}:
        raise ValueError("Функция потерь должна быть linear, huber или soft_l1")
    if not np.isfinite(f_scale) or f_scale <= 0:
        raise ValueError("Масштаб робастной функции должен быть положительным и конечным")
    if not isinstance(max_evaluations, int) or max_evaluations < 1:
        raise ValueError("Число итераций должно быть положительным целым")
    try:
        lower_beta, upper_beta = (np.broadcast_to(np.asarray(value, float), (p,)).copy() for value in bounds)
    except (ValueError, TypeError) as exc:
        raise ValueError("Задайте нижние и верхние границы для каждого параметра") from exc
    if (np.any(np.isnan(lower_beta)) or np.any(np.isnan(upper_beta))
            or np.any(lower_beta >= upper_beta)):
        raise ValueError("Нижние границы должны быть строго меньше верхних; NaN запрещён")
    parameter_bounds = lower_beta, upper_beta
    bounded = np.any(np.isfinite(lower_beta)) or np.any(np.isfinite(upper_beta))
    if method == "ols" and sx is not None:
        raise ValueError("Погрешности X учитываются только методом ODR")
    if absolute_sigma and (sy is None or (method == "odr" and sx is None)):
        raise ValueError("Для абсолютных погрешностей задайте sy (для ODR также sx)")

    def uncertainty(value):
        result = np.ones_like(x) if value is None else np.broadcast_to(np.asarray(value, float), x.shape).copy()
        if np.any(~np.isfinite(result)) or np.any(result <= 0):
            raise ValueError("Погрешности должны быть положительными и конечными")
        return result

    sigma_x, sigma_y = uncertainty(sx), uncertainty(sy)
    beta = _initial(model, x, y) if initial is None else np.asarray(initial, dtype=float)
    if beta.shape != (p,) or not np.all(np.isfinite(beta)):
        raise ValueError(f"Укажите {p} конечных начальных параметров")
    if initial is not None and (np.any(beta < lower_beta) or np.any(beta > upper_beta)):
        raise ValueError("Начальные параметры должны лежать внутри заданных границ")
    if starts > 1:
        from .curve_search import multistart_fit
        return multistart_fit(x, y, model, starts=starts, seed=start_seed,
                              options=dict(method=method, initial=initial, sx=sx, sy=sy,
                                           absolute_sigma=absolute_sigma, max_evaluations=max_evaluations,
                                           loss=loss, f_scale=f_scale, bounds=bounds))
    active = np.zeros(p, dtype=int)
    direct_ols = model.linear and method == "ols" and not bounded and loss == "linear"
    # Iterative refits can reuse a supplied solution (e.g. bootstrap); ordinary
    # unconstrained OLS always keeps its exact, fast linear solve.
    linear_data = _linear_components(model, x) if model.linear else None
    if model.linear and (initial is None or direct_ols):
        baseline, design = linear_data
        beta, linear_factor = linear_coefficients(design, y-baseline, sigma_y)
    if initial is None or not direct_ols:
        beta = np.clip(beta, lower_beta, upper_beta)
        # Move near-boundary starts inside: TRF can otherwise stop at a tiny
        # zero even when a resample's optimum lies in the interior.
        for j in range(p):
            margin = (1e-4 if initial is not None else .01) * max(1., abs(beta[j]))
            if np.isfinite(lower_beta[j]) and np.isfinite(upper_beta[j]):
                margin = min(margin, (upper_beta[j]-lower_beta[j])/2)
            if np.isfinite(lower_beta[j]) and beta[j]-lower_beta[j] <= 1e-8*max(1., abs(lower_beta[j])):
                beta[j] = lower_beta[j]+margin
            elif np.isfinite(upper_beta[j]) and upper_beta[j]-beta[j] <= 1e-8*max(1., abs(upper_beta[j])):
                beta[j] = upper_beta[j]-margin
    if not model.linear or bounded or loss != "linear":
        preliminary = least_squares(lambda b: (model.evaluate(x, b)-y)/sigma_y,
                                    beta, jac=lambda b: (linear_data[1] if linear_data is not None else
                                                         _derivatives(model, x, b, parameter_bounds))/sigma_y[:, None],
                                    max_nfev=max_evaluations, x_scale="jac",
                                    bounds=parameter_bounds, loss=loss, f_scale=f_scale)
        if not preliminary.success:
            raise ValueError(f"OLS не сошёлся: {preliminary.message}")
        beta = preliminary.x
        active = preliminary.active_mask

    adjusted_x = x.copy()
    if method == "odr":
        # Optimize standardized X corrections alongside beta. The block Jacobian has O(n*p)
        # nonzeros; no dense n-by-n arrays or deprecated scipy.odr are needed.
        n = len(x)

        def residual(z):
            b, correction = z[:p], z[p:]
            corrected = x+sigma_x*correction
            return np.r_[(model.evaluate(corrected, b)-y)/sigma_y, correction]

        def jacobian(z):
            b = z[:p]
            corrected = x+sigma_x*z[p:]
            jb = csr_matrix(_derivatives(model, corrected, b, parameter_bounds)/sigma_y[:, None])
            jx = diags(sigma_x*_x_derivative(model, corrected, b)/sigma_y)
            return vstack([hstack([jb, jx]), hstack([csr_matrix((n, p)), diags(np.ones(n))])], format="csr")

        x_lower = (np.finfo(float).tiny-x)/sigma_x if model.positive_x else np.full(n, -np.inf)
        lower = np.r_[lower_beta, x_lower]
        upper = np.r_[upper_beta, np.full(n, np.inf)]
        solution = least_squares(residual, np.r_[beta, np.zeros(n)], jac=jacobian,
                                 bounds=(lower, upper), x_scale="jac", max_nfev=max_evaluations,
                                 loss=loss, f_scale=f_scale,
                                 tr_options={"atol": 1e-12, "btol": 1e-12})
        if not solution.success:
            raise ValueError(f"ODR не сошёлся: {solution.message}")
        beta, adjusted_x = solution.x[:p], x+sigma_x*solution.x[p:]
        objective = float(2*solution.cost)
        active = solution.active_mask[:p]
        effective_sigma = np.hypot(sigma_y, sigma_x*_x_derivative(model, adjusted_x, beta))
    else:
        residual = (model.evaluate(x, beta)-y)/sigma_y
        objective = (float(residual @ residual) if model.linear and not bounded and loss == "linear"
                     else float(2*preliminary.cost))
        effective_sigma = sigma_y
    dof = len(x)-p
    variance = 1. if absolute_sigma else objective/dof
    if direct_ols:
        covariance_factor = linear_factor*np.sqrt(variance)
        covariance = covariance_factor @ covariance_factor.T
    else:
        unweighted_jac = (linear_data[1] if method == "ols" and linear_data is not None else
                          _derivatives(model, adjusted_x, beta, parameter_bounds))
        jac = unweighted_jac/effective_sigma[:, None]
        covariance, covariance_factor = parameter_covariance(jac, variance, unweighted=unweighted_jac)
    if not np.isfinite(objective) or not np.all(np.isfinite(covariance)):
        raise ValueError("Численное переполнение; измените масштаб данных или начальные параметры")
    fitted = model.evaluate(x, beta)
    errors = y-fitted
    with np.errstate(over="ignore", invalid="ignore"):
        sse = float(errors @ errors)
        total = float(np.sum((y-np.mean(y))**2))
    if not np.isfinite(sse) or not np.isfinite(total):
        raise ValueError("Численное переполнение; измените масштаб данных или начальные параметры")
    r2 = 1-sse/total if total > 0 else float("nan")
    metrics = {"N": len(x), "MAE": float(np.mean(np.abs(errors))),
               "MSE": sse/len(x), "RMSE": float(np.sqrt(sse/len(x))),
               "MedAE": float(np.median(np.abs(errors))), "MaxAE": float(np.max(np.abs(errors))),
               "Bias": float(np.mean(errors)), "SSE": sse, "R2": r2,
               "R2_adj": 1-(1-r2)*(len(x)-1)/dof}
    notices = ["Полосы: точечная локальная аппроксимация; Prediction условна по точному X."]
    active_bounds = tuple(name for name, flag in zip(model.parameters, active) if flag)
    if loss != "linear" or active_bounds:
        covariance = None
        covariance_factor = None
        notices = ["Аналитические ошибки и полосы недоступны; для полос используйте bootstrap."]
    if active_bounds:
        notices.append("Параметры на границе: " + ", ".join(active_bounds))
    if loss != "linear":
        notices.append("Робастная функция применяется отдельно к стандартизованным остаткам координат; точки не удаляются.")
    if method == "odr":
        notices.append("Метрики вычислены по вертикальным остаткам исходных точек; цель ODR хранится отдельно.")
    if sy is not None:
        notices.append("Для Prediction без prediction_sigma используется общая дисперсия вертикальных остатков.")
    if covariance_factor is not None:
        errors = np.hypot.reduce(covariance_factor, axis=1, initial=0.)
        if np.all(errors > 0):
            normalized = covariance_factor/errors[:, None]
            if np.linalg.cond(normalized @ normalized.T) > 1e12:
                notices.append("Ковариация плохо обусловлена; коэффициенты и полосы могут быть ненадёжны.")
    return FitResult(model, method, beta, covariance, x.copy(), y.copy(), fitted,
                     adjusted_x.copy(), metrics, sse/dof, objective, tuple(notices), loss, f_scale,
                     parameter_bounds, active_bounds,
                     sigma_x.copy() if sx is not None else None,
                     sigma_y.copy() if sy is not None else None, absolute_sigma, max_evaluations,
                     covariance_factor=covariance_factor)
