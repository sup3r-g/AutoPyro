"""Safe substitution of fixed/tied parameters, leaving only free coefficients."""
import ast
from copy import deepcopy
from dataclasses import replace

import numpy as np

from .curve_fitting import custom_model


def parameter_model(model, assignments=""):
    if not assignments.strip():
        return model
    if len(assignments) > 2000:
        raise ValueError("Слишком длинные связи параметров")
    expressions = {}
    try:
        for assignment in assignments.split(";"):
            if not assignment.strip():
                continue
            name, expression = assignment.split("=", 1)
            name = name.strip()
            if name not in model.parameters or name in expressions:
                raise ValueError("Неизвестный или повторный параметр в связи")
            tree = ast.parse(expression.strip(), mode="eval").body
            if any(isinstance(n, ast.Name) and n.id == "x" for n in ast.walk(tree)):
                raise ValueError("Связи коэффициентов не должны зависеть от X")
            expressions[name] = tree
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("Некорректная связь параметров") from exc
    free = tuple(p for p in model.parameters if p not in expressions)
    if not expressions:
        return model
    if not free:
        raise ValueError("Оставьте хотя бы один свободный параметр")
    resolved = {}
    visiting = set()

    class Substitute(ast.NodeTransformer):
        def visit_Name(self, node):
            return deepcopy(resolve(node.id)) if node.id in expressions else node

    def resolve(name):
        if name in visiting:
            raise ValueError("Циклическая связь параметров")
        if name not in resolved:
            visiting.add(name)
            resolved[name] = Substitute().visit(deepcopy(expressions[name]))
            if sum(1 for _ in ast.walk(resolved[name])) > 150:
                raise ValueError("Слишком сложная связь параметров")
            visiting.remove(name)
        return resolved[name]

    for name in expressions:
        resolve(name)
    formula = ast.unparse(Substitute().visit(ast.parse(model.formula, mode="eval").body))
    compiled = custom_model(formula, free, name=model.name)
    expanded = tuple((p, ast.unparse(resolved[p]) if p in resolved else p) for p in model.parameters)
    return replace(compiled, positive_x=model.positive_x, parameter_expressions=expanded)


def parameter_samples(model, coefficients):
    """Expand free coefficient samples into all original coefficients, vectorized."""
    coefficients = np.atleast_2d(coefficients)
    definitions = model.parameter_expressions or tuple((p, p) for p in model.parameters)
    result = {}
    for name, expression in definitions:
        names = {n.id for n in ast.walk(ast.parse(expression, mode="eval")) if isinstance(n, ast.Name)}
        used = tuple(p for p in model.parameters if p in names)
        if used:
            compiled = custom_model(expression, used)
            arguments = [coefficients[:, model.parameters.index(p)] for p in used]
        else:
            compiled = custom_model(f"({expression}) + 0*unused", ("unused",))
            arguments = [np.zeros(len(coefficients))]
        result[name] = compiled.evaluate(np.zeros(len(coefficients)), arguments)
    return result
