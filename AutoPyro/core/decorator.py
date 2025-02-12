from functools import wraps
from typing import Any, Callable, TypeVar

FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)


def decorator(func: F) -> Callable[[F], F]:
    @wraps(func)
    def wrapper(*args, **kwargs):
        return value

    return wrapper
