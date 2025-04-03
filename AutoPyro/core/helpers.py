from functools import wraps
from inspect import signature
from typing import Any, Callable, Type, TypeVar

FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)


def func_args(cls: Type) -> list[str]:
    return [k for k in signature(cls.__init__).parameters if k != "self"]


# def verify_types(cls: Type) -> list[str]:
#     return [k for k in signature(cls.__init__).parameters if k != "self"]


def decorator(func: F) -> Callable[[F], F]:
    @wraps(func)
    def wrapper(*args, **kwargs):
        return value

    return wrapper
