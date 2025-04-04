import inspect
import sys
from functools import wraps
from inspect import signature
from itertools import chain
from typing import Any, Callable, Literal, Type, TypeVar

FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)


# Class specific helper functions
def init_func_args(cls: Type) -> list[str]:
    return [k for k in signature(cls.__init__).parameters if k != "self"]


# def verify_types(cls: Type) -> list[str]:
#     return [k for k in signature(cls.__init__).parameters if k != "self"]


def get_all_slots(obj: object):
    return chain.from_iterable(
        getattr(cls, "__slots__", tuple()) for cls in reversed(type(obj).__mro__)
    )


# Universal helper
def get_all_of_object_type(
    module_name: str,
    object_type: Literal["class", "function", "method", "module"] = "class",
) -> dict[str, Any]:
    predicate = getattr(inspect, f"is{object_type}")

    return dict(
        inspect.getmembers(
            sys.modules[module_name],
            lambda member: predicate(member) and member.__module__ == module_name,
        )
    )


# Testing decorator capabilities
def decorator(func: F) -> Callable[[F], F]:
    @wraps(func)
    def wrapper(*args, **kwargs):
        return value

    return wrapper
