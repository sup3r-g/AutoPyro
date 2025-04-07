import inspect
import sys
from functools import wraps
from inspect import signature
from itertools import chain
from typing import Any, Callable, Literal, Type, TypeVar


Func = TypeVar("Func", bound=Callable[..., Any])


# Class specific helper functions
# cls: Type
def get_all_slots(obj: object):
    return chain.from_iterable(
        getattr(cls, "__slots__", tuple()) for cls in reversed(type(obj).__mro__)
    )


# def verify_types(cls: Type) -> list[str]:
#     return [k for k in signature(cls.__init__).parameters if k != "self"]


# Universal helper
def func_args(function: Callable) -> list[str]:
    return [k for k in signature(function).parameters if k != "self"]


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
def decorator(func: Func) -> Callable[[Func], Func]:
    @wraps(func)
    def wrapper(*args, **kwargs):
        return value

    return wrapper
