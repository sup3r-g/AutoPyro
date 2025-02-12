from inspect import signature
from typing import Type


def func_args(cls: Type) -> list[str]:
    return [k for k in signature(cls.__init__).parameters if k != "self"]


# def verify_types(cls: Type) -> list[str]:
#     return [k for k in signature(cls.__init__).parameters if k != "self"]
