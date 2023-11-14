from __future__ import annotations
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

T = TypeVar("T")

def first(iterable: Iterable[T], default: T | None = None) -> T | None:
    """
    Returns the first item in the given iterable or `default` if empty, meaningful mostly with 'for' expressions.
    """
    for i in iterable:
        return i
    return default