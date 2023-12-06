from __future__ import annotations

from datetime import timedelta
from typing import Callable, TypeVar

from hypothesis import HealthCheck, settings

_T = TypeVar("_T")


def allow_function_scoped(callable: _T, /) -> _T:
    """Allow function_scoped fixtures in tests"""
    return settings(
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )(callable)


def deadline(time: int | float | timedelta | None) -> Callable[[_T], _T]:
    """Change hypothesis deadline"""

    def inner(callable: _T, /) -> _T:
        return settings(deadline=time)(callable)

    return inner


def allow_function_scoped_deadline(
    time: int | float | timedelta | None,
) -> Callable[[_T], _T]:
    """Allow function scoped fixtures and change hypothesis deadline"""

    def inner(callable: _T, /) -> _T:
        return settings(
            suppress_health_check=[HealthCheck.function_scoped_fixture],
            deadline=time,
        )(callable)

    return inner
