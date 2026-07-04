from __future__ import annotations

from dataclasses import dataclass

from torchdcm.spec.expressions import Expression, Term


@dataclass(frozen=True)
class Beta:
    """A scalar utility coefficient."""

    name: str
    init: float = 0.0
    fixed: bool = False

    def __mul__(self, other: str | float | int) -> Expression:
        if isinstance(other, str):
            return Expression([Term(self, other, 1.0)])
        if isinstance(other, (int, float)):
            return Expression([Term(self, None, float(other))])
        return NotImplemented

    def __rmul__(self, other: str | float | int) -> Expression:
        return self.__mul__(other)

    def __add__(self, other) -> Expression:
        return Expression([Term(self, None, 1.0)]) + other

    def __radd__(self, other) -> Expression:
        return self.__add__(other)

    def __sub__(self, other) -> Expression:
        return Expression([Term(self, None, 1.0)]) - other

    def __rsub__(self, other) -> Expression:
        return (-Expression([Term(self, None, 1.0)])) + other

    def __neg__(self) -> Expression:
        return Expression([Term(self, None, -1.0)])

