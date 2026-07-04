from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchdcm.spec.parameters import Beta


@dataclass(frozen=True)
class Term:
    parameter: "Beta"
    variable: str | None = None
    multiplier: float = 1.0


@dataclass(frozen=True)
class Expression:
    terms: list[Term]

    @property
    def parameters(self) -> list["Beta"]:
        params = {}
        for term in self.terms:
            old = params.get(term.parameter.name)
            if old is not None and old != term.parameter:
                raise ValueError(f"Conflicting definitions for parameter {term.parameter.name!r}.")
            params[term.parameter.name] = term.parameter
        return list(params.values())

    def __add__(self, other) -> "Expression":
        if other == 0:
            return self
        if isinstance(other, Expression):
            return Expression([*self.terms, *other.terms])
        if hasattr(other, "name") and hasattr(other, "init"):
            return Expression([*self.terms, Term(other, None, 1.0)])
        raise TypeError(f"Cannot add {type(other)!r} to a utility expression.")

    def __radd__(self, other) -> "Expression":
        return self.__add__(other)

    def __sub__(self, other) -> "Expression":
        return self + (-_as_expression(other))

    def __rsub__(self, other) -> "Expression":
        return _as_expression(other) + (-self)

    def __neg__(self) -> "Expression":
        return Expression([Term(t.parameter, t.variable, -t.multiplier) for t in self.terms])


def _as_expression(value) -> Expression:
    if isinstance(value, Expression):
        return value
    if hasattr(value, "name") and hasattr(value, "init"):
        return Expression([Term(value, None, 1.0)])
    raise TypeError(f"Cannot convert {type(value)!r} to a utility expression.")
