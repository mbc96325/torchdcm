from __future__ import annotations

import ast
from collections import OrderedDict
from dataclasses import dataclass, field

from torchdcm.spec.expressions import Expression, Term
from torchdcm.spec.parameters import Beta


@dataclass
class UtilitySpec:
    """Alternative-specific utility specification."""

    utilities: OrderedDict[str, Expression] = field(default_factory=OrderedDict)

    def utility(self, alternative: str, expression: Expression | Beta) -> "UtilitySpec":
        if isinstance(expression, Beta):
            expression = Expression([Term(expression, None, 1.0)])
        if not isinstance(expression, Expression):
            raise TypeError("Utility expression must be built from Beta terms.")
        self.utilities[alternative] = expression
        return self

    @property
    def parameter_names(self) -> list[str]:
        names: list[str] = []
        for expr in self.utilities.values():
            for term in expr.terms:
                if term.parameter.name not in names:
                    names.append(term.parameter.name)
        return names

    @property
    def parameters(self) -> list[Beta]:
        params: OrderedDict[str, Beta] = OrderedDict()
        for expr in self.utilities.values():
            for term in expr.terms:
                old = params.get(term.parameter.name)
                if old is not None and old != term.parameter:
                    raise ValueError(f"Conflicting definitions for parameter {term.parameter.name!r}.")
                params[term.parameter.name] = term.parameter
        return list(params.values())

    @classmethod
    def from_formula(cls, utilities: dict[str, str], *, fixed: set[str] | None = None) -> "UtilitySpec":
        """Create a specification from simple linear utility formulas."""

        fixed = fixed or set()
        # All alternatives share this registry, so writing ``B_TIME * time`` in
        # several formulas creates one generic coefficient rather than copies.
        registry: dict[str, Beta] = {}

        def beta(name: str) -> Beta:
            if name not in registry:
                registry[name] = Beta(name, fixed=name in fixed)
            return registry[name]

        def parse_node(node: ast.AST) -> Expression:
            # Deliberately accept only a small linear grammar.  Rejecting calls
            # and arbitrary Python expressions keeps formula parsing auditable.
            if isinstance(node, ast.Expression):
                return parse_node(node.body)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                return parse_node(node.left) + parse_node(node.right)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
                return parse_node(node.left) - parse_node(node.right)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                return -parse_node(node.operand)
            if isinstance(node, ast.Name):
                if node.id[:1].isupper():
                    return beta(node.id)
                raise ValueError(f"Variable {node.id!r} must be multiplied by a parameter.")
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                if isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name):
                    left, right = node.left.id, node.right.id
                    if left[:1].isupper() and not right[:1].isupper():
                        return beta(left) * right
                    if right[:1].isupper() and not left[:1].isupper():
                        return beta(right) * left
                raise ValueError("Formula terms must look like PARAM * variable.")
            raise ValueError(f"Unsupported formula component: {ast.dump(node)}")

        spec = cls()
        for alt, formula in utilities.items():
            spec.utility(alt, parse_node(ast.parse(formula, mode="eval")))
        return spec
