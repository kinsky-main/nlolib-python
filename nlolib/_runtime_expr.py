"""
Private lowering of operator callables to the C API's runtime expression strings.

The C API consumes operator expressions as strings; the Python API accepts only
callables, which are translated here. Nothing in this module is public.
"""

from __future__ import annotations

import ast
import inspect
import math
import textwrap
from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable, Mapping

RUNTIME_CONTEXT_DISPERSION = "dispersion"
RUNTIME_CONTEXT_DISPERSION_FACTOR = "dispersion_factor"
RUNTIME_CONTEXT_NONLINEAR = "nonlinear"
RUNTIME_CONTEXT_LINEAR_FACTOR = "linear_factor"
RUNTIME_CONTEXT_LINEAR = "linear"
RUNTIME_CONTEXT_POTENTIAL = "potential"

_ALLOWED_FUNCS = {"exp", "log", "sqrt", "sin", "cos"}

# Per context: the symbols a parameter binds to, in binding order, and the
# aliases the context also accepts. These are what the C parser allows
# (src/physics/operator_program_compile.c, context_allows_symbol_*), minus the
# x, y, t it also accepts in the nonlinear context and A in the potential one,
# which the bindings do not expose.
_CONTEXT_PARAMETERS: dict[str, list[str]] = {
    RUNTIME_CONTEXT_DISPERSION_FACTOR: ["A", "w"],
    RUNTIME_CONTEXT_DISPERSION: ["A", "D", "h", "w", "V"],
    RUNTIME_CONTEXT_LINEAR_FACTOR: ["A", "wt", "kx", "ky", "x", "y", "t"],
    RUNTIME_CONTEXT_LINEAR: ["A", "D", "h", "wt", "kx", "ky", "x", "y", "t", "V"],
    RUNTIME_CONTEXT_POTENTIAL: ["x", "y", "t"],
    RUNTIME_CONTEXT_NONLINEAR: ["A", "I", "V"],
}

# w and wt name the same axis in every linear context.
_CONTEXT_ALIASES: dict[str, dict[str, str]] = {
    RUNTIME_CONTEXT_DISPERSION_FACTOR: {"wt": "w"},
    RUNTIME_CONTEXT_DISPERSION: {"wt": "w"},
    RUNTIME_CONTEXT_LINEAR_FACTOR: {"w": "wt"},
    RUNTIME_CONTEXT_LINEAR: {"w": "wt"},
    RUNTIME_CONTEXT_POTENTIAL: {},
    RUNTIME_CONTEXT_NONLINEAR: {},
}

# Every name that is a symbol in some context. A free name from this set that
# the current context does not list is an error, not a captured constant.
_ALL_SYMBOLS = {
    symbol
    for context, parameters in _CONTEXT_PARAMETERS.items()
    for symbol in (*parameters, *_CONTEXT_ALIASES[context])
}


def _context_symbol(context: str, name: str) -> str | None:
    """The C symbol `name` denotes in `context`, or None if it is not one."""
    if name in _CONTEXT_PARAMETERS[context]:
        return name
    return _CONTEXT_ALIASES[context].get(name)


@dataclass
class TranslatedExpression:
    expression: str
    constants: list[float]


def _is_real_scalar(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _format_real(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("runtime expression constants must be finite real scalars")
    return format(float(value), ".17g")


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _resolve_attribute_chain(node: ast.Attribute, scope: Mapping[str, Any]) -> Any:
    chain: list[str] = [node.attr]
    cursor: ast.AST = node.value
    while isinstance(cursor, ast.Attribute):
        chain.append(cursor.attr)
        cursor = cursor.value
    if not isinstance(cursor, ast.Name):
        raise ValueError("unsupported attribute chain in runtime callable")

    base_name = cursor.id
    if base_name not in scope:
        raise ValueError(f"unresolved attribute base '{base_name}' in runtime callable")
    value = scope[base_name]
    for attr in reversed(chain):
        if not hasattr(value, attr):
            raise ValueError(f"unresolved attribute '{attr}' in runtime callable")
        value = getattr(value, attr)
    return value


def _lambda_names(node: ast.Lambda) -> tuple[tuple[str, ...], frozenset[str]]:
    args = tuple(arg.arg for arg in node.args.posonlyargs + node.args.args)
    names = {
        n.id if isinstance(n, ast.Name) else n.attr
        for n in ast.walk(node.body)
        if isinstance(n, (ast.Name, ast.Attribute))
    }
    return args, frozenset(names - set(args))


def _func_names(func: Callable[..., Any]) -> tuple[tuple[str, ...], frozenset[str]]:
    code = func.__code__
    args = tuple(code.co_varnames[: code.co_argcount])
    return args, frozenset(set(code.co_names) | set(code.co_freevars))


def _first_column(func: Callable[..., Any]) -> int | None:
    """Column of the function's first instruction on its first line (3.11+)."""
    positions = getattr(func.__code__, "co_positions", None)
    if positions is None:
        return None
    first_line = func.__code__.co_firstlineno
    # Skip the zero-width bookkeeping instructions (RESUME) at column 0.
    columns = [
        col for line, _, col, end_col in positions()
        if line == first_line and col is not None and end_col is not None and end_col > col
    ]
    return min(columns) if columns else None


def _matching_lambda(func: Callable[..., Any], lambdas: list[ast.Lambda], column_shift: int = 0) -> ast.Lambda:
    """The lambda in `lambdas` that `func` was compiled from.

    getsource returns whole lines, so a line holding several lambdas -- both
    operators of one propagate call, say -- yields all of them; taking the first
    would lower every one of them to the first one's body.
    """
    if len(lambdas) == 1:
        return lambdas[0]
    want = _func_names(func)
    candidates = [node for node in lambdas if _lambda_names(node) == want] or list(lambdas)
    if len(candidates) > 1:
        column = _first_column(func)
        if column is not None:
            column -= column_shift
            by_column = [
                node for node in candidates
                if node.col_offset <= column <= (node.end_col_offset or node.col_offset)
            ]
            if by_column:
                # The innermost lambda that contains the first instruction.
                candidates = [max(by_column, key=lambda node: node.col_offset)]
    if len(candidates) > 1:
        dumps = {ast.dump(node.body) for node in candidates}
        if len(dumps) > 1:
            raise ValueError(
                "several lambdas on the operator's source line match it; "
                "put each operator on a line of its own"
            )
    return candidates[0]


def _lambdas_in_fragment(source: str) -> list[ast.Lambda]:
    """Every lambda expression that parses starting at a `lambda` keyword."""
    found: list[ast.Lambda] = []
    start = source.find("lambda")
    while start >= 0:
        for end in range(len(source), start, -1):
            try:
                tree = ast.parse("(" + source[start:end] + ")", mode="eval")
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Lambda):
                    # Offsets are into "(" + fragment; move them back to `source`.
                    if node.lineno == 1:
                        node.col_offset += start - 1
                    if node.end_lineno == 1 and node.end_col_offset is not None:
                        node.end_col_offset += start - 1
                    found.append(node)
            break
        start = source.find("lambda", start + 1)
    return found


def _extract_expr_from_callable(func: Callable[..., Any]) -> ast.AST:
    try:
        raw = inspect.getsource(func)
    except (OSError, TypeError) as exc:
        raise ValueError(
            "unable to inspect callable source; define the operator as a lambda or def in a source file"
        ) from exc

    source = textwrap.dedent(raw)
    # Columns in `source` are this much left of the columns in the file.
    raw_first = raw.splitlines()[0] if raw else ""
    source_first = source.splitlines()[0] if source else ""
    column_shift = len(raw_first) - len(source_first)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # getsource returns whole lines, so a lambda in a dict entry or a
        # one-line compound statement comes back as a fragment that does not
        # parse on its own; parse the expression at each `lambda` instead.
        lambdas = _lambdas_in_fragment(source)
        if not lambdas:
            raise ValueError(
                "could not parse the operator's source line; define it on a line of its own"
            ) from None
        return _matching_lambda(func, lambdas, column_shift).body

    lambdas = [node for node in ast.walk(tree) if isinstance(node, ast.Lambda)]
    if lambdas:
        return _matching_lambda(func, lambdas, column_shift).body

    fn_name = getattr(func, "__name__", None)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            for stmt in node.body:
                if isinstance(stmt, ast.Return):
                    return stmt.value
            raise ValueError("function callable must contain a return expression")

    raise ValueError("could not extract expression body from callable")


class _ExpressionTranslator:
    def __init__(
        self,
        context: str,
        symbol_map: Mapping[str, str],
        captured_scope: Mapping[str, Any],
        constant_bindings: Mapping[str, float] | None,
    ):
        self._context = context
        self._symbol_map = dict(symbol_map)
        self._captured_scope = dict(captured_scope)
        self._constant_bindings = dict(constant_bindings or {})
        self._constants: list[float] = []
        self._constants_by_name: dict[str, int] = {}


    def _add_named_constant(self, name: str, value: float) -> str:
        if name in self._constants_by_name:
            idx = self._constants_by_name[name]
        else:
            idx = len(self._constants)
            self._constants_by_name[name] = idx
            self._constants.append(float(value))
        return f"c{idx}"

    def _resolve_name_constant(self, name: str) -> str:
        if name in self._constant_bindings:
            value = self._constant_bindings[name]
            if not _is_real_scalar(value):
                raise ValueError(f"constant binding '{name}' must be a finite real scalar")
            return self._add_named_constant(f"binding:{name}", float(value))

        if name not in self._captured_scope:
            raise ValueError(
                f"unknown identifier '{name}' in operator; "
                "name it in constant_bindings or capture a real scalar"
            )

        value = self._captured_scope[name]
        if isinstance(value, complex):
            if value.real == 0.0 and value.imag == 1.0:
                return "i"
            if value.real == 0.0 and value.imag == -1.0:
                return "(-i)"
            raise ValueError(
                f"captured complex identifier '{name}' is unsupported; use real scalars or 'i'"
            )

        if not _is_real_scalar(value):
            raise ValueError(
                f"captured identifier '{name}' must be a finite real scalar"
            )
        return self._add_named_constant(name, float(value))

    def _translate_constant(self, value: Any) -> str:
        if isinstance(value, complex):
            if value.real == 0.0 and value.imag == 1.0:
                return "i"
            if value.real == 0.0 and value.imag == -1.0:
                return "(-i)"
            if value.imag == 0.0:
                return _format_real(float(value.real))
            raise ValueError("complex literals other than +/-1j are unsupported")
        if _is_real_scalar(value):
            return _format_real(float(value))
        raise ValueError(f"unsupported literal in runtime callable: {value!r}")

    def translate(self, node: ast.AST) -> str:
        if isinstance(node, ast.BinOp):
            left = self.translate(node.left)
            right = self.translate(node.right)
            if isinstance(node.op, ast.Add):
                return f"({left}+{right})"
            if isinstance(node.op, ast.Sub):
                return f"({left}-{right})"
            if isinstance(node.op, ast.Mult):
                return f"({left}*{right})"
            if isinstance(node.op, ast.Div):
                return f"({left}/{right})"
            if isinstance(node.op, ast.Pow):
                return f"({left}^{right})"
            raise ValueError("unsupported binary operator in runtime callable")

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                operand = self.translate(node.operand)
                return f"(-{operand})"
            if isinstance(node.op, ast.UAdd):
                return self.translate(node.operand)
            raise ValueError("unsupported unary operator in runtime callable")

        if isinstance(node, ast.Call):
            fn_name = _call_name(node.func)
            if fn_name is None or fn_name not in _ALLOWED_FUNCS:
                raise ValueError("unsupported function call in runtime callable")
            if len(node.args) != 1 or node.keywords:
                raise ValueError(f"{fn_name}() requires exactly one positional argument")
            arg_expr = self.translate(node.args[0])
            return f"{fn_name}({arg_expr})"

        if isinstance(node, ast.Name):
            if node.id in self._symbol_map:
                return self._symbol_map[node.id]

            if node.id == "i":
                return node.id
            if node.id in _ALL_SYMBOLS:
                symbol = _context_symbol(self._context, node.id)
                if symbol is None:
                    raise ValueError(
                        f"'{node.id}' is not valid in {self._context} operators; "
                        f"allowed: {', '.join(_CONTEXT_PARAMETERS[self._context])}"
                    )
                return symbol

            return self._resolve_name_constant(node.id)

        if isinstance(node, ast.Attribute):
            value = _resolve_attribute_chain(node, self._captured_scope)
            return self._translate_constant(value)

        if isinstance(node, ast.Constant):
            return self._translate_constant(node.value)

        raise ValueError(
            f"unsupported expression node '{type(node).__name__}' in runtime callable"
        )

    @property
    def constants(self) -> list[float]:
        return list(self._constants)


def translate_callable(
    func: Callable[..., Any],
    context: str,
    *,
    constant_bindings: Mapping[str, float] | None = None,
) -> TranslatedExpression:
    """Lower one operator callable to C expression text for `context`.

    A parameter named after a symbol of the context (or an alias of one) binds
    to it; the others take, in order, the next unused symbol in the context's
    parameter order. A parameter named after a symbol the context does not
    provide, or two parameters naming the same symbol, is a ValueError. Free names are captured from the callable's scope; a name in
    `constant_bindings` wins over a captured one.
    """
    if context not in _CONTEXT_PARAMETERS:
        raise ValueError(f"context must be one of: {', '.join(sorted(_CONTEXT_PARAMETERS))}")

    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    for param in params:
        if param.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise ValueError("operator callables must only use positional parameters")

    parameter_order = _CONTEXT_PARAMETERS[context]
    if len(params) < 1 or len(params) > len(parameter_order):
        raise ValueError(
            f"{context} operator must take between 1 and {len(parameter_order)} "
            f"positional arguments ({', '.join(parameter_order)})"
        )

    symbol_map: dict[str, str] = {}
    used_symbols: set[str] = set()
    for param in params:
        if param.name not in _ALL_SYMBOLS:
            continue
        symbol = _context_symbol(context, param.name)
        if symbol is None:
            raise ValueError(
                f"parameter '{param.name}' names a symbol the {context} operator does not "
                f"provide; allowed: {', '.join(parameter_order)}"
            )
        if symbol in used_symbols:
            raise ValueError(
                f"parameter '{param.name}' and another parameter both name '{symbol}' "
                f"in the {context} operator"
            )
        symbol_map[param.name] = symbol
        used_symbols.add(symbol)
    for param in params:
        if param.name in symbol_map:
            continue
        symbol = next((s for s in parameter_order if s not in used_symbols), None)
        if symbol is None:
            raise ValueError(f"unsupported parameter '{param.name}' for {context} operator")
        symbol_map[param.name] = symbol
        used_symbols.add(symbol)

    closure_vars = inspect.getclosurevars(func)
    captured_scope: dict[str, Any] = {}
    captured_scope.update(closure_vars.builtins)
    captured_scope.update(closure_vars.globals)
    captured_scope.update(closure_vars.nonlocals)

    expr_node = _extract_expr_from_callable(func)
    translator = _ExpressionTranslator(
        context=context,
        symbol_map=symbol_map,
        captured_scope=captured_scope,
        constant_bindings=constant_bindings,
    )
    expression = translator.translate(expr_node)
    return TranslatedExpression(expression=expression, constants=translator.constants)
