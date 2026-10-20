"""Updater artifacts: the object space the improver edits (TASK_017 PR A1, entry 16).

An artifact is a forecast-v2 updater written as a plain module: module-level
``init`` and ``update`` functions, arithmetic in explicit left-to-right loops,
state threaded through the state object, and no imports at all. The loader
executes the module in a sandbox namespace whose globals are exactly the
pinned builtins and ``math`` (its pure functions only), so an ``import``
statement could not run there even if one were allowed.

``validate_artifact_source`` pins that grammar by AST walk: it is a style and
protocol guard — it keeps an LLM proposal a *kind* of program (no imports, no
f-strings, no ``while``/``try``, no comprehensions, only the pinned calls and
``raise ValueError(<one argument>)``) — not a wall-clock or resource sandbox;
the arena still validates every output an artifact returns, and (f) runs each
evaluation under a timeout. Every rejection message names the construct.

The reference artifact is ``bias_share`` translated into the grammar
operation for operation (``updater/forecast.py`` with ``updater/baselines.py``
``_share_step`` inlined): the lineage's starting point. The float discipline —
the same operations in the same order — makes the translation bit-equal to the
class original on replay, which ``tests/improver/test_artifacts.py`` pins.

Imports here are only the standard library and ``evolvets.updater.interfaces``
(the contract): the direction is improver -> updater, never the reverse, and
never an arena, cache or data module (``.importlinter``).
"""

import ast
import difflib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import cast

from evolvets.updater.interfaces import (
    ConfigValue,
    Context,
    ForecastOutput,
    ForecastUpdater,
    Matured,
    Output,
    Weights,
)


class ArtifactError(ValueError):
    """A source that is not a valid updater artifact; the message names the construct."""


#: The reference artifact's configuration: ``bias_share``'s placeholder config.
REFERENCE_ARTIFACT_CONFIG: Mapping[str, float] = MappingProxyType(
    {"eta": 1.0, "alpha": 0.01, "bias_rate": 0.1}
)

#: The pure functions an artifact may call on the loader-provided ``math``.
MATH_MEMBERS: frozenset[str] = frozenset({"exp", "sqrt", "log", "isfinite", "fabs"})

#: The sandbox's entire builtin set: 17 names, each mapped to the real builtin,
#: so the names resolve both as calls and as plain references (``isinstance``
#: checks, ``ValueError`` raises). No ``__import__``, no ``open``, no ``eval``.
ARTIFACT_BUILTINS: Mapping[str, object] = MappingProxyType(
    {
        "abs": abs,
        "min": min,
        "max": max,
        "float": float,
        "int": int,
        "bool": bool,
        "len": len,
        "range": range,
        "tuple": tuple,
        "list": list,
        "dict": dict,
        "sorted": sorted,
        "enumerate": enumerate,
        "zip": zip,
        "isinstance": isinstance,
        "str": str,
        "ValueError": ValueError,
    }
)

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.FloorDiv)

_OPERATOR_SYMBOLS: dict[type[ast.AST], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Mod: "%",
    ast.FloorDiv: "//",
    ast.Pow: "**",
    ast.MatMult: "@",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.Invert: "~",
}

_STATEMENT_NAMES: dict[type[ast.stmt], str] = {
    ast.While: "while",
    ast.Try: "try",
    ast.TryStar: "try",
    ast.Import: "import",
    ast.ImportFrom: "import (from ... import ...)",
    ast.ClassDef: "class",
    ast.Break: "break",
    ast.Continue: "continue",
    ast.Delete: "delete",
    ast.Global: "global",
    ast.Nonlocal: "nonlocal",
    ast.Assert: "assert",
    ast.With: "with",
    ast.AsyncWith: "async with",
    ast.AsyncFor: "async for",
    ast.AsyncFunctionDef: "async def",
    ast.AnnAssign: "a variable annotation statement (x: T)",
}

_EXPRESSION_NAMES: dict[type[ast.expr], str] = {
    ast.JoinedStr: "an f-string",
    ast.FormattedValue: "an f-string",
    ast.NamedExpr: "the walrus operator (:=)",
    ast.ListComp: "a comprehension",
    ast.SetComp: "a comprehension",
    ast.DictComp: "a comprehension",
    ast.GeneratorExp: "a comprehension",
    ast.Starred: "a starred expression",
    ast.Await: "await",
    ast.Yield: "yield",
    ast.YieldFrom: "yield",
    ast.Lambda: "lambda",
}

_PINNED_PARAMETERS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "init": ("n_experts", "horizon", "config", "seed"),
        "update": ("state", "feedback", "context"),
    }
)


class _Validator:
    """One AST walk over one artifact; the first rule violated raises ArtifactError.

    Module body: an optional docstring, then exactly the two top-level defs
    ``init`` and ``update`` — nothing else of any kind. Parameter annotations
    are allowed (and validated as expressions); variable annotation
    statements, return annotations, decorators and default values are not.
    """

    def __init__(self, tree: ast.Module) -> None:
        self._names = frozenset(
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        )

    # --- module and functions ---------------------------------------------------------------------

    def check_module(self, tree: ast.Module) -> None:
        """Check the module body holds exactly the two pinned defs."""
        body = list(tree.body)
        if body and _is_docstring(body[0]):
            body = body[1:]
        if len(body) != 2:
            found = ", ".join(self._describe(node) for node in body) or "nothing"
            raise ArtifactError(
                f"the module must hold exactly the two top-level defs "
                f"init and update, found {found}"
            )
        for node, expected in zip(body, ("init", "update"), strict=True):
            if not isinstance(node, ast.FunctionDef):
                raise ArtifactError(
                    f"the top-level statement must be def {expected}(...), "
                    f"got {self._describe(node)}"
                )
            if node.name != expected:
                raise ArtifactError(f"a top-level def must be named {expected}, got {node.name!r}")
            self._check_signature(node, _PINNED_PARAMETERS[expected])
            self._check_statements(node.body)

    def _check_signature(self, node: ast.FunctionDef, pinned: tuple[str, ...]) -> None:
        """Check the parameter list: plain names, no defaults, optional annotations."""
        args = node.args
        parameters = [*args.posonlyargs, *args.args]
        if args.vararg is not None or args.kwarg is not None or args.kwonlyargs:
            raise ArtifactError(f"{node.name} must not take *args, keyword-only or **kwargs")
        if args.defaults or any(default is not None for default in args.kw_defaults):
            raise ArtifactError(f"{node.name} must not use default parameter values")
        if node.decorator_list:
            raise ArtifactError(f"{node.name} must not carry a decorator")
        if node.returns is not None:
            raise ArtifactError(f"{node.name} must not annotate its return")
        names = tuple(parameter.arg for parameter in parameters)
        if names != pinned:
            raise ArtifactError(
                f"{node.name} must take exactly the parameters {list(pinned)}, got {list(names)}"
            )
        for parameter in parameters:
            if parameter.annotation is not None:
                self._check_expression(parameter.annotation)

    def _check_nested_def(self, node: ast.FunctionDef) -> None:
        """Check a nested helper: plain named parameters, its body under the same rules."""
        own = tuple(parameter.arg for parameter in [*node.args.posonlyargs, *node.args.args])
        self._check_signature(node, own)
        self._check_statements(node.body)

    # --- statements ---------------------------------------------------------------

    def _check_statements(self, statements: Sequence[ast.stmt]) -> None:
        for statement in statements:
            self._check_statement(statement)

    def _check_statement(self, statement: ast.stmt) -> None:
        if isinstance(statement, ast.FunctionDef):
            self._check_nested_def(statement)
        elif isinstance(statement, ast.For):
            self._check_target(statement.target)
            self._check_expression(statement.iter)
            self._check_statements(statement.body)
            self._check_statements(statement.orelse)
        elif isinstance(statement, ast.If):
            self._check_expression(statement.test)
            self._check_statements(statement.body)
            self._check_statements(statement.orelse)
        elif isinstance(statement, ast.Return):
            if statement.value is not None:
                self._check_expression(statement.value)
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                self._check_target(target)
            self._check_expression(statement.value)
        elif isinstance(statement, ast.AugAssign):
            if not isinstance(statement.op, _ALLOWED_BINOPS):
                raise ArtifactError(
                    f"the augmented assignment operator {self._symbol(statement.op)} is not "
                    "allowed; only + - * / % //"
                )
            self._check_target(statement.target)
            self._check_expression(statement.value)
        elif isinstance(statement, ast.Pass):
            pass
        elif isinstance(statement, ast.Expr):
            value = statement.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                pass  # an inner docstring
            elif isinstance(value, ast.Call):
                self._check_call(value)
            else:
                raise ArtifactError(
                    "an expression statement must be a string constant or an allowed call, "
                    f"got {self._expression_name(value)}"
                )
        elif isinstance(statement, ast.Raise):
            self._check_raise(statement)
        else:
            raise ArtifactError(f"{self._statement_name(statement)} is not allowed in an artifact")

    def _check_raise(self, statement: ast.Raise) -> None:
        """Allow exactly ``raise ValueError(<one argument>)``; nothing else raises."""
        if statement.exc is None:
            raise ArtifactError("a bare raise is not allowed; only raise ValueError(<argument>)")
        if statement.cause is not None:
            raise ArtifactError(
                "raise ... from ... is not allowed; only raise ValueError(<argument>)"
            )
        call = statement.exc
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Name)
            or call.func.id != "ValueError"
        ):
            raised = (
                f"raise {call.func.id}(...)"
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                else self._expression_name(call)
            )
            raise ArtifactError(f"only raise ValueError(<argument>) is allowed, got {raised}")
        if len(call.args) != 1 or call.keywords:
            raise ArtifactError("raise ValueError must be given exactly one argument")
        self._check_expression(call.args[0])

    # --- expressions --------------------------------------------------------------

    def _check_expression(self, node: ast.expr) -> None:
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, (bool, str)):
                return
            if isinstance(value, (int, float)):
                if not math.isfinite(value):
                    raise ArtifactError(f"the numeric constant {value!r} must be finite")
                return
            raise ArtifactError(
                f"a constant of type {type(value).__name__} is not allowed; "
                "only str, bool, int and float"
            )
        if isinstance(node, ast.Name):
            return
        if isinstance(node, ast.Attribute):
            self._check_attribute(node)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for element in node.elts:
                self._check_expression(element)
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if key is None:
                    raise ArtifactError("dict unpacking ({**...}) is not allowed")
                self._check_expression(key)
            for entry in node.values:
                self._check_expression(entry)
        elif isinstance(node, ast.Subscript):
            self._check_expression(node.value)
            self._check_index(node.slice)
        elif isinstance(node, ast.Call):
            self._check_call(node)
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BINOPS):
                raise ArtifactError(
                    f"the operator {self._symbol(node.op)} is not allowed; only + - * / % //"
                )
            self._check_expression(node.left)
            self._check_expression(node.right)
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
                raise ArtifactError(
                    f"the unary operator {self._symbol(node.op)} is not allowed; only - + not"
                )
            self._check_expression(node.operand)
        elif isinstance(node, ast.Compare):
            self._check_expression(node.left)
            for comparator in node.comparators:
                self._check_expression(comparator)
        elif isinstance(node, ast.BoolOp):
            for operand in node.values:
                self._check_expression(operand)
        elif isinstance(node, ast.IfExp):
            self._check_expression(node.test)
            self._check_expression(node.body)
            self._check_expression(node.orelse)
        else:
            raise ArtifactError(f"{self._expression_name(node)} is not allowed in an artifact")

    def _check_attribute(self, node: ast.Attribute) -> None:
        """Allow any non-dunder attribute; on ``math`` only MATH_MEMBERS."""
        self._check_expression(node.value)
        if node.attr.startswith("__") and node.attr.endswith("__"):
            raise ArtifactError(f"the dunder attribute .{node.attr} is not allowed")
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "math"
            and node.attr not in MATH_MEMBERS
        ):
            raise ArtifactError(f"math.{node.attr} is not allowed; only {sorted(MATH_MEMBERS)}")

    def _check_call(self, node: ast.Call) -> None:
        """Allow pinned builtins, artifact-defined functions, math members and .pop/.append."""
        function = node.func
        if isinstance(function, ast.Name):
            if function.id not in ARTIFACT_BUILTINS and function.id not in self._names:
                raise ArtifactError(
                    f"the call to {function.id!r} is not allowed; only the pinned builtins "
                    f"({sorted(ARTIFACT_BUILTINS)}) and the artifact's own functions"
                )
        elif isinstance(function, ast.Attribute):
            self._check_expression(function.value)
            if isinstance(function.value, ast.Name) and function.value.id == "math":
                if function.attr not in MATH_MEMBERS:
                    raise ArtifactError(
                        f"math.{function.attr} is not allowed; only {sorted(MATH_MEMBERS)}"
                    )
            elif function.attr in ("pop", "append"):
                if len(node.args) != 1 or node.keywords:
                    raise ArtifactError(
                        f".{function.attr} must be called with exactly one argument"
                    )
            else:
                raise ArtifactError(
                    f"the method call .{function.attr} is not allowed; only .pop(x) and .append(x)"
                )
        else:
            raise ArtifactError("a call through an expression is not allowed")
        for argument in node.args:
            self._check_expression(argument)
        for keyword in node.keywords:
            if keyword.arg is None:
                raise ArtifactError("**kwargs in a call is not allowed")
            self._check_expression(keyword.value)

    def _check_index(self, node: ast.expr) -> None:
        """Allow any expression as an index, and slices whose parts are expressions."""
        if isinstance(node, ast.Slice):
            for part in (node.lower, node.upper, node.step):
                if part is not None:
                    self._check_expression(part)
        else:
            self._check_expression(node)

    def _check_target(self, node: ast.expr) -> None:
        """Allow names, subscript stores and tuple/list unpacking as assignment targets."""
        if isinstance(node, ast.Name):
            return
        if isinstance(node, (ast.Tuple, ast.List)):
            for element in node.elts:
                self._check_target(element)
            return
        if isinstance(node, ast.Subscript):
            self._check_expression(node)
            return
        if isinstance(node, ast.Starred):
            raise ArtifactError("a starred assignment target is not allowed")
        raise ArtifactError(f"an assignment target of type {type(node).__name__} is not allowed")

    # --- message helpers ----------------------------------------------------------

    def _describe(self, node: ast.stmt) -> str:
        """Name a top-level statement for a rejection message."""
        if isinstance(node, ast.FunctionDef):
            return f"def {node.name}"
        if isinstance(node, ast.AsyncFunctionDef):
            return f"async def {node.name}"
        if isinstance(node, ast.ClassDef):
            return f"class {node.name}"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "an import"
        if isinstance(node, ast.Assign):
            return "a module-level assignment"
        return f"a {self._statement_name(node)} statement"

    def _symbol(self, operator: ast.AST) -> str:
        return _OPERATOR_SYMBOLS.get(type(operator), type(operator).__name__)

    def _statement_name(self, statement: ast.stmt) -> str:
        return _STATEMENT_NAMES.get(type(statement), type(statement).__name__.lower())

    def _expression_name(self, expression: ast.expr) -> str:
        if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name):
            return f"the call to {expression.func.id!r}"
        return _EXPRESSION_NAMES.get(type(expression), f"a {type(expression).__name__} expression")


def _is_docstring(statement: ast.stmt) -> bool:
    """Whether a statement is a bare string constant (a docstring)."""
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def validate_artifact_source(source: str) -> None:
    """Raise :class:`ArtifactError` unless ``source`` is exactly the artifact grammar.

    The grammar: a module holding an optional docstring and exactly the two
    top-level defs ``init(n_experts, horizon, config, seed)`` and
    ``update(state, feedback, context)``; inside them only ``def``, ``for``,
    ``if``, ``return``, assignments, augmented assignments, ``pass``, string
    docstrings, allowed calls as statements and ``raise ValueError(<one
    argument>)``; and, within expressions, only finite str/bool/int/float
    constants, names, non-dunder attributes (``math`` restricted to
    :data:`MATH_MEMBERS`), tuples, lists, dicts, subscripts, slices, calls
    (the pinned builtins, artifact-defined functions, ``math`` members and
    one-argument ``.pop``/``.append``), the operators ``+ - * / % //``, unary
    ``- + not``, comparisons, ``and``/``or`` and conditional expressions.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ArtifactError(f"a syntax error: {error.msg} at line {error.lineno}") from error
    _Validator(tree).check_module(tree)


def normalized_source(source: str) -> str:
    """Validate ``source``, then return its canonical form (``ast.unparse`` of its tree)."""
    validate_artifact_source(source)
    return ast.unparse(ast.parse(source))


# --- the loader -------------------------------------------------------------------


def _as_forecast(output: object) -> ForecastOutput:
    """Wrap a flat artifact output in a ForecastOutput, each entry through ``float``."""
    return ForecastOutput(tuple(float(value) for value in cast("Iterable[float]", output)))


class _ArtifactUpdater:
    """The adapter around one artifact's namespace; satisfies the ForecastUpdater protocol."""

    def __init__(self, namespace: Mapping[str, object]) -> None:
        #: the sandbox namespace the artifact's functions live in; test 7 pins it
        self.namespace = namespace

    def init(
        self, n_experts: int, horizon: int, config: Mapping[str, ConfigValue], seed: int
    ) -> object:
        init_function = cast(
            "Callable[[int, int, Mapping[str, ConfigValue], int], object]",
            self.namespace["init"],
        )
        return init_function(n_experts, horizon, config, seed)

    def update(
        self, state: object, feedback: tuple[Matured, ...], context: Context
    ) -> tuple[Output, object]:
        update_function = cast(
            "Callable[[object, tuple[Matured, ...], Context], tuple[object, object]]",
            self.namespace["update"],
        )
        output, new_state = update_function(state, feedback, context)
        rows = cast("Sequence[object]", output)
        if isinstance(output, (list, tuple)) and rows and isinstance(rows[0], (list, tuple)):
            # a weights row list (the weights-only track of (f)): unchanged
            return (cast("Weights", rows), new_state)
        return (_as_forecast(output), new_state)


def artifact_updater(source: str) -> Callable[[], ForecastUpdater]:
    """Validate ``source``, then return a factory of adapters that run it as an updater.

    The artifact is compiled and executed once, in a namespace whose globals
    are exactly ``{"__builtins__": dict(ARTIFACT_BUILTINS), "math": math}``:
    no ``__import__``, no ``open``, no ``eval``. No import statement is
    allowed, so none ever needs ``__import__``; ``math`` arrives as a global.
    The functions are stateless — state travels through the state object — so
    the one namespace serves every series, one adapter per replay.

    ``update`` returns whatever the artifact returns as ``(output, state)``
    and dispatches on shape: when the first element of ``output`` is itself a
    sequence it is a weights row list and passes through unchanged (the arena
    validates it, as for any weights-v1 output); otherwise ``output`` is a
    flat per-horizon forecast and is wrapped in a ``ForecastOutput``. Because
    the flat/nested distinction is the first element's type, the two are never
    confusable (``horizon >= 1``).
    """
    validate_artifact_source(source)
    tree = ast.parse(source)
    if tree.body and _is_docstring(tree.body[0]):
        # a no-op statement whose only effect would be a __doc__ global; the
        # sandbox namespace holds exactly the seeded globals and the two defs
        tree.body = tree.body[1:]
    code = compile(tree, "<evolvets updater artifact>", "exec")
    namespace: dict[str, object] = {"__builtins__": dict(ARTIFACT_BUILTINS), "math": math}
    exec(code, namespace)
    return lambda: _ArtifactUpdater(namespace)


# --- diffs, families and constants ------------------------------------------------


def diff_lines(parent: str, child: str) -> list[str]:
    """The unified diff's body lines (the ``---``/``+++`` headers dropped); [] when equal."""
    lines = difflib.unified_diff(parent.splitlines(), child.splitlines(), lineterm="")
    return [line for line in lines if not (line.startswith("---") or line.startswith("+++"))]


class _NumericZeroer(ast.NodeTransformer):
    """Replaces every numeric constant (bools excluded) with ``0.0``, for family comparison."""

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        value = node.value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return ast.copy_location(ast.Constant(0.0), node)
        return node


def artifact_family(parent: str, child: str) -> str:
    """:code:`"params"` when only numeric constants differ, else :code:`"logic"`.

    Both sources must validate first. Every numeric constant node is replaced
    with ``0.0`` in both ASTs; equal transformed dumps mean the edit touched
    nothing but numbers — whatever the sources' texts do — and anything else
    (a renamed local, a new branch) is a logic edit.
    """
    validate_artifact_source(parent)
    validate_artifact_source(child)
    zeroed_parent = _NumericZeroer().visit(ast.parse(parent))
    zeroed_child = _NumericZeroer().visit(ast.parse(child))
    return "params" if ast.dump(zeroed_parent) == ast.dump(zeroed_child) else "logic"


def _numeric_constants(source: str) -> list[float]:
    """The numeric constants of an already-validated source, in document order (bools excluded)."""
    values: list[float] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if (
                isinstance(child, ast.Constant)
                and isinstance(child.value, (int, float))
                and not isinstance(child.value, bool)
            ):
                values.append(child.value)
            visit(child)

    visit(ast.parse(source))
    return values


def const_change_stats(parent: str, child: str) -> tuple[int, float]:
    """(how many numeric constants differ, the largest relative change clipped to [0, 1]).

    Both sources must validate first. Constants are collected in document
    order (bools excluded); with equal counts the pair is the number of
    positions that differ and the maximum of ``abs(new - old) /
    max(abs(old), 1e-12)`` clipped to ``[0, 1]``. When the counts differ, or a
    parent constant is not finite, the pair falls back to ``(child count,
    1.0)``: an edit that adds or removes a constant is not a tuning move.
    Identical sources give ``(0, 0.0)``.
    """
    validate_artifact_source(parent)
    validate_artifact_source(child)
    old = _numeric_constants(parent)
    new = _numeric_constants(child)
    if len(old) != len(new):
        return (len(new), 1.0)
    for value in old:
        if not math.isfinite(value):
            return (len(new), 1.0)
    changed = 0
    for index in range(len(old)):
        if old[index] != new[index]:
            changed = changed + 1
    top = 0.0
    for index in range(len(old)):
        relative = abs(new[index] - old[index]) / max(abs(old[index]), 1e-12)
        if relative > top:
            top = relative
    return (changed, min(max(top, 0.0), 1.0))


# --- the reference artifact -------------------------------------------------------


#: The reference artifact as written for this module, normalised at import.
#: bias_share translated into the artifact grammar, operation for operation:
#: the bias recursion over every matured origin first, then the Fixed-Share
#: step per feedback item and horizon column, then the weighted combination
#: minus the bias, and finally the applied-bias bookkeeping. The state is a
#: dict; ``weights`` and ``bias`` are indexed [h] (per horizon column), with
#: the expert index inside each row.
_REFERENCE_ARTIFACT_TEXT = '''"""Fixed-Share combination minus an online per-horizon bias estimate.

The reference updater artifact: bias_share (updater/forecast.py) translated
into the artifact grammar, Fixed-Share inlined (updater/baselines.py
_share_step). The state is a dict; everything per-series travels through it.
"""


def init(n_experts, horizon, config, seed):
    "Validate the arguments and build the initial state."

    def checked_unit(name, value):
        "Return value as a finite number in [0, 1], else raise ValueError."
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(name + " must be a number")
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(name + " must be a finite number in [0, 1]")
        return value

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an int")
    if isinstance(n_experts, bool) or not isinstance(n_experts, int) or n_experts < 1:
        raise ValueError("n_experts must be an int >= 1")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be an int >= 1")
    if "eta" not in config or "alpha" not in config or "bias_rate" not in config:
        raise ValueError("config keys must be exactly eta, alpha, bias_rate")
    if len(config) != 3:
        raise ValueError("config keys must be exactly eta, alpha, bias_rate")
    eta = config["eta"]
    if isinstance(eta, bool) or not isinstance(eta, (int, float)):
        raise ValueError("eta must be a number")
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be a finite number > 0")
    alpha = checked_unit("alpha", config["alpha"])
    bias_rate = checked_unit("bias_rate", config["bias_rate"])
    weights = []
    for h in range(horizon):
        row = []
        for e in range(n_experts):
            row.append(1.0 / n_experts)
        weights.append(row)
    bias = []
    for h in range(horizon):
        bias.append(0.0)
    return {
        "n_experts": n_experts,
        "horizon": horizon,
        "eta": eta,
        "alpha": alpha,
        "bias_rate": bias_rate,
        "weights": weights,
        "bias": bias,
        "applied": {},
    }


def update(state, feedback, context):
    "Absorb matured bias, take a Fixed-Share step, combine, and forecast."
    n_experts = state["n_experts"]
    horizon = state["horizon"]
    eta = state["eta"]
    alpha = state["alpha"]
    bias_rate = state["bias_rate"]
    weights = state["weights"]
    bias = state["bias"]
    applied = state["applied"]
    for matured in feedback:
        step = matured.step
        if step not in applied:
            raise ValueError("no update call ever applied a bias at step " + str(step))
        applied_row = applied.pop(step)
        for h in range(horizon):
            residual = matured.combined_errors[h] + applied_row[h]
            bias[h] = bias[h] + bias_rate * (residual - bias[h])
    for matured in feedback:
        errors = matured.errors
        for h in range(horizon):
            top = abs(errors[0][h])
            for e in range(1, n_experts):
                scaled = abs(errors[e][h])
                if scaled > top:
                    top = scaled
            losses = []
            if top > 0.0:
                for e in range(n_experts):
                    losses.append(abs(errors[e][h]) / top)
            else:
                for e in range(n_experts):
                    losses.append(0.0)
            row = weights[h]
            support = []
            for e in range(n_experts):
                if row[e] > 0.0:
                    support.append(e)
            low = losses[support[0]]
            for e in support[1:]:
                if losses[e] < low:
                    low = losses[e]
            scores = [0.0] * n_experts
            for e in support:
                scores[e] = row[e] * math.exp(-eta * (losses[e] - low))
            total = 0.0
            for e in range(n_experts):
                total = total + scores[e]
            for e in range(n_experts):
                row[e] = (1.0 - alpha) * (scores[e] / total) + alpha / n_experts
    out = []
    for h in range(horizon):
        base = 0.0
        for e in range(n_experts):
            base = base + weights[h][e] * context.forecasts[e][h]
        out.append(base - bias[h])
    applied[context.step] = tuple(bias)
    return (tuple(out), state)
'''

#: The starting artifact of every Phase 1 lineage, in normalised form; the
#: pinned SHA-256 of these bytes is in tests/improver/test_artifacts.py.
REFERENCE_ARTIFACT_SOURCE: str = normalized_source(_REFERENCE_ARTIFACT_TEXT)
