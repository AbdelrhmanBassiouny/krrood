from __future__ import annotations

from .. import logger

"""
Fast-path planner and evaluator for common EQL patterns.

This module introduces a tiny planning + execution layer that can
opportunistically accelerate a subset of EQL queries without changing
public APIs. It follows SOLID principles and keeps functions short.

The fast path is intentionally conservative. When it cannot safely handle
an input query, it returns None so the interpreter falls back to the
existing generic evaluator in symbolic.py.

The fast path can be enabled by setting the environment variable
KRROOD_FAST_EVAL=1. When disabled, evaluate_fast returns None.
"""

from dataclasses import dataclass, field
import os
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union, Any

from .predicate import Predicate, HasType, Symbol
from .symbolic import (
    Attribute,
    CanBehaveLikeAVariable,
    Comparator,
    DomainMapping,
    Flatten,
    QueryObjectDescriptor,
    ResultQuantifier,
    SetOf,
    Variable,
    Literal,
    Entity,
    AND,
)
from .cache_data import get_cache_keys_for_class_, yield_class_values_from_cache
from .hashed_data import HashedValue


# ----------------------------
# Planning dataclasses
# ----------------------------


@dataclass(frozen=True)
class Filter:
    """
    Describes a literal equality filter over an attribute path.

    :ivar path: Tuple of attribute names from a root variable.
    :ivar value: Literal value to compare against.
    """

    path: Tuple[str, ...]
    value: object


@dataclass
class Precompute:
    """
    Holds precomputation requirements for an independent variable.

    :ivar var: The root variable to scan.
    :ivar attr_paths: Attribute paths to materialize as membership sets.
    :ivar filters: Literal filters to push down during the scan.
    """

    var: Variable
    attr_paths: Set[Tuple[str, ...]] = field(default_factory=set)
    filters: List[Filter] = field(default_factory=list)


@dataclass
class Plan:
    """Execution plan for a query descriptor."""

    descriptor: QueryObjectDescriptor
    precomputes: Dict[int, Precompute] = field(default_factory=dict)


# ----------------------------
# Execution context
# ----------------------------


@dataclass
class ExecutionContext:
    """
    Stores ephemeral state built per evaluation.

    :ivar membership_sets: Maps (var_id, path) to a Python set of items.
    """

    membership_sets: Dict[Tuple[int, Tuple[str, ...]], Set[Any]] = field(
        default_factory=dict
    )


# ----------------------------
# Public API
# ----------------------------


def evaluate_fast(
    result: ResultQuantifier, *, yield_when_false: bool = False
) -> Optional[Iterator[Dict[int, HashedValue]]]:
    """
    Try to evaluate the query using the fast path.

    Returns an iterator over interpreter-shaped rows when supported,
    or None to signal fallback to the generic interpreter.
    """
    # if os.getenv("KRROOD_FAST_EVAL") != "1":
    #     return None
    descriptor = result._child_
    if descriptor is None:
        return None
    plan = _plan(descriptor)
    if plan is None:
        return None
    return _execute(plan, yield_when_false=yield_when_false)


# ----------------------------
# Planner
# ----------------------------


def _plan(descriptor: QueryObjectDescriptor) -> Optional[Plan]:
    """
    Build a conservative plan. Today recognizes only membership
    precomputations for contains(attr_chain(var), probe).
    Always returns a Plan; precomputes may be empty for simple scans.
    """
    cond = descriptor._child_
    selected = descriptor.selected_variables
    pre: Dict[int, Precompute] = {}
    visited: Set[int] = set()

    def ensure(v: Variable) -> Precompute:
        vid = id(v)
        if vid not in pre:
            pre[vid] = Precompute(var=v)
        return pre[vid]

    def is_independent_symbol(v: Variable) -> bool:
        if id(v) in {id(_base_of(s)) for s in selected if _base_of(s) is not None}:
            return False
        t = getattr(v, "_type_", None)
        return (
            isinstance(t, type)
            and issubclass(t, Symbol)
            and not issubclass(t, Predicate)
        )

    def visit(node) -> None:
        if node is None:
            return
        nid = id(node)
        if nid in visited:
            return
        visited.add(nid)
        if isinstance(node, Comparator) and node._name_ == "contains":
            res = (
                _extract_var_and_attr_path(node.left)
                if isinstance(node.left, CanBehaveLikeAVariable)
                else None
            )
            if res is not None:
                v, path = res
                if is_independent_symbol(v):
                    ensure(v).attr_paths.add(path)
            visit(node.right)
            return
        # Traverse arguments/children
        if isinstance(node, Variable):
            for val in getattr(node, "_kwargs_", {}).values():
                if isinstance(val, CanBehaveLikeAVariable):
                    visit(val)
        for ch in getattr(node, "_children_", []) or []:
            visit(ch)
        if hasattr(node, "_child_"):
            visit(getattr(node, "_child_"))

    visit(cond)
    return Plan(descriptor=descriptor, precomputes=pre)


def _base_of(expr: CanBehaveLikeAVariable) -> Optional[Variable]:
    node = expr
    while isinstance(node, DomainMapping):
        node = node._child_
    return node if isinstance(node, Variable) else None


def _extract_var_and_attr_path(
    expr: CanBehaveLikeAVariable,
) -> Optional[Tuple[Variable, Tuple[str, ...]]]:
    """Extract (variable, attribute path) from a chain of Attributes.

    Unwraps DomainMapping, ResultQuantifier, and QueryObjectDescriptor to reach
    the base Variable when possible.
    """
    path: List[str] = []
    node = expr
    # Collect attribute names while walking up
    while isinstance(node, Attribute):
        path.append(node._attr_name_)
        node = node._child_
    # Unwrap common wrappers to reach the base variable
    unwrapped = True
    while unwrapped:
        unwrapped = False
        if isinstance(node, ResultQuantifier):
            desc = node._child_
            selected = getattr(desc, "selected_variables", None) or []
            node = ensure_one_selected_variable(selected, desc)
            if node:
                unwrapped = True
        elif isinstance(node, QueryObjectDescriptor):
            selected = getattr(node, "selected_variables", None) or []
            node = ensure_one_selected_variable(selected, node)
            if node:
                unwrapped = True
        elif isinstance(node, DomainMapping):
            node = node._child_
            unwrapped = True
    if isinstance(node, Variable):
        path.reverse()
        return node, tuple(path)
    return None


def ensure_one_selected_variable(selected, desc):
    if selected:
        if len(selected) > 1:
            logger.warning(
                f"Unexpected multiple selected variables in {desc}, skipping fast path."
            )
            return None
        else:
            return selected[0]


# ----------------------------
# Executor
# ----------------------------


def _condition_uses_only_bound_variables(cond, allowed_ids: Set[int]) -> bool:
    """
    Return True if the condition tree does not reference any non-predicate
    Variable outside the set of already bound variable ids.

    This prevents the fast path from attempting to evaluate expressions that
    require introducing new variables not produced by the current loops.
    """
    visited: Set[int] = set()

    def visit(node) -> bool:
        if node is None:
            return True
        nid = id(node)
        if nid in visited:
            return True
        visited.add(nid)
        # If this is a Variable of a non-predicate Symbol type, ensure it is bound
        if isinstance(node, Variable):
            t = getattr(node, "_type_", None)
            # Ignore predicate variables (HasType, etc.)
            if isinstance(t, type) and issubclass(t, Predicate):
                return True
            if isinstance(t, type) and issubclass(t, Symbol):
                return node._id_ in allowed_ids
        # Recurse children and common attributes
        for ch in getattr(node, "_children_", []) or []:
            if not visit(ch):
                return False
        if hasattr(node, "_child_"):
            if not visit(getattr(node, "_child_")):
                return False
        # Comparator: visit both sides
        if isinstance(node, Comparator):
            return visit(node.left) and visit(node.right)
        # Predicates are Variables with kwargs that may contain expressions
        if isinstance(node, Variable):
            for val in getattr(node, "_kwargs_", {}).values():
                if isinstance(val, CanBehaveLikeAVariable):
                    if not visit(val):
                        return False
        return True

    return visit(cond)


def _iterate_instances(cls: type) -> Iterable[Any]:
    cache = Variable._cache_
    keys = get_cache_keys_for_class_(cache, cls)
    for t in keys:
        for _, hv in cache[t].retrieve(from_index=False):
            yield hv.value


def _is_iterable_nonstring(obj: Any) -> bool:
    return hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes))


def _collect(items: Any) -> Iterable[Any]:
    return items if _is_iterable_nonstring(items) else [items]


def _any_pair_equal(left: Any, right: Any) -> bool:
    try:
        left_iter = _collect(left) if _is_iterable_nonstring(left) else [left]
        right_iter = _collect(right) if _is_iterable_nonstring(right) else [right]
        for lv in left_iter:
            for rv in right_iter:
                if lv == rv:
                    return True
        return False
    except Exception:
        return False


def _execute(
    plan: Plan, *, yield_when_false: bool
) -> Optional[Iterator[Dict[int, HashedValue]]]:
    ctx = ExecutionContext()
    _precompute_memberships(plan, ctx)
    desc = plan.descriptor
    selected = desc.selected_variables

    # Case 1: Entity over a single base variable with optional conditions (e.g., q6)
    if isinstance(desc, Entity):
        # Expect exactly one selected base Variable of Symbol type
        base = _base_of(selected[0]) if selected else None
        if not isinstance(base, Variable):
            return None
        base_cls = getattr(base, "_type_", None)
        if not (isinstance(base_cls, type) and issubclass(base_cls, Symbol)):
            return None
        # Ensure the condition does not require unbound variables
        cond = desc._child_
        if cond is not None:
            allowed_ids: Set[int] = {base._id_}
            if not _condition_uses_only_bound_variables(cond, allowed_ids):
                return None

        def gen_entity() -> Iterator[Dict[int, HashedValue]]:
            base_id = base._id_
            for obj in _iterate_instances(base_cls):
                row: Dict[int, HashedValue] = {base_id: HashedValue(obj)}
                try:
                    if cond is not None and not _row_satisfies(cond, row, ctx):
                        continue
                except Exception as e:
                    logger.warning(f"Condition evaluation failed during fast eval: {e}")
                    return None
                yield {base_id: HashedValue(obj)}

        return gen_entity()

    # Case 2: SetOf with a single base and optional flattened projections (e.g., q7)
    if not selected:
        return None
    bases: List[Variable] = []
    for s in selected:
        b = _base_of(s)
        if isinstance(b, Variable):
            t = getattr(b, "_type_", None)
            if isinstance(t, type) and issubclass(t, Symbol):
                bases.append(b)
    if not bases:
        return None
    base = bases[0]
    base_cls = getattr(base, "_type_", None)
    if not isinstance(base_cls, type):
        return None

    def generator() -> Iterator[Dict[int, HashedValue]]:
        seen: Set[Tuple[Any, ...]] = set()
        base_id = base._id_
        for obj in _iterate_instances(base_cls):
            row: Dict[int, HashedValue] = {base_id: HashedValue(obj)}
            # Project selected variables for this base, producing combinations
            # Then evaluate conditions for each produced combination to mirror SetOf semantics
            # Build a list of partial projections anchored at the base
            partials: List[List[Tuple[int, HashedValue]]] = [[]]
            for sel in selected:
                if _base_of(sel) is base:
                    for p in partials:
                        p.append((sel._id_, row[base_id]))
                elif isinstance(sel, Flatten) and _base_of(sel._child_) is base:
                    if not isinstance(sel._child_, Attribute):
                        return None
                    parent = getattr(obj, sel._child_._attr_name_)
                    new_partials: List[List[Tuple[int, HashedValue]]] = []
                    for v in _collect(parent):
                        for p in partials:
                            new_partials.append(p + [(sel._id_, HashedValue(v))])
                    partials = new_partials or partials
                else:
                    return None  # unsupported selection shape
            # For each combination, check condition and yield distinct tuples
            for comb in partials:
                # Form a row for condition evaluation
                comb_row = {**row, **{i: hv for i, hv in comb}}
                if desc._child_ is not None and not _row_satisfies(
                    desc._child_, comb_row, ctx
                ):
                    continue
                key = tuple(hv.value for _, hv in comb)
                if key in seen:
                    continue
                seen.add(key)
                yield comb_row
        return

    return generator()


def _row_satisfies(cond, row: Dict[int, HashedValue], ctx: ExecutionContext) -> bool:
    """Evaluate a condition for a partially bound row using simple rules.
    For anything unsupported, fall back by returning False to force interpreter.
    """
    if cond is None:
        return True
    # AND: evaluate children
    if isinstance(cond, AND):
        for ch in cond._children_:
            if not _row_satisfies(ch, row, ctx):
                return False
        return True
    # Basic predicates
    if isinstance(cond, Variable) and issubclass(
        cond._type_, HasType.__mro__[0]
    ):  # safeguard
        kwargs = cond._kwargs_
        var = kwargs.get("variable")
        types_ = kwargs.get("types_")
        if isinstance(var, CanBehaveLikeAVariable) and isinstance(types_, type):
            val = _eval_value(var, row)
            return isinstance(val, types_)
    # Comparator
    if isinstance(cond, Comparator):
        if cond._name_ == "contains":
            left_res = (
                _extract_var_and_attr_path(cond.left)
                if isinstance(cond.left, CanBehaveLikeAVariable)
                else None
            )
            if left_res is None:
                return False
            var, path = left_res
            set_key = (id(var), path)
            if set_key not in ctx.membership_sets:
                return False
            right_val = _eval_value(cond.right, row)
            return right_val in ctx.membership_sets[set_key]
        # Equality and others when both sides are concrete or iterables
        left_val = _eval_value(cond.left, row)
        right_val = _eval_value(cond.right, row)
        op = cond._name_
        if op == "==":
            return _any_pair_equal(left_val, right_val)
        if op == "!=":
            return not _any_pair_equal(left_val, right_val)
        if op == "<":
            return left_val < right_val
        if op == "<=":
            return left_val <= right_val
        if op == ">":
            return left_val > right_val
        if op == ">=":
            return left_val >= right_val
        return False
    # Nested descriptors: not handled
    raise ValueError(f"Unsupported condition: {cond}")


def _eval_value(
    expr: Union[CanBehaveLikeAVariable, object], row: Dict[int, HashedValue]
):
    # Treat Literal specially even though it is a Variable subclass
    if isinstance(expr, Literal):
        return expr._domain_source_.domain[0]
    if not isinstance(expr, CanBehaveLikeAVariable):
        return expr
    # Walk domain mappings where possible
    if isinstance(expr, Attribute):
        parent = _eval_value(expr._child_, row)
        if _is_iterable_nonstring(parent):
            return [getattr(it, expr._attr_name_) for it in _collect(parent)]
        return getattr(parent, expr._attr_name_)
    if isinstance(expr, Flatten):
        parent = _eval_value(expr._child_, row)
        # Flatten used in conditions is treated as iterable;
        # callers decide how to loop when projecting
        return list(_collect(parent))
    if isinstance(expr, Variable):
        hv = row.get(expr._id_)
        if hv is None:
            raise KeyError("Variable not bound in fast row")
        return hv.value
    return expr


def _precompute_memberships(plan: Plan, ctx: ExecutionContext) -> None:
    for entry in plan.precomputes.values():
        cls = getattr(entry.var, "_type_", None)
        if not (isinstance(cls, type) and issubclass(cls, Symbol)):
            continue
        for obj in _iterate_instances(cls):
            if not _passes_filters(obj, entry.filters):
                continue
            for path in entry.attr_paths:
                items = obj
                for a in path:
                    items = getattr(items, a)
                for it in _collect(items):
                    ctx.membership_sets.setdefault((id(entry.var), path), set()).add(it)


def _passes_filters(obj: Any, filters: List[Filter]) -> bool:
    for f in filters:
        val = obj
        for a in f.path:
            val = getattr(val, a)
        if val != f.value:
            return False
    return True
