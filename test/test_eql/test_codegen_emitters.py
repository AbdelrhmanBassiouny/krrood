from __future__ import annotations

from dataset.semantic_world_like_classes import Body, View, Cabinet, Drawer
from krrood.entity_query_language.entity import symbolic_mode, a, flatten
from krrood.entity_query_language.predicate import HasType
from krrood.entity_query_language.eql_to_python import (
    _emitter_for,
    _ComparatorEmitter,
    _TruthyEmitter,
)


def test_classify_comparator_condition(handles_and_containers_world):
    # Build a tiny query containing a comparator condition
    with symbolic_mode():
        q = a(x := Body(), (x.name == "Body1"))
    # The condition node is attached to the descriptor as its child
    cond = q._child_._child_
    assert isinstance(_emitter_for(cond), _ComparatorEmitter)


def test_classify_predicate_condition(handles_and_containers_world):
    # Build a simple predicate condition (HasType)
    with symbolic_mode():
        q = a(
            x := Cabinet(world=handles_and_containers_world),
            HasType(flatten(x.drawers), Drawer),
        )
    # The AND root may wrap the predicate depending on arity; still, the predicate
    # itself must be classified correctly.
    # Pull the predicate node out of the condition tree
    cond_root = q._child_._child_
    # If the root is the predicate already, classify it; otherwise, descend children to find it
    nodes = []
    if cond_root is not None:
        nodes.append(cond_root)
        for ch in getattr(cond_root, "_children_", []) or []:
            nodes.append(ch)
    # Find the first node that classifies as Predicate
    classifications = [_emitter_for(n) for n in nodes]
    assert any(isinstance(c, _TruthyEmitter) for c in classifications)
