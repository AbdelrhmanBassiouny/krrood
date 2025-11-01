from krrood.entity_query_language.entity import let, an, entity, and_
from krrood.entity_query_language.conclusion import Add
from krrood.entity_query_language.entity import infer
from krrood.entity_query_language.predicate import HasType
from krrood.entity_query_language.rule import refinement, alternative, next_rule
from krrood.entity_query_language.symbolic import symbolic_mode, rule_mode
from ...dataset.semantic_world_like_classes import (
    Container,
    Handle,
    FixedConnection,
    PrismaticConnection,
    Drawer,
    View,
    Door,
    Body,
    RevoluteConnection,
    Wardrobe,
)


def test_generate_drawers(handles_and_containers_world):
    world = handles_and_containers_world
    container = let(type_=Container, domain=world.bodies)
    handle = let(type_=Handle, domain=world.bodies)
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    prismatic_connection = let(type_=PrismaticConnection, domain=world.connections)
    with rule_mode():
        solutions = infer(
            Drawer(handle=handle, container=container),
            and_(
                container == fixed_connection.parent,
                handle == fixed_connection.child,
                container == prismatic_connection.child,
            ),
        ).evaluate()

    all_solutions = list(solutions)

    assert (
        len(all_solutions) == 2
    ), "Should generate components for two possible drawer."
    assert all(isinstance(d, Drawer) for d in all_solutions)
    assert all_solutions[0].handle.name == "Handle3"
    assert all_solutions[0].container.name == "Container3"
    assert all_solutions[1].handle.name == "Handle1"
    assert all_solutions[1].container.name == "Container1"


def test_add_conclusion(handles_and_containers_world):
    world = handles_and_containers_world

    container = let(type_=Container, domain=world.bodies)
    handle = let(type_=Handle, domain=world.bodies)
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    prismatic_connection = let(type_=PrismaticConnection, domain=world.connections)

    with symbolic_mode():
        query = an(
            entity(
                drawers := let(type_=Drawer, domain=None),
                container == fixed_connection.parent,
                handle == fixed_connection.child,
                container == prismatic_connection.child,
            )
        )
    with rule_mode(query):
        Add(drawers, Drawer(handle=handle, container=container))

    solutions = query.evaluate()
    all_solutions = list(solutions)
    assert (
        len(all_solutions) == 2
    ), "Should generate components for two possible drawer."
    assert all(isinstance(d, Drawer) for d in all_solutions)
    assert all_solutions[0].handle.name == "Handle3"
    assert all_solutions[0].container.name == "Container3"
    assert all_solutions[1].handle.name == "Handle1"
    assert all_solutions[1].container.name == "Container1"
    # all_drawers = list(drawers._evaluate__())
    # assert len(all_drawers) == 2, "Should generate components for two possible drawer."


def test_rule_tree_with_a_refinement(doors_and_drawers_world):
    world = doors_and_drawers_world
    body = let(type_=Body, domain=world.bodies)
    handle = let(type_=Handle, domain=world.bodies)
    fixed_connection = let(type_=FixedConnection, domain=world.connections)

    with symbolic_mode():
        query = an(
            entity(
                drawers_and_doors := let(type_=View, domain=None),
                body == fixed_connection.parent,
                handle == fixed_connection.child,
            )
        )

    with rule_mode(query):
        Add(drawers_and_doors, Drawer(handle=handle, container=body))
        with refinement(body.size > 1):
            Add(drawers_and_doors, Door(handle=handle, body=body))

    # query._render_tree_()

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 3, "Should generate 1 drawer and 1 door."
    assert isinstance(all_solutions[0], Door)
    assert all_solutions[0].handle.name == "Handle2"
    assert all_solutions[0].body.name == "Body2"
    assert isinstance(all_solutions[1], Drawer)
    assert all_solutions[1].handle.name == "Handle4"
    assert all_solutions[1].container.name == "Body4"
    assert isinstance(all_solutions[2], Drawer)
    assert all_solutions[2].handle.name == "Handle1"
    assert all_solutions[2].container.name == "Container1"


def test_rule_tree_with_multiple_refinements(doors_and_drawers_world):
    world = doors_and_drawers_world
    body = let(type_=Body, domain=world.bodies)
    container = let(type_=Container, domain=world.bodies)
    handle = let(type_=Handle, domain=world.bodies)
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    revolute_connection = let(type_=RevoluteConnection, domain=world.connections)

    with symbolic_mode():
        query = an(
            entity(
                views := let(type_=View, domain=None),
                body == fixed_connection.parent,
                handle == fixed_connection.child,
            )
        )

    with rule_mode(query):
        Add(views, Drawer(handle=handle, container=body))
        with refinement(body.size > 1):
            Add(views, Door(handle=handle, body=body))
            with alternative(
                body == revolute_connection.child,
                container == revolute_connection.parent,
            ):
                Add(views, Wardrobe(handle=handle, body=body, container=container))

    # query._render_tree_()

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 3, "Should generate 1 drawer, 1 door and 1 wardrobe."
    assert isinstance(all_solutions[0], Door)
    assert all_solutions[0].handle.name == "Handle2"
    assert all_solutions[0].body.name == "Body2"
    assert isinstance(all_solutions[1], Wardrobe)
    assert all_solutions[1].handle.name == "Handle4"
    assert all_solutions[1].container.name == "Container2"
    assert all_solutions[1].body.name == "Body4"
    assert isinstance(all_solutions[2], Drawer)
    assert all_solutions[2].handle.name == "Handle1"
    assert all_solutions[2].container.name == "Container1"


def test_rule_tree_with_an_alternative(doors_and_drawers_world):
    world = doors_and_drawers_world
    body = let(type_=Body, domain=world.bodies)
    handle = let(type_=Handle, domain=world.bodies)
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    revolute_connection = let(type_=RevoluteConnection, domain=world.connections)

    with symbolic_mode():
        query = an(
            entity(
                views := let(type_=View, domain=None),
                body == fixed_connection.parent,
                handle == fixed_connection.child,
            )
        )

    with rule_mode(query):
        Add(views, Drawer(handle=handle, container=body))
        with alternative(
            body == revolute_connection.parent, handle == revolute_connection.child
        ):
            Add(views, Door(handle=handle, body=body))

    # query._render_tree_()

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 4, "Should generate 3 drawers, 1 door"
    assert isinstance(all_solutions[0], Drawer)
    assert all_solutions[0].handle.name == "Handle2"
    assert all_solutions[0].container.name == "Body2"
    assert isinstance(all_solutions[1], Door)
    assert all_solutions[1].handle.name == "Handle3"
    assert all_solutions[1].body.name == "Body3"
    assert isinstance(all_solutions[2], Drawer)
    assert all_solutions[2].handle.name == "Handle4"
    assert all_solutions[2].container.name == "Body4"
    assert isinstance(all_solutions[3], Drawer)
    assert all_solutions[3].handle.name == "Handle1"
    assert all_solutions[3].container.name == "Container1"


def test_rule_tree_with_multiple_alternatives(doors_and_drawers_world):
    world = doors_and_drawers_world
    body = let(type_=Body, domain=world.bodies)
    container = let(type_=Container, domain=world.bodies)
    handle = let(type_=Handle, domain=world.bodies)
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    prismatic_connection = let(type_=PrismaticConnection, domain=world.connections)
    revolute_connection = let(type_=RevoluteConnection, domain=world.connections)

    with symbolic_mode():
        query = infer(
            views := let(type_=View, domain=None),
            body == fixed_connection.parent,
            handle == fixed_connection.child,
            body == prismatic_connection.child,
        )

    with rule_mode(query):
        Add(views, Drawer(handle=handle, container=body))
        with alternative(
            revolute_connection.parent == body, revolute_connection.child == handle
        ):
            Add(views, Door(handle=handle, body=body))
        with alternative(
            fixed_connection.parent == body,
            fixed_connection.child == handle,
            body == revolute_connection.child,
            container == revolute_connection.parent,
        ):
            Add(views, Wardrobe(handle=handle, body=body, container=container))

    # query._render_tree_()

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 3, "Should generate 1 drawer, 1 door and 1 wardrobe."
    expected_solution_set = {
        (Door, "Handle3", "Body3"),
        (Drawer, "Handle1", "Container1"),
        (Wardrobe, "Handle4", "Body4", "Container2"),
    }
    solution_set = set()
    for s in all_solutions:
        if isinstance(s, Door):
            solution_set.add((Door, s.handle.name, s.body.name))
        elif isinstance(s, Drawer):
            solution_set.add((Drawer, s.handle.name, s.container.name))
        elif isinstance(s, Wardrobe):
            solution_set.add((Wardrobe, s.handle.name, s.body.name, s.container.name))
    assert expected_solution_set == solution_set


def test_rule_tree_with_multiple_alternatives_optimized(doors_and_drawers_world):
    world = doors_and_drawers_world
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    prismatic_connection = let(type_=PrismaticConnection, domain=world.connections)
    revolute_connection = let(type_=RevoluteConnection, domain=world.connections)

    with symbolic_mode():
        query = infer(
            views := let(type_=View, domain=None),
            HasType(fixed_connection.child, Handle),
            fixed_connection.parent == prismatic_connection.child,
        )

    with rule_mode(query):
        Add(
            views,
            Drawer(handle=fixed_connection.child, container=fixed_connection.parent),
        )
        with alternative(HasType(revolute_connection.child, Handle)):
            Add(
                views,
                Door(handle=revolute_connection.child, body=revolute_connection.parent),
            )
        with alternative(
            fixed_connection,
            fixed_connection.parent == revolute_connection.child,
            HasType(revolute_connection.parent, Container),
        ):
            Add(
                views,
                Wardrobe(
                    handle=fixed_connection.child,
                    body=fixed_connection.parent,
                    container=revolute_connection.parent,
                ),
            )

    # query._render_tree_()

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 3, "Should generate 1 drawer, 1 door and 1 wardrobe."
    expected_solution_set = {
        (Door, "Handle3", "Body3"),
        (Drawer, "Handle1", "Container1"),
        (Wardrobe, "Handle4", "Body4", "Container2"),
    }
    solution_set = set()
    for s in all_solutions:
        if isinstance(s, Door):
            solution_set.add((Door, s.handle.name, s.body.name))
        elif isinstance(s, Drawer):
            solution_set.add((Drawer, s.handle.name, s.container.name))
        elif isinstance(s, Wardrobe):
            solution_set.add((Wardrobe, s.handle.name, s.body.name, s.container.name))
    assert expected_solution_set == solution_set


def test_rule_tree_with_multiple_alternatives_better_rule_tree(doors_and_drawers_world):
    world = doors_and_drawers_world
    body = let(type_=Body, domain=world.bodies)
    container = let(type_=Container, domain=world.bodies)
    handle = let(type_=Handle, domain=world.bodies)
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    prismatic_connection = let(type_=PrismaticConnection, domain=world.connections)
    revolute_connection = let(type_=RevoluteConnection, domain=world.connections)

    with symbolic_mode():
        query = infer(
            views := let(type_=View, domain=None),
            body == fixed_connection.parent,
            handle == fixed_connection.child,
        )

    with rule_mode(query):
        with refinement(prismatic_connection.child == body):
            Add(views, Drawer(handle=handle, container=body))
            with alternative(
                body == revolute_connection.child,
                container == revolute_connection.parent,
            ):
                Add(views, Wardrobe(handle=handle, body=body, container=container))
        with alternative(
            revolute_connection.parent == body, revolute_connection.child == handle
        ):
            Add(views, Door(handle=handle, body=body))

    # query._render_tree_()

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 3, "Should generate 1 drawer, 1 door and 1 wardrobe."
    expected_solution_set = {
        (Door, "Handle3", "Body3"),
        (Drawer, "Handle1", "Container1"),
        (Wardrobe, "Handle4", "Body4", "Container2"),
    }
    solution_set = set()
    for s in all_solutions:
        if isinstance(s, Door):
            solution_set.add((Door, s.handle.name, s.body.name))
        elif isinstance(s, Drawer):
            solution_set.add((Drawer, s.handle.name, s.container.name))
        elif isinstance(s, Wardrobe):
            solution_set.add((Wardrobe, s.handle.name, s.body.name, s.container.name))
    assert expected_solution_set == solution_set


def test_rule_tree_with_multiple_alternatives_better_rule_tree_optimized(
    doors_and_drawers_world,
):
    world = doors_and_drawers_world
    fixed_connection = let(type_=FixedConnection, domain=world.connections)
    prismatic_connection = let(type_=PrismaticConnection, domain=world.connections)
    revolute_connection = let(type_=RevoluteConnection, domain=world.connections)

    with symbolic_mode():
        query = infer(
            views := let(type_=View, domain=None),
            HasType(fixed_connection.child, Handle),
        )

    with rule_mode(query):
        with refinement(prismatic_connection.child == fixed_connection.parent):
            Add(
                views,
                Drawer(
                    handle=fixed_connection.child, container=fixed_connection.parent
                ),
            )
            with alternative(
                fixed_connection.parent == revolute_connection.child,
                HasType(revolute_connection.parent, Container),
            ):
                Add(
                    views,
                    Wardrobe(
                        handle=fixed_connection.child,
                        body=fixed_connection.parent,
                        container=revolute_connection.parent,
                    ),
                )
        with next_rule(HasType(revolute_connection.child, Handle)):
            Add(
                views,
                Door(handle=revolute_connection.child, body=revolute_connection.parent),
            )

    # query._render_tree_()

    all_solutions = list(query.evaluate())
    assert len(all_solutions) == 3, "Should generate 1 drawer, 1 door and 1 wardrobe."
    expected_solution_set = {
        (Door, "Handle3", "Body3"),
        (Drawer, "Handle1", "Container1"),
        (Wardrobe, "Handle4", "Body4", "Container2"),
    }
    solution_set = set()
    for s in all_solutions:
        if isinstance(s, Door):
            solution_set.add((Door, s.handle.name, s.body.name))
        elif isinstance(s, Drawer):
            solution_set.add((Drawer, s.handle.name, s.container.name))
        elif isinstance(s, Wardrobe):
            solution_set.add((Wardrobe, s.handle.name, s.body.name, s.container.name))
    assert expected_solution_set == solution_set
