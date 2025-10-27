from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Sequence

from sqlalchemy import types, TypeDecorator
from typing_extensions import List, Optional, Type

from krrood.entity_query_language.predicate import Symbol
from krrood.ormatic.dao import DataAccessObject, AlternativeMapping, T


# check that custom enums works
class Element(Enum):
    C = "c"
    H = "h"


# Check that Types attributes work


@dataclass
class PositionTypeWrapper(Symbol):
    position_type: Type[Position]


# check that flat classes work


@dataclass
class Position(Symbol):
    x: float
    y: float
    z: float


# check that classes with optional values work


@dataclass
class Orientation(Symbol):
    x: float
    y: float
    z: float
    w: Optional[float]


# check that one to one relationship work


@dataclass
class Pose(Symbol):
    position: Position
    orientation: Orientation


# check that one to many relationship to built in types and non built in types work


@dataclass
class Positions(Symbol):
    positions: List[Position]
    some_strings: List[str]


@dataclass
class PositionsSubclassWithAnotherPosition(Positions):
    positions2: Position


# check that one to many relationships work where the many side is of the same type


@dataclass
class DoublePositionAggregator(Symbol):
    positions1: List[Position]
    positions2: List[Position]


# check that inheritance works


@dataclass
class Position4D(Position):
    w: float


# check that inheriting from an inherited class works


@dataclass
class Position5D(Position4D):
    v: float


# check with tree like classes


@dataclass
class Node(Symbol):
    parent: Optional[Node] = None


@dataclass
class NotMappedParent: ...


# check that enum references work


@dataclass
class Atom(NotMappedParent, Symbol):
    element: Element
    type: int
    charge: float
    timestamp: datetime = field(default_factory=datetime.now)


# check that custom type checks work
class PhysicalObject:
    pass


class Cup(PhysicalObject):
    pass


class Bowl(PhysicalObject):
    pass


# @dataclass
# class MultipleInheritance(Position, Orientation):
#    pass


@dataclass
class OriginalSimulatedObject(Symbol):
    concept: Optional[PhysicalObject]
    placeholder: float = field(default=0)


@dataclass
class ObjectAnnotation(Symbol):
    """
    Class for checking how classes that are explicitly mapped interact with original types.
    """

    object_reference: OriginalSimulatedObject


@dataclass
class KinematicChain(Symbol):
    name: str


@dataclass
class Torso(KinematicChain):
    """
    A Torso is a kinematic chain connecting the base of the robot with a collection of other kinematic chains.
    """

    kinematic_chains: List[KinematicChain] = field(default_factory=list)
    """
    A collection of kinematic chains that are connected to the torso.
    """


@dataclass
class Parent(Symbol):
    name: str


@dataclass
class ChildMapped(Parent):
    attribute1: int


@dataclass
class ChildNotMapped(Parent):
    attribute2: int
    unparseable: Dict[int, int]


@dataclass
class Entity(Symbol):
    name: str
    attribute_that_shouldnt_appear_at_all: float = 0


# Define a derived class


@dataclass
class DerivedEntity(Entity):
    description: str = "Default description"


@dataclass
class EntityAssociation(Symbol):
    """
    Class for checking how classes that are explicitly mapped interact with original types.
    """

    entity: Entity
    a: Sequence[str] = None


# Define an explicit mapping DAO that maps to the base entity class


@dataclass
class CustomEntity(AlternativeMapping[Entity]):
    overwritten_name: str

    @classmethod
    def create_instance(cls, obj: Entity):
        result = cls(overwritten_name=obj.name)
        return result

    def create_from_dao(self) -> T:
        return Entity(name=self.overwritten_name)


class ConceptType(TypeDecorator):
    """
    Type that casts fields that are of type `type` to their class name on serialization and converts the name
    to the class itself through the globals on load.
    """

    impl = types.String(256)

    def process_bind_param(self, value: PhysicalObject, dialect):
        return value.__class__.__module__ + "." + value.__class__.__name__

    def process_result_value(self, value: impl, dialect) -> Optional[Type]:
        if value is None:
            return None

        module_name, class_name = str(value).rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)()


@dataclass
class Reference(Symbol):
    value: int = 0
    backreference: Optional[Backreference] = None


@dataclass
class Backreference(Symbol):
    unmappable: Dict[Any, int]
    reference: Reference = None


@dataclass
class BackreferenceMapping(AlternativeMapping[Backreference]):
    values: List[int]
    reference: Reference

    @classmethod
    def create_instance(cls, obj: T):
        return cls(list(obj.unmappable.values()), obj.reference)

    def create_from_dao(self) -> T:
        return Backreference({v: v for v in self.values}, self.reference)


@dataclass
class AlternativeMappingAggregator(Symbol):
    entities1: List[Entity]
    entities2: List[Entity]


@dataclass
class ItemWithBackreference(Symbol):
    value: int = 0
    container: ContainerGeneration = None


@dataclass
class ContainerGeneration(Symbol):
    items: List[ItemWithBackreference]

    def __post_init__(self):
        for item in self.items:
            item.container = self


@dataclass
class Vector(Symbol):
    x: float


@dataclass
class VectorMapped(AlternativeMapping[Vector]):
    x: float

    @classmethod
    def create_instance(cls, obj: T):
        return VectorMapped(obj.x)

    def create_from_dao(self) -> T:
        return Vector(self.x)


@dataclass
class Rotation(Symbol):
    angle: float


@dataclass
class RotationMapped(AlternativeMapping[Rotation]):

    angle: float

    @classmethod
    def create_instance(cls, obj: T):
        return RotationMapped(obj.angle)

    def create_from_dao(self) -> T:
        pass


@dataclass
class Transformation(Symbol):
    vector: Vector
    rotation: Rotation


@dataclass
class TransformationMapped(AlternativeMapping[Transformation]):
    vector: Vector
    rotation: Rotation

    @classmethod
    def create_instance(cls, obj: T):
        return TransformationMapped(obj.vector, obj.rotation)

    def create_from_dao(self) -> T:
        return Transformation(self.vector, self.rotation)


@dataclass
class Shape(Symbol):
    name: str
    origin: Transformation


@dataclass
class Shapes(Symbol):
    shapes: List[Shape]


@dataclass
class MoreShapes(Symbol):
    shapes: List[Shapes]


@dataclass
class VectorsWithProperty(Symbol):
    _vectors: List[Vector]

    @property
    def vectors(self) -> List[Vector]:
        return self._vectors


@dataclass
class VectorsWithPropertyMapped(AlternativeMapping[VectorsWithProperty]):
    vectors: List[Vector]

    @classmethod
    def create_instance(cls, obj: T):
        return VectorsWithPropertyMapped(obj.vectors)

    def create_from_dao(self) -> T:
        return VectorsWithProperty(self.vectors)


@dataclass
class ParentBase(Symbol):
    name: str
    value: int


@dataclass
class ChildBase(ParentBase):
    pass


@dataclass
class ParentBaseMapping(AlternativeMapping[ParentBase]):
    name: str

    @classmethod
    def create_instance(cls, obj: T):
        if not isinstance(obj, Parent):
            raise TypeError(f"Expected Parent, got {type(obj)}")
        return ParentBaseMapping(obj.name)

    def create_from_dao(self) -> T:
        return ParentBase(self.name, 0)


@dataclass
class ChildBaseMapping(ParentBaseMapping, AlternativeMapping[ChildBase]):

    @classmethod
    def create_instance(cls, obj: T):
        if not isinstance(obj, ChildMapped):
            raise TypeError(f"Expected TestClass2, got {type(obj)}")
        return ChildBaseMapping(obj.name)

    def create_from_dao(self) -> T:
        return ChildBase(self.name, 0)


@dataclass
class PrivateDefaultFactory(Symbol):
    public_value: int = 0
    _private_list: List[int] = field(default_factory=list)


@dataclass
class RelationshipParent(Symbol):
    positions: Position


@dataclass
class RelationshipChild(RelationshipParent):
    """
    This class should produce a problem when reconstructed from the database as relationships must not be declared
    twice.
    """
