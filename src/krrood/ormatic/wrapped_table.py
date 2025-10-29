from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import cached_property, lru_cache

from typing_extensions import List, Dict, TYPE_CHECKING, Optional, Set

from .dao import AlternativeMapping
from .utils import InheritanceStrategy, module_and_class_name
from ..class_diagrams.class_diagram import (
    WrappedClass,
)
from ..class_diagrams.wrapped_field import WrappedField

if TYPE_CHECKING:
    from .ormatic import ORMatic


logger = logging.getLogger(__name__)


@dataclass
class ColumnConstructor:
    """
    Represents a column constructor that can be used to create a column in SQLAlchemy.
    """

    name: str
    """
    The name of the column.
    """

    type: str
    """
    The type of the column.
    Needs to be like "Mapped[<type>]".
    """

    constructor: Optional[str] = None
    """
    The constructor call for sqlalchemy of the column.
    """

    def __str__(self) -> str:
        if self.constructor:
            return f"{self.name}: {self.type} = {self.constructor}"
        else:
            return f"{self.name}: {self.type}"


@dataclass
class WrappedTable:
    """
    A class that wraps a dataclass and contains all the information needed to create a SQLAlchemy table from it.
    """

    wrapped_clazz: WrappedClass
    """
    The wrapped class that this table wraps.
    """

    ormatic: ORMatic
    """
    Reference to the ORMatic instance that created this WrappedTable.
    """

    builtin_columns: List[ColumnConstructor] = field(default_factory=list, init=False)
    """
    List of columns that can be directly mapped using builtin types
    """

    custom_columns: List[ColumnConstructor] = field(default_factory=list, init=False)
    """
    List for custom columns that need to by fully qualified as triple of (name, type, constructor)
    """

    foreign_keys: List[ColumnConstructor] = field(default_factory=list, init=False)
    """
    List of columns that represent foreign keys as triple of (name, type, constructor)
    """

    relationships: List[ColumnConstructor] = field(default_factory=list, init=False)
    """
    List of relationships that should be added to the table.
    """

    mapper_args: Dict[str, str] = field(default_factory=dict, init=False)

    primary_key_name: str = "database_id"
    """
    The name of the primary key column.
    """

    polymorphic_on_name: str = "polymorphic_type"
    """
    The name of the column that will be used to identify polymorphic identities if any present.
    """

    skip_fields: List[WrappedField] = field(default_factory=list)
    """
    A list of fields that should be skipped when processing the dataclass.
    """

    @property
    def primary_key(self):
        if self.parent_table is not None:
            column_type = f"ForeignKey({self.parent_table.full_primary_key_name})"
        else:
            column_type = "Integer"

        return ColumnConstructor(
            self.primary_key_name,
            f"Mapped[{module_and_class_name(int)}]",
            f"mapped_column({column_type}, primary_key=True, use_existing_column=True)",
        )

    @property
    def child_tables(self) -> List[WrappedTable]:
        return [
            self.ormatic.class_dependency_graph._dependency_graph[index]
            for index in self.ormatic.inheritance_graph.successors(
                self.wrapped_clazz.index
            )
        ]

    def create_mapper_args(self):

        # this is the root of an inheritance structure
        if self.parent_table is None and len(self.child_tables) > 0:
            self.custom_columns.append(
                (
                    ColumnConstructor(
                        self.polymorphic_on_name,
                        "Mapped[str]",
                        "mapped_column(String(255), nullable=False, use_existing_column=True)",
                    )
                )
            )
            self.mapper_args.update(
                {
                    "'polymorphic_on'": f"'{self.polymorphic_on_name}'",
                    "'polymorphic_identity'": f"'{self.tablename}'",
                }
            )

        # this inherits from something
        if self.parent_table is not None:
            self.mapper_args.update(
                {
                    "'polymorphic_identity'": f"'{self.tablename}'",
                }
            )
            # only needed for joined-table inheritance
            if self.ormatic.inheritance_strategy == InheritanceStrategy.JOINED:
                self.mapper_args.update(
                    {
                        "'inherit_condition'": f"{self.primary_key_name} == {self.parent_table.full_primary_key_name}"
                    }
                )

    @cached_property
    def full_primary_key_name(self):
        return f"{self.tablename}.{self.primary_key_name}"

    @cached_property
    def tablename(self):
        result = self.wrapped_clazz.clazz.__name__
        result += "DAO"
        return result

    @cached_property
    def parent_table(self) -> Optional[WrappedTable]:
        parents = self.ormatic.inheritance_graph.predecessors(self.wrapped_clazz.index)
        if len(parents) == 0:
            return None

        return self.ormatic.wrapped_tables[
            self.ormatic.class_dependency_graph._dependency_graph[parents[0]]
        ]

    @property
    def is_alternatively_mapped(self):
        return issubclass(self.wrapped_clazz.clazz, AlternativeMapping)

    @cached_property
    def fields(self) -> List[WrappedField]:
        """
        :return: The list of fields specified only in this associated dataclass that should be mapped.
        """

        # Collect all inherited field names up the chain
        inherited_names: set[str] = set()
        p = self.parent_table
        while p is not None:
            # Use the original dataclass fields of each ancestor
            inherited_names.update(f.field.name for f in p.wrapped_clazz.fields)
            p = p.parent_table

        # Keep only fields that are not inherited by name
        result = [
            f for f in self.wrapped_clazz.fields if f.field.name not in inherited_names
        ]

        # If the parent table is alternatively mapped, drop fields that do not exist
        # in the original parent class (compare by name as well)
        if self.parent_table is not None and self.parent_table.is_alternatively_mapped:
            og_parent_class = self.parent_table.wrapped_clazz.clazz.original_class()
            wrapped_og_parent_class = (
                self.ormatic.class_dependency_graph.get_wrapped_class(og_parent_class)
            )

            og_parent_field_names = {
                f.field.name for f in wrapped_og_parent_class.fields
            }
            parent_dao_field_names = {f.field.name for f in self.parent_table.fields}

            # Fields present in original parent class but removed by the DAO mapping
            removed_by_dao = og_parent_field_names - parent_dao_field_names
            result = [r for r in result if r.field.name not in removed_by_dao]

        return result

    @lru_cache(maxsize=None)
    def parse_fields(self):

        for f in self.fields:

            logger.info("=" * 80)
            logger.info(
                f"Processing Field {self.wrapped_clazz.clazz.__name__}.{f.field.name}: {f.field.type}."
            )

            # skip private fields
            if f.field.name.startswith("_"):
                logger.info(f"Skipping since the field starts with _.")
                continue

            self.parse_field(f)

        self.create_mapper_args()

    def parse_field(self, wrapped_field: WrappedField):
        """
        Parses a given `WrappedField` and determines its type or relationship to create the
        appropriate column or define relationships in an ORM context.
        The method processes several
        types of fields, such as type types, built-in types, enumerations, one-to-one relationships,
        custom types, JSON containers, and one-to-many relationships.

        This creates the right information in the right place in the table definition to be read later by the jinja
        template.

        :param wrapped_field: An instance of `WrappedField` that contains metadata about the field
            such as its data type, whether it represents a built-in or user-defined type, or if it has
            specific ORM container properties.
        """
        if wrapped_field.is_type_type:
            logger.info(f"Parsing as type.")
            self.create_type_type_column(wrapped_field)

        elif (
            wrapped_field.is_builtin_type or wrapped_field.is_enum
        ) and not wrapped_field.is_container:
            logger.info(f"Parsing as builtin type.")
            self.create_builtin_column(wrapped_field)

        # handle one to one relationships
        elif (
            wrapped_field.is_one_to_one_relationship
            and wrapped_field.type_endpoint in self.ormatic.mapped_classes
        ):
            logger.info(f"Parsing as one to one relationship.")
            self.create_one_to_one_relationship(wrapped_field)

        # handle custom types
        elif (
            wrapped_field.is_one_to_one_relationship
            and wrapped_field.type_endpoint in self.ormatic.type_mappings
        ):
            logger.info(
                f"Parsing as custom type {self.ormatic.type_mappings[wrapped_field.type_endpoint]}."
            )
            self.create_custom_type(wrapped_field)

        # handle JSON containers
        elif wrapped_field.is_collection_of_builtins:
            logger.info(f"Parsing as JSON.")
            self.create_container_of_builtins(wrapped_field)

        # handle one to many relationships
        elif wrapped_field.is_one_to_many_relationship:
            logger.info(f"Parsing as one to many relationship.")
            self.create_one_to_many_relationship(wrapped_field)
        else:
            logger.info("Skipping due to not handled type.")

    def create_builtin_column(self, wrapped_field: WrappedField):
        """
        Creates a built-in column mapping for the given wrapped field. Depending on the
        properties of the `wrapped_field`, this function determines whether it's an enum,
        a built-in type, or requires additional imports. It then constructs appropriate
        column definitions and adds them to the respective list of database mappings.

        :param wrapped_field: The WrappedField instance representing the field
            to create a built-in column for.
        """

        self.ormatic.imported_modules.add(wrapped_field.type_endpoint.__module__)
        inner_type = module_and_class_name(wrapped_field.type_endpoint)
        type_annotation = (
            f"{module_and_class_name(Optional)}[{inner_type}]"
            if wrapped_field.is_optional
            else inner_type
        )

        constructor = "mapped_column(use_existing_column=True)"
        self.builtin_columns.append(
            ColumnConstructor(
                name=wrapped_field.field.name,
                type=f"Mapped[{type_annotation}]",
                constructor=constructor,
            )
        )

    def create_type_type_column(self, wrapped_field: WrappedField):
        """
        Create a column for a field of type `Type`.
        :param wrapped_field: The field to extract type information from.
        :return:
        """
        column_name = wrapped_field.field.name
        column_type = (
            f"Mapped[TypeType]"
            if not wrapped_field.is_optional
            else f"Mapped[{module_and_class_name(Optional)}[TypeType]]"
        )
        column_constructor = f"mapped_column(TypeType, nullable={wrapped_field.is_optional}, use_existing_column=True)"
        self.custom_columns.append(
            ColumnConstructor(column_name, column_type, column_constructor)
        )

    def create_one_to_one_relationship(self, wrapped_field: WrappedField):
        """
        Create a one-to-one relationship with using the given field.
        This adds a foreign key and a relationship to this table.

        :param wrapped_field: The field to get the information from.
        """
        # create foreign key
        fk_name = f"{wrapped_field.field.name}{self.ormatic.foreign_key_postfix}"
        fk_type = (
            f"Mapped[{module_and_class_name(Optional)}[{module_and_class_name(int)}]]"
            if wrapped_field.is_optional
            else "Mapped[int]"
        )

        # get the target table
        target_wrapped_table = self.ormatic.wrapped_tables[
            self.ormatic.class_dependency_graph.get_wrapped_class(
                wrapped_field.type_endpoint
            )
        ]

        # columns have to be nullable and use_alter=True since the insertion order might be incorrect otherwise
        fk_column_constructor = f"mapped_column(ForeignKey('{target_wrapped_table.full_primary_key_name}', use_alter=True), nullable=True, use_existing_column=True)"

        self.foreign_keys.append(
            ColumnConstructor(fk_name, fk_type, fk_column_constructor)
        )

        # create relationship to remote side
        rel_name = f"{wrapped_field.field.name}"
        rel_type = f"Mapped[{target_wrapped_table.tablename}]"
        # relationships have to be post updated since since it won't work in the case of subclasses with another ref otherwise
        rel_constructor = f"relationship('{target_wrapped_table.tablename}', uselist=False, foreign_keys=[{fk_name}], post_update=True)"
        self.relationships.append(
            ColumnConstructor(rel_name, rel_type, rel_constructor)
        )

    def create_one_to_many_relationship(self, wrapped_field: WrappedField):
        """
        Creates a one-to-many relationship mapping for the given wrapped field.
        The target side of the wrapped field gets a foreign key to this table with a unique name.
        This table gets a relationship that joins the target table with the foreign key.

        :param wrapped_field: The field for the one-to-many relationship.
        """

        # get the target table
        target_wrapped_table = self.ormatic.wrapped_tables[
            self.ormatic.class_dependency_graph.get_wrapped_class(
                wrapped_field.type_endpoint
            )
        ]

        # create a foreign key to this on the remote side
        fk_name = f"{self.tablename.lower()}_{wrapped_field.field.name}{self.ormatic.foreign_key_postfix}"
        fk_type = (
            f"Mapped[{module_and_class_name(Optional)}[{module_and_class_name(int)}]]"
        )
        fk_column_constructor = f"mapped_column(ForeignKey('{self.full_primary_key_name}', use_alter=True), nullable=True, use_existing_column=True)"
        target_wrapped_table.foreign_keys.append(
            ColumnConstructor(fk_name, fk_type, fk_column_constructor)
        )

        # create a relationship with a list to collect the other side
        rel_name = f"{wrapped_field.field.name}"
        rel_type = (
            f"Mapped[{module_and_class_name(List)}[{target_wrapped_table.tablename}]]"
        )
        rel_constructor = f"relationship('{target_wrapped_table.tablename}', foreign_keys='[{target_wrapped_table.tablename}.{fk_name}]', post_update=True)"
        self.relationships.append(
            ColumnConstructor(rel_name, rel_type, rel_constructor)
        )

    def create_container_of_builtins(self, wrapped_field: WrappedField):
        """
        Create a column for a list-like of built-in values.

        :param wrapped_field: The field to extract the information from.
        """
        self.ormatic.imported_modules.add("typing_extensions")
        column_name = wrapped_field.field.name
        container = Set if issubclass(wrapped_field.container_type, set) else List
        column_type = f"Mapped[{module_and_class_name(container)}[{module_and_class_name(wrapped_field.type_endpoint)}]]"
        column_constructor = f"mapped_column(JSON, nullable={wrapped_field.is_optional}, use_existing_column=True)"
        self.custom_columns.append(
            ColumnConstructor(column_name, column_type, column_constructor)
        )

    def create_custom_type(self, wrapped_field: WrappedField):
        custom_type = self.ormatic.type_mappings[wrapped_field.type_endpoint]
        column_name = wrapped_field.field.name
        column_type = (
            f"Mapped[{custom_type.__module__}.{custom_type.__name__}]"
            if not wrapped_field.is_optional
            else f"Mapped[{module_and_class_name(Optional)}[{custom_type.__module__}.{custom_type.__name__}]]"
        )

        constructor = f"mapped_column({custom_type.__module__}.{custom_type.__name__}, nullable={wrapped_field.is_optional}, use_existing_column=True)"

        self.custom_columns.append(
            ColumnConstructor(column_name, column_type, constructor)
        )

    @property
    def base_class_name(self):
        if self.parent_table is not None:
            return self.parent_table.tablename
        else:
            return "Base"

    def __hash__(self):
        return hash(self.wrapped_clazz)
