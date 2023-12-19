# Document Store
### Dict `classtyping._SpecialGenericAlias`

A generic version of dict.

## Document

Document holds text and metadata of a document.

Examples of documents are PDFs, Word documents, etc. A collection of
related text in an NLP application can be thought of a document as
well.

### metadata `classpydantic.fields.FieldInfo`

This class holds information about a field.

`FieldInfo` is used for any field definition regardless of whether the [`Field()`][pydantic.fields.Field]
function is explicitly used.

!!! warning
    You generally shouldn't be creating `FieldInfo` directly, you'll only need to use it when accessing
    [`BaseModel`][pydantic.main.BaseModel] `.model_fields` internals.

Attributes:
    annotation: The type annotation of the field.
    default: The default value of the field.
    default_factory: The factory function used to construct the default for the field.
    alias: The alias name of the field.
    alias_priority: The priority of the field's alias.
    validation_alias: The validation alias name of the field.
    serialization_alias: The serialization alias name of the field.
    title: The title of the field.
    description: The description of the field.
    examples: List of examples of the field.
    exclude: Whether to exclude the field from the model serialization.
    discriminator: Field name for discriminating the type in a tagged union.
    json_schema_extra: Dictionary of extra JSON schema properties.
    frozen: Whether the field is frozen.
    validate_default: Whether to validate the default value of the field.
    repr: Whether to include the field in representation of the model.
    init_var: Whether the field should be included in the constructor of the dataclass.
    kw_only: Whether the field should be a keyword-only argument in the constructor of the dataclass.
    metadata: List of metadata constraints.


## DocumentStoreBase

Abstract class for a store that can store text, and metadata from
documents.

The store can be queried by text for similar documents.

### add_document `classfunction`

```
add_document(
  self,
  document: guardrails.document_store.Document
) -> None
```

Adds a document to the store.

Args:
    document: Document object to be added

Returns:
    None if the document was added successfully

### add_text `classfunction`

```
add_text(
  self,
  text: str,
  meta: Dict[Any, Any]
) -> <class 'str'>
```

Adds a text to the store.
Args:
    text: Text to add.
    meta: Metadata to associate with the text.

Returns:
    The id of the text.

### add_texts `classfunction`

```
add_texts(
  self,
  texts: Dict[str, Dict[Any, Any]]
) -> typing.List[str]
```

Adds a list of texts to the store.
Args:
    texts: List of texts to add, and their associalted metadata.
    example: [{"I am feeling good", {"sentiment": "postive"}}]

Returns:
    List of ids of the texts.

### flush `classfunction`

```
flush(

)
```

Flushes the store to disk.

### search `classfunction`

```
search(
  self,
  query: str,
  k: int = 4
) -> typing.List[guardrails.document_store.Page]
```

Searches for pages which contain the text similar to the query.

Args:
    query: Text to search for.
    k: Number of similar pages to return.

Returns:
    List[Pages] List of pages which contains similar texts


## RealEphemeralDocumentStore

EphemeralDocumentStore is a document store that stores the documents
on local disk and use a ephemeral vector store like Faiss.

### add_document `classfunction`

```
add_document(
  self,
  document: guardrails.document_store.Document
)
```

Adds a document to the store.

Args:
    document: Document object to be added

Returns:
    None if the document was added successfully

### add_text `classfunction`

```
add_text(
  self,
  text: str,
  meta: Dict[Any, Any]
) -> <class 'str'>
```

Adds a text to the store.
Args:
    text: Text to add.
    meta: Metadata to associate with the text.

Returns:
    The id of the text.

### add_texts `classfunction`

```
add_texts(
  self,
  texts: Dict[str, Dict[Any, Any]]
) -> typing.List[str]
```

Adds a list of texts to the store.
Args:
    texts: List of texts to add, and their associalted metadata.
    example: [{"I am feeling good", {"sentiment": "postive"}}]

Returns:
    List of ids of the texts.

### flush `classfunction`

```
flush(
  self,
  path: Optional[str] = None
)
```

Flushes the store to disk.

### search `classfunction`

```
search(
  self,
  query: str,
  k: int = 4
) -> typing.List[guardrails.document_store.Page]
```

Searches for pages which contain the text similar to the query.

Args:
    query: Text to search for.
    k: Number of similar pages to return.

Returns:
    List[Pages] List of pages which contains similar texts

### search_with_threshold `classfunction`

```
search_with_threshold(
  self,
  query: str,
  threshold: float,
  k: int = 4
) -> typing.List[guardrails.document_store.Page]
```


### Field `classfunction`

```
Field(
  default: 'Any' = PydanticUndefined,
  default_factory: 'typing.Callable[[], Any] | None' = PydanticUndefined,
  alias: 'str | None' = PydanticUndefined,
  alias_priority: 'int | None' = PydanticUndefined,
  validation_alias: 'str | AliasPath | AliasChoices | None' = PydanticUndefined,
  serialization_alias: 'str | None' = PydanticUndefined,
  title: 'str | None' = PydanticUndefined,
  description: 'str | None' = PydanticUndefined,
  examples: 'list[Any] | None' = PydanticUndefined,
  exclude: 'bool | None' = PydanticUndefined,
  discriminator: 'str | None' = PydanticUndefined,
  json_schema_extra: 'dict[str, Any] | typing.Callable[[dict[str, Any]], None] | None' = PydanticUndefined,
  frozen: 'bool | None' = PydanticUndefined,
  validate_default: 'bool | None' = PydanticUndefined,
  repr: 'bool' = PydanticUndefined,
  init_var: 'bool | None' = PydanticUndefined,
  kw_only: 'bool | None' = PydanticUndefined,
  pattern: 'str | None' = PydanticUndefined,
  strict: 'bool | None' = PydanticUndefined,
  gt: 'float | None' = PydanticUndefined,
  ge: 'float | None' = PydanticUndefined,
  lt: 'float | None' = PydanticUndefined,
  le: 'float | None' = PydanticUndefined,
  multiple_of: 'float | None' = PydanticUndefined,
  allow_inf_nan: 'bool | None' = PydanticUndefined,
  max_digits: 'int | None' = PydanticUndefined,
  decimal_places: 'int | None' = PydanticUndefined,
  min_length: 'int | None' = PydanticUndefined,
  max_length: 'int | None' = PydanticUndefined,
  union_mode: "Literal['smart', 'left_to_right']" = PydanticUndefined,
  **extra: 'Unpack[_EmptyKwargs]'
) -> Any
```

Usage docs: https://docs.pydantic.dev/2.4/concepts/fields

Create a field for objects that can be configured.

Used to provide extra information about a field, either for the model schema or complex validation. Some arguments
apply only to number fields (`int`, `float`, `Decimal`) and some apply only to `str`.

Note:
    - Any `_Unset` objects will be replaced by the corresponding value defined in the `_DefaultValues` dictionary. If a key for the `_Unset` object is not found in the `_DefaultValues` dictionary, it will default to `None`

Args:
    default: Default value if the field is not set.
    default_factory: A callable to generate the default value, such as :func:`~datetime.utcnow`.
    alias: An alternative name for the attribute.
    alias_priority: Priority of the alias. This affects whether an alias generator is used.
    validation_alias: 'Whitelist' validation step. The field will be the single one allowed by the alias or set of
        aliases defined.
    serialization_alias: 'Blacklist' validation step. The vanilla field will be the single one of the alias' or set
        of aliases' fields and all the other fields will be ignored at serialization time.
    title: Human-readable title.
    description: Human-readable description.
    examples: Example values for this field.
    exclude: Whether to exclude the field from the model serialization.
    discriminator: Field name for discriminating the type in a tagged union.
    json_schema_extra: Any additional JSON schema data for the schema property.
    frozen: Whether the field is frozen.
    validate_default: Run validation that isn't only checking existence of defaults. This can be set to `True` or `False`. If not set, it defaults to `None`.
    repr: A boolean indicating whether to include the field in the `__repr__` output.
    init_var: Whether the field should be included in the constructor of the dataclass.
    kw_only: Whether the field should be a keyword-only argument in the constructor of the dataclass.
    strict: If `True`, strict validation is applied to the field.
        See [Strict Mode](../concepts/strict_mode.md) for details.
    gt: Greater than. If set, value must be greater than this. Only applicable to numbers.
    ge: Greater than or equal. If set, value must be greater than or equal to this. Only applicable to numbers.
    lt: Less than. If set, value must be less than this. Only applicable to numbers.
    le: Less than or equal. If set, value must be less than or equal to this. Only applicable to numbers.
    multiple_of: Value must be a multiple of this. Only applicable to numbers.
    min_length: Minimum length for strings.
    max_length: Maximum length for strings.
    pattern: Pattern for strings.
    allow_inf_nan: Allow `inf`, `-inf`, `nan`. Only applicable to numbers.
    max_digits: Maximum number of allow digits for strings.
    decimal_places: Maximum number of decimal places allowed for numbers.
    union_mode: The strategy to apply when validating a union. Can be `smart` (the default), or `left_to_right`.
        See [Union Mode](standard_library_types.md#union-mode) for details.
    extra: Include extra fields used by the JSON schema.

        !!! warning Deprecated
            The `extra` kwargs is deprecated. Use `json_schema_extra` instead.

Returns:
    A new [`FieldInfo`][pydantic.fields.FieldInfo], the return annotation is `Any` so `Field` can be used on
        type annotated fields without causing a typing error.

### List `classtyping._SpecialGenericAlias`

A generic version of list.

### Optional `classtyping._SpecialForm`

Optional[X] is equivalent to Union[X, None].

## Page

Page holds text and metadata of a page in a document.

It also containts the coordinates of the page in the document.


## PageCoordinates

PageCoordinates(doc_id, page_num)

### count `classmethod_descriptor`

```
count(
  self,
  value
)
```

Return number of occurrences of value.

### doc_id `class_collections._tuplegetter`

Alias for field number 0

### index `classmethod_descriptor`

```
index(
  self,
  value,
  start=0,
  stop=9223372036854775807
)
```

Return first index of value.

Raises ValueError if the value is not present.

### page_num `class_collections._tuplegetter`

Alias for field number 1



## RealSQLMetadataStore
### add_docs `classfunction`

```
add_docs(
  self,
  docs: List[guardrails.document_store.Document],
  vdb_last_index: int
)
```

### get_pages_for_for_indexes `classfunction`

```
get_pages_for_for_indexes(
  self,
  indexes: List[int]
) -> typing.List[guardrails.document_store.Page]
```


## RealSqlDocument

The base class of the class hierarchy.

When called, it accepts no arguments and returns a new featureless
instance that has no instance attributes and cannot be given any.

### id `classsqlalchemy.orm.attributes.InstrumentedAttribute`

### meta `classsqlalchemy.orm.attributes.InstrumentedAttribute`

### metadata `classsqlalchemy.sql.schema.MetaData`

A collection of :class:`_schema.Table`
objects and their associated schema
constructs.

Holds a collection of :class:`_schema.Table` objects as well as
an optional binding to an :class:`_engine.Engine` or
:class:`_engine.Connection`.  If bound, the :class:`_schema.Table` objects
in the collection and their columns may participate in implicit SQL
execution.

The :class:`_schema.Table` objects themselves are stored in the
:attr:`_schema.MetaData.tables` dictionary.

:class:`_schema.MetaData` is a thread-safe object for read operations.
Construction of new tables within a single :class:`_schema.MetaData`
object,
either explicitly or via reflection, may not be completely thread-safe.

.. seealso::

    :ref:`metadata_describing` - Introduction to database metadata

### page_num `classsqlalchemy.orm.attributes.InstrumentedAttribute`

### registry `classsqlalchemy.orm.decl_api.registry`

Generalized registry for mapping classes.

The :class:`_orm.registry` serves as the basis for maintaining a collection
of mappings, and provides configurational hooks used to map classes.

The three general kinds of mappings supported are Declarative Base,
Declarative Decorator, and Imperative Mapping.   All of these mapping
styles may be used interchangeably:

* :meth:`_orm.registry.generate_base` returns a new declarative base
  class, and is the underlying implementation of the
  :func:`_orm.declarative_base` function.

* :meth:`_orm.registry.mapped` provides a class decorator that will
  apply declarative mapping to a class without the use of a declarative
  base class.

* :meth:`_orm.registry.map_imperatively` will produce a
  :class:`_orm.Mapper` for a class without scanning the class for
  declarative class attributes. This method suits the use case historically
  provided by the ``sqlalchemy.orm.mapper()`` classical mapping function,
  which is removed as of SQLAlchemy 2.0.

.. versionadded:: 1.4

.. seealso::

    :ref:`orm_mapping_classes_toplevel` - overview of class mapping
    styles.

### text `classsqlalchemy.orm.attributes.InstrumentedAttribute`

### vector_index `classsqlalchemy.orm.attributes.InstrumentedAttribute`




### abstractmethod `classfunction`

```
abstractmethod(
  funcobj
)
```

A decorator indicating abstract methods.

Requires that the metaclass is ABCMeta or derived from it.  A
class that has a metaclass derived from ABCMeta cannot be
instantiated unless all of its abstract methods are overridden.
The abstract methods can be called using any of the normal
'super' call mechanisms.  abstractmethod() may be used to declare
abstract methods for properties and descriptors.

Usage:

    class C(metaclass=ABCMeta):
        @abstractmethod
        def my_abstract_method(self, arg1, arg2, argN):
            ...

### dataclass `classfunction`

```
dataclass(
  cls=None,
  init=True,
  repr=True,
  eq=True,
  order=False,
  unsafe_hash=False,
  frozen=False,
  match_args=True,
  kw_only=False,
  slots=False,
  weakref_slot=False
)
```

Add dunder methods based on the fields defined in the class.

Examines PEP 526 __annotations__ to determine fields.

If init is true, an __init__() method is added to the class. If repr
is true, a __repr__() method is added. If order is true, rich
comparison dunder methods are added. If unsafe_hash is true, a
__hash__() method is added. If frozen is true, fields may not be
assigned to after instance creation. If match_args is true, the
__match_args__ tuple is added. If kw_only is true, then by default
all fields are keyword-only. If slots is true, a new class with a
__slots__ attribute is returned.

### declarative_base `classfunction`

```
declarative_base(
  metadata: 'Optional[MetaData]' = None,
  mapper: 'Optional[Callable[..., Mapper[Any]]]' = None,
  cls: 'Type[Any]' = <class 'object'>,
  name: 'str' = 'Base',
  class_registry: 'Optional[clsregistry._ClsRegistryType]' = None,
  type_annotation_map: 'Optional[_TypeAnnotationMapType]' = None,
  constructor: 'Callable[..., None]' = <function _declarative_constructor at 0x2b11944a0>,
  metaclass: 'Type[Any]' = <class 'sqlalchemy.orm.decl_api.DeclarativeMeta'>
) -> Any
```

Construct a base class for declarative class definitions.

The new base class will be given a metaclass that produces
appropriate :class:`~sqlalchemy.schema.Table` objects and makes
the appropriate :class:`_orm.Mapper` calls based on the
information provided declaratively in the class and any subclasses
of the class.

.. versionchanged:: 2.0 Note that the :func:`_orm.declarative_base`
   function is superseded by the new :class:`_orm.DeclarativeBase` class,
   which generates a new "base" class using subclassing, rather than
   return value of a function.  This allows an approach that is compatible
   with :pep:`484` typing tools.

The :func:`_orm.declarative_base` function is a shorthand version
of using the :meth:`_orm.registry.generate_base`
method.  That is, the following::

    from sqlalchemy.orm import declarative_base

    Base = declarative_base()

Is equivalent to::

    from sqlalchemy.orm import registry

    mapper_registry = registry()
    Base = mapper_registry.generate_base()

See the docstring for :class:`_orm.registry`
and :meth:`_orm.registry.generate_base`
for more details.

.. versionchanged:: 1.4  The :func:`_orm.declarative_base`
   function is now a specialization of the more generic
   :class:`_orm.registry` class.  The function also moves to the
   ``sqlalchemy.orm`` package from the ``declarative.ext`` package.


:param metadata:
  An optional :class:`~sqlalchemy.schema.MetaData` instance.  All
  :class:`~sqlalchemy.schema.Table` objects implicitly declared by
  subclasses of the base will share this MetaData.  A MetaData instance
  will be created if none is provided.  The
  :class:`~sqlalchemy.schema.MetaData` instance will be available via the
  ``metadata`` attribute of the generated declarative base class.

:param mapper:
  An optional callable, defaults to :class:`_orm.Mapper`. Will
  be used to map subclasses to their Tables.

:param cls:
  Defaults to :class:`object`. A type to use as the base for the generated
  declarative base class. May be a class or tuple of classes.

:param name:
  Defaults to ``Base``.  The display name for the generated
  class.  Customizing this is not required, but can improve clarity in
  tracebacks and debugging.

:param constructor:
  Specify the implementation for the ``__init__`` function on a mapped
  class that has no ``__init__`` of its own.  Defaults to an
  implementation that assigns \**kwargs for declared
  fields and relationships to an instance.  If ``None`` is supplied,
  no __init__ will be provided and construction will fall back to
  cls.__init__ by way of the normal Python semantics.

:param class_registry: optional dictionary that will serve as the
  registry of class names-> mapped classes when string names
  are used to identify classes inside of :func:`_orm.relationship`
  and others.  Allows two or more declarative base classes
  to share the same registry of class names for simplified
  inter-base relationships.

:param type_annotation_map: optional dictionary of Python types to
    SQLAlchemy :class:`_types.TypeEngine` classes or instances.  This
    is used exclusively by the :class:`_orm.MappedColumn` construct
    to produce column types based on annotations within the
    :class:`_orm.Mapped` type.


    .. versionadded:: 2.0

    .. seealso::

        :ref:`orm_declarative_mapped_column_type_map`

:param metaclass:
  Defaults to :class:`.DeclarativeMeta`.  A metaclass or __metaclass__
  compatible callable to use as the meta type of the generated
  declarative base class.

.. seealso::

    :class:`_orm.registry`

### mapped_column `classfunction`

```
mapped_column(
  __name_pos: 'Optional[Union[str, _TypeEngineArgument[Any], SchemaEventTarget]]' = None,
  __type_pos: 'Optional[Union[_TypeEngineArgument[Any], SchemaEventTarget]]' = None,
  *args: 'SchemaEventTarget',
  init: 'Union[_NoArg, bool]' = _NoArg.NO_ARG,
  repr: 'Union[_NoArg, bool]' = _NoArg.NO_ARG,
  default: 'Optional[Any]' = _NoArg.NO_ARG,
  default_factory: 'Union[_NoArg, Callable[[], _T]]' = _NoArg.NO_ARG,
  compare: 'Union[_NoArg, bool]' = _NoArg.NO_ARG,
  kw_only: 'Union[_NoArg, bool]' = _NoArg.NO_ARG,
  nullable: 'Optional[Union[bool, Literal[SchemaConst.NULL_UNSPECIFIED]]]' = <SchemaConst.NULL_UNSPECIFIED: 3>,
  primary_key: 'Optional[bool]' = False,
  deferred: 'Union[_NoArg, bool]' = _NoArg.NO_ARG,
  deferred_group: 'Optional[str]' = None,
  deferred_raiseload: 'Optional[bool]' = None,
  use_existing_column: 'bool' = False,
  name: 'Optional[str]' = None,
  type_: 'Optional[_TypeEngineArgument[Any]]' = None,
  autoincrement: '_AutoIncrementType' = 'auto',
  doc: 'Optional[str]' = None,
  key: 'Optional[str]' = None,
  index: 'Optional[bool]' = None,
  unique: 'Optional[bool]' = None,
  info: 'Optional[_InfoType]' = None,
  onupdate: 'Optional[Any]' = None,
  insert_default: 'Optional[Any]' = _NoArg.NO_ARG,
  server_default: 'Optional[_ServerDefaultArgument]' = None,
  server_onupdate: 'Optional[FetchedValue]' = None,
  active_history: 'bool' = False,
  quote: 'Optional[bool]' = None,
  system: 'bool' = False,
  comment: 'Optional[str]' = None,
  sort_order: 'Union[_NoArg, int]' = _NoArg.NO_ARG,
  **kw: 'Any'
) -> MappedColumn[Any]
```

declare a new ORM-mapped :class:`_schema.Column` construct
for use within :ref:`Declarative Table <orm_declarative_table>`
configuration.

The :func:`_orm.mapped_column` function provides an ORM-aware and
Python-typing-compatible construct which is used with
:ref:`declarative <orm_declarative_mapping>` mappings to indicate an
attribute that's mapped to a Core :class:`_schema.Column` object.  It
provides the equivalent feature as mapping an attribute to a
:class:`_schema.Column` object directly when using Declarative,
specifically when using :ref:`Declarative Table <orm_declarative_table>`
configuration.

.. versionadded:: 2.0

:func:`_orm.mapped_column` is normally used with explicit typing along with
the :class:`_orm.Mapped` annotation type, where it can derive the SQL
type and nullability for the column based on what's present within the
:class:`_orm.Mapped` annotation.   It also may be used without annotations
as a drop-in replacement for how :class:`_schema.Column` is used in
Declarative mappings in SQLAlchemy 1.x style.

For usage examples of :func:`_orm.mapped_column`, see the documentation
at :ref:`orm_declarative_table`.

.. seealso::

    :ref:`orm_declarative_table` - complete documentation

    :ref:`whatsnew_20_orm_declarative_typing` - migration notes for
    Declarative mappings using 1.x style mappings

:param __name: String name to give to the :class:`_schema.Column`.  This
 is an optional, positional only argument that if present must be the
 first positional argument passed.  If omitted, the attribute name to
 which the :func:`_orm.mapped_column`  is mapped will be used as the SQL
 column name.
:param __type: :class:`_types.TypeEngine` type or instance which will
 indicate the datatype to be associated with the :class:`_schema.Column`.
 This is an optional, positional-only argument that if present must
 immediately follow the ``__name`` parameter if present also, or otherwise
 be the first positional parameter.  If omitted, the ultimate type for
 the column may be derived either from the annotated type, or if a
 :class:`_schema.ForeignKey` is present, from the datatype of the
 referenced column.
:param \*args: Additional positional arguments include constructs such
 as :class:`_schema.ForeignKey`, :class:`_schema.CheckConstraint`,
 and :class:`_schema.Identity`, which are passed through to the constructed
 :class:`_schema.Column`.
:param nullable: Optional bool, whether the column should be "NULL" or
 "NOT NULL". If omitted, the nullability is derived from the type
 annotation based on whether or not ``typing.Optional`` is present.
 ``nullable`` defaults to ``True`` otherwise for non-primary key columns,
 and ``False`` for primary key columns.
:param primary_key: optional bool, indicates the :class:`_schema.Column`
 would be part of the table's primary key or not.
:param deferred: Optional bool - this keyword argument is consumed by the
 ORM declarative process, and is not part of the :class:`_schema.Column`
 itself; instead, it indicates that this column should be "deferred" for
 loading as though mapped by :func:`_orm.deferred`.

 .. seealso::

    :ref:`orm_queryguide_deferred_declarative`

:param deferred_group: Implies :paramref:`_orm.mapped_column.deferred`
 to ``True``, and set the :paramref:`_orm.deferred.group` parameter.

 .. seealso::

    :ref:`orm_queryguide_deferred_group`

:param deferred_raiseload: Implies :paramref:`_orm.mapped_column.deferred`
 to ``True``, and set the :paramref:`_orm.deferred.raiseload` parameter.

 .. seealso::

    :ref:`orm_queryguide_deferred_raiseload`

:param use_existing_column: if True, will attempt to locate the given
 column name on an inherited superclass (typically single inheriting
 superclass), and if present, will not produce a new column, mapping
 to the superclass column as though it were omitted from this class.
 This is used for mixins that add new columns to an inherited superclass.

 .. seealso::

    :ref:`orm_inheritance_column_conflicts`

 .. versionadded:: 2.0.0b4

:param default: Passed directly to the
 :paramref:`_schema.Column.default` parameter if the
 :paramref:`_orm.mapped_column.insert_default` parameter is not present.
 Additionally, when used with :ref:`orm_declarative_native_dataclasses`,
 indicates a default Python value that should be applied to the keyword
 constructor within the generated ``__init__()`` method.

 Note that in the case of dataclass generation when
 :paramref:`_orm.mapped_column.insert_default` is not present, this means
 the :paramref:`_orm.mapped_column.default` value is used in **two**
 places, both the ``__init__()`` method as well as the
 :paramref:`_schema.Column.default` parameter. While this behavior may
 change in a future release, for the moment this tends to "work out"; a
 default of ``None`` will mean that the :class:`_schema.Column` gets no
 default generator, whereas a default that refers to a non-``None`` Python
 or SQL expression value will be assigned up front on the object when
 ``__init__()`` is called, which is the same value that the Core
 :class:`_sql.Insert` construct would use in any case, leading to the same
 end result.

 .. note:: When using Core level column defaults that are callables to
    be interpreted by the underlying :class:`_schema.Column` in conjunction
    with :ref:`ORM-mapped dataclasses
    <orm_declarative_native_dataclasses>`, especially those that are
    :ref:`context-aware default functions <context_default_functions>`,
    **the** :paramref:`_orm.mapped_column.insert_default` **parameter must
    be used instead**.  This is necessary to disambiguate the callable from
    being interpreted as a dataclass level default.

:param insert_default: Passed directly to the
 :paramref:`_schema.Column.default` parameter; will supersede the value
 of :paramref:`_orm.mapped_column.default` when present, however
 :paramref:`_orm.mapped_column.default` will always apply to the
 constructor default for a dataclasses mapping.

:param sort_order: An integer that indicates how this mapped column
 should be sorted compared to the others when the ORM is creating a
 :class:`_schema.Table`. Among mapped columns that have the same
 value the default ordering is used, placing first the mapped columns
 defined in the main class, then the ones in the super classes.
 Defaults to 0. The sort is ascending.

 .. versionadded:: 2.0.4

:param active_history=False:

    When ``True``, indicates that the "previous" value for a
    scalar attribute should be loaded when replaced, if not
    already loaded. Normally, history tracking logic for
    simple non-primary-key scalar values only needs to be
    aware of the "new" value in order to perform a flush. This
    flag is available for applications that make use of
    :func:`.attributes.get_history` or :meth:`.Session.is_modified`
    which also need to know the "previous" value of the attribute.

    .. versionadded:: 2.0.10


:param init: Specific to :ref:`orm_declarative_native_dataclasses`,
 specifies if the mapped attribute should be part of the ``__init__()``
 method as generated by the dataclass process.
:param repr: Specific to :ref:`orm_declarative_native_dataclasses`,
 specifies if the mapped attribute should be part of the ``__repr__()``
 method as generated by the dataclass process.
:param default_factory: Specific to
 :ref:`orm_declarative_native_dataclasses`,
 specifies a default-value generation function that will take place
 as part of the ``__init__()``
 method as generated by the dataclass process.
:param compare: Specific to
 :ref:`orm_declarative_native_dataclasses`, indicates if this field
 should be included in comparison operations when generating the
 ``__eq__()`` and ``__ne__()`` methods for the mapped class.

 .. versionadded:: 2.0.0b4

:param kw_only: Specific to
 :ref:`orm_declarative_native_dataclasses`, indicates if this field
 should be marked as keyword-only when generating the ``__init__()``.

:param \**kw: All remaining keyword arguments are passed through to the
 constructor for the :class:`_schema.Column`.

### namedtuple `classfunction`

```
namedtuple(
  typename,
  field_names,
  rename=False,
  defaults=None,
  module=None
)
```

Returns a new subclass of tuple with named fields.

>>> Point = namedtuple('Point', ['x', 'y'])
>>> Point.__doc__                   # docstring for the new class
'Point(x, y)'
>>> p = Point(11, y=22)             # instantiate with positional args or keywords
>>> p[0] + p[1]                     # indexable like a plain tuple
33
>>> x, y = p                        # unpack like a regular tuple
>>> x, y
(11, 22)
>>> p.x + p.y                       # fields also accessible by name
33
>>> d = p._asdict()                 # convert to a dictionary
>>> d['x']
11
>>> Point(**d)                      # convert from a dictionary
Point(x=11, y=22)
>>> p._replace(x=100)               # _replace() is like str.replace() but targets named fields
Point(x=100, y=22)

