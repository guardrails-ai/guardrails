# History & Logs
## Call

Usage docs: https://docs.pydantic.dev/2.4/concepts/models/

A base class for creating Pydantic models.

Attributes:
    __class_vars__: The names of classvars defined on the model.
    __private_attributes__: Metadata about the private attributes of the model.
    __signature__: The signature for instantiating the model.

    __pydantic_complete__: Whether model building is completed, or if there are still undefined fields.
    __pydantic_core_schema__: The pydantic-core schema used to build the SchemaValidator and SchemaSerializer.
    __pydantic_custom_init__: Whether the model has a custom `__init__` function.
    __pydantic_decorators__: Metadata containing the decorators defined on the model.
        This replaces `Model.__validators__` and `Model.__root_validators__` from Pydantic V1.
    __pydantic_generic_metadata__: Metadata for generic models; contains data used for a similar purpose to
        __args__, __origin__, __parameters__ in typing-module generics. May eventually be replaced by these.
    __pydantic_parent_namespace__: Parent namespace of the model, used for automatic rebuilding of models.
    __pydantic_post_init__: The name of the post-init method for the model, if defined.
    __pydantic_root_model__: Whether the model is a `RootModel`.
    __pydantic_serializer__: The pydantic-core SchemaSerializer used to dump instances of the model.
    __pydantic_validator__: The pydantic-core SchemaValidator used to validate instances of the model.

    __pydantic_extra__: An instance attribute with the values of extra fields from validation when
        `model_config['extra'] == 'allow'`.
    __pydantic_fields_set__: An instance attribute with the names of fields explicitly specified during validation.
    __pydantic_private__: Instance attribute with the values of private attributes set on the model instance.

### __abstractmethods__ `classfrozenset`

frozenset() -> empty frozenset object
frozenset(iterable) -> frozenset object

Build an immutable unordered collection of unique elements.

### __annotations__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __class_getitem__ `classmethod`

```
__class_getitem__(
  typevar_values: 'type[Any] | tuple[type[Any], ...]'
) -> type[BaseModel] | _forward_ref.PydanticRecursiveRef
```

### __class_vars__ `classset`

set() -> new empty set object
set(iterable) -> new set object

Build an unordered collection of unique elements.

### __copy__ `classfunction`

```
__copy__(
  self: 'Model'
) -> Model
```

Returns a shallow copy of the model.

### __deepcopy__ `classfunction`

```
__deepcopy__(
  self: 'Model',
  memo: 'dict[int, Any] | None' = None
) -> Model
```

Returns a deep copy of the model.

### __delattr__ `classfunction`

```
__delattr__(
  self,
  item: 'str'
) -> Any
```

Implement delattr(self, name).

### __dict__ `classmappingproxy`

### __dir__ `classmethod_descriptor`

```
__dir__(
  self
)
```

Default dir() implementation.

### __doc__ `classNoneType`

### __eq__ `classfunction`

```
__eq__(
  self,
  other: 'Any'
) -> bool
```

Return self==value.

### __fields__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __fields_set__ `classproperty`

### __format__ `classmethod_descriptor`

```
__format__(
  self,
  format_spec
)
```

Default object formatter.

### __ge__ `classwrapper_descriptor`

```
__ge__(
  self,
  value
)
```

Return self>=value.

### __get_pydantic_core_schema__ `classmethod`

```
__get_pydantic_core_schema__(
  _BaseModel__source: 'type[BaseModel]',
  _BaseModel__handler: 'GetCoreSchemaHandler'
) -> CoreSchema
```

Hook into generating the model's CoreSchema.

Args:
    __source: The class we are generating a schema for.
        This will generally be the same as the `cls` argument if this is a classmethod.
    __handler: Call into Pydantic's internal JSON schema generation.
        A callable that calls into Pydantic's internal CoreSchema generation logic.

Returns:
    A `pydantic-core` `CoreSchema`.

### __get_pydantic_json_schema__ `classmethod`

```
__get_pydantic_json_schema__(
  _BaseModel__core_schema: 'CoreSchema',
  _BaseModel__handler: 'GetJsonSchemaHandler'
) -> JsonSchemaValue
```

Hook into generating the model's JSON schema.

Args:
    __core_schema: A `pydantic-core` CoreSchema.
        You can ignore this argument and call the handler with a new CoreSchema,
        wrap this CoreSchema (`{'type': 'nullable', 'schema': current_schema}`),
        or just call the handler with the original schema.
    __handler: Call into Pydantic's internal JSON schema generation.
        This will raise a `pydantic.errors.PydanticInvalidForJsonSchema` if JSON schema
        generation fails.
        Since this gets called by `BaseModel.model_json_schema` you can override the
        `schema_generator` argument to that function to change JSON schema generation globally
        for a type.

Returns:
    A JSON schema, as a Python object.

### __getattr__ `classfunction`

```
__getattr__(
  self,
  item: 'str'
) -> Any
```

### __getattribute__ `classwrapper_descriptor`

```
__getattribute__(
  self,
  name
)
```

Return getattr(self, name).

### __getstate__ `classfunction`

```
__getstate__(
  self
) -> dict[Any, Any]
```

### __gt__ `classwrapper_descriptor`

```
__gt__(
  self,
  value
)
```

Return self>value.

### __hash__ `classNoneType`

### __init__ `classfunction`

```
__init__(
  self,
  iterations: Optional[guardrails.classes.generic.stack.Stack[guardrails.classes.history.iteration.Iteration]] = None,
  inputs: Optional[guardrails.classes.history.call_inputs.CallInputs] = None,
  exception: Optional[Exception] = None
)
```

Create a new model by parsing and validating input data from keyword arguments.

Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
validated to form a valid model.

`__init__` uses `__pydantic_self__` instead of the more common `self` for the first arg to
allow `self` as a field name.

### __init_subclass__ `classbuiltin_function_or_method`

```
__init_subclass__
```

This method is called when a class is subclassed.

The default implementation does nothing. It may be
overridden to extend subclasses.

### __iter__ `classfunction`

```
__iter__(
  self
) -> TupleGenerator
```

So `dict(model)` works.

### __le__ `classwrapper_descriptor`

```
__le__(
  self,
  value
)
```

Return self<=value.

### __lt__ `classwrapper_descriptor`

```
__lt__(
  self,
  value
)
```

Return self<value.

### __module__ `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### __ne__ `classwrapper_descriptor`

```
__ne__(
  self,
  value
)
```

Return self!=value.

### __new__ `classbuiltin_function_or_method`

```
__new__(
  *args,
  **kwargs
)
```

Create and return a new object.  See help(type) for accurate signature.

### __pretty__ `classfunction`

```
__pretty__(
  self,
  fmt: 'typing.Callable[[Any], Any]',
  **kwargs: 'Any'
) -> typing.Generator[Any, None, None]
```

Used by devtools (https://python-devtools.helpmanual.io/) to pretty print objects.

### __private_attributes__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __pydantic_complete__ `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### __pydantic_core_schema__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __pydantic_custom_init__ `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### __pydantic_decorators__ `classpydantic._internal._decorators.DecoratorInfos`

Mapping of name in the class namespace to decorator info.

note that the name in the class namespace is the function or attribute name
not the field name!

### __pydantic_extra__ `classmember_descriptor`

### __pydantic_fields_set__ `classmember_descriptor`

### __pydantic_generic_metadata__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __pydantic_init_subclass__ `classmethod`

```
__pydantic_init_subclass__(
  **kwargs: 'Any'
) -> None
```

This is intended to behave just like `__init_subclass__`, but is called by `ModelMetaclass`
only after the class is actually fully initialized. In particular, attributes like `model_fields` will
be present when this is called.

This is necessary because `__init_subclass__` will always be called by `type.__new__`,
and it would require a prohibitively large refactor to the `ModelMetaclass` to ensure that
`type.__new__` was called in such a manner that the class would already be sufficiently initialized.

This will receive the same `kwargs` that would be passed to the standard `__init_subclass__`, namely,
any kwargs passed to the class definition that aren't used internally by pydantic.

Args:
    **kwargs: Any keyword arguments passed to the class definition that aren't used internally
        by pydantic.

### __pydantic_parent_namespace__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __pydantic_post_init__ `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### __pydantic_private__ `classmember_descriptor`

### __pydantic_root_model__ `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### __pydantic_serializer__ `classpydantic_core._pydantic_core.SchemaSerializer`

### __pydantic_validator__ `classpydantic_core._pydantic_core.SchemaValidator`

### __reduce__ `classmethod_descriptor`

```
__reduce__(
  self
)
```

Helper for pickle.

### __reduce_ex__ `classmethod_descriptor`

```
__reduce_ex__(
  self,
  protocol
)
```

Helper for pickle.

### __repr__ `classfunction`

```
__repr__(
  self
) -> str
```

Return repr(self).

### __repr_args__ `classfunction`

```
__repr_args__(
  self
) -> _repr.ReprArgs
```

### __repr_name__ `classfunction`

```
__repr_name__(
  self
) -> str
```

Name of the instance's class, used in __repr__.

### __repr_str__ `classfunction`

```
__repr_str__(
  self,
  join_str: 'str'
) -> str
```

### __rich_repr__ `classfunction`

```
__rich_repr__(
  self
) -> RichReprResult
```

Used by Rich (https://rich.readthedocs.io/en/stable/pretty.html) to pretty print objects.

### __setattr__ `classfunction`

```
__setattr__(
  self,
  name: 'str',
  value: 'Any'
) -> None
```

Implement setattr(self, name, value).

### __setstate__ `classfunction`

```
__setstate__(
  self,
  state: 'dict[Any, Any]'
) -> None
```

### __signature__ `classinspect.Signature`

A Signature object represents the overall signature of a function.
It stores a Parameter object for each parameter accepted by the
function, as well as information specific to the function itself.

A Signature object has the following public attributes and methods:

* parameters : OrderedDict
    An ordered mapping of parameters' names to the corresponding
    Parameter objects (keyword-only arguments are in the same order
    as listed in `code.co_varnames`).
* return_annotation : object
    The annotation for the return type of the function if specified.
    If the function has no annotation for its return type, this
    attribute is set to `Signature.empty`.
* bind(*args, **kwargs) -> BoundArguments
    Creates a mapping from positional and keyword arguments to
    parameters.
* bind_partial(*args, **kwargs) -> BoundArguments
    Creates a partial mapping from positional and keyword arguments
    to parameters (simulating 'functools.partial' behavior.)

### __sizeof__ `classmethod_descriptor`

```
__sizeof__(
  self
)
```

Size of object in memory, in bytes.

### __slots__ `classtuple`

Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.

### __str__ `classfunction`

```
__str__(
  self
) -> str
```

Return str(self).

### __subclasshook__ `classbuiltin_function_or_method`

```
__subclasshook__
```

Abstract classes can override this to customize issubclass().

This is invoked early on by abc.ABCMeta.__subclasscheck__().
It should return True, False or NotImplemented.  If it returns
NotImplemented, the normal algorithm is used.  Otherwise, it
overrides the normal algorithm (and the outcome is cached).

### __weakref__ `classgetset_descriptor`

list of weak references to the object (if defined)

### _abc_impl `class_abc._abc_data`

Internal state held by ABC machinery.

### _calculate_keys `classfunction`

```
_calculate_keys(
  self,
  *args: 'Any',
  **kwargs: 'Any'
) -> Any
```

### _copy_and_set_values `classfunction`

```
_copy_and_set_values(
  self,
  *args: 'Any',
  **kwargs: 'Any'
) -> Any
```

### _get_value `classmethod`

```
_get_value(
  *args: 'Any',
  **kwargs: 'Any'
) -> Any
```

### _iter `classfunction`

```
_iter(
  self,
  *args: 'Any',
  **kwargs: 'Any'
) -> Any
```

### compiled_instructions `classproperty`

The initial compiled instructions that were passed to the LLM on the
first call.

### compiled_prompt `classproperty`

The initial compiled prompt that was passed to the LLM on the first
call.

### completion_tokens_consumed `classproperty`

Returns the total number of completion tokens consumed during all
iterations with this call.

### construct `classmethod`

```
construct(
  _fields_set: 'set[str] | None' = None,
  **values: 'Any'
) -> Model
```

### copy `classfunction`

```
copy(
  self: 'Model',
  include: 'AbstractSetIntStr | MappingIntStrAny | None' = None,
  exclude: 'AbstractSetIntStr | MappingIntStrAny | None' = None,
  update: 'typing.Dict[str, Any] | None' = None,
  deep: 'bool' = False
) -> Model
```

Returns a copy of the model.

!!! warning "Deprecated"
    This method is now deprecated; use `model_copy` instead.

If you need `include` or `exclude`, use:

```py
data = self.model_dump(include=include, exclude=exclude, round_trip=True)
data = {**data, **(update or {})}
copied = self.model_validate(data)
```

Args:
    include: Optional set or mapping
        specifying which fields to include in the copied model.
    exclude: Optional set or mapping
        specifying which fields to exclude in the copied model.
    update: Optional dictionary of field-value pairs to override field values
        in the copied model.
    deep: If True, the values of fields that are Pydantic models will be deep copied.

Returns:
    A copy of the model with included, excluded and updated fields as specified.

### dict `classfunction`

```
dict(
  self,
  include: 'IncEx' = None,
  exclude: 'IncEx' = None,
  by_alias: 'bool' = False,
  exclude_unset: 'bool' = False,
  exclude_defaults: 'bool' = False,
  exclude_none: 'bool' = False
) -> typing.Dict[str, Any]
```

### error `classproperty`

The error message from any exception that raised and interrupted the
run.

### exception `classproperty`

The exception that interrupted the run.

### failed_validations `classproperty`

The validator logs for any validations that failed during the
entirety of the run.

### fixed_output `classproperty`

The cumulative validation output across all current iterations with
any automatic fixes applied.

### from_orm `classmethod`

```
from_orm(
  obj: 'Any'
) -> Model
```

### instructions `classproperty`

The instructions as provided by the user when intializing or calling
the Guard.

### json `classfunction`

```
json(
  self,
  include: 'IncEx' = None,
  exclude: 'IncEx' = None,
  by_alias: 'bool' = False,
  exclude_unset: 'bool' = False,
  exclude_defaults: 'bool' = False,
  exclude_none: 'bool' = False,
  encoder: 'typing.Callable[[Any], Any] | None' = PydanticUndefined,
  models_as_dict: 'bool' = PydanticUndefined,
  **dumps_kwargs: 'Any'
) -> str
```

### logs `classproperty`

Returns all logs from all iterations as a stack.

### model_computed_fields `classproperty`

Get the computed fields of this model instance.

Returns:
    A dictionary of computed field names and their corresponding `ComputedFieldInfo` objects.

### model_config `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### model_construct `classmethod`

```
model_construct(
  _fields_set: 'set[str] | None' = None,
  **values: 'Any'
) -> Model
```

Creates a new instance of the `Model` class with validated data.

Creates a new model setting `__dict__` and `__pydantic_fields_set__` from trusted or pre-validated data.
Default values are respected, but no other validation is performed.
Behaves as if `Config.extra = 'allow'` was set since it adds all passed values

Args:
    _fields_set: The set of field names accepted for the Model instance.
    values: Trusted or pre-validated data dictionary.

Returns:
    A new instance of the `Model` class with validated data.

### model_copy `classfunction`

```
model_copy(
  self: 'Model',
  update: 'dict[str, Any] | None' = None,
  deep: 'bool' = False
) -> Model
```

Usage docs: https://docs.pydantic.dev/2.4/concepts/serialization/#model_copy

Returns a copy of the model.

Args:
    update: Values to change/add in the new model. Note: the data is not validated
        before creating the new model. You should trust this data.
    deep: Set to `True` to make a deep copy of the model.

Returns:
    New model instance.

### model_dump `classfunction`

```
model_dump(
  self,
  mode: "Literal['json', 'python'] | str" = 'python',
  include: 'IncEx' = None,
  exclude: 'IncEx' = None,
  by_alias: 'bool' = False,
  exclude_unset: 'bool' = False,
  exclude_defaults: 'bool' = False,
  exclude_none: 'bool' = False,
  round_trip: 'bool' = False,
  warnings: 'bool' = True
) -> dict[str, Any]
```

Usage docs: https://docs.pydantic.dev/2.4/concepts/serialization/#modelmodel_dump

Generate a dictionary representation of the model, optionally specifying which fields to include or exclude.

Args:
    mode: The mode in which `to_python` should run.
        If mode is 'json', the dictionary will only contain JSON serializable types.
        If mode is 'python', the dictionary may contain any Python objects.
    include: A list of fields to include in the output.
    exclude: A list of fields to exclude from the output.
    by_alias: Whether to use the field's alias in the dictionary key if defined.
    exclude_unset: Whether to exclude fields that are unset or None from the output.
    exclude_defaults: Whether to exclude fields that are set to their default value from the output.
    exclude_none: Whether to exclude fields that have a value of `None` from the output.
    round_trip: Whether to enable serialization and deserialization round-trip support.
    warnings: Whether to log warnings when invalid fields are encountered.

Returns:
    A dictionary representation of the model.

### model_dump_json `classfunction`

```
model_dump_json(
  self,
  indent: 'int | None' = None,
  include: 'IncEx' = None,
  exclude: 'IncEx' = None,
  by_alias: 'bool' = False,
  exclude_unset: 'bool' = False,
  exclude_defaults: 'bool' = False,
  exclude_none: 'bool' = False,
  round_trip: 'bool' = False,
  warnings: 'bool' = True
) -> str
```

Usage docs: https://docs.pydantic.dev/2.4/concepts/serialization/#modelmodel_dump_json

Generates a JSON representation of the model using Pydantic's `to_json` method.

Args:
    indent: Indentation to use in the JSON output. If None is passed, the output will be compact.
    include: Field(s) to include in the JSON output. Can take either a string or set of strings.
    exclude: Field(s) to exclude from the JSON output. Can take either a string or set of strings.
    by_alias: Whether to serialize using field aliases.
    exclude_unset: Whether to exclude fields that have not been explicitly set.
    exclude_defaults: Whether to exclude fields that have the default value.
    exclude_none: Whether to exclude fields that have a value of `None`.
    round_trip: Whether to use serialization/deserialization between JSON and class instance.
    warnings: Whether to show any warnings that occurred during serialization.

Returns:
    A JSON string representation of the model.

### model_extra `classproperty`

Get extra fields set during validation.

Returns:
    A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`.

### model_fields `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### model_fields_set `classproperty`

Returns the set of fields that have been set on this model instance.

Returns:
    A set of strings representing the fields that have been set,
        i.e. that were not filled from defaults.

### model_json_schema `classmethod`

```
model_json_schema(
  by_alias: 'bool' = True,
  ref_template: 'str' = '#/$defs/{model}',
  schema_generator: 'type[GenerateJsonSchema]' = <class 'pydantic.json_schema.GenerateJsonSchema'>,
  mode: 'JsonSchemaMode' = 'validation'
) -> dict[str, Any]
```

Generates a JSON schema for a model class.

Args:
    by_alias: Whether to use attribute aliases or not.
    ref_template: The reference template.
    schema_generator: To override the logic used to generate the JSON schema, as a subclass of
        `GenerateJsonSchema` with your desired modifications
    mode: The mode in which to generate the schema.

Returns:
    The JSON schema for the given model class.

### model_parametrized_name `classmethod`

```
model_parametrized_name(
  params: 'tuple[type[Any], ...]'
) -> str
```

Compute the class name for parametrizations of generic classes.

This method can be overridden to achieve a custom naming scheme for generic BaseModels.

Args:
    params: Tuple of types of the class. Given a generic class
        `Model` with 2 type variables and a concrete model `Model[str, int]`,
        the value `(str, int)` would be passed to `params`.

Returns:
    String representing the new class where `params` are passed to `cls` as type variables.

Raises:
    TypeError: Raised when trying to generate concrete names for non-generic models.

### model_post_init `classfunction`

```
model_post_init(
  self: 'BaseModel',
  __context: 'Any'
) -> None
```

This function is meant to behave like a BaseModel method to initialise private attributes.

It takes context as an argument since that's what pydantic-core passes when calling it.

Args:
    self: The BaseModel instance.
    __context: The context.

### model_rebuild `classmethod`

```
model_rebuild(
  force: 'bool' = False,
  raise_errors: 'bool' = True,
  _parent_namespace_depth: 'int' = 2,
  _types_namespace: 'dict[str, Any] | None' = None
) -> bool | None
```

Try to rebuild the pydantic-core schema for the model.

This may be necessary when one of the annotations is a ForwardRef which could not be resolved during
the initial attempt to build the schema, and automatic rebuilding fails.

Args:
    force: Whether to force the rebuilding of the model schema, defaults to `False`.
    raise_errors: Whether to raise errors, defaults to `True`.
    _parent_namespace_depth: The depth level of the parent namespace, defaults to 2.
    _types_namespace: The types namespace, defaults to `None`.

Returns:
    Returns `None` if the schema is already "complete" and rebuilding was not required.
    If rebuilding _was_ required, returns `True` if rebuilding was successful, otherwise `False`.

### model_validate `classmethod`

```
model_validate(
  obj: 'Any',
  strict: 'bool | None' = None,
  from_attributes: 'bool | None' = None,
  context: 'dict[str, Any] | None' = None
) -> Model
```

Validate a pydantic model instance.

Args:
    obj: The object to validate.
    strict: Whether to raise an exception on invalid fields.
    from_attributes: Whether to extract data from object attributes.
    context: Additional context to pass to the validator.

Raises:
    ValidationError: If the object could not be validated.

Returns:
    The validated model instance.

### model_validate_json `classmethod`

```
model_validate_json(
  json_data: 'str | bytes | bytearray',
  strict: 'bool | None' = None,
  context: 'dict[str, Any] | None' = None
) -> Model
```

Validate the given JSON data against the Pydantic model.

Args:
    json_data: The JSON data to validate.
    strict: Whether to enforce types strictly.
    context: Extra variables to pass to the validator.

Returns:
    The validated Pydantic model.

Raises:
    ValueError: If `json_data` is not a JSON string.

### model_validate_strings `classmethod`

```
model_validate_strings(
  obj: 'Any',
  strict: 'bool | None' = None,
  context: 'dict[str, Any] | None' = None
) -> Model
```

Validate the given object contains string data against the Pydantic model.

Args:
    obj: The object contains string data to validate.
    strict: Whether to enforce types strictly.
    context: Extra variables to pass to the validator.

Returns:
    The validated Pydantic model.

### parse_file `classmethod`

```
parse_file(
  path: 'str | Path',
  content_type: 'str | None' = None,
  encoding: 'str' = 'utf8',
  proto: 'DeprecatedParseProtocol | None' = None,
  allow_pickle: 'bool' = False
) -> Model
```

### parse_obj `classmethod`

```
parse_obj(
  obj: 'Any'
) -> Model
```

### parse_raw `classmethod`

```
parse_raw(
  b: 'str | bytes',
  content_type: 'str | None' = None,
  encoding: 'str' = 'utf8',
  proto: 'DeprecatedParseProtocol | None' = None,
  allow_pickle: 'bool' = False
) -> Model
```

### parsed_outputs `classproperty`

The outputs from the LLM after undergoing parsing but before
validation.

### prompt `classproperty`

The prompt as provided by the user when intializing or calling the
Guard.

### prompt_params `classproperty`

The prompt parameters as provided by the user when intializing or
calling the Guard.

### prompt_tokens_consumed `classproperty`

Returns the total number of prompt tokens consumed during all
iterations with this call.

### raw_outputs `classproperty`

The exact outputs from all LLM calls.

### reask_instructions `classproperty`

The compiled instructions used during reasks.

Does not include the initial instructions.

### reask_prompts `classproperty`

The compiled prompts used during reasks.

Does not include the initial prompt.

### reasks `classproperty`

Reasks generated during validation that could not be automatically
fixed.

These would be incorporated into the prompt for the next LLM
call if additional reasks were granted.

### schema `classmethod`

```
schema(
  by_alias: 'bool' = True,
  ref_template: 'str' = '#/$defs/{model}'
) -> typing.Dict[str, Any]
```

### schema_json `classmethod`

```
schema_json(
  by_alias: 'bool' = True,
  ref_template: 'str' = '#/$defs/{model}',
  **dumps_kwargs: 'Any'
) -> str
```

### status `classproperty`

Returns the cumulative status of the run based on the validity of
the final merged output.

### tokens_consumed `classproperty`

Returns the total number of tokens consumed during all iterations
with this call.

### tree `classproperty`

Returns the tree.

### update_forward_refs `classmethod`

```
update_forward_refs(
  **localns: 'Any'
) -> None
```

### validate `classmethod`

```
validate(
  value: 'Any'
) -> Model
```

### validated_output `classproperty`

The output from the LLM after undergoing validation.

This will only have a value if the Guard is in a passing state.

### validation_output `classproperty`

The cumulative validation output across all current iterations.

Could contain ReAsks.

### validator_logs `classproperty`

The results of each individual validation performed on the LLM
responses during all iterations.


## CallInputs

Usage docs: https://docs.pydantic.dev/2.4/concepts/models/

A base class for creating Pydantic models.

Attributes:
    __class_vars__: The names of classvars defined on the model.
    __private_attributes__: Metadata about the private attributes of the model.
    __signature__: The signature for instantiating the model.

    __pydantic_complete__: Whether model building is completed, or if there are still undefined fields.
    __pydantic_core_schema__: The pydantic-core schema used to build the SchemaValidator and SchemaSerializer.
    __pydantic_custom_init__: Whether the model has a custom `__init__` function.
    __pydantic_decorators__: Metadata containing the decorators defined on the model.
        This replaces `Model.__validators__` and `Model.__root_validators__` from Pydantic V1.
    __pydantic_generic_metadata__: Metadata for generic models; contains data used for a similar purpose to
        __args__, __origin__, __parameters__ in typing-module generics. May eventually be replaced by these.
    __pydantic_parent_namespace__: Parent namespace of the model, used for automatic rebuilding of models.
    __pydantic_post_init__: The name of the post-init method for the model, if defined.
    __pydantic_root_model__: Whether the model is a `RootModel`.
    __pydantic_serializer__: The pydantic-core SchemaSerializer used to dump instances of the model.
    __pydantic_validator__: The pydantic-core SchemaValidator used to validate instances of the model.

    __pydantic_extra__: An instance attribute with the values of extra fields from validation when
        `model_config['extra'] == 'allow'`.
    __pydantic_fields_set__: An instance attribute with the names of fields explicitly specified during validation.
    __pydantic_private__: Instance attribute with the values of private attributes set on the model instance.

### __abstractmethods__ `classfrozenset`

frozenset() -> empty frozenset object
frozenset(iterable) -> frozenset object

Build an immutable unordered collection of unique elements.

### __annotations__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __class_getitem__ `classmethod`

```
__class_getitem__(
  typevar_values: 'type[Any] | tuple[type[Any], ...]'
) -> type[BaseModel] | _forward_ref.PydanticRecursiveRef
```

### __class_vars__ `classset`

set() -> new empty set object
set(iterable) -> new set object

Build an unordered collection of unique elements.

### __copy__ `classfunction`

```
__copy__(
  self: 'Model'
) -> Model
```

Returns a shallow copy of the model.

### __deepcopy__ `classfunction`

```
__deepcopy__(
  self: 'Model',
  memo: 'dict[int, Any] | None' = None
) -> Model
```

Returns a deep copy of the model.

### __delattr__ `classfunction`

```
__delattr__(
  self,
  item: 'str'
) -> Any
```

Implement delattr(self, name).

### __dict__ `classmappingproxy`

### __dir__ `classmethod_descriptor`

```
__dir__(
  self
)
```

Default dir() implementation.

### __doc__ `classNoneType`

### __eq__ `classfunction`

```
__eq__(
  self,
  other: 'Any'
) -> bool
```

Return self==value.

### __fields__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __fields_set__ `classproperty`

### __format__ `classmethod_descriptor`

```
__format__(
  self,
  format_spec
)
```

Default object formatter.

### __ge__ `classwrapper_descriptor`

```
__ge__(
  self,
  value
)
```

Return self>=value.

### __get_pydantic_core_schema__ `classmethod`

```
__get_pydantic_core_schema__(
  _BaseModel__source: 'type[BaseModel]',
  _BaseModel__handler: 'GetCoreSchemaHandler'
) -> CoreSchema
```

Hook into generating the model's CoreSchema.

Args:
    __source: The class we are generating a schema for.
        This will generally be the same as the `cls` argument if this is a classmethod.
    __handler: Call into Pydantic's internal JSON schema generation.
        A callable that calls into Pydantic's internal CoreSchema generation logic.

Returns:
    A `pydantic-core` `CoreSchema`.

### __get_pydantic_json_schema__ `classmethod`

```
__get_pydantic_json_schema__(
  _BaseModel__core_schema: 'CoreSchema',
  _BaseModel__handler: 'GetJsonSchemaHandler'
) -> JsonSchemaValue
```

Hook into generating the model's JSON schema.

Args:
    __core_schema: A `pydantic-core` CoreSchema.
        You can ignore this argument and call the handler with a new CoreSchema,
        wrap this CoreSchema (`{'type': 'nullable', 'schema': current_schema}`),
        or just call the handler with the original schema.
    __handler: Call into Pydantic's internal JSON schema generation.
        This will raise a `pydantic.errors.PydanticInvalidForJsonSchema` if JSON schema
        generation fails.
        Since this gets called by `BaseModel.model_json_schema` you can override the
        `schema_generator` argument to that function to change JSON schema generation globally
        for a type.

Returns:
    A JSON schema, as a Python object.

### __getattr__ `classfunction`

```
__getattr__(
  self,
  item: 'str'
) -> Any
```

### __getattribute__ `classwrapper_descriptor`

```
__getattribute__(
  self,
  name
)
```

Return getattr(self, name).

### __getstate__ `classfunction`

```
__getstate__(
  self
) -> dict[Any, Any]
```

### __gt__ `classwrapper_descriptor`

```
__gt__(
  self,
  value
)
```

Return self>value.

### __hash__ `classNoneType`

### __init__ `classfunction`

```
__init__(
  __pydantic_self__,
  **data: 'Any'
) -> None
```

Create a new model by parsing and validating input data from keyword arguments.

Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
validated to form a valid model.

`__init__` uses `__pydantic_self__` instead of the more common `self` for the first arg to
allow `self` as a field name.

### __init_subclass__ `classbuiltin_function_or_method`

```
__init_subclass__
```

This method is called when a class is subclassed.

The default implementation does nothing. It may be
overridden to extend subclasses.

### __iter__ `classfunction`

```
__iter__(
  self
) -> TupleGenerator
```

So `dict(model)` works.

### __le__ `classwrapper_descriptor`

```
__le__(
  self,
  value
)
```

Return self<=value.

### __lt__ `classwrapper_descriptor`

```
__lt__(
  self,
  value
)
```

Return self<value.

### __module__ `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### __ne__ `classwrapper_descriptor`

```
__ne__(
  self,
  value
)
```

Return self!=value.

### __new__ `classbuiltin_function_or_method`

```
__new__(
  *args,
  **kwargs
)
```

Create and return a new object.  See help(type) for accurate signature.

### __pretty__ `classfunction`

```
__pretty__(
  self,
  fmt: 'typing.Callable[[Any], Any]',
  **kwargs: 'Any'
) -> typing.Generator[Any, None, None]
```

Used by devtools (https://python-devtools.helpmanual.io/) to pretty print objects.

### __private_attributes__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __pydantic_complete__ `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### __pydantic_core_schema__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __pydantic_custom_init__ `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### __pydantic_decorators__ `classpydantic._internal._decorators.DecoratorInfos`

Mapping of name in the class namespace to decorator info.

note that the name in the class namespace is the function or attribute name
not the field name!

### __pydantic_extra__ `classmember_descriptor`

### __pydantic_fields_set__ `classmember_descriptor`

### __pydantic_generic_metadata__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __pydantic_init_subclass__ `classmethod`

```
__pydantic_init_subclass__(
  **kwargs: 'Any'
) -> None
```

This is intended to behave just like `__init_subclass__`, but is called by `ModelMetaclass`
only after the class is actually fully initialized. In particular, attributes like `model_fields` will
be present when this is called.

This is necessary because `__init_subclass__` will always be called by `type.__new__`,
and it would require a prohibitively large refactor to the `ModelMetaclass` to ensure that
`type.__new__` was called in such a manner that the class would already be sufficiently initialized.

This will receive the same `kwargs` that would be passed to the standard `__init_subclass__`, namely,
any kwargs passed to the class definition that aren't used internally by pydantic.

Args:
    **kwargs: Any keyword arguments passed to the class definition that aren't used internally
        by pydantic.

### __pydantic_parent_namespace__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __pydantic_post_init__ `classNoneType`

### __pydantic_private__ `classmember_descriptor`

### __pydantic_root_model__ `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### __pydantic_serializer__ `classpydantic_core._pydantic_core.SchemaSerializer`

### __pydantic_validator__ `classpydantic_core._pydantic_core.SchemaValidator`

### __reduce__ `classmethod_descriptor`

```
__reduce__(
  self
)
```

Helper for pickle.

### __reduce_ex__ `classmethod_descriptor`

```
__reduce_ex__(
  self,
  protocol
)
```

Helper for pickle.

### __repr__ `classfunction`

```
__repr__(
  self
) -> str
```

Return repr(self).

### __repr_args__ `classfunction`

```
__repr_args__(
  self
) -> _repr.ReprArgs
```

### __repr_name__ `classfunction`

```
__repr_name__(
  self
) -> str
```

Name of the instance's class, used in __repr__.

### __repr_str__ `classfunction`

```
__repr_str__(
  self,
  join_str: 'str'
) -> str
```

### __rich_repr__ `classfunction`

```
__rich_repr__(
  self
) -> RichReprResult
```

Used by Rich (https://rich.readthedocs.io/en/stable/pretty.html) to pretty print objects.

### __setattr__ `classfunction`

```
__setattr__(
  self,
  name: 'str',
  value: 'Any'
) -> None
```

Implement setattr(self, name, value).

### __setstate__ `classfunction`

```
__setstate__(
  self,
  state: 'dict[Any, Any]'
) -> None
```

### __signature__ `classinspect.Signature`

A Signature object represents the overall signature of a function.
It stores a Parameter object for each parameter accepted by the
function, as well as information specific to the function itself.

A Signature object has the following public attributes and methods:

* parameters : OrderedDict
    An ordered mapping of parameters' names to the corresponding
    Parameter objects (keyword-only arguments are in the same order
    as listed in `code.co_varnames`).
* return_annotation : object
    The annotation for the return type of the function if specified.
    If the function has no annotation for its return type, this
    attribute is set to `Signature.empty`.
* bind(*args, **kwargs) -> BoundArguments
    Creates a mapping from positional and keyword arguments to
    parameters.
* bind_partial(*args, **kwargs) -> BoundArguments
    Creates a partial mapping from positional and keyword arguments
    to parameters (simulating 'functools.partial' behavior.)

### __sizeof__ `classmethod_descriptor`

```
__sizeof__(
  self
)
```

Size of object in memory, in bytes.

### __slots__ `classtuple`

Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.

### __str__ `classfunction`

```
__str__(
  self
) -> str
```

Return str(self).

### __subclasshook__ `classbuiltin_function_or_method`

```
__subclasshook__
```

Abstract classes can override this to customize issubclass().

This is invoked early on by abc.ABCMeta.__subclasscheck__().
It should return True, False or NotImplemented.  If it returns
NotImplemented, the normal algorithm is used.  Otherwise, it
overrides the normal algorithm (and the outcome is cached).

### __weakref__ `classgetset_descriptor`

list of weak references to the object (if defined)

### _abc_impl `class_abc._abc_data`

Internal state held by ABC machinery.

### _calculate_keys `classfunction`

```
_calculate_keys(
  self,
  *args: 'Any',
  **kwargs: 'Any'
) -> Any
```

### _copy_and_set_values `classfunction`

```
_copy_and_set_values(
  self,
  *args: 'Any',
  **kwargs: 'Any'
) -> Any
```

### _get_value `classmethod`

```
_get_value(
  *args: 'Any',
  **kwargs: 'Any'
) -> Any
```

### _iter `classfunction`

```
_iter(
  self,
  *args: 'Any',
  **kwargs: 'Any'
) -> Any
```

### construct `classmethod`

```
construct(
  _fields_set: 'set[str] | None' = None,
  **values: 'Any'
) -> Model
```

### copy `classfunction`

```
copy(
  self: 'Model',
  include: 'AbstractSetIntStr | MappingIntStrAny | None' = None,
  exclude: 'AbstractSetIntStr | MappingIntStrAny | None' = None,
  update: 'typing.Dict[str, Any] | None' = None,
  deep: 'bool' = False
) -> Model
```

Returns a copy of the model.

!!! warning "Deprecated"
    This method is now deprecated; use `model_copy` instead.

If you need `include` or `exclude`, use:

```py
data = self.model_dump(include=include, exclude=exclude, round_trip=True)
data = {**data, **(update or {})}
copied = self.model_validate(data)
```

Args:
    include: Optional set or mapping
        specifying which fields to include in the copied model.
    exclude: Optional set or mapping
        specifying which fields to exclude in the copied model.
    update: Optional dictionary of field-value pairs to override field values
        in the copied model.
    deep: If True, the values of fields that are Pydantic models will be deep copied.

Returns:
    A copy of the model with included, excluded and updated fields as specified.

### dict `classfunction`

```
dict(
  self,
  include: 'IncEx' = None,
  exclude: 'IncEx' = None,
  by_alias: 'bool' = False,
  exclude_unset: 'bool' = False,
  exclude_defaults: 'bool' = False,
  exclude_none: 'bool' = False
) -> typing.Dict[str, Any]
```

### from_orm `classmethod`

```
from_orm(
  obj: 'Any'
) -> Model
```

### json `classfunction`

```
json(
  self,
  include: 'IncEx' = None,
  exclude: 'IncEx' = None,
  by_alias: 'bool' = False,
  exclude_unset: 'bool' = False,
  exclude_defaults: 'bool' = False,
  exclude_none: 'bool' = False,
  encoder: 'typing.Callable[[Any], Any] | None' = PydanticUndefined,
  models_as_dict: 'bool' = PydanticUndefined,
  **dumps_kwargs: 'Any'
) -> str
```

### model_computed_fields `classproperty`

Get the computed fields of this model instance.

Returns:
    A dictionary of computed field names and their corresponding `ComputedFieldInfo` objects.

### model_config `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### model_construct `classmethod`

```
model_construct(
  _fields_set: 'set[str] | None' = None,
  **values: 'Any'
) -> Model
```

Creates a new instance of the `Model` class with validated data.

Creates a new model setting `__dict__` and `__pydantic_fields_set__` from trusted or pre-validated data.
Default values are respected, but no other validation is performed.
Behaves as if `Config.extra = 'allow'` was set since it adds all passed values

Args:
    _fields_set: The set of field names accepted for the Model instance.
    values: Trusted or pre-validated data dictionary.

Returns:
    A new instance of the `Model` class with validated data.

### model_copy `classfunction`

```
model_copy(
  self: 'Model',
  update: 'dict[str, Any] | None' = None,
  deep: 'bool' = False
) -> Model
```

Usage docs: https://docs.pydantic.dev/2.4/concepts/serialization/#model_copy

Returns a copy of the model.

Args:
    update: Values to change/add in the new model. Note: the data is not validated
        before creating the new model. You should trust this data.
    deep: Set to `True` to make a deep copy of the model.

Returns:
    New model instance.

### model_dump `classfunction`

```
model_dump(
  self,
  mode: "Literal['json', 'python'] | str" = 'python',
  include: 'IncEx' = None,
  exclude: 'IncEx' = None,
  by_alias: 'bool' = False,
  exclude_unset: 'bool' = False,
  exclude_defaults: 'bool' = False,
  exclude_none: 'bool' = False,
  round_trip: 'bool' = False,
  warnings: 'bool' = True
) -> dict[str, Any]
```

Usage docs: https://docs.pydantic.dev/2.4/concepts/serialization/#modelmodel_dump

Generate a dictionary representation of the model, optionally specifying which fields to include or exclude.

Args:
    mode: The mode in which `to_python` should run.
        If mode is 'json', the dictionary will only contain JSON serializable types.
        If mode is 'python', the dictionary may contain any Python objects.
    include: A list of fields to include in the output.
    exclude: A list of fields to exclude from the output.
    by_alias: Whether to use the field's alias in the dictionary key if defined.
    exclude_unset: Whether to exclude fields that are unset or None from the output.
    exclude_defaults: Whether to exclude fields that are set to their default value from the output.
    exclude_none: Whether to exclude fields that have a value of `None` from the output.
    round_trip: Whether to enable serialization and deserialization round-trip support.
    warnings: Whether to log warnings when invalid fields are encountered.

Returns:
    A dictionary representation of the model.

### model_dump_json `classfunction`

```
model_dump_json(
  self,
  indent: 'int | None' = None,
  include: 'IncEx' = None,
  exclude: 'IncEx' = None,
  by_alias: 'bool' = False,
  exclude_unset: 'bool' = False,
  exclude_defaults: 'bool' = False,
  exclude_none: 'bool' = False,
  round_trip: 'bool' = False,
  warnings: 'bool' = True
) -> str
```

Usage docs: https://docs.pydantic.dev/2.4/concepts/serialization/#modelmodel_dump_json

Generates a JSON representation of the model using Pydantic's `to_json` method.

Args:
    indent: Indentation to use in the JSON output. If None is passed, the output will be compact.
    include: Field(s) to include in the JSON output. Can take either a string or set of strings.
    exclude: Field(s) to exclude from the JSON output. Can take either a string or set of strings.
    by_alias: Whether to serialize using field aliases.
    exclude_unset: Whether to exclude fields that have not been explicitly set.
    exclude_defaults: Whether to exclude fields that have the default value.
    exclude_none: Whether to exclude fields that have a value of `None`.
    round_trip: Whether to use serialization/deserialization between JSON and class instance.
    warnings: Whether to show any warnings that occurred during serialization.

Returns:
    A JSON string representation of the model.

### model_extra `classproperty`

Get extra fields set during validation.

Returns:
    A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`.

### model_fields `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### model_fields_set `classproperty`

Returns the set of fields that have been set on this model instance.

Returns:
    A set of strings representing the fields that have been set,
        i.e. that were not filled from defaults.

### model_json_schema `classmethod`

```
model_json_schema(
  by_alias: 'bool' = True,
  ref_template: 'str' = '#/$defs/{model}',
  schema_generator: 'type[GenerateJsonSchema]' = <class 'pydantic.json_schema.GenerateJsonSchema'>,
  mode: 'JsonSchemaMode' = 'validation'
) -> dict[str, Any]
```

Generates a JSON schema for a model class.

Args:
    by_alias: Whether to use attribute aliases or not.
    ref_template: The reference template.
    schema_generator: To override the logic used to generate the JSON schema, as a subclass of
        `GenerateJsonSchema` with your desired modifications
    mode: The mode in which to generate the schema.

Returns:
    The JSON schema for the given model class.

### model_parametrized_name `classmethod`

```
model_parametrized_name(
  params: 'tuple[type[Any], ...]'
) -> str
```

Compute the class name for parametrizations of generic classes.

This method can be overridden to achieve a custom naming scheme for generic BaseModels.

Args:
    params: Tuple of types of the class. Given a generic class
        `Model` with 2 type variables and a concrete model `Model[str, int]`,
        the value `(str, int)` would be passed to `params`.

Returns:
    String representing the new class where `params` are passed to `cls` as type variables.

Raises:
    TypeError: Raised when trying to generate concrete names for non-generic models.

### model_post_init `classfunction`

```
model_post_init(
  self,
  _BaseModel__context: 'Any'
) -> None
```

Override this method to perform additional initialization after `__init__` and `model_construct`.
This is useful if you want to do some validation that requires the entire model to be initialized.

### model_rebuild `classmethod`

```
model_rebuild(
  force: 'bool' = False,
  raise_errors: 'bool' = True,
  _parent_namespace_depth: 'int' = 2,
  _types_namespace: 'dict[str, Any] | None' = None
) -> bool | None
```

Try to rebuild the pydantic-core schema for the model.

This may be necessary when one of the annotations is a ForwardRef which could not be resolved during
the initial attempt to build the schema, and automatic rebuilding fails.

Args:
    force: Whether to force the rebuilding of the model schema, defaults to `False`.
    raise_errors: Whether to raise errors, defaults to `True`.
    _parent_namespace_depth: The depth level of the parent namespace, defaults to 2.
    _types_namespace: The types namespace, defaults to `None`.

Returns:
    Returns `None` if the schema is already "complete" and rebuilding was not required.
    If rebuilding _was_ required, returns `True` if rebuilding was successful, otherwise `False`.

### model_validate `classmethod`

```
model_validate(
  obj: 'Any',
  strict: 'bool | None' = None,
  from_attributes: 'bool | None' = None,
  context: 'dict[str, Any] | None' = None
) -> Model
```

Validate a pydantic model instance.

Args:
    obj: The object to validate.
    strict: Whether to raise an exception on invalid fields.
    from_attributes: Whether to extract data from object attributes.
    context: Additional context to pass to the validator.

Raises:
    ValidationError: If the object could not be validated.

Returns:
    The validated model instance.

### model_validate_json `classmethod`

```
model_validate_json(
  json_data: 'str | bytes | bytearray',
  strict: 'bool | None' = None,
  context: 'dict[str, Any] | None' = None
) -> Model
```

Validate the given JSON data against the Pydantic model.

Args:
    json_data: The JSON data to validate.
    strict: Whether to enforce types strictly.
    context: Extra variables to pass to the validator.

Returns:
    The validated Pydantic model.

Raises:
    ValueError: If `json_data` is not a JSON string.

### model_validate_strings `classmethod`

```
model_validate_strings(
  obj: 'Any',
  strict: 'bool | None' = None,
  context: 'dict[str, Any] | None' = None
) -> Model
```

Validate the given object contains string data against the Pydantic model.

Args:
    obj: The object contains string data to validate.
    strict: Whether to enforce types strictly.
    context: Extra variables to pass to the validator.

Returns:
    The validated Pydantic model.

### parse_file `classmethod`

```
parse_file(
  path: 'str | Path',
  content_type: 'str | None' = None,
  encoding: 'str' = 'utf8',
  proto: 'DeprecatedParseProtocol | None' = None,
  allow_pickle: 'bool' = False
) -> Model
```

### parse_obj `classmethod`

```
parse_obj(
  obj: 'Any'
) -> Model
```

### parse_raw `classmethod`

```
parse_raw(
  b: 'str | bytes',
  content_type: 'str | None' = None,
  encoding: 'str' = 'utf8',
  proto: 'DeprecatedParseProtocol | None' = None,
  allow_pickle: 'bool' = False
) -> Model
```

### schema `classmethod`

```
schema(
  by_alias: 'bool' = True,
  ref_template: 'str' = '#/$defs/{model}'
) -> typing.Dict[str, Any]
```

### schema_json `classmethod`

```
schema_json(
  by_alias: 'bool' = True,
  ref_template: 'str' = '#/$defs/{model}',
  **dumps_kwargs: 'Any'
) -> str
```

### update_forward_refs `classmethod`

```
update_forward_refs(
  **localns: 'Any'
) -> None
```

### validate `classmethod`

```
validate(
  value: 'Any'
) -> Model
```


## Inputs

Usage docs: https://docs.pydantic.dev/2.4/concepts/models/

A base class for creating Pydantic models.

Attributes:
    __class_vars__: The names of classvars defined on the model.
    __private_attributes__: Metadata about the private attributes of the model.
    __signature__: The signature for instantiating the model.

    __pydantic_complete__: Whether model building is completed, or if there are still undefined fields.
    __pydantic_core_schema__: The pydantic-core schema used to build the SchemaValidator and SchemaSerializer.
    __pydantic_custom_init__: Whether the model has a custom `__init__` function.
    __pydantic_decorators__: Metadata containing the decorators defined on the model.
        This replaces `Model.__validators__` and `Model.__root_validators__` from Pydantic V1.
    __pydantic_generic_metadata__: Metadata for generic models; contains data used for a similar purpose to
        __args__, __origin__, __parameters__ in typing-module generics. May eventually be replaced by these.
    __pydantic_parent_namespace__: Parent namespace of the model, used for automatic rebuilding of models.
    __pydantic_post_init__: The name of the post-init method for the model, if defined.
    __pydantic_root_model__: Whether the model is a `RootModel`.
    __pydantic_serializer__: The pydantic-core SchemaSerializer used to dump instances of the model.
    __pydantic_validator__: The pydantic-core SchemaValidator used to validate instances of the model.

    __pydantic_extra__: An instance attribute with the values of extra fields from validation when
        `model_config['extra'] == 'allow'`.
    __pydantic_fields_set__: An instance attribute with the names of fields explicitly specified during validation.
    __pydantic_private__: Instance attribute with the values of private attributes set on the model instance.

### __abstractmethods__ `classfrozenset`

frozenset() -> empty frozenset object
frozenset(iterable) -> frozenset object

Build an immutable unordered collection of unique elements.

### __annotations__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __class_getitem__ `classmethod`

```
__class_getitem__(
  typevar_values: 'type[Any] | tuple[type[Any], ...]'
) -> type[BaseModel] | _forward_ref.PydanticRecursiveRef
```

### __class_vars__ `classset`

set() -> new empty set object
set(iterable) -> new set object

Build an unordered collection of unique elements.

### __copy__ `classfunction`

```
__copy__(
  self: 'Model'
) -> Model
```

Returns a shallow copy of the model.

### __deepcopy__ `classfunction`

```
__deepcopy__(
  self: 'Model',
  memo: 'dict[int, Any] | None' = None
) -> Model
```

Returns a deep copy of the model.

### __delattr__ `classfunction`

```
__delattr__(
  self,
  item: 'str'
) -> Any
```

Implement delattr(self, name).

### __dict__ `classmappingproxy`

### __dir__ `classmethod_descriptor`

```
__dir__(
  self
)
```

Default dir() implementation.

### __doc__ `classNoneType`

### __eq__ `classfunction`

```
__eq__(
  self,
  other: 'Any'
) -> bool
```

Return self==value.

### __fields__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __fields_set__ `classproperty`

### __format__ `classmethod_descriptor`

```
__format__(
  self,
  format_spec
)
```

Default object formatter.

### __ge__ `classwrapper_descriptor`

```
__ge__(
  self,
  value
)
```

Return self>=value.

### __get_pydantic_core_schema__ `classmethod`

```
__get_pydantic_core_schema__(
  _BaseModel__source: 'type[BaseModel]',
  _BaseModel__handler: 'GetCoreSchemaHandler'
) -> CoreSchema
```

Hook into generating the model's CoreSchema.

Args:
    __source: The class we are generating a schema for.
        This will generally be the same as the `cls` argument if this is a classmethod.
    __handler: Call into Pydantic's internal JSON schema generation.
        A callable that calls into Pydantic's internal CoreSchema generation logic.

Returns:
    A `pydantic-core` `CoreSchema`.

### __get_pydantic_json_schema__ `classmethod`

```
__get_pydantic_json_schema__(
  _BaseModel__core_schema: 'CoreSchema',
  _BaseModel__handler: 'GetJsonSchemaHandler'
) -> JsonSchemaValue
```

Hook into generating the model's JSON schema.

Args:
    __core_schema: A `pydantic-core` CoreSchema.
        You can ignore this argument and call the handler with a new CoreSchema,
        wrap this CoreSchema (`{'type': 'nullable', 'schema': current_schema}`),
        or just call the handler with the original schema.
    __handler: Call into Pydantic's internal JSON schema generation.
        This will raise a `pydantic.errors.PydanticInvalidForJsonSchema` if JSON schema
        generation fails.
        Since this gets called by `BaseModel.model_json_schema` you can override the
        `schema_generator` argument to that function to change JSON schema generation globally
        for a type.

Returns:
    A JSON schema, as a Python object.

### __getattr__ `classfunction`

```
__getattr__(
  self,
  item: 'str'
) -> Any
```

### __getattribute__ `classwrapper_descriptor`

```
__getattribute__(
  self,
  name
)
```

Return getattr(self, name).

### __getstate__ `classfunction`

```
__getstate__(
  self
) -> dict[Any, Any]
```

### __gt__ `classwrapper_descriptor`

```
__gt__(
  self,
  value
)
```

Return self>value.

### __hash__ `classNoneType`

### __init__ `classfunction`

```
__init__(
  __pydantic_self__,
  **data: 'Any'
) -> None
```

Create a new model by parsing and validating input data from keyword arguments.

Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
validated to form a valid model.

`__init__` uses `__pydantic_self__` instead of the more common `self` for the first arg to
allow `self` as a field name.

### __init_subclass__ `classbuiltin_function_or_method`

```
__init_subclass__
```

This method is called when a class is subclassed.

The default implementation does nothing. It may be
overridden to extend subclasses.

### __iter__ `classfunction`

```
__iter__(
  self
) -> TupleGenerator
```

So `dict(model)` works.

### __le__ `classwrapper_descriptor`

```
__le__(
  self,
  value
)
```

Return self<=value.

### __lt__ `classwrapper_descriptor`

```
__lt__(
  self,
  value
)
```

Return self<value.

### __module__ `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### __ne__ `classwrapper_descriptor`

```
__ne__(
  self,
  value
)
```

Return self!=value.

### __new__ `classbuiltin_function_or_method`

```
__new__(
  *args,
  **kwargs
)
```

Create and return a new object.  See help(type) for accurate signature.

### __pretty__ `classfunction`

```
__pretty__(
  self,
  fmt: 'typing.Callable[[Any], Any]',
  **kwargs: 'Any'
) -> typing.Generator[Any, None, None]
```

Used by devtools (https://python-devtools.helpmanual.io/) to pretty print objects.

### __private_attributes__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __pydantic_complete__ `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### __pydantic_core_schema__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __pydantic_custom_init__ `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### __pydantic_decorators__ `classpydantic._internal._decorators.DecoratorInfos`

Mapping of name in the class namespace to decorator info.

note that the name in the class namespace is the function or attribute name
not the field name!

### __pydantic_extra__ `classmember_descriptor`

### __pydantic_fields_set__ `classmember_descriptor`

### __pydantic_generic_metadata__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __pydantic_init_subclass__ `classmethod`

```
__pydantic_init_subclass__(
  **kwargs: 'Any'
) -> None
```

This is intended to behave just like `__init_subclass__`, but is called by `ModelMetaclass`
only after the class is actually fully initialized. In particular, attributes like `model_fields` will
be present when this is called.

This is necessary because `__init_subclass__` will always be called by `type.__new__`,
and it would require a prohibitively large refactor to the `ModelMetaclass` to ensure that
`type.__new__` was called in such a manner that the class would already be sufficiently initialized.

This will receive the same `kwargs` that would be passed to the standard `__init_subclass__`, namely,
any kwargs passed to the class definition that aren't used internally by pydantic.

Args:
    **kwargs: Any keyword arguments passed to the class definition that aren't used internally
        by pydantic.

### __pydantic_parent_namespace__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __pydantic_post_init__ `classNoneType`

### __pydantic_private__ `classmember_descriptor`

### __pydantic_root_model__ `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### __pydantic_serializer__ `classpydantic_core._pydantic_core.SchemaSerializer`

### __pydantic_validator__ `classpydantic_core._pydantic_core.SchemaValidator`

### __reduce__ `classmethod_descriptor`

```
__reduce__(
  self
)
```

Helper for pickle.

### __reduce_ex__ `classmethod_descriptor`

```
__reduce_ex__(
  self,
  protocol
)
```

Helper for pickle.

### __repr__ `classfunction`

```
__repr__(
  self
) -> str
```

Return repr(self).

### __repr_args__ `classfunction`

```
__repr_args__(
  self
) -> _repr.ReprArgs
```

### __repr_name__ `classfunction`

```
__repr_name__(
  self
) -> str
```

Name of the instance's class, used in __repr__.

### __repr_str__ `classfunction`

```
__repr_str__(
  self,
  join_str: 'str'
) -> str
```

### __rich_repr__ `classfunction`

```
__rich_repr__(
  self
) -> RichReprResult
```

Used by Rich (https://rich.readthedocs.io/en/stable/pretty.html) to pretty print objects.

### __setattr__ `classfunction`

```
__setattr__(
  self,
  name: 'str',
  value: 'Any'
) -> None
```

Implement setattr(self, name, value).

### __setstate__ `classfunction`

```
__setstate__(
  self,
  state: 'dict[Any, Any]'
) -> None
```

### __signature__ `classinspect.Signature`

A Signature object represents the overall signature of a function.
It stores a Parameter object for each parameter accepted by the
function, as well as information specific to the function itself.

A Signature object has the following public attributes and methods:

* parameters : OrderedDict
    An ordered mapping of parameters' names to the corresponding
    Parameter objects (keyword-only arguments are in the same order
    as listed in `code.co_varnames`).
* return_annotation : object
    The annotation for the return type of the function if specified.
    If the function has no annotation for its return type, this
    attribute is set to `Signature.empty`.
* bind(*args, **kwargs) -> BoundArguments
    Creates a mapping from positional and keyword arguments to
    parameters.
* bind_partial(*args, **kwargs) -> BoundArguments
    Creates a partial mapping from positional and keyword arguments
    to parameters (simulating 'functools.partial' behavior.)

### __sizeof__ `classmethod_descriptor`

```
__sizeof__(
  self
)
```

Size of object in memory, in bytes.

### __slots__ `classtuple`

Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.

### __str__ `classfunction`

```
__str__(
  self
) -> str
```

Return str(self).

### __subclasshook__ `classbuiltin_function_or_method`

```
__subclasshook__
```

Abstract classes can override this to customize issubclass().

This is invoked early on by abc.ABCMeta.__subclasscheck__().
It should return True, False or NotImplemented.  If it returns
NotImplemented, the normal algorithm is used.  Otherwise, it
overrides the normal algorithm (and the outcome is cached).

### __weakref__ `classgetset_descriptor`

list of weak references to the object (if defined)

### _abc_impl `class_abc._abc_data`

Internal state held by ABC machinery.

### _calculate_keys `classfunction`

```
_calculate_keys(
  self,
  *args: 'Any',
  **kwargs: 'Any'
) -> Any
```

### _copy_and_set_values `classfunction`

```
_copy_and_set_values(
  self,
  *args: 'Any',
  **kwargs: 'Any'
) -> Any
```

### _get_value `classmethod`

```
_get_value(
  *args: 'Any',
  **kwargs: 'Any'
) -> Any
```

### _iter `classfunction`

```
_iter(
  self,
  *args: 'Any',
  **kwargs: 'Any'
) -> Any
```

### construct `classmethod`

```
construct(
  _fields_set: 'set[str] | None' = None,
  **values: 'Any'
) -> Model
```

### copy `classfunction`

```
copy(
  self: 'Model',
  include: 'AbstractSetIntStr | MappingIntStrAny | None' = None,
  exclude: 'AbstractSetIntStr | MappingIntStrAny | None' = None,
  update: 'typing.Dict[str, Any] | None' = None,
  deep: 'bool' = False
) -> Model
```

Returns a copy of the model.

!!! warning "Deprecated"
    This method is now deprecated; use `model_copy` instead.

If you need `include` or `exclude`, use:

```py
data = self.model_dump(include=include, exclude=exclude, round_trip=True)
data = {**data, **(update or {})}
copied = self.model_validate(data)
```

Args:
    include: Optional set or mapping
        specifying which fields to include in the copied model.
    exclude: Optional set or mapping
        specifying which fields to exclude in the copied model.
    update: Optional dictionary of field-value pairs to override field values
        in the copied model.
    deep: If True, the values of fields that are Pydantic models will be deep copied.

Returns:
    A copy of the model with included, excluded and updated fields as specified.

### dict `classfunction`

```
dict(
  self,
  include: 'IncEx' = None,
  exclude: 'IncEx' = None,
  by_alias: 'bool' = False,
  exclude_unset: 'bool' = False,
  exclude_defaults: 'bool' = False,
  exclude_none: 'bool' = False
) -> typing.Dict[str, Any]
```

### from_orm `classmethod`

```
from_orm(
  obj: 'Any'
) -> Model
```

### json `classfunction`

```
json(
  self,
  include: 'IncEx' = None,
  exclude: 'IncEx' = None,
  by_alias: 'bool' = False,
  exclude_unset: 'bool' = False,
  exclude_defaults: 'bool' = False,
  exclude_none: 'bool' = False,
  encoder: 'typing.Callable[[Any], Any] | None' = PydanticUndefined,
  models_as_dict: 'bool' = PydanticUndefined,
  **dumps_kwargs: 'Any'
) -> str
```

### model_computed_fields `classproperty`

Get the computed fields of this model instance.

Returns:
    A dictionary of computed field names and their corresponding `ComputedFieldInfo` objects.

### model_config `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### model_construct `classmethod`

```
model_construct(
  _fields_set: 'set[str] | None' = None,
  **values: 'Any'
) -> Model
```

Creates a new instance of the `Model` class with validated data.

Creates a new model setting `__dict__` and `__pydantic_fields_set__` from trusted or pre-validated data.
Default values are respected, but no other validation is performed.
Behaves as if `Config.extra = 'allow'` was set since it adds all passed values

Args:
    _fields_set: The set of field names accepted for the Model instance.
    values: Trusted or pre-validated data dictionary.

Returns:
    A new instance of the `Model` class with validated data.

### model_copy `classfunction`

```
model_copy(
  self: 'Model',
  update: 'dict[str, Any] | None' = None,
  deep: 'bool' = False
) -> Model
```

Usage docs: https://docs.pydantic.dev/2.4/concepts/serialization/#model_copy

Returns a copy of the model.

Args:
    update: Values to change/add in the new model. Note: the data is not validated
        before creating the new model. You should trust this data.
    deep: Set to `True` to make a deep copy of the model.

Returns:
    New model instance.

### model_dump `classfunction`

```
model_dump(
  self,
  mode: "Literal['json', 'python'] | str" = 'python',
  include: 'IncEx' = None,
  exclude: 'IncEx' = None,
  by_alias: 'bool' = False,
  exclude_unset: 'bool' = False,
  exclude_defaults: 'bool' = False,
  exclude_none: 'bool' = False,
  round_trip: 'bool' = False,
  warnings: 'bool' = True
) -> dict[str, Any]
```

Usage docs: https://docs.pydantic.dev/2.4/concepts/serialization/#modelmodel_dump

Generate a dictionary representation of the model, optionally specifying which fields to include or exclude.

Args:
    mode: The mode in which `to_python` should run.
        If mode is 'json', the dictionary will only contain JSON serializable types.
        If mode is 'python', the dictionary may contain any Python objects.
    include: A list of fields to include in the output.
    exclude: A list of fields to exclude from the output.
    by_alias: Whether to use the field's alias in the dictionary key if defined.
    exclude_unset: Whether to exclude fields that are unset or None from the output.
    exclude_defaults: Whether to exclude fields that are set to their default value from the output.
    exclude_none: Whether to exclude fields that have a value of `None` from the output.
    round_trip: Whether to enable serialization and deserialization round-trip support.
    warnings: Whether to log warnings when invalid fields are encountered.

Returns:
    A dictionary representation of the model.

### model_dump_json `classfunction`

```
model_dump_json(
  self,
  indent: 'int | None' = None,
  include: 'IncEx' = None,
  exclude: 'IncEx' = None,
  by_alias: 'bool' = False,
  exclude_unset: 'bool' = False,
  exclude_defaults: 'bool' = False,
  exclude_none: 'bool' = False,
  round_trip: 'bool' = False,
  warnings: 'bool' = True
) -> str
```

Usage docs: https://docs.pydantic.dev/2.4/concepts/serialization/#modelmodel_dump_json

Generates a JSON representation of the model using Pydantic's `to_json` method.

Args:
    indent: Indentation to use in the JSON output. If None is passed, the output will be compact.
    include: Field(s) to include in the JSON output. Can take either a string or set of strings.
    exclude: Field(s) to exclude from the JSON output. Can take either a string or set of strings.
    by_alias: Whether to serialize using field aliases.
    exclude_unset: Whether to exclude fields that have not been explicitly set.
    exclude_defaults: Whether to exclude fields that have the default value.
    exclude_none: Whether to exclude fields that have a value of `None`.
    round_trip: Whether to use serialization/deserialization between JSON and class instance.
    warnings: Whether to show any warnings that occurred during serialization.

Returns:
    A JSON string representation of the model.

### model_extra `classproperty`

Get extra fields set during validation.

Returns:
    A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`.

### model_fields `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### model_fields_set `classproperty`

Returns the set of fields that have been set on this model instance.

Returns:
    A set of strings representing the fields that have been set,
        i.e. that were not filled from defaults.

### model_json_schema `classmethod`

```
model_json_schema(
  by_alias: 'bool' = True,
  ref_template: 'str' = '#/$defs/{model}',
  schema_generator: 'type[GenerateJsonSchema]' = <class 'pydantic.json_schema.GenerateJsonSchema'>,
  mode: 'JsonSchemaMode' = 'validation'
) -> dict[str, Any]
```

Generates a JSON schema for a model class.

Args:
    by_alias: Whether to use attribute aliases or not.
    ref_template: The reference template.
    schema_generator: To override the logic used to generate the JSON schema, as a subclass of
        `GenerateJsonSchema` with your desired modifications
    mode: The mode in which to generate the schema.

Returns:
    The JSON schema for the given model class.

### model_parametrized_name `classmethod`

```
model_parametrized_name(
  params: 'tuple[type[Any], ...]'
) -> str
```

Compute the class name for parametrizations of generic classes.

This method can be overridden to achieve a custom naming scheme for generic BaseModels.

Args:
    params: Tuple of types of the class. Given a generic class
        `Model` with 2 type variables and a concrete model `Model[str, int]`,
        the value `(str, int)` would be passed to `params`.

Returns:
    String representing the new class where `params` are passed to `cls` as type variables.

Raises:
    TypeError: Raised when trying to generate concrete names for non-generic models.

### model_post_init `classfunction`

```
model_post_init(
  self,
  _BaseModel__context: 'Any'
) -> None
```

Override this method to perform additional initialization after `__init__` and `model_construct`.
This is useful if you want to do some validation that requires the entire model to be initialized.

### model_rebuild `classmethod`

```
model_rebuild(
  force: 'bool' = False,
  raise_errors: 'bool' = True,
  _parent_namespace_depth: 'int' = 2,
  _types_namespace: 'dict[str, Any] | None' = None
) -> bool | None
```

Try to rebuild the pydantic-core schema for the model.

This may be necessary when one of the annotations is a ForwardRef which could not be resolved during
the initial attempt to build the schema, and automatic rebuilding fails.

Args:
    force: Whether to force the rebuilding of the model schema, defaults to `False`.
    raise_errors: Whether to raise errors, defaults to `True`.
    _parent_namespace_depth: The depth level of the parent namespace, defaults to 2.
    _types_namespace: The types namespace, defaults to `None`.

Returns:
    Returns `None` if the schema is already "complete" and rebuilding was not required.
    If rebuilding _was_ required, returns `True` if rebuilding was successful, otherwise `False`.

### model_validate `classmethod`

```
model_validate(
  obj: 'Any',
  strict: 'bool | None' = None,
  from_attributes: 'bool | None' = None,
  context: 'dict[str, Any] | None' = None
) -> Model
```

Validate a pydantic model instance.

Args:
    obj: The object to validate.
    strict: Whether to raise an exception on invalid fields.
    from_attributes: Whether to extract data from object attributes.
    context: Additional context to pass to the validator.

Raises:
    ValidationError: If the object could not be validated.

Returns:
    The validated model instance.

### model_validate_json `classmethod`

```
model_validate_json(
  json_data: 'str | bytes | bytearray',
  strict: 'bool | None' = None,
  context: 'dict[str, Any] | None' = None
) -> Model
```

Validate the given JSON data against the Pydantic model.

Args:
    json_data: The JSON data to validate.
    strict: Whether to enforce types strictly.
    context: Extra variables to pass to the validator.

Returns:
    The validated Pydantic model.

Raises:
    ValueError: If `json_data` is not a JSON string.

### model_validate_strings `classmethod`

```
model_validate_strings(
  obj: 'Any',
  strict: 'bool | None' = None,
  context: 'dict[str, Any] | None' = None
) -> Model
```

Validate the given object contains string data against the Pydantic model.

Args:
    obj: The object contains string data to validate.
    strict: Whether to enforce types strictly.
    context: Extra variables to pass to the validator.

Returns:
    The validated Pydantic model.

### parse_file `classmethod`

```
parse_file(
  path: 'str | Path',
  content_type: 'str | None' = None,
  encoding: 'str' = 'utf8',
  proto: 'DeprecatedParseProtocol | None' = None,
  allow_pickle: 'bool' = False
) -> Model
```

### parse_obj `classmethod`

```
parse_obj(
  obj: 'Any'
) -> Model
```

### parse_raw `classmethod`

```
parse_raw(
  b: 'str | bytes',
  content_type: 'str | None' = None,
  encoding: 'str' = 'utf8',
  proto: 'DeprecatedParseProtocol | None' = None,
  allow_pickle: 'bool' = False
) -> Model
```

### schema `classmethod`

```
schema(
  by_alias: 'bool' = True,
  ref_template: 'str' = '#/$defs/{model}'
) -> typing.Dict[str, Any]
```

### schema_json `classmethod`

```
schema_json(
  by_alias: 'bool' = True,
  ref_template: 'str' = '#/$defs/{model}',
  **dumps_kwargs: 'Any'
) -> str
```

### update_forward_refs `classmethod`

```
update_forward_refs(
  **localns: 'Any'
) -> None
```

### validate `classmethod`

```
validate(
  value: 'Any'
) -> Model
```


## Iteration

Usage docs: https://docs.pydantic.dev/2.4/concepts/models/

A base class for creating Pydantic models.

Attributes:
    __class_vars__: The names of classvars defined on the model.
    __private_attributes__: Metadata about the private attributes of the model.
    __signature__: The signature for instantiating the model.

    __pydantic_complete__: Whether model building is completed, or if there are still undefined fields.
    __pydantic_core_schema__: The pydantic-core schema used to build the SchemaValidator and SchemaSerializer.
    __pydantic_custom_init__: Whether the model has a custom `__init__` function.
    __pydantic_decorators__: Metadata containing the decorators defined on the model.
        This replaces `Model.__validators__` and `Model.__root_validators__` from Pydantic V1.
    __pydantic_generic_metadata__: Metadata for generic models; contains data used for a similar purpose to
        __args__, __origin__, __parameters__ in typing-module generics. May eventually be replaced by these.
    __pydantic_parent_namespace__: Parent namespace of the model, used for automatic rebuilding of models.
    __pydantic_post_init__: The name of the post-init method for the model, if defined.
    __pydantic_root_model__: Whether the model is a `RootModel`.
    __pydantic_serializer__: The pydantic-core SchemaSerializer used to dump instances of the model.
    __pydantic_validator__: The pydantic-core SchemaValidator used to validate instances of the model.

    __pydantic_extra__: An instance attribute with the values of extra fields from validation when
        `model_config['extra'] == 'allow'`.
    __pydantic_fields_set__: An instance attribute with the names of fields explicitly specified during validation.
    __pydantic_private__: Instance attribute with the values of private attributes set on the model instance.

### __abstractmethods__ `classfrozenset`

frozenset() -> empty frozenset object
frozenset(iterable) -> frozenset object

Build an immutable unordered collection of unique elements.

### __annotations__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __class_getitem__ `classmethod`

```
__class_getitem__(
  typevar_values: 'type[Any] | tuple[type[Any], ...]'
) -> type[BaseModel] | _forward_ref.PydanticRecursiveRef
```

### __class_vars__ `classset`

set() -> new empty set object
set(iterable) -> new set object

Build an unordered collection of unique elements.

### __copy__ `classfunction`

```
__copy__(
  self: 'Model'
) -> Model
```

Returns a shallow copy of the model.

### __deepcopy__ `classfunction`

```
__deepcopy__(
  self: 'Model',
  memo: 'dict[int, Any] | None' = None
) -> Model
```

Returns a deep copy of the model.

### __delattr__ `classfunction`

```
__delattr__(
  self,
  item: 'str'
) -> Any
```

Implement delattr(self, name).

### __dict__ `classmappingproxy`

### __dir__ `classmethod_descriptor`

```
__dir__(
  self
)
```

Default dir() implementation.

### __doc__ `classNoneType`

### __eq__ `classfunction`

```
__eq__(
  self,
  other: 'Any'
) -> bool
```

Return self==value.

### __fields__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __fields_set__ `classproperty`

### __format__ `classmethod_descriptor`

```
__format__(
  self,
  format_spec
)
```

Default object formatter.

### __ge__ `classwrapper_descriptor`

```
__ge__(
  self,
  value
)
```

Return self>=value.

### __get_pydantic_core_schema__ `classmethod`

```
__get_pydantic_core_schema__(
  _BaseModel__source: 'type[BaseModel]',
  _BaseModel__handler: 'GetCoreSchemaHandler'
) -> CoreSchema
```

Hook into generating the model's CoreSchema.

Args:
    __source: The class we are generating a schema for.
        This will generally be the same as the `cls` argument if this is a classmethod.
    __handler: Call into Pydantic's internal JSON schema generation.
        A callable that calls into Pydantic's internal CoreSchema generation logic.

Returns:
    A `pydantic-core` `CoreSchema`.

### __get_pydantic_json_schema__ `classmethod`

```
__get_pydantic_json_schema__(
  _BaseModel__core_schema: 'CoreSchema',
  _BaseModel__handler: 'GetJsonSchemaHandler'
) -> JsonSchemaValue
```

Hook into generating the model's JSON schema.

Args:
    __core_schema: A `pydantic-core` CoreSchema.
        You can ignore this argument and call the handler with a new CoreSchema,
        wrap this CoreSchema (`{'type': 'nullable', 'schema': current_schema}`),
        or just call the handler with the original schema.
    __handler: Call into Pydantic's internal JSON schema generation.
        This will raise a `pydantic.errors.PydanticInvalidForJsonSchema` if JSON schema
        generation fails.
        Since this gets called by `BaseModel.model_json_schema` you can override the
        `schema_generator` argument to that function to change JSON schema generation globally
        for a type.

Returns:
    A JSON schema, as a Python object.

### __getattr__ `classfunction`

```
__getattr__(
  self,
  item: 'str'
) -> Any
```

### __getattribute__ `classwrapper_descriptor`

```
__getattribute__(
  self,
  name
)
```

Return getattr(self, name).

### __getstate__ `classfunction`

```
__getstate__(
  self
) -> dict[Any, Any]
```

### __gt__ `classwrapper_descriptor`

```
__gt__(
  self,
  value
)
```

Return self>value.

### __hash__ `classNoneType`

### __init__ `classfunction`

```
__init__(
  __pydantic_self__,
  **data: 'Any'
) -> None
```

Create a new model by parsing and validating input data from keyword arguments.

Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
validated to form a valid model.

`__init__` uses `__pydantic_self__` instead of the more common `self` for the first arg to
allow `self` as a field name.

### __init_subclass__ `classbuiltin_function_or_method`

```
__init_subclass__
```

This method is called when a class is subclassed.

The default implementation does nothing. It may be
overridden to extend subclasses.

### __iter__ `classfunction`

```
__iter__(
  self
) -> TupleGenerator
```

So `dict(model)` works.

### __le__ `classwrapper_descriptor`

```
__le__(
  self,
  value
)
```

Return self<=value.

### __lt__ `classwrapper_descriptor`

```
__lt__(
  self,
  value
)
```

Return self<value.

### __module__ `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### __ne__ `classwrapper_descriptor`

```
__ne__(
  self,
  value
)
```

Return self!=value.

### __new__ `classbuiltin_function_or_method`

```
__new__(
  *args,
  **kwargs
)
```

Create and return a new object.  See help(type) for accurate signature.

### __pretty__ `classfunction`

```
__pretty__(
  self,
  fmt: 'typing.Callable[[Any], Any]',
  **kwargs: 'Any'
) -> typing.Generator[Any, None, None]
```

Used by devtools (https://python-devtools.helpmanual.io/) to pretty print objects.

### __private_attributes__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __pydantic_complete__ `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### __pydantic_core_schema__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __pydantic_custom_init__ `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### __pydantic_decorators__ `classpydantic._internal._decorators.DecoratorInfos`

Mapping of name in the class namespace to decorator info.

note that the name in the class namespace is the function or attribute name
not the field name!

### __pydantic_extra__ `classmember_descriptor`

### __pydantic_fields_set__ `classmember_descriptor`

### __pydantic_generic_metadata__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __pydantic_init_subclass__ `classmethod`

```
__pydantic_init_subclass__(
  **kwargs: 'Any'
) -> None
```

This is intended to behave just like `__init_subclass__`, but is called by `ModelMetaclass`
only after the class is actually fully initialized. In particular, attributes like `model_fields` will
be present when this is called.

This is necessary because `__init_subclass__` will always be called by `type.__new__`,
and it would require a prohibitively large refactor to the `ModelMetaclass` to ensure that
`type.__new__` was called in such a manner that the class would already be sufficiently initialized.

This will receive the same `kwargs` that would be passed to the standard `__init_subclass__`, namely,
any kwargs passed to the class definition that aren't used internally by pydantic.

Args:
    **kwargs: Any keyword arguments passed to the class definition that aren't used internally
        by pydantic.

### __pydantic_parent_namespace__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __pydantic_post_init__ `classNoneType`

### __pydantic_private__ `classmember_descriptor`

### __pydantic_root_model__ `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### __pydantic_serializer__ `classpydantic_core._pydantic_core.SchemaSerializer`

### __pydantic_validator__ `classpydantic_core._pydantic_core.SchemaValidator`

### __reduce__ `classmethod_descriptor`

```
__reduce__(
  self
)
```

Helper for pickle.

### __reduce_ex__ `classmethod_descriptor`

```
__reduce_ex__(
  self,
  protocol
)
```

Helper for pickle.

### __repr__ `classfunction`

```
__repr__(
  self
) -> str
```

Return repr(self).

### __repr_args__ `classfunction`

```
__repr_args__(
  self
) -> _repr.ReprArgs
```

### __repr_name__ `classfunction`

```
__repr_name__(
  self
) -> str
```

Name of the instance's class, used in __repr__.

### __repr_str__ `classfunction`

```
__repr_str__(
  self,
  join_str: 'str'
) -> str
```

### __rich_repr__ `classfunction`

```
__rich_repr__(
  self
) -> RichReprResult
```

Used by Rich (https://rich.readthedocs.io/en/stable/pretty.html) to pretty print objects.

### __setattr__ `classfunction`

```
__setattr__(
  self,
  name: 'str',
  value: 'Any'
) -> None
```

Implement setattr(self, name, value).

### __setstate__ `classfunction`

```
__setstate__(
  self,
  state: 'dict[Any, Any]'
) -> None
```

### __signature__ `classinspect.Signature`

A Signature object represents the overall signature of a function.
It stores a Parameter object for each parameter accepted by the
function, as well as information specific to the function itself.

A Signature object has the following public attributes and methods:

* parameters : OrderedDict
    An ordered mapping of parameters' names to the corresponding
    Parameter objects (keyword-only arguments are in the same order
    as listed in `code.co_varnames`).
* return_annotation : object
    The annotation for the return type of the function if specified.
    If the function has no annotation for its return type, this
    attribute is set to `Signature.empty`.
* bind(*args, **kwargs) -> BoundArguments
    Creates a mapping from positional and keyword arguments to
    parameters.
* bind_partial(*args, **kwargs) -> BoundArguments
    Creates a partial mapping from positional and keyword arguments
    to parameters (simulating 'functools.partial' behavior.)

### __sizeof__ `classmethod_descriptor`

```
__sizeof__(
  self
)
```

Size of object in memory, in bytes.

### __slots__ `classtuple`

Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.

### __str__ `classfunction`

```
__str__(
  self
) -> str
```

Return str(self).

### __subclasshook__ `classbuiltin_function_or_method`

```
__subclasshook__
```

Abstract classes can override this to customize issubclass().

This is invoked early on by abc.ABCMeta.__subclasscheck__().
It should return True, False or NotImplemented.  If it returns
NotImplemented, the normal algorithm is used.  Otherwise, it
overrides the normal algorithm (and the outcome is cached).

### __weakref__ `classgetset_descriptor`

list of weak references to the object (if defined)

### _abc_impl `class_abc._abc_data`

Internal state held by ABC machinery.

### _calculate_keys `classfunction`

```
_calculate_keys(
  self,
  *args: 'Any',
  **kwargs: 'Any'
) -> Any
```

### _copy_and_set_values `classfunction`

```
_copy_and_set_values(
  self,
  *args: 'Any',
  **kwargs: 'Any'
) -> Any
```

### _get_value `classmethod`

```
_get_value(
  *args: 'Any',
  **kwargs: 'Any'
) -> Any
```

### _iter `classfunction`

```
_iter(
  self,
  *args: 'Any',
  **kwargs: 'Any'
) -> Any
```

### completion_tokens_consumed `classproperty`

Returns the number of completion/output tokens consumed during this
iteration.

### construct `classmethod`

```
construct(
  _fields_set: 'set[str] | None' = None,
  **values: 'Any'
) -> Model
```

### copy `classfunction`

```
copy(
  self: 'Model',
  include: 'AbstractSetIntStr | MappingIntStrAny | None' = None,
  exclude: 'AbstractSetIntStr | MappingIntStrAny | None' = None,
  update: 'typing.Dict[str, Any] | None' = None,
  deep: 'bool' = False
) -> Model
```

Returns a copy of the model.

!!! warning "Deprecated"
    This method is now deprecated; use `model_copy` instead.

If you need `include` or `exclude`, use:

```py
data = self.model_dump(include=include, exclude=exclude, round_trip=True)
data = {**data, **(update or {})}
copied = self.model_validate(data)
```

Args:
    include: Optional set or mapping
        specifying which fields to include in the copied model.
    exclude: Optional set or mapping
        specifying which fields to exclude in the copied model.
    update: Optional dictionary of field-value pairs to override field values
        in the copied model.
    deep: If True, the values of fields that are Pydantic models will be deep copied.

Returns:
    A copy of the model with included, excluded and updated fields as specified.

### dict `classfunction`

```
dict(
  self,
  include: 'IncEx' = None,
  exclude: 'IncEx' = None,
  by_alias: 'bool' = False,
  exclude_unset: 'bool' = False,
  exclude_defaults: 'bool' = False,
  exclude_none: 'bool' = False
) -> typing.Dict[str, Any]
```

### error `classproperty`

The error message from any exception that raised and interrupted
this iteration.

### exception `classproperty`

The exception that interrupted this iteration.

### failed_validations `classproperty`

The validator logs for any validations that failed during this
iteration.

### from_orm `classmethod`

```
from_orm(
  obj: 'Any'
) -> Model
```

### json `classfunction`

```
json(
  self,
  include: 'IncEx' = None,
  exclude: 'IncEx' = None,
  by_alias: 'bool' = False,
  exclude_unset: 'bool' = False,
  exclude_defaults: 'bool' = False,
  exclude_none: 'bool' = False,
  encoder: 'typing.Callable[[Any], Any] | None' = PydanticUndefined,
  models_as_dict: 'bool' = PydanticUndefined,
  **dumps_kwargs: 'Any'
) -> str
```

### logs `classproperty`

Returns the logs from this iteration as a stack.

### model_computed_fields `classproperty`

Get the computed fields of this model instance.

Returns:
    A dictionary of computed field names and their corresponding `ComputedFieldInfo` objects.

### model_config `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### model_construct `classmethod`

```
model_construct(
  _fields_set: 'set[str] | None' = None,
  **values: 'Any'
) -> Model
```

Creates a new instance of the `Model` class with validated data.

Creates a new model setting `__dict__` and `__pydantic_fields_set__` from trusted or pre-validated data.
Default values are respected, but no other validation is performed.
Behaves as if `Config.extra = 'allow'` was set since it adds all passed values

Args:
    _fields_set: The set of field names accepted for the Model instance.
    values: Trusted or pre-validated data dictionary.

Returns:
    A new instance of the `Model` class with validated data.

### model_copy `classfunction`

```
model_copy(
  self: 'Model',
  update: 'dict[str, Any] | None' = None,
  deep: 'bool' = False
) -> Model
```

Usage docs: https://docs.pydantic.dev/2.4/concepts/serialization/#model_copy

Returns a copy of the model.

Args:
    update: Values to change/add in the new model. Note: the data is not validated
        before creating the new model. You should trust this data.
    deep: Set to `True` to make a deep copy of the model.

Returns:
    New model instance.

### model_dump `classfunction`

```
model_dump(
  self,
  mode: "Literal['json', 'python'] | str" = 'python',
  include: 'IncEx' = None,
  exclude: 'IncEx' = None,
  by_alias: 'bool' = False,
  exclude_unset: 'bool' = False,
  exclude_defaults: 'bool' = False,
  exclude_none: 'bool' = False,
  round_trip: 'bool' = False,
  warnings: 'bool' = True
) -> dict[str, Any]
```

Usage docs: https://docs.pydantic.dev/2.4/concepts/serialization/#modelmodel_dump

Generate a dictionary representation of the model, optionally specifying which fields to include or exclude.

Args:
    mode: The mode in which `to_python` should run.
        If mode is 'json', the dictionary will only contain JSON serializable types.
        If mode is 'python', the dictionary may contain any Python objects.
    include: A list of fields to include in the output.
    exclude: A list of fields to exclude from the output.
    by_alias: Whether to use the field's alias in the dictionary key if defined.
    exclude_unset: Whether to exclude fields that are unset or None from the output.
    exclude_defaults: Whether to exclude fields that are set to their default value from the output.
    exclude_none: Whether to exclude fields that have a value of `None` from the output.
    round_trip: Whether to enable serialization and deserialization round-trip support.
    warnings: Whether to log warnings when invalid fields are encountered.

Returns:
    A dictionary representation of the model.

### model_dump_json `classfunction`

```
model_dump_json(
  self,
  indent: 'int | None' = None,
  include: 'IncEx' = None,
  exclude: 'IncEx' = None,
  by_alias: 'bool' = False,
  exclude_unset: 'bool' = False,
  exclude_defaults: 'bool' = False,
  exclude_none: 'bool' = False,
  round_trip: 'bool' = False,
  warnings: 'bool' = True
) -> str
```

Usage docs: https://docs.pydantic.dev/2.4/concepts/serialization/#modelmodel_dump_json

Generates a JSON representation of the model using Pydantic's `to_json` method.

Args:
    indent: Indentation to use in the JSON output. If None is passed, the output will be compact.
    include: Field(s) to include in the JSON output. Can take either a string or set of strings.
    exclude: Field(s) to exclude from the JSON output. Can take either a string or set of strings.
    by_alias: Whether to serialize using field aliases.
    exclude_unset: Whether to exclude fields that have not been explicitly set.
    exclude_defaults: Whether to exclude fields that have the default value.
    exclude_none: Whether to exclude fields that have a value of `None`.
    round_trip: Whether to use serialization/deserialization between JSON and class instance.
    warnings: Whether to show any warnings that occurred during serialization.

Returns:
    A JSON string representation of the model.

### model_extra `classproperty`

Get extra fields set during validation.

Returns:
    A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`.

### model_fields `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### model_fields_set `classproperty`

Returns the set of fields that have been set on this model instance.

Returns:
    A set of strings representing the fields that have been set,
        i.e. that were not filled from defaults.

### model_json_schema `classmethod`

```
model_json_schema(
  by_alias: 'bool' = True,
  ref_template: 'str' = '#/$defs/{model}',
  schema_generator: 'type[GenerateJsonSchema]' = <class 'pydantic.json_schema.GenerateJsonSchema'>,
  mode: 'JsonSchemaMode' = 'validation'
) -> dict[str, Any]
```

Generates a JSON schema for a model class.

Args:
    by_alias: Whether to use attribute aliases or not.
    ref_template: The reference template.
    schema_generator: To override the logic used to generate the JSON schema, as a subclass of
        `GenerateJsonSchema` with your desired modifications
    mode: The mode in which to generate the schema.

Returns:
    The JSON schema for the given model class.

### model_parametrized_name `classmethod`

```
model_parametrized_name(
  params: 'tuple[type[Any], ...]'
) -> str
```

Compute the class name for parametrizations of generic classes.

This method can be overridden to achieve a custom naming scheme for generic BaseModels.

Args:
    params: Tuple of types of the class. Given a generic class
        `Model` with 2 type variables and a concrete model `Model[str, int]`,
        the value `(str, int)` would be passed to `params`.

Returns:
    String representing the new class where `params` are passed to `cls` as type variables.

Raises:
    TypeError: Raised when trying to generate concrete names for non-generic models.

### model_post_init `classfunction`

```
model_post_init(
  self,
  _BaseModel__context: 'Any'
) -> None
```

Override this method to perform additional initialization after `__init__` and `model_construct`.
This is useful if you want to do some validation that requires the entire model to be initialized.

### model_rebuild `classmethod`

```
model_rebuild(
  force: 'bool' = False,
  raise_errors: 'bool' = True,
  _parent_namespace_depth: 'int' = 2,
  _types_namespace: 'dict[str, Any] | None' = None
) -> bool | None
```

Try to rebuild the pydantic-core schema for the model.

This may be necessary when one of the annotations is a ForwardRef which could not be resolved during
the initial attempt to build the schema, and automatic rebuilding fails.

Args:
    force: Whether to force the rebuilding of the model schema, defaults to `False`.
    raise_errors: Whether to raise errors, defaults to `True`.
    _parent_namespace_depth: The depth level of the parent namespace, defaults to 2.
    _types_namespace: The types namespace, defaults to `None`.

Returns:
    Returns `None` if the schema is already "complete" and rebuilding was not required.
    If rebuilding _was_ required, returns `True` if rebuilding was successful, otherwise `False`.

### model_validate `classmethod`

```
model_validate(
  obj: 'Any',
  strict: 'bool | None' = None,
  from_attributes: 'bool | None' = None,
  context: 'dict[str, Any] | None' = None
) -> Model
```

Validate a pydantic model instance.

Args:
    obj: The object to validate.
    strict: Whether to raise an exception on invalid fields.
    from_attributes: Whether to extract data from object attributes.
    context: Additional context to pass to the validator.

Raises:
    ValidationError: If the object could not be validated.

Returns:
    The validated model instance.

### model_validate_json `classmethod`

```
model_validate_json(
  json_data: 'str | bytes | bytearray',
  strict: 'bool | None' = None,
  context: 'dict[str, Any] | None' = None
) -> Model
```

Validate the given JSON data against the Pydantic model.

Args:
    json_data: The JSON data to validate.
    strict: Whether to enforce types strictly.
    context: Extra variables to pass to the validator.

Returns:
    The validated Pydantic model.

Raises:
    ValueError: If `json_data` is not a JSON string.

### model_validate_strings `classmethod`

```
model_validate_strings(
  obj: 'Any',
  strict: 'bool | None' = None,
  context: 'dict[str, Any] | None' = None
) -> Model
```

Validate the given object contains string data against the Pydantic model.

Args:
    obj: The object contains string data to validate.
    strict: Whether to enforce types strictly.
    context: Extra variables to pass to the validator.

Returns:
    The validated Pydantic model.

### parse_file `classmethod`

```
parse_file(
  path: 'str | Path',
  content_type: 'str | None' = None,
  encoding: 'str' = 'utf8',
  proto: 'DeprecatedParseProtocol | None' = None,
  allow_pickle: 'bool' = False
) -> Model
```

### parse_obj `classmethod`

```
parse_obj(
  obj: 'Any'
) -> Model
```

### parse_raw `classmethod`

```
parse_raw(
  b: 'str | bytes',
  content_type: 'str | None' = None,
  encoding: 'str' = 'utf8',
  proto: 'DeprecatedParseProtocol | None' = None,
  allow_pickle: 'bool' = False
) -> Model
```

### parsed_output `classproperty`

The output from the LLM after undergoing parsing but before
validation.

### prompt_tokens_consumed `classproperty`

Returns the number of prompt/input tokens consumed during this
iteration.

### raw_output `classproperty`

The exact output from the LLM.

### reasks `classproperty`

Reasks generated during validation.

These would be incorporated into the prompt or the next LLM
call.

### rich_group `classproperty`

### schema `classmethod`

```
schema(
  by_alias: 'bool' = True,
  ref_template: 'str' = '#/$defs/{model}'
) -> typing.Dict[str, Any]
```

### schema_json `classmethod`

```
schema_json(
  by_alias: 'bool' = True,
  ref_template: 'str' = '#/$defs/{model}',
  **dumps_kwargs: 'Any'
) -> str
```

### status `classproperty`

Representation of the end state of this iteration.

OneOf: pass, fail, error, not run

### tokens_consumed `classproperty`

Returns the total number of tokens consumed during this
iteration.

### update_forward_refs `classmethod`

```
update_forward_refs(
  **localns: 'Any'
) -> None
```

### validate `classmethod`

```
validate(
  value: 'Any'
) -> Model
```

### validated_output `classproperty`

The valid output from the LLM after undergoing validation.

Could be only a partial structure if field level reasks occur.
Could contain fixed values.

### validation_output `classproperty`

The output from the validation process.

Could be a combination of valid output and ReAsks

### validator_logs `classproperty`

The results of each individual validation performed on the LLM
response during this iteration.


## Outputs

Usage docs: https://docs.pydantic.dev/2.4/concepts/models/

A base class for creating Pydantic models.

Attributes:
    __class_vars__: The names of classvars defined on the model.
    __private_attributes__: Metadata about the private attributes of the model.
    __signature__: The signature for instantiating the model.

    __pydantic_complete__: Whether model building is completed, or if there are still undefined fields.
    __pydantic_core_schema__: The pydantic-core schema used to build the SchemaValidator and SchemaSerializer.
    __pydantic_custom_init__: Whether the model has a custom `__init__` function.
    __pydantic_decorators__: Metadata containing the decorators defined on the model.
        This replaces `Model.__validators__` and `Model.__root_validators__` from Pydantic V1.
    __pydantic_generic_metadata__: Metadata for generic models; contains data used for a similar purpose to
        __args__, __origin__, __parameters__ in typing-module generics. May eventually be replaced by these.
    __pydantic_parent_namespace__: Parent namespace of the model, used for automatic rebuilding of models.
    __pydantic_post_init__: The name of the post-init method for the model, if defined.
    __pydantic_root_model__: Whether the model is a `RootModel`.
    __pydantic_serializer__: The pydantic-core SchemaSerializer used to dump instances of the model.
    __pydantic_validator__: The pydantic-core SchemaValidator used to validate instances of the model.

    __pydantic_extra__: An instance attribute with the values of extra fields from validation when
        `model_config['extra'] == 'allow'`.
    __pydantic_fields_set__: An instance attribute with the names of fields explicitly specified during validation.
    __pydantic_private__: Instance attribute with the values of private attributes set on the model instance.

### __abstractmethods__ `classfrozenset`

frozenset() -> empty frozenset object
frozenset(iterable) -> frozenset object

Build an immutable unordered collection of unique elements.

### __annotations__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __class_getitem__ `classmethod`

```
__class_getitem__(
  typevar_values: 'type[Any] | tuple[type[Any], ...]'
) -> type[BaseModel] | _forward_ref.PydanticRecursiveRef
```

### __class_vars__ `classset`

set() -> new empty set object
set(iterable) -> new set object

Build an unordered collection of unique elements.

### __copy__ `classfunction`

```
__copy__(
  self: 'Model'
) -> Model
```

Returns a shallow copy of the model.

### __deepcopy__ `classfunction`

```
__deepcopy__(
  self: 'Model',
  memo: 'dict[int, Any] | None' = None
) -> Model
```

Returns a deep copy of the model.

### __delattr__ `classfunction`

```
__delattr__(
  self,
  item: 'str'
) -> Any
```

Implement delattr(self, name).

### __dict__ `classmappingproxy`

### __dir__ `classmethod_descriptor`

```
__dir__(
  self
)
```

Default dir() implementation.

### __doc__ `classNoneType`

### __eq__ `classfunction`

```
__eq__(
  self,
  other: 'Any'
) -> bool
```

Return self==value.

### __fields__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __fields_set__ `classproperty`

### __format__ `classmethod_descriptor`

```
__format__(
  self,
  format_spec
)
```

Default object formatter.

### __ge__ `classwrapper_descriptor`

```
__ge__(
  self,
  value
)
```

Return self>=value.

### __get_pydantic_core_schema__ `classmethod`

```
__get_pydantic_core_schema__(
  _BaseModel__source: 'type[BaseModel]',
  _BaseModel__handler: 'GetCoreSchemaHandler'
) -> CoreSchema
```

Hook into generating the model's CoreSchema.

Args:
    __source: The class we are generating a schema for.
        This will generally be the same as the `cls` argument if this is a classmethod.
    __handler: Call into Pydantic's internal JSON schema generation.
        A callable that calls into Pydantic's internal CoreSchema generation logic.

Returns:
    A `pydantic-core` `CoreSchema`.

### __get_pydantic_json_schema__ `classmethod`

```
__get_pydantic_json_schema__(
  _BaseModel__core_schema: 'CoreSchema',
  _BaseModel__handler: 'GetJsonSchemaHandler'
) -> JsonSchemaValue
```

Hook into generating the model's JSON schema.

Args:
    __core_schema: A `pydantic-core` CoreSchema.
        You can ignore this argument and call the handler with a new CoreSchema,
        wrap this CoreSchema (`{'type': 'nullable', 'schema': current_schema}`),
        or just call the handler with the original schema.
    __handler: Call into Pydantic's internal JSON schema generation.
        This will raise a `pydantic.errors.PydanticInvalidForJsonSchema` if JSON schema
        generation fails.
        Since this gets called by `BaseModel.model_json_schema` you can override the
        `schema_generator` argument to that function to change JSON schema generation globally
        for a type.

Returns:
    A JSON schema, as a Python object.

### __getattr__ `classfunction`

```
__getattr__(
  self,
  item: 'str'
) -> Any
```

### __getattribute__ `classwrapper_descriptor`

```
__getattribute__(
  self,
  name
)
```

Return getattr(self, name).

### __getstate__ `classfunction`

```
__getstate__(
  self
) -> dict[Any, Any]
```

### __gt__ `classwrapper_descriptor`

```
__gt__(
  self,
  value
)
```

Return self>value.

### __hash__ `classNoneType`

### __init__ `classfunction`

```
__init__(
  __pydantic_self__,
  **data: 'Any'
) -> None
```

Create a new model by parsing and validating input data from keyword arguments.

Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
validated to form a valid model.

`__init__` uses `__pydantic_self__` instead of the more common `self` for the first arg to
allow `self` as a field name.

### __init_subclass__ `classbuiltin_function_or_method`

```
__init_subclass__
```

This method is called when a class is subclassed.

The default implementation does nothing. It may be
overridden to extend subclasses.

### __iter__ `classfunction`

```
__iter__(
  self
) -> TupleGenerator
```

So `dict(model)` works.

### __le__ `classwrapper_descriptor`

```
__le__(
  self,
  value
)
```

Return self<=value.

### __lt__ `classwrapper_descriptor`

```
__lt__(
  self,
  value
)
```

Return self<value.

### __module__ `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### __ne__ `classwrapper_descriptor`

```
__ne__(
  self,
  value
)
```

Return self!=value.

### __new__ `classbuiltin_function_or_method`

```
__new__(
  *args,
  **kwargs
)
```

Create and return a new object.  See help(type) for accurate signature.

### __pretty__ `classfunction`

```
__pretty__(
  self,
  fmt: 'typing.Callable[[Any], Any]',
  **kwargs: 'Any'
) -> typing.Generator[Any, None, None]
```

Used by devtools (https://python-devtools.helpmanual.io/) to pretty print objects.

### __private_attributes__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __pydantic_complete__ `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### __pydantic_core_schema__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __pydantic_custom_init__ `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### __pydantic_decorators__ `classpydantic._internal._decorators.DecoratorInfos`

Mapping of name in the class namespace to decorator info.

note that the name in the class namespace is the function or attribute name
not the field name!

### __pydantic_extra__ `classmember_descriptor`

### __pydantic_fields_set__ `classmember_descriptor`

### __pydantic_generic_metadata__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __pydantic_init_subclass__ `classmethod`

```
__pydantic_init_subclass__(
  **kwargs: 'Any'
) -> None
```

This is intended to behave just like `__init_subclass__`, but is called by `ModelMetaclass`
only after the class is actually fully initialized. In particular, attributes like `model_fields` will
be present when this is called.

This is necessary because `__init_subclass__` will always be called by `type.__new__`,
and it would require a prohibitively large refactor to the `ModelMetaclass` to ensure that
`type.__new__` was called in such a manner that the class would already be sufficiently initialized.

This will receive the same `kwargs` that would be passed to the standard `__init_subclass__`, namely,
any kwargs passed to the class definition that aren't used internally by pydantic.

Args:
    **kwargs: Any keyword arguments passed to the class definition that aren't used internally
        by pydantic.

### __pydantic_parent_namespace__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __pydantic_post_init__ `classNoneType`

### __pydantic_private__ `classmember_descriptor`

### __pydantic_root_model__ `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### __pydantic_serializer__ `classpydantic_core._pydantic_core.SchemaSerializer`

### __pydantic_validator__ `classpydantic_core._pydantic_core.SchemaValidator`

### __reduce__ `classmethod_descriptor`

```
__reduce__(
  self
)
```

Helper for pickle.

### __reduce_ex__ `classmethod_descriptor`

```
__reduce_ex__(
  self,
  protocol
)
```

Helper for pickle.

### __repr__ `classfunction`

```
__repr__(
  self
) -> str
```

Return repr(self).

### __repr_args__ `classfunction`

```
__repr_args__(
  self
) -> _repr.ReprArgs
```

### __repr_name__ `classfunction`

```
__repr_name__(
  self
) -> str
```

Name of the instance's class, used in __repr__.

### __repr_str__ `classfunction`

```
__repr_str__(
  self,
  join_str: 'str'
) -> str
```

### __rich_repr__ `classfunction`

```
__rich_repr__(
  self
) -> RichReprResult
```

Used by Rich (https://rich.readthedocs.io/en/stable/pretty.html) to pretty print objects.

### __setattr__ `classfunction`

```
__setattr__(
  self,
  name: 'str',
  value: 'Any'
) -> None
```

Implement setattr(self, name, value).

### __setstate__ `classfunction`

```
__setstate__(
  self,
  state: 'dict[Any, Any]'
) -> None
```

### __signature__ `classinspect.Signature`

A Signature object represents the overall signature of a function.
It stores a Parameter object for each parameter accepted by the
function, as well as information specific to the function itself.

A Signature object has the following public attributes and methods:

* parameters : OrderedDict
    An ordered mapping of parameters' names to the corresponding
    Parameter objects (keyword-only arguments are in the same order
    as listed in `code.co_varnames`).
* return_annotation : object
    The annotation for the return type of the function if specified.
    If the function has no annotation for its return type, this
    attribute is set to `Signature.empty`.
* bind(*args, **kwargs) -> BoundArguments
    Creates a mapping from positional and keyword arguments to
    parameters.
* bind_partial(*args, **kwargs) -> BoundArguments
    Creates a partial mapping from positional and keyword arguments
    to parameters (simulating 'functools.partial' behavior.)

### __sizeof__ `classmethod_descriptor`

```
__sizeof__(
  self
)
```

Size of object in memory, in bytes.

### __slots__ `classtuple`

Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.

### __str__ `classfunction`

```
__str__(
  self
) -> str
```

Return str(self).

### __subclasshook__ `classbuiltin_function_or_method`

```
__subclasshook__
```

Abstract classes can override this to customize issubclass().

This is invoked early on by abc.ABCMeta.__subclasscheck__().
It should return True, False or NotImplemented.  If it returns
NotImplemented, the normal algorithm is used.  Otherwise, it
overrides the normal algorithm (and the outcome is cached).

### __weakref__ `classgetset_descriptor`

list of weak references to the object (if defined)

### _abc_impl `class_abc._abc_data`

Internal state held by ABC machinery.

### _all_empty `classfunction`

```
_all_empty(
  self
) -> <class 'bool'>
```

### _calculate_keys `classfunction`

```
_calculate_keys(
  self,
  *args: 'Any',
  **kwargs: 'Any'
) -> Any
```

### _copy_and_set_values `classfunction`

```
_copy_and_set_values(
  self,
  *args: 'Any',
  **kwargs: 'Any'
) -> Any
```

### _get_value `classmethod`

```
_get_value(
  *args: 'Any',
  **kwargs: 'Any'
) -> Any
```

### _iter `classfunction`

```
_iter(
  self,
  *args: 'Any',
  **kwargs: 'Any'
) -> Any
```

### construct `classmethod`

```
construct(
  _fields_set: 'set[str] | None' = None,
  **values: 'Any'
) -> Model
```

### copy `classfunction`

```
copy(
  self: 'Model',
  include: 'AbstractSetIntStr | MappingIntStrAny | None' = None,
  exclude: 'AbstractSetIntStr | MappingIntStrAny | None' = None,
  update: 'typing.Dict[str, Any] | None' = None,
  deep: 'bool' = False
) -> Model
```

Returns a copy of the model.

!!! warning "Deprecated"
    This method is now deprecated; use `model_copy` instead.

If you need `include` or `exclude`, use:

```py
data = self.model_dump(include=include, exclude=exclude, round_trip=True)
data = {**data, **(update or {})}
copied = self.model_validate(data)
```

Args:
    include: Optional set or mapping
        specifying which fields to include in the copied model.
    exclude: Optional set or mapping
        specifying which fields to exclude in the copied model.
    update: Optional dictionary of field-value pairs to override field values
        in the copied model.
    deep: If True, the values of fields that are Pydantic models will be deep copied.

Returns:
    A copy of the model with included, excluded and updated fields as specified.

### dict `classfunction`

```
dict(
  self,
  include: 'IncEx' = None,
  exclude: 'IncEx' = None,
  by_alias: 'bool' = False,
  exclude_unset: 'bool' = False,
  exclude_defaults: 'bool' = False,
  exclude_none: 'bool' = False
) -> typing.Dict[str, Any]
```

### failed_validations `classproperty`

Returns the validator logs for any validation that failed.

### from_orm `classmethod`

```
from_orm(
  obj: 'Any'
) -> Model
```

### json `classfunction`

```
json(
  self,
  include: 'IncEx' = None,
  exclude: 'IncEx' = None,
  by_alias: 'bool' = False,
  exclude_unset: 'bool' = False,
  exclude_defaults: 'bool' = False,
  exclude_none: 'bool' = False,
  encoder: 'typing.Callable[[Any], Any] | None' = PydanticUndefined,
  models_as_dict: 'bool' = PydanticUndefined,
  **dumps_kwargs: 'Any'
) -> str
```

### model_computed_fields `classproperty`

Get the computed fields of this model instance.

Returns:
    A dictionary of computed field names and their corresponding `ComputedFieldInfo` objects.

### model_config `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### model_construct `classmethod`

```
model_construct(
  _fields_set: 'set[str] | None' = None,
  **values: 'Any'
) -> Model
```

Creates a new instance of the `Model` class with validated data.

Creates a new model setting `__dict__` and `__pydantic_fields_set__` from trusted or pre-validated data.
Default values are respected, but no other validation is performed.
Behaves as if `Config.extra = 'allow'` was set since it adds all passed values

Args:
    _fields_set: The set of field names accepted for the Model instance.
    values: Trusted or pre-validated data dictionary.

Returns:
    A new instance of the `Model` class with validated data.

### model_copy `classfunction`

```
model_copy(
  self: 'Model',
  update: 'dict[str, Any] | None' = None,
  deep: 'bool' = False
) -> Model
```

Usage docs: https://docs.pydantic.dev/2.4/concepts/serialization/#model_copy

Returns a copy of the model.

Args:
    update: Values to change/add in the new model. Note: the data is not validated
        before creating the new model. You should trust this data.
    deep: Set to `True` to make a deep copy of the model.

Returns:
    New model instance.

### model_dump `classfunction`

```
model_dump(
  self,
  mode: "Literal['json', 'python'] | str" = 'python',
  include: 'IncEx' = None,
  exclude: 'IncEx' = None,
  by_alias: 'bool' = False,
  exclude_unset: 'bool' = False,
  exclude_defaults: 'bool' = False,
  exclude_none: 'bool' = False,
  round_trip: 'bool' = False,
  warnings: 'bool' = True
) -> dict[str, Any]
```

Usage docs: https://docs.pydantic.dev/2.4/concepts/serialization/#modelmodel_dump

Generate a dictionary representation of the model, optionally specifying which fields to include or exclude.

Args:
    mode: The mode in which `to_python` should run.
        If mode is 'json', the dictionary will only contain JSON serializable types.
        If mode is 'python', the dictionary may contain any Python objects.
    include: A list of fields to include in the output.
    exclude: A list of fields to exclude from the output.
    by_alias: Whether to use the field's alias in the dictionary key if defined.
    exclude_unset: Whether to exclude fields that are unset or None from the output.
    exclude_defaults: Whether to exclude fields that are set to their default value from the output.
    exclude_none: Whether to exclude fields that have a value of `None` from the output.
    round_trip: Whether to enable serialization and deserialization round-trip support.
    warnings: Whether to log warnings when invalid fields are encountered.

Returns:
    A dictionary representation of the model.

### model_dump_json `classfunction`

```
model_dump_json(
  self,
  indent: 'int | None' = None,
  include: 'IncEx' = None,
  exclude: 'IncEx' = None,
  by_alias: 'bool' = False,
  exclude_unset: 'bool' = False,
  exclude_defaults: 'bool' = False,
  exclude_none: 'bool' = False,
  round_trip: 'bool' = False,
  warnings: 'bool' = True
) -> str
```

Usage docs: https://docs.pydantic.dev/2.4/concepts/serialization/#modelmodel_dump_json

Generates a JSON representation of the model using Pydantic's `to_json` method.

Args:
    indent: Indentation to use in the JSON output. If None is passed, the output will be compact.
    include: Field(s) to include in the JSON output. Can take either a string or set of strings.
    exclude: Field(s) to exclude from the JSON output. Can take either a string or set of strings.
    by_alias: Whether to serialize using field aliases.
    exclude_unset: Whether to exclude fields that have not been explicitly set.
    exclude_defaults: Whether to exclude fields that have the default value.
    exclude_none: Whether to exclude fields that have a value of `None`.
    round_trip: Whether to use serialization/deserialization between JSON and class instance.
    warnings: Whether to show any warnings that occurred during serialization.

Returns:
    A JSON string representation of the model.

### model_extra `classproperty`

Get extra fields set during validation.

Returns:
    A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`.

### model_fields `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### model_fields_set `classproperty`

Returns the set of fields that have been set on this model instance.

Returns:
    A set of strings representing the fields that have been set,
        i.e. that were not filled from defaults.

### model_json_schema `classmethod`

```
model_json_schema(
  by_alias: 'bool' = True,
  ref_template: 'str' = '#/$defs/{model}',
  schema_generator: 'type[GenerateJsonSchema]' = <class 'pydantic.json_schema.GenerateJsonSchema'>,
  mode: 'JsonSchemaMode' = 'validation'
) -> dict[str, Any]
```

Generates a JSON schema for a model class.

Args:
    by_alias: Whether to use attribute aliases or not.
    ref_template: The reference template.
    schema_generator: To override the logic used to generate the JSON schema, as a subclass of
        `GenerateJsonSchema` with your desired modifications
    mode: The mode in which to generate the schema.

Returns:
    The JSON schema for the given model class.

### model_parametrized_name `classmethod`

```
model_parametrized_name(
  params: 'tuple[type[Any], ...]'
) -> str
```

Compute the class name for parametrizations of generic classes.

This method can be overridden to achieve a custom naming scheme for generic BaseModels.

Args:
    params: Tuple of types of the class. Given a generic class
        `Model` with 2 type variables and a concrete model `Model[str, int]`,
        the value `(str, int)` would be passed to `params`.

Returns:
    String representing the new class where `params` are passed to `cls` as type variables.

Raises:
    TypeError: Raised when trying to generate concrete names for non-generic models.

### model_post_init `classfunction`

```
model_post_init(
  self,
  _BaseModel__context: 'Any'
) -> None
```

Override this method to perform additional initialization after `__init__` and `model_construct`.
This is useful if you want to do some validation that requires the entire model to be initialized.

### model_rebuild `classmethod`

```
model_rebuild(
  force: 'bool' = False,
  raise_errors: 'bool' = True,
  _parent_namespace_depth: 'int' = 2,
  _types_namespace: 'dict[str, Any] | None' = None
) -> bool | None
```

Try to rebuild the pydantic-core schema for the model.

This may be necessary when one of the annotations is a ForwardRef which could not be resolved during
the initial attempt to build the schema, and automatic rebuilding fails.

Args:
    force: Whether to force the rebuilding of the model schema, defaults to `False`.
    raise_errors: Whether to raise errors, defaults to `True`.
    _parent_namespace_depth: The depth level of the parent namespace, defaults to 2.
    _types_namespace: The types namespace, defaults to `None`.

Returns:
    Returns `None` if the schema is already "complete" and rebuilding was not required.
    If rebuilding _was_ required, returns `True` if rebuilding was successful, otherwise `False`.

### model_validate `classmethod`

```
model_validate(
  obj: 'Any',
  strict: 'bool | None' = None,
  from_attributes: 'bool | None' = None,
  context: 'dict[str, Any] | None' = None
) -> Model
```

Validate a pydantic model instance.

Args:
    obj: The object to validate.
    strict: Whether to raise an exception on invalid fields.
    from_attributes: Whether to extract data from object attributes.
    context: Additional context to pass to the validator.

Raises:
    ValidationError: If the object could not be validated.

Returns:
    The validated model instance.

### model_validate_json `classmethod`

```
model_validate_json(
  json_data: 'str | bytes | bytearray',
  strict: 'bool | None' = None,
  context: 'dict[str, Any] | None' = None
) -> Model
```

Validate the given JSON data against the Pydantic model.

Args:
    json_data: The JSON data to validate.
    strict: Whether to enforce types strictly.
    context: Extra variables to pass to the validator.

Returns:
    The validated Pydantic model.

Raises:
    ValueError: If `json_data` is not a JSON string.

### model_validate_strings `classmethod`

```
model_validate_strings(
  obj: 'Any',
  strict: 'bool | None' = None,
  context: 'dict[str, Any] | None' = None
) -> Model
```

Validate the given object contains string data against the Pydantic model.

Args:
    obj: The object contains string data to validate.
    strict: Whether to enforce types strictly.
    context: Extra variables to pass to the validator.

Returns:
    The validated Pydantic model.

### parse_file `classmethod`

```
parse_file(
  path: 'str | Path',
  content_type: 'str | None' = None,
  encoding: 'str' = 'utf8',
  proto: 'DeprecatedParseProtocol | None' = None,
  allow_pickle: 'bool' = False
) -> Model
```

### parse_obj `classmethod`

```
parse_obj(
  obj: 'Any'
) -> Model
```

### parse_raw `classmethod`

```
parse_raw(
  b: 'str | bytes',
  content_type: 'str | None' = None,
  encoding: 'str' = 'utf8',
  proto: 'DeprecatedParseProtocol | None' = None,
  allow_pickle: 'bool' = False
) -> Model
```

### schema `classmethod`

```
schema(
  by_alias: 'bool' = True,
  ref_template: 'str' = '#/$defs/{model}'
) -> typing.Dict[str, Any]
```

### schema_json `classmethod`

```
schema_json(
  by_alias: 'bool' = True,
  ref_template: 'str' = '#/$defs/{model}',
  **dumps_kwargs: 'Any'
) -> str
```

### status `classproperty`

Representation of the end state of the validation run.

OneOf: pass, fail, error, not run

### update_forward_refs `classmethod`

```
update_forward_refs(
  **localns: 'Any'
) -> None
```

### validate `classmethod`

```
validate(
  value: 'Any'
) -> Model
```


