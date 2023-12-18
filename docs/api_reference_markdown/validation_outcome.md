# ValidationOutcome

Abstract base class for generic types.

A generic type is typically declared by inheriting from
this class parameterized with one or more type variables.
For example, a generic mapping type might be defined as::

  class Mapping(Generic[KT, VT]):
      def __getitem__(self, key: KT) -> VT:
          ...
      # Etc.

This class can then be used as follows::

  def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
      try:
          return mapping[key]
      except KeyError:
          return default

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

### from_guard_history `classmethod`

```
from_guard_history(
  call: guardrails.classes.history.call.Call,
  error_message: Optional[str]
)
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

