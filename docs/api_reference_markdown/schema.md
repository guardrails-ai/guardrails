# Schema
### Any `classtyping._SpecialForm`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
or class checks.

### Dict `classtyping._SpecialGenericAlias`

A generic version of dict.

## JsonSchema

Schema class that holds a _schema attribute.

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

## type

type(object) -> the object's type
type(name, bases, dict, **kwds) -> a new type

### __abstractmethods__ `classgetset_descriptor`

## object

The base class of the class hierarchy.

When called, it accepts no arguments and returns a new featureless
instance that has no instance attributes and cannot be given any.


### __delattr__ `classwrapper_descriptor`

```
__delattr__(
  self,
  name
)
```

Implement delattr(self, name).

### __dir__ `classmethod_descriptor`

```
__dir__(
  self
)
```

Default dir() implementation.

### __doc__ `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### __eq__ `classwrapper_descriptor`

```
__eq__(
  self,
  value
)
```

Return self==value.

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

### __getattribute__ `classwrapper_descriptor`

```
__getattribute__(
  self,
  name
)
```

Return getattr(self, name).

### __gt__ `classwrapper_descriptor`

```
__gt__(
  self,
  value
)
```

Return self>value.

### __hash__ `classwrapper_descriptor`

```
__hash__(
  self
)
```

Return hash(self).

### __init__ `classwrapper_descriptor`

```
__init__(
  self,
  *args,
  **kwargs
)
```

Initialize self.  See help(type(self)) for accurate signature.

### __init_subclass__ `classbuiltin_function_or_method`

```
__init_subclass__
```

This method is called when a class is subclassed.

The default implementation does nothing. It may be
overridden to extend subclasses.

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

### __repr__ `classwrapper_descriptor`

```
__repr__(
  self
)
```

Return repr(self).

### __setattr__ `classwrapper_descriptor`

```
__setattr__(
  self,
  name,
  value
)
```

Implement setattr(self, name, value).

### __sizeof__ `classmethod_descriptor`

```
__sizeof__(
  self
)
```

Size of object in memory, in bytes.

### __str__ `classwrapper_descriptor`

```
__str__(
  self
)
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


### __bases__ `classtuple`

Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.

### __basicsize__ `classint`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

### __call__ `classwrapper_descriptor`

```
__call__(
  self,
  *args,
  **kwargs
)
```

Call self as a function.


### __delattr__ `classwrapper_descriptor`

```
__delattr__(
  self,
  name
)
```

Implement delattr(self, name).

### __dict__ `classmappingproxy`

### __dictoffset__ `classint`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

### __dir__ `classmethod_descriptor`

```
__dir__(
  self
)
```

Specialized __dir__ implementation for types.

### __doc__ `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### __eq__ `classwrapper_descriptor`

```
__eq__(
  self,
  value
)
```

Return self==value.

### __flags__ `classint`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

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

### __getattribute__ `classwrapper_descriptor`

```
__getattribute__(
  self,
  name
)
```

Return getattr(self, name).

### __gt__ `classwrapper_descriptor`

```
__gt__(
  self,
  value
)
```

Return self>value.

### __hash__ `classwrapper_descriptor`

```
__hash__(
  self
)
```

Return hash(self).

### __init__ `classwrapper_descriptor`

```
__init__(
  self,
  *args,
  **kwargs
)
```

Initialize self.  See help(type(self)) for accurate signature.

### __init_subclass__ `classbuiltin_function_or_method`

```
__init_subclass__
```

This method is called when a class is subclassed.

The default implementation does nothing. It may be
overridden to extend subclasses.

### __instancecheck__ `classmethod_descriptor`

```
__instancecheck__(
  self,
  instance
)
```

Check if an object is an instance.

### __itemsize__ `classint`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

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

### __mro__ `classtuple`

Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.

### __name__ `classstr`

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

### __prepare__ `classbuiltin_function_or_method`

```
__prepare__
```

__prepare__() -> dict
used to create the namespace for the class statement

### __qualname__ `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

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

### __repr__ `classwrapper_descriptor`

```
__repr__(
  self
)
```

Return repr(self).

### __setattr__ `classwrapper_descriptor`

```
__setattr__(
  self,
  name,
  value
)
```

Implement setattr(self, name, value).

### __sizeof__ `classmethod_descriptor`

```
__sizeof__(
  self
)
```

Return memory consumption of the type object.

### __str__ `classwrapper_descriptor`

```
__str__(
  self
)
```

Return str(self).

### __subclasscheck__ `classmethod_descriptor`

```
__subclasscheck__(
  self,
  subclass
)
```

Check if a class is a subclass.

### __subclasses__ `classmethod_descriptor`

```
__subclasses__(
  self
)
```

Return a list of immediate subclasses.

### __subclasshook__ `classbuiltin_function_or_method`

```
__subclasshook__
```

Abstract classes can override this to customize issubclass().

This is invoked early on by abc.ABCMeta.__subclasscheck__().
It should return True, False or NotImplemented.  If it returns
NotImplemented, the normal algorithm is used.  Otherwise, it
overrides the normal algorithm (and the outcome is cached).

### __text_signature__ `classNoneType`

### __weakrefoffset__ `classint`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

### mro `classmethod_descriptor`

```
mro(
  self
)
```

Return a type's method resolution order.


### __delattr__ `classwrapper_descriptor`

```
__delattr__(
  self,
  name
)
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

### __eq__ `classwrapper_descriptor`

```
__eq__(
  self,
  value
)
```

Return self==value.

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

### __getattribute__ `classwrapper_descriptor`

```
__getattribute__(
  self,
  name
)
```

Return getattr(self, name).

### __gt__ `classwrapper_descriptor`

```
__gt__(
  self,
  value
)
```

Return self>value.

### __hash__ `classwrapper_descriptor`

```
__hash__(
  self
)
```

Return hash(self).

### __init__ `classfunction`

```
__init__(
  self,
  schema: guardrails.datatypes.Object,
  reask_prompt_template: Optional[str] = None,
  reask_instructions_template: Optional[str] = None
) -> None
```

Initialize self.  See help(type(self)) for accurate signature.

### __init_subclass__ `classbuiltin_function_or_method`

```
__init_subclass__
```

This method is called when a class is subclassed.

The default implementation does nothing. It may be
overridden to extend subclasses.

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
) -> <class 'str'>
```

Return repr(self).

### __setattr__ `classwrapper_descriptor`

```
__setattr__(
  self,
  name,
  value
)
```

Implement setattr(self, name, value).

### __sizeof__ `classmethod_descriptor`

```
__sizeof__(
  self
)
```

Size of object in memory, in bytes.

### __str__ `classwrapper_descriptor`

```
__str__(
  self
)
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

### async_validate `classfunction`

```
async_validate(
  self,
  iteration: guardrails.classes.history.iteration.Iteration,
  data: Optional[Dict[str, Any]],
  metadata: Dict
) -> typing.Any
```

Validate a dictionary of data against the schema.

Args:
    data: The data to validate.

Returns:
    The validated data.

### check_valid_reask_prompt `classfunction`

```
check_valid_reask_prompt(
  self,
  reask_prompt: Optional[str]
) -> None
```

### from_pydantic `classmethod`

```
from_pydantic(
  model: Type[pydantic.main.BaseModel],
  reask_prompt_template: Optional[str] = None,
  reask_instructions_template: Optional[str] = None
) -> typing_extensions.Self
```

### from_xml `classmethod`

```
from_xml(
  root: lxml.etree._Element,
  reask_prompt_template: Optional[str] = None,
  reask_instructions_template: Optional[str] = None
) -> typing_extensions.Self
```

Create a schema from an XML element.

### get_reask_setup `classfunction`

```
get_reask_setup(
  self,
  reasks: List[guardrails.utils.reask_utils.ReAsk],
  original_response: Any,
  use_full_schema: bool,
  prompt_params: Optional[Dict[str, Any]] = None
) -> typing.Tuple[ForwardRef('Schema'), guardrails.prompt.prompt.Prompt, guardrails.prompt.instructions.Instructions]
```

Construct a schema for reasking, and a prompt for reasking.

Args:
    reasks: List of tuples, where each tuple contains the path to the
        reasked element, and the ReAsk object (which contains the error
        message describing why the reask is necessary).
    original_response: The value that was returned from the API, with reasks.
    use_full_schema: Whether to use the full schema, or only the schema
        for the reasked elements.

Returns:
    The schema for reasking, and the prompt for reasking.

### introspect `classfunction`

```
introspect(
  self,
  data: Any
) -> typing.Tuple[typing.List[guardrails.utils.reask_utils.ReAsk], typing.Optional[typing.Dict]]
```

Inspect the data for reasks.

Args:
    data: The data to introspect.

Returns:
    A list of ReAsk objects.

### is_valid_fragment `classfunction`

```
is_valid_fragment(
  self,
  fragment: str,
  verified: set
) -> <class 'bool'>
```

Check if the fragment is a somewhat valid JSON.

### parse `classfunction`

```
parse(
  self,
  output: str,
  **kwargs
) -> typing.Tuple[typing.Union[typing.Dict, NoneType, guardrails.utils.reask_utils.NonParseableReAsk, str], typing.Union[Exception, NoneType, str, bool]]
```

Parse the output from the large language model.

Args:
    output: The output from the large language model.

Returns:
    The parsed output, and the exception that was raised (if any).

### parse_fragment `classfunction`

```
parse_fragment(
  self,
  fragment: str
)
```

Parse the fragment into a dict.

### preprocess_prompt `classfunction`

```
preprocess_prompt(
  self,
  prompt_callable: guardrails.llm_providers.PromptCallableBase,
  instructions: Optional[guardrails.prompt.instructions.Instructions],
  prompt: guardrails.prompt.prompt.Prompt
)
```

Preprocess the instructions and prompt before sending it to the
model.

Args:
    prompt_callable: The callable to be used to prompt the model.
    instructions: The instructions to preprocess.
    prompt: The prompt to preprocess.

### reask_instructions_template `classproperty`

### reask_prompt_template `classproperty`

### reask_prompt_vars `classset`

set() -> new empty set object
set(iterable) -> new set object

Build an unordered collection of unique elements.

### transpile `classfunction`

```
transpile(
  self,
  method: str = 'default'
) -> <class 'str'>
```

Convert the XML schema to a string that is used for prompting a
large language model.

Returns:
    The prompt.

### validate `classfunction`

```
validate(
  self,
  iteration: guardrails.classes.history.iteration.Iteration,
  data: Optional[Dict[str, Any]],
  metadata: Dict,
  **kwargs
) -> typing.Any
```

Validate a dictionary of data against the schema.

Args:
    data: The data to validate.

Returns:
    The validated data.


### List `classtyping._SpecialGenericAlias`

A generic version of list.

### Optional `classtyping._SpecialForm`

Optional type.

Optional[X] is equivalent to Union[X, None].

## Schema

Schema class that holds a _schema attribute.

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


### __delattr__ `classwrapper_descriptor`

```
__delattr__(
  self,
  name
)
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

### __doc__ `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### __eq__ `classwrapper_descriptor`

```
__eq__(
  self,
  value
)
```

Return self==value.

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

### __getattribute__ `classwrapper_descriptor`

```
__getattribute__(
  self,
  name
)
```

Return getattr(self, name).

### __gt__ `classwrapper_descriptor`

```
__gt__(
  self,
  value
)
```

Return self>value.

### __hash__ `classwrapper_descriptor`

```
__hash__(
  self
)
```

Return hash(self).

### __init__ `classfunction`

```
__init__(
  self,
  schema: guardrails.datatypes.DataType,
  reask_prompt_template: Optional[str] = None,
  reask_instructions_template: Optional[str] = None
) -> None
```

Initialize self.  See help(type(self)) for accurate signature.

### __init_subclass__ `classbuiltin_function_or_method`

```
__init_subclass__
```

This method is called when a class is subclassed.

The default implementation does nothing. It may be
overridden to extend subclasses.

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
) -> <class 'str'>
```

Return repr(self).

### __setattr__ `classwrapper_descriptor`

```
__setattr__(
  self,
  name,
  value
)
```

Implement setattr(self, name, value).

### __sizeof__ `classmethod_descriptor`

```
__sizeof__(
  self
)
```

Size of object in memory, in bytes.

### __str__ `classwrapper_descriptor`

```
__str__(
  self
)
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

### async_validate `classfunction`

```
async_validate(
  self,
  iteration: guardrails.classes.history.iteration.Iteration,
  data: Any,
  metadata: Dict
) -> typing.Any
```

Asynchronously validate a dictionary of data against the schema.

Args:
    data: The data to validate.

Returns:
    The validated data.

### check_valid_reask_prompt `classfunction`

```
check_valid_reask_prompt(
  self,
  reask_prompt: Optional[str]
) -> None
```

### from_xml `classmethod`

```
from_xml(
  root: lxml.etree._Element,
  reask_prompt_template: Optional[str] = None,
  reask_instructions_template: Optional[str] = None
) -> typing_extensions.Self
```

Create a schema from an XML element.

### get_reask_setup `classfunction`

```
get_reask_setup(
  self,
  reasks: Sequence[guardrails.utils.reask_utils.ReAsk],
  original_response: Any,
  use_full_schema: bool,
  prompt_params: Optional[Dict[str, Any]] = None
) -> typing.Tuple[ForwardRef('Schema'), guardrails.prompt.prompt.Prompt, guardrails.prompt.instructions.Instructions]
```

Construct a schema for reasking, and a prompt for reasking.

Args:
    reasks: List of tuples, where each tuple contains the path to the
        reasked element, and the ReAsk object (which contains the error
        message describing why the reask is necessary).
    original_response: The value that was returned from the API, with reasks.
    use_full_schema: Whether to use the full schema, or only the schema
        for the reasked elements.

Returns:
    The schema for reasking, and the prompt for reasking.

### introspect `classfunction`

```
introspect(
  self,
  data: Any
) -> typing.Tuple[typing.Sequence[guardrails.utils.reask_utils.ReAsk], typing.Union[str, typing.Dict, NoneType]]
```

Inspect the data for reasks.

Args:
    data: The data to introspect.

Returns:
    A list of ReAsk objects.

### parse `classfunction`

```
parse(
  self,
  output: str,
  **kwargs
) -> typing.Tuple[typing.Any, typing.Optional[Exception]]
```

Parse the output from the large language model.

Args:
    output: The output from the large language model.

Returns:
    The parsed output, and the exception that was raised (if any).

### preprocess_prompt `classfunction`

```
preprocess_prompt(
  self,
  prompt_callable: guardrails.llm_providers.PromptCallableBase,
  instructions: Optional[guardrails.prompt.instructions.Instructions],
  prompt: guardrails.prompt.prompt.Prompt
)
```

Preprocess the instructions and prompt before sending it to the
model.

Args:
    prompt_callable: The callable to be used to prompt the model.
    instructions: The instructions to preprocess.
    prompt: The prompt to preprocess.

### reask_instructions_template `classproperty`

### reask_prompt_template `classproperty`

### transpile `classfunction`

```
transpile(
  self,
  method: str = 'default'
) -> <class 'str'>
```

Convert the XML schema to a string that is used for prompting a
large language model.

Returns:
    The prompt.

### validate `classfunction`

```
validate(
  self,
  iteration: guardrails.classes.history.iteration.Iteration,
  data: Any,
  metadata: Dict,
  **kwargs
) -> typing.Any
```

Validate a dictionary of data against the schema.

Args:
    data: The data to validate.

Returns:
    The validated data.


## Schema2Prompt

Class that contains transpilers to go from a schema to its
representation in a prompt.

This is important for communicating the schema to a large language
model, and this class will provide multiple alternatives to do so.


### __delattr__ `classwrapper_descriptor`

```
__delattr__(
  self,
  name
)
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

### __doc__ `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### __eq__ `classwrapper_descriptor`

```
__eq__(
  self,
  value
)
```

Return self==value.

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

### __getattribute__ `classwrapper_descriptor`

```
__getattribute__(
  self,
  name
)
```

Return getattr(self, name).

### __gt__ `classwrapper_descriptor`

```
__gt__(
  self,
  value
)
```

Return self>value.

### __hash__ `classwrapper_descriptor`

```
__hash__(
  self
)
```

Return hash(self).

### __init__ `classwrapper_descriptor`

```
__init__(
  self,
  *args,
  **kwargs
)
```

Initialize self.  See help(type(self)) for accurate signature.

### __init_subclass__ `classbuiltin_function_or_method`

```
__init_subclass__
```

This method is called when a class is subclassed.

The default implementation does nothing. It may be
overridden to extend subclasses.

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

### __repr__ `classwrapper_descriptor`

```
__repr__(
  self
)
```

Return repr(self).

### __setattr__ `classwrapper_descriptor`

```
__setattr__(
  self,
  name,
  value
)
```

Implement setattr(self, name, value).

### __sizeof__ `classmethod_descriptor`

```
__sizeof__(
  self
)
```

Size of object in memory, in bytes.

### __str__ `classwrapper_descriptor`

```
__str__(
  self
)
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

### datatypes_to_xml `classfunction`

```
datatypes_to_xml(
  dt: guardrails.datatypes.DataType,
  root: Optional[lxml.etree._Element] = None,
  override_tag_name: Optional[str] = None
) -> <class 'lxml.etree._Element'>
```

Recursively convert the datatypes to XML elements.

### default `classmethod`

```
default(
  schema: guardrails.schema.JsonSchema
) -> <class 'str'>
```

Default transpiler.

Converts the XML schema to a string directly after removing:
    - Comments
    - Action attributes like 'on-fail-*'

Args:
    schema: The schema to transpile.

Returns:
    The prompt.


### Self `classtyping_extensions._SpecialForm`

Used to spell the type of "self" in classes.

Example::

  from typing import Self

  class ReturnsSelf:
      def parse(self, data: bytes) -> Self:
          ...
          return self

### Sequence `classtyping._SpecialGenericAlias`

A generic version of collections.abc.Sequence.

### Set `classtyping._SpecialGenericAlias`

A generic version of set.

## StringSchema

Schema class that holds a _schema attribute.

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


### __delattr__ `classwrapper_descriptor`

```
__delattr__(
  self,
  name
)
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

### __eq__ `classwrapper_descriptor`

```
__eq__(
  self,
  value
)
```

Return self==value.

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

### __getattribute__ `classwrapper_descriptor`

```
__getattribute__(
  self,
  name
)
```

Return getattr(self, name).

### __gt__ `classwrapper_descriptor`

```
__gt__(
  self,
  value
)
```

Return self>value.

### __hash__ `classwrapper_descriptor`

```
__hash__(
  self
)
```

Return hash(self).

### __init__ `classfunction`

```
__init__(
  self,
  schema: guardrails.datatypes.String,
  reask_prompt_template: Optional[str] = None,
  reask_instructions_template: Optional[str] = None
) -> None
```

Initialize self.  See help(type(self)) for accurate signature.

### __init_subclass__ `classbuiltin_function_or_method`

```
__init_subclass__
```

This method is called when a class is subclassed.

The default implementation does nothing. It may be
overridden to extend subclasses.

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
) -> <class 'str'>
```

Return repr(self).

### __setattr__ `classwrapper_descriptor`

```
__setattr__(
  self,
  name,
  value
)
```

Implement setattr(self, name, value).

### __sizeof__ `classmethod_descriptor`

```
__sizeof__(
  self
)
```

Size of object in memory, in bytes.

### __str__ `classwrapper_descriptor`

```
__str__(
  self
)
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

### async_validate `classfunction`

```
async_validate(
  self,
  iteration: guardrails.classes.history.iteration.Iteration,
  data: Any,
  metadata: Dict
) -> typing.Any
```

Validate a dictionary of data against the schema.

Args:
    data: The data to validate.

Returns:
    The validated data.

### check_valid_reask_prompt `classfunction`

```
check_valid_reask_prompt(
  self,
  reask_prompt: Optional[str]
) -> None
```

### from_string `classmethod`

```
from_string(
  validators: Sequence[Union[guardrails.validator_base.Validator, Tuple[Union[guardrails.validator_base.Validator, str, Callable], str]]],
  description: Optional[str] = None,
  reask_prompt_template: Optional[str] = None,
  reask_instructions_template: Optional[str] = None
)
```

### from_xml `classmethod`

```
from_xml(
  root: lxml.etree._Element,
  reask_prompt_template: Optional[str] = None,
  reask_instructions_template: Optional[str] = None
) -> typing_extensions.Self
```

Create a schema from an XML element.

### get_reask_setup `classfunction`

```
get_reask_setup(
  self,
  reasks: List[guardrails.utils.reask_utils.FieldReAsk],
  original_response: guardrails.utils.reask_utils.FieldReAsk,
  use_full_schema: bool,
  prompt_params: Optional[Dict[str, Any]] = None
) -> typing.Tuple[guardrails.schema.Schema, guardrails.prompt.prompt.Prompt, guardrails.prompt.instructions.Instructions]
```

Construct a schema for reasking, and a prompt for reasking.

Args:
    reasks: List of tuples, where each tuple contains the path to the
        reasked element, and the ReAsk object (which contains the error
        message describing why the reask is necessary).
    original_response: The value that was returned from the API, with reasks.
    use_full_schema: Whether to use the full schema, or only the schema
        for the reasked elements.

Returns:
    The schema for reasking, and the prompt for reasking.

### introspect `classfunction`

```
introspect(
  self,
  data: Union[guardrails.utils.reask_utils.ReAsk, str, NoneType]
) -> typing.Tuple[typing.List[guardrails.utils.reask_utils.FieldReAsk], typing.Optional[str]]
```

Inspect the data for reasks.

Args:
    data: The data to introspect.

Returns:
    A list of ReAsk objects.

### parse `classfunction`

```
parse(
  self,
  output: str,
  **kwargs
) -> typing.Tuple[typing.Any, typing.Optional[Exception]]
```

Parse the output from the large language model.

Args:
    output: The output from the large language model.

Returns:
    The parsed output, and the exception that was raised (if any).

### preprocess_prompt `classfunction`

```
preprocess_prompt(
  self,
  prompt_callable: guardrails.llm_providers.PromptCallableBase,
  instructions: Optional[guardrails.prompt.instructions.Instructions],
  prompt: guardrails.prompt.prompt.Prompt
)
```

Preprocess the instructions and prompt before sending it to the
model.

Args:
    prompt_callable: The callable to be used to prompt the model.
    instructions: The instructions to preprocess.
    prompt: The prompt to preprocess.

### reask_instructions_template `classproperty`

### reask_prompt_template `classproperty`

### reask_prompt_vars `classset`

set() -> new empty set object
set(iterable) -> new set object

Build an unordered collection of unique elements.

### transpile `classfunction`

```
transpile(
  self,
  method: str = 'default'
) -> <class 'str'>
```

Convert the XML schema to a string that is used for prompting a
large language model.

Returns:
    The prompt.

### validate `classfunction`

```
validate(
  self,
  iteration: guardrails.classes.history.iteration.Iteration,
  data: Any,
  metadata: Dict,
  **kwargs
) -> typing.Any
```

Validate a dictionary of data against the schema.

Args:
    data: The data to validate.

Returns:
    The validated data.


### TYPE_CHECKING `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### Tuple `classtyping._TupleType`

Tuple type; Tuple[X, Y] is the cross-product type of X and Y.

Example: Tuple[T1, T2] is a tuple of two elements corresponding
to type variables T1 and T2.  Tuple[int, float, str] is a tuple
of an int, a float and a string.

To specify a variable-length tuple of homogeneous type, use Tuple[T, ...].

### Type `classtyping._SpecialGenericAlias`

A special construct usable to annotate class objects.

For example, suppose we have the following classes::

  class User: ...  # Abstract base for User classes
  class BasicUser(User): ...
  class ProUser(User): ...
  class TeamUser(User): ...

And a function that takes a class argument that's a subclass of
User and returns an instance of the corresponding class::

  U = TypeVar('U', bound=User)
  def new_user(user_class: Type[U]) -> U:
      user = user_class()
      # (Here we could write the user object to a database)
      return user

  joe = new_user(BasicUser)

At this point the type checker knows that joe has type BasicUser.

### Union `classtyping._SpecialForm`

Union type; Union[X, Y] means either X or Y.

To define a union, use e.g. Union[int, str].  Details:
- The arguments must be types and there must be at least one.
- None as an argument is a special case and is replaced by
  type(None).
- Unions of unions are flattened, e.g.::

    Union[Union[int, str], float] == Union[int, str, float]

- Unions of a single argument vanish, e.g.::

    Union[int] == int  # The constructor actually returns int

- Redundant arguments are skipped, e.g.::

    Union[int, str, int] == Union[int, str]

- When comparing unions, the argument order is ignored, e.g.::

    Union[int, str] == Union[str, int]

- You cannot subclass or instantiate a union.
- You can use Optional[X] as a shorthand for Union[X, None].

### ValidatorSpec `classtyping._UnionGenericAlias`

### __builtins__ `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### __cached__ `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### __doc__ `classNoneType`

### __file__ `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### __loader__ `class_frozen_importlib_external.SourceFileLoader`

Concrete implementation of SourceLoader using the file system.

### __name__ `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### __package__ `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### __spec__ `class_frozen_importlib.ModuleSpec`

The specification for a module, used for loading.

A module's spec is the source for information about the module.  For
data associated with the module, including source, use the spec's
loader.

`name` is the absolute name of the module.  `loader` is the loader
to use when loading the module.  `parent` is the name of the
package the module is in.  The parent is derived from the name.

`is_package` determines if the module is considered a package or
not.  On modules this is reflected by the `__path__` attribute.

`origin` is the specific location used by the loader from which to
load the module, if that information is available.  When filename is
set, origin will match.

`has_location` indicates that a spec's "origin" reflects a location.
When this is True, `__file__` attribute of the module is set.

`cached` is the location of the cached bytecode file, if any.  It
corresponds to the `__cached__` attribute.

`submodule_search_locations` is the sequence of path entries to
search when importing submodules.  If set, is_package should be
True--and False otherwise.

Packages are simply modules that (may) have submodules.  If a spec
has a non-None value in `submodule_search_locations`, the import
system will consider modules loaded from the spec as packages.

Only finders (see importlib.abc.MetaPathFinder and
importlib.abc.PathEntryFinder) should modify ModuleSpec instances.

### check_refrain_in_dict `classfunction`

```
check_refrain_in_dict(
  schema: Dict
) -> <class 'bool'>
```

Checks if a Refrain object exists in a dict.

Args:
    schema: A dict that can contain lists, dicts or scalars.

Returns:
    True if a Refrain object exists in the dict.

### constants `classguardrails.utils.constants.ConstantsContainer`

### convert_pydantic_model_to_datatype `classfunction`

```
convert_pydantic_model_to_datatype(
  model_field: Union[pydantic.fields.FieldInfo, Type[pydantic.main.BaseModel]],
  datatype: Type[~DataTypeT] = <class 'guardrails.datatypes.Object'>,
  excluded_fields: Optional[List[str]] = None,
  name: Optional[str] = None,
  strict: bool = False
) -> ~DataTypeT
```

Create an Object from a Pydantic model.

### deepcopy `classfunction`

```
deepcopy(
  x,
  memo=None,
  _nil=[]
)
```

Deep copy operation on arbitrary Python objects.

See the module's __doc__ string for more info.

### extract_json_from_ouput `classfunction`

```
extract_json_from_ouput(
  output: str
) -> typing.Tuple[typing.Optional[typing.Dict], typing.Optional[Exception]]
```

### filter_in_dict `classfunction`

```
filter_in_dict(
  schema: Dict
) -> typing.Dict
```

Remove out all Filter objects from a dictionary.

Args:
    schema: A dictionary that can contain lists, dicts or scalars.

Returns:
    A dictionary with all Filter objects removed.

### gather_reasks `classfunction`

```
gather_reasks(
  validated_output: Union[str, Dict, guardrails.utils.reask_utils.ReAsk, NoneType]
) -> typing.Tuple[typing.List[guardrails.utils.reask_utils.ReAsk], typing.Optional[typing.Dict]]
```

Traverse output and gather all ReAsk objects.

Args:
    validated_output (Union[str, Dict, ReAsk], optional): The output of a model.
        Each value can be a ReAsk, a list, a dictionary, or a single value.

Returns:
    A list of ReAsk objects found in the output.

### get_pruned_tree `classfunction`

```
get_pruned_tree(
  root: guardrails.datatypes.Object,
  reasks: Optional[List[guardrails.utils.reask_utils.FieldReAsk]] = None
) -> <class 'guardrails.datatypes.Object'>
```

Prune tree of any elements that are not in `reasks`.

Return the tree with only the elements that are keys of `reasks` and
their parents. If `reasks` is None, return the entire tree. If an
element is removed, remove all ancestors that have no children.

Args:
    root: The XML tree.
    reasks: The elements that are to be reasked.

Returns:
    The prompt.

### logger `classlogging.Logger`

Instances of the Logger class represent a single logging channel. A
"logging channel" indicates an area of an application. Exactly how an
"area" is defined is up to the application developer. Since an
application can have any number of areas, logging channels are identified
by a unique string. Application areas can be nested (e.g. an area
of "input processing" might include sub-areas "read CSV files", "read
XLS files" and "read Gnumeric files"). To cater for this natural nesting,
channel names are organized into a namespace hierarchy where levels are
separated by periods, much like the Java or Python package namespace. So
in the instance given above, channel names might be "input" for the upper
level, and "input.csv", "input.xls" and "input.gnu" for the sub-levels.
There is no arbitrary limit to the depth of nesting.

### prune_obj_for_reasking `classfunction`

```
prune_obj_for_reasking(
  obj: Any
) -> typing.Union[NoneType, typing.Dict, typing.List, guardrails.utils.reask_utils.ReAsk]
```

After validation, we get a nested dictionary where some keys may be
ReAsk objects.

This function prunes the validated form of any object that is not a ReAsk object.
It also keeps all of the ancestors of the ReAsk objects.

Args:
    obj: The validated object.

Returns:
    The pruned validated object.

### verify_schema_against_json `classfunction`

```
verify_schema_against_json(
  schema: guardrails.datatypes.Object,
  generated_json: Dict[str, Any],
  prune_extra_keys: bool = False,
  coerce_types: bool = False,
  validate_subschema: bool = False
)
```

Verify that a JSON schema is valid for a given XML.

