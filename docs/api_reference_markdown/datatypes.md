# Data Types
### Any `classtyping._SpecialForm`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
or class checks.

## Boolean

Element tag: `<bool>`

### children `classproperty`

Return a SimpleNamespace of the children of this DataType.

### collect_validation `classfunction`

```
collect_validation(
  self,
  key: str,
  value: Any,
  schema: Dict
) -> <class 'guardrails.datatypes.FieldValidation'>
```

Gather validators on a value.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### tag `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.


## Case

Element tag: `<case>`

### children `classproperty`

Return a SimpleNamespace of the children of this DataType.

### collect_validation `classfunction`

```
collect_validation(
  self,
  key: str,
  value: Any,
  schema: Dict
) -> <class 'guardrails.datatypes.FieldValidation'>
```

Gather validators on a value.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### tag `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.


## Choice

Element tag: `<object>`

### children `classproperty`

Return a SimpleNamespace of the children of this DataType.

### collect_validation `classfunction`

```
collect_validation(
  self,
  key: str,
  value: Any,
  schema: Dict
) -> <class 'guardrails.datatypes.FieldValidation'>
```

Gather validators on a value.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### tag `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.


## Date

Element tag: `<date>`

To configure the date format, create a date-format attribute on the
element. E.g. `<date name="..." ... date-format="%Y-%m-%d" />`

### children `classproperty`

Return a SimpleNamespace of the children of this DataType.

### collect_validation `classfunction`

```
collect_validation(
  self,
  key: str,
  value: Any,
  schema: Dict
) -> <class 'guardrails.datatypes.FieldValidation'>
```

Gather validators on a value.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### tag `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.


### Dict `classtyping._SpecialGenericAlias`

A generic version of dict.

## Email

Element tag: `<email>`

### children `classproperty`

Return a SimpleNamespace of the children of this DataType.

### collect_validation `classfunction`

```
collect_validation(
  self,
  key: str,
  value: Any,
  schema: Dict
) -> <class 'guardrails.datatypes.FieldValidation'>
```

Gather validators on a value.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### tag `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.


## Enum

Element tag: `<enum>`

### children `classproperty`

Return a SimpleNamespace of the children of this DataType.

### collect_validation `classfunction`

```
collect_validation(
  self,
  key: str,
  value: Any,
  schema: Dict
) -> <class 'guardrails.datatypes.FieldValidation'>
```

Gather validators on a value.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### tag `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.


## FieldValidation

FieldValidation(key: Any, value: Any, validators: List[guardrails.validator_base.Validator], children: List[ForwardRef('FieldValidation')])


## Float

Element tag: `<float>`

### children `classproperty`

Return a SimpleNamespace of the children of this DataType.

### collect_validation `classfunction`

```
collect_validation(
  self,
  key: str,
  value: Any,
  schema: Dict
) -> <class 'guardrails.datatypes.FieldValidation'>
```

Gather validators on a value.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### tag `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.


## Integer

Element tag: `<integer>`

### children `classproperty`

Return a SimpleNamespace of the children of this DataType.

### collect_validation `classfunction`

```
collect_validation(
  self,
  key: str,
  value: Any,
  schema: Dict
) -> <class 'guardrails.datatypes.FieldValidation'>
```

Gather validators on a value.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### tag `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.


### Iterable `classtyping._SpecialGenericAlias`

A generic version of collections.abc.Iterable.

## List

Element tag: `<list>`

### children `classproperty`

Return a SimpleNamespace of the children of this DataType.

### collect_validation `classfunction`

```
collect_validation(
  self,
  key: str,
  value: Any,
  schema: Dict
) -> <class 'guardrails.datatypes.FieldValidation'>
```

Gather validators on a value.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### tag `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.


## NonScalarType
### children `classproperty`

Return a SimpleNamespace of the children of this DataType.

### collect_validation `classfunction`

```
collect_validation(
  self,
  key: str,
  value: Any,
  schema: Dict
) -> <class 'guardrails.datatypes.FieldValidation'>
```

Gather validators on a value.


## Object

Element tag: `<object>`

### children `classproperty`

Return a SimpleNamespace of the children of this DataType.

### collect_validation `classfunction`

```
collect_validation(
  self,
  key: str,
  value: Any,
  schema: Dict
) -> <class 'guardrails.datatypes.FieldValidation'>
```

Gather validators on a value.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### tag `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.


### Optional `classtyping._SpecialForm`

Optional type.

Optional[X] is equivalent to Union[X, None].

## Percentage

Element tag: `<percentage>`

### children `classproperty`

Return a SimpleNamespace of the children of this DataType.

### collect_validation `classfunction`

```
collect_validation(
  self,
  key: str,
  value: Any,
  schema: Dict
) -> <class 'guardrails.datatypes.FieldValidation'>
```

Gather validators on a value.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### tag `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.


## PythonCode

Element tag: `<pythoncode>`

### children `classproperty`

Return a SimpleNamespace of the children of this DataType.

### collect_validation `classfunction`

```
collect_validation(
  self,
  key: str,
  value: Any,
  schema: Dict
) -> <class 'guardrails.datatypes.FieldValidation'>
```

Gather validators on a value.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### tag `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.


## SQLCode

Element tag: `<sql>`

### children `classproperty`

Return a SimpleNamespace of the children of this DataType.

### collect_validation `classfunction`

```
collect_validation(
  self,
  key: str,
  value: Any,
  schema: Dict
) -> <class 'guardrails.datatypes.FieldValidation'>
```

Gather validators on a value.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### tag `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.


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

## String

Element tag: `<string>`

### children `classproperty`

Return a SimpleNamespace of the children of this DataType.

### collect_validation `classfunction`

```
collect_validation(
  self,
  key: str,
  value: Any,
  schema: Dict
) -> <class 'guardrails.datatypes.FieldValidation'>
```

Gather validators on a value.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### tag `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.


### T `classtyping.TypeVar`

Type variable.

Usage::

  T = TypeVar('T')  # Can be anything
  A = TypeVar('A', str, bytes)  # Must be str or bytes

Type variables exist primarily for the benefit of static type
checkers.  They serve as the parameters for generic types as well
as for generic function definitions.  See class Generic for more
information on generic types.  Generic functions work as follows:

  def repeat(x: T, n: int) -> List[T]:
      '''Return a list containing n references to x.'''
      return [x]*n

  def longest(x: A, y: A) -> A:
      '''Return the longest of two strings.'''
      return x if len(x) >= len(y) else y

The latter example's signature is essentially the overloading
of (str, str) -> str and (bytes, bytes) -> bytes.  Also note
that if the arguments are instances of some subclass of str,
the return type is still plain str.

At runtime, isinstance(x, T) and issubclass(C, T) will raise TypeError.

Type variables defined with covariant=True or contravariant=True
can be used to declare covariant or contravariant generic types.
See PEP 484 for more details. By default generic types are invariant
in all type variables.

Type variables can be introspected. e.g.:

  T.__name__ == 'T'
  T.__constraints__ == ()
  T.__covariant__ == False
  T.__contravariant__ = False
  A.__constraints__ == (str, bytes)

Note that only type variables defined in global scope can be pickled.

## Time

Element tag: `<time>`

To configure the date format, create a date-format attribute on the
element. E.g. `<time name="..." ... time-format="%H:%M:%S" />`

### children `classproperty`

Return a SimpleNamespace of the children of this DataType.

### collect_validation `classfunction`

```
collect_validation(
  self,
  key: str,
  value: Any,
  schema: Dict
) -> <class 'guardrails.datatypes.FieldValidation'>
```

Gather validators on a value.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### tag `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.


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

### TypedList `classtyping._SpecialGenericAlias`

A generic version of list.

## URL

Element tag: `<url>`

### children `classproperty`

Return a SimpleNamespace of the children of this DataType.

### collect_validation `classfunction`

```
collect_validation(
  self,
  key: str,
  value: Any,
  schema: Dict
) -> <class 'guardrails.datatypes.FieldValidation'>
```

Gather validators on a value.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### tag `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.


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

### cast_xml_to_string `classfunction`

```
cast_xml_to_string(
  xml_value: Union[memoryview, bytes, bytearray, str]
) -> <class 'str'>
```

Cast XML value to a string.

Args:
    xml_value (Union[memoryview, bytes, bytearray, str]): The XML value to cast.

Returns:
    str: The XML value as a string.

### dataclass `classfunction`

```
dataclass(
  cls=None,
  init=True,
  repr=True,
  eq=True,
  order=False,
  unsafe_hash=False,
  frozen=False
)
```

Returns the same class as was passed in, with dunder methods
added based on the fields defined in the class.

Examines PEP 526 __annotations__ to determine fields.

If init is true, an __init__() method is added to the class. If
repr is true, a __repr__() method is added. If order is true, rich
comparison dunder methods are added. If unsafe_hash is true, a
__hash__() method function is added. If frozen is true, fields may
not be assigned to after instance creation.

### deprecate_type `classfunction`

```
deprecate_type(
  cls: type
)
```

### deprecated_string_types `classset`

set() -> new empty set object
set(iterable) -> new set object

Build an unordered collection of unique elements.

### parse `classfunction`

```
parse(
  timestr,
  parserinfo=None,
  **kwargs
)
```

Parse a string in one of the supported formats, using the
``parserinfo`` parameters.

:param timestr:
    A string containing a date/time stamp.

:param parserinfo:
    A :class:`parserinfo` object containing parameters for the parser.
    If ``None``, the default arguments to the :class:`parserinfo`
    constructor are used.

The ``**kwargs`` parameter takes the following keyword arguments:

:param default:
    The default datetime object, if this is a datetime object and not
    ``None``, elements specified in ``timestr`` replace elements in the
    default object.

:param ignoretz:
    If set ``True``, time zones in parsed strings are ignored and a naive
    :class:`datetime` object is returned.

:param tzinfos:
    Additional time zone names / aliases which may be present in the
    string. This argument maps time zone names (and optionally offsets
    from those time zones) to time zones. This parameter can be a
    dictionary with timezone aliases mapping time zone names to time
    zones or a function taking two parameters (``tzname`` and
    ``tzoffset``) and returning a time zone.

    The timezones to which the names are mapped can be an integer
    offset from UTC in seconds or a :class:`tzinfo` object.

    .. doctest::
       :options: +NORMALIZE_WHITESPACE

        >>> from dateutil.parser import parse
        >>> from dateutil.tz import gettz
        >>> tzinfos = {"BRST": -7200, "CST": gettz("America/Chicago")}
        >>> parse("2012-01-19 17:21:00 BRST", tzinfos=tzinfos)
        datetime.datetime(2012, 1, 19, 17, 21, tzinfo=tzoffset(u'BRST', -7200))
        >>> parse("2012-01-19 17:21:00 CST", tzinfos=tzinfos)
        datetime.datetime(2012, 1, 19, 17, 21,
                          tzinfo=tzfile('/usr/share/zoneinfo/America/Chicago'))

    This parameter is ignored if ``ignoretz`` is set.

:param dayfirst:
    Whether to interpret the first value in an ambiguous 3-integer date
    (e.g. 01/05/09) as the day (``True``) or month (``False``). If
    ``yearfirst`` is set to ``True``, this distinguishes between YDM and
    YMD. If set to ``None``, this value is retrieved from the current
    :class:`parserinfo` object (which itself defaults to ``False``).

:param yearfirst:
    Whether to interpret the first value in an ambiguous 3-integer date
    (e.g. 01/05/09) as the year. If ``True``, the first number is taken to
    be the year, otherwise the last number is taken to be the year. If
    this is set to ``None``, the value is retrieved from the current
    :class:`parserinfo` object (which itself defaults to ``False``).

:param fuzzy:
    Whether to allow fuzzy parsing, allowing for string like "Today is
    January 1, 2047 at 8:21:00AM".

:param fuzzy_with_tokens:
    If ``True``, ``fuzzy`` is automatically set to True, and the parser
    will return a tuple where the first element is the parsed
    :class:`datetime.datetime` datetimestamp and the second element is
    a tuple containing the portions of the string which were ignored:

    .. doctest::

        >>> from dateutil.parser import parse
        >>> parse("Today is January 1, 2047 at 8:21:00AM", fuzzy_with_tokens=True)
        (datetime.datetime(2047, 1, 1, 8, 21), (u'Today is ', u' ', u'at '))

:return:
    Returns a :class:`datetime.datetime` object or, if the
    ``fuzzy_with_tokens`` option is ``True``, returns a tuple, the
    first element being a :class:`datetime.datetime` object, the second
    a tuple containing the fuzzy tokens.

:raises ParserError:
    Raised for invalid or unknown string formats, if the provided
    :class:`tzinfo` is not in a valid format, or if an invalid date would
    be created.

:raises OverflowError:
    Raised if the parsed date exceeds the largest valid C integer on
    your system.

### to_float `classfunction`

```
to_float(
  v: Any
) -> typing.Optional[float]
```

### to_int `classfunction`

```
to_int(
  v: Any
) -> typing.Optional[int]
```

### to_string `classfunction`

```
to_string(
  v: Any
) -> typing.Optional[str]
```

### update_deprecated_type_to_string `classfunction`

```
update_deprecated_type_to_string(
  type
)
```

### verify_metadata_requirements `classfunction`

```
verify_metadata_requirements(
  metadata: dict,
  datatypes: Union[ForwardRef('DataType'), Iterable[ForwardRef('DataType')]]
) -> typing.List[str]
```

