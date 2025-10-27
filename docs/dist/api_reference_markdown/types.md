# Types

## OnFailAction

```python
class OnFailAction(str, Enum)
```

OnFailAction is an Enum that represents the different actions that can
be taken when a validation fails.

**Attributes**:

- `REASK` _Literal["reask"]_ - On failure, Reask the LLM.
- `FIX` _Literal["fix"]_ - On failure, apply a static fix.
- `FILTER` _Literal["filter"]_ - On failure, filter out the invalid values.
- `REFRAIN` _Literal["refrain"]_ - On failure, refrain from responding;
  return an empty value.
- `NOOP` _Literal["noop"]_ - On failure, do nothing.
- `EXCEPTION` _Literal["exception"]_ - On failure, raise a ValidationError.
- `FIX_REASK` _Literal["fix_reask"]_ - On failure, apply a static fix,
  check if the fixed value passed validation, if not then reask the LLM.
- `CUSTOM` _Literal["custom"]_ - On failure, call a custom function with the
  invalid value and the FailResult's from any validators run on the value.

## RailTypes

```python
class RailTypes(str, Enum)
```

RailTypes is an Enum that represents the builtin tags for RAIL xml.

**Attributes**:

- `STRING` _Literal["string"]_ - A string value.
- `INTEGER` _Literal["integer"]_ - An integer value.
- `FLOAT` _Literal["float"]_ - A float value.
- `BOOL` _Literal["bool"]_ - A boolean value.
- `DATE` _Literal["date"]_ - A date value.
- `TIME` _Literal["time"]_ - A time value.
  DATETIME (Literal["date-time: - A datetime value.
- `PERCENTAGE` _Literal["percentage"]_ - A percentage value represented as a string.
  Example "20.5%".
- `ENUM` _Literal["enum"]_ - An enum value.
- `LIST` _Literal["list"]_ - A list/array value.
- `OBJECT` _Literal["object"]_ - An object/dictionary value.
- `CHOICE` _Literal["choice"]_ - The options for a discrimated union.
- `CASE` _Literal["case"]_ - A dictionary that contains a discrimated union.

## MessageHistory

```python
MessageHistory = List[Dict[str, Union[Prompt, str]]]
```

## ModelOrListOfModels

```python
ModelOrListOfModels = Union[Type[BaseModel], Type[List[Type[BaseModel]]]]
```

## ModelOrListOrDict

```python
ModelOrListOrDict = Union[Type[BaseModel], Type[List[Type[BaseModel]]],
                          Type[Dict[str, Type[BaseModel]]]]
```

## ModelOrModelUnion

```python
ModelOrModelUnion = Union[Type[BaseModel], Union[Type[BaseModel], Any]]
```

## PydanticValidatorTuple

```python
PydanticValidatorTuple = Tuple[Union[Validator, str, Callable], str]
```

## PydanticValidatorSpec

```python
PydanticValidatorSpec = Union[Validator, PydanticValidatorTuple]
```

## UseValidatorSpec

```python
UseValidatorSpec = Union[Validator, Type[Validator]]
```

## UseManyValidatorTuple

```python
UseManyValidatorTuple = Tuple[
    Type[Validator],
    Optional[Union[List[Any], Dict[str, Any]]],
    Optional[Dict[str, Any]],
]
```

## UseManyValidatorSpec

```python
UseManyValidatorSpec = Union[Validator, UseManyValidatorTuple]
```

## ValidatorMap

```python
ValidatorMap = Dict[str, List[Validator]]
```

