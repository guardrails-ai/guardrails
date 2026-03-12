# Actions

## ReAsk

```python
class ReAsk(IReask)
```

Base class for ReAsk objects.

**Attributes**:

- `incorrect_value` _Any_ - The value that failed validation.
- `fail_results` _List[FailResult]_ - The results of the failed validations.

## FieldReAsk

```python
class FieldReAsk(ReAsk)
```

An implementation of ReAsk that is used to reask for a specific field.
Inherits from ReAsk.

**Attributes**:

- `path` _Optional[List[Any]]_ - a list of keys that
  designated the path to the field that failed validation.

## SkeletonReAsk

```python
class SkeletonReAsk(ReAsk)
```

An implementation of ReAsk that is used to reask for structured data
when the response does not match the expected schema.

Inherits from ReAsk.

## NonParseableReAsk

```python
class NonParseableReAsk(ReAsk)
```

An implementation of ReAsk that is used to reask for structured data
when the response is not parseable as JSON.

Inherits from ReAsk.

## Filter

```python
class Filter()
```

#### apply\_filters

```python
def apply_filters(value: Any) -> Any
```

Recursively filter out any values that are instances of Filter.

## Refrain

```python
class Refrain()
```

#### apply\_refrain

```python
def apply_refrain(value: Any, output_type: OutputTypes) -> Any
```

Recursively check for any values that are instances of Refrain.

If found, return an empty value of the appropriate type.

